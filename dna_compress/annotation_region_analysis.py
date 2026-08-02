from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .probability_trace import (
    ProbabilityTraceReader,
    TRACE_EMISSION_ORDER_POSITION_MAJOR_V1,
    read_target_probability_trace_positions,
    validate_trace_compatibility,
)


ANNOTATION_REGION_SCHEMA_VERSION = 1

REGION_CLASSES = (
    "intergenic",
    "cds",
    "rna",
    "exon_non_cds",
    "intron",
    "gene_other",
    "repeat_mobile_existing",
)
CLASS_TO_ID = {name: index for index, name in enumerate(REGION_CLASSES)}
ID_TO_CLASS = {index: name for name, index in CLASS_TO_ID.items()}
PRIORITY_LOW_TO_HIGH = (
    "intergenic",
    "gene_other",
    "intron",
    "exon_non_cds",
    "rna",
    "cds",
    "repeat_mobile_existing",
)
MAPPABLE_STATUSES = {"exact_raw_match", "exact_acgt_filtered_match", "exact_coordinate_records_available"}


@dataclass(frozen=True)
class CoordinateRecord:
    record_index: int
    seqid: str
    official_length: int
    official_acgt_length: int
    local_start: int
    local_end: int
    removed_starts: np.ndarray
    removed_ends: np.ndarray
    removed_cumulative_lengths: np.ndarray


@dataclass(frozen=True)
class AnnotationIntervalIndex:
    species: str
    path: Path
    manifest_path: Path
    feature_table_path: Path
    starts: np.ndarray
    ends: np.ndarray
    class_ids: np.ndarray
    sequence_length: int
    manifest: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path: str | Path, *, repo_root: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root_candidate = Path(repo_root) / candidate
    if root_candidate.exists():
        return root_candidate
    return candidate


def read_mapping_validation(annotation_dir: str | Path, *, repo_root: str | Path | None = None) -> dict[str, dict[str, str]]:
    annotation_path = Path(annotation_dir)
    root = Path(repo_root) if repo_root is not None else annotation_path.parents[1]
    validation_path = annotation_path / "official_mapping_validation.csv"
    rows: dict[str, dict[str, str]] = {}
    if validation_path.exists():
        with validation_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                code = str(row["code"])
                row = dict(row)
                for key in ["fasta_path", "gff3_path", "seq_report_path"]:
                    if row.get(key):
                        row[key] = str(resolve_repo_path(row[key], repo_root=root))
                rows[code] = row
    for species_dir in sorted(annotation_path.iterdir() if annotation_path.exists() else []):
        if not species_dir.is_dir() or not (species_dir / "coordinate_records.tsv").exists():
            continue
        code = species_dir.name
        if code in rows:
            continue
        records = load_coordinate_records(species_dir)
        local_length = max((record.local_end for record in records.values()), default=0)
        gff_candidates = sorted(
            list(species_dir.glob("*.gff3"))
            + list(species_dir.glob("*.gff"))
            + list(species_dir.glob("**/genomic.gff"))
            + list(species_dir.glob("**/*.gff3"))
        )
        fasta_candidates = sorted(
            list(species_dir.glob("*.fna"))
            + list(species_dir.glob("*.fasta"))
            + list(species_dir.glob("**/*.fna"))
            + list(species_dir.glob("**/*.fasta"))
        )
        if not gff_candidates:
            continue
        rows[code] = {
            "code": code,
            "species": code,
            "source": "discovered_from_coordinate_records",
            "accession": "",
            "mode": "",
            "chromosome": "",
            "note": "Synthesized because coordinate_records.tsv exists but mapping validation row was unavailable.",
            "status": "exact_coordinate_records_available",
            "local_length": str(local_length),
            "official_record_count": str(len(records)),
            "official_total_length": str(sum(record.official_length for record in records.values())),
            "official_acgt_length": str(sum(record.official_acgt_length for record in records.values())),
            "local_sha256": "",
            "official_raw_sha256": "",
            "official_acgt_sha256": "",
            "exact_raw_match": "",
            "exact_acgt_match": "",
            "local_is_acgt_substring": "",
            "acgt_substring_offset": "",
            "fasta_path": str(fasta_candidates[0]) if fasta_candidates else "",
            "gff3_path": str(gff_candidates[0]),
            "seq_report_path": "",
            "error": "",
        }
    return rows


def load_coordinate_records(species_dir: str | Path) -> dict[str, CoordinateRecord]:
    species_path = Path(species_dir)
    removed_by_record: dict[int, list[tuple[int, int]]] = {}
    removed_path = species_path / "removed_non_acgt_intervals.tsv"
    if removed_path.exists():
        with removed_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                record_index = int(row["record_index"])
                removed_by_record.setdefault(record_index, []).append(
                    (int(row["official_start_1based"]), int(row["official_end_1based"]))
                )

    records: dict[str, CoordinateRecord] = {}
    coordinate_path = species_path / "coordinate_records.tsv"
    with coordinate_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            record_index = int(row["record_index"])
            removed = sorted(removed_by_record.get(record_index, []))
            starts = np.asarray([item[0] for item in removed], dtype=np.int64)
            ends = np.asarray([item[1] for item in removed], dtype=np.int64)
            lengths = ends - starts + 1 if starts.size else np.asarray([], dtype=np.int64)
            cumulative = np.cumsum(lengths, dtype=np.int64)
            seqid = str(row["seqid"])
            records[seqid] = CoordinateRecord(
                record_index=record_index,
                seqid=seqid,
                official_length=int(row["official_length"]),
                official_acgt_length=int(row["official_acgt_length"]),
                local_start=int(row["local_acgt_start_0based"]),
                local_end=int(row["local_acgt_end_0based_exclusive"]),
                removed_starts=starts,
                removed_ends=ends,
                removed_cumulative_lengths=cumulative,
            )
    return records


def removed_bases_before(record: CoordinateRecord, official_position_1based: int) -> int:
    if record.removed_ends.size == 0:
        return 0
    count = int(np.searchsorted(record.removed_ends, int(official_position_1based), side="left"))
    if count <= 0:
        return 0
    return int(record.removed_cumulative_lengths[count - 1])


def map_official_interval_to_local_acgt(
    record: CoordinateRecord,
    start_1based: int,
    end_1based: int,
    *,
    core_base_count: int | None = None,
) -> list[tuple[int, int]]:
    """Map a 1-based inclusive official interval to local ACGT-only half-open intervals."""

    start = max(1, int(start_1based))
    end = min(int(end_1based), int(record.official_length))
    if start > end:
        return []

    chunks: list[tuple[int, int]] = []
    cursor = start
    if record.removed_starts.size:
        first = int(np.searchsorted(record.removed_ends, start, side="left"))
        for removed_start, removed_end in zip(record.removed_starts[first:], record.removed_ends[first:]):
            removed_start = int(removed_start)
            removed_end = int(removed_end)
            if removed_start > end:
                break
            if removed_end < cursor:
                continue
            if cursor < removed_start:
                chunks.append((cursor, min(end, removed_start - 1)))
            cursor = max(cursor, removed_end + 1)
            if cursor > end:
                break
    if cursor <= end:
        chunks.append((cursor, end))

    local_chunks: list[tuple[int, int]] = []
    limit = int(core_base_count) if core_base_count is not None else int(record.local_end)
    for official_start, official_end in chunks:
        before = removed_bases_before(record, official_start)
        local_start = int(record.local_start) + (int(official_start) - 1) - before
        local_end = local_start + (int(official_end) - int(official_start) + 1)
        local_start = max(int(record.local_start), local_start)
        local_end = min(limit, local_end)
        if local_start < local_end:
            local_chunks.append((local_start, local_end))
    return local_chunks


def parse_gff3_attributes(value: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in value.split(";"):
        if not item:
            continue
        if "=" not in item:
            attrs[item] = ""
            continue
        key, raw = item.split("=", 1)
        attrs[key] = raw
    return attrs


def classify_feature(feature_type: str, attributes: dict[str, str]) -> str | None:
    feature = feature_type.lower()
    gbkey = attributes.get("gbkey", "").lower()
    regulatory_class = attributes.get("regulatory_class", "").lower()
    if "repeat" in feature or feature in {
        "mobile_genetic_element",
        "transposable_element",
        "insertion_sequence",
    }:
        return "repeat_mobile_existing"
    if "transpos" in feature or "mobile" in feature:
        return "repeat_mobile_existing"
    if "repeat" in gbkey or "mobile" in gbkey or "transpos" in gbkey:
        return "repeat_mobile_existing"
    if "repeat" in regulatory_class:
        return "repeat_mobile_existing"
    if feature == "cds":
        return "cds"
    if feature != "mrna" and feature.endswith("rna"):
        return "rna"
    if feature == "exon":
        return "exon_non_cds"
    if feature in {"gene", "pseudogene"} or gbkey == "gene":
        return "gene_other"
    return None


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> np.ndarray:
    sorted_intervals = sorted((int(start), int(end)) for start, end in intervals if int(start) < int(end))
    if not sorted_intervals:
        return np.empty((0, 2), dtype=np.int64)
    merged: list[list[int]] = [[sorted_intervals[0][0], sorted_intervals[0][1]]]
    for start, end in sorted_intervals[1:]:
        last = merged[-1]
        if start <= last[1]:
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return np.asarray(merged, dtype=np.int64)


def subtract_intervals(base_intervals: np.ndarray, cutter_intervals: np.ndarray) -> np.ndarray:
    if base_intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if cutter_intervals.size == 0:
        return np.asarray(base_intervals, dtype=np.int64)
    result: list[tuple[int, int]] = []
    cutter_index = 0
    cutters = np.asarray(cutter_intervals, dtype=np.int64)
    for base_start, base_end in np.asarray(base_intervals, dtype=np.int64):
        cursor = int(base_start)
        base_end = int(base_end)
        while cutter_index < cutters.shape[0] and int(cutters[cutter_index, 1]) <= cursor:
            cutter_index += 1
        scan_index = cutter_index
        while scan_index < cutters.shape[0] and int(cutters[scan_index, 0]) < base_end:
            cut_start = int(cutters[scan_index, 0])
            cut_end = int(cutters[scan_index, 1])
            if cursor < cut_start:
                result.append((cursor, min(base_end, cut_start)))
            cursor = max(cursor, cut_end)
            if cursor >= base_end:
                break
            scan_index += 1
        if cursor < base_end:
            result.append((cursor, base_end))
    return merge_intervals(result)


def _interval_base_count(intervals: np.ndarray) -> int:
    if intervals.size == 0:
        return 0
    return int(np.sum(intervals[:, 1] - intervals[:, 0], dtype=np.int64))


def _exclusive_runs_from_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if labels.size == 0:
        empty64 = np.empty((0,), dtype=np.int64)
        return empty64, empty64, np.empty((0,), dtype=np.uint8)
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), changes.astype(np.int64)])
    ends = np.concatenate([changes.astype(np.int64), np.asarray([labels.shape[0]], dtype=np.int64)])
    return starts, ends, labels[starts].astype(np.uint8, copy=False)


def build_annotation_interval_index(
    *,
    species: str,
    annotation_dir: str | Path,
    output_dir: str | Path,
    trace_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    overwrite: bool = False,
) -> AnnotationIntervalIndex:
    annotation_path = Path(annotation_dir)
    root = Path(repo_root) if repo_root is not None else annotation_path.parents[1]
    rows = read_mapping_validation(annotation_path, repo_root=root)
    if species not in rows:
        raise KeyError(f"species {species!r} not found in mapping validation")
    validation = rows[species]
    status = validation.get("status", "")
    if status not in MAPPABLE_STATUSES:
        raise ValueError(f"species {species} is not coordinate verified: status={status}")

    trace_manifest: dict[str, Any] | None = None
    if trace_dir is not None:
        trace_reader = ProbabilityTraceReader(trace_dir)
        sequence_length = int(trace_reader.manifest.core_base_count)
        trace_manifest = trace_reader.manifest.to_json_dict()
    else:
        sequence_length = int(validation["local_length"])

    species_dir = annotation_path / species
    records = load_coordinate_records(species_dir)
    gff_path = resolve_repo_path(str(validation["gff3_path"]), repo_root=root)
    output_path = Path(output_dir)
    index_dir = output_path / "annotation_interval_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / f"{species}.npz"
    manifest_path = index_dir / f"{species}.manifest.json"
    feature_table_path = index_dir / f"{species}.features.tsv.gz"
    if index_path.exists() and manifest_path.exists() and feature_table_path.exists() and not overwrite:
        return read_annotation_interval_index(index_path)

    raw_intervals: dict[str, list[tuple[int, int]]] = {name: [] for name in REGION_CLASSES}
    feature_type_counts: dict[str, int] = {}
    mapped_feature_type_counts: dict[str, int] = {}
    mapped_rows = 0
    clipped_or_split_features = 0
    skipped_unmapped_seqid = 0
    skipped_unclassified = 0

    with gzip.open(feature_table_path, "wt", encoding="utf-8", newline="") as feature_handle:
        writer = csv.DictWriter(
            feature_handle,
            fieldnames=[
                "species",
                "seqid",
                "source",
                "feature_type",
                "class_name",
                "official_start_1based",
                "official_end_1based",
                "local_start_0based",
                "local_end_0based_exclusive",
                "strand",
                "id",
                "parent",
                "gene",
                "gene_biotype",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        with gff_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 9:
                    continue
                seqid, source, feature_type, start_s, end_s, _, strand, _, attributes_s = parts
                feature_type_counts[feature_type] = feature_type_counts.get(feature_type, 0) + 1
                record = records.get(seqid)
                if record is None:
                    skipped_unmapped_seqid += 1
                    continue
                attrs = parse_gff3_attributes(attributes_s)
                class_name = classify_feature(feature_type, attrs)
                if class_name is None:
                    skipped_unclassified += 1
                    continue
                try:
                    official_start = int(start_s)
                    official_end = int(end_s)
                except ValueError:
                    continue
                chunks = map_official_interval_to_local_acgt(
                    record,
                    official_start,
                    official_end,
                    core_base_count=sequence_length,
                )
                if not chunks:
                    continue
                if len(chunks) > 1 or chunks[0] != (
                    record.local_start + official_start - 1 - removed_bases_before(record, official_start),
                    min(sequence_length, record.local_start + official_start - 1 - removed_bases_before(record, official_start) + official_end - official_start + 1),
                ):
                    clipped_or_split_features += 1
                mapped_feature_type_counts[feature_type] = mapped_feature_type_counts.get(feature_type, 0) + 1
                for local_start, local_end in chunks:
                    raw_intervals[class_name].append((local_start, local_end))
                    mapped_rows += 1
                    writer.writerow(
                        {
                            "species": species,
                            "seqid": seqid,
                            "source": source,
                            "feature_type": feature_type,
                            "class_name": class_name,
                            "official_start_1based": official_start,
                            "official_end_1based": official_end,
                            "local_start_0based": local_start,
                            "local_end_0based_exclusive": local_end,
                            "strand": strand,
                            "id": attrs.get("ID", ""),
                            "parent": attrs.get("Parent", ""),
                            "gene": attrs.get("gene", ""),
                            "gene_biotype": attrs.get("gene_biotype", ""),
                        }
                    )

    merged = {name: merge_intervals(items) for name, items in raw_intervals.items()}
    gene_cover = merged["gene_other"]
    coding_or_transcribed = merge_intervals(
        list(map(tuple, merged["cds"].tolist()))
        + list(map(tuple, merged["rna"].tolist()))
        + list(map(tuple, merged["exon_non_cds"].tolist()))
    )
    merged["intron"] = subtract_intervals(gene_cover, coding_or_transcribed)

    labels = np.zeros((sequence_length,), dtype=np.uint8)
    for class_name in PRIORITY_LOW_TO_HIGH:
        if class_name == "intergenic":
            continue
        class_id = CLASS_TO_ID[class_name]
        for start, end in merged[class_name]:
            labels[int(start) : int(end)] = class_id
    starts, ends, class_ids = _exclusive_runs_from_labels(labels)

    exclusive_counts = np.bincount(class_ids, weights=ends - starts, minlength=len(REGION_CLASSES))
    raw_class_base_counts = {
        name: _interval_base_count(intervals)
        for name, intervals in merged.items()
        if name != "intergenic"
    }
    interval_counts = {
        name: int(np.sum(class_ids == class_id))
        for name, class_id in CLASS_TO_ID.items()
    }

    manifest = {
        "schema_version": ANNOTATION_REGION_SCHEMA_VERSION,
        "species": species,
        "sequence_length": int(sequence_length),
        "region_classes": list(REGION_CLASSES),
        "exclusive_priority_low_to_high": list(PRIORITY_LOW_TO_HIGH),
        "mapping_status": status,
        "annotation_dir": str(annotation_path),
        "gff3_path": str(gff_path),
        "gff3_sha256": sha256_file(gff_path),
        "feature_table_path": str(feature_table_path),
        "source_fasta_path": validation.get("fasta_path"),
        "local_source_file": str(root / "datasets" / "DNACorpus" / species),
        "trace_dir": str(trace_dir) if trace_dir is not None else None,
        "trace_manifest": trace_manifest,
        "coordinate_record_count": len(records),
        "feature_type_counts": feature_type_counts,
        "mapped_feature_type_counts": mapped_feature_type_counts,
        "mapped_feature_rows": int(mapped_rows),
        "skipped_unmapped_seqid": int(skipped_unmapped_seqid),
        "skipped_unclassified": int(skipped_unclassified),
        "clipped_or_split_features": int(clipped_or_split_features),
        "raw_class_base_counts": raw_class_base_counts,
        "exclusive_class_base_counts": {
            ID_TO_CLASS[index]: int(count)
            for index, count in enumerate(exclusive_counts.tolist())
        },
        "exclusive_interval_counts": interval_counts,
        "repeat_mobile_existing_note": (
            "Uses only repeat/mobile-like features already present in the official GFF3; "
            "this is not a complete RepeatMasker/TRF repeat landscape."
        ),
    }
    np.savez_compressed(
        index_path,
        schema_version=np.asarray([ANNOTATION_REGION_SCHEMA_VERSION], dtype=np.int16),
        starts=starts.astype(np.int64, copy=False),
        ends=ends.astype(np.int64, copy=False),
        class_ids=class_ids.astype(np.uint8, copy=False),
        region_classes=np.asarray(REGION_CLASSES),
        sequence_length=np.asarray([sequence_length], dtype=np.int64),
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return AnnotationIntervalIndex(
        species=species,
        path=index_path,
        manifest_path=manifest_path,
        feature_table_path=feature_table_path,
        starts=starts,
        ends=ends,
        class_ids=class_ids,
        sequence_length=sequence_length,
        manifest=manifest,
    )


def read_annotation_interval_index(index_path: str | Path) -> AnnotationIntervalIndex:
    path = Path(index_path)
    manifest_path = path.with_suffix(".manifest.json")
    with np.load(path, allow_pickle=False) as data:
        starts = np.asarray(data["starts"], dtype=np.int64)
        ends = np.asarray(data["ends"], dtype=np.int64)
        class_ids = np.asarray(data["class_ids"], dtype=np.uint8)
        sequence_length = int(np.asarray(data["sequence_length"], dtype=np.int64)[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return AnnotationIntervalIndex(
        species=str(manifest["species"]),
        path=path,
        manifest_path=manifest_path,
        feature_table_path=Path(str(manifest["feature_table_path"])),
        starts=starts,
        ends=ends,
        class_ids=class_ids,
        sequence_length=sequence_length,
        manifest=manifest,
    )


def class_ids_for_positions(index: AnnotationIntervalIndex, positions: np.ndarray) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    if pos.size == 0:
        return np.empty(pos.shape, dtype=np.uint8)
    if np.any(pos < 0) or np.any(pos >= int(index.sequence_length)):
        raise ValueError("positions are outside annotation index sequence bounds")
    interval_ids = np.searchsorted(index.starts, pos, side="right") - 1
    if np.any(interval_ids < 0) or np.any(pos >= index.ends[interval_ids]):
        raise ValueError("annotation interval index does not cover all requested positions")
    return index.class_ids[interval_ids].reshape(pos.shape)


def _aggregate_position_bits_by_annotation_regions(
    *,
    annotation_index: AnnotationIntervalIndex,
    position_bits_iter: Iterable[tuple[np.ndarray, np.ndarray]],
    window_bases: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    class_count = len(REGION_CLASSES)
    base_counts = np.zeros((class_count,), dtype=np.int64)
    sum_bits = np.zeros((class_count,), dtype=np.float64)
    min_bpb = np.full((class_count,), np.inf, dtype=np.float64)
    max_bpb = np.full((class_count,), -np.inf, dtype=np.float64)

    window_counts: np.ndarray | None = None
    window_sum_bits: np.ndarray | None = None
    window_class_counts: np.ndarray | None = None
    if window_bases is not None and int(window_bases) > 0:
        wb = int(window_bases)
        window_count = int((annotation_index.sequence_length + wb - 1) // wb)
        window_counts = np.zeros((window_count,), dtype=np.int64)
        window_sum_bits = np.zeros((window_count,), dtype=np.float64)
        window_class_counts = np.zeros((window_count, class_count), dtype=np.int64)
    else:
        wb = 0

    rows_seen = 0
    chunk_count = 0
    for positions_raw, bits_raw in position_bits_iter:
        chunk_count += 1
        positions = np.asarray(positions_raw, dtype=np.int64)
        bits = np.asarray(bits_raw, dtype=np.float64)
        if positions.shape != bits.shape:
            raise ValueError("positions and bits chunks must have matching shapes")
        classes = class_ids_for_positions(annotation_index, positions)
        base_counts += np.bincount(classes, minlength=class_count)
        sum_bits += np.bincount(classes, weights=bits, minlength=class_count)
        for class_id in np.unique(classes).tolist():
            mask = classes == int(class_id)
            if np.any(mask):
                class_bits = bits[mask]
                min_bpb[int(class_id)] = min(float(min_bpb[int(class_id)]), float(np.min(class_bits)))
                max_bpb[int(class_id)] = max(float(max_bpb[int(class_id)]), float(np.max(class_bits)))
        if window_counts is not None and window_sum_bits is not None and window_class_counts is not None:
            window_ids = positions // wb
            window_counts += np.bincount(window_ids, minlength=window_counts.shape[0])
            window_sum_bits += np.bincount(window_ids, weights=bits, minlength=window_sum_bits.shape[0])
            composite = window_ids.astype(np.int64) * class_count + classes.astype(np.int64)
            composite_counts = np.bincount(composite, minlength=window_counts.shape[0] * class_count)
            window_class_counts += composite_counts.reshape(window_counts.shape[0], class_count)
        rows_seen += int(bits.shape[0])

    total_bases = int(np.sum(base_counts))
    rows: list[dict[str, Any]] = []
    exclusive_interval_counts = annotation_index.manifest.get("exclusive_interval_counts", {})
    for class_id, class_name in enumerate(REGION_CLASSES):
        count = int(base_counts[class_id])
        mean = float(sum_bits[class_id] / count) if count else None
        rows.append(
            {
                "species": annotation_index.species,
                "region_class": class_name,
                "base_count": count,
                "interval_count": int(exclusive_interval_counts.get(class_name, 0)),
                "coverage_fraction": float(count / total_bases) if total_bases else 0.0,
                "sum_bits": float(sum_bits[class_id]),
                "mean_bpb": mean,
                "min_bpb": None if not count else float(min_bpb[class_id]),
                "max_bpb": None if not count else float(max_bpb[class_id]),
            }
        )

    window_rows: list[dict[str, Any]] = []
    if window_counts is not None and window_sum_bits is not None and window_class_counts is not None:
        for window_id, count in enumerate(window_counts.tolist()):
            if int(count) <= 0:
                continue
            dominant_class_id = int(np.argmax(window_class_counts[window_id]))
            window_rows.append(
                {
                    "species": annotation_index.species,
                    "window_id": int(window_id),
                    "start": int(window_id * wb),
                    "end": int(min(annotation_index.sequence_length, (window_id + 1) * wb)),
                    "base_count": int(count),
                    "mean_bpb": float(window_sum_bits[window_id] / count),
                    "dominant_region_class": ID_TO_CLASS[dominant_class_id],
                    "dominant_region_fraction": float(window_class_counts[window_id, dominant_class_id] / count),
                }
            )
    return rows, window_rows, int(rows_seen), int(chunk_count)


def aggregate_trace_by_annotation_regions(
    *,
    trace_dir: str | Path,
    annotation_index: AnnotationIntervalIndex,
    window_bases: int | None = 8192,
    verify_trace_checksum: bool = False,
) -> dict[str, Any]:
    reader = ProbabilityTraceReader(trace_dir)
    manifest = reader.manifest
    if int(manifest.core_base_count) != int(annotation_index.sequence_length):
        raise ValueError(
            f"trace/index length mismatch for {annotation_index.species}: "
            f"{manifest.core_base_count} != {annotation_index.sequence_length}"
        )

    started = perf_counter()
    def iter_bits() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for shard in reader.iter_shards(verify_checksum=verify_trace_checksum):
            target_prob = np.asarray(shard["target_prob"], dtype=np.float64)
            positions = np.asarray(shard["emit_position"], dtype=np.int64)
            probabilities = np.clip(target_prob, np.finfo(np.float64).tiny, 1.0)
            yield positions, -np.log2(probabilities)

    rows, window_rows, rows_seen, shard_count = _aggregate_position_bits_by_annotation_regions(
        annotation_index=annotation_index,
        position_bits_iter=iter_bits(),
        window_bases=window_bases,
    )

    elapsed = perf_counter() - started
    return {
        "species": annotation_index.species,
        "schema_version": ANNOTATION_REGION_SCHEMA_VERSION,
        "trace_dir": str(trace_dir),
        "trace_manifest": manifest.to_json_dict(),
        "annotation_index_manifest": annotation_index.manifest,
        "rows_seen": int(rows_seen),
        "shard_count": int(shard_count),
        "elapsed_seconds": float(elapsed),
        "bases_per_second": float(rows_seen / elapsed) if elapsed > 0 else None,
        "region_rows": rows,
        "window_rows": window_rows,
    }


def aggregate_fused_traces_by_annotation_regions(
    *,
    left_trace_dir: str | Path,
    right_trace_dir: str | Path,
    annotation_index: AnnotationIntervalIndex,
    fusion_eta: float = 0.05,
    fusion_initial_left_weight: float = 0.5,
    window_bases: int | None = 8192,
    block_windows: int = 512,
) -> dict[str, Any]:
    left = ProbabilityTraceReader(left_trace_dir)
    right = ProbabilityTraceReader(right_trace_dir)
    compatibility_diffs = validate_trace_compatibility(left.manifest, right.manifest)
    if compatibility_diffs:
        raise ValueError(f"probability traces are not compatible: {compatibility_diffs}")
    if left.manifest.emission_order != TRACE_EMISSION_ORDER_POSITION_MAJOR_V1:
        raise ValueError("fused annotation aggregation expects position_major_v1 traces")
    if int(left.manifest.core_base_count) != int(annotation_index.sequence_length):
        raise ValueError(
            f"trace/index length mismatch for {annotation_index.species}: "
            f"{left.manifest.core_base_count} != {annotation_index.sequence_length}"
        )

    started = perf_counter()
    trace_window_bases = int(left.manifest.window_bases)
    window_count = int(np.ceil(max(left.manifest.core_base_count, 1) / max(trace_window_bases, 1)))
    left_weights = np.full((window_count,), float(fusion_initial_left_weight), dtype=np.float64)
    right_weights = 1.0 - left_weights
    eta = float(fusion_eta)
    eta_power = 1.0 - eta

    def iter_fused_bits() -> Iterable[tuple[np.ndarray, np.ndarray]]:
        for window_start in range(0, window_count, int(block_windows)):
            window_end = min(window_count, window_start + int(block_windows))
            start_position = int(window_start * trace_window_bases)
            end_position = min(int(left.manifest.core_base_count), int(window_end * trace_window_bases))
            positions = np.arange(start_position, end_position, dtype=np.int64)
            left_chunk = read_target_probability_trace_positions(left_trace_dir, positions)
            right_chunk = read_target_probability_trace_positions(right_trace_dir, positions)
            if not np.array_equal(left_chunk["target_symbol"], right_chunk["target_symbol"]):
                raise ValueError("trace target symbols diverged inside compatible hashes")
            left_prob = np.asarray(left_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
            right_prob = np.asarray(right_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
            local_window_count = int(window_end - window_start)
            local_base_count = int(end_position - start_position)
            local_windows = np.arange(local_window_count, dtype=np.int64)
            fused_bits = np.empty((local_base_count,), dtype=np.float64)
            for offset in range(trace_window_bases):
                local_rows = local_windows * trace_window_bases + int(offset)
                valid = local_rows < local_base_count
                if not bool(np.any(valid)):
                    break
                active_rows = local_rows[valid]
                window_ids = int(window_start) + local_windows[valid]
                left_weight = left_weights[window_ids]
                right_weight = right_weights[window_ids]
                left_target = left_prob[active_rows]
                right_target = right_prob[active_rows]
                fused_target = np.maximum(left_weight * left_target + right_weight * right_target, 1e-300)
                fused_bits[active_rows] = -np.log2(fused_target)
                if eta > 0.0:
                    left_new = np.power(left_weight, eta_power) * left_target
                    right_new = np.power(right_weight, eta_power) * right_target
                else:
                    left_new = left_weight * left_target
                    right_new = right_weight * right_target
                denom = np.maximum(left_new + right_new, 1e-300)
                left_weights[window_ids] = left_new / denom
                right_weights[window_ids] = right_new / denom
            yield positions, fused_bits

    rows, window_rows, rows_seen, chunk_count = _aggregate_position_bits_by_annotation_regions(
        annotation_index=annotation_index,
        position_bits_iter=iter_fused_bits(),
        window_bases=window_bases,
    )
    elapsed = perf_counter() - started
    return {
        "species": annotation_index.species,
        "schema_version": ANNOTATION_REGION_SCHEMA_VERSION,
        "fusion_policy": "online_hedge_linear_target_probability_trace",
        "fusion_eta": float(fusion_eta),
        "fusion_initial_left_weight": float(fusion_initial_left_weight),
        "fusion_final_mean_left_weight": float(left_weights.mean()) if left_weights.size else None,
        "left_trace_dir": str(left_trace_dir),
        "right_trace_dir": str(right_trace_dir),
        "left_trace_manifest": left.manifest.to_json_dict(),
        "right_trace_manifest": right.manifest.to_json_dict(),
        "annotation_index_manifest": annotation_index.manifest,
        "rows_seen": int(rows_seen),
        "chunk_count": int(chunk_count),
        "elapsed_seconds": float(elapsed),
        "bases_per_second": float(rows_seen / elapsed) if elapsed > 0 else None,
        "region_rows": rows,
        "window_rows": window_rows,
    }


def read_dnacorpus_sequence_slice(source_file: str | Path, start: int, end: int) -> str:
    if start < 0 or end < start:
        raise ValueError("invalid sequence interval")
    with Path(source_file).open("rb") as handle:
        handle.seek(int(start))
        return handle.read(int(end) - int(start)).decode("ascii").upper()


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return asdict(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")
