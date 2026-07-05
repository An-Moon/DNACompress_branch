#!/usr/bin/env python3
from __future__ import annotations

"""Convert local Carbon pretraining corpus parquet files to indexed FASTA.

Examples:

    # Build a smoke-test FASTA/index from whatever parquet files have already
    # finished downloading.
    python scripts/build_carbon_hf_fasta_index.py \
      --dataset-root /data/students/Liang_junnan/carbon-pretraining-corpus \
      --fasta-root /data/students/Liang_junnan/carbon-pretraining-corpus_fasta_smoke \
      --index-dir /data/students/Liang_junnan/carbon-pretraining-corpus_index_smoke \
      --max-files 2 \
      --max-bases 200000000 \
      --build-index

    # Full conversion after the download has completed.
    python scripts/build_carbon_hf_fasta_index.py \
      --dataset-root /data/students/Liang_junnan/carbon-pretraining-corpus \
      --fasta-root /data/students/Liang_junnan/carbon-pretraining-corpus_fasta \
      --index-dir /data/students/Liang_junnan/carbon-pretraining-corpus_index \
      --build-index

    # Prefer the smaller 10B-token subset first.
    python scripts/build_carbon_hf_fasta_index.py \
      --dataset-root /data/students/Liang_junnan/carbon-pretraining-corpus \
      --subset eukaryote_generator_10B_subset \
      --fasta-root /data/students/Liang_junnan/carbon-pretraining-corpus_10b_fasta \
      --index-dir /data/students/Liang_junnan/carbon-pretraining-corpus_10b_index \
      --build-index
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from dna_compress.fasta_fragment_index import (  # noqa: E402
    DEFAULT_ANCHOR_STRIDE,
    DEFAULT_IN_MEMORY_THRESHOLD,
    build_fasta_fragment_index,
)


DEFAULT_DATASET_ROOT = Path("/data/students/Liang_junnan/carbon-pretraining-corpus")
DEFAULT_FASTA_ROOT = Path("/data/students/Liang_junnan/carbon-pretraining-corpus_fasta")
DEFAULT_INDEX_DIR = Path("/data/students/Liang_junnan/carbon-pretraining-corpus_index")
DEFAULT_ALPHABET = "ACGTN"


@dataclass
class ConversionStats:
    parquet_files_seen: int = 0
    parquet_files_converted: int = 0
    parquet_files_skipped_existing: int = 0
    rows_seen: int = 0
    rows_written: int = 0
    raw_bases_seen: int = 0
    filtered_bases_written: int = 0
    output_fasta_files: int = 0
    seconds: float = 0.0


def _json_print(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _safe_name(value: str) -> str:
    value = value.strip().replace("/", "_")
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def _discover_parquet_files(dataset_root: Path, subsets: Iterable[str] | None) -> list[Path]:
    subset_set = {str(item).strip("/") for item in subsets or [] if str(item).strip("/")}
    files: list[Path] = []
    for path in sorted(dataset_root.rglob("*.parquet")):
        if ".cache" in path.parts:
            continue
        rel = path.relative_to(dataset_root)
        if subset_set and rel.parts[0] not in subset_set:
            continue
        files.append(path)
    return files


def _source_name_for_path(path: Path, dataset_root: Path, source_depth: int) -> str:
    rel = path.relative_to(dataset_root)
    parts = rel.parts[:-1]
    if source_depth > 0:
        parts = parts[:source_depth]
    if not parts:
        parts = (rel.stem,)
    return "carbon_" + _safe_name("_".join(parts))


def _output_path_for_parquet(path: Path, dataset_root: Path, fasta_root: Path, source_depth: int) -> Path:
    source = _source_name_for_path(path, dataset_root, source_depth)
    rel = path.relative_to(dataset_root)
    stem = _safe_name("_".join(rel.with_suffix("").parts))
    return fasta_root / source / f"{stem}.fasta"


def _sanitize_sequence(value: object, *, alphabet: set[str], rna_to_dna: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("ascii", errors="ignore")
    else:
        text = str(value)
    text = text.upper()
    if rna_to_dna:
        text = text.replace("U", "T")
    return "".join(ch for ch in text if ch in alphabet)


def _wrap_sequence(sequence: str, line_width: int) -> Iterable[str]:
    for start in range(0, len(sequence), line_width):
        yield sequence[start : start + line_width]


def _read_sequence_column(path: Path, requested: str | None) -> str:
    schema = pq.read_schema(path)
    names = set(schema.names)
    candidates = [requested] if requested else []
    candidates.extend(["sequence", "seq", "Sequence", "dna", "text"])
    for candidate in candidates:
        if candidate and candidate in names:
            return candidate
    raise ValueError(f"Could not find a sequence column in {path}; columns={schema.names}")


def convert_carbon_parquets_to_fasta(
    *,
    dataset_root: Path,
    fasta_root: Path,
    subsets: list[str],
    sequence_column: str | None,
    source_depth: int,
    alphabet: str,
    rna_to_dna: bool,
    line_width: int,
    parquet_batch_size: int,
    min_sequence_bases: int,
    max_files: int | None,
    max_rows: int | None,
    max_bases: int | None,
    overwrite: bool,
    skip_existing: bool,
) -> ConversionStats:
    started = time.time()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Carbon dataset root does not exist: {dataset_root}")
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if parquet_batch_size <= 0:
        raise ValueError("parquet_batch_size must be positive")
    if source_depth <= 0:
        raise ValueError("source_depth must be positive")

    alphabet_set = set(alphabet.upper())
    if not alphabet_set:
        raise ValueError("alphabet cannot be empty")

    parquet_files = _discover_parquet_files(dataset_root, subsets)
    if max_files is not None:
        parquet_files = parquet_files[: max(0, int(max_files))]
    stats = ConversionStats(parquet_files_seen=len(parquet_files))
    fasta_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    _json_print(
        {
            "event": "carbon_convert_start",
            "dataset_root": str(dataset_root),
            "fasta_root": str(fasta_root),
            "parquet_files": len(parquet_files),
            "subsets": subsets,
            "sequence_column": sequence_column or "auto",
            "source_depth": source_depth,
            "max_rows": max_rows,
            "max_bases": max_bases,
        }
    )

    stop = False
    for file_index, parquet_path in enumerate(parquet_files):
        output_path = _output_path_for_parquet(parquet_path, dataset_root, fasta_root, source_depth)
        source = output_path.parent.name
        if output_path.exists() and not overwrite:
            if skip_existing:
                stats.parquet_files_skipped_existing += 1
                stats.output_fasta_files += 1
                manifest_rows.append(
                    {
                        "source": source,
                        "parquet_path": str(parquet_path),
                        "fasta_path": str(output_path),
                        "status": "skipped_existing",
                    }
                )
                continue
            raise FileExistsError(f"Output FASTA already exists: {output_path}; use --overwrite or --skip-existing")

        seq_col = _read_sequence_column(parquet_path, sequence_column)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows_written_for_file = 0
        bases_written_for_file = 0
        file_started = time.time()
        pf = pq.ParquetFile(parquet_path)
        with output_path.open("w", encoding="ascii") as handle:
            for batch in pf.iter_batches(batch_size=parquet_batch_size, columns=[seq_col]):
                table = pa.Table.from_batches([batch])
                values = table.column(seq_col).to_pylist()
                for row_offset, raw_sequence in enumerate(values):
                    stats.rows_seen += 1
                    sequence = _sanitize_sequence(raw_sequence, alphabet=alphabet_set, rna_to_dna=rna_to_dna)
                    stats.raw_bases_seen += len(str(raw_sequence)) if raw_sequence is not None else 0
                    if len(sequence) < min_sequence_bases:
                        continue
                    header = (
                        f"{parquet_path.relative_to(dataset_root).as_posix()}|row={stats.rows_seen - 1}|"
                        f"file_row={row_offset}|source={source}|bases={len(sequence)}"
                    )
                    handle.write(f">{header}\n")
                    for line in _wrap_sequence(sequence, line_width):
                        handle.write(line)
                        handle.write("\n")
                    stats.rows_written += 1
                    rows_written_for_file += 1
                    stats.filtered_bases_written += len(sequence)
                    bases_written_for_file += len(sequence)
                    if max_rows is not None and stats.rows_written >= max_rows:
                        stop = True
                        break
                    if max_bases is not None and stats.filtered_bases_written >= max_bases:
                        stop = True
                        break
                if stop:
                    break

        stats.parquet_files_converted += 1
        stats.output_fasta_files += 1
        manifest_rows.append(
            {
                "source": source,
                "parquet_path": str(parquet_path),
                "fasta_path": str(output_path),
                "status": "converted",
                "sequence_column": seq_col,
                "rows_written": rows_written_for_file,
                "bases_written": bases_written_for_file,
                "seconds": time.time() - file_started,
            }
        )
        elapsed = max(time.time() - started, 1e-6)
        _json_print(
            {
                "event": "carbon_convert_progress",
                "file_index": file_index + 1,
                "file_count": len(parquet_files),
                "parquet_path": str(parquet_path),
                "fasta_path": str(output_path),
                "rows_written_for_file": rows_written_for_file,
                "bases_written_for_file": bases_written_for_file,
                "total_bases_written": stats.filtered_bases_written,
                "bases_per_second": stats.filtered_bases_written / elapsed,
            }
        )
        if stop:
            break

    stats.seconds = time.time() - started
    manifest = {
        "schema_version": 1,
        "dataset": "HuggingFaceBio/carbon-pretraining-corpus",
        "dataset_root": str(dataset_root),
        "fasta_root": str(fasta_root),
        "subsets": subsets,
        "source_depth": source_depth,
        "alphabet": alphabet,
        "rna_to_dna": rna_to_dna,
        "stats": asdict(stats),
        "files": manifest_rows,
    }
    (fasta_root / "carbon_fasta_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _json_print({"event": "carbon_convert_done", **asdict(stats)})
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Carbon HF parquet corpus to FASTA and optional indexed FASTA.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--fasta-root", default=str(DEFAULT_FASTA_ROOT))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--subset", action="append", default=[], help="Repeatable subset/config name to include.")
    parser.add_argument("--sequence-column", help="Sequence column name. Defaults to auto-detecting 'sequence'.")
    parser.add_argument("--source-depth", type=int, default=1, help="How many relative path components to fold into source names.")
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    parser.add_argument("--no-rna-to-dna", dest="rna_to_dna", action="store_false")
    parser.add_argument("--line-width", type=int, default=100)
    parser.add_argument("--parquet-batch-size", type=int, default=4096)
    parser.add_argument("--min-sequence-bases", type=int, default=1)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-bases", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--index-only", action="store_true", help="Skip conversion and only build the FASTA fragment index.")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--anchor-stride", type=int, default=DEFAULT_ANCHOR_STRIDE)
    parser.add_argument("--index-chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--index-batch-rows", type=int, default=50_000)
    parser.add_argument("--index-in-memory-threshold", type=int, default=DEFAULT_IN_MEMORY_THRESHOLD)
    parser.set_defaults(rna_to_dna=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    fasta_root = Path(args.fasta_root)
    index_dir = Path(args.index_dir)

    if not args.index_only:
        convert_carbon_parquets_to_fasta(
            dataset_root=dataset_root,
            fasta_root=fasta_root,
            subsets=list(args.subset or []),
            sequence_column=args.sequence_column,
            source_depth=args.source_depth,
            alphabet=args.alphabet,
            rna_to_dna=bool(args.rna_to_dna),
            line_width=int(args.line_width),
            parquet_batch_size=int(args.parquet_batch_size),
            min_sequence_bases=int(args.min_sequence_bases),
            max_files=args.max_files,
            max_rows=args.max_rows,
            max_bases=args.max_bases,
            overwrite=bool(args.overwrite),
            skip_existing=bool(args.skip_existing),
        )

    if args.build_index or args.index_only:
        started = time.time()
        _json_print(
            {
                "event": "carbon_index_start",
                "fasta_root": str(fasta_root),
                "index_dir": str(index_dir),
                "anchor_stride": int(args.anchor_stride),
            }
        )

        def progress(event: dict[str, object]) -> None:
            elapsed = max(time.time() - started, 1e-6)
            processed = int(event["processed_bytes"])
            _json_print(
                {
                    "event": "carbon_index_progress",
                    **event,
                    "elapsed_seconds": elapsed,
                    "bytes_per_second": processed / elapsed,
                }
            )

        stats = build_fasta_fragment_index(
            fasta_root=fasta_root,
            index_dir=index_dir,
            anchor_stride=int(args.anchor_stride),
            chunk_size=int(args.index_chunk_size),
            batch_rows=int(args.index_batch_rows),
            in_memory_threshold=int(args.index_in_memory_threshold),
            progress_callback=progress,
        )
        _json_print(
            {
                "event": "carbon_index_done",
                "seconds": time.time() - started,
                "file_count": stats.file_count,
                "record_count": stats.record_count,
                "run_count": stats.run_count,
                "anchor_count": stats.anchor_count,
                "total_size_bytes": stats.total_size_bytes,
                "source_summary": stats.source_summary,
            }
        )


if __name__ == "__main__":
    main()
