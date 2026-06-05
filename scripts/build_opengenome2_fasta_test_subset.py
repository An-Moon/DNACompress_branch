from __future__ import annotations

"""Build a real-FASTA OpenGenome2 test subset from indexed records.

Example:

    python scripts/build_opengenome2_fasta_test_subset.py \
        --index-dir /data/students/Liang_junnan/opengenome2_subset/index \
        --output-dir /data/students/Liang_junnan/opengenome2_subset/real_fasta_test_subset_100mb_per_source \
        --target-bytes-per-source 100000000 \
        --seed 0 \
        --overwrite
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import pyarrow.parquet as pq


DEFAULT_INDEX_DIR = Path("/data/students/Liang_junnan/opengenome2_subset/index")
DEFAULT_OUTPUT_DIR = Path("/data/students/Liang_junnan/opengenome2_subset/real_fasta_test_subset_1gb_per_source")
DEFAULT_TARGET_BYTES_PER_SOURCE = 1_000_000_000


def _source_seed(seed: int, source: str) -> int:
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:8], "little")


def _safe_source_filename(source: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in source)
    return f"{safe}.fasta"


def _load_manifest(index_dir: Path) -> dict[str, Any]:
    path = index_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"index manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_file_paths(index_dir: Path, fasta_root: Path) -> dict[int, Path]:
    table = pq.read_table(index_dir / "files.parquet", columns=["file_id", "rel_path"], use_threads=False)
    data = table.to_pydict()
    return {int(file_id): fasta_root / str(rel_path) for file_id, rel_path in zip(data["file_id"], data["rel_path"])}


def _scan_source_stats(index_dir: Path, *, batch_rows: int) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    parquet = pq.ParquetFile(index_dir / "records.parquet")
    for batch in parquet.iter_batches(
        columns=["source", "record_start_byte", "record_end_byte"],
        batch_size=batch_rows,
        use_threads=False,
    ):
        data = batch.to_pydict()
        for source, start, end in zip(data["source"], data["record_start_byte"], data["record_end_byte"]):
            source = str(source)
            entry = stats.setdefault(source, {"record_count": 0, "record_bytes": 0})
            entry["record_count"] += 1
            entry["record_bytes"] += max(0, int(end) - int(start))
    return stats


def _choose_plans(
    stats: dict[str, dict[str, int]],
    *,
    target_bytes_per_source: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for source, entry in sorted(stats.items()):
        record_count = int(entry["record_count"])
        total_bytes = int(entry["record_bytes"])
        include_all = total_bytes <= target_bytes_per_source
        start_ordinal = 0 if include_all else random.Random(_source_seed(seed, source)).randrange(record_count)
        plans[source] = {
            "source": source,
            "target_bytes": int(target_bytes_per_source),
            "total_record_count": record_count,
            "total_record_bytes": total_bytes,
            "include_all": include_all,
            "start_ordinal": int(start_ordinal),
            "selected_record_count": 0,
            "selected_bytes": 0,
            "bytes_written": 0,
            "spans": [],
            "done": False,
        }
    return plans


def _append_span(plan: dict[str, Any], *, file_id: int, record_id: int, start: int, end: int) -> None:
    spans = plan["spans"]
    if spans and spans[-1]["file_id"] == file_id and spans[-1]["end_record_id"] + 1 == record_id:
        spans[-1]["end_record_id"] = int(record_id)
        spans[-1]["end_byte"] = int(end)
        spans[-1]["record_count"] += 1
        return
    spans.append(
        {
            "file_id": int(file_id),
            "start_record_id": int(record_id),
            "end_record_id": int(record_id),
            "start_byte": int(start),
            "end_byte": int(end),
            "record_count": 1,
        }
    )


def _copy_record_bytes(
    *,
    source: str,
    plan: dict[str, Any],
    file_paths: dict[int, Path],
    output_handle,
    source_handles: dict[int, Any],
    file_id: int,
    record_id: int,
    start: int,
    end: int,
) -> None:
    if end <= start:
        return
    handle = source_handles.get(file_id)
    if handle is None or handle.closed:
        handle = file_paths[file_id].open("rb")
        source_handles[file_id] = handle
    handle.seek(start)
    remaining = end - start
    last_chunk = b""
    while remaining > 0:
        chunk = handle.read(min(1024 * 1024, remaining))
        if not chunk:
            raise IOError(f"unexpected EOF while reading record {record_id} from source {source!r}")
        output_handle.write(chunk)
        last_chunk = chunk
        remaining -= len(chunk)
    if last_chunk and not last_chunk.endswith(b"\n"):
        output_handle.write(b"\n")
        plan["bytes_written"] += 1
    record_bytes = end - start
    plan["selected_record_count"] += 1
    plan["selected_bytes"] += record_bytes
    plan["bytes_written"] += record_bytes
    _append_span(plan, file_id=file_id, record_id=record_id, start=start, end=end)


def build_opengenome2_fasta_test_subset(
    *,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    target_bytes_per_source: int = DEFAULT_TARGET_BYTES_PER_SOURCE,
    seed: int = 0,
    batch_rows: int = 100_000,
    overwrite: bool = False,
) -> dict[str, Any]:
    index_dir = Path(index_dir)
    output_dir = Path(output_dir)
    if target_bytes_per_source <= 0:
        raise ValueError("target_bytes_per_source must be > 0")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be > 0")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    manifest = _load_manifest(index_dir)
    fasta_root = Path(str(manifest["fasta_root"]))
    file_paths = _load_file_paths(index_dir, fasta_root)
    stats = _scan_source_stats(index_dir, batch_rows=batch_rows)
    plans = _choose_plans(stats, target_bytes_per_source=target_bytes_per_source, seed=seed)

    output_handles = {
        source: (output_dir / _safe_source_filename(source)).open("wb")
        for source in sorted(plans)
    }
    source_handles: dict[int, Any] = {}
    try:
        parquet = pq.ParquetFile(index_dir / "records.parquet")
        for pass_index in range(2):
            source_ordinals = {source: 0 for source in plans}
            for batch in parquet.iter_batches(
                columns=["record_id", "file_id", "source", "record_start_byte", "record_end_byte"],
                batch_size=batch_rows,
                use_threads=False,
            ):
                data = batch.to_pydict()
                for record_id, file_id, source, start, end in zip(
                    data["record_id"],
                    data["file_id"],
                    data["source"],
                    data["record_start_byte"],
                    data["record_end_byte"],
                ):
                    source = str(source)
                    plan = plans[source]
                    ordinal = source_ordinals[source]
                    source_ordinals[source] += 1
                    if plan["done"]:
                        continue
                    include = False
                    if plan["include_all"]:
                        include = pass_index == 0
                    elif pass_index == 0:
                        include = ordinal >= int(plan["start_ordinal"])
                    else:
                        include = ordinal < int(plan["start_ordinal"])
                    if not include:
                        continue
                    _copy_record_bytes(
                        source=source,
                        plan=plan,
                        file_paths=file_paths,
                        output_handle=output_handles[source],
                        source_handles=source_handles,
                        file_id=int(file_id),
                        record_id=int(record_id),
                        start=int(start),
                        end=int(end),
                    )
                    if (not plan["include_all"]) and int(plan["selected_bytes"]) >= target_bytes_per_source:
                        plan["done"] = True
                if all(bool(plan["done"]) or bool(plan["include_all"]) for plan in plans.values()) and pass_index == 0:
                    # All small sources are complete after pass 0, and large sources already hit target.
                    if all(bool(plan["done"]) or int(plan["selected_bytes"]) >= int(plan["total_record_bytes"]) for plan in plans.values()):
                        break
            for plan in plans.values():
                if plan["include_all"]:
                    plan["done"] = True
            if all(bool(plan["done"]) for plan in plans.values()):
                break
    finally:
        for handle in output_handles.values():
            handle.close()
        for handle in source_handles.values():
            handle.close()

    result = {
        "schema_version": 1,
        "layout": "one_real_fasta_per_source",
        "index_dir": str(index_dir),
        "fasta_root": str(fasta_root),
        "output_dir": str(output_dir),
        "target_bytes_per_source": int(target_bytes_per_source),
        "seed": int(seed),
        "sources": {
            source: {
                key: value
                for key, value in plan.items()
                if key not in {"done", "source"}
            }
            | {"output_path": str(output_dir / _safe_source_filename(source))}
            for source, plan in sorted(plans.items())
        },
        "argv": sys.argv,
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a real FASTA OpenGenome2 test subset by source.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-bytes-per-source", type=int, default=DEFAULT_TARGET_BYTES_PER_SOURCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-rows", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest = build_opengenome2_fasta_test_subset(
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        target_bytes_per_source=args.target_bytes_per_source,
        seed=args.seed,
        batch_rows=args.batch_rows,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
