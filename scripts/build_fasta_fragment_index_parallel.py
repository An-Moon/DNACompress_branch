#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.fasta_fragment_index import (  # noqa: E402
    ANCHORS_SCHEMA,
    DEFAULT_ANCHOR_STRIDE,
    DEFAULT_BATCH_ROWS,
    DEFAULT_IN_MEMORY_THRESHOLD,
    FILES_SCHEMA,
    INDEX_SCHEMA_VERSION,
    RECORDS_SCHEMA,
    RUNS_SCHEMA,
    build_fasta_fragment_index,
    discover_fasta_files,
)


SCHEMA_BY_NAME = {
    "files.parquet": FILES_SCHEMA,
    "records.parquet": RECORDS_SCHEMA,
    "acgt_runs.parquet": RUNS_SCHEMA,
    "acgt_anchors.parquet": ANCHORS_SCHEMA,
}

OFFSET_COLUMNS = {
    "files.parquet": {"file_id": "file"},
    "records.parquet": {"record_id": "record", "file_id": "file"},
    "acgt_runs.parquet": {"run_id": "run", "record_id": "record", "file_id": "file"},
    "acgt_anchors.parquet": {"run_id": "run"},
}


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def _partition_contiguous(files: list[Path], *, workers: int) -> list[list[Path]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not files:
        return []
    sizes = [path.stat().st_size for path in files]
    total = sum(sizes)
    target = max(1, math.ceil(total / workers))
    shards: list[list[Path]] = []
    current: list[Path] = []
    current_size = 0
    for path, size in zip(files, sizes):
        should_split = current and current_size + size > target and len(shards) < workers - 1
        if should_split:
            shards.append(current)
            current = []
            current_size = 0
        current.append(path)
        current_size += size
    if current:
        shards.append(current)
    return shards


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def _symlink_shard_root(*, fasta_root: Path, files: list[Path], shard_root: Path) -> None:
    _reset_dir(shard_root)
    for path in files:
        rel = path.relative_to(fasta_root)
        link = shard_root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(path)


def _build_shard(
    shard_id: int,
    fasta_root_text: str,
    file_texts: list[str],
    work_dir_text: str,
    anchor_stride: int,
    chunk_size: int,
    batch_rows: int,
    in_memory_threshold: int,
) -> dict[str, Any]:
    fasta_root = Path(fasta_root_text)
    files = [Path(item) for item in file_texts]
    work_dir = Path(work_dir_text)
    shard_root = work_dir / "roots" / f"shard_{shard_id:04d}"
    shard_index = work_dir / "shards" / f"shard_{shard_id:04d}"
    shard_log = work_dir / "logs" / f"shard_{shard_id:04d}.jsonl"
    shard_index.parent.mkdir(parents=True, exist_ok=True)
    shard_log.parent.mkdir(parents=True, exist_ok=True)
    _symlink_shard_root(fasta_root=fasta_root, files=files, shard_root=shard_root)
    if shard_index.exists():
        shutil.rmtree(shard_index)

    started = time.time()

    def progress(event: dict[str, Any]) -> None:
        elapsed = max(time.time() - started, 1e-6)
        processed = int(event["processed_bytes"])
        with shard_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "shard_progress",
                        "shard_id": shard_id,
                        **event,
                        "elapsed_seconds": elapsed,
                        "bytes_per_second": processed / elapsed,
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")

    stats = build_fasta_fragment_index(
        fasta_root=shard_root,
        index_dir=shard_index,
        anchor_stride=anchor_stride,
        chunk_size=chunk_size,
        batch_rows=batch_rows,
        in_memory_threshold=in_memory_threshold,
        progress_callback=progress,
    )
    result = {
        "event": "shard_done",
        "shard_id": shard_id,
        "seconds": time.time() - started,
        "file_count": stats.file_count,
        "record_count": stats.record_count,
        "run_count": stats.run_count,
        "anchor_count": stats.anchor_count,
        "total_size_bytes": stats.total_size_bytes,
        "index_dir": str(shard_index),
    }
    with shard_log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False))
        handle.write("\n")
    return result


def _merge_source_summary(target: dict[str, dict[str, int]], source: dict[str, dict[str, Any]]) -> None:
    for name, stats in source.items():
        entry = target.setdefault(name, {})
        for key, value in stats.items():
            entry[key] = int(entry.get(key, 0)) + int(value)


def _offset_batch(batch: pa.RecordBatch, *, schema: pa.Schema, filename: str, offsets: dict[str, int]) -> pa.Table:
    arrays = []
    column_offsets = OFFSET_COLUMNS[filename]
    for field in schema:
        array = batch.column(field.name)
        offset_name = column_offsets.get(field.name)
        if offset_name is not None and offsets[offset_name]:
            array = pc.add(array, pa.scalar(int(offsets[offset_name]), type=field.type))
        arrays.append(array)
    return pa.Table.from_arrays(arrays, schema=schema)


def _merge_parquet_file(*, shard_dirs: list[Path], output_dir: Path, filename: str, shard_offsets: list[dict[str, int]], merge_batch_rows: int) -> None:
    schema = SCHEMA_BY_NAME[filename]
    writer: pq.ParquetWriter | None = None
    try:
        for shard_dir, offsets in zip(shard_dirs, shard_offsets):
            parquet = pq.ParquetFile(shard_dir / filename)
            for batch in parquet.iter_batches(batch_size=merge_batch_rows, use_threads=False):
                table = _offset_batch(batch, schema=schema, filename=filename, offsets=offsets)
                if writer is None:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(output_dir / filename, schema)
                writer.write_table(table)
        if writer is None:
            output_dir.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(output_dir / filename, schema)
            writer.write_table(pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema))
    finally:
        if writer is not None:
            writer.close()


def _merge_shards(
    *,
    fasta_root: Path,
    shard_dirs: list[Path],
    output_dir: Path,
    anchor_stride: int,
    merge_batch_rows: int,
) -> dict[str, Any]:
    manifests = [json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8")) for shard_dir in shard_dirs]
    shard_offsets: list[dict[str, int]] = []
    file_offset = 0
    record_offset = 0
    run_offset = 0
    anchor_offset = 0
    source_summary: dict[str, dict[str, int]] = {}
    total_size = 0
    for manifest in manifests:
        shard_offsets.append({"file": file_offset, "record": record_offset, "run": run_offset, "anchor": anchor_offset})
        file_offset += int(manifest["file_count"])
        record_offset += int(manifest["record_count"])
        run_offset += int(manifest["run_count"])
        anchor_offset += int(manifest["anchor_count"])
        total_size += int(manifest["total_size_bytes"])
        _merge_source_summary(source_summary, manifest.get("source_summary", {}))

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename in ("files.parquet", "records.parquet", "acgt_runs.parquet", "acgt_anchors.parquet"):
        _json_print({"event": "merge_start", "file": filename})
        _merge_parquet_file(
            shard_dirs=shard_dirs,
            output_dir=output_dir,
            filename=filename,
            shard_offsets=shard_offsets,
            merge_batch_rows=merge_batch_rows,
        )
        _json_print({"event": "merge_done", "file": filename})

    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "fasta_root": str(fasta_root),
        "index_dir": str(output_dir),
        "anchor_stride": int(anchor_stride),
        "file_count": int(file_offset),
        "record_count": int(record_offset),
        "run_count": int(run_offset),
        "anchor_count": int(anchor_offset),
        "total_size_bytes": int(total_size),
        "source_summary": source_summary,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a FASTA fragment index with parallel shard indexing and schema-compatible merge.")
    parser.add_argument("--fasta-root", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--anchor-stride", type=int, default=DEFAULT_ANCHOR_STRIDE)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--merge-batch-rows", type=int, default=100_000)
    parser.add_argument("--in-memory-threshold", type=int, default=DEFAULT_IN_MEMORY_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    fasta_root = Path(args.fasta_root).resolve()
    index_dir = Path(args.index_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    if not fasta_root.is_dir():
        raise FileNotFoundError(f"FASTA root does not exist: {fasta_root}")
    if index_dir.exists() and any(index_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"index dir already exists and is not empty: {index_dir}; pass --overwrite")
    if work_dir.exists() and any(work_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"work dir already exists and is not empty: {work_dir}; pass --overwrite")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be positive")

    files = discover_fasta_files(fasta_root)
    if not files:
        raise FileNotFoundError(f"no FASTA files found under {fasta_root}")
    shards = _partition_contiguous(files, workers=min(int(args.workers), len(files)))
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=False)

    total_size = sum(path.stat().st_size for path in files)
    _json_print(
        {
            "event": "parallel_index_start",
            "fasta_root": str(fasta_root),
            "index_dir": str(index_dir),
            "work_dir": str(work_dir),
            "workers": int(args.workers),
            "shards": len(shards),
            "file_count": len(files),
            "total_size_bytes": total_size,
        }
    )
    for shard_id, shard_files in enumerate(shards):
        _json_print(
            {
                "event": "shard_plan",
                "shard_id": shard_id,
                "file_count": len(shard_files),
                "size_bytes": sum(path.stat().st_size for path in shard_files),
                "first_rel_path": str(shard_files[0].relative_to(fasta_root)),
                "last_rel_path": str(shard_files[-1].relative_to(fasta_root)),
            }
        )

    started = time.time()
    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [
            executor.submit(
                _build_shard,
                shard_id,
                str(fasta_root),
                [str(path) for path in shard_files],
                str(work_dir),
                int(args.anchor_stride),
                int(args.chunk_size),
                int(args.batch_rows),
                int(args.in_memory_threshold),
            )
            for shard_id, shard_files in enumerate(shards)
        ]
        for future in as_completed(futures):
            result = future.result()
            _json_print(result)

    shard_dirs = [work_dir / "shards" / f"shard_{shard_id:04d}" for shard_id in range(len(shards))]
    manifest = _merge_shards(
        fasta_root=fasta_root,
        shard_dirs=shard_dirs,
        output_dir=index_dir,
        anchor_stride=int(args.anchor_stride),
        merge_batch_rows=int(args.merge_batch_rows),
    )
    _json_print({"event": "parallel_index_done", "seconds": time.time() - started, **manifest})
    if not args.keep_work_dir:
        shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()
