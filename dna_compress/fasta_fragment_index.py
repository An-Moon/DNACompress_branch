from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from typing import Any, Callable

# Runtime cache construction can run on shared machines with tight process/thread
# limits. Keep native libraries conservative unless the caller explicitly opts in.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .megadna_loader import MEGADNA_BASE_TO_TOKEN
from .tokenization import normalize_alphabet


INDEX_SCHEMA_VERSION = 1
DEFAULT_FASTA_ROOT = Path("/data/students/Liang_junnan/opengenome2_subset/fasta")
DEFAULT_INDEX_DIR = Path("/data/students/Liang_junnan/opengenome2_subset/index")
DEFAULT_ANCHOR_STRIDE = 262_144
DEFAULT_BATCH_ROWS = 50_000
DEFAULT_IN_MEMORY_THRESHOLD = 128 * 1024 * 1024
RUNTIME_CACHE_SCHEMA_VERSION = 2
RUNTIME_CACHE_DIR_NAME = "runtime_cache_v2"
EVAL_CACHE_SCHEMA_VERSION = 1
EVAL_CACHE_SOURCE_SAMPLING_STRATEGY = "per_sample_probability"

try:
    pa.set_cpu_count(max(1, int(os.environ.get("ARROW_NUM_THREADS", "1"))))
except (AttributeError, ValueError):
    pass
try:
    pa.set_io_thread_count(max(1, int(os.environ.get("ARROW_NUM_THREADS", "1"))))
except (AttributeError, ValueError):
    pass

_BASE_TO_TOKEN_BYTE = np.zeros(256, dtype=np.uint8)
for _base, _token in MEGADNA_BASE_TO_TOKEN.items():
    _BASE_TO_TOKEN_BYTE[ord(_base)] = _token


FILES_SCHEMA = pa.schema(
    [
        ("file_id", pa.int64()),
        ("source", pa.string()),
        ("rel_path", pa.string()),
        ("size_bytes", pa.int64()),
        ("mtime_ns", pa.int64()),
    ]
)

RECORDS_SCHEMA = pa.schema(
    [
        ("record_id", pa.int64()),
        ("file_id", pa.int64()),
        ("source", pa.string()),
        ("header", pa.string()),
        ("record_start_byte", pa.int64()),
        ("sequence_start_byte", pa.int64()),
        ("record_end_byte", pa.int64()),
        ("sequence_bases", pa.int64()),
        ("acgt_bases", pa.int64()),
        ("n_bases", pa.int64()),
        ("non_acgt_bases", pa.int64()),
        ("lowercase_bases", pa.int64()),
        ("a_bases", pa.int64()),
        ("c_bases", pa.int64()),
        ("g_bases", pa.int64()),
        ("t_bases", pa.int64()),
    ]
)

RUNS_SCHEMA = pa.schema(
    [
        ("run_id", pa.int64()),
        ("record_id", pa.int64()),
        ("file_id", pa.int64()),
        ("source", pa.string()),
        ("run_start_file_byte", pa.int64()),
        ("run_start_record_base", pa.int64()),
        ("run_base_length", pa.int64()),
    ]
)

ANCHORS_SCHEMA = pa.schema(
    [
        ("run_id", pa.int64()),
        ("run_base_offset", pa.int64()),
        ("file_byte_offset", pa.int64()),
    ]
)


class _ParquetBatchWriter:
    def __init__(self, path: Path, schema: pa.Schema, batch_rows: int = DEFAULT_BATCH_ROWS) -> None:
        self.path = path
        self.schema = schema
        self.batch_rows = batch_rows
        self.columns: dict[str, list[Any]] = {field.name: [] for field in schema}
        self.writer: pq.ParquetWriter | None = None
        self.rows_written = 0

    def append(self, **row: Any) -> None:
        for field in self.schema:
            self.columns[field.name].append(row[field.name])
        if len(next(iter(self.columns.values()))) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        row_count = len(next(iter(self.columns.values())))
        if row_count == 0:
            return
        table = pa.Table.from_arrays(
            [pa.array(self.columns[field.name], type=field.type) for field in self.schema],
            schema=self.schema,
        )
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema)
        self.writer.write_table(table)
        self.rows_written += row_count
        for values in self.columns.values():
            values.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_arrays([pa.array([], type=field.type) for field in self.schema], schema=self.schema)
            self.writer = pq.ParquetWriter(self.path, self.schema)
            self.writer.write_table(table)
        self.writer.close()


@dataclass
class _RecordState:
    record_id: int
    file_id: int
    source: str
    header: str
    record_start_byte: int
    sequence_start_byte: int
    sequence_bases: int = 0
    acgt_bases: int = 0
    n_bases: int = 0
    non_acgt_bases: int = 0
    lowercase_bases: int = 0
    a_bases: int = 0
    c_bases: int = 0
    g_bases: int = 0
    t_bases: int = 0


@dataclass
class _RunState:
    run_id: int
    record_id: int
    file_id: int
    source: str
    run_start_file_byte: int
    run_start_record_base: int
    run_base_length: int = 0
    next_anchor_offset: int = 0


@dataclass
class BuildStats:
    file_count: int
    record_count: int
    run_count: int
    anchor_count: int
    total_size_bytes: int
    source_summary: dict[str, dict[str, int]]


def _byte_is_space(byte_value: int) -> bool:
    return byte_value in (9, 10, 11, 12, 13, 32)


def _byte_is_alpha(byte_value: int) -> bool:
    return (65 <= byte_value <= 90) or (97 <= byte_value <= 122)


def _to_upper(byte_value: int) -> int:
    if 97 <= byte_value <= 122:
        return byte_value - 32
    return byte_value


def _decode_header(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip()


def discover_fasta_files(fasta_root: Path) -> list[Path]:
    return sorted(path for path in fasta_root.rglob("*.fasta") if path.is_file())


def build_fasta_fragment_index(
    *,
    fasta_root: str | Path = DEFAULT_FASTA_ROOT,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    anchor_stride: int = DEFAULT_ANCHOR_STRIDE,
    chunk_size: int = 4 * 1024 * 1024,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    in_memory_threshold: int = DEFAULT_IN_MEMORY_THRESHOLD,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> BuildStats:
    if anchor_stride <= 0:
        raise ValueError("anchor_stride must be > 0")
    root = Path(fasta_root)
    output = Path(index_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"FASTA root does not exist: {root}")

    output.mkdir(parents=True, exist_ok=True)
    files_writer = _ParquetBatchWriter(output / "files.parquet", FILES_SCHEMA, batch_rows)
    records_writer = _ParquetBatchWriter(output / "records.parquet", RECORDS_SCHEMA, batch_rows)
    runs_writer = _ParquetBatchWriter(output / "acgt_runs.parquet", RUNS_SCHEMA, batch_rows)
    anchors_writer = _ParquetBatchWriter(output / "acgt_anchors.parquet", ANCHORS_SCHEMA, batch_rows)

    fasta_files = discover_fasta_files(root)
    source_summary: dict[str, dict[str, int]] = {}
    total_size = 0
    next_record_id = 0
    next_run_id = 0
    anchor_count = 0

    def close_run(run: _RunState | None) -> None:
        nonlocal next_run_id
        if run is None or run.run_base_length <= 0:
            return
        runs_writer.append(
            run_id=run.run_id,
            record_id=run.record_id,
            file_id=run.file_id,
            source=run.source,
            run_start_file_byte=run.run_start_file_byte,
            run_start_record_base=run.run_start_record_base,
            run_base_length=run.run_base_length,
        )

    def close_record(record: _RecordState | None, record_end_byte: int) -> None:
        if record is None:
            return
        records_writer.append(
            record_id=record.record_id,
            file_id=record.file_id,
            source=record.source,
            header=record.header,
            record_start_byte=record.record_start_byte,
            sequence_start_byte=record.sequence_start_byte,
            record_end_byte=record_end_byte,
            sequence_bases=record.sequence_bases,
            acgt_bases=record.acgt_bases,
            n_bases=record.n_bases,
            non_acgt_bases=record.non_acgt_bases,
            lowercase_bases=record.lowercase_bases,
            a_bases=record.a_bases,
            c_bases=record.c_bases,
            g_bases=record.g_bases,
            t_bases=record.t_bases,
        )

    for file_id, path in enumerate(fasta_files):
        rel_path = path.relative_to(root)
        source = rel_path.parts[0] if rel_path.parts else "."
        stat = path.stat()
        total_size += int(stat.st_size)
        source_stats = source_summary.setdefault(
            source,
            {
                "file_count": 0,
                "record_count": 0,
                "run_count": 0,
                "anchor_count": 0,
                "size_bytes": 0,
                "sequence_bases": 0,
                "acgt_bases": 0,
                "n_bases": 0,
                "non_acgt_bases": 0,
                "lowercase_bases": 0,
            },
        )
        source_stats["file_count"] += 1
        source_stats["size_bytes"] += int(stat.st_size)
        files_writer.append(
            file_id=file_id,
            source=source,
            rel_path=str(rel_path),
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
        )

        current_record: _RecordState | None = None
        current_run: _RunState | None = None
        offset = 0

        def finish_run() -> None:
            nonlocal current_run
            close_run(current_run)
            if current_run is not None and current_run.run_base_length > 0:
                source_stats["run_count"] += 1
            current_run = None

        def finish_record(record_end_byte: int) -> None:
            nonlocal current_record
            close_record(current_record, record_end_byte)
            if current_record is not None:
                source_stats["sequence_bases"] += current_record.sequence_bases
                source_stats["acgt_bases"] += current_record.acgt_bases
                source_stats["n_bases"] += current_record.n_bases
                source_stats["non_acgt_bases"] += current_record.non_acgt_bases
                source_stats["lowercase_bases"] += current_record.lowercase_bases
            current_record = None

        def start_run(first_base_file_offset: int) -> None:
            nonlocal current_run, next_run_id
            if current_record is None:
                raise ValueError("cannot start an ACGT run without an active FASTA record")
            current_run = _RunState(
                run_id=next_run_id,
                record_id=current_record.record_id,
                file_id=file_id,
                source=source,
                run_start_file_byte=first_base_file_offset,
                run_start_record_base=current_record.sequence_bases,
            )
            next_run_id += 1

        def add_acgt_block(line_start: int, base_positions: np.ndarray, upper: np.ndarray, lower_mask: np.ndarray) -> None:
            nonlocal anchor_count
            if current_record is None or base_positions.size == 0:
                return
            if current_run is None:
                start_run(line_start + int(base_positions[0]))
            if current_run is None:
                raise ValueError("internal error: ACGT run was not started")

            block_count = int(base_positions.size)
            contiguous = bool(block_count == int(base_positions[-1] - base_positions[0] + 1))
            block_start = int(base_positions[0])
            while current_run.next_anchor_offset <= current_run.run_base_length + block_count - 1:
                index_in_block = current_run.next_anchor_offset - current_run.run_base_length
                if contiguous:
                    file_byte_offset = line_start + block_start + index_in_block
                else:
                    file_byte_offset = line_start + int(base_positions[index_in_block])
                anchors_writer.append(
                    run_id=current_run.run_id,
                    run_base_offset=current_run.next_anchor_offset,
                    file_byte_offset=file_byte_offset,
                )
                current_run.next_anchor_offset += anchor_stride
                anchor_count += 1
                source_stats["anchor_count"] += 1

            block = upper[base_positions]
            current_run.run_base_length += block_count
            current_record.sequence_bases += block_count
            current_record.acgt_bases += block_count
            current_record.lowercase_bases += int(np.count_nonzero(lower_mask[base_positions]))
            current_record.a_bases += int(np.count_nonzero(block == ord("A")))
            current_record.c_bases += int(np.count_nonzero(block == ord("C")))
            current_record.g_bases += int(np.count_nonzero(block == ord("G")))
            current_record.t_bases += int(np.count_nonzero(block == ord("T")))

        def add_non_acgt_base(byte_value: int) -> None:
            if current_record is None:
                return
            upper_value = _to_upper(byte_value)
            if not _byte_is_alpha(byte_value):
                return
            current_record.sequence_bases += 1
            current_record.non_acgt_bases += 1
            if upper_value != byte_value:
                current_record.lowercase_bases += 1
            if upper_value == ord("N"):
                current_record.n_bases += 1

        def process_sequence_values(values: np.ndarray, values_file_start: int) -> None:
            if current_record is None or values.size == 0:
                return
            upper = values.copy()
            lower_mask = (upper >= ord("a")) & (upper <= ord("z"))
            upper[lower_mask] -= 32
            base_mask = (
                (upper == ord("A"))
                | (upper == ord("C"))
                | (upper == ord("G"))
                | (upper == ord("T"))
            )
            whitespace_mask = (
                (values == 9)
                | (values == 10)
                | (values == 11)
                | (values == 12)
                | (values == 13)
                | (values == 32)
            )
            break_positions = np.flatnonzero((~base_mask) & (~whitespace_mask))
            cursor = 0
            for break_position in np.append(break_positions, values.size):
                break_index = int(break_position)
                if break_index > cursor:
                    local_bases = np.flatnonzero(base_mask[cursor:break_index])
                    if local_bases.size:
                        add_acgt_block(values_file_start, local_bases + cursor, upper, lower_mask)
                if break_index < values.size:
                    finish_run()
                    add_non_acgt_base(int(values[break_index]))
                cursor = break_index + 1

        if int(stat.st_size) <= in_memory_threshold:
            payload = path.read_bytes()
            header_positions = [match.start() for match in re.finditer(rb"(?m)^>", payload)]
            for header_index, header_position in enumerate(header_positions):
                finish_run()
                finish_record(header_position)
                line_end = payload.find(b"\n", header_position)
                if line_end < 0:
                    line_end = len(payload)
                header = payload[header_position + 1 : line_end].rstrip(b"\r")
                sequence_start = min(line_end + 1, len(payload))
                record_end = header_positions[header_index + 1] if header_index + 1 < len(header_positions) else len(payload)
                current_record = _RecordState(
                    record_id=next_record_id,
                    file_id=file_id,
                    source=source,
                    header=_decode_header(header),
                    record_start_byte=header_position,
                    sequence_start_byte=sequence_start,
                )
                next_record_id += 1
                source_stats["record_count"] += 1
                if sequence_start < record_end:
                    process_sequence_values(np.frombuffer(payload[sequence_start:record_end], dtype=np.uint8), sequence_start)
            offset = len(payload)
        else:
            with path.open("rb") as handle:
                for raw_line in handle:
                    line_start = offset
                    offset += len(raw_line)
                    line = raw_line.rstrip(b"\r\n")
                    if line.startswith(b">"):
                        finish_run()
                        finish_record(line_start)
                        current_record = _RecordState(
                            record_id=next_record_id,
                            file_id=file_id,
                            source=source,
                            header=_decode_header(line[1:]),
                            record_start_byte=line_start,
                            sequence_start_byte=offset,
                        )
                        next_record_id += 1
                        source_stats["record_count"] += 1
                        continue
                    if current_record is None or not line:
                        continue
                    process_sequence_values(np.frombuffer(line, dtype=np.uint8), line_start)

        close_run(current_run)
        if current_run is not None and current_run.run_base_length > 0:
            source_stats["run_count"] += 1
        finish_record(int(stat.st_size))
        if progress_callback is not None:
            progress_callback(
                {
                    "file_id": file_id,
                    "file_count": len(fasta_files),
                    "rel_path": str(rel_path),
                    "size_bytes": int(stat.st_size),
                    "records": next_record_id,
                    "runs": next_run_id,
                    "anchors": anchor_count,
                    "processed_bytes": total_size,
                }
            )

    files_writer.close()
    records_writer.close()
    runs_writer.close()
    anchors_writer.close()

    stats = BuildStats(
        file_count=len(fasta_files),
        record_count=next_record_id,
        run_count=next_run_id,
        anchor_count=anchor_count,
        total_size_bytes=total_size,
        source_summary=source_summary,
    )
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "fasta_root": str(root),
        "index_dir": str(output),
        "anchor_stride": anchor_stride,
        "file_count": stats.file_count,
        "record_count": stats.record_count,
        "run_count": stats.run_count,
        "anchor_count": stats.anchor_count,
        "total_size_bytes": stats.total_size_bytes,
        "source_summary": stats.source_summary,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def load_manifest(index_dir: str | Path) -> dict[str, Any]:
    path = Path(index_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"index manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f"{path.name}.tmp.",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def _stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.tmp.", suffix=".npy", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        np.save(temp_path, array)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _arrow_column_to_numpy(table: pa.Table, name: str, dtype: np.dtype | type) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _stable_run_hash(run_ids: np.ndarray, split_seed: int) -> np.ndarray:
    values = run_ids.astype(np.uint64, copy=False) ^ np.uint64(split_seed)
    values ^= values >> np.uint64(33)
    values *= np.uint64(0xff51afd7ed558ccd)
    values ^= values >> np.uint64(33)
    values *= np.uint64(0xc4ceb9fe1a85ec53)
    values ^= values >> np.uint64(33)
    return values


def split_run_ids(
    run_ids: np.ndarray,
    *,
    split: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int = 0,
) -> np.ndarray:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: train, val, test")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")
    bucket_count = 1_000_000
    cut_train = int(round(train_ratio * bucket_count))
    cut_val = int(round((train_ratio + val_ratio) * bucket_count))
    hashed = _stable_run_hash(run_ids, split_seed)
    buckets = (hashed % np.uint64(bucket_count)).astype(np.int64)
    if split == "train":
        return buckets < cut_train
    if split == "val":
        return (buckets >= cut_train) & (buckets < cut_val)
    return buckets >= cut_val


@dataclass
class FastaIndexRuntimeCache:
    index_dir: Path
    cache_dir: Path
    fasta_root: Path
    file_paths: list[Path]
    file_sources: np.ndarray
    run_ids: np.ndarray
    run_file_ids: np.ndarray
    run_source_ids: np.ndarray
    run_start_file_offsets: np.ndarray
    run_lengths: np.ndarray
    anchor_run_ids: np.ndarray
    anchor_base_offsets: np.ndarray
    anchor_file_offsets: np.ndarray
    source_names: list[str]
    manifest: dict[str, Any]


def ensure_fasta_index_runtime_cache(index_dir: str | Path) -> Path:
    root = Path(index_dir)
    manifest = load_manifest(root)
    cache_dir = root / RUNTIME_CACHE_DIR_NAME
    metadata_path = cache_dir / "metadata.json"
    expected = {
        "schema_version": RUNTIME_CACHE_SCHEMA_VERSION,
        "index_schema_version": manifest.get("schema_version"),
        "file_count": manifest.get("file_count"),
        "run_count": manifest.get("run_count"),
        "anchor_count": manifest.get("anchor_count"),
        "total_size_bytes": manifest.get("total_size_bytes"),
    }
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        required = [
            "file_rel_paths.json",
            "source_names.json",
            "file_sources.npy",
            "run_ids.npy",
            "run_file_ids.npy",
            "run_source_ids.npy",
            "run_start_file_offsets.npy",
            "run_lengths.npy",
            "anchor_run_ids.npy",
            "anchor_base_offsets.npy",
            "anchor_file_offsets.npy",
        ]
        if existing == expected and all((cache_dir / name).exists() for name in required):
            return cache_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    files = pq.read_table(
        root / "files.parquet",
        columns=["file_id", "source", "rel_path"],
        use_threads=False,
    ).to_pydict()
    source_names = sorted(set(str(source) for source in files["source"]))
    source_to_id = {source: idx for idx, source in enumerate(source_names)}
    file_count = int(manifest["file_count"])
    file_rel_paths = [""] * file_count
    file_sources = np.empty((file_count,), dtype=np.int32)
    for file_id, source, rel_path in zip(files["file_id"], files["source"], files["rel_path"]):
        idx = int(file_id)
        file_rel_paths[idx] = str(rel_path)
        file_sources[idx] = source_to_id[str(source)]

    runs = pq.read_table(
        root / "acgt_runs.parquet",
        columns=["run_id", "file_id", "run_start_file_byte", "run_base_length"],
        use_threads=False,
    )
    run_ids = _arrow_column_to_numpy(runs, "run_id", np.int64)
    run_file_ids = _arrow_column_to_numpy(runs, "file_id", np.int64)
    run_start_file_offsets = _arrow_column_to_numpy(runs, "run_start_file_byte", np.int64)
    run_lengths = _arrow_column_to_numpy(runs, "run_base_length", np.int64)
    run_source_ids = file_sources[run_file_ids].astype(np.int32, copy=False)

    anchors = pq.read_table(
        root / "acgt_anchors.parquet",
        columns=["run_id", "run_base_offset", "file_byte_offset"],
        use_threads=False,
    )
    anchor_run_ids = _arrow_column_to_numpy(anchors, "run_id", np.int64)
    anchor_base_offsets = _arrow_column_to_numpy(anchors, "run_base_offset", np.int64)
    anchor_file_offsets = _arrow_column_to_numpy(anchors, "file_byte_offset", np.int64)

    _write_json_atomic(cache_dir / "file_rel_paths.json", {"paths": file_rel_paths})
    _write_json_atomic(cache_dir / "source_names.json", {"sources": source_names})
    _save_npy_atomic(cache_dir / "file_sources.npy", file_sources)
    _save_npy_atomic(cache_dir / "run_ids.npy", run_ids)
    _save_npy_atomic(cache_dir / "run_file_ids.npy", run_file_ids)
    _save_npy_atomic(cache_dir / "run_source_ids.npy", run_source_ids)
    _save_npy_atomic(cache_dir / "run_start_file_offsets.npy", run_start_file_offsets)
    _save_npy_atomic(cache_dir / "run_lengths.npy", run_lengths)
    _save_npy_atomic(cache_dir / "anchor_run_ids.npy", anchor_run_ids)
    _save_npy_atomic(cache_dir / "anchor_base_offsets.npy", anchor_base_offsets)
    _save_npy_atomic(cache_dir / "anchor_file_offsets.npy", anchor_file_offsets)
    _write_json_atomic(metadata_path, expected)
    return cache_dir


def wait_for_fasta_index_runtime_cache(index_dir: str | Path, *, timeout_seconds: int = 3600) -> Path:
    cache_dir = Path(index_dir) / RUNTIME_CACHE_DIR_NAME
    metadata_path = cache_dir / "metadata.json"
    started = time.time()
    while True:
        if metadata_path.exists():
            return cache_dir
        if time.time() - started > timeout_seconds:
            raise TimeoutError(f"timed out waiting for FASTA index runtime cache: {metadata_path}")
        time.sleep(1.0)


def load_fasta_index_runtime_cache(index_dir: str | Path, *, mmap_mode: str | None = "r") -> FastaIndexRuntimeCache:
    root = Path(index_dir)
    cache_dir = ensure_fasta_index_runtime_cache(root)
    manifest = load_manifest(root)
    rel_paths = json.loads((cache_dir / "file_rel_paths.json").read_text(encoding="utf-8"))["paths"]
    source_names = json.loads((cache_dir / "source_names.json").read_text(encoding="utf-8"))["sources"]
    fasta_root = Path(str(manifest["fasta_root"]))
    return FastaIndexRuntimeCache(
        index_dir=root,
        cache_dir=cache_dir,
        fasta_root=fasta_root,
        file_paths=[fasta_root / str(path) for path in rel_paths],
        file_sources=np.load(cache_dir / "file_sources.npy", mmap_mode=mmap_mode),
        run_ids=np.load(cache_dir / "run_ids.npy", mmap_mode=mmap_mode),
        run_file_ids=np.load(cache_dir / "run_file_ids.npy", mmap_mode=mmap_mode),
        run_source_ids=np.load(cache_dir / "run_source_ids.npy", mmap_mode=mmap_mode),
        run_start_file_offsets=np.load(cache_dir / "run_start_file_offsets.npy", mmap_mode=mmap_mode),
        run_lengths=np.load(cache_dir / "run_lengths.npy", mmap_mode=mmap_mode),
        anchor_run_ids=np.load(cache_dir / "anchor_run_ids.npy", mmap_mode=mmap_mode),
        anchor_base_offsets=np.load(cache_dir / "anchor_base_offsets.npy", mmap_mode=mmap_mode),
        anchor_file_offsets=np.load(cache_dir / "anchor_file_offsets.npy", mmap_mode=mmap_mode),
        source_names=[str(source) for source in source_names],
        manifest=manifest,
    )


def wait_or_build_fasta_index_runtime_cache(index_dir: str | Path, *, is_main_process: bool = True) -> Path:
    if is_main_process:
        return ensure_fasta_index_runtime_cache(index_dir)
    return wait_for_fasta_index_runtime_cache(index_dir)


class IndexedFastaFragmentSampler:
    def __init__(
        self,
        index_dir: str | Path,
        *,
        seq_length: int,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        self.index_dir = Path(index_dir)
        self.manifest = load_manifest(self.index_dir)
        if int(self.manifest.get("schema_version", -1)) != INDEX_SCHEMA_VERSION:
            raise ValueError(f"unsupported FASTA fragment index schema: {self.manifest.get('schema_version')!r}")
        self.fasta_root = Path(str(self.manifest["fasta_root"]))
        self.seq_length = seq_length

        files_table = pq.read_table(self.index_dir / "files.parquet", columns=["file_id", "rel_path"])
        files = files_table.to_pydict()
        self.file_paths = {int(file_id): self.fasta_root / str(rel) for file_id, rel in zip(files["file_id"], files["rel_path"])}

        runs_table = pq.read_table(
            self.index_dir / "acgt_runs.parquet",
            columns=["run_id", "file_id", "source", "run_base_length"],
        )
        runs = runs_table.to_pydict()
        run_lengths = np.asarray(runs["run_base_length"], dtype=np.int64)
        eligible_mask = run_lengths >= seq_length
        if not bool(np.any(eligible_mask)):
            raise ValueError("no indexed ACGT runs are long enough for seq_length")

        self.run_ids = np.asarray(runs["run_id"], dtype=np.int64)[eligible_mask]
        self.run_file_ids = np.asarray(runs["file_id"], dtype=np.int64)[eligible_mask]
        self.run_sources = np.asarray(runs["source"], dtype=object)[eligible_mask]
        self.run_lengths = run_lengths[eligible_mask]
        self.run_available = self.run_lengths - int(seq_length) + 1
        self.run_id_to_position = {int(run_id): idx for idx, run_id in enumerate(self.run_ids.tolist())}

        anchors_table = pq.read_table(
            self.index_dir / "acgt_anchors.parquet",
            columns=["run_id", "run_base_offset", "file_byte_offset"],
        )
        anchors = anchors_table.to_pydict()
        self.anchor_run_ids = np.asarray(anchors["run_id"], dtype=np.int64)
        self.anchor_base_offsets = np.asarray(anchors["run_base_offset"], dtype=np.int64)
        self.anchor_file_offsets = np.asarray(anchors["file_byte_offset"], dtype=np.int64)

        self.source_to_run_positions: dict[str, np.ndarray] = {}
        for source in sorted(set(self.run_sources.tolist())):
            self.source_to_run_positions[source] = np.flatnonzero(self.run_sources == source)
        self._configure_source_weights(source_weights)
        self._handles: dict[int, Any] = {}

    def _configure_source_weights(self, source_weights: dict[str, float] | None) -> None:
        if source_weights is None:
            self.source_names = np.array(["__all__"], dtype=object)
            self.source_weights = np.array([1.0], dtype=np.float64)
            self.source_positions = [np.arange(len(self.run_ids), dtype=np.int64)]
            return

        unknown = sorted(set(source_weights) - set(self.source_to_run_positions))
        if unknown:
            raise ValueError(f"source weights include unknown or ineligible sources: {', '.join(unknown)}")
        names: list[str] = []
        weights: list[float] = []
        positions: list[np.ndarray] = []
        for source, weight in sorted(source_weights.items()):
            value = float(weight)
            if value < 0:
                raise ValueError("source sampling weights must be non-negative")
            if value == 0:
                continue
            names.append(source)
            weights.append(value)
            positions.append(self.source_to_run_positions[source])
        if not weights or sum(weights) <= 0:
            raise ValueError("source sampling weights must sum to > 0")
        self.source_names = np.asarray(names, dtype=object)
        self.source_weights = np.asarray(weights, dtype=np.float64)
        self.source_weights = self.source_weights / self.source_weights.sum()
        self.source_positions = positions

    def summary(self) -> dict[str, Any]:
        eligible_by_source: dict[str, int] = {}
        bases_by_source: dict[str, int] = {}
        for source, positions in self.source_to_run_positions.items():
            eligible_by_source[source] = int(len(positions))
            bases_by_source[source] = int(self.run_available[positions].sum())
        return {
            "index_dir": str(self.index_dir),
            "fasta_root": str(self.fasta_root),
            "seq_length": self.seq_length,
            "eligible_run_count": int(len(self.run_ids)),
            "eligible_window_count": int(self.run_available.sum()),
            "eligible_runs_by_source": eligible_by_source,
            "eligible_windows_by_source": bases_by_source,
            "source_sampling_weights": {
                str(name): float(weight) for name, weight in zip(self.source_names.tolist(), self.source_weights.tolist())
            },
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _handle_for_file(self, file_id: int):
        handle = self._handles.get(file_id)
        if handle is None or handle.closed:
            path = self.file_paths[file_id]
            handle = path.open("rb")
            self._handles[file_id] = handle
        return handle

    def _choose_run_position(self, rng: random.Random) -> int:
        if len(self.source_positions) == 1 and self.source_names[0] == "__all__":
            positions = self.source_positions[0]
        else:
            source_index = rng.choices(range(len(self.source_positions)), weights=self.source_weights.tolist(), k=1)[0]
            positions = self.source_positions[source_index]
        weights = self.run_available[positions].astype(np.float64)
        local = rng.choices(range(len(positions)), weights=weights.tolist(), k=1)[0]
        return int(positions[local])

    def _anchor_for(self, run_id: int, base_offset: int) -> tuple[int, int]:
        start = int(np.searchsorted(self.anchor_run_ids, run_id, side="left"))
        end = int(np.searchsorted(self.anchor_run_ids, run_id, side="right"))
        if start == end:
            raise ValueError(f"run {run_id} has no anchors")
        local_offsets = self.anchor_base_offsets[start:end]
        local_index = int(np.searchsorted(local_offsets, base_offset, side="right")) - 1
        if local_index < 0:
            local_index = 0
        absolute = start + local_index
        return int(self.anchor_base_offsets[absolute]), int(self.anchor_file_offsets[absolute])

    def sample(self, *, seed: int | None = None, index: int | None = None) -> dict[str, Any]:
        rng_seed = seed if index is None else int(seed or 0) + int(index)
        rng = random.Random(rng_seed)
        run_position = self._choose_run_position(rng)
        run_id = int(self.run_ids[run_position])
        available = int(self.run_available[run_position])
        start_base = rng.randrange(available)
        anchor_base, anchor_file = self._anchor_for(run_id, start_base)
        skip_bases = start_base - anchor_base
        file_id = int(self.run_file_ids[run_position])
        handle = self._handle_for_file(file_id)
        handle.seek(anchor_file)

        tokens = bytearray()
        skipped = 0
        while len(tokens) < self.seq_length:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            for byte_value in chunk:
                upper = _to_upper(byte_value)
                token = int(_BASE_TO_TOKEN_BYTE[upper])
                if token:
                    if skipped < skip_bases:
                        skipped += 1
                    else:
                        tokens.append(token)
                        if len(tokens) >= self.seq_length:
                            break
                elif byte_value in (9, 10, 11, 12, 13, 32):
                    continue
                elif _byte_is_alpha(byte_value):
                    raise ValueError(f"encountered non-ACGT base while reading indexed run {run_id}")
        if len(tokens) != self.seq_length:
            raise ValueError(f"sampled only {len(tokens)} tokens from indexed run {run_id}, expected {self.seq_length}")
        return {
            "input_ids": torch.tensor(list(tokens), dtype=torch.long),
            "run_id": run_id,
            "file_id": file_id,
            "source": str(self.run_sources[run_position]),
            "start_base": int(start_base),
        }


class IndexedMegaDNAWindowDataset(Dataset):
    def __init__(
        self,
        *,
        index_dir: str | Path,
        seq_length: int,
        samples_per_epoch: int,
        seed: int,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        self.sampler = IndexedFastaFragmentSampler(index_dir, seq_length=seq_length, source_weights=source_weights)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be > 0")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.sampler.sample(seed=self.seed, index=int(index))
        return {"input_ids": sample["input_ids"]}

    def summary(self) -> dict[str, Any]:
        return self.sampler.summary()

    def __del__(self) -> None:
        try:
            self.sampler.close()
        except Exception:
            pass


class IndexedMegabyteWindowDataset(Dataset):
    def __init__(
        self,
        *,
        index_dir: str | Path,
        split: str,
        seq_length: int,
        token_merge_size: int,
        token_merge_alphabet: str,
        samples: int | None,
        seed: int,
        source_weights: dict[str, float] | None = None,
        source_loss_weights: dict[str, float] | None = None,
        pad_id: int | None = None,
        window_mode: str = "sliding_random",
        epoch_mode: str = "samples",
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        split_seed: int = 0,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if token_merge_size <= 0:
            raise ValueError("token_merge_size must be > 0")
        if samples is not None and samples <= 0:
            raise ValueError("samples must be > 0")
        if window_mode not in {"sliding_random", "nonoverlap_random"}:
            raise ValueError("window_mode must be one of: sliding_random, nonoverlap_random")
        if epoch_mode not in {"samples", "all_windows"}:
            raise ValueError("epoch_mode must be one of: samples, all_windows")
        if epoch_mode == "all_windows" and window_mode != "nonoverlap_random":
            raise ValueError("epoch_mode='all_windows' requires window_mode='nonoverlap_random'")
        if window_mode == "nonoverlap_random" and pad_id is None:
            raise ValueError("pad_id is required for nonoverlap_random indexed FASTA sampling")
        if window_mode != "sliding_random" and source_weights:
            raise ValueError("source_weights are supported only with sliding_random; use source_loss_weights")
        alphabet = normalize_alphabet(token_merge_alphabet)
        if set(alphabet) - {"A", "C", "G", "T", "N"}:
            raise ValueError("indexed_fasta Megabyte sampling supports only A/C/G/T/N alphabets")
        if not {"A", "C", "G", "T"}.issubset(set(alphabet)):
            raise ValueError("indexed_fasta Megabyte sampling requires A/C/G/T in token_merge_alphabet")

        self.cache = load_fasta_index_runtime_cache(index_dir)
        self.split = split
        self.seq_length = int(seq_length)
        self.token_merge_size = int(token_merge_size)
        self.base_length = self.seq_length * self.token_merge_size
        self.seed = int(seed)
        self.alphabet = alphabet
        self.pad_id = None if pad_id is None else int(pad_id)
        self.window_mode = window_mode
        self.epoch_mode = epoch_mode
        self.source_weights_config = dict(source_weights or {})
        self.source_loss_weights_config = dict(source_loss_weights or {})
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self._handles: dict[int, Any] = {}

        split_mask = split_run_ids(
            np.asarray(self.cache.run_ids),
            split=split,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
        )
        run_lengths = np.asarray(self.cache.run_lengths)
        if self.window_mode == "sliding_random":
            eligible_mask = run_lengths >= self.base_length
        else:
            eligible_mask = (run_lengths // self.token_merge_size) >= 1
        self.positions = np.flatnonzero(split_mask & eligible_mask).astype(np.int64, copy=False)
        if self.positions.size == 0:
            raise ValueError(f"no indexed FASTA runs are eligible for split={split!r} and seq_length={seq_length}")

        self.run_full_tokens = (run_lengths[self.positions] // self.token_merge_size).astype(np.int64, copy=False)
        if self.window_mode == "sliding_random":
            self.available = (self.run_full_tokens - self.seq_length + 1).astype(np.int64, copy=False)
            self.nonoverlap_padded_windows = 0
        else:
            self.available = ((self.run_full_tokens + self.seq_length - 1) // self.seq_length).astype(np.int64, copy=False)
            self.nonoverlap_padded_windows = int(np.count_nonzero(self.run_full_tokens % self.seq_length))
        self.total_candidate_windows = int(self.available.sum())
        if self.total_candidate_windows <= 0:
            raise ValueError(f"no indexed FASTA windows are eligible for split={split!r} and seq_length={seq_length}")
        self.available_cumsum = np.cumsum(self.available, dtype=np.int64)
        self.samples = self.total_candidate_windows if samples is None else int(samples)
        self.source_ids = np.asarray(self.cache.run_source_ids)[self.positions].astype(np.int32, copy=False)
        self._configure_sampling(source_weights)
        self._configure_loss_weights(source_loss_weights)

        if self.token_merge_size > 1:
            self._digit_lookup = np.full(256, -1, dtype=np.int16)
            for index, ch in enumerate(self.alphabet):
                self._digit_lookup[ord(ch)] = index
                self._digit_lookup[ord(ch.lower())] = index
            self._merge_weights = np.array(
                [len(self.alphabet) ** power for power in range(self.token_merge_size - 1, -1, -1)],
                dtype=np.uint64,
            )
        else:
            self._digit_lookup = None
            self._merge_weights = None

    def _configure_sampling(self, source_weights: dict[str, float] | None) -> None:
        if source_weights is None or not source_weights:
            cumsum = np.cumsum(self.available, dtype=np.int64)
            if int(cumsum[-1]) <= 0:
                raise ValueError("indexed FASTA sampling weights must sum to > 0")
            self.source_names = ["__all__"]
            self.source_probabilities = np.array([1.0], dtype=np.float64)
            self.group_positions = [np.arange(self.positions.size, dtype=np.int64)]
            self.group_cumsums = [cumsum]
            return

        source_to_id = {source: idx for idx, source in enumerate(self.cache.source_names)}
        unknown = sorted(set(source_weights) - set(source_to_id))
        if unknown:
            raise ValueError(f"source weights include unknown sources: {', '.join(unknown)}")
        names: list[str] = []
        probabilities: list[float] = []
        group_positions: list[np.ndarray] = []
        group_cumsums: list[np.ndarray] = []
        for source, raw_weight in sorted(source_weights.items()):
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("source sampling weights must be non-negative")
            if weight == 0:
                continue
            source_id = source_to_id[source]
            local_positions = np.flatnonzero(self.source_ids == source_id).astype(np.int64, copy=False)
            if local_positions.size == 0:
                raise ValueError(f"source {source!r} has no eligible runs for split={self.split!r}")
            cumsum = np.cumsum(self.available[local_positions], dtype=np.int64)
            if int(cumsum[-1]) <= 0:
                raise ValueError(f"source {source!r} has no eligible windows for split={self.split!r}")
            names.append(source)
            probabilities.append(weight)
            group_positions.append(local_positions)
            group_cumsums.append(cumsum)
        if not probabilities or sum(probabilities) <= 0:
            raise ValueError("source sampling weights must sum to > 0")
        self.source_names = names
        self.source_probabilities = np.asarray(probabilities, dtype=np.float64)
        self.source_probabilities /= self.source_probabilities.sum()
        self.group_positions = group_positions
        self.group_cumsums = group_cumsums

    def _configure_loss_weights(self, source_loss_weights: dict[str, float] | None) -> None:
        self.loss_weights_by_source_id = np.ones((len(self.cache.source_names),), dtype=np.float32)
        self.loss_weight_summary: dict[str, float] = {}
        if source_loss_weights is None or not source_loss_weights:
            return

        source_to_id = {source: idx for idx, source in enumerate(self.cache.source_names)}
        unknown = sorted(set(source_loss_weights) - set(source_to_id))
        if unknown:
            raise ValueError(f"source loss weights include unknown sources: {', '.join(unknown)}")
        raw_total = float(sum(float(value) for value in source_loss_weights.values()))
        if raw_total <= 0:
            raise ValueError("source loss weights must sum to > 0")

        if self.window_mode == "sliding_random":
            source_masses = np.bincount(
                self.source_ids,
                weights=(self.available.astype(np.float64) * float(self.seq_length)),
                minlength=len(self.cache.source_names),
            )
        else:
            source_masses = np.bincount(
                self.source_ids,
                weights=self.run_full_tokens.astype(np.float64),
                minlength=len(self.cache.source_names),
            )
        total_mass = float(source_masses.sum())
        if total_mass <= 0:
            raise ValueError("indexed FASTA source loss weighting requires positive token mass")

        multipliers = np.zeros((len(self.cache.source_names),), dtype=np.float32)
        for source, raw_weight in source_loss_weights.items():
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("source loss weights must be non-negative")
            if weight == 0:
                continue
            source_id = source_to_id[source]
            actual_fraction = float(source_masses[source_id]) / total_mass
            if actual_fraction <= 0:
                raise ValueError(f"source {source!r} has no token mass for split={self.split!r}")
            target_fraction = weight / raw_total
            multipliers[source_id] = np.float32(target_fraction / actual_fraction)

        if not np.any(multipliers > 0):
            raise ValueError("source loss weights must include at least one positive eligible source")
        self.loss_weights_by_source_id = multipliers
        self.loss_weight_summary = {
            self.cache.source_names[source_id]: float(multiplier)
            for source_id, multiplier in enumerate(multipliers.tolist())
            if multiplier > 0
        }

    def __len__(self) -> int:
        return self.samples

    def summary(self) -> dict[str, Any]:
        eligible_by_source: dict[str, int] = {}
        windows_by_source: dict[str, int] = {}
        for source_id, source_name in enumerate(self.cache.source_names):
            local = np.flatnonzero(self.source_ids == source_id)
            if local.size == 0:
                continue
            eligible_by_source[source_name] = int(local.size)
            windows_by_source[source_name] = int(self.available[local].sum())
        return {
            "source_mode": "indexed_fasta",
            "index_dir": str(self.cache.index_dir),
            "runtime_cache_dir": str(self.cache.cache_dir),
            "split": self.split,
            "split_seed": self.split_seed,
            "window_mode": self.window_mode,
            "epoch_mode": self.epoch_mode,
            "seq_length": self.seq_length,
            "token_merge_size": self.token_merge_size,
            "base_length": self.base_length,
            "samples": self.samples,
            "eligible_run_count": int(self.positions.size),
            "eligible_window_count": self.total_candidate_windows,
            "candidate_window_count": self.total_candidate_windows,
            "padded_window_count": self.nonoverlap_padded_windows,
            "eligible_runs_by_source": eligible_by_source,
            "eligible_windows_by_source": windows_by_source,
            "source_sampling_weights": {
                name: float(weight) for name, weight in zip(self.source_names, self.source_probabilities.tolist())
            },
            "source_loss_weights": dict(self.loss_weight_summary),
        }

    def _handle_for_file(self, file_id: int):
        handle = self._handles.get(file_id)
        if handle is None or handle.closed:
            handle = self.cache.file_paths[file_id].open("rb")
            self._handles[file_id] = handle
        return handle

    def _choose_local_position(self, rng: random.Random) -> tuple[int, int]:
        group_index = rng.choices(range(len(self.group_positions)), weights=self.source_probabilities.tolist(), k=1)[0]
        cumsum = self.group_cumsums[group_index]
        ticket = rng.randrange(int(cumsum[-1]))
        offset = int(np.searchsorted(cumsum, ticket + 1, side="left"))
        group_position = int(self.group_positions[group_index][offset])
        previous = int(cumsum[offset - 1]) if offset > 0 else 0
        token_offset = ticket - previous
        return group_position, token_offset

    def _permuted_window_ticket(self, index: int) -> int:
        total = self.total_candidate_windows
        if total <= 1:
            return 0
        cycle, local_index = divmod(int(index), total)
        rng = random.Random(self.seed + cycle * 1_000_003)
        step = rng.randrange(1, total)
        while math.gcd(step, total) != 1:
            step = rng.randrange(1, total)
        offset = rng.randrange(total)
        return (local_index * step + offset) % total

    def _choose_nonoverlap_window(self, index: int) -> tuple[int, int, int]:
        ticket = self._permuted_window_ticket(index)
        local_position = int(np.searchsorted(self.available_cumsum, ticket + 1, side="left"))
        previous = int(self.available_cumsum[local_position - 1]) if local_position > 0 else 0
        window_index = ticket - previous
        start_token = int(window_index) * self.seq_length
        full_tokens = int(self.run_full_tokens[local_position])
        tokens_to_read = min(self.seq_length, full_tokens - start_token)
        if tokens_to_read <= 0:
            raise ValueError("internal error: selected empty nonoverlap indexed FASTA window")
        return local_position, start_token, tokens_to_read

    def _anchor_for(
        self,
        run_id: int,
        base_offset: int,
        *,
        run_start_file_offset: int | None = None,
    ) -> tuple[int, int]:
        if base_offset == 0 and run_start_file_offset is not None:
            return 0, int(run_start_file_offset)
        start = int(np.searchsorted(self.cache.anchor_run_ids, run_id, side="left"))
        end = int(np.searchsorted(self.cache.anchor_run_ids, run_id, side="right"))
        if start == end:
            raise ValueError(f"run {run_id} has no anchors")
        local_offsets = self.cache.anchor_base_offsets[start:end]
        local_index = int(np.searchsorted(local_offsets, base_offset, side="right")) - 1
        if local_index < 0:
            local_index = 0
        absolute = start + local_index
        return int(self.cache.anchor_base_offsets[absolute]), int(self.cache.anchor_file_offsets[absolute])

    def _read_bases(
        self,
        *,
        run_id: int,
        file_id: int,
        base_start: int,
        target_bases: int,
        run_start_file_offset: int | None = None,
    ) -> bytes:
        anchor_base, anchor_file = self._anchor_for(
            run_id,
            base_start,
            run_start_file_offset=run_start_file_offset,
        )
        skip_bases = base_start - anchor_base
        handle = self._handle_for_file(file_id)
        handle.seek(anchor_file)

        bases = bytearray()
        skipped = 0
        read_size = min(1024 * 1024, max(4096, int((target_bases + skip_bases) * 2) + 256))
        while len(bases) < target_bases:
            remaining = target_bases - len(bases)
            chunk = handle.read(min(read_size, max(4096, remaining * 2 + 256)))
            if not chunk:
                break
            for byte_value in chunk:
                upper = _to_upper(byte_value)
                if upper in (ord("A"), ord("C"), ord("G"), ord("T")):
                    if skipped < skip_bases:
                        skipped += 1
                    else:
                        bases.append(upper)
                        if len(bases) >= target_bases:
                            break
                elif byte_value in (9, 10, 11, 12, 13, 32):
                    continue
                elif _byte_is_alpha(byte_value):
                    raise ValueError(f"encountered non-ACGT base while reading indexed run {run_id}")
        if len(bases) != target_bases:
            raise ValueError(f"sampled only {len(bases)} bases from indexed run {run_id}, expected {target_bases}")
        return bytes(bases)

    def _tokenize_bases(self, bases: bytes, *, pad_to_seq_length: bool = False) -> torch.Tensor:
        if self.token_merge_size <= 1:
            token_ids = np.frombuffer(bases, dtype=np.uint8).astype(np.int64, copy=True)
        else:
            if self._digit_lookup is None or self._merge_weights is None:
                raise ValueError("internal error: token merge lookup was not initialized")
            raw = np.frombuffer(bases, dtype=np.uint8)
            digits = self._digit_lookup[raw]
            if np.any(digits < 0):
                raise ValueError("indexed FASTA sample contained a base outside token_merge_alphabet")
            full_digit_count = (digits.shape[0] // self.token_merge_size) * self.token_merge_size
            digits = digits[:full_digit_count]
            merged = digits.reshape(-1, self.token_merge_size).astype(np.uint64, copy=False)
            token_ids = (merged * self._merge_weights).sum(axis=1, dtype=np.uint64).astype(np.int64, copy=False)
        if pad_to_seq_length:
            if self.pad_id is None:
                raise ValueError("pad_id is required to pad indexed FASTA windows")
            padded = np.full((self.seq_length,), self.pad_id, dtype=np.int64)
            usable = min(token_ids.shape[0], self.seq_length)
            padded[:usable] = token_ids[:usable]
            token_ids = padded
        return torch.as_tensor(token_ids.copy(), dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.window_mode == "sliding_random":
            rng = random.Random(self.seed + int(index))
            local_position, token_offset = self._choose_local_position(rng)
            target_bases = self.base_length
            pad_to_seq_length = False
        else:
            local_position, token_offset, tokens_to_read = self._choose_nonoverlap_window(int(index))
            target_bases = tokens_to_read * self.token_merge_size
            pad_to_seq_length = True
        run_position = int(self.positions[local_position])
        run_id = int(self.cache.run_ids[run_position])
        file_id = int(self.cache.run_file_ids[run_position])
        base_start = int(token_offset) * self.token_merge_size
        bases = self._read_bases(run_id=run_id, file_id=file_id, base_start=base_start, target_bases=target_bases)
        source_id = int(self.source_ids[local_position])
        item: dict[str, torch.Tensor] = {
            "input_ids": self._tokenize_bases(bases, pad_to_seq_length=pad_to_seq_length),
            "source_id": torch.tensor(source_id, dtype=torch.long),
        }
        if self.source_loss_weights_config:
            item["loss_weight"] = torch.tensor(float(self.loss_weights_by_source_id[source_id]), dtype=torch.float32)
        return item

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class CachedIndexedMegabyteEvalDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        source_loss_weights: dict[str, float] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"indexed FASTA eval cache metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(self.metadata.get("schema_version", -1)) != EVAL_CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported indexed FASTA eval cache schema: {self.metadata.get('schema_version')!r}")
        self.input_ids = np.load(self.cache_dir / "input_ids.npy", mmap_mode="r")
        self.valid_tokens = np.load(self.cache_dir / "valid_tokens.npy", mmap_mode="r")
        self.source_ids = np.load(self.cache_dir / "source_ids.npy", mmap_mode="r")
        self.run_ids = np.load(self.cache_dir / "run_ids.npy", mmap_mode="r")
        self.run_window_indices = np.load(self.cache_dir / "run_window_indices.npy", mmap_mode="r")
        self.source_names = [str(source) for source in self.metadata["source_names"]]
        self.source_loss_weights_config = dict(source_loss_weights or {})
        self.loss_weights_by_source_id = np.ones((len(self.source_names),), dtype=np.float32)
        self.loss_weight_summary: dict[str, float] = {}
        self._configure_loss_weights(source_loss_weights)

    def _configure_loss_weights(self, source_loss_weights: dict[str, float] | None) -> None:
        if source_loss_weights is None or not source_loss_weights:
            return
        source_to_id = {source: idx for idx, source in enumerate(self.source_names)}
        unknown = sorted(set(source_loss_weights) - set(source_to_id))
        if unknown:
            raise ValueError(f"source loss weights include unknown sources: {', '.join(unknown)}")
        raw_total = float(sum(float(value) for value in source_loss_weights.values()))
        if raw_total <= 0:
            raise ValueError("source loss weights must sum to > 0")
        token_masses = np.bincount(
            np.asarray(self.source_ids, dtype=np.int64),
            weights=np.asarray(self.valid_tokens, dtype=np.float64),
            minlength=len(self.source_names),
        )
        total_mass = float(token_masses.sum())
        if total_mass <= 0:
            raise ValueError("indexed FASTA eval cache source loss weighting requires positive token mass")
        multipliers = np.zeros((len(self.source_names),), dtype=np.float32)
        for source, raw_weight in source_loss_weights.items():
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("source loss weights must be non-negative")
            if weight == 0:
                continue
            source_id = source_to_id[source]
            actual_fraction = float(token_masses[source_id]) / total_mass
            if actual_fraction <= 0:
                raise ValueError(f"source {source!r} has no token mass in indexed FASTA eval cache")
            multipliers[source_id] = np.float32((weight / raw_total) / actual_fraction)
        if not np.any(multipliers > 0):
            raise ValueError("source loss weights must include at least one positive cached source")
        self.loss_weights_by_source_id = multipliers
        self.loss_weight_summary = {
            self.source_names[source_id]: float(multiplier)
            for source_id, multiplier in enumerate(multipliers.tolist())
            if multiplier > 0
        }

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_id = int(self.source_ids[index])
        item: dict[str, torch.Tensor] = {
            "input_ids": torch.as_tensor(np.asarray(self.input_ids[index], dtype=np.int64).copy(), dtype=torch.long),
            "source_id": torch.tensor(source_id, dtype=torch.long),
        }
        if self.source_loss_weights_config:
            item["loss_weight"] = torch.tensor(float(self.loss_weights_by_source_id[source_id]), dtype=torch.float32)
        return item

    def summary(self) -> dict[str, Any]:
        source_counts = np.bincount(
            np.asarray(self.source_ids, dtype=np.int64),
            minlength=len(self.source_names),
        )
        source_tokens = np.bincount(
            np.asarray(self.source_ids, dtype=np.int64),
            weights=np.asarray(self.valid_tokens, dtype=np.float64),
            minlength=len(self.source_names),
        )
        return {
            **dict(self.metadata),
            "cache_dir": str(self.cache_dir),
            "samples": len(self),
            "source_counts": {
                self.source_names[idx]: int(count)
                for idx, count in enumerate(source_counts.tolist())
                if count > 0
            },
            "source_valid_tokens": {
                self.source_names[idx]: float(count)
                for idx, count in enumerate(source_tokens.tolist())
                if count > 0
            },
            "source_loss_weights": dict(self.loss_weight_summary),
        }


def _eval_cache_key_payload(
    *,
    index_dir: str | Path,
    split: str,
    samples: int,
    seq_length: int,
    token_merge_size: int,
    token_merge_alphabet: str,
    pad_id: int,
    source_weights: dict[str, float] | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    eval_seed: int,
) -> dict[str, Any]:
    manifest = load_manifest(index_dir)
    manifest_digest = _stable_json_hash(manifest)
    return {
        "schema_version": EVAL_CACHE_SCHEMA_VERSION,
        "index_manifest_digest": manifest_digest,
        "index_dir": str(Path(index_dir).resolve()),
        "split": split,
        "samples": int(samples),
        "seq_length": int(seq_length),
        "token_merge_size": int(token_merge_size),
        "token_merge_alphabet": normalize_alphabet(token_merge_alphabet),
        "pad_id": int(pad_id),
        "source_weights": {str(k): float(v) for k, v in sorted(dict(source_weights or {}).items())},
        "source_sampling_strategy": EVAL_CACHE_SOURCE_SAMPLING_STRATEGY,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "split_seed": int(split_seed),
        "eval_seed": int(eval_seed),
    }


def indexed_eval_cache_path(
    *,
    cache_root: str | Path,
    key_payload: dict[str, Any],
) -> Path:
    digest = _stable_json_hash(key_payload)
    split = str(key_payload["split"])
    return Path(cache_root) / f"{split}_{digest[:20]}"


def prepare_indexed_megabyte_eval_cache(
    *,
    index_dir: str | Path,
    cache_root: str | Path,
    split: str,
    samples: int,
    seq_length: int,
    token_merge_size: int,
    token_merge_alphabet: str,
    pad_id: int,
    source_weights: dict[str, float] | None = None,
    source_loss_weights: dict[str, float] | None = None,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    split_seed: int = 0,
    eval_seed: int = 0,
    mode: str = "reuse",
    is_main_process: bool = True,
    wait_timeout_seconds: int = 3600,
) -> CachedIndexedMegabyteEvalDataset:
    if mode not in {"reuse", "refresh"}:
        raise ValueError("indexed eval cache mode must be one of: reuse, refresh")
    if samples <= 0:
        raise ValueError("samples must be > 0")
    key_payload = _eval_cache_key_payload(
        index_dir=index_dir,
        split=split,
        samples=samples,
        seq_length=seq_length,
        token_merge_size=token_merge_size,
        token_merge_alphabet=token_merge_alphabet,
        pad_id=pad_id,
        source_weights=source_weights,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        eval_seed=eval_seed,
    )
    cache_dir = indexed_eval_cache_path(cache_root=cache_root, key_payload=key_payload)
    metadata_path = cache_dir / "metadata.json"
    required = [
        "input_ids.npy",
        "valid_tokens.npy",
        "source_ids.npy",
        "run_ids.npy",
        "run_window_indices.npy",
        "metadata.json",
    ]
    cache_existed = metadata_path.exists() and all((cache_dir / name).exists() for name in required)
    cache_hit = bool(mode == "reuse" and cache_existed)
    if is_main_process:
        if mode == "refresh" and cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_hit = False
        if not (metadata_path.exists() and all((cache_dir / name).exists() for name in required)):
            _build_indexed_megabyte_eval_cache(
                index_dir=index_dir,
                cache_dir=cache_dir,
                key_payload=key_payload,
                source_weights=source_weights,
            )
    else:
        started = time.time()
        while not (metadata_path.exists() and all((cache_dir / name).exists() for name in required)):
            if time.time() - started > wait_timeout_seconds:
                raise TimeoutError(f"timed out waiting for indexed FASTA eval cache: {cache_dir}")
            time.sleep(1.0)
    dataset = CachedIndexedMegabyteEvalDataset(cache_dir, source_loss_weights=source_loss_weights)
    dataset.metadata["cache_hit"] = cache_hit
    return dataset


def _build_indexed_megabyte_eval_cache(
    *,
    index_dir: str | Path,
    cache_dir: Path,
    key_payload: dict[str, Any],
    source_weights: dict[str, float] | None,
) -> None:
    tmp_dir = cache_dir.parent / f"{cache_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    try:
        reader = IndexedMegabyteWindowDataset(
            index_dir=index_dir,
            split=str(key_payload["split"]),
            seq_length=int(key_payload["seq_length"]),
            token_merge_size=int(key_payload["token_merge_size"]),
            token_merge_alphabet=str(key_payload["token_merge_alphabet"]),
            samples=1,
            seed=int(key_payload["eval_seed"]),
            source_weights=None,
            source_loss_weights=None,
            pad_id=int(key_payload["pad_id"]),
            window_mode="nonoverlap_random",
            epoch_mode="samples",
            train_ratio=float(key_payload["train_ratio"]),
            val_ratio=float(key_payload["val_ratio"]),
            test_ratio=float(key_payload["test_ratio"]),
            split_seed=int(key_payload["split_seed"]),
        )
        source_names = list(reader.cache.source_names)
        source_to_id = {source: idx for idx, source in enumerate(source_names)}
        available_by_source = np.bincount(
            reader.source_ids.astype(np.int64, copy=False),
            weights=reader.available.astype(np.float64, copy=False),
            minlength=len(source_names),
        )
        if source_weights:
            unknown = sorted(set(source_weights) - set(source_to_id))
            if unknown:
                raise ValueError(f"source weights include unknown sources: {', '.join(unknown)}")
            active_names: list[str] = []
            active_ids: list[int] = []
            raw_probs: list[float] = []
            for source, raw_weight in sorted(source_weights.items()):
                weight = float(raw_weight)
                if weight < 0:
                    raise ValueError("source sampling weights must be non-negative")
                if weight == 0:
                    continue
                source_id = source_to_id[source]
                if available_by_source[source_id] <= 0:
                    raise ValueError(f"source {source!r} has no eligible eval windows for split={key_payload['split']!r}")
                active_names.append(source)
                active_ids.append(source_id)
                raw_probs.append(weight)
            if not raw_probs or sum(raw_probs) <= 0:
                raise ValueError("source sampling weights must sum to > 0")
            probabilities = np.asarray(raw_probs, dtype=np.float64)
            probabilities /= probabilities.sum()
        else:
            active_ids = np.flatnonzero(available_by_source > 0)
            if active_ids.size == 0:
                raise ValueError(f"no eligible eval windows for split={key_payload['split']!r}")
            active_names = [source_names[int(source_id)] for source_id in active_ids.tolist()]
            active_ids = [int(source_id) for source_id in active_ids.tolist()]
            probabilities = available_by_source[active_ids].astype(np.float64, copy=False)
            probabilities /= probabilities.sum()

        samples = int(key_payload["samples"])
        rng = np.random.default_rng(int(key_payload["eval_seed"]))
        input_ids = np.empty((samples, int(key_payload["seq_length"])), dtype=np.int64)
        valid_tokens = np.empty((samples,), dtype=np.int32)
        source_ids = np.empty((samples,), dtype=np.int32)
        run_ids = np.empty((samples,), dtype=np.int64)
        run_window_indices = np.empty((samples,), dtype=np.int64)
        selected_source_indices = rng.choice(len(active_ids), size=samples, replace=True, p=probabilities)
        source_counts = {source: 0 for source in active_names}
        source_total_windows: dict[str, int] = {}
        source_candidates: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        for source, source_id in zip(active_names, active_ids):
            local_positions = np.flatnonzero(reader.source_ids == source_id).astype(np.int64, copy=False)
            cumsum = np.cumsum(reader.available[local_positions], dtype=np.int64)
            total_windows = int(cumsum[-1])
            source_total_windows[source] = total_windows
            source_candidates[source_id] = (local_positions, cumsum, total_windows)
        seen_windows: set[tuple[int, int, int]] = set()
        repeat_count = 0
        for sample_index, source_index in enumerate(selected_source_indices.tolist()):
            source_id = int(active_ids[int(source_index)])
            source_name = source_names[source_id]
            local_positions, cumsum, total_windows = source_candidates[source_id]
            ticket = int(rng.integers(total_windows))
            offset = int(np.searchsorted(cumsum, ticket + 1, side="left"))
            previous = int(cumsum[offset - 1]) if offset > 0 else 0
            local_position = int(local_positions[offset])
            window_index = ticket - previous
            start_token = window_index * reader.seq_length
            full_tokens = int(reader.run_full_tokens[local_position])
            tokens_to_read = min(reader.seq_length, full_tokens - start_token)
            if tokens_to_read <= 0:
                raise ValueError("internal error: selected empty indexed FASTA eval window")
            run_position = int(reader.positions[local_position])
            run_id = int(reader.cache.run_ids[run_position])
            file_id = int(reader.cache.run_file_ids[run_position])
            window_key = (source_id, run_id, window_index)
            if window_key in seen_windows:
                repeat_count += 1
            else:
                seen_windows.add(window_key)
            base_start = start_token * reader.token_merge_size
            target_bases = tokens_to_read * reader.token_merge_size
            bases = reader._read_bases(
                run_id=run_id,
                file_id=file_id,
                base_start=base_start,
                target_bases=target_bases,
                run_start_file_offset=int(reader.cache.run_start_file_offsets[run_position]),
            )
            input_ids[sample_index] = reader._tokenize_bases(bases, pad_to_seq_length=True).numpy()
            valid_tokens[sample_index] = tokens_to_read
            source_ids[sample_index] = source_id
            run_ids[sample_index] = run_id
            run_window_indices[sample_index] = window_index
            source_counts[source_name] += 1

        _save_npy_atomic(tmp_dir / "input_ids.npy", input_ids)
        _save_npy_atomic(tmp_dir / "valid_tokens.npy", valid_tokens)
        _save_npy_atomic(tmp_dir / "source_ids.npy", source_ids)
        _save_npy_atomic(tmp_dir / "run_ids.npy", run_ids)
        _save_npy_atomic(tmp_dir / "run_window_indices.npy", run_window_indices)
        metadata = {
            **key_payload,
            "source_names": source_names,
            "cache_key": _stable_json_hash(key_payload),
            "cache_hit": False,
            "sample_counts_by_source": source_counts,
            "eligible_windows_by_source": source_total_windows,
            "repeat_count": int(repeat_count),
            "source_sampling_strategy": EVAL_CACHE_SOURCE_SAMPLING_STRATEGY,
            "cache_layout": "token_windows_v1",
        }
        _write_json_atomic(tmp_dir / "metadata.json", metadata)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(tmp_dir, cache_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        try:
            reader.close()  # type: ignore[name-defined]
        except Exception:
            pass


class IndexedMegabyteFileStreamDataset(IterableDataset):
    def __init__(
        self,
        *,
        index_dir: str | Path,
        split: str,
        seq_length: int,
        token_merge_size: int,
        token_merge_alphabet: str,
        samples: int | None,
        seed: int,
        source_loss_weights: dict[str, float] | None = None,
        pad_id: int | None = None,
        file_stream_windows: int = 8192,
        file_shuffle_buffer_windows: int = 8192,
        file_stream_order_seed: int = 0,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        split_seed: int = 0,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if token_merge_size <= 0:
            raise ValueError("token_merge_size must be > 0")
        if samples is not None and samples <= 0:
            raise ValueError("samples must be > 0")
        if pad_id is None:
            raise ValueError("pad_id is required for nonoverlap_file_stream indexed FASTA sampling")
        if file_stream_windows <= 0:
            raise ValueError("file_stream_windows must be > 0")
        if file_shuffle_buffer_windows < 0:
            raise ValueError("file_shuffle_buffer_windows must be >= 0")
        if ddp_rank < 0 or ddp_world_size <= 0 or ddp_rank >= ddp_world_size:
            raise ValueError("invalid DDP rank/world size for indexed FASTA file stream")
        alphabet = normalize_alphabet(token_merge_alphabet)
        if set(alphabet) - {"A", "C", "G", "T", "N"}:
            raise ValueError("indexed_fasta Megabyte sampling supports only A/C/G/T/N alphabets")
        if not {"A", "C", "G", "T"}.issubset(set(alphabet)):
            raise ValueError("indexed_fasta Megabyte sampling requires A/C/G/T in token_merge_alphabet")

        self.cache = load_fasta_index_runtime_cache(index_dir)
        self.split = split
        self.seq_length = int(seq_length)
        self.token_merge_size = int(token_merge_size)
        self.base_length = self.seq_length * self.token_merge_size
        self.seed = int(seed)
        self.alphabet = alphabet
        self.pad_id = int(pad_id)
        self.file_stream_windows = int(file_stream_windows)
        self.file_shuffle_buffer_windows = int(file_shuffle_buffer_windows)
        self.file_stream_order_seed = int(file_stream_order_seed)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self._handles: dict[int, Any] = {}
        self.source_loss_weights_config = dict(source_loss_weights or {})

        split_mask = split_run_ids(
            np.asarray(self.cache.run_ids),
            split=split,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
        )
        run_lengths = np.asarray(self.cache.run_lengths)
        eligible_mask = (run_lengths // self.token_merge_size) >= 1
        self.positions = np.flatnonzero(split_mask & eligible_mask).astype(np.int64, copy=False)
        if self.positions.size == 0:
            raise ValueError(f"no indexed FASTA runs are eligible for split={split!r} and seq_length={seq_length}")

        self.run_full_tokens = (run_lengths[self.positions] // self.token_merge_size).astype(np.int64, copy=False)
        self.available = ((self.run_full_tokens + self.seq_length - 1) // self.seq_length).astype(np.int64, copy=False)
        self.total_candidate_windows = int(self.available.sum())
        if self.total_candidate_windows <= 0:
            raise ValueError(f"no indexed FASTA windows are eligible for split={split!r} and seq_length={seq_length}")
        self.samples = self.total_candidate_windows if samples is None else int(samples)
        self.nonoverlap_padded_windows = int(np.count_nonzero(self.run_full_tokens % self.seq_length))
        self.source_ids = np.asarray(self.cache.run_source_ids)[self.positions].astype(np.int32, copy=False)
        self.file_ids = np.asarray(self.cache.run_file_ids)[self.positions].astype(np.int64, copy=False)
        self.run_start_file_offsets = np.asarray(self.cache.run_start_file_offsets)[self.positions].astype(
            np.int64,
            copy=False,
        )
        self.available_cumsum = np.cumsum(self.available, dtype=np.int64)
        self._build_stream_units()
        self._configure_loss_weights(source_loss_weights)

        if self.token_merge_size > 1:
            self._digit_lookup = np.full(256, -1, dtype=np.int16)
            for index, ch in enumerate(self.alphabet):
                self._digit_lookup[ord(ch)] = index
                self._digit_lookup[ord(ch.lower())] = index
            self._merge_weights = np.array(
                [len(self.alphabet) ** power for power in range(self.token_merge_size - 1, -1, -1)],
                dtype=np.uint64,
            )
        else:
            self._digit_lookup = None
            self._merge_weights = None

    def _build_stream_units(self) -> None:
        file_change = np.empty((self.file_ids.shape[0],), dtype=bool)
        file_change[0] = True
        file_change[1:] = self.file_ids[1:] != self.file_ids[:-1]
        self.file_run_starts = np.flatnonzero(file_change).astype(np.int64, copy=False)
        self.file_run_ends = np.empty_like(self.file_run_starts)
        self.file_run_ends[:-1] = self.file_run_starts[1:]
        self.file_run_ends[-1] = self.file_ids.shape[0]
        self.file_window_starts = np.empty_like(self.file_run_starts)
        self.file_window_starts[0] = 0
        if self.file_run_starts.shape[0] > 1:
            self.file_window_starts[1:] = self.available_cumsum[self.file_run_starts[1:] - 1]
        file_window_ends = self.available_cumsum[self.file_run_ends - 1]
        self.file_window_counts = file_window_ends - self.file_window_starts
        self.stream_file_count = int(self.file_run_starts.shape[0])

        self.file_unit_counts = (
            (self.file_window_counts + self.file_stream_windows - 1) // self.file_stream_windows
        ).astype(np.int64, copy=False)
        self.file_unit_starts = np.empty_like(self.file_unit_counts)
        self.file_unit_starts[0] = 0
        if self.file_unit_counts.shape[0] > 1:
            self.file_unit_starts[1:] = np.cumsum(self.file_unit_counts[:-1], dtype=np.int64)
        self.stream_unit_count = int(self.file_unit_counts.sum())
        if self.stream_unit_count <= 0:
            raise ValueError("indexed FASTA file stream produced no stream units")

    def _configure_loss_weights(self, source_loss_weights: dict[str, float] | None) -> None:
        self.loss_weights_by_source_id = np.ones((len(self.cache.source_names),), dtype=np.float32)
        self.loss_weight_summary: dict[str, float] = {}
        if source_loss_weights is None or not source_loss_weights:
            return

        source_to_id = {source: idx for idx, source in enumerate(self.cache.source_names)}
        unknown = sorted(set(source_loss_weights) - set(source_to_id))
        if unknown:
            raise ValueError(f"source loss weights include unknown sources: {', '.join(unknown)}")
        raw_total = float(sum(float(value) for value in source_loss_weights.values()))
        if raw_total <= 0:
            raise ValueError("source loss weights must sum to > 0")

        source_masses = np.bincount(
            self.source_ids,
            weights=self.run_full_tokens.astype(np.float64),
            minlength=len(self.cache.source_names),
        )
        total_mass = float(source_masses.sum())
        if total_mass <= 0:
            raise ValueError("indexed FASTA source loss weighting requires positive token mass")

        multipliers = np.zeros((len(self.cache.source_names),), dtype=np.float32)
        for source, raw_weight in source_loss_weights.items():
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("source loss weights must be non-negative")
            if weight == 0:
                continue
            source_id = source_to_id[source]
            actual_fraction = float(source_masses[source_id]) / total_mass
            if actual_fraction <= 0:
                raise ValueError(f"source {source!r} has no token mass for split={self.split!r}")
            target_fraction = weight / raw_total
            multipliers[source_id] = np.float32(target_fraction / actual_fraction)

        if not np.any(multipliers > 0):
            raise ValueError("source loss weights must include at least one positive eligible source")
        self.loss_weights_by_source_id = multipliers
        self.loss_weight_summary = {
            self.cache.source_names[source_id]: float(multiplier)
            for source_id, multiplier in enumerate(multipliers.tolist())
            if multiplier > 0
        }

    def __len__(self) -> int:
        return int(math.ceil(self.samples / self.ddp_world_size))

    def summary(self) -> dict[str, Any]:
        eligible_by_source: dict[str, int] = {}
        windows_by_source: dict[str, int] = {}
        for source_id, source_name in enumerate(self.cache.source_names):
            local = np.flatnonzero(self.source_ids == source_id)
            if local.size == 0:
                continue
            eligible_by_source[source_name] = int(local.size)
            windows_by_source[source_name] = int(self.available[local].sum())
        return {
            "source_mode": "indexed_fasta",
            "index_dir": str(self.cache.index_dir),
            "runtime_cache_dir": str(self.cache.cache_dir),
            "split": self.split,
            "split_seed": self.split_seed,
            "window_mode": "nonoverlap_file_stream",
            "epoch_mode": "samples" if self.samples != self.total_candidate_windows else "all_windows",
            "seq_length": self.seq_length,
            "token_merge_size": self.token_merge_size,
            "base_length": self.base_length,
            "samples": self.samples,
            "eligible_run_count": int(self.positions.size),
            "eligible_window_count": self.total_candidate_windows,
            "candidate_window_count": self.total_candidate_windows,
            "padded_window_count": self.nonoverlap_padded_windows,
            "stream_unit_count": self.stream_unit_count,
            "stream_file_count": self.stream_file_count,
            "file_stream_windows": self.file_stream_windows,
            "file_shuffle_buffer_windows": self.file_shuffle_buffer_windows,
            "file_stream_order_seed": self.file_stream_order_seed,
            "eligible_runs_by_source": eligible_by_source,
            "eligible_windows_by_source": windows_by_source,
            "source_loss_weights": dict(self.loss_weight_summary),
        }

    def _file_order(self, cycle: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + self.file_stream_order_seed + cycle * 1_000_003)
        return rng.permutation(self.stream_file_count)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        consumer_id = self.ddp_rank * num_workers + worker_id
        total_consumers = self.ddp_world_size * num_workers

        produced_before_cycle = 0
        cycle = 0
        while produced_before_cycle < self.samples:
            cycle_limit = min(self.total_candidate_windows, self.samples - produced_before_cycle)
            file_order = self._file_order(cycle)
            order_position = 0
            consumed_windows = 0
            for file_group in file_order.tolist():
                file_windows = int(self.file_window_counts[file_group])
                unit_count = int(self.file_unit_counts[file_group])
                for file_unit in range(unit_count):
                    unit_file_window_start = file_unit * self.file_stream_windows
                    unit_window_count = min(self.file_stream_windows, file_windows - unit_file_window_start)
                    unit_start = consumed_windows
                    unit_end = consumed_windows + unit_window_count
                    consumed_windows = unit_end
                    current_order_position = order_position
                    order_position += 1
                    if current_order_position % total_consumers != consumer_id:
                        continue
                    overlap_start = unit_start
                    overlap_end = min(unit_end, cycle_limit)
                    if overlap_start >= overlap_end:
                        continue
                    skip_windows = overlap_start - unit_start
                    limit_windows = overlap_end - overlap_start
                    yield from self._iter_unit_items(
                        file_group=int(file_group),
                        unit_file_window_start=int(unit_file_window_start),
                        unit_window_count=int(unit_window_count),
                        skip_windows=int(skip_windows),
                        limit_windows=int(limit_windows),
                        cycle=cycle,
                    )
                if consumed_windows >= cycle_limit:
                    break

            produced_before_cycle += int(cycle_limit)
            cycle += 1

    def _iter_unit_items(
        self,
        *,
        file_group: int,
        unit_file_window_start: int,
        unit_window_count: int,
        skip_windows: int,
        limit_windows: int,
        cycle: int,
    ):
        item_iter = self._iter_unit_items_sequential(
            file_group=file_group,
            unit_file_window_start=unit_file_window_start,
            unit_window_count=unit_window_count,
            skip_windows=skip_windows,
            limit_windows=limit_windows,
        )
        if self.file_shuffle_buffer_windows <= 0:
            yield from item_iter
            return

        rng = random.Random(
            self.seed
            + self.file_stream_order_seed
            + cycle * 1_000_003
            + int(file_group) * 97_003
            + int(unit_file_window_start) * 17
        )
        buffer: list[dict[str, torch.Tensor]] = []
        for item in item_iter:
            buffer.append(item)
            if len(buffer) >= self.file_shuffle_buffer_windows:
                rng.shuffle(buffer)
                yield from buffer
                buffer.clear()
        if buffer:
            rng.shuffle(buffer)
            yield from buffer

    def _iter_unit_items_sequential(
        self,
        *,
        file_group: int,
        unit_file_window_start: int,
        unit_window_count: int,
        skip_windows: int,
        limit_windows: int,
    ):
        unit_file_window_start = int(unit_file_window_start) + int(skip_windows)
        absolute_ticket = int(self.file_window_starts[file_group]) + unit_file_window_start
        local_position = int(np.searchsorted(self.available_cumsum, absolute_ticket + 1, side="left"))
        previous = int(self.available_cumsum[local_position - 1]) if local_position > 0 else 0
        run_window = absolute_ticket - previous

        remaining = min(int(limit_windows), int(unit_window_count) - int(skip_windows))
        while remaining > 0:
            available = int(self.available[local_position])
            take = min(remaining, available - run_window)
            yield from self._read_run_windows(local_position=local_position, start_window=run_window, window_count=take)
            remaining -= int(take)
            local_position += 1
            run_window = 0

    def _read_run_windows(self, *, local_position: int, start_window: int, window_count: int):
        run_position = int(self.positions[local_position])
        run_id = int(self.cache.run_ids[run_position])
        file_id = int(self.cache.run_file_ids[run_position])
        full_tokens = int(self.run_full_tokens[local_position])
        first_token = int(start_window) * self.seq_length
        last_window = int(start_window) + int(window_count) - 1
        end_token = min((last_window + 1) * self.seq_length, full_tokens)
        tokens_to_read = end_token - first_token
        if tokens_to_read <= 0:
            return
        bases = self._read_bases(
            run_id=run_id,
            file_id=file_id,
            base_start=first_token * self.token_merge_size,
            target_bases=tokens_to_read * self.token_merge_size,
            run_start_file_offset=int(self.cache.run_start_file_offsets[run_position]),
        )
        tokens = self._tokenize_bases(bases)
        source_id = int(self.source_ids[local_position])
        for offset in range(int(window_count)):
            window_token_start = offset * self.seq_length
            original_window_index = int(start_window) + offset
            token_count = min(self.seq_length, full_tokens - original_window_index * self.seq_length)
            item_tokens = tokens[window_token_start : window_token_start + token_count]
            if item_tokens.shape[0] != self.seq_length:
                padded = torch.full((self.seq_length,), self.pad_id, dtype=torch.long)
                padded[: item_tokens.shape[0]] = item_tokens
                item_tokens = padded
            else:
                item_tokens = item_tokens.clone()
            item: dict[str, torch.Tensor] = {
                "input_ids": item_tokens,
                "source_id": torch.tensor(source_id, dtype=torch.long),
            }
            if self.source_loss_weights_config:
                item["loss_weight"] = torch.tensor(float(self.loss_weights_by_source_id[source_id]), dtype=torch.float32)
            yield item

    def _handle_for_file(self, file_id: int):
        handle = self._handles.get(file_id)
        if handle is None or handle.closed:
            handle = self.cache.file_paths[file_id].open("rb")
            self._handles[file_id] = handle
        return handle

    def _anchor_for(
        self,
        run_id: int,
        base_offset: int,
        *,
        run_start_file_offset: int | None = None,
    ) -> tuple[int, int]:
        if base_offset == 0 and run_start_file_offset is not None:
            return 0, int(run_start_file_offset)
        start = int(np.searchsorted(self.cache.anchor_run_ids, run_id, side="left"))
        end = int(np.searchsorted(self.cache.anchor_run_ids, run_id, side="right"))
        if start == end:
            raise ValueError(f"run {run_id} has no anchors")
        local_offsets = self.cache.anchor_base_offsets[start:end]
        local_index = int(np.searchsorted(local_offsets, base_offset, side="right")) - 1
        if local_index < 0:
            local_index = 0
        absolute = start + local_index
        return int(self.cache.anchor_base_offsets[absolute]), int(self.cache.anchor_file_offsets[absolute])

    def _read_bases(
        self,
        *,
        run_id: int,
        file_id: int,
        base_start: int,
        target_bases: int,
        run_start_file_offset: int | None = None,
    ) -> bytes:
        anchor_base, anchor_file = self._anchor_for(
            run_id,
            base_start,
            run_start_file_offset=run_start_file_offset,
        )
        skip_bases = base_start - anchor_base
        handle = self._handle_for_file(file_id)
        handle.seek(anchor_file)

        bases = bytearray()
        skipped = 0
        read_size = min(1024 * 1024, max(4096, int((target_bases + skip_bases) * 2) + 256))
        while len(bases) < target_bases:
            remaining = target_bases - len(bases)
            chunk = handle.read(min(read_size, max(4096, remaining * 2 + 256)))
            if not chunk:
                break
            for byte_value in chunk:
                upper = _to_upper(byte_value)
                if upper in (ord("A"), ord("C"), ord("G"), ord("T")):
                    if skipped < skip_bases:
                        skipped += 1
                    else:
                        bases.append(upper)
                        if len(bases) >= target_bases:
                            break
                elif byte_value in (9, 10, 11, 12, 13, 32):
                    continue
                elif _byte_is_alpha(byte_value):
                    raise ValueError(f"encountered non-ACGT base while reading indexed run {run_id}")
        if len(bases) != target_bases:
            raise ValueError(f"sampled only {len(bases)} bases from indexed run {run_id}, expected {target_bases}")
        return bytes(bases)

    def _tokenize_bases(self, bases: bytes) -> torch.Tensor:
        if self.token_merge_size <= 1:
            token_ids = np.frombuffer(bases, dtype=np.uint8).astype(np.int64, copy=True)
        else:
            if self._digit_lookup is None or self._merge_weights is None:
                raise ValueError("internal error: token merge lookup was not initialized")
            raw = np.frombuffer(bases, dtype=np.uint8)
            digits = self._digit_lookup[raw]
            if np.any(digits < 0):
                raise ValueError("indexed FASTA sample contained a base outside token_merge_alphabet")
            full_digit_count = (digits.shape[0] // self.token_merge_size) * self.token_merge_size
            digits = digits[:full_digit_count]
            merged = digits.reshape(-1, self.token_merge_size).astype(np.uint64, copy=False)
            token_ids = (merged * self._merge_weights).sum(axis=1, dtype=np.uint64).astype(np.int64, copy=False)
        return torch.as_tensor(token_ids.copy(), dtype=torch.long)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class _SourceSequentialReader:
    def __init__(
        self,
        dataset: "IndexedMegabyteSourceBatchStreamDataset",
        *,
        source_id: int,
        consumer_id: int,
        total_consumers: int,
    ) -> None:
        self.dataset = dataset
        self.source_id = int(source_id)
        self.consumer_id = int(consumer_id)
        self.total_consumers = int(total_consumers)
        self.file_groups = dataset.source_file_groups.get(self.source_id, [])
        self.stride_windows = len(self.file_groups) < self.total_consumers
        if self.stride_windows:
            self.assigned_file_indices = list(range(len(self.file_groups)))
        else:
            self.assigned_file_indices = [
                index for index in range(len(self.file_groups)) if index % self.total_consumers == self.consumer_id
            ]
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.global_window_index = 0
        self.cycle = 0
        self.file_order: list[int] = []
        self.file_order_pos = 0
        self.group_run_offset = 0
        self.run_window_offset = 0
        self.exhausted_once = False
        self.read_chunk_index = 0
        self.total_source_windows = int(
            sum(int(dataset.available[local_position]) for group in self.file_groups for local_position in group.tolist())
        )
        self._reset_cycle()

    def has_data(self) -> bool:
        if not self.assigned_file_indices:
            return False
        if not self.stride_windows:
            return True
        return self.total_source_windows > self.consumer_id

    def _reset_cycle(self) -> None:
        if not self.assigned_file_indices:
            self.file_order = []
            return
        rng = random.Random(
            self.dataset.seed
            + self.dataset.source_file_order_seed
            + self.source_id * 1_000_003
            + self.cycle * 97_003
        )
        self.file_order = list(self.assigned_file_indices)
        rng.shuffle(self.file_order)
        self.file_order_pos = 0
        self.group_run_offset = 0
        self.run_window_offset = 0
        self.global_window_index = 0
        self.read_chunk_index = 0

    def next_item(self) -> dict[str, torch.Tensor] | None:
        if not self.has_data():
            return None
        while not self.buffer:
            filled = self._fill_buffer()
            if filled:
                break
            if self.dataset.epoch_mode == "all_windows":
                self.exhausted_once = True
                return None
            self.cycle += 1
            self._reset_cycle()
        if not self.buffer:
            return None
        return self.buffer.pop(0)

    def skip_items(self, count: int) -> bool:
        """Skip the specified number of items in the data stream.

        This method advances the reader's position by `count` items without returning them.
        Used primarily for training resumption from checkpoints.

        Args:
            count: Number of items to skip

        Returns:
            True if successfully skipped `count` items, False if data exhausted

        Note:
            When source_read_chunk_shuffle is enabled, complete read chunks can be
            skipped without disk reads because their internal shuffled order is never
            observed. Only the final partial chunk is materialized and discarded to
            preserve the same next-item order as physical skipping.
        """
        remaining = int(count)
        if remaining <= 0:
            return True
        if not self.has_data():
            return False
        if self.buffer:
            drop = min(remaining, len(self.buffer))
            del self.buffer[:drop]
            remaining -= drop
        while remaining > 0:
            if self.dataset.source_read_chunk_shuffle and remaining < self.dataset.source_read_chunk_windows:
                filled = self._fill_buffer()
                if not filled:
                    if self.dataset.epoch_mode == "all_windows":
                        self.exhausted_once = True
                        return False
                    self.cycle += 1
                    self._reset_cycle()
                    continue
                drop = min(remaining, len(self.buffer))
                del self.buffer[:drop]
                remaining -= drop
                continue

            skipped = (
                self._skip_next_read_chunk_items()
                if self.dataset.source_read_chunk_shuffle
                else self._skip_items_in_current_cycle(remaining)
            )
            remaining -= skipped
            if remaining <= 0:
                return True
            if skipped <= 0 or self.file_order_pos >= len(self.file_order):
                if self.dataset.epoch_mode == "all_windows":
                    self.exhausted_once = True
                    return False
                self.cycle += 1
                self._reset_cycle()
                continue
        return True

    def _skip_items_in_current_cycle(self, count: int) -> int:
        skipped = 0
        target = int(count)
        while skipped < target and self.file_order_pos < len(self.file_order):
            file_group_index = self.file_order[self.file_order_pos]
            local_positions = self.file_groups[file_group_index]
            if self.group_run_offset >= local_positions.shape[0]:
                self.file_order_pos += 1
                self.group_run_offset = 0
                self.run_window_offset = 0
                continue

            local_position = int(local_positions[self.group_run_offset])
            segment_remaining = int(self.dataset.available[local_position]) - int(self.run_window_offset)
            if segment_remaining <= 0:
                self.group_run_offset += 1
                self.run_window_offset = 0
                continue

            if not self.stride_windows:
                take = min(target - skipped, segment_remaining)
                self._advance_raw_windows(take)
                skipped += take
                continue

            assigned_in_segment = self._assigned_count_in_raw_span(self.global_window_index, segment_remaining)
            needed = target - skipped
            if assigned_in_segment <= needed:
                self._advance_raw_windows(segment_remaining)
                skipped += assigned_in_segment
                continue

            lo = 1
            hi = segment_remaining
            while lo < hi:
                mid = (lo + hi) // 2
                if self._assigned_count_in_raw_span(self.global_window_index, mid) >= needed:
                    hi = mid
                else:
                    lo = mid + 1
            self._advance_raw_windows(lo)
            skipped += needed
        return skipped

    def _assigned_count_in_raw_span(self, start: int, length: int) -> int:
        if length <= 0:
            return 0
        first_offset = (self.consumer_id - int(start)) % self.total_consumers
        if first_offset >= length:
            return 0
        return 1 + (int(length) - 1 - first_offset) // self.total_consumers

    def _advance_raw_windows(self, window_count: int) -> None:
        remaining = int(window_count)
        while remaining > 0 and self.file_order_pos < len(self.file_order):
            file_group_index = self.file_order[self.file_order_pos]
            local_positions = self.file_groups[file_group_index]
            if self.group_run_offset >= local_positions.shape[0]:
                self.file_order_pos += 1
                self.group_run_offset = 0
                self.run_window_offset = 0
                continue
            local_position = int(local_positions[self.group_run_offset])
            available = int(self.dataset.available[local_position])
            take = min(remaining, available - self.run_window_offset)
            self.run_window_offset += int(take)
            self.global_window_index += int(take)
            remaining -= int(take)
            if self.run_window_offset >= available:
                self.group_run_offset += 1
                self.run_window_offset = 0
            if self.group_run_offset >= local_positions.shape[0]:
                self.file_order_pos += 1
                self.group_run_offset = 0
                self.run_window_offset = 0

    def _skip_next_read_chunk_items(self) -> int:
        if not self.file_order:
            return 0
        target = self.dataset.source_read_chunk_windows
        skipped = 0
        while skipped < target and self.file_order_pos < len(self.file_order):
            file_group_index = self.file_order[self.file_order_pos]
            local_positions = self.file_groups[file_group_index]
            while skipped < target and self.group_run_offset < local_positions.shape[0]:
                local_position = int(local_positions[self.group_run_offset])
                available = int(self.dataset.available[local_position])
                take = min(target - skipped, available - self.run_window_offset)
                if take > 0:
                    if self.stride_windows:
                        skipped += self._assigned_count_in_raw_span(self.global_window_index, take)
                        self.global_window_index += int(take)
                    else:
                        skipped += int(take)
                self.run_window_offset += int(take)
                if self.run_window_offset >= available:
                    self.group_run_offset += 1
                    self.run_window_offset = 0
            if self.group_run_offset >= local_positions.shape[0]:
                self.file_order_pos += 1
                self.group_run_offset = 0
                self.run_window_offset = 0
        if skipped > 0:
            self.read_chunk_index += 1
        return skipped

    def _fill_buffer(self) -> bool:
        if not self.file_order:
            return False
        target = self.dataset.source_read_chunk_windows
        while len(self.buffer) < target and self.file_order_pos < len(self.file_order):
            file_group_index = self.file_order[self.file_order_pos]
            local_positions = self.file_groups[file_group_index]
            while len(self.buffer) < target and self.group_run_offset < local_positions.shape[0]:
                local_position = int(local_positions[self.group_run_offset])
                available = int(self.dataset.available[local_position])
                take = min(target - len(self.buffer), available - self.run_window_offset)
                if take > 0:
                    items = self.dataset._read_run_windows(
                        local_position=local_position,
                        start_window=self.run_window_offset,
                        window_count=take,
                    )
                    if self.stride_windows:
                        for item in items:
                            if self.global_window_index % self.total_consumers == self.consumer_id:
                                self.buffer.append(item)
                            self.global_window_index += 1
                    else:
                        self.buffer.extend(items)
                self.run_window_offset += int(take)
                if self.run_window_offset >= available:
                    self.group_run_offset += 1
                    self.run_window_offset = 0
            if self.group_run_offset >= local_positions.shape[0]:
                self.file_order_pos += 1
                self.group_run_offset = 0
                self.run_window_offset = 0
        if self.buffer and self.dataset.source_read_chunk_shuffle:
            rng = random.Random(
                self.dataset.seed
                + self.dataset.source_file_order_seed
                + self.source_id * 1_000_003
                + self.cycle * 97_003
                + self.read_chunk_index * 65_537
            )
            rng.shuffle(self.buffer)
        if self.buffer:
            self.read_chunk_index += 1
        return bool(self.buffer)


class _SourceRandomChunkReader:
    def __init__(
        self,
        dataset: "IndexedMegabyteSourceBatchStreamDataset",
        *,
        source_id: int,
        consumer_id: int,
        total_consumers: int,
    ) -> None:
        self.dataset = dataset
        self.source_id = int(source_id)
        self.consumer_id = int(consumer_id)
        self.total_consumers = int(total_consumers)
        self.local_positions = dataset.source_local_positions.get(self.source_id)
        self.chunk_prefix = dataset.source_chunk_prefixes.get(self.source_id)
        self.window_prefix = dataset.source_window_prefixes.get(self.source_id)
        self.total_chunks = 0 if self.chunk_prefix is None else int(self.chunk_prefix[-1])
        self.total_source_windows = int(dataset.source_window_counts[self.source_id])
        self.buffer: list[dict[str, torch.Tensor]] = []
        self.cycle = 0
        self.chunk_order_pos = 0
        self.chunk_order_offset = 0
        self.chunk_order_step = 1
        self.read_chunk_index = 0
        self._reset_cycle()

    def has_data(self) -> bool:
        return self.total_source_windows > self.consumer_id and self.total_chunks > 0

    def _reset_cycle(self) -> None:
        if self.total_chunks <= 0:
            return
        rng = random.Random(
            self.dataset.seed
            + self.dataset.source_file_order_seed
            + self.dataset.epoch_index * 1_000_000_007
            + self.source_id * 1_000_003
            + self.cycle * 97_003
        )
        self.chunk_order_offset = rng.randrange(self.total_chunks)
        if self.total_chunks <= 1:
            self.chunk_order_step = 1
        else:
            step = rng.randrange(1, self.total_chunks)
            while math.gcd(step, self.total_chunks) != 1:
                step = rng.randrange(1, self.total_chunks)
            self.chunk_order_step = int(step)
        self.chunk_order_pos = 0
        self.read_chunk_index = 0

    def _assigned_count_in_span(self, start: int, length: int) -> int:
        if length <= 0:
            return 0
        first_offset = (self.consumer_id - int(start)) % self.total_consumers
        if first_offset >= length:
            return 0
        return 1 + (int(length) - 1 - first_offset) // self.total_consumers

    def _chunk_for_order_position(self, order_position: int) -> tuple[int, int, int, int]:
        if self.local_positions is None or self.chunk_prefix is None or self.window_prefix is None:
            raise ValueError("source random chunk reader was not initialized")
        chunk_ordinal = (self.chunk_order_offset + int(order_position) * self.chunk_order_step) % self.total_chunks
        run_array_index = int(np.searchsorted(self.chunk_prefix, chunk_ordinal, side="right")) - 1
        if run_array_index < 0:
            run_array_index = 0
        local_position = int(self.local_positions[run_array_index])
        chunk_in_run = int(chunk_ordinal - int(self.chunk_prefix[run_array_index]))
        start_window = chunk_in_run * self.dataset.source_read_chunk_windows
        available = int(self.dataset.available[local_position])
        window_count = min(self.dataset.source_read_chunk_windows, available - start_window)
        source_window_start = int(self.window_prefix[run_array_index]) + start_window
        return local_position, int(start_window), int(window_count), int(source_window_start)

    def _fill_buffer(self) -> bool:
        while not self.buffer:
            if self.total_chunks <= 0:
                return False
            if self.chunk_order_pos >= self.total_chunks:
                self.cycle += 1
                self._reset_cycle()
                continue

            local_position, start_window, window_count, source_window_start = self._chunk_for_order_position(
                self.chunk_order_pos
            )
            self.chunk_order_pos += 1
            items = self.dataset._read_run_windows(
                local_position=local_position,
                start_window=start_window,
                window_count=window_count,
            )
            for offset, item in enumerate(items):
                source_window_ordinal = source_window_start + offset
                if source_window_ordinal % self.total_consumers == self.consumer_id:
                    self.buffer.append(item)
            if self.buffer and self.dataset.source_read_chunk_shuffle:
                rng = random.Random(
                    self.dataset.seed
                    + self.dataset.source_file_order_seed
                    + self.dataset.epoch_index * 1_000_000_007
                    + self.source_id * 1_000_003
                    + self.cycle * 97_003
                    + self.read_chunk_index * 65_537
                    + int(source_window_start)
                )
                rng.shuffle(self.buffer)
            if items:
                self.read_chunk_index += 1
        return True

    def next_item(self) -> dict[str, torch.Tensor] | None:
        if not self.has_data():
            return None
        while not self.buffer:
            if self._fill_buffer():
                break
        if not self.buffer:
            return None
        return self.buffer.pop(0)

    def skip_items(self, count: int) -> bool:
        remaining = int(count)
        if remaining <= 0:
            return True
        if not self.has_data():
            return False
        if self.buffer:
            drop = min(remaining, len(self.buffer))
            del self.buffer[:drop]
            remaining -= drop

        while remaining > 0:
            if self.total_chunks <= 0:
                return False
            if self.chunk_order_pos >= self.total_chunks:
                self.cycle += 1
                self._reset_cycle()
                continue

            _, _, window_count, source_window_start = self._chunk_for_order_position(self.chunk_order_pos)
            assigned = self._assigned_count_in_span(source_window_start, window_count)
            if assigned <= 0:
                self.chunk_order_pos += 1
                continue
            if remaining >= assigned:
                self.chunk_order_pos += 1
                self.read_chunk_index += 1
                remaining -= assigned
                continue

            filled = self._fill_buffer()
            if not filled:
                return False
            drop = min(remaining, len(self.buffer))
            del self.buffer[:drop]
            remaining -= drop
        return True


class IndexedMegabyteSourceBatchStreamDataset(IterableDataset):
    def __init__(
        self,
        *,
        index_dir: str | Path,
        split: str,
        seq_length: int,
        token_merge_size: int,
        token_merge_alphabet: str,
        samples: int | None,
        seed: int,
        batch_size: int,
        source_weights: dict[str, float] | None = None,
        source_loss_weights: dict[str, float] | None = None,
        pad_id: int | None = None,
        source_mix_chunk_batches: int = 64,
        source_read_chunk_windows: int = 8192,
        source_read_chunk_shuffle: bool = True,
        source_balance_batches: int | None = None,
        source_read_block_windows: int | None = None,
        source_file_order_seed: int = 0,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        split_seed: int = 0,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        start_batch_index: int = 0,
    ) -> None:
        if seq_length <= 0:
            raise ValueError("seq_length must be > 0")
        if token_merge_size <= 0:
            raise ValueError("token_merge_size must be > 0")
        if samples is not None and samples <= 0:
            raise ValueError("samples must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if pad_id is None:
            raise ValueError("pad_id is required for source_batch_file_stream indexed FASTA sampling")
        # Legacy parameter compatibility: accept old names but use new ones internally
        if source_balance_batches is not None:
            import warnings
            warnings.warn(
                "Parameter 'source_balance_batches' is deprecated and ignored by source_batch_file_stream "
                "probability sampling.",
                DeprecationWarning,
                stacklevel=2
            )
            source_mix_chunk_batches = int(source_balance_batches)
        if source_read_block_windows is not None:
            import warnings
            warnings.warn(
                "Parameter 'source_read_block_windows' is deprecated, use 'source_read_chunk_windows' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            source_read_chunk_windows = int(source_read_block_windows)
        if source_mix_chunk_batches <= 0:
            raise ValueError("source_mix_chunk_batches must be > 0 for backward-compatible configuration")
        if source_read_chunk_windows <= 0:
            raise ValueError("source_read_chunk_windows must be > 0")
        if ddp_rank < 0 or ddp_world_size <= 0 or ddp_rank >= ddp_world_size:
            raise ValueError("invalid DDP rank/world size for indexed FASTA source batch stream")
        if start_batch_index < 0:
            raise ValueError("start_batch_index must be >= 0")
        alphabet = normalize_alphabet(token_merge_alphabet)
        if set(alphabet) - {"A", "C", "G", "T", "N"}:
            raise ValueError("indexed_fasta Megabyte sampling supports only A/C/G/T/N alphabets")
        if not {"A", "C", "G", "T"}.issubset(set(alphabet)):
            raise ValueError("indexed_fasta Megabyte sampling requires A/C/G/T in token_merge_alphabet")

        self.cache = load_fasta_index_runtime_cache(index_dir)
        self.split = split
        self.seq_length = int(seq_length)
        self.token_merge_size = int(token_merge_size)
        self.base_length = self.seq_length * self.token_merge_size
        self.seed = int(seed)
        self.alphabet = alphabet
        self.pad_id = int(pad_id)
        self.batch_size = int(batch_size)
        self.source_mix_chunk_batches = int(source_mix_chunk_batches)
        self.source_read_chunk_windows = int(source_read_chunk_windows)
        self.source_read_chunk_shuffle = bool(source_read_chunk_shuffle)
        self.source_file_order_seed = int(source_file_order_seed)
        self.train_ratio = float(train_ratio)
        self.val_ratio = float(val_ratio)
        self.test_ratio = float(test_ratio)
        self.split_seed = int(split_seed)
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self.start_batch_index = int(start_batch_index)
        self.epoch_index = 0
        self.epoch_mode = "samples" if samples is not None else "all_windows"
        self.epoch_sample_count_mode = "configured_samples" if samples is not None else "expected_slowest_source_coverage"
        self._handles: dict[int, Any] = {}
        self.source_weights_config = dict(source_weights or {})
        self.source_loss_weights_config = dict(source_loss_weights or {})

        split_mask = split_run_ids(
            np.asarray(self.cache.run_ids),
            split=split,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            split_seed=self.split_seed,
        )
        run_lengths = np.asarray(self.cache.run_lengths)
        eligible_mask = (run_lengths // self.token_merge_size) >= 1
        self.positions = np.flatnonzero(split_mask & eligible_mask).astype(np.int64, copy=False)
        if self.positions.size == 0:
            raise ValueError(f"no indexed FASTA runs are eligible for split={split!r} and seq_length={seq_length}")

        self.run_full_tokens = (run_lengths[self.positions] // self.token_merge_size).astype(np.int64, copy=False)
        self.available = ((self.run_full_tokens + self.seq_length - 1) // self.seq_length).astype(np.int64, copy=False)
        self.total_candidate_windows = int(self.available.sum())
        if self.total_candidate_windows <= 0:
            raise ValueError(f"no indexed FASTA windows are eligible for split={split!r} and seq_length={seq_length}")
        self.nonoverlap_padded_windows = int(np.count_nonzero(self.run_full_tokens % self.seq_length))
        self.source_ids = np.asarray(self.cache.run_source_ids)[self.positions].astype(np.int32, copy=False)
        self.file_ids = np.asarray(self.cache.run_file_ids)[self.positions].astype(np.int64, copy=False)
        self.run_start_file_offsets = np.asarray(self.cache.run_start_file_offsets)[self.positions].astype(
            np.int64,
            copy=False,
        )
        self.source_file_groups: dict[int, list[np.ndarray]] = {}
        self._configure_sampling(source_weights)
        self._build_source_window_chunks()
        self._configure_loss_weights(source_loss_weights)
        self.samples = self._expected_slowest_source_samples() if samples is None else int(samples)
        self.batch_count = int(math.ceil(self.samples / self.batch_size))
        if self.start_batch_index > self.batch_count:
            raise ValueError("start_batch_index cannot exceed the dataset batch count")

        if self.token_merge_size > 1:
            self._digit_lookup = np.full(256, -1, dtype=np.int16)
            for index, ch in enumerate(self.alphabet):
                self._digit_lookup[ord(ch)] = index
                self._digit_lookup[ord(ch.lower())] = index
            self._merge_weights = np.array(
                [len(self.alphabet) ** power for power in range(self.token_merge_size - 1, -1, -1)],
                dtype=np.uint64,
            )
        else:
            self._digit_lookup = None
            self._merge_weights = None

    def _build_source_file_groups(self) -> None:
        self.source_file_groups: dict[int, list[np.ndarray]] = {}
        for source_id in np.unique(self.source_ids).astype(np.int32, copy=False).tolist():
            local_positions = np.flatnonzero(self.source_ids == source_id).astype(np.int64, copy=False)
            if local_positions.size == 0:
                continue
            local_file_ids = self.file_ids[local_positions]
            change = np.empty((local_positions.shape[0],), dtype=bool)
            change[0] = True
            change[1:] = local_file_ids[1:] != local_file_ids[:-1]
            starts = np.flatnonzero(change)
            groups: list[np.ndarray] = []
            for group_index, start in enumerate(starts.tolist()):
                end = int(starts[group_index + 1]) if group_index + 1 < starts.shape[0] else local_positions.shape[0]
                groups.append(local_positions[start:end])
            self.source_file_groups[source_id] = groups

    def _build_source_window_chunks(self) -> None:
        self.source_local_positions: dict[int, np.ndarray] = {}
        self.source_chunk_prefixes: dict[int, np.ndarray] = {}
        self.source_window_prefixes: dict[int, np.ndarray] = {}
        for source_id in self.active_source_ids.tolist():
            local_positions = np.flatnonzero(self.source_ids == source_id).astype(np.int64, copy=False)
            if local_positions.size == 0:
                continue
            available = self.available[local_positions].astype(np.int64, copy=False)
            chunk_counts = ((available + self.source_read_chunk_windows - 1) // self.source_read_chunk_windows).astype(
                np.int64,
                copy=False,
            )
            chunk_prefix = np.empty((chunk_counts.shape[0] + 1,), dtype=np.int64)
            chunk_prefix[0] = 0
            np.cumsum(chunk_counts, out=chunk_prefix[1:])
            window_prefix = np.empty((available.shape[0] + 1,), dtype=np.int64)
            window_prefix[0] = 0
            np.cumsum(available, out=window_prefix[1:])
            self.source_local_positions[source_id] = local_positions
            self.source_chunk_prefixes[source_id] = chunk_prefix
            self.source_window_prefixes[source_id] = window_prefix

    def _configure_sampling(self, source_weights: dict[str, float] | None) -> None:
        source_window_counts = np.bincount(
            self.source_ids,
            weights=self.available.astype(np.float64),
            minlength=len(self.cache.source_names),
        )
        self.source_window_counts = source_window_counts.astype(np.int64, copy=False)
        source_to_id = {source: idx for idx, source in enumerate(self.cache.source_names)}
        if source_weights:
            unknown = sorted(set(source_weights) - set(source_to_id))
            if unknown:
                raise ValueError(f"source weights include unknown sources: {', '.join(unknown)}")
            active_ids: list[int] = []
            weights: list[float] = []
            for source, raw_weight in sorted(source_weights.items()):
                weight = float(raw_weight)
                if weight < 0:
                    raise ValueError("source sampling weights must be non-negative")
                if weight == 0:
                    continue
                source_id = source_to_id[source]
                if source_window_counts[source_id] <= 0:
                    raise ValueError(f"source {source!r} has no eligible windows for split={self.split!r}")
                active_ids.append(source_id)
                weights.append(weight)
            if not weights or sum(weights) <= 0:
                raise ValueError("source sampling weights must sum to > 0")
            self.source_sampling_mode = "weighted"
        else:
            active_ids = [int(source_id) for source_id in np.flatnonzero(source_window_counts > 0).tolist()]
            weights = [float(source_window_counts[source_id]) for source_id in active_ids]
            self.source_sampling_mode = "natural"

        self.active_source_ids = np.asarray(active_ids, dtype=np.int32)
        self.source_probabilities = np.asarray(weights, dtype=np.float64)
        self.source_probabilities /= self.source_probabilities.sum()
        self.source_cdf = np.cumsum(self.source_probabilities)
        self.source_cdf[-1] = 1.0
        self.source_names = [self.cache.source_names[int(source_id)] for source_id in self.active_source_ids.tolist()]

    def _expected_slowest_source_samples(self) -> int:
        candidates: list[float] = []
        for source_id, probability in zip(self.active_source_ids.tolist(), self.source_probabilities.tolist()):
            if probability <= 0:
                continue
            windows = int(self.source_window_counts[int(source_id)])
            candidates.append(float(windows) / float(probability))
        if not candidates:
            raise ValueError("cannot compute indexed FASTA epoch length without active sources")
        return int(math.ceil(max(candidates)))

    def _configure_loss_weights(self, source_loss_weights: dict[str, float] | None) -> None:
        self.loss_weights_by_source_id = np.ones((len(self.cache.source_names),), dtype=np.float32)
        self.loss_weight_summary: dict[str, float] = {}
        if source_loss_weights is None or not source_loss_weights:
            return

        source_to_id = {source: idx for idx, source in enumerate(self.cache.source_names)}
        unknown = sorted(set(source_loss_weights) - set(source_to_id))
        if unknown:
            raise ValueError(f"source loss weights include unknown sources: {', '.join(unknown)}")
        raw_total = float(sum(float(value) for value in source_loss_weights.values()))
        if raw_total <= 0:
            raise ValueError("source loss weights must sum to > 0")

        source_masses = np.bincount(
            self.source_ids,
            weights=self.run_full_tokens.astype(np.float64),
            minlength=len(self.cache.source_names),
        )
        total_mass = float(source_masses.sum())
        if total_mass <= 0:
            raise ValueError("indexed FASTA source loss weighting requires positive token mass")

        multipliers = np.zeros((len(self.cache.source_names),), dtype=np.float32)
        for source, raw_weight in source_loss_weights.items():
            weight = float(raw_weight)
            if weight < 0:
                raise ValueError("source loss weights must be non-negative")
            if weight == 0:
                continue
            source_id = source_to_id[source]
            actual_fraction = float(source_masses[source_id]) / total_mass
            if actual_fraction <= 0:
                raise ValueError(f"source {source!r} has no token mass for split={self.split!r}")
            target_fraction = weight / raw_total
            multipliers[source_id] = np.float32(target_fraction / actual_fraction)

        if not np.any(multipliers > 0):
            raise ValueError("source loss weights must include at least one positive eligible source")
        self.loss_weights_by_source_id = multipliers
        self.loss_weight_summary = {
            self.cache.source_names[source_id]: float(multiplier)
            for source_id, multiplier in enumerate(multipliers.tolist())
            if multiplier > 0
        }

    def __len__(self) -> int:
        remaining_batches = max(0, self.batch_count - self.start_batch_index)
        return int(math.ceil(remaining_batches / self.ddp_world_size))

    def set_start_batch_index(self, start_batch_index: int) -> None:
        if start_batch_index < 0:
            raise ValueError("start_batch_index must be >= 0")
        if start_batch_index > self.batch_count:
            raise ValueError("start_batch_index cannot exceed the dataset batch count")
        self.start_batch_index = int(start_batch_index)

    def set_epoch(self, epoch_index: int) -> None:
        if epoch_index < 0:
            raise ValueError("epoch_index must be >= 0")
        self.epoch_index = int(epoch_index)

    def summary(self) -> dict[str, Any]:
        eligible_by_source: dict[str, int] = {}
        windows_by_source: dict[str, int] = {}
        file_groups_by_source: dict[str, int] = {}
        read_chunks_by_source: dict[str, int] = {}
        for source_id, source_name in enumerate(self.cache.source_names):
            local = np.flatnonzero(self.source_ids == source_id)
            if local.size == 0:
                continue
            eligible_by_source[source_name] = int(local.size)
            windows_by_source[source_name] = int(self.available[local].sum())
            file_groups_by_source[source_name] = int(len(self.source_file_groups.get(source_id, [])))
            chunk_prefix = self.source_chunk_prefixes.get(source_id)
            read_chunks_by_source[source_name] = 0 if chunk_prefix is None else int(chunk_prefix[-1])
        return {
            "source_mode": "indexed_fasta",
            "index_dir": str(self.cache.index_dir),
            "runtime_cache_dir": str(self.cache.cache_dir),
            "split": self.split,
            "split_seed": self.split_seed,
            "window_mode": "source_batch_file_stream",
            "epoch_mode": self.epoch_mode,
            "epoch_index": self.epoch_index,
            "epoch_sample_count_mode": self.epoch_sample_count_mode,
            "source_sampling_strategy": "per_sample_probability",
            "seq_length": self.seq_length,
            "token_merge_size": self.token_merge_size,
            "base_length": self.base_length,
            "samples": self.samples,
            "batches": self.batch_count,
            "start_batch_index": self.start_batch_index,
            "batch_size": self.batch_size,
            "eligible_run_count": int(self.positions.size),
            "eligible_window_count": self.total_candidate_windows,
            "candidate_window_count": self.total_candidate_windows,
            "padded_window_count": self.nonoverlap_padded_windows,
            "source_mix_chunk_batches": self.source_mix_chunk_batches,
            "source_balance_batches": self.source_mix_chunk_batches,
            "deprecated_source_mix_chunk_batches_ignored": True,
            "source_read_chunk_windows": self.source_read_chunk_windows,
            "source_read_chunk_shuffle": self.source_read_chunk_shuffle,
            "source_file_order_seed": self.source_file_order_seed,
            "source_sampling_mode": self.source_sampling_mode,
            "source_sampling_weights": {
                name: float(weight) for name, weight in zip(self.source_names, self.source_probabilities.tolist())
            },
            "eligible_runs_by_source": eligible_by_source,
            "eligible_windows_by_source": windows_by_source,
            "file_groups_by_source": file_groups_by_source,
            "read_chunks_by_source": read_chunks_by_source,
            "source_loss_weights": dict(self.loss_weight_summary),
        }

    def _source_random_values(self, global_sample_indices: np.ndarray) -> np.ndarray:
        values = global_sample_indices.astype(np.uint64, copy=False)
        values = values + np.uint64(self.seed) + np.uint64(self.source_file_order_seed) * np.uint64(0x9E3779B97F4A7C15)
        values = values + np.uint64(self.epoch_index) * np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(30)
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> np.uint64(27)
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
        return ((values >> np.uint64(11)).astype(np.float64)) * (1.0 / float(1 << 53))

    def _source_ids_for_samples(self, global_sample_indices: np.ndarray) -> np.ndarray:
        random_values = self._source_random_values(global_sample_indices)
        indices = np.searchsorted(self.source_cdf, random_values, side="right")
        indices = np.minimum(indices, self.active_source_ids.shape[0] - 1)
        return self.active_source_ids[indices].astype(np.int32, copy=False)

    def _source_ids_for_batch(self, batch_index: int) -> np.ndarray:
        start_sample = int(batch_index) * self.batch_size
        current_batch_size = min(self.batch_size, self.samples - start_sample)
        if current_batch_size <= 0:
            return np.asarray([], dtype=np.int32)
        sample_indices = np.arange(start_sample, start_sample + current_batch_size, dtype=np.uint64)
        return self._source_ids_for_samples(sample_indices)

    def _source_skip_counts_before_batch(self, start_batch_index: int, consumer_id: int, total_consumers: int) -> dict[int, int]:
        counts = {int(source_id): 0 for source_id in self.active_source_ids.tolist()}
        target_batch = min(int(start_batch_index), self.batch_count)
        if target_batch <= 0:
            return counts

        chunk_batches = 8192
        for batch_start in range(0, target_batch, chunk_batches):
            batch_end = min(target_batch, batch_start + chunk_batches)
            batch_indices = np.arange(batch_start, batch_end, dtype=np.int64)
            assigned_batches = batch_indices[batch_indices % int(total_consumers) == int(consumer_id)]
            if assigned_batches.size == 0:
                continue
            sample_offsets = np.arange(self.batch_size, dtype=np.int64)
            sample_indices = (assigned_batches[:, None] * self.batch_size + sample_offsets[None, :]).reshape(-1)
            sample_indices = sample_indices[sample_indices < self.samples].astype(np.uint64, copy=False)
            if sample_indices.size == 0:
                continue
            source_ids = self._source_ids_for_samples(sample_indices)
            bincount = np.bincount(source_ids, minlength=len(self.cache.source_names))
            for source_id in self.active_source_ids.tolist():
                counts[int(source_id)] += int(bincount[int(source_id)])
        return counts

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        consumer_id = self.ddp_rank * num_workers + worker_id
        total_consumers = self.ddp_world_size * num_workers
        readers = {
            int(source_id): _SourceRandomChunkReader(
                self,
                source_id=int(source_id),
                consumer_id=consumer_id,
                total_consumers=total_consumers,
            )
            for source_id in self.active_source_ids.tolist()
        }
        if not any(reader.has_data() for reader in readers.values()):
            raise ValueError("indexed FASTA source batch stream has no file groups assigned to this worker/rank")
        if self.start_batch_index > 0:
            skip_counts = self._source_skip_counts_before_batch(
                self.start_batch_index,
                consumer_id=consumer_id,
                total_consumers=total_consumers,
            )
            for source_id, count in skip_counts.items():
                if count > 0:
                    readers[int(source_id)].skip_items(int(count))

        for batch_index in range(self.start_batch_index, self.batch_count):
            if batch_index % total_consumers != consumer_id:
                continue
            source_ids = self._source_ids_for_batch(batch_index)
            items: list[dict[str, torch.Tensor]] = []
            missing_source = False
            for source_id in source_ids.tolist():
                item = readers[int(source_id)].next_item()
                if item is None:
                    missing_source = True
                    break
                items.append(item)
            if missing_source:
                continue
            if not items:
                continue
            batch: dict[str, torch.Tensor] = {
                "input_ids": torch.stack([item["input_ids"] for item in items], dim=0),
                "source_id": torch.stack([item["source_id"] for item in items], dim=0),
            }
            if self.source_loss_weights_config:
                batch["loss_weight"] = torch.stack([item["loss_weight"] for item in items], dim=0)
            batch["_batch_index"] = torch.tensor(int(batch_index), dtype=torch.long)
            yield batch

    def _read_run_windows(self, *, local_position: int, start_window: int, window_count: int) -> list[dict[str, torch.Tensor]]:
        run_position = int(self.positions[local_position])
        run_id = int(self.cache.run_ids[run_position])
        file_id = int(self.cache.run_file_ids[run_position])
        full_tokens = int(self.run_full_tokens[local_position])
        first_token = int(start_window) * self.seq_length
        last_window = int(start_window) + int(window_count) - 1
        end_token = min((last_window + 1) * self.seq_length, full_tokens)
        tokens_to_read = end_token - first_token
        if tokens_to_read <= 0:
            return []
        bases = self._read_bases(
            run_id=run_id,
            file_id=file_id,
            base_start=first_token * self.token_merge_size,
            target_bases=tokens_to_read * self.token_merge_size,
            run_start_file_offset=int(self.cache.run_start_file_offsets[run_position]),
        )
        tokens = self._tokenize_bases(bases)
        source_id = int(self.source_ids[local_position])
        items: list[dict[str, torch.Tensor]] = []
        for offset in range(int(window_count)):
            window_token_start = offset * self.seq_length
            original_window_index = int(start_window) + offset
            token_count = min(self.seq_length, full_tokens - original_window_index * self.seq_length)
            item_tokens = tokens[window_token_start : window_token_start + token_count]
            if item_tokens.shape[0] != self.seq_length:
                padded = torch.full((self.seq_length,), self.pad_id, dtype=torch.long)
                padded[: item_tokens.shape[0]] = item_tokens
                item_tokens = padded
            else:
                item_tokens = item_tokens.clone()
            item: dict[str, torch.Tensor] = {
                "input_ids": item_tokens,
                "source_id": torch.tensor(source_id, dtype=torch.long),
            }
            if self.source_loss_weights_config:
                item["loss_weight"] = torch.tensor(float(self.loss_weights_by_source_id[source_id]), dtype=torch.float32)
            items.append(item)
        return items

    def _handle_for_file(self, file_id: int):
        handle = self._handles.get(file_id)
        if handle is None or handle.closed:
            handle = self.cache.file_paths[file_id].open("rb")
            self._handles[file_id] = handle
        return handle

    def _anchor_for(
        self,
        run_id: int,
        base_offset: int,
        *,
        run_start_file_offset: int | None = None,
    ) -> tuple[int, int]:
        if base_offset == 0 and run_start_file_offset is not None:
            return 0, int(run_start_file_offset)
        start = int(np.searchsorted(self.cache.anchor_run_ids, run_id, side="left"))
        end = int(np.searchsorted(self.cache.anchor_run_ids, run_id, side="right"))
        if start == end:
            raise ValueError(f"run {run_id} has no anchors")
        local_offsets = self.cache.anchor_base_offsets[start:end]
        local_index = int(np.searchsorted(local_offsets, base_offset, side="right")) - 1
        if local_index < 0:
            local_index = 0
        absolute = start + local_index
        return int(self.cache.anchor_base_offsets[absolute]), int(self.cache.anchor_file_offsets[absolute])

    def _read_bases(
        self,
        *,
        run_id: int,
        file_id: int,
        base_start: int,
        target_bases: int,
        run_start_file_offset: int | None = None,
    ) -> bytes:
        anchor_base, anchor_file = self._anchor_for(
            run_id,
            base_start,
            run_start_file_offset=run_start_file_offset,
        )
        skip_bases = base_start - anchor_base
        handle = self._handle_for_file(file_id)
        handle.seek(anchor_file)

        bases = bytearray()
        skipped = 0
        read_size = min(1024 * 1024, max(4096, int((target_bases + skip_bases) * 2) + 256))
        while len(bases) < target_bases:
            remaining = target_bases - len(bases)
            chunk = handle.read(min(read_size, max(4096, remaining * 2 + 256)))
            if not chunk:
                break
            for byte_value in chunk:
                upper = _to_upper(byte_value)
                if upper in (ord("A"), ord("C"), ord("G"), ord("T")):
                    if skipped < skip_bases:
                        skipped += 1
                    else:
                        bases.append(upper)
                        if len(bases) >= target_bases:
                            break
                elif byte_value in (9, 10, 11, 12, 13, 32):
                    continue
                elif _byte_is_alpha(byte_value):
                    raise ValueError(f"encountered non-ACGT base while reading indexed run {run_id}")
        if len(bases) != target_bases:
            raise ValueError(f"sampled only {len(bases)} bases from indexed run {run_id}, expected {target_bases}")
        return bytes(bases)

    def _tokenize_bases(self, bases: bytes) -> torch.Tensor:
        if self.token_merge_size <= 1:
            token_ids = np.frombuffer(bases, dtype=np.uint8).astype(np.int64, copy=True)
        else:
            if self._digit_lookup is None or self._merge_weights is None:
                raise ValueError("internal error: token merge lookup was not initialized")
            raw = np.frombuffer(bases, dtype=np.uint8)
            digits = self._digit_lookup[raw]
            if np.any(digits < 0):
                raise ValueError("indexed FASTA sample contained a base outside token_merge_alphabet")
            full_digit_count = (digits.shape[0] // self.token_merge_size) * self.token_merge_size
            digits = digits[:full_digit_count]
            merged = digits.reshape(-1, self.token_merge_size).astype(np.uint64, copy=False)
            token_ids = (merged * self._merge_weights).sum(axis=1, dtype=np.uint64).astype(np.int64, copy=False)
        return torch.as_tensor(token_ids.copy(), dtype=torch.long)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
