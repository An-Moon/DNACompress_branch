from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .fasta_fragment_index import load_fasta_index_runtime_cache, split_run_ids
from .tokenization import normalize_alphabet, tokenize_source_array


REPACKED_SCHEMA_VERSION = 2
REPACKED_LAYOUT = "hash_partitioned_mixed_shards"
DEFAULT_REPACKED_WINDOW_DIR = Path("/data/students/Liang_junnan/opengenome2_subset/repacked_megabyte_s1024_m3_hashshard")
DEFAULT_REPACKED_SHARD_WINDOWS = 65_536
DEFAULT_REPACKED_READ_UNIT_WINDOWS = 8_192
DEFAULT_REPACKED_READ_CHUNK_WINDOWS = 8_192
DEFAULT_HASH_SHARD_COUNT = 16
DEFAULT_HASH_SHARD_SEED = 0
DEFAULT_WRITER_BUFFER_MB = 256

SCHEDULE_DTYPE = np.dtype([("shard_id", np.uint32), ("window_index", np.uint64)])


def _byte_is_alpha(byte_value: int) -> bool:
    return (65 <= byte_value <= 90) or (97 <= byte_value <= 122)


def _to_upper(byte_value: int) -> int:
    if 97 <= byte_value <= 122:
        return byte_value - 32
    return byte_value


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp_path, path)


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, path)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    work = np.asarray(values, dtype=np.uint64).copy()
    work = (work + np.uint64(0x9E3779B97F4A7C15)).astype(np.uint64, copy=False)
    work = ((work ^ (work >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)).astype(np.uint64, copy=False)
    work = ((work ^ (work >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)).astype(np.uint64, copy=False)
    return work ^ (work >> np.uint64(31))


def _window_hash(run_ids: np.ndarray, window_indices: np.ndarray, seed: int) -> np.ndarray:
    mixed = np.asarray(run_ids, dtype=np.uint64)
    mixed ^= np.asarray(window_indices, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    mixed ^= np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF)
    return _splitmix64(mixed)


def _split_hash(run_ids: np.ndarray, split_seed: int) -> np.ndarray:
    mixed = np.asarray(run_ids, dtype=np.uint64) ^ np.uint64(int(split_seed) & 0xFFFFFFFFFFFFFFFF)
    return _splitmix64(mixed) % np.uint64(1_000_000)


def _schedule_dir_name(*, split_seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> str:
    return (
        f"split_seed_{int(split_seed)}_"
        f"train_{train_ratio:g}_val_{val_ratio:g}_test_{test_ratio:g}"
    )


def _split_labels_for_run_ids(
    run_ids: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
) -> np.ndarray:
    del test_ratio
    buckets = _split_hash(run_ids, split_seed).astype(np.float64) / 1_000_000.0
    labels = np.full(run_ids.shape[0], 2, dtype=np.uint8)
    labels[buckets < train_ratio] = 0
    labels[(buckets >= train_ratio) & (buckets < train_ratio + val_ratio)] = 1
    return labels


def _read_bases_from_run(
    *,
    cache: Any,
    handles: dict[int, Any],
    run_id: int,
    file_id: int,
    base_start: int,
    target_bases: int,
    run_start_file_offset: int | None = None,
) -> bytes:
    if int(base_start) == 0 and run_start_file_offset is not None:
        anchor_file = int(run_start_file_offset)
        skip_bases = 0
    else:
        start = int(np.searchsorted(cache.anchor_run_ids, run_id, side="left"))
        end = int(np.searchsorted(cache.anchor_run_ids, run_id, side="right"))
        if start == end:
            raise ValueError(f"run {run_id} has no anchors")
        local_offsets = cache.anchor_base_offsets[start:end]
        local_index = int(np.searchsorted(local_offsets, base_start, side="right")) - 1
        if local_index < 0:
            local_index = 0
        anchor_index = start + local_index
        anchor_base = int(cache.anchor_base_offsets[anchor_index])
        anchor_file = int(cache.anchor_file_offsets[anchor_index])
        skip_bases = int(base_start) - anchor_base

    handle = handles.get(file_id)
    if handle is None or handle.closed:
        handle = cache.file_paths[file_id].open("rb")
        handles[file_id] = handle
    handle.seek(anchor_file)

    bases = bytearray()
    skipped = 0
    # Most OpenGenome2 ACGT runs are short: reading a fixed 1 MiB chunk per run
    # explodes into tens of TiB of redundant reads. Size chunks to the requested
    # run slice, while still allowing line breaks and occasional whitespace.
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
        raise ValueError(f"read only {len(bases)} bases from indexed run {run_id}, expected {target_bases}")
    return bytes(bases)


def _read_bases_from_buffer(*, buffer: bytes, byte_offset: int, target_bases: int, run_id: int) -> bytes:
    bases = bytearray()
    for byte_value in memoryview(buffer)[int(byte_offset) :]:
        upper = _to_upper(int(byte_value))
        if upper in (ord("A"), ord("C"), ord("G"), ord("T")):
            bases.append(upper)
            if len(bases) >= target_bases:
                break
        elif byte_value in (9, 10, 11, 12, 13, 32):
            continue
        elif _byte_is_alpha(int(byte_value)):
            raise ValueError(f"encountered non-ACGT base while reading indexed run {run_id}")
    if len(bases) != target_bases:
        raise ValueError(f"read only {len(bases)} bases from buffered indexed run {run_id}, expected {target_bases}")
    return bytes(bases)


def _validate_repacked_token_dtype(*, token_merge_size: int, token_merge_alphabet: str, pad_id: int) -> np.dtype:
    alphabet = normalize_alphabet(token_merge_alphabet)
    max_regular_id = len(alphabet) ** int(token_merge_size) - 1 if token_merge_size > 1 else max(map(ord, alphabet))
    max_id = max(max_regular_id, int(pad_id))
    if max_id > np.iinfo(np.uint8).max:
        raise ValueError(
            "repacked_windows v1 stores uint8 token ids; choose token_merge_size/alphabet/pad_id with max id <= 255"
        )
    return np.dtype(np.uint8)


@dataclass
class _ShardRecord:
    split: str
    source: str
    source_id: int
    shard_id: int
    window_count: int
    padded_window_count: int
    bin_path: str
    valid_tokens_path: str
    run_ids_path: str
    run_window_indices_path: str


class _ShardWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        split: str,
        source: str,
        source_id: int,
        seq_length: int,
        shard_windows: int,
        token_dtype: np.dtype,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.source = source
        self.source_id = int(source_id)
        self.seq_length = int(seq_length)
        self.shard_windows = int(shard_windows)
        self.token_dtype = np.dtype(token_dtype)
        self.shard_id = 0
        self.tokens: list[np.ndarray] = []
        self.valid_tokens: list[np.ndarray] = []
        self.run_ids: list[np.ndarray] = []
        self.run_window_indices: list[np.ndarray] = []
        self.count = 0
        self.records: list[_ShardRecord] = []

    def append(
        self,
        *,
        windows: np.ndarray,
        valid_tokens: np.ndarray,
        run_ids: np.ndarray,
        run_window_indices: np.ndarray,
    ) -> None:
        offset = 0
        total = int(windows.shape[0])
        while offset < total:
            capacity = self.shard_windows - self.count
            take = min(capacity, total - offset)
            next_offset = offset + take
            self.tokens.append(np.asarray(windows[offset:next_offset], dtype=self.token_dtype))
            self.valid_tokens.append(np.asarray(valid_tokens[offset:next_offset], dtype=np.uint16))
            self.run_ids.append(np.asarray(run_ids[offset:next_offset], dtype=np.int64))
            self.run_window_indices.append(np.asarray(run_window_indices[offset:next_offset], dtype=np.uint64))
            self.count += int(take)
            offset = next_offset
            if self.count >= self.shard_windows:
                self.flush()

    def flush(self) -> None:
        if self.count == 0:
            return
        split_source_dir = self.output_dir / self.split / self.source
        split_source_dir.mkdir(parents=True, exist_ok=True)
        stem = f"shard_{self.shard_id:06d}"
        tokens = np.concatenate(self.tokens, axis=0).astype(self.token_dtype, copy=False)
        valid = np.concatenate(self.valid_tokens, axis=0).astype(np.uint16, copy=False)
        run_ids = np.concatenate(self.run_ids, axis=0).astype(np.int64, copy=False)
        run_window_indices = np.concatenate(self.run_window_indices, axis=0).astype(np.uint64, copy=False)

        bin_path = split_source_dir / f"{stem}.bin"
        valid_path = split_source_dir / f"{stem}.valid_tokens.npy"
        run_ids_path = split_source_dir / f"{stem}.run_ids.npy"
        run_windows_path = split_source_dir / f"{stem}.run_window_indices.npy"
        _write_bytes_atomic(bin_path, tokens.reshape(-1).tobytes(order="C"))
        _save_npy_atomic(valid_path, valid)
        _save_npy_atomic(run_ids_path, run_ids)
        _save_npy_atomic(run_windows_path, run_window_indices)

        self.records.append(
            _ShardRecord(
                split=self.split,
                source=self.source,
                source_id=self.source_id,
                shard_id=self.shard_id,
                window_count=int(tokens.shape[0]),
                padded_window_count=int(np.count_nonzero(valid < self.seq_length)),
                bin_path=str(bin_path.relative_to(self.output_dir)),
                valid_tokens_path=str(valid_path.relative_to(self.output_dir)),
                run_ids_path=str(run_ids_path.relative_to(self.output_dir)),
                run_window_indices_path=str(run_windows_path.relative_to(self.output_dir)),
            )
        )
        self.shard_id += 1
        self.tokens.clear()
        self.valid_tokens.clear()
        self.run_ids.clear()
        self.run_window_indices.clear()
        self.count = 0


class _ShardWriterManager:
    def __init__(
        self,
        *,
        output_dir: Path,
        seq_length: int,
        shard_windows: int,
        token_dtype: np.dtype,
    ) -> None:
        self.output_dir = output_dir
        self.seq_length = int(seq_length)
        self.shard_windows = int(shard_windows)
        self.token_dtype = np.dtype(token_dtype)
        self.writers: dict[tuple[str, str], _ShardWriter] = {}

    def writer(self, *, split: str, source: str, source_id: int) -> _ShardWriter:
        key = (split, source)
        writer = self.writers.get(key)
        if writer is None:
            writer = _ShardWriter(
                output_dir=self.output_dir,
                split=split,
                source=source,
                source_id=source_id,
                seq_length=self.seq_length,
                shard_windows=self.shard_windows,
                token_dtype=self.token_dtype,
            )
            self.writers[key] = writer
        return writer

    def flush_source(self, source: str) -> None:
        for split, writer_source in list(self.writers):
            if writer_source == source:
                self.writers[(split, writer_source)].flush()

    def close(self) -> list[_ShardRecord]:
        records: list[_ShardRecord] = []
        for writer in self.writers.values():
            writer.flush()
            records.extend(writer.records)
        records.sort(key=lambda item: (item.split, item.source, item.shard_id))
        return records


class _HashShardWriter:
    def __init__(
        self,
        *,
        output_dir: Path,
        shard_id: int,
        seq_length: int,
        token_dtype: np.dtype,
        buffer_windows: int,
    ) -> None:
        self.output_dir = output_dir
        self.shard_id = int(shard_id)
        self.seq_length = int(seq_length)
        self.token_dtype = np.dtype(token_dtype)
        self.buffer_windows = max(1, int(buffer_windows))
        self.window_count = 0
        self.padded_window_count = 0
        self.source_counts: dict[int, int] = {}
        self.tokens: list[np.ndarray] = []
        self.source_ids: list[np.ndarray] = []
        self.valid_tokens: list[np.ndarray] = []
        self.run_ids: list[np.ndarray] = []
        self.run_window_indices: list[np.ndarray] = []
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        stem = f"shard_{self.shard_id:05d}"
        self.tokens_path = shard_dir / f"{stem}.tokens.bin"
        self.source_ids_path = shard_dir / f"{stem}.source_ids.npy"
        self.valid_tokens_path = shard_dir / f"{stem}.valid_tokens.npy"
        self.run_ids_path = shard_dir / f"{stem}.run_ids.npy"
        self.run_window_indices_path = shard_dir / f"{stem}.run_window_indices.npy"
        self.tokens_handle = self.tokens_path.open("wb")

    def append(
        self,
        *,
        windows: np.ndarray,
        source_ids: np.ndarray,
        valid_tokens: np.ndarray,
        run_ids: np.ndarray,
        run_window_indices: np.ndarray,
    ) -> None:
        if windows.shape[0] == 0:
            return
        self.tokens.append(np.asarray(windows, dtype=self.token_dtype))
        self.source_ids.append(np.asarray(source_ids, dtype=np.uint16))
        self.valid_tokens.append(np.asarray(valid_tokens, dtype=np.uint16))
        self.run_ids.append(np.asarray(run_ids, dtype=np.int64))
        self.run_window_indices.append(np.asarray(run_window_indices, dtype=np.uint64))
        if sum(chunk.shape[0] for chunk in self.tokens) >= self.buffer_windows:
            self.flush()

    def flush(self) -> None:
        if not self.tokens:
            return
        tokens = np.concatenate(self.tokens, axis=0).astype(self.token_dtype, copy=False)
        source_ids = np.concatenate(self.source_ids, axis=0).astype(np.uint16, copy=False)
        valid_tokens = np.concatenate(self.valid_tokens, axis=0).astype(np.uint16, copy=False)
        run_ids = np.concatenate(self.run_ids, axis=0).astype(np.int64, copy=False)
        run_window_indices = np.concatenate(self.run_window_indices, axis=0).astype(np.uint64, copy=False)

        self.tokens_handle.write(tokens.reshape(-1).tobytes(order="C"))
        self.window_count += int(tokens.shape[0])
        self.padded_window_count += int(np.count_nonzero(valid_tokens < self.seq_length))
        for source_id, count in zip(*np.unique(source_ids, return_counts=True)):
            self.source_counts[int(source_id)] = self.source_counts.get(int(source_id), 0) + int(count)

        self.tokens.clear()
        self.source_ids.clear()
        self.valid_tokens.clear()
        self.run_ids.clear()
        self.run_window_indices.clear()

        if not hasattr(self, "_source_chunks"):
            self._source_chunks: list[np.ndarray] = []
            self._valid_chunks: list[np.ndarray] = []
            self._run_id_chunks: list[np.ndarray] = []
            self._run_window_chunks: list[np.ndarray] = []
        self._source_chunks.append(source_ids)
        self._valid_chunks.append(valid_tokens)
        self._run_id_chunks.append(run_ids)
        self._run_window_chunks.append(run_window_indices)

    def close(self) -> dict[str, Any]:
        self.flush()
        self.tokens_handle.close()
        source_chunks = getattr(self, "_source_chunks", [])
        valid_chunks = getattr(self, "_valid_chunks", [])
        run_id_chunks = getattr(self, "_run_id_chunks", [])
        run_window_chunks = getattr(self, "_run_window_chunks", [])
        source_ids = np.concatenate(source_chunks).astype(np.uint16, copy=False) if source_chunks else np.empty(0, np.uint16)
        valid_tokens = np.concatenate(valid_chunks).astype(np.uint16, copy=False) if valid_chunks else np.empty(0, np.uint16)
        run_ids = np.concatenate(run_id_chunks).astype(np.int64, copy=False) if run_id_chunks else np.empty(0, np.int64)
        run_window_indices = (
            np.concatenate(run_window_chunks).astype(np.uint64, copy=False)
            if run_window_chunks
            else np.empty(0, np.uint64)
        )
        _save_npy_atomic(self.source_ids_path, source_ids)
        _save_npy_atomic(self.valid_tokens_path, valid_tokens)
        _save_npy_atomic(self.run_ids_path, run_ids)
        _save_npy_atomic(self.run_window_indices_path, run_window_indices)
        self._source_chunks = []
        self._valid_chunks = []
        self._run_id_chunks = []
        self._run_window_chunks = []
        return {
            "shard_id": self.shard_id,
            "window_count": int(self.window_count),
            "padded_window_count": int(self.padded_window_count),
            "tokens_path": str(self.tokens_path.relative_to(self.output_dir)),
            "source_ids_path": str(self.source_ids_path.relative_to(self.output_dir)),
            "valid_tokens_path": str(self.valid_tokens_path.relative_to(self.output_dir)),
            "run_ids_path": str(self.run_ids_path.relative_to(self.output_dir)),
            "run_window_indices_path": str(self.run_window_indices_path.relative_to(self.output_dir)),
            "source_counts": {str(key): int(value) for key, value in sorted(self.source_counts.items())},
        }


class _HashShardWriterManager:
    def __init__(
        self,
        *,
        output_dir: Path,
        seq_length: int,
        token_dtype: np.dtype,
        hash_shard_count: int,
        writer_buffer_mb: int,
    ) -> None:
        self.output_dir = output_dir
        self.seq_length = int(seq_length)
        self.token_dtype = np.dtype(token_dtype)
        self.hash_shard_count = int(hash_shard_count)
        bytes_per_window = max(1, self.seq_length * self.token_dtype.itemsize)
        buffer_windows = max(1, (int(writer_buffer_mb) * 1024 * 1024) // bytes_per_window)
        self.writers = [
            _HashShardWriter(
                output_dir=output_dir,
                shard_id=shard_id,
                seq_length=seq_length,
                token_dtype=token_dtype,
                buffer_windows=buffer_windows,
            )
            for shard_id in range(self.hash_shard_count)
        ]

    def append(
        self,
        *,
        shard_ids: np.ndarray,
        windows: np.ndarray,
        source_ids: np.ndarray,
        valid_tokens: np.ndarray,
        run_ids: np.ndarray,
        run_window_indices: np.ndarray,
    ) -> None:
        for shard_id in np.unique(shard_ids).tolist():
            mask = shard_ids == shard_id
            self.writers[int(shard_id)].append(
                windows=windows[mask],
                source_ids=source_ids[mask],
                valid_tokens=valid_tokens[mask],
                run_ids=run_ids[mask],
                run_window_indices=run_window_indices[mask],
            )

    def close(self) -> list[dict[str, Any]]:
        return [writer.close() for writer in self.writers]


def _append_split_summary(
    summary: dict[str, Any],
    *,
    split: str,
    source: str,
    windows: int,
    padded: int,
) -> None:
    split_summary = summary.setdefault(split, {"window_count": 0, "padded_window_count": 0, "sources": {}})
    split_summary["window_count"] += int(windows)
    split_summary["padded_window_count"] += int(padded)
    source_summary = split_summary["sources"].setdefault(source, {"window_count": 0, "padded_window_count": 0})
    source_summary["window_count"] += int(windows)
    source_summary["padded_window_count"] += int(padded)


def build_repacked_megabyte_windows(
    *,
    index_dir: str | Path,
    output_dir: str | Path = DEFAULT_REPACKED_WINDOW_DIR,
    seq_length: int = 1024,
    token_merge_size: int = 3,
    token_merge_alphabet: str = "ACGTN",
    pad_id: int = 125,
    shard_windows: int = DEFAULT_REPACKED_SHARD_WINDOWS,
    read_unit_windows: int = DEFAULT_REPACKED_READ_UNIT_WINDOWS,
    hash_shard_count: int = DEFAULT_HASH_SHARD_COUNT,
    hash_shard_seed: int = DEFAULT_HASH_SHARD_SEED,
    writer_buffer_mb: int = DEFAULT_WRITER_BUFFER_MB,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    split_seed: int = 0,
    overwrite: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if token_merge_size <= 0:
        raise ValueError("token_merge_size must be > 0")
    if hash_shard_count <= 0:
        raise ValueError("hash_shard_count must be > 0")
    if read_unit_windows <= 0:
        raise ValueError("read_unit_windows must be > 0")
    if writer_buffer_mb <= 0:
        raise ValueError("writer_buffer_mb must be > 0")
    token_dtype = _validate_repacked_token_dtype(
        token_merge_size=token_merge_size,
        token_merge_alphabet=token_merge_alphabet,
        pad_id=pad_id,
    )
    output = Path(output_dir)
    if output.exists():
        if overwrite:
            shutil.rmtree(output)
        elif (output / "manifest.json").exists() or any(output.iterdir()):
            raise FileExistsError(f"repacked output already exists; pass --overwrite to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cache = load_fasta_index_runtime_cache(index_dir)
    run_ids_all = np.asarray(cache.run_ids)
    manager = _HashShardWriterManager(
        output_dir=output,
        seq_length=seq_length,
        token_dtype=token_dtype,
        hash_shard_count=hash_shard_count,
        writer_buffer_mb=writer_buffer_mb,
    )
    source_summary: dict[str, dict[str, int]] = {}
    handles: dict[int, Any] = {}
    started = time.time()
    total_windows = 0
    total_padded = 0
    processed_runs = 0

    try:
        run_lengths = np.asarray(cache.run_lengths)
        source_ids = np.asarray(cache.run_source_ids)
        file_ids = np.asarray(cache.run_file_ids)
        run_start_file_offsets = np.asarray(cache.run_start_file_offsets)
        file_sizes: dict[int, int] = {}
        total_run_count = int(run_ids_all.shape[0])
        max_batch_bytes = 64 * 1024 * 1024

        def file_size(file_id: int) -> int:
            cached = file_sizes.get(file_id)
            if cached is None:
                cached = int(cache.file_paths[file_id].stat().st_size)
                file_sizes[file_id] = cached
            return cached

        def run_end_file_offset(run_position: int) -> int:
            file_id = int(file_ids[run_position])
            next_position = run_position + 1
            if next_position < total_run_count and int(file_ids[next_position]) == file_id:
                return int(run_start_file_offsets[next_position])
            return file_size(file_id)

        def emit_progress() -> None:
            if progress_callback is not None and processed_runs % 50_000 == 0:
                progress_callback(
                    {
                        "processed_runs": processed_runs,
                        "total_runs": total_run_count,
                        "window_count": total_windows,
                        "padded_window_count": total_padded,
                        "elapsed_seconds": time.time() - started,
                    }
                )

        def append_token_chunk(
            *,
            run_position: int,
            start_window: int,
            take_windows: int,
            full_tokens: int,
            token_ids: np.ndarray,
        ) -> None:
            nonlocal total_windows, total_padded

            run_id = int(run_ids_all[run_position])
            source_id = int(source_ids[run_position])
            source = str(cache.source_names[source_id])
            windows = np.full((take_windows, seq_length), int(pad_id), dtype=token_dtype)
            valid_tokens = np.empty((take_windows,), dtype=np.uint16)
            for offset in range(take_windows):
                original_window = start_window + offset
                token_start = offset * seq_length
                token_count = min(seq_length, full_tokens - original_window * seq_length)
                if token_count <= 0:
                    raise ValueError("internal error: generated empty repacked window")
                windows[offset, :token_count] = token_ids[token_start : token_start + token_count]
                valid_tokens[offset] = int(token_count)

            run_id_array = np.full((take_windows,), run_id, dtype=np.int64)
            run_window_indices = np.arange(start_window, start_window + take_windows, dtype=np.uint64)
            shard_ids = (
                _window_hash(run_id_array, run_window_indices, hash_shard_seed) % np.uint64(hash_shard_count)
            ).astype(np.int64)
            source_id_array = np.full((take_windows,), source_id, dtype=np.uint16)
            manager.append(
                shard_ids=shard_ids,
                windows=windows,
                source_ids=source_id_array,
                valid_tokens=valid_tokens,
                run_ids=run_id_array,
                run_window_indices=run_window_indices,
            )
            padded = int(np.count_nonzero(valid_tokens < seq_length))
            total_windows += int(take_windows)
            total_padded += padded
            summary = source_summary.setdefault(source, {"window_count": 0, "padded_window_count": 0})
            summary["window_count"] += int(take_windows)
            summary["padded_window_count"] += int(padded)

        def process_single_run(run_position: int) -> None:
            nonlocal processed_runs

            full_tokens = int(run_lengths[run_position]) // int(token_merge_size)
            if full_tokens <= 0:
                return
            run_id = int(run_ids_all[run_position])
            file_id = int(file_ids[run_position])
            window_count = int((full_tokens + seq_length - 1) // seq_length)
            for start_window in range(0, window_count, read_unit_windows):
                take_windows = min(int(read_unit_windows), window_count - start_window)
                first_token = start_window * seq_length
                last_window = start_window + take_windows - 1
                end_token = min((last_window + 1) * seq_length, full_tokens)
                tokens_to_read = end_token - first_token
                if tokens_to_read <= 0:
                    continue
                bases = _read_bases_from_run(
                    cache=cache,
                    handles=handles,
                    run_id=run_id,
                    file_id=file_id,
                    base_start=first_token * token_merge_size,
                    target_bases=tokens_to_read * token_merge_size,
                    run_start_file_offset=int(run_start_file_offsets[run_position]),
                )
                token_ids = tokenize_source_array(bases, token_merge_size, token_merge_alphabet).astype(
                    token_dtype,
                    copy=False,
                )
                append_token_chunk(
                    run_position=run_position,
                    start_window=start_window,
                    take_windows=take_windows,
                    full_tokens=full_tokens,
                    token_ids=token_ids,
                )
            processed_runs += 1
            emit_progress()

        def process_buffered_runs(group_positions: list[int], buffer: bytes, buffer_start: int) -> None:
            nonlocal processed_runs

            for group_position in group_positions:
                full_tokens = int(run_lengths[group_position]) // int(token_merge_size)
                if full_tokens <= 0:
                    continue
                run_id = int(run_ids_all[group_position])
                window_count = int((full_tokens + seq_length - 1) // seq_length)
                bases = _read_bases_from_buffer(
                    buffer=buffer,
                    byte_offset=int(run_start_file_offsets[group_position]) - buffer_start,
                    target_bases=full_tokens * int(token_merge_size),
                    run_id=run_id,
                )
                token_ids = tokenize_source_array(bases, token_merge_size, token_merge_alphabet).astype(
                    token_dtype,
                    copy=False,
                )
                append_token_chunk(
                    run_position=group_position,
                    start_window=0,
                    take_windows=window_count,
                    full_tokens=full_tokens,
                    token_ids=token_ids,
                )
                processed_runs += 1
                emit_progress()

        run_position = 0
        while run_position < total_run_count:
            full_tokens = int(run_lengths[run_position]) // int(token_merge_size)
            if full_tokens <= 0:
                run_position += 1
                continue
            window_count = int((full_tokens + seq_length - 1) // seq_length)
            if window_count > int(read_unit_windows):
                process_single_run(run_position)
                run_position += 1
                continue

            file_id = int(file_ids[run_position])
            first_start = int(run_start_file_offsets[run_position])
            group_positions: list[int] = []
            group_end = first_start
            cursor = run_position
            while cursor < total_run_count and int(file_ids[cursor]) == file_id:
                cursor_full_tokens = int(run_lengths[cursor]) // int(token_merge_size)
                if cursor_full_tokens <= 0:
                    cursor += 1
                    continue
                cursor_window_count = int((cursor_full_tokens + seq_length - 1) // seq_length)
                if cursor_window_count > int(read_unit_windows):
                    break
                candidate_end = run_end_file_offset(cursor)
                if group_positions and candidate_end - first_start > max_batch_bytes:
                    break
                if not group_positions and candidate_end - first_start > max_batch_bytes:
                    break
                group_positions.append(cursor)
                group_end = candidate_end
                cursor += 1

            if len(group_positions) > 1:
                handle = handles.get(file_id)
                if handle is None or handle.closed:
                    handle = cache.file_paths[file_id].open("rb")
                    handles[file_id] = handle
                handle.seek(first_start)
                buffer = handle.read(group_end - first_start)
                process_buffered_runs(group_positions, buffer, first_start)
                run_position = group_positions[-1] + 1
            else:
                process_single_run(run_position)
                run_position += 1
    finally:
        for handle in handles.values():
            handle.close()

    shards = manager.close()
    schedule_dir = build_repacked_split_schedule(
        repacked_dir=output,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
    )
    schedule_summary = json.loads((schedule_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": REPACKED_SCHEMA_VERSION,
        "format": "dna_compress.repacked_megabyte_windows",
        "layout": REPACKED_LAYOUT,
        "index_dir": str(Path(index_dir)),
        "seq_length": int(seq_length),
        "token_merge_size": int(token_merge_size),
        "token_merge_alphabet": normalize_alphabet(token_merge_alphabet),
        "pad_id": int(pad_id),
        "token_dtype": str(np.dtype(token_dtype).name),
        "hash_shard_count": int(hash_shard_count),
        "hash_shard_seed": int(hash_shard_seed),
        "writer_buffer_mb": int(writer_buffer_mb),
        "legacy_shard_windows_arg": int(shard_windows),
        "read_unit_windows": int(read_unit_windows),
        "split_seed": int(split_seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "window_count": int(total_windows),
        "padded_window_count": int(total_padded),
        "source_names": list(cache.source_names),
        "sources": source_summary,
        "shards": shards,
        "default_schedule_dir": str(schedule_dir.relative_to(output)),
        "default_schedule": schedule_summary,
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def build_repacked_split_schedule(
    *,
    repacked_dir: str | Path,
    train_ratio: float = 0.98,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01,
    split_seed: int = 0,
    overwrite: bool = True,
) -> Path:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("split ratios must sum to 1.0")
    root = Path(repacked_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards = manifest.get("shards", [])
    else:
        shard_dir = root / "shards"
        shards = [
            {
                "shard_id": int(path.name.split("_")[1].split(".")[0]),
                "run_ids_path": str(path.relative_to(root)),
            }
            for path in sorted(shard_dir.glob("shard_*.run_ids.npy"))
        ]
        for shard in shards:
            shard_id = int(shard["shard_id"])
            stem = f"shards/shard_{shard_id:05d}"
            shard["run_window_indices_path"] = f"{stem}.run_window_indices.npy"
    schedule_dir = root / "schedules" / _schedule_dir_name(
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    if schedule_dir.exists() and overwrite:
        shutil.rmtree(schedule_dir)
    elif schedule_dir.exists():
        raise FileExistsError(f"schedule already exists: {schedule_dir}")
    schedule_dir.mkdir(parents=True, exist_ok=True)

    split_entries: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_names = ("train", "val", "test")
    counts = {name: 0 for name in split_names}
    for shard in shards:
        shard_id = int(shard["shard_id"])
        run_ids = np.load(root / str(shard["run_ids_path"]), mmap_mode="r")
        labels = _split_labels_for_run_ids(
            np.asarray(run_ids),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            split_seed=split_seed,
        )
        local_indices = np.arange(run_ids.shape[0], dtype=np.uint64)
        for label, split_name in enumerate(split_names):
            selected = local_indices[labels == label]
            if selected.size == 0:
                continue
            entries = np.empty(selected.shape[0], dtype=SCHEDULE_DTYPE)
            entries["shard_id"] = np.uint32(shard_id)
            entries["window_index"] = selected
            split_entries[split_name].append(entries)
            counts[split_name] += int(selected.shape[0])

    for split_name in split_names:
        if split_entries[split_name]:
            merged = np.concatenate(split_entries[split_name])
        else:
            merged = np.empty(0, dtype=SCHEDULE_DTYPE)
        _save_npy_atomic(schedule_dir / f"{split_name}.npy", merged)

    summary = {
        "schema_version": REPACKED_SCHEMA_VERSION,
        "layout": "run_hash_split_schedule",
        "split_seed": int(split_seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "counts": counts,
    }
    (schedule_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return schedule_dir


@dataclass(frozen=True)
class _ShardInfo:
    shard_id: int
    window_count: int
    padded_window_count: int
    tokens_path: Path
    source_ids_path: Path
    valid_tokens_path: Path
    run_ids_path: Path
    run_window_indices_path: Path


class RepackedMegabyteWindowDataset(IterableDataset):
    def __init__(
        self,
        *,
        repacked_dir: str | Path,
        split: str,
        seq_length: int,
        token_merge_size: int,
        token_merge_alphabet: str,
        pad_id: int,
        samples: int | None,
        seed: int,
        read_chunk_windows: int = DEFAULT_REPACKED_READ_CHUNK_WINDOWS,
        epoch_mode: str = "samples",
        schedule_dir: str | Path | None = None,
        shard_load_mode: str = "mmap",
        shard_sampling_mode: str = "random",
        source_sampling_weights: dict[str, float] | None = None,
        source_loss_weights: dict[str, float] | None = None,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        if read_chunk_windows <= 0:
            raise ValueError("read_chunk_windows must be > 0")
        if epoch_mode not in {"samples", "all_windows"}:
            raise ValueError("epoch_mode must be one of: samples, all_windows")
        if shard_load_mode not in {"mmap", "preload"}:
            raise ValueError("shard_load_mode must be one of: mmap, preload")
        if shard_sampling_mode not in {"random", "all_shards"}:
            raise ValueError("shard_sampling_mode must be one of: random, all_shards")
        if source_sampling_weights:
            raise ValueError("repacked hash-shard layout uses source_loss_weights, not source_sampling_weights")
        if ddp_rank < 0 or ddp_world_size <= 0 or ddp_rank >= ddp_world_size:
            raise ValueError("invalid DDP rank/world size for repacked windows")
        self.root = Path(repacked_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"repacked manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("schema_version", -1)) != REPACKED_SCHEMA_VERSION:
            raise ValueError(f"unsupported repacked schema version: {self.manifest.get('schema_version')}")
        if self.manifest.get("layout") != REPACKED_LAYOUT:
            raise ValueError(f"unsupported repacked layout: {self.manifest.get('layout')}")
        if int(self.manifest["seq_length"]) != int(seq_length):
            raise ValueError("repacked seq_length does not match config.model.seq_length")
        if int(self.manifest["token_merge_size"]) != int(token_merge_size):
            raise ValueError("repacked token_merge_size does not match config.data.token_merge_size")
        if normalize_alphabet(str(self.manifest["token_merge_alphabet"])) != normalize_alphabet(token_merge_alphabet):
            raise ValueError("repacked token_merge_alphabet does not match config.data.token_merge_alphabet")
        if int(self.manifest["pad_id"]) != int(pad_id):
            raise ValueError("repacked pad_id does not match config.model.pad_id")
        if str(self.manifest["token_dtype"]) != "uint8":
            raise ValueError("repacked_windows loader expects uint8 token shards")

        self.split = split
        self.seq_length = int(seq_length)
        self.token_merge_size = int(token_merge_size)
        self.pad_id = int(pad_id)
        self.seed = int(seed)
        self.read_chunk_windows = int(read_chunk_windows)
        self.epoch_mode = epoch_mode
        self.shard_load_mode = shard_load_mode
        self.shard_sampling_mode = shard_sampling_mode
        self.ddp_rank = int(ddp_rank)
        self.ddp_world_size = int(ddp_world_size)
        self.source_loss_weights_config = dict(source_loss_weights or {})
        self.source_names = list(self.manifest.get("source_names", []))

        self.shards: list[_ShardInfo] = []
        for raw in self.manifest.get("shards", []):
            self.shards.append(
                _ShardInfo(
                    shard_id=int(raw["shard_id"]),
                    window_count=int(raw["window_count"]),
                    padded_window_count=int(raw["padded_window_count"]),
                    tokens_path=self.root / str(raw["tokens_path"]),
                    source_ids_path=self.root / str(raw["source_ids_path"]),
                    valid_tokens_path=self.root / str(raw["valid_tokens_path"]),
                    run_ids_path=self.root / str(raw["run_ids_path"]),
                    run_window_indices_path=self.root / str(raw["run_window_indices_path"]),
                )
            )
        if not self.shards:
            raise ValueError("repacked dataset has no shards")
        self.shards.sort(key=lambda shard: shard.shard_id)

        self.shard_window_counts = np.asarray([shard.window_count for shard in self.shards], dtype=np.int64)
        self.shard_ids = np.asarray([shard.shard_id for shard in self.shards], dtype=np.int32)
        self.shard_id_to_position = {int(shard.shard_id): index for index, shard in enumerate(self.shards)}
        schedule_root = Path(schedule_dir) if schedule_dir is not None else self.root / str(self.manifest["default_schedule_dir"])
        schedule_path = schedule_root / f"{split}.npy"
        if not schedule_path.exists():
            raise FileNotFoundError(f"repacked schedule not found: {schedule_path}")
        self.schedule_dir = schedule_root
        self.schedule = np.load(schedule_path, mmap_mode="r")
        if self.schedule.dtype != SCHEDULE_DTYPE:
            raise ValueError(f"unsupported repacked schedule dtype: {self.schedule.dtype}")
        self.total_windows = int(self.schedule.shape[0])
        self.samples = self.total_windows if samples is None else int(samples)
        if self.samples <= 0:
            raise ValueError("samples must be > 0")
        if self.total_windows <= 0:
            raise ValueError(f"repacked schedule has no windows for split={split!r}")
        self._valid_token_arrays: dict[int, np.ndarray] = {}
        self._source_id_arrays: dict[int, np.ndarray] = {}
        self._token_arrays: dict[int, np.ndarray] = {}
        self._preloaded_tokens: dict[int, np.ndarray] = {}
        self._build_schedule_index()
        self._configure_loss_weights(source_loss_weights)

    def _build_schedule_index(self) -> None:
        shard_ids = np.asarray(self.schedule["shard_id"], dtype=np.int32)
        unique, starts, counts = np.unique(shard_ids, return_index=True, return_counts=True)
        order = np.argsort(starts)
        self.schedule_shard_ids = unique[order].astype(np.int32, copy=False)
        self.schedule_shard_starts = starts[order].astype(np.int64, copy=False)
        self.schedule_shard_counts = counts[order].astype(np.int64, copy=False)
        self.schedule_shard_probabilities = self.schedule_shard_counts.astype(np.float64)
        self.schedule_shard_probabilities /= self.schedule_shard_probabilities.sum()

    def _configure_loss_weights(self, source_loss_weights: dict[str, float] | None) -> None:
        self.loss_weights_by_source_id = np.ones((len(self.source_names),), dtype=np.float32)
        self.loss_weight_summary: dict[str, float] = {}
        if source_loss_weights is None or not source_loss_weights:
            return

        source_to_id = {source: idx for idx, source in enumerate(self.source_names)}
        unknown = sorted(set(source_loss_weights) - set(source_to_id))
        if unknown:
            raise ValueError(f"source loss weights include unknown sources: {', '.join(unknown)}")
        raw_total = float(sum(float(value) for value in source_loss_weights.values()))
        if raw_total <= 0:
            raise ValueError("source loss weights must sum to > 0")
        source_masses = np.zeros((len(self.source_names),), dtype=np.float64)
        for shard_id, start, count in zip(
            self.schedule_shard_ids.tolist(),
            self.schedule_shard_starts.tolist(),
            self.schedule_shard_counts.tolist(),
        ):
            shard_pos = self.shard_id_to_position[int(shard_id)]
            local_window_indices = np.asarray(self.schedule["window_index"][start : start + count], dtype=np.int64)
            source_ids = np.asarray(self._source_ids_for_shard(shard_pos)[local_window_indices], dtype=np.int64)
            source_masses += np.bincount(
                source_ids,
                minlength=len(self.source_names),
            ).astype(np.float64) * float(self.seq_length)
        total_mass = float(source_masses.sum())
        multipliers = np.zeros((len(self.source_names),), dtype=np.float32)
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
            self.source_names[source_id]: float(multiplier)
            for source_id, multiplier in enumerate(multipliers.tolist())
            if multiplier > 0
        }

    def __len__(self) -> int:
        return int(math.ceil(self.samples / self.ddp_world_size))

    def summary(self) -> dict[str, Any]:
        windows_by_source: dict[str, int] = {}
        padded_by_source: dict[str, int] = {}
        for shard_id, start, count in zip(
            self.schedule_shard_ids.tolist(),
            self.schedule_shard_starts.tolist(),
            self.schedule_shard_counts.tolist(),
        ):
            shard_pos = self.shard_id_to_position[int(shard_id)]
            local_indices = np.asarray(self.schedule["window_index"][start : start + count], dtype=np.int64)
            source_ids = np.asarray(self._source_ids_for_shard(shard_pos)[local_indices], dtype=np.int64)
            valid = np.asarray(self._valid_tokens_for_shard(shard_pos)[local_indices], dtype=np.uint16)
            for source_id, source_count in zip(*np.unique(source_ids, return_counts=True)):
                source = self.source_names[int(source_id)]
                windows_by_source[source] = windows_by_source.get(source, 0) + int(source_count)
            padded_ids = source_ids[valid < self.seq_length]
            if padded_ids.size:
                for source_id, source_count in zip(*np.unique(padded_ids, return_counts=True)):
                    source = self.source_names[int(source_id)]
                    padded_by_source[source] = padded_by_source.get(source, 0) + int(source_count)
        return {
            "source_mode": "repacked_windows",
            "repacked_dir": str(self.root),
            "schedule_dir": str(self.schedule_dir),
            "split": self.split,
            "epoch_mode": self.epoch_mode,
            "layout": REPACKED_LAYOUT,
            "shard_load_mode": self.shard_load_mode,
            "shard_sampling_mode": self.shard_sampling_mode,
            "seq_length": self.seq_length,
            "token_merge_size": self.token_merge_size,
            "samples": self.samples,
            "window_count": self.total_windows,
            "shard_count": len(self.shards),
            "read_chunk_windows": self.read_chunk_windows,
            "windows_by_source": windows_by_source,
            "padded_windows_by_source": padded_by_source,
            "source_loss_weights": dict(self.loss_weight_summary),
        }

    def _valid_tokens_for_shard(self, shard_id: int) -> np.ndarray:
        valid = self._valid_token_arrays.get(shard_id)
        if valid is None:
            valid = np.load(self.shards[shard_id].valid_tokens_path, mmap_mode="r")
            self._valid_token_arrays[shard_id] = valid
        return valid

    def _source_ids_for_shard(self, shard_id: int) -> np.ndarray:
        source_ids = self._source_id_arrays.get(shard_id)
        if source_ids is None:
            source_ids = np.load(self.shards[shard_id].source_ids_path, mmap_mode="r")
            self._source_id_arrays[shard_id] = source_ids
        return source_ids

    def _tokens_for_shard(self, shard_id: int) -> np.ndarray:
        tokens = self._token_arrays.get(shard_id)
        if tokens is None:
            shard = self.shards[shard_id]
            if self.shard_load_mode == "preload":
                tokens = np.fromfile(shard.tokens_path, dtype=np.uint8).reshape(shard.window_count, self.seq_length)
            else:
                tokens = np.memmap(shard.tokens_path, mode="r", dtype=np.uint8, shape=(shard.window_count, self.seq_length))
            self._token_arrays[shard_id] = tokens
        return tokens

    def _item_for(self, *, shard_pos: int, window_index: int) -> dict[str, torch.Tensor]:
        source_id = int(self._source_ids_for_shard(shard_pos)[window_index])
        token_row = np.asarray(self._tokens_for_shard(shard_pos)[window_index], dtype=np.uint8)
        item: dict[str, torch.Tensor] = {
            "input_ids": torch.as_tensor(token_row.astype(np.int64, copy=True)),
            "source_id": torch.tensor(source_id, dtype=torch.long),
        }
        if self.source_loss_weights_config:
            item["loss_weight"] = torch.tensor(float(self.loss_weights_by_source_id[source_id]), dtype=torch.float32)
        return item

    def _yield_schedule_slice(self, *, shard_id: int, start: int, count: int, rng: random.Random, limit: int | None = None):
        take = int(count) if limit is None else min(int(limit), int(count))
        local_offsets = list(range(int(count)))
        rng.shuffle(local_offsets)
        shard_pos = self.shard_id_to_position[int(shard_id)]
        window_indices = self.schedule["window_index"][start : start + count]
        for offset in local_offsets[:take]:
            yield self._item_for(shard_pos=shard_pos, window_index=int(window_indices[offset]))

    def _choose_schedule_shard(self, rng: random.Random) -> int:
        return int(
            rng.choices(
                range(self.schedule_shard_ids.shape[0]),
                weights=self.schedule_shard_probabilities.tolist(),
                k=1,
            )[0]
        )

    def __iter__(self):
        worker = get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        num_workers = 1 if worker is None else int(worker.num_workers)
        if self.shard_load_mode == "preload" and num_workers > 1:
            raise ValueError("repacked preload mode requires num_workers=0 to avoid duplicating large shards")
        consumer_id = self.ddp_rank * num_workers + worker_id
        total_consumers = self.ddp_world_size * num_workers
        rng = random.Random(self.seed + consumer_id * 1_000_003)

        if self.epoch_mode == "all_windows":
            shard_order = np.random.default_rng(self.seed).permutation(self.schedule_shard_ids.shape[0])
            global_position = 0
            for order_index in shard_order.tolist():
                shard_id = int(self.schedule_shard_ids[order_index])
                start = int(self.schedule_shard_starts[order_index])
                count = int(self.schedule_shard_counts[order_index])
                local_rng = random.Random(self.seed + shard_id * 97_003)
                offsets = list(range(count))
                local_rng.shuffle(offsets)
                shard_pos = self.shard_id_to_position[shard_id]
                window_indices = self.schedule["window_index"][start : start + count]
                for offset in offsets:
                    if global_position % total_consumers == consumer_id:
                        yield self._item_for(shard_pos=shard_pos, window_index=int(window_indices[offset]))
                    global_position += 1
            return

        produced = 0
        target = (self.samples + total_consumers - 1 - consumer_id) // total_consumers
        while produced < target:
            order_index = self._choose_schedule_shard(rng)
            shard_id = int(self.schedule_shard_ids[order_index])
            start = int(self.schedule_shard_starts[order_index])
            count = int(self.schedule_shard_counts[order_index])
            limit = min(target - produced, count)
            for item in self._yield_schedule_slice(
                shard_id=shard_id,
                start=start,
                count=count,
                rng=rng,
                limit=limit,
            ):
                yield item
            produced += int(limit)

    def close(self) -> None:
        self._token_arrays.clear()
        self._valid_token_arrays.clear()
        self._source_id_arrays.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
