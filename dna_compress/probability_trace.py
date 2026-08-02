from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Iterator

import numpy as np

from .tokenization import normalize_alphabet


TRACE_SCHEMA_VERSION = 1
TRACE_KIND_TARGET_PROBABILITY = "target_probability"
TRACE_EMISSION_ORDER_FUSED_DEPTH_MAJOR_V1 = "fused_depth_major_v1"
TRACE_EMISSION_ORDER_POSITION_MAJOR_V1 = "position_major_v1"
TRACE_POSITION_INDEX_SCHEMA_VERSION = 1
TRACE_POSITION_INDEX_FILENAME = "position_index.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("ascii"))


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _savez_compressed_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, **arrays)
        tmp_path.replace(path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def fused_depth_major_emit_positions(
    *,
    core_base_count: int,
    window_bases: int,
    token_merge_size: int,
) -> np.ndarray:
    if int(core_base_count) < 0:
        raise ValueError("core_base_count must be non-negative")
    if int(window_bases) <= 0:
        raise ValueError("window_bases must be positive")
    if int(token_merge_size) <= 0:
        raise ValueError("token_merge_size must be positive")
    if int(window_bases) % int(token_merge_size) != 0:
        raise ValueError("window_bases must be divisible by token_merge_size")

    tokens_per_window = int(window_bases) // int(token_merge_size)
    token_count = int(core_base_count) // int(token_merge_size)
    window_count = int(np.ceil(max(token_count, 1) / max(tokens_per_window, 1))) if token_count else 0
    positions: list[int] = []
    for token_step in range(tokens_per_window):
        active_count = 0
        for window_id in range(window_count):
            token_index = window_id * tokens_per_window + token_step
            if token_index < token_count:
                active_count += 1
        if active_count <= 0:
            continue
        for base_offset in range(int(token_merge_size)):
            depth = token_step * int(token_merge_size) + base_offset
            for window_id in range(active_count):
                position = window_id * int(window_bases) + depth
                if position < int(core_base_count):
                    positions.append(position)
    return np.asarray(positions, dtype=np.int64)


def _active_window_counts_by_token_step(
    *,
    core_base_count: int,
    window_bases: int,
    token_merge_size: int,
) -> np.ndarray:
    if int(window_bases) % int(token_merge_size) != 0:
        raise ValueError("window_bases must be divisible by token_merge_size")
    tokens_per_window = int(window_bases) // int(token_merge_size)
    token_count = int(core_base_count) // int(token_merge_size)
    if token_count <= 0:
        return np.zeros((tokens_per_window,), dtype=np.int64)
    token_steps = np.arange(tokens_per_window, dtype=np.int64)
    active = np.where(
        token_steps < token_count,
        ((token_count - 1 - token_steps) // tokens_per_window) + 1,
        0,
    )
    return active.astype(np.int64, copy=False)


def fused_depth_major_row_indices_for_positions(
    positions: np.ndarray,
    *,
    core_base_count: int,
    window_bases: int,
    token_merge_size: int,
) -> np.ndarray:
    """Map emitted base positions to trace row indices without storing positions."""

    if int(core_base_count) < 0:
        raise ValueError("core_base_count must be non-negative")
    if int(window_bases) <= 0:
        raise ValueError("window_bases must be positive")
    if int(token_merge_size) <= 0:
        raise ValueError("token_merge_size must be positive")
    if int(window_bases) % int(token_merge_size) != 0:
        raise ValueError("window_bases must be divisible by token_merge_size")

    positions_array = np.asarray(positions, dtype=np.int64)
    original_shape = positions_array.shape
    flat = positions_array.reshape(-1)
    if flat.size == 0:
        return np.empty(original_shape, dtype=np.int64)
    token_count = int(core_base_count) // int(token_merge_size)
    emitted_base_count = token_count * int(token_merge_size)
    if np.any(flat < 0) or np.any(flat >= emitted_base_count):
        raise ValueError("positions must be within emitted core bases")

    active_counts = _active_window_counts_by_token_step(
        core_base_count=int(core_base_count),
        window_bases=int(window_bases),
        token_merge_size=int(token_merge_size),
    )
    row_starts = np.concatenate(
        [
            np.zeros((1,), dtype=np.int64),
            np.cumsum(active_counts[:-1] * int(token_merge_size), dtype=np.int64),
        ]
    )

    depth = flat % int(window_bases)
    token_step = depth // int(token_merge_size)
    base_offset = depth % int(token_merge_size)
    window_id = flat // int(window_bases)
    active = active_counts[token_step]
    if np.any(window_id >= active):
        raise ValueError("positions include bases not emitted by the depth-major trace order")
    row_indices = row_starts[token_step] + base_offset * active + window_id
    return np.asarray(row_indices, dtype=np.int64).reshape(original_shape)


def trace_row_indices_for_positions(
    positions: np.ndarray,
    *,
    core_base_count: int,
    window_bases: int,
    token_merge_size: int,
    emission_order: str,
) -> np.ndarray:
    if str(emission_order) == TRACE_EMISSION_ORDER_POSITION_MAJOR_V1:
        positions_array = np.asarray(positions, dtype=np.int64)
        if np.any(positions_array < 0) or np.any(positions_array >= int(core_base_count)):
            raise ValueError("positions must be within emitted core bases")
        return positions_array.copy()
    if str(emission_order) == TRACE_EMISSION_ORDER_FUSED_DEPTH_MAJOR_V1:
        return fused_depth_major_row_indices_for_positions(
            positions,
            core_base_count=int(core_base_count),
            window_bases=int(window_bases),
            token_merge_size=int(token_merge_size),
        )
    raise ValueError(f"unsupported trace emission order: {emission_order}")


def target_symbols_for_positions(sequence: str, emit_positions: np.ndarray, alphabet: str = "ACGT") -> np.ndarray:
    alphabet = normalize_alphabet(alphabet)
    base_to_symbol = {base: index for index, base in enumerate(alphabet)}
    positions = np.asarray(emit_positions, dtype=np.int64)
    symbols = np.empty((positions.shape[0],), dtype=np.int16)
    for row, position in enumerate(positions.tolist()):
        symbols[row] = base_to_symbol[sequence[position]]
    return symbols


@dataclass(frozen=True)
class ProbabilityTraceManifest:
    schema_version: int
    trace_kind: str
    model_family: str
    model_id: str
    alphabet: str
    unit_size: int
    source_sha256: str
    normalized_sequence_sha256: str
    core_base_count: int
    tail_base_count: int
    tail_sequence_sha256: str
    target_symbols_sha256: str
    emit_order_sha256: str
    row_count: int
    dtype: str
    shard_rows: int
    window_bases: int
    token_merge_size: int
    emission_order: str
    producer_config: dict[str, Any]
    shard_files: list[str]
    checksum_sha256: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["alphabet"] = normalize_alphabet(str(payload["alphabet"]))
        return _json_safe(payload)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ProbabilityTraceManifest":
        return cls(
            schema_version=int(payload["schema_version"]),
            trace_kind=str(payload["trace_kind"]),
            model_family=str(payload["model_family"]),
            model_id=str(payload["model_id"]),
            alphabet=normalize_alphabet(str(payload["alphabet"])),
            unit_size=int(payload["unit_size"]),
            source_sha256=str(payload["source_sha256"]),
            normalized_sequence_sha256=str(payload["normalized_sequence_sha256"]),
            core_base_count=int(payload["core_base_count"]),
            tail_base_count=int(payload["tail_base_count"]),
            tail_sequence_sha256=str(payload["tail_sequence_sha256"]),
            target_symbols_sha256=str(payload["target_symbols_sha256"]),
            emit_order_sha256=str(payload["emit_order_sha256"]),
            row_count=int(payload["row_count"]),
            dtype=str(payload["dtype"]),
            shard_rows=int(payload["shard_rows"]),
            window_bases=int(payload["window_bases"]),
            token_merge_size=int(payload["token_merge_size"]),
            emission_order=str(payload["emission_order"]),
            producer_config=dict(payload.get("producer_config") or {}),
            shard_files=[str(item) for item in payload.get("shard_files", [])],
            checksum_sha256=str(payload["checksum_sha256"]),
        )


class ProbabilityTraceReader:
    def __init__(self, trace_dir: str | Path) -> None:
        self.trace_dir = Path(trace_dir)
        manifest_path = self.trace_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"trace manifest not found: {manifest_path}")
        self.manifest = ProbabilityTraceManifest.from_json_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )

    def iter_shards(self, *, verify_checksum: bool = True) -> Iterator[dict[str, np.ndarray]]:
        checksum = hashlib.sha256() if verify_checksum else None
        rows_seen = 0
        for relative in self.manifest.shard_files:
            path = self.trace_dir / relative
            if checksum is not None:
                payload = path.read_bytes()
                checksum.update(payload)
            with np.load(path) as data:
                target_prob = np.asarray(data["target_prob"], dtype=np.float64)
                target_symbol = np.asarray(data["target_symbol"], dtype=np.int16)
                if "emit_position" in data:
                    emit_position = np.asarray(data["emit_position"], dtype=np.int64)
                elif self.manifest.emission_order == TRACE_EMISSION_ORDER_POSITION_MAJOR_V1:
                    emit_position = np.arange(rows_seen, rows_seen + target_prob.shape[0], dtype=np.int64)
                else:
                    raise ValueError(f"trace shard lacks emit_position: {path}")
            rows = int(target_prob.shape[0])
            if target_symbol.shape != (rows,) or emit_position.shape != (rows,):
                raise ValueError(f"invalid shard shapes in {path}")
            rows_seen += rows
            yield {
                "target_prob": target_prob,
                "target_symbol": target_symbol,
                "emit_position": emit_position,
            }
        if rows_seen != int(self.manifest.row_count):
            raise ValueError(f"trace row_count mismatch: manifest={self.manifest.row_count}, shards={rows_seen}")
        if checksum is not None and checksum.hexdigest() != self.manifest.checksum_sha256:
            raise ValueError("trace shard checksum mismatch")


@dataclass(frozen=True)
class ProbabilityTracePositionIndexShard:
    file: str
    row_start: int
    row_end: int

    @property
    def row_count(self) -> int:
        return max(0, int(self.row_end) - int(self.row_start))


@dataclass(frozen=True)
class ProbabilityTracePositionIndex:
    schema_version: int
    trace_kind: str
    trace_dir: str
    index_path: str
    row_count: int
    core_base_count: int
    window_bases: int
    token_merge_size: int
    emission_order: str
    emit_order_sha256: str
    checksum_sha256: str
    shards: list[ProbabilityTracePositionIndexShard]

    def to_json_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": int(self.schema_version),
                "trace_kind": str(self.trace_kind),
                "trace_dir": str(self.trace_dir),
                "index_path": str(self.index_path),
                "row_count": int(self.row_count),
                "core_base_count": int(self.core_base_count),
                "window_bases": int(self.window_bases),
                "token_merge_size": int(self.token_merge_size),
                "emission_order": str(self.emission_order),
                "emit_order_sha256": str(self.emit_order_sha256),
                "checksum_sha256": str(self.checksum_sha256),
                "shards": [asdict(shard) for shard in self.shards],
            }
        )

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "ProbabilityTracePositionIndex":
        return cls(
            schema_version=int(payload["schema_version"]),
            trace_kind=str(payload["trace_kind"]),
            trace_dir=str(payload["trace_dir"]),
            index_path=str(payload["index_path"]),
            row_count=int(payload["row_count"]),
            core_base_count=int(payload["core_base_count"]),
            window_bases=int(payload["window_bases"]),
            token_merge_size=int(payload["token_merge_size"]),
            emission_order=str(payload["emission_order"]),
            emit_order_sha256=str(payload["emit_order_sha256"]),
            checksum_sha256=str(payload["checksum_sha256"]),
            shards=[
                ProbabilityTracePositionIndexShard(
                    file=str(item["file"]),
                    row_start=int(item["row_start"]),
                    row_end=int(item["row_end"]),
                )
                for item in payload.get("shards", [])
            ],
        )


def build_probability_trace_position_index(
    trace_dir: str | Path,
    *,
    overwrite: bool = False,
) -> ProbabilityTracePositionIndex:
    reader = ProbabilityTraceReader(trace_dir)
    manifest = reader.manifest
    if manifest.emission_order not in {
        TRACE_EMISSION_ORDER_FUSED_DEPTH_MAJOR_V1,
        TRACE_EMISSION_ORDER_POSITION_MAJOR_V1,
    }:
        raise ValueError(f"unsupported trace emission order: {manifest.emission_order}")

    index_path = Path(trace_dir) / TRACE_POSITION_INDEX_FILENAME
    if index_path.exists() and not overwrite:
        return read_probability_trace_position_index(trace_dir)

    shards: list[ProbabilityTracePositionIndexShard] = []
    row_start = 0
    for relative in manifest.shard_files:
        row_end = min(int(manifest.row_count), row_start + int(manifest.shard_rows))
        shards.append(
            ProbabilityTracePositionIndexShard(
                file=str(relative),
                row_start=int(row_start),
                row_end=int(row_end),
            )
        )
        row_start = row_end
    if row_start != int(manifest.row_count):
        raise ValueError(f"index row_count mismatch: {row_start} != {manifest.row_count}")

    index = ProbabilityTracePositionIndex(
        schema_version=TRACE_POSITION_INDEX_SCHEMA_VERSION,
        trace_kind=manifest.trace_kind,
        trace_dir=str(Path(trace_dir)),
        index_path=str(index_path),
        row_count=int(manifest.row_count),
        core_base_count=int(manifest.core_base_count),
        window_bases=int(manifest.window_bases),
        token_merge_size=int(manifest.token_merge_size),
        emission_order=manifest.emission_order,
        emit_order_sha256=manifest.emit_order_sha256,
        checksum_sha256=manifest.checksum_sha256,
        shards=shards,
    )
    index_path.write_text(json.dumps(index.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def read_probability_trace_position_index(trace_dir: str | Path) -> ProbabilityTracePositionIndex:
    index_path = Path(trace_dir) / TRACE_POSITION_INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"trace position index not found: {index_path}")
    return ProbabilityTracePositionIndex.from_json_dict(json.loads(index_path.read_text(encoding="utf-8")))


def validate_probability_trace_position_index(
    trace_dir: str | Path,
    index: ProbabilityTracePositionIndex,
) -> None:
    manifest = ProbabilityTraceReader(trace_dir).manifest
    mismatches: list[str] = []
    for field in [
        "trace_kind",
        "row_count",
        "core_base_count",
        "window_bases",
        "token_merge_size",
        "emission_order",
        "emit_order_sha256",
        "checksum_sha256",
    ]:
        if getattr(manifest, field) != getattr(index, field):
            mismatches.append(field)
    if mismatches:
        raise ValueError(f"trace position index is stale or incompatible: {', '.join(mismatches)}")


def _trace_shard_row_bounds(manifest: ProbabilityTraceManifest) -> tuple[np.ndarray, np.ndarray]:
    starts: list[int] = []
    ends: list[int] = []
    row_start = 0
    for _ in manifest.shard_files:
        row_end = min(int(manifest.row_count), row_start + int(manifest.shard_rows))
        starts.append(row_start)
        ends.append(row_end)
        row_start = row_end
    if row_start != int(manifest.row_count):
        raise ValueError(f"manifest shard bounds do not cover row_count: {row_start} != {manifest.row_count}")
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _read_trace_rows_by_indices(
    trace_dir: str | Path,
    manifest: ProbabilityTraceManifest,
    row_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= int(manifest.row_count)):
        raise ValueError("row indices outside trace row_count")
    target_prob = np.empty((rows.shape[0],), dtype=np.float64)
    target_symbol = np.empty((rows.shape[0],), dtype=np.int16)
    emit_position = np.empty((rows.shape[0],), dtype=np.int64)
    if rows.size == 0:
        return {
            "target_prob": target_prob,
            "target_symbol": target_symbol,
            "emit_position": emit_position,
        }

    row_starts, row_ends = _trace_shard_row_bounds(manifest)
    shard_ids = np.searchsorted(row_ends, rows, side="right")
    trace_path = Path(trace_dir)
    for shard_id in np.unique(shard_ids).tolist():
        mask = shard_ids == int(shard_id)
        shard_row_start = int(row_starts[int(shard_id)])
        local_rows = rows[mask] - shard_row_start
        with np.load(trace_path / manifest.shard_files[int(shard_id)]) as data:
            target_prob[mask] = np.asarray(data["target_prob"], dtype=np.float64)[local_rows]
            target_symbol[mask] = np.asarray(data["target_symbol"], dtype=np.int16)[local_rows]
            if "emit_position" in data:
                emit_position[mask] = np.asarray(data["emit_position"], dtype=np.int64)[local_rows]
            elif manifest.emission_order == TRACE_EMISSION_ORDER_POSITION_MAJOR_V1:
                emit_position[mask] = rows[mask]
            else:
                raise ValueError(f"trace shard lacks emit_position: {trace_path / manifest.shard_files[int(shard_id)]}")
    return {
        "target_prob": target_prob,
        "target_symbol": target_symbol,
        "emit_position": emit_position,
    }


def read_target_probability_trace_positions(
    trace_dir: str | Path,
    positions: np.ndarray | list[int],
    *,
    index: ProbabilityTracePositionIndex | None = None,
    validate_index: bool = True,
) -> dict[str, np.ndarray]:
    manifest = ProbabilityTraceReader(trace_dir).manifest
    if index is not None and validate_index:
        validate_probability_trace_position_index(trace_dir, index)

    requested_positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    row_indices = trace_row_indices_for_positions(
        requested_positions,
        core_base_count=int(manifest.core_base_count),
        window_bases=int(manifest.window_bases),
        token_merge_size=int(manifest.token_merge_size),
        emission_order=str(manifest.emission_order),
    )
    target_prob = np.empty((requested_positions.shape[0],), dtype=np.float64)
    target_symbol = np.empty((requested_positions.shape[0],), dtype=np.int16)
    emit_position = np.empty((requested_positions.shape[0],), dtype=np.int64)

    if requested_positions.size == 0:
        return {
        "target_prob": target_prob,
        "target_symbol": target_symbol,
        "emit_position": emit_position,
        "row_index": row_indices,
    }

    rows = _read_trace_rows_by_indices(trace_dir, manifest, row_indices)
    target_prob[:] = rows["target_prob"]
    target_symbol[:] = rows["target_symbol"]
    emit_position[:] = rows["emit_position"]
    if not np.array_equal(emit_position, requested_positions):
        raise ValueError("position index lookup returned mismatched emit positions")
    return {
        "target_prob": target_prob,
        "target_symbol": target_symbol,
        "emit_position": emit_position,
        "row_index": row_indices,
    }


def _array_hash_init(dtype: np.dtype, shape: tuple[int, ...]) -> hashlib._Hash:
    digest = hashlib.sha256()
    digest.update(str(np.dtype(dtype)).encode("ascii"))
    digest.update(np.asarray(shape, dtype=np.int64).tobytes())
    return digest


def convert_probability_trace_to_position_major(
    source_trace_dir: str | Path,
    output_trace_dir: str | Path,
    *,
    shard_rows: int | None = None,
    dtype: str | None = None,
    overwrite: bool = False,
    verify_checksum: bool = True,
    temp_dir: str | Path | None = None,
    store_emit_position: bool = False,
) -> ProbabilityTraceManifest:
    source = ProbabilityTraceReader(source_trace_dir)
    manifest = source.manifest
    if manifest.emission_order not in {
        TRACE_EMISSION_ORDER_FUSED_DEPTH_MAJOR_V1,
        TRACE_EMISSION_ORDER_POSITION_MAJOR_V1,
    }:
        raise ValueError(f"unsupported trace emission order: {manifest.emission_order}")

    output_path = Path(output_trace_dir)
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return ProbabilityTraceManifest.from_json_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    if output_path.exists() and not manifest_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(f"partial or non-trace output exists without manifest: {output_path}")
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_dir = output_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    row_count = int(manifest.row_count)
    prob_dtype = np.dtype(dtype or manifest.dtype)
    out_shard_rows = int(shard_rows or manifest.shard_rows)
    if out_shard_rows <= 0:
        raise ValueError("shard_rows must be positive")

    temp_parent = Path(temp_dir) if temp_dir is not None else output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.", dir=temp_parent) as tmp_name:
        tmp = Path(tmp_name)
        prob_mmap = np.memmap(tmp / "target_prob.dat", dtype=prob_dtype, mode="w+", shape=(row_count,))
        symbol_mmap = np.memmap(tmp / "target_symbol.dat", dtype=np.int16, mode="w+", shape=(row_count,))
        seen_mmap = np.memmap(tmp / "seen.dat", dtype=np.uint8, mode="w+", shape=(row_count,))
        seen_mmap[:] = 0

        rows_seen = 0
        for shard in source.iter_shards(verify_checksum=bool(verify_checksum)):
            positions = np.asarray(shard["emit_position"], dtype=np.int64)
            if np.any(positions < 0) or np.any(positions >= row_count):
                raise ValueError("source trace contains emit positions outside row_count")
            prob_mmap[positions] = np.asarray(shard["target_prob"], dtype=prob_dtype)
            symbol_mmap[positions] = np.asarray(shard["target_symbol"], dtype=np.int16)
            seen_mmap[positions] = 1
            rows_seen += int(positions.shape[0])
        if rows_seen != row_count:
            raise ValueError(f"source row_count mismatch while converting: {rows_seen} != {row_count}")
        if bool(np.any(seen_mmap != 1)):
            missing = np.flatnonzero(np.asarray(seen_mmap) != 1)[:8].tolist()
            raise ValueError(f"source trace did not cover all emitted positions; first missing={missing}")
        prob_mmap.flush()
        symbol_mmap.flush()

        shard_files: list[str] = []
        shard_checksum = hashlib.sha256()
        target_symbol_hash = _array_hash_init(np.dtype(np.int16), (row_count,))
        emit_order_hash = _array_hash_init(np.dtype(np.int64), (row_count,))
        for shard_index, start in enumerate(range(0, row_count, out_shard_rows)):
            end = min(row_count, start + out_shard_rows)
            symbols = np.asarray(symbol_mmap[start:end], dtype=np.int16)
            emit_position = np.arange(start, end, dtype=np.int64)
            target_symbol_hash.update(np.ascontiguousarray(symbols).tobytes())
            emit_order_hash.update(np.ascontiguousarray(emit_position).tobytes())
            relative = Path("shards") / f"shard_{shard_index:06d}.npz"
            shard_path = output_path / relative
            shard_arrays = {
                "target_prob": np.asarray(prob_mmap[start:end], dtype=prob_dtype),
                "target_symbol": symbols,
            }
            if bool(store_emit_position):
                shard_arrays["emit_position"] = emit_position
            _savez_compressed_atomic(shard_path, **shard_arrays)
            shard_checksum.update(shard_path.read_bytes())
            shard_files.append(str(relative))

    producer_config = dict(manifest.producer_config)
    producer_config["trace_order_conversion"] = {
        "source_trace_dir": str(Path(source_trace_dir)),
        "source_emission_order": str(manifest.emission_order),
        "output_emission_order": TRACE_EMISSION_ORDER_POSITION_MAJOR_V1,
        "store_emit_position": bool(store_emit_position),
    }
    converted = ProbabilityTraceManifest(
        schema_version=int(manifest.schema_version),
        trace_kind=str(manifest.trace_kind),
        model_family=str(manifest.model_family),
        model_id=str(manifest.model_id),
        alphabet=str(manifest.alphabet),
        unit_size=int(manifest.unit_size),
        source_sha256=str(manifest.source_sha256),
        normalized_sequence_sha256=str(manifest.normalized_sequence_sha256),
        core_base_count=int(manifest.core_base_count),
        tail_base_count=int(manifest.tail_base_count),
        tail_sequence_sha256=str(manifest.tail_sequence_sha256),
        target_symbols_sha256=target_symbol_hash.hexdigest(),
        emit_order_sha256=emit_order_hash.hexdigest(),
        row_count=row_count,
        dtype=str(prob_dtype),
        shard_rows=out_shard_rows,
        window_bases=int(manifest.window_bases),
        token_merge_size=int(manifest.token_merge_size),
        emission_order=TRACE_EMISSION_ORDER_POSITION_MAJOR_V1,
        producer_config=producer_config,
        shard_files=shard_files,
        checksum_sha256=shard_checksum.hexdigest(),
    )
    manifest_path.write_text(json.dumps(converted.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return converted


def write_target_probability_trace(
    trace_dir: str | Path,
    *,
    model_family: str,
    model_id: str,
    source_payload: bytes,
    normalized_sequence: str,
    core_sequence: str,
    tail_sequence: str,
    target_prob: np.ndarray,
    target_symbol: np.ndarray,
    emit_position: np.ndarray,
    window_bases: int,
    token_merge_size: int,
    producer_config: dict[str, Any],
    dtype: str = "float32",
    shard_rows: int = 1_000_000,
    emission_order: str = TRACE_EMISSION_ORDER_FUSED_DEPTH_MAJOR_V1,
    store_emit_position: bool = True,
    overwrite: bool = False,
) -> ProbabilityTraceManifest:
    trace_path = Path(trace_dir)
    manifest_path = trace_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"trace already exists: {manifest_path}")
    if int(shard_rows) <= 0:
        raise ValueError("shard_rows must be positive")

    target_prob = np.asarray(target_prob, dtype=np.float64)
    target_symbol = np.asarray(target_symbol, dtype=np.int16)
    emit_position = np.asarray(emit_position, dtype=np.int64)
    row_count = int(target_prob.shape[0])
    if target_symbol.shape != (row_count,) or emit_position.shape != (row_count,):
        raise ValueError("target_prob, target_symbol, and emit_position must be 1D arrays with matching length")
    if row_count != len(core_sequence):
        raise ValueError(f"row_count must equal core sequence bases: {row_count} != {len(core_sequence)}")
    if np.any(~np.isfinite(target_prob)) or np.any(target_prob < 0.0):
        raise ValueError("target probabilities must be finite and non-negative")
    if str(emission_order) not in {
        TRACE_EMISSION_ORDER_FUSED_DEPTH_MAJOR_V1,
        TRACE_EMISSION_ORDER_POSITION_MAJOR_V1,
    }:
        raise ValueError(f"unsupported trace emission order: {emission_order}")
    if not bool(store_emit_position):
        if str(emission_order) != TRACE_EMISSION_ORDER_POSITION_MAJOR_V1:
            raise ValueError("store_emit_position=False is only supported for position-major traces")
        if not np.array_equal(emit_position, np.arange(row_count, dtype=np.int64)):
            raise ValueError("position-major traces without stored emit_position must have row_index == emit_position")

    trace_path.mkdir(parents=True, exist_ok=True)
    shard_dir = trace_path / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    for old_shard in shard_dir.glob("shard_*.npz"):
        old_shard.unlink()

    prob_dtype = np.dtype(dtype)
    shard_files: list[str] = []
    checksum = hashlib.sha256()
    for shard_index, start in enumerate(range(0, row_count, int(shard_rows))):
        end = min(row_count, start + int(shard_rows))
        relative = Path("shards") / f"shard_{shard_index:06d}.npz"
        shard_path = trace_path / relative
        shard_arrays = {
            "target_prob": np.asarray(target_prob[start:end], dtype=prob_dtype),
            "target_symbol": np.asarray(target_symbol[start:end], dtype=np.int16),
        }
        if bool(store_emit_position):
            shard_arrays["emit_position"] = np.asarray(emit_position[start:end], dtype=np.int64)
        _savez_compressed_atomic(shard_path, **shard_arrays)
        shard_payload = shard_path.read_bytes()
        checksum.update(shard_payload)
        shard_files.append(str(relative))

    manifest = ProbabilityTraceManifest(
        schema_version=TRACE_SCHEMA_VERSION,
        trace_kind=TRACE_KIND_TARGET_PROBABILITY,
        model_family=str(model_family),
        model_id=str(model_id),
        alphabet="ACGT",
        unit_size=1,
        source_sha256=sha256_bytes(source_payload),
        normalized_sequence_sha256=sha256_text(normalized_sequence),
        core_base_count=len(core_sequence),
        tail_base_count=len(tail_sequence),
        tail_sequence_sha256=sha256_text(tail_sequence),
        target_symbols_sha256=sha256_array(target_symbol),
        emit_order_sha256=sha256_array(emit_position),
        row_count=row_count,
        dtype=str(prob_dtype),
        shard_rows=int(shard_rows),
        window_bases=int(window_bases),
        token_merge_size=int(token_merge_size),
        emission_order=str(emission_order),
        producer_config=dict(producer_config),
        shard_files=shard_files,
        checksum_sha256=checksum.hexdigest(),
    )
    manifest_path.write_text(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_trace_compatibility(
    left: ProbabilityTraceManifest,
    right: ProbabilityTraceManifest,
) -> list[dict[str, Any]]:
    fields = [
        "schema_version",
        "trace_kind",
        "alphabet",
        "unit_size",
        "source_sha256",
        "normalized_sequence_sha256",
        "core_base_count",
        "tail_base_count",
        "tail_sequence_sha256",
        "target_symbols_sha256",
        "emit_order_sha256",
        "row_count",
        "window_bases",
        "token_merge_size",
        "emission_order",
    ]
    diffs: list[dict[str, Any]] = []
    for field in fields:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value != right_value:
            diffs.append({"field": field, "left": left_value, "right": right_value})
    return diffs


def assert_trace_compatible(left: ProbabilityTraceManifest, right: ProbabilityTraceManifest) -> None:
    diffs = validate_trace_compatibility(left, right)
    if diffs:
        preview = ", ".join(str(item["field"]) for item in diffs[:8])
        raise ValueError(f"probability traces are not compatible: {preview}")


def _iter_aligned_chunks(
    left: ProbabilityTraceReader,
    right: ProbabilityTraceReader,
    *,
    verify_checksum: bool = True,
) -> Iterator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    left_iter = left.iter_shards(verify_checksum=verify_checksum)
    right_iter = right.iter_shards(verify_checksum=verify_checksum)
    left_chunk: dict[str, np.ndarray] | None = None
    right_chunk: dict[str, np.ndarray] | None = None
    left_offset = 0
    right_offset = 0

    while True:
        if left_chunk is None or left_offset >= int(left_chunk["target_prob"].shape[0]):
            try:
                left_chunk = next(left_iter)
                left_offset = 0
            except StopIteration:
                left_chunk = None
        if right_chunk is None or right_offset >= int(right_chunk["target_prob"].shape[0]):
            try:
                right_chunk = next(right_iter)
                right_offset = 0
            except StopIteration:
                right_chunk = None
        if left_chunk is None or right_chunk is None:
            if left_chunk is not None or right_chunk is not None:
                raise ValueError("trace shard streams ended at different row counts")
            return

        left_remaining = int(left_chunk["target_prob"].shape[0]) - left_offset
        right_remaining = int(right_chunk["target_prob"].shape[0]) - right_offset
        take = min(left_remaining, right_remaining)
        left_slice = {key: value[left_offset : left_offset + take] for key, value in left_chunk.items()}
        right_slice = {key: value[right_offset : right_offset + take] for key, value in right_chunk.items()}
        left_offset += take
        right_offset += take
        yield left_slice, right_slice


def fuse_target_probability_traces(
    left_trace_dir: str | Path,
    right_trace_dir: str | Path,
    *,
    fusion_eta: float = 0.05,
    fusion_initial_lm_weight: float = 0.5,
    verify_checksum: bool = True,
) -> dict[str, Any]:
    if not (0.0 <= float(fusion_initial_lm_weight) <= 1.0):
        raise ValueError("fusion_initial_lm_weight must be in [0, 1]")
    if not (0.0 <= float(fusion_eta) < 1.0):
        raise ValueError("fusion_eta must be in [0, 1)")

    started = perf_counter()
    left = ProbabilityTraceReader(left_trace_dir)
    right = ProbabilityTraceReader(right_trace_dir)
    compatibility_diffs = validate_trace_compatibility(left.manifest, right.manifest)
    if compatibility_diffs:
        raise ValueError(f"probability traces are not compatible: {compatibility_diffs}")

    window_bases = int(left.manifest.window_bases)
    window_count = int(np.ceil(max(left.manifest.core_base_count, 1) / max(window_bases, 1)))
    lm_weights = np.full((window_count,), float(fusion_initial_lm_weight), dtype=np.float64)
    nc_weights = 1.0 - lm_weights

    fused_bits = 0.0
    left_bits = 0.0
    right_bits = 0.0
    emitted = 0
    weight_sum = 0.0
    weight_count = 0

    eta = float(fusion_eta)
    eta_power = 1.0 - eta

    if left.manifest.emission_order == TRACE_EMISSION_ORDER_POSITION_MAJOR_V1:
        block_windows = 512
        for window_start in range(0, window_count, block_windows):
            window_end = min(window_count, window_start + block_windows)
            start_position = int(window_start * window_bases)
            end_position = min(int(left.manifest.core_base_count), int(window_end * window_bases))
            positions = np.arange(start_position, end_position, dtype=np.int64)
            left_chunk = _read_trace_rows_by_indices(left_trace_dir, left.manifest, positions)
            right_chunk = _read_trace_rows_by_indices(right_trace_dir, right.manifest, positions)
            if not np.array_equal(left_chunk["target_symbol"], right_chunk["target_symbol"]):
                raise ValueError("trace target symbols diverged inside compatible hashes")
            if not np.array_equal(left_chunk["emit_position"], right_chunk["emit_position"]):
                raise ValueError("trace emit positions diverged inside compatible hashes")
            left_prob = np.asarray(left_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
            right_prob = np.asarray(right_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
            left_bits += float((-np.log2(left_prob)).sum())
            right_bits += float((-np.log2(right_prob)).sum())
            local_window_count = int(window_end - window_start)
            local_base_count = int(end_position - start_position)
            local_windows = np.arange(local_window_count, dtype=np.int64)
            for offset in range(window_bases):
                local_rows = local_windows * window_bases + int(offset)
                valid = local_rows < local_base_count
                if not bool(np.any(valid)):
                    break
                active_rows = local_rows[valid]
                window_ids = int(window_start) + local_windows[valid]
                lm_weight = lm_weights[window_ids]
                nc_weight = nc_weights[window_ids]
                lm_target = left_prob[active_rows]
                nc_target = right_prob[active_rows]
                fused_target = np.maximum(lm_weight * lm_target + nc_weight * nc_target, 1e-300)
                fused_bits += float((-np.log2(fused_target)).sum())
                if eta > 0.0:
                    lm_new = np.power(lm_weight, eta_power) * lm_target
                    nc_new = np.power(nc_weight, eta_power) * nc_target
                else:
                    lm_new = lm_weight * lm_target
                    nc_new = nc_weight * nc_target
                denom = np.maximum(lm_new + nc_new, 1e-300)
                lm_weights[window_ids] = lm_new / denom
                nc_weights[window_ids] = nc_new / denom
            emitted += local_base_count
    else:
        for left_chunk, right_chunk in _iter_aligned_chunks(
            left,
            right,
            verify_checksum=bool(verify_checksum),
        ):
            if not np.array_equal(left_chunk["target_symbol"], right_chunk["target_symbol"]):
                raise ValueError("trace target symbols diverged inside compatible hashes")
            if not np.array_equal(left_chunk["emit_position"], right_chunk["emit_position"]):
                raise ValueError("trace emit positions diverged inside compatible hashes")
            left_prob = np.asarray(left_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
            right_prob = np.asarray(right_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
            positions = np.asarray(left_chunk["emit_position"], dtype=np.int64)
            offsets = positions % window_bases
            group_starts = np.concatenate(
                (
                    np.asarray([0], dtype=np.int64),
                    np.nonzero(offsets[1:] != offsets[:-1])[0].astype(np.int64) + 1,
                )
            )
            group_ends = np.concatenate(
                (
                    group_starts[1:],
                    np.asarray([positions.shape[0]], dtype=np.int64),
                )
            )
            for start, end in zip(group_starts.tolist(), group_ends.tolist()):
                window_ids = positions[start:end] // window_bases
                if np.unique(window_ids).shape[0] != window_ids.shape[0]:
                    for row_index, window_id in enumerate(window_ids.tolist(), start=start):
                        lm_weight = float(lm_weights[window_id])
                        nc_weight = float(nc_weights[window_id])
                        lm_target = float(left_prob[row_index])
                        nc_target = float(right_prob[row_index])
                        fused_target = max(lm_weight * lm_target + nc_weight * nc_target, 1e-300)
                        fused_bits += -float(np.log2(fused_target))
                        left_bits += -float(np.log2(lm_target))
                        right_bits += -float(np.log2(nc_target))

                        if eta > 0.0:
                            lm_new = (lm_weight ** eta_power) * lm_target
                            nc_new = (nc_weight ** eta_power) * nc_target
                        else:
                            lm_new = lm_weight * lm_target
                            nc_new = nc_weight * nc_target
                        denom = max(lm_new + nc_new, 1e-300)
                        lm_weights[window_id] = lm_new / denom
                        nc_weights[window_id] = nc_new / denom
                        emitted += 1
                    continue

                lm_weight = lm_weights[window_ids]
                nc_weight = nc_weights[window_ids]
                lm_target = left_prob[start:end]
                nc_target = right_prob[start:end]
                fused_target = np.maximum(lm_weight * lm_target + nc_weight * nc_target, 1e-300)
                fused_bits += float((-np.log2(fused_target)).sum())
                left_bits += float((-np.log2(lm_target)).sum())
                right_bits += float((-np.log2(nc_target)).sum())

                if eta > 0.0:
                    lm_new = np.power(lm_weight, eta_power) * lm_target
                    nc_new = np.power(nc_weight, eta_power) * nc_target
                else:
                    lm_new = lm_weight * lm_target
                    nc_new = nc_weight * nc_target
                denom = np.maximum(lm_new + nc_new, 1e-300)
                lm_weights[window_ids] = lm_new / denom
                nc_weights[window_ids] = nc_new / denom
                emitted += int(end) - int(start)

    weight_sum = float(lm_weights.sum())
    weight_count = int(lm_weights.shape[0])
    tail_bits = 2 * int(left.manifest.tail_base_count)
    sample_bases = int(left.manifest.core_base_count) + int(left.manifest.tail_base_count)
    elapsed = perf_counter() - started
    return {
        "codec": "fused_lm_nc_prefix",
        "trace_mode": TRACE_KIND_TARGET_PROBABILITY,
        "fusion_policy": "online_hedge_linear_target_probability_trace",
        "decodable_design": "target_probability_trace_non_arithmetic",
        "decoder_realistic": False,
        "encode_arithmetic": False,
        "arithmetic_coded_bytes": None,
        "arithmetic_bits_per_base": None,
        "arithmetic_stream_count": 0,
        "fusion_eta": float(fusion_eta),
        "fusion_initial_lm_weight": float(fusion_initial_lm_weight),
        "fusion_final_mean_left_weight": weight_sum / max(weight_count, 1),
        "fusion_final_mean_lm_weight": weight_sum / max(weight_count, 1),
        "sample_bases": sample_bases,
        "core_base_count": int(left.manifest.core_base_count),
        "tail_base_count": int(left.manifest.tail_base_count),
        "tail_side_info_bits": int(tail_bits),
        "window_count": window_count,
        "window_bases": window_bases,
        "token_merge_size": int(left.manifest.token_merge_size),
        "emitted_arithmetic_symbol_count": int(emitted),
        "theoretical_bits": float(fused_bits + tail_bits),
        "core_model_theoretical_bits": float(fused_bits),
        "theoretical_bits_per_base": float(fused_bits + tail_bits) / max(sample_bases, 1),
        "core_theoretical_bits_per_base": float(fused_bits) / max(int(left.manifest.core_base_count), 1),
        "left_model_family": left.manifest.model_family,
        "left_model_id": left.manifest.model_id,
        "left_only_theoretical_bits": float(left_bits),
        "left_only_theoretical_bits_per_base": float(left_bits) / max(int(left.manifest.core_base_count), 1),
        "right_model_family": right.manifest.model_family,
        "right_model_id": right.manifest.model_id,
        "right_only_theoretical_bits": float(right_bits),
        "right_only_theoretical_bits_per_base": float(right_bits) / max(int(left.manifest.core_base_count), 1),
        "trace_left": str(Path(left_trace_dir)),
        "trace_right": str(Path(right_trace_dir)),
        "trace_schema_version": int(left.manifest.schema_version),
        "trace_emission_order": left.manifest.emission_order,
        "compression_process_seconds": float(elapsed),
        "compression_bases_per_second": sample_bases / max(elapsed, 1e-12),
        "compression_symbols_per_second": emitted / max(elapsed, 1e-12),
    }
