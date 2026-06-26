from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .compression import baseline_sizes, resolve_arithmetic_coding_metadata
from .compression_eval import autocast_context
from .config import load_experiment_config
from .fast_arithmetic import (
    BatchedStreamingArithmeticDecoder,
    BatchedStreamingArithmeticEncoder,
    fast_floor_intervals_from_probabilities,
)
from .megabyte_batched_decode import MegabyteBatchedDecodeStepper, fast_floor_frequency_rows
from .megabyte_loader import build_model, load_megabyte_checkpoint
from .tokenization import tokenize_source_bytes


WINDOW_CODEC_FORMAT_VERSION = 1
WINDOW_CODEC_V2_FORMAT_VERSION = 2
WINDOW_CODEC_V3_FORMAT_VERSION = 3
WINDOW_CODEC_NAME = "megabyte_window_fast_floor"
LENGTH_PREFIX_BYTES = 4
V2_MAGIC = b"MBW2DNA\n"
V2_HEADER_PREFIX_BYTES = 8
V3_MAGIC = b"MBW3DNA\n"
V3_HEADER_UINT64_FIELDS = (
    "tokens_per_window",
    "window_count",
    "logical_token_count",
    "base_token_count",
    "token_merge_size",
    "frequency_total",
    "compression_batch_size",
)


@dataclass(frozen=True)
class TokenWindowBatch:
    window_start: int
    tokens: torch.Tensor
    valid_lengths: torch.Tensor | None = None


@dataclass
class TokenStreamMetadata:
    original_token_count: int = 0
    symbol_count_without_eos: int = 0
    tail_padding_tokens: int = 0
    eos_appended: bool = False
    read_bytes: int = 0
    filtered_bytes: int = 0
    emitted_windows: int = 0


def resolve_device_names(device_names: list[str] | tuple[str, ...] | None, *, fallback: str | torch.device) -> list[str]:
    resolved = [str(torch.device(name)) for name in device_names] if device_names else [str(torch.device(fallback))]
    cuda_count: int | None = None
    for name in resolved:
        device = torch.device(name)
        if device.type != "cuda":
            continue
        if cuda_count is None:
            cuda_count = int(torch.cuda.device_count())
        if cuda_count <= 0:
            raise ValueError(f"CUDA device requested ({name}) but torch.cuda.device_count() is 0.")
        index = 0 if device.index is None else int(device.index)
        if index < 0 or index >= cuda_count:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            hint = (
                f" With CUDA_VISIBLE_DEVICES={visible!r}, use process-local ids "
                f"cuda:0..cuda:{cuda_count - 1}."
                if visible
                else ""
            )
            raise ValueError(
                f"CUDA device {name} is not visible in this process "
                f"(torch.cuda.device_count()={cuda_count}).{hint}"
            )
    return resolved


def batch_aligned_window_ranges(window_count: int, batch_size: int, shard_count: int) -> list[tuple[int, int]]:
    if window_count < 0:
        raise ValueError("window_count must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")
    if window_count == 0:
        return []

    batch_count = math.ceil(int(window_count) / int(batch_size))
    active_shards = min(int(shard_count), batch_count)
    ranges: list[tuple[int, int]] = []
    for shard_index in range(active_shards):
        start_batch = (shard_index * batch_count) // active_shards
        end_batch = ((shard_index + 1) * batch_count) // active_shards
        start = start_batch * int(batch_size)
        end = min(end_batch * int(batch_size), int(window_count))
        if start < end:
            ranges.append((start, end))
    return ranges


def valid_lengths_from_logical_token_count(
    *,
    window_count: int,
    tokens_per_window: int,
    logical_token_count: int,
) -> torch.Tensor:
    window_count = int(window_count)
    tokens_per_window = int(tokens_per_window)
    logical_token_count = int(logical_token_count)
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    if tokens_per_window <= 0:
        raise ValueError("tokens_per_window must be positive")
    if logical_token_count <= 0 or logical_token_count > window_count * tokens_per_window:
        raise ValueError("logical_token_count must be in [1, window_count * tokens_per_window]")
    lengths = torch.full((window_count,), tokens_per_window, dtype=torch.long)
    tail = logical_token_count - (window_count - 1) * tokens_per_window
    if tail <= 0 or tail > tokens_per_window:
        raise ValueError("logical_token_count is inconsistent with window_count")
    lengths[-1] = int(tail)
    return lengths


def _sum_float(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key, 0.0) or 0.0) for row in rows))


def _sum_int(rows: list[dict[str, Any]], key: str) -> int:
    return int(sum(int(row.get(key, 0) or 0) for row in rows))


def _aggregate_compression_metrics(
    shard_metrics: list[dict[str, Any]],
    *,
    token_count: int,
    batch_size: int,
    mode: str,
    devices: list[str],
    wall_seconds: float,
) -> dict[str, Any]:
    encoded_payload_bytes = _sum_int(shard_metrics, "compression_encoded_payload_bytes")
    emitted = _sum_int(shard_metrics, "compression_emitted_symbols")
    result: dict[str, Any] = {
        "compression_batch_size": int(batch_size),
        "compression_mode": mode,
        "compression_wall_seconds": float(wall_seconds),
        "compression_model_seconds": _sum_float(shard_metrics, "compression_model_seconds"),
        "compression_quantize_seconds": _sum_float(shard_metrics, "compression_quantize_seconds"),
        "compression_interval_transfer_seconds": _sum_float(shard_metrics, "compression_interval_transfer_seconds"),
        "compression_range_seconds": _sum_float(shard_metrics, "compression_range_seconds"),
        "compression_tokens_per_second": int(token_count) / max(float(wall_seconds), 1e-12),
        "compression_encoded_payload_bytes": encoded_payload_bytes,
        "compression_emitted_symbols": emitted,
        "window_codec_devices": devices,
        "window_codec_shard_count": len(shard_metrics),
        "window_codec_parallelism": "multi_gpu_window_shards" if len(shard_metrics) > 1 else "single_device",
        "window_codec_shards": shard_metrics,
    }
    for key in ("compression_max_memory_allocated_gb", "compression_memory_reserved_gb"):
        values = [float(row[key]) for row in shard_metrics if key in row]
        if values:
            result[key] = max(values)
    return result


def _aggregate_decode_metrics(
    shard_metrics: list[dict[str, Any]],
    *,
    token_count: int,
    batch_size: int,
    devices: list[str],
    parse_seconds: float,
    wall_seconds: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "decode_batch_size": int(batch_size),
        "decode_parse_seconds": float(parse_seconds),
        "decode_wall_seconds": float(wall_seconds),
        "decode_model_seconds": _sum_float(shard_metrics, "decode_model_seconds"),
        "decode_quantize_seconds": _sum_float(shard_metrics, "decode_quantize_seconds"),
        "decode_freq_transfer_seconds": _sum_float(shard_metrics, "decode_freq_transfer_seconds"),
        "decode_arith_seconds": _sum_float(shard_metrics, "decode_arith_seconds"),
        "decode_token_transfer_seconds": _sum_float(shard_metrics, "decode_token_transfer_seconds"),
        "decode_tokens_per_second": int(token_count) / max(float(wall_seconds), 1e-12),
        "decode_mismatches": _sum_int(shard_metrics, "decode_mismatches"),
        "window_codec_decode_devices": devices,
        "window_codec_decode_shard_count": len(shard_metrics),
        "window_codec_decode_parallelism": "multi_gpu_window_shards" if len(shard_metrics) > 1 else "single_device",
        "window_codec_decode_shards": shard_metrics,
    }
    for key in ("decode_max_memory_allocated_gb", "decode_memory_reserved_gb"):
        values = [float(row[key]) for row in shard_metrics if key in row]
        if values:
            result[key] = max(values)
    return result


def checkpoint_path(run_dir: str | Path, checkpoint_tag: str) -> Path:
    path = Path(run_dir) / f"{checkpoint_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_codec_model(run_dir: str | Path, checkpoint_tag: str, device: torch.device):
    run_dir = Path(run_dir)
    config = load_experiment_config(run_dir / "resolved_config.json")
    model = build_model(config.model).to(device)
    model_state, checkpoint_metadata, _ = load_megabyte_checkpoint(
        checkpoint_path(run_dir, checkpoint_tag),
        map_location=device,
    )
    model.load_state_dict(model_state)
    model.eval()
    return config, model, checkpoint_metadata


def load_codec_config_and_metadata(run_dir: str | Path, checkpoint_tag: str):
    run_dir = Path(run_dir)
    config = load_experiment_config(run_dir / "resolved_config.json")
    _, checkpoint_metadata, _ = load_megabyte_checkpoint(
        checkpoint_path(run_dir, checkpoint_tag),
        map_location="cpu",
    )
    return config, checkpoint_metadata


def _load_model_from_config_checkpoint(config, checkpoint_path: str | Path, device: torch.device):
    model = build_model(config.model).to(device)
    model_state, _, _ = load_megabyte_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    return model


def resolve_frequency_total(config, requested_total: int | None) -> tuple[int, dict[str, float | int]]:
    metadata = resolve_arithmetic_coding_metadata(
        vocab_size=int(config.model.vocab_size),
        requested_total=requested_total,
        target_uniform_mass=float(config.arithmetic.target_uniform_mass),
    )
    return int(metadata["arithmetic_frequency_total"]), metadata


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device: torch.device, *, prefix: str = "") -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        f"{prefix}max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        f"{prefix}memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
    }


def pack_length_prefixed_streams(streams: list[bytes]) -> bytes:
    chunks: list[bytes] = []
    for stream in streams:
        length = len(stream)
        if length >= 2**32:
            raise ValueError("window stream is too large for uint32 length prefix")
        chunks.append(length.to_bytes(LENGTH_PREFIX_BYTES, byteorder="little", signed=False))
        chunks.append(stream)
    return b"".join(chunks)


def _encode_varuint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varuint value must be non-negative")
    output = bytearray()
    value = int(value)
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _decode_varuint(payload: bytes, offset: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if offset >= len(payload):
            raise ValueError("truncated varuint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("varuint is too large")


def pack_varuint_length_prefixed_streams(streams: list[bytes]) -> bytes:
    chunks: list[bytes] = []
    for stream in streams:
        chunks.append(_encode_varuint(len(stream)))
        chunks.append(stream)
    return b"".join(chunks)


def parse_varuint_length_prefixed_streams(payload: bytes, *, expected_count: int | None = None) -> list[bytes]:
    streams: list[bytes] = []
    offset = 0
    payload_size = len(payload)
    while offset < payload_size:
        length, offset = _decode_varuint(payload, offset)
        if offset + length > payload_size:
            raise ValueError("truncated varuint length-prefixed stream")
        streams.append(payload[offset : offset + length])
        offset += length
    if expected_count is not None and len(streams) != int(expected_count):
        raise ValueError(f"stream count mismatch: {len(streams)} != {expected_count}")
    return streams


def pack_v2_window_payload(streams: list[bytes], header: dict[str, Any]) -> bytes:
    header_payload = json.dumps(header, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(header_payload) >= 2 ** (8 * V2_HEADER_PREFIX_BYTES):
        raise ValueError("v2 window codec header is too large")
    return (
        V2_MAGIC
        + len(header_payload).to_bytes(V2_HEADER_PREFIX_BYTES, byteorder="little", signed=False)
        + header_payload
        + pack_length_prefixed_streams(streams)
    )


def parse_v2_window_payload(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    if not payload.startswith(V2_MAGIC):
        raise ValueError("unsupported window codec payload; expected v2 MBW magic")
    offset = len(V2_MAGIC)
    if len(payload) < offset + V2_HEADER_PREFIX_BYTES:
        raise ValueError("truncated v2 window codec header length")
    header_length = int.from_bytes(payload[offset : offset + V2_HEADER_PREFIX_BYTES], byteorder="little", signed=False)
    offset += V2_HEADER_PREFIX_BYTES
    if len(payload) < offset + header_length:
        raise ValueError("truncated v2 window codec header")
    header = json.loads(payload[offset : offset + header_length].decode("utf-8"))
    offset += header_length
    return header, parse_length_prefixed_streams(payload[offset:])


def pack_v3_window_payload(streams: list[bytes], header: dict[str, Any]) -> bytes:
    field_values = []
    for key in V3_HEADER_UINT64_FIELDS:
        value = int(header[key])
        if value < 0 or value >= 2**64:
            raise ValueError(f"v3 header field {key} is outside uint64 range")
        field_values.append(value)
    fixed_header = b"".join(value.to_bytes(8, byteorder="little", signed=False) for value in field_values)
    return V3_MAGIC + fixed_header + pack_varuint_length_prefixed_streams(streams)


def parse_v3_window_payload(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    if not payload.startswith(V3_MAGIC):
        raise ValueError("unsupported window codec payload; expected v3 MBW magic")
    offset = len(V3_MAGIC)
    header_bytes = 8 * len(V3_HEADER_UINT64_FIELDS)
    if len(payload) < offset + header_bytes:
        raise ValueError("truncated v3 window codec header")
    values = []
    for _ in V3_HEADER_UINT64_FIELDS:
        values.append(int.from_bytes(payload[offset : offset + 8], byteorder="little", signed=False))
        offset += 8
    header = {
        "format_version": WINDOW_CODEC_V3_FORMAT_VERSION,
        "codec": WINDOW_CODEC_NAME,
        "framing": "mbw_v3_compact_varuint_length_prefixed_streams",
        "payload_header_format": "mbw_v3_compact",
        **dict(zip(V3_HEADER_UINT64_FIELDS, values)),
    }
    streams = parse_varuint_length_prefixed_streams(payload[offset:], expected_count=int(header["window_count"]))
    return header, streams


def parse_window_payload(payload: bytes) -> tuple[dict[str, Any], list[bytes]]:
    if payload.startswith(V3_MAGIC):
        return parse_v3_window_payload(payload)
    if payload.startswith(V2_MAGIC):
        return parse_v2_window_payload(payload)
    raise ValueError("unsupported window codec payload; expected v2 or v3 MBW magic")


def parse_length_prefixed_streams(payload: bytes) -> list[bytes]:
    streams: list[bytes] = []
    offset = 0
    payload_size = len(payload)
    while offset < payload_size:
        if offset + LENGTH_PREFIX_BYTES > payload_size:
            raise ValueError("truncated window length prefix")
        length = int.from_bytes(payload[offset : offset + LENGTH_PREFIX_BYTES], byteorder="little", signed=False)
        offset += LENGTH_PREFIX_BYTES
        if offset + length > payload_size:
            raise ValueError("truncated window payload")
        streams.append(payload[offset : offset + length])
        offset += length
    return streams


def payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def generate_random_windows(*, window_count: int, seq_length: int, vocab_high: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randint(0, int(vocab_high), (int(window_count), int(seq_length)), dtype=torch.long, generator=generator)


def load_token_windows(
    path: str | Path,
    *,
    tokens_per_window: int | None,
    pad_id: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    array = np.load(Path(path))
    if array.ndim == 2:
        windows = torch.as_tensor(array, dtype=torch.long).contiguous()
        return windows, {
            "token_input_path": str(path),
            "token_input_ndim": 2,
            "original_token_count": int(windows.numel()),
            "tail_padding_tokens": 0,
        }
    if array.ndim != 1:
        raise ValueError("token npy input must be a 1D token stream or a 2D [windows, tokens] array")
    if tokens_per_window is None or tokens_per_window <= 0:
        raise ValueError("--tokens-per-window is required for 1D token npy input")
    flat = torch.as_tensor(array, dtype=torch.long).contiguous()
    original_count = int(flat.numel())
    padding = (-original_count) % int(tokens_per_window)
    if padding:
        flat = torch.cat((flat, torch.full((padding,), int(pad_id), dtype=torch.long)), dim=0)
    windows = flat.view(-1, int(tokens_per_window)).contiguous()
    return windows, {
        "token_input_path": str(path),
        "token_input_ndim": 1,
        "original_token_count": original_count,
        "tail_padding_tokens": int(padding),
    }


def _alphabet_digit_lookup(token_merge_alphabet: str) -> tuple[np.ndarray, int]:
    alphabet = "".join(ch for ch in token_merge_alphabet.upper() if not ch.isspace())
    unique: list[str] = []
    seen: set[str] = set()
    for ch in alphabet:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)
    if len(unique) < 2:
        raise ValueError("token_merge_alphabet must contain at least 2 unique characters")
    lookup = np.full(256, -1, dtype=np.int16)
    for index, ch in enumerate(unique):
        lookup[ord(ch)] = index
        lookup[ord(ch.lower())] = index
    return lookup, len(unique)


def iter_token_window_batches(
    chunks,
    *,
    seq_length: int,
    batch_size: int,
    pad_id: int,
    eos_id: int | None,
    token_merge_size: int,
    token_merge_alphabet: str,
    requested_bytes: int | None = None,
    metadata: TokenStreamMetadata | None = None,
):
    if seq_length <= 0:
        raise ValueError("seq_length must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if token_merge_size <= 0:
        raise ValueError("token_merge_size must be > 0")
    meta = metadata if metadata is not None else TokenStreamMetadata()
    lookup, base = _alphabet_digit_lookup(token_merge_alphabet)
    batch_token_capacity = int(seq_length) * int(batch_size)
    token_buffer: list[int] = []
    carry_digits: list[int] = []
    window_start = 0
    remaining = None if requested_bytes is None or requested_bytes <= 0 else int(requested_bytes)

    def emit_full_batches():
        nonlocal window_start, token_buffer
        while len(token_buffer) >= batch_token_capacity:
            batch_tokens = token_buffer[:batch_token_capacity]
            del token_buffer[:batch_token_capacity]
            tokens = torch.tensor(batch_tokens, dtype=torch.long).view(int(batch_size), int(seq_length)).contiguous()
            valid_lengths = torch.full((int(batch_size),), int(seq_length), dtype=torch.long)
            batch = TokenWindowBatch(window_start=window_start, tokens=tokens, valid_lengths=valid_lengths)
            window_start += int(batch_size)
            meta.emitted_windows += int(batch_size)
            yield batch

    for raw_chunk in chunks:
        if remaining is not None and remaining <= 0:
            break
        if not raw_chunk:
            continue
        if remaining is not None and len(raw_chunk) > remaining:
            raw_chunk = raw_chunk[:remaining]
        meta.read_bytes += len(raw_chunk)
        if remaining is not None:
            remaining -= len(raw_chunk)

        raw = np.frombuffer(raw_chunk, dtype=np.uint8)
        digits = lookup[raw]
        if np.any(digits < 0):
            digits = digits[digits >= 0]
        meta.filtered_bytes += int(digits.shape[0])
        if token_merge_size <= 1:
            token_buffer.extend(int(value) for value in raw[lookup[raw] >= 0].tolist())
        else:
            if carry_digits:
                digits = np.concatenate((np.asarray(carry_digits, dtype=np.int16), digits))
            full_digit_count = (int(digits.shape[0]) // int(token_merge_size)) * int(token_merge_size)
            carry_digits = [int(value) for value in digits[full_digit_count:].tolist()]
            if full_digit_count > 0:
                merged = digits[:full_digit_count].reshape(-1, int(token_merge_size))
                for row in merged:
                    token_id = 0
                    for digit in row.tolist():
                        token_id = token_id * base + int(digit)
                    token_buffer.append(token_id)
        yield from emit_full_batches()

    meta.symbol_count_without_eos = len(token_buffer) + window_start * int(seq_length)
    if eos_id is not None:
        token_buffer.append(int(eos_id))
        meta.eos_appended = True
    meta.original_token_count = len(token_buffer) + window_start * int(seq_length)
    if token_buffer:
        valid_token_count = len(token_buffer)
        padding = (-len(token_buffer)) % int(seq_length)
        if padding:
            token_buffer.extend([int(pad_id)] * padding)
        meta.tail_padding_tokens = int(padding)
        windows = len(token_buffer) // int(seq_length)
        tokens = torch.tensor(token_buffer, dtype=torch.long).view(windows, int(seq_length)).contiguous()
        valid_lengths = torch.full((windows,), int(seq_length), dtype=torch.long)
        if windows > 0:
            valid_lengths[-1] = int(valid_token_count - (windows - 1) * int(seq_length))
        meta.emitted_windows += int(windows)
        yield TokenWindowBatch(window_start=window_start, tokens=tokens, valid_lengths=valid_lengths)
    elif window_start == 0:
        token_buffer = [int(eos_id if eos_id is not None else pad_id)]
        padding = int(seq_length) - 1
        token_buffer.extend([int(pad_id)] * padding)
        meta.tail_padding_tokens = padding
        meta.original_token_count = 1
        meta.eos_appended = eos_id is not None
        tokens = torch.tensor(token_buffer, dtype=torch.long).view(1, int(seq_length)).contiguous()
        meta.emitted_windows = 1
        yield TokenWindowBatch(window_start=0, tokens=tokens, valid_lengths=torch.tensor([1], dtype=torch.long))


def payload_to_token_windows(
    payload: bytes,
    *,
    seq_length: int,
    pad_id: int,
    eos_id: int | None,
    token_merge_size: int,
    token_merge_alphabet: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    symbols = tokenize_source_bytes(payload, token_merge_size, token_merge_alphabet)
    symbol_count_without_eos = len(symbols)
    if eos_id is not None:
        symbols.append(int(eos_id))
    original_token_count = len(symbols)
    window_count = max(1, (original_token_count + int(seq_length) - 1) // int(seq_length))
    padded_length = window_count * int(seq_length)
    tokens = torch.full((padded_length,), int(pad_id), dtype=torch.long)
    if symbols:
        tokens[: len(symbols)] = torch.tensor(symbols, dtype=torch.long)
    tail_padding_tokens = padded_length - original_token_count
    return tokens.view(window_count, int(seq_length)).contiguous(), {
        "original_token_count": int(original_token_count),
        "symbol_count_without_eos": int(symbol_count_without_eos),
        "eos_appended": eos_id is not None,
        "tail_padding_tokens": int(tail_padding_tokens),
    }


def save_token_windows(
    tokens_cpu: torch.Tensor,
    path: str | Path,
    *,
    original_token_count: int | None = None,
    flat_output_path: str | Path | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens_cpu.detach().cpu().numpy())
    if flat_output_path is not None:
        flat = tokens_cpu.reshape(-1)
        if original_token_count is not None:
            flat = flat[: int(original_token_count)]
        flat_path = Path(flat_output_path)
        flat_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(flat_path, flat.detach().cpu().numpy())


def compress_token_windows(
    *,
    model: torch.nn.Module,
    tokens_cpu: torch.Tensor,
    valid_lengths_cpu: torch.Tensor | None = None,
    batch_size: int,
    device: torch.device,
    dtype_name: str,
    frequency_total: int,
    compression_mode: str = "cached",
) -> tuple[list[bytes], dict[str, Any]]:
    if tokens_cpu.dim() != 2:
        raise ValueError("tokens_cpu must have shape [windows, tokens_per_window]")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seq_length = int(tokens_cpu.shape[1])
    if valid_lengths_cpu is None:
        valid_lengths_cpu = torch.full((int(tokens_cpu.shape[0]),), seq_length, dtype=torch.long)
    else:
        valid_lengths_cpu = torch.as_tensor(valid_lengths_cpu, dtype=torch.long).cpu().contiguous()
    if valid_lengths_cpu.dim() != 1 or int(valid_lengths_cpu.shape[0]) != int(tokens_cpu.shape[0]):
        raise ValueError("valid_lengths_cpu must have shape [windows]")
    if int(valid_lengths_cpu.min().item()) < 0 or int(valid_lengths_cpu.max().item()) > seq_length:
        raise ValueError("valid window lengths must be in [0, seq_length]")
    vocab_size = int(model.config.V)
    streams: list[bytes] = []
    model_seconds = 0.0
    quantize_seconds = 0.0
    interval_transfer_seconds = 0.0
    range_seconds = 0.0
    emitted = 0

    sync_if_cuda(device)
    started = perf_counter()
    for start in range(0, int(tokens_cpu.shape[0]), int(batch_size)):
        chunk_cpu = tokens_cpu[start : start + int(batch_size)].contiguous()
        chunk_lengths_cpu = valid_lengths_cpu[start : start + int(batch_size)].contiguous()
        chunk_size = int(chunk_cpu.shape[0])
        ids = chunk_cpu.to(device, non_blocking=True)

        if compression_mode == "full_forward":
            model_started = perf_counter()
            with torch.inference_mode(), autocast_context(device, dtype_name):
                logits = model(ids).lm_logits
            sync_if_cuda(device)
            model_seconds += perf_counter() - model_started

            quant_started = perf_counter()
            flat_probs = torch.softmax(logits.reshape(-1, vocab_size).float(), dim=-1)
            targets = ids.reshape(-1)
            lows, highs, totals = fast_floor_intervals_from_probabilities(
                flat_probs,
                targets,
                total=int(frequency_total),
            )
            lows = lows.to(torch.int32).view(chunk_size, seq_length)
            highs = highs.to(torch.int32).view(chunk_size, seq_length)
            totals = totals.to(torch.int32).view(chunk_size, seq_length)
            sync_if_cuda(device)
            quantize_seconds += perf_counter() - quant_started

            transfer_started = perf_counter()
            lows_cpu = lows.cpu()
            highs_cpu = highs.cpu()
            totals_cpu = totals.cpu()
            sync_if_cuda(device)
            interval_transfer_seconds += perf_counter() - transfer_started
        elif compression_mode == "cached":
            stepper = MegabyteBatchedDecodeStepper(model, batch_size=chunk_size, device=device, dtype_name=dtype_name)
            lows_steps: list[torch.Tensor] = []
            highs_steps: list[torch.Tensor] = []
            totals_steps: list[torch.Tensor] = []
            for step in range(seq_length):
                model_started = perf_counter()
                logits = stepper.next_logits()
                sync_if_cuda(device)
                model_seconds += perf_counter() - model_started

                quant_started = perf_counter()
                probs = torch.softmax(logits.float(), dim=-1)
                lows, highs, totals = fast_floor_intervals_from_probabilities(
                    probs,
                    ids[:, step],
                    total=int(frequency_total),
                )
                lows = lows.to(torch.int32)
                highs = highs.to(torch.int32)
                totals = totals.to(torch.int32)
                sync_if_cuda(device)
                quantize_seconds += perf_counter() - quant_started

                transfer_started = perf_counter()
                lows_steps.append(lows.cpu())
                highs_steps.append(highs.cpu())
                totals_steps.append(totals.cpu())
                sync_if_cuda(device)
                interval_transfer_seconds += perf_counter() - transfer_started
                stepper.accept_symbols(ids[:, step])
            lows_cpu = torch.stack(lows_steps, dim=1).contiguous()
            highs_cpu = torch.stack(highs_steps, dim=1).contiguous()
            totals_cpu = torch.stack(totals_steps, dim=1).contiguous()
        else:
            raise ValueError(f"unknown compression mode: {compression_mode}")

        encoder = BatchedStreamingArithmeticEncoder(chunk_size)
        timings = encoder.encode_interval_matrix_with_lengths(lows_cpu, highs_cpu, totals_cpu, chunk_lengths_cpu)
        range_seconds += timings.range_seconds
        emitted += timings.emitted_count
        streams.extend(encoder.finish())

    wall_seconds = perf_counter() - started
    encoded_payload_bytes = sum(len(stream) for stream in streams)
    token_count = int(valid_lengths_cpu.sum().item())
    return streams, {
        "compression_batch_size": int(batch_size),
        "compression_mode": compression_mode,
        "compression_wall_seconds": wall_seconds,
        "compression_model_seconds": model_seconds,
        "compression_quantize_seconds": quantize_seconds,
        "compression_interval_transfer_seconds": interval_transfer_seconds,
        "compression_range_seconds": range_seconds,
        "compression_tokens_per_second": token_count / max(wall_seconds, 1e-12),
        "compression_encoded_payload_bytes": encoded_payload_bytes,
        "compression_emitted_symbols": int(emitted),
        **memory_stats(device, prefix="compression_"),
    }


def _compress_token_windows_worker(args: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(str(args["device"]))
    model = _load_model_from_config_checkpoint(args["config"], args["checkpoint_path"], device)
    tokens_cpu = args["tokens_cpu"].contiguous()
    streams, metrics = compress_token_windows(
        model=model,
        tokens_cpu=tokens_cpu,
        valid_lengths_cpu=args.get("valid_lengths_cpu"),
        batch_size=int(args["batch_size"]),
        device=device,
        dtype_name=str(args["dtype_name"]),
        frequency_total=int(args["frequency_total"]),
        compression_mode=str(args["compression_mode"]),
    )
    metrics.update(
        {
            "shard_index": int(args["shard_index"]),
            "device": str(device),
            "window_start": int(args["window_start"]),
            "window_end": int(args["window_end"]),
            "window_count": int(tokens_cpu.shape[0]),
            "valid_token_count": int(metrics.get("compression_emitted_symbols", 0)),
        }
    )
    return {"streams": streams, "metrics": metrics}


def compress_token_windows_multi_device(
    *,
    config,
    checkpoint_path: str | Path,
    model: torch.nn.Module | None,
    tokens_cpu: torch.Tensor,
    valid_lengths_cpu: torch.Tensor | None = None,
    batch_size: int,
    devices: list[str],
    dtype_name: str,
    frequency_total: int,
    compression_mode: str = "cached",
) -> tuple[list[bytes], dict[str, Any]]:
    if len(devices) <= 1:
        device = torch.device(devices[0])
        if model is None:
            model = _load_model_from_config_checkpoint(config, checkpoint_path, device)
        return compress_token_windows(
            model=model,
            tokens_cpu=tokens_cpu,
            valid_lengths_cpu=valid_lengths_cpu,
            batch_size=batch_size,
            device=device,
            dtype_name=dtype_name,
            frequency_total=frequency_total,
            compression_mode=compression_mode,
        )

    ranges = batch_aligned_window_ranges(int(tokens_cpu.shape[0]), int(batch_size), len(devices))
    if not ranges:
        return [], _aggregate_compression_metrics(
            [],
            token_count=0,
            batch_size=batch_size,
            mode=compression_mode,
            devices=devices,
            wall_seconds=0.0,
        )

    started = perf_counter()
    worker_args: list[dict[str, Any]] = []
    for shard_index, (start, end) in enumerate(ranges):
        worker_args.append(
            {
                "shard_index": shard_index,
                "device": devices[shard_index % len(devices)],
                "config": config,
                "checkpoint_path": str(checkpoint_path),
                "tokens_cpu": tokens_cpu[start:end].contiguous(),
                "valid_lengths_cpu": (
                    torch.as_tensor(valid_lengths_cpu, dtype=torch.long).cpu()[start:end].contiguous()
                    if valid_lengths_cpu is not None
                    else None
                ),
                "batch_size": int(batch_size),
                "dtype_name": dtype_name,
                "frequency_total": int(frequency_total),
                "compression_mode": compression_mode,
                "window_start": int(start),
                "window_end": int(end),
            }
        )

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as executor:
        results = list(executor.map(_compress_token_windows_worker, worker_args))

    results.sort(key=lambda item: int(item["metrics"]["window_start"]))
    streams: list[bytes] = []
    shard_metrics: list[dict[str, Any]] = []
    for result in results:
        streams.extend(result["streams"])
        shard_metrics.append(result["metrics"])
    wall_seconds = perf_counter() - started
    metrics = _aggregate_compression_metrics(
        shard_metrics,
        token_count=_sum_int(shard_metrics, "compression_emitted_symbols"),
        batch_size=int(batch_size),
        mode=compression_mode,
        devices=devices,
        wall_seconds=wall_seconds,
    )
    return streams, metrics


_WORKER_MODEL: torch.nn.Module | None = None
_WORKER_DEVICE: torch.device | None = None
_WORKER_DTYPE_NAME: str | None = None
_WORKER_FREQUENCY_TOTAL: int | None = None
_WORKER_COMPRESSION_MODE: str | None = None
_WORKER_INIT_SECONDS: float = 0.0
_WORKER_INIT_REPORTED: bool = False


def _initialize_codec_worker(
    config,
    checkpoint_path: str,
    device_name: str,
    dtype_name: str,
    frequency_total: int,
    compression_mode: str,
) -> None:
    global _WORKER_MODEL
    global _WORKER_DEVICE
    global _WORKER_DTYPE_NAME
    global _WORKER_FREQUENCY_TOTAL
    global _WORKER_COMPRESSION_MODE
    global _WORKER_INIT_SECONDS
    global _WORKER_INIT_REPORTED
    started = perf_counter()
    device = torch.device(device_name)
    _WORKER_MODEL = _load_model_from_config_checkpoint(config, checkpoint_path, device)
    _WORKER_DEVICE = device
    _WORKER_DTYPE_NAME = dtype_name
    _WORKER_FREQUENCY_TOTAL = int(frequency_total)
    _WORKER_COMPRESSION_MODE = compression_mode
    _WORKER_INIT_SECONDS = perf_counter() - started
    _WORKER_INIT_REPORTED = False


def _worker_init_metric() -> float:
    global _WORKER_INIT_REPORTED
    if _WORKER_INIT_REPORTED:
        return 0.0
    _WORKER_INIT_REPORTED = True
    return float(_WORKER_INIT_SECONDS)


def _compress_batch_in_persistent_worker(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_DEVICE is None:
        raise RuntimeError("codec worker is not initialized")
    tokens_cpu = task["tokens_cpu"].contiguous()
    streams, metrics = compress_token_windows(
        model=_WORKER_MODEL,
        tokens_cpu=tokens_cpu,
        valid_lengths_cpu=task.get("valid_lengths_cpu"),
        batch_size=int(tokens_cpu.shape[0]),
        device=_WORKER_DEVICE,
        dtype_name=str(_WORKER_DTYPE_NAME),
        frequency_total=int(_WORKER_FREQUENCY_TOTAL),
        compression_mode=str(_WORKER_COMPRESSION_MODE),
    )
    metrics.update(
        {
            "worker_init_seconds": _worker_init_metric(),
            "device": str(_WORKER_DEVICE),
            "window_start": int(task["window_start"]),
            "window_end": int(task["window_start"]) + int(tokens_cpu.shape[0]),
            "window_count": int(tokens_cpu.shape[0]),
            "valid_token_count": int(metrics.get("compression_emitted_symbols", 0)),
        }
    )
    return {"streams": streams, "metrics": metrics}


def _decode_batch_in_persistent_worker(task: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_DEVICE is None:
        raise RuntimeError("codec worker is not initialized")
    streams = task["streams"]
    framed = pack_length_prefixed_streams(streams)
    decoded, metrics = decode_framed_token_windows(
        model=_WORKER_MODEL,
        framed_payload=framed,
        window_count=len(streams),
        tokens_per_window=int(task["tokens_per_window"]),
        valid_lengths_cpu=task.get("valid_lengths_cpu"),
        pad_id=task.get("pad_id"),
        batch_size=len(streams),
        device=_WORKER_DEVICE,
        dtype_name=str(_WORKER_DTYPE_NAME),
        frequency_total=int(_WORKER_FREQUENCY_TOTAL),
        threads=int(task.get("threads", 0)),
        expected_tokens_cpu=task.get("expected_tokens_cpu"),
    )
    metrics.update(
        {
            "worker_init_seconds": _worker_init_metric(),
            "device": str(_WORKER_DEVICE),
            "window_start": int(task["window_start"]),
            "window_end": int(task["window_start"]) + int(decoded.shape[0]),
            "window_count": int(decoded.shape[0]),
        }
    )
    return {"decoded": decoded, "metrics": metrics}


class WindowCodecPipeline:
    def __init__(
        self,
        *,
        config,
        checkpoint_path: str | Path,
        devices: list[str],
        dtype_name: str,
        frequency_total: int,
        batch_size: int,
        compression_mode: str = "cached",
    ) -> None:
        if compression_mode not in {"cached", "full_forward"}:
            raise ValueError("compression_mode must be one of: cached, full_forward")
        self.config = config
        self.checkpoint_path = str(checkpoint_path)
        self.devices = list(devices)
        self.dtype_name = dtype_name
        self.frequency_total = int(frequency_total)
        self.batch_size = int(batch_size)
        self.compression_mode = compression_mode
        self._executors: list[ProcessPoolExecutor] = []
        self._single_model: torch.nn.Module | None = None
        self._single_device: torch.device | None = None
        self._next_device_index = 0
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if len(self.devices) <= 1:
            started = perf_counter()
            self._single_device = torch.device(self.devices[0])
            self._single_model = _load_model_from_config_checkpoint(self.config, self.checkpoint_path, self._single_device)
            self._single_init_seconds = perf_counter() - started
        else:
            ctx = mp.get_context("spawn")
            for device in self.devices:
                self._executors.append(
                    ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=ctx,
                        initializer=_initialize_codec_worker,
                        initargs=(
                            self.config,
                            self.checkpoint_path,
                            device,
                            self.dtype_name,
                            self.frequency_total,
                            self.compression_mode,
                        ),
                    )
                )
        self._started = True

    def close(self) -> None:
        for executor in self._executors:
            executor.shutdown(wait=True, cancel_futures=False)
        self._executors = []
        self._single_model = None
        self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _submit(self, fn, task: dict[str, Any]):
        executor = self._executors[self._next_device_index % len(self._executors)]
        self._next_device_index += 1
        return executor.submit(fn, task)

    def compress_batches(self, batches) -> tuple[list[bytes], dict[str, Any]]:
        self.start()
        started = perf_counter()
        if len(self.devices) <= 1:
            streams_by_start: dict[int, list[bytes]] = {}
            metrics_rows: list[dict[str, Any]] = []
            init_reported = False
            assert self._single_model is not None
            assert self._single_device is not None
            for batch in batches:
                valid_lengths = (
                    batch.valid_lengths.contiguous()
                    if batch.valid_lengths is not None
                    else torch.full((int(batch.tokens.shape[0]),), int(batch.tokens.shape[1]), dtype=torch.long)
                )
                batch_streams, metrics = compress_token_windows(
                    model=self._single_model,
                    tokens_cpu=batch.tokens,
                    valid_lengths_cpu=valid_lengths,
                    batch_size=int(batch.tokens.shape[0]),
                    device=self._single_device,
                    dtype_name=self.dtype_name,
                    frequency_total=self.frequency_total,
                    compression_mode=self.compression_mode,
                )
                metrics.update(
                    {
                        "worker_init_seconds": 0.0 if init_reported else float(getattr(self, "_single_init_seconds", 0.0)),
                        "device": str(self._single_device),
                        "window_start": int(batch.window_start),
                        "window_end": int(batch.window_start) + int(batch.tokens.shape[0]),
                        "window_count": int(batch.tokens.shape[0]),
                        "valid_token_count": int(valid_lengths.sum().item()),
                    }
                )
                init_reported = True
                streams_by_start[int(batch.window_start)] = batch_streams
                metrics_rows.append(metrics)
        else:
            streams_by_start = {}
            metrics_rows = []
            pending = set()
            max_pending = max(1, len(self._executors) * 2)

            def drain(done_futures):
                for future in done_futures:
                    result = future.result()
                    metrics = result["metrics"]
                    streams_by_start[int(metrics["window_start"])] = result["streams"]
                    metrics_rows.append(metrics)

            for batch in batches:
                valid_lengths = (
                    batch.valid_lengths.contiguous()
                    if batch.valid_lengths is not None
                    else torch.full((int(batch.tokens.shape[0]),), int(batch.tokens.shape[1]), dtype=torch.long)
                )
                pending.add(
                    self._submit(
                        _compress_batch_in_persistent_worker,
                        {
                            "window_start": int(batch.window_start),
                            "tokens_cpu": batch.tokens.contiguous(),
                            "valid_lengths_cpu": valid_lengths,
                        },
                    )
                )
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    drain(done)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                drain(done)

        streams: list[bytes] = []
        for window_start in sorted(streams_by_start):
            streams.extend(streams_by_start[window_start])
        wall_seconds = perf_counter() - started
        token_count = _sum_int(metrics_rows, "compression_emitted_symbols")
        metrics = _aggregate_compression_metrics(
            metrics_rows,
            token_count=token_count,
            batch_size=self.batch_size,
            mode=self.compression_mode,
            devices=self.devices,
            wall_seconds=wall_seconds,
        )
        metrics["worker_init_seconds"] = _sum_float(metrics_rows, "worker_init_seconds")
        return streams, metrics

    def decode_streams(
        self,
        *,
        streams: list[bytes],
        tokens_per_window: int,
        valid_lengths_cpu: torch.Tensor | None = None,
        pad_id: int | None = None,
        threads: int = 0,
        expected_tokens_cpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        self.start()
        started = perf_counter()
        if valid_lengths_cpu is None:
            valid_lengths_cpu = torch.full((len(streams),), int(tokens_per_window), dtype=torch.long)
        else:
            valid_lengths_cpu = torch.as_tensor(valid_lengths_cpu, dtype=torch.long).cpu().contiguous()
        batches: list[tuple[int, list[bytes]]] = []
        for start in range(0, len(streams), self.batch_size):
            batches.append((start, streams[start : start + self.batch_size]))

        if len(self.devices) <= 1:
            decoded_by_start: dict[int, torch.Tensor] = {}
            metrics_rows: list[dict[str, Any]] = []
            init_reported = False
            assert self._single_model is not None
            assert self._single_device is not None
            for start, chunk_streams in batches:
                expected = expected_tokens_cpu[start : start + len(chunk_streams)].contiguous() if expected_tokens_cpu is not None else None
                decoded, metrics = decode_framed_token_windows(
                    model=self._single_model,
                    framed_payload=pack_length_prefixed_streams(chunk_streams),
                    window_count=len(chunk_streams),
                    tokens_per_window=int(tokens_per_window),
                    valid_lengths_cpu=valid_lengths_cpu[start : start + len(chunk_streams)].contiguous(),
                    pad_id=pad_id,
                    batch_size=len(chunk_streams),
                    device=self._single_device,
                    dtype_name=self.dtype_name,
                    frequency_total=self.frequency_total,
                    threads=int(threads),
                    expected_tokens_cpu=expected,
                )
                metrics.update(
                    {
                        "worker_init_seconds": 0.0 if init_reported else float(getattr(self, "_single_init_seconds", 0.0)),
                        "device": str(self._single_device),
                        "window_start": int(start),
                        "window_end": int(start) + int(decoded.shape[0]),
                        "window_count": int(decoded.shape[0]),
                    }
                )
                init_reported = True
                decoded_by_start[start] = decoded
                metrics_rows.append(metrics)
        else:
            decoded_by_start = {}
            metrics_rows = []
            pending = set()
            max_pending = max(1, len(self._executors) * 2)

            def drain(done_futures):
                for future in done_futures:
                    result = future.result()
                    metrics = result["metrics"]
                    decoded_by_start[int(metrics["window_start"])] = result["decoded"]
                    metrics_rows.append(metrics)

            for start, chunk_streams in batches:
                expected = expected_tokens_cpu[start : start + len(chunk_streams)].contiguous() if expected_tokens_cpu is not None else None
                pending.add(
                    self._submit(
                        _decode_batch_in_persistent_worker,
                        {
                            "window_start": int(start),
                            "streams": chunk_streams,
                            "tokens_per_window": int(tokens_per_window),
                            "valid_lengths_cpu": valid_lengths_cpu[start : start + len(chunk_streams)].contiguous(),
                            "pad_id": pad_id,
                            "threads": int(threads),
                            "expected_tokens_cpu": expected,
                        },
                    )
                )
                if len(pending) >= max_pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    drain(done)
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                drain(done)

        ordered = [decoded_by_start[start] for start in sorted(decoded_by_start)]
        output = torch.cat(ordered, dim=0) if ordered else torch.empty((0, int(tokens_per_window)), dtype=torch.long)
        wall_seconds = perf_counter() - started
        metrics = _aggregate_decode_metrics(
            metrics_rows,
            token_count=int(valid_lengths_cpu.sum().item()),
            batch_size=self.batch_size,
            devices=self.devices,
            parse_seconds=0.0,
            wall_seconds=wall_seconds,
        )
        metrics["worker_init_seconds"] = _sum_float(metrics_rows, "worker_init_seconds")
        return output, metrics


def frame_compressed_streams(streams: list[bytes]) -> tuple[bytes, dict[str, Any]]:
    started = perf_counter()
    framed = pack_length_prefixed_streams(streams)
    frame_seconds = perf_counter() - started
    framing_bytes = LENGTH_PREFIX_BYTES * len(streams)
    return framed, {
        "framing_seconds": frame_seconds,
        "framed_bytes": len(framed),
        "framing_bytes": framing_bytes,
    }


def decode_framed_token_windows(
    *,
    model: torch.nn.Module,
    framed_payload: bytes,
    window_count: int,
    tokens_per_window: int,
    valid_lengths_cpu: torch.Tensor | None = None,
    pad_id: int | None = None,
    batch_size: int,
    device: torch.device,
    dtype_name: str,
    frequency_total: int,
    threads: int = 0,
    expected_tokens_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    parse_started = perf_counter()
    streams = parse_length_prefixed_streams(framed_payload)
    parse_seconds = perf_counter() - parse_started
    if len(streams) != int(window_count):
        raise RuntimeError(f"framed stream count mismatch: {len(streams)} != {window_count}")
    if valid_lengths_cpu is None:
        valid_lengths_cpu = torch.full((len(streams),), int(tokens_per_window), dtype=torch.long)
    else:
        valid_lengths_cpu = torch.as_tensor(valid_lengths_cpu, dtype=torch.long).cpu().contiguous()
    if valid_lengths_cpu.dim() != 1 or int(valid_lengths_cpu.shape[0]) != len(streams):
        raise ValueError("valid_lengths_cpu must have shape [window_count]")
    if int(valid_lengths_cpu.min().item()) < 0 or int(valid_lengths_cpu.max().item()) > int(tokens_per_window):
        raise ValueError("valid decode lengths must be in [0, tokens_per_window]")

    output = torch.full(
        (len(streams), int(tokens_per_window)),
        int(pad_id if pad_id is not None else 0),
        dtype=torch.long,
    )
    model_seconds = 0.0
    quantize_seconds = 0.0
    freq_transfer_seconds = 0.0
    arith_seconds = 0.0
    token_transfer_seconds = 0.0
    mismatches = 0

    sync_if_cuda(device)
    started = perf_counter()
    for start in range(0, len(streams), int(batch_size)):
        chunk_streams = streams[start : start + int(batch_size)]
        chunk_size = len(chunk_streams)
        chunk_lengths_cpu = valid_lengths_cpu[start : start + chunk_size].contiguous()
        decoder = BatchedStreamingArithmeticDecoder(chunk_streams, threads=int(threads))
        stepper = MegabyteBatchedDecodeStepper(model, batch_size=chunk_size, device=device, dtype_name=dtype_name)
        for step in range(int(tokens_per_window)):
            active_cpu = chunk_lengths_cpu > int(step)
            if not bool(active_cpu.any().item()):
                break
            model_started = perf_counter()
            logits = stepper.next_logits()
            sync_if_cuda(device)
            model_seconds += perf_counter() - model_started

            quant_started = perf_counter()
            freqs_gpu, totals_gpu = fast_floor_frequency_rows(logits, total=int(frequency_total), return_totals=True)
            sync_if_cuda(device)
            quantize_seconds += perf_counter() - quant_started

            transfer_started = perf_counter()
            freqs_cpu = freqs_gpu.cpu()
            totals_cpu = totals_gpu.cpu()
            sync_if_cuda(device)
            freq_transfer_seconds += perf_counter() - transfer_started

            arith_started = perf_counter()
            decoded_active_cpu = decoder.decode_frequency_rows_with_totals_and_active(freqs_cpu, totals_cpu, active_cpu)
            arith_seconds += perf_counter() - arith_started
            symbols_cpu = torch.full((chunk_size,), int(pad_id if pad_id is not None else 0), dtype=torch.long)
            symbols_cpu[active_cpu] = decoded_active_cpu
            output[start : start + chunk_size, step] = symbols_cpu

            if expected_tokens_cpu is not None:
                expected = expected_tokens_cpu[start : start + chunk_size, step]
                if not torch.equal(symbols_cpu[active_cpu], expected[active_cpu]):
                    mismatches += int((symbols_cpu[active_cpu] != expected[active_cpu]).sum().item())

            token_started = perf_counter()
            symbols_gpu = symbols_cpu.to(device, non_blocking=True)
            sync_if_cuda(device)
            token_transfer_seconds += perf_counter() - token_started
            stepper.accept_symbols(symbols_gpu)

    wall_seconds = perf_counter() - started
    token_count = int(valid_lengths_cpu.sum().item())
    return output, {
        "decode_batch_size": int(batch_size),
        "decode_parse_seconds": parse_seconds,
        "decode_wall_seconds": wall_seconds,
        "decode_model_seconds": model_seconds,
        "decode_quantize_seconds": quantize_seconds,
        "decode_freq_transfer_seconds": freq_transfer_seconds,
        "decode_arith_seconds": arith_seconds,
        "decode_token_transfer_seconds": token_transfer_seconds,
        "decode_tokens_per_second": token_count / max(wall_seconds, 1e-12),
        "decode_mismatches": int(mismatches),
        **memory_stats(device, prefix="decode_"),
    }


def _decode_framed_token_windows_worker(args: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(str(args["device"]))
    model = _load_model_from_config_checkpoint(args["config"], args["checkpoint_path"], device)
    expected = args.get("expected_tokens_cpu")
    framed = pack_length_prefixed_streams(args["streams"])
    decoded, metrics = decode_framed_token_windows(
        model=model,
        framed_payload=framed,
        window_count=len(args["streams"]),
        tokens_per_window=int(args["tokens_per_window"]),
        valid_lengths_cpu=args.get("valid_lengths_cpu"),
        pad_id=args.get("pad_id"),
        batch_size=int(args["batch_size"]),
        device=device,
        dtype_name=str(args["dtype_name"]),
        frequency_total=int(args["frequency_total"]),
        threads=int(args["threads"]),
        expected_tokens_cpu=expected,
    )
    metrics.update(
        {
            "shard_index": int(args["shard_index"]),
            "device": str(device),
            "window_start": int(args["window_start"]),
            "window_end": int(args["window_end"]),
            "window_count": int(decoded.shape[0]),
        }
    )
    return {"decoded": decoded, "metrics": metrics}


def decode_framed_token_windows_multi_device(
    *,
    config,
    checkpoint_path: str | Path,
    model: torch.nn.Module | None,
    framed_payload: bytes,
    window_count: int,
    tokens_per_window: int,
    valid_lengths_cpu: torch.Tensor | None = None,
    pad_id: int | None = None,
    batch_size: int,
    devices: list[str],
    dtype_name: str,
    frequency_total: int,
    threads: int = 0,
    expected_tokens_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(devices) <= 1:
        device = torch.device(devices[0])
        if model is None:
            model = _load_model_from_config_checkpoint(config, checkpoint_path, device)
        return decode_framed_token_windows(
            model=model,
            framed_payload=framed_payload,
            window_count=window_count,
            tokens_per_window=tokens_per_window,
            valid_lengths_cpu=valid_lengths_cpu,
            pad_id=pad_id,
            batch_size=batch_size,
            device=device,
            dtype_name=dtype_name,
            frequency_total=frequency_total,
            threads=threads,
            expected_tokens_cpu=expected_tokens_cpu,
        )

    parse_started = perf_counter()
    streams = parse_length_prefixed_streams(framed_payload)
    parse_seconds = perf_counter() - parse_started
    if len(streams) != int(window_count):
        raise RuntimeError(f"framed stream count mismatch: {len(streams)} != {window_count}")
    if valid_lengths_cpu is None:
        valid_lengths_cpu = torch.full((int(window_count),), int(tokens_per_window), dtype=torch.long)
    else:
        valid_lengths_cpu = torch.as_tensor(valid_lengths_cpu, dtype=torch.long).cpu().contiguous()

    ranges = batch_aligned_window_ranges(int(window_count), int(batch_size), len(devices))
    output = torch.empty((int(window_count), int(tokens_per_window)), dtype=torch.long)
    started = perf_counter()
    worker_args: list[dict[str, Any]] = []
    for shard_index, (start, end) in enumerate(ranges):
        worker_args.append(
            {
                "shard_index": shard_index,
                "device": devices[shard_index % len(devices)],
                "config": config,
                "checkpoint_path": str(checkpoint_path),
                "streams": streams[start:end],
                "tokens_per_window": int(tokens_per_window),
                "valid_lengths_cpu": valid_lengths_cpu[start:end].contiguous(),
                "pad_id": pad_id,
                "batch_size": int(batch_size),
                "dtype_name": dtype_name,
                "frequency_total": int(frequency_total),
                "threads": int(threads),
                "window_start": int(start),
                "window_end": int(end),
                "expected_tokens_cpu": (
                    expected_tokens_cpu[start:end].contiguous() if expected_tokens_cpu is not None else None
                ),
            }
        )

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as executor:
        results = list(executor.map(_decode_framed_token_windows_worker, worker_args))

    results.sort(key=lambda item: int(item["metrics"]["window_start"]))
    shard_metrics: list[dict[str, Any]] = []
    for result in results:
        metrics = result["metrics"]
        start = int(metrics["window_start"])
        end = int(metrics["window_end"])
        output[start:end] = result["decoded"]
        shard_metrics.append(metrics)
    wall_seconds = perf_counter() - started
    metrics = _aggregate_decode_metrics(
        shard_metrics,
        token_count=int(valid_lengths_cpu.sum().item()),
        batch_size=int(batch_size),
        devices=devices,
        parse_seconds=parse_seconds,
        wall_seconds=wall_seconds,
    )
    return output, metrics


def build_codec_metadata(
    *,
    run_dir: str | Path,
    checkpoint_tag: str,
    checkpoint_metadata: dict[str, Any],
    dtype_name: str,
    device: torch.device,
    tokens_cpu: torch.Tensor,
    token_merge_size: int,
    frequency_total: int,
    arithmetic_metadata: dict[str, float | int],
    compression_metrics: dict[str, Any],
    framing_metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = dict(extra or {})
    encoded_token_count = int(tokens_cpu.numel())
    logical_token_count = int(extra.get("original_token_count", encoded_token_count))
    base_token_count = int(extra.get("symbol_count_without_eos", logical_token_count))
    total_bases = base_token_count * int(token_merge_size)
    metadata: dict[str, Any] = {
        "format_version": WINDOW_CODEC_FORMAT_VERSION,
        "codec": WINDOW_CODEC_NAME,
        "quantization_mode": "fast_floor",
        "framing": "uint32_le_length_prefixed_streams",
        "length_prefix_bytes_per_window": LENGTH_PREFIX_BYTES,
        "run_dir": str(run_dir),
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_step": checkpoint_metadata.get("step"),
        "device_used_for_compression": str(device),
        "dtype": dtype_name,
        "window_count": int(tokens_cpu.shape[0]),
        "tokens_per_window": int(tokens_cpu.shape[1]),
        "encoded_token_count": encoded_token_count,
        "token_count": logical_token_count,
        "token_merge_size": int(token_merge_size),
        "base_count": total_bases,
        "frequency_total": int(frequency_total),
        "framing_overhead_bpb": float(framing_metrics["framing_bytes"]) * 8.0 / max(total_bases, 1),
        "compressed_bpb_with_framing": float(framing_metrics["framed_bytes"]) * 8.0 / max(total_bases, 1),
        "compressed_bpb_payload_only": float(compression_metrics["compression_encoded_payload_bytes"]) * 8.0 / max(total_bases, 1),
        "compression_bases_per_second": total_bases / max(float(compression_metrics["compression_wall_seconds"]), 1e-12),
        **arithmetic_metadata,
        **compression_metrics,
        **framing_metrics,
    }
    metadata.update(extra)
    return metadata


def build_codec_metadata_from_counts(
    *,
    run_dir: str | Path,
    checkpoint_tag: str,
    checkpoint_metadata: dict[str, Any],
    dtype_name: str,
    device: torch.device,
    window_count: int,
    tokens_per_window: int,
    logical_token_count: int,
    base_token_count: int,
    tail_padding_tokens: int,
    token_merge_size: int,
    frequency_total: int,
    arithmetic_metadata: dict[str, float | int],
    compression_metrics: dict[str, Any],
    framing_metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = dict(extra or {})
    encoded_token_count = int(window_count) * int(tokens_per_window)
    total_bases = int(base_token_count) * int(token_merge_size)
    metadata: dict[str, Any] = {
        "format_version": WINDOW_CODEC_V3_FORMAT_VERSION,
        "codec": WINDOW_CODEC_NAME,
        "quantization_mode": "fast_floor",
        "framing": "mbw_v3_compact_varuint_length_prefixed_streams",
        "payload_header_format": "mbw_v3_compact",
        "length_prefix_bytes_per_window": "varuint",
        "run_dir": str(run_dir),
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_step": checkpoint_metadata.get("step"),
        "device_used_for_compression": str(device),
        "dtype": dtype_name,
        "window_count": int(window_count),
        "tokens_per_window": int(tokens_per_window),
        "encoded_token_count": encoded_token_count,
        "token_count": int(logical_token_count),
        "token_merge_size": int(token_merge_size),
        "base_count": total_bases,
        "tail_padding_tokens": int(tail_padding_tokens),
        "encoded_padding_tokens": 0,
        "frequency_total": int(frequency_total),
        "framing_overhead_bpb": float(framing_metrics["framing_bytes"]) * 8.0 / max(total_bases, 1),
        "compressed_bpb_with_framing": float(framing_metrics["framed_bytes"]) * 8.0 / max(total_bases, 1),
        "compressed_bpb_payload_only": float(compression_metrics["compression_encoded_payload_bytes"]) * 8.0 / max(total_bases, 1),
        "compression_bases_per_second": total_bases / max(float(compression_metrics["compression_wall_seconds"]), 1e-12),
        **arithmetic_metadata,
        **compression_metrics,
        **framing_metrics,
    }
    metadata.update(extra)
    return metadata


def compress_chunk_stream_to_v2_payload(
    *,
    pipeline: WindowCodecPipeline,
    chunks,
    config,
    run_dir: str | Path,
    checkpoint_tag: str,
    checkpoint_metadata: dict[str, Any],
    device: torch.device,
    source_name: str,
    requested_bytes: int | None,
    arithmetic_metadata: dict[str, float | int],
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    stream_metadata = TokenStreamMetadata()
    batches = iter_token_window_batches(
        chunks,
        seq_length=int(config.model.seq_length),
        batch_size=int(pipeline.batch_size),
        pad_id=int(config.model.pad_id),
        eos_id=int(config.model.eos_id),
        token_merge_size=int(config.data.token_merge_size),
        token_merge_alphabet=str(config.data.token_merge_alphabet),
        requested_bytes=requested_bytes,
        metadata=stream_metadata,
    )
    streams, compression_metrics = pipeline.compress_batches(batches)
    header = {
        "format_version": WINDOW_CODEC_V3_FORMAT_VERSION,
        "codec": WINDOW_CODEC_NAME,
        "window_count": int(stream_metadata.emitted_windows),
        "tokens_per_window": int(config.model.seq_length),
        "logical_token_count": int(stream_metadata.original_token_count),
        "base_token_count": int(stream_metadata.symbol_count_without_eos),
        "token_merge_size": int(config.data.token_merge_size),
        "compression_batch_size": int(pipeline.batch_size),
        "frequency_total": int(pipeline.frequency_total),
    }
    payload = pack_v3_window_payload(streams, header)
    framing_bytes = len(payload) - sum(len(stream) for stream in streams)
    framing_metrics = {
        "framing_seconds": 0.0,
        "framed_bytes": len(payload),
        "framing_bytes": int(framing_bytes),
    }
    metadata = build_codec_metadata_from_counts(
        run_dir=run_dir,
        checkpoint_tag=checkpoint_tag,
        checkpoint_metadata=checkpoint_metadata,
        dtype_name=pipeline.dtype_name,
        device=device,
        window_count=int(stream_metadata.emitted_windows),
        tokens_per_window=int(config.model.seq_length),
        logical_token_count=int(stream_metadata.original_token_count),
        base_token_count=int(stream_metadata.symbol_count_without_eos),
        tail_padding_tokens=int(stream_metadata.tail_padding_tokens),
        token_merge_size=int(config.data.token_merge_size),
        frequency_total=int(pipeline.frequency_total),
        arithmetic_metadata=arithmetic_metadata,
        compression_metrics=compression_metrics,
        framing_metrics=framing_metrics,
        extra={
            "source_name": source_name,
            "requested_bytes": requested_bytes,
            "stream_read_bytes": int(stream_metadata.read_bytes),
            "stream_filtered_bytes": int(stream_metadata.filtered_bytes),
            "eos_appended": bool(stream_metadata.eos_appended),
            "payload_sha256": payload_sha256(payload),
            **dict(extra_metadata or {}),
        },
    )
    return payload, metadata


def decode_window_payload_with_pipeline(
    *,
    pipeline: WindowCodecPipeline,
    payload: bytes,
    expected_tokens_cpu: torch.Tensor | None = None,
    threads: int = 0,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    parse_started = perf_counter()
    header, streams = parse_window_payload(payload)
    parse_seconds = perf_counter() - parse_started
    if int(header.get("window_count", len(streams))) != len(streams):
        raise RuntimeError(f"window stream count mismatch: {len(streams)} != {header.get('window_count')}")
    if int(header.get("format_version", 0)) >= WINDOW_CODEC_V3_FORMAT_VERSION:
        valid_lengths_cpu = valid_lengths_from_logical_token_count(
            window_count=int(header["window_count"]),
            tokens_per_window=int(header["tokens_per_window"]),
            logical_token_count=int(header["logical_token_count"]),
        )
    else:
        valid_lengths_cpu = torch.full((len(streams),), int(header["tokens_per_window"]), dtype=torch.long)
    decoded, metrics = pipeline.decode_streams(
        streams=streams,
        tokens_per_window=int(header["tokens_per_window"]),
        valid_lengths_cpu=valid_lengths_cpu,
        pad_id=int(pipeline.config.model.pad_id),
        threads=int(threads),
        expected_tokens_cpu=expected_tokens_cpu,
    )
    metrics["decode_parse_seconds"] = float(parse_seconds)
    metrics["decode_wall_seconds"] = float(metrics.get("decode_wall_seconds", 0.0)) + float(parse_seconds)
    metrics["decode_tokens_per_second"] = int(valid_lengths_cpu.sum().item()) / max(float(metrics["decode_wall_seconds"]), 1e-12)
    return decoded, metrics, header


def decode_v2_payload_with_pipeline(
    *,
    pipeline: WindowCodecPipeline,
    payload: bytes,
    expected_tokens_cpu: torch.Tensor | None = None,
    threads: int = 0,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    return decode_window_payload_with_pipeline(
        pipeline=pipeline,
        payload=payload,
        expected_tokens_cpu=expected_tokens_cpu,
        threads=threads,
    )


def actual_window_codec_metrics(
    *,
    payload: bytes,
    metadata: dict[str, Any],
    include_codec_baselines: bool = True,
) -> dict[str, Any]:
    sample_bytes = len(payload) if payload else int(metadata.get("stream_filtered_bytes", metadata.get("base_count", 0)))
    sample_bases = int(metadata["base_count"])
    compressed_bytes = int(metadata["framed_bytes"])
    compressed_bits = compressed_bytes * 8.0
    compression_process_seconds = float(metadata["compression_wall_seconds"]) + float(metadata.get("framing_seconds", 0.0))
    arithmetic_quantize_seconds = float(metadata.get("compression_quantize_seconds", 0.0))
    arithmetic_range_seconds = float(metadata.get("compression_range_seconds", 0.0))
    data_transfer_seconds = float(metadata.get("compression_interval_transfer_seconds", 0.0))
    model_forward_seconds = float(metadata.get("compression_model_seconds", 0.0))
    arithmetic_encode_seconds = arithmetic_quantize_seconds + arithmetic_range_seconds
    accounted_seconds = model_forward_seconds + data_transfer_seconds + arithmetic_encode_seconds
    python_overhead_seconds = max(0.0, compression_process_seconds - accounted_seconds)

    return {
        "mode": "windows_nonoverlap",
        "sample_bytes": sample_bytes,
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": int(metadata["token_count"]),
        "encoded_token_count": int(metadata.get("encoded_token_count", metadata["token_count"])),
        "tail_padding_tokens": int(metadata.get("tail_padding_tokens", 0)),
        "encoded_padding_tokens": int(metadata.get("encoded_padding_tokens", 0)),
        "theoretical_bits": compressed_bits,
        "theoretical_bits_per_base": compressed_bits / max(sample_bases, 1),
        "theoretical_bits_source": "actual_window_codec_bytes",
        "arithmetic_coded_bytes": compressed_bytes,
        "arithmetic_payload_bytes": int(metadata.get("compression_encoded_payload_bytes", compressed_bytes)),
        "framing_bytes": int(metadata.get("framing_bytes", 0)),
        "compressed_bpb_payload_only": float(metadata.get("compressed_bpb_payload_only", 0.0)),
        "compressed_bpb_with_framing": float(metadata.get("compressed_bpb_with_framing", compressed_bits / max(sample_bases, 1))),
        "arithmetic_bits_per_base": compressed_bits / max(sample_bases, 1),
        "model_forward_seconds": model_forward_seconds,
        "softmax_seconds": 0.0,
        "model_forward_softmax_seconds": model_forward_seconds,
        "probability_compute_seconds": model_forward_seconds,
        "data_transfer_seconds": data_transfer_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "gpu_prefix_aggregate_seconds": 0.0,
        "window_build_seconds": 0.0,
        "python_overhead_seconds": python_overhead_seconds,
        "cpu_small_alphabet_quantize_seconds": arithmetic_quantize_seconds,
        "arithmetic_quantize_seconds": arithmetic_quantize_seconds,
        "arithmetic_range_seconds": arithmetic_range_seconds,
        "arithmetic_wrapper_seconds": 0.0,
        "arithmetic_interval_transfer_seconds": data_transfer_seconds,
        "fast_floor_interval_seconds": arithmetic_quantize_seconds,
        "arithmetic_backend": "fast_cpp",
        "compression_process_seconds": compression_process_seconds,
        "compression_bytes_per_second": sample_bytes / max(compression_process_seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(compression_process_seconds, 1e-12),
        "compression_symbols_per_second": int(metadata["encoded_token_count"]) / max(compression_process_seconds, 1e-12),
        "arithmetic_coding_mode": "model_symbol",
        "arithmetic_quantization_mode": "fast_floor",
        "arithmetic_merge_size": 1,
        "emitted_arithmetic_symbol_count": int(metadata["compression_emitted_symbols"]),
        "arithmetic_frequency_total": int(metadata["arithmetic_frequency_total"]),
        "arithmetic_vocab_size": int(metadata["arithmetic_vocab_size"]),
        "arithmetic_target_uniform_mass": float(metadata["arithmetic_target_uniform_mass"]),
        "arithmetic_effective_uniform_mass": float(metadata["arithmetic_effective_uniform_mass"]),
        "window_policy": "contiguous_train_style",
        "window_stride": int(metadata["tokens_per_window"]),
        "cache_reuse": True,
        "window_codec_format_version": int(metadata["format_version"]),
        "window_codec_payload_sha256": metadata.get("payload_sha256"),
        "window_codec_framing": metadata.get("framing"),
        "window_codec_compression_mode": metadata.get("compression_mode"),
        "window_codec_quantization_mode": metadata.get("quantization_mode"),
        "window_codec_batch_size": int(metadata["compression_batch_size"]),
        "window_codec_devices": metadata.get("window_codec_devices"),
        "window_codec_shard_count": metadata.get("window_codec_shard_count"),
        "window_codec_parallelism": metadata.get("window_codec_parallelism"),
        "window_codec_window_count": int(metadata["window_count"]),
        "window_codec_tokens_per_window": int(metadata["tokens_per_window"]),
        "window_codec_encoded_token_count": int(metadata["encoded_token_count"]),
        "window_codec_logical_token_count": int(metadata["token_count"]),
        "window_codec_tail_padding_tokens": int(metadata.get("tail_padding_tokens", 0) or 0),
        "window_codec_framed_bytes": compressed_bytes,
        "window_codec_framing_bytes": int(metadata["framing_bytes"]),
        "window_codec_framing_overhead_bpb": float(metadata["framing_overhead_bpb"]),
        "window_codec_bpb_with_framing": float(metadata["compressed_bpb_with_framing"]),
        "window_codec_bpb_payload_only": float(metadata["compressed_bpb_payload_only"]),
        "window_codec_bases_per_second": float(metadata["compression_bases_per_second"]),
        **baseline_sizes(payload, include_codec_baselines=include_codec_baselines and bool(payload)),
    }
