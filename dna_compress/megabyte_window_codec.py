from __future__ import annotations

import hashlib
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
WINDOW_CODEC_NAME = "megabyte_window_fast_floor"
LENGTH_PREFIX_BYTES = 4


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
        timings = encoder.encode_interval_matrix(lows_cpu, highs_cpu, totals_cpu)
        range_seconds += timings.range_seconds
        emitted += timings.emitted_count
        streams.extend(encoder.finish())

    wall_seconds = perf_counter() - started
    encoded_payload_bytes = sum(len(stream) for stream in streams)
    token_count = int(tokens_cpu.numel())
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

    output = torch.empty((len(streams), int(tokens_per_window)), dtype=torch.long)
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
        decoder = BatchedStreamingArithmeticDecoder(chunk_streams, threads=int(threads))
        stepper = MegabyteBatchedDecodeStepper(model, batch_size=chunk_size, device=device, dtype_name=dtype_name)
        for step in range(int(tokens_per_window)):
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
            symbols_cpu = decoder.decode_frequency_rows_with_totals(freqs_cpu, totals_cpu)
            arith_seconds += perf_counter() - arith_started
            output[start : start + chunk_size, step] = symbols_cpu

            if expected_tokens_cpu is not None:
                expected = expected_tokens_cpu[start : start + chunk_size, step]
                if not torch.equal(symbols_cpu, expected):
                    mismatches += int((symbols_cpu != expected).sum().item())

            token_started = perf_counter()
            symbols_gpu = symbols_cpu.to(device, non_blocking=True)
            sync_if_cuda(device)
            token_transfer_seconds += perf_counter() - token_started
            stepper.accept_symbols(symbols_gpu)

    wall_seconds = perf_counter() - started
    token_count = int(window_count) * int(tokens_per_window)
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


def actual_window_codec_metrics(
    *,
    payload: bytes,
    metadata: dict[str, Any],
    include_codec_baselines: bool = True,
) -> dict[str, Any]:
    sample_bytes = len(payload)
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
        "theoretical_bits": compressed_bits,
        "theoretical_bits_per_base": compressed_bits / max(sample_bases, 1),
        "theoretical_bits_source": "actual_window_codec_bytes",
        "arithmetic_coded_bytes": compressed_bytes,
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
        **baseline_sizes(payload, include_codec_baselines=include_codec_baselines),
    }
