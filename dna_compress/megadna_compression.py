from __future__ import annotations

import math
from time import perf_counter
from typing import Callable

import numpy as np
import torch

from .compression import (
    ArithmeticEncoder,
    baseline_sizes,
    probabilities_to_cumulative_batch,
    resolve_arithmetic_coding_metadata,
)
from .compression_eval import (
    NON_OVERLAP_MODE,
    OVERLAP_MODE,
    SLIDING_TOKEN_MODE,
    CompressionMode,
    autocast_context,
    sample_payload,
)
from .megadna_data import encode_source_for_megadna
from .megadna_loader import MEGADNA_EOS_ID, MEGADNA_PAD_ID, MEGADNA_VOCAB


SUPPORTED_MEGADNA_COMPRESSION_MODES = (
    SLIDING_TOKEN_MODE,
    NON_OVERLAP_MODE,
    OVERLAP_MODE,
)


def _symbols_with_eos(payload: bytes, *, non_acgt_policy: str) -> tuple[bytes, list[int], int]:
    encoded = encode_source_for_megadna(payload, non_acgt_policy=non_acgt_policy)
    symbols = list(encoded.token_bytes)
    symbols.append(MEGADNA_EOS_ID)
    return encoded.original, symbols, encoded.dropped_bytes


def _encode_model_symbol_probabilities(
    *,
    probability_rows: np.ndarray,
    target_symbols: np.ndarray,
    total: int,
    encoder: ArithmeticEncoder,
) -> tuple[float, int]:
    quantize_started = perf_counter()
    cumulative_batch = probabilities_to_cumulative_batch(probability_rows, total=total)
    quantize_seconds = perf_counter() - quantize_started
    for cumulative, target in zip(cumulative_batch, target_symbols):
        encoder.update(cumulative, int(target))
    return quantize_seconds, len(target_symbols)


def _window_starts_for_overlap(total_symbols: int, seq_length: int, stride: int) -> list[int]:
    if total_symbols <= 0:
        return [0]
    if total_symbols <= seq_length:
        return [0]
    extra = total_symbols - seq_length
    num_extra_windows = math.ceil(extra / stride)
    return [0] + [stride * index for index in range(1, num_extra_windows + 1)]


def _finalize_metrics(
    *,
    original_payload: bytes,
    dropped_bytes: int,
    symbols: list[int],
    total_bits: float,
    encoded: bytes,
    mode: str,
    model_forward_seconds: float,
    softmax_seconds: float,
    data_transfer_seconds: float,
    arithmetic_encode_seconds: float,
    cpu_small_alphabet_quantize_seconds: float,
    emitted_arithmetic_symbol_count: int,
    arithmetic_metadata: dict[str, object],
    mode_details: dict[str, object],
) -> dict[str, object]:
    baseline = baseline_sizes(original_payload)
    sample_bases = max(len(original_payload), 1)
    symbol_count_without_eos = max(len(symbols) - 1, 0)
    return {
        "mode": mode,
        "sample_bytes": len(original_payload),
        "sample_bases": len(original_payload),
        "sample_symbols_with_eos": len(symbols),
        "symbol_count_without_eos": symbol_count_without_eos,
        "dropped_non_acgt_bytes": dropped_bytes,
        "theoretical_bits": total_bits,
        "theoretical_bits_per_byte": total_bits / max(len(original_payload), 1),
        "theoretical_bits_per_base": total_bits / sample_bases,
        "arithmetic_coded_bytes": len(encoded),
        "arithmetic_bits_per_byte": (len(encoded) * 8) / max(len(original_payload), 1),
        "arithmetic_bits_per_base": (len(encoded) * 8) / sample_bases,
        "ascii_bytes": baseline["ascii_bytes"],
        "two_bit_pack_bytes": baseline["two_bit_pack_bytes"],
        "gzip_bytes": baseline["gzip_bytes"],
        "bz2_bytes": baseline["bz2_bytes"],
        "lzma_bytes": baseline["lzma_bytes"],
        "model_forward_seconds": model_forward_seconds,
        "softmax_seconds": softmax_seconds,
        "data_transfer_seconds": data_transfer_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "cpu_small_alphabet_quantize_seconds": cpu_small_alphabet_quantize_seconds,
        "compression_process_seconds": (
            model_forward_seconds + softmax_seconds + data_transfer_seconds + arithmetic_encode_seconds
        ),
        "emitted_arithmetic_symbol_count": emitted_arithmetic_symbol_count,
        "arithmetic_frequency_total": arithmetic_metadata["arithmetic_frequency_total"],
        "arithmetic_vocab_size": arithmetic_metadata["arithmetic_vocab_size"],
        "arithmetic_target_uniform_mass": arithmetic_metadata["arithmetic_target_uniform_mass"],
        "arithmetic_effective_uniform_mass": arithmetic_metadata["arithmetic_effective_uniform_mass"],
        "arithmetic_coding_mode": "model_symbol",
        "arithmetic_merge_size": 1,
        "model_vocab": list(MEGADNA_VOCAB),
        "mode_details": mode_details,
    }


def compress_megadna_source(
    *,
    model: torch.nn.Module,
    source: bytes,
    seq_length: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    requested_bytes: int | None,
    mode: CompressionMode,
    overlap_stride: int = 1,
    arithmetic_frequency_total: int | None = None,
    arithmetic_target_uniform_mass: float = 0.01,
    non_acgt_policy: str = "reject",
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    sampled = sample_payload(source, requested_bytes)
    original_payload, symbols, dropped_bytes = _symbols_with_eos(sampled, non_acgt_policy=non_acgt_policy)
    symbols_tensor = torch.tensor(symbols, dtype=torch.long)

    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=len(MEGADNA_VOCAB),
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
    )
    total_bits = 0.0
    encoder = ArithmeticEncoder()
    model_forward_seconds = 0.0
    softmax_seconds = 0.0
    data_transfer_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    cpu_small_alphabet_quantize_seconds = 0.0
    emitted_arithmetic_symbol_count = 0

    if mode == SLIDING_TOKEN_MODE:
        padded = torch.full((len(symbols) + seq_length - 1,), MEGADNA_PAD_ID, dtype=torch.long)
        padded[-len(symbols) :] = symbols_tensor
        all_windows = padded.unfold(0, seq_length, 1)
        total_batches = max(1, math.ceil(len(symbols) / batch_size))

        model.eval()
        with torch.no_grad():
            for start in range(0, len(symbols), batch_size):
                target_slice = symbols_tensor[start : start + batch_size]
                batch = all_windows[start : start + target_slice.shape[0]]
                targets_np = target_slice.numpy()

                transfer_started = perf_counter()
                batch = batch.to(device, non_blocking=True)
                targets_device = target_slice.to(device, non_blocking=True)
                data_transfer_seconds += perf_counter() - transfer_started

                with autocast_context(device, dtype_name):
                    forward_started = perf_counter()
                    output = model(batch, return_loss=False)
                    model_forward_seconds += perf_counter() - forward_started

                    softmax_started = perf_counter()
                    log_probs = torch.log_softmax(output.lm_logits[:, -1, :], dim=-1)
                    softmax_seconds += perf_counter() - softmax_started

                target_log_probs = log_probs.gather(1, targets_device.unsqueeze(1)).squeeze(1)
                total_bits += float((-target_log_probs / math.log(2)).sum().item())
                transfer_started = perf_counter()
                probs_np = log_probs.float().exp().cpu().numpy()
                data_transfer_seconds += perf_counter() - transfer_started
                encode_started = perf_counter()
                quantize_seconds, emitted_count = _encode_model_symbol_probabilities(
                    probability_rows=probs_np,
                    target_symbols=targets_np,
                    total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                    encoder=encoder,
                )
                cpu_small_alphabet_quantize_seconds += quantize_seconds
                emitted_arithmetic_symbol_count += emitted_count
                arithmetic_encode_seconds += perf_counter() - encode_started

                if progress_callback is not None:
                    progress_callback((start // batch_size) + 1, total_batches)

        return _finalize_metrics(
            original_payload=original_payload,
            dropped_bytes=dropped_bytes,
            symbols=symbols,
            total_bits=total_bits,
            encoded=encoder.finish(),
            mode=mode,
            model_forward_seconds=model_forward_seconds,
            softmax_seconds=softmax_seconds,
            data_transfer_seconds=data_transfer_seconds,
            arithmetic_encode_seconds=arithmetic_encode_seconds,
            cpu_small_alphabet_quantize_seconds=cpu_small_alphabet_quantize_seconds,
            emitted_arithmetic_symbol_count=emitted_arithmetic_symbol_count,
            arithmetic_metadata=arithmetic_metadata,
            mode_details={
                "window_stride": 1,
                "window_policy": "right_aligned_sliding_context",
                "cache_reuse": False,
            },
        )

    if mode == NON_OVERLAP_MODE:
        window_starts = list(range(0, len(symbols), seq_length)) or [0]
        effective_mode = NON_OVERLAP_MODE
    elif mode == OVERLAP_MODE:
        if overlap_stride <= 0 or overlap_stride >= seq_length:
            raise ValueError("overlap_stride must satisfy 0 < overlap_stride < seq_length")
        window_starts = _window_starts_for_overlap(len(symbols), seq_length, overlap_stride)
        effective_mode = OVERLAP_MODE
    else:
        raise ValueError(f"Unsupported megaDNA compression mode '{mode}'.")

    total_batches = max(1, math.ceil(len(window_starts) / batch_size))
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(window_starts), batch_size):
            starts = window_starts[batch_start : batch_start + batch_size]
            windows = torch.full((len(starts), seq_length), MEGADNA_PAD_ID, dtype=torch.long)
            lengths: list[int] = []
            for row_index, start in enumerate(starts):
                chunk = symbols[start : start + seq_length]
                lengths.append(len(chunk))
                if chunk:
                    windows[row_index, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)

            transfer_started = perf_counter()
            batch = windows.to(device, non_blocking=True)
            data_transfer_seconds += perf_counter() - transfer_started

            with autocast_context(device, dtype_name):
                forward_started = perf_counter()
                output = model(batch, return_loss=False)
                model_forward_seconds += perf_counter() - forward_started

                softmax_started = perf_counter()
                log_probs = torch.log_softmax(output.lm_logits, dim=-1)
                softmax_seconds += perf_counter() - softmax_started

            encode_started = perf_counter()
            for row_index, (start, chunk_length) in enumerate(zip(starts, lengths)):
                if chunk_length <= 0:
                    continue
                local_start = 0
                if mode == OVERLAP_MODE and start > 0:
                    local_start = min(seq_length - overlap_stride, chunk_length)
                row_log_probs = log_probs[row_index, local_start:chunk_length, :]
                if row_log_probs.shape[0] == 0:
                    continue
                targets_device = torch.tensor(
                    symbols[start + local_start : start + chunk_length],
                    dtype=torch.long,
                    device=device,
                )
                target_log_probs = row_log_probs.gather(1, targets_device.unsqueeze(1)).squeeze(1)
                total_bits += float((-target_log_probs / math.log(2)).sum().item())
                transfer_started = perf_counter()
                probs_np = row_log_probs.float().exp().cpu().numpy()
                targets_np = targets_device.cpu().numpy()
                data_transfer_seconds += perf_counter() - transfer_started
                quantize_seconds, emitted_count = _encode_model_symbol_probabilities(
                    probability_rows=probs_np,
                    target_symbols=targets_np,
                    total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                    encoder=encoder,
                )
                cpu_small_alphabet_quantize_seconds += quantize_seconds
                emitted_arithmetic_symbol_count += emitted_count
            arithmetic_encode_seconds += perf_counter() - encode_started

            if progress_callback is not None:
                progress_callback((batch_start // batch_size) + 1, total_batches)

    mode_details: dict[str, object] = {
        "window_policy": "contiguous_train_style",
        "cache_reuse": False,
        "window_stride": seq_length if mode == NON_OVERLAP_MODE else overlap_stride,
    }
    return _finalize_metrics(
        original_payload=original_payload,
        dropped_bytes=dropped_bytes,
        symbols=symbols,
        total_bits=total_bits,
        encoded=encoder.finish(),
        mode=effective_mode,
        model_forward_seconds=model_forward_seconds,
        softmax_seconds=softmax_seconds,
        data_transfer_seconds=data_transfer_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        cpu_small_alphabet_quantize_seconds=cpu_small_alphabet_quantize_seconds,
        emitted_arithmetic_symbol_count=emitted_arithmetic_symbol_count,
        arithmetic_metadata=arithmetic_metadata,
        mode_details=mode_details,
    )
