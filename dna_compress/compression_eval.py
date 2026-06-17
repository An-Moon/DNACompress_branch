from __future__ import annotations

import math
from contextlib import nullcontext
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import torch

from .compression import (
    baseline_sizes,
    resolve_arithmetic_coding_metadata,
)
from .fast_arithmetic import (
    ARITHMETIC_QUANTIZATION_MODES,
    StreamingArithmeticEncoder,
    fast_floor_intervals_from_probabilities,
)
from .fixed_token_factorization import (
    FixedTokenArithmeticFactorizer,
    factorize_fixed_token_log_probs,
)
from .tokenization import normalize_alphabet, tokenize_source_bytes


CompressionMode = str

SLIDING_TOKEN_MODE = "sliding_token"
NON_OVERLAP_MODE = "windows_nonoverlap"
OVERLAP_MODE = "windows_overlap"
SUPPORTED_COMPRESSION_MODES = (
    SLIDING_TOKEN_MODE,
    NON_OVERLAP_MODE,
    OVERLAP_MODE,
)
MEGABYTE_ARITHMETIC_CODING_MODES = (
    "model_symbol",
    "base_prefix_exact_gpu_cpu",
)
MEGABYTE_ARITHMETIC_QUANTIZATION_MODES = ARITHMETIC_QUANTIZATION_MODES


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def sample_payload(source: bytes, requested_bytes: int | None) -> bytes:
    if requested_bytes is None or requested_bytes <= 0 or len(source) <= requested_bytes:
        return source
    return source[:requested_bytes]


def _symbols_with_optional_eos(
    payload: bytes,
    eos_id: int | None,
    token_merge_size: int,
    token_merge_alphabet: str,
) -> list[int]:
    symbols = tokenize_source_bytes(payload, token_merge_size, token_merge_alphabet)
    if eos_id is not None:
        symbols.append(eos_id)
    return symbols


def _normalized_base_sequence(
    payload: bytes,
    token_merge_alphabet: str,
    token_merge_size: int,
) -> list[str]:
    alphabet = normalize_alphabet(token_merge_alphabet)
    byte_to_base: dict[int, str] = {}
    for base in alphabet:
        byte_to_base[ord(base)] = base
        byte_to_base[ord(base.lower())] = base
    normalized = [byte_to_base[byte_value] for byte_value in payload if byte_value in byte_to_base]
    if token_merge_size <= 1:
        return normalized
    full_base_count = (len(normalized) // token_merge_size) * token_merge_size
    return normalized[:full_base_count]


class _PositionBitsProfileAccumulator:
    def __init__(self, *, seq_length: int, token_merge_size: int, alphabet: str) -> None:
        self.alphabet = normalize_alphabet(alphabet)
        self.seq_length = seq_length
        self.token_merge_size = token_merge_size
        self.window_base_length = seq_length * token_merge_size
        self._base_to_index = {base: index for index, base in enumerate(self.alphabet)}
        self._counts = np.zeros((len(self.alphabet), self.window_base_length), dtype=np.int64)
        self._sum_bits = np.zeros((len(self.alphabet), self.window_base_length), dtype=np.float64)
        self.total_bits = 0.0

    def add_token(self, *, token_bits: float, window_token_index: int, base_chars: list[str]) -> None:
        if len(base_chars) != self.token_merge_size:
            raise ValueError(
                f"Expected {self.token_merge_size} bases for one merged token, got {len(base_chars)}."
            )
        base_bits = token_bits / self.token_merge_size
        window_base_start = window_token_index * self.token_merge_size
        for base_offset, base_char in enumerate(base_chars):
            position = window_base_start + base_offset
            base_index = self._base_to_index[base_char]
            self._counts[base_index, position] += 1
            self._sum_bits[base_index, position] += base_bits
        self.total_bits += token_bits

    def as_dict(self) -> dict[str, object]:
        return {
            "alphabet": self.alphabet,
            "window_base_length": self.window_base_length,
            "counts": self._counts.tolist(),
            "sum_bits_per_base": self._sum_bits.tolist(),
            "total_bits": self.total_bits,
            "excludes_eos": True,
        }


def _build_position_bits_profile(
    *,
    payload: bytes,
    seq_length: int,
    token_merge_size: int,
    token_merge_alphabet: str,
    symbol_count_without_eos: int,
    collect_position_bits_profile: bool,
) -> tuple[_PositionBitsProfileAccumulator | None, list[str]]:
    if not collect_position_bits_profile:
        return None, []

    normalized_bases = _normalized_base_sequence(payload, token_merge_alphabet, token_merge_size)
    expected_base_count = symbol_count_without_eos * token_merge_size
    if len(normalized_bases) != expected_base_count:
        raise RuntimeError(
            "Normalized base count does not match tokenized sample size: "
            f"{len(normalized_bases)} != {expected_base_count}"
        )
    return (
        _PositionBitsProfileAccumulator(
            seq_length=seq_length,
            token_merge_size=token_merge_size,
            alphabet=token_merge_alphabet,
        ),
        normalized_bases,
    )


def _record_position_bits_profile(
    *,
    accumulator: _PositionBitsProfileAccumulator | None,
    normalized_bases: list[str],
    target_log_probs: torch.Tensor,
    global_token_start: int,
    window_token_start: int,
    token_merge_size: int,
    token_count_without_eos: int,
) -> None:
    if accumulator is None:
        return

    token_bits = (-target_log_probs / math.log(2)).detach().cpu().tolist()
    for offset, token_bit_value in enumerate(token_bits):
        global_token_index = global_token_start + offset
        if global_token_index >= token_count_without_eos:
            continue
        base_start = global_token_index * token_merge_size
        base_chars = normalized_bases[base_start : base_start + token_merge_size]
        accumulator.add_token(
            token_bits=float(token_bit_value),
            window_token_index=window_token_start + offset,
            base_chars=base_chars,
        )


def _finalize_metrics(
    *,
    payload: bytes,
    symbols: list[int],
    symbol_count_without_eos: int,
    token_merge_size: int,
    total_bits: float,
    encoded: bytes,
    mode: CompressionMode,
    model_forward_seconds: float,
    softmax_seconds: float,
    data_transfer_seconds: float,
    arithmetic_encode_seconds: float,
    arithmetic_metadata: dict[str, object],
    arithmetic_coding_mode: str = "model_symbol",
    arithmetic_quantization_mode: str = "precise",
    arithmetic_merge_size: int = 1,
    gpu_prefix_aggregate_seconds: float = 0.0,
    window_build_seconds: float = 0.0,
    python_overhead_seconds: float = 0.0,
    cpu_small_alphabet_quantize_seconds: float = 0.0,
    arithmetic_range_seconds: float = 0.0,
    arithmetic_wrapper_seconds: float = 0.0,
    arithmetic_interval_transfer_seconds: float = 0.0,
    fast_floor_interval_seconds: float = 0.0,
    arithmetic_backend: str = "python",
    emitted_arithmetic_symbol_count: int | None = None,
    mode_details: dict[str, object] | None = None,
    position_bits_profile: dict[str, object] | None = None,
    include_codec_baselines: bool = True,
) -> dict[str, object]:
    sample_bytes = len(payload)
    sample_bases = symbol_count_without_eos * token_merge_size
    model_forward_softmax_seconds = model_forward_seconds + softmax_seconds
    compression_process_seconds = (
        model_forward_seconds
        + softmax_seconds
        + window_build_seconds
        + python_overhead_seconds
        + gpu_prefix_aggregate_seconds
        + data_transfer_seconds
        + arithmetic_encode_seconds
    )
    if emitted_arithmetic_symbol_count is None:
        emitted_arithmetic_symbol_count = len(symbols)
    metrics = {
        "mode": mode,
        "sample_bytes": sample_bytes,
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": len(symbols),
        "theoretical_bits": total_bits,
        "theoretical_bits_per_base": total_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": len(encoded),
        "arithmetic_bits_per_base": (len(encoded) * 8) / max(sample_bases, 1),
        "model_forward_seconds": model_forward_seconds,
        "softmax_seconds": softmax_seconds,
        "model_forward_softmax_seconds": model_forward_softmax_seconds,
        "probability_compute_seconds": model_forward_softmax_seconds,
        "data_transfer_seconds": data_transfer_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "gpu_prefix_aggregate_seconds": gpu_prefix_aggregate_seconds,
        "window_build_seconds": window_build_seconds,
        "python_overhead_seconds": python_overhead_seconds,
        "cpu_small_alphabet_quantize_seconds": cpu_small_alphabet_quantize_seconds,
        "arithmetic_quantize_seconds": cpu_small_alphabet_quantize_seconds,
        "arithmetic_range_seconds": arithmetic_range_seconds,
        "arithmetic_wrapper_seconds": arithmetic_wrapper_seconds,
        "arithmetic_interval_transfer_seconds": arithmetic_interval_transfer_seconds,
        "fast_floor_interval_seconds": fast_floor_interval_seconds,
        "arithmetic_backend": arithmetic_backend,
        "compression_process_seconds": compression_process_seconds,
        "compression_bytes_per_second": sample_bytes / max(compression_process_seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(compression_process_seconds, 1e-12),
        "compression_symbols_per_second": emitted_arithmetic_symbol_count / max(compression_process_seconds, 1e-12),
        "arithmetic_coding_mode": arithmetic_coding_mode,
        "arithmetic_quantization_mode": arithmetic_quantization_mode,
        "arithmetic_merge_size": arithmetic_merge_size,
        "emitted_arithmetic_symbol_count": emitted_arithmetic_symbol_count,
        **arithmetic_metadata,
        **baseline_sizes(payload, include_codec_baselines=include_codec_baselines),
        **(mode_details or {}),
    }
    if position_bits_profile is not None:
        metrics["position_bits_profile"] = position_bits_profile
        metrics["position_bits_profile_total_bits"] = float(position_bits_profile["total_bits"])
        metrics["position_bits_profile_excludes_eos"] = bool(position_bits_profile.get("excludes_eos", True))
    return metrics


def _resolve_megabyte_arithmetic_metadata(
    *,
    vocab_size: int,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    arithmetic_coding_mode: str,
    factorizer: FixedTokenArithmeticFactorizer | None,
) -> dict[str, object]:
    if arithmetic_coding_mode == "model_symbol":
        metadata_vocab_size = vocab_size
    elif arithmetic_coding_mode == "base_prefix_exact_gpu_cpu":
        if factorizer is None:
            raise ValueError("factorizer is required for base_prefix_exact_gpu_cpu mode.")
        metadata_vocab_size = factorizer.max_emitted_vocab_size
    else:
        raise ValueError(f"Unsupported Megabyte arithmetic coding mode '{arithmetic_coding_mode}'.")
    return resolve_arithmetic_coding_metadata(
        vocab_size=metadata_vocab_size,
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
    )


def _encode_model_symbol_probabilities(
    *,
    probability_rows: np.ndarray,
    target_symbols: np.ndarray,
    total: int,
    encoder: StreamingArithmeticEncoder,
    quantization_mode: str = "precise",
) -> tuple[float, float, int]:
    if quantization_mode == "precise":
        timings = encoder.encode_probability_rows(probability_rows, target_symbols, total=total)
    elif quantization_mode == "fast_floor_cpu":
        timings = encoder.encode_probability_rows_fast_floor(probability_rows, target_symbols, total=total)
    else:
        raise ValueError(f"CPU probability encoding does not support quantization mode '{quantization_mode}'.")
    return timings.quantize_seconds, timings.range_seconds, timings.emitted_count


def _encode_model_symbol_fast_floor_gpu(
    *,
    log_prob_rows: torch.Tensor,
    target_symbols: torch.Tensor,
    total: int,
    encoder: StreamingArithmeticEncoder,
    device: torch.device,
) -> tuple[float, float, int, float, float]:
    if device.type != "cuda" or log_prob_rows.device.type != "cuda":
        raise ValueError("fast_floor_gpu quantization requires CUDA tensors and a CUDA device.")
    _sync_if_cuda(device)
    interval_started = perf_counter()
    lows, highs, totals = fast_floor_intervals_from_probabilities(
        log_prob_rows.float().exp(),
        target_symbols,
        total=total,
    )
    _sync_if_cuda(device)
    interval_seconds = perf_counter() - interval_started

    transfer_started = perf_counter()
    lows_cpu = lows.cpu()
    highs_cpu = highs.cpu()
    totals_cpu = totals.cpu()
    _sync_if_cuda(device)
    interval_transfer_seconds = perf_counter() - transfer_started

    timings = encoder.encode_intervals(lows_cpu, highs_cpu, totals_cpu, interval_transfer_seconds=interval_transfer_seconds)
    return (
        interval_seconds,
        timings.range_seconds,
        timings.emitted_count,
        interval_transfer_seconds,
        interval_seconds,
    )


def _encode_factorized_probabilities(
    *,
    log_prob_rows: torch.Tensor,
    target_symbols: torch.Tensor,
    factorizer: FixedTokenArithmeticFactorizer,
    total: int,
    encoder: StreamingArithmeticEncoder,
) -> tuple[float, float, float, int, float, float]:
    aggregate_started = perf_counter()
    factorized = factorize_fixed_token_log_probs(
        log_probs=log_prob_rows,
        target_token_ids=target_symbols,
        factorizer=factorizer,
    )
    gpu_prefix_aggregate_seconds = perf_counter() - aggregate_started

    transfer_started = perf_counter()
    root_probabilities = factorized.root_probabilities.cpu().numpy()
    root_symbols = factorized.root_symbols.cpu().numpy()
    regular_step_probabilities = tuple(step.cpu().numpy() for step in factorized.regular_step_probabilities)
    regular_step_symbols = tuple(step.cpu().numpy() for step in factorized.regular_step_symbols)
    regular_row_positions = factorized.regular_row_positions.cpu().numpy()
    special_step_probabilities = factorized.special_step_probabilities.cpu().numpy()
    special_step_symbols = factorized.special_step_symbols.cpu().numpy()
    special_row_positions = factorized.special_row_positions.cpu().numpy()
    data_transfer_seconds = perf_counter() - transfer_started

    row_count = int(root_symbols.shape[0])
    step_probabilities = [root_probabilities]
    step_symbols = [root_symbols]
    step_row_positions = [np.arange(row_count, dtype=np.int64)]
    for step_probabilities_batch, step_symbols_batch in zip(regular_step_probabilities, regular_step_symbols):
        step_probabilities.append(step_probabilities_batch)
        step_symbols.append(step_symbols_batch)
        step_row_positions.append(regular_row_positions)
    if special_step_probabilities.shape[0] > 0:
        step_probabilities.append(special_step_probabilities)
        step_symbols.append(special_step_symbols)
        step_row_positions.append(special_row_positions)
    timings = encoder.encode_grouped_steps(
        step_probabilities,
        step_symbols,
        step_row_positions,
        row_count=row_count,
        total=total,
    )

    total_bits = float((-factorized.target_log_probs / math.log(2)).sum().item())
    return (
        total_bits,
        data_transfer_seconds,
        gpu_prefix_aggregate_seconds,
        factorized.emitted_symbol_count,
        timings.quantize_seconds,
        timings.range_seconds,
    )


def compress_sequence_sliding_token(
    *,
    model: torch.nn.Module,
    payload: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    token_merge_size: int,
    token_merge_alphabet: str,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    arithmetic_coding_mode: str = "model_symbol",
    arithmetic_quantization_mode: str = "precise",
    arithmetic_merge_size: int = 1,
    arithmetic_backend: str = "python",
    factorizer: FixedTokenArithmeticFactorizer | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    collect_position_bits_profile: bool = False,
    include_codec_baselines: bool = True,
) -> dict[str, object]:
    del collect_position_bits_profile
    symbols = _symbols_with_optional_eos(payload, eos_id, token_merge_size, token_merge_alphabet)
    symbols_tensor = torch.tensor(symbols, dtype=torch.long)

    padded = torch.full((len(symbols) + seq_length - 1,), pad_id, dtype=torch.long)
    padded[-len(symbols) :] = symbols_tensor
    all_windows = padded.unfold(0, seq_length, 1)

    total_bits = 0.0
    total_batches = max(1, math.ceil(len(symbols) / batch_size))
    processed_batches = 0
    encoder = StreamingArithmeticEncoder(arithmetic_backend)
    model_forward_seconds = 0.0
    softmax_seconds = 0.0
    window_build_seconds = 0.0
    data_transfer_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    arithmetic_wrapper_seconds = 0.0
    arithmetic_interval_transfer_seconds = 0.0
    fast_floor_interval_seconds = 0.0
    gpu_prefix_aggregate_seconds = 0.0
    cpu_small_alphabet_quantize_seconds = 0.0
    arithmetic_range_seconds = 0.0
    emitted_arithmetic_symbol_count = 0
    arithmetic_metadata = _resolve_megabyte_arithmetic_metadata(
        vocab_size=model.vocab_size if hasattr(model, "vocab_size") else int(max(symbols, default=pad_id) + 1),
        arithmetic_frequency_total=arithmetic_frequency_total,
        arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
        arithmetic_coding_mode=arithmetic_coding_mode,
        factorizer=factorizer,
    )
    if arithmetic_quantization_mode not in MEGABYTE_ARITHMETIC_QUANTIZATION_MODES:
        raise ValueError(
            "arithmetic_quantization_mode must be one of: "
            + ", ".join(MEGABYTE_ARITHMETIC_QUANTIZATION_MODES)
        )
    if arithmetic_coding_mode != "model_symbol" and arithmetic_quantization_mode != "precise":
        raise ValueError("non-precise arithmetic quantization modes are only supported with model_symbol coding.")
    if arithmetic_quantization_mode == "fast_floor_gpu":
        raise ValueError("fast_floor_gpu is only implemented for windows_nonoverlap model_symbol compression.")

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

            if arithmetic_coding_mode == "model_symbol":
                if arithmetic_quantization_mode == "fast_floor_gpu":
                    raise ValueError("fast_floor_gpu is only implemented for windows_nonoverlap model_symbol compression.")
                target_log_probs = log_probs.gather(1, targets_device.unsqueeze(1)).squeeze(1)
                total_bits += float((-target_log_probs / math.log(2)).sum().item())
                transfer_started = perf_counter()
                probs_np = log_probs.float().exp().cpu().numpy()
                data_transfer_seconds += perf_counter() - transfer_started
                quantize_seconds, range_seconds, emitted_count = _encode_model_symbol_probabilities(
                    probability_rows=probs_np,
                    target_symbols=targets_np,
                    total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                    encoder=encoder,
                    quantization_mode=arithmetic_quantization_mode,
                )
                cpu_small_alphabet_quantize_seconds += quantize_seconds
                arithmetic_range_seconds += range_seconds
                arithmetic_encode_seconds += quantize_seconds + range_seconds
                emitted_arithmetic_symbol_count += emitted_count
            elif arithmetic_coding_mode == "base_prefix_exact_gpu_cpu":
                if factorizer is None:
                    raise ValueError("factorizer is required for base_prefix_exact_gpu_cpu mode.")
                (
                    batch_bits,
                    batch_transfer_seconds,
                    batch_gpu_aggregate_seconds,
                    emitted_count,
                    batch_quantize_seconds,
                    batch_range_seconds,
                ) = _encode_factorized_probabilities(
                    log_prob_rows=log_probs,
                    target_symbols=targets_device,
                    factorizer=factorizer,
                    total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                    encoder=encoder,
                )
                total_bits += batch_bits
                data_transfer_seconds += batch_transfer_seconds
                gpu_prefix_aggregate_seconds += batch_gpu_aggregate_seconds
                cpu_small_alphabet_quantize_seconds += batch_quantize_seconds
                arithmetic_range_seconds += batch_range_seconds
                arithmetic_encode_seconds += batch_quantize_seconds + batch_range_seconds
                emitted_arithmetic_symbol_count += emitted_count
            else:
                raise ValueError(f"Unsupported Megabyte arithmetic coding mode '{arithmetic_coding_mode}'.")

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    finish_started = perf_counter()
    encoded = encoder.finish()
    finish_seconds = perf_counter() - finish_started
    arithmetic_encode_seconds += finish_seconds
    arithmetic_wrapper_seconds += finish_seconds

    return _finalize_metrics(
        payload=payload,
        symbols=symbols,
        symbol_count_without_eos=len(symbols) - (1 if eos_id is not None else 0),
        token_merge_size=token_merge_size,
        total_bits=total_bits,
        encoded=encoded,
        mode=SLIDING_TOKEN_MODE,
        model_forward_seconds=model_forward_seconds,
        softmax_seconds=softmax_seconds,
        data_transfer_seconds=data_transfer_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        arithmetic_metadata=arithmetic_metadata,
        arithmetic_coding_mode=arithmetic_coding_mode,
        arithmetic_quantization_mode=arithmetic_quantization_mode,
        arithmetic_merge_size=arithmetic_merge_size,
        gpu_prefix_aggregate_seconds=gpu_prefix_aggregate_seconds,
        window_build_seconds=window_build_seconds,
        cpu_small_alphabet_quantize_seconds=cpu_small_alphabet_quantize_seconds,
        arithmetic_range_seconds=arithmetic_range_seconds,
        arithmetic_wrapper_seconds=arithmetic_wrapper_seconds,
        arithmetic_interval_transfer_seconds=arithmetic_interval_transfer_seconds,
        fast_floor_interval_seconds=fast_floor_interval_seconds,
        arithmetic_backend=encoder.backend,
        emitted_arithmetic_symbol_count=emitted_arithmetic_symbol_count,
        include_codec_baselines=include_codec_baselines,
        mode_details={
            "window_stride": 1,
            "window_policy": "right_aligned_sliding_context",
            "cache_reuse": False,
        },
    )


def _window_starts_for_overlap(total_symbols: int, seq_length: int, stride: int) -> list[int]:
    if total_symbols <= 0:
        return [0]
    if total_symbols <= seq_length:
        return [0]

    extra = total_symbols - seq_length
    num_extra_windows = math.ceil(extra / stride)
    return [0] + [stride * index for index in range(1, num_extra_windows + 1)]


def compress_sequence_train_windows(
    *,
    model: torch.nn.Module,
    payload: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    overlap_stride: int | None = None,
    token_merge_size: int,
    token_merge_alphabet: str,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    arithmetic_coding_mode: str = "model_symbol",
    arithmetic_quantization_mode: str = "precise",
    arithmetic_merge_size: int = 1,
    arithmetic_backend: str = "python",
    factorizer: FixedTokenArithmeticFactorizer | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    collect_position_bits_profile: bool = False,
    include_codec_baselines: bool = True,
) -> dict[str, object]:
    process_started = perf_counter()
    symbols = _symbols_with_optional_eos(payload, eos_id, token_merge_size, token_merge_alphabet)
    token_count_without_eos = len(symbols) - (1 if eos_id is not None else 0)
    position_bits_profile, normalized_bases = _build_position_bits_profile(
        payload=payload,
        seq_length=seq_length,
        token_merge_size=token_merge_size,
        token_merge_alphabet=token_merge_alphabet,
        symbol_count_without_eos=token_count_without_eos,
        collect_position_bits_profile=collect_position_bits_profile,
    )
    total_bits = 0.0
    encoder = StreamingArithmeticEncoder(arithmetic_backend)

    if overlap_stride is None:
        mode = NON_OVERLAP_MODE
        window_starts = list(range(0, len(symbols), seq_length)) or [0]
    else:
        if overlap_stride <= 0 or overlap_stride >= seq_length:
            raise ValueError("overlap_stride must satisfy 0 < overlap_stride < seq_length")
        mode = OVERLAP_MODE
        window_starts = _window_starts_for_overlap(len(symbols), seq_length, overlap_stride)

    total_batches = max(1, math.ceil(len(window_starts) / batch_size))
    processed_batches = 0
    model_forward_seconds = 0.0
    softmax_seconds = 0.0
    window_build_seconds = 0.0
    data_transfer_seconds = 0.0
    arithmetic_encode_seconds = 0.0
    arithmetic_wrapper_seconds = 0.0
    arithmetic_interval_transfer_seconds = 0.0
    fast_floor_interval_seconds = 0.0
    gpu_prefix_aggregate_seconds = 0.0
    cpu_small_alphabet_quantize_seconds = 0.0
    arithmetic_range_seconds = 0.0
    emitted_arithmetic_symbol_count = 0
    arithmetic_metadata = _resolve_megabyte_arithmetic_metadata(
        vocab_size=model.vocab_size if hasattr(model, "vocab_size") else (max(pad_id, eos_id) + 1),
        arithmetic_frequency_total=arithmetic_frequency_total,
        arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
        arithmetic_coding_mode=arithmetic_coding_mode,
        factorizer=factorizer,
    )
    if arithmetic_quantization_mode not in MEGABYTE_ARITHMETIC_QUANTIZATION_MODES:
        raise ValueError(
            "arithmetic_quantization_mode must be one of: "
            + ", ".join(MEGABYTE_ARITHMETIC_QUANTIZATION_MODES)
        )
    if arithmetic_coding_mode != "model_symbol" and arithmetic_quantization_mode != "precise":
        raise ValueError("non-precise arithmetic quantization modes are only supported with model_symbol coding.")
    if arithmetic_quantization_mode == "fast_floor_gpu" and (
        overlap_stride is not None or arithmetic_coding_mode != "model_symbol" or collect_position_bits_profile
    ):
        raise ValueError(
            "fast_floor_gpu is only implemented for windows_nonoverlap model_symbol compression "
            "without position-bits profiling."
        )

    optimized_nonoverlap_model_symbol = (
        overlap_stride is None
        and arithmetic_coding_mode == "model_symbol"
        and not collect_position_bits_profile
    )
    symbols_tensor = torch.tensor(symbols, dtype=torch.long)
    all_windows = None
    all_lengths = None
    if optimized_nonoverlap_model_symbol:
        window_build_started = perf_counter()
        window_count = len(window_starts)
        padded_length = max(window_count * seq_length, seq_length)
        padded_symbols = torch.full((padded_length,), pad_id, dtype=torch.long)
        if symbols_tensor.numel() > 0:
            padded_symbols[: symbols_tensor.numel()] = symbols_tensor
        all_windows = padded_symbols.view(window_count, seq_length)
        starts_tensor = torch.arange(0, window_count * seq_length, seq_length, dtype=torch.long)
        all_lengths = torch.clamp(symbols_tensor.numel() - starts_tensor, min=0, max=seq_length)
        window_build_seconds += perf_counter() - window_build_started

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(window_starts), batch_size):
            starts = window_starts[batch_start : batch_start + batch_size]
            window_build_started = perf_counter()
            if optimized_nonoverlap_model_symbol:
                if all_windows is None or all_lengths is None:
                    raise RuntimeError("optimized window tensors were not initialized")
                window_slice = slice(batch_start, batch_start + len(starts))
                windows = all_windows[window_slice]
                lengths_tensor = all_lengths[window_slice]
            else:
                windows = torch.full((len(starts), seq_length), pad_id, dtype=torch.long)
                lengths: list[int] = []
                for row_index, start in enumerate(starts):
                    chunk = symbols[start : start + seq_length]
                    lengths.append(len(chunk))
                    if chunk:
                        windows[row_index, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
                lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            window_build_seconds += perf_counter() - window_build_started

            _sync_if_cuda(device)
            transfer_started = perf_counter()
            batch = windows.to(device, non_blocking=True)
            _sync_if_cuda(device)
            data_transfer_seconds += perf_counter() - transfer_started

            with autocast_context(device, dtype_name):
                _sync_if_cuda(device)
                forward_started = perf_counter()
                output = model(batch, return_loss=False)
                _sync_if_cuda(device)
                model_forward_seconds += perf_counter() - forward_started

                _sync_if_cuda(device)
                softmax_started = perf_counter()
                log_probs = torch.log_softmax(output.lm_logits, dim=-1)
                _sync_if_cuda(device)
                softmax_seconds += perf_counter() - softmax_started

            if optimized_nonoverlap_model_symbol:
                valid_mask = (
                    torch.arange(seq_length, device=device).unsqueeze(0)
                    < lengths_tensor.to(device, non_blocking=True).unsqueeze(1)
                )
                if valid_mask.any():
                    flat_log_probs = log_probs[valid_mask]
                    target_slice = symbols_tensor[
                        batch_start * seq_length : batch_start * seq_length + int(lengths_tensor.sum().item())
                    ]
                    targets_device = target_slice.to(device, non_blocking=True)
                    target_log_probs = flat_log_probs.gather(1, targets_device.unsqueeze(1)).squeeze(1)
                    _sync_if_cuda(device)
                    total_bits_started = perf_counter()
                    total_bits += float((-target_log_probs / math.log(2)).sum().item())
                    _sync_if_cuda(device)
                    data_transfer_seconds += perf_counter() - total_bits_started

                    _sync_if_cuda(device)
                    if arithmetic_quantization_mode == "fast_floor_gpu":
                        encode_started = perf_counter()
                        (
                            quantize_seconds,
                            range_seconds,
                            emitted_count,
                            interval_transfer_seconds,
                            interval_seconds,
                        ) = _encode_model_symbol_fast_floor_gpu(
                            log_prob_rows=flat_log_probs,
                            target_symbols=targets_device,
                            total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                            encoder=encoder,
                            device=device,
                        )
                        encode_wall_seconds = max(0.0, perf_counter() - encode_started - interval_transfer_seconds)
                        data_transfer_seconds += interval_transfer_seconds
                        arithmetic_interval_transfer_seconds += interval_transfer_seconds
                        fast_floor_interval_seconds += interval_seconds
                    else:
                        transfer_started = perf_counter()
                        probs_np = flat_log_probs.float().exp().cpu().numpy()
                        targets_np = target_slice.numpy()
                        _sync_if_cuda(device)
                        data_transfer_seconds += perf_counter() - transfer_started

                        encode_started = perf_counter()
                        quantize_seconds, range_seconds, emitted_count = _encode_model_symbol_probabilities(
                            probability_rows=probs_np,
                            target_symbols=targets_np,
                            total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                            encoder=encoder,
                            quantization_mode=arithmetic_quantization_mode,
                        )
                        encode_wall_seconds = perf_counter() - encode_started
                    cpu_small_alphabet_quantize_seconds += quantize_seconds
                    arithmetic_range_seconds += range_seconds
                    arithmetic_encode_seconds += encode_wall_seconds
                    arithmetic_wrapper_seconds += max(0.0, encode_wall_seconds - quantize_seconds - range_seconds)
                    emitted_arithmetic_symbol_count += emitted_count

                processed_batches += 1
                if progress_callback is not None:
                    progress_callback(processed_batches, total_batches)
                continue

            for row_index, (start, chunk_length) in enumerate(zip(starts, lengths_tensor.tolist())):
                if chunk_length <= 0:
                    continue

                local_start = 0
                if overlap_stride is not None and start > 0:
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
                _record_position_bits_profile(
                    accumulator=position_bits_profile,
                    normalized_bases=normalized_bases,
                    target_log_probs=target_log_probs,
                    global_token_start=start + local_start,
                    window_token_start=local_start,
                    token_merge_size=token_merge_size,
                    token_count_without_eos=token_count_without_eos,
                )
                if arithmetic_coding_mode == "model_symbol":
                    _sync_if_cuda(device)
                    total_bits_started = perf_counter()
                    total_bits += float((-target_log_probs / math.log(2)).sum().item())
                    _sync_if_cuda(device)
                    data_transfer_seconds += perf_counter() - total_bits_started
                    _sync_if_cuda(device)
                    transfer_started = perf_counter()
                    probs_np = row_log_probs.float().exp().cpu().numpy()
                    targets_np = targets_device.cpu().numpy()
                    _sync_if_cuda(device)
                    data_transfer_seconds += perf_counter() - transfer_started
                    encode_started = perf_counter()
                    quantize_seconds, range_seconds, emitted_count = _encode_model_symbol_probabilities(
                        probability_rows=probs_np,
                        target_symbols=targets_np,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                        quantization_mode=arithmetic_quantization_mode,
                    )
                    encode_wall_seconds = perf_counter() - encode_started
                    cpu_small_alphabet_quantize_seconds += quantize_seconds
                    arithmetic_range_seconds += range_seconds
                    arithmetic_encode_seconds += encode_wall_seconds
                    arithmetic_wrapper_seconds += max(0.0, encode_wall_seconds - quantize_seconds - range_seconds)
                    emitted_arithmetic_symbol_count += emitted_count
                elif arithmetic_coding_mode == "base_prefix_exact_gpu_cpu":
                    if factorizer is None:
                        raise ValueError("factorizer is required for base_prefix_exact_gpu_cpu mode.")
                    (
                        batch_bits,
                        batch_transfer_seconds,
                        batch_gpu_aggregate_seconds,
                        emitted_count,
                        batch_quantize_seconds,
                        batch_range_seconds,
                    ) = _encode_factorized_probabilities(
                        log_prob_rows=row_log_probs,
                        target_symbols=targets_device,
                        factorizer=factorizer,
                        total=int(arithmetic_metadata["arithmetic_frequency_total"]),
                        encoder=encoder,
                    )
                    total_bits += batch_bits
                    data_transfer_seconds += batch_transfer_seconds
                    gpu_prefix_aggregate_seconds += batch_gpu_aggregate_seconds
                    cpu_small_alphabet_quantize_seconds += batch_quantize_seconds
                    arithmetic_range_seconds += batch_range_seconds
                    arithmetic_encode_seconds += batch_quantize_seconds + batch_range_seconds
                    emitted_arithmetic_symbol_count += emitted_count
                else:
                    raise ValueError(f"Unsupported Megabyte arithmetic coding mode '{arithmetic_coding_mode}'.")

            processed_batches += 1
            if progress_callback is not None:
                progress_callback(processed_batches, total_batches)

    finish_started = perf_counter()
    encoded = encoder.finish()
    finish_seconds = perf_counter() - finish_started
    arithmetic_encode_seconds += finish_seconds
    arithmetic_wrapper_seconds += finish_seconds
    measured_process_seconds = perf_counter() - process_started
    accounted_process_seconds = (
        model_forward_seconds
        + softmax_seconds
        + window_build_seconds
        + gpu_prefix_aggregate_seconds
        + data_transfer_seconds
        + arithmetic_encode_seconds
    )
    python_overhead_seconds = max(0.0, measured_process_seconds - accounted_process_seconds)

    mode_details: dict[str, object] = {
        "window_policy": "contiguous_train_style",
        "cache_reuse": False,
    }
    if overlap_stride is None:
        mode_details["window_stride"] = seq_length
    else:
        mode_details.update(
            {
                "window_stride": overlap_stride,
                "cache_note": (
                    "This evaluator recomputes each overlap window exactly. Patch-aligned overlap "
                    "makes cache reuse plausible, but hidden-state reuse is not implemented here."
                ),
            }
        )

    return _finalize_metrics(
        payload=payload,
        symbols=symbols,
        symbol_count_without_eos=token_count_without_eos,
        token_merge_size=token_merge_size,
        total_bits=total_bits,
        encoded=encoded,
        mode=mode,
        model_forward_seconds=model_forward_seconds,
        softmax_seconds=softmax_seconds,
        data_transfer_seconds=data_transfer_seconds,
        arithmetic_encode_seconds=arithmetic_encode_seconds,
        arithmetic_metadata=arithmetic_metadata,
        arithmetic_coding_mode=arithmetic_coding_mode,
        arithmetic_quantization_mode=arithmetic_quantization_mode,
        arithmetic_merge_size=arithmetic_merge_size,
        gpu_prefix_aggregate_seconds=gpu_prefix_aggregate_seconds,
        window_build_seconds=window_build_seconds,
        python_overhead_seconds=python_overhead_seconds,
        cpu_small_alphabet_quantize_seconds=cpu_small_alphabet_quantize_seconds,
        arithmetic_range_seconds=arithmetic_range_seconds,
        arithmetic_wrapper_seconds=arithmetic_wrapper_seconds,
        arithmetic_interval_transfer_seconds=arithmetic_interval_transfer_seconds,
        fast_floor_interval_seconds=fast_floor_interval_seconds,
        arithmetic_backend=encoder.backend,
        emitted_arithmetic_symbol_count=emitted_arithmetic_symbol_count,
        mode_details=mode_details,
        position_bits_profile=position_bits_profile.as_dict() if position_bits_profile is not None else None,
        include_codec_baselines=include_codec_baselines,
    )


def compress_source(
    *,
    model: torch.nn.Module,
    source: bytes,
    seq_length: int,
    pad_id: int,
    eos_id: int,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    requested_bytes: int | None,
    mode: CompressionMode,
    overlap_stride: int = 1,
    token_merge_size: int = 1,
    token_merge_alphabet: str = "ACGTN",
    arithmetic_frequency_total: int | None = None,
    arithmetic_target_uniform_mass: float = 0.01,
    arithmetic_coding_mode: str = "model_symbol",
    arithmetic_quantization_mode: str = "precise",
    arithmetic_merge_size: int = 1,
    arithmetic_backend: str = "python",
    factorizer: FixedTokenArithmeticFactorizer | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    collect_position_bits_profile: bool = False,
    include_codec_baselines: bool = True,
) -> dict[str, object]:
    payload = sample_payload(source, requested_bytes)
    if mode == SLIDING_TOKEN_MODE:
        return compress_sequence_sliding_token(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            token_merge_size=token_merge_size,
            token_merge_alphabet=token_merge_alphabet,
            arithmetic_frequency_total=arithmetic_frequency_total,
            arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
            arithmetic_coding_mode=arithmetic_coding_mode,
            arithmetic_quantization_mode=arithmetic_quantization_mode,
            arithmetic_merge_size=arithmetic_merge_size,
            arithmetic_backend=arithmetic_backend,
            factorizer=factorizer,
            progress_callback=progress_callback,
            collect_position_bits_profile=collect_position_bits_profile,
            include_codec_baselines=include_codec_baselines,
        )
    if mode == NON_OVERLAP_MODE:
        return compress_sequence_train_windows(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            overlap_stride=None,
            token_merge_size=token_merge_size,
            token_merge_alphabet=token_merge_alphabet,
            arithmetic_frequency_total=arithmetic_frequency_total,
            arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
            arithmetic_coding_mode=arithmetic_coding_mode,
            arithmetic_quantization_mode=arithmetic_quantization_mode,
            arithmetic_merge_size=arithmetic_merge_size,
            arithmetic_backend=arithmetic_backend,
            factorizer=factorizer,
            progress_callback=progress_callback,
            collect_position_bits_profile=collect_position_bits_profile,
            include_codec_baselines=include_codec_baselines,
        )
    if mode == OVERLAP_MODE:
        return compress_sequence_train_windows(
            model=model,
            payload=payload,
            seq_length=seq_length,
            pad_id=pad_id,
            eos_id=eos_id,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            overlap_stride=overlap_stride,
            token_merge_size=token_merge_size,
            token_merge_alphabet=token_merge_alphabet,
            arithmetic_frequency_total=arithmetic_frequency_total,
            arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
            arithmetic_coding_mode=arithmetic_coding_mode,
            arithmetic_quantization_mode=arithmetic_quantization_mode,
            arithmetic_merge_size=arithmetic_merge_size,
            arithmetic_backend=arithmetic_backend,
            factorizer=factorizer,
            progress_callback=progress_callback,
            collect_position_bits_profile=collect_position_bits_profile,
            include_codec_baselines=include_codec_baselines,
        )
    raise ValueError(f"Unsupported compression mode '{mode}'")
 

def summarize_per_source(
    per_source: Iterable[dict[str, object]],
) -> dict[str, object]:
    rows = list(per_source)
    total_sample_bytes = sum(int(row["sample_bytes"]) for row in rows)
    total_sample_bases = sum(int(row["sample_bases"]) for row in rows)
    total_theoretical_bits = sum(float(row["theoretical_bits"]) for row in rows)
    total_arithmetic_bytes = sum(int(row["arithmetic_coded_bytes"]) for row in rows)
    total_ascii_bytes = sum(int(row["ascii_bytes"]) for row in rows)
    total_two_bit_pack_bytes = sum(int(row["two_bit_pack_bytes"]) for row in rows)
    def _optional_int_sum(key: str) -> int | None:
        values = [row.get(key) for row in rows]
        if any(value is None for value in values):
            return None
        return sum(int(value) for value in values)

    total_gzip_bytes = _optional_int_sum("gzip_bytes")
    total_bz2_bytes = _optional_int_sum("bz2_bytes")
    total_lzma_bytes = _optional_int_sum("lzma_bytes")
    total_model_forward_seconds = sum(
        float(row.get("model_forward_seconds", row.get("model_forward_softmax_seconds", row.get("probability_compute_seconds", 0.0))))
        for row in rows
    )
    total_softmax_seconds = sum(float(row.get("softmax_seconds", 0.0)) for row in rows)
    total_window_build_seconds = sum(float(row.get("window_build_seconds", 0.0)) for row in rows)
    total_python_overhead_seconds = sum(float(row.get("python_overhead_seconds", 0.0)) for row in rows)
    total_data_transfer_seconds = sum(float(row.get("data_transfer_seconds", 0.0)) for row in rows)
    total_arithmetic_interval_transfer_seconds = sum(
        float(row.get("arithmetic_interval_transfer_seconds", 0.0)) for row in rows
    )
    total_fast_floor_interval_seconds = sum(float(row.get("fast_floor_interval_seconds", 0.0)) for row in rows)
    total_arithmetic_encode_seconds = sum(float(row.get("arithmetic_encode_seconds", 0.0)) for row in rows)
    total_arithmetic_range_seconds = sum(float(row.get("arithmetic_range_seconds", 0.0)) for row in rows)
    total_arithmetic_wrapper_seconds = sum(float(row.get("arithmetic_wrapper_seconds", 0.0)) for row in rows)
    total_gpu_prefix_aggregate_seconds = sum(float(row.get("gpu_prefix_aggregate_seconds", 0.0)) for row in rows)
    total_cpu_small_alphabet_quantize_seconds = sum(
        float(row.get("cpu_small_alphabet_quantize_seconds", 0.0)) for row in rows
    )
    total_compression_process_seconds = sum(float(row.get("compression_process_seconds", 0.0)) for row in rows)
    total_emitted_arithmetic_symbol_count = sum(int(row.get("emitted_arithmetic_symbol_count", 0) or 0) for row in rows)
    total_core_model_theoretical_bits = sum(float(row.get("core_model_theoretical_bits", row.get("theoretical_bits", 0.0))) for row in rows)
    total_tail_base_count = sum(int(row.get("tail_base_count", 0) or 0) for row in rows)
    total_tail_side_info_bits = sum(int(row.get("tail_side_info_bits", 0) or 0) for row in rows)

    summary = {
        "source_count": len(rows),
        "total_sample_bytes": total_sample_bytes,
        "total_sample_bases": total_sample_bases,
        "total_theoretical_bits": total_theoretical_bits,
        "total_theoretical_bits_per_base": total_theoretical_bits / max(total_sample_bases, 1),
        "total_arithmetic_coded_bytes": total_arithmetic_bytes,
        "total_arithmetic_bits_per_base": (total_arithmetic_bytes * 8) / max(total_sample_bases, 1),
        "total_ascii_bytes": total_ascii_bytes,
        "total_two_bit_pack_bytes": total_two_bit_pack_bytes,
        "total_gzip_bytes": total_gzip_bytes,
        "total_bz2_bytes": total_bz2_bytes,
        "total_lzma_bytes": total_lzma_bytes,
        "total_model_forward_seconds": total_model_forward_seconds,
        "total_softmax_seconds": total_softmax_seconds,
        "total_window_build_seconds": total_window_build_seconds,
        "total_python_overhead_seconds": total_python_overhead_seconds,
        "total_data_transfer_seconds": total_data_transfer_seconds,
        "total_arithmetic_interval_transfer_seconds": total_arithmetic_interval_transfer_seconds,
        "total_fast_floor_interval_seconds": total_fast_floor_interval_seconds,
        "total_arithmetic_encode_seconds": total_arithmetic_encode_seconds,
        "total_arithmetic_quantize_seconds": total_cpu_small_alphabet_quantize_seconds,
        "total_arithmetic_range_seconds": total_arithmetic_range_seconds,
        "total_arithmetic_wrapper_seconds": total_arithmetic_wrapper_seconds,
        "total_gpu_prefix_aggregate_seconds": total_gpu_prefix_aggregate_seconds,
        "total_cpu_small_alphabet_quantize_seconds": total_cpu_small_alphabet_quantize_seconds,
        "total_compression_process_seconds": total_compression_process_seconds,
        "total_compression_bytes_per_second": total_sample_bytes / max(total_compression_process_seconds, 1e-12),
        "total_compression_bases_per_second": total_sample_bases / max(total_compression_process_seconds, 1e-12),
        "total_emitted_arithmetic_symbol_count": total_emitted_arithmetic_symbol_count,
        "total_core_model_theoretical_bits": total_core_model_theoretical_bits,
        "total_tail_base_count": total_tail_base_count,
        "total_tail_side_info_bits": total_tail_side_info_bits,
    }
    if total_compression_process_seconds > 0:
        summary["total_compression_symbols_per_second"] = (
            total_emitted_arithmetic_symbol_count / total_compression_process_seconds
        )
    for key in (
        "arithmetic_frequency_total",
        "arithmetic_vocab_size",
        "arithmetic_target_uniform_mass",
        "arithmetic_effective_uniform_mass",
        "arithmetic_coding_mode",
        "arithmetic_quantization_mode",
        "arithmetic_merge_size",
        "arithmetic_backend",
    ):
        if rows and all(row.get(key) == rows[0].get(key) for row in rows):
            summary[key] = rows[0].get(key)
    return summary
