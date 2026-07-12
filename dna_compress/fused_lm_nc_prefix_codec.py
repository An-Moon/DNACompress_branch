from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from queue import Queue
import threading
from time import perf_counter
from typing import Any

import numpy as np
import torch

from .compression import baseline_sizes, resolve_arithmetic_coding_metadata
from .config import ExperimentConfig
from .fast_arithmetic import BatchedStreamingArithmeticEncoder
from .fast_nc_prefix import FusedNcPrefixStreamingEncoder
from .megabyte_batched_decode import MegabyteBatchedDecodeStepper
from .megabyte_window_codec import memory_stats, sync_if_cuda
from .noncontiguous_prefix_codec import (
    NoncontiguousPrefixConfig,
    compute_noncontiguous_prefix_probabilities,
)
from .tokenization import normalize_alphabet


_FACTOR_TABLE_CACHE: dict[tuple[int, str, str, str], list[torch.Tensor]] = {}


@dataclass(frozen=True)
class FusedTokenWindows:
    sequence: str
    core_sequence: str
    tail_sequence: str
    tokens: torch.Tensor
    token_base_symbols: torch.Tensor
    valid_token_lengths: torch.Tensor
    token_merge_size: int
    token_window_bases: int
    model_uses_ascii_tokens: bool
    model_token_alphabet: str
    filtered_out_bases: int


class MatrixBackedNcPrefixStreamingState:
    """Depth API backed by the current C++ nc_prefix full probability path.

    This keeps the fused compressor's synchronization explicit while leaving the
    native true-streaming state as a later drop-in replacement.
    """

    def __init__(self, sequence: str, config: NoncontiguousPrefixConfig) -> None:
        self.config = config
        self.result = compute_noncontiguous_prefix_probabilities(
            sequence,
            config,
            return_probabilities=True,
            summary_only=False,
        )
        self.probabilities = np.asarray(self.result.probabilities, dtype=np.float64)
        self.target_symbols = np.asarray(self.result.target_symbols, dtype=np.int64)

    @property
    def metadata(self) -> dict[str, Any]:
        return self.result.metadata

    def predict_positions(self, positions: np.ndarray) -> np.ndarray:
        return self.probabilities[np.asarray(positions, dtype=np.int64)]

    def accept_depth(self, positions: np.ndarray, symbols: np.ndarray) -> None:
        del positions, symbols


def _sync(device: torch.device) -> None:
    sync_if_cuda(device)


def _filtered_acgt(payload: bytes) -> tuple[str, int]:
    text = payload.decode("ascii", errors="ignore").upper()
    chars = [ch for ch in text if ch in {"A", "C", "G", "T"}]
    return "".join(chars), len(text) - len(chars)


def _base_symbol_lookup(alphabet: str) -> dict[str, int]:
    alphabet = normalize_alphabet(alphabet)
    if alphabet != "ACGT":
        raise ValueError("fused LM/nc_prefix compression currently requires alphabet='ACGT'")
    return {base: index for index, base in enumerate(alphabet)}


def _model_uses_ascii_single_base_tokens(model: torch.nn.Module, config: ExperimentConfig) -> bool:
    if int(config.data.token_merge_size) != 1:
        return False
    vocab_size = int(getattr(model.config, "V", getattr(model.config, "vocab_size", 0)))
    return vocab_size >= 256


def _encode_regular_tokens_and_base_symbols(
    sequence: str,
    *,
    token_merge_size: int,
    output_alphabet: str,
    model_token_alphabet: str,
    model_uses_ascii_tokens: bool,
) -> tuple[np.ndarray, np.ndarray]:
    base_to_symbol = _base_symbol_lookup(output_alphabet)
    model_base_to_symbol = {base: index for index, base in enumerate(normalize_alphabet(model_token_alphabet))}
    full_base_count = (len(sequence) // int(token_merge_size)) * int(token_merge_size)
    core = sequence[:full_base_count]
    if full_base_count == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, int(token_merge_size)), dtype=np.int64)

    base_symbols = np.asarray([base_to_symbol[base] for base in core], dtype=np.int64)
    token_base_symbols = base_symbols.reshape(-1, int(token_merge_size))
    if int(token_merge_size) == 1 and model_uses_ascii_tokens:
        tokens = np.frombuffer(core.encode("ascii"), dtype=np.uint8).astype(np.int64, copy=False)
        return tokens, token_base_symbols

    missing = sorted(set(core) - set(model_base_to_symbol))
    if missing:
        raise ValueError(
            "filtered ACGT sequence contains bases absent from the model token alphabet "
            f"{model_token_alphabet!r}: {''.join(missing)!r}"
        )
    model_digits = np.asarray([model_base_to_symbol[base] for base in core], dtype=np.int64).reshape(
        -1, int(token_merge_size)
    )
    base = len(model_token_alphabet)
    weights = np.asarray([base ** power for power in range(int(token_merge_size) - 1, -1, -1)], dtype=np.int64)
    tokens = (model_digits * weights).sum(axis=1, dtype=np.int64)
    return tokens.astype(np.int64, copy=False), token_base_symbols


def build_fused_token_windows(
    payload: bytes,
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    window_bases: int,
    alphabet: str = "ACGT",
) -> FusedTokenWindows:
    alphabet = normalize_alphabet(alphabet)
    token_merge_size = int(config.data.token_merge_size)
    if token_merge_size <= 0:
        raise ValueError("token_merge_size must be positive")
    if int(window_bases) <= 0:
        raise ValueError("window_bases must be positive")
    if int(window_bases) % token_merge_size != 0:
        raise ValueError("window_bases must be divisible by token_merge_size")

    sequence, filtered_out = _filtered_acgt(payload)
    model_uses_ascii = _model_uses_ascii_single_base_tokens(model, config)
    model_token_alphabet = normalize_alphabet(config.data.token_merge_alphabet)
    tokens_np, token_base_symbols_np = _encode_regular_tokens_and_base_symbols(
        sequence,
        token_merge_size=token_merge_size,
        output_alphabet=alphabet,
        model_token_alphabet=model_token_alphabet,
        model_uses_ascii_tokens=model_uses_ascii,
    )
    core_base_count = int(tokens_np.shape[0]) * token_merge_size
    core_sequence = sequence[:core_base_count]
    tail_sequence = sequence[core_base_count:]
    if tokens_np.shape[0] == 0:
        raise ValueError("fused compression requires at least one complete LM token")

    tokens_per_window = int(window_bases) // token_merge_size
    window_count = math.ceil(int(tokens_np.shape[0]) / tokens_per_window)
    padded_token_count = window_count * tokens_per_window
    pad_id = int(config.model.pad_id)
    padded = np.full((padded_token_count,), pad_id, dtype=np.int64)
    padded[: tokens_np.shape[0]] = tokens_np
    token_base_symbols = np.zeros((padded_token_count, token_merge_size), dtype=np.int64)
    token_base_symbols[: token_base_symbols_np.shape[0], :] = token_base_symbols_np

    valid_lengths = np.full((window_count,), tokens_per_window, dtype=np.int64)
    tail_tokens = int(tokens_np.shape[0]) - (window_count - 1) * tokens_per_window
    valid_lengths[-1] = tail_tokens
    return FusedTokenWindows(
        sequence=sequence,
        core_sequence=core_sequence,
        tail_sequence=tail_sequence,
        tokens=torch.from_numpy(padded.reshape(window_count, tokens_per_window)).long().contiguous(),
        token_base_symbols=torch.from_numpy(token_base_symbols.reshape(window_count, tokens_per_window, token_merge_size))
        .long()
        .contiguous(),
        valid_token_lengths=torch.from_numpy(valid_lengths).long().contiguous(),
        token_merge_size=token_merge_size,
        token_window_bases=int(window_bases),
        model_uses_ascii_tokens=model_uses_ascii,
        model_token_alphabet=model_token_alphabet,
        filtered_out_bases=filtered_out,
    )


def _acgt_factorization_tables(
    *,
    token_merge_size: int,
    model_token_alphabet: str,
    output_alphabet: str,
    device: torch.device,
) -> list[torch.Tensor]:
    model_token_alphabet = normalize_alphabet(model_token_alphabet)
    output_alphabet = normalize_alphabet(output_alphabet)
    cache_key = (int(token_merge_size), model_token_alphabet, output_alphabet, str(device))
    cached = _FACTOR_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    token_base = len(model_token_alphabet)
    output_indices = [model_token_alphabet.index(base) for base in output_alphabet]
    tables: list[torch.Tensor] = []
    for base_index in range(int(token_merge_size)):
        prefix_count = len(output_alphabet) ** base_index
        future_count = len(output_alphabet) ** (int(token_merge_size) - base_index - 1)
        table = torch.empty((prefix_count, len(output_alphabet), future_count), dtype=torch.long)
        for prefix_code in range(prefix_count):
            prefix_digits: list[int] = []
            cursor = prefix_code
            for power in range(base_index - 1, -1, -1):
                divisor = len(output_alphabet) ** power
                digit = cursor // divisor
                cursor %= divisor
                prefix_digits.append(output_indices[digit])
            for candidate_digit, candidate_index in enumerate(output_indices):
                for future_code in range(future_count):
                    future_digits: list[int] = []
                    cursor = future_code
                    for power in range(int(token_merge_size) - base_index - 2, -1, -1):
                        divisor = len(output_alphabet) ** power
                        digit = cursor // divisor
                        cursor %= divisor
                        future_digits.append(output_indices[digit])
                    token_digits = prefix_digits + [candidate_index] + future_digits
                    token_id = 0
                    for digit in token_digits:
                        token_id = token_id * token_base + digit
                    table[prefix_code, candidate_digit, future_code] = token_id
        tables.append(table.to(device=device))
    _FACTOR_TABLE_CACHE[cache_key] = tables
    return tables


@lru_cache(maxsize=16)
def _acgtn_merge3_acgt_token_ids_tuple() -> tuple[int, ...]:
    ids: list[int] = []
    for first in range(4):
        for second in range(4):
            for third in range(4):
                ids.append(first * 25 + second * 5 + third)
    return tuple(ids)


def _acgtn_merge3_log_probs_to_base_steps(
    logits: torch.Tensor,
    target_base_symbols: torch.Tensor,
) -> list[torch.Tensor]:
    ids = torch.tensor(_acgtn_merge3_acgt_token_ids_tuple(), dtype=torch.long, device=logits.device)
    selected = logits.float().index_select(1, ids).reshape(logits.shape[0], 4, 4, 4)
    target_base_symbols = target_base_symbols.to(device=logits.device, dtype=torch.long)
    rows = torch.arange(logits.shape[0], device=logits.device)
    first = torch.softmax(torch.logsumexp(selected, dim=(2, 3)), dim=1)
    after_first = selected[rows, target_base_symbols[:, 0]]
    second = torch.softmax(torch.logsumexp(after_first, dim=2), dim=1)
    after_second = after_first[rows, target_base_symbols[:, 1]]
    third = torch.softmax(after_second, dim=1)
    return [first, second, third]


def _regular_log_probs_to_base_steps(
    logits: torch.Tensor,
    target_base_symbols: torch.Tensor,
    *,
    token_merge_size: int,
    model_token_alphabet: str,
    output_alphabet: str,
    model_uses_ascii_tokens: bool,
    force_generic: bool = False,
) -> list[torch.Tensor]:
    if logits.dim() != 2:
        raise ValueError("logits must have shape [windows, vocab]")
    if int(token_merge_size) == 1 and model_uses_ascii_tokens:
        indices = torch.tensor([ord("A"), ord("C"), ord("G"), ord("T")], dtype=torch.long, device=logits.device)
        probs = torch.softmax(logits.float().index_select(1, indices), dim=1)
        return [probs]

    model_token_alphabet = normalize_alphabet(model_token_alphabet)
    output_alphabet = normalize_alphabet(output_alphabet)
    regular_vocab_size = len(model_token_alphabet) ** int(token_merge_size)
    if logits.shape[1] < regular_vocab_size:
        raise ValueError(
            f"LM vocab ({logits.shape[1]}) is smaller than regular ACGT token vocab ({regular_vocab_size})"
        )
    target_base_symbols = target_base_symbols.to(device=logits.device, dtype=torch.long)
    if target_base_symbols.dim() != 2 or target_base_symbols.shape[1] != int(token_merge_size):
        raise ValueError("target_base_symbols must have shape [windows, token_merge_size]")
    if (
        not force_generic
        and int(token_merge_size) == 3
        and model_token_alphabet == "ACGTN"
        and output_alphabet == "ACGT"
    ):
        return _acgtn_merge3_log_probs_to_base_steps(logits, target_base_symbols)
    tables = _acgt_factorization_tables(
        token_merge_size=token_merge_size,
        model_token_alphabet=model_token_alphabet,
        output_alphabet=output_alphabet,
        device=logits.device,
    )
    steps: list[torch.Tensor] = []
    prefix_code = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
    logits_float = logits.float()
    for base_index in range(int(token_merge_size)):
        ids = tables[base_index].index_select(0, prefix_code).reshape(logits.shape[0], len(output_alphabet), -1)
        gathered = torch.gather(logits_float, 1, ids.reshape(logits.shape[0], -1)).reshape_as(ids).float()
        masses = torch.logsumexp(gathered, dim=2)
        steps.append(torch.softmax(masses, dim=1))
        if base_index + 1 < int(token_merge_size):
            prefix_code = prefix_code * len(output_alphabet) + target_base_symbols[:, base_index]
    return steps


def _target_intervals_from_probabilities(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    total: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    probs = np.where(np.isfinite(probs) & (probs > 0.0), probs, 0.0)
    row_sums = probs.sum(axis=1, keepdims=True).clip(min=1e-300)
    probs = probs / row_sums
    freqs = np.floor(probs * float(total)).astype(np.int64)
    freqs = np.maximum(freqs, 1)
    cumulative = np.cumsum(freqs, axis=1, dtype=np.int64)
    rows = np.arange(probs.shape[0], dtype=np.int64)
    highs = cumulative[rows, targets]
    selected = freqs[rows, targets]
    lows = highs - selected
    totals = cumulative[:, -1]
    return lows.astype(np.int32), highs.astype(np.int32), totals.astype(np.int32)


def _tail_side_info_bits(tail_sequence: str) -> int:
    return 2 * len(tail_sequence)


def _compress_fused_matrix_backed_payload(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: int,
    nc_prefix_window_bases: int | None = None,
    nc_prefix_min_windows: int = 8192,
    nc_prefix_hash_bucket_count: int = 0,
    fusion_eta: float = 0.05,
    fusion_initial_lm_weight: float = 0.5,
    arithmetic_frequency_total: int | None = None,
    arithmetic_target_uniform_mass: float = 0.01,
    encode_arithmetic: bool = True,
    collect_diagnostics: bool = True,
    include_codec_baselines: bool = True,
) -> dict[str, Any]:
    if not (0.0 <= float(fusion_initial_lm_weight) <= 1.0):
        raise ValueError("fusion_initial_lm_weight must be in [0, 1]")
    if not (0.0 <= float(fusion_eta) < 1.0):
        raise ValueError("fusion_eta must be in [0, 1)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    alphabet = normalize_alphabet("ACGT")
    token_merge_size = int(config.data.token_merge_size)
    lm_seq_length = int(getattr(model.config, "T_MAX", config.model.seq_length))
    window_bases = int(nc_prefix_window_bases or (lm_seq_length * token_merge_size))
    if window_bases // max(token_merge_size, 1) != lm_seq_length:
        raise ValueError(
            "first fused implementation requires nc_prefix_window_bases to match the LM decode window: "
            f"window_bases/token_merge_size={window_bases // max(token_merge_size, 1)}, "
            f"lm_seq_length={lm_seq_length}"
        )

    process_started = perf_counter()
    windows = build_fused_token_windows(
        payload,
        model=model,
        config=config,
        window_bases=window_bases,
        alphabet=alphabet,
    )
    if len(windows.core_sequence) < int(nc_prefix_min_windows) * int(window_bases):
        raise ValueError(
            "fused compression requires enough bases for nc_prefix statistics: "
            f"core_bases={len(windows.core_sequence)}, window_bases={window_bases}, "
            f"min_windows={nc_prefix_min_windows}"
        )

    nc_started = perf_counter()
    nc_state = MatrixBackedNcPrefixStreamingState(
        windows.core_sequence,
        NoncontiguousPrefixConfig(
            window_bases=window_bases,
            alphabet=alphabet,
            backend="auto",
            min_windows=nc_prefix_min_windows,
            hash_bucket_count=nc_prefix_hash_bucket_count,
        ),
    )
    nc_prepare_seconds = perf_counter() - nc_started

    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=4,
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
    )
    frequency_total = int(arithmetic_metadata["arithmetic_frequency_total"])

    tokens_cpu = windows.tokens
    base_symbols_cpu = windows.token_base_symbols
    valid_lengths_cpu = windows.valid_token_lengths
    window_count = int(tokens_cpu.shape[0])
    tokens_per_window = int(tokens_cpu.shape[1])
    encoded_streams: list[bytes] = []
    model_seconds = 0.0
    lm_factorize_seconds = 0.0
    lm_transfer_seconds = 0.0
    nc_predict_seconds = 0.0
    fusion_seconds = 0.0
    interval_quantize_seconds = 0.0
    arithmetic_range_seconds = 0.0
    emitted = 0
    fused_bits = 0.0
    lm_bits = 0.0
    nc_bits = 0.0
    final_lm_weight_sum = 0.0
    final_weight_count = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    _sync(device)
    encode_started = perf_counter()
    for chunk_start in range(0, window_count, int(batch_size)):
        chunk_end = min(window_count, chunk_start + int(batch_size))
        chunk_tokens_cpu = tokens_cpu[chunk_start:chunk_end].contiguous()
        chunk_base_symbols = base_symbols_cpu[chunk_start:chunk_end].contiguous()
        chunk_lengths_cpu = valid_lengths_cpu[chunk_start:chunk_end].contiguous()
        chunk_size = int(chunk_tokens_cpu.shape[0])
        chunk_tokens = chunk_tokens_cpu.to(device, non_blocking=True)
        stepper = MegabyteBatchedDecodeStepper(model, batch_size=chunk_size, device=device, dtype_name=dtype_name)
        encoder = BatchedStreamingArithmeticEncoder(chunk_size) if encode_arithmetic else None
        lm_weights = np.full((chunk_size,), float(fusion_initial_lm_weight), dtype=np.float64)
        nc_weights = 1.0 - lm_weights

        for token_step in range(tokens_per_window):
            model_started = perf_counter()
            logits = stepper.next_logits()
            _sync(device)
            model_seconds += perf_counter() - model_started

            factor_started = perf_counter()
            active_token_mask = chunk_lengths_cpu > token_step
            safe_target_base_symbols = torch.where(
                active_token_mask[:, None],
                chunk_base_symbols[:, token_step, :],
                torch.zeros_like(chunk_base_symbols[:, token_step, :]),
            ).to(device=device)
            base_probability_steps = _regular_log_probs_to_base_steps(
                logits,
                safe_target_base_symbols,
                token_merge_size=token_merge_size,
                model_token_alphabet=windows.model_token_alphabet,
                output_alphabet=alphabet,
                model_uses_ascii_tokens=windows.model_uses_ascii_tokens,
            )
            _sync(device)
            lm_factorize_seconds += perf_counter() - factor_started

            for base_offset, lm_probs_gpu in enumerate(base_probability_steps):
                base_depth = token_step * token_merge_size + base_offset
                active = (chunk_lengths_cpu.numpy() > token_step)
                if not bool(np.any(active)):
                    continue
                transfer_started = perf_counter()
                lm_probs = lm_probs_gpu.detach().float().cpu().numpy()
                _sync(device)
                lm_transfer_seconds += perf_counter() - transfer_started

                positions = np.arange(chunk_start, chunk_end, dtype=np.int64) * int(window_bases) + int(base_depth)
                nc_started_step = perf_counter()
                active_rows = active.astype(bool)
                nc_probs = np.full((chunk_size, 4), 0.25, dtype=np.float64)
                active_indices = np.nonzero(active_rows)[0]
                nc_probs[active_indices] = nc_state.predict_positions(positions[active_indices])
                nc_predict_seconds += perf_counter() - nc_started_step
                targets = chunk_base_symbols[:, token_step, base_offset].numpy().astype(np.int64, copy=False)

                fusion_started = perf_counter()
                fused = lm_weights[:, None] * lm_probs + nc_weights[:, None] * nc_probs
                fused = fused / fused.sum(axis=1, keepdims=True).clip(min=1e-300)
                rows = np.arange(chunk_size, dtype=np.int64)
                lm_target = lm_probs[rows, targets].clip(min=1e-300)
                nc_target = nc_probs[rows, targets].clip(min=1e-300)
                fused_target = fused[rows, targets].clip(min=1e-300)
                fused_bits += float((-np.log2(fused_target[active_rows])).sum())
                lm_bits += float((-np.log2(lm_target[active_rows])).sum())
                nc_bits += float((-np.log2(nc_target[active_rows])).sum())
                if fusion_eta > 0.0:
                    lm_new = np.power(lm_weights, 1.0 - float(fusion_eta)) * lm_target
                    nc_new = np.power(nc_weights, 1.0 - float(fusion_eta)) * nc_target
                else:
                    lm_new = lm_weights * lm_target
                    nc_new = nc_weights * nc_target
                denom = (lm_new + nc_new).clip(min=1e-300)
                lm_weights = np.where(active_rows, lm_new / denom, lm_weights)
                nc_weights = np.where(active_rows, nc_new / denom, nc_weights)
                fusion_seconds += perf_counter() - fusion_started

                if encode_arithmetic:
                    quant_started = perf_counter()
                    lows, highs, totals = _target_intervals_from_probabilities(
                        fused,
                        targets,
                        total=frequency_total,
                    )
                    interval_quantize_seconds += perf_counter() - quant_started
                    assert encoder is not None
                    timings = encoder.encode_interval_step(lows, highs, totals, active_rows)
                    arithmetic_range_seconds += timings.range_seconds
                    emitted += timings.emitted_count
                else:
                    emitted += int(active_rows.sum())
                nc_state.accept_depth(positions[active_rows], targets[active_rows])

            stepper.accept_symbols(chunk_tokens[:, token_step])

        final_lm_weight_sum += float(lm_weights.sum())
        final_weight_count += int(lm_weights.shape[0])
        if encode_arithmetic:
            assert encoder is not None
            encoded_streams.extend(encoder.finish())

    _sync(device)
    encode_wall_seconds = perf_counter() - encode_started
    tail_bits = _tail_side_info_bits(windows.tail_sequence)
    encoded_bytes = sum(len(stream) for stream in encoded_streams)
    arithmetic_bytes = encoded_bytes + ((tail_bits + 7) // 8 if encode_arithmetic else 0)
    core_bases = len(windows.core_sequence)
    sample_bases = len(windows.sequence)
    elapsed = perf_counter() - process_started
    metrics: dict[str, Any] = {
        "codec": "fused_lm_nc_prefix",
        "decodable_design": "planned_depth_major_window_streams",
        "encode_arithmetic": bool(encode_arithmetic),
        "alphabet": alphabet,
        "sample_bases": int(sample_bases),
        "core_base_count": int(core_bases),
        "tail_base_count": int(len(windows.tail_sequence)),
        "tail_side_info_bits": int(tail_bits),
        "filtered_out_bases": int(windows.filtered_out_bases),
        "window_count": int(window_count),
        "token_merge_size": int(token_merge_size),
        "tokens_per_window": int(tokens_per_window),
        "window_bases": int(window_bases),
        "lm_seq_length": int(lm_seq_length),
        "model_uses_ascii_tokens": bool(windows.model_uses_ascii_tokens),
        "model_token_alphabet": windows.model_token_alphabet,
        "fusion_policy": "online_hedge_linear",
        "fusion_eta": float(fusion_eta),
        "fusion_initial_lm_weight": float(fusion_initial_lm_weight),
        "fusion_final_mean_lm_weight": final_lm_weight_sum / max(final_weight_count, 1),
        "theoretical_bits": float(fused_bits + tail_bits),
        "core_model_theoretical_bits": float(fused_bits),
        "theoretical_bits_per_base": float(fused_bits + tail_bits) / max(sample_bases, 1),
        "core_theoretical_bits_per_base": float(fused_bits) / max(core_bases, 1),
        "lm_only_theoretical_bits": float(lm_bits),
        "lm_only_theoretical_bits_per_base": float(lm_bits) / max(core_bases, 1),
        "nc_prefix_only_theoretical_bits": float(nc_bits),
        "nc_prefix_only_theoretical_bits_per_base": float(nc_bits) / max(core_bases, 1),
        "arithmetic_coded_bytes": int(arithmetic_bytes) if encode_arithmetic else None,
        "arithmetic_bits_per_base": (float(arithmetic_bytes) * 8.0 / max(sample_bases, 1)) if encode_arithmetic else None,
        "arithmetic_stream_count": int(len(encoded_streams)),
        "emitted_arithmetic_symbol_count": int(emitted),
        "compression_process_seconds": float(elapsed),
        "compression_core_seconds": float(encode_wall_seconds),
        "compression_bases_per_second": float(sample_bases) / max(elapsed, 1e-12),
        "model_seconds": float(model_seconds),
        "lm_factorize_seconds": float(lm_factorize_seconds),
        "lm_probability_transfer_seconds": float(lm_transfer_seconds),
        "nc_prefix_prepare_seconds": float(nc_prepare_seconds),
        "nc_prefix_predict_seconds": float(nc_predict_seconds),
        "fusion_seconds": float(fusion_seconds),
        "arithmetic_quantize_seconds": float(interval_quantize_seconds),
        "arithmetic_range_seconds": float(arithmetic_range_seconds),
        "nc_prefix_backend": "matrix_backed_current_cpp",
        "nc_prefix_metadata": nc_state.metadata,
        **arithmetic_metadata,
        **memory_stats(device, prefix="compression_"),
        **baseline_sizes(payload),
    }
    return metrics


def _compress_fused_streaming_token_payload(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: int | str,
    nc_prefix_window_bases: int | None = None,
    nc_prefix_min_windows: int = 8192,
    nc_prefix_hash_bucket_count: int = 0,
    fusion_eta: float = 0.05,
    fusion_initial_lm_weight: float = 0.5,
    arithmetic_frequency_total: int | None = None,
    arithmetic_target_uniform_mass: float = 0.01,
    encode_arithmetic: bool = True,
    collect_diagnostics: bool = True,
    include_codec_baselines: bool = True,
    pipeline_mode: str = "streaming_token_encode_overlap",
) -> dict[str, Any]:
    if not (0.0 <= float(fusion_initial_lm_weight) <= 1.0):
        raise ValueError("fusion_initial_lm_weight must be in [0, 1]")
    if not (0.0 <= float(fusion_eta) < 1.0):
        raise ValueError("fusion_eta must be in [0, 1)")
    if pipeline_mode not in {"streaming_token_encode_overlap", "streaming_token_strict"}:
        raise ValueError("pipeline_mode must be 'streaming_token_encode_overlap' or 'streaming_token_strict'")
    alphabet = normalize_alphabet("ACGT")
    token_merge_size = int(config.data.token_merge_size)
    lm_seq_length = int(getattr(model.config, "T_MAX", config.model.seq_length))
    window_bases = int(nc_prefix_window_bases or (lm_seq_length * token_merge_size))
    if window_bases // max(token_merge_size, 1) != lm_seq_length:
        raise ValueError(
            f"{pipeline_mode} requires nc_prefix_window_bases to match the LM decode window: "
            f"window_bases/token_merge_size={window_bases // max(token_merge_size, 1)}, "
            f"lm_seq_length={lm_seq_length}"
        )

    process_started = perf_counter()
    windows = build_fused_token_windows(
        payload,
        model=model,
        config=config,
        window_bases=window_bases,
        alphabet=alphabet,
    )
    if len(windows.core_sequence) < int(nc_prefix_min_windows) * int(window_bases):
        raise ValueError(
            "fused compression requires enough bases for nc_prefix statistics: "
            f"core_bases={len(windows.core_sequence)}, window_bases={window_bases}, "
            f"min_windows={nc_prefix_min_windows}"
        )

    tokens_cpu = windows.tokens
    base_symbols_cpu = windows.token_base_symbols
    valid_lengths_cpu = windows.valid_token_lengths
    window_count = int(tokens_cpu.shape[0])
    tokens_per_window = int(tokens_cpu.shape[1])
    if batch_size == "auto" or int(batch_size) <= 0:
        resolved_batch_size = window_count
    else:
        resolved_batch_size = int(batch_size)
    if resolved_batch_size != window_count:
        raise ValueError(
            f"{pipeline_mode} requires batch_size == window_count. "
            f"Got batch_size={resolved_batch_size}, window_count={window_count}. "
            "Use batch_size=auto or an explicit value equal to window_count."
        )

    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=4,
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
    )
    frequency_total = int(arithmetic_metadata["arithmetic_frequency_total"])
    native_encoder = FusedNcPrefixStreamingEncoder(
        window_count=window_count,
        window_bases=window_bases,
        hash_bucket_count=int(nc_prefix_hash_bucket_count),
        arithmetic_frequency_total=frequency_total,
        fusion_eta=float(fusion_eta),
        initial_lm_weight=float(fusion_initial_lm_weight),
        encode_arithmetic=bool(encode_arithmetic),
        collect_diagnostics=bool(collect_diagnostics),
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    _sync(device)
    encode_started = perf_counter()
    model_seconds = 0.0
    lm_factorize_seconds = 0.0
    lm_transfer_seconds = 0.0
    native_encode_seconds_observed = 0.0
    cpu_wait_for_gpu_seconds = 0.0
    gpu_queue_wait_seconds = 0.0
    token_jobs = 0
    chunk_tokens = tokens_cpu.to(device, non_blocking=True)
    stepper = MegabyteBatchedDecodeStepper(model, batch_size=window_count, device=device, dtype_name=dtype_name)
    use_encode_overlap = pipeline_mode == "streaming_token_encode_overlap" and device.type == "cuda"
    worker_error: list[BaseException] = []
    worker_stats = {"native_seconds": 0.0, "wait_seconds": 0.0, "jobs": 0}
    work_queue: Queue[Any] | None = None
    worker: threading.Thread | None = None

    if use_encode_overlap:
        work_queue = Queue(maxsize=2)

        def _token_cpu_worker() -> None:
            assert work_queue is not None
            try:
                while True:
                    item = work_queue.get()
                    try:
                        if item is None:
                            return
                        event, lm_probs_cpu, targets = item
                        wait_started = perf_counter()
                        event.synchronize()
                        worker_stats["wait_seconds"] += perf_counter() - wait_started
                        native_started_inner = perf_counter()
                        native_encoder.encode_token_step(lm_probs_cpu, targets)
                        worker_stats["native_seconds"] += perf_counter() - native_started_inner
                        worker_stats["jobs"] += 1
                    finally:
                        work_queue.task_done()
            except BaseException as error:  # pragma: no cover - exercised by integration failures
                worker_error.append(error)

        worker = threading.Thread(target=_token_cpu_worker, name="fused-nc-prefix-token-overlap-worker", daemon=True)
        worker.start()

    for token_step in range(tokens_per_window):
        active_token_mask = valid_lengths_cpu > token_step
        active_count = int(active_token_mask.sum().item())
        model_started = perf_counter()
        logits = stepper.next_logits()
        _sync(device)
        model_seconds += perf_counter() - model_started

        factor_started = perf_counter()
        safe_target_base_symbols = torch.where(
            active_token_mask[:, None],
            base_symbols_cpu[:, token_step, :],
            torch.zeros_like(base_symbols_cpu[:, token_step, :]),
        ).to(device=device)
        base_probability_steps = _regular_log_probs_to_base_steps(
            logits,
            safe_target_base_symbols,
            token_merge_size=token_merge_size,
            model_token_alphabet=windows.model_token_alphabet,
            output_alphabet=alphabet,
            model_uses_ascii_tokens=windows.model_uses_ascii_tokens,
        )
        _sync(device)
        lm_factorize_seconds += perf_counter() - factor_started

        if active_count <= 0:
            stepper.accept_symbols(chunk_tokens[:, token_step])
            continue

        transfer_started = perf_counter()
        lm_probs_gpu = torch.stack(base_probability_steps, dim=1)[:active_count].detach().float()
        if device.type == "cuda":
            lm_probs_cpu = torch.empty(
                (active_count, token_merge_size, 4),
                dtype=torch.float32,
                pin_memory=True,
            )
            lm_probs_cpu.copy_(lm_probs_gpu, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device))
            if not use_encode_overlap:
                event.synchronize()
        else:
            lm_probs_cpu = lm_probs_gpu.cpu().contiguous()
            _sync(device)
            event = None
        lm_transfer_seconds += perf_counter() - transfer_started

        targets = base_symbols_cpu[:active_count, token_step, :].to(torch.int16).contiguous()
        if use_encode_overlap:
            assert work_queue is not None and event is not None
            queue_started = perf_counter()
            work_queue.put((event, lm_probs_cpu, targets))
            gpu_queue_wait_seconds += perf_counter() - queue_started
        else:
            native_started = perf_counter()
            native_encoder.encode_token_step(lm_probs_cpu, targets)
            native_encode_seconds_observed += perf_counter() - native_started
            token_jobs += 1

        stepper.accept_symbols(chunk_tokens[:, token_step])

    if use_encode_overlap:
        assert work_queue is not None
        work_queue.join()
        work_queue.put(None)
        work_queue.join()
        assert worker is not None
        worker.join()
        if worker_error:
            raise RuntimeError("fused token overlap worker failed") from worker_error[0]
        native_encode_seconds_observed = float(worker_stats["native_seconds"])
        cpu_wait_for_gpu_seconds = float(worker_stats["wait_seconds"])
        token_jobs = int(worker_stats["jobs"])

    _sync(device)
    encode_wall_seconds = perf_counter() - encode_started
    native_result = native_encoder.finish()
    encoded_streams = list(native_result.get("streams", []))
    tail_bits = _tail_side_info_bits(windows.tail_sequence)
    encoded_bytes = sum(len(stream) for stream in encoded_streams)
    arithmetic_bytes = encoded_bytes + ((tail_bits + 7) // 8 if encode_arithmetic else 0)
    core_bases = len(windows.core_sequence)
    sample_bases = len(windows.sequence)
    elapsed = perf_counter() - process_started
    diagnostics_collected = bool(native_result.get("diagnostics_collected", True))
    fused_bits = float(native_result["fused_theoretical_bits"]) if diagnostics_collected else None
    metrics: dict[str, Any] = {
        "codec": "fused_lm_nc_prefix",
        "pipeline_mode": pipeline_mode,
        "decodable_design": "encoder_only_lm_token_ahead_overlap_native_ordered_commit"
        if use_encode_overlap
        else "decoder_realistic_token_synchronous_native_ordered_commit",
        "decoder_realistic": not bool(use_encode_overlap),
        "encoder_overlap_enabled": bool(use_encode_overlap),
        "encode_arithmetic": bool(encode_arithmetic),
        "alphabet": alphabet,
        "sample_bases": int(sample_bases),
        "core_base_count": int(core_bases),
        "tail_base_count": int(len(windows.tail_sequence)),
        "tail_side_info_bits": int(tail_bits),
        "filtered_out_bases": int(windows.filtered_out_bases),
        "window_count": int(window_count),
        "token_merge_size": int(token_merge_size),
        "tokens_per_window": int(tokens_per_window),
        "window_bases": int(window_bases),
        "lm_seq_length": int(lm_seq_length),
        "model_uses_ascii_tokens": bool(windows.model_uses_ascii_tokens),
        "model_token_alphabet": windows.model_token_alphabet,
        "batch_size": int(resolved_batch_size),
        "fusion_policy": "online_hedge_linear_native",
        "fusion_eta": float(fusion_eta),
        "fusion_initial_lm_weight": float(fusion_initial_lm_weight),
        "fusion_final_mean_lm_weight": float(native_result["fusion_final_mean_lm_weight"]),
        "diagnostics_collected": bool(diagnostics_collected),
        "theoretical_bits": float(fused_bits + tail_bits) if diagnostics_collected else None,
        "core_model_theoretical_bits": float(fused_bits) if diagnostics_collected else None,
        "theoretical_bits_per_base": (float(fused_bits + tail_bits) / max(sample_bases, 1))
        if diagnostics_collected
        else None,
        "core_theoretical_bits_per_base": (float(fused_bits) / max(core_bases, 1))
        if diagnostics_collected
        else None,
        "lm_only_theoretical_bits": native_result["lm_only_theoretical_bits"] if diagnostics_collected else None,
        "lm_only_theoretical_bits_per_base": native_result["lm_only_theoretical_bits_per_base"]
        if diagnostics_collected
        else None,
        "nc_prefix_only_theoretical_bits": native_result["nc_prefix_only_theoretical_bits"] if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits_per_base": native_result["nc_prefix_only_theoretical_bits_per_base"]
        if diagnostics_collected
        else None,
        "arithmetic_coded_bytes": int(arithmetic_bytes) if encode_arithmetic else None,
        "arithmetic_bits_per_base": (float(arithmetic_bytes) * 8.0 / max(sample_bases, 1)) if encode_arithmetic else None,
        "arithmetic_stream_count": int(len(encoded_streams)),
        "emitted_arithmetic_symbol_count": int(native_result["emitted_arithmetic_symbol_count"]),
        "compression_process_seconds": float(elapsed),
        "compression_core_seconds": float(encode_wall_seconds),
        "compression_bases_per_second": float(sample_bases) / max(elapsed, 1e-12),
        "model_seconds": float(model_seconds),
        "lm_factorize_seconds": float(lm_factorize_seconds),
        "lm_probability_transfer_seconds": float(lm_transfer_seconds),
        "native_fused_encode_seconds_observed": float(native_encode_seconds_observed),
        "native_fused_encode_seconds": float(native_result["encode_seconds"]),
        "native_finish_seconds": float(native_result["finish_seconds"]),
        "streaming_async_enabled": bool(use_encode_overlap),
        "streaming_async_jobs": int(token_jobs),
        "streaming_cpu_wait_for_gpu_seconds": float(cpu_wait_for_gpu_seconds),
        "streaming_gpu_queue_wait_seconds": float(gpu_queue_wait_seconds),
        "streaming_ring_buffer_depth_tokens": 2 if use_encode_overlap else 0,
        "nc_predict_wait_seconds": None,
        "lm_wait_seconds": None,
        "fusion_update_seconds": None,
        "nc_predict_seconds": None,
        "pipeline_depth_lag_max": 0,
        "nc_prefix_prepare_seconds": 0.0,
        "nc_prefix_predict_seconds": None,
        "fusion_seconds": None,
        "arithmetic_quantize_seconds": None,
        "arithmetic_range_seconds": None,
        "nc_prefix_backend": "streaming_token_native",
        "nc_prefix_metadata": native_result["model_metadata"],
        **arithmetic_metadata,
        **memory_stats(device, prefix="compression_"),
        **baseline_sizes(payload, include_codec_baselines=include_codec_baselines),
    }
    return metrics


def compress_fused_lm_nc_prefix_payload(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: int | str,
    nc_prefix_window_bases: int | None = None,
    nc_prefix_min_windows: int = 8192,
    nc_prefix_hash_bucket_count: int = 0,
    fusion_eta: float = 0.05,
    fusion_initial_lm_weight: float = 0.5,
    arithmetic_frequency_total: int | None = None,
    arithmetic_target_uniform_mass: float = 0.01,
    encode_arithmetic: bool = True,
    pipeline_mode: str = "streaming_token_encode_overlap",
    collect_diagnostics: bool = True,
    include_codec_baselines: bool = True,
) -> dict[str, Any]:
    if pipeline_mode in {"streaming_token_encode_overlap", "streaming_token_strict"}:
        return _compress_fused_streaming_token_payload(
            model=model,
            config=config,
            payload=payload,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            nc_prefix_window_bases=nc_prefix_window_bases,
            nc_prefix_min_windows=nc_prefix_min_windows,
            nc_prefix_hash_bucket_count=nc_prefix_hash_bucket_count,
            fusion_eta=fusion_eta,
            fusion_initial_lm_weight=fusion_initial_lm_weight,
            arithmetic_frequency_total=arithmetic_frequency_total,
            arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
            encode_arithmetic=encode_arithmetic,
            collect_diagnostics=collect_diagnostics,
            include_codec_baselines=include_codec_baselines,
            pipeline_mode=pipeline_mode,
        )
    raise ValueError(
        "pipeline_mode must be 'streaming_token_encode_overlap' or 'streaming_token_strict'; "
        "streaming_v2, streaming_v3, and matrix_debug are retired"
    )


def load_megabyte_model_for_fusion(
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[ExperimentConfig, torch.nn.Module, dict[str, Any]]:
    from .megabyte_loader import build_model, load_megabyte_checkpoint
    from .config import load_experiment_config
    from .tokenization import apply_token_merge_to_model_config

    run_dir = Path(run_dir)
    config = load_experiment_config(run_dir / "resolved_config.json")
    apply_token_merge_to_model_config(config.model, config.data)
    model = build_model(config.model).to(device)
    state, metadata, _ = load_megabyte_checkpoint(Path(checkpoint_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return config, model, metadata
