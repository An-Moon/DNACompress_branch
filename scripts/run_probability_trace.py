#!/usr/bin/env python3
from __future__ import annotations

"""Write target-probability traces for later offline fusion.

Examples:

    python scripts/run_probability_trace.py \
      --model megabyte \
      --run-dir outputs/dna_megabyte_large_opengenome2_11 \
      --checkpoint-tag best \
      --source-file outputs/nc_prefix_vs_geco2_true_speed_orsa_25m_seed12345/OrSa_5675277_25165824.seq \
      --source-format raw \
      --output-trace outputs/traces/orsa/megabyte

    python scripts/run_probability_trace.py \
      --model nc_prefix \
      --source-file outputs/nc_prefix_vs_geco2_true_speed_orsa_25m_seed12345/OrSa_5675277_25165824.seq \
      --source-format raw \
      --nc-prefix-window-bases 3072 \
      --token-merge-size 3 \
      --output-trace outputs/traces/orsa/nc_prefix

    scripts/run_evo2_1b_env_python.sh scripts/run_probability_trace.py \
      --model evo2 \
      --local-path third_party/evo2_7b_base/evo2_7b_base.pt \
      --model-name evo2_7b_base \
      --evo2-probability-mode full_forward \
      --source-file datasets/DNACorpus/BuEb \
      --source-format raw \
      --nc-prefix-window-bases 8192 \
      --output-trace outputs/traces/BuEb/evo2
"""

import argparse
import codecs
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.experiment import resolve_device  # noqa: E402
from dna_compress.fused_lm_nc_prefix_codec import (  # noqa: E402
    MegabyteStreamingAdapter,
    _carbon_conditional_log_probs_to_base_steps,
    _encode_carbon_tokens_and_base_symbols,
    _torch_dtype_from_name,
    load_carbon_adapter_for_fusion,
    load_megabyte_model_for_fusion,
)
from dna_compress.fast_nc_prefix import FusedNcPrefixStreamingEncoder  # noqa: E402
from dna_compress.megabyte_window_codec import sync_if_cuda  # noqa: E402
from dna_compress.noncontiguous_prefix_codec import DEFAULT_NC_PREFIX_MIN_WINDOWS  # noqa: E402
from dna_compress.probability_trace import (  # noqa: E402
    fused_depth_major_emit_positions,
    target_symbols_for_positions,
    write_target_probability_trace,
)
from dna_compress.tokenization import normalize_alphabet  # noqa: E402


DEFAULT_MEGABYTE_RUN_DIR = REPO_ROOT / "outputs" / "dna_megabyte_large_opengenome2_11"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write target probability traces for offline fusion.")
    parser.add_argument("--model", choices=("megabyte", "nc_prefix", "carbon", "evo2"), required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--source-format", choices=("auto", "fasta", "raw"), default="auto")
    parser.add_argument("--max-bases", type=int)
    parser.add_argument("--output-trace", required=True)
    parser.add_argument("--trace-dtype", default="float32", choices=("float16", "float32", "float64"))
    parser.add_argument("--shard-rows", type=int, default=1_000_000)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing trace manifest and shards.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", help="Megabyte inference dtype. Defaults to config.train.dtype.")
    parser.add_argument("--batch-size", default="auto", help="Megabyte window batch size.")
    parser.add_argument("--run-dir", default=str(DEFAULT_MEGABYTE_RUN_DIR))
    parser.add_argument("--checkpoint", help="Megabyte checkpoint path. Defaults to <run-dir>/<checkpoint-tag>.pt.")
    parser.add_argument("--checkpoint-tag", default="best")
    parser.add_argument(
        "--megabyte-model-window-tokens",
        type=int,
        help="Override Megabyte streaming reset length in model tokens. Used to probe context extrapolation.",
    )
    parser.add_argument(
        "--megabyte-model-window-bases",
        type=int,
        help="Override Megabyte streaming reset length in bases; converted to ceil(bases / model token_merge_size).",
    )
    parser.add_argument(
        "--megabyte-probability-mode",
        choices=("auto", "streaming_cache", "full_forward"),
        default="auto",
        help=(
            "Megabyte probability extraction mode. streaming_cache follows compression-style cache inference; "
            "full_forward uses faster teacher-forcing probes."
        ),
    )
    parser.add_argument("--local-path", help="Local model path for Carbon or Evo2.")
    parser.add_argument("--model-name", help="Model name for Carbon or Evo2.")
    parser.add_argument("--revision", default="fns", help="Carbon revision.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--carbon-probability-mode",
        choices=("streaming_cache", "full_forward"),
        default="streaming_cache",
        help=(
            "Carbon probability extraction mode. streaming_cache uses stepwise use_cache inference; "
            "full_forward teacher-forces <dna_begin> plus each 6-mer window in one forward pass."
        ),
    )
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=False, help="Evo2 use_kernels flag.")
    parser.add_argument(
        "--evo2-probability-mode",
        choices=("streaming_cache", "full_forward"),
        default="streaming_cache",
        help=(
            "Evo2 probability extraction mode. streaming_cache uses stepwise cache inference; "
            "full_forward teacher-forces each window and gathers next-base target probabilities."
        ),
    )
    parser.add_argument("--nc-prefix-window-bases", type=int, default=3072)
    parser.add_argument(
        "--token-merge-size",
        type=int,
        default=3,
        help="Emit-order token merge size for nc_prefix traces. Use the paired LM token merge size.",
    )
    parser.add_argument("--nc-prefix-backend", choices=("auto", "fast_cpp"), default="auto")
    parser.add_argument("--nc-prefix-min-windows", type=int, default=DEFAULT_NC_PREFIX_MIN_WINDOWS)
    parser.add_argument("--nc-prefix-hash-bucket-count", type=int, default=0)
    parser.add_argument("--nc-prefix-geco2-level", type=int, default=10)
    return parser


def _is_probably_fasta(data: bytes) -> bool:
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if line:
            return line.startswith(b">")
    return False


def _read_payload(path: Path, source_format: str, max_bases: int | None) -> bytes:
    data = path.read_bytes()
    use_fasta = source_format == "fasta" or (source_format == "auto" and _is_probably_fasta(data))
    if use_fasta:
        lines = []
        for raw_line in data.splitlines():
            line = raw_line.strip()
            if line and not line.startswith(b">"):
                lines.append(line)
        data = b"".join(lines)
    if max_bases is not None and int(max_bases) > 0:
        kept = []
        count = 0
        for byte_value in data.upper():
            if byte_value in b"ACGT":
                kept.append(byte_value)
                count += 1
                if count >= int(max_bases):
                    break
        data = bytes(kept)
    return data


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


def _resolve_batch_size(batch_size: str, window_count: int, fallback: int) -> int:
    batch_size_name = str(batch_size).lower()
    if batch_size_name in {"all", "full", "sequence"}:
        return int(window_count)
    if batch_size_name == "auto":
        return min(int(window_count), max(1, int(fallback)))
    resolved = int(batch_size)
    if resolved <= 0:
        raise ValueError("batch-size must be positive")
    return resolved


def _resolve_megabyte_model_window_tokens(args: argparse.Namespace, token_merge_size: int) -> int | None:
    tokens = args.megabyte_model_window_tokens
    bases = args.megabyte_model_window_bases
    if tokens is not None and bases is not None:
        from_bases = int(math.ceil(int(bases) / max(int(token_merge_size), 1)))
        if int(tokens) != from_bases:
            raise ValueError(
                "--megabyte-model-window-tokens and --megabyte-model-window-bases disagree: "
                f"{tokens} != ceil({bases}/{token_merge_size})={from_bases}"
            )
    if tokens is not None:
        if int(tokens) <= 0:
            raise ValueError("--megabyte-model-window-tokens must be positive")
        return int(tokens)
    if bases is not None:
        if int(bases) <= 0:
            raise ValueError("--megabyte-model-window-bases must be positive")
        return int(math.ceil(int(bases) / max(int(token_merge_size), 1)))
    return None


def _set_megabyte_model_window_tokens(adapter: MegabyteStreamingAdapter, tokens: int) -> None:
    adapter.seq_length = int(tokens)
    if hasattr(adapter.model.config, "_replace"):
        adapter.model.config = adapter.model.config._replace(T_MAX=int(tokens))
    elif hasattr(adapter.model.config, "T_MAX"):
        adapter.model.config.T_MAX = int(tokens)
    else:
        raise AttributeError("Megabyte model config does not expose T_MAX")


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if int(multiple) <= 0:
        return int(value)
    return int(math.ceil(int(value) / int(multiple)) * int(multiple))


def _filtered_acgt_sequence(payload: bytes) -> str:
    return "".join(chr(value).upper() for value in payload if chr(value).upper() in {"A", "C", "G", "T"})


def _base_symbols(sequence: str) -> np.ndarray:
    lookup = {"A": 0, "C": 1, "G": 2, "T": 3}
    return np.asarray([lookup[base] for base in sequence], dtype=np.int16)


def _megabyte_tokens_with_partial_tail(
    *,
    sequence: str,
    token_merge_size: int,
    model_token_alphabet: str,
    output_alphabet: str = "ACGT",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode sequence to model tokens while keeping the final partial token analyzable."""

    if int(token_merge_size) <= 0:
        raise ValueError("token_merge_size must be positive")
    if not sequence:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, int(token_merge_size)), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    output_alphabet = normalize_alphabet(output_alphabet)
    model_token_alphabet = normalize_alphabet(model_token_alphabet)
    base_to_symbol = {base: index for index, base in enumerate(output_alphabet)}
    model_base_to_symbol = {base: index for index, base in enumerate(model_token_alphabet)}
    missing = sorted(set(sequence) - set(model_base_to_symbol))
    if missing:
        raise ValueError(
            "filtered ACGT sequence contains bases absent from the model token alphabet "
            f"{model_token_alphabet!r}: {''.join(missing)!r}"
        )

    token_count = int(math.ceil(len(sequence) / int(token_merge_size)))
    padded_base_symbols = np.zeros((token_count, int(token_merge_size)), dtype=np.int64)
    padded_model_digits = np.zeros((token_count, int(token_merge_size)), dtype=np.int64)
    valid_base_lengths = np.zeros((token_count,), dtype=np.int64)
    for token_index in range(token_count):
        start = token_index * int(token_merge_size)
        token = sequence[start : start + int(token_merge_size)]
        valid_base_lengths[token_index] = len(token)
        for offset, base in enumerate(token):
            padded_base_symbols[token_index, offset] = base_to_symbol[base]
            padded_model_digits[token_index, offset] = model_base_to_symbol[base]
    base = len(model_token_alphabet)
    weights = np.asarray([base ** power for power in range(int(token_merge_size) - 1, -1, -1)], dtype=np.int64)
    tokens = (padded_model_digits * weights).sum(axis=1, dtype=np.int64)
    return tokens.astype(np.int64, copy=False), padded_base_symbols, valid_base_lengths


def _window_lengths(total_bases: int, window_bases: int) -> np.ndarray:
    if int(window_bases) <= 0:
        raise ValueError("window_bases must be positive")
    window_count = int(math.ceil(max(int(total_bases), 1) / int(window_bases))) if total_bases else 0
    lengths = np.zeros((window_count,), dtype=np.int64)
    for window_id in range(window_count):
        start = window_id * int(window_bases)
        lengths[window_id] = max(0, min(int(window_bases), int(total_bases) - start))
    return lengths


def _target_trace_arrays_from_position_probs(
    *,
    sequence: str,
    target_prob_by_position: np.ndarray,
    window_bases: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    core_sequence = sequence
    emit_position = fused_depth_major_emit_positions(
        core_base_count=len(core_sequence),
        window_bases=int(window_bases),
        token_merge_size=1,
    )
    target_symbol_by_position = _base_symbols(core_sequence)
    target_prob_by_position = np.asarray(target_prob_by_position, dtype=np.float64)
    if target_prob_by_position.shape != (len(core_sequence),):
        raise ValueError("target_prob_by_position must have one row per core base")
    if np.any(~np.isfinite(target_prob_by_position)) or np.any(target_prob_by_position < 0.0):
        bad = int(np.count_nonzero(~np.isfinite(target_prob_by_position) | (target_prob_by_position < 0.0)))
        raise ValueError(f"invalid target probabilities by position: {bad}")
    return (
        core_sequence,
        target_prob_by_position[emit_position],
        target_symbol_by_position[emit_position],
        emit_position,
    )


def _megabyte_target_probabilities_base_trace(
    *,
    adapter: MegabyteStreamingAdapter,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: str,
    trace_window_bases: int,
    model_window_tokens: int | None,
    batch_fallback: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    sequence = _filtered_acgt_sequence(payload)
    core_sequence = sequence
    token_merge_size = int(adapter.token_merge_size)
    tokens_np, token_base_symbols_np, valid_base_lengths_np = _megabyte_tokens_with_partial_tail(
        sequence=core_sequence,
        token_merge_size=token_merge_size,
        model_token_alphabet=adapter.token_alphabet,
        output_alphabet="ACGT",
    )
    token_count = int(tokens_np.shape[0])
    if token_count == 0:
        raise ValueError("megabyte trace requires at least one ACGT base")

    model_tokens_per_window = int(model_window_tokens or adapter.seq_length)
    if model_tokens_per_window <= 0:
        raise ValueError("model_window_tokens must be positive")
    model_window_count = int(math.ceil(token_count / max(model_tokens_per_window, 1)))
    padded_token_count = model_window_count * model_tokens_per_window
    padded_tokens = np.full((padded_token_count,), int(adapter.pad_id), dtype=np.int64)
    padded_tokens[:token_count] = tokens_np
    padded_base_symbols = np.zeros((padded_token_count, token_merge_size), dtype=np.int64)
    padded_base_symbols[:token_count, :] = token_base_symbols_np
    padded_valid_base_lengths = np.zeros((padded_token_count,), dtype=np.int64)
    padded_valid_base_lengths[:token_count] = valid_base_lengths_np

    tokens_cpu = torch.from_numpy(padded_tokens.reshape(model_window_count, model_tokens_per_window)).long().contiguous()
    base_symbols_cpu = (
        torch.from_numpy(padded_base_symbols.reshape(model_window_count, model_tokens_per_window, token_merge_size))
        .long()
        .contiguous()
    )
    valid_base_lengths_cpu = (
        torch.from_numpy(padded_valid_base_lengths.reshape(model_window_count, model_tokens_per_window))
        .long()
        .contiguous()
    )
    valid_token_lengths_cpu = torch.full((model_window_count,), model_tokens_per_window, dtype=torch.long)
    tail_tokens = token_count - (model_window_count - 1) * model_tokens_per_window
    valid_token_lengths_cpu[-1] = tail_tokens

    resolved_batch_size = _resolve_batch_size(batch_size, model_window_count, batch_fallback)
    target_prob_by_position = np.full((len(core_sequence),), np.nan, dtype=np.float64)
    target_symbol_by_position = _base_symbols(core_sequence)

    model_seconds = 0.0
    factorize_seconds = 0.0
    transfer_seconds = 0.0
    adapter.eval()
    for chunk_start in range(0, model_window_count, resolved_batch_size):
        chunk_end = min(model_window_count, chunk_start + resolved_batch_size)
        chunk_tokens_cpu = tokens_cpu[chunk_start:chunk_end].contiguous()
        chunk_base_symbols = base_symbols_cpu[chunk_start:chunk_end].contiguous()
        chunk_base_lengths = valid_base_lengths_cpu[chunk_start:chunk_end].contiguous()
        chunk_lengths_cpu = valid_token_lengths_cpu[chunk_start:chunk_end].contiguous()
        chunk_tokens = chunk_tokens_cpu.to(device, non_blocking=True)
        stepper = adapter.start_batch(chunk_tokens_cpu, device=device, dtype_name=dtype_name)
        for token_step in range(model_tokens_per_window):
            active_token_mask = chunk_lengths_cpu > token_step
            if not bool(active_token_mask.any().item()):
                break
            model_started = perf_counter()
            logits = stepper.next_logits()
            sync_if_cuda(device)
            model_seconds += perf_counter() - model_started

            factor_started = perf_counter()
            safe_target_base_symbols = torch.where(
                active_token_mask[:, None],
                chunk_base_symbols[:, token_step, :],
                torch.zeros_like(chunk_base_symbols[:, token_step, :]),
            ).to(device=device)
            base_probability_steps = adapter.logits_to_acgt_base_probs(
                logits,
                safe_target_base_symbols,
                output_alphabet="ACGT",
            )
            sync_if_cuda(device)
            factorize_seconds += perf_counter() - factor_started

            for base_offset, base_probs in enumerate(base_probability_steps):
                valid_base_mask = active_token_mask & (chunk_base_lengths[:, token_step] > int(base_offset))
                if not bool(valid_base_mask.any().item()):
                    continue
                active_rows_cpu = torch.nonzero(valid_base_mask, as_tuple=False).view(-1)
                active_rows = active_rows_cpu.to(device=device)
                targets = chunk_base_symbols[active_rows_cpu, token_step, base_offset].to(device=device)
                transfer_started = perf_counter()
                probs = base_probs[active_rows, targets].detach().float().cpu().numpy()
                sync_if_cuda(device)
                transfer_seconds += perf_counter() - transfer_started
                model_window_ids = chunk_start + active_rows_cpu.numpy().astype(np.int64, copy=False)
                token_indices = model_window_ids * model_tokens_per_window + int(token_step)
                positions = token_indices * token_merge_size + int(base_offset)
                target_prob_by_position[positions] = probs.astype(np.float64, copy=False)

            stepper.accept_symbols(chunk_tokens[:, token_step])

    if np.any(np.isnan(target_prob_by_position)):
        missing = int(np.count_nonzero(np.isnan(target_prob_by_position)))
        raise ValueError(f"megabyte target probability generation missed {missing} bases")

    core_sequence, target_prob, target_symbol, emit_position = _target_trace_arrays_from_position_probs(
        sequence=sequence,
        target_prob_by_position=target_prob_by_position,
        window_bases=int(trace_window_bases),
    )
    metadata = {
        "model_seconds": model_seconds,
        "factorize_seconds": factorize_seconds,
        "probability_transfer_seconds": transfer_seconds,
        "trace_generation_seconds": perf_counter() - started,
        "batch_size": resolved_batch_size,
        "batch_count": int((model_window_count + resolved_batch_size - 1) // resolved_batch_size),
        "window_count": model_window_count,
        "tokens_per_window": model_tokens_per_window,
        "model_window_tokens": model_tokens_per_window,
        "model_window_bases": model_tokens_per_window * token_merge_size,
        "trace_window_bases": int(trace_window_bases),
        "trace_token_merge_size": 1,
        "model_token_merge_size": token_merge_size,
        "includes_partial_final_model_token": bool(len(sequence) % token_merge_size),
        **adapter.metadata(),
    }
    return sequence, core_sequence, target_prob, target_symbol, emit_position, metadata


def _megabyte_target_probabilities_full_forward_base_trace(
    *,
    adapter: MegabyteStreamingAdapter,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: str,
    trace_window_bases: int,
    model_window_tokens: int | None,
    batch_fallback: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    sequence = _filtered_acgt_sequence(payload)
    core_sequence = sequence
    token_merge_size = int(adapter.token_merge_size)
    tokens_np, token_base_symbols_np, valid_base_lengths_np = _megabyte_tokens_with_partial_tail(
        sequence=core_sequence,
        token_merge_size=token_merge_size,
        model_token_alphabet=adapter.token_alphabet,
        output_alphabet="ACGT",
    )
    token_count = int(tokens_np.shape[0])
    if token_count == 0:
        raise ValueError("megabyte trace requires at least one ACGT base")

    patch_size = int(getattr(adapter.model.config, "P", getattr(adapter.config.model, "patch_size", 1)))
    requested_model_window_tokens = int(model_window_tokens or adapter.seq_length)
    model_tokens_per_window = _round_up_to_multiple(requested_model_window_tokens, patch_size)
    _set_megabyte_model_window_tokens(adapter, model_tokens_per_window)

    model_window_count = int(math.ceil(token_count / max(model_tokens_per_window, 1)))
    padded_token_count = model_window_count * model_tokens_per_window
    padded_tokens = np.full((padded_token_count,), int(adapter.pad_id), dtype=np.int64)
    padded_tokens[:token_count] = tokens_np
    padded_base_symbols = np.zeros((padded_token_count, token_merge_size), dtype=np.int64)
    padded_base_symbols[:token_count, :] = token_base_symbols_np
    padded_valid_base_lengths = np.zeros((padded_token_count,), dtype=np.int64)
    padded_valid_base_lengths[:token_count] = valid_base_lengths_np

    tokens_cpu = torch.from_numpy(padded_tokens.reshape(model_window_count, model_tokens_per_window)).long().contiguous()
    base_symbols_cpu = (
        torch.from_numpy(padded_base_symbols.reshape(model_window_count, model_tokens_per_window, token_merge_size))
        .long()
        .contiguous()
    )
    valid_base_lengths_cpu = (
        torch.from_numpy(padded_valid_base_lengths.reshape(model_window_count, model_tokens_per_window))
        .long()
        .contiguous()
    )
    resolved_batch_size = _resolve_batch_size(batch_size, model_window_count, batch_fallback)
    target_prob_by_position = np.full((len(core_sequence),), np.nan, dtype=np.float64)
    target_symbol_by_position = _base_symbols(core_sequence)
    del target_symbol_by_position

    model_seconds = 0.0
    factorize_seconds = 0.0
    transfer_seconds = 0.0
    adapter.eval()
    for chunk_start in range(0, model_window_count, resolved_batch_size):
        chunk_end = min(model_window_count, chunk_start + resolved_batch_size)
        chunk_tokens_cpu = tokens_cpu[chunk_start:chunk_end].contiguous()
        chunk_base_symbols = base_symbols_cpu[chunk_start:chunk_end].contiguous()
        chunk_base_lengths = valid_base_lengths_cpu[chunk_start:chunk_end].contiguous()
        chunk_tokens = chunk_tokens_cpu.to(device, non_blocking=True)

        model_started = perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=_torch_dtype_from_name(dtype_name),
            enabled=device.type == "cuda" and dtype_name not in {"float32", "fp32"},
        ):
            output = adapter.model(chunk_tokens, return_loss=False)
            logits = output.lm_logits
        sync_if_cuda(device)
        model_seconds += perf_counter() - model_started

        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_base_symbols = chunk_base_symbols.reshape(-1, token_merge_size)
        flat_valid_base_lengths = chunk_base_lengths.reshape(-1)
        valid_token_rows = torch.nonzero(flat_valid_base_lengths > 0, as_tuple=False).view(-1)
        if valid_token_rows.numel() == 0:
            continue
        factor_started = perf_counter()
        base_probability_steps = adapter.logits_to_acgt_base_probs(
            flat_logits.index_select(0, valid_token_rows.to(device=device)),
            flat_base_symbols.index_select(0, valid_token_rows).to(device=device),
            output_alphabet="ACGT",
        )
        sync_if_cuda(device)
        factorize_seconds += perf_counter() - factor_started

        valid_rows_np = valid_token_rows.cpu().numpy().astype(np.int64, copy=False)
        model_window_ids = valid_rows_np // model_tokens_per_window
        token_steps = valid_rows_np % model_tokens_per_window
        global_token_indices = (chunk_start + model_window_ids) * model_tokens_per_window + token_steps
        for base_offset, base_probs in enumerate(base_probability_steps):
            valid_base_mask = flat_valid_base_lengths.index_select(0, valid_token_rows) > int(base_offset)
            if not bool(valid_base_mask.any().item()):
                continue
            selected_rows = torch.nonzero(valid_base_mask, as_tuple=False).view(-1)
            local_valid_rows = valid_token_rows.index_select(0, selected_rows)
            targets = flat_base_symbols.index_select(0, local_valid_rows)[:, base_offset].to(device=device)
            transfer_started = perf_counter()
            probs = base_probs.index_select(0, selected_rows.to(device=device))[
                torch.arange(selected_rows.numel(), device=device),
                targets,
            ].detach().float().cpu().numpy()
            sync_if_cuda(device)
            transfer_seconds += perf_counter() - transfer_started
            positions = global_token_indices[selected_rows.cpu().numpy().astype(np.int64, copy=False)] * token_merge_size + int(
                base_offset
            )
            target_prob_by_position[positions] = probs.astype(np.float64, copy=False)

    if np.any(np.isnan(target_prob_by_position)):
        missing = int(np.count_nonzero(np.isnan(target_prob_by_position)))
        raise ValueError(f"megabyte full-forward target probability generation missed {missing} bases")

    core_sequence, target_prob, target_symbol, emit_position = _target_trace_arrays_from_position_probs(
        sequence=sequence,
        target_prob_by_position=target_prob_by_position,
        window_bases=int(trace_window_bases),
    )
    metadata = {
        "probability_generation_mode": "megabyte_full_forward_teacher_forcing",
        "model_seconds": model_seconds,
        "factorize_seconds": factorize_seconds,
        "probability_transfer_seconds": transfer_seconds,
        "trace_generation_seconds": perf_counter() - started,
        "batch_size": resolved_batch_size,
        "batch_count": int((model_window_count + resolved_batch_size - 1) // resolved_batch_size),
        "window_count": model_window_count,
        "tokens_per_window": model_tokens_per_window,
        "requested_model_window_tokens": requested_model_window_tokens,
        "model_window_tokens": model_tokens_per_window,
        "model_window_bases": model_tokens_per_window * token_merge_size,
        "trace_window_bases": int(trace_window_bases),
        "trace_token_merge_size": 1,
        "model_token_merge_size": token_merge_size,
        "patch_size": patch_size,
        "includes_partial_final_model_token": bool(len(sequence) % token_merge_size),
        **adapter.metadata(),
    }
    return sequence, core_sequence, target_prob, target_symbol, emit_position, metadata


def _megabyte_target_probabilities(
    *,
    adapter: MegabyteStreamingAdapter,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: str,
    window_bases: int,
    batch_fallback: int,
    model_window_tokens: int | None = None,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    if model_window_tokens is not None:
        original_seq_length = int(adapter.seq_length)
        try:
            adapter.seq_length = int(model_window_tokens)
            if hasattr(adapter.model.config, "_replace"):
                adapter.model.config = adapter.model.config._replace(T_MAX=int(model_window_tokens))
            elif hasattr(adapter.model.config, "T_MAX"):
                adapter.model.config.T_MAX = int(model_window_tokens)
            windows = adapter.build_windows(payload, window_bases=int(window_bases), alphabet="ACGT")
        finally:
            adapter.seq_length = original_seq_length
    else:
        windows = adapter.build_windows(payload, window_bases=int(window_bases), alphabet="ACGT")
    tokens_cpu = windows.tokens
    base_symbols_cpu = windows.token_base_symbols
    valid_lengths_cpu = windows.valid_token_lengths
    window_count = int(tokens_cpu.shape[0])
    tokens_per_window = int(tokens_cpu.shape[1])
    token_merge_size = int(windows.token_merge_size)
    resolved_batch_size = _resolve_batch_size(batch_size, window_count, batch_fallback)

    target_probs: list[np.ndarray] = []
    target_symbols: list[np.ndarray] = []
    emit_positions: list[np.ndarray] = []
    model_seconds = 0.0
    factorize_seconds = 0.0
    transfer_seconds = 0.0

    adapter.eval()
    for chunk_start in range(0, window_count, resolved_batch_size):
        chunk_end = min(window_count, chunk_start + resolved_batch_size)
        chunk_tokens_cpu = tokens_cpu[chunk_start:chunk_end].contiguous()
        chunk_base_symbols = base_symbols_cpu[chunk_start:chunk_end].contiguous()
        chunk_lengths_cpu = valid_lengths_cpu[chunk_start:chunk_end].contiguous()
        chunk_tokens = chunk_tokens_cpu.to(device, non_blocking=True)
        stepper = adapter.start_batch(chunk_tokens_cpu, device=device, dtype_name=dtype_name)
        for token_step in range(tokens_per_window):
            active_token_mask = chunk_lengths_cpu > token_step
            active_count = int(active_token_mask.sum().item())
            model_started = perf_counter()
            logits = stepper.next_logits()
            sync_if_cuda(device)
            model_seconds += perf_counter() - model_started

            factor_started = perf_counter()
            safe_target_base_symbols = torch.where(
                active_token_mask[:, None],
                chunk_base_symbols[:, token_step, :],
                torch.zeros_like(chunk_base_symbols[:, token_step, :]),
            ).to(device=device)
            base_probability_steps = adapter.logits_to_acgt_base_probs(
                logits,
                safe_target_base_symbols,
                output_alphabet="ACGT",
            )
            sync_if_cuda(device)
            factorize_seconds += perf_counter() - factor_started

            if active_count > 0:
                active_windows = np.arange(chunk_start, chunk_start + active_count, dtype=np.int64)
                for base_offset, base_probs in enumerate(base_probability_steps):
                    targets = chunk_base_symbols[:active_count, token_step, base_offset].to(device=device)
                    rows = torch.arange(active_count, device=device)
                    transfer_started = perf_counter()
                    probs = base_probs[:active_count][rows, targets].detach().float().cpu().numpy()
                    sync_if_cuda(device)
                    transfer_seconds += perf_counter() - transfer_started
                    positions = (
                        active_windows * int(window_bases)
                        + token_step * token_merge_size
                        + int(base_offset)
                    )
                    target_probs.append(probs.astype(np.float64, copy=False))
                    target_symbols.append(targets.detach().cpu().numpy().astype(np.int16, copy=False))
                    emit_positions.append(positions.astype(np.int64, copy=False))

            stepper.accept_symbols(chunk_tokens[:, token_step])

    target_prob = np.concatenate(target_probs) if target_probs else np.zeros((0,), dtype=np.float64)
    target_symbol = np.concatenate(target_symbols) if target_symbols else np.zeros((0,), dtype=np.int16)
    emit_position = np.concatenate(emit_positions) if emit_positions else np.zeros((0,), dtype=np.int64)
    metadata = {
        "model_seconds": model_seconds,
        "factorize_seconds": factorize_seconds,
        "probability_transfer_seconds": transfer_seconds,
        "trace_generation_seconds": perf_counter() - started,
        "batch_size": resolved_batch_size,
        "batch_count": int((window_count + resolved_batch_size - 1) // resolved_batch_size),
        "window_count": window_count,
        "tokens_per_window": tokens_per_window,
        "model_window_tokens": int(model_window_tokens or adapter.seq_length),
        "model_window_bases": int(model_window_tokens or adapter.seq_length) * token_merge_size,
        "trace_window_bases": int(window_bases),
        "trace_token_merge_size": token_merge_size,
        "model_token_merge_size": token_merge_size,
        **adapter.metadata(),
    }
    return windows.sequence, windows.core_sequence, target_prob, target_symbol, emit_position, metadata


def _nc_prefix_target_probabilities(
    *,
    payload: bytes,
    window_bases: int,
    token_merge_size: int,
    backend: str,
    min_windows: int,
    hash_bucket_count: int,
    geco2_level: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    alphabet = normalize_alphabet("ACGT")
    sequence = "".join(chr(value).upper() for value in payload if chr(value).upper() in alphabet)
    core_base_count = (len(sequence) // int(token_merge_size)) * int(token_merge_size)
    core_sequence = sequence[:core_base_count]
    emit_position = fused_depth_major_emit_positions(
        core_base_count=core_base_count,
        window_bases=int(window_bases),
        token_merge_size=int(token_merge_size),
    )
    target_symbol = target_symbols_for_positions(core_sequence, emit_position, alphabet)
    del backend, min_windows
    tokens_per_window = int(window_bases) // int(token_merge_size)
    window_count = int(np.ceil(max(core_base_count, 1) / max(int(window_bases), 1))) if core_base_count else 0
    padded_symbols = np.zeros((window_count * tokens_per_window * int(token_merge_size),), dtype=np.int16)
    padded_symbols[:core_base_count] = target_symbols_for_positions(
        core_sequence,
        np.arange(core_base_count, dtype=np.int64),
        alphabet,
    )
    base_symbols = padded_symbols.reshape(window_count, tokens_per_window, int(token_merge_size))
    valid_token_lengths = np.full((window_count,), tokens_per_window, dtype=np.int64)
    if window_count:
        token_count = core_base_count // int(token_merge_size)
        valid_token_lengths[-1] = token_count - (window_count - 1) * tokens_per_window

    encoder = FusedNcPrefixStreamingEncoder(
        window_count=window_count,
        window_bases=int(window_bases),
        hash_bucket_count=int(hash_bucket_count),
        geco2_level=int(geco2_level),
        arithmetic_frequency_total=65536,
        fusion_eta=0.05,
        initial_lm_weight=0.5,
        encode_arithmetic=False,
        collect_diagnostics=True,
    )
    target_prob_steps: list[np.ndarray] = []
    for token_step in range(tokens_per_window):
        active_count = int(np.count_nonzero(valid_token_lengths > token_step))
        if active_count <= 0:
            continue
        lm_uniform = torch.full((active_count, int(token_merge_size), 4), 0.25, dtype=torch.float32)
        targets = torch.from_numpy(base_symbols[:active_count, token_step, :].astype(np.int16, copy=False))
        step = encoder.encode_token_step_collect_targets(lm_uniform, targets)
        nc_target_probabilities = step["nc_target_probabilities"].cpu().numpy()
        for base_offset in range(int(token_merge_size)):
            target_prob_steps.append(nc_target_probabilities[:active_count, base_offset].astype(np.float64, copy=False))
    native_result = encoder.finish()
    target_prob = np.concatenate(target_prob_steps) if target_prob_steps else np.zeros((0,), dtype=np.float64)
    metadata = {
        "trace_generation_seconds": perf_counter() - started,
        "nc_prefix": native_result.get("model_metadata", {}),
        "nc_prefix_trace_source": "fused_nc_prefix_streaming_encoder_collect_targets",
        "window_count": window_count,
        "tokens_per_window": tokens_per_window,
        "nc_prefix_only_theoretical_bits": native_result.get("nc_prefix_only_theoretical_bits"),
    }
    return sequence, core_sequence, target_prob, target_symbol, emit_position, metadata


def _carbon_target_probabilities(
    *,
    adapter: Any,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: str,
    window_bases: int,
    batch_fallback: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    sequence = _filtered_acgt_sequence(payload)
    if not sequence:
        raise ValueError("Carbon trace requires at least one A/C/G/T base")
    carbon_k = int(adapter.token_merge_size)
    if carbon_k <= 0:
        raise ValueError("Carbon token merge size must be positive")
    window_lengths = _window_lengths(len(sequence), int(window_bases))
    window_count = int(window_lengths.shape[0])
    tokens_per_window = int(math.ceil(int(window_bases) / carbon_k))
    resolved_batch_size = _resolve_batch_size(batch_size, window_count, batch_fallback)

    tokens_np = np.full((window_count, tokens_per_window), int(adapter.pad_id), dtype=np.int64)
    base_symbols_np = np.zeros((window_count, tokens_per_window, carbon_k), dtype=np.int64)
    valid_base_lengths = np.zeros((window_count, tokens_per_window), dtype=np.int64)
    dna_token_to_id = adapter.tokenizer.dna_token_to_id
    for window_id, length in enumerate(window_lengths.tolist()):
        window_start = window_id * int(window_bases)
        for token_step in range(tokens_per_window):
            base_start = window_start + token_step * carbon_k
            base_end = min(window_start + int(length), base_start + carbon_k)
            valid = max(0, base_end - base_start)
            if valid <= 0:
                continue
            token = sequence[base_start:base_end] + ("A" * (carbon_k - valid))
            tokens_np[window_id, token_step] = int(dna_token_to_id[token])
            _, symbols = _encode_carbon_tokens_and_base_symbols(
                token,
                tokenizer=adapter.tokenizer,
                token_merge_size=carbon_k,
                output_alphabet="ACGT",
            )
            base_symbols_np[window_id, token_step, :] = symbols.reshape(carbon_k)
            valid_base_lengths[window_id, token_step] = valid

    target_prob_by_position = np.full((len(sequence),), np.nan, dtype=np.float64)
    model_seconds = 0.0
    factorize_seconds = 0.0
    transfer_seconds = 0.0
    adapter.eval()
    for chunk_start in range(0, window_count, resolved_batch_size):
        chunk_end = min(window_count, chunk_start + resolved_batch_size)
        chunk_tokens_cpu = torch.from_numpy(tokens_np[chunk_start:chunk_end]).long().contiguous()
        chunk_valid = valid_base_lengths[chunk_start:chunk_end]
        chunk_symbols = torch.from_numpy(base_symbols_np[chunk_start:chunk_end]).long().contiguous()
        chunk_tokens = chunk_tokens_cpu.to(device, non_blocking=True)
        stepper = adapter.start_batch(chunk_tokens_cpu, device=device, dtype_name=dtype_name)
        rows_all = np.arange(chunk_end - chunk_start, dtype=np.int64)
        for token_step in range(tokens_per_window):
            valid_lengths = chunk_valid[:, token_step]
            if int(np.count_nonzero(valid_lengths)) <= 0:
                break
            model_started = perf_counter()
            logits = stepper.next_logits()
            sync_if_cuda(device)
            model_seconds += perf_counter() - model_started

            factor_started = perf_counter()
            safe_target_base_symbols = chunk_symbols[:, token_step, :].to(device=device)
            base_probability_steps = adapter.logits_to_acgt_base_probs(
                logits,
                safe_target_base_symbols,
                output_alphabet="ACGT",
            )
            sync_if_cuda(device)
            factorize_seconds += perf_counter() - factor_started

            for base_offset, base_probs in enumerate(base_probability_steps):
                valid_rows = rows_all[valid_lengths > base_offset]
                if valid_rows.size == 0:
                    continue
                targets = chunk_symbols[valid_rows, token_step, base_offset].to(device=device)
                row_tensor = torch.as_tensor(valid_rows, dtype=torch.long, device=device)
                transfer_started = perf_counter()
                probs = base_probs[row_tensor, targets].detach().float().cpu().numpy()
                sync_if_cuda(device)
                transfer_seconds += perf_counter() - transfer_started
                positions = (
                    (chunk_start + valid_rows) * int(window_bases)
                    + token_step * carbon_k
                    + int(base_offset)
                )
                target_prob_by_position[positions] = probs.astype(np.float64, copy=False)

            stepper.accept_symbols(chunk_tokens[:, token_step])

    if np.any(np.isnan(target_prob_by_position)):
        missing = int(np.count_nonzero(np.isnan(target_prob_by_position)))
        raise RuntimeError(f"Carbon trace missed {missing} target base probabilities")
    core_sequence, target_prob, target_symbol, emit_position = _target_trace_arrays_from_position_probs(
        sequence=sequence,
        target_prob_by_position=target_prob_by_position,
        window_bases=int(window_bases),
    )
    metadata = {
        "trace_generation_seconds": perf_counter() - started,
        "model_seconds": model_seconds,
        "factorize_seconds": factorize_seconds,
        "probability_transfer_seconds": transfer_seconds,
        "batch_size": resolved_batch_size,
        "batch_count": int((window_count + resolved_batch_size - 1) // resolved_batch_size),
        "window_count": window_count,
        "window_bases": int(window_bases),
        "internal_token_merge_size": carbon_k,
        "internal_tokens_per_window": tokens_per_window,
        "carbon_trace_source": "streaming_cache_target_prob_factorized_6mer_to_base",
        **adapter.metadata(),
    }
    return sequence, core_sequence, target_prob, target_symbol, emit_position, metadata


def _carbon_target_probabilities_full_forward(
    *,
    adapter: Any,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: str,
    window_bases: int,
    batch_fallback: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    sequence = _filtered_acgt_sequence(payload)
    if not sequence:
        raise ValueError("Carbon trace requires at least one A/C/G/T base")
    carbon_k = int(adapter.token_merge_size)
    if carbon_k <= 0:
        raise ValueError("Carbon token merge size must be positive")
    window_lengths = _window_lengths(len(sequence), int(window_bases))
    window_count = int(window_lengths.shape[0])
    tokens_per_window = int(math.ceil(int(window_bases) / carbon_k))
    resolved_batch_size = _resolve_batch_size(batch_size, window_count, batch_fallback)

    tokens_np = np.full((window_count, tokens_per_window), int(adapter.pad_id), dtype=np.int64)
    base_symbols_np = np.zeros((window_count, tokens_per_window, carbon_k), dtype=np.int64)
    valid_base_lengths = np.zeros((window_count, tokens_per_window), dtype=np.int64)
    dna_token_to_id = adapter.tokenizer.dna_token_to_id
    for window_id, length in enumerate(window_lengths.tolist()):
        window_start = window_id * int(window_bases)
        for token_step in range(tokens_per_window):
            base_start = window_start + token_step * carbon_k
            base_end = min(window_start + int(length), base_start + carbon_k)
            valid = max(0, base_end - base_start)
            if valid <= 0:
                continue
            token = sequence[base_start:base_end] + ("A" * (carbon_k - valid))
            tokens_np[window_id, token_step] = int(dna_token_to_id[token])
            _, symbols = _encode_carbon_tokens_and_base_symbols(
                token,
                tokenizer=adapter.tokenizer,
                token_merge_size=carbon_k,
                output_alphabet="ACGT",
            )
            base_symbols_np[window_id, token_step, :] = symbols.reshape(carbon_k)
            valid_base_lengths[window_id, token_step] = valid

    target_prob_by_position = np.full((len(sequence),), np.nan, dtype=np.float64)
    model_seconds = 0.0
    factorize_seconds = 0.0
    transfer_seconds = 0.0
    autocast_dtype = _torch_dtype_from_name(dtype_name)
    use_autocast = device.type == "cuda" and autocast_dtype in {torch.float16, torch.bfloat16}
    adapter.eval()

    for chunk_start in range(0, window_count, resolved_batch_size):
        chunk_end = min(window_count, chunk_start + resolved_batch_size)
        chunk_size = chunk_end - chunk_start
        chunk_valid = valid_base_lengths[chunk_start:chunk_end]
        chunk_symbols = torch.from_numpy(base_symbols_np[chunk_start:chunk_end]).long().contiguous()
        input_np = np.full((chunk_size, tokens_per_window + 1), int(adapter.pad_id), dtype=np.int64)
        input_np[:, 0] = int(adapter.begin_token_id)
        input_np[:, 1:] = tokens_np[chunk_start:chunk_end]
        transfer_started = perf_counter()
        input_ids = torch.from_numpy(input_np).long().contiguous().to(device, non_blocking=True)
        sync_if_cuda(device)
        transfer_seconds += perf_counter() - transfer_started
        rows_all = np.arange(chunk_size, dtype=np.int64)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                model_started = perf_counter()
                output = adapter.model(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                logits_by_step = output.logits[:, :tokens_per_window, :]
                sync_if_cuda(device)
                model_seconds += perf_counter() - model_started

            for token_step in range(tokens_per_window):
                valid_lengths = chunk_valid[:, token_step]
                if int(np.count_nonzero(valid_lengths)) <= 0:
                    break
                factor_started = perf_counter()
                logits = logits_by_step[:, token_step, :]
                safe_target_base_symbols = chunk_symbols[:, token_step, :].to(device=device)
                base_probability_steps = adapter.logits_to_acgt_base_probs(
                    logits,
                    safe_target_base_symbols,
                    output_alphabet="ACGT",
                )
                sync_if_cuda(device)
                factorize_seconds += perf_counter() - factor_started

                for base_offset, base_probs in enumerate(base_probability_steps):
                    valid_rows = rows_all[valid_lengths > base_offset]
                    if valid_rows.size == 0:
                        continue
                    targets = chunk_symbols[valid_rows, token_step, base_offset].to(device=device)
                    row_tensor = torch.as_tensor(valid_rows, dtype=torch.long, device=device)
                    transfer_started = perf_counter()
                    probs = base_probs[row_tensor, targets].detach().float().cpu().numpy()
                    sync_if_cuda(device)
                    transfer_seconds += perf_counter() - transfer_started
                    positions = (
                        (chunk_start + valid_rows) * int(window_bases)
                        + token_step * carbon_k
                        + int(base_offset)
                    )
                    target_prob_by_position[positions] = probs.astype(np.float64, copy=False)

    if np.any(np.isnan(target_prob_by_position)):
        missing = int(np.count_nonzero(np.isnan(target_prob_by_position)))
        raise RuntimeError(f"Carbon full-forward trace missed {missing} target base probabilities")
    core_sequence, target_prob, target_symbol, emit_position = _target_trace_arrays_from_position_probs(
        sequence=sequence,
        target_prob_by_position=target_prob_by_position,
        window_bases=int(window_bases),
    )
    metadata = {
        "trace_generation_seconds": perf_counter() - started,
        "model_seconds": model_seconds,
        "factorize_seconds": factorize_seconds,
        "probability_transfer_seconds": transfer_seconds,
        "batch_size": resolved_batch_size,
        "batch_count": int((window_count + resolved_batch_size - 1) // resolved_batch_size),
        "window_count": window_count,
        "window_bases": int(window_bases),
        "internal_token_merge_size": carbon_k,
        "internal_tokens_per_window": tokens_per_window,
        "carbon_trace_source": "full_forward_target_prob_factorized_6mer_to_base",
        "carbon_alignment": "logits over <dna_begin> plus previous 6-mer gathered for next 6-mer then factorized to bases",
        **adapter.metadata(),
    }
    return sequence, core_sequence, target_prob, target_symbol, emit_position, metadata


def _load_evo2(*, model_name: str, local_path: Path, use_kernels: bool) -> Any:
    from evo2 import Evo2

    torch.serialization.add_safe_globals([codecs.encode])
    return Evo2(model_name, local_path=str(local_path), use_kernels=use_kernels)


def _set_evo2_offsets(inference_params: dict[str, Any], offset: int) -> None:
    for key in ("mha", "hcl", "hcm", "hcs"):
        if key in inference_params and hasattr(inference_params[key], "seqlen_offset"):
            inference_params[key].seqlen_offset = int(offset)


def _extract_evo2_logits(outputs: Any) -> torch.Tensor:
    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if not torch.is_tensor(logits):
        raise TypeError(f"Evo2 forward returned unsupported logits type: {type(logits)!r}")
    return logits


def _evo2_target_probabilities(
    *,
    model: Any,
    tokenizer: Any,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: str,
    window_bases: int,
    batch_fallback: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    sequence = _filtered_acgt_sequence(payload)
    if not sequence:
        raise ValueError("Evo2 trace requires at least one A/C/G/T base")
    stateful_model = model if hasattr(model, "initialize_inference_params") else getattr(model, "model", None)
    if stateful_model is None or not hasattr(stateful_model, "initialize_inference_params"):
        raise AttributeError("Evo2 model does not expose initialize_inference_params on wrapper or .model")

    window_lengths = _window_lengths(len(sequence), int(window_bases))
    window_count = int(window_lengths.shape[0])
    resolved_batch_size = _resolve_batch_size(batch_size, window_count, batch_fallback)
    pad_id = int(getattr(tokenizer, "pad_id", 0))
    target_prob_by_position = np.full((len(sequence),), np.nan, dtype=np.float64)
    model_seconds = 0.0
    transfer_seconds = 0.0
    autocast_dtype = _torch_dtype_from_name(dtype_name)
    use_autocast = device.type == "cuda" and autocast_dtype in {torch.float16, torch.bfloat16}

    for chunk_start in range(0, window_count, resolved_batch_size):
        chunk_end = min(window_count, chunk_start + resolved_batch_size)
        chunk_size = chunk_end - chunk_start
        token_batch = np.full((chunk_size, int(window_bases)), pad_id, dtype=np.int64)
        for row, window_id in enumerate(range(chunk_start, chunk_end)):
            base_start = window_id * int(window_bases)
            window_seq = sequence[base_start : base_start + int(window_lengths[window_id])]
            token_ids = tokenizer.tokenize(window_seq)
            token_batch[row, : len(token_ids)] = np.asarray(token_ids, dtype=np.int64)
            if window_seq:
                target_prob_by_position[base_start] = 0.25

        tokens_cpu = torch.from_numpy(token_batch).long().contiguous()
        tokens = tokens_cpu.to(device, non_blocking=True)
        inference_params = stateful_model.initialize_inference_params(max_seqlen=int(window_bases))
        if "mha" in inference_params and hasattr(inference_params["mha"], "max_batch_size"):
            inference_params["mha"].max_batch_size = int(chunk_size)
        rows_all = np.arange(chunk_size, dtype=np.int64)

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                for step in range(int(window_bases) - 1):
                    valid_rows = rows_all[window_lengths[chunk_start:chunk_end] > step + 1]
                    if valid_rows.size == 0:
                        break
                    _set_evo2_offsets(inference_params, step)
                    current = tokens[:, step].reshape(-1, 1)
                    target = tokens[:, step + 1]
                    model_started = perf_counter()
                    outputs, inference_params = stateful_model(current, inference_params_dict=inference_params)
                    logits = _extract_evo2_logits(outputs)[:, -1, :]
                    log_probs = torch.log_softmax(logits, dim=-1)
                    sync_if_cuda(device)
                    model_seconds += perf_counter() - model_started

                    row_tensor = torch.as_tensor(valid_rows, dtype=torch.long, device=device)
                    transfer_started = perf_counter()
                    probs = torch.gather(
                        log_probs[row_tensor],
                        dim=1,
                        index=target[row_tensor].reshape(-1, 1),
                    ).squeeze(1).float().exp().detach().cpu().numpy()
                    sync_if_cuda(device)
                    transfer_seconds += perf_counter() - transfer_started
                    positions = (chunk_start + valid_rows) * int(window_bases) + step + 1
                    target_prob_by_position[positions] = probs.astype(np.float64, copy=False)

    if np.any(np.isnan(target_prob_by_position)):
        missing = int(np.count_nonzero(np.isnan(target_prob_by_position)))
        raise RuntimeError(f"Evo2 trace missed {missing} target base probabilities")
    core_sequence, target_prob, target_symbol, emit_position = _target_trace_arrays_from_position_probs(
        sequence=sequence,
        target_prob_by_position=target_prob_by_position,
        window_bases=int(window_bases),
    )
    metadata = {
        "trace_generation_seconds": perf_counter() - started,
        "model_seconds": model_seconds,
        "probability_transfer_seconds": transfer_seconds,
        "batch_size": resolved_batch_size,
        "batch_count": int((window_count + resolved_batch_size - 1) // resolved_batch_size),
        "window_count": window_count,
        "window_bases": int(window_bases),
        "evo2_trace_source": "streaming_cache_target_prob_base_tokens_first_base_uniform",
        "first_base_probability": 0.25,
    }
    return sequence, core_sequence, target_prob, target_symbol, emit_position, metadata


def _evo2_target_probabilities_full_forward(
    *,
    model: Any,
    tokenizer: Any,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: str,
    window_bases: int,
    batch_fallback: int,
) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    started = perf_counter()
    sequence = _filtered_acgt_sequence(payload)
    if not sequence:
        raise ValueError("Evo2 trace requires at least one A/C/G/T base")

    if hasattr(model, "eval"):
        model.eval()
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "eval"):
        inner_model.eval()

    window_lengths = _window_lengths(len(sequence), int(window_bases))
    window_count = int(window_lengths.shape[0])
    resolved_batch_size = _resolve_batch_size(batch_size, window_count, batch_fallback)
    pad_id = int(getattr(tokenizer, "pad_id", 0))
    target_prob_by_position = np.full((len(sequence),), np.nan, dtype=np.float64)
    model_seconds = 0.0
    softmax_seconds = 0.0
    probability_transfer_seconds = 0.0
    autocast_dtype = _torch_dtype_from_name(dtype_name)
    use_autocast = device.type == "cuda" and autocast_dtype in {torch.float16, torch.bfloat16}

    for chunk_start in range(0, window_count, resolved_batch_size):
        chunk_end = min(window_count, chunk_start + resolved_batch_size)
        chunk_lengths = window_lengths[chunk_start:chunk_end]
        chunk_size = chunk_end - chunk_start
        max_length = int(chunk_lengths.max(initial=0))
        if max_length <= 0:
            continue
        token_batch = np.full((chunk_size, max_length), pad_id, dtype=np.int64)
        for row, window_id in enumerate(range(chunk_start, chunk_end)):
            base_start = window_id * int(window_bases)
            window_seq = sequence[base_start : base_start + int(window_lengths[window_id])]
            token_ids = tokenizer.tokenize(window_seq)
            if len(token_ids) != len(window_seq):
                raise ValueError(
                    f"Evo2 full-forward trace expects one token per base: {len(token_ids)} != {len(window_seq)}"
                )
            token_batch[row, : len(token_ids)] = np.asarray(token_ids, dtype=np.int64)
            if window_seq:
                target_prob_by_position[base_start] = 0.25

        if max_length <= 1:
            continue

        tokens_cpu = torch.from_numpy(token_batch).long().contiguous()
        transfer_started = perf_counter()
        tokens = tokens_cpu.to(device, non_blocking=True)
        sync_if_cuda(device)
        probability_transfer_seconds += perf_counter() - transfer_started

        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                model_started = perf_counter()
                outputs = model(tokens)
                logits = _extract_evo2_logits(outputs)
                sync_if_cuda(device)
                model_seconds += perf_counter() - model_started

                softmax_started = perf_counter()
                log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                sync_if_cuda(device)
                softmax_seconds += perf_counter() - softmax_started

            targets = tokens[:, 1:]
            transfer_started = perf_counter()
            gathered = torch.gather(log_probs, dim=2, index=targets.unsqueeze(-1)).squeeze(-1)
            gathered_cpu = gathered.float().exp().detach().cpu().numpy()
            sync_if_cuda(device)
            probability_transfer_seconds += perf_counter() - transfer_started

        for row, window_id in enumerate(range(chunk_start, chunk_end)):
            length = int(window_lengths[window_id])
            if length <= 1:
                continue
            base_start = window_id * int(window_bases)
            positions = base_start + np.arange(1, length, dtype=np.int64)
            target_prob_by_position[positions] = gathered_cpu[row, : length - 1].astype(np.float64, copy=False)

    if np.any(np.isnan(target_prob_by_position)):
        missing = int(np.count_nonzero(np.isnan(target_prob_by_position)))
        raise RuntimeError(f"Evo2 full-forward trace missed {missing} target base probabilities")
    core_sequence, target_prob, target_symbol, emit_position = _target_trace_arrays_from_position_probs(
        sequence=sequence,
        target_prob_by_position=target_prob_by_position,
        window_bases=int(window_bases),
    )
    metadata = {
        "trace_generation_seconds": perf_counter() - started,
        "model_seconds": model_seconds,
        "softmax_seconds": softmax_seconds,
        "probability_transfer_seconds": probability_transfer_seconds,
        "batch_size": resolved_batch_size,
        "batch_count": int((window_count + resolved_batch_size - 1) // resolved_batch_size),
        "window_count": window_count,
        "window_bases": int(window_bases),
        "evo2_trace_source": "full_forward_target_prob_base_tokens_first_base_uniform",
        "evo2_alignment": "log_softmax(logits[:, :-1]) gathered at input_ids[:, 1:]",
        "first_base_probability": 0.25,
    }
    return sequence, core_sequence, target_prob, target_symbol, emit_position, metadata


def main() -> None:
    args = _build_parser().parse_args()
    source_file = Path(args.source_file)
    payload = _read_payload(source_file, str(args.source_format), args.max_bases)
    output_trace = Path(args.output_trace)

    if args.model == "megabyte":
        device = resolve_device(str(args.device))
        run_dir = Path(args.run_dir)
        checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / f"{args.checkpoint_tag}.pt"
        config, model, checkpoint_metadata = load_megabyte_model_for_fusion(
            run_dir=run_dir,
            checkpoint_path=checkpoint,
            device=device,
        )
        dtype_name = str(args.dtype or config.train.dtype)
        adapter = MegabyteStreamingAdapter(model=model, config=config)
        window_bases = int(args.nc_prefix_window_bases or (adapter.seq_length * adapter.token_merge_size))
        model_window_tokens = _resolve_megabyte_model_window_tokens(args, int(adapter.token_merge_size))
        if model_window_tokens is not None:
            _set_megabyte_model_window_tokens(adapter, int(model_window_tokens))
        if window_bases % int(adapter.token_merge_size) == 0:
            sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = _megabyte_target_probabilities(
                adapter=adapter,
                payload=payload,
                device=device,
                dtype_name=dtype_name,
                batch_size=str(args.batch_size),
                window_bases=window_bases,
                model_window_tokens=model_window_tokens,
                batch_fallback=DEFAULT_NC_PREFIX_MIN_WINDOWS,
            )
            trace_token_merge_size = int(adapter.token_merge_size)
        else:
            probability_mode = str(args.megabyte_probability_mode)
            if probability_mode == "auto":
                probability_mode = "streaming_cache"
            if probability_mode == "streaming_cache":
                sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = (
                    _megabyte_target_probabilities_base_trace(
                        adapter=adapter,
                        payload=payload,
                        device=device,
                        dtype_name=dtype_name,
                        batch_size=str(args.batch_size),
                        trace_window_bases=window_bases,
                        model_window_tokens=model_window_tokens,
                        batch_fallback=DEFAULT_NC_PREFIX_MIN_WINDOWS,
                    )
                )
            else:
                sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = (
                    _megabyte_target_probabilities_full_forward_base_trace(
                        adapter=adapter,
                        payload=payload,
                        device=device,
                        dtype_name=dtype_name,
                        batch_size=str(args.batch_size),
                        trace_window_bases=window_bases,
                        model_window_tokens=model_window_tokens,
                        batch_fallback=DEFAULT_NC_PREFIX_MIN_WINDOWS,
                    )
                )
            trace_token_merge_size = 1
        producer_config = {
            "model": "megabyte",
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_metadata": checkpoint_metadata,
            "device": str(device),
            "dtype": dtype_name,
            **metadata,
        }
        model_id = str(checkpoint)
        token_merge_size = int(metadata.get("trace_token_merge_size", trace_token_merge_size))
    elif args.model == "nc_prefix":
        window_bases = int(args.nc_prefix_window_bases)
        token_merge_size = int(args.token_merge_size)
        sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = _nc_prefix_target_probabilities(
            payload=payload,
            window_bases=window_bases,
            token_merge_size=token_merge_size,
            backend=str(args.nc_prefix_backend),
            min_windows=int(args.nc_prefix_min_windows),
            hash_bucket_count=int(args.nc_prefix_hash_bucket_count),
            geco2_level=int(args.nc_prefix_geco2_level),
        )
        producer_config = {
            "model": "nc_prefix",
            "nc_prefix_window_bases": window_bases,
            "token_merge_size": token_merge_size,
            "nc_prefix_backend": str(args.nc_prefix_backend),
            "nc_prefix_min_windows": int(args.nc_prefix_min_windows),
            "nc_prefix_hash_bucket_count": int(args.nc_prefix_hash_bucket_count),
            "nc_prefix_geco2_level": int(args.nc_prefix_geco2_level),
            **metadata,
        }
        model_id = f"nc_prefix_w{window_bases}_tm{token_merge_size}_level{int(args.nc_prefix_geco2_level)}"
    elif args.model == "carbon":
        device = resolve_device(str(args.device))
        local_path = Path(args.local_path or "third_party/Carbon-3B")
        dtype_name = str(args.dtype or "bfloat16")
        adapter, checkpoint_metadata = load_carbon_adapter_for_fusion(
            local_path=local_path,
            device=device,
            dtype_name=dtype_name,
            model_name=str(args.model_name or "Carbon-3B"),
            revision=str(args.revision),
            context_bases=int(args.nc_prefix_window_bases),
            trust_remote_code=bool(args.trust_remote_code),
        )
        window_bases = int(args.nc_prefix_window_bases)
        token_merge_size = 1
        carbon_probability_mode = str(args.carbon_probability_mode)
        if carbon_probability_mode == "full_forward":
            sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = (
                _carbon_target_probabilities_full_forward(
                    adapter=adapter,
                    payload=payload,
                    device=device,
                    dtype_name=dtype_name,
                    batch_size=str(args.batch_size),
                    window_bases=window_bases,
                    batch_fallback=256,
                )
            )
        else:
            sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = _carbon_target_probabilities(
                adapter=adapter,
                payload=payload,
                device=device,
                dtype_name=dtype_name,
                batch_size=str(args.batch_size),
                window_bases=window_bases,
                batch_fallback=256,
            )
        producer_config = {
            "model": "carbon",
            "local_path": str(local_path),
            "model_name": str(args.model_name or "Carbon-3B"),
            "revision": str(args.revision),
            "checkpoint_metadata": checkpoint_metadata,
            "device": str(device),
            "dtype": dtype_name,
            "carbon_probability_mode": carbon_probability_mode,
            **metadata,
        }
        model_id = f"{local_path}:{carbon_probability_mode}_base_factorized_w{window_bases}"
    elif args.model == "evo2":
        device = resolve_device(str(args.device))
        local_path = Path(args.local_path or "third_party/evo2_7b_base/evo2_7b_base.pt")
        model_name = str(args.model_name or "evo2_7b_base")
        dtype_name = str(args.dtype or "bfloat16")
        model = _load_evo2(model_name=model_name, local_path=local_path, use_kernels=bool(args.use_kernels))
        window_bases = int(args.nc_prefix_window_bases)
        token_merge_size = 1
        evo2_probability_mode = str(args.evo2_probability_mode)
        if evo2_probability_mode == "full_forward":
            sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = (
                _evo2_target_probabilities_full_forward(
                    model=model,
                    tokenizer=model.tokenizer,
                    payload=payload,
                    device=device,
                    dtype_name=dtype_name,
                    batch_size=str(args.batch_size),
                    window_bases=window_bases,
                    batch_fallback=48,
                )
            )
        else:
            sequence, core_sequence, target_prob, target_symbol, emit_position, metadata = _evo2_target_probabilities(
                model=model,
                tokenizer=model.tokenizer,
                payload=payload,
                device=device,
                dtype_name=dtype_name,
                batch_size=str(args.batch_size),
                window_bases=window_bases,
                batch_fallback=48,
            )
        producer_config = {
            "model": "evo2",
            "local_path": str(local_path),
            "model_name": model_name,
            "device": str(device),
            "dtype": dtype_name,
            "use_kernels": bool(args.use_kernels),
            "evo2_probability_mode": evo2_probability_mode,
            **metadata,
        }
        model_id = f"{local_path}:{evo2_probability_mode}_base_w{window_bases}"
    else:
        raise ValueError(f"unsupported model: {args.model}")

    tail_sequence = sequence[len(core_sequence) :]
    manifest = write_target_probability_trace(
        output_trace,
        model_family=str(args.model),
        model_id=model_id,
        source_payload=payload,
        normalized_sequence=sequence,
        core_sequence=core_sequence,
        tail_sequence=tail_sequence,
        target_prob=target_prob,
        target_symbol=target_symbol,
        emit_position=emit_position,
        window_bases=window_bases,
        token_merge_size=token_merge_size,
        producer_config={
            "source_file": str(source_file),
            "source_format": str(args.source_format),
            "max_bases": args.max_bases,
            **producer_config,
        },
        dtype=str(args.trace_dtype),
        shard_rows=int(args.shard_rows),
        overwrite=bool(args.force),
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "output_trace": str(output_trace),
                    "manifest": manifest.to_json_dict(),
                    "target_probability_bits": float((-np.log2(np.clip(target_prob, 1e-300, None))).sum()),
                    "target_probability_bpb": float((-np.log2(np.clip(target_prob, 1e-300, None))).sum())
                    / max(int(manifest.core_base_count), 1),
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
