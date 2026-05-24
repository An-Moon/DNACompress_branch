from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import torch

from .compression import (
    ArithmeticEncoder,
    baseline_sizes,
    probabilities_to_cumulative_batch,
    resolve_arithmetic_coding_metadata,
)
from .compression_eval import NON_OVERLAP_MODE, autocast_context, sample_payload
from .config import ExperimentConfig
from .dnagpt_data import max_target_tokens
from .dnagpt_loader import (
    build_dnagpt_components,
    load_dnagpt_checkpoint,
)
from .dnagpt_tokenization import resolve_species_prefix_token
from .experiment import resolve_device
from .megabyte_loader import build_model, load_megabyte_checkpoint
from .tokenization import apply_token_merge_to_model_config, normalize_alphabet, tokenize_source_bytes


FUSION_STATIC_CONTEXT = "static_context"
FUSION_ORACLE_MAX = "oracle_max"
SUPPORTED_FUSION_POLICIES = (FUSION_STATIC_CONTEXT, FUSION_ORACLE_MAX)


@dataclass(frozen=True)
class UnitProbabilityResult:
    adapter_name: str
    probabilities: np.ndarray
    model_forward_seconds: float
    softmax_seconds: float
    aggregate_seconds: float
    data_transfer_seconds: float


@dataclass(frozen=True)
class FusionSourceInputs:
    payload: bytes
    normalized_sequence: str
    core_sequence: str
    target_symbols: np.ndarray
    core_base_count: int
    tail_sequence: str
    unit_size: int
    alphabet: str


@dataclass(frozen=True)
class StaticContextTable:
    context_units: int
    adapter_names: tuple[str, ...]
    global_model_index: int
    context_to_model_index: dict[tuple[int, ...], int]
    context_counts: dict[tuple[int, ...], int]
    global_bits: tuple[float, ...]
    global_count: int

    def select_model_index(self, context: tuple[int, ...]) -> int:
        return self.context_to_model_index.get(context, self.global_model_index)

    def to_json_dict(self) -> dict[str, object]:
        rows: dict[str, object] = {}
        for context, model_index in sorted(self.context_to_model_index.items()):
            key = ",".join(str(value) for value in context)
            rows[key] = {
                "model": self.adapter_names[model_index],
                "model_index": model_index,
                "count": self.context_counts.get(context, 0),
            }
        return {
            "context_units": self.context_units,
            "adapter_names": list(self.adapter_names),
            "global_model": self.adapter_names[self.global_model_index],
            "global_model_index": self.global_model_index,
            "global_bits": list(self.global_bits),
            "global_count": self.global_count,
            "contexts": rows,
        }


class ProbabilityAdapter(ABC):
    name: str
    token_size: int
    alphabet: str

    @abstractmethod
    def unit_probabilities(
        self,
        *,
        species: str,
        core_sequence: str,
        unit_size: int,
        batch_size: int,
    ) -> UnitProbabilityResult:
        raise NotImplementedError


def resolve_fusion_unit_size(token_sizes: Iterable[int], requested: str | int = "auto") -> int:
    sizes = [int(size) for size in token_sizes]
    if not sizes:
        raise ValueError("At least one token size is required.")
    if any(size <= 0 for size in sizes):
        raise ValueError("Token sizes must be positive.")

    if requested == "auto":
        unit_size = sizes[0]
        for size in sizes[1:]:
            unit_size = math.gcd(unit_size, size)
        return max(unit_size, 1)

    unit_size = int(requested)
    if unit_size <= 0:
        raise ValueError("fusion unit size must be positive.")
    invalid = [size for size in sizes if size % unit_size != 0]
    if invalid:
        raise ValueError(
            "fusion unit size must divide every model token size: "
            f"unit_size={unit_size}, token_sizes={sizes}"
        )
    return unit_size


def lcm_token_size(token_sizes: Iterable[int]) -> int:
    result = 1
    for size in token_sizes:
        result = result * int(size) // math.gcd(result, int(size))
    return result


def normalize_dna_sequence(payload: bytes, alphabet: str) -> str:
    normalized_alphabet = normalize_alphabet(alphabet)
    allowed = set(normalized_alphabet)
    text = payload.decode("ascii", errors="ignore").upper()
    return "".join(ch for ch in text if ch in allowed)


def encode_unit_symbols(sequence: str, unit_size: int, alphabet: str) -> np.ndarray:
    if unit_size <= 0:
        raise ValueError("unit_size must be positive.")
    alphabet = normalize_alphabet(alphabet)
    base = len(alphabet)
    base_to_digit = {base_char: index for index, base_char in enumerate(alphabet)}
    full_base_count = (len(sequence) // unit_size) * unit_size
    symbols: list[int] = []
    for start in range(0, full_base_count, unit_size):
        value = 0
        for base_char in sequence[start : start + unit_size]:
            value = value * base + base_to_digit[base_char]
        symbols.append(value)
    return np.asarray(symbols, dtype=np.int64)


def build_fusion_source_inputs(
    *,
    source: bytes,
    requested_bytes: int | None,
    token_sizes: Iterable[int],
    unit_size: int,
    alphabet: str,
) -> FusionSourceInputs:
    payload = sample_payload(source, requested_bytes)
    normalized_sequence = normalize_dna_sequence(payload, alphabet)
    token_lcm = lcm_token_size(token_sizes)
    core_base_count = (len(normalized_sequence) // token_lcm) * token_lcm
    core_sequence = normalized_sequence[:core_base_count]
    tail_sequence = normalized_sequence[core_base_count:]
    target_symbols = encode_unit_symbols(core_sequence, unit_size, alphabet)
    return FusionSourceInputs(
        payload=payload,
        normalized_sequence=normalized_sequence,
        core_sequence=core_sequence,
        target_symbols=target_symbols,
        core_base_count=core_base_count,
        tail_sequence=tail_sequence,
        unit_size=unit_size,
        alphabet=normalize_alphabet(alphabet),
    )


def unit_id_to_piece(unit_id: int, unit_size: int, alphabet: str) -> str:
    alphabet = normalize_alphabet(alphabet)
    base = len(alphabet)
    value = int(unit_id)
    chars = [""] * unit_size
    for index in range(unit_size - 1, -1, -1):
        chars[index] = alphabet[value % base]
        value //= base
    return "".join(chars)


def _regular_token_pieces(token_size: int, unit_size: int, alphabet: str) -> list[str]:
    unit_vocab_size = len(normalize_alphabet(alphabet)) ** unit_size
    units_per_token = token_size // unit_size
    pieces: list[str] = []
    for chunk_ids in product(range(unit_vocab_size), repeat=units_per_token):
        pieces.append("".join(unit_id_to_piece(unit_id, unit_size, alphabet) for unit_id in chunk_ids))
    return pieces


def factorize_token_probabilities_to_units(
    *,
    token_probabilities: np.ndarray,
    target_unit_symbols: np.ndarray,
    token_size: int,
    unit_size: int,
    alphabet: str,
) -> np.ndarray:
    if token_size % unit_size != 0:
        raise ValueError("unit_size must divide token_size.")
    probs = np.asarray(token_probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("token_probabilities must be 2D.")

    alphabet = normalize_alphabet(alphabet)
    unit_vocab_size = len(alphabet) ** unit_size
    units_per_token = token_size // unit_size
    expected_vocab_size = unit_vocab_size ** units_per_token
    if probs.shape[1] != expected_vocab_size:
        raise ValueError(
            f"Expected token vocab size {expected_vocab_size}, got {probs.shape[1]}."
        )
    expected_units = probs.shape[0] * units_per_token
    if int(target_unit_symbols.shape[0]) != expected_units:
        raise ValueError(
            "target unit count must equal token row count times units_per_token: "
            f"{target_unit_symbols.shape[0]} != {expected_units}"
        )

    normalized = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-300)
    token_tables = normalized.reshape((probs.shape[0],) + (unit_vocab_size,) * units_per_token)
    target_chunks = np.asarray(target_unit_symbols, dtype=np.int64).reshape(probs.shape[0], units_per_token)
    unit_rows = np.zeros((expected_units, unit_vocab_size), dtype=np.float64)

    out_index = 0
    for row_index in range(probs.shape[0]):
        table = token_tables[row_index]
        for step in range(units_per_token):
            current = table
            if step > 0:
                current = current[tuple(int(value) for value in target_chunks[row_index, :step])]
            if step < units_per_token - 1:
                axes = tuple(range(1, current.ndim))
                distribution = current.sum(axis=axes)
            else:
                distribution = current
            unit_rows[out_index] = distribution / max(float(distribution.sum()), 1e-300)
            out_index += 1
    return unit_rows


def contexts_for_targets(target_symbols: np.ndarray, context_units: int) -> list[tuple[int, ...]]:
    if context_units <= 0:
        raise ValueError("context_units must be positive.")
    symbols = [int(value) for value in target_symbols.tolist()]
    contexts: list[tuple[int, ...]] = []
    for index in range(len(symbols)):
        start = max(0, index - context_units)
        prefix = symbols[start:index]
        missing = context_units - len(prefix)
        contexts.append(tuple([-1] * missing + prefix))
    return contexts


def fit_static_context_table(
    *,
    adapter_names: list[str],
    target_symbols: np.ndarray,
    model_probabilities: list[np.ndarray],
    context_units: int,
    min_context_count: int = 1,
) -> StaticContextTable:
    if len(adapter_names) != len(model_probabilities):
        raise ValueError("adapter_names and model_probabilities length mismatch.")
    if not model_probabilities:
        raise ValueError("At least one model probability matrix is required.")

    target_symbols = np.asarray(target_symbols, dtype=np.int64)
    contexts = contexts_for_targets(target_symbols, context_units)
    model_count = len(model_probabilities)
    context_sums: dict[tuple[int, ...], list[float]] = {}
    context_counts: dict[tuple[int, ...], int] = {}
    global_sums = [0.0 for _ in range(model_count)]

    for row_index, target in enumerate(target_symbols.tolist()):
        context = contexts[row_index]
        sums = context_sums.setdefault(context, [0.0 for _ in range(model_count)])
        context_counts[context] = context_counts.get(context, 0) + 1
        for model_index, probabilities in enumerate(model_probabilities):
            probability = max(float(probabilities[row_index, int(target)]), 1e-300)
            bits = -math.log2(probability)
            sums[model_index] += bits
            global_sums[model_index] += bits

    global_count = int(target_symbols.shape[0])
    global_averages = [
        total / max(global_count, 1)
        for total in global_sums
    ]
    global_model_index = int(np.argmin(np.asarray(global_averages, dtype=np.float64)))
    context_to_model_index: dict[tuple[int, ...], int] = {}
    for context, sums in context_sums.items():
        count = context_counts[context]
        if count < min_context_count:
            continue
        averages = [total / count for total in sums]
        context_to_model_index[context] = int(np.argmin(np.asarray(averages, dtype=np.float64)))

    return StaticContextTable(
        context_units=context_units,
        adapter_names=tuple(adapter_names),
        global_model_index=global_model_index,
        context_to_model_index=context_to_model_index,
        context_counts=context_counts,
        global_bits=tuple(global_sums),
        global_count=global_count,
    )


class StaticContextAccumulator:
    def __init__(self, *, adapter_names: list[str], context_units: int, min_context_count: int = 1) -> None:
        if context_units <= 0:
            raise ValueError("context_units must be positive.")
        self.adapter_names = list(adapter_names)
        self.context_units = int(context_units)
        self.min_context_count = int(min_context_count)
        self._model_count = len(adapter_names)
        self._context_sums: dict[tuple[int, ...], list[float]] = {}
        self._context_counts: dict[tuple[int, ...], int] = {}
        self._global_sums = [0.0 for _ in range(self._model_count)]
        self._global_count = 0

    def update(self, *, target_symbols: np.ndarray, model_probabilities: list[np.ndarray]) -> None:
        if len(model_probabilities) != self._model_count:
            raise ValueError("model probability count does not match accumulator.")
        target_symbols = np.asarray(target_symbols, dtype=np.int64)
        contexts = contexts_for_targets(target_symbols, self.context_units)
        for row_index, target in enumerate(target_symbols.tolist()):
            context = contexts[row_index]
            sums = self._context_sums.setdefault(context, [0.0 for _ in range(self._model_count)])
            self._context_counts[context] = self._context_counts.get(context, 0) + 1
            self._global_count += 1
            for model_index, probabilities in enumerate(model_probabilities):
                probability = max(float(probabilities[row_index, int(target)]), 1e-300)
                bits = -math.log2(probability)
                sums[model_index] += bits
                self._global_sums[model_index] += bits

    def finalize(self) -> StaticContextTable:
        global_averages = [bits / max(self._global_count, 1) for bits in self._global_sums]
        global_model_index = int(np.argmin(np.asarray(global_averages, dtype=np.float64)))
        context_to_model_index: dict[tuple[int, ...], int] = {}
        for context, sums in self._context_sums.items():
            count = self._context_counts[context]
            if count < self.min_context_count:
                continue
            averages = [bits / count for bits in sums]
            context_to_model_index[context] = int(np.argmin(np.asarray(averages, dtype=np.float64)))
        return StaticContextTable(
            context_units=self.context_units,
            adapter_names=tuple(self.adapter_names),
            global_model_index=global_model_index,
            context_to_model_index=context_to_model_index,
            context_counts=dict(self._context_counts),
            global_bits=tuple(self._global_sums),
            global_count=self._global_count,
        )


def _tail_side_info_bits(tail_sequence: str, alphabet: str) -> int:
    if not tail_sequence:
        return 0
    bits_per_base = math.ceil(math.log2(len(normalize_alphabet(alphabet))))
    return 8 + bits_per_base * len(tail_sequence)


def _encode_policy_probabilities(
    *,
    policy: str,
    adapter_names: list[str],
    target_symbols: np.ndarray,
    model_probabilities: list[np.ndarray],
    static_table: StaticContextTable | None,
    arithmetic_total: int,
    context_units: int,
) -> tuple[float, bytes, dict[str, int], bool]:
    encoder = ArithmeticEncoder()
    choice_counts = {name: 0 for name in adapter_names}
    decodable = policy != FUSION_ORACLE_MAX
    contexts = contexts_for_targets(target_symbols, context_units)
    selected_indices = np.zeros((target_symbols.shape[0],), dtype=np.int64)

    for row_index, target in enumerate(target_symbols.tolist()):
        target = int(target)
        if policy == FUSION_STATIC_CONTEXT:
            if static_table is None:
                raise ValueError("static_table is required for static_context policy.")
            model_index = static_table.select_model_index(contexts[row_index])
        elif policy == FUSION_ORACLE_MAX:
            target_probs = [
                max(float(probabilities[row_index, target]), 1e-300)
                for probabilities in model_probabilities
            ]
            model_index = int(np.argmax(np.asarray(target_probs, dtype=np.float64)))
        else:
            raise ValueError(f"Unsupported fusion policy '{policy}'.")
        selected_indices[row_index] = model_index
        choice_counts[adapter_names[model_index]] += 1

    chosen_probabilities = np.stack(
        [
            model_probabilities[int(model_index)][row_index]
            for row_index, model_index in enumerate(selected_indices.tolist())
        ],
        axis=0,
    )
    target_probabilities = chosen_probabilities[
        np.arange(target_symbols.shape[0], dtype=np.int64),
        target_symbols,
    ].clip(min=1e-300)
    total_bits = float((-np.log2(target_probabilities)).sum())
    cumulative_batch = probabilities_to_cumulative_batch(chosen_probabilities, total=arithmetic_total)
    for cumulative, target in zip(cumulative_batch, target_symbols):
        encoder.update(cumulative, int(target))

    return total_bits, encoder.finish(), choice_counts, decodable


def compress_fusion_source(
    *,
    species: str,
    source: bytes,
    adapters: list[ProbabilityAdapter],
    unit_size: int,
    alphabet: str,
    batch_size: int,
    requested_bytes: int | None,
    policy: str,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    context_units: int,
    static_table: StaticContextTable | None = None,
) -> dict[str, object]:
    started = perf_counter()
    alphabet = normalize_alphabet(alphabet)
    incompatible = [
        f"{adapter.name}:{adapter.alphabet}"
        for adapter in adapters
        if normalize_alphabet(adapter.alphabet) != alphabet
    ]
    if incompatible:
        raise ValueError(f"Adapter alphabets must match fusion alphabet {alphabet}: {incompatible}")
    fusion_input = build_fusion_source_inputs(
        source=source,
        requested_bytes=requested_bytes,
        token_sizes=[adapter.token_size for adapter in adapters],
        unit_size=unit_size,
        alphabet=alphabet,
    )
    if fusion_input.target_symbols.shape[0] == 0:
        raise ValueError("Fusion compression requires at least one complete unit.")

    model_results = [
        adapter.unit_probabilities(
            species=species,
            core_sequence=fusion_input.core_sequence,
            unit_size=unit_size,
            batch_size=batch_size,
        )
        for adapter in adapters
    ]
    model_probabilities = [result.probabilities for result in model_results]
    target_count = int(fusion_input.target_symbols.shape[0])
    for result in model_results:
        if result.probabilities.shape[0] != target_count:
            raise RuntimeError(
                f"Adapter {result.adapter_name} emitted {result.probabilities.shape[0]} unit rows, "
                f"expected {target_count}."
            )

    unit_vocab_size = len(normalize_alphabet(alphabet)) ** unit_size
    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=unit_vocab_size,
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
    )
    encode_started = perf_counter()
    theoretical_bits, encoded, choice_counts, decodable = _encode_policy_probabilities(
        policy=policy,
        adapter_names=[adapter.name for adapter in adapters],
        target_symbols=fusion_input.target_symbols,
        model_probabilities=model_probabilities,
        static_table=static_table,
        arithmetic_total=int(arithmetic_metadata["arithmetic_frequency_total"]),
        context_units=context_units,
    )
    arithmetic_encode_seconds = perf_counter() - encode_started
    tail_side_info_bits = _tail_side_info_bits(fusion_input.tail_sequence, alphabet)
    total_bits = theoretical_bits + tail_side_info_bits
    elapsed = perf_counter() - started

    sample_bases = len(fusion_input.normalized_sequence)
    metrics: dict[str, object] = {
        "mode": NON_OVERLAP_MODE,
        "fusion_policy": policy,
        "decodable": decodable,
        "sample_bytes": len(fusion_input.payload),
        "sample_bases": sample_bases,
        "core_base_count": fusion_input.core_base_count,
        "tail_base_count": len(fusion_input.tail_sequence),
        "tail_side_info_bits": tail_side_info_bits,
        "sample_symbols_with_eos": target_count,
        "uses_eos": False,
        "unit_size": unit_size,
        "unit_vocab_size": unit_vocab_size,
        "context_units": context_units,
        "theoretical_bits": total_bits,
        "core_model_theoretical_bits": theoretical_bits,
        "theoretical_bits_per_base": total_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": len(encoded) + ((tail_side_info_bits + 7) // 8),
        "arithmetic_bits_per_base": ((len(encoded) * 8) + tail_side_info_bits) / max(sample_bases, 1),
        "arithmetic_coding_mode": f"fusion_{policy}",
        "arithmetic_merge_size": unit_size,
        "emitted_arithmetic_symbol_count": target_count,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "compression_process_seconds": elapsed,
        "compression_bytes_per_second": len(fusion_input.payload) / max(elapsed, 1e-12),
        "compression_bases_per_second": sample_bases / max(elapsed, 1e-12),
        "compression_symbols_per_second": target_count / max(elapsed, 1e-12),
        "fusion_model_choice_counts": choice_counts,
        **arithmetic_metadata,
        **baseline_sizes(fusion_input.payload),
    }
    for result in model_results:
        target_probs = result.probabilities[
            np.arange(target_count, dtype=np.int64),
            fusion_input.target_symbols,
        ].clip(min=1e-300)
        model_bits = float((-np.log2(target_probs)).sum())
        prefix = f"model_{result.adapter_name}"
        metrics[f"{prefix}_theoretical_bits"] = model_bits
        metrics[f"{prefix}_theoretical_bits_per_core_base"] = model_bits / max(fusion_input.core_base_count, 1)
        metrics[f"{prefix}_model_forward_seconds"] = result.model_forward_seconds
        metrics[f"{prefix}_softmax_seconds"] = result.softmax_seconds
        metrics[f"{prefix}_aggregate_seconds"] = result.aggregate_seconds
        metrics[f"{prefix}_data_transfer_seconds"] = result.data_transfer_seconds
    return metrics


class MegabyteProbabilityAdapter(ProbabilityAdapter):
    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        config: ExperimentConfig,
        device: torch.device,
        dtype_name: str,
    ) -> None:
        self.name = name
        self.model = model
        self.config = config
        self.device = device
        self.dtype_name = dtype_name
        self.token_size = int(config.data.token_merge_size)
        self.alphabet = normalize_alphabet(config.data.token_merge_alphabet)
        if self.token_size <= 0:
            raise ValueError("Megabyte token_merge_size must be positive.")

    @classmethod
    def from_checkpoint(
        cls,
        *,
        name: str,
        run_dir: Path,
        checkpoint_path: Path,
        device: torch.device,
        dtype_name: str | None = None,
    ) -> "MegabyteProbabilityAdapter":
        config = ExperimentConfig()
        config_path = run_dir / "resolved_config.json"
        if config_path.exists():
            from .config import load_experiment_config

            config = load_experiment_config(config_path)
        apply_token_merge_to_model_config(config.model, config.data)
        model = build_model(config.model).to(device)
        model_state, _, _ = load_megabyte_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(model_state)
        model.eval()
        return cls(
            name=name,
            model=model,
            config=config,
            device=device,
            dtype_name=dtype_name or config.train.dtype,
        )

    def unit_probabilities(
        self,
        *,
        species: str,
        core_sequence: str,
        unit_size: int,
        batch_size: int,
    ) -> UnitProbabilityResult:
        del species
        if self.token_size % unit_size != 0:
            raise ValueError("unit_size must divide Megabyte token size.")
        symbols = tokenize_source_bytes(core_sequence.encode("ascii"), self.token_size, self.alphabet)
        token_count = len(symbols)
        target_units = encode_unit_symbols(core_sequence, unit_size, self.alphabet)
        if token_count == 0:
            raise ValueError("Megabyte adapter requires at least one token.")
        if target_units.shape[0] != token_count * (self.token_size // unit_size):
            raise RuntimeError("Megabyte target unit count does not match token count.")

        seq_length = int(self.config.model.seq_length)
        pad_id = int(self.config.model.pad_id)
        all_unit_probabilities: list[np.ndarray] = []
        model_forward_seconds = 0.0
        softmax_seconds = 0.0
        aggregate_seconds = 0.0
        data_transfer_seconds = 0.0

        self.model.eval()
        with torch.no_grad():
            starts = list(range(0, token_count, seq_length))
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start : batch_start + batch_size]
                windows = torch.full((len(batch_starts), seq_length), pad_id, dtype=torch.long)
                lengths: list[int] = []
                for row_index, start in enumerate(batch_starts):
                    chunk = symbols[start : start + seq_length]
                    lengths.append(len(chunk))
                    if chunk:
                        windows[row_index, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)

                transfer_started = perf_counter()
                batch = windows.to(self.device, non_blocking=True)
                data_transfer_seconds += perf_counter() - transfer_started

                with autocast_context(self.device, self.dtype_name):
                    forward_started = perf_counter()
                    output = self.model(batch, return_loss=False)
                    model_forward_seconds += perf_counter() - forward_started

                    softmax_started = perf_counter()
                    log_probs = torch.log_softmax(output.lm_logits, dim=-1)
                    softmax_seconds += perf_counter() - softmax_started

                for row_index, (start, chunk_length) in enumerate(zip(batch_starts, lengths)):
                    if chunk_length <= 0:
                        continue
                    row_log_probs = log_probs[row_index, :chunk_length, :]
                    aggregate_started = perf_counter()
                    regular_vocab_size = len(self.alphabet) ** self.token_size
                    token_probs = row_log_probs[:, :regular_vocab_size].float().exp().cpu().numpy()
                    target_slice = target_units[
                        start * (self.token_size // unit_size) : (start + chunk_length) * (self.token_size // unit_size)
                    ]
                    unit_rows = factorize_token_probabilities_to_units(
                        token_probabilities=token_probs,
                        target_unit_symbols=target_slice,
                        token_size=self.token_size,
                        unit_size=unit_size,
                        alphabet=self.alphabet,
                    )
                    aggregate_seconds += perf_counter() - aggregate_started
                    all_unit_probabilities.append(unit_rows)

        probabilities = np.concatenate(all_unit_probabilities, axis=0) if all_unit_probabilities else np.zeros((0, 0))
        return UnitProbabilityResult(
            adapter_name=self.name,
            probabilities=probabilities,
            model_forward_seconds=model_forward_seconds,
            softmax_seconds=softmax_seconds,
            aggregate_seconds=aggregate_seconds,
            data_transfer_seconds=data_transfer_seconds,
        )


class DNAGPTProbabilityAdapter(ProbabilityAdapter):
    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        tokenizer,
        spec,
        config: ExperimentConfig,
        device: torch.device,
        dtype_name: str,
    ) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.config = config
        self.device = device
        self.dtype_name = dtype_name
        self.token_size = int(spec.kmer_size)
        self.alphabet = "ACGTN"
        self._regular_token_ids_by_unit: dict[int, torch.Tensor] = {}

    @classmethod
    def from_checkpoint(
        cls,
        *,
        name: str,
        run_dir: Path,
        checkpoint_path: Path,
        device: torch.device,
        dtype_name: str | None = None,
    ) -> "DNAGPTProbabilityAdapter":
        from .config import load_experiment_config

        config = load_experiment_config(run_dir / "resolved_config.json")
        model, tokenizer, spec = build_dnagpt_components(config.model)
        model_state, _, _ = load_dnagpt_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(model_state, strict=False)
        model = model.to(device)
        model.eval()
        return cls(
            name=name,
            model=model,
            tokenizer=tokenizer,
            spec=spec,
            config=config,
            device=device,
            dtype_name=dtype_name or config.train.dtype,
        )

    def _regular_token_ids(self, unit_size: int) -> torch.Tensor:
        cached = self._regular_token_ids_by_unit.get(unit_size)
        if cached is not None:
            return cached
        pieces = _regular_token_pieces(self.token_size, unit_size, self.alphabet)
        token_ids = [int(self.tokenizer.piece_to_id(piece)) for piece in pieces]
        if any(token_id == int(self.tokenizer.unk_id) for token_id in token_ids):
            raise ValueError("DNAGPT tokenizer is missing at least one regular DNA token.")
        tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        self._regular_token_ids_by_unit[unit_size] = tensor
        return tensor

    def unit_probabilities(
        self,
        *,
        species: str,
        core_sequence: str,
        unit_size: int,
        batch_size: int,
    ) -> UnitProbabilityResult:
        if self.token_size % unit_size != 0:
            raise ValueError("unit_size must divide DNAGPT token size.")
        pieces = [
            core_sequence[index : index + self.token_size]
            for index in range(0, len(core_sequence), self.token_size)
        ]
        token_ids = [int(self.tokenizer.piece_to_id(piece)) for piece in pieces if piece]
        token_count = len(token_ids)
        if token_count == 0:
            raise ValueError("DNAGPT adapter requires at least one token.")
        target_units = encode_unit_symbols(core_sequence, unit_size, self.alphabet)
        units_per_token = self.token_size // unit_size
        if target_units.shape[0] != token_count * units_per_token:
            raise RuntimeError("DNAGPT target unit count does not match token count.")

        prefix_token = resolve_species_prefix_token(species, self.config.data.species_prefix_map)
        prefix_ids: list[int] = []
        if prefix_token is not None:
            token_text = prefix_token if prefix_token.startswith("<") else f"<{prefix_token}>"
            prefix_ids.append(int(self.tokenizer.piece_to_id(token_text)))

        seq_length = int(self.config.model.seq_length)
        prefix_length = len(prefix_ids)
        target_capacity = max_target_tokens(seq_length, prefix_length)
        starts = list(range(0, token_count, target_capacity))
        regular_token_ids = self._regular_token_ids(unit_size)
        all_unit_probabilities: list[np.ndarray] = []
        model_forward_seconds = 0.0
        softmax_seconds = 0.0
        aggregate_seconds = 0.0
        data_transfer_seconds = 0.0

        self.model.eval()
        with torch.no_grad():
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start : batch_start + batch_size]
                batch_input = torch.full((len(batch_starts), seq_length), self.tokenizer.pad_id, dtype=torch.long)
                chunk_lengths: list[int] = []
                for row_index, start in enumerate(batch_starts):
                    chunk = token_ids[start : start + target_capacity]
                    chunk_lengths.append(len(chunk))
                    for prefix_index, prefix_id in enumerate(prefix_ids):
                        batch_input[row_index, prefix_index + 1] = int(prefix_id)
                    for offset, token_id in enumerate(chunk[:-1]):
                        batch_input[row_index, prefix_length + offset + 1] = int(token_id)

                transfer_started = perf_counter()
                batch = batch_input.to(self.device, non_blocking=True)
                data_transfer_seconds += perf_counter() - transfer_started

                with autocast_context(self.device, self.dtype_name):
                    forward_started = perf_counter()
                    logits = self.model(batch)
                    model_forward_seconds += perf_counter() - forward_started

                    softmax_started = perf_counter()
                    log_probs = torch.log_softmax(logits, dim=-1)
                    softmax_seconds += perf_counter() - softmax_started

                for row_index, (start, chunk_length) in enumerate(zip(batch_starts, chunk_lengths)):
                    if chunk_length <= 0:
                        continue
                    row_log_probs = log_probs[row_index, prefix_length : prefix_length + chunk_length, :]
                    aggregate_started = perf_counter()
                    token_probs = row_log_probs.index_select(1, regular_token_ids).float().exp().cpu().numpy()
                    target_slice = target_units[start * units_per_token : (start + chunk_length) * units_per_token]
                    unit_rows = factorize_token_probabilities_to_units(
                        token_probabilities=token_probs,
                        target_unit_symbols=target_slice,
                        token_size=self.token_size,
                        unit_size=unit_size,
                        alphabet=self.alphabet,
                    )
                    aggregate_seconds += perf_counter() - aggregate_started
                    all_unit_probabilities.append(unit_rows)

        probabilities = np.concatenate(all_unit_probabilities, axis=0) if all_unit_probabilities else np.zeros((0, 0))
        return UnitProbabilityResult(
            adapter_name=self.name,
            probabilities=probabilities,
            model_forward_seconds=model_forward_seconds,
            softmax_seconds=softmax_seconds,
            aggregate_seconds=aggregate_seconds,
            data_transfer_seconds=data_transfer_seconds,
        )


def build_adapter_from_spec(
    *,
    spec: str,
    index: int,
    device_name: str,
    dtype_name: str | None = None,
) -> ProbabilityAdapter:
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise ValueError("Model spec must be kind:run_dir[:checkpoint].")
    kind = parts[0].strip().lower()
    run_dir = Path(parts[1])
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if len(parts) == 3 and parts[2].strip():
        raw_checkpoint = parts[2].strip()
        if raw_checkpoint in {"best", "last"}:
            checkpoint_path = run_dir / f"{raw_checkpoint}.pt"
        else:
            checkpoint_path = Path(raw_checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = run_dir / checkpoint_path
    else:
        checkpoint_path = run_dir / ("last.pt" if kind == "dnagpt" else "best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(device_name)
    name = f"{kind}{index + 1}"
    if kind == "megabyte":
        return MegabyteProbabilityAdapter.from_checkpoint(
            name=name,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype_name=dtype_name,
        )
    if kind == "dnagpt":
        return DNAGPTProbabilityAdapter.from_checkpoint(
            name=name,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype_name=dtype_name,
        )
    raise ValueError(f"Unsupported fusion model kind '{kind}'.")


def write_static_context_table(path: Path, table: StaticContextTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
