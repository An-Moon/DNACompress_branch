from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from queue import Queue
import tempfile
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
_CARBON_FACTOR_TABLE_CACHE: dict[tuple[int, str, str, str, int], list[torch.Tensor]] = {}


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


def _build_token_windows_from_arrays(
    *,
    sequence: str,
    filtered_out: int,
    tokens_np: np.ndarray,
    token_base_symbols_np: np.ndarray,
    token_merge_size: int,
    window_bases: int,
    pad_id: int,
    model_uses_ascii_tokens: bool,
    model_token_alphabet: str,
) -> FusedTokenWindows:
    core_base_count = int(tokens_np.shape[0]) * int(token_merge_size)
    core_sequence = sequence[:core_base_count]
    tail_sequence = sequence[core_base_count:]
    if tokens_np.shape[0] == 0:
        raise ValueError("fused compression requires at least one complete LM token")

    tokens_per_window = int(window_bases) // int(token_merge_size)
    window_count = math.ceil(int(tokens_np.shape[0]) / tokens_per_window)
    padded_token_count = window_count * tokens_per_window
    padded = np.full((padded_token_count,), int(pad_id), dtype=np.int64)
    padded[: tokens_np.shape[0]] = tokens_np.astype(np.int64, copy=False)
    token_base_symbols = np.zeros((padded_token_count, int(token_merge_size)), dtype=np.int64)
    token_base_symbols[: token_base_symbols_np.shape[0], :] = token_base_symbols_np

    valid_lengths = np.full((window_count,), tokens_per_window, dtype=np.int64)
    tail_tokens = int(tokens_np.shape[0]) - (window_count - 1) * tokens_per_window
    valid_lengths[-1] = tail_tokens
    return FusedTokenWindows(
        sequence=sequence,
        core_sequence=core_sequence,
        tail_sequence=tail_sequence,
        tokens=torch.from_numpy(padded.reshape(window_count, tokens_per_window)).long().contiguous(),
        token_base_symbols=torch.from_numpy(token_base_symbols.reshape(window_count, tokens_per_window, int(token_merge_size)))
        .long()
        .contiguous(),
        valid_token_lengths=torch.from_numpy(valid_lengths).long().contiguous(),
        token_merge_size=int(token_merge_size),
        token_window_bases=int(window_bases),
        model_uses_ascii_tokens=bool(model_uses_ascii_tokens),
        model_token_alphabet=normalize_alphabet(model_token_alphabet),
        filtered_out_bases=int(filtered_out),
    )


def _move_tensor_tree(value: Any, device: torch.device) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _move_tensor_tree(item, device)
        return value
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree(item, device) for item in value)
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _move_tensor_tree(item, device)
        return value
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        for field_name in fields:
            setattr(value, field_name, _move_tensor_tree(getattr(value, field_name), device))
    return value


class StreamingLMBatch:
    def next_logits(self) -> torch.Tensor:
        raise NotImplementedError

    def accept_symbols(self, token_ids: torch.Tensor) -> None:
        raise NotImplementedError

    def move_state_to(self, device: torch.device) -> None:
        return None


class StreamingLMAdapter:
    name: str
    token_merge_size: int
    token_alphabet: str
    seq_length: int
    pad_id: int

    def eval(self) -> None:
        raise NotImplementedError

    def build_windows(self, payload: bytes, *, window_bases: int, alphabet: str = "ACGT") -> FusedTokenWindows:
        raise NotImplementedError

    def start_batch(self, tokens_cpu: torch.Tensor, *, device: torch.device, dtype_name: str) -> StreamingLMBatch:
        raise NotImplementedError

    def logits_to_acgt_base_probs(
        self,
        logits: torch.Tensor,
        target_base_symbols: torch.Tensor,
        *,
        output_alphabet: str,
    ) -> list[torch.Tensor]:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"lm_backend": self.name}


class MegabyteStreamingBatch(StreamingLMBatch):
    def __init__(self, stepper: MegabyteBatchedDecodeStepper) -> None:
        self.stepper = stepper

    def next_logits(self) -> torch.Tensor:
        return self.stepper.next_logits()

    def accept_symbols(self, token_ids: torch.Tensor) -> None:
        self.stepper.accept_symbols(token_ids)

    def move_state_to(self, device: torch.device) -> None:
        self.stepper.global_caches = _move_tensor_tree(self.stepper.global_caches, device)
        self.stepper.local_caches = _move_tensor_tree(self.stepper.local_caches, device)
        self.stepper.previous_patch_tokens = _move_tensor_tree(self.stepper.previous_patch_tokens, device)
        self.stepper.current_patch_tokens = _move_tensor_tree(self.stepper.current_patch_tokens, device)
        self.stepper.current_patch_context = _move_tensor_tree(self.stepper.current_patch_context, device)
        self.stepper.device = device


class MegabyteStreamingAdapter(StreamingLMAdapter):
    def __init__(self, *, model: torch.nn.Module, config: ExperimentConfig) -> None:
        self.name = "megabyte"
        self.model = model
        self.config = config
        self.token_merge_size = int(config.data.token_merge_size)
        self.token_alphabet = normalize_alphabet(config.data.token_merge_alphabet)
        self.seq_length = int(getattr(model.config, "T_MAX", config.model.seq_length))
        self.pad_id = int(config.model.pad_id)
        self.model_uses_ascii_tokens = _model_uses_ascii_single_base_tokens(model, config)

    def eval(self) -> None:
        self.model.eval()

    def build_windows(self, payload: bytes, *, window_bases: int, alphabet: str = "ACGT") -> FusedTokenWindows:
        return build_fused_token_windows(
            payload,
            model=self.model,
            config=self.config,
            window_bases=window_bases,
            alphabet=alphabet,
        )

    def start_batch(self, tokens_cpu: torch.Tensor, *, device: torch.device, dtype_name: str) -> StreamingLMBatch:
        return MegabyteStreamingBatch(
            MegabyteBatchedDecodeStepper(
                self.model,
                batch_size=int(tokens_cpu.shape[0]),
                device=device,
                dtype_name=dtype_name,
            )
        )

    def logits_to_acgt_base_probs(
        self,
        logits: torch.Tensor,
        target_base_symbols: torch.Tensor,
        *,
        output_alphabet: str,
    ) -> list[torch.Tensor]:
        return _regular_log_probs_to_base_steps(
            logits,
            target_base_symbols,
            token_merge_size=self.token_merge_size,
            model_token_alphabet=self.token_alphabet,
            output_alphabet=output_alphabet,
            model_uses_ascii_tokens=self.model_uses_ascii_tokens,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "lm_backend": self.name,
            "lm_seq_length": int(self.seq_length),
            "lm_token_merge_size": int(self.token_merge_size),
            "lm_token_alphabet": self.token_alphabet,
            "model_uses_ascii_tokens": bool(self.model_uses_ascii_tokens),
        }


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


def _carbon_factorization_tables(
    *,
    tokenizer: Any,
    token_merge_size: int,
    output_alphabet: str,
    device: torch.device,
) -> list[torch.Tensor]:
    output_alphabet = normalize_alphabet(output_alphabet)
    token_merge_size = int(token_merge_size)
    vocab_identity = id(getattr(tokenizer, "dna_token_to_id", tokenizer))
    cache_key = (token_merge_size, "ATCG", output_alphabet, str(device), vocab_identity)
    cached = _CARBON_FACTOR_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not hasattr(tokenizer, "dna_token_to_id"):
        raise ValueError("Carbon tokenizer must expose dna_token_to_id")
    dna_token_to_id = tokenizer.dna_token_to_id
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = max(int(value) for value in dna_token_to_id.values()) + 1
    tables: list[torch.Tensor] = []
    for base_index in range(token_merge_size):
        prefix_count = len(output_alphabet) ** base_index
        future_count = len(output_alphabet) ** (token_merge_size - base_index - 1)
        table = torch.empty((prefix_count, len(output_alphabet), future_count), dtype=torch.long)
        for prefix_code in range(prefix_count):
            prefix_chars: list[str] = []
            cursor = prefix_code
            for power in range(base_index - 1, -1, -1):
                divisor = len(output_alphabet) ** power
                digit = cursor // divisor
                cursor %= divisor
                prefix_chars.append(output_alphabet[digit])
            for candidate_digit, candidate_base in enumerate(output_alphabet):
                for future_code in range(future_count):
                    future_chars: list[str] = []
                    cursor = future_code
                    for power in range(token_merge_size - base_index - 2, -1, -1):
                        divisor = len(output_alphabet) ** power
                        digit = cursor // divisor
                        cursor %= divisor
                        future_chars.append(output_alphabet[digit])
                    token = "".join(prefix_chars + [candidate_base] + future_chars)
                    try:
                        token_id = int(dna_token_to_id[token])
                    except KeyError as exc:
                        raise ValueError(f"Carbon tokenizer is missing DNA token {token!r}") from exc
                    if token_id < 0 or token_id >= vocab_size:
                        raise ValueError("Carbon DNA token id exceeds tokenizer vocabulary size")
                    table[prefix_code, candidate_digit, future_code] = token_id
        tables.append(table.to(device=device))
    _CARBON_FACTOR_TABLE_CACHE[cache_key] = tables
    return tables


def _carbon_conditional_log_probs_to_base_steps(
    logits: torch.Tensor,
    target_base_symbols: torch.Tensor,
    *,
    tokenizer: Any,
    token_merge_size: int,
    output_alphabet: str = "ACGT",
) -> list[torch.Tensor]:
    if logits.dim() != 2:
        raise ValueError("Carbon logits must have shape [windows, vocab]")
    target_base_symbols = target_base_symbols.to(device=logits.device, dtype=torch.long)
    if target_base_symbols.dim() != 2 or target_base_symbols.shape[1] != int(token_merge_size):
        raise ValueError("target_base_symbols must have shape [windows, carbon_k]")
    tables = _carbon_factorization_tables(
        tokenizer=tokenizer,
        token_merge_size=int(token_merge_size),
        output_alphabet=output_alphabet,
        device=logits.device,
    )
    steps: list[torch.Tensor] = []
    prefix_code = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
    logits_float = logits.float()
    for base_index in range(int(token_merge_size)):
        ids = tables[base_index].index_select(0, prefix_code).reshape(logits.shape[0], len(output_alphabet), -1)
        gathered = torch.gather(logits_float, 1, ids.reshape(logits.shape[0], -1)).reshape_as(ids)
        masses = torch.logsumexp(gathered, dim=2)
        steps.append(torch.softmax(masses, dim=1))
        if base_index + 1 < int(token_merge_size):
            prefix_code = prefix_code * len(output_alphabet) + target_base_symbols[:, base_index]
    return steps


def _encode_carbon_tokens_and_base_symbols(
    sequence: str,
    *,
    tokenizer: Any,
    token_merge_size: int,
    output_alphabet: str,
) -> tuple[np.ndarray, np.ndarray]:
    base_to_symbol = _base_symbol_lookup(output_alphabet)
    full_base_count = (len(sequence) // int(token_merge_size)) * int(token_merge_size)
    core = sequence[:full_base_count]
    if full_base_count == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, int(token_merge_size)), dtype=np.int64)
    if not hasattr(tokenizer, "dna_token_to_id"):
        raise ValueError("Carbon tokenizer must expose dna_token_to_id")
    dna_token_to_id = tokenizer.dna_token_to_id
    tokens: list[int] = []
    base_symbols: list[int] = []
    for start in range(0, full_base_count, int(token_merge_size)):
        token = core[start : start + int(token_merge_size)]
        try:
            tokens.append(int(dna_token_to_id[token]))
        except KeyError as exc:
            raise ValueError(f"Carbon tokenizer is missing DNA token {token!r}") from exc
        base_symbols.extend(base_to_symbol[base] for base in token)
    return (
        np.asarray(tokens, dtype=np.int64),
        np.asarray(base_symbols, dtype=np.int64).reshape(-1, int(token_merge_size)),
    )


class CarbonStreamingBatch(StreamingLMBatch):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        batch_size: int,
        begin_token_id: int,
        device: torch.device,
        dtype_name: str,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype_name = dtype_name
        self.past_key_values: Any = None
        self.pending_logits: torch.Tensor | None = None
        input_ids = torch.full((int(batch_size), 1), int(begin_token_id), dtype=torch.long, device=device)
        self._forward(input_ids)

    def _forward(self, input_ids: torch.Tensor) -> None:
        autocast_dtype = _torch_dtype_from_name(self.dtype_name)
        use_autocast = self.device.type == "cuda" and autocast_dtype in {torch.float16, torch.bfloat16}
        with torch.inference_mode():
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=use_autocast):
                output = self.model(
                    input_ids=input_ids,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        self.past_key_values = output.past_key_values
        self.pending_logits = output.logits[:, -1, :]

    def next_logits(self) -> torch.Tensor:
        if self.pending_logits is None:
            raise RuntimeError("Carbon batch has no pending logits")
        return self.pending_logits

    def accept_symbols(self, token_ids: torch.Tensor) -> None:
        token_ids = token_ids.to(device=self.device, dtype=torch.long, non_blocking=True).reshape(-1, 1)
        self._forward(token_ids)


class CarbonStreamingAdapter(StreamingLMAdapter):
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        local_path: Path,
        model_name: str,
        revision: str,
        context_bases: int,
        trust_remote_code: bool,
    ) -> None:
        self.name = "carbon"
        self.model = model
        self.tokenizer = tokenizer
        self.local_path = local_path
        self.model_name = model_name
        self.revision = revision
        self.trust_remote_code = bool(trust_remote_code)
        self.token_merge_size = int(getattr(tokenizer, "k", getattr(model, "k", 6)) or 6)
        if self.token_merge_size != 6:
            raise ValueError("first Carbon fused adapter only supports k=6")
        self.token_alphabet = "ATCG"
        self.seq_length = max(1, int(context_bases) // self.token_merge_size)
        self.pad_id = int(getattr(tokenizer, "pad_token_id", getattr(model.config, "pad_token_id", 0)) or 0)
        if not hasattr(tokenizer, "dna_begin_token_id"):
            raise ValueError("Carbon tokenizer must expose dna_begin_token_id")
        self.begin_token_id = int(tokenizer.dna_begin_token_id)

    def eval(self) -> None:
        self.model.eval()

    def build_windows(self, payload: bytes, *, window_bases: int, alphabet: str = "ACGT") -> FusedTokenWindows:
        alphabet = normalize_alphabet(alphabet)
        if int(window_bases) % self.token_merge_size != 0:
            raise ValueError("Carbon fused window_bases must be divisible by 6")
        sequence, filtered_out = _filtered_acgt(payload)
        tokens_np, token_base_symbols_np = _encode_carbon_tokens_and_base_symbols(
            sequence,
            tokenizer=self.tokenizer,
            token_merge_size=self.token_merge_size,
            output_alphabet=alphabet,
        )
        return _build_token_windows_from_arrays(
            sequence=sequence,
            filtered_out=filtered_out,
            tokens_np=tokens_np,
            token_base_symbols_np=token_base_symbols_np,
            token_merge_size=self.token_merge_size,
            window_bases=window_bases,
            pad_id=self.pad_id,
            model_uses_ascii_tokens=False,
            model_token_alphabet=self.token_alphabet,
        )

    def start_batch(self, tokens_cpu: torch.Tensor, *, device: torch.device, dtype_name: str) -> StreamingLMBatch:
        return CarbonStreamingBatch(
            model=self.model,
            batch_size=int(tokens_cpu.shape[0]),
            begin_token_id=self.begin_token_id,
            device=device,
            dtype_name=dtype_name,
        )

    def logits_to_acgt_base_probs(
        self,
        logits: torch.Tensor,
        target_base_symbols: torch.Tensor,
        *,
        output_alphabet: str,
    ) -> list[torch.Tensor]:
        return _carbon_conditional_log_probs_to_base_steps(
            logits,
            target_base_symbols,
            tokenizer=self.tokenizer,
            token_merge_size=self.token_merge_size,
            output_alphabet=output_alphabet,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "lm_backend": self.name,
            "carbon_model_name": self.model_name,
            "carbon_revision": self.revision,
            "carbon_local_path": str(self.local_path),
            "carbon_k": int(self.token_merge_size),
            "carbon_context_tokens": int(self.seq_length),
            "carbon_context_bases": int(self.seq_length * self.token_merge_size),
            "carbon_factorization": "conditional_acgt_kmer_joint",
            "carbon_token_alphabet": self.token_alphabet,
            "carbon_trust_remote_code": bool(self.trust_remote_code),
        }


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
    nc_prefix_geco2_level: int = 10,
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


def _compress_fused_streaming_token_nc_full_batch_payload(
    *,
    windows: FusedTokenWindows,
    lm_adapter: StreamingLMAdapter,
    device: torch.device,
    dtype_name: str,
    lm_batch_size: int,
    window_bases: int,
    nc_prefix_hash_bucket_count: int,
    nc_prefix_geco2_level: int,
    fusion_eta: float,
    fusion_initial_lm_weight: float,
    frequency_total: int,
    arithmetic_metadata: dict[str, Any],
    arithmetic_target_uniform_mass: float,
    encode_arithmetic: bool,
    collect_diagnostics: bool,
    include_codec_baselines: bool,
    process_started: float,
    payload: bytes,
    alphabet: str,
) -> dict[str, Any]:
    tokens_cpu = windows.tokens
    base_symbols_cpu = windows.token_base_symbols
    valid_lengths_cpu = windows.valid_token_lengths
    window_count = int(tokens_cpu.shape[0])
    tokens_per_window = int(tokens_cpu.shape[1])
    token_merge_size = int(windows.token_merge_size)
    lm_seq_length = int(lm_adapter.seq_length)
    resolved_lm_batch_size = min(window_count, max(1, int(lm_batch_size)))

    _sync(device)
    encode_started = perf_counter()
    model_seconds = 0.0
    lm_factorize_seconds = 0.0
    lm_transfer_seconds = 0.0
    lm_batch_summaries: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="dna_fused_lm_probs_") as tmpdir:
        probs_path = Path(tmpdir) / "lm_probs.float32"
        lm_probs = np.memmap(
            probs_path,
            dtype=np.float32,
            mode="w+",
            shape=(tokens_per_window, window_count, token_merge_size, 4),
        )
        lm_precompute_started = perf_counter()
        for chunk_start in range(0, window_count, resolved_lm_batch_size):
            chunk_end = min(window_count, chunk_start + resolved_lm_batch_size)
            chunk_tokens_cpu = tokens_cpu[chunk_start:chunk_end].contiguous()
            chunk_base_symbols = base_symbols_cpu[chunk_start:chunk_end].contiguous()
            chunk_lengths_cpu = valid_lengths_cpu[chunk_start:chunk_end].contiguous()
            chunk_window_count = int(chunk_tokens_cpu.shape[0])
            stepper = lm_adapter.start_batch(chunk_tokens_cpu, device=device, dtype_name=dtype_name)
            chunk_model_seconds = 0.0
            chunk_factorize_seconds = 0.0
            chunk_transfer_seconds = 0.0
            for token_step in range(tokens_per_window):
                active_token_mask = chunk_lengths_cpu > token_step
                model_started = perf_counter()
                logits = stepper.next_logits()
                _sync(device)
                step_model_seconds = perf_counter() - model_started
                model_seconds += step_model_seconds
                chunk_model_seconds += step_model_seconds

                factor_started = perf_counter()
                safe_target_base_symbols = torch.where(
                    active_token_mask[:, None],
                    chunk_base_symbols[:, token_step, :],
                    torch.zeros_like(chunk_base_symbols[:, token_step, :]),
                ).to(device=device)
                base_probability_steps = lm_adapter.logits_to_acgt_base_probs(
                    logits,
                    safe_target_base_symbols,
                    output_alphabet=alphabet,
                )
                _sync(device)
                step_factor_seconds = perf_counter() - factor_started
                lm_factorize_seconds += step_factor_seconds
                chunk_factorize_seconds += step_factor_seconds

                transfer_started = perf_counter()
                lm_probs_gpu = torch.stack(base_probability_steps, dim=1).detach().float()
                lm_probs[token_step, chunk_start:chunk_end, :, :] = lm_probs_gpu.cpu().numpy()
                step_transfer_seconds = perf_counter() - transfer_started
                lm_transfer_seconds += step_transfer_seconds
                chunk_transfer_seconds += step_transfer_seconds

                stepper.accept_symbols(chunk_tokens_cpu[:, token_step])

            lm_batch_summaries.append(
                {
                    "chunk_index": len(lm_batch_summaries),
                    "window_start": int(chunk_start),
                    "window_end": int(chunk_end),
                    "window_count": int(chunk_window_count),
                    "model_seconds": float(chunk_model_seconds),
                    "lm_factorize_seconds": float(chunk_factorize_seconds),
                    "lm_probability_transfer_seconds": float(chunk_transfer_seconds),
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        lm_probs.flush()
        lm_precompute_seconds = perf_counter() - lm_precompute_started

        native_encoder = FusedNcPrefixStreamingEncoder(
            window_count=window_count,
            window_bases=window_bases,
            hash_bucket_count=int(nc_prefix_hash_bucket_count),
            geco2_level=int(nc_prefix_geco2_level),
            arithmetic_frequency_total=int(frequency_total),
            fusion_eta=float(fusion_eta),
            initial_lm_weight=float(fusion_initial_lm_weight),
            encode_arithmetic=bool(encode_arithmetic),
            collect_diagnostics=bool(collect_diagnostics),
        )
        native_started = perf_counter()
        token_jobs = 0
        for token_step in range(tokens_per_window):
            active_count = int((valid_lengths_cpu > token_step).sum().item())
            if active_count <= 0:
                continue
            lm_probs_cpu = torch.from_numpy(np.asarray(lm_probs[token_step, :active_count, :, :])).contiguous()
            targets = base_symbols_cpu[:active_count, token_step, :].to(torch.int16).contiguous()
            native_encoder.encode_token_step(lm_probs_cpu, targets)
            token_jobs += 1
        native_encode_seconds_observed = perf_counter() - native_started
        native_result = native_encoder.finish()

    _sync(device)
    encode_wall_seconds = perf_counter() - encode_started
    tail_bits = _tail_side_info_bits(windows.tail_sequence)
    chunk_streams = list(native_result.get("streams", []))
    encoded_bytes = sum(len(stream) for stream in chunk_streams)
    arithmetic_bytes = encoded_bytes + ((tail_bits + 7) // 8 if encode_arithmetic else 0)
    emitted_symbols = int(native_result["emitted_arithmetic_symbol_count"])
    core_bases = len(windows.core_sequence)
    sample_bases = len(windows.sequence)
    elapsed = perf_counter() - process_started
    diagnostics_collected = bool(native_result.get("diagnostics_collected", True))
    fused_bits = float(native_result["fused_theoretical_bits"]) if diagnostics_collected else None
    lm_bits = float(native_result["lm_only_theoretical_bits"]) if diagnostics_collected else None
    nc_bits = float(native_result["nc_prefix_only_theoretical_bits"]) if diagnostics_collected else None
    chunk_metadata = dict(native_result.get("model_metadata") or {})
    chunk_summary = {
        "chunk_index": 0,
        "window_start": 0,
        "window_end": int(window_count),
        "window_count": int(window_count),
        "core_bases": int(core_bases),
        "arithmetic_coded_bytes": int(encoded_bytes) if encode_arithmetic else None,
        "emitted_arithmetic_symbol_count": int(emitted_symbols),
        "fusion_final_mean_lm_weight": float(native_result["fusion_final_mean_lm_weight"]),
        "diagnostics_collected": bool(diagnostics_collected),
        "core_theoretical_bits_per_base": float(fused_bits) / max(core_bases, 1) if diagnostics_collected else None,
        "lm_only_theoretical_bits_per_base": float(lm_bits) / max(core_bases, 1) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits_per_base": float(nc_bits) / max(core_bases, 1) if diagnostics_collected else None,
        "model_seconds": float(model_seconds),
        "lm_factorize_seconds": float(lm_factorize_seconds),
        "lm_probability_transfer_seconds": float(lm_transfer_seconds),
        "native_fused_encode_seconds_observed": float(native_encode_seconds_observed),
        "native_fused_encode_seconds": float(native_result["encode_seconds"]),
        "native_finish_seconds": float(native_result["finish_seconds"]),
        "streaming_async_jobs": int(token_jobs),
        "streaming_gpu_queue_wait_seconds": 0.0,
        "nc_prefix_window_count": int(chunk_metadata.get("window_count", window_count)),
        "nc_prefix_hash_bucket_count": chunk_metadata.get("hash_bucket_count"),
        "nc_prefix_geco2_level": int(chunk_metadata.get("geco2_level", nc_prefix_geco2_level)),
        "nc_prefix_pipeline_block_windows": chunk_metadata.get("pipeline_block_windows"),
    }
    metrics: dict[str, Any] = {
        "codec": "fused_lm_nc_prefix",
        "lm_backend": lm_adapter.name,
        "pipeline_mode": "streaming_token_nc_full_batch",
        "decodable_design": "lm_micro_batch_precompute_nc_prefix_full_window_batch_native_ordered_commit",
        "decoder_realistic": False,
        "encoder_overlap_enabled": False,
        "encode_arithmetic": bool(encode_arithmetic),
        "alphabet": alphabet,
        "sample_bases": int(sample_bases),
        "core_base_count": int(core_bases),
        "tail_base_count": int(len(windows.tail_sequence)),
        "tail_side_info_bits": int(tail_bits),
        "filtered_out_bases": int(windows.filtered_out_bases),
        "window_count": int(window_count),
        "batch_count": 1,
        "batch_size": int(resolved_lm_batch_size),
        "batch_window_counts": [int(window_count)],
        "lm_batch_count": int(len(lm_batch_summaries)),
        "lm_batch_size": int(resolved_lm_batch_size),
        "lm_batch_window_counts": [int(item["window_count"]) for item in lm_batch_summaries],
        "token_merge_size": int(token_merge_size),
        "tokens_per_window": int(tokens_per_window),
        "window_bases": int(window_bases),
        "lm_seq_length": int(lm_seq_length),
        "model_uses_ascii_tokens": bool(windows.model_uses_ascii_tokens),
        "model_token_alphabet": windows.model_token_alphabet,
        "lm_metadata": lm_adapter.metadata(),
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
        "core_theoretical_bits_per_base": (float(fused_bits) / max(core_bases, 1)) if diagnostics_collected else None,
        "lm_only_theoretical_bits": float(lm_bits) if diagnostics_collected else None,
        "lm_only_theoretical_bits_per_base": (float(lm_bits) / max(core_bases, 1)) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits": float(nc_bits) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits_per_base": (float(nc_bits) / max(core_bases, 1)) if diagnostics_collected else None,
        "arithmetic_coded_bytes": int(arithmetic_bytes) if encode_arithmetic else None,
        "arithmetic_bits_per_base": (float(arithmetic_bytes) * 8.0 / max(sample_bases, 1)) if encode_arithmetic else None,
        "arithmetic_stream_count": int(len(chunk_streams)),
        "emitted_arithmetic_symbol_count": int(emitted_symbols),
        "compression_process_seconds": float(elapsed),
        "compression_core_seconds": float(encode_wall_seconds),
        "compression_bases_per_second": float(sample_bases) / max(elapsed, 1e-12),
        "model_seconds": float(model_seconds),
        "lm_factorize_seconds": float(lm_factorize_seconds),
        "lm_probability_transfer_seconds": float(lm_transfer_seconds),
        "lm_precompute_seconds": float(lm_precompute_seconds),
        "native_fused_encode_seconds_observed": float(native_encode_seconds_observed),
        "native_fused_encode_seconds": float(native_result["encode_seconds"]),
        "native_finish_seconds": float(native_result["finish_seconds"]),
        "streaming_async_enabled": False,
        "streaming_async_jobs": int(token_jobs),
        "streaming_cpu_wait_for_gpu_seconds": 0.0,
        "streaming_gpu_queue_wait_seconds": 0.0,
        "streaming_ring_buffer_depth_tokens": 0,
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
        "nc_prefix_backend": "streaming_token_native_full_window_batch",
        "nc_prefix_metadata": {
            "batch_scope": "full_sequence_window_batch_single_nc_prefix_state",
            "batch_count": 1,
            "batch_size": int(window_count),
            "tail_batch_window_count": int(window_count),
            "lm_micro_batch_count": int(len(lm_batch_summaries)),
            "lm_micro_batch_size": int(resolved_lm_batch_size),
            "hash_bucket_count_config": int(nc_prefix_hash_bucket_count),
            "geco2_level_config": int(nc_prefix_geco2_level),
            "chunk_summaries": [chunk_summary],
            "lm_batch_summaries": lm_batch_summaries,
        },
        **arithmetic_metadata,
        **memory_stats(device, prefix="compression_"),
        **baseline_sizes(payload, include_codec_baselines=include_codec_baselines),
    }
    return metrics


def _compress_fused_streaming_token_nc_full_batch_overlap_payload(
    *,
    windows: FusedTokenWindows,
    lm_adapter: StreamingLMAdapter,
    device: torch.device,
    dtype_name: str,
    lm_batch_size: int,
    window_bases: int,
    nc_prefix_hash_bucket_count: int,
    nc_prefix_geco2_level: int,
    fusion_eta: float,
    fusion_initial_lm_weight: float,
    frequency_total: int,
    arithmetic_metadata: dict[str, Any],
    arithmetic_target_uniform_mass: float,
    encode_arithmetic: bool,
    collect_diagnostics: bool,
    include_codec_baselines: bool,
    process_started: float,
    payload: bytes,
    alphabet: str,
) -> dict[str, Any]:
    tokens_cpu = windows.tokens
    base_symbols_cpu = windows.token_base_symbols
    valid_lengths_cpu = windows.valid_token_lengths
    window_count = int(tokens_cpu.shape[0])
    tokens_per_window = int(tokens_cpu.shape[1])
    token_merge_size = int(windows.token_merge_size)
    lm_seq_length = int(lm_adapter.seq_length)
    resolved_lm_batch_size = min(window_count, max(1, int(lm_batch_size)))
    planned_lm_batch_count = int(math.ceil(window_count / max(resolved_lm_batch_size, 1)))
    offload_lm_state = device.type == "cuda" and planned_lm_batch_count > 2
    cpu_device = torch.device("cpu")

    _sync(device)
    encode_started = perf_counter()
    model_seconds = 0.0
    lm_factorize_seconds = 0.0
    lm_transfer_seconds = 0.0
    lm_state_reload_seconds = 0.0
    lm_state_offload_seconds = 0.0
    nc_queue_wait_seconds = 0.0
    lm_batch_summaries: list[dict[str, Any]] = []
    lm_batches: list[dict[str, Any]] = []

    for chunk_start in range(0, window_count, resolved_lm_batch_size):
        chunk_end = min(window_count, chunk_start + resolved_lm_batch_size)
        batch = lm_adapter.start_batch(tokens_cpu[chunk_start:chunk_end].contiguous(), device=device, dtype_name=dtype_name)
        if offload_lm_state:
            offload_started = perf_counter()
            batch.move_state_to(cpu_device)
            _sync(device)
            lm_state_offload_seconds += perf_counter() - offload_started
            torch.cuda.empty_cache()
        lm_batches.append(
            {
                "chunk_index": len(lm_batches),
                "window_start": int(chunk_start),
                "window_end": int(chunk_end),
                "tokens_cpu": tokens_cpu[chunk_start:chunk_end].contiguous(),
                "base_symbols": base_symbols_cpu[chunk_start:chunk_end].contiguous(),
                "lengths_cpu": valid_lengths_cpu[chunk_start:chunk_end].contiguous(),
                "batch": batch,
                "model_seconds": 0.0,
                "lm_factorize_seconds": 0.0,
                "lm_probability_transfer_seconds": 0.0,
                "state_reload_seconds": 0.0,
                "state_offload_seconds": 0.0,
            }
        )

    native_encoder = FusedNcPrefixStreamingEncoder(
        window_count=window_count,
        window_bases=window_bases,
        hash_bucket_count=int(nc_prefix_hash_bucket_count),
        geco2_level=int(nc_prefix_geco2_level),
        arithmetic_frequency_total=int(frequency_total),
        fusion_eta=float(fusion_eta),
        initial_lm_weight=float(fusion_initial_lm_weight),
        encode_arithmetic=bool(encode_arithmetic),
        collect_diagnostics=bool(collect_diagnostics),
    )
    worker_error: list[BaseException] = []
    worker_stats = {"native_seconds": 0.0, "jobs": 0}
    work_queue: Queue[Any] = Queue(maxsize=2)

    def _nc_full_batch_worker() -> None:
        try:
            while True:
                item = work_queue.get()
                try:
                    if item is None:
                        return
                    lm_probs_np, targets = item
                    native_started_inner = perf_counter()
                    lm_probs_cpu = torch.from_numpy(lm_probs_np).contiguous()
                    native_encoder.encode_token_step(lm_probs_cpu, targets)
                    worker_stats["native_seconds"] += perf_counter() - native_started_inner
                    worker_stats["jobs"] += 1
                finally:
                    work_queue.task_done()
        except BaseException as error:  # pragma: no cover - exercised by integration failures
            worker_error.append(error)

    worker = threading.Thread(
        target=_nc_full_batch_worker,
        name="fused-nc-prefix-full-window-batch-worker",
        daemon=True,
    )
    worker.start()

    try:
        for token_step in range(tokens_per_window):
            active_count = int((valid_lengths_cpu > token_step).sum().item())
            if active_count <= 0:
                continue
            step_probs = np.empty((active_count, token_merge_size, 4), dtype=np.float32)
            for batch_info in lm_batches:
                chunk_start = int(batch_info["window_start"])
                chunk_end = int(batch_info["window_end"])
                active_end = min(chunk_end, active_count)
                if active_end <= chunk_start:
                    continue
                chunk_tokens_cpu = batch_info["tokens_cpu"]
                chunk_base_symbols = batch_info["base_symbols"]
                chunk_lengths_cpu = batch_info["lengths_cpu"]
                batch = batch_info["batch"]
                if offload_lm_state:
                    reload_started = perf_counter()
                    batch.move_state_to(device)
                    _sync(device)
                    reload_seconds = perf_counter() - reload_started
                    lm_state_reload_seconds += reload_seconds
                    batch_info["state_reload_seconds"] += reload_seconds

                active_token_mask = chunk_lengths_cpu > token_step
                model_started = perf_counter()
                logits = batch.next_logits()
                _sync(device)
                step_model_seconds = perf_counter() - model_started
                model_seconds += step_model_seconds
                batch_info["model_seconds"] += step_model_seconds

                factor_started = perf_counter()
                safe_target_base_symbols = torch.where(
                    active_token_mask[:, None],
                    chunk_base_symbols[:, token_step, :],
                    torch.zeros_like(chunk_base_symbols[:, token_step, :]),
                ).to(device=device)
                base_probability_steps = lm_adapter.logits_to_acgt_base_probs(
                    logits,
                    safe_target_base_symbols,
                    output_alphabet=alphabet,
                )
                _sync(device)
                step_factor_seconds = perf_counter() - factor_started
                lm_factorize_seconds += step_factor_seconds
                batch_info["lm_factorize_seconds"] += step_factor_seconds

                transfer_started = perf_counter()
                lm_probs_gpu = torch.stack(base_probability_steps, dim=1).detach().float()
                step_probs[chunk_start:active_end, :, :] = lm_probs_gpu[: active_end - chunk_start].cpu().numpy()
                step_transfer_seconds = perf_counter() - transfer_started
                lm_transfer_seconds += step_transfer_seconds
                batch_info["lm_probability_transfer_seconds"] += step_transfer_seconds

                batch.accept_symbols(chunk_tokens_cpu[:, token_step])
                if offload_lm_state:
                    offload_started = perf_counter()
                    batch.move_state_to(cpu_device)
                    _sync(device)
                    offload_seconds = perf_counter() - offload_started
                    lm_state_offload_seconds += offload_seconds
                    batch_info["state_offload_seconds"] += offload_seconds
                    torch.cuda.empty_cache()

            targets = base_symbols_cpu[:active_count, token_step, :].to(torch.int16).contiguous()
            queue_started = perf_counter()
            work_queue.put((step_probs, targets))
            nc_queue_wait_seconds += perf_counter() - queue_started
            if worker_error:
                raise RuntimeError("fused nc full-batch worker failed") from worker_error[0]

        work_queue.join()
        work_queue.put(None)
        work_queue.join()
        worker.join()
        if worker_error:
            raise RuntimeError("fused nc full-batch worker failed") from worker_error[0]
        native_result = native_encoder.finish()
    finally:
        if worker.is_alive():
            try:
                work_queue.put_nowait(None)
            except Exception:
                pass

    for batch_info in lm_batches:
        lm_batch_summaries.append(
            {
                "chunk_index": int(batch_info["chunk_index"]),
                "window_start": int(batch_info["window_start"]),
                "window_end": int(batch_info["window_end"]),
                "window_count": int(batch_info["window_end"]) - int(batch_info["window_start"]),
                "model_seconds": float(batch_info["model_seconds"]),
                "lm_factorize_seconds": float(batch_info["lm_factorize_seconds"]),
                "lm_probability_transfer_seconds": float(batch_info["lm_probability_transfer_seconds"]),
                "state_reload_seconds": float(batch_info["state_reload_seconds"]),
                "state_offload_seconds": float(batch_info["state_offload_seconds"]),
            }
        )

    _sync(device)
    encode_wall_seconds = perf_counter() - encode_started
    tail_bits = _tail_side_info_bits(windows.tail_sequence)
    chunk_streams = list(native_result.get("streams", []))
    encoded_bytes = sum(len(stream) for stream in chunk_streams)
    arithmetic_bytes = encoded_bytes + ((tail_bits + 7) // 8 if encode_arithmetic else 0)
    emitted_symbols = int(native_result["emitted_arithmetic_symbol_count"])
    core_bases = len(windows.core_sequence)
    sample_bases = len(windows.sequence)
    elapsed = perf_counter() - process_started
    diagnostics_collected = bool(native_result.get("diagnostics_collected", True))
    fused_bits = float(native_result["fused_theoretical_bits"]) if diagnostics_collected else None
    lm_bits = float(native_result["lm_only_theoretical_bits"]) if diagnostics_collected else None
    nc_bits = float(native_result["nc_prefix_only_theoretical_bits"]) if diagnostics_collected else None
    chunk_metadata = dict(native_result.get("model_metadata") or {})
    chunk_summary = {
        "chunk_index": 0,
        "window_start": 0,
        "window_end": int(window_count),
        "window_count": int(window_count),
        "core_bases": int(core_bases),
        "arithmetic_coded_bytes": int(encoded_bytes) if encode_arithmetic else None,
        "emitted_arithmetic_symbol_count": int(emitted_symbols),
        "fusion_final_mean_lm_weight": float(native_result["fusion_final_mean_lm_weight"]),
        "diagnostics_collected": bool(diagnostics_collected),
        "core_theoretical_bits_per_base": float(fused_bits) / max(core_bases, 1) if diagnostics_collected else None,
        "lm_only_theoretical_bits_per_base": float(lm_bits) / max(core_bases, 1) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits_per_base": float(nc_bits) / max(core_bases, 1) if diagnostics_collected else None,
        "model_seconds": float(model_seconds),
        "lm_factorize_seconds": float(lm_factorize_seconds),
        "lm_probability_transfer_seconds": float(lm_transfer_seconds),
        "native_fused_encode_seconds_observed": float(worker_stats["native_seconds"]),
        "native_fused_encode_seconds": float(native_result["encode_seconds"]),
        "native_finish_seconds": float(native_result["finish_seconds"]),
        "streaming_async_jobs": int(worker_stats["jobs"]),
        "streaming_gpu_queue_wait_seconds": float(nc_queue_wait_seconds),
        "nc_prefix_window_count": int(chunk_metadata.get("window_count", window_count)),
        "nc_prefix_hash_bucket_count": chunk_metadata.get("hash_bucket_count"),
        "nc_prefix_geco2_level": int(chunk_metadata.get("geco2_level", nc_prefix_geco2_level)),
        "nc_prefix_pipeline_block_windows": chunk_metadata.get("pipeline_block_windows"),
    }
    metrics: dict[str, Any] = {
        "codec": "fused_lm_nc_prefix",
        "lm_backend": lm_adapter.name,
        "pipeline_mode": "streaming_token_nc_full_batch",
        "decodable_design": (
            "lm_micro_batch_cpu_offload_nc_prefix_full_window_batch_overlap_native_ordered_commit"
            if offload_lm_state
            else "lm_micro_batch_gpu_resident_nc_prefix_full_window_batch_overlap_native_ordered_commit"
        ),
        "decoder_realistic": False,
        "encoder_overlap_enabled": True,
        "encode_arithmetic": bool(encode_arithmetic),
        "alphabet": alphabet,
        "sample_bases": int(sample_bases),
        "core_base_count": int(core_bases),
        "tail_base_count": int(len(windows.tail_sequence)),
        "tail_side_info_bits": int(tail_bits),
        "filtered_out_bases": int(windows.filtered_out_bases),
        "window_count": int(window_count),
        "batch_count": 1,
        "batch_size": int(resolved_lm_batch_size),
        "batch_window_counts": [int(window_count)],
        "lm_batch_count": int(len(lm_batch_summaries)),
        "lm_batch_size": int(resolved_lm_batch_size),
        "lm_batch_window_counts": [int(item["window_count"]) for item in lm_batch_summaries],
        "token_merge_size": int(token_merge_size),
        "tokens_per_window": int(tokens_per_window),
        "window_bases": int(window_bases),
        "lm_seq_length": int(lm_seq_length),
        "model_uses_ascii_tokens": bool(windows.model_uses_ascii_tokens),
        "model_token_alphabet": windows.model_token_alphabet,
        "lm_metadata": lm_adapter.metadata(),
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
        "core_theoretical_bits_per_base": (float(fused_bits) / max(core_bases, 1)) if diagnostics_collected else None,
        "lm_only_theoretical_bits": float(lm_bits) if diagnostics_collected else None,
        "lm_only_theoretical_bits_per_base": (float(lm_bits) / max(core_bases, 1)) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits": float(nc_bits) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits_per_base": (float(nc_bits) / max(core_bases, 1)) if diagnostics_collected else None,
        "arithmetic_coded_bytes": int(arithmetic_bytes) if encode_arithmetic else None,
        "arithmetic_bits_per_base": (float(arithmetic_bytes) * 8.0 / max(sample_bases, 1)) if encode_arithmetic else None,
        "arithmetic_stream_count": int(len(chunk_streams)),
        "emitted_arithmetic_symbol_count": int(emitted_symbols),
        "compression_process_seconds": float(elapsed),
        "compression_core_seconds": float(encode_wall_seconds),
        "compression_bases_per_second": float(sample_bases) / max(elapsed, 1e-12),
        "model_seconds": float(model_seconds),
        "lm_factorize_seconds": float(lm_factorize_seconds),
        "lm_probability_transfer_seconds": float(lm_transfer_seconds),
        "lm_state_reload_seconds": float(lm_state_reload_seconds),
        "lm_state_offload_seconds": float(lm_state_offload_seconds),
        "lm_precompute_seconds": None,
        "native_fused_encode_seconds_observed": float(worker_stats["native_seconds"]),
        "native_fused_encode_seconds": float(native_result["encode_seconds"]),
        "native_finish_seconds": float(native_result["finish_seconds"]),
        "streaming_async_enabled": True,
        "streaming_async_jobs": int(worker_stats["jobs"]),
        "streaming_cpu_wait_for_gpu_seconds": 0.0,
        "streaming_gpu_queue_wait_seconds": float(nc_queue_wait_seconds),
        "streaming_ring_buffer_depth_tokens": 2,
        "nc_predict_wait_seconds": None,
        "lm_wait_seconds": None,
        "fusion_update_seconds": None,
        "nc_predict_seconds": None,
        "pipeline_depth_lag_max": 1,
        "nc_prefix_prepare_seconds": 0.0,
        "nc_prefix_predict_seconds": None,
        "fusion_seconds": None,
        "arithmetic_quantize_seconds": None,
        "arithmetic_range_seconds": None,
        "nc_prefix_backend": "streaming_token_native_full_window_batch",
        "nc_prefix_metadata": {
            "batch_scope": "full_sequence_window_batch_single_nc_prefix_state",
            "overlap_mode": "lm_micro_batch_cpu_offload_producer_nc_prefix_worker_consumer"
            if offload_lm_state
            else "lm_micro_batch_gpu_resident_producer_nc_prefix_worker_consumer",
            "lm_state_offload_enabled": bool(offload_lm_state),
            "batch_count": 1,
            "batch_size": int(window_count),
            "tail_batch_window_count": int(window_count),
            "lm_micro_batch_count": int(len(lm_batch_summaries)),
            "lm_micro_batch_size": int(resolved_lm_batch_size),
            "hash_bucket_count_config": int(nc_prefix_hash_bucket_count),
            "geco2_level_config": int(nc_prefix_geco2_level),
            "chunk_summaries": [chunk_summary],
            "lm_batch_summaries": lm_batch_summaries,
        },
        **arithmetic_metadata,
        **memory_stats(device, prefix="compression_"),
        **baseline_sizes(payload, include_codec_baselines=include_codec_baselines),
    }
    return metrics


def _compress_fused_streaming_token_payload(
    *,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: int | str,
    model: torch.nn.Module | None,
    config: ExperimentConfig | None,
    lm_adapter: StreamingLMAdapter | None = None,
    nc_prefix_window_bases: int | None = None,
    nc_prefix_min_windows: int = 8192,
    nc_prefix_hash_bucket_count: int = 0,
    nc_prefix_geco2_level: int = 10,
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
    if pipeline_mode not in {
        "streaming_token_encode_overlap",
        "streaming_token_strict",
        "streaming_token_nc_full_batch",
    }:
        raise ValueError(
            "pipeline_mode must be 'streaming_token_encode_overlap', 'streaming_token_strict', "
            "or 'streaming_token_nc_full_batch'"
        )
    if int(nc_prefix_geco2_level) < 1 or int(nc_prefix_geco2_level) > 12:
        raise ValueError("nc_prefix_geco2_level must be in [1, 12]")
    alphabet = normalize_alphabet("ACGT")
    if lm_adapter is None:
        if model is None or config is None:
            raise ValueError("model and config are required when lm_adapter is not provided")
        lm_adapter = MegabyteStreamingAdapter(model=model, config=config)
    token_merge_size = int(lm_adapter.token_merge_size)
    lm_seq_length = int(lm_adapter.seq_length)
    window_bases = int(nc_prefix_window_bases or (lm_seq_length * token_merge_size))
    if window_bases // max(token_merge_size, 1) != lm_seq_length:
        raise ValueError(
            f"{pipeline_mode} requires nc_prefix_window_bases to match the LM decode window: "
            f"window_bases/token_merge_size={window_bases // max(token_merge_size, 1)}, "
            f"lm_seq_length={lm_seq_length}"
        )

    process_started = perf_counter()
    windows = lm_adapter.build_windows(payload, window_bases=window_bases, alphabet=alphabet)
    tokens_cpu = windows.tokens
    base_symbols_cpu = windows.token_base_symbols
    valid_lengths_cpu = windows.valid_token_lengths
    window_count = int(tokens_cpu.shape[0])
    tokens_per_window = int(tokens_cpu.shape[1])
    batch_size_name = str(batch_size).lower() if isinstance(batch_size, str) else None
    if batch_size_name in {"all", "full", "sequence"}:
        resolved_batch_size = window_count
    elif batch_size_name == "auto" or int(batch_size) <= 0:
        resolved_batch_size = min(window_count, max(1, int(nc_prefix_min_windows)))
    else:
        resolved_batch_size = int(batch_size)
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=4,
        requested_total=arithmetic_frequency_total,
        target_uniform_mass=arithmetic_target_uniform_mass,
    )
    frequency_total = int(arithmetic_metadata["arithmetic_frequency_total"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    lm_adapter.eval()
    if pipeline_mode == "streaming_token_nc_full_batch":
        return _compress_fused_streaming_token_nc_full_batch_overlap_payload(
            windows=windows,
            lm_adapter=lm_adapter,
            device=device,
            dtype_name=dtype_name,
            lm_batch_size=resolved_batch_size,
            window_bases=window_bases,
            nc_prefix_hash_bucket_count=nc_prefix_hash_bucket_count,
            nc_prefix_geco2_level=nc_prefix_geco2_level,
            fusion_eta=fusion_eta,
            fusion_initial_lm_weight=fusion_initial_lm_weight,
            frequency_total=frequency_total,
            arithmetic_metadata=arithmetic_metadata,
            arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
            encode_arithmetic=encode_arithmetic,
            collect_diagnostics=collect_diagnostics,
            include_codec_baselines=include_codec_baselines,
            process_started=process_started,
            payload=payload,
            alphabet=alphabet,
        )
    _sync(device)
    encode_started = perf_counter()
    model_seconds = 0.0
    lm_factorize_seconds = 0.0
    lm_transfer_seconds = 0.0
    native_encode_seconds_observed = 0.0
    cpu_wait_for_gpu_seconds = 0.0
    gpu_queue_wait_seconds = 0.0
    token_jobs = 0
    use_encode_overlap = pipeline_mode == "streaming_token_encode_overlap" and device.type == "cuda"
    encoded_streams: list[bytes] = []
    emitted_symbols = 0
    native_encode_seconds = 0.0
    native_finish_seconds = 0.0
    fused_bits_total = 0.0
    lm_bits_total = 0.0
    nc_bits_total = 0.0
    diagnostics_collected = bool(collect_diagnostics)
    weighted_lm_weight_sum = 0.0
    weighted_lm_weight_count = 0
    chunk_summaries: list[dict[str, Any]] = []

    for chunk_start in range(0, window_count, resolved_batch_size):
        chunk_end = min(window_count, chunk_start + resolved_batch_size)
        chunk_tokens_cpu = tokens_cpu[chunk_start:chunk_end].contiguous()
        chunk_base_symbols = base_symbols_cpu[chunk_start:chunk_end].contiguous()
        chunk_lengths_cpu = valid_lengths_cpu[chunk_start:chunk_end].contiguous()
        chunk_window_count = int(chunk_tokens_cpu.shape[0])
        chunk_core_bases = int(chunk_lengths_cpu.sum().item()) * int(token_merge_size)
        native_encoder = FusedNcPrefixStreamingEncoder(
            window_count=chunk_window_count,
            window_bases=window_bases,
            hash_bucket_count=int(nc_prefix_hash_bucket_count),
            geco2_level=int(nc_prefix_geco2_level),
            arithmetic_frequency_total=frequency_total,
            fusion_eta=float(fusion_eta),
            initial_lm_weight=float(fusion_initial_lm_weight),
            encode_arithmetic=bool(encode_arithmetic),
            collect_diagnostics=bool(collect_diagnostics),
        )
        chunk_tokens = chunk_tokens_cpu.to(device, non_blocking=True)
        stepper = lm_adapter.start_batch(chunk_tokens_cpu, device=device, dtype_name=dtype_name)
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

            worker = threading.Thread(
                target=_token_cpu_worker,
                name=f"fused-nc-prefix-token-overlap-worker-{chunk_start}",
                daemon=True,
            )
            worker.start()

        chunk_model_seconds = 0.0
        chunk_factorize_seconds = 0.0
        chunk_transfer_seconds = 0.0
        chunk_gpu_queue_wait_seconds = 0.0
        chunk_native_observed_seconds = 0.0
        chunk_token_jobs = 0
        for token_step in range(tokens_per_window):
            active_token_mask = chunk_lengths_cpu > token_step
            active_count = int(active_token_mask.sum().item())
            model_started = perf_counter()
            logits = stepper.next_logits()
            _sync(device)
            step_model_seconds = perf_counter() - model_started
            model_seconds += step_model_seconds
            chunk_model_seconds += step_model_seconds

            factor_started = perf_counter()
            safe_target_base_symbols = torch.where(
                active_token_mask[:, None],
                chunk_base_symbols[:, token_step, :],
                torch.zeros_like(chunk_base_symbols[:, token_step, :]),
            ).to(device=device)
            base_probability_steps = lm_adapter.logits_to_acgt_base_probs(
                logits,
                safe_target_base_symbols,
                output_alphabet=alphabet,
            )
            _sync(device)
            step_factor_seconds = perf_counter() - factor_started
            lm_factorize_seconds += step_factor_seconds
            chunk_factorize_seconds += step_factor_seconds

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
            step_transfer_seconds = perf_counter() - transfer_started
            lm_transfer_seconds += step_transfer_seconds
            chunk_transfer_seconds += step_transfer_seconds

            targets = chunk_base_symbols[:active_count, token_step, :].to(torch.int16).contiguous()
            if use_encode_overlap:
                assert work_queue is not None and event is not None
                queue_started = perf_counter()
                work_queue.put((event, lm_probs_cpu, targets))
                queue_wait = perf_counter() - queue_started
                gpu_queue_wait_seconds += queue_wait
                chunk_gpu_queue_wait_seconds += queue_wait
            else:
                native_started = perf_counter()
                native_encoder.encode_token_step(lm_probs_cpu, targets)
                step_native_seconds = perf_counter() - native_started
                native_encode_seconds_observed += step_native_seconds
                chunk_native_observed_seconds += step_native_seconds
                token_jobs += 1
                chunk_token_jobs += 1

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
            native_encode_seconds_observed += float(worker_stats["native_seconds"])
            chunk_native_observed_seconds = float(worker_stats["native_seconds"])
            cpu_wait_for_gpu_seconds += float(worker_stats["wait_seconds"])
            token_jobs += int(worker_stats["jobs"])
            chunk_token_jobs = int(worker_stats["jobs"])

        native_result = native_encoder.finish()
        chunk_streams = list(native_result.get("streams", []))
        encoded_streams.extend(chunk_streams)
        emitted_symbols += int(native_result["emitted_arithmetic_symbol_count"])
        native_encode_seconds += float(native_result["encode_seconds"])
        native_finish_seconds += float(native_result["finish_seconds"])
        chunk_diagnostics = bool(native_result.get("diagnostics_collected", True))
        diagnostics_collected = diagnostics_collected and chunk_diagnostics
        if chunk_diagnostics:
            fused_bits_total += float(native_result["fused_theoretical_bits"])
            lm_bits_total += float(native_result["lm_only_theoretical_bits"])
            nc_bits_total += float(native_result["nc_prefix_only_theoretical_bits"])
        weighted_lm_weight_sum += float(native_result["fusion_final_mean_lm_weight"]) * chunk_window_count
        weighted_lm_weight_count += chunk_window_count
        chunk_metadata = dict(native_result.get("model_metadata") or {})
        chunk_summaries.append(
            {
                "chunk_index": len(chunk_summaries),
                "window_start": int(chunk_start),
                "window_end": int(chunk_end),
                "window_count": int(chunk_window_count),
                "core_bases": int(chunk_core_bases),
                "arithmetic_coded_bytes": int(sum(len(stream) for stream in chunk_streams)) if encode_arithmetic else None,
                "emitted_arithmetic_symbol_count": int(native_result["emitted_arithmetic_symbol_count"]),
                "fusion_final_mean_lm_weight": float(native_result["fusion_final_mean_lm_weight"]),
                "diagnostics_collected": bool(chunk_diagnostics),
                "core_theoretical_bits_per_base": (
                    float(native_result["fused_theoretical_bits"]) / max(chunk_core_bases, 1)
                    if chunk_diagnostics
                    else None
                ),
                "lm_only_theoretical_bits_per_base": (
                    float(native_result["lm_only_theoretical_bits"]) / max(chunk_core_bases, 1)
                    if chunk_diagnostics
                    else None
                ),
                "nc_prefix_only_theoretical_bits_per_base": (
                    float(native_result["nc_prefix_only_theoretical_bits"]) / max(chunk_core_bases, 1)
                    if chunk_diagnostics
                    else None
                ),
                "model_seconds": float(chunk_model_seconds),
                "lm_factorize_seconds": float(chunk_factorize_seconds),
                "lm_probability_transfer_seconds": float(chunk_transfer_seconds),
                "native_fused_encode_seconds_observed": float(chunk_native_observed_seconds),
                "native_fused_encode_seconds": float(native_result["encode_seconds"]),
                "native_finish_seconds": float(native_result["finish_seconds"]),
                "streaming_async_jobs": int(chunk_token_jobs),
                "streaming_gpu_queue_wait_seconds": float(chunk_gpu_queue_wait_seconds),
                "nc_prefix_window_count": int(chunk_metadata.get("window_count", chunk_window_count)),
                "nc_prefix_hash_bucket_count": chunk_metadata.get("hash_bucket_count"),
                "nc_prefix_geco2_level": int(chunk_metadata.get("geco2_level", nc_prefix_geco2_level)),
                "nc_prefix_pipeline_block_windows": chunk_metadata.get("pipeline_block_windows"),
            }
        )

    _sync(device)
    encode_wall_seconds = perf_counter() - encode_started
    tail_bits = _tail_side_info_bits(windows.tail_sequence)
    encoded_bytes = sum(len(stream) for stream in encoded_streams)
    arithmetic_bytes = encoded_bytes + ((tail_bits + 7) // 8 if encode_arithmetic else 0)
    core_bases = len(windows.core_sequence)
    sample_bases = len(windows.sequence)
    elapsed = perf_counter() - process_started
    fused_bits = float(fused_bits_total) if diagnostics_collected else None
    metrics: dict[str, Any] = {
        "codec": "fused_lm_nc_prefix",
        "lm_backend": lm_adapter.name,
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
        "batch_count": int(len(chunk_summaries)),
        "batch_size": int(resolved_batch_size),
        "batch_window_counts": [int(item["window_count"]) for item in chunk_summaries],
        "token_merge_size": int(token_merge_size),
        "tokens_per_window": int(tokens_per_window),
        "window_bases": int(window_bases),
        "lm_seq_length": int(lm_seq_length),
        "model_uses_ascii_tokens": bool(windows.model_uses_ascii_tokens),
        "model_token_alphabet": windows.model_token_alphabet,
        "lm_metadata": lm_adapter.metadata(),
        "fusion_policy": "online_hedge_linear_native",
        "fusion_eta": float(fusion_eta),
        "fusion_initial_lm_weight": float(fusion_initial_lm_weight),
        "fusion_final_mean_lm_weight": weighted_lm_weight_sum / max(weighted_lm_weight_count, 1),
        "diagnostics_collected": bool(diagnostics_collected),
        "theoretical_bits": float(fused_bits + tail_bits) if diagnostics_collected else None,
        "core_model_theoretical_bits": float(fused_bits) if diagnostics_collected else None,
        "theoretical_bits_per_base": (float(fused_bits + tail_bits) / max(sample_bases, 1))
        if diagnostics_collected
        else None,
        "core_theoretical_bits_per_base": (float(fused_bits) / max(core_bases, 1))
        if diagnostics_collected
        else None,
        "lm_only_theoretical_bits": float(lm_bits_total) if diagnostics_collected else None,
        "lm_only_theoretical_bits_per_base": (float(lm_bits_total) / max(core_bases, 1)) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits": float(nc_bits_total) if diagnostics_collected else None,
        "nc_prefix_only_theoretical_bits_per_base": (float(nc_bits_total) / max(core_bases, 1))
        if diagnostics_collected
        else None,
        "arithmetic_coded_bytes": int(arithmetic_bytes) if encode_arithmetic else None,
        "arithmetic_bits_per_base": (float(arithmetic_bytes) * 8.0 / max(sample_bases, 1)) if encode_arithmetic else None,
        "arithmetic_stream_count": int(len(encoded_streams)),
        "emitted_arithmetic_symbol_count": int(emitted_symbols),
        "compression_process_seconds": float(elapsed),
        "compression_core_seconds": float(encode_wall_seconds),
        "compression_bases_per_second": float(sample_bases) / max(elapsed, 1e-12),
        "model_seconds": float(model_seconds),
        "lm_factorize_seconds": float(lm_factorize_seconds),
        "lm_probability_transfer_seconds": float(lm_transfer_seconds),
        "native_fused_encode_seconds_observed": float(native_encode_seconds_observed),
        "native_fused_encode_seconds": float(native_encode_seconds),
        "native_finish_seconds": float(native_finish_seconds),
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
        "nc_prefix_metadata": {
            "batch_scope": "per_window_batch_independent_nc_prefix_state",
            "batch_count": int(len(chunk_summaries)),
            "batch_size": int(resolved_batch_size),
            "tail_batch_window_count": int(chunk_summaries[-1]["window_count"]) if chunk_summaries else 0,
            "hash_bucket_count_config": int(nc_prefix_hash_bucket_count),
            "geco2_level_config": int(nc_prefix_geco2_level),
            "chunk_summaries": chunk_summaries,
        },
        **arithmetic_metadata,
        **memory_stats(device, prefix="compression_"),
        **baseline_sizes(payload, include_codec_baselines=include_codec_baselines),
    }
    return metrics


def compress_fused_lm_nc_prefix_payload(
    *,
    payload: bytes,
    device: torch.device,
    dtype_name: str,
    batch_size: int | str,
    model: torch.nn.Module | None = None,
    config: ExperimentConfig | None = None,
    lm_adapter: StreamingLMAdapter | None = None,
    nc_prefix_window_bases: int | None = None,
    nc_prefix_min_windows: int = 8192,
    nc_prefix_hash_bucket_count: int = 0,
    nc_prefix_geco2_level: int = 10,
    fusion_eta: float = 0.05,
    fusion_initial_lm_weight: float = 0.5,
    arithmetic_frequency_total: int | None = None,
    arithmetic_target_uniform_mass: float = 0.01,
    encode_arithmetic: bool = True,
    pipeline_mode: str = "streaming_token_encode_overlap",
    collect_diagnostics: bool = True,
    include_codec_baselines: bool = True,
) -> dict[str, Any]:
    if pipeline_mode in {"streaming_token_encode_overlap", "streaming_token_strict", "streaming_token_nc_full_batch"}:
        return _compress_fused_streaming_token_payload(
            model=model,
            config=config,
            lm_adapter=lm_adapter,
            payload=payload,
            device=device,
            dtype_name=dtype_name,
            batch_size=batch_size,
            nc_prefix_window_bases=nc_prefix_window_bases,
            nc_prefix_min_windows=nc_prefix_min_windows,
            nc_prefix_hash_bucket_count=nc_prefix_hash_bucket_count,
            nc_prefix_geco2_level=nc_prefix_geco2_level,
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
        "pipeline_mode must be 'streaming_token_encode_overlap', 'streaming_token_strict', "
        "or 'streaming_token_nc_full_batch'; "
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


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    torch_dtype = getattr(torch, str(dtype_name), None)
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if str(dtype_name) == "float16":
        return torch.float16
    if str(dtype_name) == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_carbon_adapter_for_fusion(
    *,
    local_path: str | Path,
    device: torch.device,
    dtype_name: str,
    model_name: str = "Carbon-500M",
    revision: str = "fns",
    context_bases: int = 3072,
    trust_remote_code: bool = True,
) -> tuple[CarbonStreamingAdapter, dict[str, Any]]:
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Carbon local model directory not found: {local_path}")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Carbon fused adapter requires transformers to be installed.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(local_path),
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(local_path),
        revision=revision,
        trust_remote_code=trust_remote_code,
        torch_dtype=_torch_dtype_from_name(dtype_name),
    ).to(device)
    model.eval()
    if hasattr(model, "setup_tokenizer"):
        model.setup_tokenizer(tokenizer)
    elif not hasattr(model, "tokenizer"):
        model.tokenizer = tokenizer
    if not hasattr(model, "forward"):
        raise TypeError("Loaded Carbon model does not expose a forward method")
    adapter = CarbonStreamingAdapter(
        model=model,
        tokenizer=tokenizer,
        local_path=local_path,
        model_name=model_name,
        revision=revision,
        context_bases=context_bases,
        trust_remote_code=trust_remote_code,
    )
    metadata = {
        "lm_backend": "carbon",
        "carbon_model_name": model_name,
        "carbon_revision": revision,
        "carbon_local_path": str(local_path),
        "carbon_context_bases": int(context_bases),
        "carbon_dtype": str(dtype_name),
        "carbon_trust_remote_code": bool(trust_remote_code),
        "carbon_k": int(adapter.token_merge_size),
        "carbon_context_tokens": int(adapter.seq_length),
    }
    return adapter, metadata
