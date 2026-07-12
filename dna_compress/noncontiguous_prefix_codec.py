from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .compression import resolve_arithmetic_coding_metadata
from .fast_arithmetic import StreamingArithmeticEncoder
from .fast_nc_prefix import (
    NC_PREFIX_BACKENDS,
    compute_fast_nc_prefix,
    resolve_nc_prefix_backend,
)
from .fusion_compression import ProbabilityAdapter, UnitProbabilityResult
from .tokenization import normalize_alphabet


DEFAULT_NC_PREFIX_MIN_WINDOWS = 8192


@dataclass(frozen=True)
class NoncontiguousPrefixConfig:
    window_bases: int = 3072
    alphabet: str = "ACGT"
    backend: str = "auto"
    min_windows: int = DEFAULT_NC_PREFIX_MIN_WINDOWS
    hash_bucket_count: int = 0


@dataclass(frozen=True)
class NoncontiguousPrefixResult:
    probabilities: np.ndarray
    target_symbols: np.ndarray
    emit_order: np.ndarray
    metadata: dict[str, Any]
    bpb_values: np.ndarray | None = None

    @property
    def bpb(self) -> np.ndarray:
        if self.bpb_values is not None:
            return self.bpb_values
        rows = np.arange(self.target_symbols.shape[0], dtype=np.int64)
        target_probabilities = self.probabilities[rows, self.target_symbols].clip(min=1e-300)
        return -np.log2(target_probabilities).astype(np.float64, copy=False)


def _validate_config(config: NoncontiguousPrefixConfig) -> NoncontiguousPrefixConfig:
    if int(config.window_bases) <= 0:
        raise ValueError("window_bases must be positive")
    alphabet = normalize_alphabet(config.alphabet)
    if alphabet != "ACGT":
        raise ValueError("nc_prefix currently supports alphabet='ACGT' only")
    backend = str(config.backend)
    if backend not in NC_PREFIX_BACKENDS:
        raise ValueError(f"backend must be one of: {', '.join(NC_PREFIX_BACKENDS)}")
    if int(config.min_windows) <= 0:
        raise ValueError("min_windows must be positive")
    if int(config.hash_bucket_count) < 0:
        raise ValueError("hash_bucket_count must be non-negative; use 0 for GECO2 default")
    return NoncontiguousPrefixConfig(
        window_bases=int(config.window_bases),
        alphabet=alphabet,
        backend=backend,
        min_windows=int(config.min_windows),
        hash_bucket_count=int(config.hash_bucket_count),
    )


def compute_noncontiguous_prefix_probabilities(
    sequence: str,
    config: NoncontiguousPrefixConfig | None = None,
    *,
    return_probabilities: bool = True,
    summary_only: bool = False,
) -> NoncontiguousPrefixResult:
    cfg = _validate_config(config or NoncontiguousPrefixConfig())
    base_to_symbol = {base: index for index, base in enumerate(cfg.alphabet)}
    invalid = sorted(set(sequence) - set(base_to_symbol))
    if invalid:
        raise ValueError(f"sequence contains bases outside alphabet {cfg.alphabet!r}: {''.join(invalid)!r}")
    symbols = np.asarray([base_to_symbol[base] for base in sequence], dtype=np.int16)
    n = int(symbols.shape[0])
    vocab_size = len(cfg.alphabet)
    if n == 0:
        raise ValueError("sequence must contain at least one base")
    min_required_bases = int(cfg.window_bases) * int(cfg.min_windows)
    if n < min_required_bases:
        raise ValueError(
            "nc_prefix requires a larger sequence to provide enough non-contiguous prefix statistics: "
            f"sequence_bases={n}, window_bases={cfg.window_bases}, min_windows={cfg.min_windows}, "
            f"min_required_bases={min_required_bases}"
        )

    resolve_nc_prefix_backend(cfg.backend)
    fast_result = compute_fast_nc_prefix(
        symbols,
        window_bases=cfg.window_bases,
        vocab_size=vocab_size,
        return_probabilities=return_probabilities,
        summary_only=summary_only,
        hash_bucket_count=cfg.hash_bucket_count,
    )
    probabilities = fast_result["probabilities"].cpu().numpy()
    bpb_values = fast_result["bpb"].cpu().numpy()
    target_symbols = fast_result["target_symbols"].cpu().numpy()
    emit_order = fast_result["emit_order"].cpu().numpy()
    metadata = dict(fast_result["metadata"])
    metadata.update(
        {
            "codec": "nc_prefix",
            "alphabet": cfg.alphabet,
            "min_windows": int(cfg.min_windows),
            "min_required_bases": int(min_required_bases),
            "return_probabilities": bool(return_probabilities),
            "summary_only": bool(summary_only),
            "hash_bucket_count_config": int(cfg.hash_bucket_count),
        }
    )
    return NoncontiguousPrefixResult(
        probabilities=probabilities,
        target_symbols=target_symbols,
        emit_order=emit_order,
        metadata=metadata,
        bpb_values=bpb_values,
    )


def compress_noncontiguous_prefix_sequence(
    sequence: str,
    config: NoncontiguousPrefixConfig | None = None,
    *,
    arithmetic_frequency_total: int | None = None,
    arithmetic_target_uniform_mass: float = 0.01,
    encode_arithmetic: bool = True,
) -> dict[str, Any]:
    started = perf_counter()
    cfg = _validate_config(config or NoncontiguousPrefixConfig())
    result = compute_noncontiguous_prefix_probabilities(
        sequence,
        cfg,
        return_probabilities=encode_arithmetic,
        summary_only=not encode_arithmetic,
    )
    if encode_arithmetic:
        vocab_size = result.probabilities.shape[1]
        arithmetic_metadata = resolve_arithmetic_coding_metadata(
            vocab_size=vocab_size,
            requested_total=arithmetic_frequency_total,
            target_uniform_mass=arithmetic_target_uniform_mass,
        )
        total = int(arithmetic_metadata["arithmetic_frequency_total"])
        ordered_probabilities = result.probabilities[result.emit_order]
        ordered_symbols = result.target_symbols[result.emit_order]
        encoder = StreamingArithmeticEncoder("auto")
        timings = encoder.encode_probability_rows(ordered_probabilities, ordered_symbols, total=total)
        encoded = encoder.finish()
        arithmetic_backend = encoder.backend
        arithmetic_bits_per_base = (len(encoded) * 8.0) / max(int(result.target_symbols.shape[0]), 1)
        arithmetic_coded_bytes = len(encoded)
        arithmetic_symbol_count = int(ordered_symbols.shape[0])
        arithmetic_timing_fields = {
            "arithmetic_quantize_seconds": timings.quantize_seconds,
            "arithmetic_range_seconds": timings.range_seconds,
            "arithmetic_interval_transfer_seconds": timings.interval_transfer_seconds,
            "arithmetic_encode_seconds": timings.encode_seconds,
        }
    else:
        arithmetic_metadata = {
            "arithmetic_frequency_total": arithmetic_frequency_total,
            "arithmetic_target_uniform_mass": arithmetic_target_uniform_mass,
        }
        arithmetic_backend = None
        arithmetic_bits_per_base = None
        arithmetic_coded_bytes = None
        arithmetic_symbol_count = 0
        arithmetic_timing_fields = {
            "arithmetic_quantize_seconds": None,
            "arithmetic_range_seconds": None,
            "arithmetic_interval_transfer_seconds": None,
            "arithmetic_encode_seconds": None,
        }
    elapsed = perf_counter() - started
    base_count = int(result.metadata["base_count"])
    theoretical_bits = float(result.metadata["theoretical_bits"])
    return {
        "codec": "nc_prefix",
        "decodable_design": "lockstep_window_group",
        "encode_arithmetic": bool(encode_arithmetic),
        "sample_bases": base_count,
        "theoretical_bits": theoretical_bits,
        "theoretical_bits_per_base": theoretical_bits / max(base_count, 1),
        "arithmetic_coded_bytes": arithmetic_coded_bytes,
        "arithmetic_bits_per_base": arithmetic_bits_per_base,
        "emitted_arithmetic_symbol_count": arithmetic_symbol_count,
        "arithmetic_backend": arithmetic_backend,
        **arithmetic_timing_fields,
        "compression_process_seconds": elapsed,
        "compression_bases_per_second": base_count / max(elapsed, 1e-12),
        "model_metadata": result.metadata,
        **arithmetic_metadata,
    }


class NoncontiguousPrefixProbabilityAdapter(ProbabilityAdapter):
    def __init__(
        self,
        *,
        name: str = "nc_prefix",
        window_bases: int = 3072,
        alphabet: str = "ACGT",
        backend: str = "auto",
        min_windows: int = DEFAULT_NC_PREFIX_MIN_WINDOWS,
        hash_bucket_count: int = 0,
    ) -> None:
        self.name = name
        self.token_size = 1
        self.alphabet = normalize_alphabet(alphabet)
        self.config = _validate_config(
            NoncontiguousPrefixConfig(
                window_bases=window_bases,
                alphabet=self.alphabet,
                backend=backend,
                min_windows=min_windows,
                hash_bucket_count=hash_bucket_count,
            )
        )

    def unit_probabilities(
        self,
        *,
        species: str,
        core_sequence: str,
        unit_size: int,
        batch_size: int,
    ) -> UnitProbabilityResult:
        del species, batch_size
        if int(unit_size) != 1:
            raise ValueError("nc_prefix only supports unit_size=1")
        result = compute_noncontiguous_prefix_probabilities(core_sequence, self.config, return_probabilities=True)
        return UnitProbabilityResult(
            adapter_name=self.name,
            probabilities=result.probabilities,
            model_forward_seconds=0.0,
            softmax_seconds=0.0,
            aggregate_seconds=float(result.metadata["compute_seconds"]),
            data_transfer_seconds=0.0,
            metadata={
                "nc_prefix": result.metadata,
                "model_window_bases": int(self.config.window_bases),
            },
        )
