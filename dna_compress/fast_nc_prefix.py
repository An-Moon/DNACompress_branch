from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


NC_PREFIX_BACKENDS = ("auto", "fast_cpp")

_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None


def _extension_source_path() -> Path:
    return Path(__file__).resolve().parent / "native" / "fast_nc_prefix.cpp"


def load_fast_nc_prefix_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise _EXTENSION_ERROR

    try:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="dna_compress_fast_nc_prefix_nc_geco2_levels_v6",
            sources=[str(_extension_source_path())],
            extra_cflags=["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
            with_cuda=False,
            verbose=False,
        )
        return _EXTENSION
    except Exception as error:  # pragma: no cover - environment dependent
        _EXTENSION_ERROR = error
        raise


def resolve_nc_prefix_backend(requested_backend: str) -> str:
    if requested_backend not in NC_PREFIX_BACKENDS:
        raise ValueError(f"nc_prefix backend must be one of: {', '.join(NC_PREFIX_BACKENDS)}")
    load_fast_nc_prefix_extension()
    return "fast_cpp"


def compute_fast_nc_prefix(
    symbols: np.ndarray,
    *,
    window_bases: int,
    vocab_size: int,
    return_probabilities: bool,
    summary_only: bool = False,
    hash_bucket_count: int = 0,
    geco2_level: int = 10,
    stream_arithmetic_total: int = 0,
) -> dict[str, Any]:
    if int(hash_bucket_count) < 0:
        raise ValueError("hash_bucket_count must be non-negative; use 0 for GECO2 default")
    if int(geco2_level) < 1 or int(geco2_level) > 12:
        raise ValueError("geco2_level must be in [1, 12]")
    if int(stream_arithmetic_total) < 0:
        raise ValueError("stream_arithmetic_total must be non-negative")
    extension = load_fast_nc_prefix_extension()
    symbol_tensor = torch.from_numpy(np.ascontiguousarray(symbols, dtype=np.int16))
    if hasattr(extension, "compute_nc_prefix_current"):
        return extension.compute_nc_prefix_current(
            symbol_tensor,
            int(window_bases),
            int(vocab_size),
            bool(return_probabilities),
            bool(summary_only),
            int(hash_bucket_count),
            int(geco2_level),
            int(stream_arithmetic_total),
        )

    order_tensor = torch.empty((0,), dtype=torch.int64)
    return extension.compute_nc_prefix(
        symbol_tensor,
        order_tensor,
        int(window_bases),
        int(vocab_size),
        0.0,
        0.0,
        0.0,
        False,
        bool(return_probabilities),
        bool(summary_only),
        2,
        int(geco2_level),
        False,
        False,
        0,
        int(hash_bucket_count),
        0,
        "cache_pipeline",
        "normal",
        0,
    )


class FusedNcPrefixStreamingEncoder:
    def __init__(
        self,
        *,
        window_count: int,
        window_bases: int,
        hash_bucket_count: int,
        geco2_level: int = 10,
        arithmetic_frequency_total: int,
        fusion_eta: float,
        initial_lm_weight: float,
        encode_arithmetic: bool,
        collect_diagnostics: bool = True,
    ) -> None:
        extension = load_fast_nc_prefix_extension()
        self._encoder = extension.FusedNcPrefixStreamingEncoder(
            int(window_count),
            int(window_bases),
            int(hash_bucket_count),
            int(geco2_level),
            int(arithmetic_frequency_total),
            float(fusion_eta),
            float(initial_lm_weight),
            bool(encode_arithmetic),
            bool(collect_diagnostics),
        )

    def encode_base_step(self, lm_probabilities, target_symbols) -> dict[str, Any]:
        if not isinstance(lm_probabilities, torch.Tensor):
            lm_probabilities = torch.as_tensor(lm_probabilities)
        if not isinstance(target_symbols, torch.Tensor):
            target_symbols = torch.as_tensor(target_symbols)
        if lm_probabilities.device.type != "cpu" or target_symbols.device.type != "cpu":
            raise ValueError("streaming fused encoder expects CPU tensors")
        return dict(self._encoder.encode_base_step(lm_probabilities.contiguous(), target_symbols.contiguous()))

    def encode_token_step(self, lm_probabilities, target_symbols) -> dict[str, Any]:
        if not isinstance(lm_probabilities, torch.Tensor):
            lm_probabilities = torch.as_tensor(lm_probabilities)
        if not isinstance(target_symbols, torch.Tensor):
            target_symbols = torch.as_tensor(target_symbols)
        if lm_probabilities.device.type != "cpu" or target_symbols.device.type != "cpu":
            raise ValueError("streaming fused encoder expects CPU tensors")
        return dict(self._encoder.encode_token_step(lm_probabilities.contiguous(), target_symbols.contiguous()))

    def encode_token_step_collect_targets(self, lm_probabilities, target_symbols) -> dict[str, Any]:
        if not isinstance(lm_probabilities, torch.Tensor):
            lm_probabilities = torch.as_tensor(lm_probabilities)
        if not isinstance(target_symbols, torch.Tensor):
            target_symbols = torch.as_tensor(target_symbols)
        if lm_probabilities.device.type != "cpu" or target_symbols.device.type != "cpu":
            raise ValueError("streaming fused encoder expects CPU tensors")
        return dict(
            self._encoder.encode_token_step(
                lm_probabilities.contiguous(),
                target_symbols.contiguous(),
                True,
                False,
            )
        )

    def encode_token_step_collect_probabilities(self, lm_probabilities, target_symbols) -> dict[str, Any]:
        if not isinstance(lm_probabilities, torch.Tensor):
            lm_probabilities = torch.as_tensor(lm_probabilities)
        if not isinstance(target_symbols, torch.Tensor):
            target_symbols = torch.as_tensor(target_symbols)
        if lm_probabilities.device.type != "cpu" or target_symbols.device.type != "cpu":
            raise ValueError("streaming fused encoder expects CPU tensors")
        return dict(
            self._encoder.encode_token_step(
                lm_probabilities.contiguous(),
                target_symbols.contiguous(),
                True,
                True,
            )
        )

    def freeze_background_and_reset_targets(
        self, target_window_count: int, initial_lm_weight: float = 0.0
    ) -> dict[str, Any]:
        return dict(
            self._encoder.freeze_background_and_reset_targets(
                int(target_window_count), float(initial_lm_weight)
            )
        )

    def train_background_token_step(self, target_symbols) -> dict[str, Any]:
        if not isinstance(target_symbols, torch.Tensor):
            target_symbols = torch.as_tensor(target_symbols)
        if target_symbols.device.type != "cpu":
            raise ValueError("background training expects CPU target symbols")
        return dict(self._encoder.train_background_token_step(target_symbols.contiguous()))

    def finish(self) -> dict[str, Any]:
        return dict(self._encoder.finish())

    def metadata(self) -> dict[str, Any]:
        return dict(self._encoder.metadata())
