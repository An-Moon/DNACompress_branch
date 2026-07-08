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
            name="dna_compress_fast_nc_prefix_current",
            sources=[str(_extension_source_path())],
            extra_cflags=["-O3"],
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
) -> dict[str, Any]:
    extension = load_fast_nc_prefix_extension()
    symbol_tensor = torch.from_numpy(np.ascontiguousarray(symbols, dtype=np.int16))
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
        2,
        10,
        False,
        False,
        0,
        0,
        0,
    )
