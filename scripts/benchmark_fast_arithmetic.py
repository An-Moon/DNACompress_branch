from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.fast_arithmetic import StreamingArithmeticEncoder


def _run_backend(probabilities: np.ndarray, symbols: np.ndarray, *, total: int, backend: str) -> dict[str, object]:
    encoder = StreamingArithmeticEncoder(backend)
    started = perf_counter()
    timings = encoder.encode_probability_rows(probabilities, symbols, total=total)
    encoded = encoder.finish()
    elapsed = perf_counter() - started
    return {
        "requested_backend": backend,
        "actual_backend": encoder.backend,
        "rows": int(probabilities.shape[0]),
        "vocab_size": int(probabilities.shape[1]),
        "encoded_bytes": len(encoded),
        "elapsed_seconds": elapsed,
        "quantize_seconds": timings.quantize_seconds,
        "range_seconds": timings.range_seconds,
        "rows_per_second": float(probabilities.shape[0]) / max(elapsed, 1e-12),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Python vs C++ arithmetic encoding on synthetic probabilities.")
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--vocab-size", type=int, default=56)
    parser.add_argument("--total", type=int, default=1 << 15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    logits = rng.normal(size=(args.rows, args.vocab_size)).astype(args.dtype)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    symbols = rng.integers(0, args.vocab_size, size=(args.rows,), dtype=np.int64)

    python_result = _run_backend(probabilities, symbols, total=args.total, backend="python")
    fast_result = _run_backend(probabilities, symbols, total=args.total, backend="fast_cpp")
    speedup = python_result["elapsed_seconds"] / max(float(fast_result["elapsed_seconds"]), 1e-12)
    print(json.dumps(
        {
            "python": python_result,
            "fast_cpp": fast_result,
            "speedup": speedup,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
