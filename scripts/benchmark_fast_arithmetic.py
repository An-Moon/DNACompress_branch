from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.fast_arithmetic import StreamingArithmeticEncoder, fast_floor_intervals_from_probabilities


def _run_precise_backend(probabilities: np.ndarray, symbols: np.ndarray, *, total: int, backend: str) -> dict[str, object]:
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


def _run_fast_floor_cpu(probabilities: np.ndarray, symbols: np.ndarray, *, total: int) -> dict[str, object]:
    encoder = StreamingArithmeticEncoder("fast_cpp")
    started = perf_counter()
    timings = encoder.encode_probability_rows_fast_floor(probabilities, symbols, total=total)
    encoded = encoder.finish()
    elapsed = perf_counter() - started
    return {
        "requested_backend": "fast_cpp",
        "actual_backend": encoder.backend,
        "rows": int(probabilities.shape[0]),
        "vocab_size": int(probabilities.shape[1]),
        "encoded_bytes": len(encoded),
        "elapsed_seconds": elapsed,
        "quantize_seconds": timings.quantize_seconds,
        "range_seconds": timings.range_seconds,
        "interval_transfer_seconds": timings.interval_transfer_seconds,
        "rows_per_second": float(probabilities.shape[0]) / max(elapsed, 1e-12),
    }


def _run_fast_floor_gpu(probabilities: np.ndarray, symbols: np.ndarray, *, total: int) -> dict[str, object] | None:
    if not torch.cuda.is_available():
        return None
    encoder = StreamingArithmeticEncoder("fast_cpp")
    probabilities_gpu = torch.from_numpy(probabilities).cuda()
    symbols_gpu = torch.from_numpy(symbols).cuda()
    torch.cuda.synchronize()
    started = perf_counter()
    interval_started = perf_counter()
    lows, highs, totals = fast_floor_intervals_from_probabilities(probabilities_gpu, symbols_gpu, total=total)
    torch.cuda.synchronize()
    interval_seconds = perf_counter() - interval_started
    transfer_started = perf_counter()
    lows_cpu = lows.cpu()
    highs_cpu = highs.cpu()
    totals_cpu = totals.cpu()
    torch.cuda.synchronize()
    interval_transfer_seconds = perf_counter() - transfer_started
    timings = encoder.encode_intervals(
        lows_cpu,
        highs_cpu,
        totals_cpu,
        interval_transfer_seconds=interval_transfer_seconds,
    )
    encoded = encoder.finish()
    elapsed = perf_counter() - started
    return {
        "requested_backend": "fast_cpp",
        "actual_backend": encoder.backend,
        "rows": int(probabilities.shape[0]),
        "vocab_size": int(probabilities.shape[1]),
        "encoded_bytes": len(encoded),
        "elapsed_seconds": elapsed,
        "quantize_seconds": interval_seconds,
        "fast_floor_interval_seconds": interval_seconds,
        "range_seconds": timings.range_seconds,
        "interval_transfer_seconds": interval_transfer_seconds,
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

    python_result = _run_precise_backend(probabilities, symbols, total=args.total, backend="python")
    fast_result = _run_precise_backend(probabilities, symbols, total=args.total, backend="fast_cpp")
    fast_floor_cpu_result = _run_fast_floor_cpu(probabilities, symbols, total=args.total)
    fast_floor_gpu_result = _run_fast_floor_gpu(probabilities, symbols, total=args.total)
    speedup = python_result["elapsed_seconds"] / max(float(fast_result["elapsed_seconds"]), 1e-12)
    fast_floor_cpu_speedup = fast_result["elapsed_seconds"] / max(
        float(fast_floor_cpu_result["elapsed_seconds"]), 1e-12
    )
    print(json.dumps(
        {
            "python": python_result,
            "precise_fast_cpp": fast_result,
            "fast_floor_cpu": fast_floor_cpu_result,
            "fast_floor_gpu": fast_floor_gpu_result,
            "speedup": speedup,
            "fast_floor_cpu_speedup_vs_precise_fast_cpp": fast_floor_cpu_speedup,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
