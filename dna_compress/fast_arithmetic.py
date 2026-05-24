from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import torch

from .compression import ArithmeticEncoder, probabilities_to_cumulative_batch


ARITHMETIC_BACKENDS = ("python", "fast_cpp", "auto")

_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None


@dataclass
class EncodeTimings:
    quantize_seconds: float
    range_seconds: float
    emitted_count: int

    @property
    def encode_seconds(self) -> float:
        return self.quantize_seconds + self.range_seconds


def _extension_source_path() -> Path:
    return Path(__file__).resolve().parent / "native" / "fast_arithmetic.cpp"


def load_fast_arithmetic_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise _EXTENSION_ERROR

    try:
        from torch.utils.cpp_extension import load

        _EXTENSION = load(
            name="dna_compress_fast_arithmetic",
            sources=[str(_extension_source_path())],
            extra_cflags=["-O3"],
            with_cuda=False,
            verbose=False,
        )
        return _EXTENSION
    except Exception as error:  # pragma: no cover - environment dependent
        _EXTENSION_ERROR = error
        raise


def _as_numpy_probabilities(probability_rows) -> np.ndarray:
    if isinstance(probability_rows, torch.Tensor):
        if probability_rows.device.type != "cpu":
            raise ValueError("probability_rows must already be on CPU")
        return np.ascontiguousarray(probability_rows.detach().numpy())
    return np.ascontiguousarray(np.asarray(probability_rows))


def _as_numpy_ints(symbols) -> np.ndarray:
    if isinstance(symbols, torch.Tensor):
        if symbols.device.type != "cpu":
            raise ValueError("symbols must already be on CPU")
        return np.ascontiguousarray(symbols.detach().numpy(), dtype=np.int64)
    return np.ascontiguousarray(np.asarray(symbols, dtype=np.int64))


def _as_cpu_probability_tensor(probability_rows) -> torch.Tensor:
    if isinstance(probability_rows, torch.Tensor):
        if probability_rows.device.type != "cpu":
            raise ValueError("probability_rows must already be on CPU")
        tensor = probability_rows.detach()
        if tensor.dtype not in {torch.float32, torch.float64}:
            tensor = tensor.float()
        return tensor.contiguous()
    array = np.ascontiguousarray(np.asarray(probability_rows))
    if array.dtype not in (np.float32, np.float64):
        array = array.astype(np.float32, copy=False)
    return torch.from_numpy(array)


def _as_cpu_symbol_tensor(symbols) -> torch.Tensor:
    if isinstance(symbols, torch.Tensor):
        if symbols.device.type != "cpu":
            raise ValueError("symbols must already be on CPU")
        tensor = symbols.detach()
        if tensor.dtype not in {torch.int32, torch.int64}:
            tensor = tensor.long()
        return tensor.contiguous()
    return torch.from_numpy(np.ascontiguousarray(np.asarray(symbols, dtype=np.int64)))


def resolve_arithmetic_backend(requested_backend: str) -> str:
    if requested_backend not in ARITHMETIC_BACKENDS:
        raise ValueError(f"arithmetic backend must be one of: {', '.join(ARITHMETIC_BACKENDS)}")
    if requested_backend == "python":
        return "python"
    try:
        load_fast_arithmetic_extension()
        return "fast_cpp"
    except Exception:
        if requested_backend == "auto":
            return "python"
        raise


class StreamingArithmeticEncoder:
    def __init__(self, backend: str = "python") -> None:
        self.backend = resolve_arithmetic_backend(backend)
        if self.backend == "fast_cpp":
            self._encoder = load_fast_arithmetic_extension().FastArithmeticEncoder()
        else:
            self._encoder = ArithmeticEncoder()

    def encode_probability_rows(self, probability_rows, target_symbols, *, total: int) -> EncodeTimings:
        if self.backend == "fast_cpp":
            result = self._encoder.encode_probability_rows(
                _as_cpu_probability_tensor(probability_rows),
                _as_cpu_symbol_tensor(target_symbols),
                int(total),
            )
            return EncodeTimings(
                quantize_seconds=float(result["quantize_seconds"]),
                range_seconds=float(result["range_seconds"]),
                emitted_count=int(result["emitted_count"]),
            )

        probabilities = _as_numpy_probabilities(probability_rows)
        symbols = _as_numpy_ints(target_symbols)
        quantize_started = perf_counter()
        cumulative_batch = probabilities_to_cumulative_batch(probabilities, total=total)
        quantize_seconds = perf_counter() - quantize_started

        range_started = perf_counter()
        for cumulative, target in zip(cumulative_batch, symbols):
            self._encoder.update(cumulative, int(target))
        range_seconds = perf_counter() - range_started
        return EncodeTimings(
            quantize_seconds=quantize_seconds,
            range_seconds=range_seconds,
            emitted_count=int(symbols.shape[0]),
        )

    def encode_grouped_steps(
        self,
        step_probabilities: Iterable,
        step_symbols: Iterable,
        step_row_positions: Iterable,
        *,
        row_count: int,
        total: int,
    ) -> EncodeTimings:
        probabilities = list(step_probabilities)
        symbols = list(step_symbols)
        positions = list(step_row_positions)
        if not (len(probabilities) == len(symbols) == len(positions)):
            raise ValueError("step probability, symbol, and row-position lists must have equal length")

        if self.backend == "fast_cpp":
            result = self._encoder.encode_grouped_steps(
                [_as_cpu_probability_tensor(step) for step in probabilities],
                [_as_cpu_symbol_tensor(step) for step in symbols],
                [_as_cpu_symbol_tensor(step) for step in positions],
                int(row_count),
                int(total),
            )
            return EncodeTimings(
                quantize_seconds=float(result["quantize_seconds"]),
                range_seconds=float(result["range_seconds"]),
                emitted_count=int(result["emitted_count"]),
            )

        probability_arrays = [_as_numpy_probabilities(step) for step in probabilities]
        symbol_arrays = [_as_numpy_ints(step) for step in symbols]
        position_arrays = [_as_numpy_ints(step) for step in positions]

        quantize_started = perf_counter()
        cumulative_steps = [
            probabilities_to_cumulative_batch(step, total=total)
            for step in probability_arrays
        ]
        quantize_seconds = perf_counter() - quantize_started

        emitted_count = 0
        range_started = perf_counter()
        for row_index in range(row_count):
            for cumulative_batch, symbol_batch, row_positions in zip(
                cumulative_steps,
                symbol_arrays,
                position_arrays,
            ):
                position = int(row_positions[row_index])
                if position < 0:
                    continue
                self._encoder.update(cumulative_batch[position], int(symbol_batch[position]))
                emitted_count += 1
        range_seconds = perf_counter() - range_started
        return EncodeTimings(
            quantize_seconds=quantize_seconds,
            range_seconds=range_seconds,
            emitted_count=emitted_count,
        )

    def finish(self) -> bytes:
        return bytes(self._encoder.finish())


def fast_decode_probability_rows(encoded: bytes, probability_rows, *, total: int) -> np.ndarray:
    extension = load_fast_arithmetic_extension()
    decoder = extension.FastArithmeticDecoder(encoded)
    decoded = decoder.decode_probability_rows(_as_cpu_probability_tensor(probability_rows), int(total))
    return decoded.cpu().numpy()
