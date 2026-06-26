from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import torch

from .compression import ArithmeticEncoder, probabilities_to_cumulative_batch


ARITHMETIC_BACKENDS = ("python", "fast_cpp", "auto")
ARITHMETIC_QUANTIZATION_MODES = ("precise", "fast_floor_cpu", "fast_floor_gpu")

_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None


@dataclass
class EncodeTimings:
    quantize_seconds: float
    range_seconds: float
    emitted_count: int
    interval_transfer_seconds: float = 0.0
    fast_floor_interval_seconds: float = 0.0

    @property
    def encode_seconds(self) -> float:
        return self.quantize_seconds + self.interval_transfer_seconds + self.range_seconds


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

    def encode_probability_rows_fast_floor(self, probability_rows, target_symbols, *, total: int) -> EncodeTimings:
        if self.backend != "fast_cpp":
            raise ValueError("fast_floor_cpu quantization requires arithmetic backend 'fast_cpp' or 'auto'")
        result = self._encoder.encode_probability_rows_fast_floor(
            _as_cpu_probability_tensor(probability_rows),
            _as_cpu_symbol_tensor(target_symbols),
            int(total),
        )
        quantize_seconds = float(result["quantize_seconds"])
        return EncodeTimings(
            quantize_seconds=quantize_seconds,
            range_seconds=float(result["range_seconds"]),
            emitted_count=int(result["emitted_count"]),
            fast_floor_interval_seconds=quantize_seconds,
        )

    def encode_intervals(self, lows, highs, totals, *, interval_transfer_seconds: float = 0.0) -> EncodeTimings:
        if self.backend != "fast_cpp":
            raise ValueError("interval encoding requires arithmetic backend 'fast_cpp' or 'auto'")
        result = self._encoder.encode_intervals(
            _as_cpu_symbol_tensor(lows),
            _as_cpu_symbol_tensor(highs),
            _as_cpu_symbol_tensor(totals),
        )
        return EncodeTimings(
            quantize_seconds=0.0,
            range_seconds=float(result["range_seconds"]),
            emitted_count=int(result["emitted_count"]),
            interval_transfer_seconds=float(interval_transfer_seconds),
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


def fast_decode_probability_rows_fast_floor(encoded: bytes, probability_rows, *, total: int) -> np.ndarray:
    extension = load_fast_arithmetic_extension()
    decoder = extension.FastArithmeticDecoder(encoded)
    decoded = decoder.decode_probability_rows_fast_floor(_as_cpu_probability_tensor(probability_rows), int(total))
    return decoded.cpu().numpy()


class StreamingArithmeticDecoder:
    def __init__(self, encoded: bytes) -> None:
        self._decoder = load_fast_arithmetic_extension().FastArithmeticDecoder(bytes(encoded))

    def decode_frequency_row(self, frequencies) -> int:
        return int(self._decoder.decode_frequency_row(_as_cpu_symbol_tensor(frequencies)))


class BatchedStreamingArithmeticEncoder:
    def __init__(self, stream_count: int) -> None:
        if stream_count <= 0:
            raise ValueError("batched arithmetic encoder requires at least one stream")
        self._encoder = load_fast_arithmetic_extension().BatchedFastArithmeticEncoder(int(stream_count))

    @property
    def size(self) -> int:
        return int(self._encoder.size())

    def encode_interval_matrix(self, lows, highs, totals) -> EncodeTimings:
        result = self._encoder.encode_interval_matrix(
            _as_cpu_symbol_tensor(lows),
            _as_cpu_symbol_tensor(highs),
            _as_cpu_symbol_tensor(totals),
        )
        return EncodeTimings(
            quantize_seconds=0.0,
            range_seconds=float(result["range_seconds"]),
            emitted_count=int(result["emitted_count"]),
        )

    def encode_interval_matrix_with_lengths(self, lows, highs, totals, lengths) -> EncodeTimings:
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.as_tensor(lengths)
        if lengths.device.type != "cpu":
            raise ValueError("row lengths must already be on CPU")
        if lengths.dim() != 1:
            raise ValueError("row lengths must be a 1D tensor")
        if lengths.dtype not in {torch.int64, torch.int32}:
            lengths = lengths.to(torch.int64)
        result = self._encoder.encode_interval_matrix_with_lengths(
            _as_cpu_symbol_tensor(lows),
            _as_cpu_symbol_tensor(highs),
            _as_cpu_symbol_tensor(totals),
            lengths.contiguous(),
        )
        return EncodeTimings(
            quantize_seconds=0.0,
            range_seconds=float(result["range_seconds"]),
            emitted_count=int(result["emitted_count"]),
        )

    def finish(self) -> list[bytes]:
        return [bytes(stream) for stream in self._encoder.finish()]


class BatchedStreamingArithmeticDecoder:
    def __init__(self, encoded_streams: Iterable[bytes], *, threads: int = 0) -> None:
        streams = [bytes(stream) for stream in encoded_streams]
        if not streams:
            raise ValueError("batched arithmetic decoder requires at least one stream")
        self._decoder = load_fast_arithmetic_extension().BatchedFastArithmeticDecoder(streams, int(threads))

    @property
    def size(self) -> int:
        return int(self._decoder.size())

    def decode_frequency_rows(self, frequencies) -> torch.Tensor:
        if not isinstance(frequencies, torch.Tensor):
            frequencies = torch.as_tensor(frequencies)
        if frequencies.device.type != "cpu":
            raise ValueError("frequency rows must already be on CPU")
        if frequencies.dim() != 2:
            raise ValueError("frequency rows must be a 2D tensor")
        if frequencies.dtype not in {torch.int64, torch.int32, torch.int16, torch.uint16, torch.uint8}:
            frequencies = frequencies.to(torch.int32)
        return self._decoder.decode_frequency_rows(frequencies.contiguous())

    def decode_frequency_rows_with_totals(self, frequencies, totals) -> torch.Tensor:
        if not isinstance(frequencies, torch.Tensor):
            frequencies = torch.as_tensor(frequencies)
        if not isinstance(totals, torch.Tensor):
            totals = torch.as_tensor(totals)
        if frequencies.device.type != "cpu" or totals.device.type != "cpu":
            raise ValueError("frequency rows and totals must already be on CPU")
        if frequencies.dim() != 2 or totals.dim() != 1 or totals.shape[0] != frequencies.shape[0]:
            raise ValueError("frequency rows must be [batch, vocab] and totals must be [batch]")
        if frequencies.dtype not in {torch.int64, torch.int32, torch.uint16}:
            frequencies = frequencies.to(torch.int32)
        if totals.dtype not in {torch.int64, torch.int32}:
            totals = totals.to(torch.int32)
        return self._decoder.decode_frequency_rows_with_totals(frequencies.contiguous(), totals.contiguous())

    def decode_frequency_rows_with_totals_and_active(self, frequencies, totals, active) -> torch.Tensor:
        if not isinstance(frequencies, torch.Tensor):
            frequencies = torch.as_tensor(frequencies)
        if not isinstance(totals, torch.Tensor):
            totals = torch.as_tensor(totals)
        if not isinstance(active, torch.Tensor):
            active = torch.as_tensor(active)
        if frequencies.device.type != "cpu" or totals.device.type != "cpu" or active.device.type != "cpu":
            raise ValueError("frequency rows, totals, and active mask must already be on CPU")
        if frequencies.dim() != 2 or totals.dim() != 1 or active.dim() != 1:
            raise ValueError("frequency rows must be [batch, vocab], totals [batch], active [batch]")
        if totals.shape[0] != frequencies.shape[0] or active.shape[0] != frequencies.shape[0]:
            raise ValueError("frequency rows, totals, and active mask must have matching batch size")
        if frequencies.dtype not in {torch.int64, torch.int32, torch.uint16}:
            frequencies = frequencies.to(torch.int32)
        if totals.dtype not in {torch.int64, torch.int32}:
            totals = totals.to(torch.int32)
        if active.dtype not in {torch.bool, torch.uint8, torch.int32, torch.int64}:
            active = active.to(torch.bool)
        return self._decoder.decode_frequency_rows_with_totals_and_active(
            frequencies.contiguous(),
            totals.contiguous(),
            active.contiguous(),
        )


def fast_floor_intervals_from_probabilities(
    probability_rows: torch.Tensor,
    target_symbols: torch.Tensor,
    *,
    total: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probability_rows.dim() != 2:
        raise ValueError("probability_rows must be a 2D tensor")
    if target_symbols.dim() != 1 or target_symbols.shape[0] != probability_rows.shape[0]:
        raise ValueError("target_symbols must be a 1D tensor matching probability rows")
    if target_symbols.device != probability_rows.device:
        target_symbols = target_symbols.to(probability_rows.device)

    probs = probability_rows.float()
    probs = torch.where(torch.isfinite(probs) & (probs > 0), probs, torch.zeros((), dtype=probs.dtype, device=probs.device))
    freqs = torch.floor(probs * float(total)).to(torch.int64)
    freqs = torch.clamp(freqs, min=1)
    cumulative = torch.cumsum(freqs, dim=1)
    symbols = target_symbols.to(dtype=torch.long)
    if torch.any(symbols < 0) or torch.any(symbols >= probability_rows.shape[1]):
        raise ValueError("target symbol is outside the probability row vocabulary")
    high = cumulative.gather(1, symbols.unsqueeze(1)).squeeze(1)
    selected_freq = freqs.gather(1, symbols.unsqueeze(1)).squeeze(1)
    low = high - selected_freq
    row_totals = cumulative[:, -1]
    return low.contiguous(), high.contiguous(), row_totals.contiguous()
