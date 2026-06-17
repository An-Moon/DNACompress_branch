from __future__ import annotations

import unittest

import numpy as np
import torch

from dna_compress.compression import (
    MIN_FREQUENCY_TOTAL,
    probabilities_to_cumulative_batch,
    resolve_frequency_total,
)
from dna_compress.fast_arithmetic import (
    BatchedStreamingArithmeticEncoder,
    BatchedStreamingArithmeticDecoder,
    StreamingArithmeticDecoder,
    StreamingArithmeticEncoder,
    fast_decode_probability_rows_fast_floor,
    fast_decode_probability_rows,
    fast_floor_intervals_from_probabilities,
    load_fast_arithmetic_extension,
)


class ArithmeticCodingTests(unittest.TestCase):
    def test_resolve_frequency_total_for_dnagpt_vocab(self) -> None:
        self.assertEqual(resolve_frequency_total(19564, None, 0.01), 2097152)

    def test_resolve_frequency_total_keeps_small_vocab_floor(self) -> None:
        self.assertEqual(resolve_frequency_total(259, None, 0.01), MIN_FREQUENCY_TOTAL)

    def test_probabilities_to_cumulative_batch_large_vocab(self) -> None:
        vocab_size = 19564
        frequency_total = 2097152
        probabilities = np.full((1, vocab_size), 1.0 / vocab_size, dtype=np.float64)

        cumulative = probabilities_to_cumulative_batch(probabilities, total=frequency_total)

        self.assertEqual(cumulative.shape, (1, vocab_size + 1))
        self.assertEqual(int(cumulative[0, 0]), 0)
        self.assertEqual(int(cumulative[0, -1]), frequency_total)
        self.assertTrue(np.all(np.diff(cumulative[0]) > 0))

    def test_larger_frequency_total_reduces_quantization_penalty(self) -> None:
        vocab_size = 19564
        logits = np.linspace(0.0, -12.0, vocab_size, dtype=np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()

        entropy_bits = float(-(probabilities * np.log2(probabilities)).sum())
        low_total = probabilities_to_cumulative_batch(probabilities, total=1 << 15)
        high_total = probabilities_to_cumulative_batch(probabilities, total=1 << 21)

        low_quantized = np.diff(low_total).astype(np.float64)
        low_quantized /= low_quantized.sum()
        high_quantized = np.diff(high_total).astype(np.float64)
        high_quantized /= high_quantized.sum()

        low_penalty = float(-(probabilities * np.log2(low_quantized)).sum()) - entropy_bits
        high_penalty = float(-(probabilities * np.log2(high_quantized)).sum()) - entropy_bits

        self.assertGreater(low_penalty, 0.1)
        self.assertLess(high_penalty, low_penalty * 0.2)

    def test_fast_cpp_round_trip_probability_rows(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        rng = np.random.default_rng(7)
        probabilities = rng.random((128, 5), dtype=np.float64)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        symbols = rng.integers(0, probabilities.shape[1], size=probabilities.shape[0], dtype=np.int64)

        encoder = StreamingArithmeticEncoder("fast_cpp")
        timings = encoder.encode_probability_rows(probabilities, symbols, total=MIN_FREQUENCY_TOTAL)
        encoded = encoder.finish()
        decoded = fast_decode_probability_rows(encoded, probabilities, total=MIN_FREQUENCY_TOTAL)

        self.assertEqual(encoder.backend, "fast_cpp")
        self.assertEqual(timings.emitted_count, len(symbols))
        self.assertGreater(len(encoded), 0)
        np.testing.assert_array_equal(decoded, symbols)

    def test_fast_cpp_round_trip_probability_rows_fast_floor(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        rng = np.random.default_rng(11)
        for vocab_size in (5, 37, 131):
            probabilities = rng.random((256, vocab_size), dtype=np.float32)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            symbols = rng.integers(0, vocab_size, size=probabilities.shape[0], dtype=np.int64)

            encoder = StreamingArithmeticEncoder("fast_cpp")
            timings = encoder.encode_probability_rows_fast_floor(probabilities, symbols, total=MIN_FREQUENCY_TOTAL)
            encoded = encoder.finish()
            decoded = fast_decode_probability_rows_fast_floor(encoded, probabilities, total=MIN_FREQUENCY_TOTAL)

            self.assertEqual(timings.emitted_count, len(symbols))
            self.assertGreater(timings.fast_floor_interval_seconds, 0.0)
            self.assertGreater(len(encoded), 0)
            np.testing.assert_array_equal(decoded, symbols)

    def test_fast_floor_intervals_are_valid(self) -> None:
        probabilities = torch.tensor(
            [
                [0.1, 0.2, 0.7],
                [float("nan"), -1.0, 0.9],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        symbols = torch.tensor([2, 0, 1], dtype=torch.long)

        lows, highs, totals = fast_floor_intervals_from_probabilities(
            probabilities,
            symbols,
            total=MIN_FREQUENCY_TOTAL,
        )

        self.assertTrue(torch.all(lows >= 0))
        self.assertTrue(torch.all(highs > lows))
        self.assertTrue(torch.all(highs <= totals))
        self.assertTrue(torch.all(totals >= probabilities.shape[1]))

    def test_streaming_decoder_decodes_frequency_rows(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        frequency_rows = [
            torch.tensor([1, 4, 2, 7], dtype=torch.int64),
            torch.tensor([3, 1, 6, 2], dtype=torch.int64),
            torch.tensor([5, 5, 1, 9], dtype=torch.int64),
        ]
        symbols = [3, 0, 2]
        lows = []
        highs = []
        totals = []
        for freqs, symbol in zip(frequency_rows, symbols):
            cumulative = torch.cat((torch.zeros(1, dtype=torch.int64), torch.cumsum(freqs, dim=0)))
            lows.append(cumulative[symbol])
            highs.append(cumulative[symbol + 1])
            totals.append(cumulative[-1])

        encoder = StreamingArithmeticEncoder("fast_cpp")
        encoder.encode_intervals(torch.stack(lows), torch.stack(highs), torch.stack(totals))
        encoded = encoder.finish()
        decoder = StreamingArithmeticDecoder(encoded)
        decoded = [decoder.decode_frequency_row(freqs) for freqs in frequency_rows]

        self.assertEqual(decoded, symbols)

    def test_batched_streaming_decoder_decodes_frequency_rows(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        rng = np.random.default_rng(17)
        steps = 9
        batch_size = 12
        vocab_size = 19
        freqs = rng.integers(1, 500, size=(steps, batch_size, vocab_size), dtype=np.int64)
        symbols = rng.integers(0, vocab_size, size=(steps, batch_size), dtype=np.int64)
        encoded_streams = []
        for batch_index in range(batch_size):
            encoder = StreamingArithmeticEncoder("fast_cpp")
            lows = []
            highs = []
            totals = []
            for step in range(steps):
                row = torch.from_numpy(freqs[step, batch_index])
                cumulative = torch.cat((torch.zeros(1, dtype=torch.int64), torch.cumsum(row, dim=0)))
                symbol = int(symbols[step, batch_index])
                lows.append(cumulative[symbol])
                highs.append(cumulative[symbol + 1])
                totals.append(cumulative[-1])
            encoder.encode_intervals(torch.stack(lows), torch.stack(highs), torch.stack(totals))
            encoded_streams.append(encoder.finish())

        for dtype, threads in ((torch.uint16, 1), (torch.int32, 4)):
            decoder = BatchedStreamingArithmeticDecoder(encoded_streams, threads=threads)
            decoded_steps = []
            for step in range(steps):
                decoded_steps.append(decoder.decode_frequency_rows(torch.from_numpy(freqs[step]).to(dtype)))
            decoded = torch.stack(decoded_steps).numpy()
            np.testing.assert_array_equal(decoded, symbols)

        decoder = BatchedStreamingArithmeticDecoder(encoded_streams, threads=0)
        decoded_steps = []
        for step in range(steps):
            rows = torch.from_numpy(freqs[step]).to(torch.uint16)
            totals = rows.to(torch.int32).sum(dim=1)
            decoded_steps.append(decoder.decode_frequency_rows_with_totals(rows, totals))
        decoded = torch.stack(decoded_steps).numpy()
        np.testing.assert_array_equal(decoded, symbols)

    def test_batched_streaming_encoder_roundtrip_interval_matrix(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        rng = np.random.default_rng(23)
        batch_size = 7
        steps = 11
        vocab_size = 13
        freqs = rng.integers(1, 600, size=(batch_size, steps, vocab_size), dtype=np.int64)
        symbols = rng.integers(0, vocab_size, size=(batch_size, steps), dtype=np.int64)
        cumulative = np.concatenate(
            [np.zeros((batch_size, steps, 1), dtype=np.int64), np.cumsum(freqs, axis=2)],
            axis=2,
        )
        lows = np.take_along_axis(cumulative, symbols[:, :, None], axis=2).squeeze(2)
        highs = np.take_along_axis(cumulative, symbols[:, :, None] + 1, axis=2).squeeze(2)
        totals = cumulative[:, :, -1]

        encoder = BatchedStreamingArithmeticEncoder(batch_size)
        timings = encoder.encode_interval_matrix(
            torch.from_numpy(lows).to(torch.int32),
            torch.from_numpy(highs).to(torch.int32),
            torch.from_numpy(totals).to(torch.int32),
        )
        encoded_streams = encoder.finish()
        decoder = BatchedStreamingArithmeticDecoder(encoded_streams, threads=2)
        decoded_steps = []
        for step in range(steps):
            decoded_steps.append(decoder.decode_frequency_rows(torch.from_numpy(freqs[:, step]).to(torch.uint16)))
        decoded = torch.stack(decoded_steps, dim=1).numpy()

        self.assertEqual(len(encoded_streams), batch_size)
        self.assertEqual(timings.emitted_count, batch_size * steps)
        np.testing.assert_array_equal(decoded, symbols)

    def test_fast_floor_gpu_intervals_match_cpu_when_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")

        rng = np.random.default_rng(13)
        probabilities_np = rng.random((512, 17), dtype=np.float32)
        probabilities_np /= probabilities_np.sum(axis=1, keepdims=True)
        symbols_np = rng.integers(0, probabilities_np.shape[1], size=probabilities_np.shape[0], dtype=np.int64)
        probabilities_cpu = torch.from_numpy(probabilities_np)
        symbols_cpu = torch.from_numpy(symbols_np)
        cpu_intervals = fast_floor_intervals_from_probabilities(
            probabilities_cpu,
            symbols_cpu,
            total=MIN_FREQUENCY_TOTAL,
        )
        gpu_intervals = fast_floor_intervals_from_probabilities(
            probabilities_cpu.cuda(),
            symbols_cpu.cuda(),
            total=MIN_FREQUENCY_TOTAL,
        )

        for cpu_tensor, gpu_tensor in zip(cpu_intervals, gpu_intervals):
            torch.testing.assert_close(cpu_tensor, gpu_tensor.cpu(), rtol=0, atol=0)

    def test_auto_backend_returns_working_encoder(self) -> None:
        probabilities = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        symbols = np.array([1, 0], dtype=np.int64)
        encoder = StreamingArithmeticEncoder("auto")
        timings = encoder.encode_probability_rows(probabilities, symbols, total=MIN_FREQUENCY_TOTAL)
        encoded = encoder.finish()

        self.assertIn(encoder.backend, {"python", "fast_cpp"})
        self.assertEqual(timings.emitted_count, 2)
        self.assertGreater(len(encoded), 0)


if __name__ == "__main__":
    unittest.main()
