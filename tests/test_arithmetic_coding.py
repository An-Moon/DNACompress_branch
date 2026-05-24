from __future__ import annotations

import unittest

import numpy as np

from dna_compress.compression import (
    MIN_FREQUENCY_TOTAL,
    probabilities_to_cumulative_batch,
    resolve_frequency_total,
)
from dna_compress.fast_arithmetic import (
    StreamingArithmeticEncoder,
    fast_decode_probability_rows,
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
