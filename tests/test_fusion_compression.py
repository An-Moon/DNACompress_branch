from __future__ import annotations

import unittest

import numpy as np

from dna_compress.fusion_compression import (
    FUSION_ORACLE_MAX,
    FUSION_STATIC_CONTEXT,
    ProbabilityAdapter,
    UnitProbabilityResult,
    compress_fusion_source,
    encode_unit_symbols,
    factorize_token_probabilities_to_units,
    fit_static_context_table,
    resolve_fusion_unit_size,
)


class _StaticAdapter(ProbabilityAdapter):
    def __init__(self, name: str, token_size: int, probabilities: np.ndarray) -> None:
        self.name = name
        self.token_size = token_size
        self.alphabet = "AC"
        self._probabilities = probabilities

    def unit_probabilities(
        self,
        *,
        species: str,
        core_sequence: str,
        unit_size: int,
        batch_size: int,
    ) -> UnitProbabilityResult:
        del species, core_sequence, unit_size, batch_size
        return UnitProbabilityResult(
            adapter_name=self.name,
            probabilities=self._probabilities,
            model_forward_seconds=0.0,
            softmax_seconds=0.0,
            aggregate_seconds=0.0,
            data_transfer_seconds=0.0,
        )


class FusionCompressionTests(unittest.TestCase):
    def test_resolve_fusion_unit_size_auto_and_explicit_validation(self) -> None:
        self.assertEqual(resolve_fusion_unit_size([6, 3], "auto"), 3)
        self.assertEqual(resolve_fusion_unit_size([5, 3], "auto"), 1)
        self.assertEqual(resolve_fusion_unit_size([6, 3], 1), 1)
        with self.assertRaises(ValueError):
            resolve_fusion_unit_size([6, 3], 4)

    def test_factorize_token_probabilities_to_units_matches_token_probability(self) -> None:
        # Alphabet AC, token_size=2, unit_size=1. Token ids in lexical unit order:
        # AA, AC, CA, CC.
        token_probabilities = np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64)
        target_units = np.asarray([1, 0], dtype=np.int64)  # CA
        unit_rows = factorize_token_probabilities_to_units(
            token_probabilities=token_probabilities,
            target_unit_symbols=target_units,
            token_size=2,
            unit_size=1,
            alphabet="AC",
        )

        emitted_probability = unit_rows[0, 1] * unit_rows[1, 0]
        self.assertAlmostEqual(emitted_probability, 0.3, places=8)
        self.assertAlmostEqual(float(unit_rows[0].sum()), 1.0, places=8)
        self.assertAlmostEqual(float(unit_rows[1].sum()), 1.0, places=8)

    def test_static_context_table_chooses_lower_average_bits_and_fallback(self) -> None:
        target_symbols = np.asarray([0, 1, 0, 1], dtype=np.int64)
        model_a = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.8, 0.2],
            ],
            dtype=np.float64,
        )
        model_b = np.asarray(
            [
                [0.6, 0.4],
                [0.1, 0.9],
                [0.6, 0.4],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        table = fit_static_context_table(
            adapter_names=["a", "b"],
            target_symbols=target_symbols,
            model_probabilities=[model_a, model_b],
            context_units=1,
        )

        self.assertEqual(table.select_model_index((-1,)), 0)
        self.assertEqual(table.select_model_index((0,)), 1)
        self.assertEqual(table.select_model_index((99,)), table.global_model_index)

    def test_oracle_bits_are_no_worse_than_single_models(self) -> None:
        source = b"ACAC"
        targets = encode_unit_symbols("ACAC", 1, "AC")
        self.assertEqual(targets.tolist(), [0, 1, 0, 1])
        model_a = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.8, 0.2],
            ],
            dtype=np.float64,
        )
        model_b = np.asarray(
            [
                [0.4, 0.6],
                [0.1, 0.9],
                [0.4, 0.6],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        adapters = [
            _StaticAdapter("a", 1, model_a),
            _StaticAdapter("b", 1, model_b),
        ]
        metrics = compress_fusion_source(
            species="test",
            source=source,
            adapters=adapters,
            unit_size=1,
            alphabet="AC",
            batch_size=2,
            requested_bytes=None,
            policy=FUSION_ORACLE_MAX,
            arithmetic_frequency_total=None,
            arithmetic_target_uniform_mass=0.01,
            context_units=1,
        )

        self.assertFalse(bool(metrics["decodable"]))
        self.assertLessEqual(
            float(metrics["core_model_theoretical_bits"]),
            float(metrics["model_a_theoretical_bits"]),
        )
        self.assertLessEqual(
            float(metrics["core_model_theoretical_bits"]),
            float(metrics["model_b_theoretical_bits"]),
        )
        self.assertGreater(int(metrics["arithmetic_coded_bytes"]), 0)
        self.assertEqual(metrics["arithmetic_coding_mode"], "fusion_oracle_max")

    def test_static_context_compression_uses_table(self) -> None:
        source = b"ACAC"
        model_a = np.asarray(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.9, 0.1],
                [0.8, 0.2],
            ],
            dtype=np.float64,
        )
        model_b = np.asarray(
            [
                [0.4, 0.6],
                [0.1, 0.9],
                [0.4, 0.6],
                [0.1, 0.9],
            ],
            dtype=np.float64,
        )
        target_symbols = encode_unit_symbols("ACAC", 1, "AC")
        table = fit_static_context_table(
            adapter_names=["a", "b"],
            target_symbols=target_symbols,
            model_probabilities=[model_a, model_b],
            context_units=1,
        )
        metrics = compress_fusion_source(
            species="test",
            source=source,
            adapters=[_StaticAdapter("a", 1, model_a), _StaticAdapter("b", 1, model_b)],
            unit_size=1,
            alphabet="AC",
            batch_size=2,
            requested_bytes=None,
            policy=FUSION_STATIC_CONTEXT,
            arithmetic_frequency_total=None,
            arithmetic_target_uniform_mass=0.01,
            context_units=1,
            static_table=table,
        )

        self.assertTrue(bool(metrics["decodable"]))
        self.assertEqual(metrics["arithmetic_coding_mode"], "fusion_static_context")
        self.assertEqual(metrics["fusion_model_choice_counts"], {"a": 2, "b": 2})


if __name__ == "__main__":
    unittest.main()
