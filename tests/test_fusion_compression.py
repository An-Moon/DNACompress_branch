from __future__ import annotations

import unittest

import numpy as np
import torch

from dna_compress.config import ExperimentConfig
from dna_compress.fusion_compression import (
    FUSION_ORACLE_MAX,
    FUSION_STATIC_CONTEXT,
    MegabyteProbabilityAdapter,
    ProbabilityAdapter,
    UnitProbabilityResult,
    _MegabyteWindowMemoryState,
    _megabyte_forward_with_window_memory,
    compress_fusion_source,
    encode_unit_symbols,
    factorize_token_probabilities_to_units,
    fit_static_context_table,
    resolve_fusion_unit_size,
)
from dna_compress.megabyte_loader import build_model
from dna_compress.tokenization import apply_token_merge_to_model_config


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
    def _small_megabyte_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.model.implementation = "megabyte_in_action"
        config.model.patch_size = 4
        config.model.global_dim = 8
        config.model.local_dim = 8
        config.model.seq_length = 16
        config.model.global_heads = 2
        config.model.global_layers = 2
        config.model.local_heads = 2
        config.model.local_layers = 1
        config.model.flash_attn = False
        config.train.dtype = "float32"
        config.data.token_merge_size = 1
        config.data.token_merge_alphabet = "ACGT"
        apply_token_merge_to_model_config(config.model, config.data)
        return config

    def test_megabyte_window_memory_first_window_matches_forward(self) -> None:
        torch.manual_seed(5)
        config = self._small_megabyte_config()
        model = build_model(config.model).eval()
        ids = torch.randint(0, 4, (1, config.model.seq_length), dtype=torch.long)

        with torch.no_grad():
            expected = model(ids, return_loss=False).lm_logits
            actual, state, metadata = _megabyte_forward_with_window_memory(
                model,
                ids,
                _MegabyteWindowMemoryState(),
            )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        self.assertEqual(metadata["global_memory_layers_used"], 0)
        self.assertEqual(metadata["local_memory_used"], 0)
        torch.testing.assert_close(state.previous_patch_ids, ids.reshape(1, -1, int(config.model.patch_size))[:, -1, :])

    def test_megabyte_previous_window_memory_uses_shifted_input_without_kv(self) -> None:
        torch.manual_seed(10)
        config = self._small_megabyte_config()
        model = build_model(config.model).eval()
        first = torch.randint(0, 4, (1, config.model.seq_length), dtype=torch.long)
        second = torch.randint(0, 4, (1, config.model.seq_length), dtype=torch.long)

        with torch.no_grad():
            independent = model(second, return_loss=False).lm_logits
            _, state, _ = _megabyte_forward_with_window_memory(
                model,
                first,
                _MegabyteWindowMemoryState(),
            )
            streamed_tail, _, metadata = _megabyte_forward_with_window_memory(
                model,
                second,
                state,
            )

        self.assertEqual(metadata["global_memory_layers_used"], 0)
        self.assertEqual(metadata["local_memory_used"], 1)
        self.assertGreater(float((streamed_tail - independent).abs().max()), 0.0)

    def test_megabyte_previous_window_kv_matches_two_window_tail(self) -> None:
        torch.manual_seed(12)
        config = self._small_megabyte_config()
        model = build_model(config.model).eval()
        first = torch.randint(0, 4, (1, config.model.seq_length), dtype=torch.long)
        second = torch.randint(0, 4, (1, config.model.seq_length), dtype=torch.long)

        with torch.no_grad():
            combined = torch.cat([first, second], dim=1)
            expected_tail = model(combined, return_loss=False).lm_logits[:, config.model.seq_length :, :]
            _, state, first_metadata = _megabyte_forward_with_window_memory(
                model,
                first,
                _MegabyteWindowMemoryState(),
                stitch_global_kv=True,
            )
            stitched_tail, _, second_metadata = _megabyte_forward_with_window_memory(
                model,
                second,
                state,
                stitch_global_kv=True,
            )

        torch.testing.assert_close(stitched_tail, expected_tail, atol=1e-5, rtol=1e-5)
        self.assertEqual(first_metadata["global_memory_layers_used"], 0)
        self.assertEqual(second_metadata["global_memory_layers_used"], config.model.global_layers)
        self.assertEqual(second_metadata["local_memory_used"], 1)

    def test_megabyte_previous_window_memory_adapter_streams_windows(self) -> None:
        torch.manual_seed(11)
        config = self._small_megabyte_config()
        model = build_model(config.model).eval()
        adapter = MegabyteProbabilityAdapter(
            name="megabyte_previous_window",
            model=model,
            config=config,
            device=torch.device("cpu"),
            dtype_name="float32",
            window_context_mode="previous_window_memory",
        )

        result = adapter.unit_probabilities(
            species="test",
            core_sequence="ACGT" * 10,
            unit_size=1,
            batch_size=8,
        )

        self.assertEqual(result.probabilities.shape, (40, 4))
        self.assertIsNotNone(result.metadata)
        assert result.metadata is not None
        self.assertEqual(result.metadata["effective_window_batch_size"], 1)
        self.assertEqual(result.metadata["memory_kv_slots"], 0)
        self.assertEqual(result.metadata["cross_window_memory_summary"], "previous_last_patch")
        self.assertEqual(result.metadata["cross_window_memory_source"], "previous_last_patch_shifted_input")
        self.assertEqual(result.metadata["cross_window_local_conditioning"], "previous_last_patch_shifted_input")
        self.assertEqual(result.metadata["global_memory_windows"], 0)
        self.assertEqual(result.metadata["local_memory_windows"], 2)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, rtol=1e-6)

    def test_megabyte_probe_seq_length_override_controls_real_window_length(self) -> None:
        torch.manual_seed(13)
        config = self._small_megabyte_config()
        model = build_model(config.model).eval()
        adapter = MegabyteProbabilityAdapter(
            name="megabyte_half_window",
            model=model,
            config=config,
            device=torch.device("cpu"),
            dtype_name="float32",
            window_context_mode="previous_window_memory",
            probe_seq_length=8,
        )

        result = adapter.unit_probabilities(
            species="test",
            core_sequence="ACGT" * 10,
            unit_size=1,
            batch_size=8,
        )

        self.assertEqual(result.probabilities.shape, (40, 4))
        self.assertIsNotNone(result.metadata)
        assert result.metadata is not None
        self.assertEqual(result.metadata["model_seq_length_tokens"], 8)
        self.assertEqual(result.metadata["model_window_bases"], 8)
        self.assertEqual(result.metadata["memory_kv_slots"], 0)
        self.assertEqual(result.metadata["memory_extra_prediction_tokens"], 0)
        self.assertEqual(result.metadata["effective_window_batch_size"], 1)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, rtol=1e-6)

    def test_megabyte_paired_previous_window_kv_adapter_streams_pairs(self) -> None:
        torch.manual_seed(14)
        config = self._small_megabyte_config()
        model = build_model(config.model).eval()
        adapter = MegabyteProbabilityAdapter(
            name="megabyte_paired_kv",
            model=model,
            config=config,
            device=torch.device("cpu"),
            dtype_name="float32",
            window_context_mode="paired_previous_window_kv",
            probe_seq_length=8,
        )

        result = adapter.unit_probabilities(
            species="test",
            core_sequence="ACGT" * 10,
            unit_size=1,
            batch_size=8,
        )

        self.assertEqual(result.probabilities.shape, (40, 4))
        self.assertIsNotNone(result.metadata)
        assert result.metadata is not None
        self.assertEqual(result.metadata["effective_window_batch_size"], 1)
        self.assertEqual(result.metadata["memory_kv_slots"], 2)
        self.assertEqual(result.metadata["cross_window_reset"], "paired_windows")
        self.assertEqual(result.metadata["cross_window_memory_summary"], "previous_window_global_kv")
        self.assertEqual(result.metadata["global_memory_windows"], 2)
        self.assertEqual(result.metadata["local_memory_windows"], 2)
        self.assertEqual(result.metadata["global_memory_layer_uses"], 4)
        np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, rtol=1e-6)

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
