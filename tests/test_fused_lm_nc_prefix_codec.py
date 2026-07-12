from __future__ import annotations

import unittest

import numpy as np
import torch

from dna_compress.config import ExperimentConfig
from dna_compress.fused_lm_nc_prefix_codec import (
    _regular_log_probs_to_base_steps,
    compress_fused_lm_nc_prefix_payload,
)
from dna_compress.fast_nc_prefix import FusedNcPrefixStreamingEncoder
from dna_compress.megabyte_loader import build_model
from dna_compress.noncontiguous_prefix_codec import (
    NoncontiguousPrefixConfig,
    compute_noncontiguous_prefix_probabilities,
)
from dna_compress.tokenization import apply_token_merge_to_model_config


class FusedLmNcPrefixCodecTests(unittest.TestCase):
    def _small_megabyte_config(self, *, token_merge_size: int = 2) -> ExperimentConfig:
        config = ExperimentConfig()
        config.model.implementation = "megabyte_in_action"
        config.model.patch_size = 4
        config.model.global_dim = 8
        config.model.local_dim = 8
        config.model.seq_length = 8
        config.model.global_heads = 2
        config.model.global_layers = 1
        config.model.local_heads = 2
        config.model.local_layers = 1
        config.model.flash_attn = False
        config.model.nugget_enabled = False
        config.train.dtype = "float32"
        config.train.eval_batch_size = 2
        config.data.token_merge_size = int(token_merge_size)
        config.data.token_merge_alphabet = "ACGT"
        apply_token_merge_to_model_config(config.model, config.data)
        return config

    def test_token_probability_factorization_matches_regular_token_probability(self) -> None:
        torch.manual_seed(123)
        for merge_size in (1, 2, 3):
            vocab = 4**merge_size + 2
            logits = torch.randn(5, vocab)
            target_chunks = torch.randint(0, 4, (5, merge_size), dtype=torch.long)
            weights = torch.tensor([4**power for power in range(merge_size - 1, -1, -1)], dtype=torch.long)
            targets = (target_chunks * weights).sum(dim=1)

            steps = _regular_log_probs_to_base_steps(
                logits,
                target_chunks,
                token_merge_size=merge_size,
                model_token_alphabet="ACGT",
                output_alphabet="ACGT",
                model_uses_ascii_tokens=False,
            )

            log_probs = torch.log_softmax(logits.float(), dim=1)
            regular_probs = log_probs[:, : 4**merge_size].exp()
            expected = regular_probs[torch.arange(targets.numel()), targets] / regular_probs.sum(dim=1)
            actual = torch.ones_like(expected)
            rows = torch.arange(targets.numel())
            for index, step in enumerate(steps):
                actual = actual * step[rows, target_chunks[:, index]]

            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_token_probability_factorization_supports_acgtn_model_tokens(self) -> None:
        torch.manual_seed(456)
        merge_size = 3
        model_alphabet = "ACGTN"
        vocab = len(model_alphabet) ** merge_size + 2
        logits = torch.randn(4, vocab)
        output_digits = torch.randint(0, 4, (4, merge_size), dtype=torch.long)
        weights = torch.tensor([25, 5, 1], dtype=torch.long)
        targets = (output_digits * weights).sum(dim=1)

        steps = _regular_log_probs_to_base_steps(
            logits,
            output_digits,
            token_merge_size=merge_size,
            model_token_alphabet=model_alphabet,
            output_alphabet="ACGT",
            model_uses_ascii_tokens=False,
        )

        regular = torch.log_softmax(logits.float(), dim=1)[:, : 5**merge_size].exp().reshape(4, 5, 5, 5)
        acgt = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        acgt_mass = regular.index_select(1, acgt).index_select(2, acgt).index_select(3, acgt)
        expected = acgt_mass[
            torch.arange(4),
            output_digits[:, 0],
            output_digits[:, 1],
            output_digits[:, 2],
        ] / acgt_mass.sum(dim=(1, 2, 3))
        actual = torch.ones_like(expected)
        rows = torch.arange(4)
        for index, step in enumerate(steps):
            actual = actual * step[rows, output_digits[:, index]]

        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_acgtn_merge3_fast_factorization_matches_generic(self) -> None:
        torch.manual_seed(457)
        logits = torch.randn(6, 127)
        target_base_symbols = torch.randint(0, 4, (6, 3), dtype=torch.long)
        fast = _regular_log_probs_to_base_steps(
            logits,
            target_base_symbols,
            token_merge_size=3,
            model_token_alphabet="ACGTN",
            output_alphabet="ACGT",
            model_uses_ascii_tokens=False,
        )
        generic = _regular_log_probs_to_base_steps(
            logits,
            target_base_symbols,
            token_merge_size=3,
            model_token_alphabet="ACGTN",
            output_alphabet="ACGT",
            model_uses_ascii_tokens=False,
            force_generic=True,
        )

        for actual, expected in zip(fast, generic):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_fused_skip_arithmetic_matches_arithmetic_theoretical_bits(self) -> None:
        torch.manual_seed(321)
        config = self._small_megabyte_config(token_merge_size=2)
        model = build_model(config.model).eval()
        payload = ("ACGT" * 16).encode("ascii")

        common = dict(
            model=model,
            config=config,
            payload=payload,
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size="auto",
            nc_prefix_window_bases=16,
            nc_prefix_min_windows=1,
            nc_prefix_hash_bucket_count=1024,
            fusion_eta=0.05,
            fusion_initial_lm_weight=0.5,
        )
        without_arithmetic = compress_fused_lm_nc_prefix_payload(
            **common,
            encode_arithmetic=False,
        )
        with_arithmetic = compress_fused_lm_nc_prefix_payload(
            **common,
            encode_arithmetic=True,
        )

        self.assertEqual(without_arithmetic["core_base_count"], 64)
        self.assertEqual(with_arithmetic["core_base_count"], 64)
        self.assertAlmostEqual(
            without_arithmetic["core_theoretical_bits_per_base"],
            with_arithmetic["core_theoretical_bits_per_base"],
            places=10,
        )
        self.assertIsNotNone(with_arithmetic["arithmetic_bits_per_base"])
        self.assertEqual(with_arithmetic["emitted_arithmetic_symbol_count"], 64)

    def test_native_split_encoder_requires_predict_then_update(self) -> None:
        encoder = FusedNcPrefixStreamingEncoder(
            window_count=2,
            window_bases=4,
            hash_bucket_count=1024,
            arithmetic_frequency_total=65536,
            fusion_eta=0.05,
            initial_lm_weight=0.5,
            encode_arithmetic=False,
            collect_diagnostics=True,
        )
        lm_probs = torch.full((2, 4), 0.25, dtype=torch.float32)
        targets = torch.tensor([0, 1], dtype=torch.int16)

        predicted = encoder.predict_base_step(2)
        self.assertEqual(predicted["active_windows"], 2)
        with self.assertRaisesRegex(RuntimeError, "predict_base_step"):
            encoder.predict_base_step(2)

        updated = encoder.fuse_encode_update_base_step(lm_probs, targets)
        self.assertEqual(updated["active_windows"], 2)
        with self.assertRaisesRegex(RuntimeError, "requires a pending"):
            encoder.fuse_encode_update_base_step(lm_probs, targets)

        finished = encoder.finish()
        self.assertEqual(finished["base_count"], 2)

    def test_nc_only_diagnostic_matches_standalone_nc_prefix(self) -> None:
        torch.manual_seed(777)
        config = self._small_megabyte_config(token_merge_size=2)
        model = build_model(config.model).eval()
        sequence = "ACGT" * 16
        fused = compress_fused_lm_nc_prefix_payload(
            model=model,
            config=config,
            payload=sequence.encode("ascii"),
            device=torch.device("cpu"),
            dtype_name="float32",
            batch_size="auto",
            nc_prefix_window_bases=16,
            nc_prefix_min_windows=1,
            nc_prefix_hash_bucket_count=1024,
            encode_arithmetic=False,
        )
        standalone = compute_noncontiguous_prefix_probabilities(
            sequence,
            NoncontiguousPrefixConfig(
                window_bases=16,
                alphabet="ACGT",
                min_windows=1,
                hash_bucket_count=1024,
            ),
            return_probabilities=True,
        )

        expected = float(standalone.metadata["theoretical_bits"]) / len(sequence)
        self.assertTrue(np.isfinite(fused["nc_prefix_only_theoretical_bits_per_base"]))
        self.assertAlmostEqual(fused["nc_prefix_only_theoretical_bits_per_base"], expected, places=10)

    def test_streaming_v2_matches_matrix_debug_fused_bits(self) -> None:
        torch.manual_seed(888)
        config = self._small_megabyte_config(token_merge_size=2)
        model = build_model(config.model).eval()
        payload = ("ACGT" * 16).encode("ascii")
        common = dict(
            model=model,
            config=config,
            payload=payload,
            device=torch.device("cpu"),
            dtype_name="float32",
            nc_prefix_window_bases=16,
            nc_prefix_min_windows=1,
            nc_prefix_hash_bucket_count=1024,
            encode_arithmetic=False,
        )
        streaming = compress_fused_lm_nc_prefix_payload(
            **common,
            batch_size="auto",
            pipeline_mode="streaming_v2",
        )
        streaming_v3 = compress_fused_lm_nc_prefix_payload(
            **common,
            batch_size="auto",
            pipeline_mode="streaming_v3",
        )
        matrix = compress_fused_lm_nc_prefix_payload(
            **common,
            batch_size=4,
            pipeline_mode="matrix_debug",
        )
        self.assertAlmostEqual(
            streaming["core_theoretical_bits_per_base"],
            matrix["core_theoretical_bits_per_base"],
            places=8,
        )
        self.assertAlmostEqual(
            streaming_v3["core_theoretical_bits_per_base"],
            matrix["core_theoretical_bits_per_base"],
            places=8,
        )
        self.assertEqual(streaming_v3["pipeline_mode"], "streaming_v3")
        self.assertIsNotNone(streaming_v3["nc_predict_seconds"])
        self.assertIsNotNone(streaming_v3["fusion_update_seconds"])


if __name__ == "__main__":
    unittest.main()
