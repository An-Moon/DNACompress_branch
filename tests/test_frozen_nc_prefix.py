from __future__ import annotations

import unittest

import numpy as np
import torch

from dna_compress.fast_nc_prefix import FusedNcPrefixStreamingEncoder


BASE_TO_SYMBOL = {"A": 0, "C": 1, "G": 2, "T": 3}


def symbols(sequences: list[str]) -> torch.Tensor:
    return torch.tensor(
        [[BASE_TO_SYMBOL[base] for base in sequence] for sequence in sequences],
        dtype=torch.int16,
    )


def frozen_target_probabilities(background: list[str], targets: list[str], *, specialized_training: bool = True):
    window_bases = len(background[0])
    encoder = FusedNcPrefixStreamingEncoder(
        window_count=len(background),
        window_bases=window_bases,
        hash_bucket_count=256,
        geco2_level=10,
        arithmetic_frequency_total=65536,
        fusion_eta=0.05,
        initial_lm_weight=0.0,
        encode_arithmetic=False,
        collect_diagnostics=False,
    )
    background_symbols = symbols(background)
    if specialized_training:
        encoder.train_background_token_step(background_symbols)
    else:
        uniform_background = torch.full(
            (len(background), window_bases, 4), 0.25, dtype=torch.float64
        )
        encoder.encode_token_step(uniform_background, background_symbols)
    frozen = encoder.freeze_background_and_reset_targets(len(targets))
    target_symbols = symbols(targets)
    uniform_targets = torch.full((len(targets), window_bases, 4), 0.25, dtype=torch.float64)
    result = encoder.encode_token_step_collect_targets(uniform_targets, target_symbols)
    finished = encoder.finish()
    return result["nc_target_probabilities"].numpy(), frozen, finished


def frozen_full_probabilities(background: list[str], targets: list[str]):
    window_bases = len(background[0])
    encoder = FusedNcPrefixStreamingEncoder(
        window_count=len(background),
        window_bases=window_bases,
        hash_bucket_count=256,
        geco2_level=10,
        arithmetic_frequency_total=65536,
        fusion_eta=0.05,
        initial_lm_weight=0.0,
        encode_arithmetic=False,
        collect_diagnostics=False,
    )
    encoder.train_background_token_step(symbols(background))
    encoder.freeze_background_and_reset_targets(len(targets))
    target_symbols = symbols(targets)
    uniform_targets = torch.full((len(targets), window_bases, 4), 0.25, dtype=torch.float64)
    result = encoder.encode_token_step_collect_probabilities(uniform_targets, target_symbols)
    encoder.finish()
    return result["nc_probabilities"].numpy(), result["nc_target_probabilities"].numpy()


class FrozenNcPrefixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.background = [
            "ACGT" * 8,
            "AAAAACCCCCGGGGGTTTTTACGTACGTACGT",
            "TGCATGCA" * 4,
            "CCCCGGGG" * 4,
        ]
        self.reference = "ACGT" * 8
        self.variant = self.reference[:16] + "T" + self.reference[17:]

    def test_freeze_resets_accounting_and_records_background(self) -> None:
        probabilities, frozen, finished = frozen_target_probabilities(
            self.background, [self.reference, self.variant]
        )
        self.assertEqual(probabilities.shape, (2, 32))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertEqual(frozen["background_window_count"], 4)
        self.assertEqual(frozen["background_base_count"], 128)
        self.assertTrue(frozen["frozen_counters"])
        self.assertEqual(finished["base_count"], 64)
        self.assertTrue(finished["model_metadata"]["frozen_counters"])

    def test_targets_do_not_change_each_others_frozen_predictions(self) -> None:
        paired, _, _ = frozen_target_probabilities(
            self.background, [self.reference, self.variant]
        )
        reference_alone, _, _ = frozen_target_probabilities(self.background, [self.reference])
        reordered, _, _ = frozen_target_probabilities(
            self.background, [self.variant, self.reference]
        )
        np.testing.assert_array_equal(paired[0], reference_alone[0])
        np.testing.assert_array_equal(paired, reordered[::-1])

    def test_variant_cannot_affect_pre_observation_predictions(self) -> None:
        paired, _, _ = frozen_target_probabilities(
            self.background, [self.reference, self.variant]
        )
        np.testing.assert_array_equal(paired[0, :16], paired[1, :16])

    def test_specialized_background_training_matches_generic_path(self) -> None:
        specialized, _, _ = frozen_target_probabilities(
            self.background, [self.reference, self.variant], specialized_training=True
        )
        generic, _, _ = frozen_target_probabilities(
            self.background, [self.reference, self.variant], specialized_training=False
        )
        np.testing.assert_array_equal(specialized, generic)

    def test_full_probabilities_are_normalized_and_match_target_trace(self) -> None:
        full, target = frozen_full_probabilities(
            self.background, [self.reference, self.variant]
        )
        self.assertEqual(full.shape, (2, 32, 4))
        np.testing.assert_allclose(full.sum(axis=2), 1.0, rtol=0.0, atol=1e-15)
        target_symbols = symbols([self.reference, self.variant]).numpy()
        gathered = np.take_along_axis(full, target_symbols[:, :, None], axis=2)[:, :, 0]
        np.testing.assert_array_equal(gathered, target)
        np.testing.assert_array_equal(full[0, :17], full[1, :17])


if __name__ == "__main__":
    unittest.main()
