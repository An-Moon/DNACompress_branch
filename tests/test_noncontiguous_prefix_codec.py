from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dna_compress.fast_nc_prefix import load_fast_nc_prefix_extension
from dna_compress.noncontiguous_prefix_codec import (
    NoncontiguousPrefixConfig,
    NoncontiguousPrefixProbabilityAdapter,
    compress_noncontiguous_prefix_sequence,
    compute_noncontiguous_prefix_probabilities,
)
from scripts.run_dna_region_bpb_probe import build_region_adapters, run_probe_for_source


class NoncontiguousPrefixCodecTests(unittest.TestCase):
    def test_same_depth_predictions_do_not_see_same_depth_targets(self) -> None:
        result = compute_noncontiguous_prefix_probabilities(
            "AG",
            NoncontiguousPrefixConfig(window_bases=1, alphabet="ACGT", min_windows=1),
        )
        np.testing.assert_allclose(result.probabilities[0], np.full((4,), 0.25), rtol=1e-12)
        np.testing.assert_allclose(result.probabilities[1], np.full((4,), 0.25), rtol=1e-12)
        self.assertEqual(result.emit_order.tolist(), [0, 1])

    def test_fast_cpp_is_deterministic_and_summary_matches(self) -> None:
        try:
            load_fast_nc_prefix_extension()
        except Exception as error:
            self.skipTest(f"fast nc_prefix extension is unavailable: {error}")

        config = NoncontiguousPrefixConfig(window_bases=4, alphabet="ACGT", min_windows=1, backend="fast_cpp")
        sequence = "ACGTACGTACGTACGT"
        first = compute_noncontiguous_prefix_probabilities(sequence, config)
        second = compute_noncontiguous_prefix_probabilities(sequence, config)
        np.testing.assert_allclose(first.probabilities, second.probabilities, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.bpb, second.bpb, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(first.emit_order, second.emit_order)
        self.assertEqual(first.metadata["backend"], "fast_cpp")
        self.assertEqual(first.metadata["update_mode"], "cache_pipeline")
        self.assertFalse(first.metadata["fine_timing_enabled"])
        self.assertEqual(len(first.metadata["weight_history"]), 0)

        summary = compute_noncontiguous_prefix_probabilities(
            sequence,
            config,
            return_probabilities=False,
            summary_only=True,
        )
        self.assertEqual(summary.probabilities.shape, (0, 4))
        self.assertEqual(summary.bpb.shape, (0,))
        self.assertAlmostEqual(
            summary.metadata["theoretical_bits_per_base"],
            first.metadata["theoretical_bits_per_base"],
            places=12,
        )
        self.assertEqual(summary.metadata["artifact_tensor_bytes"], 0)

    def test_cache_pipeline_metadata_and_hash_bucket_config(self) -> None:
        sequence = ("ACGTGCAATTCG" * 64)[:640]
        common = {
            "window_bases": 8,
            "alphabet": "ACGT",
            "min_windows": 1,
            "backend": "fast_cpp",
        }
        for block_windows in (64, 256):
            with patch.dict(
                "os.environ",
                {"DNA_COMPRESS_NC_PREFIX_PIPELINE_BLOCK_WINDOWS": str(block_windows)},
            ):
                result = compute_noncontiguous_prefix_probabilities(
                    sequence,
                    NoncontiguousPrefixConfig(**common),
                    return_probabilities=False,
                    summary_only=True,
                )
            self.assertEqual(result.metadata["pipeline_block_windows"], block_windows)
            self.assertEqual(result.metadata["hash_bucket_count_requested"], 0)
            self.assertEqual(result.metadata["hash_bucket_count"], 33554471)
            self.assertEqual(result.metadata["large_table_alignment_bytes"], 2 * 1024 * 1024)
            self.assertTrue(result.metadata["models"][-1]["storage_2mb_aligned"])

        custom = compute_noncontiguous_prefix_probabilities(
            sequence,
            NoncontiguousPrefixConfig(**common, hash_bucket_count=1024),
            return_probabilities=False,
            summary_only=True,
        )
        self.assertEqual(custom.metadata["hash_bucket_count_requested"], 1024)
        self.assertEqual(custom.metadata["hash_bucket_count"], 1024)

    def test_fine_timing_is_opt_in(self) -> None:
        sequence = "ACGTACGTACGT"
        config = NoncontiguousPrefixConfig(window_bases=4, alphabet="ACGT", min_windows=1, backend="fast_cpp")
        with patch.dict("os.environ", {"DNA_COMPRESS_NC_PREFIX_PROFILE_TIMING": "1"}):
            result = compute_noncontiguous_prefix_probabilities(sequence, config, return_probabilities=False)
        self.assertTrue(result.metadata["fine_timing_enabled"])
        self.assertGreater(result.metadata["timing"]["timed_stage_seconds"], 0.0)
        self.assertGreater(len(result.metadata["weight_history"]), 0)

    def test_compression_theoretical_bits_matches_per_base_bpb(self) -> None:
        config = NoncontiguousPrefixConfig(window_bases=4, alphabet="ACGT", min_windows=1)
        result = compute_noncontiguous_prefix_probabilities("ACGTACGTACGT", config)
        metrics = compress_noncontiguous_prefix_sequence("ACGTACGTACGT", config)
        self.assertAlmostEqual(float(result.bpb.sum()), float(metrics["theoretical_bits"]), places=9)
        self.assertEqual(int(metrics["emitted_arithmetic_symbol_count"]), 12)
        self.assertGreater(int(metrics["arithmetic_coded_bytes"]), 0)

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_noncontiguous_prefix_probabilities("AC", NoncontiguousPrefixConfig(window_bases=0, min_windows=1))
        with self.assertRaises(ValueError):
            compute_noncontiguous_prefix_probabilities("AX", NoncontiguousPrefixConfig(alphabet="ACGT", min_windows=1))
        with self.assertRaises(ValueError):
            compute_noncontiguous_prefix_probabilities(
                "ACGT",
                NoncontiguousPrefixConfig(window_bases=4, alphabet="ACGT", min_windows=1, hash_bucket_count=-1),
            )
        adapter = NoncontiguousPrefixProbabilityAdapter(alphabet="ACGT", min_windows=1)
        with self.assertRaises(ValueError):
            adapter.unit_probabilities(species="x", core_sequence="ACGT", unit_size=2, batch_size=1)

    def test_default_min_windows_rejects_small_region(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_required_bases=25165824"):
            compute_noncontiguous_prefix_probabilities("ACGT" * 12500, NoncontiguousPrefixConfig())

    def test_region_probe_builds_nc_prefix_adapter_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.txt"
            source_path.write_text("ACGTACGTACGT", encoding="ascii")
            args = SimpleNamespace(
                device="cpu",
                model=["nc_prefix"],
                run_dir=None,
                model_kind="nc_prefix",
                checkpoint=None,
                checkpoint_tag="best",
                nc_prefix_window_bases=4,
                nc_prefix_min_windows=1,
                nc_prefix_backend="fast_cpp",
                nc_prefix_hash_bucket_count=1024,
                alphabet="ACGT",
                region_start=0,
                region_bases=12,
                random_region=False,
                seed=1,
                batch_size=2,
                plot_window_bases=4,
                smooth_window_bases=2,
                model_window_smooth_bases=2,
                plot_max_points=1000,
                plot_individual_windows=False,
                max_individual_window_plots=1,
                record_dtype="float32",
                write_per_base_csv=False,
                compute_only=True,
            )
            adapters = build_region_adapters(args)
            self.assertEqual(adapters[0].name, "nc_prefix")
            result = run_probe_for_source(
                args,
                source_info={
                    "dataset": "dnacorpus",
                    "source": "fake_source",
                    "species": "Fake",
                    "paths": [source_path],
                    "fasta": False,
                    "alphabet": "ACGT",
                },
                adapters=adapters,
                output_dir=root / "out",
            )
            self.assertIn("nc_prefix", result["models"])
            self.assertTrue((root / "out" / "models" / "nc_prefix" / "bpb.npz").exists())
            metadata = result["models"]["nc_prefix"]["metadata"]
            self.assertEqual(metadata["nc_prefix"]["window_bases"], 4)
            self.assertEqual(metadata["nc_prefix"]["hash_bucket_count"], 1024)


if __name__ == "__main__":
    unittest.main()
