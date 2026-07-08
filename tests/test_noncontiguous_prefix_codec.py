from __future__ import annotations

import tempfile
import shutil
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dna_compress.noncontiguous_prefix_codec import (
    NoncontiguousPrefixConfig,
    NoncontiguousPrefixProbabilityAdapter,
    compress_noncontiguous_prefix_sequence,
    compute_noncontiguous_prefix_probabilities,
)
from dna_compress.fast_nc_prefix import load_fast_nc_prefix_extension
from scripts.run_dna_region_bpb_probe import build_region_adapters, run_probe_for_source


class NoncontiguousPrefixCodecTests(unittest.TestCase):
    def test_same_depth_predictions_do_not_see_same_depth_targets(self) -> None:
        config = NoncontiguousPrefixConfig(
            window_bases=1,
            alphabet="ACGT",
            min_windows=1,
        )
        result = compute_noncontiguous_prefix_probabilities("AG", config)
        np.testing.assert_allclose(result.probabilities[0], np.full((4,), 0.25), rtol=1e-12)
        np.testing.assert_allclose(result.probabilities[1], np.full((4,), 0.25), rtol=1e-12)
        self.assertEqual(result.emit_order.tolist(), [0, 1])

    def test_probability_simulation_is_deterministic_for_decoder_replay(self) -> None:
        config = NoncontiguousPrefixConfig(
            window_bases=3,
            alphabet="ACGT",
            min_windows=1,
        )
        first = compute_noncontiguous_prefix_probabilities("ACGTACGTAC", config)
        second = compute_noncontiguous_prefix_probabilities("ACGTACGTAC", config)
        np.testing.assert_allclose(first.probabilities, second.probabilities, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(first.emit_order, second.emit_order)

    def test_fast_cpp_is_deterministic_on_small_sequence(self) -> None:
        try:
            load_fast_nc_prefix_extension()
        except Exception as error:
            self.skipTest(f"fast nc_prefix extension is unavailable: {error}")

        kwargs = dict(
            window_bases=4,
            alphabet="ACGT",
            min_windows=1,
        )
        sequence = "ACGTACGTACGTACGT"
        first = compute_noncontiguous_prefix_probabilities(
            sequence,
            NoncontiguousPrefixConfig(**kwargs, backend="fast_cpp"),
        )
        second = compute_noncontiguous_prefix_probabilities(
            sequence,
            NoncontiguousPrefixConfig(**kwargs, backend="fast_cpp"),
        )
        np.testing.assert_allclose(first.probabilities, second.probabilities, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.bpb, second.bpb, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(first.emit_order, second.emit_order)
        self.assertEqual(first.metadata["backend"], "fast_cpp")

    def test_fast_cpp_summary_without_probability_matrix_matches_bpb(self) -> None:
        try:
            load_fast_nc_prefix_extension()
        except Exception as error:
            self.skipTest(f"fast nc_prefix extension is unavailable: {error}")

        config = NoncontiguousPrefixConfig(
            window_bases=4,
            alphabet="ACGT",
            min_windows=1,
            backend="fast_cpp",
        )
        full = compute_noncontiguous_prefix_probabilities("ACGTACGTACGT", config)
        summary = compute_noncontiguous_prefix_probabilities("ACGTACGTACGT", config, return_probabilities=False)
        self.assertEqual(summary.probabilities.shape, (0, 4))
        np.testing.assert_allclose(summary.bpb, full.bpb, rtol=1e-12, atol=1e-12)

    def test_nc_prefix_is_acgt_only_and_deterministic(self) -> None:
        try:
            load_fast_nc_prefix_extension()
        except Exception as error:
            self.skipTest(f"fast nc_prefix extension is unavailable: {error}")

        config = NoncontiguousPrefixConfig(
            window_bases=4,
            alphabet="ACGT",
            min_windows=1,
            backend="fast_cpp",
        )
        sequence = "ACGTACGTACGTACGT"
        first = compute_noncontiguous_prefix_probabilities(sequence, config)
        second = compute_noncontiguous_prefix_probabilities(sequence, config)
        np.testing.assert_allclose(first.probabilities, second.probabilities, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(first.bpb, second.bpb, rtol=0.0, atol=0.0)
        self.assertEqual(first.metadata["preset"], "nc_prefix")
        self.assertEqual(first.metadata["predictor_count"], 8)
        self.assertEqual(first.metadata["alphabet"], "ACGT")

        summary = compute_noncontiguous_prefix_probabilities(sequence, config, return_probabilities=False)
        self.assertEqual(summary.probabilities.shape, (0, 4))
        np.testing.assert_allclose(summary.bpb, first.bpb, rtol=1e-12, atol=1e-12)

        with self.assertRaisesRegex(ValueError, "alphabet='ACGT'"):
            compute_noncontiguous_prefix_probabilities(
                sequence + "N",
                NoncontiguousPrefixConfig(
                    window_bases=4,
                    alphabet="ACGTN",
                    min_windows=1,
                    backend="fast_cpp",
                ),
            )

    def test_nc_prefix_matches_geco2_estimate_on_random_acgt(self) -> None:
        geco2 = shutil.which("GeCo2") or "/home/Liang_junnan/miniconda3/bin/GeCo2"
        if not Path(geco2).exists():
            self.skipTest("GeCo2 binary is unavailable")
        try:
            load_fast_nc_prefix_extension()
        except Exception as error:
            self.skipTest(f"fast nc_prefix extension is unavailable: {error}")

        rng = np.random.default_rng(7)
        alphabet = np.asarray(list("ACGT"))
        sequence = "".join(alphabet[rng.integers(0, 4, size=2000)].tolist())
        with tempfile.TemporaryDirectory() as tmpdir:
            seq_path = Path(tmpdir) / "seq.txt"
            seq_path.write_text(sequence, encoding="ascii")
            subprocess.run(
                [geco2, "-F", "-e", "-l", "10", str(seq_path)],
                cwd=tmpdir,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            iae = np.asarray([float(item) for item in (Path(str(seq_path) + ".iae")).read_text().split()], dtype=np.float64)

        result = compute_noncontiguous_prefix_probabilities(
            sequence,
            NoncontiguousPrefixConfig(
                window_bases=len(sequence),
                alphabet="ACGT",
                min_windows=1,
                backend="fast_cpp",
            ),
            return_probabilities=False,
        )
        self.assertEqual(result.metadata["preset"], "nc_prefix")
        self.assertEqual(result.metadata["algorithm"], "geco2_level10_per_window_weights")
        self.assertEqual(result.metadata["fusion_mode"], "geco2_serial_weight_power_times_model_probability")
        self.assertLess(abs(float(result.metadata["geco2_quantized_bits_per_base"]) - float(iae.mean())), 1e-3)
        np.testing.assert_allclose(result.bpb[:16], iae[:16], rtol=0.0, atol=1.1e-2)

    def test_nc_prefix_parallel_uses_per_window_weights(self) -> None:
        try:
            load_fast_nc_prefix_extension()
        except Exception as error:
            self.skipTest(f"fast nc_prefix extension is unavailable: {error}")

        result = compute_noncontiguous_prefix_probabilities(
            "ACGT" * 8,
            NoncontiguousPrefixConfig(
                window_bases=4,
                alphabet="ACGT",
                min_windows=1,
                backend="fast_cpp",
            ),
            return_probabilities=False,
        )
        self.assertEqual(result.metadata["window_count"], 8)
        self.assertEqual(result.metadata["weight_scope"], "per_window_local_weights")
        self.assertEqual(
            result.metadata["fusion_mode"],
            "geco2_per_window_weight_power_times_model_probability_depth_major_shared_counters",
        )

    def test_compression_theoretical_bits_matches_per_base_bpb(self) -> None:
        config = NoncontiguousPrefixConfig(
            window_bases=4,
            alphabet="ACGT",
            min_windows=1,
        )
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


if __name__ == "__main__":
    unittest.main()
