from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dna_compress.config import ExperimentConfig
from dna_compress.fused_lm_nc_prefix_codec import compress_fused_lm_nc_prefix_payload
from dna_compress.megabyte_loader import build_model
from dna_compress.probability_trace import (
    ProbabilityTraceReader,
    build_probability_trace_position_index,
    convert_probability_trace_to_position_major,
    fused_depth_major_emit_positions,
    fused_depth_major_row_indices_for_positions,
    fuse_target_probability_traces,
    read_target_probability_trace_positions,
    target_symbols_for_positions,
    validate_trace_compatibility,
    write_target_probability_trace,
)
from dna_compress.tokenization import apply_token_merge_to_model_config
from scripts.run_probability_trace import (
    _megabyte_target_probabilities,
    _megabyte_tokens_with_partial_tail,
    _nc_prefix_target_probabilities,
    _target_trace_arrays_from_position_probs,
)


class ProbabilityTraceTests(unittest.TestCase):
    def test_megabyte_base_trace_keeps_non_divisible_8192_tail(self) -> None:
        sequence = "ACGT" * 2048
        tokens, base_symbols, valid_lengths = _megabyte_tokens_with_partial_tail(
            sequence=sequence,
            token_merge_size=3,
            model_token_alphabet="ACGTN",
        )
        self.assertEqual(tokens.shape[0], 2731)
        self.assertEqual(base_symbols.shape, (2731, 3))
        self.assertEqual(int(valid_lengths[-1]), 2)

        core, probs, symbols, positions = _target_trace_arrays_from_position_probs(
            sequence=sequence,
            target_prob_by_position=np.full((len(sequence),), 0.5, dtype=np.float64),
            window_bases=8192,
        )
        self.assertEqual(core, sequence)
        self.assertEqual(probs.shape[0], len(sequence))
        self.assertEqual(symbols.shape[0], len(sequence))
        np.testing.assert_array_equal(np.sort(positions), np.arange(len(sequence), dtype=np.int64))

    def _write_fake_trace(
        self,
        root: Path,
        name: str,
        *,
        probs: np.ndarray,
        symbols: np.ndarray | None = None,
        shard_rows: int = 3,
    ):
        sequence = "ACGTACGT"
        positions = fused_depth_major_emit_positions(core_base_count=8, window_bases=4, token_merge_size=2)
        target_symbols = (
            target_symbols_for_positions(sequence, positions)
            if symbols is None
            else np.asarray(symbols, dtype=np.int16)
        )
        return write_target_probability_trace(
            root / name,
            model_family=name,
            model_id=name,
            source_payload=sequence.encode("ascii"),
            normalized_sequence=sequence,
            core_sequence=sequence,
            tail_sequence="",
            target_prob=np.asarray(probs, dtype=np.float64),
            target_symbol=target_symbols,
            emit_position=positions,
            window_bases=4,
            token_merge_size=2,
            producer_config={"test": name},
            shard_rows=shard_rows,
            overwrite=True,
        )

    def test_write_read_manifest_and_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probs = np.linspace(0.1, 0.8, 8, dtype=np.float64)
            manifest = self._write_fake_trace(root, "fake", probs=probs, shard_rows=3)
            reader = ProbabilityTraceReader(root / "fake")
            self.assertEqual(reader.manifest.row_count, 8)
            self.assertEqual(reader.manifest.checksum_sha256, manifest.checksum_sha256)
            loaded = np.concatenate([shard["target_prob"] for shard in reader.iter_shards()])
            np.testing.assert_allclose(loaded, probs.astype(np.float32), rtol=1e-6)

    def test_depth_major_position_index_lookup(self) -> None:
        positions = fused_depth_major_emit_positions(core_base_count=10, window_bases=4, token_merge_size=2)
        rows = fused_depth_major_row_indices_for_positions(
            positions,
            core_base_count=10,
            window_bases=4,
            token_merge_size=2,
        )
        np.testing.assert_array_equal(rows, np.arange(positions.shape[0], dtype=np.int64))

    def test_position_index_reads_selected_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probs = np.linspace(0.1, 0.8, 8, dtype=np.float64)
            self._write_fake_trace(root, "fake", probs=probs, shard_rows=3)
            index = build_probability_trace_position_index(root / "fake")
            selected_positions = np.asarray([7, 0, 4, 3], dtype=np.int64)
            result = read_target_probability_trace_positions(root / "fake", selected_positions, index=index)
            all_positions = fused_depth_major_emit_positions(core_base_count=8, window_bases=4, token_merge_size=2)
            rows = fused_depth_major_row_indices_for_positions(
                selected_positions,
                core_base_count=8,
                window_bases=4,
                token_merge_size=2,
            )
            np.testing.assert_array_equal(all_positions[rows], selected_positions)
            np.testing.assert_allclose(result["target_prob"], probs.astype(np.float32)[rows], rtol=1e-6)
            np.testing.assert_array_equal(result["emit_position"], selected_positions)

    def test_compatibility_reports_target_symbol_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            probs = np.full((8,), 0.5, dtype=np.float64)
            left = self._write_fake_trace(root, "left", probs=probs)
            symbols = target_symbols_for_positions("ACGTACGT", fused_depth_major_emit_positions(core_base_count=8, window_bases=4, token_merge_size=2))
            symbols[0] = (int(symbols[0]) + 1) % 4
            right = self._write_fake_trace(root, "right", probs=probs, symbols=symbols)
            diffs = validate_trace_compatibility(left, right)
            self.assertIn("target_symbols_sha256", {item["field"] for item in diffs})

    def test_offline_fusion_is_independent_of_shard_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left_probs = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float64)
            right_probs = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
            self._write_fake_trace(root, "left", probs=left_probs, shard_rows=3)
            self._write_fake_trace(root, "right", probs=right_probs, shard_rows=5)
            first = fuse_target_probability_traces(root / "left", root / "right", fusion_eta=0.05)

            self._write_fake_trace(root, "left_again", probs=left_probs, shard_rows=8)
            self._write_fake_trace(root, "right_again", probs=right_probs, shard_rows=2)
            second = fuse_target_probability_traces(root / "left_again", root / "right_again", fusion_eta=0.05)
            self.assertAlmostEqual(first["core_model_theoretical_bits"], second["core_model_theoretical_bits"], places=6)
            self.assertAlmostEqual(first["fusion_final_mean_lm_weight"], second["fusion_final_mean_lm_weight"], places=6)
            self.assertIsNone(first["arithmetic_bits_per_base"])
            self.assertEqual(first["decodable_design"], "target_probability_trace_non_arithmetic")

    def test_position_major_conversion_preserves_lookup_and_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            left_probs = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], dtype=np.float64)
            right_probs = np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
            self._write_fake_trace(root, "left", probs=left_probs, shard_rows=3)
            self._write_fake_trace(root, "right", probs=right_probs, shard_rows=5)
            depth_major = fuse_target_probability_traces(root / "left", root / "right", fusion_eta=0.05)

            left_manifest = convert_probability_trace_to_position_major(
                root / "left",
                root / "left_position_major",
                overwrite=True,
            )
            right_manifest = convert_probability_trace_to_position_major(
                root / "right",
                root / "right_position_major",
                overwrite=True,
            )
            self.assertEqual(left_manifest.emission_order, "position_major_v1")
            self.assertFalse(validate_trace_compatibility(left_manifest, right_manifest))

            selected_positions = np.asarray([7, 0, 4, 3], dtype=np.int64)
            old_rows = read_target_probability_trace_positions(root / "left", selected_positions)
            new_rows = read_target_probability_trace_positions(root / "left_position_major", selected_positions)
            np.testing.assert_allclose(old_rows["target_prob"], new_rows["target_prob"], rtol=0, atol=0)
            np.testing.assert_array_equal(old_rows["target_symbol"], new_rows["target_symbol"])
            np.testing.assert_array_equal(new_rows["row_index"], selected_positions)

            position_major = fuse_target_probability_traces(
                root / "left_position_major",
                root / "right_position_major",
                fusion_eta=0.05,
            )
            self.assertAlmostEqual(
                depth_major["core_model_theoretical_bits"],
                position_major["core_model_theoretical_bits"],
                places=6,
            )
            self.assertAlmostEqual(
                depth_major["fusion_final_mean_lm_weight"],
                position_major["fusion_final_mean_lm_weight"],
                places=6,
            )

    def _small_megabyte_config(self) -> ExperimentConfig:
        config = ExperimentConfig()
        config.model.implementation = "megabyte_in_action"
        config.model.patch_size = 4
        config.model.global_dim = 8
        config.model.local_dim = 8
        config.model.seq_length = 4
        config.model.global_heads = 2
        config.model.global_layers = 1
        config.model.local_heads = 2
        config.model.local_layers = 1
        config.model.flash_attn = False
        config.model.nugget_enabled = False
        config.train.dtype = "float32"
        config.data.token_merge_size = 2
        config.data.token_merge_alphabet = "ACGT"
        apply_token_merge_to_model_config(config.model, config.data)
        return config

    def test_small_model_offline_trace_fusion_matches_live_diagnostics(self) -> None:
        torch.manual_seed(21)
        config = self._small_megabyte_config()
        model = build_model(config.model).eval()
        payload = b"ACGTACGTACGTACGT"
        device = torch.device("cpu")

        live = compress_fused_lm_nc_prefix_payload(
            model=model,
            config=config,
            payload=payload,
            device=device,
            dtype_name="float32",
            batch_size="all",
            nc_prefix_window_bases=8,
            nc_prefix_min_windows=1,
            nc_prefix_hash_bucket_count=0,
            nc_prefix_geco2_level=10,
            fusion_eta=0.05,
            fusion_initial_lm_weight=0.5,
            encode_arithmetic=False,
            collect_diagnostics=True,
            include_codec_baselines=False,
            pipeline_mode="streaming_token_strict",
        )

        from dna_compress.fused_lm_nc_prefix_codec import MegabyteStreamingAdapter

        adapter = MegabyteStreamingAdapter(model=model, config=config)
        sequence, core, lm_prob, lm_symbols, lm_positions, _ = _megabyte_target_probabilities(
            adapter=adapter,
            payload=payload,
            device=device,
            dtype_name="float32",
            batch_size="all",
            window_bases=8,
            batch_fallback=1,
        )
        nc_sequence, nc_core, nc_prob, nc_symbols, nc_positions, _ = _nc_prefix_target_probabilities(
            payload=payload,
            window_bases=8,
            token_merge_size=2,
            backend="fast_cpp",
            min_windows=1,
            hash_bucket_count=0,
            geco2_level=10,
        )
        self.assertEqual(sequence, nc_sequence)
        self.assertEqual(core, nc_core)
        np.testing.assert_array_equal(lm_symbols, nc_symbols)
        np.testing.assert_array_equal(lm_positions, nc_positions)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_target_probability_trace(
                root / "megabyte",
                model_family="megabyte",
                model_id="small",
                source_payload=payload,
                normalized_sequence=sequence,
                core_sequence=core,
                tail_sequence=sequence[len(core) :],
                target_prob=lm_prob,
                target_symbol=lm_symbols,
                emit_position=lm_positions,
                window_bases=8,
                token_merge_size=2,
                producer_config={},
                overwrite=True,
            )
            write_target_probability_trace(
                root / "nc_prefix",
                model_family="nc_prefix",
                model_id="small",
                source_payload=payload,
                normalized_sequence=nc_sequence,
                core_sequence=nc_core,
                tail_sequence=nc_sequence[len(nc_core) :],
                target_prob=nc_prob,
                target_symbol=nc_symbols,
                emit_position=nc_positions,
                window_bases=8,
                token_merge_size=2,
                producer_config={},
                overwrite=True,
            )
            offline = fuse_target_probability_traces(root / "megabyte", root / "nc_prefix")

        self.assertAlmostEqual(
            float(offline["core_theoretical_bits_per_base"]),
            float(live["core_theoretical_bits_per_base"]),
            places=5,
        )
        self.assertAlmostEqual(
            float(offline["fusion_final_mean_lm_weight"]),
            float(live["fusion_final_mean_lm_weight"]),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
