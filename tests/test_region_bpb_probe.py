from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dna_region_bpb_probe import (
    Evo2RegionAdapter,
    Geco2RegionAdapter,
    bpb_for_adapter,
    extract_filtered_region,
    filtered_length,
    model_window_average,
    moving_average,
    run_probe_for_source,
    stable_random_start,
    window_rows,
)
from dna_compress.fusion_compression import ProbabilityAdapter, UnitProbabilityResult


class FakeAdapter(ProbabilityAdapter):
    name = "fake"
    token_size = 1
    alphabet = "ACGT"

    def unit_probabilities(self, *, species: str, core_sequence: str, unit_size: int, batch_size: int):
        del species, unit_size, batch_size
        rows = []
        for base in core_sequence:
            if base == "A":
                rows.append([0.5, 0.25, 0.125, 0.125])
            elif base == "C":
                rows.append([0.25, 0.5, 0.125, 0.125])
            elif base == "G":
                rows.append([0.25, 0.125, 0.5, 0.125])
            else:
                rows.append([0.25, 0.125, 0.125, 0.5])
        return UnitProbabilityResult(
            adapter_name=self.name,
            probabilities=np.asarray(rows, dtype=np.float64),
            model_forward_seconds=0.1,
            softmax_seconds=0.2,
            aggregate_seconds=0.3,
            data_transfer_seconds=0.4,
        )


class FakeEvo2Tokenizer:
    pad_id = 1

    def tokenize(self, sequence: str):
        return [ord(base) for base in sequence]


class FakeEvo2Model:
    def __call__(self, input_ids):
        batch, length = input_ids.shape
        logits = torch.zeros((batch, length, 128), dtype=torch.float32, device=input_ids.device)
        return logits, None


class RegionBpbProbeTests(unittest.TestCase):
    def test_flat_filtered_region_uses_filtered_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat.txt"
            path.write_bytes(b"xxACgtNN--TA")
            self.assertEqual(filtered_length([path], alphabet="ACGTN", fasta=False), 8)
            self.assertEqual(
                extract_filtered_region([path], alphabet="ACGTN", fasta=False, start=2, length=4),
                b"GTNN",
            )

    def test_fasta_filtered_region_skips_headers_and_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "source.fasta"
            path.write_bytes(b">record 1\nACGT\n>record 2\nNNtaXX\n")
            self.assertEqual(filtered_length([path], alphabet="ACGTN", fasta=True), 8)
            self.assertEqual(
                extract_filtered_region([path], alphabet="ACGTN", fasta=True, start=3, length=5),
                b"TNNTA",
            )

    def test_stable_random_start_is_reproducible_and_bounded(self) -> None:
        first = stable_random_start(source_name="gtdb", source_length=1000, region_bases=100, seed=7)
        second = stable_random_start(source_name="gtdb", source_length=1000, region_bases=100, seed=7)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, 900)
        self.assertEqual(stable_random_start(source_name="short", source_length=50, region_bases=100, seed=7), 0)

    def test_window_rows_and_model_window_average(self) -> None:
        bpb = np.asarray([1.0, 2.0, 3.0, 5.0, 7.0])
        offsets = np.asarray([0, 1, 2, 4, 5])
        rows = window_rows(bpb, offsets, source_start=10, window_bases=3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["base_count"], 3)
        self.assertAlmostEqual(rows[0]["mean_bpb"], 2.0)
        self.assertEqual(rows[1]["source_start"], 13)
        means, counts = model_window_average(bpb, offsets, model_window_bases=3)
        self.assertEqual(counts.tolist(), [1, 2, 2])
        self.assertAlmostEqual(float(means[1]), 3.5)

    def test_moving_average_preserves_length(self) -> None:
        values = np.asarray([1.0, 2.0, 10.0, 2.0])
        smoothed = moving_average(values, 3)
        self.assertEqual(smoothed.shape, values.shape)
        self.assertAlmostEqual(float(smoothed[1]), (1.0 + 2.0 + 10.0) / 3.0)

    def test_bpb_for_adapter_filters_and_scores_targets(self) -> None:
        bpb, offsets, metadata = bpb_for_adapter(
            FakeAdapter(),
            species="sp",
            region_sequence="ANCG",
            region_offsets=np.arange(4, dtype=np.int64),
            batch_size=2,
        )
        self.assertEqual(offsets.tolist(), [0, 2, 3])
        self.assertEqual(bpb.shape[0], 3)
        self.assertTrue(np.allclose(bpb, np.ones(3)))
        self.assertEqual(metadata["filtered_out_bases"], 1)

    def test_evo2_adapter_scores_next_base_and_drops_window_first_base(self) -> None:
        adapter = Evo2RegionAdapter(
            name="evo2_fake",
            evo2_model=FakeEvo2Model(),
            tokenizer=FakeEvo2Tokenizer(),
            local_path=Path("fake.pt"),
            model_name="evo2_7b_base",
            context_bases=4,
            device=torch.device("cpu"),
            requested_device="cpu",
            dtype_name="float32",
            use_kernels=False,
        )
        bpb, offsets, metadata = adapter.region_bpb(
            region_sequence="ACGTAC",
            region_offsets=np.arange(6, dtype=np.int64),
            batch_size=2,
        )
        self.assertEqual(offsets.tolist(), [1, 2, 3, 5])
        self.assertTrue(np.allclose(bpb, np.full((4,), np.log2(128.0))))
        self.assertEqual(metadata["dropped_context_bases"], 2)
        self.assertEqual(metadata["valid_base_count"], 4)
        self.assertEqual(metadata["evo2_alignment"], "log_softmax(logits[:, :-1]) gathered at input_ids[:, 1:]")

    def test_run_probe_for_source_with_fake_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "source.txt"
            source_path.write_bytes(b"ACGTNNACGT")
            args = SimpleNamespace(
                region_start=1,
                region_bases=6,
                random_region=False,
                seed=1,
                batch_size=2,
                plot_window_bases=3,
                smooth_window_bases=2,
                model_window_smooth_bases=2,
                plot_individual_windows=True,
                max_individual_window_plots=1,
                record_dtype="float16",
                write_per_base_csv=False,
                compute_only=False,
            )
            result = run_probe_for_source(
                args,
                source_info={
                    "dataset": "dnacorpus",
                    "source": "fake_source",
                    "species": "Fake",
                    "paths": [source_path],
                    "fasta": False,
                    "alphabet": "ACGTN",
                },
                adapters=[FakeAdapter()],
                output_dir=root / "out",
            )
            self.assertEqual(result["region_bases"], 6)
            self.assertTrue((root / "out" / "region_bpb.json").exists())
            self.assertTrue((root / "out" / "models" / "fake" / "bpb.npz").exists())
            self.assertTrue((root / "out" / "region_bpb_curve.png").exists())
            self.assertEqual(result["outputs"]["individual_window_count"], 1)
            self.assertIsNone(result["outputs"]["per_base_csv"])

    def test_geco2_adapter_with_fake_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_bin = root / "fake_geco2.py"
            fake_bin.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "path = Path(sys.argv[-1])\n"
                "data = path.read_bytes()\n"
                "out = path.with_name(path.name + '.co')\n"
                "out.write_bytes(b'G' * (3 + (len(data) + 1) // 2))\n",
                encoding="utf-8",
            )
            fake_bin.chmod(0o755)
            adapter = Geco2RegionAdapter(
                name="geco2_test",
                binary=str(fake_bin),
                level=5,
                pseudo_window_bases=4,
                profile_mode="prefix_delta",
                use_dnacorpus_paper_levels=True,
                temp_root=root,
                keep_temp=False,
            )
            bpb, offsets, metadata = adapter.region_bpb(
                region_sequence="ACGTACGTAC",
                region_offsets=np.arange(10, dtype=np.int64),
                pseudo_window_bases=4,
                dataset="opengenome2",
                species=None,
                source="fake",
            )
            self.assertEqual(bpb.shape[0], 10)
            self.assertEqual(offsets.tolist(), list(range(10)))
            self.assertEqual(metadata["continuous_compressed_bytes"], 8)
            self.assertEqual(metadata["geco2_profile_mode"], "prefix_delta")
            self.assertEqual(len(metadata["geco2_prefix_results"]), 3)

    def test_geco2_uses_dnacorpus_paper_level_by_species(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_path = root / "levels.txt"
            fake_bin = root / "fake_geco2.py"
            fake_bin.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "level = sys.argv[sys.argv.index('-l') + 1]\n"
                f"Path({str(log_path)!r}).write_text(level, encoding='utf-8')\n"
                "p = Path(sys.argv[-1])\n"
                "p.with_name(p.name + '.co').write_bytes(b'1234')\n",
                encoding="utf-8",
            )
            fake_bin.chmod(0o755)
            adapter = Geco2RegionAdapter(
                name="geco2_test",
                binary=str(fake_bin),
                level=5,
                pseudo_window_bases=16,
                profile_mode="constant",
                use_dnacorpus_paper_levels=True,
                temp_root=root,
                keep_temp=False,
            )
            _, _, metadata = adapter.region_bpb(
                region_sequence="ACGTACGT",
                region_offsets=np.arange(8, dtype=np.int64),
                pseudo_window_bases=16,
                dataset="dnacorpus",
                species="BuEb",
                source="BuEb",
            )
            self.assertEqual(metadata["geco2_level"], 1)
            self.assertEqual(log_path.read_text(encoding="utf-8"), "1")

    def test_geco2_estimate_mode_reads_per_base_iae(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fake_bin = root / "fake_geco2.py"
            fake_bin.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "import sys\n"
                "p = Path(sys.argv[-1])\n"
                "data = p.read_bytes()\n"
                "p.with_name(p.name + '.co').write_bytes(b'1234')\n"
                "if '-e' in sys.argv:\n"
                "    p.with_name(p.name + '.iae').write_text(''.join('1.5\\n' for _ in data), encoding='utf-8')\n",
                encoding="utf-8",
            )
            fake_bin.chmod(0o755)
            adapter = Geco2RegionAdapter(
                name="geco2_test",
                binary=str(fake_bin),
                level=5,
                pseudo_window_bases=16,
                profile_mode="estimate",
                use_dnacorpus_paper_levels=False,
                temp_root=root,
                keep_temp=False,
            )
            bpb, offsets, metadata = adapter.region_bpb(
                region_sequence="ACGTACGT",
                region_offsets=np.arange(8, dtype=np.int64),
                pseudo_window_bases=16,
                dataset="dnacorpus",
                species="BuEb",
                source="BuEb",
            )
            self.assertTrue(np.allclose(bpb, np.full(8, 1.5)))
            self.assertEqual(offsets.tolist(), list(range(8)))
            self.assertEqual(metadata["geco2_profile_mode"], "estimate")


if __name__ == "__main__":
    unittest.main()
