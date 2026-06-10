from __future__ import annotations

from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dna_compress.experiment import compute_language_model_loss
from dna_compress.fasta_fragment_index import (
    IndexedFastaFragmentSampler,
    IndexedMegabyteFileStreamDataset,
    IndexedMegabyteSourceBatchStreamDataset,
    IndexedMegabyteWindowDataset,
    build_fasta_fragment_index,
    ensure_fasta_index_runtime_cache,
    load_fasta_index_runtime_cache,
    prepare_indexed_megabyte_eval_cache,
    split_run_ids,
)
from dna_compress.repacked_windows import (
    RepackedMegabyteWindowDataset,
    build_repacked_megabyte_windows,
    build_repacked_split_schedule,
)
from scripts.run_megadna_experiment import _apply_overrides as apply_megadna_overrides
from scripts.run_megadna_experiment import _build_parser as build_megadna_parser
from scripts.run_dna_experiment import _apply_overrides as apply_dna_overrides
from scripts.run_dna_experiment import _build_parser as build_dna_parser
from scripts.run_dna_experiment import _validate_config_for_megabyte
from scripts.build_opengenome2_fasta_test_subset import build_opengenome2_fasta_test_subset
from dna_compress.config import ExperimentConfig


class FastaFragmentIndexTests(unittest.TestCase):
    def _build_small_index(self, tmpdir: str) -> tuple[Path, Path]:
        fasta_root = Path(tmpdir) / "fasta"
        index_dir = Path(tmpdir) / "index"
        (fasta_root / "source_a").mkdir(parents=True)
        (fasta_root / "source_b").mkdir(parents=True)
        (fasta_root / "source_a" / "a.fasta").write_text(
            ">rec1 first\n"
            "acgtNNACGTry\n"
            ">rec2 second\n"
            "AAAA\n"
            "CCCC\n",
            encoding="utf-8",
        )
        (fasta_root / "source_b" / "b.fasta").write_text(
            ">long single line\n" + ("tttt" * 20) + "N" + ("gggg" * 20) + "\n",
            encoding="utf-8",
        )
        build_fasta_fragment_index(
            fasta_root=fasta_root,
            index_dir=index_dir,
            anchor_stride=8,
            chunk_size=17,
            batch_rows=2,
        )
        return fasta_root, index_dir

    def _build_nonoverlap_index(self, tmpdir: str) -> tuple[Path, Path]:
        fasta_root = Path(tmpdir) / "fasta"
        index_dir = Path(tmpdir) / "index"
        (fasta_root / "source_a").mkdir(parents=True)
        (fasta_root / "source_b").mkdir(parents=True)
        (fasta_root / "source_a" / "a.fasta").write_text(
            ">a\n"
            "AAAACCCCGG\n",
            encoding="utf-8",
        )
        (fasta_root / "source_b" / "b.fasta").write_text(
            ">b\n"
            "TTTTTTTTTTTTTTTTTTTTTTTT\n",
            encoding="utf-8",
        )
        build_fasta_fragment_index(
            fasta_root=fasta_root,
            index_dir=index_dir,
            anchor_stride=4,
            chunk_size=11,
            batch_rows=2,
        )
        return fasta_root, index_dir

    def _build_file_stream_index(self, tmpdir: str) -> tuple[Path, Path]:
        fasta_root = Path(tmpdir) / "fasta"
        index_dir = Path(tmpdir) / "index"
        (fasta_root / "source_a").mkdir(parents=True)
        (fasta_root / "source_b").mkdir(parents=True)
        (fasta_root / "source_a" / "a.fasta").write_text(
            ">a1\n"
            "AAAACCCCGGGGTTTTAA\n",
            encoding="utf-8",
        )
        (fasta_root / "source_b" / "b.fasta").write_text(
            ">b1\n"
            "TGCACAGTGTACCTGA\n",
            encoding="utf-8",
        )
        build_fasta_fragment_index(
            fasta_root=fasta_root,
            index_dir=index_dir,
            anchor_stride=4,
            chunk_size=9,
            batch_rows=2,
        )
        return fasta_root, index_dir

    def _build_source_batch_index(self, tmpdir: str) -> tuple[Path, Path]:
        fasta_root = Path(tmpdir) / "fasta"
        index_dir = Path(tmpdir) / "index"
        for source in ("source_a", "source_b"):
            (fasta_root / source).mkdir(parents=True)
        for index in range(4):
            (fasta_root / "source_a" / f"a{index}.fasta").write_text(
                f">a{index}\n" + "A" * 18 + "\n",
                encoding="utf-8",
            )
            (fasta_root / "source_b" / f"b{index}.fasta").write_text(
                f">b{index}\n" + "T" * 18 + "\n",
                encoding="utf-8",
            )
        build_fasta_fragment_index(
            fasta_root=fasta_root,
            index_dir=index_dir,
            anchor_stride=4,
            chunk_size=9,
            batch_rows=2,
        )
        return fasta_root, index_dir

    def test_build_index_records_acgt_runs_and_splits_non_acgt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)

            records = pq.read_table(index_dir / "records.parquet").to_pydict()
            runs = pq.read_table(index_dir / "acgt_runs.parquet").to_pydict()
            anchors = pq.read_table(index_dir / "acgt_anchors.parquet").to_pydict()

            self.assertEqual(len(records["record_id"]), 3)
            self.assertEqual(sum(records["lowercase_bases"]), 166)
            self.assertEqual(sum(records["n_bases"]), 3)
            self.assertIn(4, runs["run_base_length"])
            self.assertIn(8, runs["run_base_length"])
            self.assertGreaterEqual(len(anchors["run_id"]), len(runs["run_id"]))

    def test_sampler_returns_fixed_length_megadna_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)
            sampler = IndexedFastaFragmentSampler(index_dir, seq_length=12)

            sample = sampler.sample(seed=123)

            self.assertEqual(sample["input_ids"].shape[0], 12)
            self.assertTrue(set(sample["input_ids"].tolist()).issubset({1, 2, 3, 4}))
            self.assertIn(sample["source"], {"source_a", "source_b"})
            sampler.close()

    def test_source_weights_validate_unknown_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)

            with self.assertRaises(ValueError):
                IndexedFastaFragmentSampler(index_dir, seq_length=4, source_weights={"missing": 1.0})

            sampler = IndexedFastaFragmentSampler(index_dir, seq_length=4, source_weights={"source_b": 1.0})
            self.assertEqual(sampler.sample(seed=1)["source"], "source_b")
            sampler.close()

    def test_run_hash_splits_do_not_overlap(self) -> None:
        run_ids = torch.arange(10_000, dtype=torch.long).numpy()

        train = split_run_ids(run_ids, split="train", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, split_seed=7)
        val = split_run_ids(run_ids, split="val", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, split_seed=7)
        test = split_run_ids(run_ids, split="test", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, split_seed=7)

        self.assertFalse(bool((train & val).any()))
        self.assertFalse(bool((train & test).any()))
        self.assertFalse(bool((val & test).any()))
        self.assertTrue(bool((train | val | test).all()))

    def test_runtime_cache_builds_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)

            cache_dir = ensure_fasta_index_runtime_cache(index_dir)
            second_cache_dir = ensure_fasta_index_runtime_cache(index_dir)
            cache = load_fasta_index_runtime_cache(index_dir)

            self.assertEqual(cache_dir, second_cache_dir)
            self.assertTrue((cache_dir / "metadata.json").exists())
            self.assertEqual(len(cache.file_paths), 2)
            self.assertGreater(cache.run_ids.shape[0], 0)

    def test_indexed_megabyte_dataset_returns_ascii_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)
            dataset = IndexedMegabyteWindowDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=3,
                seed=11,
                source_weights={"source_b": 1.0},
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            item = dataset[0]["input_ids"]

            self.assertEqual(tuple(item.shape), (4,))
            self.assertTrue(set(item.tolist()).issubset({ord("A"), ord("C"), ord("G"), ord("T")}))

    def test_indexed_megabyte_dataset_returns_merged_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)
            dataset = IndexedMegabyteWindowDataset(
                index_dir=index_dir,
                split="train",
                seq_length=2,
                token_merge_size=3,
                token_merge_alphabet="ACGTN",
                samples=1,
                seed=5,
                source_weights={"source_a": 1.0},
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            item = dataset[0]["input_ids"]

            self.assertEqual(item.tolist(), [0, 6])

    def test_indexed_megabyte_dataset_validates_source_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)

            with self.assertRaises(ValueError):
                IndexedMegabyteWindowDataset(
                    index_dir=index_dir,
                    split="train",
                    seq_length=4,
                    token_merge_size=1,
                    token_merge_alphabet="ACGTN",
                    samples=1,
                    seed=1,
                    source_weights={"missing": 1.0},
                    train_ratio=1.0,
                    val_ratio=0.0,
                    test_ratio=0.0,
                )

    def test_indexed_megabyte_dataset_works_with_dataloader_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_small_index(tmpdir)
            dataset = IndexedMegabyteWindowDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=2,
                seed=11,
                source_weights={"source_b": 1.0},
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            loader = DataLoader(dataset, batch_size=2, num_workers=1)

            batch = next(iter(loader))

            self.assertEqual(tuple(batch["input_ids"].shape), (2, 4))

    def test_indexed_megabyte_nonoverlap_windows_pad_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_nonoverlap_index(tmpdir)
            dataset = IndexedMegabyteWindowDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=3,
                pad_id=257,
                window_mode="nonoverlap_random",
                epoch_mode="all_windows",
                source_loss_weights={"source_a": 1.0, "source_b": 1.0},
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            windows = [dataset[index]["input_ids"].tolist() for index in range(len(dataset))]
            nonpad_counts = sorted(sum(token != 257 for token in window) for window in windows)

            self.assertEqual(len(dataset), 9)
            self.assertEqual(nonpad_counts[:1], [2])
            self.assertIn([ord("G"), ord("G"), 257, 257], windows)
            self.assertEqual(dataset.summary()["padded_window_count"], 1)

    def test_indexed_megabyte_nonoverlap_merged_tokens_pad_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_root = Path(tmpdir) / "fasta"
            index_dir = Path(tmpdir) / "index"
            (fasta_root / "source_a").mkdir(parents=True)
            (fasta_root / "source_a" / "a.fasta").write_text(">a\nAAACCCGGGT\n", encoding="utf-8")
            build_fasta_fragment_index(fasta_root=fasta_root, index_dir=index_dir, anchor_stride=3, chunk_size=7)
            dataset = IndexedMegabyteWindowDataset(
                index_dir=index_dir,
                split="train",
                seq_length=2,
                token_merge_size=3,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=3,
                pad_id=125,
                window_mode="nonoverlap_random",
                epoch_mode="all_windows",
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            windows = {tuple(dataset[index]["input_ids"].tolist()) for index in range(len(dataset))}

            self.assertEqual(windows, {(0, 31), (62, 125)})

    def test_indexed_megabyte_nonoverlap_order_is_reproducible_and_unique_per_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_nonoverlap_index(tmpdir)
            kwargs = dict(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=19,
                pad_id=257,
                window_mode="nonoverlap_random",
                epoch_mode="all_windows",
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            first = IndexedMegabyteWindowDataset(**kwargs)
            second = IndexedMegabyteWindowDataset(**kwargs)

            first_windows = [tuple(first[index]["input_ids"].tolist()) for index in range(len(first))]
            second_windows = [tuple(second[index]["input_ids"].tolist()) for index in range(len(second))]
            tickets = [first._permuted_window_ticket(index) for index in range(len(first))]

            self.assertEqual(first_windows, second_windows)
            self.assertEqual(len(first_windows), len(second_windows))
            self.assertEqual(len(set(tickets)), len(first))

    def test_indexed_megabyte_source_loss_weights_return_expected_multipliers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_nonoverlap_index(tmpdir)
            dataset = IndexedMegabyteWindowDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=7,
                pad_id=257,
                window_mode="nonoverlap_random",
                epoch_mode="all_windows",
                source_loss_weights={"source_a": 0.5, "source_b": 0.5},
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            weights_by_first_token: dict[int, set[float]] = {}
            for index in range(len(dataset)):
                item = dataset[index]
                first_token = int(item["input_ids"][0].item())
                weights_by_first_token.setdefault(first_token, set()).add(round(float(item["loss_weight"].item()), 6))

            self.assertEqual(weights_by_first_token[ord("A")], {1.7})
            self.assertEqual(weights_by_first_token[ord("T")], {0.708333})

            with self.assertRaises(ValueError):
                IndexedMegabyteWindowDataset(
                    index_dir=index_dir,
                    split="train",
                    seq_length=4,
                    token_merge_size=1,
                    token_merge_alphabet="ACGTN",
                    samples=1,
                    seed=7,
                    pad_id=257,
                    window_mode="nonoverlap_random",
                    source_loss_weights={"missing": 1.0},
                    train_ratio=1.0,
                    val_ratio=0.0,
                    test_ratio=0.0,
                )

    def test_indexed_megabyte_file_stream_covers_nonoverlap_windows_with_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            dataset = IndexedMegabyteFileStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=5,
                pad_id=257,
                file_stream_windows=2,
                file_shuffle_buffer_windows=0,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            windows = [tuple(item["input_ids"].tolist()) for item in dataset]

            self.assertEqual(len(windows), 9)
            self.assertEqual(len(set(windows)), 9)
            self.assertIn((ord("A"), ord("A"), 257, 257), windows)
            self.assertEqual(dataset.summary()["stream_unit_count"], 5)
            self.assertEqual(dataset.summary()["file_stream_windows"], 2)

    def test_indexed_megabyte_file_stream_dataloader_workers_do_not_duplicate_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            dataset = IndexedMegabyteFileStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=7,
                pad_id=257,
                file_stream_windows=2,
                file_shuffle_buffer_windows=0,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            loader = DataLoader(dataset, batch_size=1, num_workers=2)

            windows = [tuple(batch["input_ids"][0].tolist()) for batch in loader]

            self.assertEqual(len(windows), 9)
            self.assertEqual(len(set(windows)), 9)

    def test_indexed_megabyte_file_stream_ddp_like_ranks_do_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            common = dict(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=11,
                pad_id=257,
                file_stream_windows=2,
                file_shuffle_buffer_windows=0,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
                ddp_world_size=2,
            )
            rank0 = IndexedMegabyteFileStreamDataset(ddp_rank=0, **common)
            rank1 = IndexedMegabyteFileStreamDataset(ddp_rank=1, **common)

            windows0 = {tuple(item["input_ids"].tolist()) for item in rank0}
            windows1 = {tuple(item["input_ids"].tolist()) for item in rank1}

            self.assertFalse(windows0 & windows1)
            self.assertEqual(len(windows0 | windows1), 9)

    def test_indexed_megabyte_file_stream_shuffle_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            kwargs = dict(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=13,
                pad_id=257,
                file_stream_windows=5,
                file_shuffle_buffer_windows=2,
                file_stream_order_seed=3,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            first = IndexedMegabyteFileStreamDataset(**kwargs)
            second = IndexedMegabyteFileStreamDataset(**kwargs)

            first_windows = [tuple(item["input_ids"].tolist()) for item in first]
            second_windows = [tuple(item["input_ids"].tolist()) for item in second]

            self.assertEqual(first_windows, second_windows)

    def test_indexed_megabyte_source_batch_stream_samples_sources_by_probability(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)
            dataset = IndexedMegabyteSourceBatchStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=4096,
                seed=17,
                batch_size=32,
                source_weights={"source_a": 3.0, "source_b": 1.0},
                pad_id=257,
                source_balance_batches=4,
                source_read_block_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            batches = list(dataset)
            windows = torch.cat([batch["input_ids"] for batch in batches], dim=0)
            starts = windows[:, 0].tolist()
            a_count = starts.count(ord("A"))
            observed_a_fraction = a_count / len(starts)

            self.assertEqual(tuple(batches[0]["input_ids"].shape), (32, 4))
            self.assertEqual(len(starts), 4096)
            self.assertGreater(observed_a_fraction, 0.70)
            self.assertLess(observed_a_fraction, 0.80)
            self.assertEqual(dataset.summary()["source_balance_batches"], 4)
            self.assertEqual(dataset.summary()["source_mix_chunk_batches"], 4)
            self.assertTrue(dataset.summary()["deprecated_source_mix_chunk_batches_ignored"])
            self.assertEqual(dataset.summary()["source_sampling_strategy"], "per_sample_probability")

    def test_indexed_megabyte_source_batch_stream_all_windows_uses_expected_slowest_source_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)

            natural = IndexedMegabyteSourceBatchStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=17,
                batch_size=8,
                source_weights=None,
                pad_id=257,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            weighted = IndexedMegabyteSourceBatchStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=None,
                seed=17,
                batch_size=8,
                source_weights={"source_a": 3.0, "source_b": 1.0},
                pad_id=257,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            self.assertEqual(natural.samples, natural.total_candidate_windows)
            self.assertEqual(weighted.samples, 80)
            self.assertEqual(weighted.summary()["epoch_sample_count_mode"], "expected_slowest_source_coverage")

    def test_indexed_megabyte_source_batch_stream_epoch_controls_random_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)
            common = dict(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=64,
                seed=43,
                batch_size=8,
                source_weights={"source_a": 1.0, "source_b": 1.0},
                pad_id=257,
                source_read_chunk_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            first = IndexedMegabyteSourceBatchStreamDataset(**common)
            second = IndexedMegabyteSourceBatchStreamDataset(**common)
            next_epoch = IndexedMegabyteSourceBatchStreamDataset(**common)
            next_epoch.set_epoch(1)

            first_starts = torch.cat([batch["input_ids"] for batch in first], dim=0)[:, 0].tolist()
            second_starts = torch.cat([batch["input_ids"] for batch in second], dim=0)[:, 0].tolist()
            next_epoch_starts = torch.cat([batch["input_ids"] for batch in next_epoch], dim=0)[:, 0].tolist()

            self.assertEqual(first_starts, second_starts)
            self.assertNotEqual(first_starts, next_epoch_starts)

    def test_indexed_megabyte_source_batch_stream_shuffles_read_chunk_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_root = Path(tmpdir) / "fasta"
            index_dir = Path(tmpdir) / "index"
            (fasta_root / "source_a").mkdir(parents=True)
            (fasta_root / "source_a" / "a.fasta").write_text(
                ">a\n"
                "AAAACCCCGGGGTTTT\n",
                encoding="utf-8",
            )
            build_fasta_fragment_index(
                fasta_root=fasta_root,
                index_dir=index_dir,
                anchor_stride=4,
                chunk_size=11,
                batch_rows=2,
            )
            common = dict(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=4,
                seed=37,
                batch_size=1,
                source_weights={"source_a": 1.0},
                pad_id=257,
                source_mix_chunk_batches=4,
                source_read_chunk_windows=4,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            sequential = IndexedMegabyteSourceBatchStreamDataset(**common, source_read_chunk_shuffle=False)
            shuffled = IndexedMegabyteSourceBatchStreamDataset(**common, source_read_chunk_shuffle=True)
            sequential_starts = [int(batch["input_ids"][0, 0].item()) for batch in sequential]
            shuffled_starts = [int(batch["input_ids"][0, 0].item()) for batch in shuffled]

            self.assertEqual(sequential_starts, [ord("A"), ord("C"), ord("G"), ord("T")])
            self.assertCountEqual(shuffled_starts, sequential_starts)
            self.assertNotEqual(shuffled_starts, sequential_starts)

    def test_indexed_megabyte_source_batch_stream_mixes_sources_at_window_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)
            dataset = IndexedMegabyteSourceBatchStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=32,
                seed=41,
                batch_size=8,
                source_weights={"source_a": 1.0, "source_b": 1.0},
                pad_id=257,
                source_mix_chunk_batches=4,
                source_read_chunk_windows=2,
                source_read_chunk_shuffle=False,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            starts = torch.cat([batch["input_ids"] for batch in dataset], dim=0)[:, 0].tolist()

            self.assertIn(ord("A"), starts)
            self.assertIn(ord("T"), starts)

    def test_indexed_megabyte_source_batch_stream_dataloader_workers_emit_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)
            dataset = IndexedMegabyteSourceBatchStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=32,
                seed=19,
                batch_size=8,
                source_weights={"source_a": 1.0, "source_b": 1.0},
                pad_id=257,
                source_balance_batches=2,
                source_read_block_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            loader = DataLoader(dataset, batch_size=None, num_workers=2)

            batches = list(loader)

            self.assertEqual(sum(batch["input_ids"].shape[0] for batch in batches), 32)
            self.assertTrue(all(tuple(batch["input_ids"].shape[1:]) == (4,) for batch in batches))

    def test_indexed_megabyte_source_batch_stream_resumes_from_batch_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)
            common = dict(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=32,
                seed=31,
                batch_size=4,
                source_weights={"source_a": 1.0, "source_b": 1.0},
                pad_id=257,
                source_balance_batches=2,
                source_read_block_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            full_dataset = IndexedMegabyteSourceBatchStreamDataset(**common)
            resumed_dataset = IndexedMegabyteSourceBatchStreamDataset(**common, start_batch_index=3)

            full_batches = list(full_dataset)
            resumed_batches = list(resumed_dataset)

            self.assertEqual([int(batch["_batch_index"].item()) for batch in resumed_batches], [3, 4, 5, 6, 7])
            torch.testing.assert_close(
                torch.cat([batch["input_ids"] for batch in resumed_batches], dim=0),
                torch.cat([batch["input_ids"] for batch in full_batches[3:]], dim=0),
            )
            resumed_dataset.set_start_batch_index(0)
            self.assertEqual(int(next(iter(resumed_dataset))["_batch_index"].item()), 0)

    def test_indexed_megabyte_source_batch_stream_strides_small_source_across_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_root = Path(tmpdir) / "fasta"
            index_dir = Path(tmpdir) / "index"
            (fasta_root / "source_a").mkdir(parents=True)
            (fasta_root / "source_b").mkdir(parents=True)
            (fasta_root / "source_a" / "one_file.fasta").write_text(">a\n" + "A" * 64 + "\n", encoding="utf-8")
            for index in range(4):
                (fasta_root / "source_b" / f"b{index}.fasta").write_text(
                    f">b{index}\n" + "T" * 64 + "\n",
                    encoding="utf-8",
                )
            build_fasta_fragment_index(
                fasta_root=fasta_root,
                index_dir=index_dir,
                anchor_stride=8,
                chunk_size=11,
                batch_rows=2,
            )
            dataset = IndexedMegabyteSourceBatchStreamDataset(
                index_dir=index_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                samples=32,
                seed=29,
                batch_size=4,
                source_weights={"source_a": 1.0, "source_b": 1.0},
                pad_id=257,
                source_balance_batches=1,
                source_read_block_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            loader = DataLoader(dataset, batch_size=None, num_workers=4)

            windows = torch.cat([batch["input_ids"] for batch in loader], dim=0)
            starts = windows[:, 0].tolist()

            self.assertEqual(len(starts), 32)
            self.assertIn(ord("A"), starts)
            self.assertIn(ord("T"), starts)

    def test_indexed_megabyte_source_batch_stream_validates_unknown_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)

            with self.assertRaises(ValueError):
                IndexedMegabyteSourceBatchStreamDataset(
                    index_dir=index_dir,
                    split="train",
                    seq_length=4,
                    token_merge_size=1,
                    token_merge_alphabet="ACGTN",
                    samples=8,
                    seed=23,
                    batch_size=4,
                    source_weights={"missing": 1.0},
                    pad_id=257,
                    train_ratio=1.0,
                    val_ratio=0.0,
                    test_ratio=0.0,
                )

    def test_indexed_megabyte_eval_cache_reuses_random_source_weighted_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_source_batch_index(tmpdir)
            cache_root = Path(tmpdir) / "eval_cache"

            dataset = prepare_indexed_megabyte_eval_cache(
                index_dir=index_dir,
                cache_root=cache_root,
                split="val",
                samples=16,
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=257,
                source_weights={"source_a": 3.0, "source_b": 1.0},
                train_ratio=0.0,
                val_ratio=1.0,
                test_ratio=0.0,
                split_seed=0,
                eval_seed=13,
                mode="refresh",
            )

            summary = dataset.summary()
            batch = next(iter(DataLoader(dataset, batch_size=4)))
            self.assertEqual(len(dataset), 16)
            self.assertEqual(tuple(dataset[0]["input_ids"].shape), (4,))
            self.assertEqual(tuple(batch["input_ids"].shape), (4, 4))
            self.assertEqual(summary["source_counts"]["source_a"], 12)
            self.assertEqual(summary["source_counts"]["source_b"], 4)
            self.assertTrue(np.any(np.asarray(dataset.run_window_indices) != 0))
            self.assertFalse(summary["cache_hit"])

            reused = prepare_indexed_megabyte_eval_cache(
                index_dir=index_dir,
                cache_root=cache_root,
                split="val",
                samples=16,
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=257,
                source_weights={"source_a": 3.0, "source_b": 1.0},
                train_ratio=0.0,
                val_ratio=1.0,
                test_ratio=0.0,
                split_seed=0,
                eval_seed=13,
                mode="reuse",
            )

            self.assertTrue(reused.summary()["cache_hit"])
            np.testing.assert_array_equal(np.asarray(dataset.input_ids), np.asarray(reused.input_ids))

    def test_real_fasta_test_subset_keeps_original_contiguous_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_root = Path(tmpdir) / "fasta"
            index_dir = Path(tmpdir) / "index"
            (fasta_root / "source_a").mkdir(parents=True)
            (fasta_root / "source_b").mkdir(parents=True)
            (fasta_root / "source_a" / "a.fasta").write_text(
                ">a0 original\nAAAA\n"
                ">a1 original\nCCCC\n"
                ">a2 original\nGGGG\n"
                ">a3 original\nTTTT\n",
                encoding="utf-8",
            )
            (fasta_root / "source_b" / "b.fasta").write_text(
                ">b0 keep\nACGTACGT\n",
                encoding="utf-8",
            )
            build_fasta_fragment_index(
                fasta_root=fasta_root,
                index_dir=index_dir,
                anchor_stride=4,
                chunk_size=11,
                batch_rows=2,
            )

            output_dir = Path(tmpdir) / "subset"
            manifest = build_opengenome2_fasta_test_subset(
                index_dir=index_dir,
                output_dir=output_dir,
                target_bytes_per_source=30,
                seed=0,
                batch_rows=2,
                overwrite=False,
            )

            source_a = manifest["sources"]["source_a"]
            source_b = manifest["sources"]["source_b"]
            self.assertGreaterEqual(source_a["selected_bytes"], 30)
            self.assertFalse(source_a["include_all"])
            self.assertTrue(source_b["include_all"])
            self.assertEqual(source_b["selected_record_count"], 1)
            self.assertTrue((output_dir / "source_a.fasta").read_text(encoding="utf-8").startswith(">a"))
            self.assertEqual((output_dir / "source_b.fasta").read_text(encoding="utf-8"), ">b0 keep\nACGTACGT\n")
            for span in source_a["spans"]:
                self.assertEqual(span["record_count"], span["end_record_id"] - span["start_record_id"] + 1)

    def test_build_repacked_windows_and_iterate_all_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            repacked_dir = Path(tmpdir) / "repacked"

            manifest = build_repacked_megabyte_windows(
                index_dir=index_dir,
                output_dir=repacked_dir,
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=125,
                shard_windows=3,
                read_unit_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            dataset = RepackedMegabyteWindowDataset(
                repacked_dir=repacked_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=125,
                samples=None,
                seed=7,
                read_chunk_windows=2,
                epoch_mode="all_windows",
            )

            windows = [item["input_ids"].tolist() for item in dataset]

            self.assertEqual(manifest["window_count"], 9)
            self.assertEqual(len(windows), 9)
            self.assertIn([ord("A"), ord("A"), 125, 125], windows)
            self.assertTrue((repacked_dir / "manifest.json").exists())

    def test_repacked_hash_shards_schedule_and_source_loss_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            repacked_dir = Path(tmpdir) / "repacked"
            manifest = build_repacked_megabyte_windows(
                index_dir=index_dir,
                output_dir=repacked_dir,
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=125,
                hash_shard_count=2,
                read_unit_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )

            self.assertEqual(manifest["layout"], "hash_partitioned_mixed_shards")
            self.assertTrue((repacked_dir / manifest["default_schedule_dir"] / "train.npy").exists())
            mixed_shards = [
                shard for shard in manifest["shards"] if sum(1 for value in shard["source_counts"].values() if value > 0) > 1
            ]
            self.assertGreaterEqual(len(mixed_shards), 1)

            with self.assertRaises(ValueError):
                RepackedMegabyteWindowDataset(
                    repacked_dir=repacked_dir,
                    split="train",
                    seq_length=4,
                    token_merge_size=1,
                    token_merge_alphabet="ACGTN",
                    pad_id=125,
                    samples=2,
                    seed=1,
                    source_sampling_weights={"missing": 1.0},
                )

            dataset = RepackedMegabyteWindowDataset(
                repacked_dir=repacked_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=125,
                samples=4,
                seed=1,
                read_chunk_windows=2,
                source_loss_weights={"source_a": 0.5, "source_b": 0.5},
            )
            observed = list(dataset)

            self.assertEqual(len(observed), 4)
            self.assertTrue(all("loss_weight" in item for item in observed))
            self.assertTrue(all(tuple(item["input_ids"].shape) == (4,) for item in observed))

    def test_repacked_dataloader_workers_cover_all_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            repacked_dir = Path(tmpdir) / "repacked"
            build_repacked_megabyte_windows(
                index_dir=index_dir,
                output_dir=repacked_dir,
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=125,
                shard_windows=3,
                read_unit_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            dataset = RepackedMegabyteWindowDataset(
                repacked_dir=repacked_dir,
                split="train",
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=125,
                samples=None,
                seed=7,
                read_chunk_windows=1,
                epoch_mode="all_windows",
            )
            loader = DataLoader(dataset, batch_size=1, num_workers=2)

            count = sum(int(batch["input_ids"].shape[0]) for batch in loader)

            self.assertEqual(count, 9)

    def test_repacked_split_schedule_can_be_rebuilt_without_rewriting_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, index_dir = self._build_file_stream_index(tmpdir)
            repacked_dir = Path(tmpdir) / "repacked"
            build_repacked_megabyte_windows(
                index_dir=index_dir,
                output_dir=repacked_dir,
                seq_length=4,
                token_merge_size=1,
                token_merge_alphabet="ACGTN",
                pad_id=125,
                hash_shard_count=2,
                read_unit_windows=2,
                train_ratio=1.0,
                val_ratio=0.0,
                test_ratio=0.0,
            )
            schedule_dir = build_repacked_split_schedule(
                repacked_dir=repacked_dir,
                train_ratio=0.5,
                val_ratio=0.25,
                test_ratio=0.25,
                split_seed=99,
            )

            entries = {}
            for split in ("train", "val", "test"):
                array = np.load(schedule_dir / f"{split}.npy")
                entries[split] = {(int(row["shard_id"]), int(row["window_index"])) for row in array}

            self.assertFalse(entries["train"] & entries["val"])
            self.assertFalse(entries["train"] & entries["test"])
            self.assertFalse(entries["val"] & entries["test"])
            self.assertEqual(len(entries["train"] | entries["val"] | entries["test"]), 9)

    def test_weighted_language_model_loss_ignores_zero_weight_samples(self) -> None:
        class FakeModel(torch.nn.Module):
            def forward(self, ids, return_loss=False):
                logits = torch.zeros((*ids.shape, 300), dtype=torch.float32)
                logits[0, :, ord("A")] = 10.0
                logits[1, :, ord("T")] = 10.0
                if return_loss:
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        ids.reshape(-1),
                        ignore_index=257,
                    )
                else:
                    loss = None
                return SimpleNamespace(lm_logits=logits, loss=loss)

        ids = torch.tensor(
            [
                [ord("A"), ord("A"), 257, 257],
                [ord("A"), ord("A"), 257, 257],
            ],
            dtype=torch.long,
        )
        batch = {"input_ids": ids, "loss_weight": torch.tensor([1.0, 0.0], dtype=torch.float32)}

        loss, valid_tokens, weighted_tokens = compute_language_model_loss(FakeModel(), ids, batch, pad_id=257)

        self.assertEqual(valid_tokens, 4)
        self.assertEqual(weighted_tokens, 2.0)
        self.assertLess(float(loss.item()), 1.0)


class MegaDNAIndexedFastaCliTests(unittest.TestCase):
    def test_megadna_cli_parses_indexed_fasta_options(self) -> None:
        parser = build_megadna_parser()
        args = parser.parse_args(
            [
                "--sequence-source-mode",
                "indexed_fasta",
                "--fasta-index-dir",
                "/tmp/index",
                "--source-sampling-weights-json",
                '{"gtdb_v220": 0.5, "metagenomes": 0.2}',
                "--indexed-eval-samples",
                "17",
            ]
        )
        config = ExperimentConfig()
        apply_megadna_overrides(config, args)

        self.assertEqual(config.data.sequence_source_mode, "indexed_fasta")
        self.assertEqual(config.data.fasta_index_dir, "/tmp/index")
        self.assertEqual(config.data.source_sampling_weights["gtdb_v220"], 0.5)
        self.assertEqual(config.data.indexed_eval_samples, 17)


class MegabyteIndexedFastaCliTests(unittest.TestCase):
    def test_dna_cli_parses_indexed_fasta_options(self) -> None:
        parser = build_dna_parser()
        args = parser.parse_args(
            [
                "--config",
                "dummy.json",
                "--sequence-source-mode",
                "indexed_fasta",
                "--fasta-index-dir",
                "/tmp/index",
                "--source-sampling-weights-json",
                '{"gtdb_v220": 0.5, "metagenomes": 0.2}',
                "--indexed-eval-samples",
                "17",
                "--indexed-eval-cache-dir",
                "/tmp/eval_cache",
                "--indexed-eval-cache-mode",
                "refresh",
                "--indexed-eval-random-seed",
                "11",
                "--indexed-split-seed",
                "9",
                "--indexed-window-mode",
                "nonoverlap_file_stream",
                "--indexed-train-epoch-mode",
                "all_windows",
                "--indexed-file-stream-windows",
                "123",
                "--indexed-file-shuffle-buffer-windows",
                "45",
                "--indexed-file-stream-order-seed",
                "6",
                "--indexed-source-mix-chunk-batches",
                "7",
                "--indexed-source-read-chunk-windows",
                "89",
                "--no-indexed-source-read-chunk-shuffle",
                "--indexed-source-file-order-seed",
                "10",
                "--source-loss-weights-json",
                '{"gtdb_v220": 0.5, "metagenomes": 0.5}',
            ]
        )
        config = ExperimentConfig()
        apply_dna_overrides(config, args)

        self.assertEqual(config.data.sequence_source_mode, "indexed_fasta")
        self.assertEqual(config.data.fasta_index_dir, "/tmp/index")
        self.assertEqual(config.data.source_sampling_weights["metagenomes"], 0.2)
        self.assertEqual(config.data.indexed_eval_samples, 17)
        self.assertEqual(config.data.indexed_eval_cache_dir, "/tmp/eval_cache")
        self.assertEqual(config.data.indexed_eval_cache_mode, "refresh")
        self.assertEqual(config.data.indexed_eval_random_seed, 11)
        self.assertEqual(config.data.indexed_split_seed, 9)
        self.assertEqual(config.data.indexed_window_mode, "nonoverlap_file_stream")
        self.assertEqual(config.data.indexed_train_epoch_mode, "all_windows")
        self.assertEqual(config.data.indexed_file_stream_windows, 123)
        self.assertEqual(config.data.indexed_file_shuffle_buffer_windows, 45)
        self.assertEqual(config.data.indexed_file_stream_order_seed, 6)
        self.assertEqual(config.data.indexed_source_mix_chunk_batches, 7)
        self.assertEqual(config.data.indexed_source_read_chunk_windows, 89)
        self.assertFalse(config.data.indexed_source_read_chunk_shuffle)
        self.assertEqual(config.data.indexed_source_file_order_seed, 10)
        self.assertEqual(config.data.source_loss_weights["gtdb_v220"], 0.5)

    def test_indexed_fasta_compress_mode_is_rejected(self) -> None:
        config = ExperimentConfig()
        config.data.sequence_source_mode = "indexed_fasta"
        config.data.fasta_index_dir = "/tmp/index"

        with self.assertRaises(ValueError):
            _validate_config_for_megabyte(config, mode="compress")

        _validate_config_for_megabyte(config, mode="all")

    def test_indexed_fasta_nonoverlap_all_windows_rejects_source_sampling_weights(self) -> None:
        config = ExperimentConfig()
        config.data.sequence_source_mode = "indexed_fasta"
        config.data.fasta_index_dir = "/tmp/index"
        config.data.indexed_window_mode = "nonoverlap_file_stream"
        config.data.indexed_train_epoch_mode = "all_windows"
        config.data.source_sampling_weights = {"gtdb_v220": 1.0}

        with self.assertRaises(ValueError):
            _validate_config_for_megabyte(config, mode="all")

    def test_indexed_fasta_source_batch_stream_accepts_source_sampling_weights(self) -> None:
        config = ExperimentConfig()
        config.data.sequence_source_mode = "indexed_fasta"
        config.data.fasta_index_dir = "/tmp/index"
        config.data.indexed_window_mode = "source_batch_file_stream"
        config.data.indexed_train_epoch_mode = "samples"
        config.data.source_sampling_weights = {"gtdb_v220": 1.0}

        _validate_config_for_megabyte(config, mode="all")

    def test_dna_cli_parses_repacked_window_options(self) -> None:
        parser = build_dna_parser()
        args = parser.parse_args(
            [
                "--config",
                "dummy.json",
                "--sequence-source-mode",
                "repacked_windows",
                "--repacked-window-dir",
                "/tmp/repacked",
                "--repacked-schedule-dir",
                "/tmp/repacked/schedules/split_seed_0_train_0.98_val_0.01_test_0.01",
                "--repacked-eval-samples",
                "23",
                "--repacked-train-epoch-mode",
                "all_windows",
                "--repacked-read-chunk-windows",
                "456",
                "--repacked-shard-load-mode",
                "mmap",
                "--repacked-shard-sampling-mode",
                "random",
                "--source-loss-weights-json",
                '{"gtdb_v220": 0.7, "metagenomes": 0.3}',
            ]
        )
        config = ExperimentConfig()
        apply_dna_overrides(config, args)

        self.assertEqual(config.data.sequence_source_mode, "repacked_windows")
        self.assertEqual(config.data.repacked_window_dir, "/tmp/repacked")
        self.assertEqual(
            config.data.repacked_schedule_dir,
            "/tmp/repacked/schedules/split_seed_0_train_0.98_val_0.01_test_0.01",
        )
        self.assertEqual(config.data.repacked_eval_samples, 23)
        self.assertEqual(config.data.repacked_train_epoch_mode, "all_windows")
        self.assertEqual(config.data.repacked_read_chunk_windows, 456)
        self.assertEqual(config.data.repacked_shard_load_mode, "mmap")
        self.assertEqual(config.data.repacked_shard_sampling_mode, "random")
        self.assertEqual(config.data.source_loss_weights["gtdb_v220"], 0.7)

    def test_repacked_windows_compress_mode_is_rejected(self) -> None:
        config = ExperimentConfig()
        config.data.sequence_source_mode = "repacked_windows"
        config.data.repacked_window_dir = "/tmp/repacked"

        with self.assertRaises(ValueError):
            _validate_config_for_megabyte(config, mode="compress")

        _validate_config_for_megabyte(config, mode="all")


if __name__ == "__main__":
    unittest.main()
