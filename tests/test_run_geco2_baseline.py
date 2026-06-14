from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from DNACompress.scripts.run_geco2_dnacorpus_baseline import (
    DEFAULT_MODE_NAME,
    build_geco2_command,
    resolve_geco2_level,
    run_geco2_baseline,
    run_geco2_file,
)


@dataclass
class _FakeSplits:
    train_sources: list[bytes]
    val_sources: list[bytes]
    test_sources: list[bytes]
    summary: dict[str, object]


class RunGeco2BaselineTests(unittest.TestCase):
    def test_build_geco2_command_uses_force_verbose_and_level(self) -> None:
        command = build_geco2_command("/usr/bin/GeCo2", level=5, input_path=Path("sample.seq"))

        self.assertEqual(command, ["/usr/bin/GeCo2", "-F", "-v", "-l", "5", "sample.seq"])

    def test_resolve_geco2_level_uses_paper_mode_by_species(self) -> None:
        self.assertEqual(resolve_geco2_level(species="HoSa", source_name="HoSa", default_level=5), 12)
        self.assertEqual(resolve_geco2_level(species="Unknown", source_name="Unknown", default_level=5), 5)

    def test_run_geco2_file_reads_created_co_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.seq"
            input_path.write_bytes(b"ACGT")

            def fake_run(command, check, capture_output, text):
                del command, check, capture_output, text
                Path(str(input_path) + ".co").write_bytes(b"compressed")
                return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

            with patch("scripts.run_geco2_baseline.subprocess.run", side_effect=fake_run):
                result = run_geco2_file(binary="GeCo2", input_path=input_path, level=5)

            self.assertEqual(result["compressed_bytes"], len(b"compressed"))
            self.assertEqual(result["returncode"], 0)

    def test_run_geco2_baseline_merges_mode_and_preserves_existing_results(self) -> None:
        metrics = {
            "models": [
                {
                    "config": {
                        "data": {
                            "dataset_dir": "datasets/DNACorpus",
                            "species": ["SpA"],
                            "train_ratio": 0.6,
                            "val_ratio": 0.2,
                            "test_ratio": 0.2,
                            "compression_sample_bytes": 8,
                        }
                    }
                }
            ],
            "dataset": {"species": [{"species": "SpA", "source_name": "SpA"}]},
            "results": {
                "train": {
                    "static_context": {
                        "aggregate": {"source_count": 1},
                        "per_source": [{"species": "SpA", "source_name": "SpA"}],
                    }
                }
            },
        }
        fake_splits = _FakeSplits(
            train_sources=[b"ACGTACGTACGT"],
            val_sources=[b"ACGT"],
            test_sources=[b"ACGT"],
            summary={"species": [{"species": "SpA", "source_name": "SpA"}]},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            compression_json = Path(tmpdir) / "compression_compare.json"
            output_json = Path(tmpdir) / "out.json"
            compression_json.write_text(json.dumps(metrics), encoding="utf-8")

            with (
                patch("scripts.run_geco2_baseline.load_splits", return_value=fake_splits),
                patch(
                    "scripts.run_geco2_baseline.compress_payload_with_geco2",
                    return_value={
                        "compressed_bytes": 3,
                        "seconds": 0.5,
                        "command": ["GeCo2"],
                        "returncode": 0,
                        "stdout_tail": "",
                        "stderr_tail": "",
                    },
                ),
                patch("scripts.run_geco2_baseline.generate_artifacts_for_compression_compare", return_value=[]),
            ):
                merged = run_geco2_baseline(
                    compression_json=compression_json,
                    output_json=output_json,
                    binary="GeCo2",
                    level=5,
                    split_names=["train"],
                    compression_sample_bytes=8,
                )

            self.assertIn("static_context", merged["results"]["train"])
            self.assertIn(DEFAULT_MODE_NAME, merged["results"]["train"])
            aggregate = merged["results"]["train"][DEFAULT_MODE_NAME]["aggregate"]
            self.assertEqual(aggregate["source_count"], 1)
            self.assertAlmostEqual(float(aggregate["total_arithmetic_bits_per_base"]), 3.0)
            aggregate_csv = Path(tmpdir) / "compression_aggregate_by_split_mode.csv"
            per_source_csv = Path(tmpdir) / "compression_per_source_by_split_mode.csv"
            self.assertTrue(aggregate_csv.exists())
            self.assertIn(f",{DEFAULT_MODE_NAME},", per_source_csv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
