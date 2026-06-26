from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.plot_compression_curves import _build_split_mode_rows, generate_artifacts_for_compression_compare, generate_curves_for_root


class _FakeAxis:
    def __init__(self) -> None:
        self.transAxes = object()

    def plot(self, *args, **kwargs) -> None:
        del args, kwargs

    def text(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_ylabel(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_title(self, *args, **kwargs) -> None:
        del args, kwargs

    def grid(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_xlabel(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_xticks(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_xticklabels(self, *args, **kwargs) -> None:
        del args, kwargs

    def tick_params(self, *args, **kwargs) -> None:
        del args, kwargs

    def legend(self, *args, **kwargs) -> None:
        del args, kwargs


class _FakeFigure:
    def tight_layout(self) -> None:
        return None

    def savefig(self, path: str | Path, dpi: int = 160) -> None:
        del dpi
        Path(path).write_bytes(b"fake-png")


class _FakePyplot:
    def subplots(self, rows: int, cols: int, figsize: tuple[float, float], sharex: bool = False):
        del cols, figsize, sharex
        return _FakeFigure(), [_FakeAxis() for _ in range(rows)]

    def close(self, figure: _FakeFigure) -> None:
        del figure


class PlotCompressionCurvesTests(unittest.TestCase):
    def test_build_split_mode_rows_derives_expected_metrics(self) -> None:
        compression_compare = {
            "dataset": {
                "species": [
                    {"species": "BuEb", "source_name": "BuEb", "total_size": 18_940},
                    {"species": "HoSa", "source_name": "HoSa", "total_size": 189_752_667},
                ]
            },
            "results": {
                "train": {
                    "geco2_paper_modes": {
                        "per_source": [
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_coded_bytes": 210,
                                "arithmetic_bits_per_base": 1.68,
                                "geco2_level": 12,
                            },
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_coded_bytes": 260,
                                "arithmetic_bits_per_base": 2.08,
                                "geco2_level": 1,
                            },
                        ]
                    },
                    "windows_nonoverlap": {
                        "per_source": [
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "arithmetic_bits_per_base": 1.5,
                                "compressed_bpb_payload_only": 1.35,
                                "arithmetic_payload_bytes": 169,
                                "framing_bytes": 19,
                                "theoretical_bits_per_base": 1.4,
                                "compression_bases_per_second": 2_000_000.0,
                                "compression_bytes_per_second": 500_000.0,
                            },
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "arithmetic_bits_per_base": 1.25,
                                "compressed_bpb_payload_only": 1.1,
                                "arithmetic_payload_bytes": 138,
                                "framing_bytes": 18,
                                "theoretical_bits_per_base": 1.2,
                                "compression_bases_per_second": 3_000_000.0,
                                "compression_bytes_per_second": 750_000.0,
                            },
                        ]
                    }
                }
            },
        }

        rows = _build_split_mode_rows(
            compression_compare=compression_compare,
            split_name="train",
            mode_name="windows_nonoverlap",
        )

        self.assertEqual([row["source_name"] for row in rows], ["HoSa", "BuEb"])
        self.assertAlmostEqual(float(rows[1]["vs_2bit_percent"]), 62.5)
        self.assertAlmostEqual(float(rows[1]["payload_only_bits_per_base"]), 1.1)
        self.assertAlmostEqual(float(rows[1]["payload_only_vs_2bit_percent"]), 55.0)
        self.assertEqual(int(rows[1]["arithmetic_payload_bytes"]), 138)
        self.assertEqual(int(rows[1]["framing_bytes"]), 18)
        self.assertAlmostEqual(float(rows[1]["paper_baseline_bpb"]), 4686 * 8 / 18_940)
        self.assertEqual(int(rows[1]["paper_baseline_compressed_bytes"]), 4686)
        self.assertEqual(int(rows[1]["paper_baseline_geco2_mode"]), 1)
        self.assertAlmostEqual(float(rows[1]["compression_mbases_per_second"]), 3.0)
        self.assertAlmostEqual(float(rows[0]["compression_mbytes_per_second"]), 0.5)

    def test_generate_artifacts_for_compression_compare_writes_csv_and_png(self) -> None:
        compression_compare = {
            "dataset": {
                "species": [
                    {"species": "BuEb", "source_name": "BuEb", "total_size": 18_940},
                    {"species": "HoSa", "source_name": "HoSa", "total_size": 189_752_667},
                ]
            },
            "results": {
                "train": {
                    "geco2_paper_modes": {
                        "per_source": [
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_coded_bytes": 260,
                                "arithmetic_bits_per_base": 2.08,
                                "geco2_level": 1,
                            },
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_coded_bytes": 210,
                                "arithmetic_bits_per_base": 1.68,
                                "geco2_level": 12,
                            },
                        ]
                    },
                    "windows_nonoverlap": {
                        "per_source": [
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_bits_per_base": 1.25,
                                "compressed_bpb_payload_only": 1.1,
                                "arithmetic_payload_bytes": 138,
                                "framing_bytes": 18,
                                "theoretical_bits_per_base": 1.2,
                                "compression_bases_per_second": 3_000_000.0,
                                "compression_bytes_per_second": 750_000.0,
                            },
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_bits_per_base": 1.5,
                                "compressed_bpb_payload_only": 1.35,
                                "arithmetic_payload_bytes": 169,
                                "framing_bytes": 19,
                                "theoretical_bits_per_base": 1.4,
                                "compression_bases_per_second": 2_000_000.0,
                                "compression_bytes_per_second": 500_000.0,
                            },
                        ]
                    }
                }
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            stats_dir = Path(tmpdir)
            compression_compare_path = stats_dir / "compression_compare.json"
            compression_compare_path.write_text(__import__("json").dumps(compression_compare), encoding="utf-8")

            with patch("scripts.plot_compression_curves._load_matplotlib_pyplot", return_value=_FakePyplot()):
                generated_paths = generate_artifacts_for_compression_compare(compression_compare_path)

            self.assertEqual(len(generated_paths), 5)
            csv_path = stats_dir / "compression_curves" / "windows_nonoverlap_compression_curve_data.csv"
            png_path = stats_dir / "compression_curves" / "windows_nonoverlap_compression_curves.png"
            payload_only_png_path = stats_dir / "compression_curves" / "windows_nonoverlap_payload_only_compression_curves.png"
            paper_baseline_path = stats_dir / "compression_curves" / "baselines" / "paper_baseline.csv"
            geco2_baseline_path = stats_dir / "compression_curves" / "baselines" / "geco2_experiment_baseline.csv"
            self.assertTrue(csv_path.exists())
            self.assertTrue(png_path.exists())
            self.assertTrue(payload_only_png_path.exists())
            self.assertTrue(paper_baseline_path.exists())
            self.assertTrue(geco2_baseline_path.exists())

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["source_name"], "BuEb")
            self.assertEqual(rows[1]["vs_2bit_percent"], "62.5")
            self.assertEqual(rows[1]["payload_only_bits_per_base"], "1.1")
            self.assertEqual(rows[1]["payload_only_vs_2bit_percent"], "55.00000000000001")
            self.assertEqual(rows[1]["arithmetic_payload_bytes"], "138")
            self.assertEqual(rows[1]["framing_bytes"], "18")
            self.assertEqual(rows[1]["paper_baseline_compressed_bytes"], "4686")

    def test_generate_curves_for_root_auto_geco2_baseline_writes_experiment_overlay(self) -> None:
        compression_compare = {
            "dataset": {
                "species": [
                    {"species": "BuEb", "source_name": "BuEb", "total_size": 18_940},
                    {"species": "HoSa", "source_name": "HoSa", "total_size": 189_752_667},
                ]
            },
            "results": {
                "full": {
                    "windows_nonoverlap": {
                        "per_source": [
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_bits_per_base": 1.25,
                                "compressed_bpb_payload_only": 1.1,
                            },
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_bits_per_base": 1.5,
                                "compressed_bpb_payload_only": 1.35,
                            },
                        ]
                    }
                }
            },
        }
        baseline_compare = {
            "results": {
                "full": {
                    "geco2_paper_modes": {
                        "per_source": [
                            {
                                "species": "BuEb",
                                "source_name": "BuEb",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_coded_bytes": 260,
                                "arithmetic_bits_per_base": 2.08,
                                "geco2_level": 1,
                            },
                            {
                                "species": "HoSa",
                                "source_name": "HoSa",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_coded_bytes": 210,
                                "arithmetic_bits_per_base": 1.68,
                                "geco2_level": 12,
                            },
                        ]
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            compression_compare_path = root_dir / "compression_compare.json"
            baseline_path = root_dir / "baseline.json"
            compression_compare_path.write_text(__import__("json").dumps(compression_compare), encoding="utf-8")
            baseline_path.write_text(__import__("json").dumps(baseline_compare), encoding="utf-8")

            def resolve_baseline(selector: str | None) -> Path | None:
                self.assertEqual(selector, "dnacorpus_fullsplit")
                return baseline_path

            with patch("scripts.plot_compression_curves.resolve_geco2_baseline_path", side_effect=resolve_baseline):
                with patch("scripts.plot_compression_curves._load_matplotlib_pyplot", return_value=_FakePyplot()):
                    generated_paths = generate_curves_for_root(root_dir, geco2_baseline_selector="auto")

            self.assertEqual(len(generated_paths), 5)
            csv_path = root_dir / "compression_curves" / "full_windows_nonoverlap_compression_curve_data.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[1]["source_name"], "BuEb")
            self.assertEqual(rows[1]["experiment_baseline_bpb"], "2.08")
            self.assertEqual(rows[1]["experiment_baseline_geco2_mode"], "1")

    def test_generate_curves_for_root_auto_geco2_baseline_detects_opengenome2(self) -> None:
        compression_compare = {
            "input_dir": "/data/opengenome2_subset/fasta_test_subset_100mb_per_source",
            "dataset": {
                "dataset_dir": "/data/opengenome2_subset/fasta_test_subset_100mb_per_source",
                "species": [
                    {"species": "gtdb_v220", "source_name": "gtdb_v220", "total_size": 1000},
                ],
            },
            "results": {
                "test": {
                    "windows_nonoverlap": {
                        "per_source": [
                            {
                                "species": "gtdb_v220",
                                "source_name": "gtdb_v220",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_bits_per_base": 1.25,
                                "compressed_bpb_payload_only": 1.1,
                            },
                        ]
                    }
                }
            },
        }
        baseline_compare = {
            "results": {
                "test": {
                    "geco2_paper_modes": {
                        "per_source": [
                            {
                                "species": "gtdb_v220",
                                "source_name": "gtdb_v220",
                                "sample_bytes": 1000,
                                "sample_bases": 1000,
                                "arithmetic_coded_bytes": 200,
                                "arithmetic_bits_per_base": 1.6,
                                "geco2_level": "opengenome2_100mb",
                            },
                        ]
                    }
                }
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            compression_compare_path = root_dir / "compression_compare.json"
            baseline_path = root_dir / "opengenome2_baseline.json"
            compression_compare_path.write_text(__import__("json").dumps(compression_compare), encoding="utf-8")
            baseline_path.write_text(__import__("json").dumps(baseline_compare), encoding="utf-8")

            def resolve_baseline(selector: str | None) -> Path | None:
                self.assertEqual(selector, "opengenome2_100mb")
                return baseline_path

            with patch("scripts.plot_compression_curves.resolve_geco2_baseline_path", side_effect=resolve_baseline):
                with patch("scripts.plot_compression_curves._load_matplotlib_pyplot", return_value=_FakePyplot()):
                    generate_curves_for_root(root_dir, geco2_baseline_selector="auto")

            csv_path = root_dir / "compression_curves" / "test_windows_nonoverlap_compression_curve_data.csv"
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["experiment_baseline_bpb"], "1.6")
            self.assertEqual(rows[0]["experiment_baseline_geco2_mode"], "opengenome2_100mb")


if __name__ == "__main__":
    unittest.main()
