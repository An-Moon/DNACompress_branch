#!/usr/bin/env python3
from __future__ import annotations

"""Fuse two target-probability traces without loading the source models."""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import fuse_target_probability_traces  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline target-probability trace fusion.")
    parser.add_argument("--trace-a", required=True)
    parser.add_argument("--trace-b", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--fusion-eta", type=float, default=0.05)
    parser.add_argument("--fusion-initial-lm-weight", type=float, default=0.5)
    parser.add_argument(
        "--no-verify-checksum",
        action="store_true",
        help="Skip shard checksum verification while reading traces.",
    )
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_outputs(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    row = {
        "codec": metrics.get("codec"),
        "trace_mode": metrics.get("trace_mode"),
        "sample_bases": metrics.get("sample_bases"),
        "core_base_count": metrics.get("core_base_count"),
        "window_count": metrics.get("window_count"),
        "window_bases": metrics.get("window_bases"),
        "token_merge_size": metrics.get("token_merge_size"),
        "theoretical_bits_per_base": metrics.get("theoretical_bits_per_base"),
        "core_theoretical_bits_per_base": metrics.get("core_theoretical_bits_per_base"),
        "lm_only_theoretical_bits_per_base": metrics.get("lm_only_theoretical_bits_per_base"),
        "nc_prefix_only_theoretical_bits_per_base": metrics.get("nc_prefix_only_theoretical_bits_per_base"),
        "arithmetic_bits_per_base": metrics.get("arithmetic_bits_per_base"),
        "fusion_final_mean_lm_weight": metrics.get("fusion_final_mean_lm_weight"),
        "compression_bases_per_second": metrics.get("compression_bases_per_second"),
        "compression_process_seconds": metrics.get("compression_process_seconds"),
        "trace_left": metrics.get("trace_left"),
        "trace_right": metrics.get("trace_right"),
    }
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _add_model_aliases(metrics: dict[str, Any]) -> None:
    left_family = str(metrics.get("left_model_family", ""))
    right_family = str(metrics.get("right_model_family", ""))
    if left_family == "megabyte":
        metrics["lm_only_theoretical_bits"] = metrics.get("left_only_theoretical_bits")
        metrics["lm_only_theoretical_bits_per_base"] = metrics.get("left_only_theoretical_bits_per_base")
    if right_family == "megabyte":
        metrics["lm_only_theoretical_bits"] = metrics.get("right_only_theoretical_bits")
        metrics["lm_only_theoretical_bits_per_base"] = metrics.get("right_only_theoretical_bits_per_base")
    if left_family == "nc_prefix":
        metrics["nc_prefix_only_theoretical_bits"] = metrics.get("left_only_theoretical_bits")
        metrics["nc_prefix_only_theoretical_bits_per_base"] = metrics.get("left_only_theoretical_bits_per_base")
    if right_family == "nc_prefix":
        metrics["nc_prefix_only_theoretical_bits"] = metrics.get("right_only_theoretical_bits")
        metrics["nc_prefix_only_theoretical_bits_per_base"] = metrics.get("right_only_theoretical_bits_per_base")


def main() -> None:
    args = _build_parser().parse_args()
    metrics = fuse_target_probability_traces(
        args.trace_a,
        args.trace_b,
        fusion_eta=float(args.fusion_eta),
        fusion_initial_lm_weight=float(args.fusion_initial_lm_weight),
        verify_checksum=not bool(args.no_verify_checksum),
    )
    _add_model_aliases(metrics)
    output_json = Path(args.output_json)
    _write_outputs(output_json, metrics)
    print(json.dumps(_json_safe({"output_json": str(output_json), **metrics}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
