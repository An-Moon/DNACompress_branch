#!/usr/bin/env python3
from __future__ import annotations

"""Offline-fuse Carbon 3B and nc_prefix DNACorpus target-probability traces."""

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


DEFAULT_SPECIES_ORDER = [
    "HoSa",
    "GaGa",
    "AnCa",
    "DaRe",
    "OrSa",
    "DrMe",
    "EnIn",
    "ScPo",
    "WaMe",
    "PlFa",
    "EsCo",
    "HaHi",
    "HePy",
    "AeCa",
    "YeMi",
    "AgPh",
    "BuEb",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline online-hedge fusion for Carbon 3B and nc_prefix DNACorpus traces."
    )
    parser.add_argument(
        "--carbon-trace-root",
        default="outputs/carbon3b_dnacorpus_w8192_target_traces_position_major/traces",
        help="Position-major Carbon trace root.",
    )
    parser.add_argument(
        "--nc-prefix-trace-root",
        default="outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full_position_major/traces",
        help="Position-major nc_prefix trace root.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dnacorpus_w8192_comprehensive_analysis_v1/fusion_online_hedge_eta0.05_init0.5",
        help="Directory for per-species fusion JSON files and summary tables.",
    )
    parser.add_argument("--fusion-eta", type=float, default=0.05)
    parser.add_argument("--fusion-initial-carbon-weight", type=float, default=0.5)
    parser.add_argument("--species", nargs="*", default=None, help="Optional species list; defaults to DNACorpus order.")
    parser.add_argument(
        "--verify-checksum",
        action="store_true",
        help="Verify trace shard checksums while reading. Slower for full DNACorpus.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-species JSON outputs.")
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _weighted_mean(rows: list[dict[str, Any]], key: str) -> float:
    total_bases = sum(int(row["sample_bases"]) for row in rows)
    if total_bases <= 0:
        return float("nan")
    return sum(float(row[key]) * int(row["sample_bases"]) for row in rows) / total_bases


def _pick_trace_pair(args: argparse.Namespace, species: str) -> tuple[Path, Path, str]:
    return Path(args.carbon_trace_root) / species, Path(args.nc_prefix_trace_root) / species, "position_major"


def _row_from_metrics(
    *,
    name: str,
    metrics: dict[str, Any],
    initial_weight: float,
    trace_selection: str,
) -> dict[str, Any]:
    return {
        "species": name,
        "sample_bases": int(metrics["sample_bases"]),
        "core_base_count": int(metrics["core_base_count"]),
        "window_bases": int(metrics["window_bases"]),
        "trace_emission_order": str(metrics.get("trace_emission_order", "")),
        "trace_selection": str(trace_selection),
        "fusion_eta": float(metrics["fusion_eta"]),
        "fusion_initial_carbon_weight": float(initial_weight),
        "fusion_final_mean_carbon_weight": float(metrics["fusion_final_mean_carbon_weight"]),
        "fused_theoretical_bpb": float(metrics["theoretical_bits_per_base"]),
        "carbon3b_only_theoretical_bpb": float(metrics["left_only_theoretical_bits_per_base"]),
        "nc_prefix_only_theoretical_bpb": float(metrics["right_only_theoretical_bits_per_base"]),
        "fused_minus_carbon3b_bpb": float(metrics["theoretical_bits_per_base"])
        - float(metrics["left_only_theoretical_bits_per_base"]),
        "fused_minus_nc_prefix_bpb": float(metrics["theoretical_bits_per_base"])
        - float(metrics["right_only_theoretical_bits_per_base"]),
        "process_seconds": float(metrics["compression_process_seconds"]),
        "bases_per_second": float(metrics["compression_bases_per_second"]),
        "trace_left": metrics["trace_left"],
        "trace_right": metrics["trace_right"],
    }


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    per_species_dir = output_dir / "per_species"
    per_species_dir.mkdir(parents=True, exist_ok=True)

    species = args.species or DEFAULT_SPECIES_ORDER
    rows: list[dict[str, Any]] = []
    per_species_metrics: dict[str, dict[str, Any]] = {}
    for name in species:
        carbon_trace, nc_trace, trace_selection = _pick_trace_pair(args, name)
        if not (carbon_trace / "manifest.json").exists() or not (nc_trace / "manifest.json").exists():
            print(f"[skip] {name}: missing manifest", file=sys.stderr)
            continue
        per_species_json = per_species_dir / f"{name}.json"
        if bool(args.skip_existing) and per_species_json.exists():
            metrics = json.loads(per_species_json.read_text(encoding="utf-8"))
            trace_selection = str(metrics.get("trace_selection", trace_selection))
            print(f"[reuse] {name} {trace_selection}", file=sys.stderr, flush=True)
        else:
            print(f"[fuse] {name} {trace_selection}", file=sys.stderr, flush=True)
            metrics = fuse_target_probability_traces(
                carbon_trace,
                nc_trace,
                fusion_eta=float(args.fusion_eta),
                fusion_initial_lm_weight=float(args.fusion_initial_carbon_weight),
                verify_checksum=bool(args.verify_checksum),
            )
            metrics["left_model_label"] = "Carbon 3B"
            metrics["right_model_label"] = "nc_prefix"
            metrics["fusion_final_mean_carbon_weight"] = metrics["fusion_final_mean_left_weight"]
            metrics["trace_selection"] = trace_selection
            per_species_json.write_text(
                json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        metrics.setdefault("left_model_label", "Carbon 3B")
        metrics.setdefault("right_model_label", "nc_prefix")
        metrics.setdefault("fusion_final_mean_carbon_weight", metrics.get("fusion_final_mean_left_weight"))
        metrics.setdefault("trace_selection", trace_selection)
        per_species_metrics[name] = metrics
        rows.append(
            _row_from_metrics(
                name=name,
                metrics=metrics,
                initial_weight=float(args.fusion_initial_carbon_weight),
                trace_selection=trace_selection,
            )
        )

    fieldnames = [
        "species",
        "sample_bases",
        "core_base_count",
        "window_bases",
        "trace_emission_order",
        "trace_selection",
        "fusion_eta",
        "fusion_initial_carbon_weight",
        "fusion_final_mean_carbon_weight",
        "fused_theoretical_bpb",
        "carbon3b_only_theoretical_bpb",
        "nc_prefix_only_theoretical_bpb",
        "fused_minus_carbon3b_bpb",
        "fused_minus_nc_prefix_bpb",
        "process_seconds",
        "bases_per_second",
        "trace_left",
        "trace_right",
    ]
    summary_csv = output_dir / "carbon3b_nc_prefix_fusion_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "fusion_policy": "online_hedge_linear_target_probability_trace",
        "decodable_design": "target_probability_trace_non_arithmetic",
        "species_count": len(rows),
        "total_bases": sum(int(row["sample_bases"]) for row in rows),
        "fusion_eta": float(args.fusion_eta),
        "fusion_initial_carbon_weight": float(args.fusion_initial_carbon_weight),
        "weighted_fused_theoretical_bpb": _weighted_mean(rows, "fused_theoretical_bpb"),
        "weighted_carbon3b_only_theoretical_bpb": _weighted_mean(rows, "carbon3b_only_theoretical_bpb"),
        "weighted_nc_prefix_only_theoretical_bpb": _weighted_mean(rows, "nc_prefix_only_theoretical_bpb"),
        "summary_csv": str(summary_csv),
        "per_species_dir": str(per_species_dir),
        "per_species": per_species_metrics,
    }
    summary_json = output_dir / "carbon3b_nc_prefix_fusion_summary.json"
    summary_json.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_safe({k: v for k, v in summary.items() if k != "per_species"}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
