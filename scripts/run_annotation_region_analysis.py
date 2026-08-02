#!/usr/bin/env python3
from __future__ import annotations

"""Build DNACorpus annotation region indexes and aggregate trace or fused-trace bpb.

Examples:

    python scripts/run_annotation_region_analysis.py \
      --species EsCo --species BuEb \
      --output-dir outputs/nc_prefix_annotation_region_analysis_v1

    python scripts/run_annotation_region_analysis.py \
      --all-mappable \
      --output-dir outputs/nc_prefix_annotation_region_analysis_v1

    python scripts/run_annotation_region_analysis.py \
      --all-mappable \
      --output-dir outputs/dnacorpus_w8192_comprehensive_analysis_v1/fusion_online_hedge_eta0.05_init0.5_position_major_v1/annotation_region_analysis \
      --fused-left-trace-root outputs/carbon3b_dnacorpus_w8192_target_traces_position_major/traces \
      --fused-right-trace-root outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full_position_major/traces
"""

import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.annotation_region_analysis import (  # noqa: E402
    MAPPABLE_STATUSES,
    aggregate_fused_traces_by_annotation_regions,
    aggregate_trace_by_annotation_regions,
    build_annotation_interval_index,
    json_default,
    read_mapping_validation,
    write_csv,
)


DEFAULT_ANNOTATION_DIR = REPO_ROOT / "datasets" / "DNACorpus_annotations_official"
DEFAULT_TRACE_ROOT = (
    REPO_ROOT / "outputs" / "nc_prefix_dnacorpus_best_available_w8192_target_traces_full_position_major" / "traces"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "nc_prefix_annotation_region_analysis_v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze target-probability traces or fused trace probabilities by mapped DNACorpus GFF3 regions."
    )
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--fused-left-trace-root", type=Path, help="Left trace root for offline fused region aggregation.")
    parser.add_argument("--fused-right-trace-root", type=Path, help="Right trace root for offline fused region aggregation.")
    parser.add_argument("--fusion-eta", type=float, default=0.05)
    parser.add_argument("--fusion-initial-left-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--species", action="append", default=[], help="DNACorpus species code to analyze.")
    parser.add_argument("--all-mappable", action="store_true", help="Analyze every coordinate-verified species.")
    parser.add_argument("--build-only", action="store_true", help="Only build annotation indexes.")
    parser.add_argument("--aggregate-only", action="store_true", help="Only aggregate using existing annotation indexes.")
    parser.add_argument("--force", action="store_true", help="Rebuild existing annotation indexes.")
    parser.add_argument("--window-bases", type=int, default=8192, help="Optional window size for window summary.")
    parser.add_argument(
        "--verify-trace-checksum",
        action="store_true",
        help="Verify trace shard checksum while aggregating. This is slower on large traces.",
    )
    return parser


def _select_species(args: argparse.Namespace) -> list[str]:
    rows = read_mapping_validation(args.annotation_dir, repo_root=REPO_ROOT)
    selected = list(args.species)
    if args.all_mappable or not selected:
        selected = [code for code, row in rows.items() if row.get("status") in MAPPABLE_STATUSES]
    return selected


def _write_skipped(output_dir: Path, skipped: list[dict[str, str]]) -> None:
    if not skipped:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skipped_species.json").write_text(json.dumps(skipped, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    if args.build_only and args.aggregate_only:
        raise SystemExit("--build-only and --aggregate-only are mutually exclusive")
    fused_mode = args.fused_left_trace_root is not None or args.fused_right_trace_root is not None
    if fused_mode and (args.fused_left_trace_root is None or args.fused_right_trace_root is None):
        raise SystemExit("provide both --fused-left-trace-root and --fused-right-trace-root")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    species_list = _select_species(args)

    summary_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    run_results: dict[str, object] = {
        "schema_version": 1,
        "annotation_dir": str(args.annotation_dir),
        "trace_root": str(args.trace_root),
        "fused_left_trace_root": str(args.fused_left_trace_root) if args.fused_left_trace_root else None,
        "fused_right_trace_root": str(args.fused_right_trace_root) if args.fused_right_trace_root else None,
        "fusion_eta": float(args.fusion_eta),
        "fusion_initial_left_weight": float(args.fusion_initial_left_weight),
        "output_dir": str(output_dir),
        "species": species_list,
        "build_only": bool(args.build_only),
        "aggregate_only": bool(args.aggregate_only),
        "window_bases": int(args.window_bases) if args.window_bases else None,
        "results": {},
        "skipped": [],
    }
    skipped: list[dict[str, str]] = []

    for species in species_list:
        trace_dir = (
            args.fused_left_trace_root / species
            if fused_mode and args.fused_left_trace_root is not None
            else args.trace_root / species
        )
        index_path = output_dir / "annotation_interval_index" / f"{species}.npz"
        try:
            if args.aggregate_only:
                from dna_compress.annotation_region_analysis import read_annotation_interval_index

                index = read_annotation_interval_index(index_path)
            else:
                index = build_annotation_interval_index(
                    species=species,
                    annotation_dir=args.annotation_dir,
                    output_dir=output_dir,
                    trace_dir=trace_dir if trace_dir.exists() else None,
                    repo_root=REPO_ROOT,
                    overwrite=bool(args.force),
                )
            if args.build_only:
                run_results["results"][species] = {"annotation_index": str(index.path)}
                print(json.dumps({"event": "built_index", "species": species, "path": str(index.path)}))
                continue
            if not trace_dir.exists():
                raise FileNotFoundError(f"trace dir missing: {trace_dir}")
            if fused_mode:
                assert args.fused_left_trace_root is not None
                assert args.fused_right_trace_root is not None
                right_trace_dir = args.fused_right_trace_root / species
                if not right_trace_dir.exists():
                    raise FileNotFoundError(f"right trace dir missing: {right_trace_dir}")
                result = aggregate_fused_traces_by_annotation_regions(
                    left_trace_dir=trace_dir,
                    right_trace_dir=right_trace_dir,
                    annotation_index=index,
                    fusion_eta=float(args.fusion_eta),
                    fusion_initial_left_weight=float(args.fusion_initial_left_weight),
                    window_bases=int(args.window_bases) if args.window_bases else None,
                )
            else:
                result = aggregate_trace_by_annotation_regions(
                    trace_dir=trace_dir,
                    annotation_index=index,
                    window_bases=int(args.window_bases) if args.window_bases else None,
                    verify_trace_checksum=bool(args.verify_trace_checksum),
                )
            species_json = output_dir / "per_species" / f"{species}.json"
            species_json.parent.mkdir(parents=True, exist_ok=True)
            species_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
            summary_rows.extend(result["region_rows"])
            window_rows.extend(result["window_rows"])
            run_results["results"][species] = {
                "annotation_index": str(index.path),
                "per_species_json": str(species_json),
                "elapsed_seconds": result["elapsed_seconds"],
                "bases_per_second": result["bases_per_second"],
            }
            print(
                json.dumps(
                    {
                        "event": "aggregated_species",
                        "species": species,
                        "rows_seen": result["rows_seen"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "bases_per_second": result["bases_per_second"],
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001 - batch runner should keep other species moving.
            row = {"species": species, "reason": str(exc)}
            skipped.append(row)
            run_results["skipped"].append(row)
            print(json.dumps({"event": "skip_species", **row}), file=sys.stderr)

    if summary_rows:
        fieldnames = [
            "species",
            "region_class",
            "base_count",
            "interval_count",
            "coverage_fraction",
            "sum_bits",
            "mean_bpb",
            "min_bpb",
            "max_bpb",
        ]
        write_csv(output_dir / "region_class_summary.csv", summary_rows, fieldnames)
        (output_dir / "region_class_summary.json").write_text(
            json.dumps(summary_rows, indent=2, ensure_ascii=False, default=json_default),
            encoding="utf-8",
        )
    if window_rows:
        write_csv(
            output_dir / "region_window_summary.csv",
            window_rows,
            [
                "species",
                "window_id",
                "start",
                "end",
                "base_count",
                "mean_bpb",
                "dominant_region_class",
                "dominant_region_fraction",
            ],
        )
    _write_skipped(output_dir, skipped)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_results, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
