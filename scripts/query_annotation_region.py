#!/usr/bin/env python3
from __future__ import annotations

"""Query sequence, annotation labels, and trace-derived bpb for a DNACorpus interval.

Examples:

    python scripts/query_annotation_region.py \
      --species EsCo --start 180 --end 260 \
      --analysis-dir outputs/nc_prefix_annotation_region_analysis_v1

    python scripts/query_annotation_region.py \
      --species HoSa --feature-type CDS --limit-bases 1000 \
      --analysis-dir outputs/nc_prefix_annotation_region_analysis_v1 \
      --output-csv outputs/nc_prefix_annotation_region_analysis_v1/query_examples/hosa_cds.csv
"""

import argparse
import csv
import gzip
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.annotation_region_analysis import (  # noqa: E402
    ID_TO_CLASS,
    class_ids_for_positions,
    read_annotation_interval_index,
    read_dnacorpus_sequence_slice,
)
from dna_compress.probability_trace import (  # noqa: E402
    read_target_probability_trace_positions,
)


DEFAULT_ANALYSIS_DIR = REPO_ROOT / "outputs" / "nc_prefix_annotation_region_analysis_v1"
DEFAULT_TRACE_ROOT = (
    REPO_ROOT / "outputs" / "nc_prefix_dnacorpus_best_available_w8192_target_traces_full_position_major" / "traces"
)
DEFAULT_DNACORPUS_DIR = REPO_ROOT / "datasets" / "DNACorpus"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query trace-derived bpb and annotation labels for a DNACorpus interval."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--dnacorpus-dir", type=Path, default=DEFAULT_DNACORPUS_DIR)
    parser.add_argument("--species", required=True)
    parser.add_argument("--start", type=int, help="0-based local DNACorpus start.")
    parser.add_argument("--end", type=int, help="0-based local DNACorpus end, exclusive.")
    parser.add_argument("--feature-type", help="Query the first mapped feature of this GFF3 type.")
    parser.add_argument("--feature-class", help="Optionally restrict --feature-type by mapped class name.")
    parser.add_argument("--limit-bases", type=int, default=10000, help="Maximum bases to print/query.")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV output. Defaults to stdout.")
    return parser


def _feature_interval(index_path: Path, feature_type: str, feature_class: str | None) -> tuple[int, int]:
    feature_table = index_path.with_suffix(".features.tsv.gz")
    with gzip.open(feature_table, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["feature_type"] != feature_type:
                continue
            if feature_class and row["class_name"] != feature_class:
                continue
            return int(row["local_start_0based"]), int(row["local_end_0based_exclusive"])
    detail = f" and class {feature_class!r}" if feature_class else ""
    raise ValueError(f"no mapped feature_type {feature_type!r}{detail} found")


def main() -> None:
    args = _build_parser().parse_args()
    index_path = args.analysis_dir / "annotation_interval_index" / f"{args.species}.npz"
    index = read_annotation_interval_index(index_path)

    if args.feature_type:
        start, end = _feature_interval(index_path, args.feature_type, args.feature_class)
    else:
        if args.start is None or args.end is None:
            raise SystemExit("provide --start/--end or --feature-type")
        start, end = int(args.start), int(args.end)
    if start < 0 or end <= start:
        raise SystemExit("invalid query interval")
    if args.limit_bases and end - start > int(args.limit_bases):
        end = start + int(args.limit_bases)

    positions = np.arange(start, end, dtype=np.int64)
    classes = class_ids_for_positions(index, positions)
    trace_dir = args.trace_root / args.species
    probs = read_target_probability_trace_positions(trace_dir, positions)
    sequence = read_dnacorpus_sequence_slice(args.dnacorpus_dir / args.species, start, end)
    bpb = -np.log2(np.clip(probs["target_prob"], np.finfo(np.float64).tiny, 1.0))

    rows = []
    for offset, position in enumerate(positions.tolist()):
        rows.append(
            {
                "species": args.species,
                "position": int(position),
                "base": sequence[offset],
                "region_class": ID_TO_CLASS[int(classes[offset])],
                "target_symbol": int(probs["target_symbol"][offset]),
                "target_prob": float(probs["target_prob"][offset]),
                "bpb": float(bpb[offset]),
                "trace_row_index": int(probs["row_index"][offset]),
            }
        )

    fieldnames = ["species", "position", "base", "region_class", "target_symbol", "target_prob", "bpb", "trace_row_index"]
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
