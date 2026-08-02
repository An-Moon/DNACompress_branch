#!/usr/bin/env python3
from __future__ import annotations

"""Query target probabilities at sequence positions from a probability trace.

Examples:

    python scripts/query_probability_trace_positions.py \
      --trace-dir outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full/traces/OrSa \
      --position 0 --position 8192 --range 100000:100100:10

    python scripts/query_probability_trace_positions.py \
      --trace-dir outputs/traces/orsa/nc_prefix \
      --positions-file positions.txt \
      --output-csv queried_positions.csv
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import (  # noqa: E402
    read_target_probability_trace_positions,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query target probabilities by sequence position.")
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--position", type=int, action="append", default=[], help="0-based sequence position to query.")
    parser.add_argument("--range", action="append", default=[], help="Python-style start:end[:step] position range.")
    parser.add_argument("--positions-file", help="Text file with one integer position per line.")
    parser.add_argument("--output-csv", help="Optional output CSV. Defaults to stdout.")
    return parser


def _parse_range(spec: str) -> list[int]:
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"invalid range {spec!r}; expected start:end[:step]")
    start = int(parts[0])
    end = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    if step <= 0:
        raise ValueError("range step must be positive")
    return list(range(start, end, step))


def main() -> None:
    args = _build_parser().parse_args()
    trace_dir = Path(args.trace_dir)
    positions: list[int] = [int(item) for item in args.position]
    for item in args.range:
        positions.extend(_parse_range(str(item)))
    if args.positions_file:
        with Path(args.positions_file).open() as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    positions.append(int(stripped))
    if not positions:
        raise SystemExit("provide --position, --range, or --positions-file")

    result = read_target_probability_trace_positions(trace_dir, np.asarray(positions, dtype=np.int64))

    rows = [
        {
            "position": int(position),
            "row_index": int(row_index),
            "target_symbol": int(symbol),
            "target_prob": float(prob),
        }
        for position, row_index, symbol, prob in zip(
            result["emit_position"].tolist(),
            result["row_index"].tolist(),
            result["target_symbol"].tolist(),
            result["target_prob"].tolist(),
        )
    ]
    fieldnames = ["position", "row_index", "target_symbol", "target_prob"]
    if args.output_csv:
        with Path(args.output_csv).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
