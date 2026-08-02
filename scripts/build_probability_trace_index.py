#!/usr/bin/env python3
from __future__ import annotations

"""Build compact position indexes for target-probability traces.

Examples:

    python scripts/build_probability_trace_index.py \
      --trace-root outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full/traces

    python scripts/build_probability_trace_index.py \
      --trace-dir outputs/traces/orsa/nc_prefix \
      --check-sample 32
"""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import (  # noqa: E402
    build_probability_trace_position_index,
    read_target_probability_trace_positions,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build compact trace position indexes.")
    parser.add_argument("--trace-dir", action="append", default=[], help="Trace directory containing manifest.json.")
    parser.add_argument("--trace-root", help="Directory whose children are trace directories.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing position indexes.")
    parser.add_argument("--check-sample", type=int, default=0, help="Optionally read this many evenly spaced positions after indexing.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    trace_dirs = [Path(item) for item in args.trace_dir]
    if args.trace_root:
        root = Path(args.trace_root)
        trace_dirs.extend(sorted(path for path in root.iterdir() if (path / "manifest.json").exists()))
    if not trace_dirs:
        raise SystemExit("provide --trace-dir or --trace-root")

    for trace_dir in trace_dirs:
        index = build_probability_trace_position_index(trace_dir, overwrite=bool(args.force))
        print(f"{trace_dir}: wrote {index.index_path} rows={index.row_count} shards={len(index.shards)}")
        if int(args.check_sample) > 0 and index.row_count > 0:
            count = min(int(args.check_sample), int(index.core_base_count))
            if count > 0:
                step = max(1, int(index.core_base_count) // count)
                positions = list(range(0, int(index.core_base_count), step))[:count]
                result = read_target_probability_trace_positions(trace_dir, positions, index=index)
                print(f"{trace_dir}: checked {len(result['target_prob'])} positions")


if __name__ == "__main__":
    main()
