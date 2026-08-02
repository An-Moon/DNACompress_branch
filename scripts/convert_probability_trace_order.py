#!/usr/bin/env python3
from __future__ import annotations

"""Convert target-probability traces to the canonical position-major order.

Examples:

    python scripts/convert_probability_trace_order.py \
      --trace-dir outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full/traces/BuEb \
      --output-dir outputs/trace_position_major_smoke/nc_prefix/BuEb \
      --overwrite

    python scripts/convert_probability_trace_order.py \
      --trace-root outputs/carbon3b_dnacorpus_w8192_target_traces/traces \
      --output-root outputs/carbon3b_dnacorpus_w8192_target_traces_position_major/traces
"""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import convert_probability_trace_to_position_major  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert target-probability traces to position_major_v1.")
    parser.add_argument("--trace-dir", action="append", default=[], help="Input trace directory containing manifest.json.")
    parser.add_argument("--trace-root", help="Input root whose child directories are traces.")
    parser.add_argument("--output-dir", action="append", default=[], help="Output trace dir for the matching --trace-dir.")
    parser.add_argument("--output-root", help="Output root used with --trace-root or multiple --trace-dir.")
    parser.add_argument("--species", nargs="*", default=None, help="Optional child names to convert when --trace-root is used.")
    parser.add_argument("--shard-rows", type=int, default=None, help="Rows per output shard; defaults to input manifest.")
    parser.add_argument("--dtype", default=None, help="Output probability dtype; defaults to input manifest dtype.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output trace directories.")
    parser.add_argument("--verify-checksum", action="store_true", help="Verify source shard checksum while converting.")
    parser.add_argument("--temp-dir", help="Directory for temporary memmap files.")
    parser.add_argument(
        "--store-emit-position",
        action="store_true",
        help="Store emit_position arrays in output shards. Defaults to compact position-major shards without them.",
    )
    return parser


def _resolve_jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    trace_dirs = [Path(item) for item in args.trace_dir]
    output_dirs = [Path(item) for item in args.output_dir]
    if trace_dirs and output_dirs and len(trace_dirs) != len(output_dirs):
        raise SystemExit("--output-dir count must match --trace-dir count")
    if trace_dirs:
        if output_dirs:
            jobs.extend(zip(trace_dirs, output_dirs))
        elif args.output_root:
            root = Path(args.output_root)
            jobs.extend((trace_dir, root / trace_dir.name) for trace_dir in trace_dirs)
        else:
            raise SystemExit("provide --output-dir or --output-root for --trace-dir")
    if args.trace_root:
        if not args.output_root:
            raise SystemExit("provide --output-root with --trace-root")
        input_root = Path(args.trace_root)
        names = args.species or sorted(path.name for path in input_root.iterdir() if (path / "manifest.json").exists())
        output_root = Path(args.output_root)
        jobs.extend((input_root / name, output_root / name) for name in names)
    if not jobs:
        raise SystemExit("provide --trace-dir or --trace-root")
    return jobs


def main() -> None:
    args = _build_parser().parse_args()
    jobs = _resolve_jobs(args)
    rows = []
    started_all = perf_counter()
    for index, (source, output) in enumerate(jobs, 1):
        started = perf_counter()
        print(f"[{index}/{len(jobs)}] convert {source} -> {output}", file=sys.stderr, flush=True)
        manifest = convert_probability_trace_to_position_major(
            source,
            output,
            shard_rows=args.shard_rows,
            dtype=args.dtype,
            overwrite=bool(args.overwrite),
            verify_checksum=bool(args.verify_checksum),
            temp_dir=args.temp_dir,
            store_emit_position=bool(args.store_emit_position),
        )
        elapsed = perf_counter() - started
        rows.append(
            {
                "source_trace_dir": str(source),
                "output_trace_dir": str(output),
                "model_family": manifest.model_family,
                "model_id": manifest.model_id,
                "row_count": int(manifest.row_count),
                "window_bases": int(manifest.window_bases),
                "token_merge_size": int(manifest.token_merge_size),
                "emission_order": manifest.emission_order,
                "store_emit_position": bool(args.store_emit_position),
                "elapsed_seconds": float(elapsed),
                "rows_per_second": float(manifest.row_count / elapsed) if elapsed > 0 else None,
            }
        )
    summary = {
        "schema_version": 1,
        "conversion": "target_probability_trace_to_position_major_v1",
        "trace_count": len(rows),
        "elapsed_seconds": float(perf_counter() - started_all),
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
