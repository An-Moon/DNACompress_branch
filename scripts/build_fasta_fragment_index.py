#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.fasta_fragment_index import (  # noqa: E402
    DEFAULT_ANCHOR_STRIDE,
    DEFAULT_FASTA_ROOT,
    DEFAULT_INDEX_DIR,
    DEFAULT_IN_MEMORY_THRESHOLD,
    build_fasta_fragment_index,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a random-access FASTA fragment index.")
    parser.add_argument("--fasta-root", default=str(DEFAULT_FASTA_ROOT))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--anchor-stride", type=int, default=DEFAULT_ANCHOR_STRIDE)
    parser.add_argument("--chunk-size", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--batch-rows", type=int, default=50_000)
    parser.add_argument("--in-memory-threshold", type=int, default=DEFAULT_IN_MEMORY_THRESHOLD)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    started = time.time()
    print(
        json.dumps(
            {
                "event": "start",
                "fasta_root": args.fasta_root,
                "index_dir": args.index_dir,
                "anchor_stride": args.anchor_stride,
                "chunk_size": args.chunk_size,
                "in_memory_threshold": args.in_memory_threshold,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def progress(event: dict[str, object]) -> None:
        elapsed = max(time.time() - started, 1e-6)
        processed = int(event["processed_bytes"])
        rate = processed / elapsed
        print(
            json.dumps(
                {
                    "event": "progress",
                    **event,
                    "elapsed_seconds": elapsed,
                    "bytes_per_second": rate,
                    "gib_per_hour": rate * 3600 / 1024**3,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    stats = build_fasta_fragment_index(
        fasta_root=args.fasta_root,
        index_dir=args.index_dir,
        anchor_stride=args.anchor_stride,
        chunk_size=args.chunk_size,
        batch_rows=args.batch_rows,
        in_memory_threshold=args.in_memory_threshold,
        progress_callback=progress,
    )
    print(
        json.dumps(
            {
                "event": "done",
                "seconds": time.time() - started,
                "file_count": stats.file_count,
                "record_count": stats.record_count,
                "run_count": stats.run_count,
                "anchor_count": stats.anchor_count,
                "total_size_bytes": stats.total_size_bytes,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
