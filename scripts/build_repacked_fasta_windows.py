from __future__ import annotations

"""Build fixed-size Megabyte training windows from an indexed FASTA dataset.

Example:

    python scripts/build_repacked_fasta_windows.py \
        --index-dir /data/students/Liang_junnan/opengenome2_subset/index \
        --output-dir /data/students/Liang_junnan/opengenome2_subset/repacked_megabyte_s1024_m3_hashshard \
        --seq-length 1024 \
        --token-merge-size 3 \
        --token-merge-alphabet ACGTN \
        --pad-id 125 \
        --hash-shard-count 16 \
        --hash-shard-seed 0 \
        --writer-buffer-mb 256 \
        --read-unit-windows 8192 \
        --train-ratio 0.98 \
        --val-ratio 0.01 \
        --test-ratio 0.01 \
        --split-seed 0
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.repacked_windows import (  # noqa: E402
    DEFAULT_HASH_SHARD_COUNT,
    DEFAULT_HASH_SHARD_SEED,
    DEFAULT_REPACKED_READ_UNIT_WINDOWS,
    DEFAULT_REPACKED_WINDOW_DIR,
    DEFAULT_WRITER_BUFFER_MB,
    build_repacked_megabyte_windows,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repack indexed OpenGenome2 FASTA into fixed Megabyte windows.")
    parser.add_argument(
        "--index-dir",
        default="/data/students/Liang_junnan/opengenome2_subset/index",
        help="Existing FASTA fragment index directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_REPACKED_WINDOW_DIR),
        help="Output repacked window dataset directory.",
    )
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--token-merge-size", type=int, default=3)
    parser.add_argument("--token-merge-alphabet", default="ACGTN")
    parser.add_argument("--pad-id", type=int, default=125)
    parser.add_argument("--hash-shard-count", type=int, default=DEFAULT_HASH_SHARD_COUNT)
    parser.add_argument("--hash-shard-seed", type=int, default=DEFAULT_HASH_SHARD_SEED)
    parser.add_argument("--writer-buffer-mb", type=int, default=DEFAULT_WRITER_BUFFER_MB)
    parser.add_argument("--read-unit-windows", type=int, default=DEFAULT_REPACKED_READ_UNIT_WINDOWS)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output dir before building.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    started = time.time()

    def progress(event: dict[str, object]) -> None:
        elapsed = float(event.get("elapsed_seconds", time.time() - started))
        windows = int(event.get("window_count", 0))
        rate = windows / max(elapsed, 1e-6)
        print(
            "[progress] "
            f"runs={event.get('processed_runs')}/{event.get('total_runs')} "
            f"windows={windows} "
            f"padded={event.get('padded_window_count')} "
            f"elapsed={elapsed:.1f}s "
            f"windows_per_sec={rate:.1f}",
            flush=True,
        )

    manifest = build_repacked_megabyte_windows(
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
        token_merge_size=args.token_merge_size,
        token_merge_alphabet=args.token_merge_alphabet,
        pad_id=args.pad_id,
        hash_shard_count=args.hash_shard_count,
        hash_shard_seed=args.hash_shard_seed,
        writer_buffer_mb=args.writer_buffer_mb,
        read_unit_windows=args.read_unit_windows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        overwrite=args.overwrite,
        progress_callback=progress,
    )
    print(
        json.dumps(
            {k: manifest[k] for k in ["window_count", "padded_window_count", "default_schedule_dir", "elapsed_seconds"]},
            indent=2,
        )
    )
    print(f"manifest: {Path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()
