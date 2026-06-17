#!/usr/bin/env python3
"""Example:
.venv/bin/python scripts/run_megabyte_window_compress.py \
  --run-dir outputs/dna_megabyte_large_opengenome2_4 \
  --input-tokens-npy /path/to/token_windows.npy \
  --compression-batch-size 8192 \
  --output outputs/example.megabyte_windows.mbw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import resolve_device
from dna_compress.megabyte_window_codec import (
    build_codec_metadata,
    compress_token_windows,
    frame_compressed_streams,
    generate_random_windows,
    load_codec_model,
    load_token_windows,
    payload_sha256,
    resolve_frequency_total,
)


DEFAULT_RUN_DIR = Path("outputs/dna_megabyte_large_opengenome2_4")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress independent MEGABYTE token windows into a framed arithmetic bitstream."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--input-tokens-npy", default=None, help="1D token stream or 2D [windows, tokens] array.")
    parser.add_argument("--random-windows", type=int, default=0, help="Generate this many random windows instead.")
    parser.add_argument("--tokens-per-window", type=int, default=None, help="Defaults to model seq_length.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compression-batch-size", type=int, default=8192)
    parser.add_argument("--compression-mode", choices=["cached", "full_forward"], default="cached")
    parser.add_argument("--frequency-total", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output .mbw payload path.")
    parser.add_argument("--metadata-output", default=None, help="Defaults to <output>.json.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if bool(args.input_tokens_npy) == bool(args.random_windows):
        raise SystemExit("Specify exactly one of --input-tokens-npy or --random-windows.")

    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    config, model, checkpoint_metadata = load_codec_model(run_dir, args.checkpoint_tag, device)
    dtype_name = args.dtype or config.train.dtype
    tokens_per_window = int(args.tokens_per_window or config.model.seq_length)
    if tokens_per_window > int(config.model.seq_length):
        raise ValueError("--tokens-per-window cannot exceed model seq_length")

    if args.input_tokens_npy:
        tokens_cpu, token_metadata = load_token_windows(
            args.input_tokens_npy,
            tokens_per_window=tokens_per_window,
            pad_id=int(config.model.pad_id),
        )
    else:
        tokens_cpu = generate_random_windows(
            window_count=int(args.random_windows),
            seq_length=tokens_per_window,
            vocab_high=int(config.model.pad_id),
            seed=int(args.seed),
        )
        token_metadata = {
            "token_input_path": None,
            "token_input_ndim": 2,
            "original_token_count": int(tokens_cpu.numel()),
            "tail_padding_tokens": 0,
            "random_seed": int(args.seed),
        }

    frequency_total, arithmetic_metadata = resolve_frequency_total(config, args.frequency_total)
    streams, compression_metrics = compress_token_windows(
        model=model,
        tokens_cpu=tokens_cpu,
        batch_size=int(args.compression_batch_size),
        device=device,
        dtype_name=dtype_name,
        frequency_total=frequency_total,
        compression_mode=args.compression_mode,
    )
    framed, framing_metrics = frame_compressed_streams(streams)
    metadata = build_codec_metadata(
        run_dir=run_dir,
        checkpoint_tag=args.checkpoint_tag,
        checkpoint_metadata=checkpoint_metadata,
        dtype_name=dtype_name,
        device=device,
        tokens_cpu=tokens_cpu,
        token_merge_size=int(config.data.token_merge_size),
        frequency_total=frequency_total,
        arithmetic_metadata=arithmetic_metadata,
        compression_metrics=compression_metrics,
        framing_metrics=framing_metrics,
        extra={
            **token_metadata,
            "payload_sha256": payload_sha256(framed),
        },
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(framed)
    metadata_path = Path(args.metadata_output) if args.metadata_output else output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"type": "compression_complete", "output": str(output_path), "metadata": str(metadata_path), **metadata}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
