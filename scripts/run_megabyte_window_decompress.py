#!/usr/bin/env python3
"""Example:
.venv/bin/python scripts/run_megabyte_window_decompress.py \
  --input outputs/example.megabyte_windows.mbw \
  --output-tokens-npy outputs/example.decoded_windows.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from dna_compress.compression_eval import resolve_device
from dna_compress.megabyte_window_codec import (
    WINDOW_CODEC_NAME,
    decode_framed_token_windows,
    load_codec_model,
    payload_sha256,
    save_token_windows,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decode a framed MEGABYTE window payload back to token windows.")
    parser.add_argument("--input", required=True, help="Input .mbw payload path.")
    parser.add_argument("--metadata", default=None, help="Defaults to <input>.json.")
    parser.add_argument("--run-dir", default=None, help="Override run_dir stored in metadata.")
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default=None, help="Override checkpoint tag.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--decode-batch-size", type=int, default=None, help="Defaults to compression_batch_size in metadata.")
    parser.add_argument("--allow-mismatched-batch", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--output-tokens-npy", required=True, help="Output 2D [windows, tokens] token array.")
    parser.add_argument("--output-flat-tokens-npy", default=None, help="Optional trimmed 1D token stream.")
    parser.add_argument("--expected-tokens-npy", default=None, help="Optional expected tokens for mismatch reporting.")
    parser.add_argument("--metrics-output", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_path = Path(args.input)
    metadata_path = Path(args.metadata) if args.metadata else input_path.with_suffix(input_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("codec") != WINDOW_CODEC_NAME:
        raise ValueError(f"unsupported codec metadata: {metadata.get('codec')}")

    compression_batch_size = int(metadata["compression_batch_size"])
    decode_batch_size = int(args.decode_batch_size or compression_batch_size)
    if decode_batch_size != compression_batch_size and not args.allow_mismatched_batch:
        raise ValueError(
            "decode batch size must match compression_batch_size for deterministic arithmetic decode "
            f"({decode_batch_size} != {compression_batch_size}); use --allow-mismatched-batch only for experiments."
        )

    payload = input_path.read_bytes()
    expected_hash = metadata.get("payload_sha256")
    actual_hash = payload_sha256(payload)
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("payload sha256 does not match metadata")

    run_dir = Path(args.run_dir or metadata["run_dir"])
    checkpoint_tag = args.checkpoint_tag or metadata["checkpoint_tag"]
    device = resolve_device(args.device)
    config, model, _ = load_codec_model(run_dir, checkpoint_tag, device)
    dtype_name = args.dtype or metadata.get("dtype") or config.train.dtype

    expected_tokens = None
    if args.expected_tokens_npy:
        expected_tokens = torch.as_tensor(np.load(args.expected_tokens_npy), dtype=torch.long)

    decoded, metrics = decode_framed_token_windows(
        model=model,
        framed_payload=payload,
        window_count=int(metadata["window_count"]),
        tokens_per_window=int(metadata["tokens_per_window"]),
        batch_size=decode_batch_size,
        device=device,
        dtype_name=dtype_name,
        frequency_total=int(metadata["frequency_total"]),
        threads=int(args.threads),
        expected_tokens_cpu=expected_tokens,
    )
    save_token_windows(
        decoded,
        args.output_tokens_npy,
        original_token_count=metadata.get("original_token_count"),
        flat_output_path=args.output_flat_tokens_npy,
    )

    metrics = {
        "type": "decode_complete",
        "input": str(input_path),
        "metadata": str(metadata_path),
        "output_tokens_npy": str(args.output_tokens_npy),
        "output_flat_tokens_npy": args.output_flat_tokens_npy,
        "roundtrip_checked": expected_tokens is not None,
        "roundtrip_ok": expected_tokens is None or int(metrics["decode_mismatches"]) == 0,
        "decode_bases_per_second": float(metrics["decode_tokens_per_second"]) * int(metadata["token_merge_size"]),
        **metrics,
    }
    if args.metrics_output:
        Path(args.metrics_output).write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
