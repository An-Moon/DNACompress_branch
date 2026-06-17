#!/usr/bin/env python3
"""Example:
.venv/bin/python scripts/run_megabyte_window_roundtrip.py \
  --run-dir outputs/dna_megabyte_large_opengenome2_4 \
  --random-windows 8192 \
  --compression-batch-sizes 4096 8192 \
  --output-prefix outputs/window_codec_smoke
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
    decode_framed_token_windows,
    frame_compressed_streams,
    generate_random_windows,
    load_codec_model,
    load_token_windows,
    payload_sha256,
    resolve_frequency_total,
    save_token_windows,
)


DEFAULT_RUN_DIR = Path("outputs/dna_megabyte_large_opengenome2_4")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress and decode MEGABYTE token windows in one roundtrip run.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--input-tokens-npy", default=None)
    parser.add_argument("--random-windows", type=int, default=128)
    parser.add_argument("--tokens-per-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compression-batch-sizes", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--decode-batch-sizes", type=int, nargs="+", default=None, help="Defaults to the compression batch sizes.")
    parser.add_argument("--compression-mode", choices=["cached", "full_forward"], default="cached")
    parser.add_argument("--allow-mismatched-batch", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--frequency-total", type=int, default=None)
    parser.add_argument("--output-prefix", default=None, help="Optional prefix for best roundtrip payload/metadata/decoded tokens.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
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
    total_bases = int(tokens_cpu.numel()) * int(config.data.token_merge_size)
    header = {
        "type": "header",
        "run_dir": str(run_dir),
        "checkpoint_tag": args.checkpoint_tag,
        "checkpoint_step": checkpoint_metadata.get("step"),
        "device": str(device),
        "dtype": dtype_name,
        "window_count": int(tokens_cpu.shape[0]),
        "tokens_per_window": int(tokens_cpu.shape[1]),
        "token_count": int(tokens_cpu.numel()),
        "base_count": total_bases,
        "frequency_total": int(frequency_total),
        **token_metadata,
        **arithmetic_metadata,
    }
    print(json.dumps(header, ensure_ascii=False, sort_keys=True), flush=True)

    best_roundtrip: dict[str, object] | None = None
    best_payload: bytes | None = None
    best_metadata: dict[str, object] | None = None
    best_decoded = None

    decode_batch_sizes = args.decode_batch_sizes or args.compression_batch_sizes
    for compression_batch_size in args.compression_batch_sizes:
        streams, compression_metrics = compress_token_windows(
            model=model,
            tokens_cpu=tokens_cpu,
            batch_size=int(compression_batch_size),
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
            extra={**token_metadata, "payload_sha256": payload_sha256(framed)},
        )
        compression_row = {"type": "compression", **metadata}
        print(json.dumps(compression_row, ensure_ascii=False, sort_keys=True), flush=True)

        for decode_batch_size in decode_batch_sizes:
            if int(decode_batch_size) != int(compression_batch_size) and not args.allow_mismatched_batch:
                continue
            decoded, decode_metrics = decode_framed_token_windows(
                model=model,
                framed_payload=framed,
                window_count=int(tokens_cpu.shape[0]),
                tokens_per_window=int(tokens_cpu.shape[1]),
                batch_size=int(decode_batch_size),
                device=device,
                dtype_name=dtype_name,
                frequency_total=frequency_total,
                threads=int(args.threads),
                expected_tokens_cpu=tokens_cpu,
            )
            roundtrip_ok = int(decode_metrics["decode_mismatches"]) == 0
            row = {
                "type": "decode",
                "source_compression_batch_size": int(compression_batch_size),
                "source_compression_mode": args.compression_mode,
                "roundtrip_ok": roundtrip_ok,
                "decode_bases_per_second": float(decode_metrics["decode_tokens_per_second"]) * int(config.data.token_merge_size),
                **decode_metrics,
            }
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            if roundtrip_ok and (
                best_roundtrip is None
                or float(row["decode_tokens_per_second"]) > float(best_roundtrip["decode_tokens_per_second"])
            ):
                best_roundtrip = row
                best_payload = framed
                best_metadata = metadata
                best_decoded = decoded

    if best_roundtrip is None:
        raise RuntimeError("no roundtrip decode completed without mismatches")
    print(json.dumps({"type": "best_roundtrip", **best_roundtrip}, ensure_ascii=False, sort_keys=True), flush=True)

    if args.output_prefix:
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        payload_path = prefix.with_suffix(".mbw")
        metadata_path = prefix.with_suffix(".mbw.json")
        decoded_path = prefix.with_suffix(".decoded_tokens.npy")
        payload_path.write_bytes(best_payload or b"")
        metadata_path.write_text(json.dumps(best_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        save_token_windows(
            best_decoded,
            decoded_path,
            original_token_count=best_metadata.get("original_token_count") if best_metadata else None,
        )
        print(
            json.dumps(
                {
                    "type": "saved_best_roundtrip",
                    "payload": str(payload_path),
                    "metadata": str(metadata_path),
                    "decoded_tokens": str(decoded_path),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
