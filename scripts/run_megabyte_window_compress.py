#!/usr/bin/env python3
"""Example:
.venv/bin/python scripts/run_megabyte_window_compress.py \
  --run-dir outputs/dna_megabyte_large_opengenome2_4 \
  --input-tokens-npy /path/to/token_windows.npy \
  --window-codec-devices cuda:0 cuda:1 \
  --window-codec-compression-mode cached \
  --compression-batch-size 8192 \
  --output outputs/example.megabyte_windows.mbw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import resolve_device
from dna_compress.megabyte_window_codec import (
    TokenWindowBatch,
    WindowCodecPipeline,
    build_codec_metadata_from_counts,
    compress_chunk_stream_to_v2_payload,
    generate_random_windows,
    load_codec_config_and_metadata,
    load_token_windows,
    pack_v3_window_payload,
    payload_sha256,
    resolve_frequency_total,
    resolve_device_names,
    valid_lengths_from_logical_token_count,
)


DEFAULT_RUN_DIR = Path("outputs/dna_megabyte_large_opengenome2_4")


def _iter_file_chunks(path: Path, chunk_bytes: int = 8 << 20):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            yield chunk


def _iter_fasta_sequence_chunks(path: Path, alphabet: str):
    allowed = {ord(base) for base in alphabet.upper()}
    allowed.update(ord(base.lower()) for base in alphabet.upper())
    delete = bytes(byte for byte in range(256) if byte not in allowed)
    with path.open("rb") as handle:
        for line in handle:
            content = line.rstrip(b"\r\n")
            if content.startswith(b">"):
                continue
            cleaned = content.upper().translate(None, delete)
            if cleaned:
                yield cleaned


def _iter_token_window_batches(tokens_cpu: torch.Tensor, batch_size: int):
    valid_lengths = torch.full((int(tokens_cpu.shape[0]),), int(tokens_cpu.shape[1]), dtype=torch.long)
    for start in range(0, int(tokens_cpu.shape[0]), int(batch_size)):
        yield TokenWindowBatch(
            window_start=start,
            tokens=tokens_cpu[start : start + int(batch_size)].contiguous(),
            valid_lengths=valid_lengths[start : start + int(batch_size)].contiguous(),
        )


def _iter_token_window_batches_with_lengths(tokens_cpu: torch.Tensor, valid_lengths: torch.Tensor, batch_size: int):
    for start in range(0, int(tokens_cpu.shape[0]), int(batch_size)):
        yield TokenWindowBatch(
            window_start=start,
            tokens=tokens_cpu[start : start + int(batch_size)].contiguous(),
            valid_lengths=valid_lengths[start : start + int(batch_size)].contiguous(),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compress independent MEGABYTE token windows into a framed arithmetic bitstream."
    )
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-codec-devices", nargs="+", default=None)
    parser.add_argument("--window-codec-workers-per-device", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--input-tokens-npy", default=None, help="1D token stream or 2D [windows, tokens] array.")
    parser.add_argument("--input-bytes", default=None, help="Raw already-filtered sequence bytes to compress.")
    parser.add_argument("--input-fasta", default=None, help="FASTA input; headers and non-alphabet symbols are skipped.")
    parser.add_argument("--random-windows", type=int, default=0, help="Generate this many random windows instead.")
    parser.add_argument("--tokens-per-window", type=int, default=None, help="Defaults to model seq_length.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compression-batch-size", type=int, default=8192)
    parser.add_argument("--window-codec-compression-mode", "--compression-mode", choices=["cached", "full_forward"], default="cached")
    parser.add_argument("--frequency-total", type=int, default=None)
    parser.add_argument("--output", required=True, help="Output .mbw payload path.")
    parser.add_argument("--metadata-output", default=None, help="Defaults to <output>.json.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_count = sum(bool(value) for value in (args.input_tokens_npy, args.input_bytes, args.input_fasta)) + int(bool(args.random_windows))
    if input_count != 1:
        raise SystemExit("Specify exactly one of --input-tokens-npy, --input-bytes, --input-fasta, or --random-windows.")

    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    device_names = resolve_device_names(args.window_codec_devices, fallback=device)
    if int(args.window_codec_workers_per_device) != 1:
        raise ValueError("--window-codec-workers-per-device values other than 1 are not supported yet.")
    config, checkpoint_metadata = load_codec_config_and_metadata(run_dir, args.checkpoint_tag)
    dtype_name = args.dtype or config.train.dtype
    tokens_per_window = int(args.tokens_per_window or config.model.seq_length)
    if tokens_per_window > int(config.model.seq_length):
        raise ValueError("--tokens-per-window cannot exceed model seq_length")

    tokens_cpu = None
    token_metadata = None
    if args.input_tokens_npy:
        tokens_cpu, token_metadata = load_token_windows(
            args.input_tokens_npy,
            tokens_per_window=tokens_per_window,
            pad_id=int(config.model.pad_id),
        )
    elif args.random_windows:
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
    with WindowCodecPipeline(
        config=config,
        checkpoint_path=run_dir / f"{args.checkpoint_tag}.pt",
        devices=device_names,
        dtype_name=dtype_name,
        frequency_total=frequency_total,
        batch_size=int(args.compression_batch_size),
        compression_mode=args.window_codec_compression_mode,
    ) as pipeline:
        if args.input_bytes or args.input_fasta:
            source_path = Path(args.input_bytes or args.input_fasta)
            chunks = (
                _iter_fasta_sequence_chunks(source_path, config.data.token_merge_alphabet)
                if args.input_fasta
                else _iter_file_chunks(source_path)
            )
            framed, metadata = compress_chunk_stream_to_v2_payload(
                pipeline=pipeline,
                chunks=chunks,
                config=config,
                run_dir=run_dir,
                checkpoint_tag=args.checkpoint_tag,
                checkpoint_metadata=checkpoint_metadata,
                device=torch.device(device_names[0]),
                source_name=source_path.stem,
                requested_bytes=None,
                arithmetic_metadata=arithmetic_metadata,
                extra_metadata={"token_input_path": str(source_path)},
            )
        else:
            assert tokens_cpu is not None
            assert token_metadata is not None
            valid_lengths = valid_lengths_from_logical_token_count(
                window_count=int(tokens_cpu.shape[0]),
                tokens_per_window=int(tokens_cpu.shape[1]),
                logical_token_count=int(token_metadata["original_token_count"]),
            )
            batches = _iter_token_window_batches_with_lengths(tokens_cpu, valid_lengths, int(args.compression_batch_size))
            streams, compression_metrics = pipeline.compress_batches(batches)
            header = {
                "format_version": 3,
                "codec": "megabyte_window_fast_floor",
                "window_count": int(tokens_cpu.shape[0]),
                "tokens_per_window": int(tokens_cpu.shape[1]),
                "logical_token_count": int(token_metadata["original_token_count"]),
                "base_token_count": int(token_metadata["original_token_count"]),
                "token_merge_size": int(config.data.token_merge_size),
                "compression_batch_size": int(args.compression_batch_size),
                "frequency_total": int(frequency_total),
            }
            framed = pack_v3_window_payload(streams, header)
            framing_metrics = {
                "framing_seconds": 0.0,
                "framed_bytes": len(framed),
                "framing_bytes": len(framed) - sum(len(stream) for stream in streams),
            }
            metadata = build_codec_metadata_from_counts(
                run_dir=run_dir,
                checkpoint_tag=args.checkpoint_tag,
                checkpoint_metadata=checkpoint_metadata,
                dtype_name=dtype_name,
                device=torch.device(device_names[0]),
                window_count=int(tokens_cpu.shape[0]),
                tokens_per_window=int(tokens_cpu.shape[1]),
                logical_token_count=int(token_metadata["original_token_count"]),
                base_token_count=int(token_metadata["original_token_count"]),
                tail_padding_tokens=int(token_metadata.get("tail_padding_tokens", 0)),
                token_merge_size=int(config.data.token_merge_size),
                frequency_total=frequency_total,
                arithmetic_metadata=arithmetic_metadata,
                compression_metrics=compression_metrics,
                framing_metrics=framing_metrics,
                extra={**token_metadata, "payload_sha256": payload_sha256(framed)},
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
