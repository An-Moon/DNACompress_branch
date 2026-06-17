#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression import resolve_arithmetic_coding_metadata
from dna_compress.compression_eval import NON_OVERLAP_MODE, autocast_context, compress_source, resolve_device
from dna_compress.config import load_experiment_config
from dna_compress.megabyte_loader import build_model, load_megabyte_checkpoint
from dna_compress.megabyte_serial_decode import MegabyteSerialDecoder, encode_symbols_with_serial_model
from dna_compress.tokenization import normalize_alphabet, tokenize_source_bytes


DEFAULT_RUN_DIR = Path("outputs/dna_megabyte_large_opengenome2_4")
DEFAULT_FASTA = Path("/data/students/Liang_junnan/opengenome2_subset/fasta_test_subset_100mb_per_source/gtdb_v220.fasta")


def _load_fasta_sequence(path: Path, *, alphabet: str) -> bytes:
    allowed = set(normalize_alphabet(alphabet).encode("ascii"))
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(b">"):
                continue
            chunks.append(bytes(byte for byte in line.upper() if byte in allowed))
    return b"".join(chunks)


def _checkpoint_path(run_dir: Path, checkpoint_tag: str) -> Path:
    path = run_dir / f"{checkpoint_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _load_model(run_dir: Path, checkpoint_tag: str, device: torch.device):
    config = load_experiment_config(run_dir / "resolved_config.json")
    model = build_model(config.model).to(device)
    model_state, checkpoint_metadata, _ = load_megabyte_checkpoint(_checkpoint_path(run_dir, checkpoint_tag), map_location=device)
    model.load_state_dict(model_state)
    model.eval()
    return config, model, checkpoint_metadata


def _symbols_from_source(source: bytes, *, token_count: int, merge_size: int, alphabet: str, eos_id: int) -> list[int]:
    requested_bases = token_count * merge_size
    symbols = tokenize_source_bytes(source[:requested_bases], merge_size, alphabet)
    if len(symbols) < token_count:
        raise ValueError(f"FASTA source yielded only {len(symbols)} merged tokens, requested {token_count}")
    return symbols[:token_count] + [int(eos_id)]


def _run_no_cache_decode_sim(
    *,
    model: torch.nn.Module,
    symbols: list[int],
    seq_length: int,
    pad_id: int,
    device: torch.device,
    dtype_name: str,
) -> dict[str, float]:
    started = perf_counter()
    with torch.no_grad(), autocast_context(device, dtype_name):
        for index in range(len(symbols)):
            window_start = (index // seq_length) * seq_length
            offset = index - window_start
            window_symbols = symbols[window_start : window_start + seq_length]
            if len(window_symbols) < seq_length:
                window_symbols = window_symbols + [pad_id] * (seq_length - len(window_symbols))
            ids = torch.tensor([window_symbols], dtype=torch.long, device=device)
            _ = model(ids).lm_logits[0, offset]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    wall_seconds = perf_counter() - started
    return {
        "wall_seconds": wall_seconds,
        "tokens_per_second": len(symbols) / max(wall_seconds, 1e-12),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark cached serial MEGABYTE arithmetic decode.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--input-fasta", default=str(DEFAULT_FASTA))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--token-counts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--compression-batch-size", type=int, default=None)
    parser.add_argument("--arithmetic-frequency-total", type=int, default=None)
    parser.add_argument("--arithmetic-target-uniform-mass", type=float, default=None)
    parser.add_argument("--skip-batch-compression", action="store_true")
    parser.add_argument("--include-no-cache", action="store_true")
    parser.add_argument("--roundtrip", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    config, model, checkpoint_metadata = _load_model(run_dir, args.checkpoint_tag, device)
    dtype_name = args.dtype or config.train.dtype
    merge_size = int(config.data.token_merge_size)
    alphabet = config.data.token_merge_alphabet
    batch_size = int(args.compression_batch_size or config.train.eval_batch_size)
    target_uniform_mass = (
        float(args.arithmetic_target_uniform_mass)
        if args.arithmetic_target_uniform_mass is not None
        else float(config.arithmetic.target_uniform_mass)
    )
    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=int(config.model.vocab_size),
        requested_total=args.arithmetic_frequency_total or config.arithmetic.frequency_total,
        target_uniform_mass=target_uniform_mass,
    )
    total = int(arithmetic_metadata["arithmetic_frequency_total"])
    source = _load_fasta_sequence(Path(args.input_fasta), alphabet=alphabet)

    header = {
        "run_dir": str(run_dir),
        "checkpoint_tag": args.checkpoint_tag,
        "checkpoint_step": checkpoint_metadata.get("step"),
        "input_fasta": str(args.input_fasta),
        "device": str(device),
        "dtype": dtype_name,
        "token_merge_size": merge_size,
        "arithmetic_quantization_mode": "fast_floor_gpu" if device.type == "cuda" else "fast_floor_cpu",
        **arithmetic_metadata,
    }
    print(json.dumps({"type": "header", **header}, ensure_ascii=False), flush=True)

    for token_count in args.token_counts:
        symbols = _symbols_from_source(
            source,
            token_count=token_count,
            merge_size=merge_size,
            alphabet=alphabet,
            eos_id=int(config.model.eos_id),
        )
        sample_bytes = token_count * merge_size
        row: dict[str, object] = {
            "type": "benchmark",
            "requested_tokens_without_eos": token_count,
            "encoded_tokens_with_eos": len(symbols),
            "sample_bases": sample_bytes,
        }

        if not args.skip_batch_compression:
            compression = compress_source(
                model=model,
                source=source,
                seq_length=int(config.model.seq_length),
                pad_id=int(config.model.pad_id),
                eos_id=int(config.model.eos_id),
                device=device,
                dtype_name=dtype_name,
                batch_size=batch_size,
                requested_bytes=sample_bytes,
                mode=NON_OVERLAP_MODE,
                token_merge_size=merge_size,
                token_merge_alphabet=alphabet,
                arithmetic_frequency_total=total,
                arithmetic_target_uniform_mass=target_uniform_mass,
                arithmetic_coding_mode="model_symbol",
                arithmetic_quantization_mode="fast_floor_gpu" if device.type == "cuda" else "fast_floor_cpu",
                arithmetic_merge_size=merge_size,
                arithmetic_backend="fast_cpp",
                include_codec_baselines=False,
            )
            row["batch_compression"] = {
                key: compression.get(key)
                for key in (
                    "bits_per_base",
                    "compression_process_seconds",
                    "compression_bases_per_second",
                    "model_forward_seconds",
                    "arithmetic_quantize_seconds",
                    "arithmetic_interval_transfer_seconds",
                    "arithmetic_range_seconds",
                    "python_overhead_seconds",
                )
            }

        encoded, encode_timings = encode_symbols_with_serial_model(
            model,
            symbols,
            device=device,
            dtype_name=dtype_name,
            arithmetic_frequency_total=total,
        )
        decoder = MegabyteSerialDecoder(
            model,
            device=device,
            dtype_name=dtype_name,
            arithmetic_frequency_total=total,
        )
        decoded, decode_timings = decoder.decode(encoded, token_count=len(symbols))
        if args.roundtrip and decoded != symbols:
            raise RuntimeError("cached serial decode roundtrip mismatch")
        row["cached_serial_encode"] = encode_timings
        row["cached_serial_decode"] = decode_timings
        row["encoded_bytes"] = len(encoded)
        row["roundtrip_ok"] = decoded == symbols

        if args.include_no_cache:
            row["no_cache_decode_sim"] = _run_no_cache_decode_sim(
                model=model,
                symbols=symbols,
                seq_length=int(config.model.seq_length),
                pad_id=int(config.model.pad_id),
                device=device,
                dtype_name=dtype_name,
            )

        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
