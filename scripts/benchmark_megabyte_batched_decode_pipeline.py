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
from dna_compress.compression_eval import resolve_device
from dna_compress.config import load_experiment_config
from dna_compress.fast_arithmetic import BatchedStreamingArithmeticDecoder
from dna_compress.megabyte_batched_decode import MegabyteBatchedDecodeStepper, fast_floor_frequency_rows
from dna_compress.megabyte_loader import build_model, load_megabyte_checkpoint


DEFAULT_RUN_DIR = Path("outputs/dna_megabyte_large_opengenome2_4")


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synthetic_streams(batch_size: int) -> list[bytes]:
    streams: list[bytes] = []
    for index in range(batch_size):
        value = (0x9E3779B97F4A7C15 * (index + 1)) & ((1 << 64) - 1)
        streams.append(value.to_bytes(8, byteorder="little", signed=False))
    return streams


def _memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "memory_reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
    }


def _result_row(
    *,
    mode: str,
    batch_size: int,
    token_count: int,
    token_merge_size: int,
    wall_seconds: float,
    device: torch.device,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    logical_tokens = batch_size * token_count
    row: dict[str, object] = {
        "type": "benchmark",
        "mode": mode,
        "batch_size": batch_size,
        "token_steps": token_count,
        "logical_tokens": logical_tokens,
        "pipeline_wall_seconds": wall_seconds,
        "wall_seconds": wall_seconds,
        "step_ms": wall_seconds / max(token_count, 1) * 1000.0,
        "tokens_per_second": logical_tokens / max(wall_seconds, 1e-12),
        "bases_per_second": logical_tokens * token_merge_size / max(wall_seconds, 1e-12),
        **_memory_stats(device),
    }
    if extra:
        row.update(extra)
    return row


def _run_model_only(
    *,
    model: torch.nn.Module,
    batch_size: int,
    token_count: int,
    warmup_tokens: int,
    device: torch.device,
    dtype_name: str,
    token_merge_size: int,
) -> dict[str, object]:
    stepper = MegabyteBatchedDecodeStepper(model, batch_size=batch_size, device=device, dtype_name=dtype_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if warmup_tokens > 0:
        stepper.run_random_tokens(token_count=warmup_tokens)
    _sync(device)
    stepper = MegabyteBatchedDecodeStepper(model, batch_size=batch_size, device=device, dtype_name=dtype_name)
    started = perf_counter()
    checksum = stepper.run_random_tokens(token_count=token_count)
    _sync(device)
    wall_seconds = perf_counter() - started
    return _result_row(
        mode="model_only",
        batch_size=batch_size,
        token_count=token_count,
        token_merge_size=token_merge_size,
        wall_seconds=wall_seconds,
        device=device,
        extra={"checksum": checksum, **stepper.timings.as_dict()},
    )


def _random_frequency_rows(batch_size: int, vocab_size: int, *, frequency_total: int) -> torch.Tensor:
    # Keep totals in the same rough range as fast-floor rows without paying softmax cost.
    high = max(3, min(2048, frequency_total // 8))
    freqs = torch.randint(1, high, (batch_size, vocab_size), dtype=torch.int32)
    if frequency_total <= 65535:
        return freqs.to(torch.uint16)
    return freqs


def _run_arith_only(
    *,
    batch_size: int,
    vocab_size: int,
    token_count: int,
    token_merge_size: int,
    threads: int,
    frequency_total: int,
    device: torch.device,
) -> dict[str, object]:
    decoder = BatchedStreamingArithmeticDecoder(_synthetic_streams(batch_size), threads=threads)
    freqs = _random_frequency_rows(batch_size, vocab_size, frequency_total=frequency_total)
    decoder.decode_frequency_rows(freqs)
    started = perf_counter()
    checksum = 0
    for _ in range(token_count):
        decoded = decoder.decode_frequency_rows(freqs)
        checksum += int(decoded[0].item())
    wall_seconds = perf_counter() - started
    return _result_row(
        mode="arith_only",
        batch_size=batch_size,
        token_count=token_count,
        token_merge_size=token_merge_size,
        wall_seconds=wall_seconds,
        device=device,
        extra={
            "arith_decode_seconds": wall_seconds,
            "cpu_threads": threads,
            "frequency_dtype": str(freqs.dtype).replace("torch.", ""),
            "checksum": checksum,
        },
    )


def _run_transfer_only(
    *,
    batch_size: int,
    vocab_size: int,
    token_count: int,
    token_merge_size: int,
    frequency_total: int,
    device: torch.device,
) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    logits = torch.randn(batch_size, vocab_size, device=device)
    token_cpu = torch.empty(batch_size, dtype=torch.int64)
    total_quant_seconds = 0.0
    total_freq_transfer_seconds = 0.0
    total_token_transfer_seconds = 0.0
    _sync(device)
    started = perf_counter()
    for _ in range(token_count):
        quant_started = perf_counter()
        freqs_gpu = fast_floor_frequency_rows(logits, total=frequency_total)
        _sync(device)
        total_quant_seconds += perf_counter() - quant_started

        transfer_started = perf_counter()
        freqs_cpu = freqs_gpu.cpu()
        _sync(device)
        total_freq_transfer_seconds += perf_counter() - transfer_started

        token_started = perf_counter()
        _ = token_cpu.to(device, non_blocking=True)
        _sync(device)
        total_token_transfer_seconds += perf_counter() - token_started
        logits = logits + (float(freqs_cpu[0, 0].item()) * 0.0)
    wall_seconds = perf_counter() - started
    return _result_row(
        mode="transfer_only",
        batch_size=batch_size,
        token_count=token_count,
        token_merge_size=token_merge_size,
        wall_seconds=wall_seconds,
        device=device,
        extra={
            "quantize_seconds": total_quant_seconds,
            "freq_transfer_seconds": total_freq_transfer_seconds,
            "token_transfer_seconds": total_token_transfer_seconds,
            "frequency_dtype": "uint16" if frequency_total <= 65535 else "int32",
        },
    )


def _run_pipeline(
    *,
    model: torch.nn.Module,
    batch_size: int,
    token_count: int,
    warmup_tokens: int,
    token_merge_size: int,
    threads: int,
    frequency_total: int,
    device: torch.device,
    dtype_name: str,
) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run_steps(steps: int) -> tuple[float, dict[str, float], int]:
        decoder = BatchedStreamingArithmeticDecoder(_synthetic_streams(batch_size), threads=threads)
        stepper = MegabyteBatchedDecodeStepper(model, batch_size=batch_size, device=device, dtype_name=dtype_name)
        total_model_seconds = 0.0
        total_quant_seconds = 0.0
        total_freq_transfer_seconds = 0.0
        total_arith_seconds = 0.0
        total_token_transfer_seconds = 0.0
        checksum = 0
        _sync(device)
        started = perf_counter()
        for _ in range(steps):
            model_started = perf_counter()
            logits = stepper.next_logits()
            _sync(device)
            total_model_seconds += perf_counter() - model_started

            quant_started = perf_counter()
            freqs_gpu, totals_gpu = fast_floor_frequency_rows(logits, total=frequency_total, return_totals=True)
            _sync(device)
            total_quant_seconds += perf_counter() - quant_started

            transfer_started = perf_counter()
            freqs_cpu = freqs_gpu.cpu()
            totals_cpu = totals_gpu.cpu()
            _sync(device)
            total_freq_transfer_seconds += perf_counter() - transfer_started

            arith_started = perf_counter()
            symbols_cpu = decoder.decode_frequency_rows_with_totals(freqs_cpu, totals_cpu)
            total_arith_seconds += perf_counter() - arith_started
            checksum += int(symbols_cpu[0].item())

            token_started = perf_counter()
            symbols_gpu = symbols_cpu.to(device, non_blocking=True)
            _sync(device)
            total_token_transfer_seconds += perf_counter() - token_started
            stepper.accept_symbols(symbols_gpu)
        wall = perf_counter() - started
        return wall, {
            "model_seconds": total_model_seconds,
            "quantize_seconds": total_quant_seconds,
            "freq_transfer_seconds": total_freq_transfer_seconds,
            "arith_decode_seconds": total_arith_seconds,
            "token_transfer_seconds": total_token_transfer_seconds,
            **stepper.timings.as_dict(),
        }, checksum

    if warmup_tokens > 0:
        run_steps(warmup_tokens)
    wall_seconds, timings, checksum = run_steps(token_count)
    return _result_row(
        mode="pipeline",
        batch_size=batch_size,
        token_count=token_count,
        token_merge_size=token_merge_size,
        wall_seconds=wall_seconds,
        device=device,
        extra={
            **timings,
            "cpu_threads": threads,
            "frequency_dtype": "uint16" if frequency_total <= 65535 else "int32",
            "checksum": checksum,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark batched MEGABYTE decode-side pipeline with synthetic arithmetic streams.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4096, 6144, 8192, 10240, 12288, 16384])
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--frequency-total", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["model_only", "arith_only", "transfer_only", "pipeline"],
        default="pipeline",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    config, model, checkpoint_metadata = _load_model(run_dir, args.checkpoint_tag, device)
    dtype_name = args.dtype or config.train.dtype
    token_merge_size = int(config.data.token_merge_size)
    arithmetic_metadata = resolve_arithmetic_coding_metadata(
        vocab_size=int(config.model.vocab_size),
        requested_total=args.frequency_total or config.arithmetic.frequency_total,
        target_uniform_mass=float(config.arithmetic.target_uniform_mass),
    )
    frequency_total = int(arithmetic_metadata["arithmetic_frequency_total"])
    header = {
        "type": "header",
        "run_dir": str(run_dir),
        "checkpoint_tag": args.checkpoint_tag,
        "checkpoint_step": checkpoint_metadata.get("step"),
        "device": str(device),
        "dtype": dtype_name,
        "mode": args.mode,
        "token_merge_size": token_merge_size,
        "seq_length": int(config.model.seq_length),
        "patch_size": int(config.model.patch_size),
        "vocab_size": int(config.model.vocab_size),
        "frequency_total": frequency_total,
        "cpu_threads": int(args.threads),
        **arithmetic_metadata,
    }
    print(json.dumps(header, ensure_ascii=False, sort_keys=True), flush=True)

    best: dict[str, object] | None = None
    for batch_size in args.batch_sizes:
        try:
            if args.mode == "model_only":
                row = _run_model_only(
                    model=model,
                    batch_size=batch_size,
                    token_count=int(args.tokens),
                    warmup_tokens=int(args.warmup_tokens),
                    device=device,
                    dtype_name=dtype_name,
                    token_merge_size=token_merge_size,
                )
            elif args.mode == "arith_only":
                row = _run_arith_only(
                    batch_size=batch_size,
                    vocab_size=int(config.model.vocab_size),
                    token_count=int(args.tokens),
                    token_merge_size=token_merge_size,
                    threads=int(args.threads),
                    frequency_total=frequency_total,
                    device=device,
                )
            elif args.mode == "transfer_only":
                row = _run_transfer_only(
                    batch_size=batch_size,
                    vocab_size=int(config.model.vocab_size),
                    token_count=int(args.tokens),
                    token_merge_size=token_merge_size,
                    frequency_total=frequency_total,
                    device=device,
                )
            else:
                row = _run_pipeline(
                    model=model,
                    batch_size=batch_size,
                    token_count=int(args.tokens),
                    warmup_tokens=int(args.warmup_tokens),
                    token_merge_size=token_merge_size,
                    threads=int(args.threads),
                    frequency_total=frequency_total,
                    device=device,
                    dtype_name=dtype_name,
                )
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            if best is None or float(row["tokens_per_second"]) > float(best["tokens_per_second"]):
                best = row
        except torch.cuda.OutOfMemoryError as error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {"type": "oom", "batch_size": batch_size, "message": str(error).splitlines()[0]},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            break
    if best is not None:
        print(json.dumps({"type": "best", **best}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
