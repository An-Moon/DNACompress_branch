#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import resolve_device
from dna_compress.config import load_experiment_config
from dna_compress.megabyte_loader import build_model, load_megabyte_checkpoint
from dna_compress.megabyte_serial_decode import _autocast_context, _transformer_step


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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _batched_cached_decode_inference_pass(
    *,
    model: torch.nn.Module,
    tokens: torch.Tensor,
    token_count: int,
    device: torch.device,
    dtype_name: str,
) -> float:
    config = model.config
    batch_size = tokens.shape[0]
    patch_size = int(config.P)
    seq_length = int(config.T_MAX)
    pad_id = int(config.pad_id)
    global_caches = None
    local_caches = None
    current_patch_context = None
    checksum = torch.zeros((), dtype=torch.float32, device=device)

    with torch.inference_mode(), _autocast_context(device, dtype_name):
        for index in range(token_count):
            window_offset = index % seq_length
            local_index = index % patch_size
            if window_offset == 0:
                global_caches = None
                local_caches = None
                current_patch_context = None

            if local_index == 0 or current_patch_context is None:
                if window_offset == 0:
                    patch_ids = torch.full((batch_size, patch_size), pad_id, dtype=torch.long, device=device)
                else:
                    patch_ids = tokens[:, index - patch_size : index]
                patch_embed = model.to_embed(patch_ids)
                global_in = rearrange(patch_embed, "b p d -> b 1 (p d)")
                global_out, global_caches = _transformer_step(model.g_transformer, global_in, global_caches)
                current_patch_context = model.gl_linear(global_out).view(batch_size, patch_size, int(config.D_L))
                local_caches = None

            if window_offset == 0:
                previous_ids = torch.full((batch_size, 1), pad_id, dtype=torch.long, device=device)
            else:
                previous_ids = tokens[:, index - 1 : index]

            local_embed = model.to_l_embed(model.to_embed(previous_ids))
            local_in = current_patch_context[:, local_index : local_index + 1, :] + local_embed
            local_out, local_caches = _transformer_step(model.l_transformer, local_in, local_caches)
            logits = model.to_logits(local_out[:, -1, :])
            checksum = checksum + logits[:, 0].float().sum()

    # Keep the logits path live so compilers/runtimes cannot elide the projection.
    return float(checksum.detach().cpu())


def _benchmark_batch(
    *,
    model: torch.nn.Module,
    batch_size: int,
    token_count: int,
    warmup_tokens: int,
    device: torch.device,
    dtype_name: str,
) -> dict[str, object]:
    config = model.config
    vocab_high = max(1, int(config.pad_id))
    total_tokens = max(token_count, warmup_tokens)
    tokens = torch.randint(
        low=0,
        high=vocab_high,
        size=(batch_size, total_tokens),
        dtype=torch.long,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if warmup_tokens > 0:
        _batched_cached_decode_inference_pass(
            model=model,
            tokens=tokens,
            token_count=warmup_tokens,
            device=device,
            dtype_name=dtype_name,
        )
    _synchronize(device)
    started = perf_counter()
    checksum = _batched_cached_decode_inference_pass(
        model=model,
        tokens=tokens,
        token_count=token_count,
        device=device,
        dtype_name=dtype_name,
    )
    _synchronize(device)
    wall_seconds = perf_counter() - started
    logical_tokens = batch_size * token_count
    result: dict[str, object] = {
        "batch_size": batch_size,
        "token_steps": token_count,
        "logical_tokens": logical_tokens,
        "wall_seconds": wall_seconds,
        "tokens_per_second": logical_tokens / max(wall_seconds, 1e-12),
        "step_seconds": wall_seconds / max(token_count, 1),
        "checksum": checksum,
    }
    if device.type == "cuda":
        result["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
        result["memory_reserved_gb"] = torch.cuda.memory_reserved(device) / (1024**3)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark batched cached MEGABYTE decode-side inference with random tokens.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 64, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    parser.add_argument("--stop-on-oom", action="store_true", default=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    device = resolve_device(args.device)
    config, model, checkpoint_metadata = _load_model(run_dir, args.checkpoint_tag, device)
    dtype_name = args.dtype or config.train.dtype
    token_merge_size = int(config.data.token_merge_size)
    print(
        json.dumps(
            {
                "type": "header",
                "run_dir": str(run_dir),
                "checkpoint_tag": args.checkpoint_tag,
                "checkpoint_step": checkpoint_metadata.get("step"),
                "device": str(device),
                "dtype": dtype_name,
                "token_merge_size": token_merge_size,
                "seq_length": int(config.model.seq_length),
                "patch_size": int(config.model.patch_size),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )

    best: dict[str, object] | None = None
    for batch_size in args.batch_sizes:
        try:
            result = _benchmark_batch(
                model=model,
                batch_size=batch_size,
                token_count=int(args.tokens),
                warmup_tokens=int(args.warmup_tokens),
                device=device,
                dtype_name=dtype_name,
            )
            result["type"] = "benchmark"
            result["bases_per_second"] = float(result["tokens_per_second"]) * token_merge_size
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
            if best is None or float(result["tokens_per_second"]) > float(best["tokens_per_second"]):
                best = result
        except torch.cuda.OutOfMemoryError as error:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(
                json.dumps(
                    {
                        "type": "oom",
                        "batch_size": batch_size,
                        "message": str(error).splitlines()[0],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.stop_on_oom:
                break

    if best is not None:
        print(json.dumps({"type": "best", **best}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
