from __future__ import annotations

"""Fine-tune or evaluate official megaDNA on DNACorpus.

Upstream note:
    The official `lingxusb/megaDNA` repository does not include a training
    script or optimizer/batch/learning-rate defaults. It only ships inference
    notebooks plus the model class. The reusable defaults we can read from the
    official 145M checkpoint are architectural: vocab size 6, pad id 0,
    eos token id 5, 3 stages, `max_seq_len=(128, 64, 16)`, stage dimensions
    `(512, 256, 196)`, and depth `(8, 8, 8)`. The examples below therefore set
    experiment hyperparameters explicitly when they differ from this script's
    own defaults.

Examples:

    # Evaluate the official 145M checkpoint on a small HoSa slice.
    python scripts/run_megadna_experiment.py \
      --mode eval \
      --species HoSa \
      --seq-length 3072 \
      --eval-batch-size 2 \
      --max-train-bytes 4096 \
      --max-val-bytes 4096 \
      --max-test-bytes 4096 \
      --non-acgt-policy filter \
      --device cpu \
      --no-timestamp-output

    # Fine-tune from the official 145M checkpoint, with common non-default overrides.
    python scripts/run_megadna_experiment.py \
      --dataset-dir datasets/DNACorpus \
      --sequence-source-mode auto \
      --multi-sequence-mode separate \
      --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
      --checkpoint outputs/megadna_145m_all_20260522_154950/best.pt \
      --init-from resume \
      --seq-length 131072 \
      --print-config \
      --batch-size 2 \
      --eval-batch-size 2 \
      --learning-rate 1e-4 \
      --train-samples-per-epoch 600000 \
      --compression-sample-bytes 100000 \
      --train-ratio 0.6 \
      --val-ratio 0.2 \
      --test-ratio 0.2 \
      --lr-scheduler cosine \
      --lr-min-ratio 0.1 \
      --num-workers 4 \
      --prefetch-factor 4 \
      --persistent-workers \
      --pin-memory \
      --log-interval 25 \
      --eval-interval 2500 \
      --non-acgt-policy filter \
      --device cuda:1 \
      --run-name megadna_145m_all \
      --output-dir outputs/megadna_145m_all \
      --wandb-project dna-compress \
      --wandb-name megadna_145m_all_resume

    # Resume from this script's checkpoint.
    python scripts/run_megadna_experiment.py \
      --mode all \
      --checkpoint outputs/megadna_145m_all_20260522_154950/best.pt \
      --init-from resume \
      --device cuda:1 \
      --wandb-project dna-compress \
      --wandb-name megadna_145m_all_resume 

Optional generic overrides (repeatable):

    --override train.epochs=2 --override data.species='["GaGa","DrMe"]'
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.config import ExperimentConfig, load_experiment_config, save_experiment_config
from dna_compress.data import load_splits
from dna_compress.experiment import (
    autocast_context,
    build_lr_scheduler,
    init_wandb_run,
    log_wandb_metrics,
    open_training_log_file,
    resolve_device,
    seed_everything,
    write_training_log_event,
)
from dna_compress.megadna_data import (
    RandomMegaDNAWindowDataset,
    SequentialMegaDNAWindowDataset,
    encode_sources_for_megadna,
)
from dna_compress.megadna_loader import (
    MEGADNA_EOS_ID,
    MEGADNA_PAD_ID,
    default_megadna_weight_path,
    load_megadna_model,
    wrap_megadna_for_target_aligned_logits,
)


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if (value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _set_nested_attr(config: Any, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2:
        raise ValueError(f"Override key must include section and field: {dotted_key}")
    target = config
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config section/path: {'.'.join(parts[:-1])}")
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise ValueError(f"Unknown config field: {dotted_key}")
    setattr(target, parts[-1], value)


def _apply_if_not_none(config: Any, dotted_key: str, value: Any) -> None:
    if value is not None:
        _set_nested_attr(config, dotted_key, value)


def _parse_sequence_include(values: list[str] | None) -> dict[str, list[str]] | None:
    if values is None:
        return None
    parsed: dict[str, list[str]] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid sequence include '{item}'. Expected species=key1,key2,...")
        species, raw_keys = item.split("=", 1)
        species_name = species.strip()
        keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        if not species_name or not keys:
            raise ValueError(f"Invalid sequence include '{item}'. Expected species=key1,key2,...")
        parsed.setdefault(species_name, [])
        for key in keys:
            if key not in parsed[species_name]:
                parsed[species_name].append(key)
    return parsed


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species
    config.model.implementation = "megadna"
    config.model.vocab_size = len(("**", "A", "T", "C", "G", "#"))
    config.model.pad_id = MEGADNA_PAD_ID
    config.model.eos_id = MEGADNA_EOS_ID
    config.data.token_merge_size = 1
    config.data.token_merge_alphabet = "ATCG"

    _apply_if_not_none(config, "model.pretrained_weight_path", args.base_weight)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)

    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.sequence_source_mode", args.sequence_source_mode)
    _apply_if_not_none(config, "data.multi_sequence_mode", args.multi_sequence_mode)
    _apply_if_not_none(config, "data.clean_cache_enabled", args.clean_cache_enabled)
    _apply_if_not_none(config, "data.clean_cache_dir", args.clean_cache_dir)
    if args.sequence_include is not None:
        config.data.sequence_include_map = _parse_sequence_include(args.sequence_include) or {}
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.max_train_bytes_per_species", args.max_train_bytes)
    _apply_if_not_none(config, "data.max_val_bytes_per_species", args.max_val_bytes)
    _apply_if_not_none(config, "data.max_test_bytes_per_species", args.max_test_bytes)
    _apply_if_not_none(config, "data.train_samples_per_epoch", args.train_samples_per_epoch)
    _apply_if_not_none(config, "data.train_sampling_strategy", args.train_sampling_strategy)
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)

    _apply_if_not_none(config, "train.seed", args.seed)
    _apply_if_not_none(config, "train.device", args.device)
    _apply_if_not_none(config, "train.dtype", args.dtype)
    _apply_if_not_none(config, "train.init_from", args.init_from)
    _apply_if_not_none(config, "train.epochs", args.epochs)
    _apply_if_not_none(config, "train.batch_size", args.batch_size)
    _apply_if_not_none(config, "train.eval_batch_size", args.eval_batch_size)
    _apply_if_not_none(config, "train.learning_rate", args.learning_rate)
    _apply_if_not_none(config, "train.weight_decay", args.weight_decay)
    _apply_if_not_none(config, "train.lr_scheduler", args.lr_scheduler)
    _apply_if_not_none(config, "train.lr_warmup_steps", args.lr_warmup_steps)
    _apply_if_not_none(config, "train.lr_min_ratio", args.lr_min_ratio)
    _apply_if_not_none(config, "train.grad_clip_norm", args.grad_clip_norm)
    _apply_if_not_none(config, "train.num_workers", args.num_workers)
    _apply_if_not_none(config, "train.prefetch_factor", args.prefetch_factor)
    _apply_if_not_none(config, "train.persistent_workers", args.persistent_workers)
    _apply_if_not_none(config, "train.pin_memory", args.pin_memory)
    _apply_if_not_none(config, "train.log_interval", args.log_interval)
    _apply_if_not_none(config, "train.eval_interval", args.eval_interval)

    _apply_if_not_none(config, "output.run_name", args.run_name)
    _apply_if_not_none(config, "output.output_dir", args.output_dir)
    _apply_if_not_none(config, "output.tracking_backend", args.tracking_backend)
    _apply_if_not_none(config, "output.wandb_project", args.wandb_project)
    _apply_if_not_none(config, "output.wandb_entity", args.wandb_entity)
    _apply_if_not_none(config, "output.wandb_name", args.wandb_name)
    _apply_if_not_none(config, "output.wandb_group", args.wandb_group)
    _apply_if_not_none(config, "output.wandb_tags", args.wandb_tags)
    _apply_if_not_none(config, "output.wandb_mode", args.wandb_mode)
    if args.wandb_enabled is not None:
        config.output.wandb_enabled = args.wandb_enabled
    elif args.wandb_project is not None:
        config.output.wandb_enabled = True

    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected section.key=value")
        key, raw_value = item.split("=", 1)
        _set_nested_attr(config, key.strip(), _parse_scalar(raw_value.strip()))


def _default_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.implementation = "megadna"
    config.model.pretrained_weight_path = str(default_megadna_weight_path())
    config.model.seq_length = 1024
    config.model.vocab_size = 6
    config.model.pad_id = MEGADNA_PAD_ID
    config.model.eos_id = MEGADNA_EOS_ID
    config.data.token_merge_size = 1
    config.data.token_merge_alphabet = "ATCG"
    config.data.train_samples_per_epoch = 1024
    config.data.compression_sample_bytes = 16_384
    config.train.init_from = "pretrained"
    config.train.batch_size = 1
    config.train.eval_batch_size = 1
    config.train.learning_rate = 1e-5
    config.train.dtype = "bfloat16"
    config.output.run_name = "megadna_145m"
    config.output.output_dir = "outputs/megadna_145m"
    return config


def _apply_timestamp_to_output_dir(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if not args.timestamp_output:
        return
    if config.train.init_from == "resume" and args.checkpoint is not None:
        return
    config.output.output_dir = f"{config.output.output_dir}_{datetime.now().strftime(args.timestamp_format)}"


def _validate(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if config.model.seq_length <= 0:
        raise ValueError("model.seq_length must be > 0")
    if config.train.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("train.dtype must be one of: float32, float16, bfloat16")
    if config.train.init_from not in {"pretrained", "resume"}:
        raise ValueError("megaDNA init_from must be one of: pretrained, resume")
    if args.non_acgt_policy not in {"reject", "filter"}:
        raise ValueError("--non-acgt-policy must be reject or filter")
    if config.data.train_sampling_strategy not in {"proportional", "uniform", "sqrt"}:
        raise ValueError("data.train_sampling_strategy must be one of: proportional, uniform, sqrt")


def _checkpoint_path(args: argparse.Namespace, output_dir: Path, tag: str = "best") -> Path | None:
    if args.checkpoint:
        return Path(args.checkpoint)
    candidate = output_dir / f"{tag}.pt"
    return candidate if candidate.exists() else None


def _load_model_and_checkpoint(config: ExperimentConfig, args: argparse.Namespace, device: torch.device):
    base_weight = Path(config.model.pretrained_weight_path or default_megadna_weight_path())
    raw_model = load_megadna_model(base_weight, device=device)
    checkpoint_payload = None
    checkpoint_path = _checkpoint_path(args, Path(config.output.output_dir), tag="last" if config.train.init_from == "resume" else "best")
    if checkpoint_path is not None:
        payload = torch.load(checkpoint_path, map_location=device)
        if isinstance(payload, dict) and "model_state" in payload:
            raw_model.load_state_dict(payload["model_state"], strict=True)
            checkpoint_payload = payload
        elif isinstance(payload, torch.nn.Module):
            raw_model = payload.to(device)
        else:
            raise ValueError(f"Unsupported megaDNA checkpoint format: {checkpoint_path}")
    return raw_model, checkpoint_payload, checkpoint_path


def _loader_kwargs(config: ExperimentConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "num_workers": config.train.num_workers,
        "pin_memory": bool(config.train.pin_memory),
    }
    if config.train.num_workers > 0:
        kwargs["prefetch_factor"] = config.train.prefetch_factor
        kwargs["persistent_workers"] = bool(config.train.persistent_workers)
    return kwargs


def _evaluate(model: torch.nn.Module, dataloader: DataLoader, *, device: torch.device, dtype_name: str) -> dict[str, float]:
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    total_bases = 0
    started = time.time()
    with torch.no_grad():
        for batch in dataloader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            valid_tokens = int((ids != MEGADNA_PAD_ID).sum().item())
            with autocast_context(device, dtype_name):
                output = model(ids, return_loss=True)
            total_nats += float(output.loss.item()) * valid_tokens
            total_tokens += valid_tokens
            total_bases += valid_tokens  # megaDNA token_size=1, so 1 token == 1 base
    return {
        "loss_nats_per_token": total_nats / max(total_tokens, 1),
        "bits_per_base": (total_nats / max(total_tokens, 1)) / 0.6931471805599453,
        "tokens": total_tokens,
        "bases": total_bases,
        "seconds": time.time() - started,
    }


def _save_checkpoint(path: Path, raw_model: torch.nn.Module, optimizer: AdamW, step: int, best_val_bpb: float, scheduler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "best_val_bpb": best_val_bpb,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "model_family": "megadna",
    }
    torch.save(payload, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune or evaluate official megaDNA on DNACorpus.")
    parser.add_argument("--config", help="Optional JSON config. Defaults are megaDNA-specific when omitted.")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--checkpoint", help="This script's best.pt/last.pt checkpoint to load.")
    parser.add_argument("--base-weight", default=str(default_megadna_weight_path()), help="Official full-object megaDNA .pt weight.")
    parser.add_argument("--non-acgt-policy", choices=["reject", "filter"], default="reject")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timestamp-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestamp-format", default="%Y%m%d_%H%M%S")
    parser.add_argument("--override", action="append", default=[])

    data = parser.add_argument_group("data")
    data.add_argument("--dataset-dir")
    data.add_argument("--species", nargs="+")
    data.add_argument("--sequence-source-mode", choices=["auto", "flat_file", "fasta_dir"])
    data.add_argument("--multi-sequence-mode", choices=["separate", "concat"])
    data.add_argument("--clean-cache-enabled", dest="clean_cache_enabled", action="store_true", default=None)
    data.add_argument("--no-clean-cache", dest="clean_cache_enabled", action="store_false")
    data.add_argument("--clean-cache-dir")
    data.add_argument("--sequence-include", action="append")
    data.add_argument("--train-ratio", type=float)
    data.add_argument("--val-ratio", type=float)
    data.add_argument("--test-ratio", type=float)
    data.add_argument("--max-train-bytes", type=int)
    data.add_argument("--max-val-bytes", type=int)
    data.add_argument("--max-test-bytes", type=int)
    data.add_argument("--train-samples-per-epoch", type=int)
    data.add_argument("--train-sampling-strategy", choices=["proportional", "uniform", "sqrt"])
    data.add_argument("--compression-sample-bytes", type=int)

    train = parser.add_argument_group("train")
    train.add_argument("--seq-length", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    train.add_argument("--init-from", choices=["pretrained", "resume"])
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--eval-batch-size", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--weight-decay", type=float)
    train.add_argument("--lr-scheduler", choices=["none", "linear", "cosine"])
    train.add_argument("--lr-warmup-steps", type=int)
    train.add_argument("--lr-min-ratio", type=float)
    train.add_argument("--grad-clip-norm", type=float)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--prefetch-factor", type=int, help="DataLoader prefetch factor per worker (effective when num_workers > 0).")
    train.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader workers alive between epochs (effective when num_workers > 0).",
    )
    train.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pinned host memory for faster host-to-device transfer on CUDA.",
    )
    train.add_argument("--log-interval", type=int)
    train.add_argument("--eval-interval", type=int)

    output = parser.add_argument_group("output")
    output.add_argument("--run-name")
    output.add_argument("--output-dir")
    output.add_argument(
        "--tracking-backend",
        choices=["swanlab", "wandb", "both"],
        help="Experiment tracking backend. Reuses the wandb_* project/name fields for compatibility.",
    )
    output.add_argument("--wandb-project", help="Enable realtime W&B logging and set project name.")
    output.add_argument("--wandb-entity", help="Optional W&B entity/team.")
    output.add_argument("--wandb-name", help="Optional W&B run name.")
    output.add_argument("--wandb-group", help="Optional W&B group.")
    output.add_argument("--wandb-tags", nargs="+", help="Optional W&B tags.")
    output.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    output.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable realtime W&B logging.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_experiment_config(args.config) if args.config else _default_config()
    _apply_overrides(config, args)
    _apply_timestamp_to_output_dir(config, args)
    _validate(config, args)
    if args.print_config or args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))
    if args.dry_run:
        print("Dry-run completed: config resolved and validated.")
        return

    seed_everything(config.train.seed)
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, output_dir / "resolved_config.json")

    # --- file-based training log + wandb ---
    train_log_handle: object = None
    wandb_run = None
    training_log_path, train_log_handle = open_training_log_file(output_dir)
    wandb_run = init_wandb_run(config, output_dir)

    print("[megadna] loading splits...", flush=True)
    split_started = time.time()
    splits = load_splits(config.data, seq_length=config.model.seq_length)
    train_sources, train_encoding = encode_sources_for_megadna(splits.train_sources, non_acgt_policy=args.non_acgt_policy)
    val_sources, val_encoding = encode_sources_for_megadna(splits.val_sources, non_acgt_policy=args.non_acgt_policy)
    test_sources, test_encoding = encode_sources_for_megadna(splits.test_sources, non_acgt_policy=args.non_acgt_policy)
    split_entries = splits.summary["species"]
    total_train_bytes = int(sum(int(item["train_bytes"]) for item in split_entries))
    total_val_bytes = int(sum(int(item["val_bytes"]) for item in split_entries))
    total_test_bytes = int(sum(int(item["test_bytes"]) for item in split_entries))
    clean_cache_summary = splits.summary.get("clean_cache", {})
    print(
        "[megadna] data splits loaded: "
        f"sources={len(split_entries)} "
        f"train_bytes={total_train_bytes} "
        f"val_bytes={total_val_bytes} "
        f"test_bytes={total_test_bytes} "
        f"elapsed={time.time() - split_started:.1f}s",
        flush=True,
    )
    if isinstance(clean_cache_summary, dict) and int(clean_cache_summary.get("applicable_sources", 0)) > 0:
        print(
            "[cache] clean "
            f"enabled={bool(clean_cache_summary.get('enabled'))} "
            f"dir={clean_cache_summary.get('cache_dir')} "
            f"hits={int(clean_cache_summary.get('hits', 0))} "
            f"created={int(clean_cache_summary.get('created', 0))} "
            f"rebuilt={int(clean_cache_summary.get('rebuilt', 0))} "
            f"disabled={int(clean_cache_summary.get('disabled', 0))}",
            flush=True,
        )

    print("[megadna] building datasets...", flush=True)
    train_dataset = RandomMegaDNAWindowDataset(
        train_sources,
        seq_length=config.model.seq_length,
        samples_per_epoch=config.data.train_samples_per_epoch,
        seed=config.train.seed,
        sampling_strategy=config.data.train_sampling_strategy,
    )
    val_dataset = SequentialMegaDNAWindowDataset(val_sources, seq_length=config.model.seq_length)
    test_dataset = SequentialMegaDNAWindowDataset(test_sources, seq_length=config.model.seq_length)
    loader_kwargs = _loader_kwargs(config)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.train.eval_batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.train.eval_batch_size, shuffle=False, **loader_kwargs)
    print(
        f"[megadna] datasets ready: train_samples={len(train_dataset)} "
        f"val_windows={len(val_dataset)} test_windows={len(test_dataset)}",
        flush=True,
    )

    device = resolve_device(config.train.device)
    raw_model, checkpoint_payload, checkpoint_path = _load_model_and_checkpoint(config, args, device)
    model = wrap_megadna_for_target_aligned_logits(raw_model).to(device)
    optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=config.train.lr_scheduler,
        warmup_steps=config.train.lr_warmup_steps,
        total_steps=max(1, config.train.epochs * len(train_loader)),
        min_ratio=config.train.lr_min_ratio,
    )
    best_val_bpb = float("inf")
    global_step = 0
    if config.train.init_from == "resume" and isinstance(checkpoint_payload, dict):
        if isinstance(checkpoint_payload.get("optimizer_state"), dict):
            optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
        if scheduler is not None and isinstance(checkpoint_payload.get("scheduler_state"), dict):
            scheduler.load_state_dict(checkpoint_payload["scheduler_state"])
        best_val_bpb = float(checkpoint_payload.get("best_val_bpb", float("inf")))
        global_step = int(checkpoint_payload.get("step", 0))

    summary = {
        "device": str(device),
        "base_weight": config.model.pretrained_weight_path,
        "loaded_checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "non_acgt_policy": args.non_acgt_policy,
        "encoding": {"train": train_encoding, "val": val_encoding, "test": test_encoding},
        "dataset": splits.summary,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics: dict[str, object] = {"summary": summary}
    if args.mode in {"train", "all"}:
        print("[megadna] starting training...", flush=True)
        model.train()
        started = time.time()
        for epoch in range(config.train.epochs):
            for batch in train_loader:
                ids = batch["input_ids"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, config.train.dtype):
                    output = model(ids, return_loss=True)
                output.loss.backward()
                if config.train.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip_norm)
                grad_norm_val = sum(
                    p.grad.detach().norm() for p in model.parameters() if p.grad is not None
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if global_step % config.train.log_interval == 0:
                    loss_item = float(output.loss.item())
                    bpb = loss_item / 0.6931471805599453
                    tokens_per_second = (
                        config.train.batch_size * config.model.seq_length * config.train.log_interval
                    ) / max(time.time() - started, 1e-6)
                    if train_log_handle is not None:
                        write_training_log_event(
                            train_log_handle,
                            {
                                "event": "train",
                                "step": global_step,
                                "epoch": epoch + 1,
                                "loss_nats_per_token": loss_item,
                                "bits_per_base": bpb,
                                "grad_norm": float(grad_norm_val),
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "tokens_per_second": float(tokens_per_second),
                            },
                        )
                    log_wandb_metrics(
                        wandb_run,
                        {
                            "epoch": epoch + 1,
                            "train/loss": loss_item,
                            "train/bpb": bpb,
                            "train/grad_norm": float(grad_norm_val),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/tokens_per_second": float(tokens_per_second),
                        },
                        step=global_step,
                    )
                    print(
                        f"[train] epoch={epoch + 1} step={global_step} "
                        f"loss/token={loss_item:.4f} bits/base={bpb:.4f} "
                        f"grad_norm={float(grad_norm_val):.4f} "
                        f"tokens/s={tokens_per_second:.1f} "
                        f"lr={optimizer.param_groups[0]['lr']:.6g}",
                        flush=True,
                    )
                    started = time.time()
                if global_step % config.train.eval_interval == 0:
                    val_metrics = _evaluate(model, val_loader, device=device, dtype_name=config.train.dtype)
                    if train_log_handle is not None:
                        write_training_log_event(
                            train_log_handle,
                            {
                                "event": "eval",
                                "split": "val",
                                "step": global_step,
                                "epoch": epoch + 1,
                                "loss_nats_per_token": float(val_metrics["loss_nats_per_token"]),
                                "bits_per_base": float(val_metrics["bits_per_base"]),
                                "tokens": int(val_metrics["tokens"]),
                                "bases": int(val_metrics["bases"]),
                            },
                        )
                    log_wandb_metrics(
                        wandb_run,
                        {
                            "epoch": epoch + 1,
                            "eval/loss": float(val_metrics["loss_nats_per_token"]),
                            "eval/bpb": float(val_metrics["bits_per_base"]),
                            "val/loss": float(val_metrics["loss_nats_per_token"]),
                            "val/bpb": float(val_metrics["bits_per_base"]),
                        },
                        step=global_step,
                    )
                    print(
                        f"[eval] step={global_step} val_loss/token={val_metrics['loss_nats_per_token']:.4f} "
                        f"val_bits/base={val_metrics['bits_per_base']:.4f}",
                        flush=True,
                    )
                    if val_metrics["bits_per_base"] < best_val_bpb:
                        best_val_bpb = val_metrics["bits_per_base"]
                        _save_checkpoint(output_dir / "best.pt", raw_model, optimizer, global_step, best_val_bpb, scheduler)
            # end-of-epoch eval
            val_metrics = _evaluate(model, val_loader, device=device, dtype_name=config.train.dtype)
            if train_log_handle is not None:
                write_training_log_event(
                    train_log_handle,
                    {
                        "event": "eval",
                        "split": "val",
                        "step": global_step,
                        "epoch": epoch + 1,
                        "loss_nats_per_token": float(val_metrics["loss_nats_per_token"]),
                        "bits_per_base": float(val_metrics["bits_per_base"]),
                        "tokens": int(val_metrics["tokens"]),
                        "bases": int(val_metrics["bases"]),
                    },
                )
            log_wandb_metrics(
                wandb_run,
                {
                    "epoch": epoch + 1,
                    "eval/loss": float(val_metrics["loss_nats_per_token"]),
                    "eval/bpb": float(val_metrics["bits_per_base"]),
                    "val/loss": float(val_metrics["loss_nats_per_token"]),
                    "val/bpb": float(val_metrics["bits_per_base"]),
                },
                step=global_step,
            )
            print(
                f"[eval] epoch={epoch + 1} val_loss/token={val_metrics['loss_nats_per_token']:.4f} "
                f"val_bits/base={val_metrics['bits_per_base']:.4f}",
                flush=True,
            )
            if val_metrics["bits_per_base"] < best_val_bpb:
                best_val_bpb = val_metrics["bits_per_base"]
                _save_checkpoint(output_dir / "best.pt", raw_model, optimizer, global_step, best_val_bpb, scheduler)
            _save_checkpoint(output_dir / "last.pt", raw_model, optimizer, global_step, best_val_bpb, scheduler)
        metrics["train_last_step"] = global_step

    if args.mode in {"eval", "all"}:
        val_metrics = _evaluate(model, val_loader, device=device, dtype_name=config.train.dtype)
        test_metrics = _evaluate(model, test_loader, device=device, dtype_name=config.train.dtype)
        metrics["val"] = val_metrics
        metrics["test"] = test_metrics
        if train_log_handle is not None:
            for split_name, split_metrics in [("val", val_metrics), ("test", test_metrics)]:
                write_training_log_event(
                    train_log_handle,
                    {
                        "event": "eval",
                        "split": split_name,
                        "step": global_step,
                        "loss_nats_per_token": float(split_metrics["loss_nats_per_token"]),
                        "bits_per_base": float(split_metrics["bits_per_base"]),
                        "tokens": int(split_metrics["tokens"]),
                        "bases": int(split_metrics["bases"]),
                    },
                )
        log_wandb_metrics(
            wandb_run,
            {
                "val/loss": float(val_metrics["loss_nats_per_token"]),
                "val/bpb": float(val_metrics["bits_per_base"]),
                "test/loss": float(test_metrics["loss_nats_per_token"]),
                "test/bpb": float(test_metrics["bits_per_base"]),
            },
            step=global_step,
        )
        (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2, ensure_ascii=False))

    if train_log_handle is not None:
        train_log_handle.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
