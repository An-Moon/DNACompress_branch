from __future__ import annotations

"""Run standalone megaDNA compression on DNACorpus splits.

Examples:

    # Compress a small test slice with the official 145M checkpoint.
    python scripts/run_megadna_compression.py \
      --dataset-dir datasets/DNACorpus \
      --species HoSa \
      --split test \
      --seq-length 1024 \
      --eval-batch-size 1 \
      --compression-sample-bytes 4096 \
      --compression-modes windows_nonoverlap \
      --non-acgt-policy filter \
      --device cpu \
      --output-json outputs/megadna_145m/compression_test.json

    # Compress from a fine-tuned run directory.
    python scripts/run_megadna_compression.py \
      --run-dir outputs/megadna_hosa \
      --checkpoint-tag best \
      --split train val test \
      --compression-modes windows_nonoverlap windows_overlap \
      --overlap-stride 512 \
      --non-acgt-policy filter \
      --device cuda:0

    # Explicit paths.
    python scripts/run_megadna_compression.py \
      --config outputs/megadna_hosa/resolved_config.json \
      --checkpoint outputs/megadna_hosa/best.pt \
      --split test
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import (
    NON_OVERLAP_MODE,
    OVERLAP_MODE,
    SLIDING_TOKEN_MODE,
    summarize_per_source,
)
from dna_compress.config import ExperimentConfig, load_experiment_config
from dna_compress.data import load_splits
from dna_compress.megadna_compression import (
    SUPPORTED_MEGADNA_COMPRESSION_MODES,
    compress_megadna_source,
)
from dna_compress.megadna_loader import (
    MEGADNA_EOS_ID,
    MEGADNA_PAD_ID,
    default_megadna_weight_path,
    load_megadna_model,
    wrap_megadna_for_target_aligned_logits,
)
from dna_compress.experiment import resolve_device


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
    config.train.eval_batch_size = 1
    config.train.dtype = "bfloat16"
    config.output.output_dir = "outputs/megadna_145m"
    return config


def _resolve_config_path(args: argparse.Namespace) -> Path | None:
    if args.config:
        return Path(args.config)
    if args.run_dir:
        return Path(args.run_dir) / "resolved_config.json"
    return None


def _checkpoint_path(args: argparse.Namespace, config: ExperimentConfig) -> Path | None:
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.run_dir:
        candidate = Path(args.run_dir) / f"{args.checkpoint_tag}.pt"
        return candidate if candidate.exists() else None
    return None


def _apply_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.species is not None:
        config.data.species = args.species
    config.model.implementation = "megadna"
    config.model.vocab_size = 6
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
    _apply_if_not_none(config, "data.compression_sample_bytes", args.compression_sample_bytes)
    _apply_if_not_none(config, "train.device", args.device)
    _apply_if_not_none(config, "train.dtype", args.dtype)
    _apply_if_not_none(config, "train.eval_batch_size", args.eval_batch_size)
    _apply_if_not_none(config, "arithmetic.frequency_total", args.arithmetic_frequency_total)
    _apply_if_not_none(config, "arithmetic.target_uniform_mass", args.arithmetic_target_uniform_mass)
    _apply_if_not_none(config, "output.output_dir", args.output_dir)

    for item in args.override:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected section.key=value")
        key, raw_value = item.split("=", 1)
        _set_nested_attr(config, key.strip(), _parse_scalar(raw_value.strip()))


def _load_model(config: ExperimentConfig, checkpoint_path: Path | None, device: torch.device) -> tuple[torch.nn.Module, dict[str, object]]:
    raw_model = load_megadna_model(config.model.pretrained_weight_path or default_megadna_weight_path(), device=device)
    checkpoint_metadata: dict[str, object] = {}
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            raw_model.load_state_dict(checkpoint["model_state"], strict=True)
            checkpoint_metadata = {
                "checkpoint_step": checkpoint.get("step"),
                "best_val_bpb": checkpoint.get("best_val_bpb"),
            }
        elif isinstance(checkpoint, torch.nn.Module):
            raw_model = checkpoint.to(device)
        else:
            raise ValueError(f"Unsupported megaDNA checkpoint format: {checkpoint_path}")
    return wrap_megadna_for_target_aligned_logits(raw_model).to(device), checkpoint_metadata


def _normalize_splits(raw_splits: list[str]) -> list[str]:
    if "all" in raw_splits:
        return ["train", "val", "test"]
    return raw_splits


def _sources_for_split(splits, split_name: str) -> list[bytes]:
    if split_name == "train":
        return splits.train_sources
    if split_name == "val":
        return splits.val_sources
    if split_name == "test":
        return splits.test_sources
    raise ValueError(f"Unsupported split '{split_name}'")


def _source_entries(splits) -> list[dict[str, object]]:
    return [dict(item) for item in splits.summary["species"]]


def _resolve_output_json(args: argparse.Namespace, config: ExperimentConfig) -> Path:
    if args.output_json:
        return Path(args.output_json)
    if args.run_dir:
        return Path(args.run_dir) / "megadna_compression.json"
    return Path(config.output.output_dir) / "megadna_compression.json"


def _validate(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if config.model.seq_length <= 0:
        raise ValueError("model.seq_length must be > 0")
    if config.train.eval_batch_size <= 0:
        raise ValueError("train.eval_batch_size must be > 0")
    if args.overlap_stride is not None and args.overlap_stride <= 0:
        raise ValueError("--overlap-stride must be > 0")
    if OVERLAP_MODE in args.compression_modes:
        stride = args.overlap_stride or max(1, config.model.seq_length // 2)
        if stride >= config.model.seq_length:
            raise ValueError("overlap stride must be smaller than seq_length")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone megaDNA compression on DNACorpus splits.")
    parser.add_argument("--run-dir")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--base-weight", default=str(default_megadna_weight_path()))
    parser.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--compression-modes",
        nargs="+",
        default=[NON_OVERLAP_MODE],
        choices=list(SUPPORTED_MEGADNA_COMPRESSION_MODES),
    )
    parser.add_argument("--overlap-stride", type=int)
    parser.add_argument("--non-acgt-policy", choices=["reject", "filter"], default="reject")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--override", action="append", default=[])

    data = parser.add_argument_group("model/data")
    data.add_argument("--seq-length", type=int)
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
    data.add_argument("--compression-sample-bytes", type=int)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device")
    runtime.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    runtime.add_argument("--eval-batch-size", type=int)
    runtime.add_argument("--output-dir")
    runtime.add_argument("--arithmetic-frequency-total", type=int)
    runtime.add_argument("--arithmetic-target-uniform-mass", type=float)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config_path = _resolve_config_path(args)
    if config_path is not None and not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_experiment_config(config_path) if config_path is not None else _default_config()
    _apply_overrides(config, args)
    _validate(config, args)
    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    device = resolve_device(config.train.device)
    checkpoint_path = _checkpoint_path(args, config)
    model, checkpoint_metadata = _load_model(config, checkpoint_path, device)

    print("[megadna-compress] loading splits...", flush=True)
    started = time.time()
    splits = load_splits(config.data, seq_length=config.model.seq_length)
    print(f"[megadna-compress] loaded {len(splits.summary['species'])} sources in {time.time() - started:.1f}s", flush=True)

    overlap_stride = args.overlap_stride or max(1, config.model.seq_length // 2)
    requested_splits = _normalize_splits(args.split)
    metrics: dict[str, object] = {
        "device": str(device),
        "base_weight": config.model.pretrained_weight_path,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        **checkpoint_metadata,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "non_acgt_policy": args.non_acgt_policy,
        "overlap_stride_tokens": overlap_stride,
        "resolved_config": config.to_dict(),
        "dataset": splits.summary,
        "results": {},
    }

    entries = _source_entries(splits)
    for split_name in requested_splits:
        split_sources = _sources_for_split(splits, split_name)
        split_result: dict[str, object] = {}
        for mode in args.compression_modes:
            per_source: list[dict[str, object]] = []
            for source_index, (entry, source) in enumerate(zip(entries, split_sources), start=1):
                source_name = str(entry.get("source_name", entry.get("species", source_index)))

                def _on_progress(batch_done: int, batch_total: int, *, si: int = source_index, sn: str = source_name) -> None:
                    ratio = 100.0 * batch_done / max(batch_total, 1)
                    print(
                        f"\r[megadna-compress] split={split_name} mode={mode} "
                        f"source={si}/{len(split_sources)}({sn}) batch={batch_done}/{batch_total} ({ratio:5.1f}%)",
                        end="",
                        flush=True,
                    )

                row = compress_megadna_source(
                    model=model,
                    source=source,
                    seq_length=config.model.seq_length,
                    device=device,
                    dtype_name=config.train.dtype,
                    batch_size=config.train.eval_batch_size,
                    requested_bytes=config.data.compression_sample_bytes,
                    mode=mode,
                    overlap_stride=overlap_stride,
                    arithmetic_frequency_total=config.arithmetic.frequency_total,
                    arithmetic_target_uniform_mass=config.arithmetic.target_uniform_mass,
                    non_acgt_policy=args.non_acgt_policy,
                    progress_callback=_on_progress,
                )
                print()
                per_source.append({"species": str(entry["species"]), "source_name": source_name, **row})
            split_result[mode] = {"aggregate": summarize_per_source(per_source), "per_source": per_source}
        metrics["results"][split_name] = split_result

    output_json = _resolve_output_json(args, config)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved megaDNA compression metrics to {output_json}")


if __name__ == "__main__":
    main()
