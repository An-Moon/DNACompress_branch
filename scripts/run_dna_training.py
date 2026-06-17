from __future__ import annotations

"""Run DNA Megabyte training.

Complete example (train + eval + compression, with common overrides):

    python scripts/run_dna_training.py \
        --config configs/dna_megabyte_large.json \
        --mode all \
        --init-from scratch \
        --pretrained-weight-path outputs/dna_megabyte_huge_ensembl_all/best.pt \
        --seed 42 \
        --dataset-dir datasets/DNACorpus \
        --sequence-source-mode auto \
        --multi-sequence-mode separate \
        --dtype bfloat16 \
        --epochs 1 \
        --batch-size 32 \
        --eval-batch-size 32 \
        --learning-rate 1e-4 \
        --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
        --train-samples-per-epoch 600000 \
        --compression-sample-bytes 100000 \
        --print-config \
        --seq-length 1024 \
        --token-merge-size 3 \
        --weight-decay 0.01 \
        --log-interval 25 \
        --eval-interval 2500 \
        --train-ratio 0.6 \
        --val-ratio 0.2 \
        --test-ratio 0.2 \
        --lr-scheduler cosine \
        --lr-warmup-steps 0 \
        --lr-min-ratio 0.1 \
        --grad-clip-norm 1.0 \
        --num-workers 4 
            
        --wandb-project dna-compress \
        --wandb-name dna_megabyte_huge_ensembl_all_resume \
        --gpu-ids 2 3 \
        --species homo_sapiens mus_musculus bos_taurus danio_rerio \
                  drosophila_melanogaster caenorhabditis_elegans \
                  saccharomyces_cerevisiae arabidopsis_thaliana \
        --init-from pretrained \
        --pretrained-weight-path outputs/dna_megabyte_huge_b128_ensembl_all/best.pt 

OpenGenome2 indexed FASTA variant (train + eval; compression is recorded as skipped):
  
    CUDA_VISIBLE_DEVICES=3 python scripts/run_dna_training.py \
        --config configs/dna_megabyte_large.json \
        --mode all \
        --init-from resume \
        --pretrained-weight-path outputs/dna_megabyte_large_20260616_144744_20260616_171309/last.pt \
        --seed 42 \
        --sequence-source-mode indexed_fasta \
        --fasta-index-dir /data/students/Liang_junnan/opengenome2_subset/index \
        --indexed-eval-samples 1024 \
        --indexed-eval-cache-dir /data/students/Liang_junnan/opengenome2_subset/eval_cache \
        --indexed-eval-cache-mode reuse \
        --indexed-eval-random-seed 0 \
        --indexed-split-seed 0 \
        --indexed-window-mode source_batch_file_stream \
        --indexed-train-epoch-mode all_windows \
        --indexed-file-stream-order-seed 0 \
        --indexed-source-read-chunk-windows 8192 \
        --indexed-source-read-chunk-shuffle \
        --indexed-source-file-order-seed 0 \
        --source-sampling-weights-json '{"gtdb_v220":0.35,"metagenomes":0.3,"ncbi_eukaryotic_genomes":0.25,"plasmids_phage":0.1}' \
        --dtype float16 \
        --epochs 2 \
        --batch-size 32 \
        --eval-batch-size 32 \
        --learning-rate 3e-5 \
        --print-config \
        --seq-length 1024 \
        --token-merge-size 3 \
        --weight-decay 0.01 \
        --log-interval 25 \
        --eval-interval 2500 \
        --train-ratio 0.98 \
        --val-ratio 0.01 \
        --test-ratio 0.01 \
        --lr-scheduler cosine \
        --lr-warmup-steps 0 \
        --lr-min-ratio 0.1 \
        --grad-clip-norm 1.0 \
        --num-workers 2 \
        --no-persistent-workers \
        --wandb-project dna-compress \
        --wandb-name dna_megabyte_large_opengenome2_9

OpenGenome2 repacked window variant (train + eval; compression is recorded as skipped):
  
    CUDA_VISIBLE_DEVICES=3 python scripts/run_dna_training.py \
        --config configs/dna_megabyte_large.json \
        --mode all \
        --init-from scratch \
        --seed 42 \
        --sequence-source-mode repacked_windows \
        --repacked-window-dir /data/students/Liang_junnan/opengenome2_subset/repacked_megabyte_s1024_m3_hashshard \
        --repacked-schedule-dir /data/students/Liang_junnan/opengenome2_subset/repacked_megabyte_s1024_m3_hashshard/schedules/split_seed_0_train_0.98_val_0.01_test_0.01 \
        --repacked-eval-samples 1024 \
        --repacked-train-epoch-mode samples \
        --repacked-read-chunk-windows 8192 \
        --repacked-shard-load-mode mmap \
        --repacked-shard-sampling-mode random \
        --source-loss-weights-json '{"gtdb_v220":0.35,"metagenomes":0.3,"ncbi_eukaryotic_genomes":0.25,"plasmids_phage":0.1}' \
        --dtype bfloat16 \
        --epochs 2 \
        --batch-size 32 \
        --eval-batch-size 32 \
        --learning-rate 1e-4 \
        --print-config \
        --seq-length 1024 \
        --token-merge-size 3 \
        --weight-decay 0.01 \
        --log-interval 25 \
        --eval-interval 2500 \
        --train-ratio 0.98 \
        --val-ratio 0.01 \
        --test-ratio 0.01 \
        --lr-scheduler cosine \
        --lr-warmup-steps 1024 \
        --lr-min-ratio 0.1 \
        --grad-clip-norm 1.0 \
        --num-workers 4 

Multi-GPU DDP example (2 GPUs):

    torchrun --nproc_per_node=2 scripts/run_dna_training.py \
        --config configs/dna_megabyte_quick.json \
        --mode train \
        --dataset-dir datasets/ensembl_raw \
        --sequence-source-mode auto \
        --multi-sequence-mode separate \
        --species homo_sapiens mus_musculus bos_taurus danio_rerio \
                  drosophila_melanogaster caenorhabditis_elegans \
                  saccharomyces_cerevisiae arabidopsis_thaliana \
        --device cuda \
        --num-workers 8 \
        --prefetch-factor 4 \
        --persistent-workers \
        --pin-memory \
        --gpus 0 1
    
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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.config import load_experiment_config
from dna_compress.tokenization import apply_token_merge_to_model_config, normalize_alphabet


def _resolve_resume_checkpoint_path(explicit_path: str | None, output_dir: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    default_resume_path = Path(output_dir) / "last.pt"
    if default_resume_path.exists():
        return default_resume_path
    raise FileNotFoundError(
        "train.init_from='resume' but no checkpoint path was provided and output_dir/last.pt does not exist."
    )


def _load_resume_default_config(base_config: Any, args: argparse.Namespace) -> tuple[Any, Path | None, Path | None]:
    requested_init_from = args.init_from if args.init_from is not None else base_config.train.init_from
    if requested_init_from != "resume":
        return base_config, None, None

    checkpoint_path = _resolve_resume_checkpoint_path(
        explicit_path=args.pretrained_weight_path or base_config.model.pretrained_weight_path,
        output_dir=args.output_dir or base_config.output.output_dir,
    )
    resolved_config_path = checkpoint_path.parent / "resolved_config.json"
    if not resolved_config_path.exists():
        raise FileNotFoundError(
            f"train.init_from='resume' requires resume defaults at {resolved_config_path}, "
            "but the file does not exist."
        )
    return load_experiment_config(resolved_config_path), checkpoint_path, resolved_config_path


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


def _parse_gpu_ids(values: list[str]) -> list[int]:
    gpu_ids: list[int] = []
    for value in values:
        for token in value.split(","):
            item = token.strip()
            if not item:
                continue
            try:
                gpu_id = int(item)
            except ValueError as error:
                raise ValueError(f"Invalid GPU id '{item}'. Expected integers like 0 1 or 0,1.") from error
            if gpu_id < 0:
                raise ValueError(f"Invalid GPU id '{item}'. GPU id must be >= 0.")
            gpu_ids.append(gpu_id)

    if not gpu_ids:
        raise ValueError("--gpus was provided but no valid GPU ids were parsed.")

    # Preserve order while dropping duplicates.
    deduplicated_gpu_ids: list[int] = []
    for gpu_id in gpu_ids:
        if gpu_id not in deduplicated_gpu_ids:
            deduplicated_gpu_ids.append(gpu_id)
    return deduplicated_gpu_ids


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


def _parse_weight_map(raw: str | None, *, option_name: str, key_label: str) -> dict[str, float] | None:
    if raw is None:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{option_name} must be a JSON object")
    parsed: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{key_label} keys must be non-empty strings")
        weight = float(value)
        if weight < 0:
            raise ValueError(f"{key_label} weights must be non-negative")
        parsed[key] = weight
    if sum(parsed.values()) <= 0:
        raise ValueError(f"{key_label} weights must sum to > 0")
    return parsed


def _parse_source_sampling_weights(raw: str | None) -> dict[str, float] | None:
    return _parse_weight_map(raw, option_name="--source-sampling-weights-json", key_label="source sampling")


def _parse_source_loss_weights(raw: str | None) -> dict[str, float] | None:
    return _parse_weight_map(raw, option_name="--source-loss-weights-json", key_label="source loss")


def _parse_repacked_source_sampling_weights(raw: str | None) -> dict[str, float] | None:
    return _parse_weight_map(
        raw,
        option_name="--repacked-source-sampling-weights-json",
        key_label="repacked source sampling",
    )


def _set_nested_attr(config: Any, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if len(parts) < 2:
        raise ValueError(f"Override key must include section and field: {dotted_key}")

    target: Any = config
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise ValueError(f"Unknown config section/path: {'.'.join(parts[:-1])}")
        target = getattr(target, part)

    field_name = parts[-1]
    if not hasattr(target, field_name):
        raise ValueError(f"Unknown config field: {dotted_key}")
    setattr(target, field_name, value)


def _apply_if_not_none(config: Any, dotted_key: str, value: Any) -> None:
    if value is None:
        return
    _set_nested_attr(config, dotted_key, value)


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    # Generic overrides take top priority after explicit CLI flags.
    if args.species is not None:
        config.data.species = args.species

    _apply_if_not_none(config, "model.implementation", args.implementation)
    _apply_if_not_none(config, "model.pretrained_weight_path", args.pretrained_weight_path)
    _apply_if_not_none(config, "model.patch_size", args.patch_size)
    _apply_if_not_none(config, "model.seq_length", args.seq_length)
    _apply_if_not_none(config, "model.global_dim", args.global_dim)
    _apply_if_not_none(config, "model.local_dim", args.local_dim)
    _apply_if_not_none(config, "model.global_heads", args.global_heads)
    _apply_if_not_none(config, "model.global_layers", args.global_layers)
    _apply_if_not_none(config, "model.local_heads", args.local_heads)
    _apply_if_not_none(config, "model.local_layers", args.local_layers)
    _apply_if_not_none(config, "model.attn_dropout", args.attn_dropout)
    _apply_if_not_none(config, "model.ff_dropout", args.ff_dropout)
    _apply_if_not_none(config, "model.flash_attn", args.flash_attn)
    _apply_if_not_none(config, "model.input_causal_conv_kernel_size", args.input_causal_conv_kernel_size)

    _apply_if_not_none(config, "data.dataset_dir", args.dataset_dir)
    _apply_if_not_none(config, "data.sequence_source_mode", args.sequence_source_mode)
    _apply_if_not_none(config, "data.fasta_index_dir", args.fasta_index_dir)
    _apply_if_not_none(config, "data.indexed_eval_samples", args.indexed_eval_samples)
    _apply_if_not_none(config, "data.indexed_eval_cache_dir", args.indexed_eval_cache_dir)
    _apply_if_not_none(config, "data.indexed_eval_cache_mode", args.indexed_eval_cache_mode)
    _apply_if_not_none(config, "data.indexed_eval_random_seed", args.indexed_eval_random_seed)
    _apply_if_not_none(config, "data.indexed_split_seed", args.indexed_split_seed)
    _apply_if_not_none(config, "data.indexed_window_mode", args.indexed_window_mode)
    _apply_if_not_none(config, "data.indexed_train_epoch_mode", args.indexed_train_epoch_mode)
    _apply_if_not_none(config, "data.indexed_file_stream_windows", args.indexed_file_stream_windows)
    _apply_if_not_none(config, "data.indexed_file_shuffle_buffer_windows", args.indexed_file_shuffle_buffer_windows)
    _apply_if_not_none(config, "data.indexed_file_stream_order_seed", args.indexed_file_stream_order_seed)
    _apply_if_not_none(config, "data.indexed_source_mix_chunk_batches", args.indexed_source_mix_chunk_batches)
    _apply_if_not_none(config, "data.indexed_source_read_chunk_windows", args.indexed_source_read_chunk_windows)
    _apply_if_not_none(config, "data.indexed_source_read_chunk_shuffle", args.indexed_source_read_chunk_shuffle)
    _apply_if_not_none(config, "data.indexed_source_balance_batches", args.indexed_source_balance_batches)
    _apply_if_not_none(config, "data.indexed_source_read_block_windows", args.indexed_source_read_block_windows)
    if args.indexed_source_mix_chunk_batches is None and args.indexed_source_balance_batches is not None:
        config.data.indexed_source_mix_chunk_batches = int(args.indexed_source_balance_batches)
    elif args.indexed_source_mix_chunk_batches is not None:
        config.data.indexed_source_balance_batches = None
    if args.indexed_source_read_chunk_windows is None and args.indexed_source_read_block_windows is not None:
        config.data.indexed_source_read_chunk_windows = int(args.indexed_source_read_block_windows)
    elif args.indexed_source_read_chunk_windows is not None:
        config.data.indexed_source_read_block_windows = None
    _apply_if_not_none(config, "data.indexed_source_file_order_seed", args.indexed_source_file_order_seed)
    _apply_if_not_none(config, "data.repacked_window_dir", args.repacked_window_dir)
    _apply_if_not_none(config, "data.repacked_schedule_dir", args.repacked_schedule_dir)
    _apply_if_not_none(config, "data.repacked_eval_samples", args.repacked_eval_samples)
    _apply_if_not_none(config, "data.repacked_train_epoch_mode", args.repacked_train_epoch_mode)
    _apply_if_not_none(config, "data.repacked_read_chunk_windows", args.repacked_read_chunk_windows)
    _apply_if_not_none(config, "data.repacked_shard_load_mode", args.repacked_shard_load_mode)
    _apply_if_not_none(config, "data.repacked_shard_sampling_mode", args.repacked_shard_sampling_mode)
    if args.repacked_source_sampling_weights_json is not None:
        config.data.repacked_source_sampling_weights = (
            _parse_repacked_source_sampling_weights(args.repacked_source_sampling_weights_json) or {}
        )
    if args.source_sampling_weights_json is not None:
        config.data.source_sampling_weights = _parse_source_sampling_weights(args.source_sampling_weights_json) or {}
    if args.source_loss_weights_json is not None:
        config.data.source_loss_weights = _parse_source_loss_weights(args.source_loss_weights_json) or {}
    _apply_if_not_none(config, "data.multi_sequence_mode", args.multi_sequence_mode)
    _apply_if_not_none(config, "data.clean_cache_enabled", args.clean_cache_enabled)
    _apply_if_not_none(config, "data.clean_cache_dir", args.clean_cache_dir)
    if args.sequence_include is not None:
        config.data.sequence_include_map = _parse_sequence_include(args.sequence_include)
    _apply_if_not_none(config, "data.train_ratio", args.train_ratio)
    _apply_if_not_none(config, "data.val_ratio", args.val_ratio)
    _apply_if_not_none(config, "data.test_ratio", args.test_ratio)
    _apply_if_not_none(config, "data.max_train_bytes_per_species", args.max_train_bytes)
    _apply_if_not_none(config, "data.max_val_bytes_per_species", args.max_val_bytes)
    _apply_if_not_none(config, "data.max_test_bytes_per_species", args.max_test_bytes)
    _apply_if_not_none(config, "data.train_samples_per_epoch", args.train_samples_per_epoch)
    _apply_if_not_none(config, "data.train_sampling_strategy", args.train_sampling_strategy)
    _apply_if_not_none(config, "data.token_merge_size", args.token_merge_size)
    _apply_if_not_none(config, "data.token_merge_alphabet", args.token_merge_alphabet)
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
    if args.gpus is not None:
        config.train.gpu_ids = _parse_gpu_ids(args.gpus)

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


def _validate_config_for_megabyte(config: Any, mode: str = "all") -> None:
    if config.model.implementation not in {
        "megabyte",
        "megabyte_in_action",
        "megabyte_in_action_causal_conv",
        "megabyte_relative",
    }:
        raise ValueError(
            "model.implementation must be one of 'megabyte', 'megabyte_in_action', "
            "'megabyte_in_action_causal_conv', or 'megabyte_relative' "
            f"for this project, got '{config.model.implementation}'."
        )

    if config.model.seq_length <= 0 or config.model.patch_size <= 0:
        raise ValueError("model.seq_length and model.patch_size must be > 0")

    if config.model.seq_length % config.model.patch_size != 0:
        raise ValueError(
            f"model.seq_length ({config.model.seq_length}) must be divisible by "
            f"model.patch_size ({config.model.patch_size}) for Megabyte."
        )

    if not (0.0 <= config.model.attn_dropout < 1.0):
        raise ValueError("model.attn_dropout must be in [0.0, 1.0)")

    if not (0.0 <= config.model.ff_dropout < 1.0):
        raise ValueError("model.ff_dropout must be in [0.0, 1.0)")

    if config.model.input_causal_conv_kernel_size <= 0:
        raise ValueError("model.input_causal_conv_kernel_size must be >= 1")

    ratio_sum = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"data split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={config.data.train_ratio}, val={config.data.val_ratio}, test={config.data.test_ratio})."
        )

    if config.train.dtype not in {"float32", "float16", "bfloat16"}:
        raise ValueError("train.dtype must be one of: float32, float16, bfloat16")

    if config.train.init_from not in {"scratch", "pretrained", "resume"}:
        raise ValueError("train.init_from must be one of: scratch, pretrained, resume")

    if config.train.batch_size <= 0 or config.train.eval_batch_size <= 0:
        raise ValueError("train.batch_size and train.eval_batch_size must be > 0")

    if config.data.train_sampling_strategy not in {"proportional", "uniform", "sqrt"}:
        raise ValueError("data.train_sampling_strategy must be one of: proportional, uniform, sqrt")

    if config.data.token_merge_size <= 0:
        raise ValueError("data.token_merge_size must be >= 1")

    normalize_alphabet(config.data.token_merge_alphabet)
    if config.data.sequence_source_mode not in {"auto", "flat_file", "fasta_dir", "indexed_fasta", "repacked_windows"}:
        raise ValueError(
            "data.sequence_source_mode must be one of: auto, flat_file, fasta_dir, indexed_fasta, repacked_windows"
        )
    if config.data.sequence_source_mode == "indexed_fasta":
        if config.data.indexed_source_balance_batches is not None:
            config.data.indexed_source_mix_chunk_batches = int(config.data.indexed_source_balance_batches)
        if config.data.indexed_source_read_block_windows is not None:
            config.data.indexed_source_read_chunk_windows = int(config.data.indexed_source_read_block_windows)
        if mode == "compress":
            raise ValueError("indexed_fasta mode does not support --mode compress; use train, eval, or all.")
        if not config.data.fasta_index_dir:
            raise ValueError("data.fasta_index_dir is required when sequence_source_mode is indexed_fasta")
        if config.data.indexed_eval_samples <= 0:
            raise ValueError("data.indexed_eval_samples must be > 0")
        if config.data.indexed_eval_cache_mode not in {"reuse", "refresh", "off"}:
            raise ValueError("data.indexed_eval_cache_mode must be one of: reuse, refresh, off")
        if config.data.indexed_eval_cache_mode != "off" and not config.data.indexed_eval_cache_dir:
            raise ValueError("data.indexed_eval_cache_dir is required unless indexed_eval_cache_mode='off'")
        if not isinstance(config.data.indexed_eval_random_seed, int):
            raise ValueError("data.indexed_eval_random_seed must be an integer")
        if not isinstance(config.data.indexed_split_seed, int):
            raise ValueError("data.indexed_split_seed must be an integer")
        if config.data.indexed_window_mode not in {
            "sliding_random",
            "nonoverlap_random",
            "nonoverlap_file_stream",
            "source_batch_file_stream",
        }:
            raise ValueError(
                "data.indexed_window_mode must be one of: sliding_random, nonoverlap_random, "
                "nonoverlap_file_stream, source_batch_file_stream"
            )
        if config.data.indexed_train_epoch_mode not in {"samples", "all_windows"}:
            raise ValueError("data.indexed_train_epoch_mode must be one of: samples, all_windows")
        if config.data.indexed_train_epoch_mode == "all_windows" and config.data.indexed_window_mode not in {
            "nonoverlap_random",
            "nonoverlap_file_stream",
            "source_batch_file_stream",
        }:
            raise ValueError(
                "data.indexed_train_epoch_mode='all_windows' requires "
                "indexed_window_mode='nonoverlap_random', 'nonoverlap_file_stream', or 'source_batch_file_stream'"
            )
        if config.data.indexed_file_stream_windows <= 0:
            raise ValueError("data.indexed_file_stream_windows must be > 0")
        if config.data.indexed_file_shuffle_buffer_windows < 0:
            raise ValueError("data.indexed_file_shuffle_buffer_windows must be >= 0")
        if not isinstance(config.data.indexed_file_stream_order_seed, int):
            raise ValueError("data.indexed_file_stream_order_seed must be an integer")
        if config.data.indexed_source_mix_chunk_batches <= 0:
            raise ValueError("data.indexed_source_mix_chunk_batches must be > 0")
        if config.data.indexed_source_read_chunk_windows <= 0:
            raise ValueError("data.indexed_source_read_chunk_windows must be > 0")
        if not isinstance(config.data.indexed_source_file_order_seed, int):
            raise ValueError("data.indexed_source_file_order_seed must be an integer")
        if not isinstance(config.data.source_sampling_weights, dict):
            raise ValueError("data.source_sampling_weights must be a dict[str, float]")
        if not isinstance(config.data.source_loss_weights, dict):
            raise ValueError("data.source_loss_weights must be a dict[str, float]")
        if config.data.indexed_window_mode not in {"sliding_random", "source_batch_file_stream"} and config.data.source_sampling_weights:
            raise ValueError(
                "source_sampling_weights are supported only with indexed_window_mode='sliding_random' "
                "or 'source_batch_file_stream'; use --source-loss-weights-json for other nonoverlap indexed modes."
            )
    if config.data.sequence_source_mode == "repacked_windows":
        if mode == "compress":
            raise ValueError("repacked_windows mode does not support --mode compress; use train, eval, or all.")
        if not config.data.repacked_window_dir:
            raise ValueError("data.repacked_window_dir is required when sequence_source_mode is repacked_windows")
        if config.data.repacked_eval_samples <= 0:
            raise ValueError("data.repacked_eval_samples must be > 0")
        if config.data.repacked_train_epoch_mode not in {"samples", "all_windows"}:
            raise ValueError("data.repacked_train_epoch_mode must be one of: samples, all_windows")
        if config.data.repacked_read_chunk_windows <= 0:
            raise ValueError("data.repacked_read_chunk_windows must be > 0")
        if config.data.repacked_shard_load_mode not in {"mmap", "preload"}:
            raise ValueError("data.repacked_shard_load_mode must be one of: mmap, preload")
        if config.data.repacked_shard_load_mode == "preload" and config.train.num_workers > 0:
            raise ValueError("repacked preload mode requires train.num_workers=0 to avoid duplicating large shards")
        if config.data.repacked_shard_sampling_mode not in {"random", "all_shards"}:
            raise ValueError("data.repacked_shard_sampling_mode must be one of: random, all_shards")
        if not isinstance(config.data.repacked_source_sampling_weights, dict):
            raise ValueError("data.repacked_source_sampling_weights must be a dict[str, float]")
        if config.data.repacked_source_sampling_weights:
            raise ValueError(
                "repacked_source_sampling_weights are not used by hash-shard repacked_windows; "
                "use --source-loss-weights-json to adjust source contribution."
            )
        if not isinstance(config.data.source_loss_weights, dict):
            raise ValueError("data.source_loss_weights must be a dict[str, float]")
        if config.data.source_sampling_weights:
            raise ValueError(
                "source_sampling_weights are for indexed_fasta sliding_random; "
                "use --repacked-source-sampling-weights-json for repacked_windows."
            )
    if config.data.multi_sequence_mode not in {"separate", "concat"}:
        raise ValueError("data.multi_sequence_mode must be one of: separate, concat")
    if not isinstance(config.data.sequence_include_map, dict):
        raise ValueError("data.sequence_include_map must be a dict[str, list[str]]")
    for species_name, keys in config.data.sequence_include_map.items():
        if not isinstance(species_name, str) or not species_name:
            raise ValueError("data.sequence_include_map keys must be non-empty strings")
        if not isinstance(keys, list) or not keys or any((not isinstance(key, str) or not key.strip()) for key in keys):
            raise ValueError(f"data.sequence_include_map[{species_name!r}] must be a non-empty list of strings")

    if config.train.lr_scheduler not in {"none", "linear", "cosine"}:
        raise ValueError("train.lr_scheduler must be one of: none, linear, cosine")

    if config.train.lr_warmup_steps < 0:
        raise ValueError("train.lr_warmup_steps must be >= 0")

    if not (0.0 <= config.train.lr_min_ratio <= 1.0):
        raise ValueError("train.lr_min_ratio must be in [0.0, 1.0]")

    if config.train.num_workers < 0:
        raise ValueError("train.num_workers must be >= 0")

    if config.train.prefetch_factor <= 0:
        raise ValueError("train.prefetch_factor must be >= 1")
    if config.arithmetic.frequency_total is not None and config.arithmetic.frequency_total <= 0:
        raise ValueError("arithmetic.frequency_total must be > 0 when provided")
    if not (0.0 < config.arithmetic.target_uniform_mass <= 1.0):
        raise ValueError("arithmetic.target_uniform_mass must be in (0.0, 1.0]")

    if config.train.gpu_ids is not None:
        if len(config.train.gpu_ids) == 0:
            raise ValueError("train.gpu_ids cannot be an empty list when provided")
        if any((not isinstance(gpu_id, int) or gpu_id < 0) for gpu_id in config.train.gpu_ids):
            raise ValueError("train.gpu_ids must be a list of non-negative integers")


def _apply_timestamp_to_output_dir(config: Any, args: argparse.Namespace) -> None:
    if not args.timestamp_output:
        return
    if config.train.init_from == "resume" and not config.model.pretrained_weight_path:
        return
    timestamp = datetime.now().strftime(args.timestamp_format)
    config.output.output_dir = f"{config.output.output_dir}_{timestamp}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/eval/compress Megabyte on DNACorpus with config + CLI overrides.",
        epilog=(
            "Examples:\n"
            "  python scripts/run_dna_training.py --config configs/dna_megabyte_quick.json --mode all\n"
            "  python scripts/run_dna_training.py --config configs/dna_megabyte_quick.json --mode train "
            "--batch-size 8 --learning-rate 1e-4 --seq-length 512\n"
            "  torchrun --nproc_per_node=2 scripts/run_dna_training.py --config configs/dna_megabyte_quick.json "
            "--mode train --device cuda --gpus 0 1 --num-workers 8 --prefetch-factor 4 --persistent-workers\n"
            "  python scripts/run_dna_training.py --config configs/dna_megabyte_quick.json "
            "--override train.epochs=2 --override data.species='[\"HoSa\",\"YeMi\"]'"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to experiment JSON config.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["train", "eval", "compress", "all"],
        help="Which stage to run.",
    )
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config before running.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate config, then exit.")
    parser.add_argument(
        "--timestamp-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append timestamp suffix to output_dir (default: enabled). Use --no-timestamp-output to disable.",
    )
    parser.add_argument(
        "--timestamp-format",
        default="%Y%m%d_%H%M%S",
        help="strftime format for output timestamp, default: %%Y%%m%%d_%%H%%M%%S",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Generic override in form section.key=value. Can be repeated.",
    )

    model_group = parser.add_argument_group("model overrides")
    model_group.add_argument(
        "--implementation",
        choices=["megabyte", "megabyte_in_action", "megabyte_in_action_causal_conv", "megabyte_relative"],
    )
    model_group.add_argument("--pretrained-weight-path")
    model_group.add_argument("--patch-size", type=int)
    model_group.add_argument("--seq-length", type=int)
    model_group.add_argument("--global-dim", type=int)
    model_group.add_argument("--local-dim", type=int)
    model_group.add_argument("--global-heads", type=int)
    model_group.add_argument("--global-layers", type=int)
    model_group.add_argument("--local-heads", type=int)
    model_group.add_argument("--local-layers", type=int)
    model_group.add_argument("--attn-dropout", type=float, help="Attention dropout used in both global and local transformers.")
    model_group.add_argument("--ff-dropout", type=float, help="Feed-forward dropout used in both global and local transformers.")
    model_group.add_argument(
        "--flash-attn",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable flash-attn package kernels for Megabyte-in-Action attention. Default is disabled.",
    )
    model_group.add_argument(
        "--input-causal-conv-kernel-size",
        type=int,
        help="Kernel size for the causal Conv1d added before Megabyte-in-Action token embeddings are consumed.",
    )

    data_group = parser.add_argument_group("data overrides")
    data_group.add_argument("--dataset-dir")
    data_group.add_argument("--species", nargs="+", help="Species list, e.g. --species HoSa YeMi")
    data_group.add_argument(
        "--sequence-source-mode",
        choices=["auto", "flat_file", "fasta_dir", "indexed_fasta", "repacked_windows"],
    )
    data_group.add_argument("--fasta-index-dir")
    data_group.add_argument("--source-sampling-weights-json")
    data_group.add_argument("--indexed-eval-samples", type=int)
    data_group.add_argument("--indexed-eval-cache-dir")
    data_group.add_argument("--indexed-eval-cache-mode", choices=["reuse", "refresh", "off"])
    data_group.add_argument("--indexed-eval-random-seed", type=int)
    data_group.add_argument("--indexed-split-seed", type=int)
    data_group.add_argument(
        "--indexed-window-mode",
        choices=["sliding_random", "nonoverlap_random", "nonoverlap_file_stream", "source_batch_file_stream"],
    )
    data_group.add_argument("--indexed-train-epoch-mode", choices=["samples", "all_windows"])
    data_group.add_argument("--indexed-file-stream-windows", type=int)
    data_group.add_argument("--indexed-file-shuffle-buffer-windows", type=int)
    data_group.add_argument("--indexed-file-stream-order-seed", type=int)
    data_group.add_argument(
        "--indexed-source-mix-chunk-batches",
        type=int,
        help="Deprecated compatibility option; source_batch_file_stream now samples sources per sample by probability.",
    )
    data_group.add_argument("--indexed-source-read-chunk-windows", type=int)
    data_group.add_argument(
        "--indexed-source-read-chunk-shuffle",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shuffle each per-source read chunk in memory after sequential disk read.",
    )
    data_group.add_argument("--indexed-source-balance-batches", type=int, help="Legacy alias for --indexed-source-mix-chunk-batches.")
    data_group.add_argument("--indexed-source-read-block-windows", type=int, help="Legacy alias for --indexed-source-read-chunk-windows.")
    data_group.add_argument("--indexed-source-file-order-seed", type=int)
    data_group.add_argument("--repacked-window-dir")
    data_group.add_argument("--repacked-schedule-dir")
    data_group.add_argument("--repacked-eval-samples", type=int)
    data_group.add_argument("--repacked-train-epoch-mode", choices=["samples", "all_windows"])
    data_group.add_argument("--repacked-read-chunk-windows", type=int)
    data_group.add_argument("--repacked-shard-load-mode", choices=["mmap", "preload"])
    data_group.add_argument("--repacked-shard-sampling-mode", choices=["random", "all_shards"])
    data_group.add_argument("--repacked-source-sampling-weights-json")
    data_group.add_argument("--source-loss-weights-json")
    data_group.add_argument("--multi-sequence-mode", choices=["separate", "concat"])
    data_group.add_argument("--clean-cache-enabled", dest="clean_cache_enabled", action="store_true", default=None)
    data_group.add_argument("--no-clean-cache", dest="clean_cache_enabled", action="store_false")
    data_group.add_argument("--clean-cache-dir")
    data_group.add_argument(
        "--sequence-include",
        action="append",
        help="Repeatable sequence selector in form species=key1,key2,...",
    )
    data_group.add_argument("--train-ratio", type=float)
    data_group.add_argument("--val-ratio", type=float)
    data_group.add_argument("--test-ratio", type=float)
    data_group.add_argument("--max-train-bytes", type=int)
    data_group.add_argument("--max-val-bytes", type=int)
    data_group.add_argument("--max-test-bytes", type=int)
    data_group.add_argument("--train-samples-per-epoch", type=int)
    data_group.add_argument("--train-sampling-strategy", choices=["proportional", "uniform", "sqrt"])
    data_group.add_argument("--token-merge-size", type=int, help="Merge this many DNA bases into one token. 1 keeps byte-level tokens.")
    data_group.add_argument("--token-merge-alphabet", help="DNA alphabet used for merged-token encoding, e.g. ACGTN")
    data_group.add_argument("--compression-sample-bytes", type=int)

    train_group = parser.add_argument_group("train overrides")
    train_group.add_argument("--seed", type=int)
    train_group.add_argument("--device", help="auto/cpu/cuda/cuda:0 ...")
    train_group.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    train_group.add_argument("--init-from", choices=["scratch", "pretrained", "resume"])
    train_group.add_argument("--epochs", type=int)
    train_group.add_argument("--batch-size", type=int)
    train_group.add_argument("--eval-batch-size", type=int)
    train_group.add_argument("--learning-rate", type=float)
    train_group.add_argument("--weight-decay", type=float)
    train_group.add_argument("--lr-scheduler", choices=["none", "linear", "cosine"])
    train_group.add_argument("--lr-warmup-steps", type=int)
    train_group.add_argument("--lr-min-ratio", type=float)
    train_group.add_argument("--grad-clip-norm", type=float)
    train_group.add_argument("--num-workers", type=int)
    train_group.add_argument("--prefetch-factor", type=int, help="DataLoader prefetch factor per worker (effective when num_workers > 0).")
    train_group.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader workers alive between epochs (effective when num_workers > 0).",
    )
    train_group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pinned host memory for faster host-to-device transfer on CUDA.",
    )
    train_group.add_argument("--log-interval", type=int)
    train_group.add_argument("--eval-interval", type=int)
    train_group.add_argument(
        "--gpus",
        "--gpu-ids",
        nargs="+",
        help="GPU ids to use, e.g. --gpus 0 1 or --gpus 0,1. For DDP, launch with torchrun and set nproc_per_node to len(gpu_ids).",
    )

    output_group = parser.add_argument_group("output overrides")
    output_group.add_argument("--run-name")
    output_group.add_argument("--output-dir")
    output_group.add_argument(
        "--tracking-backend",
        choices=["swanlab", "wandb", "both"],
        help="Experiment tracking backend. Reuses the wandb_* project/name fields for compatibility.",
    )
    output_group.add_argument("--wandb-project", help="Enable realtime W&B logging and set project name.")
    output_group.add_argument("--wandb-entity", help="Optional W&B entity/team.")
    output_group.add_argument("--wandb-name", help="Optional W&B run name.")
    output_group.add_argument("--wandb-group", help="Optional W&B group.")
    output_group.add_argument("--wandb-tags", nargs="+", help="Optional W&B tags.")
    output_group.add_argument("--wandb-mode", choices=["online", "offline", "disabled"])
    output_group.add_argument(
        "--wandb-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force enable/disable realtime W&B logging.",
    )

    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    seed_config = load_experiment_config(args.config)
    config, resume_checkpoint_path, resume_config_path = _load_resume_default_config(seed_config, args)
    if resume_checkpoint_path is not None:
        print(
            f"[startup] resume checkpoint path: {resume_checkpoint_path} | "
            f"resume config path: {resume_config_path} | resume defaults loaded=True",
            flush=True,
        )
    _apply_overrides(config, args)
    _apply_timestamp_to_output_dir(config, args)
    _validate_config_for_megabyte(config, mode=args.mode)
    apply_token_merge_to_model_config(config.model, config.data)

    if args.print_config or args.dry_run:
        print(json.dumps(config.to_dict(), indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry-run completed: config resolved and validated.")
        return

    print("[startup] importing training runtime...", flush=True)
    from dna_compress.experiment import run_experiment

    metrics = run_experiment(config, mode=args.mode)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
