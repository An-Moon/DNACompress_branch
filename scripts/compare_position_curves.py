from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import autocast_context, resolve_device
from dna_compress.config import ExperimentConfig, load_experiment_config
from dna_compress.data import load_splits
from dna_compress.dnagpt_data import max_target_tokens
from dna_compress.dnagpt_loader import build_dnagpt_components, load_dnagpt_checkpoint
from dna_compress.dnagpt_prefix_coding import (
    build_dnagpt_prefix_trie,
    factorize_dnagpt_log_probs_to_base_prefix_stream,
)
from dna_compress.dnagpt_tokenization import tokenize_dna_source
from dna_compress.fixed_token_factorization import (
    build_fixed_token_arithmetic_factorizer,
    factorize_fixed_token_log_probs,
)
from dna_compress.megabyte_loader import build_model
from dna_compress.tokenization import apply_token_merge_to_model_config, normalize_alphabet, tokenize_source_bytes


def _load_megabyte_model(run_dir: Path, checkpoint_path: Path, device: torch.device) -> tuple[ExperimentConfig, torch.nn.Module]:
    config = load_experiment_config(run_dir / "resolved_config.json")
    apply_token_merge_to_model_config(config.model, config.data)
    model = build_model(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"Megabyte checkpoint is missing model_state: {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return config, model


def _load_dnagpt_model(run_dir: Path, checkpoint_path: Path, device: torch.device):
    config = load_experiment_config(run_dir / "resolved_config.json")
    model, tokenizer, spec = build_dnagpt_components(config.model)
    model_state, metadata, _ = load_dnagpt_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    return config, model, tokenizer, spec, metadata


def _split_payload(config: ExperimentConfig, *, species: str, split: str, length: int) -> tuple[bytes, dict[str, Any]]:
    data_config = config.data
    data_config.species = [species]
    data_config.max_train_bytes_per_species = None
    data_config.max_val_bytes_per_species = None
    data_config.max_test_bytes_per_species = None
    splits = load_splits(data_config, seq_length=config.model.seq_length)
    if split == "train":
        source = splits.train_sources[0]
    elif split == "val":
        source = splits.val_sources[0]
    elif split == "test":
        source = splits.test_sources[0]
    else:
        raise ValueError(f"Unsupported split: {split}")
    if len(source) < length:
        raise ValueError(f"{species} {split} split has only {len(source)} bytes, requested {length}")
    return source[:length], dict(splits.summary["species"][0])


def _token_bases(sequence: bytes, *, token_merge_size: int, alphabet: str) -> list[str]:
    allowed = set(normalize_alphabet(alphabet))
    normalized = [chr(byte).upper() for byte in sequence if chr(byte).upper() in allowed]
    full_count = (len(normalized) // token_merge_size) * token_merge_size
    return normalized[:full_count]


def megabyte_position_bits(
    *,
    model: torch.nn.Module,
    config: ExperimentConfig,
    payload: bytes,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    token_merge_size = config.data.token_merge_size
    symbols = tokenize_source_bytes(payload, token_merge_size, config.data.token_merge_alphabet)
    bases = _token_bases(payload, token_merge_size=token_merge_size, alphabet=config.data.token_merge_alphabet)
    expected_bases = len(symbols) * token_merge_size
    if len(bases) != expected_bases:
        raise RuntimeError(f"Megabyte token/base length mismatch: {len(bases)} != {expected_bases}")

    seq_length = config.model.seq_length
    pad_id = config.model.pad_id
    bits_per_base = [float("nan")] * len(bases)
    window_starts = list(range(0, len(symbols), seq_length)) or [0]
    factorizer = build_fixed_token_arithmetic_factorizer(
        vocab_size=config.model.vocab_size,
        special_token_ids=[config.model.pad_id, config.model.eos_id],
        model_merge_size=token_merge_size,
        arithmetic_merge_size=1,
        alphabet=normalize_alphabet(config.data.token_merge_alphabet),
    ).to(device)

    with torch.no_grad():
        for batch_start in range(0, len(window_starts), batch_size):
            starts = window_starts[batch_start : batch_start + batch_size]
            windows = torch.full((len(starts), seq_length), pad_id, dtype=torch.long)
            lengths: list[int] = []
            for row_index, start in enumerate(starts):
                chunk = symbols[start : start + seq_length]
                lengths.append(len(chunk))
                if chunk:
                    windows[row_index, : len(chunk)] = torch.tensor(chunk, dtype=torch.long)
            batch = windows.to(device, non_blocking=True)
            with autocast_context(device, config.train.dtype):
                output = model(batch, return_loss=False)
                log_probs = torch.log_softmax(output.lm_logits, dim=-1)
            for row_index, (start, chunk_length) in enumerate(zip(starts, lengths)):
                if chunk_length <= 0:
                    continue
                targets = torch.tensor(symbols[start : start + chunk_length], dtype=torch.long, device=device)
                row_log_probs = log_probs[row_index, :chunk_length, :]
                factorized = factorize_fixed_token_log_probs(
                    log_probs=row_log_probs,
                    target_token_ids=targets,
                    factorizer=factorizer,
                )
                root_probs = factorized.root_probabilities
                root_symbols = factorized.root_symbols
                first_base_bits = (
                    -torch.log(root_probs.gather(1, root_symbols.unsqueeze(1)).squeeze(1).clamp_min(1e-30))
                    / math.log(2)
                )
                per_token_base_bits = [first_base_bits]
                for step_probabilities, step_symbols in zip(
                    factorized.regular_step_probabilities,
                    factorized.regular_step_symbols,
                ):
                    per_token_base_bits.append(
                        -torch.log(
                            step_probabilities.gather(1, step_symbols.unsqueeze(1)).squeeze(1).clamp_min(1e-30)
                        )
                        / math.log(2)
                    )
                base_bits_by_token = torch.stack(per_token_base_bits, dim=1).float().cpu().tolist()
                for offset, token_base_bits in enumerate(base_bits_by_token):
                    base_start = (start + offset) * token_merge_size
                    for base_offset in range(token_merge_size):
                        bits_per_base[base_start + base_offset] = float(token_base_bits[base_offset])
    return bits_per_base


def dnagpt_position_bits(
    *,
    model: torch.nn.Module,
    tokenizer,
    spec,
    config: ExperimentConfig,
    payload: bytes,
    species: str,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    tokenized = tokenize_dna_source(
        species=species,
        source=payload,
        tokenizer=tokenizer,
        kmer_size=spec.kmer_size,
        species_prefix_map=config.data.species_prefix_map,
        drop_tail_to_full_kmer=False,
    )
    bits_per_base = [float("nan")] * tokenized.total_bases
    target_capacity = max_target_tokens(config.model.seq_length, len(tokenized.prefix_ids))
    starts = list(range(0, len(tokenized.dna_token_ids), target_capacity)) or [0]
    trie = build_dnagpt_prefix_trie(tokenizer).to(device)

    with torch.no_grad():
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start : batch_start + batch_size]
            inputs = torch.full((len(batch_starts), config.model.seq_length), tokenizer.pad_id, dtype=torch.long)
            lengths: list[int] = []
            base_offsets: list[int] = []
            for row_index, start in enumerate(batch_starts):
                target_length = min(target_capacity, len(tokenized.dna_token_ids) - start)
                lengths.append(target_length)
                base_offsets.append(sum(tokenized.dna_token_base_lengths[:start]))
                context = list(tokenized.prefix_ids)
                context.extend(tokenized.dna_token_ids[start : start + max(target_length - 1, 0)])
                if context:
                    inputs[row_index, 1 : len(context) + 1] = torch.tensor(context, dtype=torch.long)
            batch = inputs.to(device, non_blocking=True)
            with autocast_context(device, config.train.dtype):
                logits = model(batch)
                log_probs = torch.log_softmax(logits, dim=-1)
            prefix_length = len(tokenized.prefix_ids)
            for row_index, (start, target_length, base_start_offset) in enumerate(zip(batch_starts, lengths, base_offsets)):
                if target_length <= 0:
                    continue
                targets = torch.tensor(
                    tokenized.dna_token_ids[start : start + target_length],
                    dtype=torch.long,
                    device=device,
                )
                row_log_probs = log_probs[row_index, prefix_length : prefix_length + target_length, :]
                factorized = factorize_dnagpt_log_probs_to_base_prefix_stream(
                    log_probs=row_log_probs,
                    target_token_ids=targets,
                    trie=trie,
                )
                cursor = base_start_offset
                for offset in range(target_length):
                    base_len = tokenized.dna_token_base_lengths[start + offset]
                    valid_mask = factorized.emitted_valid_mask[offset]
                    emitted_probabilities = factorized.emitted_probabilities[offset][valid_mask]
                    emitted_symbols = factorized.emitted_symbols[offset][valid_mask]
                    base_step_index = 0
                    for probability_row, symbol in zip(emitted_probabilities, emitted_symbols):
                        symbol_value = int(symbol.item())
                        if symbol_value == 0:
                            continue
                        if base_step_index >= base_len:
                            break
                        probability = float(probability_row[symbol_value].clamp_min(1e-30).item())
                        if cursor + base_step_index < len(bits_per_base):
                            bits_per_base[cursor + base_step_index] = -math.log2(probability)
                        base_step_index += 1
                    if base_step_index != base_len:
                        raise RuntimeError(
                            f"DNAGPT base-prefix factorization produced {base_step_index} base steps "
                            f"for a token of length {base_len}."
                        )
                    cursor += base_len
    return bits_per_base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    radius = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        lower = max(0, index - radius)
        upper = min(len(values), index + radius + 1)
        finite = [value for value in values[lower:upper] if not math.isnan(value)]
        smoothed.append(sum(finite) / len(finite) if finite else float("nan"))
    return smoothed


def _write_plots(out_dir: Path, rows: list[dict[str, object]], smooth_window: int) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["position_one_based"]) for row in rows]
    mega_bits = [float(row["megabyte_bits_per_base"]) for row in rows]
    dnagpt_bits = [float(row["dnagpt_bits_per_base"]) for row in rows]
    mega_prob = [float(row["megabyte_true_base_probability"]) for row in rows]
    dnagpt_prob = [float(row["dnagpt_true_base_probability"]) for row in rows]
    delta_bits = [float(row["megabyte_minus_dnagpt_bits_per_base"]) for row in rows]

    paths: list[Path] = []
    fig, axes = plt.subplots(3, 1, figsize=(15, 10.5), sharex=True)
    axes[0].plot(x, mega_bits, color="#1f77b4", alpha=0.28, linewidth=0.7, label="MEGABYTE raw")
    axes[0].plot(x, dnagpt_bits, color="#ff7f0e", alpha=0.28, linewidth=0.7, label="DNA-GPT raw")
    axes[0].plot(x, _moving_average(mega_bits, smooth_window), color="#1f77b4", linewidth=1.8, label=f"MEGABYTE {smooth_window}-bp MA")
    axes[0].plot(x, _moving_average(dnagpt_bits, smooth_window), color="#ff7f0e", linewidth=1.8, label=f"DNA-GPT {smooth_window}-bp MA")
    axes[0].set_ylabel("Bits/base")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(x, _moving_average(mega_prob, smooth_window), color="#1f77b4", linewidth=1.6, label="MEGABYTE")
    axes[1].plot(x, _moving_average(dnagpt_prob, smooth_window), color="#ff7f0e", linewidth=1.6, label="DNA-GPT")
    axes[1].set_ylabel("True-base probability")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].axhline(0.0, color="#333333", linewidth=0.8)
    axes[2].plot(x, _moving_average(delta_bits, smooth_window), color="#2ca02c", linewidth=1.5)
    axes[2].set_ylabel("MEGABYTE - DNA-GPT bits/base")
    axes[2].set_xlabel("DaRe val position, 1-based")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("DaRe val first 6000 bases: position probability curves")
    fig.tight_layout()
    path = out_dir / "dare_val_first6000_position_curves.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths.append(path)

    fig, axis = plt.subplots(figsize=(15, 4.6))
    axis.axhline(0.0, color="#333333", linewidth=0.8)
    axis.fill_between(x, _moving_average(delta_bits, smooth_window), 0, color="#2ca02c", alpha=0.25)
    axis.plot(x, _moving_average(delta_bits, smooth_window), color="#2ca02c", linewidth=1.4)
    axis.set_xlabel("DaRe val position, 1-based")
    axis.set_ylabel("MEGABYTE - DNA-GPT bits/base")
    axis.set_title("Negative values favor MEGABYTE; positive values favor DNA-GPT")
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    delta_path = out_dir / "dare_val_first6000_delta_bits.png"
    fig.savefig(delta_path, dpi=170)
    plt.close(fig)
    paths.append(delta_path)
    return paths


def _summary(rows: list[dict[str, object]], *, smooth_window: int) -> dict[str, object]:
    mega = [float(row["megabyte_bits_per_base"]) for row in rows]
    dnagpt = [float(row["dnagpt_bits_per_base"]) for row in rows]
    delta = [m - d for m, d in zip(mega, dnagpt)]
    mega_wins = sum(1 for value in delta if value < 0)
    dnagpt_wins = sum(1 for value in delta if value > 0)
    ties = len(delta) - mega_wins - dnagpt_wins

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def corr(a: list[float], b: list[float]) -> float:
        mean_a = mean(a)
        mean_b = mean(b)
        numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
        denom_a = math.sqrt(sum((x - mean_a) ** 2 for x in a))
        denom_b = math.sqrt(sum((y - mean_b) ** 2 for y in b))
        return numerator / (denom_a * denom_b) if denom_a > 0 and denom_b > 0 else float("nan")

    smooth_delta = _moving_average(delta, smooth_window)
    segments: list[dict[str, object]] = []
    current_sign = None
    current_start = 0
    for index, value in enumerate(smooth_delta):
        sign = "megabyte" if value < 0 else "dnagpt" if value > 0 else "tie"
        if current_sign is None:
            current_sign = sign
            current_start = index
        elif sign != current_sign:
            segments.append({"winner": current_sign, "start": current_start + 1, "end": index})
            current_sign = sign
            current_start = index
    if current_sign is not None:
        segments.append({"winner": current_sign, "start": current_start + 1, "end": len(smooth_delta)})
    long_segments = [
        segment for segment in segments
        if int(segment["end"]) - int(segment["start"]) + 1 >= smooth_window
    ]
    return {
        "position_count": len(rows),
        "smooth_window": smooth_window,
        "megabyte_mean_bits_per_base": mean(mega),
        "dnagpt_mean_bits_per_base": mean(dnagpt),
        "megabyte_minus_dnagpt_mean_bits_per_base": mean(delta),
        "megabyte_win_positions": mega_wins,
        "dnagpt_win_positions": dnagpt_wins,
        "tie_positions": ties,
        "megabyte_win_fraction": mega_wins / len(rows),
        "dnagpt_win_fraction": dnagpt_wins / len(rows),
        "bits_curve_correlation": corr(mega, dnagpt),
        "long_smoothed_winner_segments": long_segments[:20],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare local position probability curves for two DNA models.")
    parser.add_argument("--megabyte-run-dir", default="outputs/dna_megabyte_large_b128_ensembl_all_finetune")
    parser.add_argument("--dnagpt-run-dir", default="outputs/dna_dnagpt_0p1bm_all_finetune")
    parser.add_argument("--megabyte-checkpoint", default=None)
    parser.add_argument("--dnagpt-checkpoint", default=None)
    parser.add_argument("--dnagpt-checkpoint-fallback", default="outputs/dna_dnagpt_0p1bm_all_finetuned_1/last.pt")
    parser.add_argument("--species", default="DaRe")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--length", type=int, default=6000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--smooth-window", type=int, default=101)
    parser.add_argument("--out-dir", default="outputs/model_fusion_position_curves/dare_val_first6000")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    megabyte_run_dir = Path(args.megabyte_run_dir)
    dnagpt_run_dir = Path(args.dnagpt_run_dir)
    megabyte_checkpoint = Path(args.megabyte_checkpoint) if args.megabyte_checkpoint else megabyte_run_dir / "best.pt"
    dnagpt_checkpoint = Path(args.dnagpt_checkpoint) if args.dnagpt_checkpoint else dnagpt_run_dir / "best.pt"
    dnagpt_checkpoint_note = "requested"
    if not dnagpt_checkpoint.exists() and args.dnagpt_checkpoint_fallback:
        fallback = Path(args.dnagpt_checkpoint_fallback)
        if fallback.exists():
            dnagpt_checkpoint_note = f"fallback_for_missing_requested_checkpoint:{dnagpt_checkpoint}"
            dnagpt_checkpoint = fallback
    if not megabyte_checkpoint.exists():
        raise FileNotFoundError(f"Megabyte checkpoint not found: {megabyte_checkpoint}")
    if not dnagpt_checkpoint.exists():
        raise FileNotFoundError(f"DNAGPT checkpoint not found: {dnagpt_checkpoint}")

    megabyte_config = load_experiment_config(megabyte_run_dir / "resolved_config.json")
    payload, source_metadata = _split_payload(megabyte_config, species=args.species, split=args.split, length=args.length)
    sequence = payload.decode("ascii").upper()
    (out_dir / "dare_val_first6000_sequence.txt").write_text(sequence, encoding="ascii")

    print(f"[position] device={device} species={args.species} split={args.split} length={len(payload)}")
    print(f"[position] loading MEGABYTE from {megabyte_checkpoint}", flush=True)
    megabyte_config, megabyte_model = _load_megabyte_model(megabyte_run_dir, megabyte_checkpoint, device)
    megabyte_bits = megabyte_position_bits(
        model=megabyte_model,
        config=megabyte_config,
        payload=payload,
        device=device,
        batch_size=args.batch_size,
    )
    del megabyte_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[position] loading DNA-GPT from {dnagpt_checkpoint}", flush=True)
    dnagpt_config, dnagpt_model, tokenizer, spec, dnagpt_metadata = _load_dnagpt_model(dnagpt_run_dir, dnagpt_checkpoint, device)
    dnagpt_bits = dnagpt_position_bits(
        model=dnagpt_model,
        tokenizer=tokenizer,
        spec=spec,
        config=dnagpt_config,
        payload=payload,
        species=args.species,
        device=device,
        batch_size=args.batch_size,
    )

    if len(megabyte_bits) != args.length or len(dnagpt_bits) != args.length:
        raise RuntimeError(f"Expected {args.length} aligned bases, got {len(megabyte_bits)} and {len(dnagpt_bits)}")

    rows: list[dict[str, object]] = []
    for index, (base, mega_bit, dnagpt_bit) in enumerate(zip(sequence, megabyte_bits, dnagpt_bits), start=1):
        rows.append(
            {
                "position_one_based": index,
                "position_zero_based": index - 1,
                "base": base,
                "megabyte_bits_per_base": mega_bit,
                "dnagpt_bits_per_base": dnagpt_bit,
                "megabyte_true_base_probability": 2.0 ** (-mega_bit),
                "dnagpt_true_base_probability": 2.0 ** (-dnagpt_bit),
                "megabyte_minus_dnagpt_bits_per_base": mega_bit - dnagpt_bit,
                "winner": "megabyte" if mega_bit < dnagpt_bit else "dnagpt" if dnagpt_bit < mega_bit else "tie",
            }
        )

    csv_path = out_dir / "dare_val_first6000_position_curves.csv"
    _write_csv(csv_path, rows)
    plot_paths = _write_plots(out_dir, rows, args.smooth_window)
    summary = _summary(rows, smooth_window=args.smooth_window)
    metadata = {
        "species": args.species,
        "split": args.split,
        "length": args.length,
        "source_metadata": source_metadata,
        "device": str(device),
        "megabyte_run_dir": str(megabyte_run_dir),
        "megabyte_checkpoint": str(megabyte_checkpoint),
        "dnagpt_run_dir": str(dnagpt_run_dir),
        "dnagpt_checkpoint": str(dnagpt_checkpoint),
        "dnagpt_checkpoint_note": dnagpt_checkpoint_note,
        "dnagpt_checkpoint_metadata": dnagpt_metadata,
        "alignment": (
            "per-base bits are from hierarchical prefix factorization; Megabyte 3-mer tokens are "
            "factorized into one-base conditional steps, and DNA-GPT dynamic k-mers are factorized "
            "through the tokenizer prefix trie with token-stop symbols omitted from base positions"
        ),
        "outputs": {
            "sequence": str(out_dir / "dare_val_first6000_sequence.txt"),
            "csv": str(csv_path),
            "plots": [str(path) for path in plot_paths],
        },
        "summary": summary,
    }
    summary_path = out_dir / "dare_val_first6000_position_curves_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[position] wrote {csv_path}")
    for path in plot_paths:
        print(f"[position] wrote {path}")
    print(f"[position] wrote {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
