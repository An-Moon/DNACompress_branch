#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate synchronous MEGABYTE LM + nc_prefix fused arithmetic compression."""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from dna_compress.experiment import resolve_device  # noqa: E402
from dna_compress.fused_lm_nc_prefix_codec import (  # noqa: E402
    compress_fused_lm_nc_prefix_payload,
    load_megabyte_model_for_fusion,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a synchronous LM x nc_prefix fused compression evaluation on one sequence file."
    )
    parser.add_argument("--run-dir", required=True, help="MEGABYTE training output directory with resolved_config.json.")
    parser.add_argument("--checkpoint", help="Checkpoint path. Defaults to <run-dir>/<checkpoint-tag>.pt.")
    parser.add_argument("--checkpoint-tag", default="best", help="Checkpoint basename under --run-dir when --checkpoint is not set.")
    parser.add_argument("--source-file", required=True, help="Input FASTA or raw sequence file.")
    parser.add_argument("--source-format", choices=("auto", "fasta", "raw"), default="auto")
    parser.add_argument("--max-bases", type=int, help="Optional cap after ACGT filtering, useful for quick tests.")
    parser.add_argument("--output-json", help="Output metrics JSON path.")
    parser.add_argument("--output-dir", default="outputs/fused_lm_nc_prefix")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", help="Inference dtype. Defaults to config.train.dtype.")
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Window batch size. token streaming default 'auto' means all windows in one batch.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("streaming_token_encode_overlap", "streaming_token_strict"),
        default="streaming_token_encode_overlap",
    )
    parser.add_argument("--nc-prefix-window-bases", type=int)
    parser.add_argument("--nc-prefix-min-windows", type=int, default=8192)
    parser.add_argument("--nc-prefix-hash-bucket-count", type=int, default=0)
    parser.add_argument("--fusion-eta", type=float, default=0.05)
    parser.add_argument("--fusion-initial-lm-weight", type=float, default=0.5)
    parser.add_argument("--arithmetic-frequency-total", type=int)
    parser.add_argument("--arithmetic-target-uniform-mass", type=float, default=0.01)
    parser.add_argument("--skip-arithmetic", action="store_true")
    parser.add_argument(
        "--fast-runtime",
        action="store_true",
        help="Disable diagnostic bpb accumulation and codec baselines for fastest runtime measurement.",
    )
    parser.add_argument("--skip-codec-baselines", action="store_true")
    return parser


def _is_probably_fasta(data: bytes) -> bool:
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        return line.startswith(b">")
    return False


def _read_payload(path: Path, source_format: str, max_bases: int | None) -> bytes:
    data = path.read_bytes()
    use_fasta = source_format == "fasta" or (source_format == "auto" and _is_probably_fasta(data))
    if use_fasta:
        lines = []
        for raw_line in data.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(b">"):
                continue
            lines.append(line)
        data = b"".join(lines)
    if max_bases is not None and int(max_bases) > 0:
        kept = []
        count = 0
        for byte_value in data.upper():
            if byte_value in b"ACGT":
                kept.append(byte_value)
                count += 1
                if count >= int(max_bases):
                    break
        data = bytes(kept)
    return data


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_outputs(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    row = {
        "codec": metrics.get("codec"),
        "source_file": metrics.get("source_file"),
        "checkpoint": metrics.get("checkpoint"),
        "sample_bases": metrics.get("sample_bases"),
        "core_base_count": metrics.get("core_base_count"),
        "window_count": metrics.get("window_count"),
        "window_bases": metrics.get("window_bases"),
        "token_merge_size": metrics.get("token_merge_size"),
        "theoretical_bits_per_base": metrics.get("theoretical_bits_per_base"),
        "core_theoretical_bits_per_base": metrics.get("core_theoretical_bits_per_base"),
        "lm_only_theoretical_bits_per_base": metrics.get("lm_only_theoretical_bits_per_base"),
        "nc_prefix_only_theoretical_bits_per_base": metrics.get("nc_prefix_only_theoretical_bits_per_base"),
        "arithmetic_bits_per_base": metrics.get("arithmetic_bits_per_base"),
        "compression_bases_per_second": metrics.get("compression_bases_per_second"),
        "compression_process_seconds": metrics.get("compression_process_seconds"),
        "compression_core_seconds": metrics.get("compression_core_seconds"),
        "nc_prefix_prepare_seconds": metrics.get("nc_prefix_prepare_seconds"),
        "model_seconds": metrics.get("model_seconds"),
        "fusion_final_mean_lm_weight": metrics.get("fusion_final_mean_lm_weight"),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / f"{args.checkpoint_tag}.pt"
    source_file = Path(args.source_file)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else Path(args.output_dir) / f"{source_file.stem}_fused_lm_nc_prefix.json"
    )

    device = resolve_device(str(args.device))
    config, model, checkpoint_metadata = load_megabyte_model_for_fusion(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        device=device,
    )
    dtype_name = str(args.dtype or config.train.dtype)
    batch_size = "auto" if str(args.batch_size) == "auto" else int(args.batch_size)
    payload = _read_payload(source_file, str(args.source_format), args.max_bases)

    metrics = compress_fused_lm_nc_prefix_payload(
        model=model,
        config=config,
        payload=payload,
        device=device,
        dtype_name=dtype_name,
        batch_size=batch_size,
        nc_prefix_window_bases=args.nc_prefix_window_bases,
        nc_prefix_min_windows=int(args.nc_prefix_min_windows),
        nc_prefix_hash_bucket_count=int(args.nc_prefix_hash_bucket_count),
        fusion_eta=float(args.fusion_eta),
        fusion_initial_lm_weight=float(args.fusion_initial_lm_weight),
        arithmetic_frequency_total=args.arithmetic_frequency_total,
        arithmetic_target_uniform_mass=float(args.arithmetic_target_uniform_mass),
        encode_arithmetic=not bool(args.skip_arithmetic),
        pipeline_mode=str(args.pipeline_mode),
        collect_diagnostics=not bool(args.fast_runtime),
        include_codec_baselines=not (bool(args.fast_runtime) or bool(args.skip_codec_baselines)),
    )
    metrics.update(
        {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_metadata": checkpoint_metadata,
            "source_file": str(source_file),
            "source_format": str(args.source_format),
            "device": str(device),
            "dtype": dtype_name,
            "batch_size": batch_size,
            "pipeline_mode": str(args.pipeline_mode),
            "max_bases": args.max_bases,
            "fast_runtime": bool(args.fast_runtime),
            "skip_codec_baselines": bool(args.skip_codec_baselines),
        }
    )
    _write_outputs(output_json, metrics)
    print(json.dumps(_json_safe({"output_json": str(output_json), **metrics}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
