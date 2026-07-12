#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark true encoding speed of nc_prefix against GeCo2 on the same region.

This script intentionally does not draw plots or write per-base artifacts.  It
extracts one ACGT region, runs nc_prefix probability modelling plus arithmetic
encoding in memory, then runs the GeCo2 binary on the same flat sequence file.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.fast_arithmetic import load_fast_arithmetic_extension  # noqa: E402
from dna_compress.fast_nc_prefix import load_fast_nc_prefix_extension  # noqa: E402
from dna_compress.noncontiguous_prefix_codec import (  # noqa: E402
    NoncontiguousPrefixConfig,
    compress_noncontiguous_prefix_sequence,
)
from scripts.run_dna_region_bpb_probe import (  # noqa: E402
    extract_filtered_region,
    filtered_length,
    resolve_region_sources,
    stable_random_start,
)


DEFAULT_WINDOW_BASES = 3072
DEFAULT_WINDOW_COUNT = 8192
DEFAULT_REGION_BASES = DEFAULT_WINDOW_BASES * DEFAULT_WINDOW_COUNT


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_geco2_binary(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        candidate = Path(binary)
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"Could not find GeCo2 binary: {binary}")
    return resolved


def _tail(text: str, max_chars: int = 4000) -> str:
    return text if len(text) <= max_chars else text[-max_chars:]


def _run_geco2(
    *,
    binary: str,
    input_path: Path,
    level: int,
    base_count: int,
) -> dict[str, Any]:
    compressed_path = Path(str(input_path) + ".co")
    if compressed_path.exists():
        compressed_path.unlink()
    command = [binary, "-F", "-v", "-l", str(level), str(input_path)]
    started = perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    seconds = perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"GeCo2 failed with return code {completed.returncode}: {_tail(completed.stderr or completed.stdout)}")
    if not compressed_path.exists():
        candidates = sorted(input_path.parent.glob("*.co"))
        if len(candidates) == 1:
            compressed_path = candidates[0]
        else:
            raise FileNotFoundError(f"GeCo2 did not create expected output: {compressed_path}")
    compressed_bytes = int(compressed_path.stat().st_size)
    return {
        "codec": "geco2",
        "command": command,
        "returncode": int(completed.returncode),
        "seconds": seconds,
        "bases_per_second": base_count / max(seconds, 1e-12),
        "compressed_bytes": compressed_bytes,
        "bits_per_base": (compressed_bytes * 8.0) / max(base_count, 1),
        "compressed_path": str(compressed_path),
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _run_nc_prefix(
    *,
    sequence: str,
    window_bases: int,
    min_windows: int,
    hash_bucket_count: int,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
) -> dict[str, Any]:
    result = compress_noncontiguous_prefix_sequence(
        sequence,
        NoncontiguousPrefixConfig(
            window_bases=window_bases,
            alphabet="ACGT",
            backend="fast_cpp",
            min_windows=min_windows,
            hash_bucket_count=hash_bucket_count,
        ),
        arithmetic_frequency_total=arithmetic_frequency_total,
        arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
        encode_arithmetic=True,
    )
    metadata = dict(result["model_metadata"])
    return {
        "codec": "nc_prefix",
        "seconds": float(result["compression_process_seconds"]),
        "bases_per_second": float(result["compression_bases_per_second"]),
        "sample_bases": int(result["sample_bases"]),
        "theoretical_bits_per_base": float(result["theoretical_bits_per_base"]),
        "arithmetic_bits_per_base": float(result["arithmetic_bits_per_base"]),
        "arithmetic_coded_bytes": int(result["arithmetic_coded_bytes"]),
        "model_compute_seconds": float(metadata["compute_seconds"]),
        "arithmetic_quantize_seconds": float(result["arithmetic_quantize_seconds"]),
        "arithmetic_range_seconds": float(result["arithmetic_range_seconds"]),
        "arithmetic_encode_seconds": float(result["arithmetic_encode_seconds"]),
        "window_count": int(metadata["window_count"]),
        "window_bases": int(window_bases),
        "update_mode": str(metadata.get("update_mode", "cache_pipeline")),
        "threads": int(metadata.get("threads_requested", 0)),
        "hash_bucket_count_requested": int(metadata.get("hash_bucket_count_requested", hash_bucket_count)),
        "hash_bucket_count_effective": int(metadata.get("hash_bucket_count", 0)),
        "model_metadata": metadata,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark nc_prefix true encoding speed against GeCo2.")
    parser.add_argument("--dataset", choices=("dnacorpus",), default="dnacorpus")
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--species", default="OrSa")
    parser.add_argument("--region-start", type=int)
    parser.add_argument("--random-region", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--region-bases", type=int, default=DEFAULT_REGION_BASES)
    parser.add_argument("--window-bases", type=int, default=DEFAULT_WINDOW_BASES)
    parser.add_argument("--min-windows", type=int, default=DEFAULT_WINDOW_COUNT)
    parser.add_argument("--nc-prefix-hash-bucket-count", type=int, default=0, help="ctx17 hash bucket count. Use 0 for the GECO2 default.")
    parser.add_argument("--geco2-bin", default="GeCo2")
    parser.add_argument("--geco2-level", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arithmetic-frequency-total", type=int)
    parser.add_argument("--arithmetic-target-uniform-mass", type=float, default=0.01)
    parser.add_argument("--keep-sample", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geco2_binary = _resolve_geco2_binary(args.geco2_bin)

    source_args = argparse.Namespace(
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        input_dir=None,
        source=None,
        species=[args.species],
        sequence_source_mode="auto",
        multi_sequence_mode="separate",
        sequence_include=None,
    )
    source_info = resolve_region_sources(source_args, "ACGT")[0]
    length_started = perf_counter()
    source_length = filtered_length(source_info["paths"], alphabet="ACGT", fasta=bool(source_info["fasta"]))
    source_length_seconds = perf_counter() - length_started
    requested_bases = min(int(args.region_bases), int(source_length))
    if args.region_start is not None:
        region_start = int(args.region_start)
    elif args.random_region and source_length > requested_bases:
        region_start = stable_random_start(
            source_name=str(args.species),
            source_length=int(source_length),
            region_bases=int(requested_bases),
            seed=int(args.seed),
        )
    else:
        region_start = 0
    actual_bases = min(requested_bases, int(source_length) - int(region_start))
    if actual_bases <= 0:
        raise ValueError("selected region is empty")

    read_started = perf_counter()
    region_bytes = extract_filtered_region(
        source_info["paths"],
        alphabet="ACGT",
        fasta=bool(source_info["fasta"]),
        start=region_start,
        length=actual_bases,
    )
    region_read_seconds = perf_counter() - read_started
    if len(region_bytes) != actual_bases:
        raise RuntimeError(f"expected {actual_bases} ACGT bases, got {len(region_bytes)}")
    sequence = region_bytes.decode("ascii")

    sample_path = output_dir / f"{args.species}_{region_start}_{actual_bases}.seq"
    sample_path.write_bytes(region_bytes)

    preload_started = perf_counter()
    load_fast_nc_prefix_extension()
    load_fast_arithmetic_extension()
    preload_seconds = perf_counter() - preload_started

    rows: list[dict[str, Any]] = []
    nc_results: list[dict[str, Any]] = []
    geco2_results: list[dict[str, Any]] = []
    for repeat_index in range(int(args.repeat)):
        nc = _run_nc_prefix(
            sequence=sequence,
            window_bases=int(args.window_bases),
            min_windows=int(args.min_windows),
            hash_bucket_count=int(args.nc_prefix_hash_bucket_count),
            arithmetic_frequency_total=args.arithmetic_frequency_total,
            arithmetic_target_uniform_mass=float(args.arithmetic_target_uniform_mass),
        )
        nc_results.append(nc)
        rows.append(
            {
                "repeat": repeat_index,
                "codec": "nc_prefix",
                "seconds": nc["seconds"],
                "bases_per_second": nc["bases_per_second"],
                "mbases_per_second": nc["bases_per_second"] / 1_000_000.0,
                "bits_per_base": nc["arithmetic_bits_per_base"],
                "theoretical_bits_per_base": nc["theoretical_bits_per_base"],
                "model_compute_seconds": nc["model_compute_seconds"],
                "arithmetic_encode_seconds": nc["arithmetic_encode_seconds"],
                "arithmetic_quantize_seconds": nc["arithmetic_quantize_seconds"],
                "arithmetic_range_seconds": nc["arithmetic_range_seconds"],
                "update_mode": nc["update_mode"],
                "threads": nc["threads"],
                "hash_bucket_count_requested": nc["hash_bucket_count_requested"],
                "hash_bucket_count_effective": nc["hash_bucket_count_effective"],
            }
        )

        geco2 = _run_geco2(
            binary=geco2_binary,
            input_path=sample_path,
            level=int(args.geco2_level),
            base_count=int(actual_bases),
        )
        geco2_results.append(geco2)
        rows.append(
            {
                "repeat": repeat_index,
                "codec": "geco2",
                "seconds": geco2["seconds"],
                "bases_per_second": geco2["bases_per_second"],
                "mbases_per_second": geco2["bases_per_second"] / 1_000_000.0,
                "bits_per_base": geco2["bits_per_base"],
                "geco2_level": int(args.geco2_level),
                "compressed_bytes": geco2["compressed_bytes"],
            }
        )

    _write_csv(output_dir / "speed_rows.csv", rows)

    def aggregate(codec: str) -> dict[str, Any]:
        codec_rows = [row for row in rows if row["codec"] == codec]
        return {
            "codec": codec,
            "repeat_count": len(codec_rows),
            "mean_seconds": sum(float(row["seconds"]) for row in codec_rows) / max(len(codec_rows), 1),
            "mean_bases_per_second": sum(float(row["bases_per_second"]) for row in codec_rows) / max(len(codec_rows), 1),
            "mean_mbases_per_second": sum(float(row["mbases_per_second"]) for row in codec_rows) / max(len(codec_rows), 1),
            "mean_bits_per_base": sum(float(row["bits_per_base"]) for row in codec_rows) / max(len(codec_rows), 1),
            "best_bases_per_second": max((float(row["bases_per_second"]) for row in codec_rows), default=0.0),
        }

    summary = {
        "dataset": args.dataset,
        "species": args.species,
        "source_length": int(source_length),
        "source_length_seconds": source_length_seconds,
        "region_start": int(region_start),
        "region_bases": int(actual_bases),
        "region_read_seconds": region_read_seconds,
        "window_bases": int(args.window_bases),
        "min_windows": int(args.min_windows),
        "nc_prefix_update_mode": "cache_pipeline",
        "nc_prefix_threads": 0,
        "nc_prefix_hash_bucket_count": int(args.nc_prefix_hash_bucket_count),
        "geco2_binary": geco2_binary,
        "geco2_level": int(args.geco2_level),
        "repeat": int(args.repeat),
        "jit_preload_seconds": preload_seconds,
        "sample_path": str(sample_path),
        "sample_kept": bool(args.keep_sample),
        "aggregates": {
            "nc_prefix": aggregate("nc_prefix"),
            "geco2": aggregate("geco2"),
        },
        "rows_csv": str(output_dir / "speed_rows.csv"),
        "nc_prefix_last_model_metadata": nc_results[-1]["model_metadata"] if nc_results else None,
        "geco2_last_stdout_tail": geco2_results[-1]["stdout_tail"] if geco2_results else None,
        "geco2_last_stderr_tail": geco2_results[-1]["stderr_tail"] if geco2_results else None,
    }
    _write_json(output_dir / "speed_summary.json", summary)
    print(json.dumps(_json_safe(summary["aggregates"]), ensure_ascii=False, indent=2))
    print(f"summary_json={output_dir / 'speed_summary.json'}")
    print(f"rows_csv={output_dir / 'speed_rows.csv'}")

    if not args.keep_sample:
        sample_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
