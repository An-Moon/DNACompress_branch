#!/usr/bin/env python3
from __future__ import annotations

"""Offline arithmetic-coding evaluation for the nc_prefix statistical codec."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.noncontiguous_prefix_codec import (  # noqa: E402
    DEFAULT_NC_PREFIX_MIN_WINDOWS,
    NoncontiguousPrefixConfig,
    compress_noncontiguous_prefix_sequence,
)
from dna_compress.tokenization import normalize_alphabet  # noqa: E402
from scripts.run_dna_region_bpb_probe import (  # noqa: E402
    DEFAULT_REGION_BASES,
    extract_filtered_region,
    filtered_length,
    resolve_region_sources,
    stable_random_start,
    write_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate nc_prefix lockstep-window statistical compression.")
    parser.add_argument("--dataset", choices=("dnacorpus", "opengenome2"), required=True)
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--input-dir", help="OpenGenome2 FASTA subset directory. Defaults to --dataset-dir.")
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--species", nargs="+")
    parser.add_argument("--sequence-source-mode", choices=("auto", "flat_file", "fasta_dir"), default="auto")
    parser.add_argument("--multi-sequence-mode", choices=("separate", "concat"), default="separate")
    parser.add_argument("--sequence-include", action="append")
    parser.add_argument("--alphabet", default="ACGTN")
    parser.add_argument("--region-start", type=int)
    parser.add_argument("--region-bases", type=int, default=DEFAULT_REGION_BASES)
    parser.add_argument("--random-region", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nc-prefix-window-bases", type=int, default=3072)
    parser.add_argument("--nc-prefix-backend", choices=("auto", "fast_cpp"), default="auto")
    parser.add_argument("--backend", dest="nc_prefix_backend", choices=("auto", "fast_cpp"), help=argparse.SUPPRESS)
    parser.add_argument("--nc-prefix-min-windows", type=int, default=DEFAULT_NC_PREFIX_MIN_WINDOWS)
    parser.add_argument("--arithmetic-frequency-total", type=int)
    parser.add_argument("--arithmetic-target-uniform-mass", type=float, default=0.01)
    parser.add_argument("--skip-arithmetic", action="store_true", help="Only evaluate model probabilities/bpb; do not emit the arithmetic-coded byte stream.")
    return parser


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


def _selected_region(args: argparse.Namespace, source_info: dict[str, Any]) -> tuple[str, int, int, int | None, float, float]:
    source_length: int | None = None
    length_seconds = 0.0
    requested_bases = int(args.region_bases)
    needs_source_length = bool(args.random_region or requested_bases <= 0)
    if needs_source_length:
        started = perf_counter()
        source_length = filtered_length(
            source_info["paths"],
            alphabet=str(source_info["alphabet"]),
            fasta=bool(source_info["fasta"]),
        )
        length_seconds = perf_counter() - started
        if requested_bases <= 0 or requested_bases > source_length:
            requested_bases = source_length
    if args.random_region:
        if source_length is None:
            raise RuntimeError("source length should have been computed for random region selection")
        region_start = stable_random_start(
            source_name=str(source_info["source"]),
            source_length=source_length,
            region_bases=requested_bases,
            seed=int(args.seed),
        )
    elif args.region_start is not None:
        region_start = int(args.region_start)
    else:
        region_start = 0
    if source_length is not None and region_start > source_length:
        raise ValueError(f"region start {region_start} exceeds source length {source_length}")
    actual_bases = min(requested_bases, source_length - region_start) if source_length is not None else requested_bases
    read_started = perf_counter()
    region_bytes = extract_filtered_region(
        source_info["paths"],
        alphabet=str(source_info["alphabet"]),
        fasta=bool(source_info["fasta"]),
        start=region_start,
        length=actual_bases,
    )
    read_seconds = perf_counter() - read_started
    return region_bytes.decode("ascii"), int(region_start), int(actual_bases), source_length, length_seconds, read_seconds


def main() -> None:
    args = _build_parser().parse_args()
    alphabet = normalize_alphabet(args.alphabet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = NoncontiguousPrefixConfig(
        window_bases=int(args.nc_prefix_window_bases),
        alphabet=alphabet,
        backend=str(args.nc_prefix_backend),
        min_windows=int(args.nc_prefix_min_windows),
    )
    parameters = {
        "dataset": args.dataset,
        "dataset_dir": args.dataset_dir,
        "input_dir": args.input_dir,
        "source": args.source,
        "species": args.species,
        "alphabet": alphabet,
        "region_start": args.region_start,
        "region_bases": int(args.region_bases),
        "random_region": bool(args.random_region),
        "seed": int(args.seed),
        "nc_prefix": {
            "window_bases": int(args.nc_prefix_window_bases),
            "backend": str(args.nc_prefix_backend),
            "min_windows": int(args.nc_prefix_min_windows),
            "min_required_bases": int(args.nc_prefix_min_windows) * int(args.nc_prefix_window_bases),
            "algorithm": "geco2_level10_per_window_weights",
        },
        "arithmetic_frequency_total": args.arithmetic_frequency_total,
        "arithmetic_target_uniform_mass": float(args.arithmetic_target_uniform_mass),
        "encode_arithmetic": not bool(args.skip_arithmetic),
    }
    _write_json(output_dir / "parameters.json", parameters)

    rows: list[dict[str, Any]] = []
    source_infos = resolve_region_sources(args, alphabet)
    total_started = perf_counter()
    for source_info in source_infos:
        region_sequence, region_start, actual_bases, source_length, length_seconds, read_seconds = _selected_region(args, source_info)
        compression = compress_noncontiguous_prefix_sequence(
            region_sequence,
            config,
            arithmetic_frequency_total=args.arithmetic_frequency_total,
            arithmetic_target_uniform_mass=float(args.arithmetic_target_uniform_mass),
            encode_arithmetic=not bool(args.skip_arithmetic),
        )
        row = {
            "dataset": source_info["dataset"],
            "source": source_info["source"],
            "species": source_info.get("species"),
            "region_start": region_start,
            "region_bases": len(region_sequence),
            "filtered_source_bases": source_length,
            "filtered_source_bases_known": source_length is not None,
            "theoretical_bits_per_base": compression["theoretical_bits_per_base"],
            "arithmetic_bits_per_base": compression["arithmetic_bits_per_base"],
            "arithmetic_coded_bytes": compression["arithmetic_coded_bytes"],
            "arithmetic_backend": compression["arithmetic_backend"],
            "arithmetic_quantize_seconds": compression["arithmetic_quantize_seconds"],
            "arithmetic_range_seconds": compression["arithmetic_range_seconds"],
            "arithmetic_interval_transfer_seconds": compression["arithmetic_interval_transfer_seconds"],
            "arithmetic_encode_seconds": compression["arithmetic_encode_seconds"],
            "compression_bases_per_second": compression["compression_bases_per_second"],
            "source_length_seconds": length_seconds,
            "region_read_seconds": read_seconds,
            "compression_process_seconds": compression["compression_process_seconds"],
            "window_bases": int(args.nc_prefix_window_bases),
            "backend": str(args.nc_prefix_backend),
            "min_windows": int(args.nc_prefix_min_windows),
            "min_required_bases": int(args.nc_prefix_min_windows) * int(args.nc_prefix_window_bases),
            "window_count": int(compression["model_metadata"]["window_count"]),
            "algorithm": "geco2_level10_per_window_weights",
            "encode_arithmetic": not bool(args.skip_arithmetic),
        }
        rows.append(row)
        model_metadata_path = output_dir / f"{str(source_info['source']).replace('/', '_')}_model_metadata.json"
        _write_json(model_metadata_path, dict(compression["model_metadata"]))

    per_source_csv = output_dir / "nc_prefix_per_source.csv"
    write_csv(per_source_csv, rows)
    total_bases = sum(int(row["region_bases"]) for row in rows)
    total_theoretical_bits = sum(float(row["theoretical_bits_per_base"]) * int(row["region_bases"]) for row in rows)
    arithmetic_rows = [row for row in rows if row["arithmetic_bits_per_base"] is not None]
    total_arithmetic_bits = (
        sum(float(row["arithmetic_bits_per_base"]) * int(row["region_bases"]) for row in arithmetic_rows)
        if arithmetic_rows
        else None
    )
    summary = {
        "codec": "nc_prefix",
        "source_count": len(rows),
        "total_bases": total_bases,
        "total_theoretical_bits_per_base": total_theoretical_bits / max(total_bases, 1),
        "total_arithmetic_bits_per_base": total_arithmetic_bits / max(total_bases, 1) if total_arithmetic_bits is not None else None,
        "encode_arithmetic": not bool(args.skip_arithmetic),
        "wall_seconds": perf_counter() - total_started,
        "parameters_json": str(output_dir / "parameters.json"),
        "per_source_csv": str(per_source_csv),
    }
    _write_json(output_dir / "nc_prefix_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
