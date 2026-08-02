#!/usr/bin/env python3
from __future__ import annotations

"""Run full-sequence nc_prefix DNACorpus compression and save target-probability traces."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_compression_curves import GECO2_PAPER_BASELINE_BY_SOURCE  # noqa: E402
from dna_compress.probability_trace import convert_probability_trace_to_position_major  # noqa: E402


DNACORPUS_BEST_AVAILABLE_GECO2_LEVEL_BY_SOURCE = {
    **{source: int(row["mode"]) for source, row in GECO2_PAPER_BASELINE_BY_SOURCE.items()},
    # These two sources are absent from the GECO2 paper-mode table. Local
    # DNACorpus sweeps/results show AnCa benefits strongly from level 10, while
    # WaMe remains best among observed settings at the level-5 fallback.
    "AnCa": 10,
    "WaMe": 5,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run nc_prefix full DNACorpus compression with per-source traces.")
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--output-dir", default="outputs/nc_prefix_dnacorpus_full_batch_w8192_target_traces")
    parser.add_argument("--source", nargs="+", help="Optional subset of DNACorpus flat-file source names.")
    parser.add_argument("--nc-prefix-window-bases", type=int, default=8192)
    parser.add_argument("--nc-prefix-backend", choices=("auto", "fast_cpp"), default="auto")
    parser.add_argument("--nc-prefix-min-windows", type=int, default=1)
    parser.add_argument("--nc-prefix-hash-bucket-count", type=int, default=0)
    parser.add_argument(
        "--nc-prefix-preset",
        choices=("fixed", "geco2_paper", "dnacorpus_best_available"),
        default="fixed",
        help=(
            "fixed uses --nc-prefix-geco2-level for all sources; geco2_paper uses "
            "the paper source map with fallback level 5; dnacorpus_best_available "
            "uses the paper map plus local AnCa/WaMe overrides."
        ),
    )
    parser.add_argument("--nc-prefix-geco2-level", type=int, default=10)
    parser.add_argument("--token-merge-size", type=int, default=1)
    parser.add_argument("--trace-dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--shard-rows", type=int, default=1_000_000)
    parser.add_argument("--position-major-output-dir", help="Defaults to <output-dir>_position_major.")
    parser.add_argument(
        "--store-position-major-emit-position",
        action="store_true",
        help="Store emit_position arrays in converted position-major shards.",
    )
    parser.add_argument("--arithmetic-frequency-total", type=int)
    parser.add_argument("--arithmetic-target-uniform-mass", type=float, default=0.01)
    parser.add_argument("--skip-arithmetic", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _species_paths(dataset_dir: Path, requested: list[str] | None) -> list[Path]:
    paths = [path for path in sorted(dataset_dir.iterdir()) if path.is_file()]
    if requested:
        wanted = set(requested)
        paths = [path for path in paths if path.name in wanted]
    return paths


def _is_complete(metrics_dir: Path, trace_dir: Path) -> bool:
    summary = _json(metrics_dir / "nc_prefix_summary.json")
    manifest = _json(trace_dir / "manifest.json")
    return bool(
        summary
        and summary.get("total_theoretical_bits_per_base") is not None
        and manifest
        and manifest.get("row_count", 0)
    )


def _is_position_major_complete(trace_dir: Path) -> bool:
    manifest = _json(trace_dir / "manifest.json")
    return bool(
        manifest
        and int(manifest.get("row_count") or 0) > 0
        and manifest.get("emission_order") == "position_major_v1"
    )


def _resolve_geco2_level(source: str, preset: str, configured_level: int) -> int:
    if preset == "fixed":
        return int(configured_level)
    if preset == "geco2_paper":
        baseline = GECO2_PAPER_BASELINE_BY_SOURCE.get(source)
        return int(baseline["mode"]) if baseline is not None else 5
    if preset == "dnacorpus_best_available":
        return int(DNACORPUS_BEST_AVAILABLE_GECO2_LEVEL_BY_SOURCE.get(source, 5))
    raise ValueError(f"unknown nc_prefix preset: {preset}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _default_position_major_output_dir(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}_position_major")


def _summarize(output_dir: Path, position_major_output_dir: Path, species_paths: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    for source_path in species_paths:
        source = source_path.name
        metrics_dir = output_dir / "per_species" / source
        depth_trace_dir = output_dir / "traces" / source
        trace_dir = position_major_output_dir / "traces" / source
        summary = _json(metrics_dir / "nc_prefix_summary.json") or {}
        manifest = _json(trace_dir / "manifest.json") or {}
        per_source_rows = []
        csv_path = metrics_dir / "nc_prefix_per_source.csv"
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                per_source_rows = list(csv.DictReader(handle))
        row0 = per_source_rows[0] if per_source_rows else {}
        rows.append(
            {
                "source": source,
                "sample_bases": summary.get("total_bases") or row0.get("region_bases"),
                "theoretical_bits_per_base": summary.get("total_theoretical_bits_per_base")
                or row0.get("theoretical_bits_per_base"),
                "arithmetic_bits_per_base": summary.get("total_arithmetic_bits_per_base")
                or row0.get("arithmetic_bits_per_base"),
                "compression_bases_per_second": row0.get("compression_bases_per_second"),
                "compression_process_seconds": row0.get("compression_process_seconds"),
                "window_bases": row0.get("window_bases") or manifest.get("window_bases"),
                "window_count": row0.get("window_count"),
                "geco2_level": row0.get("geco2_level"),
                "hash_bucket_count_effective": row0.get("hash_bucket_count_effective"),
                "trace_dir": str(trace_dir),
                "depth_major_trace_dir": str(depth_trace_dir),
                "trace_row_count": manifest.get("row_count"),
                "trace_dtype": manifest.get("dtype"),
                "trace_emission_order": manifest.get("emission_order"),
                "trace_shard_count": len(manifest.get("shard_files") or []),
                "trace_checksum_sha256": manifest.get("checksum_sha256"),
            }
        )
    rows = [row for row in rows if row.get("sample_bases") or row.get("trace_row_count")]
    total_bases = 0
    theoretical_bits = 0.0
    arithmetic_bits = 0.0
    arithmetic_bases = 0
    for row in rows:
        bases = int(float(row.get("sample_bases") or 0))
        total_bases += bases
        if row.get("theoretical_bits_per_base") not in {None, ""}:
            theoretical_bits += float(row["theoretical_bits_per_base"]) * bases
        if row.get("arithmetic_bits_per_base") not in {None, ""}:
            arithmetic_bits += float(row["arithmetic_bits_per_base"]) * bases
            arithmetic_bases += bases
    for summary_dir in [output_dir, position_major_output_dir]:
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(summary_dir / "nc_prefix_dnacorpus_full_w8192_summary.csv", rows)
        aggregate = {
            "codec": "nc_prefix",
            "source_count": len(rows),
            "total_bases": total_bases,
            "weighted_theoretical_bits_per_base": theoretical_bits / max(total_bases, 1),
            "weighted_arithmetic_bits_per_base": arithmetic_bits / max(arithmetic_bases, 1) if arithmetic_bases else None,
            "summary_csv": str(summary_dir / "nc_prefix_dnacorpus_full_w8192_summary.csv"),
            "trace_order": "position_major_v1",
            "rows": rows,
        }
        (summary_dir / "nc_prefix_dnacorpus_full_w8192_summary.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _convert_trace(depth_trace_dir: Path, position_trace_dir: Path, args: argparse.Namespace) -> None:
    convert_probability_trace_to_position_major(
        depth_trace_dir,
        position_trace_dir,
        shard_rows=int(args.shard_rows),
        dtype=str(args.trace_dtype),
        overwrite=True,
        verify_checksum=False,
        store_emit_position=bool(args.store_position_major_emit_position),
    )


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    position_major_output_dir = (
        Path(args.position_major_output_dir)
        if args.position_major_output_dir
        else _default_position_major_output_dir(output_dir)
    )
    per_species_root = output_dir / "per_species"
    trace_root = output_dir / "traces"
    position_trace_root = position_major_output_dir / "traces"
    log_root = output_dir / "logs"
    per_species_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)
    position_trace_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    species_paths = _species_paths(Path(args.dataset_dir), args.source)
    run_parameters = {
        **vars(args),
        "depth_major_output_dir": str(output_dir),
        "position_major_output_dir": str(position_major_output_dir),
        "analysis_trace_order": "position_major_v1",
    }
    for path in [output_dir / "run_parameters.json", position_major_output_dir / "run_parameters.json"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_parameters, ensure_ascii=False, indent=2), encoding="utf-8")

    for source_path in species_paths:
        source = source_path.name
        metrics_dir = per_species_root / source
        trace_dir = trace_root / source
        position_trace_dir = position_trace_root / source
        geco2_level = _resolve_geco2_level(
            source,
            str(args.nc_prefix_preset),
            int(args.nc_prefix_geco2_level),
        )
        depth_complete = _is_complete(metrics_dir, trace_dir)
        position_complete = _is_position_major_complete(position_trace_dir)
        if args.skip_existing and depth_complete and position_complete:
            print({"event": "skip_existing", "source": source, "trace_dir": str(position_trace_dir)}, flush=True)
            continue

        compress_cmd = [
            sys.executable,
            "scripts/run_nc_prefix_compression.py",
            "--dataset",
            "dnacorpus",
            "--dataset-dir",
            str(args.dataset_dir),
            "--species",
            source,
            "--alphabet",
            "ACGT",
            "--region-bases",
            "0",
            "--output-dir",
            str(metrics_dir),
            "--nc-prefix-window-bases",
            str(args.nc_prefix_window_bases),
            "--nc-prefix-backend",
            str(args.nc_prefix_backend),
            "--nc-prefix-min-windows",
            str(args.nc_prefix_min_windows),
            "--nc-prefix-hash-bucket-count",
            str(args.nc_prefix_hash_bucket_count),
            "--nc-prefix-geco2-level",
            str(geco2_level),
            "--arithmetic-target-uniform-mass",
            str(args.arithmetic_target_uniform_mass),
        ]
        if args.arithmetic_frequency_total is not None:
            compress_cmd.extend(["--arithmetic-frequency-total", str(args.arithmetic_frequency_total)])
        if args.skip_arithmetic:
            compress_cmd.append("--skip-arithmetic")

        trace_cmd = [
            sys.executable,
            "scripts/run_probability_trace.py",
            "--model",
            "nc_prefix",
            "--source-file",
            str(source_path),
            "--source-format",
            "raw",
            "--output-trace",
            str(trace_dir),
            "--nc-prefix-window-bases",
            str(args.nc_prefix_window_bases),
            "--token-merge-size",
            str(args.token_merge_size),
            "--nc-prefix-backend",
            str(args.nc_prefix_backend),
            "--nc-prefix-min-windows",
            str(args.nc_prefix_min_windows),
            "--nc-prefix-hash-bucket-count",
            str(args.nc_prefix_hash_bucket_count),
            "--nc-prefix-geco2-level",
            str(geco2_level),
            "--trace-dtype",
            str(args.trace_dtype),
            "--shard-rows",
            str(args.shard_rows),
            "--force",
        ]

        started = time.perf_counter()
        log_path = log_root / f"{source}.log"
        print({"event": "start_source", "source": source, "geco2_level": geco2_level, "log": str(log_path)}, flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            if not (args.skip_existing and depth_complete):
                log.write(json.dumps({"event": "compress_command", "command": compress_cmd}) + "\n")
                log.flush()
                compress = subprocess.run(compress_cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                if compress.returncode != 0:
                    log.write(json.dumps({"event": "compress_failed", "returncode": compress.returncode}) + "\n")
                    log.flush()
                    raise SystemExit(compress.returncode)
                log.write(json.dumps({"event": "depth_major_trace_command", "command": trace_cmd}) + "\n")
                log.flush()
                trace = subprocess.run(trace_cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                if trace.returncode != 0:
                    log.write(json.dumps({"event": "trace_failed", "returncode": trace.returncode}) + "\n")
                    log.flush()
                    raise SystemExit(trace.returncode)
            log.write(
                json.dumps(
                    {
                        "event": "convert_position_major",
                        "source_trace_dir": str(trace_dir),
                        "output_trace_dir": str(position_trace_dir),
                    }
                )
                + "\n"
            )
            log.flush()
            _convert_trace(trace_dir, position_trace_dir, args)
        print(
            {
                "event": "finish_source",
                "source": source,
                "seconds": time.perf_counter() - started,
                "metrics_dir": str(metrics_dir),
                "trace_dir": str(position_trace_dir),
                "depth_major_trace_dir": str(trace_dir),
            },
            flush=True,
        )
        _summarize(output_dir, position_major_output_dir, species_paths)

    _summarize(output_dir, position_major_output_dir, species_paths)


if __name__ == "__main__":
    main()
