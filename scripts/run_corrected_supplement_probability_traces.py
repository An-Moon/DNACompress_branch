#!/usr/bin/env python3
from __future__ import annotations

"""Run target-probability traces for dnacorpus_corrected_supplement_v1.

The model probability implementation stays centralized in
``scripts/run_probability_trace.py``.  This script only enumerates supplement
sources, launches the single-source entrypoint, converts traces to compact
position-major order, and writes a summary.

Examples:
    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python scripts/run_corrected_supplement_probability_traces.py \
      --model carbon \
      --dataset-root datasets/dnacorpus_corrected_supplement_v1 \
      --source EnIn \
      --output-dir outputs/carbon3b_corrected_supplement_w8192_full_forward_target_traces_gpu2 \
      --window-bases 8192 --batch-size 32 \
      --carbon-probability-mode full_forward \
      --local-path third_party/Carbon-3B --model-name Carbon-3B --device cuda:0
"""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import (  # noqa: E402
    ProbabilityTraceReader,
    convert_probability_trace_to_position_major,
)
from scripts.plot_compression_curves import GECO2_PAPER_BASELINE_BY_SOURCE  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate corrected supplement target-probability traces.")
    parser.add_argument("--dataset-root", default="datasets/dnacorpus_corrected_supplement_v1")
    parser.add_argument("--output-dir", required=True, help="Depth-major trace root.")
    parser.add_argument("--position-major-output-dir", help="Defaults to <output-dir>_position_major.")
    parser.add_argument("--model", choices=("carbon", "nc_prefix", "megabyte", "evo2"), required=True)
    parser.add_argument("--source", nargs="+", help="Optional subset, e.g. HePy ScPo PlFa EnIn.")
    parser.add_argument("--window-bases", type=int, default=8192)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trace-dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--shard-rows", type=int, default=1_000_000)
    parser.add_argument(
        "--store-position-major-emit-position",
        action="store_true",
        help="Store emit_position arrays in converted position-major shards.",
    )
    parser.add_argument("--local-path")
    parser.add_argument("--model-name")
    parser.add_argument("--revision", default="fns")
    parser.add_argument(
        "--carbon-probability-mode",
        choices=("streaming_cache", "full_forward"),
        default="full_forward",
        help="Carbon probability extraction mode passed to run_probability_trace.py.",
    )
    parser.add_argument("--run-dir", default="outputs/dna_megabyte_large_opengenome2_11")
    parser.add_argument("--checkpoint-tag", default="best")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--megabyte-probability-mode",
        choices=("auto", "streaming_cache", "full_forward"),
        default="auto",
        help="Megabyte probability extraction mode passed to run_probability_trace.py.",
    )
    parser.add_argument("--megabyte-model-window-tokens", type=int)
    parser.add_argument("--megabyte-model-window-bases", type=int)
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--evo2-probability-mode",
        choices=("streaming_cache", "full_forward"),
        default="full_forward",
        help="Evo2 probability extraction mode passed to run_probability_trace.py.",
    )
    parser.add_argument("--token-merge-size", type=int, default=1)
    parser.add_argument("--nc-prefix-backend", choices=("auto", "fast_cpp"), default="auto")
    parser.add_argument("--nc-prefix-min-windows", type=int, default=1)
    parser.add_argument("--nc-prefix-hash-bucket-count", type=int, default=0)
    parser.add_argument(
        "--nc-prefix-preset",
        choices=("fixed", "geco2_paper"),
        default="geco2_paper",
        help="fixed uses --nc-prefix-geco2-level; geco2_paper uses the DNACorpus paper map.",
    )
    parser.add_argument("--nc-prefix-geco2-level", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    return parser


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_sources(dataset_root: Path, requested: list[str] | None) -> list[dict[str, Any]]:
    summary_path = dataset_root / "metadata" / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"supplement summary not found: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if requested:
        wanted = set(requested)
        rows = [row for row in rows if row.get("id") in wanted]
        missing = sorted(wanted - {str(row.get("id")) for row in rows})
        if missing:
            raise SystemExit(f"unknown supplement source(s): {' '.join(missing)}")
    return rows


def _default_position_major_output_dir(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}_position_major")


def _is_complete(trace_dir: Path) -> bool:
    manifest = _json(trace_dir / "manifest.json")
    return bool(manifest and int(manifest.get("row_count") or 0) > 0)


def _is_position_major_complete(trace_dir: Path) -> bool:
    manifest = _json(trace_dir / "manifest.json")
    return bool(
        manifest
        and int(manifest.get("row_count") or 0) > 0
        and manifest.get("emission_order") == "position_major_v1"
    )


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


def _resolve_geco2_level(source: str, preset: str, configured_level: int) -> int:
    if preset == "fixed":
        return int(configured_level)
    baseline = GECO2_PAPER_BASELINE_BY_SOURCE.get(source)
    if baseline is None:
        raise ValueError(f"no GECO2 paper level is available for {source}; use --nc-prefix-preset fixed")
    return int(baseline["mode"])


def _trace_bpb(trace_dir: Path) -> tuple[float, float]:
    reader = ProbabilityTraceReader(trace_dir)
    total_bits = 0.0
    total_rows = 0
    for shard in reader.iter_shards(verify_checksum=False):
        probs = np.asarray(shard["target_prob"], dtype=np.float64)
        if np.any(~np.isfinite(probs)) or np.any(probs <= 0.0) or np.any(probs > 1.0):
            raise ValueError(f"invalid target probabilities in {trace_dir}")
        total_bits += float(np.sum(-np.log2(probs)))
        total_rows += int(probs.shape[0])
    if total_rows != int(reader.manifest.row_count):
        raise ValueError(f"trace row_count mismatch while summarizing {trace_dir}")
    return total_bits, total_bits / max(total_rows, 1)


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


def _trace_command(args: argparse.Namespace, source_row: dict[str, Any], trace_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_probability_trace.py",
        "--model",
        str(args.model),
        "--source-file",
        str(source_row["clean_acgt_fasta_path"]),
        "--source-format",
        "fasta",
        "--output-trace",
        str(trace_dir),
        "--nc-prefix-window-bases",
        str(args.window_bases),
        "--batch-size",
        str(args.batch_size),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--trace-dtype",
        str(args.trace_dtype),
        "--shard-rows",
        str(args.shard_rows),
        "--force",
    ]
    if args.model == "carbon":
        cmd.extend(
            [
                "--local-path",
                str(args.local_path or "third_party/Carbon-3B"),
                "--model-name",
                str(args.model_name or "Carbon-3B"),
                "--revision",
                str(args.revision),
                "--carbon-probability-mode",
                str(args.carbon_probability_mode),
            ]
        )
    elif args.model == "nc_prefix":
        source = str(source_row["id"])
        geco2_level = _resolve_geco2_level(
            source,
            str(args.nc_prefix_preset),
            int(args.nc_prefix_geco2_level),
        )
        cmd.extend(
            [
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
            ]
        )
    elif args.model == "megabyte":
        cmd.extend(["--run-dir", str(args.run_dir), "--checkpoint-tag", str(args.checkpoint_tag)])
        if args.checkpoint:
            cmd.extend(["--checkpoint", str(args.checkpoint)])
        cmd.extend(["--megabyte-probability-mode", str(args.megabyte_probability_mode)])
        if args.megabyte_model_window_tokens is not None:
            cmd.extend(["--megabyte-model-window-tokens", str(args.megabyte_model_window_tokens)])
        if args.megabyte_model_window_bases is not None:
            cmd.extend(["--megabyte-model-window-bases", str(args.megabyte_model_window_bases)])
    elif args.model == "evo2":
        cmd.extend(
            [
                "--local-path",
                str(args.local_path or "third_party/evo2_7b_base/evo2_7b_base.pt"),
                "--model-name",
                str(args.model_name or "evo2_7b_base"),
            ]
        )
        cmd.append("--use-kernels" if args.use_kernels else "--no-use-kernels")
        cmd.extend(["--evo2-probability-mode", str(args.evo2_probability_mode)])
    return cmd


def _summarize(
    output_dir: Path,
    position_major_output_dir: Path,
    source_rows: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        source = str(source_row["id"])
        depth_trace_dir = output_dir / "traces" / source
        trace_dir = position_major_output_dir / "traces" / source
        manifest = _json(trace_dir / "manifest.json")
        if not manifest:
            continue
        producer = dict(manifest.get("producer_config") or {})
        total_bits, bpb = _trace_bpb(trace_dir)
        expected_bases = int(source_row.get("acgt_bases") or 0)
        row_count = int(manifest.get("row_count") or 0)
        rows.append(
            {
                "source": source,
                "organism": source_row.get("organism"),
                "accession": source_row.get("accession"),
                "model_family": manifest.get("model_family"),
                "model_id": manifest.get("model_id"),
                "record_count": source_row.get("record_count"),
                "total_bases": source_row.get("total_bases"),
                "acgt_bases": expected_bases,
                "n_bases": source_row.get("n_bases"),
                "n_fraction": source_row.get("n_fraction"),
                "gff3_feature_count": source_row.get("gff3_feature_count"),
                "core_base_count": manifest.get("core_base_count"),
                "tail_base_count": manifest.get("tail_base_count"),
                "row_count": row_count,
                "row_count_matches_acgt_bases": row_count == expected_bases,
                "theoretical_bits": total_bits,
                "theoretical_bits_per_base": bpb,
                "window_bases": manifest.get("window_bases"),
                "window_count": producer.get("window_count")
                or math.ceil(max(row_count, 1) / max(int(manifest.get("window_bases") or 1), 1)),
                "token_merge_size": manifest.get("token_merge_size"),
                "batch_size": producer.get("batch_size"),
                "batch_count": producer.get("batch_count"),
                "probability_generation_mode": producer.get("probability_generation_mode"),
                "trace_dtype": manifest.get("dtype"),
                "trace_emission_order": manifest.get("emission_order"),
                "trace_shard_count": len(manifest.get("shard_files") or []),
                "trace_checksum_sha256": manifest.get("checksum_sha256"),
                "trace_generation_seconds": producer.get("trace_generation_seconds"),
                "selected_fasta_path": source_row.get("selected_fasta_path"),
                "clean_acgt_fasta_path": source_row.get("clean_acgt_fasta_path"),
                "selected_gff3_path": source_row.get("selected_gff3_path"),
                "sequence_records_path": source_row.get("sequence_records_path"),
                "non_acgt_intervals_path": source_row.get("non_acgt_intervals_path"),
                "trace_dir": str(trace_dir),
                "depth_major_trace_dir": str(depth_trace_dir),
            }
        )
    total_rows = sum(int(row.get("row_count") or 0) for row in rows)
    total_bits = sum(float(row.get("theoretical_bits") or 0.0) for row in rows)
    aggregate = {
        "schema_version": 1,
        "dataset": "dnacorpus_corrected_supplement_v1",
        "source_count": len(rows),
        "total_core_base_count": total_rows,
        "weighted_theoretical_bits_per_base": total_bits / max(total_rows, 1),
        "trace_order": "position_major_v1",
        "summary_csv": "trace_summary.csv",
        "rows": rows,
    }
    for summary_dir in [output_dir, position_major_output_dir]:
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(summary_dir / "trace_summary.csv", rows)
        (summary_dir / "trace_summary.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    args = _build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    position_major_output_dir = (
        Path(args.position_major_output_dir)
        if args.position_major_output_dir
        else _default_position_major_output_dir(output_dir)
    )
    source_rows = _read_sources(dataset_root, args.source)
    trace_root = output_dir / "traces"
    position_trace_root = position_major_output_dir / "traces"
    log_root = output_dir / "logs"
    trace_root.mkdir(parents=True, exist_ok=True)
    position_trace_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    run_parameters = {
        **vars(args),
        "dataset_root": str(dataset_root),
        "depth_major_output_dir": str(output_dir),
        "position_major_output_dir": str(position_major_output_dir),
        "analysis_trace_order": "position_major_v1",
        "probability_entrypoint": "scripts/run_probability_trace.py",
    }
    for path in [output_dir / "run_parameters.json", position_major_output_dir / "run_parameters.json"]:
        path.write_text(json.dumps(run_parameters, ensure_ascii=False, indent=2), encoding="utf-8")

    for source_row in source_rows:
        source = str(source_row["id"])
        clean_fasta = Path(str(source_row["clean_acgt_fasta_path"]))
        if not clean_fasta.exists():
            raise FileNotFoundError(f"clean ACGT FASTA not found for {source}: {clean_fasta}")
        trace_dir = trace_root / source
        position_trace_dir = position_trace_root / source
        depth_complete = _is_complete(trace_dir)
        position_complete = _is_position_major_complete(position_trace_dir)
        if args.skip_existing and depth_complete and position_complete:
            print({"event": "skip_existing", "source": source, "trace_dir": str(position_trace_dir)}, flush=True)
            continue

        log_path = log_root / f"{source}.log"
        started = perf_counter()
        print({"event": "start_source", "source": source, "model": args.model, "log": str(log_path)}, flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            if not (args.skip_existing and depth_complete):
                cmd = _trace_command(args, source_row, trace_dir)
                log.write(json.dumps({"event": "depth_major_trace_command", "command": cmd}) + "\n")
                log.flush()
                result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0:
                    log.write(json.dumps({"event": "trace_failed", "returncode": result.returncode}) + "\n")
                    log.flush()
                    raise SystemExit(result.returncode)
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
                "seconds": perf_counter() - started,
                "trace_dir": str(position_trace_dir),
                "depth_major_trace_dir": str(trace_dir),
            },
            flush=True,
        )
        _summarize(output_dir, position_major_output_dir, source_rows)

    _summarize(output_dir, position_major_output_dir, source_rows)


if __name__ == "__main__":
    main()
