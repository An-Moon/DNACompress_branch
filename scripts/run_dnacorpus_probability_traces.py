#!/usr/bin/env python3
from __future__ import annotations

"""Run per-source DNACorpus target-probability trace generation.

Example:
    CUDA_VISIBLE_DEVICES=0 scripts/run_evo2_1b_env_python.sh \
      scripts/run_dnacorpus_probability_traces.py \
      --model evo2 --dataset-dir datasets/DNACorpus --source BuEb \
      --output-dir outputs/evo2_7b_dnacorpus_w8192_target_traces_gpu0 \
      --window-bases 8192 --batch-size 48 \
      --evo2-probability-mode full_forward \
      --local-path third_party/evo2_7b_base/evo2_7b_base.pt \
      --model-name evo2_7b_base --device cuda:0

    CUDA_VISIBLE_DEVICES=2 python scripts/run_dnacorpus_probability_traces.py \
      --model carbon --dataset-dir datasets/DNACorpus --source BuEb \
      --output-dir outputs/carbon3b_dnacorpus_w8192_full_forward_target_traces \
      --window-bases 8192 --batch-size 32 \
      --carbon-probability-mode full_forward \
      --local-path third_party/Carbon-3B --model-name Carbon-3B --device cuda:0
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import convert_probability_trace_to_position_major  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DNACorpus target-probability traces per source.")
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=("carbon", "evo2", "megabyte"), required=True)
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--window-bases", type=int, default=8192)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trace-dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--shard-rows", type=int, default=1_000_000)
    parser.add_argument("--position-major-output-dir", help="Defaults to <output-dir>_position_major.")
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
        default="streaming_cache",
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
        default="streaming_cache",
        help="Evo2 probability extraction mode passed to run_probability_trace.py.",
    )
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


def _default_position_major_output_dir(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}_position_major")


def _summarize(output_dir: Path, position_major_output_dir: Path, species_paths: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    for source_path in species_paths:
        source = source_path.name
        depth_trace_dir = output_dir / "traces" / source
        trace_dir = position_major_output_dir / "traces" / source
        manifest = _json(trace_dir / "manifest.json")
        if not manifest:
            continue
        producer = dict(manifest.get("producer_config") or {})
        rows.append(
            {
                "source": source,
                "model_family": manifest.get("model_family"),
                "model_id": manifest.get("model_id"),
                "core_base_count": manifest.get("core_base_count"),
                "tail_base_count": manifest.get("tail_base_count"),
                "row_count": manifest.get("row_count"),
                "window_bases": manifest.get("window_bases"),
                "token_merge_size": manifest.get("token_merge_size"),
                "dtype": manifest.get("dtype"),
                "emission_order": manifest.get("emission_order"),
                "shard_count": len(manifest.get("shard_files") or []),
                "checksum_sha256": manifest.get("checksum_sha256"),
                "trace_generation_seconds": producer.get("trace_generation_seconds"),
                "batch_size": producer.get("batch_size"),
                "batch_count": producer.get("batch_count"),
                "trace_dir": str(trace_dir),
                "depth_major_trace_dir": str(depth_trace_dir),
            }
        )
    for summary_dir in [output_dir, position_major_output_dir]:
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(summary_dir / "trace_summary.csv", rows)
        aggregate = {
            "source_count": len(rows),
            "total_core_base_count": sum(int(row.get("core_base_count") or 0) for row in rows),
            "summary_csv": str(summary_dir / "trace_summary.csv"),
            "trace_order": "position_major_v1",
            "rows": rows,
        }
        (summary_dir / "trace_summary.json").write_text(
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


def _trace_command(args: argparse.Namespace, source_path: Path, trace_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_probability_trace.py",
        "--model",
        str(args.model),
        "--source-file",
        str(source_path),
        "--source-format",
        "raw",
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
            ]
        )
        cmd.extend(["--carbon-probability-mode", str(args.carbon_probability_mode)])
    elif args.model == "evo2":
        cmd.extend(
            [
                "--local-path",
                str(args.local_path or "third_party/evo2_7b_base/evo2_7b_base.pt"),
                "--model-name",
                str(args.model_name or "evo2_7b_base"),
            ]
        )
        if args.use_kernels:
            cmd.append("--use-kernels")
        else:
            cmd.append("--no-use-kernels")
        cmd.extend(["--evo2-probability-mode", str(args.evo2_probability_mode)])
    elif args.model == "megabyte":
        cmd.extend(["--run-dir", str(args.run_dir), "--checkpoint-tag", str(args.checkpoint_tag)])
        if args.checkpoint:
            cmd.extend(["--checkpoint", str(args.checkpoint)])
        cmd.extend(["--megabyte-probability-mode", str(args.megabyte_probability_mode)])
        if args.megabyte_model_window_tokens is not None:
            cmd.extend(["--megabyte-model-window-tokens", str(args.megabyte_model_window_tokens)])
        if args.megabyte_model_window_bases is not None:
            cmd.extend(["--megabyte-model-window-bases", str(args.megabyte_model_window_bases)])
    return cmd


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    position_major_output_dir = (
        Path(args.position_major_output_dir)
        if args.position_major_output_dir
        else _default_position_major_output_dir(output_dir)
    )
    trace_root = output_dir / "traces"
    position_trace_root = position_major_output_dir / "traces"
    log_root = output_dir / "logs"
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
        trace_dir = trace_root / source
        position_trace_dir = position_trace_root / source
        depth_complete = _is_complete(trace_dir)
        position_complete = _is_position_major_complete(position_trace_dir)
        if args.skip_existing and depth_complete and position_complete:
            print({"event": "skip_existing", "source": source, "trace_dir": str(position_trace_dir)}, flush=True)
            continue
        log_path = log_root / f"{source}.log"
        started = perf_counter()
        print({"event": "start_source", "source": source, "log": str(log_path)}, flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            if not (args.skip_existing and depth_complete):
                cmd = _trace_command(args, source_path, trace_dir)
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
        _summarize(output_dir, position_major_output_dir, species_paths)

    _summarize(output_dir, position_major_output_dir, species_paths)


if __name__ == "__main__":
    main()
