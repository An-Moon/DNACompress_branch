#!/usr/bin/env python3
from __future__ import annotations

"""Run the three-region DNACorpus context-length compression probe.

The script deliberately keeps one experiment in one entry point: it selects
three deterministic, non-overlapping regions, writes the region manifest,
generates reusable target-probability traces, converts them to position-major
order, and computes lightweight offline fusion summaries.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
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
    validate_trace_compatibility,
)


DEFAULT_WINDOWS = (8192, 16384, 32768, 65536, 131072)
DEFAULT_SEGMENT_BASES = 16 * DEFAULT_WINDOWS[-1]
DEFAULT_SELECTION_SEED = 20260812
SEGMENT_NAMES = ("segment_a", "segment_b", "segment_c")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DNACorpus three-region context-length probability probe.")
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--source", default="OrSa")
    parser.add_argument("--output-dir", default="outputs/dnacorpus_context_length_probe_orsa_v2")
    parser.add_argument("--segment-bases", type=int, default=DEFAULT_SEGMENT_BASES)
    parser.add_argument("--window-bases", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument(
        "--alignment-bases",
        type=int,
        default=DEFAULT_WINDOWS[-1],
        help="Fixed region alignment for the complete experiment; keep unchanged for subset/smoke runs.",
    )
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("stap", "carbon", "carbon_kv", "evo2", "evo2_optimized"),
        default=["stap"],
    )
    parser.add_argument("--segments", nargs="+", choices=SEGMENT_NAMES, default=list(SEGMENT_NAMES))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--python",
        default="/home/Hu_xuanwei/.conda/envs/qwen_vl/bin/python",
        help="Python executable for STAP and Carbon trace generation.",
    )
    parser.add_argument("--trace-dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--shard-rows", type=int, default=1_000_000)
    parser.add_argument("--carbon-batch-size", default="auto")
    parser.add_argument("--evo2-batch-size", default="auto")
    parser.add_argument(
        "--evo2-use-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Evo2 optimized kernels; keep disabled for the baseline-compatible path.",
    )
    parser.add_argument(
        "--evo2-filter-chunk-systems",
        type=int,
        default=0,
        help="Memory-safe Evo2 compute_filter chunk size along independent systems; zero disables it.",
    )
    parser.add_argument(
        "--evo2-ungated-hcs-kernel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Vortex Triton HCS for Evo2's non-gated short FIR.",
    )
    parser.add_argument("--evo2-ungated-hcs-chunk-channels", type=int, default=1024)
    parser.add_argument("--evo2-hcm-chunk-channels", type=int, default=128)
    parser.add_argument("--evo2-hcl-chunk-channels", type=int, default=128)
    parser.add_argument("--carbon-local-path", default="/home/Hu_xuanwei/model/Carbon-3B")
    parser.add_argument("--carbon-model-name", default="Carbon-3B")
    parser.add_argument("--carbon-revision", default="fns")
    parser.add_argument("--evo2-local-path", default="/home/Hu_xuanwei/model/evo2_7b/evo2_7b.pt")
    parser.add_argument("--evo2-model-name", default="evo2_7b")
    parser.add_argument(
        "--evo2-python",
        help="Optional Python executable or wrapper for Evo2. Defaults to the current interpreter.",
    )
    parser.add_argument("--fusion-eta", type=float, default=0.05)
    parser.add_argument("--fusion-initial-lm-weight", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    return parser


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _aligned_random_start(
    lo: int, hi: int, segment_bases: int, alignment: int, rng: random.Random
) -> int:
    first = int(math.ceil(lo / alignment) * alignment)
    last = int(((hi - segment_bases) // alignment) * alignment)
    if first > last:
        raise ValueError(
            f"stratum [{lo}, {hi}) cannot hold an aligned {segment_bases}-base segment at alignment {alignment}"
        )
    aligned_count = ((last - first) // alignment) + 1
    return first + rng.randrange(aligned_count) * alignment


def select_three_regions(
    source_bases: int,
    segment_bases: int,
    alignment: int,
    selection_seed: int = DEFAULT_SELECTION_SEED,
) -> list[dict[str, int | str]]:
    if source_bases < 3 * segment_bases:
        raise ValueError(f"source has {source_bases} bases; three non-overlapping segments require {3 * segment_bases}")
    regions: list[dict[str, int | str]] = []
    rng = random.Random(int(selection_seed))
    for index, name in enumerate(SEGMENT_NAMES):
        lo = (index * source_bases) // 3
        hi = ((index + 1) * source_bases) // 3
        start = _aligned_random_start(lo, hi, segment_bases, alignment, rng)
        regions.append({"name": name, "stratum_start": lo, "stratum_end": hi, "start": start, "end": start + segment_bases})
    for left, right in zip(regions, regions[1:]):
        if int(left["end"]) > int(right["start"]):
            raise AssertionError(f"selected regions overlap: {left} and {right}")
    return regions


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_regions(
    source_path: Path,
    output_dir: Path,
    segment_bases: int,
    alignment: int,
    selection_seed: int,
    force: bool,
) -> dict[str, Any]:
    source = source_path.read_bytes().upper()
    if any(base not in b"ACGT" for base in source):
        raise ValueError(f"{source_path} is not an A/C/G/T-only flat sequence")
    regions = select_three_regions(len(source), segment_bases, alignment, selection_seed)
    segment_dir = output_dir / "segments"
    entries = []
    for region in regions:
        payload = source[int(region["start"]) : int(region["end"])]
        path = segment_dir / f"{region['name']}.seq"
        if path.exists() and not force:
            existing = path.read_bytes()
            if existing != payload:
                raise ValueError(f"existing segment differs from deterministic selection: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        entries.append({**region, "bases": len(payload), "path": str(path), "sha256": _sha256(payload)})
    manifest = {
        "experiment": "dnacorpus_context_length_probe_v1",
        "selection_policy": "three_equal_coordinate_strata_seeded_random_aligned_nonoverlap",
        "selection_seed": int(selection_seed),
        "source": source_path.name,
        "source_path": str(source_path),
        "source_bases": len(source),
        "source_sha256": _sha256(source),
        "segment_bases": segment_bases,
        "alignment_bases": alignment,
        "segments": entries,
    }
    _write_json(output_dir / "region_manifest.json", manifest)
    return manifest


def _trace_dir(output_dir: Path, order: str, segment: str, window_bases: int, model: str) -> Path:
    return output_dir / f"traces_{order}" / segment / f"w{window_bases:06d}" / model


def _trace_complete(path: Path) -> bool:
    try:
        payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        return int(payload.get("row_count") or 0) > 0
    except Exception:
        return False


def _model_command(args: argparse.Namespace, model: str, segment_path: Path, output_trace: Path, window_bases: int) -> list[str]:
    runner = [str(args.evo2_python)] if model in {"evo2", "evo2_optimized"} and args.evo2_python else [str(args.python)]
    runner_model = "evo2" if model == "evo2_optimized" else model
    cmd = runner + [
        "scripts/run_probability_trace.py",
        "--model", "nc_prefix" if runner_model == "stap" else ("carbon" if runner_model == "carbon_kv" else runner_model),
        "--source-file", str(segment_path),
        "--source-format", "raw",
        "--output-trace", str(output_trace),
        "--nc-prefix-window-bases", str(window_bases),
        "--trace-dtype", str(args.trace_dtype),
        "--shard-rows", str(args.shard_rows),
        "--force",
    ]
    if model == "stap":
        cmd.extend([
            "--token-merge-size", "1", "--nc-prefix-backend", "auto", "--nc-prefix-min-windows", "1",
            "--nc-prefix-geco2-level", "10",
        ])
    elif model in {"carbon", "carbon_kv"}:
        probability_mode = "streaming_cache" if model == "carbon_kv" else "full_forward"
        cmd.extend([
            "--local-path", str(args.carbon_local_path), "--model-name", str(args.carbon_model_name),
            "--revision", str(args.carbon_revision), "--carbon-probability-mode", probability_mode,
            "--batch-size", str(args.carbon_batch_size), "--device", str(args.device), "--dtype", str(args.dtype),
        ])
    else:
        cmd.extend([
            "--local-path", str(args.evo2_local_path), "--model-name", str(args.evo2_model_name),
            "--evo2-probability-mode", "full_forward", "--batch-size", str(args.evo2_batch_size),
            "--device", str(args.device), "--dtype", str(args.dtype),
            "--evo2-filter-chunk-systems", str(args.evo2_filter_chunk_systems),
            "--evo2-ungated-hcs-chunk-channels", str(args.evo2_ungated_hcs_chunk_channels),
            "--evo2-hcm-chunk-channels", str(args.evo2_hcm_chunk_channels),
            "--evo2-hcl-chunk-channels", str(args.evo2_hcl_chunk_channels),
            "--evo2-ungated-hcs-kernel" if bool(args.evo2_ungated_hcs_kernel) else "--no-evo2-ungated-hcs-kernel",
            "--use-kernels" if bool(args.evo2_use_kernels) else "--no-use-kernels",
        ])
    return cmd


def _load_position_trace(path: Path) -> tuple[Any, np.ndarray]:
    reader = ProbabilityTraceReader(path)
    probs = np.concatenate([np.asarray(shard["target_prob"], dtype=np.float64) for shard in reader.iter_shards()])
    return reader.manifest, probs.clip(min=1e-300)


def _model_summary(trace_dir: Path, common_block_bases: int) -> dict[str, Any]:
    manifest, probs = _load_position_trace(trace_dir)
    positions = np.arange(probs.shape[0], dtype=np.int64)
    common = (positions % int(common_block_bases)) != 0
    bits = -np.log2(probs)
    return {
        "trace": str(trace_dir),
        "trace_checksum_sha256": manifest.checksum_sha256,
        "row_count": int(probs.shape[0]),
        "native_bpb": float(bits.mean()),
        "common_mask_bases": int(common.sum()),
        "common_mask_bpb": float(bits[common].mean()),
    }


def _fusion_summary(left_dir: Path, right_dir: Path, common_block_bases: int, eta: float, initial: float) -> dict[str, Any]:
    left_manifest, left = _load_position_trace(left_dir)
    right_manifest, right = _load_position_trace(right_dir)
    diffs = validate_trace_compatibility(left_manifest, right_manifest)
    if diffs:
        raise ValueError(f"incompatible fusion traces: {diffs}")
    window_bases = int(left_manifest.window_bases)
    if left.shape != right.shape:
        raise ValueError("fusion traces have different row counts")
    window_count = int(math.ceil(left.shape[0] / window_bases))
    weights = np.full(window_count, float(initial), dtype=np.float64)
    fused_bits = np.empty(left.shape[0], dtype=np.float64)
    power = 1.0 - float(eta)
    for depth in range(window_bases):
        positions = np.arange(depth, left.shape[0], window_bases, dtype=np.int64)
        window_ids = positions // window_bases
        lm_weight = weights[window_ids]
        stat_weight = 1.0 - lm_weight
        fused = np.maximum(lm_weight * left[positions] + stat_weight * right[positions], 1e-300)
        fused_bits[positions] = -np.log2(fused)
        lm_new = np.power(lm_weight, power) * left[positions]
        stat_new = np.power(stat_weight, power) * right[positions]
        weights[window_ids] = lm_new / np.maximum(lm_new + stat_new, 1e-300)
    positions = np.arange(left.shape[0], dtype=np.int64)
    common = (positions % int(common_block_bases)) != 0
    return {
        "fusion_policy": "online_hedge_linear_target_probability_trace",
        "fusion_eta": float(eta),
        "fusion_initial_lm_weight": float(initial),
        "left_trace": str(left_dir),
        "right_trace": str(right_dir),
        "left_checksum_sha256": left_manifest.checksum_sha256,
        "right_checksum_sha256": right_manifest.checksum_sha256,
        "native_bpb": float(fused_bits.mean()),
        "common_mask_bases": int(common.sum()),
        "common_mask_bpb": float(fused_bits[common].mean()),
        "final_mean_lm_weight": float(weights.mean()),
    }


def main() -> None:
    args = _parser().parse_args()
    windows = sorted(set(int(value) for value in args.window_bases))
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("window lengths must be positive")
    alignment = int(args.alignment_bases)
    if alignment <= 0 or any(alignment % value for value in windows):
        raise ValueError("every requested window length must divide --alignment-bases")
    if int(args.segment_bases) % alignment:
        raise ValueError("segment length must be divisible by the largest window")

    output_dir = Path(args.output_dir)
    source_path = Path(args.dataset_dir) / str(args.source)
    manifest = prepare_regions(
        source_path, output_dir, int(args.segment_bases), alignment, int(args.selection_seed), bool(args.force)
    )
    if args.prepare_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    selected = {str(entry["name"]): entry for entry in manifest["segments"] if str(entry["name"]) in args.segments}
    summary_rows: list[dict[str, Any]] = []
    for segment_name, entry in selected.items():
        segment_path = Path(str(entry["path"]))
        for window_bases in windows:
            model_summaries: dict[str, Any] = {}
            for model in args.models:
                depth_dir = _trace_dir(output_dir, "depth_major", segment_name, window_bases, model)
                position_dir = _trace_dir(output_dir, "position_major", segment_name, window_bases, model)
                if not _trace_complete(depth_dir) or bool(args.force):
                    log_path = output_dir / "logs" / segment_name / f"w{window_bases:06d}_{model}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    cmd = _model_command(args, model, segment_path, depth_dir, window_bases)
                    started_at = datetime.now(timezone.utc)
                    wall_started = perf_counter()
                    with log_path.open("w", encoding="utf-8") as log:
                        log.write(json.dumps({"command": cmd}) + "\n")
                        log.flush()
                        completed = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                    ended_at = datetime.now(timezone.utc)
                    _write_json(
                        output_dir / "timings" / segment_name / f"w{window_bases:06d}_{model}.json",
                        {
                            "segment": segment_name,
                            "window_bases": window_bases,
                            "model": model,
                            "started_at_utc": started_at.isoformat(),
                            "ended_at_utc": ended_at.isoformat(),
                            "wall_seconds": perf_counter() - wall_started,
                            "returncode": int(completed.returncode),
                            "command": cmd,
                            "log": str(log_path),
                        },
                    )
                    if completed.returncode != 0:
                        raise SystemExit(f"trace generation failed; inspect {log_path}")
                if not _trace_complete(position_dir) or bool(args.force):
                    convert_probability_trace_to_position_major(
                        depth_dir, position_dir, shard_rows=int(args.shard_rows), dtype=str(args.trace_dtype),
                        overwrite=True, verify_checksum=True, store_emit_position=False,
                    )
                model_summaries[model] = _model_summary(position_dir, min(windows))

            for model in ("stap", "carbon", "carbon_kv", "evo2", "evo2_optimized"):
                position_dir = _trace_dir(output_dir, "position_major", segment_name, window_bases, model)
                if _trace_complete(position_dir):
                    model_summaries[model] = _model_summary(position_dir, min(windows))

            fusions: dict[str, Any] = {}
            stap_dir = _trace_dir(output_dir, "position_major", segment_name, window_bases, "stap")
            for lm in ("carbon", "carbon_kv", "evo2", "evo2_optimized"):
                lm_dir = _trace_dir(output_dir, "position_major", segment_name, window_bases, lm)
                if _trace_complete(lm_dir) and _trace_complete(stap_dir):
                    fusions[f"{lm}_stap"] = _fusion_summary(
                        lm_dir, stap_dir, min(windows), float(args.fusion_eta), float(args.fusion_initial_lm_weight)
                    )
            row = {
                "source": str(args.source), "segment": segment_name, "segment_start": int(entry["start"]),
                "segment_bases": int(entry["bases"]), "window_bases": window_bases,
                "window_count": int(entry["bases"]) // window_bases,
                "models": model_summaries, "fusions": fusions,
            }
            summary_rows.append(row)
            _write_json(output_dir / "summaries" / segment_name / f"w{window_bases:06d}.json", row)

    summary_rows = []
    for summary_path in sorted((output_dir / "summaries").glob("*/w*.json")):
        summary_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
    _write_json(
        output_dir / "summary.json",
        {
            "source": str(args.source), "window_bases": windows, "segment_bases": int(args.segment_bases),
            "common_mask_block_bases": min(windows), "trace_policy": "retain_source_target_probability_traces",
            "fusion_trace_policy": "summary_only_recomputable_from_source_traces", "rows": summary_rows,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "row_count": len(summary_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
