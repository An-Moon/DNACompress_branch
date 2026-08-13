#!/usr/bin/env python3
from __future__ import annotations

"""Nested 3/4/5/6-segment robustness probe for the context-length sweep.

The frozen three segments from a completed context probe form S3. Three new
segments are added sequentially by filling the largest currently uncovered
coordinate gap. The added start is sampled uniformly from valid aligned starts
inside that gap with a fixed seed. Thus S3 is a strict subset of S4, S5 and S6.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from time import perf_counter

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import (  # noqa: E402
    ProbabilityTraceReader,
    convert_probability_trace_to_position_major,
)


WINDOWS = (8192, 16384, 32768, 65536, 131072)
BASE_NAMES = ("segment_a", "segment_b", "segment_c")
ADDED_NAMES = ("segment_d", "segment_e", "segment_f")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Nested segment-count robustness probe.")
    p.add_argument("--source", required=True)
    p.add_argument("--base-experiment-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--selection-seed", type=int, default=20260812)
    p.add_argument("--models", nargs="+", choices=("evo2_optimized", "carbon"), required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--evo2-python", default="/home/Hu_xuanwei/.conda/envs/evo2_py311/bin/python")
    p.add_argument("--carbon-python", default="/home/Hu_xuanwei/.conda/envs/qwen_vl/bin/python")
    p.add_argument("--prepare-only", action="store_true")
    return p


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def valid_aligned_range(lo: int, hi: int, length: int, alignment: int) -> tuple[int, int] | None:
    first = math.ceil(lo / alignment) * alignment
    last = ((hi - length) // alignment) * alignment
    return None if first > last else (first, last)


def prepare(args: argparse.Namespace) -> dict:
    base_manifest = json.loads((args.base_experiment_dir / "region_manifest.json").read_text())
    source_path = REPO_ROOT / "datasets" / "DNACorpus" / args.source
    source = source_path.read_bytes().upper()
    if sha256(source) != base_manifest["source_sha256"]:
        raise ValueError("source checksum differs from frozen three-segment experiment")
    length = int(base_manifest["segment_bases"])
    alignment = int(base_manifest["alignment_bases"])
    regions = [dict(x) for x in base_manifest["segments"]]
    rng = random.Random(int(args.selection_seed))

    for name in ADDED_NAMES:
        occupied = sorted((int(x["start"]), int(x["end"])) for x in regions)
        gaps: list[tuple[int, int]] = []
        cursor = 0
        for start, end in occupied:
            if cursor < start:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < len(source):
            gaps.append((cursor, len(source)))
        candidates = []
        for lo, hi in gaps:
            bounds = valid_aligned_range(lo, hi, length, alignment)
            if bounds is not None:
                first, last = bounds
                candidates.append((((last - first) // alignment) + 1, hi - lo, lo, hi, first, last))
        if not candidates:
            raise ValueError("no uncovered aligned gap can hold another segment")
        _, _, lo, hi, first, last = max(candidates)
        count = ((last - first) // alignment) + 1
        start = first + rng.randrange(count) * alignment
        payload = source[start : start + length]
        path = args.output_dir / "segments" / f"{name}.seq"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        regions.append({
            "name": name,
            "selection_step": len(regions) + 1,
            "selected_gap_start": lo,
            "selected_gap_end": hi,
            "start": start,
            "end": start + length,
            "bases": length,
            "path": str(path),
            "sha256": sha256(payload),
        })

    manifest = {
        "experiment": "dnacorpus_segment_count_robustness_v1",
        "selection_policy": "frozen_S3_plus_seeded_random_aligned_largest_gap_fill",
        "selection_seed": int(args.selection_seed),
        "source": args.source,
        "source_path": str(source_path),
        "source_bases": len(source),
        "source_sha256": sha256(source),
        "segment_bases": length,
        "alignment_bases": alignment,
        "nested_sets": {f"S{k}": [str(x["name"]) for x in regions[:k]] for k in range(3, 7)},
        "segments": regions,
        "base_experiment_dir": str(args.base_experiment_dir),
    }
    write_json(args.output_dir / "region_manifest.json", manifest)
    return manifest


def complete(path: Path) -> bool:
    try:
        return int(json.loads((path / "manifest.json").read_text())["row_count"]) > 0
    except Exception:
        return False


def command(args: argparse.Namespace, model: str, source: Path, output: Path, window: int) -> list[str]:
    common = [
        "scripts/run_probability_trace.py", "--source-file", str(source), "--source-format", "raw",
        "--output-trace", str(output), "--nc-prefix-window-bases", str(window),
        "--trace-dtype", "float32", "--shard-rows", "1000000", "--force",
        "--batch-size", "1", "--device", args.device, "--dtype", "bfloat16",
    ]
    if model == "evo2_optimized":
        return [args.evo2_python, *common, "--model", "evo2",
                "--local-path", "/home/Hu_xuanwei/model/evo2_7b/evo2_7b.pt", "--model-name", "evo2_7b",
                "--evo2-probability-mode", "full_forward", "--use-kernels", "--evo2-ungated-hcs-kernel",
                "--evo2-ungated-hcs-chunk-channels", "1024", "--evo2-hcm-chunk-channels", "128",
                "--evo2-hcl-chunk-channels", "128"]
    return [args.carbon_python, *common, "--model", "carbon",
            "--local-path", "/home/Hu_xuanwei/model/Carbon-3B", "--model-name", "Carbon-3B",
            "--revision", "fns", "--carbon-probability-mode", "full_forward"]


def bpb(trace: Path) -> tuple[float, int]:
    reader = ProbabilityTraceReader(trace)
    total_bits = 0.0
    rows = 0
    for shard in reader.iter_shards():
        prob = np.asarray(shard["target_prob"], dtype=np.float64)
        total_bits += float((-np.log2(np.maximum(prob, 1e-300))).sum())
        rows += int(prob.size)
    return total_bits / rows, rows


def summarize(args: argparse.Namespace, manifest: dict) -> None:
    rows = []
    for model in ("evo2_optimized", "carbon"):
        for window in WINDOWS:
            values = []
            for segment in manifest["segments"]:
                name = segment["name"]
                if name in BASE_NAMES:
                    trace = args.base_experiment_dir / "traces_position_major" / name / f"w{window:06d}" / model
                else:
                    trace = args.output_dir / "traces_position_major" / name / f"w{window:06d}" / model
                if complete(trace):
                    value, count = bpb(trace)
                    values.append({"segment": name, "bpb": value, "bases": count})
            for k in range(3, min(6, len(values)) + 1):
                subset = values[:k]
                weights = np.asarray([x["bases"] for x in subset], dtype=np.float64)
                bpbs = np.asarray([x["bpb"] for x in subset], dtype=np.float64)
                rows.append({
                    "model": model, "window_bases": window, "segment_count": k,
                    "segments": [x["segment"] for x in subset],
                    "total_bases": int(weights.sum()), "mean_bpb": float(np.average(bpbs, weights=weights)),
                    "segment_sd_bpb": float(np.std(bpbs, ddof=1)) if k > 1 else 0.0,
                    "segment_sem_bpb": float(np.std(bpbs, ddof=1) / math.sqrt(k)) if k > 1 else 0.0,
                    "segment_min_bpb": float(bpbs.min()), "segment_max_bpb": float(bpbs.max()),
                })
    write_json(args.output_dir / "summary.json", {"manifest": manifest, "rows": rows})


def main() -> None:
    args = parser().parse_args()
    manifest = prepare(args)
    if args.prepare_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    for model in args.models:
        for segment in manifest["segments"][3:]:
            source = Path(segment["path"])
            for window in WINDOWS:
                depth = args.output_dir / "traces_depth_major" / segment["name"] / f"w{window:06d}" / model
                position = args.output_dir / "traces_position_major" / segment["name"] / f"w{window:06d}" / model
                if not complete(depth):
                    log = args.output_dir / "logs" / segment["name"] / f"w{window:06d}_{model}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    started = perf_counter()
                    cmd = command(args, model, source, depth, window)
                    with log.open("w") as handle:
                        handle.write(json.dumps({"command": cmd}) + "\n"); handle.flush()
                        result = subprocess.run(cmd, cwd=REPO_ROOT, stdout=handle, stderr=subprocess.STDOUT)
                    write_json(args.output_dir / "timings" / segment["name"] / f"w{window:06d}_{model}.json",
                               {"returncode": result.returncode, "wall_seconds": perf_counter() - started,
                                "model": model, "window_bases": window, "segment": segment["name"], "command": cmd})
                    if result.returncode:
                        raise SystemExit(f"failed; inspect {log}")
                if not complete(position):
                    convert_probability_trace_to_position_major(depth, position, shard_rows=1_000_000,
                                                                dtype="float32", overwrite=True,
                                                                verify_checksum=True, store_emit_position=False)
                summarize(args, manifest)
    summarize(args, manifest)


if __name__ == "__main__":
    main()
