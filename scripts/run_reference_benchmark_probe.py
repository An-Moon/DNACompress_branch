#!/usr/bin/env python3
from __future__ import annotations

"""Run existing region/full probes over reference_benchmark_v1 FASTA sources.

Examples:

    python scripts/run_reference_benchmark_probe.py \
      --mode region \
      --subset minimal \
      --output-dir outputs/reference_benchmark_v1_region_smoke \
      -- --model nc_prefix --region-bases 50000 --compute-only

    python scripts/run_reference_benchmark_probe.py \
      --mode full \
      --source e_coli_k12 s_cerevisiae homo_sapiens_chr7 \
      --output-dir outputs/reference_benchmark_v1_full_smoke \
      -- --model nc_prefix --compute-only
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "reference_benchmark_v1"


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError(f"Invalid manifest: {path}")
    return data


def _sources_from_manifest(manifest: dict[str, Any], *, subset: str, requested: list[str] | None) -> list[str]:
    available: list[str] = []
    for entry in manifest["entries"]:
        source_id = str(entry["id"])
        if requested and source_id not in requested:
            continue
        if subset != "all" and subset not in set(entry.get("subsets", [])):
            continue
        available.append(source_id)
    missing = sorted(set(requested or []) - set(available))
    if missing:
        raise ValueError(f"Requested sources are not in subset {subset}: {missing}")
    return available


def _sources_from_summary(summary_path: Path, *, subset_sources: list[str]) -> list[str]:
    processed = set()
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("id"):
                    processed.add(row["id"])
    return [source for source in subset_sources if source in processed]


def _json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--manifest")
    parser.add_argument("--mode", choices=("region", "full"), required=True)
    parser.add_argument("--subset", choices=("minimal", "primary", "all"), default="minimal")
    parser.add_argument("--source", nargs="+")
    parser.add_argument(
        "--sequence-view",
        choices=("acgtn", "acgt"),
        default="acgtn",
        help="Use selected/fasta with N retained, or clean/acgt with N removed for tools such as GeCo2.",
    )
    parser.add_argument("--processed-only", action="store_true", default=True)
    parser.add_argument("--include-unprocessed", dest="processed_only", action="store_false")
    parser.add_argument("--output-dir", required=True)
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return args, passthrough


def main() -> None:
    args, passthrough = _parse_args()
    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else root / "manifest.yaml"
    manifest = _read_manifest(manifest_path)
    sources = _sources_from_manifest(manifest, subset=str(args.subset), requested=args.source)
    if args.processed_only:
        sources = _sources_from_summary(root / "metadata" / "summary.csv", subset_sources=sources)
    if not sources:
        raise RuntimeError("No benchmark sources selected. Run prepare_reference_benchmark.py --process first or pass --include-unprocessed.")
    input_dir = root / ("clean/acgt" if args.sequence_view == "acgt" else "selected/fasta")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Benchmark FASTA view does not exist: {input_dir}")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "scripts" / ("run_dna_region_bpb_probe.py" if args.mode == "region" else "run_dna_full_bpb_probe.py")
    results: list[dict[str, Any]] = []
    for source in sources:
        source_output_dir = output_dir / source
        command = [
            sys.executable,
            str(script),
            "--dataset",
            "opengenome2",
            "--input-dir",
            str(input_dir),
            "--source",
            source,
            "--output-dir",
            str(source_output_dir),
            *passthrough,
        ]
        _json_print({"event": "probe_start", "mode": args.mode, "source": source, "sequence_view": args.sequence_view, "command": command})
        completed = subprocess.run(command, check=False)
        row = {"source": source, "returncode": completed.returncode, "output_dir": str(source_output_dir)}
        results.append(row)
        _json_print({"event": "probe_done", **row})
        if completed.returncode != 0:
            raise RuntimeError(f"Probe failed for {source} with exit code {completed.returncode}")
    summary_path = output_dir / "reference_benchmark_probe_runs.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "returncode", "output_dir"])
        writer.writeheader()
        writer.writerows(results)
    _json_print({"event": "probe_summary_written", "path": str(summary_path)})


if __name__ == "__main__":
    main()
