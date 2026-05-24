from __future__ import annotations

"""Reproduce the GeCo2 paper baseline by compressing each DNACorpus species in full.

Unlike ``scripts/run_geco2_baseline.py``, this runner ignores the train/val/test
splitting in ``compression_compare.json`` and feeds the entire per-species file
under ``datasets/DNACorpus/<species>`` to GeCo2 with the per-species ``-l`` mode
listed in Pratas et al., 2019 (Table 4 of the GeCo2 paper).

Example:
    python scripts/run_geco2_paper_baseline.py \
        --dataset-dir datasets/DNACorpus \
        --output-dir outputs/dna_geco2_paper_baseline/statistics
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import shutil
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_compression_curves import GECO2_PAPER_BASELINE_BY_SOURCE
from scripts.run_geco2_baseline import (
    _resolve_geco2_binary,
    _tail_text,
    build_geco2_command,
)


DEFAULT_DATASET_DIR = Path("datasets/DNACorpus")
DEFAULT_OUTPUT_DIR = Path("outputs/dna_geco2_paper_baseline/statistics")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _run_geco2_on_file(
    *,
    binary: str,
    source_path: Path,
    level: int,
    temp_root: Path | None,
    keep_temp: bool,
) -> dict[str, Any]:
    if keep_temp:
        temp_dir = Path(tempfile.mkdtemp(prefix="geco2_paper_", dir=str(temp_root) if temp_root else None))
        cleanup_ctx = None
    else:
        cleanup_ctx = tempfile.TemporaryDirectory(prefix="geco2_paper_", dir=str(temp_root) if temp_root else None)
        temp_dir = Path(cleanup_ctx.__enter__())
    try:
        # GeCo2 writes <input>.co next to the input. Stage a copy so we never
        # touch the original dataset file.
        staged = temp_dir / source_path.name
        shutil.copyfile(source_path, staged)
        command = build_geco2_command(binary, level=level, input_path=staged)
        started = perf_counter()
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        elapsed = perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"GeCo2 failed (rc={completed.returncode}) on {source_path}: "
                f"{_tail_text(completed.stderr or completed.stdout)}"
            )
        compressed_path = Path(str(staged) + ".co")
        if not compressed_path.exists():
            candidates = sorted(temp_dir.glob("*.co"))
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"GeCo2 did not create a unique .co for {source_path}: {candidates}"
                )
            compressed_path = candidates[0]
        return {
            "command": command,
            "returncode": completed.returncode,
            "seconds": elapsed,
            "stdout_tail": _tail_text(completed.stdout),
            "stderr_tail": _tail_text(completed.stderr),
            "input_size": source_path.stat().st_size,
            "compressed_bytes": compressed_path.stat().st_size,
            "compressed_path": str(compressed_path) if keep_temp else None,
            "temp_dir": str(temp_dir) if keep_temp else None,
        }
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.__exit__(None, None, None)


def run_paper_baseline(
    *,
    dataset_dir: Path,
    output_dir: Path,
    binary: str,
    species_filter: list[str] | None,
    fallback_level: int,
    temp_root: Path | None,
    keep_temp: bool,
) -> dict[str, Any]:
    species_table = GECO2_PAPER_BASELINE_BY_SOURCE
    if species_filter:
        missing = [name for name in species_filter if name not in species_table]
        if missing:
            raise ValueError(f"Unknown species (not in paper table): {missing}")
        ordered = [name for name in species_filter]
    else:
        ordered = list(species_table.keys())

    rows: list[dict[str, Any]] = []
    total_input_bytes = 0
    total_compressed_bytes = 0
    total_paper_bytes = 0
    total_seconds = 0.0

    for index, species in enumerate(ordered, start=1):
        source_path = dataset_dir / species
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing dataset file: {source_path}")
        paper = species_table.get(species)
        level = int(paper["mode"]) if paper is not None else fallback_level
        paper_bytes = int(paper["compressed_bytes"]) if paper is not None else None
        input_size = source_path.stat().st_size
        print(
            f"[geco2_paper] {index}/{len(ordered)} species={species} "
            f"level={level} input_bytes={input_size}",
            flush=True,
        )
        result = _run_geco2_on_file(
            binary=binary,
            source_path=source_path,
            level=level,
            temp_root=temp_root,
            keep_temp=keep_temp,
        )
        compressed_bytes = int(result["compressed_bytes"])
        compressed_bits = compressed_bytes * 8
        delta_bytes = compressed_bytes - paper_bytes if paper_bytes is not None else None
        rel_error = (delta_bytes / paper_bytes) if paper_bytes else None
        bpb = compressed_bits / max(input_size, 1)
        paper_bpb = (paper_bytes * 8) / max(input_size, 1) if paper_bytes is not None else None
        row: dict[str, Any] = {
            "species": species,
            "geco2_level": level,
            "input_bytes": input_size,
            "compressed_bytes": compressed_bytes,
            "compressed_bits_per_base": bpb,
            "paper_compressed_bytes": paper_bytes,
            "paper_bits_per_base": paper_bpb,
            "delta_bytes_vs_paper": delta_bytes,
            "relative_error_vs_paper": rel_error,
            "elapsed_seconds": result["seconds"],
            "geco2_command": result["command"],
            "geco2_returncode": result["returncode"],
            "geco2_stderr_tail": result["stderr_tail"],
        }
        rows.append(row)
        total_input_bytes += input_size
        total_compressed_bytes += compressed_bytes
        if paper_bytes is not None:
            total_paper_bytes += paper_bytes
        total_seconds += float(result["seconds"])

    aggregate: dict[str, Any] = {
        "species_count": len(rows),
        "total_input_bytes": total_input_bytes,
        "total_compressed_bytes": total_compressed_bytes,
        "total_paper_compressed_bytes": total_paper_bytes if total_paper_bytes else None,
        "total_bits_per_base": (total_compressed_bytes * 8) / max(total_input_bytes, 1),
        "total_paper_bits_per_base": (total_paper_bytes * 8) / max(total_input_bytes, 1)
        if total_paper_bytes
        else None,
        "total_delta_bytes_vs_paper": (total_compressed_bytes - total_paper_bytes)
        if total_paper_bytes
        else None,
        "total_relative_error_vs_paper": (
            (total_compressed_bytes - total_paper_bytes) / total_paper_bytes
        )
        if total_paper_bytes
        else None,
        "total_elapsed_seconds": total_seconds,
    }

    metrics: dict[str, Any] = {
        "binary": binary,
        "dataset_dir": str(dataset_dir),
        "fallback_level": fallback_level,
        "per_species": rows,
        "aggregate": aggregate,
        "paper_table": species_table,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "geco2_paper_full.json"
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "geco2_paper_full_per_species.csv", rows)
    _write_csv(
        output_dir / "geco2_paper_full_aggregate.csv",
        [{"metric": key, "value": value} for key, value in aggregate.items()],
    )
    print(f"[geco2_paper] saved metrics to {json_path}", flush=True)
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GeCo2 on full DNACorpus species files using the paper's per-species modes."
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--geco2-bin", default="GeCo2")
    parser.add_argument(
        "--species",
        nargs="+",
        help="Optional subset of species (default: all entries in paper table).",
    )
    parser.add_argument(
        "--fallback-level",
        type=int,
        default=5,
        help="Used only if --species names something missing from the paper table.",
    )
    parser.add_argument("--temp-root")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    binary = _resolve_geco2_binary(args.geco2_bin)
    metrics = run_paper_baseline(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        binary=binary,
        species_filter=args.species,
        fallback_level=args.fallback_level,
        temp_root=Path(args.temp_root) if args.temp_root else None,
        keep_temp=args.keep_temp,
    )
    aggregate = metrics["aggregate"]
    print(
        "[geco2_paper] aggregate: "
        f"input_bytes={aggregate['total_input_bytes']} "
        f"compressed_bytes={aggregate['total_compressed_bytes']} "
        f"bpb={aggregate['total_bits_per_base']:.6f} "
        f"paper_bpb={aggregate['total_paper_bits_per_base']} "
        f"delta_vs_paper={aggregate['total_delta_bytes_vs_paper']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
