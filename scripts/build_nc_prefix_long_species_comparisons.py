#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.noncontiguous_prefix_codec import (  # noqa: E402
    NoncontiguousPrefixConfig,
    compute_noncontiguous_prefix_probabilities,
)
from scripts.run_dna_region_bpb_probe import (  # noqa: E402
    _json_safe,
    _load_model_artifact_payload,
    extract_filtered_region,
    filtered_length,
    model_window_average,
    moving_average,
    resolve_region_sources,
    smooth_defined_positions,
    stable_random_start,
    window_rows,
    write_csv,
    write_model_artifact,
)


DEFAULT_SPECIES = (
    "HePy",
    "WaMe",
    "EsCo",
    "EnIn",
    "OrSa",
    "ScPo",
    "BuEb",
    "DaRe",
    "HoSa",
    "YeMi",
    "AnCa",
    "HaHi",
    "PlFa",
    "AeCa",
    "AgPh",
    "DrMe",
    "GaGa",
)
DEFAULT_WINDOW_BASES = 3072
DEFAULT_MAX_WINDOWS = 8192


def _safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_full_result(full_root: Path, species: str) -> dict[str, Any]:
    path = full_root / species / "region_bpb.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing full-source result for {species}: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    result["_json_path"] = str(path)
    return result


def _slice_full_model(
    *,
    full_result: dict[str, Any],
    model_name: str,
    region_start: int,
    region_bases: int,
    output_dir: Path,
    record_dtype: str,
    window_bases: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    models = full_result.get("models") if isinstance(full_result.get("models"), dict) else {}
    summary = models.get(model_name)
    if not isinstance(summary, dict):
        raise KeyError(f"Model {model_name!r} missing from full-source result")
    payload = _load_model_artifact_payload(summary, json_dir=Path(full_result["_json_path"]).parent)
    if payload is None:
        raise FileNotFoundError(f"Could not load artifact for {model_name}")
    full_offsets = np.asarray(payload["offsets"], dtype=np.int64)
    end = region_start + region_bases
    start_index = int(np.searchsorted(full_offsets, region_start, side="left"))
    end_index = int(np.searchsorted(full_offsets, end, side="left"))
    absolute_offsets = full_offsets[start_index:end_index]
    bpb = np.asarray(payload["bpb"], dtype=np.float64)[start_index:end_index]
    relative_offsets = absolute_offsets - region_start
    metadata = dict(summary.get("metadata") or {})
    metadata["source_full_result_json"] = str(full_result["_json_path"])
    metadata["source_full_model"] = model_name
    metadata["model_window_bases"] = int(summary.get("model_window_bases") or metadata.get("model_window_bases") or window_bases)
    rows = window_rows(bpb, relative_offsets, source_start=region_start, window_bases=window_bases)
    for row in rows:
        row["model"] = model_name
        row["plot_window_bases"] = window_bases
    means, counts = model_window_average(bpb, relative_offsets, int(metadata["model_window_bases"]))
    total_bits = float(np.sum(bpb))
    artifact_summary = write_model_artifact(
        output_dir,
        model_name=model_name,
        bpb=bpb,
        offsets=relative_offsets,
        metadata=metadata,
        window_summary=rows,
        worst_window=max(rows, key=lambda item: float(item["mean_bpb"])) if rows else None,
        model_window_means=means,
        model_window_counts=counts,
        total_bits=total_bits,
        total_bpb=total_bits / max(int(bpb.size), 1),
        adapter_wall_seconds=0.0,
        record_dtype=record_dtype,
        store_offsets=False,
        offset_start=0,
        statistics_sample_size=200000,
    )
    plot_payload = {
        "bpb": bpb,
        "offsets": relative_offsets,
        "model_window_means": means,
        "model_window_counts": counts,
    }
    return artifact_summary, plot_payload


def _compute_nc_prefix(
    *,
    sequence: str,
    species: str,
    region_start: int,
    region_bases: int,
    output_dir: Path,
    record_dtype: str,
    window_bases: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    min_windows = max(1, int(region_bases) // int(window_bases))
    started = perf_counter()
    result = compute_noncontiguous_prefix_probabilities(
        sequence,
        NoncontiguousPrefixConfig(
            window_bases=window_bases,
            alphabet="ACGT",
            min_windows=min_windows,
            backend="fast_cpp",
        ),
        return_probabilities=False,
    )
    elapsed = perf_counter() - started
    bpb = np.asarray(result.bpb, dtype=np.float64)
    offsets = np.arange(bpb.shape[0], dtype=np.int64)
    metadata = dict(result.metadata)
    metadata.update(
        {
            "species": species,
            "region_start": int(region_start),
            "region_bases": int(region_bases),
            "model_window_bases": int(window_bases),
            "wrapper_compute_seconds": float(elapsed),
        }
    )
    rows = window_rows(bpb, offsets, source_start=region_start, window_bases=window_bases)
    for row in rows:
        row["model"] = "nc_prefix_current"
        row["plot_window_bases"] = window_bases
    means, counts = model_window_average(bpb, offsets, window_bases)
    total_bits = float(np.sum(bpb))
    artifact_summary = write_model_artifact(
        output_dir,
        model_name="nc_prefix_current",
        bpb=bpb,
        offsets=offsets,
        metadata=metadata,
        window_summary=rows,
        worst_window=max(rows, key=lambda item: float(item["mean_bpb"])) if rows else None,
        model_window_means=means,
        model_window_counts=counts,
        total_bits=total_bits,
        total_bpb=total_bits / max(int(bpb.size), 1),
        adapter_wall_seconds=elapsed,
        record_dtype=record_dtype,
        store_offsets=False,
        offset_start=0,
        statistics_sample_size=200000,
    )
    plot_payload = {
        "bpb": bpb,
        "offsets": offsets,
        "model_window_means": means,
        "model_window_counts": counts,
    }
    return artifact_summary, plot_payload


def _plot_clear(
    *,
    output_dir: Path,
    species: str,
    region_start: int,
    region_bases: int,
    window_bases: int,
    plot_payloads: dict[str, dict[str, Any]],
    model_summaries: dict[str, dict[str, Any]],
    top_smooth_windows: int,
) -> Path:
    labels = {
        "geco22": "GECO2 full-source cached slice",
        "megabyte1": "MEGABYTE cached",
        "nc_prefix_current": "nc_prefix current",
    }
    colors = {
        "geco22": "#2ca02c",
        "megabyte1": "#ff7f0e",
        "nc_prefix_current": "#d62728",
    }
    window_count = (region_bases + window_bases - 1) // window_bases
    x_mid_mbase = (region_start + (np.arange(window_count, dtype=np.float64) * window_bases + window_bases / 2.0)) / 1_000_000.0
    x_mid_mbase[-1] = (region_start + ((window_count - 1) * window_bases + region_bases) / 2.0) / 1_000_000.0
    window_mean_by_model: dict[str, np.ndarray] = {}
    for model_name, payload in plot_payloads.items():
        bpb = np.asarray(payload["bpb"], dtype=np.float64)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        ids = offsets // window_bases
        sums = np.bincount(ids, weights=bpb, minlength=window_count)
        counts = np.bincount(ids, minlength=window_count)
        means = np.full((window_count,), np.nan, dtype=np.float64)
        active = counts > 0
        means[active] = sums[active] / counts[active]
        window_mean_by_model[model_name] = means

    csv_path = output_dir / "region_bpb_combined_window_means_clear.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["window_index", "source_midpoint_mbase"] + [f"{model}_bpb" for model in plot_payloads]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, midpoint in enumerate(x_mid_mbase):
            row = {"window_index": index, "source_midpoint_mbase": f"{midpoint:.9f}"}
            for model_name in plot_payloads:
                value = window_mean_by_model[model_name][index]
                row[f"{model_name}_bpb"] = f"{value:.9f}" if np.isfinite(value) else ""
            writer.writerow(row)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    ax = axes[0]
    for model_name in plot_payloads:
        y = window_mean_by_model[model_name]
        color = colors.get(model_name)
        mean = float(model_summaries.get(model_name, {}).get("total_bpb", np.nanmean(y)))
        ax.plot(x_mid_mbase, y, color=color, linewidth=0.7, alpha=0.14)
        smooth = moving_average(y, top_smooth_windows)
        ax.plot(
            x_mid_mbase,
            smooth,
            color=color,
            linewidth=2.4 if model_name == "nc_prefix_current" else 1.8,
            label=f"{labels.get(model_name, model_name)}  mean {mean:.3f}",
        )
        ax.axhline(mean, color=color, linestyle="--", linewidth=0.9, alpha=0.30)
    ax.set_title(
        f"{species} same-region BPB comparison, {region_bases:,} bp, "
        f"{window_count:,} windows ({top_smooth_windows}-window MA)"
    )
    ax.set_xlabel("Source position (Mbases)")
    ax.set_ylabel("bits/base")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_ylim(bottom=0.0)

    ax2 = axes[1]
    for model_name, payload in plot_payloads.items():
        means = np.asarray(payload["model_window_means"], dtype=np.float64)
        counts = np.asarray(payload["model_window_counts"], dtype=np.int64)
        active = np.isfinite(means) & (counts > 0)
        if not bool(np.any(active)):
            continue
        x = np.arange(means.shape[0])[active]
        color = colors.get(model_name)
        ax2.scatter(x, means[active], s=5, alpha=0.20 if model_name == "nc_prefix_current" else 0.14, color=color)
        smooth = smooth_defined_positions(means, counts, 64)
        ax2.plot(
            x,
            smooth[active],
            linewidth=2.1 if model_name == "nc_prefix_current" else 1.7,
            color=color,
            label=labels.get(model_name, model_name),
        )
    ax2.set_title(f"Average BPB by position inside the {window_bases} bp window")
    ax2.set_xlabel("Position in model window (bases)")
    ax2.set_ylabel("bits/base")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", framealpha=0.9)

    png_path = output_dir / "region_bpb_combined_curve_clear.png"
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    return png_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build long DNACorpus comparisons for nc_prefix current vs GECO2 and MEGABYTE.")
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--full-root", default="outputs/dna_megabyte_large_opengenome2_9/full_bpb_probe_dnacorpus")
    parser.add_argument("--output-dir", default="outputs/nc_prefix_current_vs_geco2_megabyte_dnacorpus_long_seed12345")
    parser.add_argument("--species", nargs="+", default=list(DEFAULT_SPECIES))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--window-bases", type=int, default=DEFAULT_WINDOW_BASES)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--record-dtype", choices=("float16", "float32", "float64"), default="float16")
    parser.add_argument("--top-smooth-windows", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    full_root = Path(args.full_root)
    target_bases = int(args.window_bases) * int(args.max_windows)
    summary_rows: list[dict[str, Any]] = []
    source_args = SimpleNamespace(
        dataset="dnacorpus",
        dataset_dir=args.dataset_dir,
        input_dir=None,
        source=None,
        species=None,
        sequence_source_mode="auto",
        multi_sequence_mode="separate",
        sequence_include=None,
    )

    for species in args.species:
        species_started = perf_counter()
        source_args.species = [species]
        source_info = resolve_region_sources(source_args, "ACGTN")[0]
        source_length = filtered_length(source_info["paths"], alphabet="ACGTN", fasta=bool(source_info["fasta"]))
        region_bases = min(int(source_length), target_bases)
        region_start = 0 if source_length <= target_bases else stable_random_start(
            source_name=species,
            source_length=source_length,
            region_bases=target_bases,
            seed=int(args.seed),
        )
        output_dir = output_root / species
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_json = output_dir / "region_bpb_combined.json"
        if combined_json.exists() and not args.overwrite:
            combined = json.loads(combined_json.read_text(encoding="utf-8"))
            row = {
                "species": species,
                "region_start": int(combined["region_start"]),
                "region_bases": int(combined["region_bases"]),
                "window_count": int(combined["region_window_count"]),
                "skipped_existing": True,
            }
            for model, model_summary in combined.get("models", {}).items():
                row[f"{model}_bpb"] = model_summary.get("total_bpb")
            summary_rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            continue

        region_sequence = extract_filtered_region(
            source_info["paths"],
            alphabet="ACGT",
            fasta=bool(source_info["fasta"]),
            start=region_start,
            length=region_bases,
        ).decode("ascii")
        if len(region_sequence) != region_bases:
            raise RuntimeError(f"{species}: expected {region_bases} ACGT bases, got {len(region_sequence)}")

        full_result = _load_full_result(full_root, species)
        model_summaries: dict[str, dict[str, Any]] = {}
        plot_payloads: dict[str, dict[str, Any]] = {}
        for model_name in ("geco22", "megabyte1"):
            model_summary, plot_payload = _slice_full_model(
                full_result=full_result,
                model_name=model_name,
                region_start=region_start,
                region_bases=region_bases,
                output_dir=output_dir,
                record_dtype=str(args.record_dtype),
                window_bases=int(args.window_bases),
            )
            model_summaries[model_name] = model_summary
            plot_payloads[model_name] = plot_payload

        nc_summary, nc_payload = _compute_nc_prefix(
            sequence=region_sequence,
            species=species,
            region_start=region_start,
            region_bases=region_bases,
            output_dir=output_dir,
            record_dtype=str(args.record_dtype),
            window_bases=int(args.window_bases),
        )
        model_summaries["nc_prefix_current"] = nc_summary
        plot_payloads["nc_prefix_current"] = nc_payload

        curve_png = _plot_clear(
            output_dir=output_dir,
            species=species,
            region_start=region_start,
            region_bases=region_bases,
            window_bases=int(args.window_bases),
            plot_payloads=plot_payloads,
            model_summaries=model_summaries,
            top_smooth_windows=int(args.top_smooth_windows),
        )
        combined = {
            "combined_plot": True,
            "dataset": "dnacorpus",
            "source": species,
            "species": species,
            "region_start": int(region_start),
            "region_bases": int(region_bases),
            "region_window_count": int((region_bases + int(args.window_bases) - 1) // int(args.window_bases)),
            "source_length": int(source_length),
            "window_bases": int(args.window_bases),
            "max_windows": int(args.max_windows),
            "seed": int(args.seed),
            "models": model_summaries,
            "model_count": len(model_summaries),
            "outputs": {
                "json": str(combined_json),
                "clear_curve_png": str(curve_png),
                "clear_window_mean_csv": str(output_dir / "region_bpb_combined_window_means_clear.csv"),
                "clear_top_smooth_windows": int(args.top_smooth_windows),
            },
        }
        _write_json(combined_json, combined)
        row = {
            "species": species,
            "source_length": int(source_length),
            "region_start": int(region_start),
            "region_bases": int(region_bases),
            "window_count": int(combined["region_window_count"]),
            "geco22_bpb": model_summaries["geco22"]["total_bpb"],
            "megabyte1_bpb": model_summaries["megabyte1"]["total_bpb"],
            "nc_prefix_current_bpb": model_summaries["nc_prefix_current"]["total_bpb"],
            "nc_prefix_compute_seconds": model_summaries["nc_prefix_current"]["adapter_wall_seconds"],
            "wall_seconds": perf_counter() - species_started,
            "curve_png": str(curve_png),
        }
        summary_rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    write_csv(output_root / "summary.csv", summary_rows)
    _write_json(
        output_root / "summary.json",
        {
            "species_count": len(summary_rows),
            "window_bases": int(args.window_bases),
            "max_windows": int(args.max_windows),
            "target_bases": int(target_bases),
            "rows": summary_rows,
        },
    )


if __name__ == "__main__":
    main()
