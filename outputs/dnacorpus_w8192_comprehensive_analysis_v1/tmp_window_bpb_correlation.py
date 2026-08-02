#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import (  # noqa: E402
    ProbabilityTraceManifest,
    ProbabilityTraceReader,
    TRACE_EMISSION_ORDER_POSITION_MAJOR_V1,
)


ANALYSIS_ROOT = REPO_ROOT / "outputs" / "dnacorpus_w8192_comprehensive_analysis_v1"
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "window_bpb_correlation_v1"

TRACE_ROOTS = {
    "carbon3b": REPO_ROOT / "outputs" / "carbon3b_dnacorpus_w8192_target_traces_position_major" / "traces",
    "evo2_7b": REPO_ROOT
    / "outputs"
    / "evo2_7b_dnacorpus_w8192_full_forward_bs12_target_traces_position_major"
    / "traces",
    "nc_prefix": REPO_ROOT
    / "outputs"
    / "nc_prefix_dnacorpus_best_available_w8192_target_traces_full_position_major"
    / "traces",
}

CORRECTED_TRACE_ROOTS = {
    "carbon3b": REPO_ROOT
    / "outputs"
    / "carbon3b_corrected_supplement_w8192_full_forward_target_traces_position_major"
    / "traces",
    "evo2_7b": REPO_ROOT
    / "outputs"
    / "evo2_7b_corrected_supplement_w8192_full_forward_bs12_target_traces_position_major"
    / "traces",
    "nc_prefix": REPO_ROOT
    / "outputs"
    / "nc_prefix_dnacorpus_corrected_supplement_w8192_target_traces_position_major"
    / "traces",
}

MODEL_LABELS = {
    "carbon3b": "Carbon 3B",
    "evo2_7b": "Evo2 7B",
    "nc_prefix": "nc_prefix",
}

MODEL_COLORS = {
    "carbon3b": "#E69F00",
    "evo2_7b": "#5B5FC7",
    "nc_prefix": "#0072B2",
}

PAIRS = [
    ("carbon3b__nc_prefix", "carbon3b", "nc_prefix", "Carbon 3B vs nc_prefix"),
    ("evo2_7b__nc_prefix", "evo2_7b", "nc_prefix", "Evo2 7B vs nc_prefix"),
    ("carbon3b__evo2_7b", "carbon3b", "evo2_7b", "Carbon 3B vs Evo2 7B"),
]

DEFAULT_SPECIES_ORDER = [
    "HoSa",
    "GaGa",
    "AnCa",
    "DaRe",
    "OrSa",
    "DrMe",
    "EnIn",
    "ScPo",
    "WaMe",
    "PlFa",
    "EsCo",
    "HaHi",
    "HePy",
    "AeCa",
    "YeMi",
    "AgPh",
    "BuEb",
]

CORRECTED_SUPPLEMENT_SPECIES = {"EnIn", "HePy", "PlFa", "ScPo"}
SCOPES = {
    "all17": lambda species: True,
    "main13_excluding_corrected_supplement": lambda species: species not in CORRECTED_SUPPLEMENT_SPECIES,
}

EPS_STD = 1e-12


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _read_manifest(trace_dir: Path) -> ProbabilityTraceManifest:
    return ProbabilityTraceReader(trace_dir).manifest


def _trace_dir(model: str, species: str) -> Path:
    if species in CORRECTED_SUPPLEMENT_SPECIES:
        return CORRECTED_TRACE_ROOTS[model] / species
    return TRACE_ROOTS[model] / species


def _available_species() -> list[str]:
    species_sets = []
    for model in TRACE_ROOTS:
        available = {path.name for path in TRACE_ROOTS[model].iterdir() if (path / "manifest.json").exists()}
        corrected_available = {
            path.name for path in CORRECTED_TRACE_ROOTS[model].iterdir() if (path / "manifest.json").exists()
        }
        available.difference_update(CORRECTED_SUPPLEMENT_SPECIES)
        available.update(corrected_available.intersection(CORRECTED_SUPPLEMENT_SPECIES))
        species_sets.append(available)
    common = set.intersection(*species_sets)
    ordered = [species for species in DEFAULT_SPECIES_ORDER if species in common]
    ordered.extend(sorted(common.difference(ordered)))
    return ordered


def _validate_manifests(species: str) -> dict[str, ProbabilityTraceManifest]:
    manifests = {model: _read_manifest(_trace_dir(model, species)) for model in TRACE_ROOTS}
    baseline_model = "carbon3b"
    baseline = manifests[baseline_model]
    fields = ["core_base_count", "row_count", "window_bases", "emission_order"]
    for model, manifest in manifests.items():
        for field in fields:
            left = getattr(baseline, field)
            right = getattr(manifest, field)
            if left != right:
                raise ValueError(
                    f"{species}: {model}.{field}={right} does not match "
                    f"{baseline_model}.{field}={left}"
                )
        if manifest.emission_order != TRACE_EMISSION_ORDER_POSITION_MAJOR_V1:
            raise ValueError(f"{species}: expected position-major trace for {model}, got {manifest.emission_order}")
    return manifests


def _compute_window_bpb(
    trace_dir: Path,
    *,
    verify_checksum: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reader = ProbabilityTraceReader(trace_dir)
    manifest = reader.manifest
    window_bases = int(manifest.window_bases)
    core_base_count = int(manifest.core_base_count)
    window_count = int(math.ceil(core_base_count / window_bases)) if core_base_count else 0
    window_bit_sums = np.zeros((window_count,), dtype=np.float64)
    window_counts = np.zeros((window_count,), dtype=np.int64)
    total_bits = 0.0
    total_count = 0
    started = perf_counter()
    for shard in reader.iter_shards(verify_checksum=verify_checksum):
        target_prob = np.asarray(shard["target_prob"], dtype=np.float64)
        bits = -np.log2(np.clip(target_prob, np.finfo(np.float32).tiny, 1.0))
        emit_position = np.asarray(shard["emit_position"], dtype=np.int64)
        window_ids = emit_position // window_bases
        total_bits += float(np.sum(bits, dtype=np.float64))
        total_count += int(bits.shape[0])
        window_bit_sums += np.bincount(window_ids, weights=bits, minlength=window_count)[:window_count]
        window_counts += np.bincount(window_ids, minlength=window_count)[:window_count].astype(np.int64)
    window_bpb = np.divide(
        window_bit_sums,
        window_counts,
        out=np.full((window_count,), np.nan, dtype=np.float64),
        where=window_counts > 0,
    )
    stats = {
        "core_base_count": core_base_count,
        "row_count": int(manifest.row_count),
        "window_bases": window_bases,
        "window_count": window_count,
        "full_bpb": float(total_bits / total_count) if total_count else float("nan"),
        "load_seconds": perf_counter() - started,
    }
    return window_bpb, window_counts, stats


def _load_or_build_species_cache(
    species: str,
    *,
    output_dir: Path,
    verify_checksum: bool,
    rebuild_cache: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    cache_dir = output_dir / "window_bpb_npz"
    cache_path = cache_dir / f"{species}.npz"
    manifests = _validate_manifests(species)
    manifest_signature = {
        model: {
            "trace_dir": str(TRACE_ROOTS[model] / species),
            "selected_trace_dir": str(_trace_dir(model, species)),
            "checksum_sha256": manifest.checksum_sha256,
            "core_base_count": int(manifest.core_base_count),
            "row_count": int(manifest.row_count),
            "window_bases": int(manifest.window_bases),
            "emission_order": manifest.emission_order,
        }
        for model, manifest in manifests.items()
    }
    if cache_path.exists() and not rebuild_cache:
        with np.load(cache_path, allow_pickle=False) as data:
            cached_signature = json.loads(str(data["manifest_signature"].item()))
            if cached_signature == manifest_signature:
                arrays = {model: np.asarray(data[f"{model}_window_bpb"], dtype=np.float64) for model in TRACE_ROOTS}
                counts = np.asarray(data["window_counts"], dtype=np.int64)
                cache_stats = json.loads(str(data["stats"].item()))
                cache_stats["cache_hit"] = True
                return arrays, counts, cache_stats

    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    counts_by_model: dict[str, np.ndarray] = {}
    model_stats: dict[str, Any] = {}
    for model, root in TRACE_ROOTS.items():
        window_bpb, window_counts, stats = _compute_window_bpb(
            _trace_dir(model, species), verify_checksum=verify_checksum
        )
        arrays[model] = window_bpb
        counts_by_model[model] = window_counts
        model_stats[model] = stats

    baseline_counts = counts_by_model["carbon3b"]
    for model, counts in counts_by_model.items():
        if counts.shape != baseline_counts.shape or not np.array_equal(counts, baseline_counts):
            raise ValueError(f"{species}: window count layout mismatch for {model}")
        if arrays[model].shape != arrays["carbon3b"].shape:
            raise ValueError(f"{species}: window bpb shape mismatch for {model}")

    stats_payload = {
        "cache_hit": False,
        "models": model_stats,
        "core_base_count": int(manifests["carbon3b"].core_base_count),
        "row_count": int(manifests["carbon3b"].row_count),
        "window_bases": int(manifests["carbon3b"].window_bases),
        "window_count": int(arrays["carbon3b"].shape[0]),
    }
    np.savez_compressed(
        cache_path,
        manifest_signature=np.asarray(json.dumps(manifest_signature, sort_keys=True)),
        stats=np.asarray(json.dumps(stats_payload, sort_keys=True)),
        window_counts=baseline_counts,
        **{f"{model}_window_bpb": values for model, values in arrays.items()},
    )
    return arrays, baseline_counts, stats_payload


def _zscore(values: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    finite = np.isfinite(values)
    valid_n = int(np.count_nonzero(finite))
    z = np.full(values.shape, np.nan, dtype=np.float64)
    if valid_n < 2:
        return z, float("nan"), float("nan"), valid_n
    mean = float(np.mean(values[finite], dtype=np.float64))
    std = float(np.std(values[finite], dtype=np.float64))
    if not np.isfinite(std) or std <= EPS_STD:
        return z, mean, std, valid_n
    z[finite] = (values[finite] - mean) / std
    return z, mean, std, valid_n


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int, str]:
    finite = np.isfinite(x) & np.isfinite(y)
    valid_n = int(np.count_nonzero(finite))
    if valid_n < 2:
        return float("nan"), valid_n, "insufficient_windows"
    x_valid = x[finite].astype(np.float64, copy=False)
    y_valid = y[finite].astype(np.float64, copy=False)
    x_std = float(np.std(x_valid))
    y_std = float(np.std(y_valid))
    if x_std <= EPS_STD or y_std <= EPS_STD:
        return float("nan"), valid_n, "insufficient_variance"
    return float(np.corrcoef(x_valid, y_valid)[0, 1]), valid_n, "ok"


def _species_rows(
    species: str,
    arrays: dict[str, np.ndarray],
    counts: np.ndarray,
    stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    z_by_model: dict[str, np.ndarray] = {}
    model_summary: dict[str, Any] = {}
    for model, values in arrays.items():
        z, mean, std, valid_n = _zscore(values)
        z_by_model[model] = z
        model_summary[model] = {
            "mean_window_bpb": mean,
            "std_window_bpb": std,
            "valid_windows": valid_n,
            "full_bpb": float(stats.get("models", {}).get(model, {}).get("full_bpb", float("nan"))),
        }

    rows: list[dict[str, Any]] = []
    core_base_count = int(stats["core_base_count"])
    window_count = int(stats["window_count"])
    for pair_key, model_a, model_b, pair_label in PAIRS:
        raw_r, raw_n, raw_status = _pearson(arrays[model_a], arrays[model_b])
        z_r, z_n, z_status = _pearson(z_by_model[model_a], z_by_model[model_b])
        rows.append(
            {
                "species": species,
                "pair_key": pair_key,
                "pair_label": pair_label,
                "model_a": model_a,
                "model_b": model_b,
                "model_a_label": MODEL_LABELS[model_a],
                "model_b_label": MODEL_LABELS[model_b],
                "core_base_count": core_base_count,
                "window_bases": int(stats["window_bases"]),
                "window_count": window_count,
                "effective_windows": z_n,
                "pearson_z_window_bpb_corr": z_r,
                "pearson_raw_window_bpb_corr": raw_r,
                "status": z_status if z_status != "ok" else raw_status,
                "model_a_mean_window_bpb": model_summary[model_a]["mean_window_bpb"],
                "model_a_std_window_bpb": model_summary[model_a]["std_window_bpb"],
                "model_b_mean_window_bpb": model_summary[model_b]["mean_window_bpb"],
                "model_b_std_window_bpb": model_summary[model_b]["std_window_bpb"],
                "model_a_full_bpb": model_summary[model_a]["full_bpb"],
                "model_b_full_bpb": model_summary[model_b]["full_bpb"],
            }
        )
    return rows, z_by_model, model_summary


def _fisher_weighted(rows: list[dict[str, Any]], *, corr_field: str) -> float:
    zs: list[float] = []
    weights: list[float] = []
    for row in rows:
        r = row.get(corr_field)
        weight = float(row.get("effective_windows") or 0)
        if r is None or not np.isfinite(float(r)) or weight <= 0:
            continue
        clipped = float(np.clip(float(r), -0.999999999999, 0.999999999999))
        zs.append(float(np.arctanh(clipped)))
        weights.append(weight)
    if not zs:
        return float("nan")
    return float(np.tanh(np.average(np.asarray(zs, dtype=np.float64), weights=np.asarray(weights, dtype=np.float64))))


def _weighted_rows(per_species_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, predicate in SCOPES.items():
        for pair_key, model_a, model_b, pair_label in PAIRS:
            rows = [
                row
                for row in per_species_rows
                if row["pair_key"] == pair_key
                and predicate(str(row["species"]))
                and np.isfinite(float(row["pearson_z_window_bpb_corr"]))
            ]
            output.append(
                {
                    "scope": scope,
                    "pair_key": pair_key,
                    "pair_label": pair_label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_label": MODEL_LABELS[model_a],
                    "model_b_label": MODEL_LABELS[model_b],
                    "species_count": len(rows),
                    "total_effective_windows": int(sum(int(row["effective_windows"]) for row in rows)),
                    "total_core_base_count": int(sum(int(row["core_base_count"]) for row in rows)),
                    "weighted_pearson_z_window_bpb_corr": _fisher_weighted(
                        rows, corr_field="pearson_z_window_bpb_corr"
                    ),
                    "weighted_pearson_raw_window_bpb_corr": _fisher_weighted(
                        rows, corr_field="pearson_raw_window_bpb_corr"
                    ),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["species"]), str(row["pair_key"])): row for row in rows}


def _weighted_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(row["scope"]), str(row["pair_key"])): row for row in rows}


def _plot_heatmap(
    output_path: Path,
    *,
    scope: str,
    species_order: list[str],
    per_species_rows: list[dict[str, Any]],
    weighted_rows: list[dict[str, Any]],
) -> None:
    row_lookup = _row_lookup(per_species_rows)
    weighted = _weighted_lookup(weighted_rows)
    scope_predicate = SCOPES[scope]
    rows = [species for species in species_order if scope_predicate(species)] + ["weighted"]
    columns = [pair_label for _, _, _, pair_label in PAIRS]
    values = np.full((len(rows), len(columns)), np.nan, dtype=np.float64)
    for row_index, species in enumerate(rows):
        for col_index, (pair_key, _, _, _) in enumerate(PAIRS):
            if species == "weighted":
                item = weighted.get((scope, pair_key))
                if item is not None:
                    values[row_index, col_index] = float(item["weighted_pearson_z_window_bpb_corr"])
            else:
                item = row_lookup.get((species, pair_key))
                if item is not None:
                    values[row_index, col_index] = float(item["pearson_z_window_bpb_corr"])

    fig_height = max(4.5, 0.42 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(9.2, fig_height), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#E5E5E5")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            value = values[row_index, col_index]
            text = "" if not np.isfinite(value) else f"{value:.2f}"
            ax.text(col_index, row_index, text, ha="center", va="center", fontsize=9, color="#111111")
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label("Pearson r of z-scored window bpb", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_weighted_bar(output_path: Path, weighted_rows: list[dict[str, Any]]) -> None:
    scopes = ["all17", "main13_excluding_corrected_supplement"]
    scope_labels = ["all17", "main13"]
    x = np.arange(len(PAIRS), dtype=np.float64)
    width = 0.34
    lookup = _weighted_lookup(weighted_rows)
    fig, ax = plt.subplots(figsize=(9.4, 4.6), constrained_layout=True)
    for offset, scope, label, color in [
        (-width / 2, scopes[0], scope_labels[0], "#555555"),
        (width / 2, scopes[1], scope_labels[1], "#009E73"),
    ]:
        values = [
            float(lookup.get((scope, pair_key), {}).get("weighted_pearson_z_window_bpb_corr", float("nan")))
            for pair_key, _, _, _ in PAIRS
        ]
        bars = ax.bar(x + offset, values, width=width, label=label, color=color, alpha=0.86)
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.025 if value >= 0 else -0.045),
                    f"{value:.2f}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=10,
                )
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Weighted Pearson r")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, _, label in PAIRS], rotation=12, ha="right")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _scatter_sample(x: np.ndarray, y: np.ndarray, max_points: int = 20000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    indices = np.flatnonzero(finite)
    if indices.shape[0] > max_points:
        pick = np.linspace(0, indices.shape[0] - 1, num=max_points, dtype=np.int64)
        indices = indices[pick]
    order = indices.astype(np.float64)
    if order.size:
        order = order / max(float(np.max(order)), 1.0)
    return x[indices], y[indices], order


def _plot_representative_scatter(
    output_path: Path,
    *,
    species: str,
    z_by_species: dict[str, dict[str, np.ndarray]],
    per_species_rows: list[dict[str, Any]],
) -> None:
    if species not in z_by_species:
        return
    lookup = _row_lookup(per_species_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), constrained_layout=True)
    for ax, (pair_key, model_a, model_b, pair_label) in zip(axes, PAIRS):
        x, y, order = _scatter_sample(z_by_species[species][model_a], z_by_species[species][model_b])
        ax.scatter(x, y, c=order, cmap="viridis", s=6, alpha=0.32, linewidths=0)
        ax.axhline(0.0, color="#BBBBBB", linewidth=0.7)
        ax.axvline(0.0, color="#BBBBBB", linewidth=0.7)
        row = lookup.get((species, pair_key), {})
        corr = row.get("pearson_z_window_bpb_corr", float("nan"))
        corr_text = "r=nan" if not np.isfinite(float(corr)) else f"r={float(corr):.3f}"
        ax.set_title(f"{pair_label}\n{corr_text}", fontsize=10)
        ax.set_xlabel(f"{MODEL_LABELS[model_a]} z-bpb")
        ax.set_ylabel(f"{MODEL_LABELS[model_b]} z-bpb")
        ax.grid(color="#EEEEEE", linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.species:
        species_order = list(args.species)
    elif args.smoke:
        species_order = ["AgPh", "BuEb", "OrSa"]
    else:
        species_order = _available_species()

    missing = {
        model: [species for species in species_order if not (_trace_dir(model, species) / "manifest.json").exists()]
        for model in TRACE_ROOTS
    }
    missing = {model: items for model, items in missing.items() if items}
    if missing:
        raise FileNotFoundError(f"missing trace manifests: {missing}")

    per_species_rows: list[dict[str, Any]] = []
    z_by_species: dict[str, dict[str, np.ndarray]] = {}
    species_summaries: dict[str, Any] = {}
    started = perf_counter()
    for index, species in enumerate(species_order, start=1):
        print(f"[{index}/{len(species_order)}] loading {species}", flush=True)
        arrays, counts, stats = _load_or_build_species_cache(
            species,
            output_dir=output_dir,
            verify_checksum=bool(args.verify_shard_checksum),
            rebuild_cache=bool(args.rebuild_cache),
        )
        rows, z_by_model, model_summary = _species_rows(species, arrays, counts, stats)
        per_species_rows.extend(rows)
        z_by_species[species] = z_by_model
        species_summaries[species] = {
            "stats": stats,
            "model_summary": model_summary,
            "window_count": int(counts.shape[0]),
            "nonempty_windows": int(np.count_nonzero(counts > 0)),
        }

    weighted = _weighted_rows(per_species_rows)
    _write_csv(output_dir / "per_species_window_bpb_correlation.csv", per_species_rows)
    _write_csv(output_dir / "weighted_window_bpb_correlation.csv", weighted)

    _plot_heatmap(
        output_dir / "window_bpb_corr_heatmap_all17.png",
        scope="all17",
        species_order=species_order,
        per_species_rows=per_species_rows,
        weighted_rows=weighted,
    )
    _plot_heatmap(
        output_dir / "window_bpb_corr_heatmap_main13.png",
        scope="main13_excluding_corrected_supplement",
        species_order=species_order,
        per_species_rows=per_species_rows,
        weighted_rows=weighted,
    )
    _plot_weighted_bar(output_dir / "weighted_window_bpb_corr_bar.png", weighted)
    for species in ["OrSa", "HoSa", "EsCo"]:
        _plot_representative_scatter(
            output_dir / f"representative_scatter_{species}.png",
            species=species,
            z_by_species=z_by_species,
            per_species_rows=per_species_rows,
        )

    manifest = {
        "script": str(SCRIPT_PATH),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "trace_roots": TRACE_ROOTS,
        "corrected_trace_roots": CORRECTED_TRACE_ROOTS,
        "model_labels": MODEL_LABELS,
        "pairs": [
            {"pair_key": pair_key, "model_a": model_a, "model_b": model_b, "pair_label": label}
            for pair_key, model_a, model_b, label in PAIRS
        ],
        "species_order": species_order,
        "corrected_supplement_species": sorted(CORRECTED_SUPPLEMENT_SPECIES),
        "scopes": list(SCOPES.keys()),
        "verify_shard_checksum": bool(args.verify_shard_checksum),
        "rebuild_cache": bool(args.rebuild_cache),
        "elapsed_seconds": perf_counter() - started,
        "species_summaries": species_summaries,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute DNACorpus three-model window-bpb correlations from target-probability traces."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--species", nargs="*", default=None, help="Optional species subset in output order.")
    parser.add_argument("--smoke", action="store_true", help="Run only AgPh, BuEb, and OrSa.")
    parser.add_argument("--verify-shard-checksum", action="store_true", help="Verify trace shard checksums while loading.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute cached window-bpb arrays.")
    return parser.parse_args()


def main() -> None:
    manifest = run(parse_args())
    print(f"wrote {manifest['output_dir']}")


if __name__ == "__main__":
    main()
