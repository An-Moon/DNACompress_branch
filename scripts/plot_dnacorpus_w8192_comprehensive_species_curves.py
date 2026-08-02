#!/usr/bin/env python3
from __future__ import annotations

"""Plot per-species window-bpb curves from target-probability traces.

Examples:

    python scripts/plot_dnacorpus_w8192_comprehensive_species_curves.py \
      --summary-json outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full/nc_prefix_dnacorpus_full_w8192_summary.json \
      --output-dir outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full/species_curves

    python scripts/plot_dnacorpus_w8192_comprehensive_species_curves.py \
      --summary-json outputs/carbon3b_dnacorpus_w8192_target_traces/trace_summary.json \
      --output-dir outputs/carbon3b_dnacorpus_w8192_target_traces/species_curves_v3 \
      --model-label "Carbon 3B" \
      --output-prefix carbon3b_w8192

    python scripts/plot_dnacorpus_w8192_comprehensive_species_curves.py \
      --summary-json outputs/carbon3b_dnacorpus_w8192_target_traces/trace_summary.json \
      --compare-summary-json outputs/nc_prefix_dnacorpus_best_available_w8192_target_traces_full/nc_prefix_dnacorpus_full_w8192_summary.json \
      --output-dir outputs/dnacorpus_w8192_comprehensive_analysis_v1 \
      --model-label "Carbon 3B" \
      --compare-model-label nc_prefix \
      --output-prefix dnacorpus_w8192
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.probability_trace import (  # noqa: E402
    ProbabilityTraceReader,
    _read_trace_rows_by_indices,
    validate_trace_compatibility,
)


DEFAULT_ORDER = [
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot per-species nc_prefix trace bpb curves.")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-csv", help="Aggregate CSV path. Defaults to <output-dir>/<output-prefix>_species_curve_summary.csv.")
    parser.add_argument("--curve-data-dir", help="Compressed per-species curve npz directory. Defaults to <output-dir>/curve_data.")
    parser.add_argument("--model-label", default="nc_prefix", help="Model label used in plot titles and legends.")
    parser.add_argument(
        "--output-prefix",
        default="dnacorpus_w8192",
        help="Filename prefix for plots, caches, and summaries.",
    )
    parser.add_argument("--compare-summary-json", help="Optional second summary JSON. When set, plots both traces on each species figure.")
    parser.add_argument("--compare-curve-data-dir", help="Optional curve npz directory for the comparison trace.")
    parser.add_argument("--compare-model-label", default="comparison", help="Model label for --compare-summary-json.")
    parser.add_argument(
        "--fusion-summary-csv",
        help="Optional online-hedge fusion CSV with trace_left/trace_right columns; adds a third curve.",
    )
    parser.add_argument("--fusion-curve-data-dir", help="Optional curve npz directory for fusion curves.")
    parser.add_argument("--fusion-model-label", default="fused", help="Model label for --fusion-summary-csv.")
    parser.add_argument("--extra-summary-json", action="append", default=[], help="Additional trace summary JSON to overlay. May be repeated.")
    parser.add_argument("--extra-model-label", action="append", default=[], help="Label for each --extra-summary-json, in the same order.")
    parser.add_argument("--extra-curve-data-dir", action="append", default=[], help="Curve npz directory for each --extra-summary-json.")
    parser.add_argument("--extra-fusion-summary-csv", action="append", default=[], help="Additional fusion CSV to overlay. May be repeated.")
    parser.add_argument(
        "--extra-fusion-model-label",
        action="append",
        default=[],
        help="Label for each --extra-fusion-summary-csv, in the same order.",
    )
    parser.add_argument("--extra-fusion-curve-data-dir", action="append", default=[], help="Curve npz directory for each --extra-fusion-summary-csv.")
    parser.add_argument("--reuse-curve-data", action=argparse.BooleanOptionalAction, default=True, help="Reuse existing curve npz files when present.")
    parser.add_argument("--verify-shard-checksum", action="store_true", help="Verify full trace checksum while plotting.")
    parser.add_argument("--rolling-windows", type=int, default=64, help="Rolling mean width for visual smoothing.")
    parser.add_argument(
        "--dense-full-position-window-threshold",
        type=int,
        default=1024,
        help="Use every emitted position in the top full-sequence curve when the species has at most this many windows.",
    )
    parser.add_argument(
        "--dense-full-position-max-bases",
        type=int,
        default=5_000_000,
        help="Maximum core bases for caching and plotting every emitted position in the top full-sequence curve.",
    )
    parser.add_argument(
        "--dense-position-smooth-bases",
        type=int,
        default=0,
        help=(
            "Rolling mean width, in bases, for dense per-position full-sequence curves. "
            "Use 0 for an automatic width based on species length."
        ),
    )
    parser.add_argument(
        "--dense-plot-max-points",
        type=int,
        default=4096,
        help="Maximum points used to render dense smoothed full-sequence curves; data and smoothing still use all positions.",
    )
    return parser


def _as_path(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base / path


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    width = int(width)
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.shape[0])
    if width <= 1 or n == 0:
        return arr.copy()
    left = width // 2
    right = width - left
    starts = np.maximum(0, np.arange(n, dtype=np.int64) - left)
    ends = np.minimum(n, np.arange(n, dtype=np.int64) + right)
    finite = np.isfinite(arr)
    clean = np.where(finite, arr, 0.0)
    prefix = np.empty((n + 1,), dtype=np.float64)
    count_prefix = np.empty((n + 1,), dtype=np.int64)
    prefix[0] = 0.0
    count_prefix[0] = 0
    np.cumsum(clean, dtype=np.float64, out=prefix[1:])
    np.cumsum(finite.astype(np.int64), dtype=np.int64, out=count_prefix[1:])
    sums = prefix[ends] - prefix[starts]
    counts = count_prefix[ends] - count_prefix[starts]
    return np.divide(sums, counts, out=np.full((n,), np.nan, dtype=np.float64), where=counts > 0)


def _effective_dense_position_smooth_bases(requested_bases: int, sample_bases: int) -> int:
    requested_bases = int(requested_bases)
    if requested_bases > 0:
        return requested_bases
    # Match the calmer visual role of the 64-window moving average used for large species,
    # while avoiding a nearly flat line on the shortest viral/phage sequences.
    return int(max(4096, min(131072, int(sample_bases) // 32)))


def _thin_line_for_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    max_points = int(max_points)
    if max_points <= 0 or x_arr.shape[0] <= max_points:
        return x_arr, y_arr
    indices = np.linspace(0, x_arr.shape[0] - 1, num=max_points, dtype=np.int64)
    return x_arr[indices], y_arr[indices]


def _trace_curve_stats(
    trace_dir: Path,
    *,
    verify_checksum: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    reader = ProbabilityTraceReader(trace_dir)
    manifest = reader.manifest
    window_bases = int(manifest.window_bases)
    window_count = int(math.ceil(int(manifest.core_base_count) / window_bases)) if manifest.core_base_count else 0
    window_bit_sums = np.zeros((window_count,), dtype=np.float64)
    window_counts = np.zeros((window_count,), dtype=np.int64)
    offset_bit_sums = np.zeros((window_bases,), dtype=np.float64)
    offset_counts = np.zeros((window_bases,), dtype=np.int64)
    started = perf_counter()
    total_bits = 0.0
    total_count = 0
    for shard in reader.iter_shards(verify_checksum=verify_checksum):
        target_prob = np.asarray(shard["target_prob"], dtype=np.float64)
        bits = -np.log2(np.clip(target_prob, np.finfo(np.float32).tiny, 1.0))
        emit_position = np.asarray(shard["emit_position"], dtype=np.int64)
        total_bits += float(np.sum(bits, dtype=np.float64))
        total_count += int(bits.shape[0])
        window_ids = emit_position // window_bases
        offsets = emit_position % window_bases
        window_bit_sums += np.bincount(window_ids, weights=bits, minlength=window_count)[:window_count]
        window_counts += np.bincount(window_ids, minlength=window_count)[:window_count].astype(np.int64)
        offset_bit_sums += np.bincount(offsets, weights=bits, minlength=window_bases)[:window_bases]
        offset_counts += np.bincount(offsets, minlength=window_bases)[:window_bases].astype(np.int64)
    window_bpb = np.divide(
        window_bit_sums,
        window_counts,
        out=np.full_like(window_bit_sums, np.nan),
        where=window_counts > 0,
    )
    offset_bpb = np.divide(
        offset_bit_sums,
        offset_counts,
        out=np.full_like(offset_bit_sums, np.nan),
        where=offset_counts > 0,
    )
    midpoint_mbase = (
        np.arange(window_count, dtype=np.float64) * window_bases + (window_counts.astype(np.float64) / 2.0)
    ) / 1e6
    full_bpb = float(total_bits / total_count) if total_count else float("nan")
    return midpoint_mbase, window_bpb, np.arange(window_bases, dtype=np.int64), offset_bpb, full_bpb, perf_counter() - started


def _fusion_trace_curve_stats(
    left_trace_dir: Path,
    right_trace_dir: Path,
    *,
    fusion_eta: float,
    fusion_initial_left_weight: float,
    dense_full_position: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, float, float]:
    left = ProbabilityTraceReader(left_trace_dir)
    right = ProbabilityTraceReader(right_trace_dir)
    diffs = validate_trace_compatibility(left.manifest, right.manifest)
    if diffs:
        raise ValueError(f"fusion trace compatibility failed for {left_trace_dir} vs {right_trace_dir}: {diffs}")
    if left.manifest.emission_order != "position_major_v1":
        raise ValueError("fusion curve plotting expects position_major_v1 traces")

    window_bases = int(left.manifest.window_bases)
    core_base_count = int(left.manifest.core_base_count)
    window_count = int(math.ceil(core_base_count / window_bases)) if core_base_count else 0
    window_bit_sums = np.zeros((window_count,), dtype=np.float64)
    window_counts = np.zeros((window_count,), dtype=np.int64)
    offset_bit_sums = np.zeros((window_bases,), dtype=np.float64)
    offset_counts = np.zeros((window_bases,), dtype=np.int64)
    dense_bpb = np.full((core_base_count,), np.nan, dtype=np.float32) if dense_full_position else None

    left_weights = np.full((window_count,), float(fusion_initial_left_weight), dtype=np.float64)
    right_weights = 1.0 - left_weights
    eta = float(fusion_eta)
    eta_power = 1.0 - eta
    fused_bits = 0.0
    emitted = 0
    started = perf_counter()

    block_windows = 512
    for window_start in range(0, window_count, block_windows):
        window_end = min(window_count, window_start + block_windows)
        start_position = int(window_start * window_bases)
        end_position = min(core_base_count, int(window_end * window_bases))
        positions = np.arange(start_position, end_position, dtype=np.int64)
        left_chunk = _read_trace_rows_by_indices(left_trace_dir, left.manifest, positions)
        right_chunk = _read_trace_rows_by_indices(right_trace_dir, right.manifest, positions)
        if not np.array_equal(left_chunk["target_symbol"], right_chunk["target_symbol"]):
            raise ValueError("fusion trace target symbols diverged")
        if not np.array_equal(left_chunk["emit_position"], right_chunk["emit_position"]):
            raise ValueError("fusion trace emit positions diverged")
        left_prob = np.asarray(left_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
        right_prob = np.asarray(right_chunk["target_prob"], dtype=np.float64).clip(min=1e-300)
        local_window_count = int(window_end - window_start)
        local_base_count = int(end_position - start_position)
        local_windows = np.arange(local_window_count, dtype=np.int64)
        for offset in range(window_bases):
            local_rows = local_windows * window_bases + int(offset)
            valid = local_rows < local_base_count
            if not bool(np.any(valid)):
                break
            active_rows = local_rows[valid]
            window_ids = int(window_start) + local_windows[valid]
            left_weight = left_weights[window_ids]
            right_weight = right_weights[window_ids]
            left_target = left_prob[active_rows]
            right_target = right_prob[active_rows]
            fused_target = np.maximum(left_weight * left_target + right_weight * right_target, 1e-300)
            bits = -np.log2(fused_target)

            window_bit_sums[window_ids] += bits
            window_counts[window_ids] += 1
            offset_bit_sums[offset] += float(np.sum(bits, dtype=np.float64))
            offset_counts[offset] += int(bits.shape[0])
            if dense_bpb is not None:
                dense_bpb[start_position + active_rows] = bits.astype(np.float32, copy=False)
            fused_bits += float(np.sum(bits, dtype=np.float64))
            emitted += int(bits.shape[0])

            if eta > 0.0:
                left_new = np.power(left_weight, eta_power) * left_target
                right_new = np.power(right_weight, eta_power) * right_target
            else:
                left_new = left_weight * left_target
                right_new = right_weight * right_target
            denom = np.maximum(left_new + right_new, 1e-300)
            left_weights[window_ids] = left_new / denom
            right_weights[window_ids] = right_new / denom

    window_bpb = np.divide(
        window_bit_sums,
        window_counts,
        out=np.full_like(window_bit_sums, np.nan),
        where=window_counts > 0,
    )
    offset_bpb = np.divide(
        offset_bit_sums,
        offset_counts,
        out=np.full_like(offset_bit_sums, np.nan),
        where=offset_counts > 0,
    )
    midpoint_mbase = (
        np.arange(window_count, dtype=np.float64) * window_bases + (window_counts.astype(np.float64) / 2.0)
    ) / 1e6
    dense_position_mbase = np.arange(core_base_count, dtype=np.float64) / 1e6 if dense_bpb is not None else None
    full_bpb = float(fused_bits / emitted) if emitted else float("nan")
    return (
        midpoint_mbase,
        window_bpb,
        np.arange(window_bases, dtype=np.int64),
        offset_bpb,
        dense_position_mbase,
        dense_bpb,
        full_bpb,
        perf_counter() - started,
    )


def _should_use_dense_full_position_curve(
    *,
    sample_bases: int,
    window_count: int,
    max_bases: int,
    window_threshold: int,
) -> bool:
    if max_bases <= 0 or window_threshold <= 0:
        return False
    return int(sample_bases) <= int(max_bases) and int(window_count) <= int(window_threshold)


def _trace_dense_position_bpb(
    trace_dir: Path,
    *,
    verify_checksum: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    reader = ProbabilityTraceReader(trace_dir)
    manifest = reader.manifest
    core_base_count = int(manifest.core_base_count)
    dense_bpb = np.full((core_base_count,), np.nan, dtype=np.float32)
    started = perf_counter()
    for shard in reader.iter_shards(verify_checksum=verify_checksum):
        target_prob = np.asarray(shard["target_prob"], dtype=np.float64)
        bits = -np.log2(np.clip(target_prob, np.finfo(np.float32).tiny, 1.0)).astype(np.float32, copy=False)
        emit_position = np.asarray(shard["emit_position"], dtype=np.int64)
        dense_bpb[emit_position] = bits
    dense_position_mbase = np.arange(core_base_count, dtype=np.float64) / 1e6
    return dense_position_mbase, dense_bpb, perf_counter() - started


def _style_axis(ax) -> None:
    ax.grid(True, color="#d6d6d6", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_color("#222222")
        ax.spines[spine].set_linewidth(0.8)


def _apply_bpb_ylim(ax, *series: np.ndarray, pad: float = 0.08) -> None:
    finite_parts = [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for values in series]
    finite = np.concatenate([values for values in finite_parts if values.size]) if any(values.size for values in finite_parts) else np.asarray([])
    if not finite.size:
        return
    ymin = max(0.0, float(np.nanpercentile(finite, 1.0)) - pad)
    ymax = float(np.nanpercentile(finite, 99.0)) + pad
    if ymin >= ymax:
        ymin = max(0.0, float(np.nanmin(finite)) - pad)
        ymax = float(np.nanmax(finite)) + pad
    ax.set_ylim(ymin, ymax)


def _set_top_axis_ylim(ax, window_bpb: np.ndarray, window_smooth: np.ndarray, full_bpb: float) -> None:
    finite = np.asarray(window_bpb, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return
    if finite.size < 256:
        ymin = min(float(np.nanmin(finite)), float(full_bpb)) - 0.04
        ymax = max(float(np.nanmax(finite)), float(full_bpb)) + 0.04
        if ymax - ymin < 0.18:
            center = 0.5 * (ymin + ymax)
            ymin = center - 0.09
            ymax = center + 0.09
        ax.set_ylim(max(0.0, ymin), ymax)
        return
    top_ymax = max(2.05, float(np.nanpercentile(finite, 99.0)) + 0.08)
    ax.set_ylim(0.0, top_ymax)


def _full_bpb_from_row(row: dict[str, Any], fallback: float | None) -> float:
    if "theoretical_bits_per_base" in row:
        return float(row["theoretical_bits_per_base"])
    if fallback is None:
        return float("nan")
    return float(fallback)


def _sample_bases_from_row(row: dict[str, Any]) -> int:
    return int(row.get("sample_bases", row.get("core_base_count", row.get("row_count"))))


def _curve_cache_path(curve_data_dir: Path, output_prefix: str, source: str) -> Path:
    return curve_data_dir / f"{output_prefix}_curve_{source}.npz"


def _label_slug(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _load_or_compute_curve(
    *,
    row: dict[str, Any],
    source: str,
    repo_root: Path,
    curve_data_dir: Path,
    output_prefix: str,
    reuse_curve_data: bool,
    verify_checksum: bool,
    dense_full_position_max_bases: int,
    dense_full_position_window_threshold: int,
) -> dict[str, Any]:
    trace_dir = _as_path(str(row["trace_dir"]), repo_root)
    curve_data_path = _curve_cache_path(curve_data_dir, output_prefix, source)
    full_bpb_from_trace: float | None = None
    sample_bases = _sample_bases_from_row(row)
    window_bases = int(row["window_bases"])
    window_count = int(row.get("window_count", math.ceil(sample_bases / max(window_bases, 1))))
    want_dense_full_position = _should_use_dense_full_position_curve(
        sample_bases=sample_bases,
        window_count=window_count,
        max_bases=int(dense_full_position_max_bases),
        window_threshold=int(dense_full_position_window_threshold),
    )
    dense_position_mbase: np.ndarray | None = None
    dense_bpb: np.ndarray | None = None
    dense_read_seconds = 0.0
    if reuse_curve_data and curve_data_path.exists():
        with np.load(curve_data_path) as data:
            if "offset_bpb" not in data:
                raise ValueError(f"curve cache lacks offset_bpb; rerun with --no-reuse-curve-data: {curve_data_path}")
            midpoint_mbase = np.asarray(data["midpoint_mbase"], dtype=np.float64)
            window_bpb = np.asarray(data["window_bpb"], dtype=np.float64)
            offset_position = np.asarray(data["offset_position"], dtype=np.int64)
            offset_bpb = np.asarray(data["offset_bpb"], dtype=np.float64)
            if "full_sequence_theoretical_bpb" in data:
                full_bpb_from_trace = float(np.asarray(data["full_sequence_theoretical_bpb"], dtype=np.float64))
            if want_dense_full_position and "dense_bpb" in data:
                dense_bpb = np.asarray(data["dense_bpb"], dtype=np.float32)
                if "dense_position_mbase" in data:
                    dense_position_mbase = np.asarray(data["dense_position_mbase"], dtype=np.float64)
                else:
                    dense_position_mbase = np.arange(dense_bpb.shape[0], dtype=np.float64) / 1e6
        read_seconds = 0.0
    else:
        midpoint_mbase, window_bpb, offset_position, offset_bpb, full_bpb_from_trace, read_seconds = _trace_curve_stats(
            trace_dir,
            verify_checksum=verify_checksum,
        )
    if want_dense_full_position and dense_bpb is None:
        dense_position_mbase, dense_bpb, dense_read_seconds = _trace_dense_position_bpb(
            trace_dir,
            verify_checksum=verify_checksum,
        )
    full_bpb = _full_bpb_from_row(row, full_bpb_from_trace)
    window_mean_bpb = float(np.nanmean(window_bpb))
    geco2_level = row.get("geco2_level")
    cache_payload = {
        "source": np.asarray(source),
        "midpoint_mbase": midpoint_mbase.astype(np.float64),
        "window_bpb": window_bpb.astype(np.float32),
        "offset_position": offset_position.astype(np.int32),
        "offset_bpb": offset_bpb.astype(np.float32),
        "full_sequence_theoretical_bpb": np.asarray(full_bpb, dtype=np.float64),
        "single_window_mean_bpb": np.asarray(window_mean_bpb, dtype=np.float64),
        "window_bases": np.asarray(int(row["window_bases"]), dtype=np.int64),
        "geco2_level": np.asarray(-1 if geco2_level in {None, ""} else int(geco2_level), dtype=np.int64),
        "dense_full_position_enabled": np.asarray(bool(dense_bpb is not None), dtype=np.bool_),
    }
    if dense_bpb is not None:
        if dense_position_mbase is None:
            dense_position_mbase = np.arange(dense_bpb.shape[0], dtype=np.float64) / 1e6
        cache_payload["dense_position_mbase"] = dense_position_mbase.astype(np.float64)
        cache_payload["dense_bpb"] = dense_bpb.astype(np.float32, copy=False)
    np.savez_compressed(curve_data_path, **cache_payload)
    return {
        "source": source,
        "trace_dir": trace_dir,
        "curve_data_path": curve_data_path,
        "midpoint_mbase": midpoint_mbase,
        "window_bpb": window_bpb,
        "offset_position": offset_position,
        "offset_bpb": offset_bpb,
        "dense_position_mbase": dense_position_mbase,
        "dense_bpb": dense_bpb,
        "full_bpb": full_bpb,
        "window_mean_bpb": window_mean_bpb,
        "window_std_bpb": float(np.nanstd(window_bpb)),
        "read_seconds": read_seconds + dense_read_seconds,
        "dense_read_seconds": dense_read_seconds,
        "dense_full_position_enabled": bool(dense_bpb is not None),
    }


def _load_or_compute_fusion_curve(
    *,
    row: dict[str, Any],
    source: str,
    repo_root: Path,
    curve_data_dir: Path,
    output_prefix: str,
    reuse_curve_data: bool,
    dense_full_position_max_bases: int,
    dense_full_position_window_threshold: int,
) -> dict[str, Any]:
    left_trace_dir = _as_path(str(row["trace_left"]), repo_root)
    right_trace_dir = _as_path(str(row["trace_right"]), repo_root)
    curve_data_path = _curve_cache_path(curve_data_dir, output_prefix, source)
    sample_bases = int(row.get("sample_bases", row.get("core_base_count", row.get("row_count"))))
    window_bases = int(row["window_bases"])
    window_count = int(math.ceil(sample_bases / max(window_bases, 1)))
    want_dense_full_position = _should_use_dense_full_position_curve(
        sample_bases=sample_bases,
        window_count=window_count,
        max_bases=int(dense_full_position_max_bases),
        window_threshold=int(dense_full_position_window_threshold),
    )

    dense_position_mbase: np.ndarray | None = None
    dense_bpb: np.ndarray | None = None
    full_bpb_from_curve: float | None = None
    read_seconds = 0.0
    if reuse_curve_data and curve_data_path.exists():
        with np.load(curve_data_path) as data:
            if "offset_bpb" not in data:
                raise ValueError(f"fusion curve cache lacks offset_bpb; rerun with --no-reuse-curve-data: {curve_data_path}")
            midpoint_mbase = np.asarray(data["midpoint_mbase"], dtype=np.float64)
            window_bpb = np.asarray(data["window_bpb"], dtype=np.float64)
            offset_position = np.asarray(data["offset_position"], dtype=np.int64)
            offset_bpb = np.asarray(data["offset_bpb"], dtype=np.float64)
            if "full_sequence_theoretical_bpb" in data:
                full_bpb_from_curve = float(np.asarray(data["full_sequence_theoretical_bpb"], dtype=np.float64))
            if want_dense_full_position and "dense_bpb" in data:
                dense_bpb = np.asarray(data["dense_bpb"], dtype=np.float32)
                if "dense_position_mbase" in data:
                    dense_position_mbase = np.asarray(data["dense_position_mbase"], dtype=np.float64)
                else:
                    dense_position_mbase = np.arange(dense_bpb.shape[0], dtype=np.float64) / 1e6
    else:
        (
            midpoint_mbase,
            window_bpb,
            offset_position,
            offset_bpb,
            dense_position_mbase,
            dense_bpb,
            full_bpb_from_curve,
            read_seconds,
        ) = _fusion_trace_curve_stats(
            left_trace_dir,
            right_trace_dir,
            fusion_eta=float(row.get("fusion_eta", 0.05)),
            fusion_initial_left_weight=float(row.get("fusion_initial_carbon_weight", row.get("fusion_initial_lm_weight", 0.5))),
            dense_full_position=want_dense_full_position,
        )

    full_bpb = float(row.get("fused_theoretical_bpb", row.get("theoretical_bits_per_base", full_bpb_from_curve)))
    window_mean_bpb = float(np.nanmean(window_bpb))
    cache_payload = {
        "source": np.asarray(source),
        "midpoint_mbase": midpoint_mbase.astype(np.float64),
        "window_bpb": window_bpb.astype(np.float32),
        "offset_position": offset_position.astype(np.int32),
        "offset_bpb": offset_bpb.astype(np.float32),
        "full_sequence_theoretical_bpb": np.asarray(full_bpb, dtype=np.float64),
        "single_window_mean_bpb": np.asarray(window_mean_bpb, dtype=np.float64),
        "window_bases": np.asarray(window_bases, dtype=np.int64),
        "fusion_eta": np.asarray(float(row.get("fusion_eta", 0.05)), dtype=np.float64),
        "fusion_initial_left_weight": np.asarray(
            float(row.get("fusion_initial_carbon_weight", row.get("fusion_initial_lm_weight", 0.5))),
            dtype=np.float64,
        ),
        "dense_full_position_enabled": np.asarray(bool(dense_bpb is not None), dtype=np.bool_),
    }
    if dense_bpb is not None:
        if dense_position_mbase is None:
            dense_position_mbase = np.arange(dense_bpb.shape[0], dtype=np.float64) / 1e6
        cache_payload["dense_position_mbase"] = dense_position_mbase.astype(np.float64)
        cache_payload["dense_bpb"] = dense_bpb.astype(np.float32, copy=False)
    np.savez_compressed(curve_data_path, **cache_payload)
    return {
        "source": source,
        "trace_dir": None,
        "curve_data_path": curve_data_path,
        "midpoint_mbase": midpoint_mbase,
        "window_bpb": window_bpb,
        "offset_position": offset_position,
        "offset_bpb": offset_bpb,
        "dense_position_mbase": dense_position_mbase,
        "dense_bpb": dense_bpb,
        "full_bpb": full_bpb,
        "window_mean_bpb": window_mean_bpb,
        "window_std_bpb": float(np.nanstd(window_bpb)),
        "read_seconds": read_seconds,
        "dense_read_seconds": 0.0,
        "dense_full_position_enabled": bool(dense_bpb is not None),
    }


def _set_top_axis_ylim_multi(ax, series: list[dict[str, Any]]) -> None:
    finite_parts = []
    for item in series:
        top_values = [item["window_bpb"], item["window_smooth"]]
        if item.get("dense_bpb") is not None:
            top_values = [item.get("dense_smooth", item["dense_bpb"]), np.asarray([item["full_bpb"]], dtype=np.float64)]
        for values in top_values:
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                finite_parts.append(arr)
    if not finite_parts:
        return
    finite = np.concatenate(finite_parts)
    if finite.size < 256:
        ymin = float(np.nanmin(finite)) - 0.04
        ymax = float(np.nanmax(finite)) + 0.04
    else:
        ymin = max(0.0, float(np.nanpercentile(finite, 1.0)) - 0.10)
        ymax = float(np.nanpercentile(finite, 99.0)) + 0.10
        if ymax - ymin < 0.24:
            center = 0.5 * (ymin + ymax)
            ymin = center - 0.12
            ymax = center + 0.12
    ax.set_ylim(max(0.0, ymin), ymax)


def _series_line_style(label: str) -> Any:
    normalized = str(label).lower()
    if "evo2" in normalized and "+ nc" in normalized:
        return (0, (5.0, 2.0, 1.0, 2.0))
    if "evo2" in normalized:
        return (0, (2.0, 1.6))
    if "carbon" in normalized and "+ nc" in normalized:
        return "solid"
    if "carbon" in normalized:
        return (0, (4.0, 2.0))
    return "solid"


def _find_gain_pairs(
    labels: list[str],
    items: list[dict[str, Any]],
    colors: list[str],
) -> list[tuple[str, dict[str, Any], dict[str, Any], str]]:
    by_label = {str(label).lower(): (label, item, color) for label, item, color in zip(labels, items, colors)}
    pairs: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    carbon = by_label.get("carbon 3b")
    carbon_nc = by_label.get("carbon 3b + nc")
    if carbon is not None and carbon_nc is not None:
        pairs.append(("Carbon + nc delta", carbon[1], carbon_nc[1], carbon_nc[2]))
    evo2 = by_label.get("evo2 7b")
    evo2_nc = by_label.get("evo2 7b + nc")
    if evo2 is not None and evo2_nc is not None:
        pairs.append(("Evo2 + nc delta", evo2[1], evo2_nc[1], evo2_nc[2]))
    return pairs


def _top_curve_xy(
    item: dict[str, Any],
    *,
    effective_dense_smooth_bases: int,
    rolling_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if item.get("dense_bpb") is not None and item.get("dense_position_mbase") is not None:
        y = (
            item["dense_smooth"]
            if effective_dense_smooth_bases > 1 and item["dense_bpb"].shape[0] >= effective_dense_smooth_bases
            else item["dense_bpb"]
        )
        return np.asarray(item["dense_position_mbase"], dtype=np.float64), np.asarray(y, dtype=np.float64)
    y = item["window_smooth"] if rolling_windows > 1 and item["window_bpb"].shape[0] >= rolling_windows else item["window_bpb"]
    return np.asarray(item["midpoint_mbase"], dtype=np.float64), np.asarray(y, dtype=np.float64)


def _apply_gain_ylim(ax, *series: np.ndarray) -> None:
    finite_parts = [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for values in series]
    finite = np.concatenate([values for values in finite_parts if values.size]) if any(values.size for values in finite_parts) else np.asarray([])
    if not finite.size:
        return
    ymin = float(np.nanpercentile(finite, 1.0))
    ymax = float(np.nanpercentile(finite, 99.0))
    pad = max(0.01, 0.12 * max(abs(ymin), abs(ymax), ymax - ymin))
    ax.set_ylim(min(0.0, ymin - pad), max(0.0, ymax + pad))


def _plot_species_comparison(
    *,
    source: str,
    series: list[tuple[str, dict[str, Any], str]],
    rolling_windows: int,
    dense_position_smooth_bases: int,
    dense_plot_max_points: int,
    sample_bases: int,
    window_bases: int,
    output_path: Path,
) -> None:
    labels = [label for label, _, _ in series]
    items = [item for _, item, _ in series]
    colors = [color for _, _, color in series]
    effective_dense_smooth_bases = _effective_dense_position_smooth_bases(
        dense_position_smooth_bases,
        sample_bases,
    )
    for item in items:
        item["window_smooth"] = _rolling_mean(item["window_bpb"], rolling_windows)
        item["offset_smooth"] = _rolling_mean(item["offset_bpb"], rolling_windows)
        if item.get("dense_bpb") is not None:
            item["dense_smooth"] = _rolling_mean(
                np.asarray(item["dense_bpb"], dtype=np.float64),
                effective_dense_smooth_bases,
            )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans"],
            "font.size": 16,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "agg.path.chunksize": 20000,
        }
    )
    fig, (ax_top, ax_gain, ax_bottom) = plt.subplots(
        3,
        1,
        figsize=(15.2, 12.1),
        dpi=160,
        gridspec_kw={"height_ratios": [1.05, 0.55, 1.0], "hspace": 0.30},
    )
    fig.patch.set_facecolor("white")
    for ax in [ax_top, ax_gain, ax_bottom]:
        ax.set_facecolor("white")
        _style_axis(ax)

    for item, label, color in zip(items, labels, colors):
        linestyle = _series_line_style(label)
        if item.get("dense_bpb") is not None and item.get("dense_position_mbase") is not None:
            top_y = (
                item["dense_smooth"]
                if effective_dense_smooth_bases > 1 and item["dense_bpb"].shape[0] >= effective_dense_smooth_bases
                else item["dense_bpb"]
            )
            plot_x, plot_y = _thin_line_for_plot(
                item["dense_position_mbase"],
                top_y,
                max_points=dense_plot_max_points,
            )
            ax_top.plot(
                plot_x,
                plot_y,
                color=color,
                linewidth=1.45,
                linestyle=linestyle,
                alpha=0.84,
                label=f"{label} full mean {item['full_bpb']:.3f}",
            )
        else:
            marker_style = "o" if item["window_bpb"].shape[0] < 256 else None
            marker_size = 2.0 if item["window_bpb"].shape[0] < 256 else 0.0
            ax_top.plot(
                item["midpoint_mbase"],
                item["window_bpb"],
                color=color,
                linewidth=0.38,
                alpha=0.08,
                marker=marker_style,
                markersize=marker_size,
            )
            top_y = item["window_smooth"] if rolling_windows > 1 and item["window_bpb"].shape[0] >= rolling_windows else item["window_bpb"]
            ax_top.plot(
                item["midpoint_mbase"],
                top_y,
                color=color,
                linewidth=1.45,
                linestyle=linestyle,
                alpha=0.84,
                label=f"{label} full mean {item['full_bpb']:.3f}",
            )
        ax_top.axhline(item["full_bpb"], color=color, linewidth=0.75, linestyle=linestyle, alpha=0.24)

        ax_bottom.scatter(item["offset_position"], item["offset_bpb"], s=3, color=color, alpha=0.035, edgecolors="none")
        bottom_y = item["offset_smooth"] if rolling_windows > 1 and item["offset_bpb"].shape[0] >= rolling_windows else item["offset_bpb"]
        ax_bottom.plot(
            item["offset_position"],
            bottom_y,
            color=color,
            linewidth=1.25,
            linestyle=linestyle,
            alpha=0.84,
            label=f"{label} position mean {item['window_mean_bpb']:.3f}",
        )
        ax_bottom.axhline(item["window_mean_bpb"], color=color, linewidth=0.75, linestyle=linestyle, alpha=0.22)

    top_x_arrays = [
        item["dense_position_mbase"]
        if item.get("dense_position_mbase") is not None
        else item["midpoint_mbase"]
        for item in items
    ]
    xmin = min(float(np.nanmin(values)) for values in top_x_arrays if values.size)
    xmax = max(float(np.nanmax(values)) for values in top_x_arrays if values.size)
    ax_top.set_xlim(xmin, xmax if xmax > xmin else 0.01)
    _set_top_axis_ylim_multi(ax_top, items)
    ax_top.set_ylabel("bits/base")
    ax_top.tick_params(axis="x", labelbottom=False)

    gain_series: list[np.ndarray] = []
    for gain_label, base_item, fused_item, color in _find_gain_pairs(labels, items, colors):
        base_x, base_y = _top_curve_xy(
            base_item,
            effective_dense_smooth_bases=effective_dense_smooth_bases,
            rolling_windows=rolling_windows,
        )
        fused_x, fused_y = _top_curve_xy(
            fused_item,
            effective_dense_smooth_bases=effective_dense_smooth_bases,
            rolling_windows=rolling_windows,
        )
        if base_y.shape != fused_y.shape:
            continue
        gain = fused_y - base_y
        gain_series.append(gain)
        plot_x, plot_gain = _thin_line_for_plot(base_x, gain, max_points=dense_plot_max_points)
        ax_gain.plot(plot_x, plot_gain, color=color, linewidth=1.35, alpha=0.9, label=gain_label)
        ax_gain.axhline(float(fused_item["full_bpb"]) - float(base_item["full_bpb"]), color=color, linewidth=0.8, linestyle="--", alpha=0.30)
    ax_gain.axhline(0.0, color="#4b5563", linewidth=0.8, linestyle="-", alpha=0.55)
    ax_gain.set_xlim(xmin, xmax if xmax > xmin else 0.01)
    _apply_gain_ylim(ax_gain, *gain_series)
    ax_gain.set_ylabel("bpb delta")
    ax_gain.set_xlabel("Source position (Mbases)")
    if gain_series:
        ax_gain.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#c8c8c8", fontsize=12)

    ax_bottom.set_xlim(0, int(window_bases) - 1)
    _apply_bpb_ylim(
        ax_bottom,
        *[values for item in items for values in (item["offset_bpb"], item["offset_smooth"])],
        pad=0.04,
    )
    ax_bottom.set_ylabel("bits/base")
    ax_bottom.set_xlabel("Position in model window (bases)")
    legend_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, linestyle=_series_line_style(label), alpha=0.9)
        for label, color in zip(labels, colors)
    ]
    fig.legend(
        legend_handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=min(5, max(1, len(labels))),
        frameon=True,
        facecolor="white",
        edgecolor="#c8c8c8",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.060)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _plot_species(
    *,
    source: str,
    midpoint_mbase: np.ndarray,
    window_bpb: np.ndarray,
    offset_position: np.ndarray,
    offset_bpb: np.ndarray,
    full_bpb: float,
    window_mean_bpb: float,
    rolling_windows: int,
    dense_position_mbase: np.ndarray | None,
    dense_bpb: np.ndarray | None,
    dense_position_smooth_bases: int,
    dense_plot_max_points: int,
    row: dict[str, Any],
    model_label: str,
    output_path: Path,
) -> None:
    window_smooth = _rolling_mean(window_bpb, rolling_windows)
    offset_smooth = _rolling_mean(offset_bpb, rolling_windows)
    effective_dense_smooth_bases = _effective_dense_position_smooth_bases(
        dense_position_smooth_bases,
        int(row["sample_bases"]),
    )
    dense_smooth = (
        _rolling_mean(np.asarray(dense_bpb, dtype=np.float64), effective_dense_smooth_bases)
        if dense_bpb is not None
        else None
    )
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "agg.path.chunksize": 20000,
        }
    )
    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(15.2, 9.7),
        dpi=160,
        gridspec_kw={"height_ratios": [1.05, 1.0], "hspace": 0.28},
    )
    fig.patch.set_facecolor("white")
    for ax in [ax_top, ax_bottom]:
        ax.set_facecolor("white")
        _style_axis(ax)

    current_color = "#d62728"
    raw_color = "#d62728"
    full_color = "#2ca02c"

    if dense_bpb is not None and dense_position_mbase is not None:
        top_y = (
            dense_smooth
            if dense_smooth is not None and effective_dense_smooth_bases > 1 and dense_bpb.shape[0] >= effective_dense_smooth_bases
            else dense_bpb
        )
        plot_x, plot_y = _thin_line_for_plot(
            dense_position_mbase,
            top_y,
            max_points=dense_plot_max_points,
        )
        ax_top.plot(
            plot_x,
            plot_y,
            color=current_color,
            linewidth=2.15,
            label=f"{model_label} full-sequence per-base mean {full_bpb:.3f}",
        )
        ax_top.set_xlim(
            float(np.nanmin(dense_position_mbase)),
            float(np.nanmax(dense_position_mbase)) if dense_position_mbase.size > 1 else 0.01,
        )
    else:
        marker_style = "o" if window_bpb.shape[0] < 256 else None
        marker_size = 3.0 if window_bpb.shape[0] < 256 else 0.0
        ax_top.plot(
            midpoint_mbase,
            window_bpb,
            color=raw_color,
            linewidth=0.75,
            alpha=0.23,
            marker=marker_style,
            markersize=marker_size,
        )
        top_y = window_smooth if rolling_windows > 1 and window_bpb.shape[0] >= rolling_windows else window_bpb
        ax_top.plot(
            midpoint_mbase,
            top_y,
            color=current_color,
            linewidth=2.4 if top_y is window_smooth else 2.0,
            label=f"{model_label} full-sequence windows mean {full_bpb:.3f}",
        )
        ax_top.set_xlim(
            float(np.nanmin(midpoint_mbase)),
            float(np.nanmax(midpoint_mbase)) if midpoint_mbase.size > 1 else 0.01,
        )
    ax_top.axhline(full_bpb, color=full_color, linewidth=1.1, linestyle="--", alpha=0.45)
    _set_top_axis_ylim(
        ax_top,
        dense_smooth if dense_smooth is not None else window_bpb,
        dense_smooth if dense_smooth is not None else window_smooth,
        full_bpb,
    )
    ax_top.set_ylabel("bits/base")
    ax_top.set_xlabel("Source position (Mbases)")

    ax_bottom.scatter(offset_position, offset_bpb, s=5, color=raw_color, alpha=0.12, edgecolors="none")
    if rolling_windows > 1 and offset_bpb.shape[0] >= rolling_windows:
        ax_bottom.plot(
            offset_position,
            offset_smooth,
            color=current_color,
            linewidth=2.0,
            label=f"{model_label} position-average",
        )
    else:
        ax_bottom.plot(
            offset_position,
            offset_bpb,
            color=current_color,
            linewidth=1.6,
            label=f"{model_label} position-average",
        )
    ax_bottom.axhline(window_mean_bpb, color=full_color, linewidth=1.1, linestyle="--", alpha=0.45)
    ax_bottom.set_xlim(0, int(row["window_bases"]) - 1)
    _apply_bpb_ylim(ax_bottom, offset_bpb, offset_smooth, pad=0.04)
    ax_bottom.set_ylabel("bits/base")
    ax_bottom.set_xlabel("Position in model window (bases)")
    ax_bottom.set_title("Average BPB by position inside the 8,192 bp window")
    ax_bottom.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#c8c8c8")

    sample_mb = int(row["sample_bases"]) / 1e6
    delta = (full_bpb - window_mean_bpb) * 1000.0
    fig.suptitle(
        (
            f"{source} {model_label} w8192 BPB, per-base full sequence ({effective_dense_smooth_bases}-base MA)"
            if dense_bpb is not None
            else f"{source} {model_label} w8192 BPB, 8,192 bp window means ({rolling_windows}-window MA)"
        ),
        y=0.985,
        fontsize=17,
    )
    metadata_parts = [
        f"full-sequence mean {full_bpb:.3f}",
        f"window-position mean {window_mean_bpb:.3f}",
    ]
    if row.get("geco2_level") not in {None, ""}:
        metadata_parts.append(f"DNACorpus best level L{row.get('geco2_level')}")
    metadata_parts.extend([f"bases={sample_mb:.3f}M", f"delta={delta:+.3f} mbpb"])
    fig.text(
        0.5,
        0.955,
        "; ".join(metadata_parts),
        ha="center",
        va="top",
        fontsize=10,
        color="#4b5563",
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.895, bottom=0.065)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = REPO_ROOT
    summary_path = _as_path(args.summary_json, repo_root)
    output_dir = _as_path(args.output_dir, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_data_dir = _as_path(args.curve_data_dir, repo_root) if args.curve_data_dir else output_dir / "curve_data"
    curve_data_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(args.output_prefix)
    model_label = str(args.model_label)
    summary_csv = _as_path(args.summary_csv, repo_root) if args.summary_csv else output_dir / f"{output_prefix}_species_curve_summary.csv"

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = list(payload["rows"])
    order_index = {source: index for index, source in enumerate(DEFAULT_ORDER)}
    rows.sort(key=lambda item: order_index.get(str(item["source"]), 999))

    if args.compare_summary_json:
        if len(args.extra_model_label) != len(args.extra_summary_json):
            raise SystemExit("--extra-model-label must be provided once for each --extra-summary-json")
        if args.extra_curve_data_dir and len(args.extra_curve_data_dir) != len(args.extra_summary_json):
            raise SystemExit("--extra-curve-data-dir must be provided once for each --extra-summary-json")
        if len(args.extra_fusion_model_label) != len(args.extra_fusion_summary_csv):
            raise SystemExit("--extra-fusion-model-label must be provided once for each --extra-fusion-summary-csv")
        if args.extra_fusion_curve_data_dir and len(args.extra_fusion_curve_data_dir) != len(args.extra_fusion_summary_csv):
            raise SystemExit("--extra-fusion-curve-data-dir must be provided once for each --extra-fusion-summary-csv")

        compare_summary_path = _as_path(args.compare_summary_json, repo_root)
        compare_payload = json.loads(compare_summary_path.read_text(encoding="utf-8"))
        compare_rows_by_source = {str(row["source"]): row for row in compare_payload["rows"]}
        left_rows_by_source = {str(row["source"]): row for row in rows}
        extra_trace_specs: list[tuple[str, dict[str, dict[str, Any]], Path]] = []
        for index, (summary_text, label) in enumerate(zip(args.extra_summary_json, args.extra_model_label)):
            extra_summary_path = _as_path(summary_text, repo_root)
            extra_payload = json.loads(extra_summary_path.read_text(encoding="utf-8"))
            if args.extra_curve_data_dir:
                extra_curve_data_dir = _as_path(args.extra_curve_data_dir[index], repo_root)
            else:
                extra_curve_data_dir = output_dir / f"curve_data_{_label_slug(label)}"
            extra_curve_data_dir.mkdir(parents=True, exist_ok=True)
            extra_trace_specs.append(
                (
                    str(label),
                    {str(row["source"]): row for row in extra_payload["rows"]},
                    extra_curve_data_dir,
                )
            )
        fusion_rows_by_source: dict[str, dict[str, Any]] = {}
        if args.fusion_summary_csv:
            fusion_summary_path = _as_path(args.fusion_summary_csv, repo_root)
            with fusion_summary_path.open(newline="", encoding="utf-8") as handle:
                fusion_rows_by_source = {str(row["species"]): row for row in csv.DictReader(handle)}
        extra_fusion_specs: list[tuple[str, dict[str, dict[str, Any]], Path]] = []
        for index, (summary_text, label) in enumerate(zip(args.extra_fusion_summary_csv, args.extra_fusion_model_label)):
            extra_fusion_summary_path = _as_path(summary_text, repo_root)
            with extra_fusion_summary_path.open(newline="", encoding="utf-8") as handle:
                extra_rows_by_source = {str(row["species"]): row for row in csv.DictReader(handle)}
            if args.extra_fusion_curve_data_dir:
                extra_curve_data_dir = _as_path(args.extra_fusion_curve_data_dir[index], repo_root)
            else:
                extra_curve_data_dir = output_dir / f"curve_data_{_label_slug(label)}"
            extra_curve_data_dir.mkdir(parents=True, exist_ok=True)
            extra_fusion_specs.append((str(label), extra_rows_by_source, extra_curve_data_dir))
        sources = sorted(
            set(left_rows_by_source) & set(compare_rows_by_source),
            key=lambda source: order_index.get(source, 999),
        )
        if not sources:
            raise SystemExit("no overlapping sources between --summary-json and --compare-summary-json")

        if args.curve_data_dir:
            left_curve_data_dir = curve_data_dir
        else:
            left_curve_data_dir = output_dir / "curve_data_left"
            left_curve_data_dir.mkdir(parents=True, exist_ok=True)
        compare_curve_data_dir = (
            _as_path(args.compare_curve_data_dir, repo_root)
            if args.compare_curve_data_dir
            else output_dir / "curve_data_right"
        )
        compare_curve_data_dir.mkdir(parents=True, exist_ok=True)
        fusion_curve_data_dir = (
            _as_path(args.fusion_curve_data_dir, repo_root)
            if args.fusion_curve_data_dir
            else output_dir / "curve_data_fusion"
        )
        if fusion_rows_by_source:
            fusion_curve_data_dir.mkdir(parents=True, exist_ok=True)

        comparison_rows: list[dict[str, Any]] = []
        for idx, source in enumerate(sources, 1):
            left_row = left_rows_by_source[source]
            right_row = compare_rows_by_source[source]
            fusion_row = fusion_rows_by_source.get(source)
            sample_bases = _sample_bases_from_row(left_row)
            right_sample_bases = _sample_bases_from_row(right_row)
            if sample_bases != right_sample_bases:
                raise ValueError(f"sample base mismatch for {source}: {sample_bases} != {right_sample_bases}")
            if fusion_row is not None and sample_bases != int(fusion_row["sample_bases"]):
                raise ValueError(f"fusion sample base mismatch for {source}: {sample_bases} != {fusion_row['sample_bases']}")
            window_bases = int(left_row["window_bases"])
            if window_bases != int(right_row["window_bases"]):
                raise ValueError(f"window_bases mismatch for {source}: {window_bases} != {right_row['window_bases']}")
            left = _load_or_compute_curve(
                row=left_row,
                source=source,
                repo_root=repo_root,
                curve_data_dir=left_curve_data_dir,
                output_prefix=f"{output_prefix}_left",
                reuse_curve_data=bool(args.reuse_curve_data),
                verify_checksum=bool(args.verify_shard_checksum),
                dense_full_position_max_bases=int(args.dense_full_position_max_bases),
                dense_full_position_window_threshold=int(args.dense_full_position_window_threshold),
            )
            right = _load_or_compute_curve(
                row=right_row,
                source=source,
                repo_root=repo_root,
                curve_data_dir=compare_curve_data_dir,
                output_prefix=f"{output_prefix}_right",
                reuse_curve_data=bool(args.reuse_curve_data),
                verify_checksum=bool(args.verify_shard_checksum),
                dense_full_position_max_bases=int(args.dense_full_position_max_bases),
                dense_full_position_window_threshold=int(args.dense_full_position_window_threshold),
            )
            fusion = None
            if fusion_row is not None:
                fusion = _load_or_compute_fusion_curve(
                    row=fusion_row,
                    source=source,
                    repo_root=repo_root,
                    curve_data_dir=fusion_curve_data_dir,
                    output_prefix=f"{output_prefix}_fusion",
                    reuse_curve_data=bool(args.reuse_curve_data),
                    dense_full_position_max_bases=int(args.dense_full_position_max_bases),
                    dense_full_position_window_threshold=int(args.dense_full_position_window_threshold),
                )
            extra_curves: list[tuple[str, dict[str, Any]]] = []
            for extra_label, extra_rows_by_source, extra_curve_data_dir in extra_trace_specs:
                extra_row = extra_rows_by_source.get(source)
                if extra_row is None:
                    continue
                extra_sample_bases = _sample_bases_from_row(extra_row)
                if sample_bases != extra_sample_bases:
                    raise ValueError(f"{extra_label} sample base mismatch for {source}: {sample_bases} != {extra_sample_bases}")
                if window_bases != int(extra_row["window_bases"]):
                    raise ValueError(f"{extra_label} window_bases mismatch for {source}: {window_bases} != {extra_row['window_bases']}")
                extra_curves.append(
                    (
                        extra_label,
                        _load_or_compute_curve(
                            row=extra_row,
                            source=source,
                            repo_root=repo_root,
                            curve_data_dir=extra_curve_data_dir,
                            output_prefix=f"{output_prefix}_{_label_slug(extra_label)}",
                            reuse_curve_data=bool(args.reuse_curve_data),
                            verify_checksum=bool(args.verify_shard_checksum),
                            dense_full_position_max_bases=int(args.dense_full_position_max_bases),
                            dense_full_position_window_threshold=int(args.dense_full_position_window_threshold),
                        ),
                    )
                )
            extra_fusion_curves: list[tuple[str, dict[str, Any]]] = []
            for extra_fusion_label, extra_rows_by_source, extra_curve_data_dir in extra_fusion_specs:
                extra_fusion_row = extra_rows_by_source.get(source)
                if extra_fusion_row is None:
                    continue
                if sample_bases != int(extra_fusion_row["sample_bases"]):
                    raise ValueError(
                        f"{extra_fusion_label} sample base mismatch for {source}: "
                        f"{sample_bases} != {extra_fusion_row['sample_bases']}"
                    )
                extra_fusion_curves.append(
                    (
                        extra_fusion_label,
                        _load_or_compute_fusion_curve(
                            row=extra_fusion_row,
                            source=source,
                            repo_root=repo_root,
                            curve_data_dir=extra_curve_data_dir,
                            output_prefix=f"{output_prefix}_{_label_slug(extra_fusion_label)}",
                            reuse_curve_data=bool(args.reuse_curve_data),
                            dense_full_position_max_bases=int(args.dense_full_position_max_bases),
                            dense_full_position_window_threshold=int(args.dense_full_position_window_threshold),
                        ),
                    )
                )
            plot_path = output_dir / f"{output_prefix}_curve_{source}.png"
            plot_series: list[tuple[str, dict[str, Any], str]] = [
                (str(args.compare_model_label), right, "#0072B2"),
                (model_label, left, "#E69F00"),
            ]
            if fusion is not None:
                plot_series.append((str(args.fusion_model_label), fusion, "#009E73"))
            for extra_label, extra_curve in extra_curves:
                plot_series.append((extra_label, extra_curve, "#5B5FC7"))
            for extra_fusion_label, extra_fusion_curve in extra_fusion_curves:
                plot_series.append((extra_fusion_label, extra_fusion_curve, "#CC79A7"))
            _plot_species_comparison(
                source=source,
                series=plot_series,
                rolling_windows=int(args.rolling_windows),
                dense_position_smooth_bases=int(args.dense_position_smooth_bases),
                dense_plot_max_points=int(args.dense_plot_max_points),
                sample_bases=sample_bases,
                window_bases=window_bases,
                output_path=plot_path,
            )
            comparison_rows.append(
                {
                    "source": source,
                    "sample_bases": sample_bases,
                    "window_bases": window_bases,
                    "left_model_label": model_label,
                    "right_model_label": str(args.compare_model_label),
                    "fusion_model_label": str(args.fusion_model_label) if fusion is not None else "",
                    "left_full_sequence_bpb": left["full_bpb"],
                    "right_full_sequence_bpb": right["full_bpb"],
                    "fusion_full_sequence_bpb": "" if fusion is None else fusion["full_bpb"],
                    "left_minus_right_full_bpb": left["full_bpb"] - right["full_bpb"],
                    "fusion_minus_left_full_bpb": "" if fusion is None else fusion["full_bpb"] - left["full_bpb"],
                    "fusion_minus_right_full_bpb": "" if fusion is None else fusion["full_bpb"] - right["full_bpb"],
                    "left_window_mean_bpb": left["window_mean_bpb"],
                    "right_window_mean_bpb": right["window_mean_bpb"],
                    "fusion_window_mean_bpb": "" if fusion is None else fusion["window_mean_bpb"],
                    "left_minus_right_window_mean_bpb": left["window_mean_bpb"] - right["window_mean_bpb"],
                    "fusion_minus_left_window_mean_bpb": "" if fusion is None else fusion["window_mean_bpb"] - left["window_mean_bpb"],
                    "fusion_minus_right_window_mean_bpb": "" if fusion is None else fusion["window_mean_bpb"] - right["window_mean_bpb"],
                    "left_curve_data_path": str(left["curve_data_path"]),
                    "right_curve_data_path": str(right["curve_data_path"]),
                    "fusion_curve_data_path": "" if fusion is None else str(fusion["curve_data_path"]),
                    **{
                        f"extra_{_label_slug(label)}_full_sequence_bpb": curve["full_bpb"]
                        for label, curve in extra_curves
                    },
                    **{
                        f"extra_{_label_slug(label)}_curve_data_path": str(curve["curve_data_path"])
                        for label, curve in extra_curves
                    },
                    **{
                        f"extra_fusion_{_label_slug(label)}_full_sequence_bpb": curve["full_bpb"]
                        for label, curve in extra_fusion_curves
                    },
                    **{
                        f"extra_fusion_{_label_slug(label)}_curve_data_path": str(curve["curve_data_path"])
                        for label, curve in extra_fusion_curves
                    },
                    "plot_path": str(plot_path),
                    "left_trace_read_seconds": left["read_seconds"],
                    "right_trace_read_seconds": right["read_seconds"],
                    "fusion_trace_read_seconds": "" if fusion is None else fusion["read_seconds"],
                    **{
                        f"extra_{_label_slug(label)}_trace_read_seconds": curve["read_seconds"]
                        for label, curve in extra_curves
                    },
                    **{
                        f"extra_fusion_{_label_slug(label)}_trace_read_seconds": curve["read_seconds"]
                        for label, curve in extra_fusion_curves
                    },
                    "dense_full_position_curve": bool(
                        left["dense_full_position_enabled"]
                        or right["dense_full_position_enabled"]
                        or (fusion is not None and fusion["dense_full_position_enabled"])
                        or any(curve["dense_full_position_enabled"] for _, curve in extra_curves)
                        or any(curve["dense_full_position_enabled"] for _, curve in extra_fusion_curves)
                    ),
                    "requested_dense_position_smooth_bases": int(args.dense_position_smooth_bases),
                    "effective_dense_position_smooth_bases": _effective_dense_position_smooth_bases(
                        int(args.dense_position_smooth_bases),
                        sample_bases,
                    ),
                    "dense_plot_max_points": int(args.dense_plot_max_points),
                }
            )
            print(
                f"[{idx:02d}/{len(sources)}] {source}: plotted {plot_path} "
                f"{model_label}={left['full_bpb']:.6f} {args.compare_model_label}={right['full_bpb']:.6f} "
                f"{args.fusion_model_label + '=' + format(fusion['full_bpb'], '.6f') if fusion is not None else ''} "
                f"{' '.join(label + '=' + format(curve['full_bpb'], '.6f') for label, curve in extra_curves + extra_fusion_curves)} "
                f"delta={left['full_bpb'] - right['full_bpb']:+.6f}",
                flush=True,
            )

        comparison_csv = summary_csv
        fieldnames = list(dict.fromkeys(key for row in comparison_rows for key in row.keys()))
        with comparison_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_rows)
        (output_dir / f"{output_prefix}_species_curve_summary.json").write_text(
            json.dumps({"rows": comparison_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {comparison_csv}")
        return

    aggregate_rows = []
    for idx, row in enumerate(rows, 1):
        source = str(row["source"])
        sample_bases = int(row.get("sample_bases", row.get("core_base_count", row.get("row_count"))))
        curve = _load_or_compute_curve(
            row=row,
            source=source,
            repo_root=repo_root,
            curve_data_dir=curve_data_dir,
            output_prefix=output_prefix,
            reuse_curve_data=bool(args.reuse_curve_data),
            verify_checksum=bool(args.verify_shard_checksum),
            dense_full_position_max_bases=int(args.dense_full_position_max_bases),
            dense_full_position_window_threshold=int(args.dense_full_position_window_threshold),
        )
        window_count = int(row.get("window_count", len(curve["window_bpb"])))
        geco2_level = row.get("geco2_level")
        plot_path = output_dir / f"{output_prefix}_curve_{source}.png"
        _plot_species(
            source=source,
            midpoint_mbase=curve["midpoint_mbase"],
            window_bpb=curve["window_bpb"],
            offset_position=curve["offset_position"],
            offset_bpb=curve["offset_bpb"],
            full_bpb=curve["full_bpb"],
            window_mean_bpb=curve["window_mean_bpb"],
            rolling_windows=int(args.rolling_windows),
            dense_position_mbase=curve["dense_position_mbase"],
            dense_bpb=curve["dense_bpb"],
            dense_position_smooth_bases=int(args.dense_position_smooth_bases),
            dense_plot_max_points=int(args.dense_plot_max_points),
            row={**row, "sample_bases": sample_bases, "window_count": window_count},
            model_label=model_label,
            output_path=plot_path,
        )
        aggregate_rows.append(
            {
                "source": source,
                "sample_bases": sample_bases,
                "window_bases": int(row["window_bases"]),
                "window_count": window_count,
                "geco2_level": "" if geco2_level in {None, ""} else int(geco2_level),
                "model_label": model_label,
                "single_window_mean_bpb": curve["window_mean_bpb"],
                "single_window_std_bpb": curve["window_std_bpb"],
                "full_sequence_theoretical_bpb": curve["full_bpb"],
                "full_sequence_arithmetic_bpb": "" if "arithmetic_bits_per_base" not in row else float(row["arithmetic_bits_per_base"]),
                "delta_full_minus_window_mean_bpb": curve["full_bpb"] - curve["window_mean_bpb"],
                "curve_data_path": str(curve["curve_data_path"]),
                "plot_path": str(plot_path),
                "trace_read_seconds": curve["read_seconds"],
                "dense_full_position_curve": bool(curve["dense_full_position_enabled"]),
                "requested_dense_position_smooth_bases": int(args.dense_position_smooth_bases),
                "effective_dense_position_smooth_bases": _effective_dense_position_smooth_bases(
                    int(args.dense_position_smooth_bases),
                    sample_bases,
                ),
                "dense_plot_max_points": int(args.dense_plot_max_points),
            }
        )
        print(
            f"[{idx:02d}/{len(rows)}] {source}: plotted {plot_path} "
            f"mean={curve['window_mean_bpb']:.6f} full={curve['full_bpb']:.6f} "
            f"delta={(curve['full_bpb'] - curve['window_mean_bpb']) * 1000.0:+.3f} mbpb "
            f"dense={curve['dense_full_position_enabled']} read={curve['read_seconds']:.1f}s",
            flush=True,
        )

    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    (output_dir / f"{output_prefix}_species_curve_summary.json").write_text(
        json.dumps({"rows": aggregate_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
