#!/usr/bin/env python3
from __future__ import annotations

"""Plot comprehensive DNACorpus w8192 model/fusion comparison bars."""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
CORRECTED_SUPPLEMENT_SPECIES = {"EnIn", "HePy", "PlFa", "ScPo"}
PLOT_EXCLUDED_SPECIES: set[str] = set()
PLOT_SPECIES_ORDER = [source for source in DEFAULT_ORDER if source not in PLOT_EXCLUDED_SPECIES]
REGION_SPECIES_WITH_ANNOTATIONS = [
    source for source in PLOT_SPECIES_ORDER if source not in {"AnCa", "WaMe"}
]
EUKARYOTE_PROKARYOTE_SEPARATOR = ("PlFa", "EsCo")

REGION_ORDER = [
    "intergenic",
    "intron",
    "exon_non_cds",
    "rna",
    "cds",
    "repeat_mobile_existing",
]

REGION_LABELS = {
    "intergenic": "Intergenic",
    "intron": "Intron",
    "exon_non_cds": "Exon non-CDS",
    "rna": "non-mRNA RNA",
    "cds": "CDS",
    "repeat_mobile_existing": "Repeat/mobile existing",
}

SMALL_REGION_BASE_THRESHOLD = 1000

COMPREHENSIVE_DIR = REPO_ROOT / "outputs" / "dnacorpus_w8192_comprehensive_analysis_v1"
GROUP_GAP = 0.85
WEIGHTED_GAP = 1.35
BAR_WIDTH = 0.12

SERIES = [
    ("nc_prefix", "nc_prefix", "#0072B2"),
    ("carbon3b", "Carbon 3B", "#E69F00"),
    ("carbon_nc_fused", "Carbon+nc", "#009E73"),
    ("evo2_7b", "Evo2 7B", "#56B4E9"),
    ("evo2_nc_fused", "Evo2+nc", "#785EF0"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot comprehensive DNACorpus w8192 comparison bars.")
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=COMPREHENSIVE_DIR,
    )
    parser.add_argument("--output-prefix", default="")
    parser.add_argument(
        "--carbon-total-csv",
        type=Path,
        default=COMPREHENSIVE_DIR / "dnacorpus_w8192_species_curve_summary.csv",
    )
    parser.add_argument(
        "--carbon-region-csv",
        type=Path,
        default=COMPREHENSIVE_DIR / "dnacorpus_w8192_carbon_nc_region_summary_with_corrected_supplement.csv",
    )
    parser.add_argument(
        "--carbon-fusion-csv",
        type=Path,
        default=COMPREHENSIVE_DIR
        / "fusion_online_hedge_eta0.05_init0.5_position_major_v1"
        / "carbon3b_nc_prefix_fusion_summary_with_corrected_supplement.csv",
    )
    parser.add_argument(
        "--carbon-fused-region-csv",
        type=Path,
        default=COMPREHENSIVE_DIR
        / "fusion_online_hedge_eta0.05_init0.5_position_major_v1"
        / "annotation_region_analysis"
        / "region_class_summary_with_corrected_supplement.csv",
    )
    parser.add_argument(
        "--evo2-fusion-csv",
        type=Path,
        default=COMPREHENSIVE_DIR
        / "evo2_7b_nc_prefix_fusion_eta0.05_init0.5_position_major_v1"
        / "evo2_7b_nc_prefix_fusion_summary_with_corrected_supplement.csv",
    )
    parser.add_argument(
        "--evo2-region-csv",
        type=Path,
        default=COMPREHENSIVE_DIR
        / "evo2_7b_annotation_region_analysis"
        / "region_class_summary_with_corrected_supplement.csv",
    )
    parser.add_argument(
        "--evo2-fused-region-csv",
        type=Path,
        default=COMPREHENSIVE_DIR
        / "evo2_7b_nc_prefix_fusion_eta0.05_init0.5_annotation_region_analysis"
        / "region_class_summary_with_corrected_supplement.csv",
    )
    parser.add_argument(
        "--min-region-bases",
        type=int,
        default=SMALL_REGION_BASE_THRESHOLD,
        help="Exclude per-species annotation regions with fewer bases than this from bars and weighted values.",
    )
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _has_number(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _style_axis(ax) -> None:
    ax.grid(axis="y", color="#d6d6d6", linewidth=0.8, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_color("#333333")
        ax.spines[spine].set_linewidth(0.85)


def _set_plot_style(*, region: bool = False) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 16 if not region else 14,
            "axes.titlesize": 17,
            "axes.labelsize": 18 if not region else 15,
            "xtick.labelsize": 13 if not region else 11,
            "ytick.labelsize": 15 if not region else 12,
            "legend.fontsize": 15 if not region else 13,
        }
    )


def _x_layout(species: list[str]) -> tuple[np.ndarray, float, float | None]:
    x = np.arange(len(species), dtype=np.float64)
    separator_x: float | None = None
    left, right = EUKARYOTE_PROKARYOTE_SEPARATOR
    if left in species and right in species:
        left_index = species.index(left)
        right_index = species.index(right)
        if right_index == left_index + 1:
            x[right_index:] += GROUP_GAP
            separator_x = 0.5 * (x[left_index] + x[right_index])
    weighted_x = x[-1] + WEIGHTED_GAP + GROUP_GAP
    return x, weighted_x, separator_x


def _mark_small_region_counts(ax, x: np.ndarray, base_counts: np.ndarray) -> None:
    small_mask = (base_counts > 0) & (base_counts < SMALL_REGION_BASE_THRESHOLD)
    if not np.any(small_mask):
        return
    ymin, ymax = ax.get_ylim()
    y = ymin + 0.035 * (ymax - ymin)
    ax.scatter(
        x[small_mask],
        np.full((int(np.sum(small_mask)),), y, dtype=np.float64),
        marker="v",
        s=28,
        color="#6b7280",
        edgecolor="white",
        linewidth=0.4,
        zorder=5,
        clip_on=False,
    )


def _draw_species_group_separator(ax, species: list[str], *, label: bool = False) -> None:
    left, right = EUKARYOTE_PROKARYOTE_SEPARATOR
    if left not in species or right not in species:
        return
    left_index = species.index(left)
    right_index = species.index(right)
    if right_index != left_index + 1:
        return
    boundary = left_index + 0.5
    ax.axvline(boundary, color="#6b7280", linewidth=1.1, linestyle=(0, (3, 4)), alpha=0.55, zorder=3)


def _species_label(source: str, bases: int) -> str:
    if bases >= 1_000_000:
        return f"{source}\n{bases / 1e6:.1f}M"
    return f"{source}\n{bases / 1e3:.0f}K"


def _weighted(values: list[tuple[float, int]]) -> float | None:
    clean = [(float(value), int(weight)) for value, weight in values if np.isfinite(float(value)) and int(weight) > 0]
    total = sum(weight for _, weight in clean)
    if total <= 0:
        return None
    return sum(value * weight for value, weight in clean) / total


def _collect_total_rows(args: argparse.Namespace) -> tuple[list[str], dict[str, int], dict[str, dict[str, float]]]:
    carbon_rows = _read_csv(_resolve(args.carbon_total_csv))
    carbon_fusion_rows = {row["species"]: row for row in _read_csv(_resolve(args.carbon_fusion_csv))}
    evo2_fusion_rows = {row["species"]: row for row in _read_csv(_resolve(args.evo2_fusion_csv))}
    order = {species: index for index, species in enumerate(PLOT_SPECIES_ORDER)}
    carbon_rows = [
        row for row in carbon_rows
        if row["source"] in order and row["source"] not in PLOT_EXCLUDED_SPECIES
    ]
    carbon_rows = sorted(carbon_rows, key=lambda row: order.get(row["source"], 999))

    bases: dict[str, int] = {}
    values: dict[str, dict[str, float]] = {key: {} for key, _, _ in SERIES}
    for row in carbon_rows:
        species = row["source"]
        bases[species] = int(row["sample_bases"])
        values["carbon3b"][species] = float(row["left_full_sequence_bpb"])
        values["nc_prefix"][species] = float(row["right_full_sequence_bpb"])
        if species in carbon_fusion_rows:
            values["carbon_nc_fused"][species] = float(carbon_fusion_rows[species]["fused_theoretical_bpb"])
        if species in evo2_fusion_rows:
            values["evo2_7b"][species] = float(evo2_fusion_rows[species]["lm_only_theoretical_bpb"])
            values["evo2_nc_fused"][species] = float(evo2_fusion_rows[species]["fused_theoretical_bpb"])
    return [row["source"] for row in carbon_rows], bases, values


def _plot_total(args: argparse.Namespace) -> dict[str, Any]:
    comparison_dir = _resolve(args.comparison_dir)
    species, bases, values = _collect_total_rows(args)
    x, weighted_x, separator_x = _x_layout(species)
    width = BAR_WIDTH

    _set_plot_style()
    fig, ax = plt.subplots(figsize=(22.0, 8.0), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    _style_axis(ax)

    offsets = (np.arange(len(SERIES), dtype=np.float64) - (len(SERIES) - 1) / 2.0) * width
    finite_values: list[float] = []
    weighted_rows: dict[str, float | None] = {}
    for series_index, (key, label, color) in enumerate(SERIES):
        y = np.asarray([values[key].get(item, np.nan) for item in species], dtype=np.float64)
        mask = np.isfinite(y)
        finite_values.extend(y[mask].tolist())
        ax.bar(x[mask] + offsets[series_index], y[mask], width=width, color=color, alpha=0.88, label=label)
        weighted_rows[key] = _weighted([(values[key].get(item, np.nan), bases[item]) for item in species])
        if weighted_rows[key] is not None:
            weighted_value = float(weighted_rows[key])
            ax.bar(
                weighted_x + offsets[series_index],
                weighted_value,
                width=width,
                color=color,
                alpha=0.88,
            )
            finite_values.append(weighted_value)

    total_bases = sum(bases.values())
    ax.set_xticks(np.append(x, weighted_x))
    ax.set_xticklabels([*[_species_label(item, bases[item]) for item in species], _species_label("Weighted", total_bases)])
    ax.set_xlim(-0.55, weighted_x + 0.55)
    if separator_x is not None:
        ax.axvline(separator_x, color="#6b7280", linewidth=1.1, linestyle=(0, (3, 4)), alpha=0.55, zorder=3)
    ax.axvline(
        0.5 * (x[-1] + weighted_x),
        color="#6b7280",
        linewidth=1.1,
        linestyle=(0, (3, 4)),
        alpha=0.55,
        zorder=3,
    )
    ax.set_ylabel("Bits per base (lower is better)")
    if finite_values:
        ax.set_ylim(max(0.0, min(finite_values) - 0.08), max(finite_values) + 0.08)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        frameon=True,
        facecolor="white",
        edgecolor="#c8c8c8",
        ncol=5,
    )
    output_path = comparison_dir / f"{args.output_prefix}_total_bpb_by_species.png"
    summary_path = comparison_dir / f"{args.output_prefix}_total_bpb_by_species.json"
    fig.subplots_adjust(left=0.070, right=0.990, top=0.875, bottom=0.17)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary = {
        "plot_path": str(output_path),
        "series_order": [key for key, _, _ in SERIES],
        "species_count": len(species),
        "total_bases": total_bases,
        "weighted_bpb": weighted_rows,
        "species_group_separator": {
            "between": list(EUKARYOTE_PROKARYOTE_SEPARATOR),
            "label": "eukaryotes_to_prokaryotes_and_viral",
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _collect_region_values(args: argparse.Namespace) -> tuple[list[str], dict[str, dict[tuple[str, str], tuple[float, int]]]]:
    carbon_region = _read_csv(_resolve(args.carbon_region_csv))
    carbon_fused_region = _read_csv(_resolve(args.carbon_fused_region_csv))
    evo2_region = _read_csv(_resolve(args.evo2_region_csv))
    evo2_fused_region = _read_csv(_resolve(args.evo2_fused_region_csv))

    values: dict[str, dict[tuple[str, str], tuple[float, int]]] = {key: {} for key, _, _ in SERIES}
    species_seen = set()
    for row in carbon_region:
        key = (row["species"], row["region_class"])
        count = int(row["base_count"])
        species_seen.add(row["species"])
        if _has_number(row.get("carbon3b_mean_bpb")):
            values["carbon3b"][key] = (float(row["carbon3b_mean_bpb"]), count)
        if _has_number(row.get("nc_prefix_mean_bpb")):
            values["nc_prefix"][key] = (float(row["nc_prefix_mean_bpb"]), count)
    for row in carbon_fused_region:
        key = (row["species"], row["region_class"])
        if not _has_number(row.get("mean_bpb")):
            continue
        values["carbon_nc_fused"][key] = (float(row["mean_bpb"]), int(row["base_count"]))
        species_seen.add(row["species"])
    for row in evo2_region:
        key = (row["species"], row["region_class"])
        if not _has_number(row.get("mean_bpb")):
            continue
        values["evo2_7b"][key] = (float(row["mean_bpb"]), int(row["base_count"]))
        species_seen.add(row["species"])
    for row in evo2_fused_region:
        key = (row["species"], row["region_class"])
        if not _has_number(row.get("mean_bpb")):
            continue
        values["evo2_nc_fused"][key] = (float(row["mean_bpb"]), int(row["base_count"]))
        species_seen.add(row["species"])
    species = list(REGION_SPECIES_WITH_ANNOTATIONS)
    return species, values


def _plot_regions(args: argparse.Namespace) -> dict[str, Any]:
    comparison_dir = _resolve(args.comparison_dir)
    species, values = _collect_region_values(args)
    x, weighted_x, separator_x = _x_layout(species)
    width = BAR_WIDTH

    _set_plot_style(region=True)
    fig, axes = plt.subplots(2, 3, figsize=(23.0, 12.5), dpi=180, sharex=False)
    fig.patch.set_facecolor("white")
    axes_flat = axes.reshape(-1)
    offsets = (np.arange(len(SERIES), dtype=np.float64) - (len(SERIES) - 1) / 2.0) * width
    global_rows: list[dict[str, Any]] = []
    small_region_rows: list[dict[str, Any]] = []

    for ax, region in zip(axes_flat, REGION_ORDER):
        _style_axis(ax)
        finite_values: list[float] = []
        weighted_region: dict[str, float | None] = {}
        region_base_counts = np.zeros((len(species),), dtype=np.int64)
        for series_index, (series_key, label, color) in enumerate(SERIES):
            y = np.asarray(
                [values[series_key].get((item, region), (np.nan, 0))[0] for item in species],
                dtype=np.float64,
            )
            counts = np.asarray(
                [values[series_key].get((item, region), (np.nan, 0))[1] for item in species],
                dtype=np.int64,
            )
            region_base_counts = np.maximum(region_base_counts, counts)
            mask = np.isfinite(y) & (counts >= int(args.min_region_bases))
            finite_values.extend(y[mask].tolist())
            ax.bar(x[mask] + offsets[series_index], y[mask], width=width, color=color, alpha=0.88, label=label)
            weighted_region[series_key] = _weighted([(float(y[idx]), int(counts[idx])) for idx in np.flatnonzero(mask)])
            if weighted_region[series_key] is not None:
                weighted_value = float(weighted_region[series_key])
                ax.bar(
                    weighted_x + offsets[series_index],
                    weighted_value,
                    width=width,
                    color=color,
                    alpha=0.88,
                )
                finite_values.append(weighted_value)
        ax.set_xticks(np.append(x, weighted_x))
        ax.set_xticklabels([*species, "Weighted"], rotation=45, ha="right")
        ax.set_xlim(-0.55, weighted_x + 0.55)
        ax.set_ylabel("Bits/base")
        ax.set_title(REGION_LABELS.get(region, region))
        if separator_x is not None:
            ax.axvline(
                separator_x,
                color="#6b7280",
                linewidth=1.1,
                linestyle=(0, (3, 4)),
                alpha=0.55,
                zorder=3,
            )
        ax.axvline(
            0.5 * (x[-1] + weighted_x),
            color="#6b7280",
            linewidth=1.1,
            linestyle=(0, (3, 4)),
            alpha=0.55,
            zorder=3,
        )
        if finite_values:
            ymin = max(0.0, min(finite_values) - 0.06)
            ymax = max(finite_values) + 0.08
            if ymax - ymin < 0.20:
                center = 0.5 * (ymin + ymax)
                ymin = max(0.0, center - 0.10)
                ymax = center + 0.10
            ax.set_ylim(ymin, ymax)
        small_species = [
            {"species": species[idx], "base_count": int(region_base_counts[idx])}
            for idx in np.flatnonzero((region_base_counts > 0) & (region_base_counts < int(args.min_region_bases)))
        ]
        if small_species:
            _mark_small_region_counts(ax, x, region_base_counts)
            small_region_rows.append({"region_class": region, "species": small_species})
        global_rows.append(
            {"region_class": region, "weighted_bpb": weighted_region, "small_base_count_species": small_species}
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#c8c8c8",
    )
    output_path = comparison_dir / f"{args.output_prefix}_annotation_region_bpb_by_species_subplots.png"
    summary_path = comparison_dir / f"{args.output_prefix}_annotation_region_bpb_by_species_subplots.json"
    fig.subplots_adjust(left=0.052, right=0.990, top=0.875, bottom=0.08, wspace=0.18, hspace=0.34)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    summary = {
        "plot_path": str(output_path),
        "series_order": [key for key, _, _ in SERIES],
        "supported_species": species,
        "global_by_region": global_rows,
        "small_region_base_threshold": int(args.min_region_bases),
        "small_region_rows": small_region_rows,
        "notes": [
            "AnCa and WaMe are omitted from annotation-region plots because they have no coordinate-verified region data.",
            "A subtle dashed separator is drawn between the eukaryotic/protist group and EsCo/prokaryotic/viral entries.",
            "The RNA panel is restricted to non-mRNA RNA features such as tRNA/rRNA/ncRNA labels.",
            "Empty bars mean zero exclusive bases for that region under the priority label scheme, not necessarily biological absence.",
            f"Small gray triangle markers flag regions with fewer than {int(args.min_region_bases)} exclusive bases; these regions are excluded from bars and weighted values.",
            "repeat_mobile_existing is sparse and reflects only repeat/mobile labels present in official GFF3.",
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = _build_parser().parse_args()
    comparison_dir = _resolve(args.comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "display_name": "DNACorpus w8192 comprehensive model comparison",
        "directory_role": "comprehensive_result_comparison",
        "series_order": [key for key, _, _ in SERIES],
    }
    (comparison_dir / "comprehensive_comparison_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total = _plot_total(args)
    region = _plot_regions(args)
    print(json.dumps({"total": total, "region": region}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
