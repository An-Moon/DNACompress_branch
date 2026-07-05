from __future__ import annotations

"""Build lightweight figures and tables used by the root README.

The script only reads existing experiment outputs. It does not run training,
compression, or model inference.
"""

import csv
from pathlib import Path
import shutil
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "docs" / "assets"

MEGABYTE_RUN = REPO_ROOT / "outputs" / "dna_megabyte_large_opengenome2_9"
MEGABYTE_DNACORPUS_SPEED = MEGABYTE_RUN / "statistics_dnacorpus_full" / "compression_speed_summary.csv"
MEGABYTE_DNACORPUS_RATIO = MEGABYTE_RUN / "statistics_dnacorpus_full" / "compression_ratio_summary.csv"
MEGABYTE_OG2_SPEED = MEGABYTE_RUN / "statistics_opengenome2_fasta_100mb" / "compression_speed_summary.csv"
GECO2_DNACORPUS_FULL = REPO_ROOT / "outputs" / "dna_geco2_dnacorpus_fullsplit" / "compression_aggregate_by_split_mode.csv"
MODEL_REGION_BAR_CSV = (
    MEGABYTE_RUN
    / "region_bpb_compare_dnacorpus_50kb_seed12345_evo2_1b_7b_megabyte_geco2"
    / "combined_species_average_bpb_bar_values.csv"
)

SOURCE_FIGURES = {
    "megabyte_vs_geco2_bpb.png": (
        MEGABYTE_RUN
        / "statistics_dnacorpus_full"
        / "compression_curves"
        / "full_windows_nonoverlap_payload_only_compression_curves.png"
    ),
    "carbon_evo2_geco2_megabyte_bpb_bar.png": (
        MEGABYTE_RUN
        / "region_bpb_compare_dnacorpus_50kb_seed12345_evo2_1b_7b_megabyte_geco2"
        / "combined_species_average_bpb_bar.png"
    ),
    "orsa_long_context_bpb_curve.png": (
        MEGABYTE_RUN
        / "full_bpb_probe_dnacorpus"
        / "OrSa"
        / "region_bpb_curve.png"
    ),
    "orsa_combined_region_bpb_curve.png": (
        MEGABYTE_RUN
        / "region_bpb_compare_dnacorpus_50kb_seed12345_evo2_1b_7b_megabyte_geco2"
        / "OrSa"
        / "region_bpb_combined_curve.png"
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing {key!r} in row from CSV")
    return float(value)


def _first_row(path: Path, *, row_type: str | None = None) -> dict[str, str]:
    for row in _read_csv(path):
        if row_type is None or row.get("row_type") == row_type:
            return row
    raise ValueError(f"No matching row in {path}")


def _metric_rows() -> list[dict[str, Any]]:
    mb_full_speed = _first_row(MEGABYTE_DNACORPUS_SPEED, row_type="aggregate")
    mb_full_ratio = _first_row(MEGABYTE_DNACORPUS_RATIO, row_type="aggregate")
    mb_og2_speed = _first_row(MEGABYTE_OG2_SPEED, row_type="aggregate")
    geco2_full = _first_row(GECO2_DNACORPUS_FULL)
    mean_row = next(row for row in _read_csv(MODEL_REGION_BAR_CSV) if row.get("species") == "Mean")

    return [
        {
            "metric": "megabyte_dnacorpus_full_bpb",
            "value": _float(mb_full_ratio, "arithmetic_bits_per_base"),
            "unit": "bits/base",
            "source": str(MEGABYTE_DNACORPUS_RATIO.relative_to(REPO_ROOT)),
        },
        {
            "metric": "megabyte_dnacorpus_full_mbases_per_second",
            "value": _float(mb_full_speed, "compression_bases_per_second") / 1_000_000.0,
            "unit": "Mbases/s",
            "source": str(MEGABYTE_DNACORPUS_SPEED.relative_to(REPO_ROOT)),
        },
        {
            "metric": "megabyte_opengenome2_100mb_mbases_per_second",
            "value": _float(mb_og2_speed, "compression_bases_per_second") / 1_000_000.0,
            "unit": "Mbases/s",
            "source": str(MEGABYTE_OG2_SPEED.relative_to(REPO_ROOT)),
        },
        {
            "metric": "megabyte_opengenome2_100mb_bpb",
            "value": _float(mb_og2_speed, "sample_bases")
            and _float(_first_row(MEGABYTE_RUN / "statistics_opengenome2_fasta_100mb" / "compression_ratio_summary.csv", row_type="aggregate"), "arithmetic_bits_per_base"),
            "unit": "bits/base",
            "source": str((MEGABYTE_RUN / "statistics_opengenome2_fasta_100mb" / "compression_ratio_summary.csv").relative_to(REPO_ROOT)),
        },
        {
            "metric": "geco2_dnacorpus_full_bpb",
            "value": _float(geco2_full, "total_arithmetic_bits_per_base"),
            "unit": "bits/base",
            "source": str(GECO2_DNACORPUS_FULL.relative_to(REPO_ROOT)),
        },
        {
            "metric": "geco2_dnacorpus_full_mbases_per_second",
            "value": _float(geco2_full, "total_compression_bases_per_second") / 1_000_000.0,
            "unit": "Mbases/s",
            "source": str(GECO2_DNACORPUS_FULL.relative_to(REPO_ROOT)),
        },
        {
            "metric": "region_mean_carbon_500m_bpb",
            "value": _float(mean_row, "carbon_500m_bpb"),
            "unit": "bits/base",
            "source": str(MODEL_REGION_BAR_CSV.relative_to(REPO_ROOT)),
        },
        {
            "metric": "region_mean_evo2_1b_base_bpb",
            "value": _float(mean_row, "evo2_1b_base_bpb"),
            "unit": "bits/base",
            "source": str(MODEL_REGION_BAR_CSV.relative_to(REPO_ROOT)),
        },
        {
            "metric": "region_mean_evo2_7b_base_bpb",
            "value": _float(mean_row, "evo2_7b_base_bpb"),
            "unit": "bits/base",
            "source": str(MODEL_REGION_BAR_CSV.relative_to(REPO_ROOT)),
        },
        {
            "metric": "region_mean_geco2_bpb",
            "value": _float(mean_row, "geco22_bpb"),
            "unit": "bits/base",
            "source": str(MODEL_REGION_BAR_CSV.relative_to(REPO_ROOT)),
        },
        {
            "metric": "region_mean_megabyte_bpb",
            "value": _float(mean_row, "megabyte1_bpb"),
            "unit": "bits/base",
            "source": str(MODEL_REGION_BAR_CSV.relative_to(REPO_ROOT)),
        },
    ]


def _copy_reference_figures() -> None:
    for output_name, source in SOURCE_FIGURES.items():
        if not source.exists():
            raise FileNotFoundError(f"Missing README source figure: {source}")
        shutil.copy2(source, ASSET_DIR / output_name)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_metrics_summary(rows: list[dict[str, Any]]) -> None:
    path = ASSET_DIR / "readme_metrics_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "unit", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _write_speed_plot(rows: list[dict[str, Any]]) -> None:
    plt = _load_pyplot()

    value_by_metric = {row["metric"]: float(row["value"]) for row in rows}
    labels = [
        "Megabyte\nDNACorpus full",
        "GECO2\nDNACorpus full",
        "Megabyte\nOpenGenome2 100MB/src",
    ]
    values = [
        value_by_metric["megabyte_dnacorpus_full_mbases_per_second"],
        value_by_metric["geco2_dnacorpus_full_mbases_per_second"],
        value_by_metric["megabyte_opengenome2_100mb_mbases_per_second"],
    ]
    colors = ["#2563eb", "#64748b", "#0f766e"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_ylabel("Compression throughput (Mbases/s)")
    ax.set_title("Compression speed from existing experiment summaries")
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.035,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.text(
        0.01,
        -0.24,
        "Megabyte values use the fastest valid summaries for the selected task; some slower runs overlapped other jobs.",
        transform=ax.transAxes,
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "compression_speed_comparison.png", bbox_inches="tight")
    plt.close(fig)


def _write_codec_diagram() -> None:
    plt = _load_pyplot()
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(10.4, 4.9), dpi=180)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    def box(x: float, y: float, w: float, h: float, label: str, color: str) -> None:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=1.2,
                edgecolor="#0f172a",
                facecolor=color,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10, wrap=True)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#334155"},
        )

    box(0.35, 3.15, 1.55, 0.82, "DNA sequence", "#dbeafe")
    box(2.25, 3.15, 1.85, 0.82, "Split into\nindependent windows", "#e0f2fe")
    box(4.55, 3.15, 1.85, 0.82, "Parallel LM\nprobabilities", "#dcfce7")
    box(6.85, 3.15, 1.55, 0.82, "Arithmetic\nencode", "#fef3c7")
    box(8.75, 3.15, 0.9, 0.82, ".mbw\nbits", "#fee2e2")

    arrow(1.9, 3.56, 2.25, 3.56)
    arrow(4.1, 3.56, 4.55, 3.56)
    arrow(6.4, 3.56, 6.85, 3.56)
    arrow(8.4, 3.56, 8.75, 3.56)

    for idx, x in enumerate([2.32, 2.85, 3.38]):
        box(x, 2.35, 0.42, 0.34, f"W{idx + 1}", "#f8fafc")
    ax.text(3.92, 2.52, "...", ha="left", va="center", fontsize=12, color="#334155")
    arrow(3.72, 2.7, 4.55, 3.28)

    box(6.85, 0.92, 1.55, 0.82, "Arithmetic\ndecode", "#fef3c7")
    box(4.55, 0.92, 1.85, 0.82, "Same LM\nprobabilities", "#dcfce7")
    box(2.25, 0.92, 1.85, 0.82, "Recovered\nwindows", "#e0f2fe")
    box(0.35, 0.92, 1.55, 0.82, "Original DNA", "#dbeafe")
    arrow(8.75, 3.08, 8.0, 1.78)
    arrow(6.85, 1.33, 6.4, 1.33)
    arrow(4.55, 1.33, 4.1, 1.33)
    arrow(2.25, 1.33, 1.9, 1.33)

    ax.text(0.35, 4.45, "Compression", fontsize=13, fontweight="bold", color="#0f172a")
    ax.text(0.35, 0.2, "Decompression is the reversible path with the same model and window boundaries.", fontsize=10, color="#334155")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "lm_window_codec_diagram.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rows = _metric_rows()
    _copy_reference_figures()
    _write_metrics_summary(rows)
    _write_speed_plot(rows)
    _write_codec_diagram()
    print(f"Wrote README assets to {ASSET_DIR}")
    for row in rows:
        value = row["value"]
        print(f"{row['metric']}: {float(value):.6g} {row['unit']}")


if __name__ == "__main__":
    main()
