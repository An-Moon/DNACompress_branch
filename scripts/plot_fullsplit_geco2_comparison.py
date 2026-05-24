from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MODEL_DIR = Path("outputs/dna_megabyte_large_b128_ensembl_all_finetune/statistics_fullsplit_nocodec")
DEFAULT_GECO2_DIR = Path("outputs/dna_geco2_paper_modes_0p6_0p2_0p2_fullsplit")
DEFAULT_DNACORPUS_ORDER = (
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
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _model_sidecar_csv(model_dir: Path, filename: str) -> Path:
    root_path = model_dir / filename
    if root_path.exists():
        return root_path
    statistics_path = model_dir / "statistics" / filename
    if statistics_path.exists():
        return statistics_path
    raise FileNotFoundError(f"Missing {filename} in {model_dir} or {model_dir / 'statistics'}")


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _default_model_label(model_dir: Path) -> str:
    name = model_dir.name.lower()
    if "dnagpt" in name:
        return "DNA-GPT"
    if "megabyte" in name:
        return "MEGABYTE"
    return "Model"


def _species_order(dataset_rows: list[dict[str, str]], requested_order: list[str] | None) -> dict[str, int]:
    if requested_order:
        order = {name: index for index, name in enumerate(requested_order)}
        fallback_start = len(order)
        for row in dataset_rows:
            name = row.get("source_name") or row.get("species") or ""
            if name and name not in order:
                order[name] = fallback_start
                fallback_start += 1
        return order
    return {
        row.get("source_name") or row.get("species") or str(index): index
        for index, row in enumerate(dataset_rows)
    }


def build_rows(
    *,
    model_dir: Path,
    geco2_dir: Path,
    species_order: list[str] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model_metrics = json.loads((model_dir / "compression_compare.json").read_text(encoding="utf-8"))
    model_rows = []
    for split, split_payload in model_metrics.get("results", {}).items():
        mode_payload = split_payload.get("windows_nonoverlap", {})
        for row in mode_payload.get("per_source", []):
            model_rows.append({"split": split, **row})
    dataset_rows = _read_csv(_model_sidecar_csv(model_dir, "dataset_splits.csv"))
    geco2_split_rows = _read_csv(geco2_dir / "geco2_split_results.csv")
    paper_rows = _read_csv(geco2_dir / "paper_total_baseline.csv")

    order_map = _species_order(dataset_rows, species_order)
    paper_by_source = {
        row["source_name"]: row
        for row in paper_rows
    }
    geco2_by_split_source = {
        (row["split"], row["source_name"]): row
        for row in geco2_split_rows
    }

    comparison_rows: list[dict[str, object]] = []
    for row in model_rows:
        split = row["split"]
        source = row.get("source_name") or row["species"]
        model_bpb = float(row["arithmetic_bits_per_base"])
        geco2_split = geco2_by_split_source.get((split, source))
        paper = paper_by_source.get(source)
        geco2_split_bpb = _float(geco2_split.get("bits_per_base")) if geco2_split else None
        paper_bpb = _float(paper.get("paper_total_bits_per_base")) if paper else None
        sample_bytes = int(row["sample_bytes"])

        comparison_rows.append(
            {
                "split": split,
                "species": row["species"],
                "source_name": source,
                "source_order": order_map.get(source, 9999),
                "sample_bytes": sample_bytes,
                "model_bits_per_base": model_bpb,
                "model_compressed_bytes": int(row["arithmetic_coded_bytes"]),
                "model_compression_seconds": float(row["compression_process_seconds"]),
                "model_bases_per_second": float(row["compression_bases_per_second"]),
                "geco2_split_bits_per_base": geco2_split_bpb,
                "geco2_split_compressed_bytes": int(geco2_split["compressed_bytes"]) if geco2_split else None,
                "geco2_split_seconds": _float(geco2_split.get("compression_seconds")) if geco2_split else None,
                "geco2_split_bases_per_second": _float(geco2_split.get("compression_bases_per_second")) if geco2_split else None,
                "paper_total_bits_per_base": paper_bpb,
                "paper_total_compressed_bytes": int(paper["paper_total_compressed_bytes"]) if paper else None,
                "paper_total_geco2_level": int(paper["paper_total_geco2_level"]) if paper else None,
                "model_vs_geco2_split_delta_bpb": model_bpb - geco2_split_bpb if geco2_split_bpb is not None else None,
                "model_vs_geco2_split_percent": model_bpb / geco2_split_bpb * 100.0 if geco2_split_bpb else None,
                "model_vs_paper_total_percent": model_bpb / paper_bpb * 100.0 if paper_bpb else None,
            }
        )

    aggregate_rows: list[dict[str, object]] = []
    model_agg = {row["split"]: row for row in _read_csv(_model_sidecar_csv(model_dir, "compression_aggregate_by_split_mode.csv"))}
    geco2_agg = {row["split"]: row for row in _read_csv(geco2_dir / "geco2_split_aggregate.csv")}
    total_paper_bytes = sum(int(row["paper_total_compressed_bytes"]) for row in paper_rows)
    total_paper_size = sum(int(row["total_size"]) for row in paper_rows)
    paper_total_bpb = total_paper_bytes * 8 / total_paper_size
    for split in ("train", "val", "test"):
        model = model_agg[split]
        geco2 = geco2_agg[split]
        aggregate_rows.append(
            {
                "split": split,
                "model_bits_per_base": float(model["total_arithmetic_bits_per_base"]),
                "geco2_split_bits_per_base": float(geco2["total_bits_per_base"]),
                "paper_total_bits_per_base": paper_total_bpb,
                "model_compressed_bytes": int(model["total_arithmetic_coded_bytes"]),
                "geco2_split_compressed_bytes": int(geco2["total_compressed_bytes"]),
                "paper_total_compressed_bytes": total_paper_bytes,
                "model_compression_seconds": float(model["total_compression_process_seconds"]),
                "geco2_split_seconds": float(geco2["total_compression_seconds"]),
                "model_bases_per_second": float(model["total_compression_bases_per_second"]),
                "geco2_split_bases_per_second": float(geco2["total_compression_bases_per_second"]),
            }
        )
    split_order = {"train": 0, "val": 1, "test": 2}
    comparison_rows.sort(
        key=lambda row: (
            split_order.get(str(row["split"]), 999),
            int(row["source_order"]),
            str(row["source_name"]),
        )
    )
    return comparison_rows, aggregate_rows


def plot_split_rows(*, rows: list[dict[str, object]], out_dir: Path, model_label: str) -> list[Path]:
    plt = _load_pyplot()
    generated: list[Path] = []
    for split in ("train", "val", "test"):
        split_rows = sorted(
            [row for row in rows if row["split"] == split],
            key=lambda row: int(row["source_order"]),
        )
        labels = [str(row["source_name"]) for row in split_rows]
        x = list(range(len(labels)))
        fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, height_ratios=[3, 2])

        axes[0].plot(x, [row["model_bits_per_base"] for row in split_rows], marker="o", label=model_label)
        axes[0].plot(x, [row["geco2_split_bits_per_base"] for row in split_rows], marker="s", label="GeCo2 split experiment")
        axes[0].plot(x, [row["paper_total_bits_per_base"] for row in split_rows], marker="^", linestyle="--", label="GeCo2 paper total")
        axes[0].set_ylabel("Bits per base")
        axes[0].set_title(f"{split} compression bits/base")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend()

        axes[1].bar(
            [value - 0.18 for value in x],
            [row["model_bases_per_second"] / 1_000_000 for row in split_rows],
            width=0.36,
            label=model_label,
        )
        axes[1].bar(
            [value + 0.18 for value in x],
            [row["geco2_split_bases_per_second"] / 1_000_000 for row in split_rows],
            width=0.36,
            label="GeCo2 split",
        )
        axes[1].set_ylabel("M bases/s")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend()
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")

        fig.tight_layout()
        path = out_dir / f"{split}_fullsplit_geco2_comparison.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        generated.append(path)
    return generated


def plot_aggregate(*, rows: list[dict[str, object]], out_dir: Path, model_label: str) -> Path:
    plt = _load_pyplot()
    labels = [str(row["split"]) for row in rows]
    x = list(range(len(labels)))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar([v - width for v in x], [row["model_bits_per_base"] for row in rows], width=width, label=model_label)
    axes[0].bar(x, [row["geco2_split_bits_per_base"] for row in rows], width=width, label="GeCo2 split experiment")
    axes[0].bar([v + width for v in x], [row["paper_total_bits_per_base"] for row in rows], width=width, label="GeCo2 paper total")
    axes[0].set_ylabel("Bits per base")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar([v - width / 2 for v in x], [row["model_bases_per_second"] / 1_000_000 for row in rows], width=width, label=model_label)
    axes[1].bar([v + width / 2 for v in x], [row["geco2_split_bases_per_second"] / 1_000_000 for row in rows], width=width, label="GeCo2 split")
    axes[1].set_ylabel("M bases/s")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    fig.suptitle("Aggregate full-split compression comparison")
    fig.tight_layout()
    path = out_dir / "aggregate_fullsplit_geco2_comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--geco2-dir", default=str(DEFAULT_GECO2_DIR))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--species-order",
        nargs="+",
        default=list(DEFAULT_DNACORPUS_ORDER),
        help="Plot order for source names. Defaults to docs/DNACORPUS_SPECIES_NOTES.md order.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Label used for the model series in legends. Defaults to an inferred label from --model-dir.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    geco2_dir = Path(args.geco2_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "geco2_comparison_curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows, aggregate_rows = build_rows(
        model_dir=model_dir,
        geco2_dir=geco2_dir,
        species_order=args.species_order,
    )
    model_label = args.model_label or _default_model_label(model_dir)
    comparison_csv = out_dir / "fullsplit_geco2_comparison_by_source.csv"
    aggregate_csv = out_dir / "fullsplit_geco2_comparison_aggregate.csv"
    _write_csv(comparison_csv, comparison_rows)
    _write_csv(aggregate_csv, aggregate_rows)
    generated = [comparison_csv, aggregate_csv]
    generated.extend(plot_split_rows(rows=comparison_rows, out_dir=out_dir, model_label=model_label))
    generated.append(plot_aggregate(rows=aggregate_rows, out_dir=out_dir, model_label=model_label))
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
