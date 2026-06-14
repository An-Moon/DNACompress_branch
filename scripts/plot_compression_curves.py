from __future__ import annotations

"""Generate per-source compression comparison plots from exported compression results.

This script scans an output directory recursively for ``compression_compare.json`` files.
For each statistics directory that contains one, it generates per split/mode artifacts:

- ``<split>_<mode>_compression_curves.png``: a 3-panel plot showing
  arithmetic bits-per-base, percentage of 2-bit encoding, and compression speed.
- ``<split>_<mode>_compression_curve_data.csv``: the tabular values used in the plot.

Example:

    python scripts/plot_compression_curves.py \
      --root-dir outputs/dna_dnagpt_0p1bm_all_finetune
"""

import argparse
import csv
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
GECO2_EXPERIMENT_MODE_NAME = "geco2_paper_modes"
GECO2_BASELINE_ALIASES: dict[str, Path] = {
    "dnacorpus": Path("outputs/dna_geco2_dnacorpus_0p6_0p2_0p2/compression_compare.json"),
    "dnacorpus_0p6_0p2_0p2": Path("outputs/dna_geco2_dnacorpus_0p6_0p2_0p2/compression_compare.json"),
    "dnacorpus_split": Path("outputs/dna_geco2_dnacorpus_0p6_0p2_0p2/compression_compare.json"),
    "dnacorpus_full": Path("outputs/dna_geco2_dnacorpus_fullsplit/compression_compare.json"),
    "dnacorpus_fullsplit": Path("outputs/dna_geco2_dnacorpus_fullsplit/compression_compare.json"),
    "opengenome2": Path("outputs/dna_geco2_opengenome2_subset_100mb_per_source/compression_compare.json"),
    "opengenome2_100mb": Path("outputs/dna_geco2_opengenome2_subset_100mb_per_source/compression_compare.json"),
    "opengenome2_subset_100mb_per_source": Path(
        "outputs/dna_geco2_opengenome2_subset_100mb_per_source/compression_compare.json"
    ),
}

DNACORPUS_SOURCE_ORDER = {
    source_name: index
    for index, source_name in enumerate(
        (
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
    )
}

GECO2_PAPER_BASELINE_BY_SOURCE: dict[str, dict[str, int]] = {
    "HoSa": {"compressed_bytes": 38_845_642, "mode": 12},
    "GaGa": {"compressed_bytes": 33_877_671, "mode": 11},
    "DaRe": {"compressed_bytes": 11_488_819, "mode": 10},
    "OrSa": {"compressed_bytes": 8_646_543, "mode": 10},
    "DrMe": {"compressed_bytes": 7_481_093, "mode": 10},
    "EnIn": {"compressed_bytes": 5_170_889, "mode": 9},
    "ScPo": {"compressed_bytes": 2_518_963, "mode": 8},
    "PlFa": {"compressed_bytes": 1_925_726, "mode": 7},
    "EsCo": {"compressed_bytes": 1_098_552, "mode": 6},
    "HaHi": {"compressed_bytes": 902_831, "mode": 5},
    "AeCa": {"compressed_bytes": 380_115, "mode": 5},
    "HePy": {"compressed_bytes": 375_481, "mode": 4},
    "YeMi": {"compressed_bytes": 16_798, "mode": 3},
    "AgPh": {"compressed_bytes": 10_708, "mode": 2},
    "BuEb": {"compressed_bytes": 4_686, "mode": 1},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_geco2_baseline_path(selector: str | None) -> Path | None:
    if selector is None:
        return None
    normalized = selector.strip()
    if not normalized or normalized.lower() in {"none", "off", "false", "no"}:
        return None

    alias_path = GECO2_BASELINE_ALIASES.get(normalized)
    if alias_path is not None:
        return REPO_ROOT / alias_path

    if normalized.startswith("dna_geco2"):
        return REPO_ROOT / "outputs" / normalized / "compression_compare.json"

    path = Path(normalized).expanduser()
    if path.is_dir() or path.suffix != ".json":
        return path / "compression_compare.json"
    return path


def _load_matplotlib_pyplot():
    try:
        matplotlib = importlib.import_module("matplotlib")
    except ImportError as error:
        raise RuntimeError(
            "Compression curve export requires matplotlib. Install it or rerun without this script."
        ) from error
    matplotlib.use("Agg")
    return importlib.import_module("matplotlib.pyplot")


def _sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return sanitized or "artifact"


def _artifact_stem(split_name: str, mode_name: str) -> str:
    sanitized_split = _sanitize_filename(split_name)
    sanitized_mode = _sanitize_filename(mode_name)
    if sanitized_split == "train" and sanitized_mode.startswith("windows_"):
        return sanitized_mode
    return f"{sanitized_split}_{sanitized_mode}"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _source_order_map(compression_compare: dict[str, Any]) -> dict[str, int]:
    dataset = compression_compare.get("dataset")
    if not isinstance(dataset, dict):
        return {}

    species_rows = dataset.get("species")
    if not isinstance(species_rows, list):
        return {}

    dataset_order: dict[str, int] = {}
    source_names: set[str] = set()
    for index, row in enumerate(species_rows):
        if not isinstance(row, dict):
            continue
        source_name = row.get("source_name")
        species = row.get("species")
        if isinstance(source_name, str):
            source_names.add(source_name)
            if source_name not in dataset_order:
                dataset_order[source_name] = index
        if isinstance(species, str):
            source_names.add(species)
            if species not in dataset_order:
                dataset_order[species] = index

    if source_names and source_names.issubset(DNACORPUS_SOURCE_ORDER):
        return {
            source_name: DNACORPUS_SOURCE_ORDER[source_name]
            for source_name in source_names
        }

    return dataset_order


def _source_metadata_map(compression_compare: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dataset = compression_compare.get("dataset")
    if not isinstance(dataset, dict):
        return {}

    species_rows = dataset.get("species")
    if not isinstance(species_rows, list):
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    for row in species_rows:
        if not isinstance(row, dict):
            continue
        source_name = row.get("source_name")
        species = row.get("species")
        if isinstance(source_name, str):
            metadata.setdefault(source_name, row)
        if isinstance(species, str):
            metadata.setdefault(species, row)
    return metadata


def _geco2_paper_baseline_payload(
    *,
    source_name: str,
    species_name: str,
    total_size: float | None,
) -> dict[str, float | int | None]:
    for key in (source_name, species_name):
        baseline = GECO2_PAPER_BASELINE_BY_SOURCE.get(key)
        if baseline is None:
            continue
        compressed_bytes = int(baseline["compressed_bytes"])
        mode = int(baseline["mode"])
        if total_size is None or total_size <= 0:
            return {
                "paper_baseline_compressed_bytes": compressed_bytes,
                "paper_baseline_geco2_mode": mode,
                "paper_baseline_bpb": None,
                "paper_baseline_percent": None,
            }
        bpb = compressed_bytes * 8.0 / total_size
        return {
            "paper_baseline_compressed_bytes": compressed_bytes,
            "paper_baseline_geco2_mode": mode,
            "paper_baseline_bpb": bpb,
            "paper_baseline_percent": bpb / 2.0 * 100.0,
        }
    return {
        "paper_baseline_compressed_bytes": None,
        "paper_baseline_geco2_mode": None,
        "paper_baseline_bpb": None,
        "paper_baseline_percent": None,
    }


def _experiment_baseline_rows(compression_compare: dict[str, Any]) -> list[dict[str, Any]]:
    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return []

    rows: list[dict[str, Any]] = []
    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        mode_name, mode_payload = _select_geco2_mode_payload(split_payload)
        if mode_payload is None:
            continue
        per_source = mode_payload.get("per_source")
        if not isinstance(per_source, list):
            continue
        for row in per_source:
            if not isinstance(row, dict):
                continue
            source_name = str(row.get("source_name") or row.get("species") or "unknown")
            species_name = str(row.get("species") or source_name)
            bpb = _safe_float(row.get("arithmetic_bits_per_base"))
            compressed_bytes = _safe_float(row.get("arithmetic_coded_bytes"))
            payload = {
                "split": split_name,
                "species": species_name,
                "source_name": source_name,
                "sample_bytes": row.get("sample_bytes"),
                "sample_bases": row.get("sample_bases"),
                "experiment_baseline_compressed_bytes": int(compressed_bytes) if compressed_bytes is not None else None,
                "experiment_baseline_geco2_mode": row.get("geco2_level") or mode_name,
                "experiment_baseline_bpb": bpb,
                "experiment_baseline_percent": bpb / 2.0 * 100.0 if bpb is not None else None,
            }
            rows.append(payload)
    return rows


def _select_geco2_mode_payload(split_payload: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    preferred = split_payload.get(GECO2_EXPERIMENT_MODE_NAME)
    if isinstance(preferred, dict):
        return GECO2_EXPERIMENT_MODE_NAME, preferred
    for mode_name, mode_payload in split_payload.items():
        if str(mode_name).lower().startswith("geco2") and isinstance(mode_payload, dict):
            return str(mode_name), mode_payload
    return None, None


def _experiment_baseline_map(compression_compare: dict[str, Any], split_name: str) -> dict[str, dict[str, Any]]:
    rows = [row for row in _experiment_baseline_rows(compression_compare) if row.get("split") == split_name]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_name = row.get("source_name")
        species = row.get("species")
        if isinstance(source_name, str):
            result[source_name] = row
        if isinstance(species, str):
            result[species] = row
    return result


def _baseline_rows(compression_compare: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_map = _source_metadata_map(compression_compare)
    order_map = _source_order_map(compression_compare)
    source_names = sorted(
        {
            key
            for key in metadata_map
            if isinstance(metadata_map[key].get("source_name", metadata_map[key].get("species")), str)
        },
        key=lambda key: (order_map.get(key, 10**9), key),
    )

    paper_rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for source_name in source_names:
        metadata = metadata_map[source_name]
        species_name = str(metadata.get("species") or source_name)
        canonical_source_name = str(metadata.get("source_name") or source_name)
        if canonical_source_name in seen_sources:
            continue
        seen_sources.add(canonical_source_name)
        total_size = _safe_float(metadata.get("total_size"))
        paper_payload = _geco2_paper_baseline_payload(
            source_name=canonical_source_name,
            species_name=species_name,
            total_size=total_size,
        )
        if paper_payload["paper_baseline_compressed_bytes"] is not None:
            paper_rows.append(
                {
                    "species": species_name,
                    "source_name": canonical_source_name,
                    "total_size": total_size,
                    **paper_payload,
                }
            )

    return paper_rows, _experiment_baseline_rows(compression_compare)


def _paper_baseline_rows(compression_compare: dict[str, Any]) -> list[dict[str, Any]]:
    paper_rows, _ = _baseline_rows(compression_compare)
    return paper_rows


def _resolve_run_name(stats_dir: Path, compression_compare: dict[str, Any]) -> str:
    run_metadata_path = stats_dir / "run_metadata.json"
    if run_metadata_path.exists():
        run_metadata = _read_json(run_metadata_path)
        if isinstance(run_metadata.get("name"), str) and run_metadata["name"]:
            return str(run_metadata["name"])

    resolved_config = compression_compare.get("resolved_config")
    if isinstance(resolved_config, dict):
        output_cfg = resolved_config.get("output")
        if isinstance(output_cfg, dict):
            wandb_name = output_cfg.get("wandb_name")
            run_name = output_cfg.get("run_name")
            if isinstance(wandb_name, str) and wandb_name:
                return wandb_name
            if isinstance(run_name, str) and run_name:
                return run_name

    return stats_dir.name


def _build_split_mode_rows(
    *,
    compression_compare: dict[str, Any],
    experiment_baseline_compare: dict[str, Any] | None = None,
    split_name: str,
    mode_name: str,
) -> list[dict[str, Any]]:
    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return []

    split_payload = results.get(split_name)
    if not isinstance(split_payload, dict):
        return []

    mode_payload = split_payload.get(mode_name)
    if not isinstance(mode_payload, dict):
        return []

    per_source = mode_payload.get("per_source")
    if not isinstance(per_source, list):
        return []

    order_map = _source_order_map(compression_compare)
    metadata_map = _source_metadata_map(compression_compare)
    experiment_map = (
        _experiment_baseline_map(experiment_baseline_compare, split_name)
        if experiment_baseline_compare is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    for item in per_source:
        if not isinstance(item, dict):
            continue

        source_name = str(item.get("source_name") or item.get("species") or "unknown")
        species_name = str(item.get("species") or source_name)
        metadata = metadata_map.get(source_name) or metadata_map.get(species_name) or {}
        total_size = _safe_float(metadata.get("total_size"))
        arithmetic_bpb = _safe_float(item.get("arithmetic_bits_per_base"))
        theoretical_bpb = _safe_float(item.get("theoretical_bits_per_base"))
        compression_bases_per_second = _safe_float(item.get("compression_bases_per_second"))
        compression_bytes_per_second = _safe_float(item.get("compression_bytes_per_second"))
        paper_baseline = _geco2_paper_baseline_payload(
            source_name=source_name,
            species_name=species_name,
            total_size=total_size,
        )
        experiment_baseline = experiment_map.get(source_name) or experiment_map.get(species_name) or {}

        row = {
            "split": split_name,
            "mode": mode_name,
            "species": species_name,
            "source_name": source_name,
            "total_size": total_size,
            "sample_bytes": item.get("sample_bytes"),
            "sample_bases": item.get("sample_bases"),
            "arithmetic_bits_per_base": arithmetic_bpb,
            "theoretical_bits_per_base": theoretical_bpb,
            "vs_2bit_percent": (arithmetic_bpb / 2.0 * 100.0) if arithmetic_bpb is not None else None,
            **paper_baseline,
            "experiment_baseline_compressed_bytes": experiment_baseline.get("experiment_baseline_compressed_bytes"),
            "experiment_baseline_geco2_mode": experiment_baseline.get("experiment_baseline_geco2_mode"),
            "experiment_baseline_bpb": experiment_baseline.get("experiment_baseline_bpb"),
            "experiment_baseline_percent": experiment_baseline.get("experiment_baseline_percent"),
            "compression_bases_per_second": compression_bases_per_second,
            "compression_mbases_per_second": (
                compression_bases_per_second / 1_000_000.0 if compression_bases_per_second is not None else None
            ),
            "compression_bytes_per_second": compression_bytes_per_second,
            "compression_mbytes_per_second": (
                compression_bytes_per_second / 1_000_000.0 if compression_bytes_per_second is not None else None
            ),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            order_map.get(str(row["source_name"]), order_map.get(str(row["species"]), 10**9)),
            str(row["source_name"]),
        )
    )
    return rows


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "mode",
        "species",
        "source_name",
        "total_size",
        "sample_bytes",
        "sample_bases",
        "arithmetic_bits_per_base",
        "theoretical_bits_per_base",
        "vs_2bit_percent",
        "paper_baseline_compressed_bytes",
        "paper_baseline_geco2_mode",
        "paper_baseline_percent",
        "paper_baseline_bpb",
        "experiment_baseline_compressed_bytes",
        "experiment_baseline_geco2_mode",
        "experiment_baseline_percent",
        "experiment_baseline_bpb",
        "compression_bases_per_second",
        "compression_mbases_per_second",
        "compression_bytes_per_second",
        "compression_mbytes_per_second",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _plot_series(
    axis: Any,
    x_values: list[int],
    y_values: list[float],
    ylabel: str,
    title: str,
    color: str,
    *,
    label: str,
) -> None:
    finite_values = [value for value in y_values if not math.isnan(value)]
    if finite_values:
        axis.plot(x_values, y_values, marker="o", linewidth=1.8, markersize=4.5, color=color, label=label)
    else:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25, linewidth=0.6)


def _plot_baseline(axis: Any, x_values: list[int], y_values: list[float], *, label: str, color: str) -> bool:
    finite_values = [value for value in y_values if not math.isnan(value)]
    if not finite_values:
        return False
    axis.plot(
        x_values,
        y_values,
        linestyle="--",
        linewidth=1.6,
        marker="s",
        markersize=4.0,
        color=color,
        alpha=0.9,
        label=label,
    )
    return True


def _write_plot_png(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    run_name: str,
    split_name: str,
    mode_name: str,
) -> None:
    plt = _load_matplotlib_pyplot()

    labels = [str(row["source_name"]) for row in rows]
    x_values = list(range(len(rows)))
    arithmetic_bpb = [
        float(value) if isinstance(value := row.get("arithmetic_bits_per_base"), (int, float)) else float("nan")
        for row in rows
    ]
    vs_2bit_percent = [
        float(value) if isinstance(value := row.get("vs_2bit_percent"), (int, float)) else float("nan")
        for row in rows
    ]
    paper_baseline_percent = [
        float(value) if isinstance(value := row.get("paper_baseline_percent"), (int, float)) else float("nan")
        for row in rows
    ]
    paper_baseline_bpb = [
        float(value) if isinstance(value := row.get("paper_baseline_bpb"), (int, float)) else float("nan")
        for row in rows
    ]
    experiment_baseline_percent = [
        float(value) if isinstance(value := row.get("experiment_baseline_percent"), (int, float)) else float("nan")
        for row in rows
    ]
    experiment_baseline_bpb = [
        float(value) if isinstance(value := row.get("experiment_baseline_bpb"), (int, float)) else float("nan")
        for row in rows
    ]
    speed_mbases = [
        float(value) if isinstance(value := row.get("compression_mbases_per_second"), (int, float)) else float("nan")
        for row in rows
    ]

    figure_width = max(12.0, min(28.0, len(rows) * 0.8))
    figure, axes = plt.subplots(3, 1, figsize=(figure_width, 13.0), sharex=True)
    series_label = "GeCo2" if mode_name == GECO2_EXPERIMENT_MODE_NAME else "Model"
    _plot_series(
        axes[0],
        x_values,
        arithmetic_bpb,
        ylabel="Arithmetic BPB",
        title=f"{run_name} | {split_name} | {mode_name} | Compression Ratio (BPB)",
        color="#1f77b4",
        label=series_label,
    )
    arithmetic_has_baseline = _plot_baseline(
        axes[0],
        x_values,
        paper_baseline_bpb,
        label="GeCo2 paper total",
        color="#d62728",
    )
    arithmetic_has_experiment_baseline = _plot_baseline(
        axes[0],
        x_values,
        experiment_baseline_bpb,
        label="GeCo2 experiment split",
        color="#9467bd",
    )
    _plot_series(
        axes[1],
        x_values,
        vs_2bit_percent,
        ylabel="% of 2-bit",
        title="Compression Ratio Relative to 2-bit Encoding",
        color="#ff7f0e",
        label=series_label,
    )
    percent_has_baseline = _plot_baseline(
        axes[1],
        x_values,
        paper_baseline_percent,
        label="GeCo2 paper total",
        color="#d62728",
    )
    percent_has_experiment_baseline = _plot_baseline(
        axes[1],
        x_values,
        experiment_baseline_percent,
        label="GeCo2 experiment split",
        color="#9467bd",
    )
    _plot_series(
        axes[2],
        x_values,
        speed_mbases,
        ylabel="Speed (Mbases/s)",
        title="Compression Speed",
        color="#2ca02c",
        label=series_label,
    )
    if arithmetic_has_baseline or arithmetic_has_experiment_baseline:
        axes[0].legend(loc="best")
    if percent_has_baseline or percent_has_experiment_baseline:
        axes[1].legend(loc="best")

    axes[2].set_xlabel("DNA Source")
    axes[2].set_xticks(x_values)
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=9)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_baseline_tables(
    output_dir: Path,
    compression_compare: dict[str, Any],
    experiment_baseline_compare: dict[str, Any] | None = None,
) -> list[Path]:
    paper_rows = _paper_baseline_rows(compression_compare)
    experiment_rows = (
        _experiment_baseline_rows(experiment_baseline_compare)
        if experiment_baseline_compare is not None
        else _experiment_baseline_rows(compression_compare)
    )
    baseline_dir = output_dir / "baselines"
    paper_path = baseline_dir / "paper_baseline.csv"
    experiment_path = baseline_dir / "geco2_experiment_baseline.csv"
    _write_baseline_csv(paper_path, paper_rows)
    _write_baseline_csv(experiment_path, experiment_rows)
    return [paper_path, experiment_path]


def _write_baseline_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_artifacts_for_compression_compare(
    compression_compare_path: Path,
    *,
    out_dir_name: str = "compression_curves",
    baseline_compression_compare_path: Path | None = None,
    include_geco2_modes: bool = False,
) -> list[Path]:
    compression_compare = _read_json(compression_compare_path)
    experiment_baseline_compare = (
        _read_json(baseline_compression_compare_path) if baseline_compression_compare_path is not None else None
    )
    stats_dir = compression_compare_path.parent
    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return []

    run_name = _resolve_run_name(stats_dir, compression_compare)
    output_dir = stats_dir / out_dir_name
    generated_paths: list[Path] = _write_baseline_tables(
        output_dir,
        compression_compare,
        experiment_baseline_compare=experiment_baseline_compare,
    )

    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        for mode_name in split_payload.keys():
            if str(mode_name) == GECO2_EXPERIMENT_MODE_NAME and not include_geco2_modes:
                continue
            rows = _build_split_mode_rows(
                compression_compare=compression_compare,
                experiment_baseline_compare=experiment_baseline_compare,
                split_name=str(split_name),
                mode_name=str(mode_name),
            )
            if not rows:
                continue

            artifact_stem = _artifact_stem(str(split_name), str(mode_name))
            csv_path = output_dir / f"{artifact_stem}_compression_curve_data.csv"
            png_path = output_dir / f"{artifact_stem}_compression_curves.png"
            _write_rows_csv(csv_path, rows)
            _write_plot_png(
                path=png_path,
                rows=rows,
                run_name=run_name,
                split_name=str(split_name),
                mode_name=str(mode_name),
            )
            generated_paths.extend([csv_path, png_path])

    return generated_paths


def generate_curves_for_root(
    root_dir: Path,
    *,
    out_dir_name: str = "compression_curves",
    baseline_compression_compare_path: Path | None = None,
    include_geco2_modes: bool = False,
) -> list[Path]:
    generated_paths: list[Path] = []
    for compression_compare_path in sorted(root_dir.rglob("compression_compare.json")):
        generated_paths.extend(
            generate_artifacts_for_compression_compare(
                compression_compare_path,
                out_dir_name=out_dir_name,
                baseline_compression_compare_path=baseline_compression_compare_path,
                include_geco2_modes=include_geco2_modes,
            )
        )
    return generated_paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recursively generate per-source compression comparison plots from compression_compare.json files."
    )
    parser.add_argument("--root-dir", required=True, help="Root output directory to scan recursively.")
    parser.add_argument(
        "--out-dir-name",
        default="compression_curves",
        help="Artifact subdirectory name created inside each statistics directory.",
    )
    parser.add_argument(
        "--baseline-compression-json",
        help="Optional compression_compare.json containing reusable GeCo2 baseline results to overlay.",
    )
    parser.add_argument(
        "--geco2-baseline",
        help=(
            "Reusable GeCo2 baseline selector to overlay. Known aliases: "
            + ", ".join(sorted(GECO2_BASELINE_ALIASES))
            + ". A dna_geco2_* directory name, directory path, or JSON path also works. "
            "--baseline-compression-json takes precedence."
        ),
    )
    parser.add_argument(
        "--include-geco2-modes",
        action="store_true",
        help="Also generate plots for geco2_paper_modes entries found in the input JSON.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

    generated_paths = generate_curves_for_root(
        root_dir,
        out_dir_name=args.out_dir_name,
        baseline_compression_compare_path=(
            Path(args.baseline_compression_json)
            if args.baseline_compression_json
            else resolve_geco2_baseline_path(args.geco2_baseline)
        ),
        include_geco2_modes=args.include_geco2_modes,
    )
    if not generated_paths:
        print(f"[done] no compression_compare.json files found under {root_dir}")
        return

    print(f"[done] generated {len(generated_paths)} artifacts under {root_dir}")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
