from __future__ import annotations

"""Summarize and plot the OrSa aligned context-length probe.

The script only reads frozen summary/manifest files.  It writes recomputable CSV,
JSON, PNG, and PDF artifacts; it never rewrites source probability traces.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "outputs/dnacorpus_context_length_probe_orsa_v2"
MODEL_SPECS = (
    ("carbon", "carbon_stap", "Carbon-3B", "#4477AA"),
    ("evo2_optimized", "evo2_optimized_stap", "Evo2-7B optimized", "#CC6677"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the DNACorpus OrSa context-length probe.")
    parser.add_argument("--root-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_segment: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for model, fusion, label, _ in MODEL_SPECS:
            if model not in row["models"] or fusion not in row["fusions"]:
                continue
            standalone = row["models"][model]["common_mask_bpb"]
            fused = row["fusions"][fusion]["common_mask_bpb"]
            item = {
                "source": row["source"],
                "segment": row["segment"],
                "segment_start": row["segment_start"],
                "window_bases": row["window_bases"],
                "model": model,
                "model_label": label,
                "standalone_bpb": standalone,
                "fused_bpb": fused,
                "fused_minus_standalone_bpb": fused - standalone,
                "improvement_bpb": standalone - fused,
                "common_mask_bases": row["models"][model]["common_mask_bases"],
            }
            per_segment.append(item)
            grouped[(model, row["window_bases"])].append(item)

    aggregate: list[dict[str, Any]] = []
    for (model, window), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        def stats(key: str) -> tuple[float, float]:
            xs = [float(value[key]) for value in values]
            return mean(xs), stdev(xs) if len(xs) > 1 else 0.0

        standalone_mean, standalone_sd = stats("standalone_bpb")
        fused_mean, fused_sd = stats("fused_bpb")
        gain_mean, gain_sd = stats("improvement_bpb")
        aggregate.append({
            "model": model,
            "model_label": values[0]["model_label"],
            "window_bases": window,
            "segment_count": len(values),
            "standalone_bpb_mean": standalone_mean,
            "standalone_bpb_sd": standalone_sd,
            "fused_bpb_mean": fused_mean,
            "fused_bpb_sd": fused_sd,
            "improvement_bpb_mean": gain_mean,
            "improvement_bpb_sd": gain_sd,
        })
    return per_segment, aggregate


def _efficiency(root: Path, summary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []
    for row in summary_rows:
        for model in ("stap", "carbon", "evo2", "evo2_optimized"):
            if model not in row["models"]:
                continue
            manifest = root / "traces_depth_major" / row["segment"] / f"w{row['window_bases']:06d}" / model / "manifest.json"
            if not manifest.exists():
                continue
            config = _load(manifest).get("producer_config", {})
            generation = config.get("trace_generation_seconds")
            model_seconds = config.get("model_seconds")
            raw.append({
                "segment": row["segment"], "window_bases": row["window_bases"], "model": model,
                "scored_bases": row["segment_bases"], "trace_generation_seconds": generation,
                "model_seconds": model_seconds,
                "trace_bases_per_second": row["segment_bases"] / generation if generation else None,
                "model_bases_per_second": row["segment_bases"] / model_seconds if model_seconds else None,
                "factorize_seconds": config.get("factorize_seconds"),
                "probability_transfer_seconds": config.get("probability_transfer_seconds"),
            })
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in raw:
        groups[(item["model"], item["window_bases"])].append(item)
    agg: list[dict[str, Any]] = []
    for (model, window), values in sorted(groups.items()):
        out: dict[str, Any] = {"model": model, "window_bases": window, "segment_count": len(values)}
        for key in ("trace_generation_seconds", "model_seconds", "trace_bases_per_second", "model_bases_per_second", "factorize_seconds", "probability_transfer_seconds"):
            xs = [float(v[key]) for v in values if v[key] is not None]
            out[f"{key}_mean"] = mean(xs) if xs else None
            out[f"{key}_sd"] = stdev(xs) if len(xs) > 1 else (0.0 if xs else None)
        agg.append(out)
    return raw, agg


def _plot(out: Path, aggregate: list[dict[str, Any]], per_segment: list[dict[str, Any]], efficiency: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    cjk_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    font_family = "DejaVu Sans"
    if cjk_path.exists():
        font_manager.fontManager.addfont(cjk_path)
        font_family = font_manager.FontProperties(fname=cjk_path).get_name()
    plt.rcParams.update({"font.family": font_family, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
    by_model = defaultdict(list)
    for row in aggregate:
        by_model[row["model"]].append(row)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model, _, label, color in MODEL_SPECS:
        rows = sorted(by_model[model], key=lambda x: x["window_bases"])
        x = [r["window_bases"] for r in rows]
        for key, suffix, linestyle, marker in (("standalone_bpb", "standalone", "-", "o"), ("fused_bpb", "+ STAP", "--", "s")):
            y = [r[f"{key}_mean"] for r in rows]; sd = [r[f"{key}_sd"] for r in rows]
            suffix_cn = "单独" if suffix == "standalone" else "+ STAP"
            axes[0].plot(x, y, linestyle, marker=marker, color=color, label=f"{label} {suffix_cn}")
            axes[0].fill_between(x, [a-b for a,b in zip(y,sd)], [a+b for a,b in zip(y,sd)], color=color, alpha=.10)
        gain = [r["improvement_bpb_mean"] for r in rows]; gain_sd = [r["improvement_bpb_sd"] for r in rows]
        axes[1].plot(x, gain, "-o", color=color, label=label)
        axes[1].fill_between(x, [a-b for a,b in zip(gain,gain_sd)], [a+b for a,b in zip(gain,gain_sd)], color=color, alpha=.12)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xticks([6144,12288,24576,49152,98304,196608]); ax.set_xticklabels(["6k","12k","25k","49k","98k","197k"]); ax.grid(alpha=.25)
        ax.set_xlabel("窗口长度（bp）"); ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("每碱基比特数（越低越好）"); axes[0].set_title("上下文扩展与 STAP 融合")
    axes[1].axhline(0, color="black", lw=.8); axes[1].set_ylabel("单独模型 − 融合（bpb）"); axes[1].set_title("STAP 的互补压缩收益")
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"context_length_main.{ext}", bbox_inches="tight")
    plt.close(fig)

    evo = [r for r in per_segment if r["model"] == "evo2_optimized"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for segment, color in zip(("segment_a","segment_b","segment_c"), ("#228833","#AA3377","#EE7733")):
        rows = sorted((r for r in evo if r["segment"] == segment), key=lambda x: x["window_bases"])
        ax.plot([r["window_bases"] for r in rows], [r["standalone_bpb"] for r in rows], "-o", color=color, label=segment.replace("segment_", "片段 ").upper())
    ax.set_xscale("log", base=2); ax.set_xticks([6144,12288,24576,49152,98304,196608]); ax.set_xticklabels(["6k","12k","25k","49k","98k","197k"])
    ax.set_xlabel("窗口长度（bp）"); ax.set_ylabel("Evo2 单独模型 bpb"); ax.set_title("三个片段均出现 196k 拐点"); ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"evo2_per_segment.{ext}", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"stap":"#228833", "carbon":"#4477AA", "evo2":"#999999", "evo2_optimized":"#CC6677"}
    labels = {"stap":"STAP", "carbon":"Carbon-3B", "evo2":"Evo2 参考后端", "evo2_optimized":"Evo2 优化后端"}
    for model in colors:
        rows = sorted((r for r in efficiency if r["model"] == model), key=lambda x: x["window_bases"])
        if rows:
            ax.plot([r["window_bases"] for r in rows], [r["trace_bases_per_second_mean"] for r in rows], "-o", color=colors[model], label=labels[model])
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([6144,12288,24576,49152,98304,196608]); ax.set_xticklabels(["6k","12k","25k","49k","98k","197k"])
    ax.set_xlabel("窗口长度（bp）"); ax.set_ylabel("Trace 生成吞吐量（bp/s，对数轴）"); ax.set_title("目标概率 trace 生成效率"); ax.grid(alpha=.25, which="both"); ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(out / f"efficiency_trace_throughput.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _args(); root = args.root_dir.resolve(); out = (args.output_dir or root / "analysis").resolve(); out.mkdir(parents=True, exist_ok=True)
    summary = _load(root / "summary.json")
    per_segment, aggregate = _aggregate(summary["rows"])
    efficiency_raw, efficiency = _efficiency(root, summary["rows"])
    _write_csv(out / "per_segment_metrics.csv", per_segment); _write_csv(out / "aggregate_metrics.csv", aggregate)
    _write_csv(out / "efficiency_per_segment.csv", efficiency_raw); _write_csv(out / "efficiency_aggregate.csv", efficiency)
    result = {"source_summary": str(root / "summary.json"), "metric": "common_mask_bpb", "uncertainty": "sample_sd_across_three_fixed_segments", "aggregate": aggregate, "efficiency": efficiency}
    (out / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _plot(out, aggregate, per_segment, efficiency)
    print(f"Wrote analysis artifacts to {out}")


if __name__ == "__main__":
    main()
