from __future__ import annotations

"""Full-source DNA BPB probe with compact reusable artifacts.

This script is the full-size companion to run_dna_region_bpb_probe.py. It writes
the same region_bpb.json + models/<model>/bpb.npz layout, but the region is the
entire filtered source and large plotting/statistics outputs are downsampled.

Examples:

    # Full DNACorpus, Megabyte + continuous GeCo2 per-base estimate.
    python scripts/run_dna_full_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
      --model megabyte:outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217:best \
      --model geco2 \
      --geco2-pseudo-window-bases 3072 \
      --device cuda:1 \
      --batch-size 32 \
      --chunk-bases 3145728 \
      --summary-window-bases 1000000 \
      --plot-max-points 200000 \
      --output-dir outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217/full_bpb_probe_dnacorpus

    # Full-source Evo2 artifact. Use CUDA_VISIBLE_DEVICES to avoid physical GPU 0;
    # process-local cuda:0 is then the selected visible card.
    # Evo2 7B can use the normal .venv. Evo2 1B requires Transformer Engine / FP8;
    # use scripts/run_evo2_1b_env_python.sh for the isolated .conda_evo2_1b env.
    CUDA_VISIBLE_DEVICES=3 python scripts/run_dna_full_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species BuEb \
      --model evo2:third_party/evo2_7b_base/evo2_7b_base.pt:evo2_7b_base \
      --evo2-context-bases 8192 \
      --device cuda:0 \
      --batch-size 1 \
      --output-dir outputs/evo2_7b_base_full_bpb_probe_dnacorpus

    CUDA_VISIBLE_DEVICES=3 scripts/run_evo2_1b_env_python.sh scripts/run_dna_full_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species BuEb \
      --model evo2:third_party/evo2_1b_base:evo2_1b_base \
      --evo2-context-bases 3072 \
      --device cuda:0 \
      --batch-size 32 \
      --chunk-bases 3145728 \
      --summary-window-bases 1000000 \
      --plot-max-points 200000 \
      --output-dir outputs/evo2_1b_base_full_bpb_probe_dnacorpus

    # Slice a region later without recomputing.
    python scripts/run_dna_region_bpb_probe.py \
      --from-full-result-dir outputs/.../full_bpb_probe_dnacorpus \
      --region-start 1000000 \
      --region-bases 50000 \
      --output-dir outputs/.../region_from_full
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dna_region_bpb_probe import (
    DEFAULT_EVO2_LOCAL_PATH,
    Geco2RegionAdapter,
    _compact_metadata,
    _geco2_pseudo_window_bases,
    _record_dtype,
    _round_float,
    _safe_label,
    _tail_text,
    _write_compact_json,
    bpb_for_adapter,
    build_region_adapters,
    extract_filtered_region,
    filtered_length,
    normalize_alphabet,
    plot_curves,
    region_identity,
    resolve_region_sources,
    stable_random_start,
    write_csv,
    write_model_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dnacorpus", "opengenome2"), required=True)
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--input-dir")
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--species", nargs="+")
    parser.add_argument("--sequence-source-mode", choices=("auto", "flat_file", "fasta_dir"), default="auto")
    parser.add_argument("--multi-sequence-mode", choices=("separate", "concat"), default="separate")
    parser.add_argument("--sequence-include", action="append")
    parser.add_argument("--alphabet", default="ACGTN")

    parser.add_argument("--model", action="append", help="Repeatable model spec: kind:run_dir[:checkpoint_or_tag].")
    parser.add_argument("--model-kind", choices=("megabyte", "megadna", "dnagpt", "geco2", "evo2"), default="megabyte")
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-tag", default="best")
    parser.add_argument("--base-weight")
    parser.add_argument("--seq-length", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--geco2-bin", default="GeCo2")
    parser.add_argument("--geco2-level", type=int, default=5)
    parser.add_argument("--geco2-profile-mode", choices=("estimate",), default="estimate")
    parser.add_argument("--geco2-pseudo-window-bases", type=int, default=3072)
    parser.add_argument("--no-geco2-dnacorpus-paper-levels", dest="geco2_dnacorpus_paper_levels", action="store_false")
    parser.add_argument("--geco2-temp-root")
    parser.add_argument("--geco2-keep-temp", action="store_true")
    parser.set_defaults(geco2_dnacorpus_paper_levels=True)
    parser.add_argument(
        "--evo2-local-path",
        default=str(DEFAULT_EVO2_LOCAL_PATH),
        help="Local Evo2 .pt checkpoint path or model directory. For 1B use --model evo2:third_party/evo2_1b_base:evo2_1b_base under scripts/run_evo2_1b_env_python.sh.",
    )
    parser.add_argument("--evo2-model-name", default="evo2_7b_base", help="Evo2 model name, e.g. evo2_7b_base or evo2_1b_base.")
    parser.add_argument("--evo2-context-bases", type=int, default=8192)
    parser.add_argument("--evo2-use-kernels", action="store_true", help="Enable optional Vortex Triton kernels for Evo2.")

    parser.add_argument("--chunk-bases", type=int, default=3_145_728, help="Model scoring chunk size in bases, aligned down to model windows.")
    parser.add_argument("--summary-window-bases", type=int, default=1_000_000, help="Downsampled CSV/statistical window size.")
    parser.add_argument("--plot-max-points", type=int, default=200_000)
    parser.add_argument("--smooth-window-bases", type=int, default=50_000)
    parser.add_argument("--model-window-smooth-bases", type=int, default=64)
    parser.add_argument("--record-dtype", choices=("float16", "float32", "float64"), default="float16")
    parser.add_argument("--statistics-sample-size", type=int, default=1_000_000)
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _aligned_chunk_bases(requested: int, model_window_bases: int, token_size: int) -> int:
    unit = max(1, int(model_window_bases), int(token_size))
    value = max(unit, int(requested))
    aligned = (value // unit) * unit
    return max(unit, aligned)


def _window_rows_contiguous(values: np.ndarray, *, source_start: int, window_bases: int) -> list[dict[str, Any]]:
    if values.size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for start in range(0, int(values.size), int(window_bases)):
        end = min(start + int(window_bases), int(values.size))
        window = values[start:end]
        rows.append(
            {
                "window_index": len(rows),
                "region_start": int(start),
                "region_end_exclusive": int(end),
                "source_start": int(source_start + start),
                "source_end_exclusive": int(source_start + end),
                "base_count": int(window.size),
                "sum_bits": float(np.sum(window, dtype=np.float64)),
                "mean_bpb": float(np.mean(window, dtype=np.float64)),
            }
        )
    return rows


def _model_window_average_contiguous(
    chunks: list[np.ndarray],
    *,
    chunk_starts: list[int],
    model_window_bases: int,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((int(model_window_bases),), dtype=np.float64)
    counts = np.zeros((int(model_window_bases),), dtype=np.int64)
    for chunk, start in zip(chunks, chunk_starts):
        if chunk.size == 0:
            continue
        positions = (np.arange(int(start), int(start) + int(chunk.size), dtype=np.int64) % int(model_window_bases))
        sums += np.bincount(positions, weights=chunk.astype(np.float64, copy=False), minlength=int(model_window_bases))
        counts += np.bincount(positions, minlength=int(model_window_bases)).astype(np.int64, copy=False)
    means = np.full((int(model_window_bases),), np.nan, dtype=np.float64)
    active = counts > 0
    means[active] = sums[active] / counts[active]
    return means, counts


def _downsample_contiguous(values: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or values.size <= max_points:
        return np.arange(int(values.size), dtype=np.int64), values.astype(np.float64, copy=False)
    bins = np.linspace(0, int(values.size), int(max_points) + 1, dtype=np.int64)
    keep = bins[1:] > bins[:-1]
    starts = bins[:-1][keep]
    ends = bins[1:][keep]
    y = np.empty((starts.shape[0],), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        y[index] = float(np.mean(values[start:end], dtype=np.float64))
    return starts.astype(np.int64, copy=False), y


def _score_model_full(
    adapter: Any,
    *,
    sequence: str,
    source_info: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    model_window_bases = max(1, int(getattr(getattr(adapter, "config", None), "model", None).seq_length) * int(getattr(adapter, "token_size", 1))) if hasattr(getattr(adapter, "config", None), "model") else int(args.summary_window_bases)
    token_size = int(getattr(adapter, "token_size", 1))
    chunk_bases = _aligned_chunk_bases(int(args.chunk_bases), model_window_bases, token_size)
    chunks: list[np.ndarray] = []
    chunk_starts: list[int] = []
    timing: dict[str, float] = {
        "model_forward_seconds": 0.0,
        "softmax_seconds": 0.0,
        "factorization_seconds": 0.0,
        "data_transfer_seconds": 0.0,
    }
    total_started = perf_counter()
    for start in range(0, len(sequence), chunk_bases):
        end = min(start + chunk_bases, len(sequence))
        chunk_sequence = sequence[start:end]
        region_offsets = np.arange(start, end, dtype=np.int64)
        bpb, offsets, metadata = bpb_for_adapter(
            adapter,
            species=source_info.get("species") or source_info.get("source"),
            region_sequence=chunk_sequence,
            region_offsets=region_offsets,
            batch_size=int(args.batch_size),
        )
        for key in timing:
            timing[key] += float(metadata.get(key, 0.0))
        chunks.append(np.asarray(bpb, dtype=_record_dtype(args.record_dtype)))
        chunk_starts.append(int(offsets[0]) if offsets.size else start)
        model_window_bases = int(metadata.get("model_window_bases", model_window_bases))
    if not chunks:
        raise ValueError(f"{adapter.name} produced no BPB values.")
    bpb_all = np.concatenate(chunks)
    means, counts = _model_window_average_contiguous(chunks, chunk_starts=chunk_starts, model_window_bases=model_window_bases)
    rows = _window_rows_contiguous(bpb_all.astype(np.float64, copy=False), source_start=0, window_bases=int(args.summary_window_bases))
    metadata = {
        "adapter_name": adapter.name,
        "adapter_class": type(adapter).__name__,
        "alphabet": getattr(adapter, "alphabet", source_info.get("alphabet")),
        "token_size": token_size,
        "valid_base_count": int(bpb_all.size),
        "filtered_out_bases": int(len(sequence) - bpb_all.size),
        "trimmed_tail_bases": int(len(sequence) - bpb_all.size),
        "model_window_bases": int(model_window_bases),
        **timing,
        "chunk_bases": int(chunk_bases),
        "full_source_wall_seconds": perf_counter() - total_started,
    }
    return bpb_all, metadata, rows, means, counts, float(perf_counter() - total_started)


def _score_geco2_full(
    adapter: Geco2RegionAdapter,
    *,
    sequence: str,
    source_info: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    if not sequence:
        raise ValueError("GeCo2 requires at least one base.")
    started = perf_counter()
    level, level_source = adapter._level_for_source(
        dataset=str(source_info.get("dataset")) if source_info.get("dataset") is not None else None,
        species=str(source_info.get("species")) if source_info.get("species") is not None else None,
        source=str(source_info.get("source")) if source_info.get("source") is not None else None,
    )
    cleanup_context = None
    if adapter.keep_temp:
        temp_dir = Path(tempfile.mkdtemp(prefix="geco2_full_bpb_", dir=str(adapter.temp_root) if adapter.temp_root else None))
    else:
        cleanup_context = tempfile.TemporaryDirectory(prefix="geco2_full_bpb_", dir=str(adapter.temp_root) if adapter.temp_root else None)
        temp_dir = Path(cleanup_context.__enter__())
    try:
        input_path = temp_dir / f"{_safe_label(str(source_info.get('source', 'source')))}.seq"
        input_path.write_text(sequence, encoding="ascii")
        command = adapter._command(input_path, level=level, estimate=True)
        geco_started = perf_counter()
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        geco_seconds = perf_counter() - geco_started
        if completed.returncode != 0:
            raise RuntimeError(f"GeCo2 failed with return code {completed.returncode}: {_tail_text(completed.stderr or completed.stdout)}")
        estimate_path = Path(str(input_path) + ".iae")
        compressed_path = Path(str(input_path) + ".co")
        if not estimate_path.exists():
            raise FileNotFoundError(f"GeCo2 did not create expected estimate file: {estimate_path}")
        bpb = np.fromiter((float(line.strip() or "0") for line in estimate_path.open("r", encoding="utf-8")), dtype=_record_dtype(args.record_dtype))
        if bpb.shape[0] != len(sequence):
            raise RuntimeError(f"GeCo2 .iae row count {bpb.shape[0]} does not match source bases {len(sequence)}")
        model_window_bases = _geco2_pseudo_window_bases(args, len(sequence))
        chunks = [bpb]
        means, counts = _model_window_average_contiguous(chunks, chunk_starts=[0], model_window_bases=model_window_bases)
        rows = _window_rows_contiguous(bpb.astype(np.float64, copy=False), source_start=0, window_bases=int(args.summary_window_bases))
        compressed_bytes = int(compressed_path.stat().st_size) if compressed_path.exists() else 0
        metadata = {
            "adapter_name": adapter.name,
            "adapter_class": type(adapter).__name__,
            "alphabet": adapter.alphabet,
            "token_size": 1,
            "valid_base_count": int(bpb.size),
            "filtered_out_bases": 0,
            "trimmed_tail_bases": 0,
            "model_window_bases": int(model_window_bases),
            "model_forward_seconds": 0.0,
            "softmax_seconds": 0.0,
            "factorization_seconds": 0.0,
            "data_transfer_seconds": 0.0,
            "geco2_binary": adapter.binary,
            "geco2_level": int(level),
            "geco2_level_source": level_source,
            "geco2_profile_mode": "estimate",
            "geco2_pseudo_window_bases": int(model_window_bases),
            "continuous_compressed_bytes": compressed_bytes,
            "continuous_bits_per_base": compressed_bytes * 8.0 / max(len(sequence), 1),
            "geco2_compression_seconds": geco_seconds,
            "geco2_wall_seconds": perf_counter() - started,
            "geco2_note": "Full-source GeCo2 was run once with -e; per-base BPB comes from the .iae file.",
        }
        return bpb, metadata, rows, means, counts, float(perf_counter() - started)
    finally:
        if cleanup_context is not None:
            cleanup_context.__exit__(None, None, None)


def run_full_source(args: argparse.Namespace, *, source_info: dict[str, Any], adapters: list[Any], output_dir: Path) -> dict[str, Any]:
    total_started = perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    length_started = perf_counter()
    source_length = filtered_length(source_info["paths"], alphabet=str(source_info["alphabet"]), fasta=bool(source_info["fasta"]))
    length_seconds = perf_counter() - length_started
    read_started = perf_counter()
    sequence = extract_filtered_region(
        source_info["paths"],
        alphabet=str(source_info["alphabet"]),
        fasta=bool(source_info["fasta"]),
        start=0,
        length=source_length,
    ).decode("ascii")
    read_seconds = perf_counter() - read_started

    model_summaries: dict[str, Any] = {}
    plot_payload: dict[str, dict[str, Any]] = {}
    window_rows_all: list[dict[str, Any]] = []
    model_window_rows: list[dict[str, Any]] = []
    for adapter in adapters:
        if isinstance(adapter, Geco2RegionAdapter):
            bpb, metadata, rows, means, counts, wall_seconds = _score_geco2_full(adapter, sequence=sequence, source_info=source_info, args=args)
        else:
            bpb, metadata, rows, means, counts, wall_seconds = _score_model_full(adapter, sequence=sequence, source_info=source_info, args=args)
        for row in rows:
            row["model"] = adapter.name
            row["plot_window_bases"] = int(args.summary_window_bases)
            window_rows_all.append(row)
        for position, (mean_value, count) in enumerate(zip(means, counts)):
            if int(count) > 0:
                model_window_rows.append({"model": adapter.name, "model_window_position": position, "base_count": int(count), "mean_bpb": float(mean_value)})
        total_bits = float(np.sum(bpb, dtype=np.float64))
        worst_row = max(rows, key=lambda item: float(item["mean_bpb"])) if rows else None
        model_summaries[adapter.name] = write_model_artifact(
            output_dir,
            model_name=adapter.name,
            bpb=bpb,
            offsets=np.zeros((0,), dtype=np.int64),
            metadata=metadata,
            window_summary=rows,
            worst_window=worst_row,
            model_window_means=means,
            model_window_counts=counts,
            total_bits=total_bits,
            total_bpb=total_bits / max(int(bpb.size), 1),
            adapter_wall_seconds=wall_seconds,
            record_dtype=str(args.record_dtype),
            store_offsets=False,
            offset_start=0,
            statistics_sample_size=int(args.statistics_sample_size),
        )
        plot_offsets, plot_bpb = _downsample_contiguous(bpb, int(args.plot_max_points))
        plot_payload[adapter.name] = {"bpb": plot_bpb, "offsets": plot_offsets, "model_window_means": means, "model_window_counts": counts}

    windows_csv = output_dir / "full_bpb_windows_downsampled.csv"
    model_window_csv = output_dir / "region_bpb_model_window_average.csv"
    curve_png = output_dir / "region_bpb_curve.png"
    json_path = output_dir / "region_bpb.json"
    write_csv(windows_csv, window_rows_all)
    write_csv(model_window_csv, model_window_rows)
    if not args.compute_only:
        plot_scale = max(1.0, source_length / max(1, int(args.plot_max_points)))
        plot_smooth_points = max(1, int(round(int(args.smooth_window_bases) / plot_scale)))
        plot_curves(
            curve_png,
            region_base_count=source_length,
            source_start=0,
            per_model=plot_payload,
            smooth_window_bases=plot_smooth_points,
            model_window_smooth_bases=int(args.model_window_smooth_bases),
            window_boundary_bases=None,
            max_points_per_model=int(args.plot_max_points),
        )
    source_paths = [str(path) for path in source_info["paths"]]
    result = {
        "dataset": source_info["dataset"],
        "source": source_info["source"],
        "species": source_info.get("species"),
        "source_path_count": len(source_paths),
        "source_paths_preview": source_paths[:4],
        "source_is_fasta": bool(source_info["fasta"]),
        "alphabet": source_info["alphabet"],
        "filtered_source_bases": int(source_length),
        "filtered_source_bases_known": True,
        "filtered_source_bases_lower_bound": int(source_length),
        "region_start": 0,
        "requested_region_bases": int(source_length),
        "region_bases": int(source_length),
        "region_identity": region_identity(
            dataset=source_info["dataset"],
            source=source_info["source"],
            species=source_info.get("species"),
            alphabet=source_info["alphabet"],
            region_start=0,
            region_bases=int(source_length),
        ),
        "random_region": False,
        "seed": 0,
        "model_count": len(model_summaries),
        "models": model_summaries,
        "full_source_bpb": True,
        "timing": {
            "source_length_seconds": _round_float(length_seconds),
            "region_read_seconds": _round_float(read_seconds),
            "total_wall_seconds": _round_float(perf_counter() - total_started),
        },
        "outputs": {
            "json": str(json_path),
            "model_artifact_dir": str(output_dir / "models"),
            "per_base_csv": None,
            "windows_csv": str(windows_csv),
            "model_window_average_csv": str(model_window_csv),
            "curve_png": str(curve_png) if not args.compute_only else None,
            "record_dtype": str(args.record_dtype),
            "offsets_stored": False,
            "summary_window_bases": int(args.summary_window_bases),
            "plot_max_points": int(args.plot_max_points),
        },
    }
    _write_compact_json(json_path, result)
    return result


def main() -> None:
    args = parse_args()
    alphabet = normalize_alphabet(args.alphabet)
    source_infos = resolve_region_sources(args, alphabet)
    adapters = build_region_adapters(args)
    output_root = Path(args.output_dir)
    multiple_sources = len(source_infos) > 1
    summaries: list[dict[str, Any]] = []
    for source_info in source_infos:
        source_label = _safe_label(str(source_info["source"]))
        output_dir = output_root / source_label if multiple_sources else output_root
        result = run_full_source(args, source_info=source_info, adapters=adapters, output_dir=output_dir)
        summaries.append(
            {
                "source": result["source"],
                "species": result.get("species"),
                "region_signature": result.get("region_identity", {}).get("signature"),
                "region_bpb_json": result["outputs"]["json"],
                "curve_png": result["outputs"]["curve_png"],
                "region_bases": result["region_bases"],
                "model_names": sorted(result["models"]),
            }
        )
    if multiple_sources:
        summary_path = output_root / "full_bpb_batch_summary.json"
        _write_compact_json(summary_path, {"source_count": len(summaries), "results": summaries})
    print(json.dumps({"output_dir": str(output_root), "source_count": len(summaries), "results": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
