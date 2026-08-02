#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_compression_curves import GECO2_PAPER_BASELINE_BY_SOURCE  # noqa: E402

DEFAULT_RUN_DIR = "outputs/dna_megabyte_large_opengenome2_11"
DEFAULT_OUTPUT_DIR = "outputs/fused_lm_nc_prefix_dnacorpus_full_geco2_paper_opengenome2_11_async"


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _species_paths(dataset_dir: Path, requested: list[str] | None) -> list[Path]:
    paths = [path for path in sorted(dataset_dir.iterdir()) if path.is_file()]
    if requested:
        wanted = set(requested)
        paths = [path for path in paths if path.name in wanted]
    return paths


def _query_gpus() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[dict[str, int]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        rows.append({"index": int(parts[0]), "util": int(parts[1]), "memory": int(parts[2])})
    return rows


def _parse_device_index(device: str) -> int | None:
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return None


def _wait_for_device(args: argparse.Namespace) -> str:
    explicit_index = _parse_device_index(str(args.device))
    while True:
        rows = _query_gpus()
        candidates = rows
        if explicit_index is not None:
            candidates = [row for row in rows if row["index"] == explicit_index]
        for row in candidates:
            if row["util"] <= int(args.idle_gpu_util) and row["memory"] <= int(args.idle_gpu_memory_mib):
                return f"cuda:{row['index']}"
        print(
            {
                "event": "waiting_for_idle_gpu",
                "requested_device": args.device,
                "idle_gpu_util": int(args.idle_gpu_util),
                "idle_gpu_memory_mib": int(args.idle_gpu_memory_mib),
                "gpus": rows,
                "sleep_seconds": int(args.idle_poll_seconds),
            },
            flush=True,
        )
        time.sleep(float(args.idle_poll_seconds))


def _reference_maps(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    lm_map: dict[str, dict[str, Any]] = {}
    lm_root = run_dir / "statistics_dnacorpus_full/window_codec/full/windows_nonoverlap"
    for path in lm_root.glob("*_*.mbw.json"):
        data = _json(path)
        if not data:
            continue
        source = str(data.get("source_name") or path.name.split("_", 1)[0])
        lm_map[source] = data

    geco_map: dict[str, dict[str, Any]] = {}
    geco_path = REPO_ROOT / "outputs/dna_geco2_dnacorpus_fullsplit/compression_compare.json"
    geco = _json(geco_path) or {}
    per_source = (
        ((geco.get("results") or {}).get("full") or {}).get("geco2_paper_modes") or {}
    ).get("per_source") or []
    for row in per_source:
        if isinstance(row, dict):
            geco_map[str(row.get("source_name") or row.get("species"))] = row
    return lm_map, geco_map


def _resolve_geco2_level(source: str, preset: str, configured_level: int) -> int:
    if preset != "geco2_paper":
        return int(configured_level)
    baseline = GECO2_PAPER_BASELINE_BY_SOURCE.get(source)
    if baseline is not None:
        return int(baseline["mode"])
    return 5


def _summarize(output_dir: Path, species_paths: list[Path], run_dir: Path) -> None:
    lm_map, geco_map = _reference_maps(run_dir)
    rows: list[dict[str, Any]] = []
    for source_path in species_paths:
        source = source_path.name
        fused = _json(output_dir / "per_species" / f"{source}.json")
        if not fused:
            continue
        lm = lm_map.get(source, {})
        geco = geco_map.get(source, {})
        fused_bpb = fused.get("arithmetic_bits_per_base")
        lm_bpb = lm.get("arithmetic_bits_per_base") or lm.get("compressed_bpb_with_framing")
        nc_diag_bpb = fused.get("nc_prefix_only_theoretical_bits_per_base")
        geco_bpb = geco.get("arithmetic_bits_per_base") or geco.get("theoretical_bits_per_base")
        nc_prefix_metadata = fused.get("nc_prefix_metadata") or {}
        rows.append(
            {
                "source": source,
                "sample_bases": fused.get("sample_bases"),
                "window_count": fused.get("window_count"),
                "batch_count": fused.get("batch_count"),
                "batch_window_counts": " ".join(map(str, fused.get("batch_window_counts") or [])),
                "lm_batch_count": fused.get("lm_batch_count"),
                "lm_batch_size": fused.get("lm_batch_size"),
                "lm_batch_window_counts": " ".join(map(str, fused.get("lm_batch_window_counts") or [])),
                "nc_prefix_batch_scope": nc_prefix_metadata.get("batch_scope"),
                "nc_prefix_batch_size": nc_prefix_metadata.get("batch_size"),
                "nc_prefix_preset": fused.get("nc_prefix_preset"),
                "nc_prefix_geco2_level": fused.get("nc_prefix_geco2_level"),
                "fused_arithmetic_bpb": fused_bpb,
                "fused_core_theoretical_bpb": fused.get("core_theoretical_bits_per_base"),
                "fused_speed_bases_per_second": fused.get("compression_bases_per_second"),
                "fused_process_seconds": fused.get("compression_process_seconds"),
                "fused_model_seconds": fused.get("model_seconds"),
                "fused_native_seconds": fused.get("native_fused_encode_seconds"),
                "fused_final_lm_weight": fused.get("fusion_final_mean_lm_weight"),
                "lm_full_arithmetic_bpb": lm_bpb,
                "lm_full_speed_bases_per_second": lm.get("compression_bases_per_second"),
                "nc_prefix_diag_bpb": nc_diag_bpb,
                "geco2_full_bpb": geco_bpb,
                "geco2_full_speed_bases_per_second": geco.get("compression_bases_per_second"),
                "delta_vs_lm_bpb": (float(fused_bpb) - float(lm_bpb)) if fused_bpb is not None and lm_bpb is not None else None,
                "delta_vs_nc_diag_bpb": (float(fused_bpb) - float(nc_diag_bpb))
                if fused_bpb is not None and nc_diag_bpb is not None
                else None,
                "delta_vs_geco2_bpb": (float(fused_bpb) - float(geco_bpb))
                if fused_bpb is not None and geco_bpb is not None
                else None,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fused_dnacorpus_full_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    total_bases = sum(int(row["sample_bases"] or 0) for row in rows)
    aggregate = {
        "completed_sources": len(rows),
        "total_bases": total_bases,
        "weighted_fused_arithmetic_bpb": _weighted(rows, "fused_arithmetic_bpb"),
        "weighted_lm_full_arithmetic_bpb": _weighted(rows, "lm_full_arithmetic_bpb"),
        "weighted_nc_prefix_diag_bpb": _weighted(rows, "nc_prefix_diag_bpb"),
        "weighted_geco2_full_bpb": _weighted(rows, "geco2_full_bpb"),
        "rows": rows,
    }
    (output_dir / "fused_dnacorpus_full_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _weighted(rows: list[dict[str, Any]], key: str) -> float | None:
    numerator = 0.0
    denominator = 0
    for row in rows:
        value = row.get(key)
        bases = int(row.get("sample_bases") or 0)
        if value is None or bases <= 0:
            continue
        numerator += float(value) * bases
        denominator += bases
    return numerator / denominator if denominator else None


def _is_complete(path: Path) -> bool:
    data = _json(path)
    return bool(data and data.get("arithmetic_bits_per_base") is not None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fused LM x nc_prefix compression over DNACorpus full sources.")
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-tag", default="best")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--device", default="auto-idle", help="'auto-idle' chooses the first idle GPU; cuda:N waits for that GPU.")
    parser.add_argument("--idle-gpu-util", type=int, default=5)
    parser.add_argument("--idle-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--idle-poll-seconds", type=int, default=300)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--batch-size", default="8192")
    parser.add_argument(
        "--pipeline-mode",
        choices=("streaming_token_encode_overlap", "streaming_token_strict", "streaming_token_nc_full_batch"),
        default="streaming_token_encode_overlap",
    )
    parser.add_argument("--nc-prefix-window-bases", type=int, default=3072)
    parser.add_argument("--nc-prefix-min-windows", type=int, default=8192)
    parser.add_argument("--nc-prefix-hash-bucket-count", type=int, default=0)
    parser.add_argument("--nc-prefix-preset", choices=("level", "geco2_paper"), default="geco2_paper")
    parser.add_argument("--nc-prefix-geco2-level", type=int, default=10, help="Fixed GECO2 level with --nc-prefix-preset level. With --nc-prefix-preset geco2_paper, DNACorpus uses the paper mode map and falls back to level 5 for sources missing from that map.")
    parser.add_argument("--fusion-eta", type=float, default=0.05)
    parser.add_argument("--fusion-initial-lm-weight", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    per_species_dir = output_dir / "per_species"
    log_dir = output_dir / "logs"
    per_species_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    species_paths = _species_paths(Path(args.dataset_dir), args.source)
    (output_dir / "run_parameters.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    for source_path in species_paths:
        source = source_path.name
        output_json = per_species_dir / f"{source}.json"
        geco2_level = _resolve_geco2_level(source, str(args.nc_prefix_preset), int(args.nc_prefix_geco2_level))
        if args.skip_existing and _is_complete(output_json):
            print({"event": "skip_existing", "source": source, "output_json": str(output_json)}, flush=True)
            continue
        device = _wait_for_device(args)
        command = [
            sys.executable,
            "scripts/run_fused_lm_nc_prefix_compression.py",
            "--run-dir",
            str(args.run_dir),
            "--checkpoint-tag",
            str(args.checkpoint_tag),
            "--source-file",
            str(source_path),
            "--source-format",
            "raw",
            "--output-json",
            str(output_json),
            "--device",
            device,
            "--dtype",
            str(args.dtype),
            "--batch-size",
            str(args.batch_size),
            "--pipeline-mode",
            str(args.pipeline_mode),
            "--nc-prefix-window-bases",
            str(args.nc_prefix_window_bases),
            "--nc-prefix-min-windows",
            str(args.nc_prefix_min_windows),
            "--nc-prefix-hash-bucket-count",
            str(args.nc_prefix_hash_bucket_count),
            "--nc-prefix-preset",
            str(args.nc_prefix_preset),
            "--nc-prefix-geco2-level",
            str(geco2_level),
            "--fusion-eta",
            str(args.fusion_eta),
            "--fusion-initial-lm-weight",
            str(args.fusion_initial_lm_weight),
            "--skip-codec-baselines",
        ]
        log_path = log_dir / f"{source}.log"
        print({"event": "start_source", "source": source, "device": device, "log": str(log_path)}, flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(
                json.dumps(
                    {
                        "event": "command",
                        "command": command,
                        "device": device,
                        "nc_prefix_preset": str(args.nc_prefix_preset),
                        "nc_prefix_geco2_level": int(geco2_level),
                    }
                )
                + "\n"
            )
            log_handle.flush()
            completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        elapsed = time.perf_counter() - started
        print(
            {
                "event": "finish_source",
                "source": source,
                "returncode": completed.returncode,
                "seconds": elapsed,
                "output_json": str(output_json),
            },
            flush=True,
        )
        _summarize(output_dir, species_paths, run_dir)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)

    _summarize(output_dir, species_paths, run_dir)


if __name__ == "__main__":
    main()
