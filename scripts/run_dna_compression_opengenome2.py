from __future__ import annotations

"""Compress per-source FASTA sequence content with a trained Megabyte checkpoint.

The FASTA reader skips record headers and formatting bytes. Only sequence
symbols from the configured alphabet are passed to the model.

Example:
    python scripts/run_dna_compression_opengenome2.py \
      --run-dir outputs/dna_megabyte_large_opengenome2_4 \
      --input-dir /data/students/Liang_junnan/opengenome2_subset/fasta_test_subset_100mb_per_source \
      --device cuda:0 \
      --eval-batch-size 128 \
      --geco2-baseline opengenome2_100mb
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress import load_experiment_config
from dna_compress.compression_eval import (
    MEGABYTE_ARITHMETIC_CODING_MODES,
    NON_OVERLAP_MODE,
    compress_source,
    resolve_device,
    summarize_per_source,
)
from dna_compress.fast_arithmetic import ARITHMETIC_BACKENDS
from dna_compress.fixed_token_factorization import build_fixed_token_arithmetic_factorizer
from dna_compress.megabyte_loader import build_model
from dna_compress.tokenization import apply_token_merge_to_model_config, normalize_alphabet
from scripts.export_statistics import write_compression_report_tables
from scripts.plot_compression_curves import generate_artifacts_for_compression_compare, resolve_geco2_baseline_path


DEFAULT_RUN_DIR = Path("outputs/dna_megabyte_large_opengenome2_4")
DEFAULT_INPUT_DIR = Path("/data/students/Liang_junnan/opengenome2_subset/fasta_test_subset_100mb_per_source")
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "statistics_opengenome2_fasta_100mb"


def _safe_source_filename(source: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in source)
    return f"{safe}.fasta"


def _read_manifest(input_dir: Path) -> dict[str, Any] | None:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _source_files(input_dir: Path, manifest: dict[str, Any] | None) -> list[tuple[str, Path, dict[str, Any]]]:
    if manifest is not None and isinstance(manifest.get("sources"), dict):
        files: list[tuple[str, Path, dict[str, Any]]] = []
        for source, payload in manifest["sources"].items():
            entry = payload if isinstance(payload, dict) else {}
            output_path = entry.get("output_path")
            path = Path(output_path) if isinstance(output_path, str) else input_dir / _safe_source_filename(str(source))
            if not path.exists():
                path = input_dir / _safe_source_filename(str(source))
            files.append((str(source), path, dict(entry)))
        return files
    return [(path.stem, path, {}) for path in sorted(input_dir.glob("*.fasta"))]


def _load_fasta_sequence(path: Path, *, alphabet: str) -> tuple[bytes, dict[str, int]]:
    allowed = {ord(base) for base in normalize_alphabet(alphabet)}
    allowed.update(ord(base.lower()) for base in normalize_alphabet(alphabet))
    delete = bytes(byte for byte in range(256) if byte not in allowed)
    payload = bytearray()
    header_bytes = 0
    newline_bytes = 0
    other_sequence_bytes = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.endswith(b"\n"):
                newline_bytes += 1
            if line.endswith(b"\r\n"):
                newline_bytes += 1
            content = line.rstrip(b"\r\n")
            if content.startswith(b">"):
                header_bytes += len(content)
                continue
            cleaned = content.upper().translate(None, delete)
            other_sequence_bytes += len(content) - len(cleaned)
            payload.extend(cleaned)
    return bytes(payload), {
        "fasta_file_bytes": path.stat().st_size,
        "fasta_sequence_bytes": len(payload),
        "fasta_header_bytes": header_bytes,
        "fasta_newline_bytes": newline_bytes,
        "fasta_other_sequence_bytes": other_sequence_bytes,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance((value := row.get(key)), (dict, list, tuple))
                    else value
                    for key in fieldnames
                }
            )


def _load_model(config, checkpoint_path: Path, device: torch.device):
    model = build_model(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint '{checkpoint_path}' is missing 'model_state'")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _checkpoint_path(run_dir: Path, checkpoint_tag: str) -> Path:
    path = run_dir / f"{checkpoint_tag}.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compress FASTA sequence content with a Megabyte checkpoint.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint-tag", choices=["best", "last"], default="best")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override config train.eval_batch_size. Leave unset to use the checkpoint config.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--arithmetic-coding-mode",
        choices=list(MEGABYTE_ARITHMETIC_CODING_MODES),
        default="model_symbol",
    )
    parser.add_argument("--arithmetic-merge-size", type=int, default=3)
    parser.add_argument("--arithmetic-backend", choices=list(ARITHMETIC_BACKENDS), default="auto")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--mode-name", default=NON_OVERLAP_MODE)
    parser.add_argument("--compression-sample-bytes", type=int, default=0)
    parser.add_argument(
        "--sequence-alphabet",
        default=None,
        help="Sequence symbols to keep from FASTA lines. Defaults to config data.token_merge_alphabet.",
    )
    parser.add_argument(
        "--baseline-compression-json",
        default=None,
        help=(
            "Optional explicit GeCo2 compression_compare.json to overlay in generated plots. "
            "Takes precedence over --geco2-baseline."
        ),
    )
    parser.add_argument(
        "--geco2-baseline",
        default="opengenome2_100mb",
        help=(
            "Reusable GeCo2 baseline selector for plot overlays. Use opengenome2_100mb, none, "
            "a dna_geco2_* directory name, a directory path, or a JSON path."
        ),
    )
    parser.add_argument(
        "--compression-curves-out-dir-name",
        default="compression_curves",
        help="Artifact subdirectory name for generated compression curve CSV/PNG files.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = Path(args.run_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    config = load_experiment_config(run_dir / "resolved_config.json")
    config.train.device = args.device
    if args.eval_batch_size is not None:
        config.train.eval_batch_size = args.eval_batch_size
    if args.dtype is not None:
        config.train.dtype = args.dtype
    config.arithmetic.coding_mode = args.arithmetic_coding_mode
    config.arithmetic.merge_size = args.arithmetic_merge_size
    config.arithmetic.backend = args.arithmetic_backend
    apply_token_merge_to_model_config(config.model, config.data)

    alphabet = args.sequence_alphabet or config.data.token_merge_alphabet
    device = resolve_device(config.train.device)
    checkpoint_path = _checkpoint_path(run_dir, args.checkpoint_tag)
    model, checkpoint = _load_model(config, checkpoint_path, device)
    factorizer = None
    if config.arithmetic.coding_mode == "base_prefix_exact_gpu_cpu":
        factorizer = build_fixed_token_arithmetic_factorizer(
            vocab_size=config.model.vocab_size,
            special_token_ids=[config.model.pad_id, config.model.eos_id],
            model_merge_size=config.data.token_merge_size,
            arithmetic_merge_size=config.arithmetic.merge_size,
            alphabet=normalize_alphabet(config.data.token_merge_alphabet),
        ).to(device)
    manifest = _read_manifest(input_dir)
    sources = _source_files(input_dir, manifest)
    if not sources:
        raise FileNotFoundError(f"no FASTA files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_source: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for index, (source, path, manifest_entry) in enumerate(sources, start=1):
        print(f"[dna_fasta] loading {index}/{len(sources)} source={source} path={path}", flush=True)
        source_wall_started = time.perf_counter()
        fasta_read_started = time.perf_counter()
        payload, fasta_stats = _load_fasta_sequence(path, alphabet=alphabet)
        fasta_read_seconds = time.perf_counter() - fasta_read_started
        requested_bytes = args.compression_sample_bytes if args.compression_sample_bytes > 0 else None
        print(
            f"[dna_fasta] compress source={source} sequence_bytes={len(payload)} "
            f"fasta_file_bytes={fasta_stats['fasta_file_bytes']}",
            flush=True,
        )

        def _progress(done: int, total: int, *, source_name: str = source) -> None:
            ratio = 100.0 * done / max(total, 1)
            print(
                f"\r[dna_fasta] source={source_name} batch={done}/{total} ({ratio:5.1f}%)",
                end="",
                flush=True,
            )

        compression_started = time.perf_counter()
        metrics = compress_source(
            model=model,
            source=payload,
            seq_length=config.model.seq_length,
            pad_id=config.model.pad_id,
            eos_id=config.model.eos_id,
            device=device,
            dtype_name=config.train.dtype,
            batch_size=config.train.eval_batch_size,
            requested_bytes=requested_bytes,
            mode=args.mode_name,
            token_merge_size=config.data.token_merge_size,
            token_merge_alphabet=config.data.token_merge_alphabet,
            arithmetic_frequency_total=config.arithmetic.frequency_total,
            arithmetic_target_uniform_mass=config.arithmetic.target_uniform_mass,
            arithmetic_coding_mode=config.arithmetic.coding_mode,
            arithmetic_merge_size=config.arithmetic.merge_size,
            arithmetic_backend=config.arithmetic.backend,
            factorizer=factorizer,
            include_codec_baselines=False,
            progress_callback=_progress,
        )
        compression_wall_seconds = time.perf_counter() - compression_started
        total_wall_seconds = time.perf_counter() - source_wall_started
        print()
        row = {
            "species": source,
            "source_name": source,
            "source_path": str(path),
            **metrics,
            **fasta_stats,
            "fasta_text_skipped_bytes": fasta_stats["fasta_header_bytes"]
            + fasta_stats["fasta_newline_bytes"]
            + fasta_stats["fasta_other_sequence_bytes"],
            "manifest_selected_record_count": manifest_entry.get("selected_record_count"),
            "manifest_selected_bytes": manifest_entry.get("selected_bytes"),
            "manifest_bytes_written": manifest_entry.get("bytes_written"),
            "fasta_read_seconds": fasta_read_seconds,
            "compression_wall_seconds": compression_wall_seconds,
            "total_wall_seconds_including_fasta_read": total_wall_seconds,
            "wall_seconds_including_fasta_read": total_wall_seconds,
        }
        per_source.append(row)
        dataset_rows.append(
            {
                "species": source,
                "source_name": source,
                "source_mode": "fasta_sequence_content",
                "source_path": str(path),
                "total_size": len(payload),
                **fasta_stats,
            }
        )

        output_json = output_dir / "compression_compare.json"
        partial_metrics = {
            "device": str(device),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": checkpoint.get("step"),
            "best_val_bpb": checkpoint.get("best_val_bpb"),
            "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "resolved_config": config.to_dict(),
            "input_dir": str(input_dir),
            "sequence_content_alphabet": alphabet,
            "dataset": {
                "dataset_dir": str(input_dir),
                "sequence_source_mode": "fasta_sequence_content",
                "species": dataset_rows,
            },
            "results": {
                args.split_name: {
                    args.mode_name: {
                        "aggregate": summarize_per_source(per_source),
                        "per_source": per_source,
                    }
                }
            },
        }
        output_json.write_text(json.dumps(partial_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_csv(output_dir / "compression_per_source_by_split_mode.csv", [{**item, "split": args.split_name, "mode": args.mode_name} for item in per_source])
        _write_csv(output_dir / "compression_aggregate_by_split_mode.csv", [{"split": args.split_name, "mode": args.mode_name, **summarize_per_source(per_source)}])
        write_compression_report_tables(output_dir, partial_metrics)

    output_json = output_dir / "compression_compare.json"
    if not args.no_plots:
        baseline_path = (
            Path(args.baseline_compression_json)
            if args.baseline_compression_json
            else resolve_geco2_baseline_path(args.geco2_baseline)
        )
        if baseline_path is not None and not baseline_path.exists():
            print(f"[dna_fasta] skip baseline overlay: not found {baseline_path}", flush=True)
            baseline_path = None
        generated = generate_artifacts_for_compression_compare(
            output_json,
            out_dir_name=args.compression_curves_out_dir_name,
            baseline_compression_compare_path=baseline_path,
        )
        print(f"[dna_fasta] generated {len(generated)} plot/table artifacts", flush=True)
    print(f"[dna_fasta] saved metrics to {output_json}", flush=True)


if __name__ == "__main__":
    main()
