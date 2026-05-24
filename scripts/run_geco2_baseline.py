from __future__ import annotations

"""Run GeCo2 as an external train/val/test compression baseline.

The runner appends a ``geco2_paper_modes`` mode to an existing ``compression_compare.json``
so GeCo2 can be compared with the same per-split tables and plots as model
compression runs.
"""

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
from time import perf_counter
import shutil
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression import baseline_sizes
from dna_compress.compression_eval import sample_payload, summarize_per_source
from dna_compress.config import DataConfig
from dna_compress.data import load_splits
from scripts.plot_compression_curves import GECO2_PAPER_BASELINE_BY_SOURCE, generate_artifacts_for_compression_compare


DEFAULT_COMPRESSION_JSON = Path("outputs/dna_fusion_megabyte_dnagpt/statistics/compression_compare.json")
DEFAULT_MODE_NAME = "geco2_paper_modes"
DEFAULT_LEVEL = 5


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _flatten_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


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
            writer.writerow({key: _flatten_value(row.get(key)) for key in fieldnames})


def _tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _data_config_from_metrics(metrics: dict[str, Any]) -> DataConfig:
    data_payload: dict[str, Any] | None = None
    models = metrics.get("models")
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            config = model.get("config")
            if not isinstance(config, dict):
                continue
            candidate = config.get("data")
            if isinstance(candidate, dict):
                data_payload = candidate
                break

    if data_payload is None:
        resolved_config = metrics.get("resolved_config")
        if isinstance(resolved_config, dict) and isinstance(resolved_config.get("data"), dict):
            data_payload = resolved_config["data"]

    if data_payload is None:
        raise ValueError("Could not find data config in compression_compare.json")

    config = DataConfig()
    valid_fields = {field.name for field in fields(DataConfig)}
    for key, value in data_payload.items():
        if key in valid_fields:
            setattr(config, key, value)
    return config


def _sources_for_split(splits, split_name: str) -> list[bytes]:
    if split_name == "train":
        return splits.train_sources
    if split_name == "val":
        return splits.val_sources
    if split_name == "test":
        return splits.test_sources
    raise ValueError(f"Unsupported split '{split_name}'")


def _normalize_splits(raw_splits: list[str]) -> list[str]:
    if "all" in raw_splits:
        return ["train", "val", "test"]
    return raw_splits


def _source_entries(splits) -> list[dict[str, Any]]:
    return [dict(item) for item in splits.summary["species"]]


def _resolve_geco2_binary(requested_binary: str) -> str:
    resolved = shutil.which(requested_binary)
    if resolved is None:
        raise FileNotFoundError(f"Could not find GeCo2 binary: {requested_binary}")
    return resolved


def build_geco2_command(binary: str, *, level: int, input_path: Path) -> list[str]:
    return [binary, "-F", "-v", "-l", str(level), str(input_path)]


def resolve_geco2_level(*, species: str, source_name: str, default_level: int) -> int:
    for key in (source_name, species):
        baseline = GECO2_PAPER_BASELINE_BY_SOURCE.get(key)
        if baseline is not None:
            return int(baseline["mode"])
    return int(default_level)


def run_geco2_file(
    *,
    binary: str,
    input_path: Path,
    level: int,
) -> dict[str, Any]:
    command = build_geco2_command(binary, level=level, input_path=input_path)
    started = perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = perf_counter() - started
    compressed_path = Path(str(input_path) + ".co")

    result: dict[str, Any] = {
        "command": command,
        "returncode": completed.returncode,
        "seconds": elapsed,
        "stdout_tail": _tail_text(completed.stdout),
        "stderr_tail": _tail_text(completed.stderr),
        "compressed_path": str(compressed_path),
    }
    if completed.returncode != 0:
        raise RuntimeError(
            "GeCo2 failed with return code "
            f"{completed.returncode}: {_tail_text(completed.stderr or completed.stdout)}"
        )
    if not compressed_path.exists():
        candidates = sorted(input_path.parent.glob("*.co"))
        if len(candidates) == 1:
            compressed_path = candidates[0]
            result["compressed_path"] = str(compressed_path)
        else:
            raise FileNotFoundError(f"GeCo2 did not create expected output: {compressed_path}")

    result["compressed_bytes"] = compressed_path.stat().st_size
    return result


def compress_payload_with_geco2(
    *,
    payload: bytes,
    binary: str,
    level: int,
    temp_root: Path | None = None,
    keep_temp: bool = False,
    label: str = "sample",
) -> dict[str, Any]:
    if keep_temp:
        temp_dir = Path(tempfile.mkdtemp(prefix="geco2_baseline_", dir=str(temp_root) if temp_root else None))
        cleanup = False
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="geco2_baseline_", dir=str(temp_root) if temp_root else None)
        temp_dir = Path(temp_context.__enter__())
        cleanup = True

    try:
        sample_path = temp_dir / f"{label}.seq"
        sample_path.write_bytes(payload)
        result = run_geco2_file(binary=binary, input_path=sample_path, level=level)
        result["input_path"] = str(sample_path)
        result["temp_dir"] = str(temp_dir)
        return result
    finally:
        if cleanup:
            temp_context.__exit__(None, None, None)


def _build_source_metrics(
    *,
    species: str,
    source_name: str,
    payload: bytes,
    geco2_result: dict[str, Any],
    level: int,
    binary: str,
) -> dict[str, Any]:
    compressed_bytes = int(geco2_result["compressed_bytes"])
    compressed_bits = float(compressed_bytes * 8)
    sample_bases = len(payload)
    seconds = float(geco2_result["seconds"])
    return {
        "species": species,
        "source_name": source_name,
        "mode": DEFAULT_MODE_NAME,
        "geco2_level": level,
        "geco2_binary": binary,
        "geco2_command": geco2_result["command"],
        "geco2_returncode": geco2_result["returncode"],
        "geco2_stdout_tail": geco2_result["stdout_tail"],
        "geco2_stderr_tail": geco2_result["stderr_tail"],
        "sample_bytes": len(payload),
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": sample_bases,
        "uses_eos": False,
        "theoretical_bits": compressed_bits,
        "theoretical_bits_per_base": compressed_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": compressed_bytes,
        "arithmetic_bits_per_base": compressed_bits / max(sample_bases, 1),
        "arithmetic_coding_mode": DEFAULT_MODE_NAME,
        "arithmetic_merge_size": 1,
        "emitted_arithmetic_symbol_count": sample_bases,
        "compression_process_seconds": seconds,
        "compression_bytes_per_second": len(payload) / max(seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(seconds, 1e-12),
        "compression_symbols_per_second": sample_bases / max(seconds, 1e-12),
        **baseline_sizes(payload),
    }


def run_geco2_baseline(
    *,
    compression_json: Path,
    output_json: Path,
    binary: str,
    level: int,
    split_names: list[str],
    compression_sample_bytes: int | None,
    max_sources: int | None = None,
    temp_root: Path | None = None,
    keep_temp: bool = False,
) -> dict[str, Any]:
    metrics = _read_json(compression_json)
    data_config = _data_config_from_metrics(metrics)
    if compression_sample_bytes is not None:
        data_config.compression_sample_bytes = compression_sample_bytes

    print("[geco2] loading data splits...", flush=True)
    splits = load_splits(data_config)
    entries = _source_entries(splits)
    if max_sources is not None:
        entries = entries[:max_sources]

    results = metrics.setdefault("results", {})
    if not isinstance(results, dict):
        raise ValueError("compression_compare.json has non-dict results")

    for split_name in split_names:
        sources = _sources_for_split(splits, split_name)
        if max_sources is not None:
            sources = sources[:max_sources]
        if len(sources) != len(entries):
            raise RuntimeError(f"Split {split_name} has {len(sources)} sources but dataset has {len(entries)} entries")

        per_source: list[dict[str, Any]] = []
        for source_index, (entry, source) in enumerate(zip(entries, sources), start=1):
            species = str(entry["species"])
            source_name = str(entry.get("source_name", species))
            level_for_source = resolve_geco2_level(
                species=species,
                source_name=source_name,
                default_level=level,
            )
            payload = sample_payload(source, data_config.compression_sample_bytes)
            print(
                f"[geco2] split={split_name} source={source_index}/{len(sources)}({source_name}) "
                f"level={level_for_source} bytes={len(payload)}",
                flush=True,
            )
            geco2_result = compress_payload_with_geco2(
                payload=payload,
                binary=binary,
                level=level_for_source,
                temp_root=temp_root,
                keep_temp=keep_temp,
                label=f"{split_name}_{source_index}_{source_name}".replace(":", "_"),
            )
            per_source.append(
                _build_source_metrics(
                    species=species,
                    source_name=source_name,
                    payload=payload,
                    geco2_result=geco2_result,
                    level=level_for_source,
                    binary=binary,
                )
            )

        aggregate = summarize_per_source(per_source)
        aggregate["geco2_level"] = "paper_modes"
        aggregate["geco2_binary"] = binary
        split_payload = results.setdefault(split_name, {})
        if not isinstance(split_payload, dict):
            raise ValueError(f"results.{split_name} is not a dict")
        split_payload[DEFAULT_MODE_NAME] = {
            "aggregate": aggregate,
            "per_source": per_source,
        }

    metrics["geco2_baseline"] = {
        "mode": DEFAULT_MODE_NAME,
        "level": "paper_modes",
        "fallback_level": level,
        "binary": binary,
        "compression_sample_bytes": data_config.compression_sample_bytes,
        "splits": split_names,
    }
    _write_json(output_json, metrics)
    _write_statistics_tables(output_json, metrics)
    generated_curves = generate_artifacts_for_compression_compare(output_json)
    if generated_curves:
        print(f"[geco2] generated {len(generated_curves)} compression curve artifacts", flush=True)
    return metrics


def _write_statistics_tables(output_json: Path, metrics: dict[str, Any]) -> None:
    results = metrics.get("results")
    if not isinstance(results, dict):
        return

    aggregate_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []
    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        for mode_name, mode_payload in split_payload.items():
            if not isinstance(mode_payload, dict):
                continue
            aggregate = mode_payload.get("aggregate")
            if isinstance(aggregate, dict):
                aggregate_rows.append({"split": split_name, "mode": mode_name, **aggregate})
            per_source = mode_payload.get("per_source")
            if isinstance(per_source, list):
                for row in per_source:
                    if isinstance(row, dict):
                        per_source_rows.append({**row, "split": split_name, "mode": mode_name})

    _write_csv(output_json.parent / "compression_aggregate_by_split_mode.csv", aggregate_rows)
    _write_csv(output_json.parent / "compression_per_source_by_split_mode.csv", per_source_rows)

    dataset = metrics.get("dataset")
    dataset_rows = []
    if isinstance(dataset, dict) and isinstance(dataset.get("species"), list):
        for row in dataset["species"]:
            if isinstance(row, dict):
                dataset_rows.append(dict(row))
    _write_csv(output_json.parent / "dataset_splits.csv", dataset_rows)

    summary_rows: list[dict[str, Any]] = []
    geco2_baseline = metrics.get("geco2_baseline")
    if isinstance(geco2_baseline, dict):
        for key, value in sorted(geco2_baseline.items()):
            summary_rows.append({"metric": f"geco2_baseline.{key}", "value": value})
    for row in aggregate_rows:
        prefix = f"compression_compare.{row.get('split')}.{row.get('mode')}"
        for key in (
            "total_theoretical_bits_per_base",
            "total_arithmetic_bits_per_base",
            "total_sample_bases",
            "total_arithmetic_coded_bytes",
            "total_compression_bases_per_second",
        ):
            if key in row:
                summary_rows.append({"metric": f"{prefix}.{key}", "value": row[key]})
    _write_csv(output_json.parent / "summary_metrics.csv", summary_rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GeCo2 train/val/test baseline and merge it into compression JSON.")
    parser.add_argument("--compression-json", default=str(DEFAULT_COMPRESSION_JSON))
    parser.add_argument("--output-json", default=None, help="Defaults to --compression-json in-place.")
    parser.add_argument("--geco2-bin", default="GeCo2")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL, help="Fallback -l for species absent from paper table.")
    parser.add_argument("--split", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test", "all"])
    parser.add_argument("--compression-sample-bytes", type=int, default=60000)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--temp-root")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    compression_json = Path(args.compression_json)
    output_json = Path(args.output_json) if args.output_json is not None else compression_json
    binary = _resolve_geco2_binary(args.geco2_bin)
    split_names = _normalize_splits(args.split)
    metrics = run_geco2_baseline(
        compression_json=compression_json,
        output_json=output_json,
        binary=binary,
        level=args.level,
        split_names=split_names,
        compression_sample_bytes=args.compression_sample_bytes,
        max_sources=args.max_sources,
        temp_root=Path(args.temp_root) if args.temp_root is not None else None,
        keep_temp=args.keep_temp,
    )
    print(f"Saved GeCo2 baseline metrics to {output_json}")
    for split_name in split_names:
        aggregate = metrics["results"][split_name][DEFAULT_MODE_NAME]["aggregate"]
        print(
            f"[geco2] {split_name}: sources={aggregate['source_count']} "
            f"bpb={aggregate['total_arithmetic_bits_per_base']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
