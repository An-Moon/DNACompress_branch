from __future__ import annotations

"""Export the same payload used for W&B upload into local files.

Outputs:
- run_metadata.json
- resolved_config.json (copied when available)
- summary_metrics.csv
- dataset_splits.csv
- compression_per_source_legacy.csv
- compression_ratio_summary.csv
- compression_speed_summary.csv
- compression_aggregate_by_split_mode.csv
- compression_per_source_by_split_mode.csv

Example:
    python scripts/export_statistics.py \
      --run-dir outputs/dna_nugget_modified_r0.25
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_dict(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_dict(next_prefix, nested, out)
        return
    if isinstance(value, list):
        return
    out[prefix] = value


def _dataset_summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dataset = metrics.get("dataset")
    if not isinstance(dataset, dict):
        return result

    species_rows = dataset.get("species")
    if isinstance(species_rows, list):
        result["dataset.species_count"] = len(species_rows)
        total_size = 0
        total_train = 0
        total_val = 0
        total_test = 0
        total_full = 0
        for row in species_rows:
            if not isinstance(row, dict):
                continue
            total_size += int(row.get("total_size", 0) or 0)
            total_train += int(row.get("train_bytes", 0) or 0)
            total_val += int(row.get("val_bytes", 0) or 0)
            total_test += int(row.get("test_bytes", 0) or 0)
            total_full += int(row.get("full_bytes", 0) or 0)
        result["dataset.total_size_bytes"] = total_size
        result["dataset.total_train_bytes"] = total_train
        result["dataset.total_val_bytes"] = total_val
        result["dataset.total_test_bytes"] = total_test
        if total_full:
            result["dataset.total_full_bytes"] = total_full

    alphabet_bytes = dataset.get("alphabet_bytes")
    if isinstance(alphabet_bytes, list):
        result["dataset.alphabet_size"] = len(alphabet_bytes)

    return result


def _collect_summary_metrics(metrics: dict[str, Any], compression_compare: dict[str, Any] | None) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for key in ["device", "model_parameters", "best_val_bits_per_base"]:
        if key in metrics:
            summary[key] = metrics[key]

    validation = metrics.get("validation")
    if isinstance(validation, dict):
        for k, v in validation.items():
            summary[f"validation.{k}"] = v

    test = metrics.get("test")
    if isinstance(test, dict):
        for k, v in test.items():
            summary[f"test.{k}"] = v

    compression = metrics.get("compression")
    if isinstance(compression, dict):
        aggregate = compression.get("aggregate")
        if isinstance(aggregate, dict):
            for k, v in aggregate.items():
                summary[f"compression.aggregate.{k}"] = v

    summary.update(_dataset_summary_from_metrics(metrics))

    if isinstance(compression_compare, dict):
        for key in ["checkpoint_step", "best_val_bpb", "overlap_stride_tokens", "overlap_stride_patches"]:
            if key in compression_compare:
                summary[f"compression_compare.{key}"] = compression_compare[key]
        arithmetic = compression_compare.get("arithmetic")
        if isinstance(arithmetic, dict):
            for key, value in arithmetic.items():
                summary[f"compression_compare.arithmetic.{key}"] = value

    return summary


def _build_dataset_table_rows(dataset: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(dataset, dict):
        return rows

    species_rows = dataset.get("species")
    if not isinstance(species_rows, list):
        return rows

    for row in species_rows:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "species": row.get("species"),
                "source_name": row.get("source_name"),
                "source_mode": row.get("source_mode"),
                "selected_sequence_count": row.get("selected_sequence_count"),
                "sequence_keys": "|".join(row.get("sequence_keys", [])) if isinstance(row.get("sequence_keys"), list) else row.get("sequence_keys"),
                "sequence_files": "|".join(row.get("sequence_files", [])) if isinstance(row.get("sequence_files"), list) else row.get("sequence_files"),
                "total_size": row.get("total_size"),
                "full_bytes": row.get("full_bytes"),
                "train_bytes": row.get("train_bytes"),
                "val_bytes": row.get("val_bytes"),
                "test_bytes": row.get("test_bytes"),
            }
        )
    return rows


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_number(value: Any) -> int | None:
    numeric = _number(value)
    if numeric is None:
        return None
    return int(numeric)


def _row_value(row: dict[str, Any], row_type: str, source_key: str, aggregate_key: str | None = None) -> Any:
    if row_type == "aggregate":
        return row.get(aggregate_key or f"total_{source_key}")
    return row.get(source_key)


def _ratio_row(
    *,
    split_name: str,
    mode_name: str,
    row_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    sample_bytes = _number(_row_value(payload, row_type, "sample_bytes"))
    sample_bases = _number(_row_value(payload, row_type, "sample_bases"))
    theoretical_bits = _number(_row_value(payload, row_type, "theoretical_bits"))
    theoretical_bpb = _number(_row_value(payload, row_type, "theoretical_bits_per_base"))
    arithmetic_bytes = _number(_row_value(payload, row_type, "arithmetic_coded_bytes"))
    arithmetic_bpb = _number(_row_value(payload, row_type, "arithmetic_bits_per_base"))
    ascii_bytes = _number(_row_value(payload, row_type, "ascii_bytes"))
    two_bit_pack_bytes = _number(_row_value(payload, row_type, "two_bit_pack_bytes"))
    gzip_bytes = _number(_row_value(payload, row_type, "gzip_bytes"))
    bz2_bytes = _number(_row_value(payload, row_type, "bz2_bytes"))
    lzma_bytes = _number(_row_value(payload, row_type, "lzma_bytes"))

    if theoretical_bpb is None and theoretical_bits is not None and sample_bases:
        theoretical_bpb = theoretical_bits / sample_bases
    if arithmetic_bpb is None and arithmetic_bytes is not None and sample_bases:
        arithmetic_bpb = arithmetic_bytes * 8.0 / sample_bases

    row = {
        "split": split_name,
        "mode": mode_name,
        "row_type": row_type,
        "species": payload.get("species") if row_type == "source" else None,
        "source_name": payload.get("source_name") if row_type == "source" else "ALL",
        "source_count": payload.get("source_count") if row_type == "aggregate" else None,
        "sample_bytes": _int_number(sample_bytes),
        "sample_bases": _int_number(sample_bases),
        "sample_symbols_with_eos": _row_value(
            payload,
            row_type,
            "sample_symbols_with_eos",
            "total_emitted_arithmetic_symbol_count",
        ),
        "theoretical_bits": theoretical_bits,
        "theoretical_bits_per_base": theoretical_bpb,
        "arithmetic_coded_bytes": _int_number(arithmetic_bytes),
        "arithmetic_bits_per_base": arithmetic_bpb,
        "arithmetic_vs_2bit_percent": arithmetic_bpb / 2.0 * 100.0 if arithmetic_bpb is not None else None,
        "arithmetic_bytes_ratio_vs_ascii": _safe_div(arithmetic_bytes, ascii_bytes)
        if arithmetic_bytes is not None and ascii_bytes is not None
        else None,
        "arithmetic_size_reduction_vs_ascii_percent": (1.0 - arithmetic_bytes / ascii_bytes) * 100.0
        if arithmetic_bytes is not None and ascii_bytes
        else None,
        "arithmetic_vs_two_bit_pack_ratio": _safe_div(arithmetic_bytes, two_bit_pack_bytes)
        if arithmetic_bytes is not None and two_bit_pack_bytes is not None
        else None,
        "arithmetic_vs_gzip_ratio": _safe_div(arithmetic_bytes, gzip_bytes)
        if arithmetic_bytes is not None and gzip_bytes is not None
        else None,
        "arithmetic_vs_bz2_ratio": _safe_div(arithmetic_bytes, bz2_bytes)
        if arithmetic_bytes is not None and bz2_bytes is not None
        else None,
        "arithmetic_vs_lzma_ratio": _safe_div(arithmetic_bytes, lzma_bytes)
        if arithmetic_bytes is not None and lzma_bytes is not None
        else None,
        "ascii_bytes": _int_number(ascii_bytes),
        "two_bit_pack_bytes": _int_number(two_bit_pack_bytes),
        "gzip_bytes": _int_number(gzip_bytes),
        "bz2_bytes": _int_number(bz2_bytes),
        "lzma_bytes": _int_number(lzma_bytes),
        "arithmetic_coding_mode": payload.get("arithmetic_coding_mode"),
        "arithmetic_merge_size": payload.get("arithmetic_merge_size"),
        "arithmetic_backend": payload.get("arithmetic_backend"),
        "arithmetic_frequency_total": payload.get("arithmetic_frequency_total"),
        "arithmetic_vocab_size": payload.get("arithmetic_vocab_size"),
        "arithmetic_target_uniform_mass": payload.get("arithmetic_target_uniform_mass"),
        "arithmetic_effective_uniform_mass": payload.get("arithmetic_effective_uniform_mass"),
    }
    for key in (
        "total_bits_per_base",
        "core_model_theoretical_bits",
        "tail_base_count",
        "tail_side_info_bits",
        "side_info_bytes",
        "nugget_code_bytes",
        "nugget_hidden_bytes",
        "nugget_metadata_bytes",
        "nugget_score_bytes",
        "window_policy",
        "window_stride",
        "cache_reuse",
    ):
        if key in payload:
            row[key] = payload.get(key)
    return row


def _speed_row(
    *,
    split_name: str,
    mode_name: str,
    row_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    sample_bytes = _number(_row_value(payload, row_type, "sample_bytes"))
    sample_bases = _number(_row_value(payload, row_type, "sample_bases"))
    emitted_symbols = _number(
        _row_value(payload, row_type, "emitted_arithmetic_symbol_count", "total_emitted_arithmetic_symbol_count")
    )
    model_forward_seconds = _number(_row_value(payload, row_type, "model_forward_seconds"))
    softmax_seconds = _number(_row_value(payload, row_type, "softmax_seconds"))
    model_forward_softmax_seconds = (
        model_forward_seconds + softmax_seconds
        if model_forward_seconds is not None and softmax_seconds is not None
        else _number(_row_value(payload, row_type, "model_forward_softmax_seconds"))
    )
    data_transfer_seconds = _number(_row_value(payload, row_type, "data_transfer_seconds"))
    gpu_prefix_aggregate_seconds = _number(_row_value(payload, row_type, "gpu_prefix_aggregate_seconds"))
    quantize_seconds = _number(
        _row_value(payload, row_type, "arithmetic_quantize_seconds", "total_arithmetic_quantize_seconds")
    )
    if quantize_seconds is None:
        quantize_seconds = _number(
            _row_value(
                payload,
                row_type,
                "cpu_small_alphabet_quantize_seconds",
                "total_cpu_small_alphabet_quantize_seconds",
            )
        )
    range_seconds = _number(_row_value(payload, row_type, "arithmetic_range_seconds"))
    arithmetic_encode_seconds = _number(_row_value(payload, row_type, "arithmetic_encode_seconds"))
    compression_process_seconds = _number(_row_value(payload, row_type, "compression_process_seconds"))
    if compression_process_seconds is None:
        parts = [
            model_forward_seconds,
            softmax_seconds,
            gpu_prefix_aggregate_seconds,
            data_transfer_seconds,
            arithmetic_encode_seconds,
        ]
        if all(value is not None for value in parts):
            compression_process_seconds = sum(float(value) for value in parts if value is not None)

    wall_seconds = _number(payload.get("compression_wall_seconds"))
    wall_seconds_source = "compression_wall_seconds" if wall_seconds is not None else None
    if wall_seconds is None:
        wall_seconds = _number(payload.get("wall_seconds_including_fasta_read"))
        wall_seconds_source = "wall_seconds_including_fasta_read" if wall_seconds is not None else None
    if wall_seconds is None and row_type == "aggregate":
        wall_seconds = _number(payload.get("total_wall_seconds"))
        wall_seconds_source = "total_wall_seconds" if wall_seconds is not None else None

    unaccounted_wall_seconds = (
        wall_seconds - compression_process_seconds
        if wall_seconds is not None and compression_process_seconds is not None
        else None
    )

    row = {
        "split": split_name,
        "mode": mode_name,
        "row_type": row_type,
        "species": payload.get("species") if row_type == "source" else None,
        "source_name": payload.get("source_name") if row_type == "source" else "ALL",
        "source_count": payload.get("source_count") if row_type == "aggregate" else None,
        "sample_bytes": _int_number(sample_bytes),
        "sample_bases": _int_number(sample_bases),
        "emitted_arithmetic_symbol_count": _int_number(emitted_symbols),
        "model_forward_seconds": model_forward_seconds,
        "softmax_seconds": softmax_seconds,
        "model_forward_softmax_seconds": model_forward_softmax_seconds,
        "data_transfer_seconds": data_transfer_seconds,
        "gpu_prefix_aggregate_seconds": gpu_prefix_aggregate_seconds,
        "arithmetic_quantize_seconds": quantize_seconds,
        "cpu_small_alphabet_quantize_seconds": quantize_seconds,
        "arithmetic_range_seconds": range_seconds,
        "arithmetic_encode_seconds": arithmetic_encode_seconds,
        "compression_process_seconds": compression_process_seconds,
        "recorded_wall_seconds": wall_seconds,
        "recorded_wall_seconds_source": wall_seconds_source,
        "wall_unaccounted_seconds": unaccounted_wall_seconds,
        "fasta_read_seconds": payload.get("fasta_read_seconds"),
        "compression_bytes_per_second": _safe_div(sample_bytes, compression_process_seconds)
        if sample_bytes is not None and compression_process_seconds is not None
        else None,
        "compression_bases_per_second": _safe_div(sample_bases, compression_process_seconds)
        if sample_bases is not None and compression_process_seconds is not None
        else None,
        "compression_symbols_per_second": _safe_div(emitted_symbols, compression_process_seconds)
        if emitted_symbols is not None and compression_process_seconds is not None
        else None,
        "wall_bytes_per_second": _safe_div(sample_bytes, wall_seconds)
        if sample_bytes is not None and wall_seconds is not None
        else None,
        "wall_bases_per_second": _safe_div(sample_bases, wall_seconds)
        if sample_bases is not None and wall_seconds is not None
        else None,
        "model_forward_process_fraction": _safe_div(model_forward_seconds, compression_process_seconds)
        if model_forward_seconds is not None and compression_process_seconds is not None
        else None,
        "data_transfer_process_fraction": _safe_div(data_transfer_seconds, compression_process_seconds)
        if data_transfer_seconds is not None and compression_process_seconds is not None
        else None,
        "arithmetic_quantize_process_fraction": _safe_div(quantize_seconds, compression_process_seconds)
        if quantize_seconds is not None and compression_process_seconds is not None
        else None,
        "arithmetic_range_process_fraction": _safe_div(range_seconds, compression_process_seconds)
        if range_seconds is not None and compression_process_seconds is not None
        else None,
        "arithmetic_encode_process_fraction": _safe_div(arithmetic_encode_seconds, compression_process_seconds)
        if arithmetic_encode_seconds is not None and compression_process_seconds is not None
        else None,
        "wall_unaccounted_fraction": _safe_div(unaccounted_wall_seconds, wall_seconds)
        if unaccounted_wall_seconds is not None and wall_seconds is not None
        else None,
        "arithmetic_backend": payload.get("arithmetic_backend"),
        "arithmetic_coding_mode": payload.get("arithmetic_coding_mode"),
        "arithmetic_merge_size": payload.get("arithmetic_merge_size"),
    }
    return row


def build_compression_report_tables(
    compression_compare: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ratio_rows: list[dict[str, Any]] = []
    speed_rows: list[dict[str, Any]] = []
    if not isinstance(compression_compare, dict):
        return ratio_rows, speed_rows

    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return ratio_rows, speed_rows

    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        for mode_name, mode_payload in split_payload.items():
            if not isinstance(mode_payload, dict):
                continue

            per_source = [row for row in mode_payload.get("per_source", []) if isinstance(row, dict)]
            aggregate = mode_payload.get("aggregate")
            if isinstance(aggregate, dict):
                aggregate_payload = dict(aggregate)
                wall_values = [_number(row.get("compression_wall_seconds")) for row in per_source]
                if not any(value is not None for value in wall_values):
                    wall_values = [_number(row.get("wall_seconds_including_fasta_read")) for row in per_source]
                if any(value is not None for value in wall_values):
                    aggregate_payload["total_wall_seconds"] = sum(float(value) for value in wall_values if value is not None)
                ratio_rows.append(
                    _ratio_row(
                        split_name=str(split_name),
                        mode_name=str(mode_name),
                        row_type="aggregate",
                        payload=aggregate_payload,
                    )
                )
                speed_rows.append(
                    _speed_row(
                        split_name=str(split_name),
                        mode_name=str(mode_name),
                        row_type="aggregate",
                        payload=aggregate_payload,
                    )
                )

            for source_row in per_source:
                ratio_rows.append(
                    _ratio_row(
                        split_name=str(split_name),
                        mode_name=str(mode_name),
                        row_type="source",
                        payload=source_row,
                    )
                )
                speed_rows.append(
                    _speed_row(
                        split_name=str(split_name),
                        mode_name=str(mode_name),
                        row_type="source",
                        payload=source_row,
                    )
                )

    return ratio_rows, speed_rows


def write_compression_report_tables(out_dir: Path, compression_compare: dict[str, Any] | None) -> tuple[Path, Path]:
    ratio_rows, speed_rows = build_compression_report_tables(compression_compare)
    ratio_path = out_dir / "compression_ratio_summary.csv"
    speed_path = out_dir / "compression_speed_summary.csv"
    _write_csv(ratio_path, ratio_rows)
    _write_csv(speed_path, speed_rows)
    return ratio_path, speed_path


def _build_compression_tables(compression_compare: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aggregate_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []

    if not isinstance(compression_compare, dict):
        return aggregate_rows, per_source_rows

    results = compression_compare.get("results")
    if not isinstance(results, dict):
        return aggregate_rows, per_source_rows

    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        for mode_name, mode_payload in split_payload.items():
            if not isinstance(mode_payload, dict):
                continue

            aggregate = mode_payload.get("aggregate")
            if isinstance(aggregate, dict):
                aggregate_row = {"split": split_name, "mode": mode_name}
                aggregate_row.update(aggregate)
                aggregate_rows.append(aggregate_row)

            per_source = mode_payload.get("per_source")
            if not isinstance(per_source, list):
                continue

            for source_row in per_source:
                if not isinstance(source_row, dict):
                    continue

                row = {
                    "split": split_name,
                    "mode": mode_name,
                    "species": source_row.get("species"),
                    "source_name": source_row.get("source_name"),
                    "sample_bytes": source_row.get("sample_bytes"),
                    "sample_bases": source_row.get("sample_bases"),
                    "theoretical_bits_per_base": source_row.get("theoretical_bits_per_base"),
                    "arithmetic_bits_per_base": source_row.get("arithmetic_bits_per_base"),
                    "arithmetic_coding_mode": source_row.get("arithmetic_coding_mode"),
                    "arithmetic_merge_size": source_row.get("arithmetic_merge_size"),
                    "arithmetic_backend": source_row.get("arithmetic_backend"),
                    "arithmetic_frequency_total": source_row.get("arithmetic_frequency_total"),
                    "arithmetic_vocab_size": source_row.get("arithmetic_vocab_size"),
                    "arithmetic_target_uniform_mass": source_row.get("arithmetic_target_uniform_mass"),
                    "arithmetic_effective_uniform_mass": source_row.get("arithmetic_effective_uniform_mass"),
                    "emitted_arithmetic_symbol_count": source_row.get("emitted_arithmetic_symbol_count"),
                    "nugget_latent_mode": source_row.get("nugget_latent_mode"),
                    "nugget_code_dim": source_row.get("nugget_code_dim"),
                    "nugget_flatten_bottleneck_dim": source_row.get("nugget_flatten_bottleneck_dim"),
                    "nugget_flatten_input_dim": source_row.get("nugget_flatten_input_dim"),
                    "nugget_flatten_max_nuggets": source_row.get("nugget_flatten_max_nuggets"),
                    "nugget_code_bytes": source_row.get("nugget_code_bytes"),
                    "nugget_hidden_bytes": source_row.get("nugget_hidden_bytes"),
                    "nugget_metadata_bytes": source_row.get("nugget_metadata_bytes"),
                    "nugget_score_bytes": source_row.get("nugget_score_bytes"),
                    "nugget_valid_count": source_row.get("nugget_valid_count"),
                    "side_info_bytes": source_row.get("side_info_bytes"),
                    "total_bits_per_base": source_row.get("total_bits_per_base"),
                    "core_model_theoretical_bits": source_row.get("core_model_theoretical_bits"),
                    "tail_base_count": source_row.get("tail_base_count"),
                    "tail_side_info_bits": source_row.get("tail_side_info_bits"),
                    "gpu_prefix_aggregate_seconds": source_row.get("gpu_prefix_aggregate_seconds"),
                    "cpu_small_alphabet_quantize_seconds": source_row.get("cpu_small_alphabet_quantize_seconds"),
                    "arithmetic_quantize_seconds": source_row.get("arithmetic_quantize_seconds"),
                    "arithmetic_range_seconds": source_row.get("arithmetic_range_seconds"),
                    "data_transfer_seconds": source_row.get("data_transfer_seconds"),
                    "arithmetic_encode_seconds": source_row.get("arithmetic_encode_seconds"),
                    "compression_process_seconds": source_row.get("compression_process_seconds"),
                    "ascii_bytes": source_row.get("ascii_bytes"),
                    "two_bit_pack_bytes": source_row.get("two_bit_pack_bytes"),
                    "gzip_bytes": source_row.get("gzip_bytes"),
                    "bz2_bytes": source_row.get("bz2_bytes"),
                    "lzma_bytes": source_row.get("lzma_bytes"),
                }

                arithmetic_bpb = source_row.get("arithmetic_bits_per_base")
                if isinstance(arithmetic_bpb, (int, float)):
                    row["arithmetic_vs_2bit_ratio"] = _safe_div(float(arithmetic_bpb), 2.0)

                sample_bytes = source_row.get("sample_bytes")
                arithmetic_bytes = source_row.get("arithmetic_coded_bytes")
                gzip_bytes = source_row.get("gzip_bytes")
                if isinstance(sample_bytes, (int, float)) and isinstance(arithmetic_bytes, (int, float)):
                    row["arithmetic_bytes_ratio_vs_ascii"] = _safe_div(float(arithmetic_bytes), float(sample_bytes))
                if isinstance(gzip_bytes, (int, float)) and isinstance(arithmetic_bytes, (int, float)):
                    row["arithmetic_vs_gzip_ratio"] = _safe_div(float(arithmetic_bytes), float(gzip_bytes))

                per_source_rows.append(row)

    return aggregate_rows, per_source_rows


def _build_legacy_compression_rows(metrics: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(metrics, dict):
        return rows

    compression = metrics.get("compression")
    if not isinstance(compression, dict):
        return rows

    per_source = compression.get("per_source")
    if not isinstance(per_source, list):
        return rows

    for row in per_source:
        if not isinstance(row, dict):
            continue
        rows.append({"split": "test", "mode": "legacy", **row})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    rows = [{"metric": key, "value": value} for key, value in sorted(summary.items())]
    _write_csv(path, rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export W&B upload payload into local JSON/CSV files.")
    parser.add_argument("--run-dir", required=True, help="Experiment output directory containing JSON outputs.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for exported files. Defaults to <run-dir>/statistics.",
    )
    parser.add_argument(
        "--resolved-config",
        default="resolved_config.json",
        help="Resolved config file name under run-dir.",
    )
    parser.add_argument(
        "--metrics-json",
        default="metrics.json",
        help="Metrics JSON file name under run-dir.",
    )
    parser.add_argument(
        "--compression-json",
        default="compression_compare.json",
        help="Compression compare JSON file name under run-dir.",
    )
    parser.add_argument("--project", default="", help="Optional project metadata to store in run_metadata.json.")
    parser.add_argument("--entity", default="", help="Optional entity metadata to store in run_metadata.json.")
    parser.add_argument("--name", default=None, help="Optional run name metadata. Defaults to run-dir folder name.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found or not a directory: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "statistics")
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / args.resolved_config
    metrics_path = run_dir / args.metrics_json
    compression_compare_path = run_dir / args.compression_json

    resolved_config = _read_json_if_exists(resolved_config_path)
    metrics = _read_json_if_exists(metrics_path)
    compression_compare = _read_json_if_exists(compression_compare_path)

    if metrics is None and compression_compare is None:
        raise ValueError("Neither metrics.json nor compression_compare.json found. Nothing to export.")

    run_name = args.name if args.name else run_dir.name

    run_metadata = {
        "project": args.project,
        "entity": args.entity,
        "name": run_name,
        "run_dir": str(run_dir),
        "has_resolved_config": resolved_config is not None,
        "has_metrics_json": metrics is not None,
        "has_compression_compare_json": compression_compare is not None,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if isinstance(resolved_config, dict):
        (out_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        model_config = resolved_config.get("model")
        if isinstance(model_config, dict):
            (out_dir / "model_config.json").write_text(
                json.dumps(model_config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    summary_source = metrics if isinstance(metrics, dict) else {}
    summary_metrics = _collect_summary_metrics(summary_source, compression_compare)
    if isinstance(resolved_config, dict):
        flat_config: dict[str, Any] = {}
        _flatten_dict("config", resolved_config, flat_config)
        summary_metrics.update(flat_config)
    _write_summary_csv(out_dir / "summary_metrics.csv", summary_metrics)

    dataset_payload = metrics.get("dataset") if isinstance(metrics, dict) else None
    if dataset_payload is None and isinstance(compression_compare, dict):
        dataset_payload = compression_compare.get("dataset")
    dataset_rows = _build_dataset_table_rows(dataset_payload)
    _write_csv(out_dir / "dataset_splits.csv", dataset_rows)

    legacy_rows = _build_legacy_compression_rows(metrics if isinstance(metrics, dict) else None)
    _write_csv(out_dir / "compression_per_source_legacy.csv", legacy_rows)

    aggregate_rows, per_source_rows = _build_compression_tables(compression_compare)
    _write_csv(out_dir / "compression_aggregate_by_split_mode.csv", aggregate_rows)
    _write_csv(out_dir / "compression_per_source_by_split_mode.csv", per_source_rows)
    write_compression_report_tables(out_dir, compression_compare)

    print(f"Exported payload files to: {out_dir}")


if __name__ == "__main__":
    main()
