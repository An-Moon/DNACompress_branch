from __future__ import annotations

"""Run extensible unit-level probability fusion compression.

Example:
    python scripts/run_fusion_compression.py \
    --model megabyte:outputs/dna_megabyte_large_b128_ensembl_all_finetune:best.pt \
    --model dnagpt:outputs/dna_dnagpt_0p1bm_all_finetuned_1:last.pt \
    --calibration-split train \
    --split val test \
    --fusion-policy static_context oracle_max \
    --fusion-unit-size auto \
    --context-units 1 \
    --compression-modes windows_nonoverlap \
    --compression-sample-bytes 60000 \
    --eval-batch-size 32 \
    --device auto \
    --output-json outputs/dna_fusion_megabyte_dnagpt/statistics/compression_compare.json
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.compression_eval import NON_OVERLAP_MODE, summarize_per_source
from dna_compress.config import ExperimentConfig
from dna_compress.data import load_splits
from dna_compress.fusion_compression import (
    FUSION_STATIC_CONTEXT,
    SUPPORTED_FUSION_POLICIES,
    StaticContextAccumulator,
    build_adapter_from_spec,
    build_fusion_source_inputs,
    compress_fusion_source,
    resolve_fusion_unit_size,
    write_static_context_table,
)
from dna_compress.tokenization import normalize_alphabet
from scripts.plot_compression_curves import generate_artifacts_for_compression_compare


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


def _source_entries(splits) -> list[dict[str, object]]:
    return [dict(item) for item in splits.summary["species"]]


def _adapter_config(adapter) -> ExperimentConfig:
    config = getattr(adapter, "config", None)
    if not isinstance(config, ExperimentConfig):
        raise TypeError(f"Adapter {adapter.name} does not expose an ExperimentConfig.")
    return config


def _apply_data_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.dataset_dir is not None:
        config.data.dataset_dir = args.dataset_dir
    if args.species is not None:
        config.data.species = args.species
    if args.train_ratio is not None:
        config.data.train_ratio = args.train_ratio
    if args.val_ratio is not None:
        config.data.val_ratio = args.val_ratio
    if args.test_ratio is not None:
        config.data.test_ratio = args.test_ratio
    if args.max_train_bytes is not None:
        config.data.max_train_bytes_per_species = args.max_train_bytes
    if args.max_val_bytes is not None:
        config.data.max_val_bytes_per_species = args.max_val_bytes
    if args.max_test_bytes is not None:
        config.data.max_test_bytes_per_species = args.max_test_bytes
    if args.compression_sample_bytes is not None:
        config.data.compression_sample_bytes = args.compression_sample_bytes


def _resolve_output_json(args: argparse.Namespace, adapters) -> Path:
    if args.output_json is not None:
        return Path(args.output_json)
    model_part = "_".join(adapter.name.rstrip("0123456789") or adapter.name for adapter in adapters)
    return Path("outputs") / f"dna_fusion_{model_part}" / "statistics" / "compression_compare.json"


def _calibrate_static_table(
    *,
    adapters,
    splits,
    calibration_split: str,
    unit_size: int,
    alphabet: str,
    batch_size: int,
    requested_bytes: int | None,
    context_units: int,
    min_context_count: int,
) -> Any:
    sources = _sources_for_split(splits, calibration_split)
    entries = _source_entries(splits)
    accumulator = StaticContextAccumulator(
        adapter_names=[adapter.name for adapter in adapters],
        context_units=context_units,
        min_context_count=min_context_count,
    )
    token_sizes = [adapter.token_size for adapter in adapters]
    for source_index, (entry, source) in enumerate(zip(entries, sources), start=1):
        species = str(entry["species"])
        source_name = str(entry.get("source_name", species))
        print(
            f"[calibrate] split={calibration_split} source={source_index}/{len(sources)}({source_name})",
            flush=True,
        )
        fusion_input = build_fusion_source_inputs(
            source=source,
            requested_bytes=requested_bytes,
            token_sizes=token_sizes,
            unit_size=unit_size,
            alphabet=alphabet,
        )
        if fusion_input.target_symbols.shape[0] == 0:
            continue
        model_probabilities = [
            adapter.unit_probabilities(
                species=species,
                core_sequence=fusion_input.core_sequence,
                unit_size=unit_size,
                batch_size=batch_size,
            ).probabilities
            for adapter in adapters
        ]
        accumulator.update(
            target_symbols=fusion_input.target_symbols,
            model_probabilities=model_probabilities,
        )
    return accumulator.finalize()


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
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flatten_value(row.get(key)) for key in fieldnames})


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    _write_csv(path, [{"metric": key, "value": value} for key, value in sorted(summary.items())])


def _dataset_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    species_rows = dataset.get("species")
    if not isinstance(species_rows, list):
        return []
    rows: list[dict[str, Any]] = []
    for row in species_rows:
        if not isinstance(row, dict):
            continue
        rows.append(
            {
                "species": row.get("species"),
                "source_name": row.get("source_name"),
                "source_mode": row.get("source_mode"),
                "selected_sequence_count": row.get("selected_sequence_count"),
                "sequence_keys": "|".join(row.get("sequence_keys", []))
                if isinstance(row.get("sequence_keys"), list)
                else row.get("sequence_keys"),
                "sequence_files": "|".join(row.get("sequence_files", []))
                if isinstance(row.get("sequence_files"), list)
                else row.get("sequence_files"),
                "total_size": row.get("total_size"),
                "train_bytes": row.get("train_bytes"),
                "val_bytes": row.get("val_bytes"),
                "test_bytes": row.get("test_bytes"),
            }
        )
    return rows


def _augment_aggregate(aggregate: dict[str, object], per_source: list[dict[str, object]], adapters) -> dict[str, object]:
    result = dict(aggregate)
    for adapter in adapters:
        key = f"model_{adapter.name}_theoretical_bits"
        total = sum(float(row.get(key, 0.0) or 0.0) for row in per_source)
        total_bases = sum(int(row.get("core_base_count", 0) or 0) for row in per_source)
        result[f"total_{key}"] = total
        result[f"total_model_{adapter.name}_theoretical_bits_per_core_base"] = total / max(total_bases, 1)
    choice_totals: dict[str, int] = {}
    for row in per_source:
        counts = row.get("fusion_model_choice_counts")
        if not isinstance(counts, dict):
            continue
        for model_name, count in counts.items():
            choice_totals[str(model_name)] = choice_totals.get(str(model_name), 0) + int(count)
    if choice_totals:
        result["fusion_model_choice_counts"] = choice_totals
    return result


def _run_split(
    *,
    adapters,
    split_name: str,
    splits,
    policies: list[str],
    unit_size: int,
    alphabet: str,
    batch_size: int,
    requested_bytes: int | None,
    arithmetic_frequency_total: int | None,
    arithmetic_target_uniform_mass: float,
    context_units: int,
    static_table,
) -> dict[str, object]:
    sources = _sources_for_split(splits, split_name)
    entries = _source_entries(splits)
    split_result: dict[str, object] = {}

    for policy in policies:
        per_source: list[dict[str, object]] = []
        for source_index, (entry, source) in enumerate(zip(entries, sources), start=1):
            species = str(entry["species"])
            source_name = str(entry.get("source_name", species))
            print(
                (
                    f"[compress] split={split_name} policy={policy} "
                    f"source={source_index}/{len(sources)}({source_name})"
                ),
                flush=True,
            )
            metrics = compress_fusion_source(
                species=species,
                source=source,
                adapters=adapters,
                unit_size=unit_size,
                alphabet=alphabet,
                batch_size=batch_size,
                requested_bytes=requested_bytes,
                policy=policy,
                arithmetic_frequency_total=arithmetic_frequency_total,
                arithmetic_target_uniform_mass=arithmetic_target_uniform_mass,
                context_units=context_units,
                static_table=static_table if policy == FUSION_STATIC_CONTEXT else None,
            )
            per_source.append({"species": species, "source_name": source_name, **metrics})

        aggregate = _augment_aggregate(summarize_per_source(per_source), per_source, adapters)
        split_result[policy] = {
            "aggregate": aggregate,
            "per_source": per_source,
        }

    return split_result


def _write_statistics_tables(output_json: Path, metrics: dict[str, object]) -> None:
    results = metrics.get("results")
    if not isinstance(results, dict):
        return
    aggregate_rows: list[dict[str, Any]] = []
    per_source_rows: list[dict[str, Any]] = []
    for split_name, split_payload in results.items():
        if not isinstance(split_payload, dict):
            continue
        for policy_name, policy_payload in split_payload.items():
            if not isinstance(policy_payload, dict):
                continue
            aggregate = policy_payload.get("aggregate")
            if isinstance(aggregate, dict):
                aggregate_rows.append({"split": split_name, "policy": policy_name, **aggregate})
            per_source = policy_payload.get("per_source")
            if isinstance(per_source, list):
                for row in per_source:
                    if isinstance(row, dict):
                        per_source_rows.append({"split": split_name, "policy": policy_name, **row})

    # Standard names match the other compression runners. The "mode" value is
    # the fusion policy name because fusion evaluates policies over one window mode.
    standard_aggregate_rows = [
        {"split": row.pop("split"), "mode": row.pop("policy"), **row}
        for row in [dict(item) for item in aggregate_rows]
    ]
    standard_per_source_rows = [
        {"split": row.pop("split"), "mode": row.pop("policy"), **row}
        for row in [dict(item) for item in per_source_rows]
    ]
    _write_csv(output_json.parent / "compression_aggregate_by_split_mode.csv", standard_aggregate_rows)
    _write_csv(output_json.parent / "compression_per_source_by_split_mode.csv", standard_per_source_rows)
    _write_csv(output_json.parent / "compression_per_source_legacy.csv", [])

    # Fusion-specific aliases are kept because they make the policy dimension explicit.
    _write_csv(output_json.parent / "fusion_aggregate_by_split_policy.csv", aggregate_rows)
    _write_csv(output_json.parent / "fusion_per_source_by_split_policy.csv", per_source_rows)

    dataset = metrics.get("dataset")
    _write_csv(output_json.parent / "dataset_splits.csv", _dataset_rows(dataset if isinstance(dataset, dict) else {}))

    summary: dict[str, Any] = {}
    fusion = metrics.get("fusion")
    if isinstance(fusion, dict):
        for key, value in fusion.items():
            if key != "static_context_table":
                summary[f"fusion.{key}"] = _flatten_value(value)
    for row in standard_aggregate_rows:
        prefix = f"compression_compare.{row.get('split')}.{row.get('mode')}"
        for key in (
            "total_theoretical_bits_per_base",
            "total_arithmetic_bits_per_base",
            "total_sample_bases",
            "total_emitted_arithmetic_symbol_count",
            "total_compression_bases_per_second",
        ):
            if key in row:
                summary[f"{prefix}.{key}"] = row[key]
    _write_summary_csv(output_json.parent / "summary_metrics.csv", summary)

    run_metadata = {
        "project": "dna-compress",
        "entity": "",
        "name": output_json.parent.parent.name if output_json.parent.name == "statistics" else output_json.parent.name,
        "run_dir": str(output_json.parent.parent if output_json.parent.name == "statistics" else output_json.parent),
        "has_resolved_config": False,
        "has_metrics_json": False,
        "has_compression_compare_json": True,
        "compression_json": str(output_json),
    }
    (output_json.parent / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run extensible DNA unit probability fusion compression.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable model spec kind:run_dir[:checkpoint]. kind is megabyte or dnagpt.",
    )
    parser.add_argument("--calibration-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--split", nargs="+", default=["test"], choices=["train", "val", "test", "all"])
    parser.add_argument("--compression-modes", nargs="+", default=[NON_OVERLAP_MODE], choices=[NON_OVERLAP_MODE])
    parser.add_argument(
        "--fusion-policy",
        nargs="+",
        default=[FUSION_STATIC_CONTEXT, "oracle_max"],
        choices=list(SUPPORTED_FUSION_POLICIES),
    )
    parser.add_argument("--fusion-unit-size", default="auto", help="auto or a positive integer.")
    parser.add_argument("--context-units", type=int, default=1)
    parser.add_argument("--min-context-count", type=int, default=1)
    parser.add_argument("--alphabet", default="ACGTN")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--output-json")
    parser.add_argument("--print-config", action="store_true")

    data_group = parser.add_argument_group("data overrides")
    data_group.add_argument("--dataset-dir")
    data_group.add_argument("--species", nargs="+")
    data_group.add_argument("--train-ratio", type=float)
    data_group.add_argument("--val-ratio", type=float)
    data_group.add_argument("--test-ratio", type=float)
    data_group.add_argument("--max-train-bytes", type=int)
    data_group.add_argument("--max-val-bytes", type=int)
    data_group.add_argument("--max-test-bytes", type=int)
    data_group.add_argument("--compression-sample-bytes", type=int)

    arithmetic_group = parser.add_argument_group("arithmetic")
    arithmetic_group.add_argument("--arithmetic-frequency-total", type=int)
    arithmetic_group.add_argument("--arithmetic-target-uniform-mass", type=float, default=0.01)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.context_units <= 0:
        raise ValueError("--context-units must be positive")
    if args.min_context_count <= 0:
        raise ValueError("--min-context-count must be positive")

    adapters = [
        build_adapter_from_spec(
            spec=model_spec,
            index=index,
            device_name=args.device,
            dtype_name=args.dtype,
        )
        for index, model_spec in enumerate(args.model)
    ]
    if len(adapters) < 2:
        raise ValueError("Fusion compression requires at least two --model entries.")

    token_sizes = [adapter.token_size for adapter in adapters]
    requested_unit_size: str | int = args.fusion_unit_size
    if requested_unit_size != "auto":
        requested_unit_size = int(requested_unit_size)
    unit_size = resolve_fusion_unit_size(token_sizes, requested_unit_size)
    alphabet = normalize_alphabet(args.alphabet)
    incompatible_alphabets = [
        f"{adapter.name}:{adapter.alphabet}"
        for adapter in adapters
        if normalize_alphabet(adapter.alphabet) != alphabet
    ]
    if incompatible_alphabets:
        raise ValueError(
            "All adapters must use the fusion alphabet. "
            f"fusion_alphabet={alphabet}, incompatible={incompatible_alphabets}"
        )
    base_config = _adapter_config(adapters[0])
    _apply_data_overrides(base_config, args)
    batch_size = args.eval_batch_size or int(base_config.train.eval_batch_size)
    requested_splits = _normalize_splits(args.split)
    output_json = _resolve_output_json(args, adapters)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    static_table_path = output_json.parent / "static_context_table.json"

    if args.print_config:
        print(
            json.dumps(
                {
                    "models": [
                        {
                            "name": adapter.name,
                            "token_size": adapter.token_size,
                            "alphabet": adapter.alphabet,
                        }
                        for adapter in adapters
                    ],
                    "unit_size": unit_size,
                    "alphabet": alphabet,
                    "data": base_config.data.__dict__,
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    print("[fusion] loading data splits...", flush=True)
    splits = load_splits(base_config.data, seq_length=base_config.model.seq_length)
    static_table = None
    if FUSION_STATIC_CONTEXT in args.fusion_policy:
        static_table = _calibrate_static_table(
            adapters=adapters,
            splits=splits,
            calibration_split=args.calibration_split,
            unit_size=unit_size,
            alphabet=alphabet,
            batch_size=batch_size,
            requested_bytes=base_config.data.compression_sample_bytes,
            context_units=args.context_units,
            min_context_count=args.min_context_count,
        )
        write_static_context_table(static_table_path, static_table)
        print(f"[fusion] saved static context table to {static_table_path}", flush=True)

    metrics: dict[str, object] = {
        "device": args.device,
        "fusion": {
            "unit_size": unit_size,
            "token_sizes": token_sizes,
            "alphabet": alphabet,
            "policies": args.fusion_policy,
            "context_units": args.context_units,
            "min_context_count": args.min_context_count,
            "calibration_split": args.calibration_split,
            "static_context_table": str(static_table_path) if static_table is not None else None,
        },
        "models": [
            {
                "name": adapter.name,
                "token_size": adapter.token_size,
                "alphabet": adapter.alphabet,
                "config": _adapter_config(adapter).to_dict(),
            }
            for adapter in adapters
        ],
        "dataset": splits.summary,
        "results": {},
    }

    for split_name in requested_splits:
        metrics["results"][split_name] = _run_split(
            adapters=adapters,
            split_name=split_name,
            splits=splits,
            policies=args.fusion_policy,
            unit_size=unit_size,
            alphabet=alphabet,
            batch_size=batch_size,
            requested_bytes=base_config.data.compression_sample_bytes,
            arithmetic_frequency_total=args.arithmetic_frequency_total,
            arithmetic_target_uniform_mass=args.arithmetic_target_uniform_mass,
            context_units=args.context_units,
            static_table=static_table,
        )

    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_statistics_tables(output_json, metrics)
    generated_curves = generate_artifacts_for_compression_compare(output_json)
    if generated_curves:
        print(f"[fusion] generated {len(generated_curves)} compression curve artifacts", flush=True)
    print(f"Saved fusion compression metrics to {output_json}")


if __name__ == "__main__":
    main()
