from __future__ import annotations

"""Probe model BPB over a continuous DNA region.

Examples:

    # Megabyte on all DNACorpus species. Each species is written under output-dir/<species>/.
    python scripts/run_dna_region_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
      --model megabyte:outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217:best \
      --model geco2 \
      --geco2-pseudo-window-bases 3072 \
      --checkpoint-tag best \
      --random-region \
      --region-bases 50000 \
      --device cuda:1 \
      --batch-size 32 \
      --output-dir outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217/region_bpb_probe_dnacorpus

    # DNACorpus GeCo2 per-base BPB. Default uses GeCo2 -e once per region and
    # stores compact per-base arrays under models/geco2*/bpb.npz. Pseudo windows
    # are only for statistics/plots; set them to the 3072-base model window.
    python scripts/run_dna_region_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
      --model-kind geco2 \
      --model geco2 \
      --geco2-bin GeCo2 \
      --geco2-pseudo-window-bases 3072 \
      --random-region \
      --region-bases 50000 \
      --device cuda:2 \
      --batch-size 32 \
      --output-dir outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217/region_bpb_probe_dnacorpus_geco2

    # OpenGenome2 FASTA subset with multiple models/sources.
    python scripts/run_dna_region_bpb_probe.py \
      --dataset opengenome2 \
      --input-dir /data/students/Liang_junnan/opengenome2_subset/fasta_test_subset_100mb_per_source \
      --source gtdb_v220 ncbi_eukaryotic_genomes \
      --model megabyte:outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217:best \
      --model geco2 \
      --geco2-pseudo-window-bases 3072 \
      --random-region \
      --region-bases 50000 \
      --device cuda:2 \
      --batch-size 32 \
      --output-dir outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217/region_bpb_probe_og2

    # Evo2 7B base local checkpoint on a short DNACorpus region.
    # 7B can run in the normal DNACompress .venv. Evo2 1B base requires
    # Transformer Engine / FP8; use scripts/run_evo2_1b_env_python.sh, which
    # points at the isolated .conda_evo2_1b environment and sets CUDA library
    # paths without touching the training environment.
    CUDA_VISIBLE_DEVICES=3 python scripts/run_dna_region_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species OrSa HoSa DaRe ScPo EsCo YeMi BuEb AgPh GaGa DrMe EnIn PlFa HePy AeCa HaHi AnCa WaMe \
      --model-kind evo2 \
      --evo2-local-path third_party/evo2_7b_base/evo2_7b_base.pt \
      --evo2-context-bases 3072 \
      --random-region \
      --seed 12345 \
      --region-bases 50000 \
      --device cuda:0 \
      --batch-size 1 \
      --output-dir outputs/evo2_7b_base_region_bpb_probe

    # Evo2 1B base local checkpoint. The directory form resolves to
    # third_party/evo2_1b_base/evo2_1b_base.pt.
    CUDA_VISIBLE_DEVICES=3 scripts/run_evo2_1b_env_python.sh scripts/run_dna_region_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species BuEb \
      --model evo2:third_party/evo2_1b_base:evo2_1b_base \
      --evo2-context-bases 3072 \
      --random-region \
      --seed 12345 \
      --region-bases 50000 \
      --device cuda:0 \
      --batch-size 32 \
      --output-dir outputs/evo2_1b_base_region_bpb_probe

    # Evo2 + Megabyte comparison on the same region.
    CUDA_VISIBLE_DEVICES=3 python scripts/run_dna_region_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species BuEb \
      --model evo2:third_party/evo2_7b_base/evo2_7b_base.pt:evo2_7b_base \
      --model megabyte:outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217:best \
      --geco2-pseudo-window-bases 3072 \
      --evo2-context-bases 3072 \
      --random-region \
      --seed 12345 \
      --region-bases 50000 \
      --device cuda:0 \
      --batch-size 1 \
      --output-dir outputs/evo2_megabyte_geco2_region_bpb_probe

    # Carbon-500M fns branch. Download once with:
    #   env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    #     HF_ENDPOINT=https://hf-mirror.com \
    #     huggingface-cli download HuggingFaceBio/Carbon-500M --revision fns \
    #       --local-dir third_party/Carbon-500M-fns --local-dir-use-symlinks False
    CUDA_VISIBLE_DEVICES=3 python scripts/run_dna_region_bpb_probe.py \
      --dataset dnacorpus \
      --dataset-dir datasets/DNACorpus \
      --species HoSa \
      --random-region \
      --seed 12345 \
      --region-bases 50000 \
      --model carbon:third_party/Carbon-500M-fns \
      --carbon-context-bases 49152 \
      --device cuda:0 \
      --batch-size 4 \
      --output-dir outputs/carbon_500m_region_bpb_probe

    # Combine Carbon with existing Evo2 / Megabyte / GeCo2 artifacts from the same region.
    python scripts/run_dna_region_bpb_probe.py \
      --plot-only \
      --result-dir outputs/carbon_500m_region_bpb_probe \
      --result-dir outputs/evo2_7b_base_region_bpb_probe \
      --result-dir outputs/evo2_1b_base_region_bpb_probe \
      --result-dir outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217/full_bpb_probe_dnacorpus \
      --combine-matching-regions \
      --output-dir outputs/carbon_evo2_megabyte_geco2_region_compare

    # For DNACorpus, GeCo2 uses previous per-species paper-mode levels by default.
    # Add --no-geco2-dnacorpus-paper-levels to force the uniform --geco2-level.

    # Redraw plots only from existing compact artifacts, without model forward or GeCo2 compression.
    python scripts/run_dna_region_bpb_probe.py \
      --plot-only \
      --result-dir outputs/dna_megabyte_large_20260616_144744_20260616_171309_20260617_140217/region_bpb_probe_og2 \
      --combine-matching-regions \
      --plot-individual-windows \
      --max-individual-window-plots 8 \
      --smooth-window-bases 512 \
      --model-window-smooth-bases 64

    # Fast modular workflow: compute first, then redraw/compose later. Re-run
    # another model into the same output directory to combine artifacts.
    python scripts/run_dna_region_bpb_probe.py ... --compute-only
    python scripts/run_dna_region_bpb_probe.py --plot-only --result-dir outputs/.../region_bpb_probe
"""

import argparse
import codecs
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from time import perf_counter
from typing import Any, Iterable, Iterator

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.config import ExperimentConfig, load_experiment_config
from dna_compress.data import _discover_fasta_files, _resolve_sequence_source_mode, _sequence_key_from_path
from dna_compress.experiment import autocast_context, resolve_device
from dna_compress.fusion_compression import (
    DNAGPTProbabilityAdapter,
    MegabyteProbabilityAdapter,
    ProbabilityAdapter,
    UnitProbabilityResult,
    build_adapter_from_spec,
    encode_unit_symbols,
)
from dna_compress.megadna_loader import (
    MEGADNA_BASE_TO_TOKEN,
    MEGADNA_EOS_ID,
    MEGADNA_PAD_ID,
    default_megadna_weight_path,
    load_megadna_model,
    wrap_megadna_for_target_aligned_logits,
)
from dna_compress.tokenization import normalize_alphabet
from scripts.plot_compression_curves import GECO2_PAPER_BASELINE_BY_SOURCE


DEFAULT_REGION_BASES = 49152
DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_EVO2_LOCAL_PATH = REPO_ROOT / "third_party" / "evo2_7b_base" / "evo2_7b_base.pt"
DEFAULT_EVO2_1B_LOCAL_PATH = REPO_ROOT / "third_party" / "evo2_1b_base" / "evo2_1b_base.pt"
DEFAULT_CARBON_LOCAL_PATH = REPO_ROOT / "third_party" / "Carbon-500M-fns"
DEFAULT_CARBON_MODEL_NAME = "Carbon-500M"
DEFAULT_CARBON_REVISION = "fns"
COMPACT_METADATA_DROP_KEYS = {
    "geco2_full_stdout_tail",
    "geco2_full_stderr_tail",
    "geco2_prefix_results",
    "geco2_full_command",
}


def _open_binary(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _translate_table(alphabet: str) -> bytes:
    allowed = {ord(base) for base in normalize_alphabet(alphabet)}
    allowed.update(ord(base.lower()) for base in normalize_alphabet(alphabet))
    return bytes(byte for byte in range(256) if byte not in allowed)


def _iter_filtered_chunks(path: Path, *, alphabet: str, fasta: bool, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> Iterator[bytes]:
    delete = _translate_table(alphabet)
    if fasta:
        with _open_binary(path) as handle:
            for line in handle:
                content = line.rstrip(b"\r\n")
                if not content or content.startswith(b">"):
                    continue
                cleaned = content.upper().translate(None, delete)
                if cleaned:
                    yield cleaned
        return

    with _open_binary(path) as handle:
        while True:
            payload = handle.read(chunk_bytes)
            if not payload:
                break
            cleaned = payload.upper().translate(None, delete)
            if cleaned:
                yield cleaned


def filtered_length(paths: Iterable[Path], *, alphabet: str, fasta: bool) -> int:
    return sum(len(chunk) for path in paths for chunk in _iter_filtered_chunks(path, alphabet=alphabet, fasta=fasta))


def extract_filtered_region(
    paths: Iterable[Path],
    *,
    alphabet: str,
    fasta: bool,
    start: int,
    length: int,
) -> bytes:
    if start < 0:
        raise ValueError("--region-start must be non-negative")
    if length <= 0:
        return b""

    seen = 0
    remaining = int(length)
    out = bytearray()
    for path in paths:
        for chunk in _iter_filtered_chunks(path, alphabet=alphabet, fasta=fasta):
            chunk_len = len(chunk)
            if seen + chunk_len <= start:
                seen += chunk_len
                continue
            local_start = max(start - seen, 0)
            take = min(chunk_len - local_start, remaining)
            if take > 0:
                out.extend(chunk[local_start : local_start + take])
                remaining -= take
                if remaining <= 0:
                    return bytes(out)
            seen += chunk_len
    return bytes(out)


def _read_manifest(input_dir: Path) -> dict[str, Any] | None:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _safe_source_filename(source: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in source)
    return f"{safe}.fasta"


def _safe_label(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe.strip("_") or "source"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in COMPACT_METADATA_DROP_KEYS:
            continue
        if isinstance(value, str) and len(value) > 500:
            compact[key] = value[:500] + "...<truncated>"
        elif isinstance(value, list) and len(value) > 64:
            compact[key] = value[:64]
            compact[f"{key}_truncated_count"] = len(value) - 64
        else:
            compact[key] = value
    return _json_safe(compact)


def _record_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.dtype(np.float16)
    if dtype_name == "float32":
        return np.dtype(np.float32)
    if dtype_name == "float64":
        return np.dtype(np.float64)
    raise ValueError("--record-dtype must be one of: float16, float32, float64")


def _offset_dtype(max_offset: int) -> np.dtype:
    return np.dtype(np.uint32 if max_offset <= np.iinfo(np.uint32).max else np.uint64)


def _opengenome2_sources(input_dir: Path) -> dict[str, Path]:
    manifest = _read_manifest(input_dir)
    if manifest is not None and isinstance(manifest.get("sources"), dict):
        sources: dict[str, Path] = {}
        for source, payload in manifest["sources"].items():
            entry = payload if isinstance(payload, dict) else {}
            output_path = entry.get("output_path")
            path = Path(output_path) if isinstance(output_path, str) else input_dir / _safe_source_filename(str(source))
            if not path.exists():
                path = input_dir / _safe_source_filename(str(source))
            sources[str(source)] = path
        return sources
    return {path.stem: path for path in sorted(input_dir.glob("*.fasta"))}


def _parse_sequence_include(values: list[str] | None, species: str | None) -> set[str] | None:
    if not values:
        return None
    selected: set[str] = set()
    for item in values:
        if "=" in item:
            item_species, raw_keys = item.split("=", 1)
            if species is not None and item_species.strip() != species:
                continue
        else:
            raw_keys = item
        selected.update(key.strip() for key in raw_keys.split(",") if key.strip())
    return selected or None


def _resolve_single_region_source(
    args: argparse.Namespace,
    alphabet: str,
    *,
    species_name: str | None = None,
    source_name: str | None = None,
) -> dict[str, Any]:
    if args.dataset == "opengenome2":
        input_dir = Path(args.input_dir or args.dataset_dir)
        sources = _opengenome2_sources(input_dir)
        if source_name is None:
            if len(sources) != 1:
                raise ValueError("--source is required for OpenGenome2 when the input directory contains multiple sources")
            resolved_source_name, path = next(iter(sources.items()))
        else:
            if source_name not in sources:
                available = ", ".join(sorted(sources)[:20])
                raise ValueError(f"OpenGenome2 source {source_name!r} not found. Available: {available}")
            resolved_source_name = source_name
            path = sources[source_name]
        return {
            "dataset": "opengenome2",
            "source": resolved_source_name,
            "species": None,
            "paths": [path],
            "fasta": True,
            "alphabet": alphabet,
        }

    dataset_dir = Path(args.dataset_dir)
    if species_name is None:
        raise ValueError("--species is required for DNACorpus")
    species_path = dataset_dir / species_name
    config = ExperimentConfig().data
    config.dataset_dir = str(dataset_dir)
    config.sequence_source_mode = args.sequence_source_mode
    config.multi_sequence_mode = args.multi_sequence_mode
    mode = _resolve_sequence_source_mode(species_path, config)
    if mode == "flat_file":
        return {
            "dataset": "dnacorpus",
            "source": source_name or species_name,
            "species": species_name,
            "paths": [species_path],
            "fasta": False,
            "alphabet": alphabet,
        }

    fasta_files = _discover_fasta_files(species_path)
    selected = _parse_sequence_include(args.sequence_include, species_name)
    keyed = [(path, _sequence_key_from_path(path)) for path in fasta_files]
    if selected is not None:
        keyed = [(path, key) for path, key in keyed if key in selected]
    if args.multi_sequence_mode == "concat":
        if not keyed:
            raise ValueError(f"No DNACorpus FASTA sequences selected for {species_name}")
        return {
            "dataset": "dnacorpus",
            "source": source_name or species_name,
            "species": species_name,
            "paths": [path for path, _ in keyed],
            "fasta": True,
            "alphabet": alphabet,
            "sequence_keys": [key for _, key in keyed],
        }

    if source_name is not None:
        matches = [
            (path, key)
            for path, key in keyed
            if source_name in {key, path.stem, path.name, f"{species_name}:{key}"}
        ]
    else:
        matches = keyed
    if len(matches) != 1:
        candidates = ", ".join(key for _, key in keyed[:20])
        raise ValueError(
            "DNACorpus FASTA separate mode needs exactly one --source or --sequence-include match. "
            f"Candidates: {candidates}"
        )
    path, key = matches[0]
    return {
        "dataset": "dnacorpus",
        "source": f"{species_name}:{key}",
        "species": species_name,
        "paths": [path],
        "fasta": True,
        "alphabet": source_info["alphabet"],
        "sequence_keys": [key],
    }


def resolve_region_sources(args: argparse.Namespace, alphabet: str) -> list[dict[str, Any]]:
    if args.dataset == "opengenome2":
        sources = list(args.source or [])
        if not sources:
            return [_resolve_single_region_source(args, alphabet)]
        return [_resolve_single_region_source(args, alphabet, source_name=source) for source in sources]

    species_values = list(args.species or [])
    if not species_values:
        raise ValueError("--species is required for DNACorpus")
    source_values = list(args.source or [])
    if source_values and len(species_values) > 1:
        if len(source_values) != len(species_values):
            raise ValueError("When multiple DNACorpus --species and --source are both provided, their counts must match.")
        return [
            _resolve_single_region_source(args, alphabet, species_name=species, source_name=source)
            for species, source in zip(species_values, source_values)
        ]
    source_name = source_values[0] if source_values else None
    return [
        _resolve_single_region_source(args, alphabet, species_name=species, source_name=source_name)
        for species in species_values
    ]


def stable_random_start(*, source_name: str, source_length: int, region_bases: int, seed: int) -> int:
    if source_length <= region_bases:
        return 0
    digest = hashlib.sha256(f"{seed}:{source_name}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "little", signed=False)
    return value % (source_length - region_bases + 1)


def region_identity(
    *,
    dataset: Any,
    source: Any,
    species: Any,
    alphabet: Any,
    region_start: int,
    region_bases: int,
) -> dict[str, Any]:
    payload = {
        "dataset": str(dataset) if dataset is not None else None,
        "source": str(source) if source is not None else None,
        "species": str(species) if species is not None else None,
        "alphabet": str(alphabet) if alphabet is not None else None,
        "region_start": int(region_start),
        "region_bases": int(region_bases),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload["signature"] = hashlib.sha256(encoded).hexdigest()[:20]
    return payload


def result_region_identity(result: dict[str, Any]) -> dict[str, Any]:
    existing = result.get("region_identity")
    if isinstance(existing, dict) and existing.get("signature"):
        return dict(existing)
    return region_identity(
        dataset=result.get("dataset"),
        source=result.get("source"),
        species=result.get("species"),
        alphabet=result.get("alphabet"),
        region_start=int(result.get("region_start", 0)),
        region_bases=int(result.get("region_bases", 0)),
    )


def _checkpoint_path(run_dir: Path, checkpoint: str | None, checkpoint_tag: str) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / path
        return path
    return run_dir / f"{checkpoint_tag}.pt"


class MegaDNARegionAdapter(ProbabilityAdapter):
    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        seq_length: int,
        device: torch.device,
        dtype_name: str,
        checkpoint_path: Path | None,
        base_weight: Path | None,
    ) -> None:
        self.name = name
        self.model = model
        self.seq_length = int(seq_length)
        self.device = device
        self.dtype_name = dtype_name
        self.checkpoint_path = checkpoint_path
        self.base_weight = base_weight
        self.token_size = 1
        self.alphabet = "ATCG"

    @classmethod
    def from_args(
        cls,
        *,
        name: str,
        run_dir: Path | None,
        checkpoint_path: Path | None,
        base_weight: Path | None,
        device: torch.device,
        dtype_name: str,
        seq_length: int | None,
    ) -> "MegaDNARegionAdapter":
        config = ExperimentConfig()
        if run_dir is not None and (run_dir / "resolved_config.json").exists():
            config = load_experiment_config(run_dir / "resolved_config.json")
        resolved_seq_length = int(seq_length or config.model.seq_length or 1024)
        resolved_weight = base_weight or Path(config.model.pretrained_weight_path or default_megadna_weight_path())
        raw_model = load_megadna_model(resolved_weight, device=device)
        if checkpoint_path is not None and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                raw_model.load_state_dict(checkpoint["model_state"], strict=True)
            elif isinstance(checkpoint, torch.nn.Module):
                raw_model = checkpoint.to(device)
            else:
                raise ValueError(f"Unsupported megaDNA checkpoint format: {checkpoint_path}")
        model = wrap_megadna_for_target_aligned_logits(raw_model).to(device)
        model.eval()
        return cls(
            name=name,
            model=model,
            seq_length=resolved_seq_length,
            device=device,
            dtype_name=dtype_name,
            checkpoint_path=checkpoint_path,
            base_weight=resolved_weight,
        )

    def unit_probabilities(
        self,
        *,
        species: str,
        core_sequence: str,
        unit_size: int,
        batch_size: int,
    ) -> UnitProbabilityResult:
        del species
        if unit_size != 1:
            raise ValueError("MegaDNA region adapter only supports unit_size=1.")
        token_ids = [MEGADNA_BASE_TO_TOKEN[base] for base in core_sequence]
        if not token_ids:
            raise ValueError("MegaDNA adapter requires at least one A/T/C/G base.")
        pad_id = MEGADNA_PAD_ID
        all_rows: list[np.ndarray] = []
        model_forward_seconds = 0.0
        softmax_seconds = 0.0
        aggregate_seconds = 0.0
        data_transfer_seconds = 0.0

        self.model.eval()
        with torch.no_grad():
            starts = list(range(0, len(token_ids), self.seq_length))
            for batch_start in range(0, len(starts), batch_size):
                batch_starts = starts[batch_start : batch_start + batch_size]
                windows = torch.full((len(batch_starts), self.seq_length), pad_id, dtype=torch.long)
                lengths: list[int] = []
                for row_index, start in enumerate(batch_starts):
                    chunk = token_ids[start : start + self.seq_length]
                    lengths.append(len(chunk))
                    windows[row_index, : len(chunk)] = torch.as_tensor(chunk, dtype=torch.long)

                transfer_started = perf_counter()
                batch = windows.to(self.device, non_blocking=True)
                data_transfer_seconds += perf_counter() - transfer_started

                with autocast_context(self.device, self.dtype_name):
                    forward_started = perf_counter()
                    output = self.model(batch, return_loss=False)
                    model_forward_seconds += perf_counter() - forward_started

                    softmax_started = perf_counter()
                    log_probs = torch.log_softmax(output.lm_logits, dim=-1)
                    softmax_seconds += perf_counter() - softmax_started

                aggregate_started = perf_counter()
                for row_index, length in enumerate(lengths):
                    if length <= 0:
                        continue
                    rows = log_probs[row_index, :length, [1, 3, 4, 2]].float().exp().cpu().numpy()
                    rows = rows / rows.sum(axis=1, keepdims=True).clip(min=1e-300)
                    all_rows.append(rows)
                aggregate_seconds += perf_counter() - aggregate_started

        probabilities = np.concatenate(all_rows, axis=0) if all_rows else np.zeros((0, 4), dtype=np.float64)
        return UnitProbabilityResult(
            adapter_name=self.name,
            probabilities=probabilities,
            model_forward_seconds=model_forward_seconds,
            softmax_seconds=softmax_seconds,
            aggregate_seconds=aggregate_seconds,
            data_transfer_seconds=data_transfer_seconds,
        )


class Evo2RegionAdapter:
    def __init__(
        self,
        *,
        name: str,
        evo2_model: Any,
        tokenizer: Any,
        local_path: Path,
        model_name: str,
        context_bases: int,
        device: torch.device,
        requested_device: str,
        dtype_name: str,
        use_kernels: bool,
    ) -> None:
        self.name = name
        self.evo2_model = evo2_model
        self.tokenizer = tokenizer
        self.local_path = local_path
        self.model_name = model_name
        self.context_bases = int(context_bases)
        self.device = device
        self.requested_device = requested_device
        self.dtype_name = dtype_name
        self.use_kernels = bool(use_kernels)
        self.token_size = 1
        self.alphabet = "ACGTN"

    @classmethod
    def from_args(
        cls,
        *,
        name: str,
        local_path: Path,
        model_name: str,
        context_bases: int,
        requested_device: str,
        dtype_name: str,
        use_kernels: bool,
    ) -> "Evo2RegionAdapter":
        if not local_path.exists():
            raise FileNotFoundError(f"Evo2 local checkpoint not found: {local_path}")
        try:
            from evo2 import Evo2
        except ImportError as exc:
            raise ImportError(
                "Evo2 is not installed or cannot import in this Python environment. "
                "For evo2_1b_base use scripts/run_evo2_1b_env_python.sh; it uses the "
                "isolated .conda_evo2_1b environment with Transformer Engine, "
                "flash-attn, pyarrow, matplotlib, and pandas installed."
            ) from exc
        torch.serialization.add_safe_globals([codecs.encode])
        # Vortex initializes Evo2 modules on process-local cuda:0 when CUDA is
        # available, so CUDA_VISIBLE_DEVICES should be used to pick the physical GPU.
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            evo2_model = Evo2(model_name, local_path=str(local_path), use_kernels=use_kernels)
        except ImportError as exc:
            raise ImportError(
                f"Failed to initialize {model_name!r}. Evo2 1B requires Transformer "
                "Engine / FP8 and usually will not run in the normal DNACompress .venv. "
                "Use scripts/run_evo2_1b_env_python.sh to run this probe in the "
                "isolated .conda_evo2_1b environment."
            ) from exc
        return cls(
            name=name,
            evo2_model=evo2_model,
            tokenizer=evo2_model.tokenizer,
            local_path=local_path,
            model_name=model_name,
            context_bases=context_bases,
            device=device,
            requested_device=requested_device,
            dtype_name=dtype_name,
            use_kernels=use_kernels,
        )

    def _filter_sequence(self, region_sequence: str, region_offsets: np.ndarray) -> tuple[str, np.ndarray, int]:
        allowed = set(self.alphabet)
        keep_mask = np.asarray([base.upper() in allowed for base in region_sequence], dtype=bool)
        kept = "".join(base.upper() for base, keep in zip(region_sequence, keep_mask) if keep)
        return kept, region_offsets[keep_mask], int(len(region_sequence) - int(keep_mask.sum()))

    @staticmethod
    def _extract_logits(outputs: Any) -> torch.Tensor:
        logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if not torch.is_tensor(logits):
            raise TypeError(f"Evo2 forward returned unsupported logits type: {type(logits)!r}")
        return logits

    def region_bpb(
        self,
        *,
        region_sequence: str,
        region_offsets: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        model_sequence, model_offsets, filtered_out = self._filter_sequence(region_sequence, region_offsets)
        if len(model_sequence) < 2:
            raise ValueError("Evo2 adapter requires at least two usable A/C/G/T/N bases.")
        context_bases = max(2, int(self.context_bases))
        starts = list(range(0, len(model_sequence), context_bases))
        bpb_chunks: list[np.ndarray] = []
        offset_chunks: list[np.ndarray] = []
        dropped_context_bases = 0
        model_forward_seconds = 0.0
        softmax_seconds = 0.0
        aggregate_seconds = 0.0
        data_transfer_seconds = 0.0

        with torch.inference_mode():
            for batch_start in range(0, len(starts), int(batch_size)):
                batch_starts = starts[batch_start : batch_start + int(batch_size)]
                chunks = [model_sequence[start : start + context_bases] for start in batch_starts]
                lengths = [len(chunk) for chunk in chunks]
                max_length = max(lengths)
                input_ids = torch.full((len(chunks), max_length), int(self.tokenizer.pad_id), dtype=torch.long)
                for row_index, chunk in enumerate(chunks):
                    token_ids = self.tokenizer.tokenize(chunk)
                    input_ids[row_index, : len(token_ids)] = torch.as_tensor(token_ids, dtype=torch.long)

                transfer_started = perf_counter()
                batch = input_ids.to(self.device, non_blocking=True)
                data_transfer_seconds += perf_counter() - transfer_started

                with autocast_context(self.device, self.dtype_name):
                    forward_started = perf_counter()
                    outputs, _ = self.evo2_model(batch)
                    logits = self._extract_logits(outputs)
                    model_forward_seconds += perf_counter() - forward_started

                    softmax_started = perf_counter()
                    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                    softmax_seconds += perf_counter() - softmax_started

                aggregate_started = perf_counter()
                targets = batch[:, 1:]
                gathered = torch.gather(log_probs, dim=2, index=targets.unsqueeze(-1)).squeeze(-1)
                gathered_cpu = gathered.float().cpu().numpy()
                aggregate_seconds += perf_counter() - aggregate_started

                for row_index, (start, length) in enumerate(zip(batch_starts, lengths)):
                    if length <= 1:
                        dropped_context_bases += length
                        continue
                    dropped_context_bases += 1
                    scored_log_probs = gathered_cpu[row_index, : length - 1]
                    bpb_chunks.append((-scored_log_probs / math.log(2.0)).astype(np.float64, copy=False))
                    offset_chunks.append(model_offsets[start + 1 : start + length])

        bpb = np.concatenate(bpb_chunks) if bpb_chunks else np.zeros((0,), dtype=np.float64)
        offsets = np.concatenate(offset_chunks) if offset_chunks else np.zeros((0,), dtype=np.int64)
        metadata = {
            "adapter_name": self.name,
            "adapter_class": type(self).__name__,
            "alphabet": self.alphabet,
            "token_size": 1,
            "valid_base_count": int(bpb.shape[0]),
            "filtered_out_bases": int(filtered_out),
            "trimmed_tail_bases": 0,
            "dropped_context_bases": int(dropped_context_bases),
            "model_window_bases": int(context_bases),
            "model_forward_seconds": model_forward_seconds,
            "softmax_seconds": softmax_seconds,
            "factorization_seconds": aggregate_seconds,
            "data_transfer_seconds": data_transfer_seconds,
            "evo2_model_name": self.model_name,
            "evo2_local_path": str(self.local_path),
            "evo2_context_bases": int(context_bases),
            "evo2_requested_device": self.requested_device,
            "evo2_runtime_device": str(self.device),
            "evo2_dtype": self.dtype_name,
            "evo2_use_kernels": self.use_kernels,
            "evo2_alignment": "log_softmax(logits[:, :-1]) gathered at input_ids[:, 1:]",
        }
        return bpb, offsets, metadata


class CarbonRegionAdapter:
    def __init__(
        self,
        *,
        name: str,
        model: Any,
        tokenizer: Any,
        local_path: Path,
        model_name: str,
        revision: str,
        context_bases: int,
        device: torch.device,
        dtype_name: str,
        trust_remote_code: bool,
    ) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.local_path = local_path
        self.model_name = model_name
        self.revision = revision
        self.context_bases = int(context_bases)
        self.device = device
        self.dtype_name = dtype_name
        self.trust_remote_code = bool(trust_remote_code)
        self.token_size = 1
        self.alphabet = "ACGTN"

    @classmethod
    def from_args(
        cls,
        *,
        name: str,
        local_path: Path,
        model_name: str,
        revision: str,
        context_bases: int,
        device: torch.device,
        dtype_name: str,
        trust_remote_code: bool,
    ) -> "CarbonRegionAdapter":
        if not local_path.exists():
            raise FileNotFoundError(
                f"Carbon local model directory not found: {local_path}. "
                "Download it with HF_ENDPOINT=https://hf-mirror.com huggingface-cli download "
                "HuggingFaceBio/Carbon-500M --revision fns --local-dir third_party/Carbon-500M-fns."
            )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Carbon adapter requires transformers to be installed.") from exc

        torch_dtype = getattr(torch, dtype_name, None)
        if not isinstance(torch_dtype, torch.dtype):
            torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            str(local_path),
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(local_path),
            revision=revision,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        ).to(device)
        model.eval()
        if not hasattr(model, "score_sequence"):
            raise TypeError(
                "Loaded Carbon model does not expose score_sequence(); use the HuggingFaceBio/Carbon-500M "
                "fns revision/local directory."
            )
        if hasattr(model, "setup_tokenizer"):
            model.setup_tokenizer(tokenizer)
        elif not hasattr(model, "tokenizer"):
            model.tokenizer = tokenizer

        return cls(
            name=name,
            model=model,
            tokenizer=tokenizer,
            local_path=local_path,
            model_name=model_name,
            revision=revision,
            context_bases=context_bases,
            device=device,
            dtype_name=dtype_name,
            trust_remote_code=trust_remote_code,
        )

    def _filter_sequence(self, region_sequence: str, region_offsets: np.ndarray) -> tuple[str, np.ndarray, int]:
        allowed = set(self.alphabet)
        keep_mask = np.asarray([base.upper() in allowed for base in region_sequence], dtype=bool)
        kept = "".join(base.upper() for base, keep in zip(region_sequence, keep_mask) if keep)
        return kept, region_offsets[keep_mask], int(len(region_sequence) - int(keep_mask.sum()))

    @staticmethod
    def _as_probability_list(actual_probs: Any) -> list[torch.Tensor]:
        if torch.is_tensor(actual_probs):
            return [actual_probs]
        if isinstance(actual_probs, (tuple, list)):
            return list(actual_probs)
        raise TypeError(f"Carbon score_sequence returned unsupported actual_probs type: {type(actual_probs)!r}")

    def region_bpb(
        self,
        *,
        region_sequence: str,
        region_offsets: np.ndarray,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        model_sequence, model_offsets, filtered_out = self._filter_sequence(region_sequence, region_offsets)
        if not model_sequence:
            raise ValueError("Carbon adapter requires at least one usable A/C/G/T/N base.")

        context_bases = max(1, int(self.context_bases))
        starts = list(range(0, len(model_sequence), context_bases))
        bpb_chunks: list[np.ndarray] = []
        offset_chunks: list[np.ndarray] = []
        score_sequence_seconds = 0.0
        aggregate_seconds = 0.0

        with torch.inference_mode():
            for batch_start in range(0, len(starts), int(batch_size)):
                batch_starts = starts[batch_start : batch_start + int(batch_size)]
                chunks = [model_sequence[start : start + context_bases] for start in batch_starts]

                score_started = perf_counter()
                _bp_probs, actual_probs = self.model.score_sequence(chunks)
                score_sequence_seconds += perf_counter() - score_started

                aggregate_started = perf_counter()
                probability_rows = self._as_probability_list(actual_probs)
                if len(probability_rows) != len(chunks):
                    raise ValueError(
                        f"Carbon score_sequence returned {len(probability_rows)} probability rows for {len(chunks)} chunks."
                    )
                for start, chunk, probs in zip(batch_starts, chunks, probability_rows):
                    prob_tensor = probs.detach().float().clamp_min(1e-12)
                    values = (-torch.log2(prob_tensor)).cpu().numpy().astype(np.float64, copy=False)
                    length = len(chunk)
                    bpb_chunks.append(values[:length])
                    offset_chunks.append(model_offsets[start : start + length])
                aggregate_seconds += perf_counter() - aggregate_started

        bpb = np.concatenate(bpb_chunks) if bpb_chunks else np.zeros((0,), dtype=np.float64)
        offsets = np.concatenate(offset_chunks) if offset_chunks else np.zeros((0,), dtype=np.int64)
        metadata = {
            "adapter_name": self.name,
            "adapter_class": type(self).__name__,
            "alphabet": self.alphabet,
            "token_size": 1,
            "valid_base_count": int(bpb.shape[0]),
            "filtered_out_bases": int(filtered_out),
            "trimmed_tail_bases": 0,
            "dropped_context_bases": 0,
            "model_window_bases": int(context_bases),
            "model_forward_seconds": score_sequence_seconds,
            "score_sequence_seconds": score_sequence_seconds,
            "softmax_seconds": 0.0,
            "factorization_seconds": aggregate_seconds,
            "data_transfer_seconds": 0.0,
            "carbon_model_name": self.model_name,
            "carbon_revision": self.revision,
            "carbon_local_path": str(self.local_path),
            "carbon_context_bases": int(context_bases),
            "carbon_runtime_device": str(self.device),
            "carbon_dtype": self.dtype_name,
            "carbon_trust_remote_code": self.trust_remote_code,
            "carbon_tokenizer_k": int(getattr(self.tokenizer, "k", getattr(self.model, "k", 0)) or 0),
            "carbon_alignment": "official fns score_sequence actual_probs converted with -log2(clamp(prob, 1e-12))",
            "carbon_n_behavior": "official score_sequence uses max per-base probability for N.",
        }
        return bpb, offsets, metadata


def _tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _resolve_geco2_binary(requested_binary: str) -> str:
    resolved = shutil.which(requested_binary)
    if resolved is None:
        raise FileNotFoundError(f"Could not find GeCo2 binary: {requested_binary}")
    return resolved


class Geco2RegionAdapter:
    def __init__(
        self,
        *,
        name: str,
        binary: str,
        level: int,
        pseudo_window_bases: int | None,
        profile_mode: str,
        use_dnacorpus_paper_levels: bool,
        temp_root: Path | None,
        keep_temp: bool,
    ) -> None:
        self.name = name
        self.binary = _resolve_geco2_binary(binary)
        self.default_level = int(level)
        self.pseudo_window_bases = pseudo_window_bases
        self.profile_mode = profile_mode
        self.use_dnacorpus_paper_levels = bool(use_dnacorpus_paper_levels)
        self.temp_root = temp_root
        self.keep_temp = bool(keep_temp)
        self.token_size = 1
        self.alphabet = "ACGTN"

    def _level_for_source(self, *, dataset: str | None, species: str | None, source: str | None) -> tuple[int, str]:
        if self.use_dnacorpus_paper_levels and dataset == "dnacorpus":
            keys = [key for key in (source, species) if key]
            if source and ":" in source:
                keys.append(source.split(":", 1)[0])
            for key in keys:
                baseline = GECO2_PAPER_BASELINE_BY_SOURCE.get(str(key))
                if baseline is not None:
                    return int(baseline["mode"]), f"dnacorpus_paper:{key}"
        return self.default_level, "configured"

    def _command(self, input_path: Path, *, level: int, estimate: bool) -> list[str]:
        command = [self.binary, "-F", "-v"]
        if estimate:
            command.append("-e")
        command.extend(["-l", str(level), str(input_path)])
        return command

    def _compress_payload(self, payload: bytes, *, label: str, level: int, estimate: bool) -> dict[str, Any]:
        if self.keep_temp:
            temp_dir = Path(tempfile.mkdtemp(prefix="geco2_region_", dir=str(self.temp_root) if self.temp_root else None))
            cleanup_context = None
        else:
            cleanup_context = tempfile.TemporaryDirectory(prefix="geco2_region_", dir=str(self.temp_root) if self.temp_root else None)
            temp_dir = Path(cleanup_context.__enter__())
        try:
            input_path = temp_dir / f"{_safe_label(label)}.seq"
            input_path.write_bytes(payload)
            command = self._command(input_path, level=level, estimate=estimate)
            started = perf_counter()
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            seconds = perf_counter() - started
            compressed_path = Path(str(input_path) + ".co")
            if completed.returncode != 0:
                raise RuntimeError(
                    "GeCo2 failed with return code "
                    f"{completed.returncode}: {_tail_text(completed.stderr or completed.stdout)}"
                )
            if not compressed_path.exists():
                candidates = sorted(input_path.parent.glob("*.co"))
                if len(candidates) == 1:
                    compressed_path = candidates[0]
                else:
                    raise FileNotFoundError(f"GeCo2 did not create expected output: {compressed_path}")
            result = {
                "compressed_bytes": int(compressed_path.stat().st_size),
                "seconds": seconds,
                "command": command,
                "returncode": int(completed.returncode),
                "stdout_tail": _tail_text(completed.stdout),
                "stderr_tail": _tail_text(completed.stderr),
                "input_path": str(input_path),
                "compressed_path": str(compressed_path),
                "temp_dir": str(temp_dir),
            }
            estimate_path = Path(str(input_path) + ".iae")
            if estimate:
                if not estimate_path.exists():
                    raise FileNotFoundError(f"GeCo2 did not create expected estimate file: {estimate_path}")
                result["estimate_path"] = str(estimate_path)
                with estimate_path.open("r", encoding="utf-8") as handle:
                    values = [float(line.strip() or "0") for line in handle]
                result["estimate_bpb"] = np.asarray(values, dtype=np.float64)
            return result
        finally:
            if cleanup_context is not None:
                cleanup_context.__exit__(None, None, None)

    def region_bpb(
        self,
        *,
        region_sequence: str,
        region_offsets: np.ndarray,
        pseudo_window_bases: int,
        dataset: str | None,
        species: str | None,
        source: str | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        del region_offsets
        payload = region_sequence.encode("ascii")
        if not payload:
            raise ValueError("GeCo2 adapter requires at least one base in the selected region.")
        total_started = perf_counter()
        level, level_source = self._level_for_source(dataset=dataset, species=species, source=source)
        full_result = self._compress_payload(payload, label=f"{self.name}_full", level=level, estimate=self.profile_mode == "estimate")
        compression_seconds = float(full_result["seconds"])
        compressed_bytes = int(full_result["compressed_bytes"])
        n = len(payload)
        if self.profile_mode == "estimate":
            bpb = np.asarray(full_result["estimate_bpb"], dtype=np.float64)
            if bpb.shape[0] != n:
                raise RuntimeError(f"GeCo2 .iae row count {bpb.shape[0]} does not match region bases {n}.")
            prefix_results: list[dict[str, Any]] = []
        elif self.profile_mode == "constant":
            bpb = np.full((n,), compressed_bytes * 8.0 / max(n, 1), dtype=np.float64)
            prefix_results: list[dict[str, Any]] = []
        else:
            step = max(1, int(pseudo_window_bases))
            endpoints = list(range(step, n, step))
            endpoints.append(n)
            previous_size = 0
            previous_endpoint = 0
            values = np.zeros((n,), dtype=np.float64)
            prefix_results = []
            for endpoint in endpoints:
                if endpoint == n:
                    result = full_result
                else:
                    result = self._compress_payload(payload[:endpoint], label=f"{self.name}_prefix_{endpoint}", level=level, estimate=False)
                    compression_seconds += float(result["seconds"])
                size = int(result["compressed_bytes"])
                delta_bytes = size - previous_size
                span = endpoint - previous_endpoint
                values[previous_endpoint:endpoint] = delta_bytes * 8.0 / max(span, 1)
                prefix_results.append(
                    {
                        "endpoint_bases": int(endpoint),
                        "compressed_bytes": size,
                        "delta_bytes": int(delta_bytes),
                        "span_bases": int(span),
                        "seconds": float(result["seconds"]),
                    }
                )
                previous_size = size
                previous_endpoint = endpoint
            bpb = values
        metadata = {
            "adapter_name": self.name,
            "adapter_class": type(self).__name__,
            "alphabet": self.alphabet,
            "token_size": 1,
            "valid_base_count": int(n),
            "filtered_out_bases": 0,
            "trimmed_tail_bases": 0,
            "model_window_bases": int(pseudo_window_bases),
            "model_forward_seconds": 0.0,
            "softmax_seconds": 0.0,
            "factorization_seconds": 0.0,
            "data_transfer_seconds": 0.0,
            "geco2_binary": self.binary,
            "geco2_level": level,
            "geco2_level_source": level_source,
            "geco2_default_level": self.default_level,
            "geco2_use_dnacorpus_paper_levels": self.use_dnacorpus_paper_levels,
            "geco2_profile_mode": self.profile_mode,
            "geco2_pseudo_window_bases": int(pseudo_window_bases),
            "continuous_compressed_bytes": compressed_bytes,
            "continuous_bits": compressed_bytes * 8.0,
            "continuous_bits_per_base": compressed_bytes * 8.0 / max(n, 1),
            "geco2_compression_seconds": compression_seconds,
            "geco2_wall_seconds": perf_counter() - total_started,
            "geco2_full_command": full_result["command"],
            "geco2_full_stdout_tail": full_result["stdout_tail"],
            "geco2_full_stderr_tail": full_result["stderr_tail"],
            "geco2_prefix_results": prefix_results,
            "geco2_note": "GeCo2 is compressed continuously; per-base BPB is assigned from continuous prefix-size deltas over pseudo windows.",
        }
        if self.profile_mode == "estimate":
            metadata["geco2_note"] = "GeCo2 was run once with -e/--estimate; per-base BPB comes from the .iae information-content file."
        offsets = np.arange(n, dtype=np.int64)
        return bpb, offsets, metadata


def _parse_model_spec(spec: str) -> tuple[str, Path | None, str | None]:
    parts = spec.split(":", 2)
    if not parts or not parts[0].strip():
        raise ValueError(f"Invalid --model spec: {spec}")
    kind = parts[0].strip().lower()
    run_dir = Path(parts[1]) if len(parts) >= 2 and parts[1].strip() else None
    checkpoint = parts[2].strip() if len(parts) == 3 and parts[2].strip() else None
    return kind, run_dir, checkpoint


def _resolve_evo2_local_path(path_or_dir: Path, model_name: str) -> Path:
    """Resolve Evo2 checkpoint files from either a .pt path or model directory."""
    if path_or_dir.is_file():
        return path_or_dir
    if path_or_dir.is_dir():
        candidates = [
            path_or_dir / f"{model_name}.pt",
            path_or_dir / "evo2_7b_base.pt",
            path_or_dir / "evo2_1b_base.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return path_or_dir


def _evo2_path_from_args(args: argparse.Namespace, model_name: str) -> Path:
    requested = Path(args.evo2_local_path)
    if model_name == "evo2_1b_base" and requested == DEFAULT_EVO2_LOCAL_PATH:
        return DEFAULT_EVO2_1B_LOCAL_PATH
    return requested


def _carbon_path_from_args(args: argparse.Namespace) -> Path:
    return Path(args.carbon_local_path)


def build_region_adapters(args: argparse.Namespace) -> list[Any]:
    device = resolve_device(args.device)
    specs = list(args.model or [])
    if not specs:
        if args.run_dir is None and args.model_kind not in {"megadna", "geco2", "evo2", "carbon"}:
            raise ValueError("--run-dir is required unless --model is provided")
        if args.run_dir is None:
            specs.append(args.model_kind)
        else:
            checkpoint_part = args.checkpoint or args.checkpoint_tag
            specs.append(f"{args.model_kind}:{args.run_dir}:{checkpoint_part}")

    adapters: list[Any] = []
    for index, spec in enumerate(specs):
        kind, run_dir, checkpoint = _parse_model_spec(spec)
        name = f"{kind}{index + 1}"
        if kind in {"megabyte", "dnagpt"}:
            if run_dir is None:
                raise ValueError(f"{kind} model spec requires a run directory: {spec}")
            checkpoint_text = checkpoint or ("last" if kind == "dnagpt" else args.checkpoint_tag)
            adapters.append(
                build_adapter_from_spec(
                    spec=f"{kind}:{run_dir}:{checkpoint_text}",
                    index=index,
                    device_name=args.device,
                    dtype_name=args.dtype,
                )
            )
            continue
        if kind == "megadna":
            checkpoint_path: Path | None = None
            if run_dir is not None:
                checkpoint_path = _checkpoint_path(run_dir, checkpoint if checkpoint not in {"best", "last"} else None, checkpoint or args.checkpoint_tag)
            elif checkpoint:
                checkpoint_path = Path(checkpoint)
            adapters.append(
                MegaDNARegionAdapter.from_args(
                    name=name,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    base_weight=Path(args.base_weight) if args.base_weight else None,
                    device=device,
                    dtype_name=args.dtype,
                    seq_length=args.seq_length,
                )
            )
            continue
        if kind == "geco2":
            binary = str(run_dir) if run_dir is not None else str(args.geco2_bin)
            level = int(checkpoint) if checkpoint is not None and str(checkpoint).isdigit() else int(args.geco2_level)
            adapters.append(
                Geco2RegionAdapter(
                    name=name,
                    binary=binary,
                    level=level,
                    pseudo_window_bases=args.geco2_pseudo_window_bases,
                    profile_mode=args.geco2_profile_mode,
                    use_dnacorpus_paper_levels=bool(args.geco2_dnacorpus_paper_levels),
                    temp_root=Path(args.geco2_temp_root) if args.geco2_temp_root else None,
                    keep_temp=bool(args.geco2_keep_temp),
                )
            )
            continue
        if kind == "evo2":
            model_name = checkpoint or args.evo2_model_name
            local_path = _resolve_evo2_local_path(run_dir or _evo2_path_from_args(args, model_name), model_name)
            adapters.append(
                Evo2RegionAdapter.from_args(
                    name=name,
                    local_path=local_path,
                    model_name=model_name,
                    context_bases=int(args.evo2_context_bases),
                    requested_device=str(args.device),
                    dtype_name=str(args.dtype),
                    use_kernels=bool(args.evo2_use_kernels),
                )
            )
            continue
        if kind == "carbon":
            local_path = run_dir or _carbon_path_from_args(args)
            revision = checkpoint or args.carbon_revision
            adapters.append(
                CarbonRegionAdapter.from_args(
                    name=name,
                    local_path=local_path,
                    model_name=args.carbon_model_name,
                    revision=revision,
                    context_bases=int(args.carbon_context_bases),
                    device=device,
                    dtype_name=str(args.dtype),
                    trust_remote_code=bool(args.carbon_trust_remote_code),
                )
            )
            continue
        raise ValueError(f"Unsupported model kind {kind!r}")
    return adapters


def _adapter_window_bases(adapter: ProbabilityAdapter, species: str | None) -> int:
    if isinstance(adapter, DNAGPTProbabilityAdapter):
        prefix_token = None
        try:
            from dna_compress.dnagpt_tokenization import resolve_species_prefix_token
            from dna_compress.dnagpt_data import max_target_tokens

            prefix_token = resolve_species_prefix_token(species or "", adapter.config.data.species_prefix_map)
            prefix_length = 1 if prefix_token is not None else 0
            return max(1, max_target_tokens(int(adapter.config.model.seq_length), prefix_length) * int(adapter.token_size))
        except Exception:
            pass
    seq_length = int(getattr(getattr(adapter, "config", None), "model", None).seq_length) if hasattr(getattr(adapter, "config", None), "model") else int(getattr(adapter, "seq_length", 1024))
    return max(1, seq_length * int(adapter.token_size))


def _geco2_pseudo_window_bases(args: argparse.Namespace, region_base_count: int) -> int:
    if args.geco2_pseudo_window_bases is not None:
        return max(1, int(args.geco2_pseudo_window_bases))
    if args.plot_window_bases is not None:
        return max(1, int(args.plot_window_bases))
    return max(1, min(1024, max(region_base_count, 1)))


def _model_sequence(sequence: str, offsets: np.ndarray, adapter: ProbabilityAdapter) -> tuple[str, np.ndarray, int]:
    allowed = set(normalize_alphabet(adapter.alphabet))
    keep_mask = np.asarray([base in allowed for base in sequence], dtype=bool)
    kept = "".join(base for base, keep in zip(sequence, keep_mask) if keep)
    kept_offsets = offsets[keep_mask]
    core_base_count = (len(kept) // int(adapter.token_size)) * int(adapter.token_size)
    return kept[:core_base_count], kept_offsets[:core_base_count], len(kept) - core_base_count


def bpb_for_adapter(
    adapter: ProbabilityAdapter,
    *,
    species: str | None,
    region_sequence: str,
    region_offsets: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model_sequence, model_offsets, trimmed_tail = _model_sequence(region_sequence, region_offsets, adapter)
    if not model_sequence:
        raise ValueError(f"{adapter.name} has no usable bases in the selected region for alphabet {adapter.alphabet!r}")
    target_symbols = encode_unit_symbols(model_sequence, 1, adapter.alphabet)
    result = adapter.unit_probabilities(
        species=species or "",
        core_sequence=model_sequence,
        unit_size=1,
        batch_size=batch_size,
    )
    if result.probabilities.shape[0] != target_symbols.shape[0]:
        raise RuntimeError(
            f"{adapter.name} produced {result.probabilities.shape[0]} probability rows for "
            f"{target_symbols.shape[0]} target bases"
        )
    target_probs = result.probabilities[np.arange(target_symbols.shape[0], dtype=np.int64), target_symbols].clip(min=1e-300)
    bpb = -np.log2(target_probs).astype(np.float64, copy=False)
    metadata = {
        "adapter_name": adapter.name,
        "adapter_class": type(adapter).__name__,
        "alphabet": adapter.alphabet,
        "token_size": int(adapter.token_size),
        "valid_base_count": int(bpb.shape[0]),
        "filtered_out_bases": int(len(region_sequence) - len(model_offsets) - trimmed_tail),
        "trimmed_tail_bases": int(trimmed_tail),
        "model_window_bases": int(_adapter_window_bases(adapter, species)),
        "model_forward_seconds": result.model_forward_seconds,
        "softmax_seconds": result.softmax_seconds,
        "factorization_seconds": result.aggregate_seconds,
        "data_transfer_seconds": result.data_transfer_seconds,
    }
    return bpb, model_offsets, metadata


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(np.float64, copy=True)
    window = min(int(window), int(values.size))
    kernel = np.ones(window, dtype=np.float64)
    sums = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones_like(values, dtype=np.float64), kernel, mode="same")
    return sums / counts.clip(min=1.0)


def window_rows(bpb: np.ndarray, offsets: np.ndarray, *, source_start: int, window_bases: int) -> list[dict[str, Any]]:
    if window_bases <= 0:
        raise ValueError("window_bases must be positive")
    rows: list[dict[str, Any]] = []
    if bpb.size == 0:
        return rows
    max_offset = int(offsets.max()) + 1
    for start in range(0, max_offset, window_bases):
        end = start + window_bases
        mask = (offsets >= start) & (offsets < end)
        if not bool(mask.any()):
            continue
        window_values = bpb[mask]
        rows.append(
            {
                "window_index": len(rows),
                "region_start": int(start),
                "region_end_exclusive": int(end),
                "source_start": int(source_start + start),
                "source_end_exclusive": int(source_start + end),
                "base_count": int(window_values.size),
                "sum_bits": float(window_values.sum()),
                "mean_bpb": float(window_values.mean()),
            }
        )
    return rows


def model_window_average(bpb: np.ndarray, offsets: np.ndarray, model_window_bases: int) -> tuple[np.ndarray, np.ndarray]:
    if bpb.size == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    positions = offsets % int(model_window_bases)
    sums = np.bincount(positions, weights=bpb, minlength=model_window_bases)
    counts = np.bincount(positions, minlength=model_window_bases)
    means = np.full((model_window_bases,), np.nan, dtype=np.float64)
    active = counts > 0
    means[active] = sums[active] / counts[active]
    return means, counts.astype(np.int64)


def smooth_defined_positions(values: np.ndarray, counts: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64, copy=True)
    finite = np.isfinite(values) & (counts > 0)
    if not bool(np.any(finite)):
        return np.full_like(values, np.nan, dtype=np.float64)
    dense = np.where(finite, values, 0.0).astype(np.float64, copy=False)
    weights = finite.astype(np.float64)
    window = max(1, min(int(window), int(values.size)))
    if window <= 1:
        return np.where(finite, values, np.nan)
    kernel = np.ones(window, dtype=np.float64)
    sums = np.convolve(dense, kernel, mode="same")
    weight_sums = np.convolve(weights, kernel, mode="same")
    out = sums / weight_sums.clip(min=1.0)
    out[weight_sums <= 0] = np.nan
    return out


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def write_per_base_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "source_base_offset", "region_base_offset", "base", "bpb", "window_index", "window_offset"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_compact_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def _round_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return numeric
    return round(numeric, digits)


def _compact_window_rows(rows: list[dict[str, Any]], *, max_rows: int = 64) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows[:max_rows]:
        compact.append(
            {
                "window_index": int(row["window_index"]),
                "source_start": int(row["source_start"]),
                "source_end_exclusive": int(row["source_end_exclusive"]),
                "base_count": int(row["base_count"]),
                "mean_bpb": _round_float(row["mean_bpb"]),
            }
        )
    return compact


def write_model_artifact(
    output_dir: Path,
    *,
    model_name: str,
    bpb: np.ndarray,
    offsets: np.ndarray,
    metadata: dict[str, Any],
    window_summary: list[dict[str, Any]],
    worst_window: dict[str, Any] | None,
    model_window_means: np.ndarray,
    model_window_counts: np.ndarray,
    total_bits: float,
    total_bpb: float,
    adapter_wall_seconds: float,
    record_dtype: str,
    store_offsets: bool = True,
    offset_start: int = 0,
    statistics_sample_size: int = 0,
) -> dict[str, Any]:
    """Write compact reusable per-base BPB arrays for one model."""
    model_dir = output_dir / "models" / _safe_label(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    npz_path = model_dir / "bpb.npz"
    summary_path = model_dir / "summary.json"
    bpb_dtype = _record_dtype(record_dtype)
    max_offset = int(np.max(offsets)) if offsets.size else 0
    max_count = int(np.max(model_window_counts)) if model_window_counts.size else 0
    arrays: dict[str, np.ndarray] = {
        "bpb": np.asarray(bpb, dtype=bpb_dtype),
        "model_window_means": np.asarray(model_window_means, dtype=bpb_dtype),
        "model_window_counts": np.asarray(model_window_counts, dtype=_offset_dtype(max_count)),
    }
    if store_offsets:
        arrays["offsets"] = np.asarray(offsets, dtype=_offset_dtype(max_offset))
    np.savez_compressed(npz_path, **arrays)
    compact_metadata = _compact_metadata(metadata)
    statistics_values = bpb
    if statistics_sample_size > 0 and bpb.size > statistics_sample_size:
        sample_indices = np.linspace(0, bpb.size - 1, int(statistics_sample_size), dtype=np.int64)
        statistics_values = bpb[sample_indices]
    summary = {
        "model": model_name,
        "artifact_npz": str(npz_path),
        "summary_json": str(summary_path),
        "record_dtype": record_dtype,
        "base_count": int(bpb.size),
        "offset_start": int(offset_start),
        "offsets_stored": bool(store_offsets),
        "model_window_bases": int(metadata.get("model_window_bases", model_window_means.shape[0])),
        "total_bits": _round_float(total_bits),
        "total_bpb": _round_float(total_bpb),
        "statistics": {key: _round_float(value) for key, value in _quantiles(statistics_values).items()},
        "statistics_sampled": bool(statistics_values.size != bpb.size),
        "statistics_sample_size": int(statistics_values.size),
        "worst_plot_window": _json_safe(worst_window),
        "window_summary_preview": _compact_window_rows(window_summary),
        "window_count": int(len(window_summary)),
        "adapter_wall_seconds": _round_float(adapter_wall_seconds),
        "metadata": compact_metadata,
    }
    _write_compact_json(summary_path, summary)
    return summary


def _resolve_artifact_path(path_text: str | None, *, json_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.exists():
        return path
    candidate = json_dir / path
    if candidate.exists():
        return candidate
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    candidate = json_dir / "models" / path.name
    if candidate.exists():
        return candidate
    return path


def _load_model_artifact_payload(summary: dict[str, Any], *, json_dir: Path) -> dict[str, Any] | None:
    artifact_path = _resolve_artifact_path(str(summary.get("artifact_npz") or ""), json_dir=json_dir)
    if artifact_path is not None and not artifact_path.exists():
        model_name = str(summary.get("model") or "")
        if model_name:
            relocated = json_dir / "models" / _safe_label(model_name) / artifact_path.name
            if relocated.exists():
                artifact_path = relocated
    if artifact_path is None or not artifact_path.exists():
        summary_path = _resolve_artifact_path(str(summary.get("summary_json") or ""), json_dir=json_dir)
        if summary_path is not None and summary_path.exists():
            nested = json.loads(summary_path.read_text(encoding="utf-8"))
            artifact_path = _resolve_artifact_path(str(nested.get("artifact_npz") or ""), json_dir=summary_path.parent)
            if artifact_path is not None and not artifact_path.exists():
                model_name = str(nested.get("model") or summary.get("model") or "")
                if model_name:
                    relocated = summary_path.parent.parent / _safe_label(model_name) / artifact_path.name
                    if relocated.exists():
                        artifact_path = relocated
    if artifact_path is None or not artifact_path.exists():
        return None
    with np.load(artifact_path) as data:
        bpb = np.asarray(data["bpb"], dtype=np.float64)
        if "offsets" in data:
            offsets = np.asarray(data["offsets"], dtype=np.int64)
        else:
            offset_start = int(summary.get("offset_start", 0))
            offsets = np.arange(offset_start, offset_start + bpb.shape[0], dtype=np.int64)
        means = np.asarray(data["model_window_means"], dtype=np.float64)
        counts = np.asarray(data["model_window_counts"], dtype=np.int64)
    return {
        "bpb": bpb,
        "offsets": offsets,
        "model_window_means": means,
        "model_window_counts": counts,
    }


def downsample_for_plot(offsets: np.ndarray, bpb: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or bpb.size <= max_points:
        return offsets, bpb
    bins = np.linspace(0, bpb.size, int(max_points) + 1, dtype=np.int64)
    keep = bins[1:] > bins[:-1]
    starts = bins[:-1][keep]
    ends = bins[1:][keep]
    y = np.empty((starts.shape[0],), dtype=np.float64)
    x = np.empty((starts.shape[0],), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        y[index] = float(np.mean(bpb[start:end]))
        x[index] = float(offsets[start])
    return x, y


def plot_curves(
    path: Path,
    *,
    region_base_count: int,
    source_start: int,
    per_model: dict[str, dict[str, Any]],
    smooth_window_bases: int,
    model_window_smooth_bases: int,
    window_boundary_bases: int | None,
    max_points_per_model: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    ax = axes[0]
    if window_boundary_bases is not None and window_boundary_bases > 0:
        for boundary in range(window_boundary_bases, int(region_base_count), window_boundary_bases):
            ax.axvline(boundary, color="0.75", linewidth=0.6, alpha=0.45, zorder=0)
    for model_name, payload in per_model.items():
        bpb = payload["bpb"]
        offsets = payload["offsets"]
        if bpb.size == 0:
            continue
        plot_offsets, plot_bpb = downsample_for_plot(offsets, bpb, max_points_per_model)
        effective_smooth = max(1, int(round(smooth_window_bases * (plot_bpb.size / max(bpb.size, 1)))))
        smooth = moving_average(plot_bpb, effective_smooth)
        ax.plot(plot_offsets, plot_bpb, alpha=0.12, linewidth=0.45)
        ax.plot(plot_offsets, smooth, linewidth=1.7, label=f"{model_name} smooth")
        ax.axhline(float(bpb.mean()), linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(f"Region BPB, source offset {source_start:,}, bases {region_base_count:,}")
    ax.set_xlabel("Region base offset")
    ax.set_ylabel("bits/base")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax2 = axes[1]
    for model_name, payload in per_model.items():
        means = payload["model_window_means"]
        counts = payload["model_window_counts"]
        active = counts > 0
        if not bool(np.any(active)):
            continue
        x = np.arange(means.shape[0])[active]
        smoothed = smooth_defined_positions(means, counts, model_window_smooth_bases)
        ax2.scatter(x, means[active], s=5, alpha=0.18)
        ax2.plot(x, smoothed[active], linewidth=1.7, label=f"{model_name} smooth")
    ax2.set_title("Average BPB by model-window position")
    ax2.set_xlabel("Position in model window (bases)")
    ax2.set_ylabel("bits/base")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_individual_windows(
    output_dir: Path,
    *,
    per_model: dict[str, dict[str, Any]],
    source_name: str,
    source_start: int,
    region_base_count: int,
    plot_window_bases: int,
    smooth_window_bases: int,
    max_windows: int,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if plot_window_bases <= 0 or region_base_count <= 0:
        return paths
    window_count = int(math.ceil(region_base_count / plot_window_bases))
    if max_windows > 0:
        window_count = min(window_count, int(max_windows))
    for window_index in range(window_count):
        start = window_index * plot_window_bases
        end = min(region_base_count, start + plot_window_bases)
        fig, ax = plt.subplots(1, 1, figsize=(11, 4.5), constrained_layout=True)
        for model_name, payload in per_model.items():
            bpb = payload["bpb"]
            offsets = payload["offsets"]
            mask = (offsets >= start) & (offsets < end)
            if not bool(mask.any()):
                continue
            x = offsets[mask] - start
            y = bpb[mask]
            ax.plot(x, y, alpha=0.18, linewidth=0.5)
            ax.plot(x, moving_average(y, smooth_window_bases), linewidth=1.7, label=f"{model_name} smooth")
            ax.axhline(float(y.mean()), linestyle="--", linewidth=1.0, alpha=0.55)
        ax.set_title(
            f"{source_name} window {window_index}, source offsets "
            f"{source_start + start:,}-{source_start + end:,}"
        )
        ax.set_xlabel("Offset in plotted window")
        ax.set_ylabel("bits/base")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        path = output_dir / f"window_{window_index:05d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dnacorpus", "opengenome2"))
    parser.add_argument("--dataset-dir", default="datasets/DNACorpus")
    parser.add_argument("--input-dir", help="OpenGenome2 FASTA subset directory. Defaults to --dataset-dir.")
    parser.add_argument("--source", nargs="+")
    parser.add_argument("--species", nargs="+")
    parser.add_argument("--sequence-source-mode", choices=("auto", "flat_file", "fasta_dir"), default="auto")
    parser.add_argument("--multi-sequence-mode", choices=("separate", "concat"), default="separate")
    parser.add_argument("--sequence-include", action="append")
    parser.add_argument("--alphabet", default="ACGTN")

    parser.add_argument("--region-start", type=int)
    parser.add_argument("--region-bases", type=int, default=DEFAULT_REGION_BASES)
    parser.add_argument("--random-region", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--model", action="append", help="Repeatable model spec: kind:run_dir[:checkpoint_or_tag].")
    parser.add_argument("--model-kind", choices=("megabyte", "megadna", "dnagpt", "geco2", "evo2", "carbon"), default="megabyte")
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-tag", default="best")
    parser.add_argument("--base-weight", help="MegaDNA base checkpoint path.")
    parser.add_argument("--seq-length", type=int, help="Override MegaDNA seq length.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--geco2-bin", default="GeCo2")
    parser.add_argument("--geco2-level", type=int, default=5)
    parser.add_argument("--geco2-profile-mode", choices=("estimate", "constant", "prefix_delta"), default="estimate")
    parser.add_argument("--geco2-pseudo-window-bases", type=int)
    parser.add_argument(
        "--no-geco2-dnacorpus-paper-levels",
        dest="geco2_dnacorpus_paper_levels",
        action="store_false",
        help="Disable DNACorpus per-species GeCo2 levels from the previous paper-mode experiments.",
    )
    parser.add_argument("--geco2-temp-root")
    parser.add_argument("--geco2-keep-temp", action="store_true")
    parser.set_defaults(geco2_dnacorpus_paper_levels=True)
    parser.add_argument(
        "--evo2-local-path",
        default=str(DEFAULT_EVO2_LOCAL_PATH),
        help="Local Evo2 .pt checkpoint path or model directory. For 1B use third_party/evo2_1b_base/evo2_1b_base.pt or --model evo2:third_party/evo2_1b_base:evo2_1b_base.",
    )
    parser.add_argument("--evo2-model-name", default="evo2_7b_base", help="Evo2 model name, e.g. evo2_7b_base or evo2_1b_base.")
    parser.add_argument("--evo2-context-bases", type=int, default=8192)
    parser.add_argument("--evo2-use-kernels", action="store_true", help="Enable optional Vortex Triton kernels for Evo2.")
    parser.add_argument(
        "--carbon-local-path",
        default=str(DEFAULT_CARBON_LOCAL_PATH),
        help="Local Carbon-500M fns model directory downloaded from HuggingFaceBio/Carbon-500M revision fns.",
    )
    parser.add_argument("--carbon-model-name", default=DEFAULT_CARBON_MODEL_NAME)
    parser.add_argument("--carbon-revision", default=DEFAULT_CARBON_REVISION)
    parser.add_argument("--carbon-context-bases", type=int, default=49152)
    parser.add_argument("--carbon-trust-remote-code", dest="carbon_trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-carbon-trust-remote-code", dest="carbon_trust_remote_code", action="store_false")

    parser.add_argument("--plot-window-bases", type=int)
    parser.add_argument("--smooth-window-bases", type=int, default=512)
    parser.add_argument("--model-window-smooth-bases", type=int, default=64)
    parser.add_argument("--plot-max-points", type=int, default=200000, help="Maximum plotted per-base/downsampled points per model.")
    parser.add_argument("--plot-individual-windows", action="store_true")
    parser.add_argument("--max-individual-window-plots", type=int, default=64)
    parser.add_argument("--compute-only", action="store_true", help="Only compute compact model BPB artifacts; skip plotting.")
    parser.add_argument("--record-dtype", choices=("float16", "float32", "float64"), default="float16")
    parser.add_argument("--write-per-base-csv", action="store_true", help="Also write legacy per-base CSV. Off by default to keep records small.")
    parser.add_argument("--plot-only", action="store_true", help="Only redraw plots from existing region_bpb.json compact artifacts or legacy CSV outputs.")
    parser.add_argument("--combine-matching-regions", action="store_true", help="For --plot-only, group result JSONs with the same resolved source/start/length and draw all model artifacts together.")
    parser.add_argument("--result-json", nargs="+", help="Existing region_bpb.json paths for --plot-only.")
    parser.add_argument("--result-dir", nargs="+", help="Existing result directories containing region_bpb.json for --plot-only.")
    parser.add_argument("--from-full-result-json", nargs="+", help="Slice region data from full-size region_bpb.json artifacts without recomputing models.")
    parser.add_argument("--from-full-result-dir", nargs="+", help="Directories containing full-size region_bpb.json artifacts to slice.")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _window_boundary_bases(args: argparse.Namespace, model_summaries: dict[str, Any]) -> int | None:
    if args.plot_window_bases is not None:
        return int(args.plot_window_bases)
    values = [
        int(payload["model_window_bases"])
        for payload in model_summaries.values()
        if isinstance(payload, dict) and payload.get("model_window_bases")
    ]
    if not values:
        return None
    return values[0] if all(value == values[0] for value in values) else None


def _load_per_base_payload(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    json_dir = Path(result.get("_json_path", "region_bpb.json")).parent
    models = result.get("models", {}) if isinstance(result.get("models"), dict) else {}
    payload: dict[str, dict[str, Any]] = {}
    for model, summary in models.items():
        if not isinstance(summary, dict):
            continue
        artifact_payload = _load_model_artifact_payload(summary, json_dir=json_dir)
        if artifact_payload is not None:
            payload[str(model)] = artifact_payload
    if payload:
        return payload

    outputs = result.get("outputs", {}) if isinstance(result.get("outputs"), dict) else {}
    per_base_path = Path(outputs.get("per_base_csv") or Path(outputs.get("json", "region_bpb.json")).with_name("region_bpb_per_base.csv"))
    if not per_base_path.exists():
        per_base_path = Path(result.get("_json_path", "region_bpb.json")).with_name("region_bpb_per_base.csv")
    if not per_base_path.exists():
        raise FileNotFoundError(f"Could not find compact model artifacts or legacy per-base CSV for {json_dir}")
    grouped: dict[str, dict[str, list[Any]]] = {}
    with per_base_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            bucket = grouped.setdefault(model, {"offsets": [], "bpb": []})
            bucket["offsets"].append(int(row["region_base_offset"]))
            bucket["bpb"].append(float(row["bpb"]))

    for model, values in grouped.items():
        offsets = np.asarray(values["offsets"], dtype=np.int64)
        bpb = np.asarray(values["bpb"], dtype=np.float64)
        order = np.argsort(offsets)
        offsets = offsets[order]
        bpb = bpb[order]
        model_window_bases = int(models.get(model, {}).get("model_window_bases") or max(int(result.get("region_bases", 1)), 1))
        means, counts = model_window_average(bpb, offsets, model_window_bases)
        payload[model] = {
            "bpb": bpb,
            "offsets": offsets,
            "model_window_means": means,
            "model_window_counts": counts,
        }
    return payload


def _discover_plot_only_jsons(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.result_json or []:
        paths.append(Path(item))
    for item in args.result_dir or []:
        directory = Path(item)
        direct = directory / "region_bpb.json"
        if direct.exists():
            paths.append(direct)
        elif directory.is_dir():
            paths.extend(sorted(directory.glob("**/region_bpb.json")))
        else:
            paths.append(direct)
    if not paths:
        if not args.output_dir:
            raise ValueError("--plot-only requires --result-json, --result-dir, or --output-dir to discover results.")
        output_dir = Path(args.output_dir)
        direct = output_dir / "region_bpb.json"
        if direct.exists():
            paths.append(direct)
        else:
            paths.extend(sorted(output_dir.glob("**/region_bpb.json")))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _discover_full_result_jsons(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.from_full_result_json or []:
        paths.append(Path(item))
    for item in args.from_full_result_dir or []:
        directory = Path(item)
        direct = directory / "region_bpb.json"
        if direct.exists():
            paths.append(direct)
        elif directory.is_dir():
            paths.extend(sorted(directory.glob("**/region_bpb.json")))
        else:
            paths.append(direct)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def redraw_from_result(args: argparse.Namespace, json_path: Path) -> dict[str, Any]:
    result = json.loads(json_path.read_text(encoding="utf-8"))
    result["_json_path"] = str(json_path)
    result["region_identity"] = result_region_identity(result)
    output_dir = json_path.parent
    plot_payload = _load_per_base_payload(result)
    curve_png = output_dir / "region_bpb_curve.png"
    window_boundary = _window_boundary_bases(args, result.get("models", {}) if isinstance(result.get("models"), dict) else {})
    plot_curves(
        curve_png,
        region_base_count=int(result.get("region_bases", 0)),
        source_start=int(result.get("region_start", 0)),
        per_model=plot_payload,
        smooth_window_bases=int(args.smooth_window_bases),
        model_window_smooth_bases=int(args.model_window_smooth_bases),
        window_boundary_bases=window_boundary,
        max_points_per_model=int(args.plot_max_points),
    )
    individual_paths: list[str] = []
    if args.plot_individual_windows:
        plot_window_bases = int(args.plot_window_bases or window_boundary or next(iter(result.get("models", {}).values())).get("model_window_bases", 1024))
        individual_paths = plot_individual_windows(
            output_dir / "individual_windows",
            per_model=plot_payload,
            source_name=str(result.get("source", "source")),
            source_start=int(result.get("region_start", 0)),
            region_base_count=int(result.get("region_bases", 0)),
            plot_window_bases=plot_window_bases,
            smooth_window_bases=int(args.smooth_window_bases),
            max_windows=int(args.max_individual_window_plots),
        )
    outputs = result.setdefault("outputs", {})
    outputs["curve_png"] = str(curve_png)
    if individual_paths:
        outputs["individual_window_dir"] = str(output_dir / "individual_windows")
        outputs["individual_window_count"] = len(individual_paths)
    _write_compact_json(json_path, result)
    return {"json": str(json_path), "curve_png": str(curve_png), "individual_windows": len(individual_paths)}


def redraw_combined_matching_regions(args: argparse.Namespace, json_paths: list[Path]) -> list[dict[str, Any]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in json_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        result["_json_path"] = str(path)
        result["region_identity"] = result_region_identity(result)
        loaded.append((path, result))

    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, result in loaded:
        identity = result_region_identity(result)
        groups.setdefault(str(identity["signature"]), []).append((path, result))

    summaries: list[dict[str, Any]] = []
    multiple_groups = len(groups) > 1
    default_root = None
    if loaded:
        default_root = loaded[0][0].parent.parent / "combined_region_bpb_plots"
    output_root = Path(args.output_dir) if args.output_dir else default_root
    if output_root is None:
        raise ValueError("--plot-only could not determine an output directory for combined plots.")

    for signature, items in sorted(groups.items()):
        reference = items[0][1]
        identity = result_region_identity(reference)
        source_label = _safe_label(str(identity.get("source") or identity.get("species") or signature))
        output_dir = output_root / source_label if multiple_groups else output_root
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_payload: dict[str, dict[str, Any]] = {}
        combined_models: dict[str, Any] = {}
        source_jsons: list[str] = []
        for path, result in items:
            source_jsons.append(str(path))
            payloads = _load_per_base_payload(result)
            models = result.get("models", {}) if isinstance(result.get("models"), dict) else {}
            for model_name, payload in payloads.items():
                model_summary = models.get(model_name, {"model": model_name})
                metadata = model_summary.get("metadata", {}) if isinstance(model_summary, dict) else {}
                label = str(metadata.get("evo2_model_name") or metadata.get("carbon_model_name") or model_name)
                if label in plot_payload:
                    label = f"{label}@{_safe_label(path.parent.name)}"
                plot_payload[label] = payload
                combined_models[label] = model_summary

        if not plot_payload:
            continue
        curve_png = output_dir / "region_bpb_combined_curve.png"
        window_boundary = _window_boundary_bases(args, combined_models)
        plot_curves(
            curve_png,
            region_base_count=int(reference.get("region_bases", 0)),
            source_start=int(reference.get("region_start", 0)),
            per_model=plot_payload,
            smooth_window_bases=int(args.smooth_window_bases),
            model_window_smooth_bases=int(args.model_window_smooth_bases),
            window_boundary_bases=window_boundary,
            max_points_per_model=int(args.plot_max_points),
        )
        individual_paths: list[str] = []
        if args.plot_individual_windows:
            plot_window_bases = int(args.plot_window_bases or window_boundary or next(iter(combined_models.values())).get("model_window_bases", 1024))
            individual_paths = plot_individual_windows(
                output_dir / "individual_windows",
                per_model=plot_payload,
                source_name=str(reference.get("source", "source")),
                source_start=int(reference.get("region_start", 0)),
                region_base_count=int(reference.get("region_bases", 0)),
                plot_window_bases=plot_window_bases,
                smooth_window_bases=int(args.smooth_window_bases),
                max_windows=int(args.max_individual_window_plots),
            )

        combined_json = output_dir / "region_bpb_combined.json"
        combined = {
            "combined_plot": True,
            "region_identity": identity,
            "dataset": reference.get("dataset"),
            "source": reference.get("source"),
            "species": reference.get("species"),
            "region_start": int(reference.get("region_start", 0)),
            "region_bases": int(reference.get("region_bases", 0)),
            "random_region_values": sorted({bool(result.get("random_region")) for _, result in items}),
            "seed_values": sorted({int(result.get("seed", 0)) for _, result in items}),
            "source_result_jsons": source_jsons,
            "model_count": len(combined_models),
            "models": combined_models,
            "outputs": {
                "json": str(combined_json),
                "curve_png": str(curve_png),
                "individual_window_dir": str(output_dir / "individual_windows") if individual_paths else None,
                "individual_window_count": len(individual_paths),
            },
        }
        _write_compact_json(combined_json, combined)
        summaries.append(
            {
                "region_signature": signature,
                "source": reference.get("source"),
                "species": reference.get("species"),
                "json": str(combined_json),
                "curve_png": str(curve_png),
                "model_names": sorted(combined_models),
                "source_result_count": len(items),
            }
        )
    return summaries


def run_probe_from_full_result(args: argparse.Namespace, *, full_json_path: Path, output_dir: Path) -> dict[str, Any]:
    total_started = perf_counter()
    full_result = json.loads(full_json_path.read_text(encoding="utf-8"))
    full_result["_json_path"] = str(full_json_path)
    payloads = _load_per_base_payload(full_result)
    if not payloads:
        raise ValueError(f"No model artifacts found in {full_json_path}")
    source_length = int(full_result.get("region_bases") or max(int(payload["bpb"].shape[0]) for payload in payloads.values()))
    requested_bases = int(args.region_bases)
    if requested_bases <= 0 or requested_bases > source_length:
        requested_bases = source_length
    if args.random_region:
        region_start = stable_random_start(
            source_name=str(full_result.get("source", full_json_path.parent.name)),
            source_length=source_length,
            region_bases=requested_bases,
            seed=args.seed,
        )
    elif args.region_start is not None:
        region_start = int(args.region_start)
    else:
        region_start = 0
    if region_start > source_length:
        raise ValueError(f"region start {region_start} exceeds source length {source_length}")
    actual_bases = min(requested_bases, source_length - region_start)
    region_end = region_start + actual_bases

    output_dir.mkdir(parents=True, exist_ok=True)
    model_summaries: dict[str, Any] = {}
    plot_payload: dict[str, dict[str, Any]] = {}
    window_csv_rows: list[dict[str, Any]] = []
    model_window_rows: list[dict[str, Any]] = []
    for model_name, payload in payloads.items():
        full_offsets = payload["offsets"]
        start_index = int(np.searchsorted(full_offsets, region_start, side="left"))
        end_index = int(np.searchsorted(full_offsets, region_end, side="left"))
        absolute_offsets = full_offsets[start_index:end_index]
        if absolute_offsets.size == 0:
            continue
        bpb = payload["bpb"][start_index:end_index]
        relative_offsets = absolute_offsets - region_start
        summary = (full_result.get("models") or {}).get(model_name, {})
        metadata = dict(summary.get("metadata") or {})
        model_window_bases = int(summary.get("model_window_bases") or metadata.get("model_window_bases") or max(actual_bases, 1))
        metadata["model_window_bases"] = model_window_bases
        metadata["source_full_result_json"] = str(full_json_path)
        metadata["source_full_model"] = model_name
        plot_window_bases = int(args.plot_window_bases or model_window_bases)
        rows = window_rows(bpb, relative_offsets, source_start=region_start, window_bases=plot_window_bases)
        for row in rows:
            row["model"] = model_name
            row["plot_window_bases"] = plot_window_bases
            window_csv_rows.append(row)
        worst_row = max(rows, key=lambda item: float(item["mean_bpb"])) if rows else None
        means, counts = model_window_average(bpb, absolute_offsets, model_window_bases)
        for position, (mean_value, count) in enumerate(zip(means, counts)):
            if int(count) > 0:
                model_window_rows.append(
                    {
                        "model": model_name,
                        "model_window_position": position,
                        "base_count": int(count),
                        "mean_bpb": float(mean_value),
                    }
                )
        total_bits = float(np.sum(bpb))
        model_summaries[model_name] = write_model_artifact(
            output_dir,
            model_name=model_name,
            bpb=bpb,
            offsets=relative_offsets,
            metadata=metadata,
            window_summary=rows,
            worst_window=worst_row,
            model_window_means=means,
            model_window_counts=counts,
            total_bits=total_bits,
            total_bpb=total_bits / max(int(bpb.size), 1),
            adapter_wall_seconds=0.0,
            record_dtype=str(args.record_dtype),
        )
        plot_payload[model_name] = {
            "bpb": bpb,
            "offsets": relative_offsets,
            "model_window_means": means,
            "model_window_counts": counts,
        }

    windows_csv = output_dir / "region_bpb_windows.csv"
    model_window_csv = output_dir / "region_bpb_model_window_average.csv"
    curve_png = output_dir / "region_bpb_curve.png"
    json_path = output_dir / "region_bpb.json"
    write_csv(windows_csv, window_csv_rows)
    write_csv(model_window_csv, model_window_rows)
    window_boundary = _window_boundary_bases(args, model_summaries)
    if not bool(getattr(args, "compute_only", False)):
        plot_curves(
            curve_png,
            region_base_count=actual_bases,
            source_start=region_start,
            per_model=plot_payload,
            smooth_window_bases=int(args.smooth_window_bases),
            model_window_smooth_bases=int(args.model_window_smooth_bases),
            window_boundary_bases=window_boundary,
            max_points_per_model=int(args.plot_max_points),
        )
    source_paths_preview = list(full_result.get("source_paths_preview") or [])
    result = {
        "dataset": full_result.get("dataset"),
        "source": full_result.get("source"),
        "species": full_result.get("species"),
        "source_path_count": full_result.get("source_path_count"),
        "source_paths_preview": source_paths_preview,
        "source_is_fasta": bool(full_result.get("source_is_fasta")),
        "alphabet": full_result.get("alphabet"),
        "filtered_source_bases": source_length,
        "filtered_source_bases_known": True,
        "filtered_source_bases_lower_bound": int(region_end),
        "region_start": int(region_start),
        "requested_region_bases": int(args.region_bases),
        "region_bases": int(actual_bases),
        "region_identity": region_identity(
            dataset=full_result.get("dataset"),
            source=full_result.get("source"),
            species=full_result.get("species"),
            alphabet=full_result.get("alphabet"),
            region_start=int(region_start),
            region_bases=int(actual_bases),
        ),
        "random_region": bool(args.random_region),
        "seed": int(args.seed),
        "model_count": len(model_summaries),
        "models": model_summaries,
        "sliced_from_full_result_json": str(full_json_path),
        "timing": {"total_wall_seconds": _round_float(perf_counter() - total_started)},
        "outputs": {
            "json": str(json_path),
            "model_artifact_dir": str(output_dir / "models"),
            "per_base_csv": None,
            "windows_csv": str(windows_csv),
            "model_window_average_csv": str(model_window_csv),
            "curve_png": str(curve_png) if not bool(getattr(args, "compute_only", False)) else None,
            "individual_window_dir": None,
            "individual_window_count": 0,
            "record_dtype": str(args.record_dtype),
        },
    }
    _write_compact_json(json_path, result)
    return result


def run_probe_for_source(
    args: argparse.Namespace,
    *,
    source_info: dict[str, Any],
    adapters: list[Any],
    output_dir: Path,
) -> dict[str, Any]:
    total_started = perf_counter()
    record_dtype = str(getattr(args, "record_dtype", "float16"))
    write_legacy_per_base_csv = bool(getattr(args, "write_per_base_csv", False))
    compute_only = bool(getattr(args, "compute_only", False))
    source_length: int | None = None
    length_seconds = 0.0
    requested_bases = int(args.region_bases)
    needs_source_length = bool(args.random_region or requested_bases <= 0)
    if needs_source_length:
        length_started = perf_counter()
        source_length = filtered_length(source_info["paths"], alphabet=str(source_info["alphabet"]), fasta=bool(source_info["fasta"]))
        length_seconds = perf_counter() - length_started
        if requested_bases <= 0 or requested_bases > source_length:
            requested_bases = source_length
    if args.random_region:
        assert source_length is not None
        region_start = stable_random_start(
            source_name=str(source_info["source"]),
            source_length=source_length,
            region_bases=requested_bases,
            seed=args.seed,
        )
    elif args.region_start is not None:
        region_start = int(args.region_start)
    else:
        region_start = 0
    if source_length is not None and region_start > source_length:
        raise ValueError(f"region start {region_start} exceeds source length {source_length}")
    actual_bases = min(requested_bases, source_length - region_start) if source_length is not None else requested_bases

    read_started = perf_counter()
    region_bytes = extract_filtered_region(
        source_info["paths"],
        alphabet=str(source_info["alphabet"]),
        fasta=bool(source_info["fasta"]),
        start=region_start,
        length=actual_bases,
    )
    read_seconds = perf_counter() - read_started
    region_sequence = region_bytes.decode("ascii")
    region_offsets = np.arange(len(region_sequence), dtype=np.int64)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "region_bpb.json"
    existing_result: dict[str, Any] | None = None
    existing_models: dict[str, Any] = {}
    plot_payload: dict[str, dict[str, Any]] = {}
    if json_path.exists():
        try:
            candidate = json.loads(json_path.read_text(encoding="utf-8"))
            same_region = (
                candidate.get("dataset") == source_info.get("dataset")
                and candidate.get("source") == source_info.get("source")
                and int(candidate.get("region_start", -1)) == int(region_start)
                and int(candidate.get("region_bases", -1)) == int(len(region_sequence))
            )
            if same_region:
                candidate["_json_path"] = str(json_path)
                existing_result = candidate
                existing_models = dict(candidate.get("models", {}) if isinstance(candidate.get("models"), dict) else {})
                plot_payload = _load_per_base_payload(candidate)
        except Exception:
            existing_result = None
            existing_models = {}
            plot_payload = {}

    per_base_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = dict(existing_models)

    for adapter in adapters:
        model_started = perf_counter()
        if isinstance(adapter, Geco2RegionAdapter):
            bpb, offsets, adapter_meta = adapter.region_bpb(
                region_sequence=region_sequence,
                region_offsets=region_offsets,
                pseudo_window_bases=_geco2_pseudo_window_bases(args, len(region_sequence)),
                dataset=str(source_info.get("dataset")) if source_info.get("dataset") is not None else None,
                species=str(source_info.get("species")) if source_info.get("species") is not None else None,
                source=str(source_info.get("source")) if source_info.get("source") is not None else None,
            )
        elif isinstance(adapter, Evo2RegionAdapter):
            bpb, offsets, adapter_meta = adapter.region_bpb(
                region_sequence=region_sequence,
                region_offsets=region_offsets,
                batch_size=args.batch_size,
            )
        elif isinstance(adapter, CarbonRegionAdapter):
            bpb, offsets, adapter_meta = adapter.region_bpb(
                region_sequence=region_sequence,
                region_offsets=region_offsets,
                batch_size=args.batch_size,
            )
        else:
            bpb, offsets, adapter_meta = bpb_for_adapter(
                adapter,
                species=source_info.get("species") or source_info.get("source"),
                region_sequence=region_sequence,
                region_offsets=region_offsets,
                batch_size=args.batch_size,
            )
        adapter_total_seconds = perf_counter() - model_started
        model_window_bases = int(adapter_meta["model_window_bases"])
        plot_window_bases = int(args.plot_window_bases or model_window_bases)
        rows = window_rows(bpb, offsets, source_start=region_start, window_bases=plot_window_bases)
        for row in rows:
            row["model"] = adapter.name
            row["plot_window_bases"] = plot_window_bases
        worst_row = max(rows, key=lambda item: float(item["mean_bpb"])) if rows else None
        means, counts = model_window_average(bpb, offsets, model_window_bases)

        if write_legacy_per_base_csv:
            for index, value in enumerate(bpb):
                region_offset = int(offsets[index])
                per_base_rows.append(
                    {
                        "model": adapter.name,
                        "source_base_offset": int(region_start + region_offset),
                        "region_base_offset": region_offset,
                        "base": region_sequence[region_offset],
                        "bpb": float(value),
                        "window_index": int(region_offset // plot_window_bases),
                        "window_offset": int(region_offset % plot_window_bases),
                    }
                )

        total_bits = float(bpb.sum())
        total_bpb = total_bits / max(int(bpb.size), 1)
        model_summaries[adapter.name] = write_model_artifact(
            output_dir,
            model_name=adapter.name,
            bpb=bpb,
            offsets=offsets,
            metadata=adapter_meta,
            window_summary=rows,
            worst_window=worst_row,
            model_window_means=means,
            model_window_counts=counts,
            total_bits=total_bits,
            total_bpb=total_bpb,
            adapter_wall_seconds=adapter_total_seconds,
            record_dtype=record_dtype,
        )
        plot_payload[adapter.name] = {
            "bpb": bpb,
            "offsets": offsets,
            "model_window_means": means,
            "model_window_counts": counts,
        }

    windows_csv = output_dir / "region_bpb_windows.csv"
    model_window_csv = output_dir / "region_bpb_model_window_average.csv"
    curve_png = output_dir / "region_bpb_curve.png"
    per_base_csv = output_dir / "region_bpb_per_base.csv"
    window_csv_rows: list[dict[str, Any]] = []
    model_window_rows: list[dict[str, Any]] = []
    for model_name, payload in plot_payload.items():
        plot_window_bases = int(args.plot_window_bases or model_summaries.get(model_name, {}).get("model_window_bases") or len(region_sequence) or 1)
        for row in window_rows(payload["bpb"], payload["offsets"], source_start=region_start, window_bases=plot_window_bases):
            row["model"] = model_name
            row["plot_window_bases"] = plot_window_bases
            window_csv_rows.append(row)
        for position, (mean_value, count) in enumerate(zip(payload["model_window_means"], payload["model_window_counts"])):
            if int(count) > 0:
                model_window_rows.append(
                    {
                        "model": model_name,
                        "model_window_position": position,
                        "base_count": int(count),
                        "mean_bpb": float(mean_value),
                    }
                )

    if write_legacy_per_base_csv:
        write_per_base_csv(per_base_csv, per_base_rows)
    write_csv(windows_csv, window_csv_rows)
    write_csv(model_window_csv, model_window_rows)
    window_boundary = _window_boundary_bases(args, model_summaries)
    individual_paths: list[str] = []
    if not compute_only:
        plot_curves(
            curve_png,
            region_base_count=len(region_sequence),
            source_start=region_start,
            per_model=plot_payload,
            smooth_window_bases=int(args.smooth_window_bases),
            model_window_smooth_bases=int(args.model_window_smooth_bases),
            window_boundary_bases=window_boundary,
            max_points_per_model=int(getattr(args, "plot_max_points", 200000)),
        )
    if args.plot_individual_windows and not compute_only:
        plot_window_bases = int(args.plot_window_bases or window_boundary or next(iter(model_summaries.values())).get("model_window_bases", 1024))
        individual_paths = plot_individual_windows(
            output_dir / "individual_windows",
            per_model=plot_payload,
            source_name=str(source_info["source"]),
            source_start=region_start,
            region_base_count=len(region_sequence),
            plot_window_bases=plot_window_bases,
            smooth_window_bases=int(args.smooth_window_bases),
            max_windows=int(args.max_individual_window_plots),
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
        "filtered_source_bases": int(source_length) if source_length is not None else None,
        "filtered_source_bases_known": source_length is not None,
        "filtered_source_bases_lower_bound": int(region_start + len(region_sequence)),
        "region_start": int(region_start),
        "requested_region_bases": int(args.region_bases),
        "region_bases": int(len(region_sequence)),
        "region_identity": region_identity(
            dataset=source_info["dataset"],
            source=source_info["source"],
            species=source_info.get("species"),
            alphabet=source_info["alphabet"],
            region_start=int(region_start),
            region_bases=int(len(region_sequence)),
        ),
        "random_region": bool(args.random_region),
        "seed": int(args.seed),
        "model_count": len(adapters),
        "models": model_summaries,
        "timing": {
            "source_length_seconds": _round_float(length_seconds),
            "region_read_seconds": _round_float(read_seconds),
            "total_wall_seconds": _round_float(perf_counter() - total_started),
        },
        "outputs": {
            "json": str(json_path),
            "model_artifact_dir": str(output_dir / "models"),
            "per_base_csv": str(per_base_csv) if args.write_per_base_csv else None,
            "windows_csv": str(windows_csv),
            "model_window_average_csv": str(model_window_csv),
            "curve_png": str(curve_png) if not compute_only else None,
            "individual_window_dir": str(output_dir / "individual_windows") if individual_paths else None,
            "individual_window_count": len(individual_paths),
            "record_dtype": record_dtype,
        },
    }
    if existing_result is not None:
        result["previous_model_count"] = len(existing_models)
    _write_compact_json(json_path, result)
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.plot_only:
        plot_jsons = _discover_plot_only_jsons(args)
        if args.combine_matching_regions:
            summaries = redraw_combined_matching_regions(args, plot_jsons)
        else:
            summaries = [redraw_from_result(args, path) for path in plot_jsons]
        if not summaries:
            raise FileNotFoundError("No region_bpb.json files found for --plot-only.")
        print(json.dumps({"plot_only": True, "results": summaries}, ensure_ascii=False))
        return

    full_jsons = _discover_full_result_jsons(args)
    if full_jsons:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when slicing from full-size results")
        output_root = Path(args.output_dir)
        multiple = len(full_jsons) > 1
        summaries = []
        for full_json in full_jsons:
            source_label = _safe_label(full_json.parent.name)
            output_dir = output_root / source_label if multiple else output_root
            result = run_probe_from_full_result(args, full_json_path=full_json, output_dir=output_dir)
            summaries.append(
                {
                    "source": result["source"],
                    "species": result.get("species"),
                    "region_signature": result.get("region_identity", {}).get("signature"),
                    "region_bpb_json": result["outputs"]["json"],
                    "curve_png": result["outputs"]["curve_png"],
                    "model_names": sorted(result["models"]),
                }
            )
        print(json.dumps({"sliced_from_full": True, "output_dir": str(output_root), "source_count": len(summaries), "results": summaries}, ensure_ascii=False))
        return

    if args.dataset is None:
        raise ValueError("--dataset is required unless --plot-only is used")
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --plot-only uses --result-json or --result-dir")
    alphabet = normalize_alphabet(args.alphabet)
    source_infos = resolve_region_sources(args, alphabet)
    adapters = build_region_adapters(args)
    output_root = Path(args.output_dir)
    multiple_sources = len(source_infos) > 1
    summaries: list[dict[str, Any]] = []
    for source_info in source_infos:
        source_label = _safe_label(str(source_info["source"]))
        output_dir = output_root / source_label if multiple_sources else output_root
        result = run_probe_for_source(args, source_info=source_info, adapters=adapters, output_dir=output_dir)
        summaries.append(
                {
                    "source": result["source"],
                    "species": result.get("species"),
                    "region_signature": result.get("region_identity", {}).get("signature"),
                    "region_bpb_json": result["outputs"]["json"],
                    "curve_png": result["outputs"]["curve_png"],
                    "model_names": sorted(result["models"]),
            }
        )

    if multiple_sources:
        summary_path = output_root / "region_bpb_batch_summary.json"
        _write_compact_json(summary_path, {"source_count": len(summaries), "results": summaries})
    print(json.dumps({"output_dir": str(output_root), "source_count": len(summaries), "results": summaries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
