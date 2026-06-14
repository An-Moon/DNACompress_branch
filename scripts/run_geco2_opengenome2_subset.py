from __future__ import annotations

"""Run GeCo2 on a directory of per-source FASTA files and emit plot-ready metrics.

Example:
    python scripts/run_geco2_fasta_subset.py \
      --input-dir /data/students/Liang_junnan/opengenome2_subset/fasta_test_subset_100mb_per_source \
      --output-dir outputs/dna_geco2_opengenome2_subset_100mb_per_source \
      --level 5
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_compression_curves import generate_artifacts_for_compression_compare


DEFAULT_INPUT_DIR = Path("/data/students/Liang_junnan/opengenome2_subset/fasta_test_subset_100mb_per_source")
DEFAULT_OUTPUT_DIR = Path("outputs/opengenome2_geco2_fasta_test_subset_100mb_per_source")
DEFAULT_MODE_NAME = "geco2_level5"


def _tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _resolve_geco2_binary(requested_binary: str) -> str:
    resolved = shutil.which(requested_binary)
    if resolved is None:
        raise FileNotFoundError(f"Could not find GeCo2 binary: {requested_binary}")
    return resolved


def _build_geco2_command(binary: str, *, level: int, input_path: Path) -> list[str]:
    return [binary, "-F", "-v", "-l", str(level), str(input_path)]


def _run_geco2_on_file(
    *,
    binary: str,
    source_path: Path,
    level: int,
    temp_root: Path | None,
    keep_temp: bool,
) -> dict[str, Any]:
    if keep_temp:
        temp_dir = Path(tempfile.mkdtemp(prefix="geco2_fasta_", dir=str(temp_root) if temp_root else None))
        cleanup_ctx = None
    else:
        cleanup_ctx = tempfile.TemporaryDirectory(prefix="geco2_fasta_", dir=str(temp_root) if temp_root else None)
        temp_dir = Path(cleanup_ctx.__enter__())
    try:
        staged = temp_dir / source_path.name
        shutil.copyfile(source_path, staged)
        command = _build_geco2_command(binary, level=level, input_path=staged)
        started = perf_counter()
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        elapsed = perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"GeCo2 failed (rc={completed.returncode}) on {source_path}: "
                f"{_tail_text(completed.stderr or completed.stdout)}"
            )
        compressed_path = Path(str(staged) + ".co")
        if not compressed_path.exists():
            candidates = sorted(temp_dir.glob("*.co"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"GeCo2 did not create a unique .co for {source_path}: {candidates}")
            compressed_path = candidates[0]
        return {
            "command": command,
            "returncode": completed.returncode,
            "seconds": elapsed,
            "stdout_tail": _tail_text(completed.stdout),
            "stderr_tail": _tail_text(completed.stderr),
            "compressed_bytes": compressed_path.stat().st_size,
            "compressed_path": str(compressed_path) if keep_temp else None,
            "temp_dir": str(temp_dir) if keep_temp else None,
        }
    finally:
        if cleanup_ctx is not None:
            cleanup_ctx.__exit__(None, None, None)


def _optional_int_sum(rows: list[dict[str, Any]], key: str) -> int | None:
    values = [row.get(key) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def _summarize_per_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_sample_bytes = sum(int(row["sample_bytes"]) for row in rows)
    total_sample_bases = sum(int(row["sample_bases"]) for row in rows)
    total_theoretical_bits = sum(float(row["theoretical_bits"]) for row in rows)
    total_arithmetic_bytes = sum(int(row["arithmetic_coded_bytes"]) for row in rows)
    total_seconds = sum(float(row.get("compression_process_seconds", 0.0)) for row in rows)
    total_symbols = sum(int(row.get("emitted_arithmetic_symbol_count", 0) or 0) for row in rows)
    return {
        "source_count": len(rows),
        "total_sample_bytes": total_sample_bytes,
        "total_sample_bases": total_sample_bases,
        "total_theoretical_bits": total_theoretical_bits,
        "total_theoretical_bits_per_base": total_theoretical_bits / max(total_sample_bases, 1),
        "total_arithmetic_coded_bytes": total_arithmetic_bytes,
        "total_arithmetic_bits_per_base": (total_arithmetic_bytes * 8) / max(total_sample_bases, 1),
        "total_ascii_bytes": sum(int(row["ascii_bytes"]) for row in rows),
        "total_two_bit_pack_bytes": sum(int(row["two_bit_pack_bytes"]) for row in rows),
        "total_gzip_bytes": _optional_int_sum(rows, "gzip_bytes"),
        "total_bz2_bytes": _optional_int_sum(rows, "bz2_bytes"),
        "total_lzma_bytes": _optional_int_sum(rows, "lzma_bytes"),
        "total_compression_process_seconds": total_seconds,
        "total_compression_bytes_per_second": total_sample_bytes / max(total_seconds, 1e-12),
        "total_compression_bases_per_second": total_sample_bases / max(total_seconds, 1e-12),
        "total_emitted_arithmetic_symbol_count": total_symbols,
        "total_compression_symbols_per_second": total_symbols / max(total_seconds, 1e-12),
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


def _read_manifest(input_dir: Path) -> dict[str, Any] | None:
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _safe_source_filename(source: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in source)
    return f"{safe}.fasta"


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


def _count_fasta_acgt_bases(path: Path) -> int:
    count = 0
    in_header = False
    at_line_start = True
    valid = set(b"ACGTacgt")
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            for byte in chunk:
                if at_line_start and byte == ord(">"):
                    in_header = True
                if byte in (10, 13):
                    in_header = False
                    at_line_start = True
                    continue
                if in_header:
                    at_line_start = False
                    continue
                if byte in valid:
                    count += 1
                if byte not in (32, 9):
                    at_line_start = False
    return count


def _source_row(
    *,
    source: str,
    path: Path,
    manifest_entry: dict[str, Any],
    result: dict[str, Any],
    level: int,
    binary: str,
) -> dict[str, Any]:
    sample_bytes = int(path.stat().st_size)
    sample_bases = _count_fasta_acgt_bases(path)
    compressed_bytes = int(result["compressed_bytes"])
    compressed_bits = float(compressed_bytes * 8)
    seconds = float(result["seconds"])
    return {
        "species": source,
        "source_name": source,
        "source_path": str(path),
        "mode": f"geco2_level{level}",
        "geco2_level": level,
        "geco2_binary": binary,
        "geco2_command": result["command"],
        "geco2_returncode": result["returncode"],
        "geco2_stdout_tail": result["stdout_tail"],
        "geco2_stderr_tail": result["stderr_tail"],
        "sample_bytes": sample_bytes,
        "sample_bases": sample_bases,
        "sample_symbols_with_eos": sample_bases,
        "uses_eos": False,
        "theoretical_bits": compressed_bits,
        "theoretical_bits_per_base": compressed_bits / max(sample_bases, 1),
        "arithmetic_coded_bytes": compressed_bytes,
        "arithmetic_bits_per_base": compressed_bits / max(sample_bases, 1),
        "arithmetic_coding_mode": f"geco2_level{level}",
        "arithmetic_merge_size": 1,
        "emitted_arithmetic_symbol_count": sample_bases,
        "compression_process_seconds": seconds,
        "compression_bytes_per_second": sample_bytes / max(seconds, 1e-12),
        "compression_bases_per_second": sample_bases / max(seconds, 1e-12),
        "compression_symbols_per_second": sample_bases / max(seconds, 1e-12),
        "ascii_bytes": sample_bases,
        "two_bit_pack_bytes": (sample_bases * 2 + 7) // 8,
        "gzip_bytes": None,
        "bz2_bytes": None,
        "lzma_bytes": None,
        "manifest_selected_record_count": manifest_entry.get("selected_record_count"),
        "manifest_selected_bytes": manifest_entry.get("selected_bytes"),
        "manifest_bytes_written": manifest_entry.get("bytes_written"),
    }


def run_geco2_fasta_subset(
    *,
    input_dir: Path,
    output_dir: Path,
    binary: str,
    level: int,
    split_name: str,
    temp_root: Path | None,
    keep_temp: bool,
) -> dict[str, Any]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    manifest = _read_manifest(input_dir)
    sources = _source_files(input_dir, manifest)
    if not sources:
        raise FileNotFoundError(f"no .fasta files found under {input_dir}")

    mode_name = f"geco2_level{level}"
    per_source: list[dict[str, Any]] = []
    for index, (source, path, entry) in enumerate(sources, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"missing FASTA for source {source}: {path}")
        print(
            f"[geco2_fasta] {index}/{len(sources)} source={source} level={level} "
            f"bytes={path.stat().st_size}",
            flush=True,
        )
        result = _run_geco2_on_file(
            binary=binary,
            source_path=path,
            level=level,
            temp_root=temp_root,
            keep_temp=keep_temp,
        )
        per_source.append(
            _source_row(
                source=source,
                path=path,
                manifest_entry=entry,
                result=result,
                level=level,
                binary=binary,
            )
        )

    aggregate = _summarize_per_source(per_source)
    aggregate["geco2_level"] = level
    aggregate["geco2_binary"] = binary
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "run_name": output_dir.name,
        "input_dir": str(input_dir),
        "manifest_path": str(input_dir / "manifest.json") if (input_dir / "manifest.json").exists() else None,
        "geco2_baseline": {
            "mode": mode_name,
            "level": level,
            "binary": binary,
            "input_dir": str(input_dir),
            "split": split_name,
        },
        "dataset": {
            "dataset_dir": str(input_dir),
            "sequence_source_mode": "fasta_dir",
            "species": [
                {
                    "species": row["species"],
                    "source_name": row["source_name"],
                    "path": row["source_path"],
                    "total_size": row["sample_bases"],
                    "file_size_bytes": row["sample_bytes"],
                }
                for row in per_source
            ],
        },
        "results": {
            split_name: {
                mode_name: {
                    "aggregate": aggregate,
                    "per_source": per_source,
                }
            }
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "compression_compare.json"
    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_csv(output_dir / "geco2_fasta_per_source.csv", [{**row, "split": split_name} for row in per_source])
    _write_csv(output_dir / "geco2_fasta_aggregate.csv", [{"split": split_name, "mode": mode_name, **aggregate}])
    generated = generate_artifacts_for_compression_compare(output_json)
    print(f"[geco2_fasta] saved metrics to {output_json}", flush=True)
    print(f"[geco2_fasta] generated {len(generated)} plot/table artifacts", flush=True)
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GeCo2 on per-source FASTA files and generate compression plots.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--geco2-bin", default="GeCo2")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--temp-root")
    parser.add_argument("--keep-temp", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    binary = _resolve_geco2_binary(args.geco2_bin)
    metrics = run_geco2_fasta_subset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        binary=binary,
        level=args.level,
        split_name=args.split_name,
        temp_root=Path(args.temp_root) if args.temp_root else None,
        keep_temp=args.keep_temp,
    )
    aggregate = metrics["results"][args.split_name][f"geco2_level{args.level}"]["aggregate"]
    print(
        "[geco2_fasta] aggregate: "
        f"sources={aggregate['source_count']} "
        f"bases={aggregate['total_sample_bases']} "
        f"compressed_bytes={aggregate['total_arithmetic_coded_bytes']} "
        f"bpb={aggregate['total_arithmetic_bits_per_base']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
