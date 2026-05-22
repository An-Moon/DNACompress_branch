from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dna_compress.megadna_loader import (
    MEGADNA_VOCAB,
    decode_megadna_tokens,
    default_megadna_weight_path,
    encode_megadna_sequence,
    load_megadna_model,
)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test the isolated official megaDNA 145M deployment.")
    parser.add_argument("--weight", default=str(default_megadna_weight_path()), help="Path to megaDNA .pt checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device, or 'auto'.")
    parser.add_argument("--primer", default="ATCGATCG", help="Uppercase A/T/C/G primer sequence.")
    parser.add_argument("--generate-len", type=int, default=0, help="If >0, run model.generate to this total length.")
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--filter-thres", type=float, default=0.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device = _resolve_device(args.device)
    weight_path = Path(args.weight)

    model = load_megadna_model(weight_path, device=device)
    input_ids = encode_megadna_sequence(args.primer, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids, return_value="logits")

    print(f"weight_path={weight_path}")
    print(f"device={device}")
    print(f"parameters={_parameter_count(model)}")
    print(f"vocab={list(MEGADNA_VOCAB)}")
    print(f"input_shape={tuple(input_ids.shape)} input_min={int(input_ids.min())} input_max={int(input_ids.max())}")
    print(f"logits_shape={tuple(logits.shape)}")

    if args.generate_len > 0:
        if args.generate_len < input_ids.shape[-1]:
            raise ValueError("--generate-len must be >= primer length.")
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                seq_len=args.generate_len,
                temperature=args.temperature,
                filter_thres=args.filter_thres,
            )
        print(f"generated_shape={tuple(generated.shape)}")
        print(f"generated={decode_megadna_tokens(generated[0])}")


if __name__ == "__main__":
    main()
