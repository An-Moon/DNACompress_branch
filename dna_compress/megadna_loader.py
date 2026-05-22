from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Iterable

import torch
import torch.nn.functional as F


MEGADNA_REPO_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "megaDNA"
MEGADNA_WEIGHT_NAME = "megaDNA_phage_145M.pt"
MEGADNA_VOCAB = ("**", "A", "T", "C", "G", "#")
MEGADNA_BASE_TO_TOKEN = {"A": 1, "T": 2, "C": 3, "G": 4}
MEGADNA_DNA_TOKEN_IDS = frozenset(MEGADNA_BASE_TO_TOKEN.values())
MEGADNA_PAD_ID = 0
MEGADNA_EOS_ID = 5


def ensure_megadna_on_path() -> None:
    repo_path = str(MEGADNA_REPO_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


def default_megadna_weight_path() -> Path:
    return MEGADNA_REPO_ROOT / "checkpoints" / MEGADNA_WEIGHT_NAME


def load_megadna_model(
    path: str | Path | None = None,
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    ensure_megadna_on_path()
    checkpoint_path = Path(path) if path is not None else default_megadna_weight_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"megaDNA checkpoint not found: {checkpoint_path}")

    map_location = torch.device(device)
    try:
        model = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"Expected megaDNA checkpoint to contain a torch.nn.Module, got {type(model)!r}.")
    model.eval()
    return model


def _coerce_sequence_text(sequence: str | bytes) -> str:
    if isinstance(sequence, bytes):
        return sequence.decode("ascii")
    if isinstance(sequence, str):
        return sequence
    raise TypeError(f"megaDNA sequence must be str or bytes, got {type(sequence)!r}.")


def encode_megadna_sequence(
    sequence: str | bytes,
    *,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    text = _coerce_sequence_text(sequence)
    if not text:
        raise ValueError("megaDNA sequence must not be empty.")

    token_ids: list[int] = []
    for index, base in enumerate(text):
        try:
            token_ids.append(MEGADNA_BASE_TO_TOKEN[base])
        except KeyError as error:
            raise ValueError(
                f"megaDNA only supports uppercase A/T/C/G bases; found {base!r} at position {index}."
            ) from error

    return torch.tensor(token_ids, dtype=torch.long, device=device)


def encode_megadna_source_bytes(source: bytes, *, strict: bool = True) -> bytes:
    token_ids = bytearray()
    for index, byte_value in enumerate(source):
        try:
            token_ids.append(MEGADNA_BASE_TO_TOKEN[chr(byte_value)])
        except KeyError as error:
            if not strict:
                continue
            raise ValueError(
                "megaDNA only supports uppercase A/T/C/G source bytes; "
                f"found byte {byte_value!r} at position {index}."
            ) from error
    return bytes(token_ids)


def decode_megadna_tokens(tokens: Iterable[int] | torch.Tensor) -> str:
    if isinstance(tokens, torch.Tensor):
        token_values = tokens.detach().cpu().reshape(-1).tolist()
    else:
        token_values = list(tokens)

    decoded: list[str] = []
    for index, token in enumerate(token_values):
        token_id = int(token)
        if token_id < 0 or token_id >= len(MEGADNA_VOCAB):
            raise ValueError(f"megaDNA token id at position {index} is outside the vocabulary: {token_id}.")
        decoded.append(MEGADNA_VOCAB[token_id])
    return "".join(decoded)


class MegaDNATargetAlignedModel(torch.nn.Module):
    """Expose megaDNA logits aligned with each target position.

    Official megaDNA `return_value="logits"` follows generation semantics:
    `logits[:, -1]` predicts the next token after the input prefix. Existing
    compression/evaluation code in this repository expects `lm_logits[:, i]`
    to score the token at input position `i`. This wrapper prepends the model's
    start-token logits and drops the final next-token row.
    """

    vocab_size = len(MEGADNA_VOCAB)
    pad_id = MEGADNA_PAD_ID
    eos_id = MEGADNA_EOS_ID

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor, return_loss: bool = False):
        if ids.ndim != 2:
            raise ValueError(f"megaDNA adapter expects [batch, seq] ids, got shape {tuple(ids.shape)}.")
        if ids.shape[-1] == 0:
            raise ValueError("megaDNA adapter does not support empty input windows.")

        next_token_logits = self.model(ids, return_value="logits")
        start_logits = self.model.forward_empty(ids.shape[0])[:, :1, :]
        if next_token_logits.shape[1] > 1:
            lm_logits = torch.cat((start_logits, next_token_logits[:, :-1, :]), dim=1)
        else:
            lm_logits = start_logits

        loss = None
        if return_loss:
            loss = F.cross_entropy(
                lm_logits.reshape(-1, lm_logits.shape[-1]),
                ids.reshape(-1),
                ignore_index=self.pad_id,
            )
        return SimpleNamespace(lm_logits=lm_logits, loss=loss)


def wrap_megadna_for_target_aligned_logits(model: torch.nn.Module) -> MegaDNATargetAlignedModel:
    return MegaDNATargetAlignedModel(model)
