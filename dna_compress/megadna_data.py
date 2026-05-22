from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .megadna_loader import MEGADNA_BASE_TO_TOKEN, MEGADNA_PAD_ID


NonACGTPolicy = str


@dataclass(frozen=True)
class EncodedMegaDNASource:
    original: bytes
    token_bytes: bytes
    dropped_bytes: int


def encode_source_for_megadna(source: bytes, *, non_acgt_policy: NonACGTPolicy) -> EncodedMegaDNASource:
    if non_acgt_policy not in {"reject", "filter"}:
        raise ValueError("non_acgt_policy must be one of: reject, filter")

    token_ids = bytearray()
    kept_original = bytearray()
    dropped = 0
    for index, byte_value in enumerate(source):
        token_id = MEGADNA_BASE_TO_TOKEN.get(chr(byte_value))
        if token_id is None:
            if non_acgt_policy == "filter":
                dropped += 1
                continue
            raise ValueError(
                "megaDNA only supports uppercase A/T/C/G source bytes; "
                f"found byte {byte_value!r} at position {index}."
            )
        token_ids.append(token_id)
        kept_original.append(byte_value)
    return EncodedMegaDNASource(original=bytes(kept_original), token_bytes=bytes(token_ids), dropped_bytes=dropped)


def encode_sources_for_megadna(
    sources: list[bytes],
    *,
    non_acgt_policy: NonACGTPolicy,
) -> tuple[list[EncodedMegaDNASource], dict[str, int]]:
    encoded = [encode_source_for_megadna(source, non_acgt_policy=non_acgt_policy) for source in sources]
    usable = [source for source in encoded if len(source.token_bytes) > 0]
    return usable, {
        "source_count": len(sources),
        "usable_source_count": len(usable),
        "dropped_bytes": sum(source.dropped_bytes for source in encoded),
        "encoded_bases": sum(len(source.token_bytes) for source in usable),
    }


class RandomMegaDNAWindowDataset(Dataset):
    def __init__(
        self,
        sources: list[EncodedMegaDNASource],
        *,
        seq_length: int,
        samples_per_epoch: int,
        seed: int,
        sampling_strategy: str = "proportional",
    ) -> None:
        self.sources = [
            np.frombuffer(source.token_bytes, dtype=np.uint8).astype(np.int64, copy=True)
            for source in sources
            if len(source.token_bytes) >= seq_length
        ]
        if not self.sources:
            raise ValueError("no megaDNA train sources are long enough for the configured seq_length")
        self.seq_length = seq_length
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self.available = [int(source.shape[0]) - seq_length + 1 for source in self.sources]
        self.sampling_strategy = sampling_strategy
        if sampling_strategy == "proportional":
            self.source_weights = [float(count) for count in self.available]
        elif sampling_strategy == "uniform":
            self.source_weights = [1.0 for _ in self.available]
        elif sampling_strategy == "sqrt":
            self.source_weights = [float(count) ** 0.5 for count in self.available]
        else:
            raise ValueError("sampling_strategy must be one of: proportional, uniform, sqrt")
        if sum(self.source_weights) <= 0:
            raise ValueError("sampling source weights must sum to > 0")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + index)
        source_index = rng.choices(range(len(self.sources)), weights=self.source_weights, k=1)[0]
        source = self.sources[source_index]
        start = rng.randrange(self.available[source_index])
        window = source[start : start + self.seq_length]
        return {"input_ids": torch.as_tensor(window, dtype=torch.long)}


class SequentialMegaDNAWindowDataset(Dataset):
    def __init__(
        self,
        sources: list[EncodedMegaDNASource],
        *,
        seq_length: int,
        pad_id: int = MEGADNA_PAD_ID,
    ) -> None:
        self.sources = [
            np.frombuffer(source.token_bytes, dtype=np.uint8).astype(np.int64, copy=True)
            for source in sources
            if len(source.token_bytes) > 0
        ]
        self.seq_length = seq_length
        self.pad_id = pad_id
        self.index: list[tuple[int, int]] = []
        for source_index, source in enumerate(self.sources):
            if source.shape[0] <= seq_length:
                self.index.append((source_index, 0))
                continue
            for start in range(0, int(source.shape[0]), seq_length):
                self.index.append((source_index, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        source_index, start = self.index[index]
        source = self.sources[source_index]
        chunk = source[start : start + self.seq_length]
        ids = torch.full((self.seq_length,), self.pad_id, dtype=torch.long)
        ids[: chunk.shape[0]] = torch.as_tensor(chunk, dtype=torch.long)
        return {"input_ids": ids}
