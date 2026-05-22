from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import torch

from dna_compress.megadna_loader import (
    MEGADNA_WEIGHT_NAME,
    decode_megadna_tokens,
    default_megadna_weight_path,
    encode_megadna_source_bytes,
    encode_megadna_sequence,
    wrap_megadna_for_target_aligned_logits,
)


class MegaDNALoaderTests(unittest.TestCase):
    def test_encode_decode_uppercase_dna_round_trip(self) -> None:
        tokens = encode_megadna_sequence("ATCG")

        self.assertTrue(torch.equal(tokens, torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        self.assertEqual(decode_megadna_tokens(tokens), "ATCG")

    def test_encode_accepts_ascii_bytes(self) -> None:
        tokens = encode_megadna_sequence(b"GCTA")

        self.assertEqual(tokens.tolist(), [4, 3, 2, 1])

    def test_encode_rejects_ambiguous_bases_and_lowercase(self) -> None:
        for sequence in ("ANCG", "AtCG"):
            with self.subTest(sequence=sequence):
                with self.assertRaises(ValueError):
                    encode_megadna_sequence(sequence)

    def test_encode_rejects_empty_sequence(self) -> None:
        with self.assertRaises(ValueError):
            encode_megadna_sequence("")

    def test_decode_rejects_tokens_outside_vocab(self) -> None:
        with self.assertRaises(ValueError):
            decode_megadna_tokens([1, 6])

    def test_source_byte_encoding_can_filter_non_acgt(self) -> None:
        self.assertEqual(list(encode_megadna_source_bytes(b"ATNCG", strict=False)), [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            encode_megadna_source_bytes(b"ATNCG", strict=True)

    def test_default_weight_path_is_independent_of_cwd(self) -> None:
        expected = Path(__file__).resolve().parents[1] / "third_party" / "megaDNA" / "checkpoints" / MEGADNA_WEIGHT_NAME

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                self.assertEqual(default_megadna_weight_path(), expected)
            finally:
                os.chdir(original_cwd)


class _FakeMegaDNA(torch.nn.Module):
    def forward_empty(self, batch_size: int) -> torch.Tensor:
        logits = torch.zeros((batch_size, 1, 6), dtype=torch.float32)
        logits[:, :, 1] = 10.0
        return logits

    def forward(self, ids: torch.Tensor, return_value: str = "logits") -> torch.Tensor:
        del return_value
        batch_size, seq_length = ids.shape
        logits = torch.zeros((batch_size, seq_length, 6), dtype=torch.float32)
        for index in range(seq_length):
            logits[:, index, (index + 2) % 6] = 10.0
        return logits


class MegaDNATargetAlignedAdapterTests(unittest.TestCase):
    def test_adapter_prepends_start_logits_and_drops_final_next_token_logits(self) -> None:
        model = wrap_megadna_for_target_aligned_logits(_FakeMegaDNA())
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        output = model(ids, return_loss=True)

        self.assertEqual(tuple(output.lm_logits.shape), (1, 3, 6))
        self.assertEqual(output.lm_logits[0].argmax(dim=-1).tolist(), [1, 2, 3])
        self.assertIsNotNone(output.loss)


if __name__ == "__main__":
    unittest.main()
