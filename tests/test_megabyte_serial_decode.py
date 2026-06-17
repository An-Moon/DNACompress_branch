from __future__ import annotations

import unittest

import torch

from dna_compress.config import ModelConfig
from dna_compress.fast_arithmetic import load_fast_arithmetic_extension
from dna_compress.megabyte_batched_decode import MegabyteBatchedDecodeStepper, fast_floor_frequency_rows
from dna_compress.megabyte_loader import build_model
from dna_compress.megabyte_serial_decode import (
    MegabyteSerialDecoder,
    encode_symbols_with_serial_model,
    fast_floor_frequency_row,
)
from dna_compress.megabyte_window_codec import (
    compress_token_windows,
    decode_framed_token_windows,
    frame_compressed_streams,
    parse_length_prefixed_streams,
    pack_length_prefixed_streams,
)


def _small_model():
    config = ModelConfig(
        implementation="megabyte_in_action",
        vocab_size=17,
        patch_size=4,
        global_dim=8,
        local_dim=16,
        seq_length=8,
        global_heads=4,
        global_layers=2,
        local_heads=4,
        local_layers=2,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=False,
        pad_id=15,
        eos_id=16,
    )
    model = build_model(config)
    model.eval()
    return model


class MegabyteSerialDecodeTests(unittest.TestCase):
    def test_fast_floor_frequency_row_is_positive(self) -> None:
        logits = torch.tensor([0.0, float("nan"), -1000.0, 8.0], dtype=torch.float32)
        freqs = fast_floor_frequency_row(logits, total=1 << 15)

        self.assertEqual(freqs.shape, logits.shape)
        self.assertTrue(torch.all(freqs >= 1))
        self.assertGreater(int(freqs.sum().item()), 0)

    def test_fast_floor_frequency_rows_prefers_uint16(self) -> None:
        logits = torch.randn(3, 7)
        freqs = fast_floor_frequency_rows(logits, total=1 << 15)

        self.assertEqual(freqs.shape, logits.shape)
        self.assertEqual(freqs.dtype, torch.uint16)
        self.assertTrue(torch.all(freqs.to(torch.int32) >= 1))

    def test_cached_step_logits_match_full_forward(self) -> None:
        torch.manual_seed(3)
        model = _small_model()
        device = torch.device("cpu")
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long, device=device)

        with torch.no_grad():
            full_logits = model(ids).lm_logits[0]

        decoder = MegabyteSerialDecoder(
            model,
            device=device,
            dtype_name="float32",
            arithmetic_frequency_total=1 << 15,
        )
        cached_logits = []
        for symbol in ids[0].tolist():
            cached_logits.append(decoder.next_logits().detach().cpu())
            decoder.accept_symbol(symbol)

        torch.testing.assert_close(
            torch.stack(cached_logits),
            full_logits.detach().cpu(),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_batched_step_logits_match_serial_stepper(self) -> None:
        torch.manual_seed(4)
        model = _small_model()
        device = torch.device("cpu")
        ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
                [1, 3, 5, 7, 2, 4, 6, 8],
            ],
            dtype=torch.long,
            device=device,
        )
        batched = MegabyteBatchedDecodeStepper(
            model,
            batch_size=ids.shape[0],
            device=device,
            dtype_name="float32",
        )
        serial_decoders = [
            MegabyteSerialDecoder(
                model,
                device=device,
                dtype_name="float32",
                arithmetic_frequency_total=1 << 15,
            )
            for _ in range(ids.shape[0])
        ]

        for position in range(ids.shape[1]):
            batched_logits = batched.next_logits().detach().cpu()
            serial_logits = []
            for row, serial_decoder in enumerate(serial_decoders):
                serial_logits.append(serial_decoder.next_logits().detach().cpu())
                serial_decoder.accept_symbol(int(ids[row, position]))
            batched.accept_symbols(ids[:, position])
            torch.testing.assert_close(
                batched_logits,
                torch.stack(serial_logits),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_fast_floor_serial_roundtrip(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        torch.manual_seed(5)
        model = _small_model()
        device = torch.device("cpu")
        symbols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16]
        total = 1 << 15

        encoded, encode_timings = encode_symbols_with_serial_model(
            model,
            symbols,
            device=device,
            dtype_name="float32",
            arithmetic_frequency_total=total,
        )
        decoder = MegabyteSerialDecoder(
            model,
            device=device,
            dtype_name="float32",
            arithmetic_frequency_total=total,
        )
        decoded, decode_timings = decoder.decode(encoded, token_count=len(symbols))

        self.assertEqual(decoded, symbols)
        self.assertGreater(len(encoded), 0)
        self.assertGreater(encode_timings["tokens_per_second"], 0.0)
        self.assertGreater(decode_timings["tokens_per_second"], 0.0)

    def test_length_prefixed_window_framing_roundtrip(self) -> None:
        streams = [b"abc", b"", b"defgh"]

        framed = pack_length_prefixed_streams(streams)
        parsed = parse_length_prefixed_streams(framed)

        self.assertEqual(parsed, streams)

    def test_batched_window_codec_roundtrip(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        torch.manual_seed(6)
        model = _small_model()
        device = torch.device("cpu")
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
            ],
            dtype=torch.long,
        )
        total = 1 << 15

        streams, compression_metrics = compress_token_windows(
            model=model,
            tokens_cpu=tokens,
            batch_size=2,
            device=device,
            dtype_name="float32",
            frequency_total=total,
            compression_mode="cached",
        )
        framed, framing_metrics = frame_compressed_streams(streams)
        decoded, decode_metrics = decode_framed_token_windows(
            model=model,
            framed_payload=framed,
            window_count=tokens.shape[0],
            tokens_per_window=tokens.shape[1],
            batch_size=2,
            device=device,
            dtype_name="float32",
            frequency_total=total,
            expected_tokens_cpu=tokens,
        )

        self.assertTrue(torch.equal(decoded, tokens))
        self.assertEqual(decode_metrics["decode_mismatches"], 0)
        self.assertGreater(compression_metrics["compression_tokens_per_second"], 0.0)
        self.assertGreater(framing_metrics["framed_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
