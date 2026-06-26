from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from dna_compress.config import ModelConfig
from dna_compress.fast_arithmetic import (
    BatchedStreamingArithmeticDecoder,
    BatchedStreamingArithmeticEncoder,
    load_fast_arithmetic_extension,
)
from dna_compress.megabyte_batched_decode import MegabyteBatchedDecodeStepper, fast_floor_frequency_rows
from dna_compress.megabyte_loader import build_model
from dna_compress.megabyte_serial_decode import (
    MegabyteSerialDecoder,
    encode_symbols_with_serial_model,
    fast_floor_frequency_row,
)
from dna_compress.megabyte_window_codec import (
    TokenWindowBatch,
    WindowCodecPipeline,
    batch_aligned_window_ranges,
    compress_token_windows,
    decode_window_payload_with_pipeline,
    decode_framed_token_windows,
    frame_compressed_streams,
    pack_v3_window_payload,
    pack_v2_window_payload,
    parse_v3_window_payload,
    parse_v2_window_payload,
    parse_length_prefixed_streams,
    pack_length_prefixed_streams,
    valid_lengths_from_logical_token_count,
)


def _small_model():
    config = _small_model_config()
    model = build_model(config)
    model.eval()
    return model


def _small_model_config():
    return ModelConfig(
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

    def test_batch_aligned_window_ranges(self) -> None:
        self.assertEqual(batch_aligned_window_ranges(0, 4, 2), [])
        self.assertEqual(batch_aligned_window_ranges(3, 4, 8), [(0, 3)])
        self.assertEqual(batch_aligned_window_ranges(16, 4, 2), [(0, 8), (8, 16)])
        self.assertEqual(batch_aligned_window_ranges(17, 4, 2), [(0, 8), (8, 17)])
        self.assertEqual(batch_aligned_window_ranges(25, 4, 3), [(0, 8), (8, 16), (16, 25)])

        for window_count, batch_size, shard_count in ((17, 4, 2), (31, 8, 4), (9, 3, 5)):
            ranges = batch_aligned_window_ranges(window_count, batch_size, shard_count)
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], window_count)
            for left, right in zip(ranges, ranges[1:]):
                self.assertEqual(left[1], right[0])
                self.assertEqual(left[1] % batch_size, 0)
            for start, end in ranges[:-1]:
                self.assertEqual(start % batch_size, 0)
                self.assertEqual(end % batch_size, 0)

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

    def test_length_aware_arithmetic_roundtrip(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        lows = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.int32)
        highs = lows + 1
        totals = torch.full_like(lows, 8)
        lengths = torch.tensor([4, 2], dtype=torch.long)

        encoder = BatchedStreamingArithmeticEncoder(2)
        timings = encoder.encode_interval_matrix_with_lengths(lows, highs, totals, lengths)
        streams = encoder.finish()
        decoder = BatchedStreamingArithmeticDecoder(streams)

        decoded_steps = []
        for step in range(4):
            active = lengths > step
            if not bool(active.any().item()):
                break
            decoded = decoder.decode_frequency_rows_with_totals_and_active(
                torch.ones((2, 8), dtype=torch.uint16),
                torch.full((2,), 8, dtype=torch.int32),
                active,
            )
            decoded_steps.append((step, active, decoded))

        self.assertEqual(timings.emitted_count, 6)
        self.assertEqual([int(x) for x in decoded_steps[0][2]], [0, 1])
        self.assertEqual([int(x) for x in decoded_steps[1][2]], [1, 2])
        self.assertEqual([int(x) for x in decoded_steps[2][2]], [2])
        self.assertEqual([int(x) for x in decoded_steps[3][2]], [3])

    def test_v3_window_payload_roundtrip_is_compact(self) -> None:
        streams = [b"abc", b"", b"defgh"]
        header = {
            "tokens_per_window": 8,
            "window_count": 3,
            "logical_token_count": 17,
            "base_token_count": 16,
            "token_merge_size": 3,
            "frequency_total": 1 << 15,
            "compression_batch_size": 2,
        }

        v2 = pack_v2_window_payload(streams, {"format_version": 2, **header})
        v3 = pack_v3_window_payload(streams, header)
        parsed_header, parsed_streams = parse_v3_window_payload(v3)

        self.assertEqual(parsed_streams, streams)
        self.assertEqual(parsed_header["format_version"], 3)
        self.assertEqual(parsed_header["logical_token_count"], 17)
        self.assertLess(len(v3), len(v2))

    def test_v2_window_codec_pipeline_roundtrip(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        torch.manual_seed(7)
        model = _small_model()
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
                [1, 3, 5, 7, 2, 4, 6, 8],
            ],
            dtype=torch.long,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "best.pt"
            torch.save({"model_state": model.state_dict()}, checkpoint_path)
            config = SimpleNamespace(model=_small_model_config())
            with WindowCodecPipeline(
                config=config,
                checkpoint_path=checkpoint_path,
                devices=["cpu"],
                dtype_name="float32",
                frequency_total=1 << 15,
                batch_size=2,
                compression_mode="cached",
            ) as pipeline:
                streams, compression_metrics = pipeline.compress_batches(
                    [
                        TokenWindowBatch(0, tokens[:2].contiguous()),
                        TokenWindowBatch(2, tokens[2:].contiguous()),
                    ]
                )
                payload = pack_v2_window_payload(
                    streams,
                    {
                        "format_version": 2,
                        "codec": "megabyte_window_fast_floor",
                        "window_count": int(tokens.shape[0]),
                        "tokens_per_window": int(tokens.shape[1]),
                        "compression_batch_size": 2,
                        "compression_mode": "cached",
                        "frequency_total": 1 << 15,
                    },
                )
                header, parsed_streams = parse_v2_window_payload(payload)
                decoded, decode_metrics = pipeline.decode_streams(
                    streams=parsed_streams,
                    tokens_per_window=int(header["tokens_per_window"]),
                    expected_tokens_cpu=tokens,
                )

        self.assertEqual(header["window_count"], tokens.shape[0])
        self.assertTrue(torch.equal(decoded, tokens))
        self.assertGreater(compression_metrics["compression_tokens_per_second"], 0.0)
        self.assertEqual(decode_metrics["decode_mismatches"], 0)

    def test_v3_window_codec_partial_tail_does_not_encode_pad(self) -> None:
        try:
            load_fast_arithmetic_extension()
        except Exception as error:
            self.skipTest(f"fast arithmetic extension is unavailable: {error}")

        torch.manual_seed(8)
        model = _small_model()
        pad_id = _small_model_config().pad_id
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [9, 10, 16, pad_id, pad_id, pad_id, pad_id, pad_id],
            ],
            dtype=torch.long,
        )
        valid_lengths = torch.tensor([8, 3], dtype=torch.long)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "best.pt"
            torch.save({"model_state": model.state_dict()}, checkpoint_path)
            config = SimpleNamespace(model=_small_model_config())
            with WindowCodecPipeline(
                config=config,
                checkpoint_path=checkpoint_path,
                devices=["cpu"],
                dtype_name="float32",
                frequency_total=1 << 15,
                batch_size=2,
                compression_mode="cached",
            ) as pipeline:
                streams, compression_metrics = pipeline.compress_batches(
                    [TokenWindowBatch(0, tokens.contiguous(), valid_lengths=valid_lengths)]
                )
                payload = pack_v3_window_payload(
                    streams,
                    {
                        "tokens_per_window": int(tokens.shape[1]),
                        "window_count": int(tokens.shape[0]),
                        "logical_token_count": int(valid_lengths.sum().item()),
                        "base_token_count": int(valid_lengths.sum().item() - 1),
                        "token_merge_size": 3,
                        "frequency_total": 1 << 15,
                        "compression_batch_size": 2,
                    },
                )
                decoded, decode_metrics, header = decode_window_payload_with_pipeline(
                    pipeline=pipeline,
                    payload=payload,
                    expected_tokens_cpu=tokens,
                )

        self.assertEqual(header["format_version"], 3)
        self.assertEqual(compression_metrics["compression_emitted_symbols"], 11)
        self.assertTrue(torch.equal(decoded, tokens))
        self.assertEqual(decode_metrics["decode_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
