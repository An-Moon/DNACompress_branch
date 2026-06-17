from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter

import torch
from einops import rearrange

from .fast_arithmetic import (
    StreamingArithmeticDecoder,
    StreamingArithmeticEncoder,
    fast_floor_intervals_from_probabilities,
)


@dataclass
class AttentionCache:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None


@dataclass
class MegabyteSerialDecodeTimings:
    global_step_seconds: float = 0.0
    local_step_seconds: float = 0.0
    logits_seconds: float = 0.0
    quantize_seconds: float = 0.0
    transfer_seconds: float = 0.0
    arithmetic_seconds: float = 0.0

    @property
    def model_seconds(self) -> float:
        return self.global_step_seconds + self.local_step_seconds + self.logits_seconds

    @property
    def total_seconds(self) -> float:
        return (
            self.global_step_seconds
            + self.local_step_seconds
            + self.logits_seconds
            + self.quantize_seconds
            + self.transfer_seconds
            + self.arithmetic_seconds
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "global_step_seconds": self.global_step_seconds,
            "local_step_seconds": self.local_step_seconds,
            "logits_seconds": self.logits_seconds,
            "quantize_seconds": self.quantize_seconds,
            "transfer_seconds": self.transfer_seconds,
            "arithmetic_seconds": self.arithmetic_seconds,
            "model_seconds": self.model_seconds,
            "total_accounted_seconds": self.total_seconds,
        }


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _attention_step(attention, x: torch.Tensor, cache: AttentionCache, *, alibi_slopes: torch.Tensor) -> tuple[torch.Tensor, AttentionCache]:
    heads = attention.heads
    x_norm = attention.norm(x)
    q = attention.to_q(x_norm)
    k_new, v_new = attention.to_kv(x_norm).chunk(2, dim=-1)
    q = rearrange(q, "b n (h d) -> b h n d", h=heads)
    if cache.k is None:
        k = k_new
        v = v_new
    else:
        k = torch.cat((cache.k, k_new), dim=1)
        v = torch.cat((cache.v, v_new), dim=1)
    out = attention.attend(q, k, v, alibi_slopes=alibi_slopes)
    out = rearrange(out, "b h n d -> b n (h d)")
    return attention.to_out(out), AttentionCache(k=k, v=v)


def _transformer_step(transformer, x: torch.Tensor, caches: list[AttentionCache] | None) -> tuple[torch.Tensor, list[AttentionCache]]:
    if caches is None:
        caches = [AttentionCache() for _ in transformer.layers]
    if len(caches) != len(transformer.layers):
        raise ValueError("cache layer count does not match transformer layer count")
    alibi_slopes = transformer.alibi_slopes.to(device=x.device, dtype=torch.float32)
    next_caches: list[AttentionCache] = []
    for layer_cache, (attn, ff) in zip(caches, transformer.layers):
        attn_out, next_cache = _attention_step(attn, x, layer_cache, alibi_slopes=alibi_slopes)
        x = attn_out + x
        x = ff(x) + x
        next_caches.append(next_cache)
    return x, next_caches


def fast_floor_frequency_row(logits: torch.Tensor, *, total: int) -> torch.Tensor:
    if logits.dim() != 1:
        raise ValueError("logits must be a 1D tensor")
    probs = torch.softmax(logits.float(), dim=-1)
    probs = torch.where(torch.isfinite(probs) & (probs > 0), probs, torch.zeros_like(probs))
    freqs = torch.floor(probs * int(total)).clamp_min(1).to(torch.int64)
    if int(freqs.sum().item()) <= 0:
        freqs = torch.ones_like(freqs, dtype=torch.int64)
    return freqs.contiguous()


class MegabyteSerialDecoder:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        dtype_name: str = "float32",
        arithmetic_frequency_total: int,
    ) -> None:
        if model.__class__.__name__ != "Megabyte":
            raise ValueError("MegabyteSerialDecoder expects a megabyte_in_action Megabyte model")
        self.model = model
        self.device = device
        self.dtype_name = dtype_name
        self.total = int(arithmetic_frequency_total)
        self.config = model.config
        self.P = int(self.config.P)
        self.V = int(self.config.V)
        self.D_G = int(self.config.D_G)
        self.pad_id = int(self.config.pad_id)
        self.timings = MegabyteSerialDecodeTimings()
        self.reset_window()

    def reset_window(self) -> None:
        self.global_caches: list[AttentionCache] | None = None
        self.local_caches: list[AttentionCache] | None = None
        self.current_patch_tokens: list[int] = []
        self.previous_patch_tokens: list[int] | None = None
        self.current_patch_local_context: torch.Tensor | None = None
        self.token_index = 0

    def _embed_patch(self, tokens: list[int] | None) -> torch.Tensor:
        if tokens is None:
            ids = torch.full((1, self.P), self.pad_id, dtype=torch.long, device=self.device)
        else:
            if len(tokens) != self.P:
                raise ValueError("global patch embedding requires a full previous patch")
            ids = torch.tensor(tokens, dtype=torch.long, device=self.device).view(1, self.P)
        embedded = self.model.to_embed(ids)
        return rearrange(embedded, "b p d -> b (p d)")

    def _start_patch(self) -> None:
        _sync_if_cuda(self.device)
        started = perf_counter()
        global_in = self._embed_patch(self.previous_patch_tokens).unsqueeze(1)
        with torch.no_grad(), _autocast_context(self.device, self.dtype_name):
            global_out, self.global_caches = _transformer_step(
                self.model.g_transformer,
                global_in,
                self.global_caches,
            )
            local_context = self.model.gl_linear(global_out).view(1, self.P, self.config.D_L)
        _sync_if_cuda(self.device)
        self.timings.global_step_seconds += perf_counter() - started
        self.current_patch_local_context = local_context
        self.local_caches = None
        self.current_patch_tokens = []

    def next_logits(self) -> torch.Tensor:
        local_index = self.token_index % self.P
        if local_index == 0 or self.current_patch_local_context is None:
            self._start_patch()

        if self.token_index == 0:
            prev_symbol = self.pad_id
        elif local_index == 0:
            if self.previous_patch_tokens is None:
                raise RuntimeError("missing previous patch at patch boundary")
            prev_symbol = self.previous_patch_tokens[-1]
        else:
            prev_symbol = self.current_patch_tokens[-1]

        token_id = torch.tensor([[prev_symbol]], dtype=torch.long, device=self.device)
        _sync_if_cuda(self.device)
        started = perf_counter()
        with torch.no_grad(), _autocast_context(self.device, self.dtype_name):
            prev_embed = self.model.to_embed(token_id)
            local_embed = self.model.to_l_embed(prev_embed)
            local_in = self.current_patch_local_context[:, local_index : local_index + 1, :] + local_embed
            local_out, self.local_caches = _transformer_step(
                self.model.l_transformer,
                local_in,
                self.local_caches,
            )
        _sync_if_cuda(self.device)
        self.timings.local_step_seconds += perf_counter() - started

        _sync_if_cuda(self.device)
        started = perf_counter()
        with torch.no_grad(), _autocast_context(self.device, self.dtype_name):
            logits = self.model.to_logits(local_out[:, -1, :]).squeeze(0)
        _sync_if_cuda(self.device)
        self.timings.logits_seconds += perf_counter() - started
        return logits

    def accept_symbol(self, symbol: int) -> None:
        self.current_patch_tokens.append(int(symbol))
        self.token_index += 1
        if len(self.current_patch_tokens) == self.P:
            self.previous_patch_tokens = list(self.current_patch_tokens)

    def decode_symbol_from_arithmetic(self, arithmetic_decoder: StreamingArithmeticDecoder) -> int:
        logits = self.next_logits()
        _sync_if_cuda(self.device)
        started = perf_counter()
        freqs = fast_floor_frequency_row(logits, total=self.total)
        _sync_if_cuda(self.device)
        self.timings.quantize_seconds += perf_counter() - started

        started = perf_counter()
        freqs_cpu = freqs.cpu()
        _sync_if_cuda(self.device)
        self.timings.transfer_seconds += perf_counter() - started

        started = perf_counter()
        symbol = arithmetic_decoder.decode_frequency_row(freqs_cpu)
        self.timings.arithmetic_seconds += perf_counter() - started
        self.accept_symbol(symbol)
        return symbol

    def decode(self, encoded: bytes, *, token_count: int) -> tuple[list[int], dict[str, float]]:
        arithmetic_decoder = StreamingArithmeticDecoder(encoded)
        decoded: list[int] = []
        self.timings = MegabyteSerialDecodeTimings()
        self.reset_window()
        started = perf_counter()
        for index in range(token_count):
            if index > 0 and index % int(self.config.T_MAX) == 0:
                self.reset_window()
            decoded.append(self.decode_symbol_from_arithmetic(arithmetic_decoder))
        wall_seconds = perf_counter() - started
        timings = self.timings.as_dict()
        timings["wall_seconds"] = wall_seconds
        timings["tokens_per_second"] = token_count / max(wall_seconds, 1e-12)
        return decoded, timings


def encode_symbols_with_serial_model(
    model: torch.nn.Module,
    symbols: list[int],
    *,
    device: torch.device,
    dtype_name: str = "float32",
    arithmetic_frequency_total: int,
) -> tuple[bytes, dict[str, float]]:
    """Encode a token stream with the same cached step path used by serial decode."""
    stepper = MegabyteSerialDecoder(
        model,
        device=device,
        dtype_name=dtype_name,
        arithmetic_frequency_total=arithmetic_frequency_total,
    )
    encoder = StreamingArithmeticEncoder("fast_cpp")
    stepper.timings = MegabyteSerialDecodeTimings()
    stepper.reset_window()
    started = perf_counter()
    for index, symbol in enumerate(symbols):
        if index > 0 and index % int(stepper.config.T_MAX) == 0:
            stepper.reset_window()
        logits = stepper.next_logits()

        _sync_if_cuda(device)
        quant_started = perf_counter()
        probabilities = torch.softmax(logits.float(), dim=-1).unsqueeze(0)
        target = torch.tensor([int(symbol)], dtype=torch.long, device=device)
        lows, highs, totals = fast_floor_intervals_from_probabilities(
            probabilities,
            target,
            total=arithmetic_frequency_total,
        )
        _sync_if_cuda(device)
        stepper.timings.quantize_seconds += perf_counter() - quant_started

        transfer_started = perf_counter()
        lows_cpu = lows.cpu()
        highs_cpu = highs.cpu()
        totals_cpu = totals.cpu()
        _sync_if_cuda(device)
        transfer_seconds = perf_counter() - transfer_started
        stepper.timings.transfer_seconds += transfer_seconds

        arithmetic_started = perf_counter()
        encoder.encode_intervals(lows_cpu, highs_cpu, totals_cpu, interval_transfer_seconds=transfer_seconds)
        stepper.timings.arithmetic_seconds += perf_counter() - arithmetic_started
        stepper.accept_symbol(int(symbol))

    encoded = encoder.finish()
    wall_seconds = perf_counter() - started
    timings = stepper.timings.as_dict()
    timings["wall_seconds"] = wall_seconds
    timings["tokens_per_second"] = len(symbols) / max(wall_seconds, 1e-12)
    return encoded, timings
