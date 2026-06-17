from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
from einops import rearrange

from .megabyte_serial_decode import AttentionCache, _autocast_context, _transformer_step


@dataclass
class BatchedDecodeStepTimings:
    global_seconds: float = 0.0
    local_seconds: float = 0.0
    logits_seconds: float = 0.0

    @property
    def model_seconds(self) -> float:
        return self.global_seconds + self.local_seconds + self.logits_seconds

    def as_dict(self) -> dict[str, float]:
        return {
            "model_global_seconds": self.global_seconds,
            "model_local_seconds": self.local_seconds,
            "model_logits_seconds": self.logits_seconds,
            "model_seconds": self.model_seconds,
        }


def fast_floor_frequency_rows(
    logits: torch.Tensor,
    *,
    total: int,
    prefer_uint16: bool = True,
    return_totals: bool = False,
) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("logits must be a 2D tensor")
    probs = torch.softmax(logits.float(), dim=-1)
    probs = torch.where(torch.isfinite(probs) & (probs > 0), probs, torch.zeros_like(probs))
    freqs = torch.floor(probs * int(total)).clamp_min(1)
    row_totals = freqs.sum(dim=1).to(torch.int32).contiguous() if return_totals else None
    if prefer_uint16 and int(total) <= 65535:
        out = freqs.to(torch.uint16).contiguous()
    else:
        out = freqs.to(torch.int32).contiguous()
    if return_totals:
        return out, row_totals
    return out


class MegabyteBatchedDecodeStepper:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        batch_size: int,
        device: torch.device,
        dtype_name: str = "float32",
    ) -> None:
        if model.__class__.__name__ != "Megabyte":
            raise ValueError("MegabyteBatchedDecodeStepper expects a megabyte_in_action Megabyte model")
        self.model = model
        self.batch_size = int(batch_size)
        self.device = device
        self.dtype_name = dtype_name
        self.config = model.config
        self.P = int(self.config.P)
        self.V = int(self.config.V)
        self.D_G = int(self.config.D_G)
        self.D_L = int(self.config.D_L)
        self.seq_length = int(self.config.T_MAX)
        self.pad_id = int(self.config.pad_id)
        self.timings = BatchedDecodeStepTimings()
        self.reset_window()

    def reset_window(self) -> None:
        self.global_caches: list[AttentionCache] | None = None
        self.local_caches: list[AttentionCache] | None = None
        self.previous_patch_tokens: torch.Tensor | None = None
        self.current_patch_tokens = torch.empty((self.batch_size, self.P), dtype=torch.long, device=self.device)
        self.current_patch_context: torch.Tensor | None = None
        self.token_index = 0

    def _start_patch(self) -> None:
        if self.previous_patch_tokens is None:
            patch_ids = torch.full((self.batch_size, self.P), self.pad_id, dtype=torch.long, device=self.device)
        else:
            patch_ids = self.previous_patch_tokens
        started = perf_counter()
        with torch.inference_mode(), _autocast_context(self.device, self.dtype_name):
            patch_embed = self.model.to_embed(patch_ids)
            global_in = rearrange(patch_embed, "b p d -> b 1 (p d)")
            global_out, self.global_caches = _transformer_step(
                self.model.g_transformer,
                global_in,
                self.global_caches,
            )
            self.current_patch_context = self.model.gl_linear(global_out).view(self.batch_size, self.P, self.D_L)
        self.timings.global_seconds += perf_counter() - started
        self.local_caches = None

    def next_logits(self) -> torch.Tensor:
        if self.token_index > 0 and self.token_index % self.seq_length == 0:
            self.reset_window()
        local_index = self.token_index % self.P
        if local_index == 0 or self.current_patch_context is None:
            self._start_patch()

        if self.token_index == 0:
            previous_ids = torch.full((self.batch_size, 1), self.pad_id, dtype=torch.long, device=self.device)
        elif local_index == 0:
            if self.previous_patch_tokens is None:
                raise RuntimeError("missing previous patch at patch boundary")
            previous_ids = self.previous_patch_tokens[:, -1:].contiguous()
        else:
            previous_ids = self.current_patch_tokens[:, local_index - 1 : local_index].contiguous()

        started = perf_counter()
        with torch.inference_mode(), _autocast_context(self.device, self.dtype_name):
            local_embed = self.model.to_l_embed(self.model.to_embed(previous_ids))
            local_in = self.current_patch_context[:, local_index : local_index + 1, :] + local_embed
            local_out, self.local_caches = _transformer_step(
                self.model.l_transformer,
                local_in,
                self.local_caches,
            )
        self.timings.local_seconds += perf_counter() - started

        started = perf_counter()
        with torch.inference_mode(), _autocast_context(self.device, self.dtype_name):
            logits = self.model.to_logits(local_out[:, -1, :])
        self.timings.logits_seconds += perf_counter() - started
        return logits

    def accept_symbols(self, symbols: torch.Tensor) -> None:
        if symbols.device != self.device:
            symbols = symbols.to(self.device, non_blocking=True)
        symbols = symbols.to(dtype=torch.long).view(self.batch_size)
        local_index = self.token_index % self.P
        self.current_patch_tokens[:, local_index] = symbols
        self.token_index += 1
        if local_index == self.P - 1:
            self.previous_patch_tokens = self.current_patch_tokens.clone()
            self.current_patch_tokens = torch.empty((self.batch_size, self.P), dtype=torch.long, device=self.device)

    def run_random_tokens(self, *, token_count: int, vocab_high: int | None = None) -> float:
        vocab_limit = int(vocab_high or self.pad_id)
        checksum = torch.zeros((), dtype=torch.float32, device=self.device)
        for _ in range(token_count):
            logits = self.next_logits()
            checksum = checksum + logits[:, 0].float().sum()
            symbols = torch.randint(0, vocab_limit, (self.batch_size,), dtype=torch.long, device=self.device)
            self.accept_symbols(symbols)
        return float(checksum.detach().cpu())
