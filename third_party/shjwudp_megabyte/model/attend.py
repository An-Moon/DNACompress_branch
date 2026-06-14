# MIT License
# Copyright (c) 2023 Phil Wang

from packaging import version

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:
    _flash_attn_func = None

# helpers

def exists(val):
    return val is not None

# main class

class Attend(nn.Module):
    def __init__(
        self,
        causal = False,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.causal = causal
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

    def get_mask(self, i, j, device):
        return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

    def _alibi_bias_from_slopes(self, alibi_slopes, q_len, k_len, device, dtype):
        positions = torch.arange(k_len, device=device, dtype=dtype)
        slopes = alibi_slopes.to(device=device, dtype=dtype)
        return positions.view(1, 1, k_len) * slopes.view(-1, 1, 1)

    def package_flash_attn(self, q, k, v, alibi_slopes = None):
        if _flash_attn_func is None or not q.is_cuda:
            return None

        q = rearrange(q, 'b h n d -> b n h d')
        if k.ndim == 3:
            k = rearrange(k, 'b n d -> b n 1 d')
        else:
            k = rearrange(k, 'b h n d -> b n h d')
        if v.ndim == 3:
            v = rearrange(v, 'b n d -> b n 1 d')
        else:
            v = rearrange(v, 'b h n d -> b n h d')

        slopes = None
        if exists(alibi_slopes):
            slopes = alibi_slopes.to(device=q.device, dtype=torch.float32)

        out = _flash_attn_func(
            q,
            k,
            v,
            dropout_p = self.dropout if self.training else 0.,
            causal = self.causal,
            alibi_slopes = slopes,
        )
        return rearrange(out, 'b n h d -> b h n d')

    def flash_attn(self, q, k, v, mask = None, attn_bias = None, alibi_slopes = None):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        if is_cuda and not exists(attn_bias) and not exists(mask):
            out = self.package_flash_attn(q, k, v, alibi_slopes=alibi_slopes)
            if exists(out):
                return out

        # single headed key / values

        if k.ndim == 3:
            k = rearrange(k, 'b n d -> b 1 n d')

        if v.ndim == 3:
            v = rearrange(v, 'b n d -> b 1 n d')

        if k.shape[1] == 1 and heads > 1:
            k = k.expand(-1, heads, -1, -1)

        if v.shape[1] == 1 and heads > 1:
            v = v.expand(-1, heads, -1, -1)

        if exists(alibi_slopes) and not exists(attn_bias):
            attn_bias = self._alibi_bias_from_slopes(alibi_slopes, q_len, k_len, device, q.dtype)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        if exists(mask) and mask.ndim != 4:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask = mask.expand(-1, heads, q_len, -1)

        causal = self.causal

        # handle attention bias

        if exists(attn_bias):
            mask_value = -torch.finfo(q.dtype).max // 2
            causal_mask = self.get_mask(q_len, k_len, device)
            attn_bias = attn_bias.masked_fill(causal_mask, mask_value)

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value)

            mask = attn_bias
            causal = False

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = mask,
            dropout_p = self.dropout if self.training else 0.,
            is_causal = causal
        )

        return out

    def forward(self, q, k, v, mask = None, attn_bias = None, alibi_slopes = None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        if self.flash:
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias, alibi_slopes = alibi_slopes)

        # similarity

        sim = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # attention bias

        if exists(attn_bias):
            sim = sim + attn_bias
        elif exists(alibi_slopes):
            sim = sim + self._alibi_bias_from_slopes(alibi_slopes, q_len, k_len, device, sim.dtype)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(q_len, k_len, device)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out
