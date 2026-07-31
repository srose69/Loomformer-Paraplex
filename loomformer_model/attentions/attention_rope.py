from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .. import state as S

def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))


def _yarn_find_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float,
                                max_position_embeddings: int) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, *, device, dtype) -> torch.Tensor:
    if low == high:
        high += 0.001
    x = (torch.arange(dim, device=device, dtype=dtype) - low) / (high - low)
    return torch.clamp(x, 0.0, 1.0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class YaRNRotaryEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inv_freq, attention_factor = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_factor = float(attention_factor)
        cos, sin = self._build_cos_sin_cache(S.SEQ_LEN, inv_freq.device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def _compute_inv_freq() -> Tuple[torch.Tensor, float]:
        rotary_dim = S.HEAD_DIM
        pos_freqs = S.ROPE_THETA ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (S.ROPE_FACTOR * pos_freqs)
        low, high = _yarn_find_correction_range(
            S.ROPE_BETA_FAST, S.ROPE_BETA_SLOW, rotary_dim, S.ROPE_THETA, S.ROPE_ORIGINAL_SEQ_LEN
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, rotary_dim // 2, device=pos_freqs.device, dtype=torch.float32
        )
        inv_freq = inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        attention_factor = S.ROPE_ATTENTION_FACTOR
        if attention_factor is None:
            attention_factor = _yarn_get_mscale(S.ROPE_FACTOR)
        return inv_freq, float(attention_factor)

    def _build_cos_sin_cache(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().mul(self.attention_factor)
            sin = emb.sin().mul(self.attention_factor)
        return cos, sin

    def _ensure_cache_device(self, device: torch.device) -> None:
        if self.cos_cached.device == device and self.sin_cached.device == device:
            return
        self.cos_cached = self.cos_cached.to(device=device)
        self.sin_cached = self.sin_cached.to(device=device)

    def _cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cache_device(position_ids.device)
        if position_ids.dim() == 1:
            cos = self.cos_cached[position_ids].to(dtype=dtype).unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[position_ids].to(dtype=dtype).unsqueeze(0).unsqueeze(2)
        elif position_ids.dim() == 2:
            cos = self.cos_cached[position_ids].to(dtype=dtype).unsqueeze(2)
            sin = self.sin_cached[position_ids].to(dtype=dtype).unsqueeze(2)
        else:
            raise ValueError(f"position_ids must be 1D or 2D, got shape {tuple(position_ids.shape)}")
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin(position_ids, q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

__all__ = ('_yarn_get_mscale', '_yarn_find_correction_dim', '_yarn_find_correction_range', '_yarn_linear_ramp_mask', '_rotate_half', 'YaRNRotaryEmbedding')
