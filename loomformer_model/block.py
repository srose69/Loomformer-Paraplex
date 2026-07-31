from __future__ import annotations

from typing import Optional

import torch.nn as nn

from . import state as S
from .paraplex import ParaplexFFN
from .attentions.attention_dense import GroupedQueryCausalSelfAttention
from .attentions.attention_sparse import InheritedContextMixer, StridedGroupedQueryCausalSelfAttention

class Block(nn.Module):
    def __init__(
        self,
        active_ordinal: Optional[int],
        token_stride: int,
        token_schedule: str,
        ablation: bool = False,
    ) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(S.N)
        if active_ordinal is None:
            self.attn = InheritedContextMixer()
        elif token_stride == 1:
            self.attn = GroupedQueryCausalSelfAttention()
        else:
            offset = active_ordinal % token_stride if token_schedule == "staggered" else 0
            self.attn = StridedGroupedQueryCausalSelfAttention(token_stride, offset)
        self.ln_ffn = nn.LayerNorm(S.N)
        self.ffn = ParaplexFFN(ablation=ablation)

__all__ = ("Block",)
