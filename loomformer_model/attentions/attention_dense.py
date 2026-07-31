from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from loomformer_runtime.layouts import PackedAttentionLayout, PackedChunkLayout, _unpacked_attention_layout
from loomformer_runtime.profiling import profile_region as _profile_region
from .. import state as S
from ..primitives import fanin_std, init_linear_residual, residual_std
from ..types import InferenceKVRuntime
from .attention_rope import YaRNRotaryEmbedding
from .attention_backends import (
    _attention_contexts_packed_sdpa, _attention_contexts_sdpa,
    _chunk_attention_list, _pack_selected_chunk_history,
    _pack_selected_chunk_kv, _packed_chunk_mask,
    _varlen_attention_contexts, _varlen_backend,
    _varlen_value_fusion_enabled,
)

class GroupedQueryCausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_weight = nn.Parameter(torch.empty(S.N + 2 * S.KV_DIM, S.N))
        self.o = nn.Linear(S.N, S.N, bias=False)
        self.rope = YaRNRotaryEmbedding()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.qkv_weight[:S.N], mean=0.0, std=fanin_std(S.N))
        nn.init.normal_(self.qkv_weight[S.N:S.N + S.KV_DIM], mean=0.0, std=fanin_std(S.N))
        nn.init.normal_(self.qkv_weight[S.N + S.KV_DIM:], mean=0.0, std=residual_std(S.N))
        init_linear_residual(self.o)

    def _cache_capacity(self) -> int:
        return S.SEQ_LEN

    @staticmethod
    def _split_q_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if D != S.N:
            raise ValueError(f"expected q last dim {S.N}, got {D}")
        return x.view(B, T, S.N_Q_HEADS, S.HEAD_DIM)

    @staticmethod
    def _split_kv_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if D != S.KV_DIM:
            raise ValueError(f"expected kv last dim {S.KV_DIM}, got {D}")
        return x.view(B, T, S.N_KV_HEADS, S.HEAD_DIM)

    @staticmethod
    def _merge_q_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, G, Dh = x.shape
        if G != S.N_Q_HEADS or Dh != S.HEAD_DIM:
            raise ValueError("bad query-head shape")
        return x.reshape(B, T, S.N)

    @staticmethod
    def _kv_to_q_heads(x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(S.GQA_GROUP_SIZE, dim=2)

    def _qkv(self, z: torch.Tensor):
        qkv = F.linear(z, self.qkv_weight)
        return torch.split(qkv, (S.N, S.KV_DIM, S.KV_DIM), dim=-1)

    def forward(self, z: torch.Tensor, attn_mask: Optional[Any] = None,
                position_ids: Optional[torch.Tensor] = None,
                inherited_context=None, selected_layout=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del inherited_context, selected_layout
        B, T, D = z.shape
        if T > S.SEQ_LEN:
            raise ValueError(f"sequence length {T} exceeds configured seq_len {S.SEQ_LEN}")
        if position_ids is None:
            position_ids = torch.arange(T, device=z.device, dtype=torch.long)
        q_p, k_p, v_p = self._qkv(z)
        q = self._split_q_heads(q_p)
        k_compact = self._split_kv_heads(k_p)
        v_compact = self._split_kv_heads(v_p)
        q, k_compact = self.rope(q, k_compact, position_ids)
        packed = attn_mask if isinstance(attn_mask, PackedAttentionLayout) else None

        flash_layout = packed
        if flash_layout is None and attn_mask is None:
            flash_layout = _unpacked_attention_layout(B, T, z.device)
        varlen_backend = _varlen_backend(q) if flash_layout is not None else None
        if S.ATTN_IMPL == "flash" and varlen_backend is None:
            raise RuntimeError(
                "attn_impl='flash' requires a validated varlen forward+backward "
                "backend (FlashAttention or Transformer Engine), fp16/bf16 CUDA "
                "tensors, and compact causal layout metadata")

        if varlen_backend is not None:
            with _profile_region(f"loom.attn.{varlen_backend}_varlen_flat"):
                q_flat = q.reshape(B * T, S.N_Q_HEADS, S.HEAD_DIM)
                k_flat = k_compact.reshape(B * T, S.N_KV_HEADS, S.HEAD_DIM)
                v_flat = v_compact.reshape(B * T, S.N_KV_HEADS, S.HEAD_DIM)
                fused_value = None
                if _varlen_value_fusion_enabled(varlen_backend, q):
                    fused_value = torch.cat(
                        (k_compact, v_compact), dim=-1
                    ).reshape(B * T, S.N_KV_HEADS, 2 * S.HEAD_DIM)
                kctx_flat, c_flat = _varlen_attention_contexts(
                    varlen_backend,
                    q_flat, k_flat, v_flat,
                    flash_layout.cu_seqlens, flash_layout.cu_seqlens,
                    flash_layout.max_seqlen, flash_layout.max_seqlen,
                    fused_value=fused_value)
                k_ctx = kctx_flat.view(B, T, S.N_Q_HEADS, S.HEAD_DIM)
                c = c_flat.view(B, T, S.N_Q_HEADS, S.HEAD_DIM)
        elif packed is not None and S.ATTN_IMPL != "manual":
            with _profile_region("loom.attn.sdpa_packed_fallback"):
                k_ctx, c = _attention_contexts_packed_sdpa(
                    q, k_compact, v_compact, packed)
        elif S.ATTN_IMPL in ("auto", "sdpa"):
            k = self._kv_to_q_heads(k_compact)
            v = self._kv_to_q_heads(v_compact)
            qg = q.transpose(1, 2)
            kg = k.transpose(1, 2)
            vg = v.transpose(1, 2)
            dense_mask = attn_mask if isinstance(attn_mask, torch.Tensor) else None
            with _profile_region("loom.attn.sdpa_flat"):
                kctx_g, c_g = _attention_contexts_sdpa(
                    qg, kg, vg, attn_mask=dense_mask, is_causal=dense_mask is None,
                    cat_label="loom.attn.cat_kv_value_flat")
            k_ctx = kctx_g.transpose(1, 2).contiguous()
            c = c_g.transpose(1, 2).contiguous()
        else:
            k = self._kv_to_q_heads(k_compact)
            v = self._kv_to_q_heads(v_compact)
            qg = q.transpose(1, 2)
            kg = k.transpose(1, 2)
            vg = v.transpose(1, 2)
            scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(S.HEAD_DIM)
            if packed is not None:
                pos = torch.arange(T, device=z.device)
                causal = pos[None, :] <= pos[:, None]
                m = (
                    packed.segment_ids.unsqueeze(2) == packed.segment_ids.unsqueeze(1)
                ).unsqueeze(1) & causal.view(1, 1, T, T)
            else:
                if isinstance(attn_mask, torch.Tensor):
                    m = attn_mask
                else:
                    # Explicit manual mode is a debugging fallback.  Do not
                    # pin a [SEQ_LEN,SEQ_LEN] buffer in every attention layer.
                    m = torch.ones(T, T, dtype=torch.bool, device=z.device).tril_(
                    ).view(1, 1, T, T)
            scores = scores.masked_fill(~m, float("-inf"))
            # fp32 softmax accumulation regardless of qg/kg/vg's dtype -- see
            # the sibling manual-attention path below for why (bf16 exp/sum
            # loses precision right where softmax needs it most).
            a = torch.softmax(scores.float(), dim=-1).to(vg.dtype)
            c_g = torch.matmul(a, vg)
            kctx_g = torch.matmul(a, kg)
            c = c_g.transpose(1, 2).contiguous()
            k_ctx = kctx_g.transpose(1, 2).contiguous()
        return self.o(self._merge_q_heads(c)), q, k_ctx, c

    @staticmethod
    def _online_chunk(
        qg: torch.Tensor,
        k_compact: torch.Tensor,
        v_compact: torch.Tensor,
        running: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stable online-softmax update over one compact-GQA KV chunk."""
        k = GroupedQueryCausalSelfAttention._kv_to_q_heads(k_compact)
        v = GroupedQueryCausalSelfAttention._kv_to_q_heads(v_compact)
        kg = k.transpose(1, 2)
        vg = v.transpose(1, 2)
        scores = torch.matmul(qg, kg.transpose(-1, -2)).float() / math.sqrt(S.HEAD_DIM)
        chunk_max = scores.amax(dim=-1)
        if running is None:
            new_max = chunk_max
            probs = torch.exp(scores - new_max.unsqueeze(-1))
            denom = probs.sum(dim=-1)
            c_num = torch.matmul(probs, vg.float())
            k_num = torch.matmul(probs, kg.float())
            return new_max, denom, c_num, k_num
        old_max, old_denom, old_c, old_k = running
        new_max = torch.maximum(old_max, chunk_max)
        old_scale = torch.exp(old_max - new_max)
        probs = torch.exp(scores - new_max.unsqueeze(-1))
        denom = old_denom * old_scale + probs.sum(dim=-1)
        c_num = old_c * old_scale.unsqueeze(-1) + torch.matmul(probs, vg.float())
        k_num = old_k * old_scale.unsqueeze(-1) + torch.matmul(probs, kg.float())
        return new_max, denom, c_num, k_num

    def _step_cpu_kv(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        cache_len: int,
        runtime: InferenceKVRuntime,
    ):
        """Attend to pinned CPU KV through an overlapped staging ring."""
        if q.device.type != "cuda":
            raise ValueError("CPU KV streaming requires a CUDA compute device")
        B = q.shape[0]
        if k_cache is None:
            shape = (B, self._cache_capacity(), S.N_KV_HEADS, S.HEAD_DIM)
            k_cache = torch.zeros(shape, dtype=k_new.dtype, device="cpu", pin_memory=True)
            v_cache = torch.zeros(shape, dtype=v_new.dtype, device="cpu", pin_memory=True)
        elif not k_cache.is_pinned() or not v_cache.is_pinned():
            k_cache = k_cache.pin_memory()
            v_cache = v_cache.pin_memory()

        qg = q.transpose(1, 2)
        running = None
        C = int(runtime.chunk_size)
        preload = runtime.preload_for(cache_len)
        buffers, copy_stream = runtime.cpu_staging(B, k_new.dtype, preload)
        compute_stream = torch.cuda.current_stream(q.device)

        if cache_len + 1 <= C:
            kb, vb = buffers[0]
            ready = torch.cuda.Event()
            with torch.cuda.stream(copy_stream):
                if cache_len:
                    kb[:, :cache_len].copy_(k_cache[:, :cache_len], non_blocking=True)
                    vb[:, :cache_len].copy_(v_cache[:, :cache_len], non_blocking=True)
                ready.record(copy_stream)
            compute_stream.wait_event(ready)
            kb[:, cache_len:cache_len + 1].copy_(k_new)
            vb[:, cache_len:cache_len + 1].copy_(v_new)
            k = self._kv_to_q_heads(kb[:, :cache_len + 1])
            v = self._kv_to_q_heads(vb[:, :cache_len + 1])
            kg = k.transpose(1, 2)
            vg = v.transpose(1, 2)
            scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(S.HEAD_DIM)
            a = torch.softmax(scores.float(), dim=-1).to(vg.dtype)
            c_g = torch.matmul(a, vg)
            kctx_g = torch.matmul(a, kg)
            k_cache[:, cache_len].copy_(k_new[:, 0], non_blocking=False)
            v_cache[:, cache_len].copy_(v_new[:, 0], non_blocking=False)
            return c_g, kctx_g, k_cache, v_cache

        ready: Dict[int, torch.cuda.Event] = {}
        free: List[Optional[torch.cuda.Event]] = [None] * len(buffers)
        offsets = list(range(0, cache_len, C))
        preload = min(len(offsets), preload)

        def enqueue(j: int, buffer_index: int) -> None:
            off = offsets[j]
            n = min(C, cache_len - off)
            kb, vb = buffers[buffer_index]
            with torch.cuda.stream(copy_stream):
                if free[buffer_index] is not None:
                    copy_stream.wait_event(free[buffer_index])
                kb[:, :n].copy_(k_cache[:, off:off + n], non_blocking=True)
                vb[:, :n].copy_(v_cache[:, off:off + n], non_blocking=True)
                event = torch.cuda.Event()
                event.record(copy_stream)
                ready[j] = event

        for j in range(preload):
            enqueue(j, j % len(buffers))
        # "Preload" means ready, not merely queued on the copy stream. Waiting
        # for the last initial event establishes the measured lead before the
        # online reducer starts; the +2 calibration margin can then absorb
        # transfer jitter instead of disappearing during the first chunks.
        if preload > 1:
            compute_stream.wait_event(ready[preload - 1])

        for j, off in enumerate(offsets):
            bi = j % len(buffers)
            if j not in ready:
                enqueue(j, bi)
            compute_stream.wait_event(ready.pop(j))
            n = min(C, cache_len - off)
            kb, vb = buffers[bi]
            running = self._online_chunk(qg, kb[:, :n], vb[:, :n], running)
            free[bi] = torch.cuda.Event()
            free[bi].record(compute_stream)
            nxt = j + preload
            if nxt < len(offsets) and nxt not in ready:
                enqueue(nxt, nxt % len(buffers))

        # The current token is already on the compute GPU. Fold it into the
        # same online reduction without a pointless round trip through RAM.
        running = self._online_chunk(qg, k_new, v_new, running)
        _, denom, c_num, k_num = running
        c_g = (c_num / denom.unsqueeze(-1)).to(q.dtype)
        kctx_g = (k_num / denom.unsqueeze(-1)).to(q.dtype)

        # Persist the new compact KV only after it has contributed locally.
        # The destination is pinned, so future H2D reads remain asynchronous.
        k_cache[:, cache_len].copy_(k_new[:, 0], non_blocking=False)
        v_cache[:, cache_len].copy_(v_new[:, 0], non_blocking=False)
        return c_g, kctx_g, k_cache, v_cache

    def _step_remote_cuda_kv(
        self,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        cache_len: int,
        storage: torch.device,
    ):
        """Execute the complete attention reduction beside remote-resident KV."""
        B = q.shape[0]
        q_remote = q.to(storage, non_blocking=True)
        k_new_remote = k_new.to(storage, non_blocking=True)
        v_new_remote = v_new.to(storage, non_blocking=True)
        if k_cache is None:
            shape = (B, self._cache_capacity(), S.N_KV_HEADS, S.HEAD_DIM)
            k_cache = torch.zeros(shape, dtype=k_new.dtype, device=storage)
            v_cache = torch.zeros(shape, dtype=v_new.dtype, device=storage)
        k_cache[:, cache_len] = k_new_remote[:, 0]
        v_cache[:, cache_len] = v_new_remote[:, 0]
        new_len = cache_len + 1
        k = self._kv_to_q_heads(k_cache[:, :new_len])
        v = self._kv_to_q_heads(v_cache[:, :new_len])
        qg = q_remote.transpose(1, 2)
        kg = k.transpose(1, 2)
        vg = v.transpose(1, 2)
        scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(S.HEAD_DIM)
        a = torch.softmax(scores.float(), dim=-1).to(vg.dtype)
        c_g = torch.matmul(a, vg).to(q.device, non_blocking=True)
        kctx_g = torch.matmul(a, kg).to(q.device, non_blocking=True)
        return c_g, kctx_g, k_cache, v_cache

    def step(
        self,
        z: torch.Tensor,
        position_id: int,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        cache_len: int,
        kv_runtime: Optional[InferenceKVRuntime] = None,
        inherited_context=None,
        held_context=None,
    ):
        del inherited_context, held_context
        capacity = self._cache_capacity()
        if cache_len >= capacity:
            raise ValueError(f"attention cache exceeded capacity={capacity}")
        B = z.shape[0]
        q_p, k_p, v_p = self._qkv(z)
        q = self._split_q_heads(q_p)
        k_new_q = self._kv_to_q_heads(self._split_kv_heads(k_p))
        pos = torch.tensor([int(position_id)], device=z.device, dtype=torch.long)
        q, k_new_q = self.rope(q, k_new_q, pos)
        k_new = k_new_q.view(B, 1, S.N_KV_HEADS, S.GQA_GROUP_SIZE, S.HEAD_DIM)[:, :, :, 0, :]
        v_new = self._split_kv_heads(v_p)
        storage = z.device if kv_runtime is None else kv_runtime.storage_device
        if storage.type == "cpu" and z.device.type == "cuda":
            c_g, kctx_g, k_cache, v_cache = self._step_cpu_kv(
                q, k_new, v_new, k_cache, v_cache, cache_len, kv_runtime)
        elif storage != z.device:
            c_g, kctx_g, k_cache, v_cache = self._step_remote_cuda_kv(
                q, k_new, v_new, k_cache, v_cache, cache_len, storage)
        else:
            if k_cache is None:
                k_cache = z.new_zeros(B, capacity, S.N_KV_HEADS, S.HEAD_DIM)
                v_cache = z.new_zeros(B, capacity, S.N_KV_HEADS, S.HEAD_DIM)
            k_cache[:, cache_len] = k_new[:, 0]
            v_cache[:, cache_len] = v_new[:, 0]
            new_len = cache_len + 1
            k_all = k_cache[:, :new_len]
            v_all = v_cache[:, :new_len]
            k = self._kv_to_q_heads(k_all)
            v = self._kv_to_q_heads(v_all)
            qg = q.transpose(1, 2)
            kg = k.transpose(1, 2)
            vg = v.transpose(1, 2)
            scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(S.HEAD_DIM)
            a = torch.softmax(scores.float(), dim=-1).to(vg.dtype)
            c_g = torch.matmul(a, vg)
            kctx_g = torch.matmul(a, kg)
        c = c_g.transpose(1, 2).contiguous()
        k_ctx = kctx_g.transpose(1, 2).contiguous()
        # Возвращаем ПОЛНЫЙ буфер (не срез) -- вызывающий код хранит его в LayerCache и
        # переиспользует на следующем шаге; срез k_all был только для этого forward.
        context = (q, k_ctx, c)
        return (
            self.o(self._merge_q_heads(c)), q, k_ctx, c,
            k_cache, v_cache, cache_len + 1, context,
        )

    def forward_chunk(
        self,
        z: torch.Tensor,
        past_k_chunks: tuple,
        past_v_chunks: tuple,
        position_ids: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        inherited_context=None,
        held_context=None,
        past_document_chunks=(),
        past_position_chunks=(),
        strided_chunk_layout=None,
    ):
        del (
            inherited_context, held_context, past_document_chunks,
            past_position_chunks, strided_chunk_layout,
        )
        B, T, _ = z.shape
        q_p, k_p, v_p = self._qkv(z)
        q = self._split_q_heads(q_p)
        k_new = self._split_kv_heads(k_p)
        v_new = self._split_kv_heads(v_p).contiguous()
        q, k_new = self.rope(q, k_new, position_ids)
        k_new = k_new.contiguous()
        k_chunks = (*past_k_chunks, k_new)
        v_chunks = (*past_v_chunks, v_new)

        varlen_backend = _varlen_backend(q)
        if S.ATTN_IMPL == "flash" and varlen_backend is None:
            raise RuntimeError(
                "attn_impl='flash' requires a validated FlashAttention or "
                "Transformer Engine varlen forward+backward backend")
        if varlen_backend is not None:
            with _profile_region(f"loom.attn.{varlen_backend}_varlen_chunk"):
                q_packed = q.reshape(B * T, S.N_Q_HEADS, S.HEAD_DIM)
                fused_value = None
                if _varlen_value_fusion_enabled(varlen_backend, q):
                    fused_value = _pack_selected_chunk_kv(
                        k_chunks, v_chunks, packed_chunk)
                    k_packed = fused_value[..., :S.HEAD_DIM]
                    v_packed = k_packed  # unused by the fused branch
                else:
                    k_packed = _pack_selected_chunk_history(k_chunks, packed_chunk)
                    v_packed = _pack_selected_chunk_history(v_chunks, packed_chunk)
                kctx, value_ctx = _varlen_attention_contexts(
                    varlen_backend,
                    q_packed, k_packed, v_packed,
                    packed_chunk.cu_seqlens_q, packed_chunk.cu_seqlens_k,
                    packed_chunk.max_seqlen_q, packed_chunk.max_seqlen_k,
                    fused_value=fused_value)
                k_ctx = kctx.view(B, T, S.N_Q_HEADS, S.HEAD_DIM)
                c = value_ctx.view(B, T, S.N_Q_HEADS, S.HEAD_DIM)
        elif S.ATTN_IMPL in ("auto", "sdpa"):
            kg = self._kv_to_q_heads(torch.cat(k_chunks, dim=1)).transpose(1, 2)
            vg = self._kv_to_q_heads(torch.cat(v_chunks, dim=1)).transpose(1, 2)
            qg = q.transpose(1, 2)
            chunk_mask = _packed_chunk_mask(
                attention_layout, packed_chunk.start, packed_chunk.end, z.device)
            with _profile_region("loom.attn.sdpa_chunk"):
                kctx_g, c_g = _attention_contexts_sdpa(
                    qg, kg, vg, attn_mask=chunk_mask, is_causal=chunk_mask is None,
                    cat_label="loom.attn.cat_kv_value_chunk")
            k_ctx = kctx_g.transpose(1, 2).contiguous()
            c = c_g.transpose(1, 2).contiguous()
        else:
            chunk_mask = _packed_chunk_mask(
                attention_layout, packed_chunk.start, packed_chunk.end, z.device)
            k_ctx, c = _chunk_attention_list(q.contiguous(), k_chunks, v_chunks, chunk_mask)
        return self.o(self._merge_q_heads(c)), q, k_ctx, c, k_new, v_new

__all__ = ("GroupedQueryCausalSelfAttention",)
