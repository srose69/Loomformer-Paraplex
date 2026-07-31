from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from inline_kernels import sample_hold
from loomformer_runtime.layouts import PackedAttentionLayout, PackedChunkLayout, _cu_seqlens_from_counts
from .. import state as S
from .attention_backends import (
    _attention_contexts_sdpa, _pack_selected_chunk_history,
    _pack_selected_chunk_kv, _varlen_attention_contexts,
    _varlen_backend, _varlen_value_fusion_enabled,
)
from .attention_dense import GroupedQueryCausalSelfAttention

@dataclass(frozen=True)
class SelectedTokenLayout:
    mask: torch.Tensor
    indices: torch.Tensor
    positions: torch.Tensor
    segments: torch.Tensor
    documents: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    row_cu_seqlens: torch.Tensor
    ranks: torch.Tensor


@dataclass(frozen=True)
class StridedChunkLayout:
    selected: SelectedTokenLayout
    history: Optional[PackedChunkLayout]
    key_positions: torch.Tensor


def _selected_token_layout(
    position_ids: torch.Tensor,
    packed: Optional[PackedAttentionLayout],
    stride: int,
    offset: int,
) -> SelectedTokenLayout:
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    B, T = position_ids.shape
    mask = position_ids.remainder(stride).eq(offset)
    indices = mask.reshape(-1).nonzero().flatten()
    positions = position_ids.reshape(-1).index_select(0, indices)
    if packed is None:
        segments_full = torch.zeros(B, T, dtype=torch.int32, device=position_ids.device)
        max_document = T
    else:
        segments_full = packed.segment_ids
        max_document = packed.max_seqlen
    row_grid = torch.arange(
        B, device=position_ids.device, dtype=torch.int64).unsqueeze(1)
    full_documents = (
        row_grid.bitwise_left_shift(32)
        | segments_full.to(torch.int64).bitwise_and(0xFFFFFFFF)
    )
    _, full_counts = torch.unique_consecutive(
        full_documents.reshape(-1), return_counts=True)
    doc_cu = _cu_seqlens_from_counts(full_counts)
    segments = segments_full.reshape(-1).index_select(0, indices)
    documents = full_documents.reshape(-1).index_select(0, indices)
    prefix = torch.zeros(B * T + 1, dtype=torch.int32, device=position_ids.device)
    torch.cumsum(mask.reshape(-1).to(torch.int32), dim=0, out=prefix[1:])
    doc_counts = prefix.index_select(0, doc_cu[1:].to(torch.int64))
    doc_counts -= prefix.index_select(0, doc_cu[:-1].to(torch.int64))
    doc_counts = doc_counts[doc_counts.ne(0)]
    cu_seqlens = _cu_seqlens_from_counts(doc_counts)
    row_ends = torch.arange(
        T, (B + 1) * T, T, dtype=torch.int64, device=position_ids.device)
    row_counts = prefix.index_select(0, row_ends).clone()
    row_starts = torch.cat((prefix.new_zeros(1), row_counts[:-1]))
    row_counts -= row_starts
    max_selected = max(1, (max_document + stride - 1 - offset) // stride)
    ranks = torch.full(
        (B * T,), -1, dtype=torch.int32, device=position_ids.device)
    ranks.index_copy_(
        0, indices, torch.arange(indices.numel(), dtype=torch.int32, device=indices.device))
    return SelectedTokenLayout(
        mask=mask,
        indices=indices,
        positions=positions,
        segments=segments,
        documents=documents,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_selected,
        row_cu_seqlens=_cu_seqlens_from_counts(row_counts),
        ranks=ranks,
    )


def _selected_sdpa_contexts(
    q: torch.Tensor,
    k_compact: torch.Tensor,
    v_compact: torch.Tensor,
    selected: SelectedTokenLayout,
    batch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    total = q.shape[0]
    row_cu = selected.row_cu_seqlens
    row_counts = row_cu[1:] - row_cu[:-1]
    width = int(row_counts.max().item())
    rows = torch.repeat_interleave(
        torch.arange(batch, device=q.device, dtype=torch.int64),
        row_counts.to(torch.int64),
    )
    ranks = torch.arange(total, device=q.device) - row_cu[:-1].to(torch.int64).index_select(0, rows)
    destinations = rows * width + ranks

    def pad(x: torch.Tensor) -> torch.Tensor:
        shape = (batch * width, x.shape[-2], x.shape[-1])
        return x.new_zeros(shape).index_copy(0, destinations, x).view(
            batch, width, x.shape[-2], x.shape[-1])

    q_pad = pad(q)
    k_pad = pad(k_compact)
    v_pad = pad(v_compact)
    pos_pad = selected.positions.new_full((batch * width,), -1).index_copy(
        0, destinations, selected.positions).view(batch, width)
    seg_pad = selected.segments.new_full((batch * width,), -1).index_copy(
        0, destinations, selected.segments).view(batch, width)
    valid = pos_pad.ge(0)
    allowed = (
        valid[:, :, None]
        & valid[:, None, :]
        & seg_pad[:, :, None].eq(seg_pad[:, None, :])
        & pos_pad[:, None, :].le(pos_pad[:, :, None])
    ).unsqueeze(1)
    qg = q_pad.transpose(1, 2)
    kg = GroupedQueryCausalSelfAttention._kv_to_q_heads(k_pad).transpose(1, 2)
    vg = GroupedQueryCausalSelfAttention._kv_to_q_heads(v_pad).transpose(1, 2)
    kctx_g, c_g = _attention_contexts_sdpa(
        qg, kg, vg, attn_mask=allowed, is_causal=False,
        cat_label="loom.attn.cat_kv_value_sparse")
    k_ctx = kctx_g.transpose(1, 2).reshape(
        batch * width, S.N_Q_HEADS, S.HEAD_DIM).index_select(0, destinations)
    c = c_g.transpose(1, 2).reshape(
        batch * width, S.N_Q_HEADS, S.HEAD_DIM).index_select(0, destinations)
    return k_ctx, c


def _selected_chunk_sdpa_contexts(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_positions: torch.Tensor,
    k_positions: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_counts = cu_q[1:] - cu_q[:-1]
    k_counts = cu_k[1:] - cu_k[:-1]
    documents = q_counts.numel()
    q_width = int(q_counts.max().item())
    k_width = int(k_counts.max().item())

    def destinations(counts: torch.Tensor, width: int) -> torch.Tensor:
        rows = torch.repeat_interleave(
            torch.arange(documents, device=q.device, dtype=torch.int64),
            counts.to(torch.int64),
        )
        cu = _cu_seqlens_from_counts(counts)
        ranks = torch.arange(rows.numel(), device=q.device) - cu[:-1].to(
            torch.int64).index_select(0, rows)
        return rows * width + ranks

    q_dst = destinations(q_counts, q_width)
    k_dst = destinations(k_counts, k_width)

    def pad(x: torch.Tensor, width: int, dst: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(documents * width, x.shape[-2], x.shape[-1])
        return out.index_copy(0, dst, x).view(
            documents, width, x.shape[-2], x.shape[-1])

    q_pad = pad(q, q_width, q_dst)
    k_pad = pad(k, k_width, k_dst)
    v_pad = pad(v, k_width, k_dst)
    qp = q_positions.new_full((documents * q_width,), -1).index_copy(
        0, q_dst, q_positions).view(documents, q_width)
    kp = k_positions.new_full((documents * k_width,), -1).index_copy(
        0, k_dst, k_positions).view(documents, k_width)
    allowed = (
        qp.ge(0)[:, :, None]
        & kp.ge(0)[:, None, :]
        & kp[:, None, :].le(qp[:, :, None])
    ).unsqueeze(1)
    qg = q_pad.transpose(1, 2)
    kg = GroupedQueryCausalSelfAttention._kv_to_q_heads(k_pad).transpose(1, 2)
    vg = GroupedQueryCausalSelfAttention._kv_to_q_heads(v_pad).transpose(1, 2)
    kctx_g, c_g = _attention_contexts_sdpa(
        qg, kg, vg, attn_mask=allowed, is_causal=False,
        cat_label="loom.attn.cat_kv_value_sparse_chunk")
    k_ctx = kctx_g.transpose(1, 2).reshape(
        documents * q_width, S.N_Q_HEADS, S.HEAD_DIM).index_select(0, q_dst)
    c = c_g.transpose(1, 2).reshape(
        documents * q_width, S.N_Q_HEADS, S.HEAD_DIM).index_select(0, q_dst)
    return k_ctx, c


class StridedGroupedQueryCausalSelfAttention(GroupedQueryCausalSelfAttention):
    def __init__(self, stride: int, offset: int) -> None:
        self.token_stride = int(stride)
        self.token_offset = int(offset)
        super().__init__()

    def _cache_capacity(self) -> int:
        remaining = S.SEQ_LEN - self.token_offset
        return max(1, (remaining + self.token_stride - 1) // self.token_stride)

    def _chunk_layout(
        self,
        position_ids: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
    ) -> SelectedTokenLayout:
        local_packed = None
        if attention_layout is not None:
            local_packed = PackedAttentionLayout(
                segment_ids=attention_layout.segment_ids[:, packed_chunk.start:packed_chunk.end],
                position_ids=position_ids.to(torch.int32),
                cu_seqlens=position_ids.new_empty(0, dtype=torch.int32),
                max_seqlen=attention_layout.max_seqlen,
            )
        return _selected_token_layout(
            position_ids, local_packed, self.token_stride, self.token_offset)

    def _selected_contexts(
        self,
        z: torch.Tensor,
        position_ids: torch.Tensor,
        packed: Optional[PackedAttentionLayout],
        selected: Optional[SelectedTokenLayout] = None,
    ):
        B, T, _ = z.shape
        if selected is None:
            selected = _selected_token_layout(
                position_ids, packed, self.token_stride, self.token_offset)
        z_selected = z.reshape(B * T, S.N).index_select(0, selected.indices)
        q_p, k_p, v_p = self._qkv(z_selected)
        if selected.indices.numel() == 0:
            q = q_p.view(-1, S.N_Q_HEADS, S.HEAD_DIM)
            dependency = (k_p.sum() + v_p.sum()) * 0
            return selected, q, q + dependency, q + dependency
        q = q_p.view(-1, S.N_Q_HEADS, S.HEAD_DIM)
        k = k_p.view(-1, S.N_KV_HEADS, S.HEAD_DIM)
        v = v_p.view(-1, S.N_KV_HEADS, S.HEAD_DIM)
        q, k = self.rope(
            q.unsqueeze(0), k.unsqueeze(0), selected.positions.unsqueeze(0))
        q, k = q.squeeze(0), k.squeeze(0)
        backend = _varlen_backend(q)
        if S.ATTN_IMPL == "flash" and backend is None:
            raise RuntimeError(
                "attn_impl='flash' requires a validated varlen forward+backward backend")
        if backend is not None:
            fused_value = (
                torch.cat((k, v), dim=-1)
                if _varlen_value_fusion_enabled(backend, q)
                else None
            )
            k_ctx, c = _varlen_attention_contexts(
                backend, q, k, v,
                selected.cu_seqlens, selected.cu_seqlens,
                selected.max_seqlen, selected.max_seqlen,
                fused_value=fused_value)
        else:
            k_ctx, c = _selected_sdpa_contexts(q, k, v, selected, B)
        return selected, q, k_ctx, c

    def build_chunk_layout(
        self,
        position_ids: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        past_document_chunks: tuple,
        past_position_chunks: tuple,
    ) -> StridedChunkLayout:
        selected = self._chunk_layout(
            position_ids, attention_layout, packed_chunk)
        if selected.indices.numel() == 0:
            return StridedChunkLayout(
                selected, None, selected.positions.new_empty(0))

        document_chunks = (*past_document_chunks, selected.documents)
        position_chunks = (*past_position_chunks, selected.positions)
        q_documents, q_counts = torch.unique_consecutive(
            selected.documents, return_counts=True)
        selectors = []
        destinations = []
        source_positions = []
        k_document_indices = []
        for documents, positions in zip(document_chunks, position_chunks):
            doc_indices = torch.searchsorted(q_documents, documents)
            bounded = doc_indices.clamp_max(q_documents.numel() - 1)
            keep = doc_indices.lt(q_documents.numel()) & q_documents.index_select(
                0, bounded).eq(documents)
            selector = keep.nonzero().flatten()
            selectors.append(selector.to(torch.int32))
            if selector.numel() == 0:
                continue
            k_document_indices.append(doc_indices.index_select(0, selector))
            source_positions.append(positions.index_select(0, selector))

        all_k_docs = torch.cat(k_document_indices)
        k_counts = torch.bincount(
            all_k_docs, minlength=q_documents.numel()).to(torch.int32)
        cu_k = _cu_seqlens_from_counts(k_counts)
        for doc_indices, positions in zip(k_document_indices, source_positions):
            ranks = (positions - self.token_offset).div(
                self.token_stride, rounding_mode="floor")
            destinations.append(
                (cu_k[:-1].to(torch.int64).index_select(0, doc_indices) + ranks).to(
                    torch.int32))
        piece_sizes = tuple(selector.numel() for selector in selectors)
        piece_offsets = torch.zeros(
            len(piece_sizes) + 1, dtype=torch.int32, device=position_ids.device)
        if piece_sizes:
            torch.cumsum(
                torch.tensor(
                    piece_sizes, dtype=torch.int32, device=position_ids.device),
                dim=0, out=piece_offsets[1:])
        destinations_t = torch.cat(destinations)
        source_positions_t = torch.cat(source_positions)
        key_positions = source_positions_t.new_empty(
            destinations_t.numel()).index_copy(
                0, destinations_t.to(torch.int64), source_positions_t)
        history = PackedChunkLayout(
            start=packed_chunk.start,
            end=packed_chunk.end,
            selectors=torch.cat(selectors),
            destinations=destinations_t,
            piece_sizes=piece_sizes,
            piece_offsets=piece_offsets,
            cu_seqlens_q=_cu_seqlens_from_counts(q_counts),
            cu_seqlens_k=cu_k,
            max_seqlen_q=selected.max_seqlen,
            max_seqlen_k=selected.max_seqlen,
        )
        return StridedChunkLayout(selected, history, key_positions)

    def forward(
        self, z: torch.Tensor, attn_mask=None, position_ids=None,
        inherited_context=None, selected_layout=None,
    ):
        B, T, _ = z.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=z.device).view(1, T).expand(B, T)
        packed = attn_mask if isinstance(attn_mask, PackedAttentionLayout) else None
        selected, q_selected, k_selected, c_selected = self._selected_contexts(
            z, position_ids, packed, selected=selected_layout)
        flat_size = B * T

        def scatter(x: torch.Tensor) -> torch.Tensor:
            shape = (flat_size, x.shape[-2], x.shape[-1])
            return x.new_zeros(shape).index_copy(0, selected.indices, x).view(
                B, T, x.shape[-2], x.shape[-1])

        residual_selected = self.o(c_selected.reshape(-1, S.N))
        if inherited_context is not None:
            q_full, k_full, c_full, residual = sample_hold(
                q_selected, k_selected, c_selected, residual_selected,
                inherited_context, selected.ranks.view(B, T))
        else:
            q_own, k_own, c_own = map(
                scatter, (q_selected, k_selected, c_selected))
            columns = torch.arange(T, device=z.device).view(1, T)
            source_columns = columns - position_ids.remainder(self.token_stride)
            source = (
                torch.arange(B, device=z.device).view(B, 1) * T + source_columns
            ).reshape(-1)
            q_full = q_own.reshape(flat_size, S.N_Q_HEADS, S.HEAD_DIM).index_select(
                0, source).view_as(q_own)
            k_full = k_own.reshape(flat_size, S.N_Q_HEADS, S.HEAD_DIM).index_select(
                0, source).view_as(k_own)
            c_full = c_own.reshape(flat_size, S.N_Q_HEADS, S.HEAD_DIM).index_select(
                0, source).view_as(c_own)
            residual = residual_selected.new_zeros(flat_size, S.N).index_copy(
                0, selected.indices, residual_selected).view(B, T, S.N)
        return residual, q_full, k_full, c_full

    def step(
        self,
        z: torch.Tensor,
        position_id: int,
        k_cache,
        v_cache,
        cache_len: int,
        kv_runtime=None,
        inherited_context=None,
        held_context=None,
    ):
        selected = int(position_id) % self.token_stride == self.token_offset
        if selected:
            return super().step(
                z, position_id, k_cache, v_cache, cache_len,
                kv_runtime=kv_runtime,
            )
        context = inherited_context if inherited_context is not None else held_context
        if context is None:
            raise RuntimeError("sample-and-hold has no prior attention context")
        q, k_ctx, c = context
        return (
            torch.zeros_like(z), q, k_ctx, c,
            k_cache, v_cache, cache_len, context,
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
        B, T, _ = z.shape
        chunk_layout = (
            strided_chunk_layout
            if strided_chunk_layout is not None
            else self.build_chunk_layout(
                position_ids, attention_layout, packed_chunk,
                past_document_chunks, past_position_chunks)
        )
        selected = chunk_layout.selected
        z_selected = z.reshape(B * T, S.N).index_select(0, selected.indices)
        if selected.indices.numel() == 0:
            q_p, k_p, v_p = self._qkv(z_selected)
            projection_dependency = (q_p.sum() + k_p.sum() + v_p.sum()) * 0
            output_dependency = self.o(z_selected).sum() * 0
            context = inherited_context
            if context is None:
                if held_context is None:
                    raise RuntimeError("sample-and-hold has no prior attention context")
                context = tuple(x[:, None].expand(-1, T, -1, -1) for x in held_context)
            k_empty = k_p.view(1, 0, S.N_KV_HEADS, S.HEAD_DIM)
            v_empty = v_p.view(1, 0, S.N_KV_HEADS, S.HEAD_DIM)
            residual = torch.zeros_like(z) + projection_dependency + output_dependency
            return residual, *context, k_empty, v_empty

        q_p, k_p, v_p = self._qkv(z_selected)
        q = q_p.view(-1, S.N_Q_HEADS, S.HEAD_DIM)
        k_new = k_p.view(-1, S.N_KV_HEADS, S.HEAD_DIM)
        v_new = v_p.view(-1, S.N_KV_HEADS, S.HEAD_DIM)
        q, k_new = self.rope(
            q.unsqueeze(0), k_new.unsqueeze(0), selected.positions.unsqueeze(0))
        q = q.squeeze(0).contiguous()
        k_new = k_new.squeeze(0).contiguous()
        v_new = v_new.contiguous()
        k_chunks = (*past_k_chunks, k_new.unsqueeze(0))
        v_chunks = (*past_v_chunks, v_new.unsqueeze(0))
        history_plan = chunk_layout.history
        if history_plan is None:
            raise RuntimeError("non-empty strided chunk has no history plan")
        backend = _varlen_backend(q)
        if S.ATTN_IMPL == "flash" and backend is None:
            raise RuntimeError(
                "attn_impl='flash' requires a validated varlen forward+backward backend")
        if backend is not None:
            fused_value = None
            if _varlen_value_fusion_enabled(backend, q):
                fused_value = _pack_selected_chunk_kv(
                    k_chunks, v_chunks, history_plan)
                k_packed = fused_value[..., :S.HEAD_DIM]
                v_packed = k_packed
            else:
                k_packed = _pack_selected_chunk_history(k_chunks, history_plan)
                v_packed = _pack_selected_chunk_history(v_chunks, history_plan)
            k_ctx, c = _varlen_attention_contexts(
                backend, q, k_packed, v_packed,
                history_plan.cu_seqlens_q, history_plan.cu_seqlens_k,
                history_plan.max_seqlen_q, history_plan.max_seqlen_k,
                fused_value=fused_value)
        else:
            k_packed = _pack_selected_chunk_history(k_chunks, history_plan)
            v_packed = _pack_selected_chunk_history(v_chunks, history_plan)
            k_ctx, c = _selected_chunk_sdpa_contexts(
                q, k_packed, v_packed, selected.positions, chunk_layout.key_positions,
                history_plan.cu_seqlens_q, history_plan.cu_seqlens_k)

        flat_size = B * T

        def scatter(x: torch.Tensor) -> torch.Tensor:
            return x.new_zeros(
                flat_size, x.shape[-2], x.shape[-1]).index_copy(
                0, selected.indices, x).view(B, T, x.shape[-2], x.shape[-1])

        residual_selected = self.o(c.reshape(-1, S.N))
        if inherited_context is not None:
            q_full, k_full, c_full, residual = sample_hold(
                q, k_ctx, c, residual_selected, inherited_context,
                selected.ranks.view(B, T))
        else:
            q_own, k_own, c_own = map(scatter, (q, k_ctx, c))
            columns = torch.arange(T, device=z.device).view(1, T)
            source_columns = columns - position_ids.remainder(self.token_stride)
            local = source_columns.ge(0)
            source = (
                torch.arange(B, device=z.device).view(B, 1) * T
                + source_columns.clamp_min(0)
            ).reshape(-1)
            held = tuple(
                x.reshape(flat_size, S.N_Q_HEADS, S.HEAD_DIM).index_select(
                    0, source).view(B, T, S.N_Q_HEADS, S.HEAD_DIM)
                for x in (q_own, k_own, c_own)
            )
            if held_context is not None:
                previous = tuple(
                    x[:, None].expand(-1, T, -1, -1) for x in held_context)
                held = tuple(
                    torch.where(local.unsqueeze(-1).unsqueeze(-1), own, prior)
                    for own, prior in zip(held, previous)
                )
            q_full, k_full, c_full = held
            residual = residual_selected.new_zeros(flat_size, S.N).index_copy(
                0, selected.indices, residual_selected).view(B, T, S.N)
        return (
            residual, q_full, k_full, c_full,
            k_new.unsqueeze(0), v_new.unsqueeze(0),
        )


class InheritedContextMixer(nn.Module):
    def _outputs(self, z: torch.Tensor, inherited_context):
        if inherited_context is None:
            raise RuntimeError("an inherited-context layer requires an earlier attention layer")
        q, k_ctx, c = inherited_context
        return torch.zeros_like(z), q, k_ctx, c

    def forward(
        self, z: torch.Tensor, attn_mask=None, position_ids=None,
        inherited_context=None, selected_layout=None,
    ):
        del attn_mask, position_ids, selected_layout
        return self._outputs(z, inherited_context)

    def forward_chunk(
        self, z: torch.Tensor, past_k_chunks: tuple, past_v_chunks: tuple,
        position_ids: torch.Tensor, attention_layout, packed_chunk,
        inherited_context=None,
        held_context=None,
        past_document_chunks=(),
        past_position_chunks=(),
        strided_chunk_layout=None,
    ):
        del (
            past_k_chunks, past_v_chunks, position_ids, attention_layout,
            packed_chunk, held_context, past_document_chunks, past_position_chunks,
            strided_chunk_layout,
        )
        attn_out, q, k_ctx, c = self._outputs(z, inherited_context)
        empty_shape = (z.shape[0], 0, S.N_KV_HEADS, S.HEAD_DIM)
        empty = z.new_empty(empty_shape)
        return attn_out, q, k_ctx, c, empty, empty

    def step(
        self, z: torch.Tensor, position_id: int,
        k_cache, v_cache, cache_len: int, kv_runtime=None, inherited_context=None,
        held_context=None,
    ):
        del position_id, kv_runtime, held_context
        attn_out, q, k_ctx, c = self._outputs(z, inherited_context)
        context = (q, k_ctx, c)
        return attn_out, q, k_ctx, c, k_cache, v_cache, cache_len, context

__all__ = ('SelectedTokenLayout', 'StridedChunkLayout', '_selected_token_layout', '_selected_sdpa_contexts', '_selected_chunk_sdpa_contexts', 'StridedGroupedQueryCausalSelfAttention', 'InheritedContextMixer')
