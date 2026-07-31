from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

@dataclass(frozen=True)
class PackedAttentionLayout:
    """Compact document layout shared by every attention consumer.

    ``segment_ids`` is O(B*T), while ``cu_seqlens`` is O(number of documents).
    Neither tensor contains an attention matrix.  Row-major flattening of
    [B,T,...] places every document contiguously in exactly the order described
    by ``cu_seqlens``.
    """

    segment_ids: torch.Tensor       # [B,T] int32, monotonically increasing per row
    position_ids: torch.Tensor      # [B,T] int32, reset to zero at each segment
    cu_seqlens: torch.Tensor        # [num_documents+1] int32
    max_seqlen: int
    chunk_plans: Tuple["PackedChunkLayout", ...] = ()

    @property
    def shape(self) -> Tuple[int, int]:
        return tuple(self.segment_ids.shape)

    def slice_batch(self, item) -> "PackedAttentionLayout":
        segment_ids = self.segment_ids[item]
        position_ids = self.position_ids[item]
        if segment_ids.dim() == 1:
            segment_ids = segment_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)
        return packed_layout_from_segment_ids(
            segment_ids, max_seqlen=self.max_seqlen,
            position_ids=position_ids)

    def __getitem__(self, item) -> "PackedAttentionLayout":
        return self.slice_batch(item)

    def to(self, device: torch.device, non_blocking: bool = False) -> "PackedAttentionLayout":
        return PackedAttentionLayout(
            segment_ids=self.segment_ids.to(device, non_blocking=non_blocking),
            position_ids=self.position_ids.to(device, non_blocking=non_blocking),
            cu_seqlens=self.cu_seqlens.to(device, non_blocking=non_blocking),
            max_seqlen=self.max_seqlen,
            chunk_plans=tuple(
                p.to(device, non_blocking=non_blocking)
                for p in self.chunk_plans),
        )

    def pin_memory(self) -> "PackedAttentionLayout":
        if self.segment_ids.device.type != "cpu":
            return self

        return PackedAttentionLayout(
            segment_ids=self.segment_ids.pin_memory(),
            position_ids=self.position_ids.pin_memory(),
            cu_seqlens=self.cu_seqlens.pin_memory(),
            max_seqlen=self.max_seqlen,
            chunk_plans=tuple(p.pin_memory() for p in self.chunk_plans),
        )


@dataclass(frozen=True)
class PackedChunkLayout:
    """Varlen metadata and gather plan for one temporal query chunk.

    K/V history remains as disjoint autograd-connected chunks.  ``selectors``
    gathers only document prefixes used by this query chunk; ``destinations``
    writes each gathered piece into one document-major THD output allocation.
    Completed documents from old chunks are never copied.
    """

    start: int
    end: int
    selectors: torch.Tensor
    destinations: torch.Tensor
    piece_sizes: Tuple[int, ...]
    piece_offsets: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    # None marks legacy/caller-built plans that did not encode fire metadata;
    # Model then derives boundaries with the historical policy.
    ends_with_fire: Optional[bool] = None

    def to(self, device: torch.device, non_blocking: bool = False) -> "PackedChunkLayout":
        return PackedChunkLayout(
            start=self.start,
            end=self.end,
            selectors=self.selectors.to(device, non_blocking=non_blocking),
            destinations=self.destinations.to(device, non_blocking=non_blocking),
            piece_sizes=self.piece_sizes,
            piece_offsets=self.piece_offsets.to(device, non_blocking=non_blocking),
            cu_seqlens_q=self.cu_seqlens_q.to(device, non_blocking=non_blocking),
            cu_seqlens_k=self.cu_seqlens_k.to(device, non_blocking=non_blocking),
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            ends_with_fire=self.ends_with_fire,
        )

    def pin_memory(self) -> "PackedChunkLayout":
        if self.selectors.device.type != "cpu":
            return self
        return PackedChunkLayout(
            start=self.start,
            end=self.end,
            selectors=self.selectors.pin_memory(),
            destinations=self.destinations.pin_memory(),
            piece_sizes=self.piece_sizes,
            piece_offsets=self.piece_offsets.pin_memory(),
            cu_seqlens_q=self.cu_seqlens_q.pin_memory(),
            cu_seqlens_k=self.cu_seqlens_k.pin_memory(),
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            ends_with_fire=self.ends_with_fire,
        )


def _cu_seqlens_from_counts(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(dtype=torch.int32)
    cu = torch.zeros(counts.numel() + 1, dtype=torch.int32, device=counts.device)
    torch.cumsum(counts, dim=0, out=cu[1:])
    return cu


def packed_layout_from_segment_ids(
    segment_ids: torch.Tensor,
    max_seqlen: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> PackedAttentionLayout:
    if segment_ids.dim() != 2:
        raise ValueError(f"segment_ids must be [B,T], got {tuple(segment_ids.shape)}")
    B, T = segment_ids.shape
    seg = segment_ids.to(dtype=torch.int32)
    if position_ids is None:
        idx = torch.arange(T, device=seg.device, dtype=torch.int32).unsqueeze(0).expand(B, T)
        new_seg = torch.ones_like(seg, dtype=torch.bool)
        new_seg[:, 1:] = seg[:, 1:] != seg[:, :-1]
        starts = torch.cummax(
            torch.where(new_seg, idx, torch.zeros_like(idx)), dim=1).values
        positions = idx - starts
    else:
        if position_ids.shape != segment_ids.shape:
            raise ValueError(
                f"position_ids must match segment_ids, got {tuple(position_ids.shape)} "
                f"vs {tuple(segment_ids.shape)}")
        positions = position_ids.to(device=seg.device, dtype=torch.int32)
    row = torch.arange(B, device=seg.device, dtype=torch.int64).unsqueeze(1)
    codes = row * (T + 1) + seg.to(torch.int64)
    _, counts = torch.unique_consecutive(codes.reshape(-1), return_counts=True)
    if max_seqlen is None:
        max_seqlen = int(counts.max().item()) if counts.numel() else 0
    return PackedAttentionLayout(
        seg, positions, _cu_seqlens_from_counts(counts), int(max_seqlen))


def _unpacked_attention_layout(batch: int, length: int, device: torch.device) -> PackedAttentionLayout:
    seg = torch.zeros(batch, length, dtype=torch.int32, device=device)
    positions = torch.arange(
        length, dtype=torch.int32, device=device).unsqueeze(0).expand(batch, length)
    counts = torch.full((batch,), length, dtype=torch.int32, device=device)
    return PackedAttentionLayout(
        seg, positions, _cu_seqlens_from_counts(counts), int(length))


@torch._dynamo.disable
def build_packed_chunk_layout(
    layout: PackedAttentionLayout,
    start: int,
    end: int,
    chunk_ranges: Tuple[Tuple[int, int], ...],
    ends_with_fire: Optional[bool] = None,
) -> PackedChunkLayout:
    """Build a reusable, tensor-only gather plan for one Tria temporal chunk."""
    segment_ids = layout.segment_ids
    B, T = segment_ids.shape
    if not (0 <= start < end <= T):
        raise ValueError(f"bad packed chunk [{start},{end}) for T={T}")
    if not chunk_ranges or chunk_ranges[-1] != (start, end):
        raise ValueError("chunk_ranges must end with the current query chunk")

    q_seg = segment_ids[:, start:end].to(torch.int64)
    first_active = q_seg[:, :1]
    last_active = q_seg[:, -1:]
    active_counts = (last_active - first_active + 1).flatten()
    row_doc_offsets = torch.zeros(B, dtype=torch.int64, device=segment_ids.device)
    if B > 1:
        torch.cumsum(active_counts[:-1], dim=0, out=row_doc_offsets[1:])
    q_doc = row_doc_offsets[:, None] + (q_seg - first_active)
    q_counts = torch.bincount(q_doc.reshape(-1))

    selectors: List[torch.Tensor] = []
    source_docs: List[torch.Tensor] = []
    source_positions: List[torch.Tensor] = []
    for chunk_start, chunk_end in chunk_ranges:
        chunk_seg = segment_ids[:, chunk_start:chunk_end].to(torch.int64)
        keep = (chunk_seg >= first_active) & (chunk_seg <= last_active)
        local = keep.reshape(-1).nonzero(as_tuple=False).flatten().to(torch.int32)
        selectors.append(local)
        if local.numel() == 0:
            continue
        b_idx, local_pos = keep.nonzero(as_tuple=True)
        docs = (
            row_doc_offsets.index_select(0, b_idx)
            + chunk_seg[b_idx, local_pos]
            - first_active.flatten().index_select(0, b_idx)
        )
        abs_pos = local_pos.to(torch.int64) + int(chunk_start)
        source_docs.append(docs)
        source_positions.append(
            layout.position_ids[b_idx, abs_pos].to(torch.int64))

    if not source_docs:
        raise RuntimeError("packed chunk has queries but selected no K/V tokens")
    source_doc = torch.cat(source_docs)
    k_counts = torch.bincount(source_doc)
    if q_counts.numel() != k_counts.numel():
        raise RuntimeError("packed Q/K document count mismatch")
    cu_k = _cu_seqlens_from_counts(k_counts)
    destination_parts = tuple(
        (
            cu_k[:-1].to(torch.int64).index_select(0, docs)
            + positions
        ).to(torch.int32)
        for docs, positions in zip(source_docs, source_positions)
    )

    piece_sizes = tuple(x.numel() for x in selectors)
    piece_offsets = torch.zeros(
        len(piece_sizes) + 1, dtype=torch.int32, device=segment_ids.device)
    if piece_sizes:
        sizes_tensor = torch.tensor(
            piece_sizes, dtype=torch.int32, device=segment_ids.device)
        torch.cumsum(sizes_tensor, dim=0, out=piece_offsets[1:])

    return PackedChunkLayout(
        start=int(start),
        end=int(end),
        selectors=torch.cat(selectors),
        destinations=torch.cat(destination_parts),
        piece_sizes=piece_sizes,
        piece_offsets=piece_offsets,
        cu_seqlens_q=_cu_seqlens_from_counts(q_counts),
        cu_seqlens_k=cu_k,
        # FlashAttention accepts upper bounds.  Reusing static bounds avoids
        # two GPU->CPU synchronizations per temporal chunk.
        max_seqlen_q=int(end - start),
        max_seqlen_k=int(layout.max_seqlen),
        ends_with_fire=ends_with_fire,
    )


def temporal_chunk_stops(
    idx: torch.Tensor,
    window: int,
    hard_fire_enabled: bool,
    carry_token_id: Optional[int],
    compiling: bool = False,
) -> List[int]:
    """Return inclusive Tria chunk endpoints using the model's exact policy."""
    T = idx.shape[1]
    if compiling:
        boundaries = list(range(window - 1, T, window)) if hard_fire_enabled else []
    else:
        boundary_mask = torch.zeros(T, dtype=torch.bool, device=idx.device)
        if hard_fire_enabled:
            boundary_mask[window - 1:T:window] = True
            boundary_mask[-1] = False
        if carry_token_id is not None:
            boundary_mask |= idx.eq(int(carry_token_id)).any(dim=0)
        boundaries = boundary_mask.nonzero(as_tuple=False).flatten().tolist()
    if not boundaries or boundaries[-1] != T - 1:
        boundaries.append(T - 1)
    return boundaries

__all__ = ('PackedAttentionLayout', 'PackedChunkLayout', '_cu_seqlens_from_counts', 'packed_layout_from_segment_ids', '_unpacked_attention_layout', 'build_packed_chunk_layout', 'temporal_chunk_stops')
