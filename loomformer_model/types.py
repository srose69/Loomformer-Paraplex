from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import torch

from . import state as S

@dataclass
class LayerCache:
    k: Optional[torch.Tensor] = None            # [B,SEQ_LEN,N_KV_HEADS,HEAD_DIM] -- preallocated
    v: Optional[torch.Tensor] = None            # [B,SEQ_LEN,N_KV_HEADS,HEAD_DIM] -- preallocated
    phase_trace: Optional[torch.Tensor] = None  # [B,HIDDEN]
    cache_pos: int = 0
    cache_capacity: int = 0
    attn_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    @property
    def cache_len(self) -> int:
        return self.cache_pos


@dataclass
class InferenceKVRuntime:
    """Placement and staging policy for inference KV.

    ``storage_device`` is where layer K/V buffers remain for the whole turn.
    CPU storage streams pinned chunks through a calibrated compute-GPU staging ring;
    CUDA storage executes the parameter-free attention reduction on that GPU
    and returns only the compact contexts to the model device.
    """
    storage_device: torch.device
    compute_device: torch.device
    chunk_size: int = 1024
    preload_chunks: int = 1
    copy_us: float = 0.0
    consume_us: float = 0.0
    safety_chunks: int = 2
    _cpu_buffers: dict = field(default_factory=dict, repr=False)
    _copy_streams: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.storage_device = torch.device(self.storage_device)
        self.compute_device = torch.device(self.compute_device)
        self.chunk_size = max(1, int(self.chunk_size))
        self.preload_chunks = max(1, int(self.preload_chunks))
        self.copy_us = max(0.0, float(self.copy_us))
        self.consume_us = max(0.0, float(self.consume_us))
        self.safety_chunks = max(0, int(self.safety_chunks))

    @property
    def is_local(self) -> bool:
        return self.storage_device == self.compute_device

    def preload_for(self, cache_len: int) -> int:
        chunks = max(1, math.ceil(max(1, int(cache_len)) / self.chunk_size))
        if self.copy_us <= 0.0 or self.consume_us <= 0.0:
            return min(chunks, self.preload_chunks)
        if self.copy_us <= self.consume_us:
            required = 1
        else:
            required = math.ceil(
                chunks - (chunks - 1) * self.consume_us / self.copy_us)
        return min(chunks, max(1, required + self.safety_chunks))

    def cpu_staging(
        self,
        batch: int,
        dtype: torch.dtype,
        preload_chunks: Optional[int] = None,
    ) -> Tuple[list, torch.cuda.Stream]:
        preload = self.preload_chunks if preload_chunks is None else max(1, int(preload_chunks))
        key = (
            int(batch), dtype, int(self.chunk_size), preload,
            str(self.compute_device),
        )
        buffers = self._cpu_buffers.get(key)
        if buffers is None:
            shape = (batch, self.chunk_size, S.N_KV_HEADS, S.HEAD_DIM)
            buffers = [
                (
                    torch.empty(shape, dtype=dtype, device=self.compute_device),
                    torch.empty(shape, dtype=dtype, device=self.compute_device),
                )
                for _ in range(max(2, preload + 1))
            ]
            self._cpu_buffers[key] = buffers
        stream = self._copy_streams.get(str(self.compute_device))
        if stream is None:
            stream = torch.cuda.Stream(device=self.compute_device)
            self._copy_streams[str(self.compute_device)] = stream
        return buffers, stream


@dataclass
class TriaTemporalState:
    carry: Optional[torch.Tensor] = None          # [B,H,3,3] last document_carry
    refeed_pending: Optional[torch.Tensor] = None  # [B] bool: feed `carry` into L0 of current token


@dataclass
class TrainChunkLayerState:
    # Keep only disjoint per-chunk K/V tensors. Returning/storing a growing
    # concatenated prefix at every chunk retains O(T^2/W) overlapping views in
    # the autograd graph. The full prefix is rebuilt only inside the checkpointed
    # block and never escapes it, so persistent state stays O(T).
    k_chunks: tuple = ()                       # tuple[[B,Q,N_KV_HEADS,HEAD_DIM], ...]
    v_chunks: tuple = ()                       # tuple[[B,Q,N_KV_HEADS,HEAD_DIM], ...]
    phase_trace: Optional[torch.Tensor] = None  # [B,HIDDEN] -- ParaplexFFN continuity across chunks
    document_chunks: tuple = ()
    position_chunks: tuple = ()
    attn_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

__all__ = ('LayerCache', 'InferenceKVRuntime', 'TriaTemporalState', 'TrainChunkLayerState')
