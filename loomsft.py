#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supervised fine-tuning for LoomFormer.

This module owns exactly three SFT-specific things:

  1. the sft_format.json schema check (validate_example),
  2. a one-time tokenized cache of the dataset (SFTCache: flat ids / loss mask /
     example offsets on disk, mmap- or RAM-backed),
  3. a stream (SFTStream) that packs cached examples into fixed-length rows and
     exposes the SAME interface loomformer.py's ShardStream/TokenStream do.

Everything else -- DDP, torch.compile, custom-op graph capture, prefetching,
LR schedule, eval, checkpoints/runpoints/resume -- is loomformer.py's training
loop, reached through lf.train_async(). Nothing is reimplemented here.

Packing uses <eos> as the per-example separator, which makes loomformer's
existing build_doc_reset_state() produce exactly the per-example position reset
and compact varlen document layout SFT needs (the chat template itself never
emits <eos>, only <|im_start|>/<|im_end|>). Attention consumes cu_seqlens on
optimized GPUs and never materializes a block-diagonal T² mask. The only
SFT-specific training tensor is the per-token loss mask.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import queue
import sys
import threading
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import loomformer as lf

IGNORE_INDEX = lf.IGNORE_INDEX

# ============================================================================
# schema validation (sft_format.json, enforced here rather than just documented)
# ============================================================================

ROLE_ENUM = {"system", "user", "assistant", "tool"}


def validate_example(ex: Dict, line_ctx: str = "") -> None:
    msgs = ex.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError(f"{line_ctx}messages must be a non-empty list")
    if msgs[0].get("role") not in ("system", "user"):
        raise ValueError(f"{line_ctx}first turn role must be system/user, got {msgs[0].get('role')!r}")
    open_calls = set()
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role not in ROLE_ENUM:
            raise ValueError(f"{line_ctx}turn {i}: unknown role {role!r}")
        content = m.get("content")
        tool_calls = m.get("tool_calls")
        if role == "assistant":
            if content is None and not tool_calls:
                raise ValueError(f"{line_ctx}turn {i}: assistant content is null with no tool_calls")
            for tc in (tool_calls or []):
                if "id" not in tc or "function" not in tc:
                    raise ValueError(f"{line_ctx}turn {i}: malformed tool_calls entry {tc!r}")
                open_calls.add(tc["id"])
        else:
            if tool_calls:
                raise ValueError(f"{line_ctx}turn {i}: tool_calls only allowed on assistant turns")
            if role != "tool" and content is None:
                raise ValueError(f"{line_ctx}turn {i}: {role} turn must have non-null content")
        if role == "tool":
            tcid = m.get("tool_call_id")
            if tcid not in open_calls:
                raise ValueError(f"{line_ctx}turn {i}: tool_call_id {tcid!r} has no matching preceding tool_calls entry")
            open_calls.discard(tcid)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            yield lineno, raw


def _iter_examples(path: str):
    """Yield numbered examples from JSONL, an Arrow, or a Parquet ``messages`` column.

    Reads record-batch-by-record-batch rather than materializing the whole
    table: pyarrow's nested (struct-in-list) type concatenation across
    internal batches is unreliable at scale (ArrowNotImplementedError:
    "Nested data conversions not implemented for chunked array outputs"),
    but every individual RecordBatch is itself unchunked, so per-batch
    to_pylist() always works.
    """
    if path.endswith(".arrow") or path.endswith(".feather"):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        i = 0
        with pa.memory_map(path, "r") as src:
            try:
                reader = ipc.open_file(src)
                batches = [reader.get_batch(b) for b in range(reader.num_record_batches)]
            except pa.lib.ArrowInvalid:
                src.seek(0)
                reader = ipc.open_stream(src)
                batches = reader
            for batch in batches:
                if "messages" not in batch.schema.names:
                    raise ValueError(f"{path}: expected a 'messages' column, got {batch.schema.names}")
                for row in batch.column("messages").to_pylist():
                    i += 1
                    yield i, {"messages": row}
    elif path.endswith(".parquet"):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        if "messages" not in pf.schema_arrow.names:
            raise ValueError(f"{path}: expected a 'messages' column, got {pf.schema_arrow.names}")
        i = 0
        for batch in pf.iter_batches(columns=["messages"]):
            for row in batch.column("messages").to_pylist():
                i += 1
                yield i, {"messages": row}
    else:
        for lineno, raw in _iter_jsonl(path):
            yield lineno, json.loads(raw)


# ============================================================================
# tokenized cache
#
# Render + tokenize the dataset exactly ONCE into three flat arrays on disk
# (ids, per-token loss mask, per-example offsets), keyed by dataset + tokenizer
# + seq_len. Later runs mmap it: no re-tokenization, no Jinja, and -- with
# dataset_cache: mmap -- no multi-GB resident pool either. Everything the hot
# path touches from here on is vectorized numpy over these arrays.
# ============================================================================

CACHE_VERSION = 1


def _cache_key(dataset: str, cfg: "lf.Config") -> str:
    st = os.stat(dataset)
    parts = [f"v{CACHE_VERSION}", os.path.abspath(dataset), str(st.st_size),
             str(st.st_mtime_ns), str(cfg.tokenizer), str(int(cfg.vocab)), str(int(cfg.seq_len))]
    for dep in (str(cfg.tokenizer or ""), "chat_template.jinja"):
        if dep and os.path.exists(dep):
            parts.append(f"{dep}:{os.stat(dep).st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class SFTCache:
    """Flat (ids, loss_mask, offsets) view of a tokenized SFT dataset."""

    __slots__ = ("ids", "mask", "off", "path")

    _FLUSH_TOKENS = 1 << 22  # write-out granularity while building (~4M tokens)

    def __init__(self, ids: np.ndarray, mask: np.ndarray, off: np.ndarray, path: str):
        self.ids = ids
        self.mask = mask
        self.off = off
        self.path = path

    @property
    def n_examples(self) -> int:
        return int(len(self.off) - 1)

    @property
    def lengths(self) -> np.ndarray:
        return (self.off[1:] - self.off[:-1]).astype(np.int64)

    @staticmethod
    def build_or_load(dataset: str, cfg: "lf.Config", tok, device: torch.device) -> "SFTCache":
        cache_dir = f"{dataset}.sftcache-{_cache_key(dataset, cfg)}"
        meta_path = os.path.join(cache_dir, "meta.json")
        if not os.path.exists(meta_path) and lf.ddp_is_main():
            SFTCache._build(dataset, cfg, tok, cache_dir)
        lf.ddp_barrier(device)
        if not os.path.exists(meta_path):
            raise RuntimeError(f"{cache_dir}: cache build did not complete on the main rank")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        mode = str(getattr(cfg, "dataset_cache", None) or "mmap").lower()
        if mode not in ("mmap", "ram"):
            raise ValueError(f"dataset_cache must be 'mmap' or 'ram', got {mode!r}")
        n_tok, n_ex = int(meta["tokens"]), int(meta["examples"])

        def _open(name: str, dtype, count: int) -> np.ndarray:
            p = os.path.join(cache_dir, name)
            if mode == "ram":
                return np.fromfile(p, dtype=dtype, count=count)
            return np.memmap(p, dtype=dtype, mode="r", shape=(count,))

        cache = SFTCache(_open("ids.i32", np.int32, n_tok), _open("mask.u8", np.int8, n_tok),
                         _open("off.i64", np.int64, n_ex + 1), cache_dir)
        lf.ddp_print(f"[sft-cache] {mode} {cache_dir}: {n_ex:,} examples, {n_tok:,} tokens, "
                     f"{int(meta['loss_tokens']):,} loss-carrying "
                     f"({int(meta['loss_tokens']) / max(1, n_tok):.1%})")
        return cache

    @staticmethod
    def _build(dataset: str, cfg: "lf.Config", tok, cache_dir: str) -> None:
        chat = lf.ChatTemplate(tok)
        seq_len = int(cfg.seq_len)
        tmp_dir = cache_dir + ".partial"
        os.makedirs(tmp_dir, exist_ok=True)
        ids_path = os.path.join(tmp_dir, "ids.i32")
        mask_path = os.path.join(tmp_dir, "mask.u8")
        t0 = time.time()
        n_seen = n_kept = n_long = n_no_loss = 0
        n_tok = n_loss_tok = 0
        offsets = [0]
        buf_ids: List[np.ndarray] = []
        buf_mask: List[np.ndarray] = []
        buffered = 0
        print(f"[sft-cache] building {cache_dir} (one-time tokenization of {dataset})", flush=True)
        with open(ids_path, "wb") as f_ids, open(mask_path, "wb") as f_mask:
            def _flush() -> None:
                nonlocal buffered
                if not buf_ids:
                    return
                np.concatenate(buf_ids).astype(np.int32, copy=False).tofile(f_ids)
                np.concatenate(buf_mask).astype(np.int8, copy=False).tofile(f_mask)
                buf_ids.clear()
                buf_mask.clear()
                buffered = 0

            for line_ctx, ex in _iter_examples(dataset):
                n_seen += 1
                validate_example(ex, line_ctx=f"{dataset}:{line_ctx}: ")
                ids, mask = chat.render_training_ids(ex["messages"], tools=ex.get("tools"))
                # A packed row is seq_len+1 tokens wide and every example is
                # followed by one <eos> separator, so seq_len is the hard cap.
                if len(ids) > seq_len:
                    n_long += 1
                    continue
                if not any(mask):
                    n_no_loss += 1
                    continue  # zero loss-carrying tokens teaches nothing
                buf_ids.append(np.asarray(ids, dtype=np.int32))
                buf_mask.append(np.asarray(mask, dtype=np.int8))
                buffered += len(ids)
                n_tok += len(ids)
                n_loss_tok += int(sum(mask))
                n_kept += 1
                offsets.append(n_tok)
                if buffered >= SFTCache._FLUSH_TOKENS:
                    _flush()
                if n_seen % 50000 == 0:
                    print(f"[sft-cache] {n_seen:,} read, {n_kept:,} kept, {n_tok:,} tokens "
                          f"({time.time() - t0:.0f}s)", flush=True)
            _flush()
        if n_kept == 0:
            raise ValueError(f"{dataset}: no examples survived validation/rendering at seq_len={seq_len}")
        np.asarray(offsets, dtype=np.int64).tofile(os.path.join(tmp_dir, "off.i64"))
        with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"version": CACHE_VERSION, "dataset": os.path.abspath(dataset),
                       "seq_len": seq_len, "examples": n_kept, "tokens": n_tok,
                       "loss_tokens": n_loss_tok, "read": n_seen,
                       "dropped_too_long": n_long, "dropped_no_loss": n_no_loss}, f)
        os.replace(tmp_dir, cache_dir)  # atomic: a cache dir exists only when complete
        print(f"[sft-cache] done in {time.time() - t0:.0f}s: {n_kept:,} kept, {n_long:,} dropped "
              f"(> seq_len={seq_len}), {n_no_loss:,} dropped (no loss tokens)", flush=True)


# ============================================================================
# vectorized packing
#
# All of this is ragged-array arithmetic (cumsum/repeat/searchsorted). There is
# no per-token or per-example Python iteration on the batch path: one batch is
# two gathers plus a handful of index computations.
# ============================================================================

def _ragged_arange(counts: np.ndarray) -> np.ndarray:
    """Concatenated [0..counts[0]-1, 0..counts[1]-1, ...] without a Python loop."""
    total = int(counts.sum())
    ends = np.cumsum(counts)
    return np.arange(total, dtype=np.int64) - np.repeat(ends - counts, counts)


def _ragged_slice(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenation of ranges [starts[i], starts[i]+counts[i])."""
    return np.repeat(starts, counts) + _ragged_arange(counts)


def _group_exclusive_cumsum(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Exclusive cumsum of ``values`` restarted at every group boundary."""
    incl = np.cumsum(values)
    group_ends = np.cumsum(counts) - 1
    group_base = np.concatenate(([0], incl[group_ends[:-1]])) if len(counts) > 1 else np.zeros(1, np.int64)
    return incl - values - np.repeat(group_base, counts)


def _need_pad_id(tok) -> int:
    tid = tok.special_id("<pad>")
    if tid is None:
        raise ValueError(
            "tokenizer is missing <pad>. Retrain it with "
            "loomformer.train_tokenizer(..., special_tokens=loomformer.DEFAULT_SPECIAL_TOKENS)."
        )
    return tid


def cached_token_count(path: str, cfg: "lf.Config") -> int:
    """Exact token count of a split, read from the cache metadata.

    Used by loomformer's budget report instead of re-reading the dataset.
    """
    base, _, frag = str(path).partition("#")
    meta_path = os.path.join(f"{base}.sftcache-{_cache_key(base, cfg)}", "meta.json")
    if not os.path.exists(meta_path):
        return 0  # first run: the cache is built by the stream a moment later
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    tokens = int(meta["tokens"])
    pct = float(getattr(cfg, "auto_val_split_pct", 0.0) or 0.0)
    if pct <= 0.0 or not frag:
        return tokens
    # Example counts are exact; the token split is proportional to them.
    n = int(meta["examples"])
    n_val = lf._split_count(n, pct)
    share = (n_val if frag.lower() == "val" else n - n_val) / max(1, n)
    return int(tokens * share)


class SFTStream:
    """Packed SFT batches behind loomformer's stream interface.

    Yields ``(ids, loss_mask, packed_layout)`` where tensors have shape
    ``[B, seq_len+1]`` and varlen/chunk metadata is prepared by the background
    CPU packer; everything else (attention, LR, DDP, compile, graph, eval,
    checkpoints) is loomformer's training loop, unchanged. Examples are joined
    by a single <eos>, which is exactly the boundary token
    ``build_doc_reset_state`` splits on, so packed examples cannot attend across
    each other and every example restarts at position 0.

    Path syntax: ``<dataset>`` or ``<dataset>#val`` / ``<dataset>#train`` --
    the fragment selects one side of the ``auto_val_split_pct`` example split
    over one shared tokenized cache (non-destructive; the dataset file is never
    rewritten, unlike the pretraining shard split).
    """

    def __init__(self, path: str, cfg: "lf.Config", device: torch.device, tokenizer=None):
        base, _, frag = str(path).partition("#")
        frag = frag.lower()
        if frag not in ("", "train", "val"):
            raise ValueError(f"unknown SFT split fragment {frag!r} in {path!r} (use #train or #val)")
        self.cfg = cfg
        self.device = device
        self.tok = tokenizer if tokenizer is not None else lf.build_tokenizer(cfg)
        self._eos_id = lf._tok_special_id(self.tok, "<eos>")
        if self._eos_id is None:
            raise ValueError(
                "SFT packing needs <eos> as the example separator, but the tokenizer has none. "
                "Retrain it with loomformer.train_tokenizer(..., "
                "special_tokens=loomformer.DEFAULT_SPECIAL_TOKENS)."
            )
        self._pad_id = _need_pad_id(self.tok)
        self._carry_id = lf._tok_special_id(self.tok, "<CARRY>")
        self.cache = SFTCache.build_or_load(base, cfg, self.tok, device)
        self._lengths = self.cache.lengths
        self.T1 = int(cfg.seq_len) + 1
        self.is_val = frag == "val"

        n = self.cache.n_examples
        pct = float(getattr(cfg, "auto_val_split_pct", 0.0) or 0.0)
        n_val = lf._split_count(n, pct) if pct > 0.0 else 0
        lo, hi = (n - n_val, n) if self.is_val else (0, n - n_val)
        if hi <= lo:
            raise ValueError(f"{path}: empty split (examples={n}, auto_val_split_pct={pct})")
        # DDP: contiguous, disjoint example ranges -- same partitioning idea as
        # ShardStream's row plan, without any cross-rank duplication.
        rank = lf.ddp_rank() if lf.ddp_is_distributed() else 0
        world = lf.ddp_world_size() if lf.ddp_is_distributed() else 1
        span = hi - lo
        r_lo = lo + span * rank // world
        r_hi = lo + span * (rank + 1) // world
        if r_hi <= r_lo:
            raise ValueError(f"rank {rank} received no SFT examples from {path!r}")
        self._examples = np.arange(r_lo, r_hi, dtype=np.int64)
        self._rng = np.random.default_rng(int(cfg.seed) + 7919 * rank)
        self._row_cursor = 0
        self._plan_epoch()
        lf.ddp_print(f"[sft-data] rank={rank} split={frag or 'train'} examples={len(self._examples):,} "
                     f"rows/epoch={self.n_rows:,} pack_fill={self._fill:.1%}")

        self._ram_queue: "queue.Queue" = queue.Queue(maxsize=max(1, int(cfg.prefetch_batches)))
        self._stop = threading.Event()
        self._producer_error: Optional[BaseException] = None
        self._gpu_ids = None
        self._gpu_mask = None
        self._gpu_layouts = None
        self._gpu_pos = 0
        self._rank = rank
        # The producer starts on first use, not here: resume fast-forward and the
        # full-split eval drive the packing plan directly from the calling thread,
        # and a background producer racing them would silently reorder data.
        self._producer: Optional[threading.Thread] = None

    # -- packing plan ------------------------------------------------------

    def _plan_epoch(self) -> None:
        """Greedy first-fit-in-order packing plan for one pass over the split.

        Computed once per epoch, not per batch: row boundaries are found with
        O(rows) C-level searchsorted calls over one cumsum, so the per-batch
        path below only does index arithmetic and two gathers.
        """
        order = self._rng.permutation(self._examples) if not self.is_val else self._examples
        # +1 per example for its <eos> separator; the cache guarantees len <= seq_len,
        # so every example fits in a row of T1 = seq_len + 1 tokens.
        costs = self._lengths[order] + 1
        incl = np.cumsum(costs)
        starts = [0]
        pos = 0
        n = len(order)
        while pos < n:
            budget = (incl[pos - 1] if pos else 0) + self.T1
            nxt = int(np.searchsorted(incl, budget, side="right"))
            pos = nxt if nxt > pos else pos + 1
            starts.append(pos)
        self._order = order
        self._row_ptr = np.asarray(starts, dtype=np.int64)
        self.n_rows = len(self._row_ptr) - 1
        self._row_cursor = 0
        used = int(costs.sum())
        self._fill = used / float(max(1, self.n_rows * self.T1))

    def _take_rows(self, count: int) -> np.ndarray:
        """Next ``count`` row indices, re-planning (reshuffling) at epoch end."""
        out = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            take = min(count - filled, self.n_rows - self._row_cursor)
            out[filled:filled + take] = np.arange(self._row_cursor, self._row_cursor + take)
            self._row_cursor += take
            filled += take
            if self._row_cursor >= self.n_rows:
                if self.is_val:
                    self._row_cursor = 0  # deterministic cycle, no reshuffle
                else:
                    self._plan_epoch()
        return out

    def fast_forward(self, n_batches: int) -> None:
        """Skip ``n_batches`` already-seen batches without packing any of them.

        Epoch plans are a pure function of the seed, so replaying the cursor
        (and the reshuffles it crosses) reproduces the original data order.
        """
        if self._producer is not None:
            raise RuntimeError("fast_forward must run before the data producer starts")
        rows = int(n_batches) * int(self.cfg.batch_size)
        while rows > 0:
            take = min(rows, self.n_rows - self._row_cursor)
            self._row_cursor += take
            rows -= take
            if self._row_cursor >= self.n_rows:
                if self.is_val:
                    self._row_cursor = 0
                else:
                    self._plan_epoch()

    # -- batch materialization (vectorized) --------------------------------

    def _pack_batch_np(self, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        B = len(rows)
        ids = np.full((B, self.T1), self._pad_id, dtype=np.int64)
        mask = np.zeros((B, self.T1), dtype=np.int8)

        ex_start = self._row_ptr[rows]
        ex_count = self._row_ptr[rows + 1] - ex_start
        ex = self._order[_ragged_slice(ex_start, ex_count)]
        row_of_ex = np.repeat(np.arange(B, dtype=np.int64), ex_count)
        ex_len = self._lengths[ex]
        # column where each example starts inside its row (its <eos> costs 1)
        col0 = _group_exclusive_cumsum(ex_len + 1, ex_count)

        local = _ragged_arange(ex_len)
        dst_row = np.repeat(row_of_ex, ex_len)
        dst_col = np.repeat(col0, ex_len) + local
        src = np.repeat(self.cache.off[ex], ex_len) + local
        ids[dst_row, dst_col] = self.cache.ids[src]
        mask[dst_row, dst_col] = self.cache.mask[src]

        ids[row_of_ex, col0 + ex_len] = self._eos_id  # separator: never a target
        mask[row_of_ex, col0] = 0  # never predict an example's first token from the previous <eos>
        # Exact max document length for x=ids[:,:-1].  Computing this beside
        # the CPU packing plan avoids the GPU max().item() synchronization that
        # FlashAttention/Transformer Engine would otherwise need every step.
        x = ids[:, :-1]
        boundary = x == self._eos_id
        seg = np.cumsum(boundary, axis=1, dtype=np.int64) - boundary
        max_seqlen = 0
        for row_seg in seg:
            changes = np.flatnonzero(row_seg[1:] != row_seg[:-1]) + 1
            edges = np.concatenate(([0], changes, [row_seg.size]))
            max_seqlen = max(max_seqlen, int(np.diff(edges).max()))
        return ids, mask, max_seqlen

    # -- loomformer stream interface ---------------------------------------

    def _sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, "lf.PackedAttentionLayout"]:
        ids, mask, max_seqlen = self._pack_batch_np(
            self._take_rows(int(self.cfg.batch_size)))
        t_ids = torch.from_numpy(ids)
        t_mask = torch.from_numpy(mask)
        x = t_ids[:, :-1]
        _position_ids, layout = lf.build_doc_reset_state(
            x, self._eos_id, max_seqlen=max_seqlen)
        if self.device.type == "cuda":
            t_ids, t_mask = t_ids.pin_memory(), t_mask.pin_memory()
            layout = layout.pin_memory()
        return t_ids, t_mask, layout

    def _attach_chunk_plans(
        self,
        ids: torch.Tensor,
        layout: "lf.PackedAttentionLayout",
    ) -> "lf.PackedAttentionLayout":
        if not (
            bool(getattr(self.cfg, "tria_carry_enabled", False))
            and bool(getattr(self.cfg, "tria_temporal_enabled", True))
        ):
            return layout
        x = ids[:, :-1]
        window = int(self.cfg.tria_temporal_window)
        stops = lf.temporal_chunk_stops(
            x, window, True, self._carry_id,
            compiling=bool(getattr(self.cfg, "compile", False)))
        ranges = []
        plans = []
        start = 0
        for stop in stops:
            end = min(int(stop) + 1, x.shape[1])
            if end <= start:
                continue
            ranges.append((start, end))
            plans.append(lf.build_packed_chunk_layout(
                layout, start, end, tuple(ranges)))
            start = end
        if start != x.shape[1]:
            raise RuntimeError(
                f"CPU SFT chunk plan stopped at {start}, expected {x.shape[1]}")
        enriched = replace(layout, chunk_plans=tuple(plans))
        return enriched.pin_memory() if self.device.type == "cuda" else enriched

    def _produce_cpu_batches(self) -> None:
        try:
            while not self._stop.is_set():
                self._ram_queue.put(self._sample_batch())
        except BaseException as exc:  # noqa: BLE001 -- must reach the consumer
            self._producer_error = exc
            try:
                self._ram_queue.put_nowait(None)
            except queue.Full:
                pass

    def _ensure_producer(self) -> None:
        if self._producer is None:
            self._producer = threading.Thread(target=self._produce_cpu_batches, daemon=True,
                                              name=f"sft-data-rank-{self._rank}")
            self._producer.start()

    def _get_raw_cpu_batch(self) -> Tuple[torch.Tensor, torch.Tensor, "lf.PackedAttentionLayout"]:
        self._ensure_producer()
        batch = self._ram_queue.get()
        if batch is None:
            raise RuntimeError("SFT data producer failed") from self._producer_error
        return batch

    def _attach_cpu_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, "lf.PackedAttentionLayout"],
    ) -> Tuple[torch.Tensor, torch.Tensor, "lf.PackedAttentionLayout"]:
        ids, mask, layout = batch
        return ids, mask, self._attach_chunk_plans(ids, layout)

    def _get_cpu_batch(self) -> Tuple[torch.Tensor, torch.Tensor, "lf.PackedAttentionLayout"]:
        return self._attach_cpu_batch(self._get_raw_cpu_batch())

    def sample_device_batch(self) -> Tuple[torch.Tensor, torch.Tensor, "lf.PackedAttentionLayout"]:
        ids, mask, layout = self._get_cpu_batch()
        return (
            ids.to(self.device, non_blocking=True),
            mask.to(self.device, non_blocking=True),
            layout.to(self.device, non_blocking=True),
        )

    def _gpu_chunk_size(self) -> int:
        return max(1, min(int(self.cfg.gpu_prefetch_batches), int(self.cfg.prefetch_batches)))

    async def _load_gpu_chunk(self, count: int) -> None:
        loop = asyncio.get_running_loop()
        # Dequeue serially to preserve the deterministic packing order, then
        # build independent integer gather plans in parallel.
        raw_batches = []
        for _ in range(count):
            raw_batches.append(
                await loop.run_in_executor(None, self._get_raw_cpu_batch))
        batches = await asyncio.gather(*[
            loop.run_in_executor(None, self._attach_cpu_batch, batch)
            for batch in raw_batches
        ])
        ids_l, mask_l, layouts = [], [], []
        for ids, mask, layout in batches:
            ids_l.append(ids)
            mask_l.append(mask)
            layouts.append(layout)
        self._gpu_ids = torch.stack(ids_l).to(self.device, non_blocking=True)
        self._gpu_mask = torch.stack(mask_l).to(self.device, non_blocking=True)
        # Keep future plans in pinned RAM. Moving every prefetched batch's
        # gather indices to VRAM would multiply metadata by gpu_prefetch_batches
        # for no compute benefit; transfer only the batch being yielded.
        self._gpu_layouts = layouts
        self._gpu_pos = 0

    async def prime(self) -> None:
        count = self._gpu_chunk_size()
        await self._load_gpu_chunk(count)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        lf.ddp_print(f"[sft-data] ready: RAM={int(self.cfg.prefetch_batches)} batches, GPU={count} batches")

    async def batches(self, n: int):
        chunk = self._gpu_chunk_size()
        yielded = 0
        while yielded < n:
            if self._gpu_ids is None or self._gpu_pos >= self._gpu_ids.shape[0]:
                await self._load_gpu_chunk(min(chunk, n - yielded))
            while self._gpu_pos < self._gpu_ids.shape[0] and yielded < n:
                out = (
                    self._gpu_ids[self._gpu_pos],
                    self._gpu_mask[self._gpu_pos],
                    self._gpu_layouts[self._gpu_pos].to(
                        self.device, non_blocking=True),
                )
                self._gpu_pos += 1
                yielded += 1
                yield out

    def close(self) -> None:
        self._stop.set()


def train_sft(
    cfg: "lf.Config",
    train_path: str,
    val_path: Optional[str],
    init_checkpoint: Optional[str],
    device: torch.device,
    ckpt_out: str,
    resume: Optional[str] = None,
    resume_step: Optional[int] = None,
) -> None:
    """Run SFT on loomformer's training loop.

    The only SFT-specific setup is selecting the SFT stream (dataset_format) and
    pointing the loop at a pretrained checkpoint for weight initialization; DDP,
    compile, graph capture, prefetching, eval, checkpoints, runpoints and resume
    are the pretraining implementations, used as-is.
    """
    cfg.dataset_format = "sft"
    if val_path:
        cfg.val_dataset = val_path
    elif float(getattr(cfg, "auto_val_split_pct", 0.0) or 0.0) > 0.0:
        # Non-destructive example-level split over the shared tokenized cache.
        cfg.val_dataset = f"{train_path}#val"
        train_path = f"{train_path}#train"
    asyncio.run(lf.train_async(
        cfg, train_path, device, ckpt_out, ablation=False,
        resume=resume, resume_step=resume_step, init_weights=init_checkpoint,
    ))


# ============================================================================
# smoke test
# ============================================================================

def smoke_test() -> None:
    import tempfile

    cfg = lf.Config(vocab=256, seq_len=320, batch_size=2, model_dim=16, n_q_heads=2,
                     head_dim=8, n_kv_heads=1, hidden=32, layers=1,
                     steps=3, warmup_steps=1, log_every=1)
    lf.apply_config(cfg)

    d = tempfile.mkdtemp()
    corpus_dir = os.path.join(d, "raw")
    os.makedirs(corpus_dir, exist_ok=True)
    with open(os.path.join(corpus_dir, "a.txt"), "w") as f:
        f.write("hello world " * 200)
    tok_path = os.path.join(d, "tok.json")
    lf.train_tokenizer(corpus_dir, 256, tok_path)
    cfg.tokenizer = tok_path
    tok = lf.build_tokenizer(cfg)

    examples = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello world"}]},
        {"messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "call a tool"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "<think>done</think>ok!"},
        ], "tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]},
    ]
    sft_path = os.path.join(d, "train.jsonl")
    with open(sft_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    for ex in examples:
        validate_example(ex)
    chat = lf.ChatTemplate(tok)
    ids, mask = chat.render_training_ids(examples[1]["messages"], tools=examples[1].get("tools"))
    assert len(ids) == len(mask) and len(ids) > 0
    print(f"[smoke] ChatTemplate.render_training_ids OK: {len(ids)} tokens, {sum(mask)} loss-carrying")
    parsed = chat.parse_tool_calls(tok.decode(ids))
    assert parsed and parsed[0]["function"]["name"] == "f"
    print(f"[smoke] ChatTemplate.parse_tool_calls OK: {parsed}")

    dev = lf.device_auto("cpu")
    cfg.dataset_format = "sft"
    stream = lf.make_stream(sft_path, cfg, dev)
    assert stream.cache.n_examples == 2, stream.cache.n_examples
    print(f"[smoke] SFTCache OK: {stream.cache.n_examples} examples, "
          f"{int(stream.cache.lengths.sum())} tokens")

    ids, loss_mask, cpu_layout = stream._sample_batch()
    assert ids.shape == (cfg.batch_size, cfg.seq_len + 1), ids.shape
    assert loss_mask.shape == ids.shape

    x, y, position_ids, attn_layout = lf.split_train_batch(
        (ids.to(dev), loss_mask.to(dev), cpu_layout.to(dev)), stream._eos_id)
    assert x.shape == (cfg.batch_size, cfg.seq_len)
    assert isinstance(attn_layout, lf.PackedAttentionLayout)
    assert attn_layout.segment_ids.shape == (cfg.batch_size, cfg.seq_len)
    assert attn_layout.cu_seqlens.dtype == torch.int32
    assert attn_layout.segment_ids.numel() == cfg.batch_size * cfg.seq_len
    assert attn_layout.max_seqlen == cpu_layout.max_seqlen
    n_loss = int((y != IGNORE_INDEX).sum().item())
    assert n_loss > 0, "packed batch carries no loss tokens"
    # Only assistant tokens may be targets: everything the renderer masked out
    # (prompt, padding, separators) must be IGNORE_INDEX.
    assert torch.equal((y != IGNORE_INDEX), loss_mask.to(dev)[:, 1:].bool())
    print(f"[smoke] packing + compact layout OK: x={tuple(x.shape)} "
          f"segments={attn_layout.cu_seqlens.numel() - 1} loss_tokens={n_loss}")

    # Segment ids exactly as build_doc_reset_state derives them: the exclusive
    # cumsum keeps each <eos> separator inside the example it terminates.
    boundary = (x[0] == stream._eos_id).long()
    seg = torch.cumsum(boundary, 0) - boundary
    assert int(position_ids[0].max().item()) < cfg.seq_len
    assert bool((position_ids[0][seg == 0] == torch.arange(int((seg == 0).sum()))).all())
    print(f"[smoke] per-example position reset OK: {int(seg.max().item()) + 1} segments in row 0")

    model = lf.Model(cfg).to(dev)
    with torch.no_grad(), lf.amp_autocast(dev):
        loss = model(x, attn_mask=attn_layout, position_ids=position_ids, labels=y,
                     ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss)
    print(f"[smoke] masked forward OK, loss={loss.item():.4f}")

    # Cross-example isolation, empirically: perturbing the first packed example
    # (its <eos> included) must not move a single logit of the later ones.
    first_len = int((seg == 0).sum())
    later = (seg > 0)
    if first_len > 0 and bool(later.any()):
        with torch.no_grad():
            one_layout = attn_layout[:1]
            base = model(x[:1], attn_mask=one_layout, position_ids=position_ids[:1])
            x2 = x[:1].clone()
            x2[0, :first_len] = (x2[0, :first_len] + 1) % int(cfg.vocab)
            other = model(x2, attn_mask=one_layout, position_ids=position_ids[:1])
        delta = (base[0, later] - other[0, later]).abs().max().item()
        assert delta < 1e-4, f"packed examples are NOT isolated: max delta {delta}"
        print(f"[smoke] cross-example isolation OK (max delta {delta:.2e})")
    stream.close()
    print("[smoke] ALL OK")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    """Thin front-end for `loomformer.py --sft-dataset ...`.

    Every flag is forwarded verbatim, so SFT runs go through the same CLI as
    pretraining: same config loading, same multi-GPU self-launch, same
    checkpoint/resume handling. `--smoke-test` is the only local action.
    """
    if "--smoke-test" in sys.argv[1:]:
        smoke_test()
        return
    if not any(a == "--sft-dataset" or a.startswith("--sft-dataset=") for a in sys.argv[1:]):
        raise SystemExit("loomsft: --sft-dataset is required (see loomformer.py --help)")
    lf.main()


if __name__ == "__main__":
    main()
