#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loomsft.py -- SFT on top of LoomFormer.

Consumes the jsonl schema documented in sft.md / sft_format.json (OpenAI-style
messages, with first-class tool_calls/tool_call_id). Does NOT convert other formats
into this shape -- get your data there yourself (see sft.md, "Out of scope").

Packs multiple examples into every training row instead of padding each example to
seq_len alone: real SFT examples are usually far shorter than seq_len, so a
pad-to-seq_len batch wastes most of its compute on padding -- exactly the throughput
stall packing exists to avoid. Packing is done correctly, not just concatenated:

  - block-diagonal causal attention mask: packed examples must not attend across each
    other's boundary (a naive concat would let example B see example A's tokens).
  - per-example position reset: every packed segment gets local position ids 0..len-1,
    matching what it sees unpacked / at inference time.
  - the next-token target at the LAST position of every packed segment is excluded from
    the loss: under block-diagonal attention that position structurally cannot see the
    next example, so "predicting" its first token is not a learnable, meaningful signal.

Both attn_mask and position_ids are threaded through loomformer.Model.forward (added
there for this purpose); passing neither reproduces the exact pretrain path unchanged.

CLI:
  loomsft.py --sft-dataset train.jsonl --config sft.yaml --init-checkpoint pretrain.pt \
             --checkpoint sft.pt [--val-dataset held_out.jsonl] [--steps N]
  loomsft.py --smoke-test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import loomformer as lf

IGNORE_INDEX = -100

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


# ============================================================================
# chat-template rendering: delegated to loomformer.ChatTemplate (single source
# of truth shared with loomchat.py -- see chat_template.jinja).
# ============================================================================


# ============================================================================
# packing
# ============================================================================

class PackedRow:
    __slots__ = ("ids", "loss_mask", "position_ids", "seg_id")

    def __init__(self, T: int):
        self.ids = np.zeros(T, dtype=np.int64)
        self.loss_mask = np.zeros(T, dtype=np.int64)
        self.position_ids = np.zeros(T, dtype=np.int64)
        self.seg_id = np.full(T, -1, dtype=np.int64)  # -1 = padding, never attended across


def _need_pad_id(tok) -> int:
    tid = tok.special_id("<pad>")
    if tid is None:
        raise ValueError(
            "tokenizer is missing <pad>. Retrain it with "
            "loomformer.train_tokenizer(..., special_tokens=loomformer.DEFAULT_SPECIAL_TOKENS)."
        )
    return tid


# ============================================================================
# <CARRY> placement: loomformer.Model._forward_chunked now cuts a training
# chunk wherever <CARRY> sits (spec: "if <CARRY> then refeed", no fixed grid --
# see loomformer.py's chunk-boundary comment). This is the SFT-side half of
# that contract: place <CARRY> by turn/content density, not by a token-count
# grid, while never firing it closer than CARRY_MIN_GAP tokens apart (a
# denser explicit refeed schedule buys nothing but chunk-count blowup).
# ============================================================================

CARRY_MIN_GAP = 48        # spec: never fire <CARRY> more often than every 48 tokens
# datasets/sft/train.jsonl: mean assistant turn ~930 tok, median ~980 (p90 1402) --
# each explicit <CARRY> forces its own chunk boundary in _forward_chunked, and
# those chunks are daisy-chained through accT_seed (not independent regions), so
# a too-dense internal stride multiplies backward-graph memory instead of the
# usual grad-checkpointing win. 256 gives ~3-4 internal refeeds across a
# median-length response instead of ~10.
CARRY_TARGET_STRIDE = 256  # density target inside long assistant spans, floor-clamped to CARRY_MIN_GAP


def _insert_carry_tokens(
    ids: List[int], mask: List[int], im_end_id: int, carry_id: int,
    min_gap: int = CARRY_MIN_GAP, target_stride: int = CARRY_TARGET_STRIDE,
) -> Tuple[List[int], List[int]]:
    """Places <CARRY> right after every AGENT (assistant) turn -- never after
    user/system/tool turns -- and, inside assistant spans longer than
    target_stride, at ~target_stride-token intervals within the response
    itself (dense token streams, not a fixed grid). Never places two <CARRY>
    closer than min_gap tokens. Loss mask of an inserted <CARRY> inherits from
    its neighbors: 1 only if it sits inside an assistant-authored span (both
    neighbors loss-carrying), 0 at turn boundaries -- structural, like every
    other template control token (<|im_start|>, role headers) that
    render_training_ids never loss-masks."""
    out_ids: List[int] = []
    out_mask: List[int] = []
    since_last = min_gap  # a <CARRY> is allowed right at the first eligible boundary
    n = len(ids)
    for k in range(n):
        out_ids.append(ids[k])
        out_mask.append(mask[k])
        since_last += 1
        if k == n - 1:
            continue
        # mask[k] == 1 on an <|im_end|> means render_training_ids counted THIS
        # turn as assistant-authored (it includes the closing <|im_end|> itself
        # in the loss span) -- i.e. this is an agent turn's end, not any turn's.
        turn_boundary = ids[k] == im_end_id and mask[k] == 1
        dense_point = (
            mask[k] == 1 and mask[k + 1] == 1 and since_last >= target_stride
        )
        if (turn_boundary or dense_point) and since_last >= min_gap:
            carry_loss = 1 if (mask[k] == 1 and mask[k + 1] == 1) else 0
            out_ids.append(carry_id)
            out_mask.append(carry_loss)
            since_last = 0
    return out_ids, out_mask


class SFTPackedStream:
    """Streams an sft.md-shaped jsonl -- never materializes the whole file (a
    training corpus is not guaranteed to be "eval/train-sft sized"; it needs to
    scale the same way loomformer.py's on-the-fly pretrain streams do). Render +
    tokenize happens lazily, one line at a time, wrapping back to the start of
    the file for additional epochs. Shuffling is a bounded reservoir buffer
    (tf.data/WebDataset ShuffleBuffer pattern): memory is O(shuffle_buffer)
    examples, not O(dataset) -- the only cost is buffer-local shuffling instead
    of a true full-dataset shuffle, which is the standard, correct trade-off at
    this scale."""

    def __init__(self, path: str, cfg: "lf.Config", tok, device: torch.device,
                 shuffle: bool = True, shuffle_buffer: int = 4096):
        self.cfg = cfg
        self.tok = tok
        self.device = device
        self.pad_id = _need_pad_id(tok)
        self.chat = lf.ChatTemplate(tok)
        self.carry_id = tok.special_id("<CARRY>")
        if self.carry_id is None:
            print("[loomsft] tokenizer has no <CARRY> token -- Tria temporal refeed stays on "
                  "the implicit dense grid only (see loomformer.DEFAULT_SPECIAL_TOKENS)")
        self.path = path
        self._shuffle = shuffle
        self._file = open(path, "r", encoding="utf-8")
        self._n_seen = 0
        self.n_dropped = 0

        self._buffer: List[Tuple[List[int], List[int]]] = []
        if shuffle:
            # Prime the reservoir once up front so even the very first draws are
            # already buffer-shuffled, not a suspiciously-sequential prefix.
            cap = max(1, int(shuffle_buffer))
            for _ in range(cap):
                ex = self._read_and_render_one()
                if ex is None:  # dataset smaller than the buffer: that's fine, just stop
                    break
                self._buffer.append(ex)
        if not self._buffer and not shuffle:
            # Sequential (val) mode never fills a buffer; just prove the file
            # has at least one usable example before training starts on it.
            probe = self._read_and_render_one()
            if probe is None:
                raise ValueError(f"{path}: no examples fit within seq_len={cfg.seq_len}")
            self._buffer.append(probe)
        elif not self._buffer:
            raise ValueError(f"{path}: no examples fit within seq_len={cfg.seq_len}")

    def _read_and_render_one(self) -> Optional[Tuple[List[int], List[int]]]:
        """Reads forward from the current file position, transparently wrapping
        to the start for additional epochs. Returns None only if an entire pass
        over the file produced not a single example that fits seq_len."""
        wrapped = False
        while True:
            raw = self._file.readline()
            if not raw:
                if wrapped:
                    return None
                self._file.seek(0)
                wrapped = True
                continue
            raw = raw.strip()
            if not raw:
                continue
            self._n_seen += 1
            ex = json.loads(raw)
            validate_example(ex, line_ctx=f"{self.path}:{self._n_seen}: ")
            # One rendered example == one packed segment. ChatTemplate prepends
            # <bos> (the same fixed position-0 attention-sink anchor pretraining
            # uses) inside render_text, so it lands at LOCAL position 0 of every
            # segment once _pack_one_row resets position_ids per segment below --
            # not just once at the front of the whole packed row.
            ids, mask = self.chat.render_training_ids(ex["messages"], tools=ex.get("tools"))
            if self.carry_id is not None:
                ids, mask = _insert_carry_tokens(ids, mask, self.chat.im_end_id, self.carry_id)
            if len(ids) > self.cfg.seq_len:
                self.n_dropped += 1  # longer than one packed row: dropped, not silently
                continue             # truncated mid tool-call / mid answer.
            return ids, mask

    def _next_example(self) -> Tuple[List[int], List[int]]:
        if not self._shuffle:
            ex = self._read_and_render_one()
            return ex if ex is not None else self._buffer[0]  # single-example fallback, see __init__
        nxt = self._read_and_render_one()
        if nxt is None:
            # Dataset exhausted and smaller than the buffer itself: sample straight
            # from what the reservoir already holds instead of stalling forever.
            return self._buffer[random.randrange(len(self._buffer))]
        i = random.randrange(len(self._buffer))
        out = self._buffer[i]
        self._buffer[i] = nxt
        return out

    def _pack_one_row(self) -> PackedRow:
        T = self.cfg.seq_len
        row = PackedRow(T)
        row.ids[:] = self.pad_id
        cursor = 0
        seg = 0
        guard = 0
        while cursor < T:
            guard += 1
            if guard > 4 * T:  # pathological: examples that never fit remaining space
                break
            ids, mask = self._next_example()
            L = len(ids)
            if cursor + L > T:
                if cursor == 0:
                    ids, mask = ids[:T], mask[:T]  # single example longer than a fresh
                    L = T                          # row: truncate rather than spin forever
                else:
                    break
            row.ids[cursor:cursor + L] = ids
            row.loss_mask[cursor:cursor + L] = mask
            row.position_ids[cursor:cursor + L] = np.arange(L)
            row.seg_id[cursor:cursor + L] = seg
            cursor += L
            seg += 1
        return row

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        B, T = self.cfg.batch_size, self.cfg.seq_len
        rows = [self._pack_one_row() for _ in range(B)]
        ids = np.stack([r.ids for r in rows])              # (B,T)
        loss_mask = np.stack([r.loss_mask for r in rows])   # (B,T)
        pos = np.stack([r.position_ids for r in rows])      # (B,T)
        seg = np.stack([r.seg_id for r in rows])            # (B,T)

        x = torch.from_numpy(ids[:, :-1]).to(self.device)
        y = torch.from_numpy(ids[:, 1:]).to(self.device)
        pos_ids = torch.from_numpy(pos[:, :-1]).to(self.device)
        # target loss at position i predicts token i+1: valid only if (a) the TARGET
        # token is an assistant-loss token, AND (b) i and i+1 are the same segment (else
        # the model is being asked to predict across a boundary it cannot attend across).
        seg_t = torch.from_numpy(seg).to(self.device)
        same_seg = (seg_t[:, :-1] == seg_t[:, 1:])
        loss_valid = torch.from_numpy(loss_mask[:, 1:]).to(self.device).bool() & same_seg
        y_masked = torch.where(loss_valid, y, torch.full_like(y, IGNORE_INDEX))

        seg_x = seg_t[:, :-1]                               # (B,T-1)
        allowed = (seg_x[:, None, :, None] == seg_x[:, None, None, :])         # (B,1,T-1,T-1)
        causal = torch.tril(torch.ones(seg_x.shape[1], seg_x.shape[1], dtype=torch.bool, device=self.device))
        attn_mask = allowed & causal[None, None]
        return {"x": x, "y": y_masked, "position_ids": pos_ids, "attn_mask": attn_mask, "seg_id": seg_x}

    async def _produce(self, queue: "asyncio.Queue", n: int):
        loop = asyncio.get_event_loop()
        for _ in range(n):
            batch = await loop.run_in_executor(None, self.sample_batch)
            await queue.put(batch)
        await queue.put(None)

    async def batches(self, n: int):
        # NOTE: a bare `await queue.get()` here would hang forever if _produce's
        # background executor call ever raises (sample_batch's exception is stored
        # on the producer Task and never surfaces until awaited) -- the consumer
        # would sit blocked on the queue with zero CPU/GPU activity, silently,
        # since the producer never reaches its final `queue.put(None)` sentinel
        # either. Race queue.get() against the producer task itself so a producer
        # exception is re-raised here immediately instead of deadlocking.
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)
        producer = asyncio.create_task(self._produce(queue, n))
        try:
            while True:
                get_task = asyncio.ensure_future(queue.get())
                done, _pending = await asyncio.wait(
                    {get_task, producer}, return_when=asyncio.FIRST_COMPLETED)
                if get_task in done:
                    batch = get_task.result()
                    if batch is None:
                        break
                    yield batch
                else:
                    get_task.cancel()
                    producer.result()  # re-raises the producer's exception, if any
                    break
        finally:
            if not producer.done():
                producer.cancel()


# ============================================================================
# train / eval
# ============================================================================

@torch.no_grad()
def eval_sft(model: torch.nn.Module, stream: SFTPackedStream, device: torch.device, n_batches: int = 8) -> float:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for _ in range(n_batches):
        b = stream.sample_batch()
        with lf.amp_autocast(device):
            logits = model(b["x"], attn_mask=b["attn_mask"], position_ids=b["position_ids"])
        model.last_tria_depth_carry = None
        model.last_tria_fire_mask = None
        model.last_tria_document_carry_stats = None
        ntok = int((b["y"] != IGNORE_INDEX).sum().item())
        if ntok == 0:
            continue
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), b["y"].reshape(-1),
                                ignore_index=IGNORE_INDEX, reduction="sum")
        tot_loss += float(loss.item())
        tot_tok += ntok
    model.train()
    return tot_loss / max(1, tot_tok)


async def train_sft_async(
    cfg: "lf.Config",
    train_path: str,
    val_path: Optional[str],
    init_checkpoint: str,
    device: torch.device,
    ckpt_out: str,
) -> None:
    lf.set_seed(cfg.seed)
    tok = lf.build_tokenizer(cfg)
    # Must run before apply_config: otherwise tria_temporal_auto recalibrates a
    # fresh W/alpha instead of the geometry init_checkpoint's Tria carry was
    # actually trained under (same ordering as loomformer.train_async).
    lf.restore_temporal_tria_from_checkpoint(cfg, init_checkpoint)
    lf.apply_config(cfg)

    model = lf.Model().to(device)
    lf.load_model_checkpoint(model, init_checkpoint, ablation=False, device=device)
    # SFT: refeed fires ONLY on explicit <CARRY>, never on the pretrain-only dense
    # W-token deadline -- the model has no fixed-grid dependency to begin with.
    model.tria_hard_fire_enabled = False

    train_stream = SFTPackedStream(train_path, cfg, tok, device, shuffle=True)
    val_stream = SFTPackedStream(val_path, cfg, tok, device, shuffle=False) if val_path else None

    params = [p for p in model.parameters() if p.requires_grad]
    opt_cls, opt_name = lf.optimizer_class_from_name(getattr(cfg, "optimizer", "adamw"))
    opt = opt_cls(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    n_params = lf.count_params(model)
    print(f"--- loomsft ({n_params:,} params | init={init_checkpoint} | train={train_path} | "
          f"val={val_path or 'none'}) ---")

    accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    step = 0
    trias_since_log = torch.zeros((), dtype=torch.long, device=device)
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    micro = 0
    loss_sum = 0.0
    batch_iter = train_stream.batches(int(cfg.steps) * accum_steps).__aiter__()
    async for batch in batch_iter:
        with lf.amp_autocast(device):
            logits = model(batch["x"], attn_mask=batch["attn_mask"], position_ids=batch["position_ids"])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), batch["y"].reshape(-1),
                                ignore_index=IGNORE_INDEX)
        if model.last_tria_fire_mask is not None:
            with torch.no_grad():
                trias_since_log.add_(model.last_tria_fire_mask.detach().sum())
        (loss / accum_steps).backward()
        model.last_tria_depth_carry = None
        model.last_tria_fire_mask = None
        model.last_tria_document_carry_stats = None
        loss_sum += float(loss.item())
        micro += 1
        if micro < accum_steps:
            continue
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        lr = lf.lr_at(cfg, step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        train_loss = loss_sum / micro
        micro = 0
        loss_sum = 0.0
        if step % cfg.log_every == 0:
            trias_log = int(trias_since_log.item())
            trias_since_log.zero_()
            msg = f"[loomsft] step {step:6d}  train_loss {train_loss:.4f}  trias: {trias_log:d}  lr {lr:.2e}  ({time.time()-t0:.0f}s)"
            if val_stream is not None and step % (cfg.log_every * 5) == 0:
                msg += f"  eval_loss {eval_sft(model, val_stream, device):.4f}"
            print(msg)
        step += 1

    torch.save({"cfg": asdict(cfg), "model_kind": "loomformer", "ffn_type": "paraplex",
                "ablation": False, "model": model.state_dict()}, ckpt_out)
    print(f"[loomsft] saved -> {ckpt_out}")


# ============================================================================
# smoke test
# ============================================================================

def smoke_test() -> None:
    import tempfile

    cfg = lf.Config(vocab=256, seq_len=64, batch_size=2, model_dim=16, n_q_heads=2,
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
    stream = SFTPackedStream(sft_path, cfg, tok, dev, shuffle=True)
    b = stream.sample_batch()
    assert b["x"].shape == (2, 63) and b["y"].shape == (2, 63)
    assert b["attn_mask"].shape == (2, 1, 63, 63)
    print(f"[smoke] packing OK: x={tuple(b['x'].shape)} attn_mask={tuple(b['attn_mask'].shape)} "
          f"loss_tokens={(b['y'] != IGNORE_INDEX).sum().item()}")

    model = lf.Model().to(dev)
    with torch.no_grad():
        logits = model(b["x"], attn_mask=b["attn_mask"], position_ids=b["position_ids"])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b["y"].reshape(-1), ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss)
    print(f"[smoke] forward with packed attn_mask/position_ids OK, loss={loss.item():.4f}")

    # cross-segment isolation, empirically: perturbing segment 0's tokens must not change
    # segment 1's logits AT ALL under the block-diagonal mask.
    seg = stream._pack_one_row()
    ids_t = torch.from_numpy(seg.ids[None, :-1]).to(dev)
    pos_t = torch.from_numpy(seg.position_ids[None, :-1]).to(dev)
    seg_t = torch.from_numpy(seg.seg_id[None, :-1])
    same = seg_t[:, None, :, None] == seg_t[:, None, None, :]
    causal = torch.tril(torch.ones(ids_t.shape[1], ids_t.shape[1], dtype=torch.bool))
    amask = (same & causal[None, None]).to(dev)
    with torch.no_grad():
        base = model(ids_t, attn_mask=amask, position_ids=pos_t)
        ids2 = ids_t.clone()
        first_seg_len = int((seg.seg_id == 0).sum())
        if first_seg_len > 0:
            ids2[0, :first_seg_len] = (ids2[0, :first_seg_len] + 1) % lf.VOCAB
            other = model(ids2, attn_mask=amask, position_ids=pos_t)
            later = seg.seg_id[:-1] != 0
            if later.any():
                delta = (base[0, later] - other[0, later]).abs().max().item()
                assert delta < 1e-4, f"packed segments are NOT isolated: max delta {delta}"
                print(f"[smoke] cross-segment isolation OK (max delta {delta:.2e})")
    print("[smoke] ALL OK")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="loomsft: SFT training for LoomFormer")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--sft-dataset", type=str, default=None)
    ap.add_argument("--val-dataset", type=str, default=None)
    ap.add_argument("--init-checkpoint", type=str, default=None, help="pretrained LoomFormer checkpoint to start SFT from")
    ap.add_argument("--checkpoint", type=str, default="loomsft.pt")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    cfg = lf.Config.from_yaml(args.config) if args.config else lf.Config()
    if args.steps is not None:
        cfg.steps = args.steps
    device_pref = args.device if args.device is not None else cfg.device
    dev = lf.device_auto(device_pref)

    assert args.sft_dataset, "--sft-dataset is required"
    assert args.init_checkpoint, "--init-checkpoint is required (SFT starts from a pretrained LoomFormer checkpoint)"
    asyncio.run(train_sft_async(cfg, args.sft_dataset, args.val_dataset, args.init_checkpoint, dev, args.checkpoint))


if __name__ == "__main__":
    main()
