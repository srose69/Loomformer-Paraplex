#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loomsft.py -- SFT on top of LoomFormer.

Consumes the jsonl schema documented in sft_format.json (OpenAI-style messages,
with first-class tool_calls/tool_call_id). Does NOT convert other formats into
this shape -- get your data there yourself.

The previous version re-rendered the Jinja template and re-tokenized from raw
JSON on every single batch draw, for the lifetime of training -- across many
epochs over a fixed, static file, that is the same handful of megabytes of
text being re-parsed and re-templated thousands of times over. Jinja rendering
is an inherently per-example Python-level operation (it doesn't batch), so the
only way to stop paying for it repeatedly is to pay for it once and cache the
result. A 150k-line dataset renders+tokenizes in a few minutes on CPU, held
entirely in RAM (a few hundred MB of int32 token ids) -- after that, every
training step is pure array slicing and concatenation, no JSON, no Jinja, no
tokenizer calls, ever again.

No <CARRY> handling. Tria's temporal refeed runs on its normal dense grid,
exactly like pretrain -- SFT does not place, mask, or otherwise think about
<CARRY> at all. (If that ever changes, it belongs back in here as a targeted
addition, not as the default.)

Packing: multiple pre-tokenized examples per row instead of padding each to
seq_len alone (real SFT examples are usually much shorter than seq_len; a
pad-to-seq_len batch wastes most of its compute on padding). Packing is
block-diagonal, not a naive concat:
  - packed examples cannot attend across each other's boundary
  - every packed segment gets its own local position ids (0..len-1)
  - the last position of every segment is excluded from the loss: under
    block-diagonal attention it structurally cannot see the next example, so
    "predicting" that example's first token is not a learnable signal.

CLI:
  loomsft.py --sft-dataset train.jsonl --config sft.yaml --init-checkpoint pretrain.pt \
             --checkpoint sft.pt [--val-dataset held_out.jsonl] [--steps N]
  loomsft.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import random
import threading
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


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            yield lineno, raw


def _iter_examples(path: str):
    """Reads jsonl OR a HF-style .arrow file with a 'messages' column (same
    dual-format convenience as loomformer's own dataset loading) -- but always
    the sft_format.json schema either way."""
    if path.endswith(".arrow") or path.endswith(".feather"):
        import pyarrow as pa
        import pyarrow.ipc as ipc
        with pa.memory_map(path, "r") as src:
            try:
                reader = ipc.open_file(src)
            except pa.lib.ArrowInvalid:
                src.seek(0)
                reader = ipc.open_stream(src)
            table = reader.read_all()
        if "messages" not in table.column_names:
            raise ValueError(f"{path}: expected a 'messages' column, got {table.column_names}")
        for i, row in enumerate(table.column("messages").to_pylist(), 1):
            yield i, {"messages": row}
    else:
        for lineno, raw in _iter_jsonl(path):
            yield lineno, json.loads(raw)


# ============================================================================
# preprocessing cache: render + tokenize once, hold pre-tokenized examples in
# memory. This is the whole fix -- everything downstream of this function is
# plain array bookkeeping, no per-example Python work left in the hot path.
# ============================================================================

class TokenizedExample:
    __slots__ = ("ids", "mask")

    def __init__(self, ids: np.ndarray, mask: np.ndarray):
        self.ids = ids
        self.mask = mask


def preprocess_dataset(
    path: str, chat: "lf.ChatTemplate", seq_len: int, verbose: bool = True,
) -> List[TokenizedExample]:
    t0 = time.time()
    out: List[TokenizedExample] = []
    n_seen = 0
    n_dropped = 0
    n_no_loss = 0
    for line_ctx, ex in _iter_examples(path):
        n_seen += 1
        validate_example(ex, line_ctx=f"{path}:{line_ctx}: ")
        ids, mask = chat.render_training_ids(ex["messages"], tools=ex.get("tools"))
        # A packed row needs room for at least one more token after this example
        # (the next segment, or the final target shift) -- same "> seq_len"
        # boundary the old code used, just checked once here instead of per draw.
        if len(ids) > seq_len - 1:
            n_dropped += 1
            continue
        if not any(mask):
            n_no_loss += 1
            continue  # an example with zero loss-carrying tokens teaches nothing
        out.append(TokenizedExample(np.asarray(ids, dtype=np.int32), np.asarray(mask, dtype=np.int8)))
        if verbose and n_seen % 20000 == 0:
            print(f"[loomsft] preprocessing {path}: {n_seen} read, {len(out)} kept "
                  f"({time.time() - t0:.0f}s)", flush=True)
    if not out:
        raise ValueError(f"{path}: no examples fit within seq_len={seq_len} after validation/rendering")
    if verbose:
        print(f"[loomsft] preprocessed {path}: {len(out)} kept, {n_dropped} dropped "
              f"(> seq_len-1={seq_len - 1} tokens), {n_no_loss} dropped (no loss-carrying "
              f"tokens) -- {time.time() - t0:.0f}s total", flush=True)
    return out


# ============================================================================
# packing
# ============================================================================

class PackedRow:
    __slots__ = ("ids", "loss_mask", "position_ids", "seg_id")

    def __init__(self, T: int, pad_id: int):
        self.ids = np.full(T, pad_id, dtype=np.int64)
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


class Packer:
    """Greedy first-fit packing over an in-memory pool of already-tokenized
    examples. Holds a shuffled permutation of the pool and walks it; reshuffles
    and starts over once exhausted (a fresh "epoch" of example ORDER -- the
    pool itself, and every example's tokenization, is fixed and was computed
    exactly once in preprocess_dataset)."""

    def __init__(self, pool: List[TokenizedExample], seq_len: int, pad_id: int,
                 shuffle: bool, seed: int = 0):
        if not pool:
            raise ValueError("empty example pool")
        self.pool = pool
        self.T = seq_len
        self.pad_id = pad_id
        self.shuffle = shuffle
        self._rng = random.Random(seed)
        self._order = list(range(len(pool)))
        if shuffle:
            self._rng.shuffle(self._order)
        self._cursor = 0

    def _next_example(self) -> TokenizedExample:
        if self._cursor >= len(self._order):
            self._cursor = 0
            if self.shuffle:
                self._rng.shuffle(self._order)
        ex = self.pool[self._order[self._cursor]]
        self._cursor += 1
        return ex

    def pack_one_row(self) -> PackedRow:
        T = self.T
        row = PackedRow(T, self.pad_id)
        cursor = 0
        seg = 0
        # A single dataset pass already guarantees every example is <= T-1 tokens
        # (preprocess_dataset dropped anything longer), so this can never spin:
        # each iteration either places >=1 example or the row is full.
        while cursor < T:
            ex = self._next_example()
            L = len(ex.ids)
            if cursor + L > T:
                if cursor == 0:
                    break  # unreachable given the preprocessing guarantee; defensive only
                break
            row.ids[cursor:cursor + L] = ex.ids
            row.loss_mask[cursor:cursor + L] = ex.mask
            row.position_ids[cursor:cursor + L] = np.arange(L)
            row.seg_id[cursor:cursor + L] = seg
            cursor += L
            seg += 1
        return row

    def sample_batch(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        rows = [self.pack_one_row() for _ in range(batch_size)]
        ids = np.stack([r.ids for r in rows])
        loss_mask = np.stack([r.loss_mask for r in rows])
        pos = np.stack([r.position_ids for r in rows])
        seg = np.stack([r.seg_id for r in rows])

        x = torch.from_numpy(ids[:, :-1]).to(device)
        y = torch.from_numpy(ids[:, 1:]).to(device)
        pos_ids = torch.from_numpy(pos[:, :-1]).to(device)
        seg_t = torch.from_numpy(seg).to(device)
        same_seg = (seg_t[:, :-1] == seg_t[:, 1:])
        loss_valid = torch.from_numpy(loss_mask[:, 1:]).to(device).bool() & same_seg
        y_masked = torch.where(loss_valid, y, torch.full_like(y, IGNORE_INDEX))

        seg_x = seg_t[:, :-1]
        allowed = (seg_x[:, None, :, None] == seg_x[:, None, None, :])
        causal = torch.tril(torch.ones(seg_x.shape[1], seg_x.shape[1], dtype=torch.bool, device=device))
        attn_mask = allowed & causal[None, None]
        return {"x": x, "y": y_masked, "position_ids": pos_ids, "attn_mask": attn_mask, "seg_id": seg_x}


def print_sft_header(cfg: "lf.Config", device: torch.device, init_checkpoint: str,
                     train_path: str, val_path: Optional[str]) -> None:
    """1:1 with loomformer.print_architecture_report, plus SFT-specific fields
    (init checkpoint, train/val dataset paths)."""
    width = 64
    rule = "=" * width
    print(rule)
    print(f" LoomSFT  ·  {device}  ·  amp={lf.AMP_DTYPE}  ·  init={init_checkpoint}")
    print(rule)
    grp = f"x{lf.GQA_GROUP_SIZE}" if lf.GQA_GROUP_SIZE else "x1"
    print(f"  shape    d_model={lf.N}  heads={lf.N_Q_HEADS}q/{lf.N_KV_HEADS}kv({grp})  "
          f"head_dim={lf.HEAD_DIM}  layers={lf.LAYERS}")
    print(f"  ffn      hidden={lf.HIDDEN}  phase={lf.PHASE_SECTORS}  attn={lf.ATTN_IMPL}")
    print(f"  rope     yarn  theta={lf.ROPE_THETA:g}  factor={lf.ROPE_FACTOR:g}x  "
          f"orig_len={lf.ROPE_ORIGINAL_SEQ_LEN}")
    if lf.HEAD_DIM < 8:
        print(f"  WARNING: head_dim={lf.HEAD_DIM} is extremely small for LM attention.")
    print(f"  train    {train_path}")
    print(f"  val      {val_path}" if val_path else "  val      (none -- training loss only)")


def print_sft_training_budget(cfg: "lf.Config", model, train_pool: List[TokenizedExample]) -> None:
    """Same shape as loomformer.print_training_budget, but every number here
    comes from the ACTUAL preprocessed pool (real render+tokenize output),
    not an estimate -- this only makes sense printed after preprocess_dataset,
    never before it."""
    pool_tokens = int(sum(len(ex.ids) for ex in train_pool))
    pool_loss_tokens = int(sum(int(ex.mask.sum()) for ex in train_pool))
    accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    tokens_per_step = int(cfg.batch_size) * int(cfg.seq_len) * accum_steps
    run_tokens = int(cfg.steps) * tokens_per_step
    run_epochs = run_tokens / max(1, pool_tokens)
    print(f"  budget   {run_tokens:,} tokens over {cfg.steps:,} steps "
          f"({run_epochs:.3f} epochs of {pool_tokens:,} pool tokens, "
          f"{len(train_pool):,} examples)")
    params = lf.count_params(model)
    tpp = run_tokens / max(1, params)
    epoch_tokens_per_param = pool_tokens / max(1, params)
    print(f"           loomformer: {params:,} params  ·  {tpp:.1f} tok/param  ·  "
          f"{epoch_tokens_per_param:.1f} data-tok/param")
    loss_frac = pool_loss_tokens / max(1, pool_tokens)
    avg_ex_len = pool_tokens / max(1, len(train_pool))
    packing_eff = avg_ex_len / max(1, int(cfg.seq_len))
    print(f"           pool: {pool_loss_tokens:,} loss-carrying tokens ({loss_frac:.1%} of pool)  ·  "
          f"avg example {avg_ex_len:.0f} tok  ·  ~{packing_eff:.1%} of a bare row before packing gains")


class BatchPrefetcher:
    """Packing (Python loop over segments + broadcasting the block-diagonal
    mask) is real CPU work, and without overlap the GPU sits idle for exactly
    that long on EVERY step -- a wave in GPU utilization at the per-step
    period, not something that shows up in a log averaged every log_every
    steps. One background thread stays one batch ahead in a bounded queue;
    the training loop only ever waits on a queue.get(), which returns
    immediately once the GPU is done with the previous step's forward/backward
    (i.e. the two overlap instead of serializing)."""

    def __init__(self, packer: Packer, batch_size: int, device: torch.device, depth: int = 2):
        self.packer = packer
        self.batch_size = batch_size
        self.device = device
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._err: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                batch = self.packer.sample_batch(self.batch_size, self.device)
                self._q.put(batch)
        except BaseException as e:  # noqa: BLE001 -- must reach the consumer, not vanish in the thread
            self._err = e
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass

    def next(self) -> Dict[str, torch.Tensor]:
        batch = self._q.get()
        if batch is None and self._err is not None:
            raise RuntimeError("BatchPrefetcher background thread failed") from self._err
        return batch

    def stop(self) -> None:
        self._stop.set()


# ============================================================================
# train / eval
# ============================================================================

@torch.no_grad()
def eval_sft(model: torch.nn.Module, packer: Packer, batch_size: int,
             device: torch.device, n_batches: int = 8) -> float:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for _ in range(n_batches):
        b = packer.sample_batch(batch_size, device)
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


def train_sft(
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
    # actually trained under (same ordering loomformer.train_async uses).
    lf.restore_temporal_tria_from_checkpoint(cfg, init_checkpoint)
    lf.apply_config(cfg)
    chat = lf.ChatTemplate(tok)
    pad_id = _need_pad_id(tok)

    model = lf.Model().to(device)
    lf.load_model_checkpoint(model, init_checkpoint, ablation=False, device=device)

    train_pool = preprocess_dataset(train_path, chat, cfg.seq_len)
    train_packer = Packer(train_pool, cfg.seq_len, pad_id, shuffle=True, seed=cfg.seed)
    val_packer = None
    if val_path:
        val_pool = preprocess_dataset(val_path, chat, cfg.seq_len)
        val_packer = Packer(val_pool, cfg.seq_len, pad_id, shuffle=False, seed=cfg.seed)

    print_sft_header(cfg, device, init_checkpoint, train_path, val_path)
    print_sft_training_budget(cfg, model, train_pool)
    print("=" * 64)

    params = [p for p in model.parameters() if p.requires_grad]
    opt_cls, opt_name = lf.optimizer_class_from_name(getattr(cfg, "optimizer", "adamw"))
    opt = opt_cls(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    prefetcher = BatchPrefetcher(train_packer, cfg.batch_size, device, depth=max(2, accum_steps // 4 + 1))
    step = 0
    trias_since_log = torch.zeros((), dtype=torch.long, device=device)
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    try:
        while step < int(cfg.steps):
            loss_sum = 0.0
            for micro in range(accum_steps):
                batch = prefetcher.next()
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
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            lr = lf.lr_at(cfg, step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            opt.zero_grad(set_to_none=True)
            train_loss = loss_sum / accum_steps
            if step % cfg.log_every == 0:
                trias_log = int(trias_since_log.item())
                trias_since_log.zero_()
                msg = f"[loomsft] step {step:6d}  train_loss {train_loss:.4f}  trias: {trias_log:d}  lr {lr:.2e}  ({time.time()-t0:.0f}s)"
                if val_packer is not None and step % (cfg.log_every * 5) == 0:
                    msg += f"  eval_loss {eval_sft(model, val_packer, cfg.batch_size, device):.4f}"
                print(msg, flush=True)
            step += 1
    finally:
        prefetcher.stop()

    torch.save({"cfg": asdict(cfg), "model_kind": "loomformer", "ffn_type": "paraplex",
                "ablation": False, "model": model.state_dict()}, ckpt_out)
    print(f"[loomsft] saved -> {ckpt_out}")


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

    pool = preprocess_dataset(sft_path, chat, cfg.seq_len, verbose=False)
    assert len(pool) == 2
    print(f"[smoke] preprocess_dataset OK: {len(pool)} examples cached")

    dev = lf.device_auto("cpu")
    pad_id = _need_pad_id(tok)
    packer = Packer(pool, cfg.seq_len, pad_id, shuffle=True, seed=0)
    b = packer.sample_batch(2, dev)
    assert b["x"].shape == (2, cfg.seq_len - 1) and b["y"].shape == (2, cfg.seq_len - 1)
    assert b["attn_mask"].shape == (2, 1, cfg.seq_len - 1, cfg.seq_len - 1)
    print(f"[smoke] packing OK: x={tuple(b['x'].shape)} attn_mask={tuple(b['attn_mask'].shape)} "
          f"loss_tokens={(b['y'] != IGNORE_INDEX).sum().item()}")

    model = lf.Model().to(dev)
    with torch.no_grad():
        logits = model(b["x"], attn_mask=b["attn_mask"], position_ids=b["position_ids"])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b["y"].reshape(-1), ignore_index=IGNORE_INDEX)
    assert torch.isfinite(loss)
    print(f"[smoke] forward with packed attn_mask/position_ids OK, loss={loss.item():.4f}")

    # cross-segment isolation, empirically: perturbing segment 0's tokens must not
    # change segment 1's logits AT ALL under the block-diagonal mask.
    seg_row = packer.pack_one_row()
    ids_t = torch.from_numpy(seg_row.ids[None, :-1]).to(dev)
    pos_t = torch.from_numpy(seg_row.position_ids[None, :-1]).to(dev)
    seg_t = torch.from_numpy(seg_row.seg_id[None, :-1])
    same = seg_t[:, None, :, None] == seg_t[:, None, None, :]
    causal = torch.tril(torch.ones(ids_t.shape[1], ids_t.shape[1], dtype=torch.bool))
    amask = (same & causal[None, None]).to(dev)
    with torch.no_grad():
        base = model(ids_t, attn_mask=amask, position_ids=pos_t)
        ids2 = ids_t.clone()
        first_seg_len = int((seg_row.seg_id == 0).sum())
        if first_seg_len > 0:
            ids2[0, :first_seg_len] = (ids2[0, :first_seg_len] + 1) % lf.VOCAB
            other = model(ids2, attn_mask=amask, position_ids=pos_t)
            later = seg_row.seg_id[:-1] != 0
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
    train_sft(cfg, args.sft_dataset, args.val_dataset, args.init_checkpoint, dev, args.checkpoint)


if __name__ == "__main__":
    main()
