#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profile_loom_train.py -- короткий torch.profiler для LoomFormer train-step.

Зачем: поймать жирные ATen/CUDA ops, лишние материализации и VRAM пики на 1-2
реальных train шагах, без полного train_async/eval/checkpoint шума.

Типичный запуск рядом с loomformer.py:
  python profile_loom_train.py --config fss1str_tiny_chk.yaml --dataset ./datasets/fss1str --warmup 2 --steps 2 --memory --shapes --top 60

Trace для Perfetto/chrome://tracing:
  python profile_loom_train.py --config fss1str_tiny_chk.yaml --dataset ./datasets/fss1str --warmup 2 --steps 1 --memory --shapes --trace loom_step.json
"""
from __future__ import annotations

import argparse
import gc
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F
import torch.profiler as tprof


ap = argparse.ArgumentParser(description="Profile 1-2 LoomFormer train steps with torch.profiler")
ap.add_argument("--config", required=True, help="YAML config for loomformer.Config.from_yaml")
ap.add_argument("--dataset", default=None, help="train dataset path; defaults to cfg.train_dataset")
ap.add_argument("--sft-dataset", default=None, help="profile loomsft.py path using this SFT jsonl dataset")
ap.add_argument("--init-checkpoint", default=None, help="pretrained checkpoint to load for --sft-dataset")
ap.add_argument("--sft-shuffle-buffer", type=int, default=256, help="SFT shuffle buffer for profiling")
ap.add_argument("--sft-workers", type=int, default=2, help="SFT tokenizer worker threads for profiling")
ap.add_argument("--device", default=None, help="cpu | cuda | cuda:0 | cuda:1; default: cfg.device/device_auto")
ap.add_argument("--steps", type=int, default=2, help="profiled optimizer steps")
ap.add_argument("--warmup", type=int, default=2, help="unprofiled warmup optimizer steps before profiling")
ap.add_argument("--top", type=int, default=60, help="rows in profiler tables")
ap.add_argument("--shapes", action="store_true", help="group ops by input shapes")
ap.add_argument("--memory", action="store_true", help="enable torch.profiler memory accounting")
ap.add_argument("--stack", action="store_true", help="record Python stack frames; expensive, use only for 1 step")
ap.add_argument("--trace", default=None, help="export Chrome/Perfetto trace JSON")
ap.add_argument("--out", default=None, help="also write text summary to this file")
ap.add_argument("--gelu", action="store_true", help="profile GELU baseline instead of Paraplex")
ap.add_argument("--ablation", action="store_true", help="construct Paraplex ablation model")
ap.add_argument("--compile", action="store_true", help="torch.compile(model) before profiling")
ap.add_argument("--graph", action="store_true", help="force cfg.graph=True: graph_helper custom_ops + torch.compile, same path as training")
ap.add_argument("--no-graph", action="store_true", help="force cfg.graph=False")
ap.add_argument("--grad-accum", type=int, default=None, help="override cfg.grad_accum_steps for profiling")
ap.add_argument("--no-tria", action="store_true", help="force cfg.tria_carry_enabled=False")
ap.add_argument("--tria", action="store_true", help="force cfg.tria_carry_enabled=True")
ap.add_argument("--no-cuda-tria", action="store_true", help="force cfg.use_cuda_tria=False")
ap.add_argument("--cuda-tria", action="store_true", help="force cfg.use_cuda_tria=True")
ap.add_argument("--temporal", action="store_true", help="force cfg.tria_temporal_enabled=True")
ap.add_argument("--no-temporal", action="store_true", help="force cfg.tria_temporal_enabled=False")
ap.add_argument("--tria-window", type=int, default=None, help="set fixed tria_temporal_window and disable auto calibration")
ap.add_argument("--no-cuda-beta", action="store_true", help="force cfg.use_cuda_beta_space=False")
ap.add_argument("--no-cuda-phase", action="store_true", help="force cfg.use_cuda_phase_sin=False")
ap.add_argument("--amp", default=None, choices=["fp32", "off", "bf16", "fp16"], help="override cfg.amp_dtype")
ap.add_argument("--optimizer", default=None, choices=["adamw", "atom"], help="override cfg.optimizer")
ap.add_argument("--seed", type=int, default=None, help="override cfg.seed")
ap.add_argument("--empty-cache", action="store_true", help="torch.cuda.empty_cache() before profiled region")
ap.add_argument("--no-trias-count", action="store_true", help="skip would-carry counter to keep profiler cleaner")
args = ap.parse_args()

import loomformer as lf  # must be next to this script or on PYTHONPATH

sft_mode = args.sft_dataset is not None
if sft_mode:
    import loomsft


def resolve_device(pref: Optional[str]) -> torch.device:
    dev = lf.device_auto(pref)
    if dev.type == "cuda":
        idx = 0 if dev.index is None else int(dev.index)
        n = torch.cuda.device_count()
        if idx < 0 or idx >= n:
            raise RuntimeError(
                f"Requested {dev}, but only {n} CUDA device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
                "If CUDA_VISIBLE_DEVICES=2, use --device cuda:0 inside this process."
            )
        torch.cuda.set_device(idx)
        dev = torch.device(f"cuda:{idx}")
    return dev


def fmt_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if abs(x) < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TiB"


cfg = lf.Config.from_yaml(args.config)
if args.dataset is not None:
    cfg.train_dataset = args.dataset
if args.device is not None:
    cfg.device = args.device
if args.seed is not None:
    cfg.seed = int(args.seed)
if args.tria:
    cfg.tria_carry_enabled = True
if args.no_tria:
    cfg.tria_carry_enabled = False
if args.cuda_tria:
    cfg.use_cuda_tria = True
if args.no_cuda_tria:
    cfg.use_cuda_tria = False
if args.temporal:
    cfg.tria_temporal_enabled = True
if args.no_temporal:
    cfg.tria_temporal_enabled = False
if args.tria_window is not None:
    if args.tria_window <= 0:
        raise SystemExit("--tria-window must be positive")
    cfg.tria_temporal_window = int(args.tria_window)
    cfg.tria_temporal_auto = False
if args.no_cuda_beta:
    cfg.use_cuda_beta_space = False
if args.no_cuda_phase:
    cfg.use_cuda_phase_sin = False
if args.amp is not None:
    cfg.amp_dtype = "fp32" if args.amp == "off" else args.amp
if args.optimizer is not None:
    cfg.optimizer = args.optimizer
if args.graph:
    cfg.graph = True
if args.no_graph:
    cfg.graph = False

if sft_mode:
    cfg.train_dataset = args.sft_dataset
elif not cfg.train_dataset:
    raise SystemExit("dataset is required: pass --dataset or set train_dataset/dataset in YAML")
if sft_mode and not args.init_checkpoint:
    raise SystemExit("--sft-dataset requires --init-checkpoint")

lf.set_seed(int(cfg.seed))
device = resolve_device(cfg.device)
tok = lf.build_tokenizer(cfg)
if sft_mode:
    # Match loomsft.py startup: checkpoint owns the temporal Tria geometry.
    lf.restore_temporal_tria_from_checkpoint(cfg, args.init_checkpoint)
lf.apply_config(cfg)

model = lf.Model(cfg, ablation=bool(args.ablation)).to(device)
if sft_mode:
    lf.load_model_checkpoint(model, args.init_checkpoint, ablation=False, device=device)


def prepare_graph_custom_ops(model_base: torch.nn.Module) -> None:
    """Register graph_helper custom_ops exactly like loomformer.py train does.

    apply_config(cfg) installs capture hooks when cfg.graph is true; this warmup
    makes the hooks see real config-shaped tensors, finalizes registration, then
    drops the throwaway grads before profiling starts.
    """
    if not bool(getattr(cfg, "graph", False)):
        return
    import graph_helper

    was_training = model_base.training
    model_base.train()
    max_attempts = 5
    last_attempt = 0
    for attempt in range(1, max_attempts + 1):
        last_attempt = attempt
        warm_batch = torch.randint(
            0, lf.VOCAB, (int(cfg.batch_size), lf.SEQ_LEN + 1),
            device=device, dtype=torch.long,
        )
        wx, wy = warm_batch[:, :-1], warm_batch[:, 1:]
        with lf.amp_autocast(device):
            logits = model_base(wx)
        loss = F.cross_entropy(logits.float().reshape(-1, lf.VOCAB), wy.reshape(-1))
        loss.backward()
        model_base.zero_grad(set_to_none=True)
        graph_helper.finalize_registration(lf, lf.tria)
        del warm_batch, wx, wy, logits, loss
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if graph_helper.is_finalized():
            break
    if not was_training:
        model_base.eval()

    registered, missing, fallback_only = graph_helper.registration_summary()
    if getattr(lf, "PHASE_GRAD_MODE", None) == "secant" and "phase_sin" in missing:
        missing = [n for n in missing if n != "phase_sin"]
        fallback_only = list(fallback_only) + ["phase_sin(secant mode)"]
    print(f"[graph] registered after {last_attempt} warmup attempt(s): {', '.join(registered) or '(none)'}")
    if fallback_only:
        print(f"[graph] fallback-only, not registered (expected, not a problem): {', '.join(fallback_only)}")
    if missing:
        print(f"[graph] NOT registered after {max_attempts} attempts (worth investigating): {', '.join(missing)}")


if bool(getattr(cfg, "graph", False)) and not sft_mode:
    prepare_graph_custom_ops(model)
    model = lf.maybe_compile(model, device, use_graph=True)
elif args.compile:
    model = torch.compile(model)
model.train()
raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

params = [p for p in model.parameters() if p.requires_grad]
try:
    OptimizerClass, optimizer_name = lf.optimizer_class_from_name(cfg.optimizer)
except AttributeError:
    OptimizerClass, optimizer_name = torch.optim.AdamW, "adamw"
if sft_mode:
    stream = loomsft.SFTPackedStream(
        args.sft_dataset, cfg, tok, device, shuffle=True,
        shuffle_buffer=int(args.sft_shuffle_buffer),
    )
else:
    stream = lf.make_stream(cfg.train_dataset, cfg, device)
accum_steps = int(args.grad_accum if args.grad_accum is not None else getattr(cfg, "grad_accum_steps", 1) or 1)
eos_id = None if sft_mode else (getattr(stream, "_eos_id", None) if bool(getattr(cfg, "doc_reset_attn", True)) else None)

tria_agg = getattr(raw_model, "tria_agg", None)
no_decay_param_ids = set()
if tria_agg is not None:
    no_decay_param_ids.add(id(tria_agg.pool.logit_scale_raw))
if no_decay_param_ids:
    decay_params = [p for p in params if id(p) not in no_decay_param_ids]
    no_decay_params = [p for p in params if id(p) in no_decay_param_ids]
    opt = OptimizerClass(
        [
            {"params": decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
else:
    opt = OptimizerClass(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

tag = "gelu" if args.gelu else ("paraplex-ablation" if args.ablation else "paraplex")
print(
    f"profile: {'sft-' if sft_mode else ''}{tag} device={device} amp={cfg.amp_dtype} optimizer={optimizer_name} "
    f"tria={getattr(cfg, 'tria_carry_enabled', False)} cuda_tria={getattr(cfg, 'use_cuda_tria', False)} "
    f"temporal={getattr(cfg, 'tria_temporal_enabled', False)} W={getattr(cfg, 'tria_temporal_window', None)} "
    f"chunks={(int(cfg.seq_len) + int(cfg.tria_temporal_window) - 1) // int(cfg.tria_temporal_window) if getattr(cfg, 'tria_temporal_enabled', False) else 1} "
    f"checkpoint={getattr(cfg, 'grad_checkpointing', False)} "
    f"warmup={args.warmup} steps={args.steps} accum={accum_steps}\n"
)

tria_count_dev = torch.zeros((), device=device, dtype=torch.long) if device.type == "cuda" else None
last_tokens = 0
chunk_stats_seen: list[dict] = []
sft_mask_stats_seen: list[dict] = []
profiled_loss_tokens = 0


def clear_tria_refs(raw_model) -> None:
    # Match the current training loop: release large Tria side-state after each backward.
    if hasattr(raw_model, "last_tria_depth_carry"):
        raw_model.last_tria_depth_carry = None
    if hasattr(raw_model, "last_tria_document_carry"):
        raw_model.last_tria_document_carry = None
    if hasattr(raw_model, "last_tria_document_carry_stats"):
        raw_model.last_tria_document_carry_stats = None
    if hasattr(raw_model, "last_tria_fire_mask"):
        raw_model.last_tria_fire_mask = None


def maybe_count_trias(raw_model) -> None:
    global tria_count_dev
    doc_stats = getattr(raw_model, "last_tria_document_carry_stats", None)
    if isinstance(doc_stats, dict):
        chunk_stats_seen.append(dict(doc_stats))
    fire_mask = getattr(raw_model, "last_tria_fire_mask", None)
    if args.no_trias_count or fire_mask is None:
        return
    with torch.no_grad():
        fired = fire_mask.detach().sum()
        if device.type == "cuda":
            tria_count_dev.add_(fired.to(dtype=torch.long))
        else:
            # CPU path: keep it simple; profiling CPU is not the target here.
            pass


def maybe_record_sft_masks(batch: dict) -> None:
    if not sft_mode or len(sft_mask_stats_seen) >= 4:
        return
    y = batch["y"]
    seg = batch["seg_id"]
    attn_mask = batch.get("attn_mask")
    boundaries = (seg[0, 1:] != seg[0, :-1]).nonzero(as_tuple=False).flatten()[:16].detach().cpu().tolist()
    sft_mask_stats_seen.append({
        "loss_tokens": int((y != loomsft.IGNORE_INDEX).sum().item()),
        "boundaries": int((seg[0, 1:] != seg[0, :-1]).sum().item()),
        "first_boundary_pos": boundaries,
        "attn_mask": None if attn_mask is None else tuple(attn_mask.shape),
    })


def one_microstep() -> tuple[torch.Tensor, int, int]:
    with tprof.record_function("loom.sample_batch"):
        if sft_mode:
            batch = loomsft.move_batch_to_device(stream.sample_batch(), device)
            x, y = batch["x"], batch["y"]
            position_ids, attn_mask = batch["position_ids"], batch["attn_mask"]
            maybe_record_sft_masks(batch)
            loss_tokens = int((y != loomsft.IGNORE_INDEX).sum().item())
        else:
            batch = stream.sample_device_batch()
            x, y = batch[:, :-1], batch[:, 1:]
            position_ids, attn_mask = lf.build_doc_reset_state(x, eos_id)
            loss_tokens = int(y.numel())
    with tprof.record_function("loom.forward"):
        with lf.amp_autocast(device):
            ignore_index = loomsft.IGNORE_INDEX if sft_mode else -100
            lm_loss = model(x, attn_mask=attn_mask, position_ids=position_ids,
                             labels=y, ignore_index=ignore_index)
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    maybe_count_trias(raw_model)
    return lm_loss, int(y.numel()), loss_tokens


def one_train_step(profiled: bool = False) -> tuple[torch.Tensor, int, int]:
    global last_tokens
    opt.zero_grad(set_to_none=True)
    loss_for_print: Optional[torch.Tensor] = None
    tokens = 0
    loss_tokens = 0
    for micro_idx in range(accum_steps):
        loss, ntok, nloss = one_microstep()
        tokens += ntok
        loss_tokens += nloss
        loss_for_print = loss.detach() if loss_for_print is None else loss_for_print + loss.detach()
        with tprof.record_function("loom.backward"):
            (loss / float(accum_steps)).backward()
        clear_tria_refs(model._orig_mod if hasattr(model, "_orig_mod") else model)
    with tprof.record_function("loom.optimizer_step"):
        if getattr(cfg, "grad_clip", 0.0) and float(cfg.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip))
        step_index = getattr(one_train_step, "_step_index", 0)
        lr = lf.lr_at(cfg, step_index)
        for group in opt.param_groups:
            group["lr"] = lr * float(group.get("lr_mult", 1.0))
        opt.step()
    one_train_step._step_index = step_index + 1
    last_tokens = tokens
    assert loss_for_print is not None
    return loss_for_print / float(accum_steps), tokens, loss_tokens


if bool(getattr(cfg, "graph", False)) and not sft_mode:
    # Single, isolated forward+backward on the JUST-compiled model -- no
    # grad-accum loop, exactly one .backward() call. Matches
    # train_one_async's own compile-warmup pass in loomformer.py, which
    # this script's args.warmup loop does NOT: one_train_step() runs
    # accum_steps separate .backward() calls per iteration (grad
    # accumulation), so its own first invocation makes Dynamo trace TWO
    # backward passes back-to-back instead of one -- a real, structural
    # difference from how real training first exercises the compiled
    # graph. See the conversation this was diagnosed in: a real training
    # run with the exact same graph:true config trained without this
    # error, this profiling script's own warmup didn't have this pass.
    _warm_batch = torch.randint(0, lf.VOCAB, (int(cfg.batch_size), lf.SEQ_LEN + 1), device=device, dtype=torch.long)
    _wx, _wy = _warm_batch[:, :-1], _warm_batch[:, 1:]
    _wpos, _wmask = lf.build_doc_reset_state(_wx, eos_id)
    with lf.amp_autocast(device):
        _wlogits = model(_wx, attn_mask=_wmask, position_ids=_wpos)
        _wloss = F.cross_entropy(_wlogits.float().reshape(-1, lf.VOCAB), _wy.reshape(-1))
    _wloss.backward()
    opt.zero_grad(set_to_none=True)
    clear_tria_refs(raw_model)
    del _warm_batch, _wx, _wy, _wpos, _wmask, _wlogits, _wloss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

if (not sft_mode) and hasattr(torch, "compile") and device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 7:
    _warm_batch2 = torch.randint(0, lf.VOCAB, (int(cfg.batch_size), lf.SEQ_LEN + 1), device=device, dtype=torch.long)
    _wx2, _wy2 = _warm_batch2[:, :-1], _warm_batch2[:, 1:]
    _wpos2, _wmask2 = lf.build_doc_reset_state(_wx2, eos_id)
    with lf.amp_autocast(device):
        _wlogits2 = model(_wx2, attn_mask=_wmask2, position_ids=_wpos2)
        _wloss2 = F.cross_entropy(_wlogits2.float().reshape(-1, lf.VOCAB), _wy2.reshape(-1))
    _wloss2.backward()
    opt.zero_grad(set_to_none=True)
    clear_tria_refs(raw_model)
    del _warm_batch2, _wx2, _wy2, _wpos2, _wmask2, _wlogits2, _wloss2
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

print("warmup...")
for _ in range(max(0, int(args.warmup))):
    one_train_step(profiled=False)
if device.type == "cuda":
    torch.cuda.synchronize()
    if args.empty_cache:
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

activities = [tprof.ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(tprof.ProfilerActivity.CUDA)

print("profiling...")
t0 = time.time()
with tprof.profile(
    activities=activities,
    record_shapes=bool(args.shapes),
    profile_memory=bool(args.memory),
    with_stack=bool(args.stack),
    with_flops=False,
    acc_events=True,
) as prof:
    last_loss_tensor: Optional[torch.Tensor] = None
    profiled_tokens = 0
    profiled_loss_tokens = 0
    for step in range(max(1, int(args.steps))):
        with tprof.record_function(f"loom.train_step_{step + 1}"):
            last_loss_tensor, ntok, nloss = one_train_step(profiled=True)
            profiled_tokens += ntok
            profiled_loss_tokens += nloss
        prof.step()

if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - t0
last_loss = float("nan") if last_loss_tensor is None else float(last_loss_tensor.item())

lines: list[str] = []
lines.append(
    f"loss_last={last_loss:.6f} "
    f"profiled_tokens={profiled_tokens:,} "
    f"profiled_loss_tokens={profiled_loss_tokens:,} "
    f"elapsed={elapsed:.3f}s")
if device.type == "cuda":
    lines.append(
        "cuda_mem: "
        f"allocated={fmt_bytes(torch.cuda.memory_allocated(device))}  "
        f"reserved={fmt_bytes(torch.cuda.memory_reserved(device))}  "
        f"peak_allocated={fmt_bytes(torch.cuda.max_memory_allocated(device))}  "
        f"peak_reserved={fmt_bytes(torch.cuda.max_memory_reserved(device))}"
    )
    if not args.no_trias_count and tria_count_dev is not None:
        trias = int(tria_count_dev.item())
        lines.append(f"trias_profiled_region={trias}  rate={trias / max(1, profiled_tokens):.6e}")
if chunk_stats_seen:
    max_abs = max(float(s.get("max_abs", 0.0)) for s in chunk_stats_seen)
    total_resets = sum(int(s.get("reset_count", 0)) for s in chunk_stats_seen)
    total_fires = sum(int(s.get("fire_count", 0)) for s in chunk_stats_seen)
    lines.append(
        "tria_doc_stats: "
        f"steps_seen={len(chunk_stats_seen)} "
        f"max_abs={max_abs:.6e} "
        f"resets_total={total_resets} "
        f"fires_total={total_fires} "
        f"last={chunk_stats_seen[-1]}"
    )
if sft_mask_stats_seen:
    lines.append("sft_masks:")
    for i, stats in enumerate(sft_mask_stats_seen):
        lines.append(
            f"  batch{i}: "
            f"loss_tokens={stats['loss_tokens']} "
            f"boundaries={stats['boundaries']} "
            f"attn_mask={stats['attn_mask']} "
            f"first_boundary_pos={stats['first_boundary_pos']}"
        )

sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
lines.append("\n== top ops by self time ==")
lines.append(prof.key_averages(group_by_input_shape=bool(args.shapes)).table(sort_by=sort_key, row_limit=int(args.top)))

# Fwd/bwd/opt breakdown by record_function regions
ka = prof.key_averages(group_by_input_shape=False)
regions = {"loom.forward": 0.0, "loom.backward": 0.0, "loom.optimizer_step": 0.0}
for e in ka:
    if e.key in regions:
        regions[e.key] = e.device_time_total if device.type == "cuda" else e.cpu_time_total
lines.append("\n== fwd/bwd/opt breakdown (total time) ==")
for name, t_us in regions.items():
    lines.append(f"  {name:30s}  {t_us / 1000:.3f} ms")

loom_labeled = [
    e for e in ka
    if e.key.startswith("loom.attn.")
    or e.key.startswith("loom.ffn.")
    or e.key.startswith("loom.model.cat")
    or e.key.startswith("loom.tria.cat")
]
if loom_labeled:
    loom_labeled.sort(
        key=lambda e: e.device_time_total if device.type == "cuda" else e.cpu_time_total,
        reverse=True,
    )
    lines.append("\n== loom labeled regions (total time) ==")
    for e in loom_labeled:
        t = e.device_time_total if device.type == "cuda" else e.cpu_time_total
        lines.append(f"  {e.key:45s}  {t / 1000:.3f} ms  ({e.count} calls)")

compiled_markers = ("triton", "inductor", "cudagraph", "cudaGraph", "CompiledFunction", "compiled_autograd")
compiled_ka = [e for e in ka if any(m in e.key for m in compiled_markers)]
if compiled_ka:
    compiled_ka.sort(key=lambda e: e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total, reverse=True)
    total_compiled = sum(e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total for e in compiled_ka)
    lines.append(f"\n== compiled/graph-ish events by self time (total: {total_compiled / 1000:.3f} ms) ==")
    for e in compiled_ka[:int(args.top)]:
        t = e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total
        lines.append(f"  {e.key:55s}  {t / 1000:10.3f} ms  ({e.count} calls)")

# Backward-only ops: autograd backward kernels have 'Backward' in key or
# 'backward' in the CUDA kernel name. loom.backward region itself shows ~0
# because autograd launches CUDA kernels outside the Python call stack.
bwd_keys = set()
for e in ka:
    k = e.key
    if "Backward" in k or "backward" in k or "_backward" in k:
        bwd_keys.add(k)
if bwd_keys:
    bwd_ka = [e for e in ka if e.key in bwd_keys]
    bwd_ka.sort(key=lambda e: e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total, reverse=True)
    total_bwd = sum(e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total for e in bwd_ka)
    lines.append(f"\n== backward ops by self time (total: {total_bwd / 1000:.3f} ms) ==")
    for e in bwd_ka[:int(args.top)]:
        t = e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total
        calls = e.count
        lines.append(f"  {e.key:55s}  {t / 1000:10.3f} ms  ({calls} calls)")

# Forward-only ops: everything except backward ops, record_function regions,
# autograd engine bookkeeping, and optimizer step.
fwd_ka = []
for e in ka:
    k = e.key
    if k in bwd_keys or "Backward" in k or "backward" in k:
        continue
    if k.startswith("loom.") or k.startswith("autograd::engine") or k.startswith("Optimizer"):
        continue
    fwd_ka.append(e)
fwd_ka.sort(key=lambda e: e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total, reverse=True)
total_fwd = sum(e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total for e in fwd_ka)
lines.append(f"\n== forward ops by self time (total: {total_fwd / 1000:.3f} ms) ==")
for e in fwd_ka[:int(args.top)]:
    t = e.self_device_time_total if device.type == "cuda" else e.self_cpu_time_total
    calls = e.count
    lines.append(f"  {e.key:55s}  {t / 1000:10.3f} ms  ({calls} calls)")

if args.memory:
    mem_key = "self_cuda_memory_usage" if device.type == "cuda" else "self_cpu_memory_usage"
    lines.append("\n== top ops by self memory ==")
    lines.append(prof.key_averages(group_by_input_shape=bool(args.shapes)).table(sort_by=mem_key, row_limit=int(args.top)))

text = "\n".join(lines)
print(text)
if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nsummary saved: {args.out}")
if args.trace:
    prof.export_chrome_trace(args.trace)
    print(f"trace saved: {args.trace}")
