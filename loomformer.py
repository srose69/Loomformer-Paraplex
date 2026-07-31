#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import gc
import contextlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP

import tria  # sibling module, same directory -- see tria.py's own docstring

if __name__ == "__main__":
    sys.modules.setdefault("loomformer", sys.modules[__name__])


from loomformer_runtime.distributed import (
    configure_cuda_math,
    set_seed,
    device_auto,
    ddp_world_size,
    ddp_rank,
    ddp_local_rank,
    ddp_is_distributed,
    ddp_is_main,
    ddp_print,
    ddp_trace,
    ddp_barrier,
    ddp_mean_float,
    ddp_sum_int,
    ddp_weighted_mean,
    ddp_unwrap_model,
    ddp_sync_mutable_buffers,
    ddp_assert_config_consensus,
    ddp_static_graph_policy,
    maybe_launch_or_init_ddp,
)
def amp_autocast(dev: torch.device):
    # AMP_DTYPE is set from Config/CLI in apply_config().
    # "fp32"/"off" = no autocast; "bf16" = CUDA BF16 autocast when supported.
    amp = str(globals().get("AMP_DTYPE", "fp32") or "fp32").lower()
    if dev.type != "cuda":
        return contextlib.nullcontext()
    if amp in ("fp32", "float32", "off", "none", "false", "0", "no"):
        return contextlib.nullcontext()
    if amp in ("bf16", "bfloat16"):
        if torch.cuda.is_bf16_supported():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()
    if amp in ("fp16", "float16", "half"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"amp_dtype must be fp32/off, bf16, or fp16; got {amp!r}")




@dataclass
class Config:
    # tokenizer / data
    vocab: int = 256
    tokenizer: Optional[str] = None
    tied_embeddings: bool = True
    doc_reset_attn: bool = True
    # Device selector. Like CLI --device: cpu | cuda:0 | cuda:1 | cudas | cuda:0,cuda:1.
    # "cudas" self-launches torchrun over all visible CUDA devices; a comma list
    # self-launches DDP over that CUDA_VISIBLE_DEVICES subset.
    device: Optional[str] = None
    # Optional train dataset path. Pure alias for --dataset: CLI wins when both are set.
    train_dataset: Optional[str] = None
    # Optional held-out dataset. When set, training logs use this stream for
    # eval_loss/bits/bpb instead of sampling from the train dataset.
    val_dataset: Optional[str] = None
    # Destructive pretrain-only holdout split. If >0 and val_dataset is unset,
    # split this percent of EACH top-level corpus file into <train_dataset>/val/val_split.*,
    # rewriting the original files as train-only. 1.0 => 1% total for equal-sharded corpora.
    auto_val_split_pct: float = 0.0
    # "auto" infers from the path's extension (.bin -> prepared tokens, .txt/.jsonl/
    # .parquet/.arrow -> on-the-fly RawCorpus). Force one explicitly if a path is
    # ambiguous (e.g. a directory of mixed files).
    dataset_format: str = "auto"  # auto | bin | txt | jsonl | parquet | arrow
    text_field: str = "text"       # column/key holding the text in jsonl/parquet/arrow rows
    seq_len: int = 128
    batch_size: int = 32

    # model shape
    # model_dim is the residual/model width (aka d_model, hidden_size elsewhere).
    # If head_dim is set instead, model_dim is derived as n_q_heads * head_dim.
    model_dim: Optional[int] = 12
    n_q_heads: int = 6
    head_dim: Optional[int] = None
    # GQA (grouped-query attention): n_q_heads query heads share fewer key/value heads.
    # Set ONE of n_kv_heads or gqa_group_size (the other is derived); set neither for
    # plain multi-head attention (n_kv_heads = n_q_heads, group=1).
    #   n_kv_heads     -- how many KV heads to have, directly (e.g. 2).
    #   gqa_group_size -- how many query heads share EACH kv head (e.g. 2 query heads
    #                     per kv head => n_kv_heads = n_q_heads / gqa_group_size).
    n_kv_heads: Optional[int] = 3
    gqa_group_size: Optional[int] = None
    hidden: Optional[int] = 66
    hidden_mult: Optional[float] = None  # if hidden is null: hidden = round_up(model_dim * hidden_mult, n_q_heads)
    layers: int = 3
    # phase sectoring: "head" = neuron hears only its query head's Q/Kctx/C (+full U);
    # "open" = own-head Q, but Kctx/C from ALL heads (+full U) — cross-head synthesis in phase.
    phase_sectors: str = "head"
    # residual-producing matrices (attn.v, attn.o, ffn.w2) init: "beta" = DeepNorm-style
    # down-scaled init (beta=(8N)^-1/4); "fanin" = plain fan-in, no down-scaling. Ablation
    # control for testing whether beta-scaling is still load-bearing now that the skip
    # term itself is DepthAttn (softmax-over-history) instead of a fixed alpha*h.
    residual_init: str = "beta"
    # The shared depth readout is cheap, but it is also a single low-rank failure
    # point used by every attention and FFN residual.  "per-sublayer" gives each
    # of the 2*layers calls its own output projection.  qkv_rms fixes the scale of
    # the vectors entering the existing depth-attention kernel; residual_rms
    # caps runaway post-LayerNorm branches without amplifying quiet branches.
    depth_attn_readout: str = "shared"  # shared | per-sublayer
    depth_attn_qkv_rms: bool = False
    residual_branch_rms_cap: Optional[float] = None
    # outer activation (stays OUTSIDE the primitive for both FFN types, per the original
    # design: "activations are outside, that's what's inside" -- w2(activation(p))):
    #   "gelu"  = default, unchanged.
    #   "powlu"       = Power Linear Unit (Jiang et al., Ant Group, arXiv:2605.25704, 2026),
    #     UNGATED single-input form (paper's Eq.1, x1=x2=x): PowLU(x) = x*x^(m/(sqrt(x)+1))*
    #     sigmoid(x) for x>0, x^2*sigmoid(x) for x<=0. Tames SwiGLU's x^2 blow-up toward
    #     near-linear growth on large positive inputs. Needs a base-clamp before pow (torch.
    #     where computes both branches' backward; fractional power of a negative base = NaN
    #     that poisons the gradient even in the unselected branch).
    #   "pvpowlu" = PowLU's GATED form (paper's practical x1*f(x2) wiring), but x2 is
    #     REUSED from an already-computed, guaranteed-positive quantity instead of a new
    #     weight matrix: Paraplex reuses amp=softplus(p_real), which is already in hand.
    #     x2>0 by construction -> no clamp/where needed at all, physically not
    #     just numerically -- the x<=0 branch can never fire.
    activation: str = "gelu"
    powlu_m: float = 3.0
    phase_grad_floor: float = 0.05
    phase_grad_mode: str = "floor"
    use_cuda_phase_sin: bool = True
    use_cuda_beta_space: bool = True
    use_cuda_pvpowlu: bool = True
    use_cuda_depth_attn: bool = True
    # AMP/autocast mode: "bf16" (default), "fp32"/"off" (no autocast), or "fp16".
    amp_dtype: str = "fp32"
    dataset_cache: str = "mmap"
    # auto: FlashAttention varlen on supported CUDA GPUs, compact SDPA/manual
    # fallback everywhere else.  "flash" is strict (fail instead of silently
    # falling back), while sdpa/manual are explicit diagnostic overrides.
    attn_impl: str = "auto"
    attn_layers: Optional[List[int]] = None
    attn_token_stride: int = 1
    attn_token_schedule: str = "shared"
    # SDPA-only compute dtype: "model" keeps q/k/v dtype, "fp32"/"fp16"/"bf16"
    # force attention compute dtype, "auto" keeps BF16 only when the efficient
    # backend accepts it and otherwise promotes SDPA inputs to FP32.
    attn_sdpa_compute_dtype: str = "auto"
    # True: one SDPA with value=[K;V]. False: two SDPA calls with value_dim=head_dim.
    # The split path can have a cheaper mem-efficient backward on some GPUs.
    attn_sdpa_value_fusion: bool = True
    # Debug/perf fallback for old GPUs where FP32 mem-efficient backward is slow:
    # recompute softmax in a custom autograd backward instead of using SDPA bwd.
    attn_sdpa_recompute_backward: bool = False
    rope_theta: float = 10000.0
    rope_factor: float = 4.0
    rope_original_seq_len: Optional[int] = None
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_attention_factor: Optional[float] = None

    # training
    steps: int = 2000
    lr: float = 2e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Divide the loss by this factor during backward, then restore the
    # gradient scale after clipping. Values >1 protect deep BF16/FP32
    # backward graphs from overflowing before global clipping can run.
    backward_scale: float = 1.0
    grad_accum_steps: int = 1
    prefetch_batches: int = 256
    gpu_prefetch_batches: int = 8
    grad_checkpointing: bool = False
    fsdp_full_shard: bool = False
    optimizer_zero_shard: bool = False
    save_every: int = 0 
    runpoints_path: Optional[str] = None  
    save_initial_checkpoint: bool = False
    save_final_checkpoint: bool = True
    # Fresh weight initialization and final checkpoint target may live in YAML,
    # so a complete train/SFT run needs only `--train --config ...`.
    init_checkpoint: Optional[str] = None
    checkpoint: Optional[str] = None
    tria_carry_enabled: bool = False
    # ===AUTO GENERATED=== bookkeeping written by loomcloner.py --scan/--clone.
    # cloned/cloned_from/cloned_mapping are informational only (which donor,
    # which mappings/*.json). train_lr is the one that's actually consumed:
    # a list of {"name": <LoomFormer param-name suffix or exact global name>,
    # "train": bool, "lr": float} entries, matched by suffix against every
    # blocks.{i}.<name> and by exact match against global names (emb.weight,
    # head.weight) -- see apply_train_lr_overrides().
    cloned: bool = False
    cloned_from: Optional[str] = None
    cloned_mapping: Optional[str] = None
    train_lr: Optional[List[Dict[str, Any]]] = None
    # --resume as a config field, not just a CLI flag -- loomcloner.py --clone
    # writes this in automatically so `--train --config X.yaml` alone resumes
    # the cloned checkpoint without needing a separate --resume on the CLI.
    # An explicit --resume on the command line still overrides this.
    resume: Optional[str] = None
    # Dataset cursor policy for --resume:
    #   auto     -- restore this dataset's saved cursor, or start a new one;
    #   continue -- restore it, with legacy global-step replay as fallback;
    #   restart  -- keep checkpoint step/LR schedule but start data at draw 0.
    resume_data_stream: str = "auto"
    # False (default): ParaplexFFN's amp gate is self-referential, amp=softplus(p_real)
    #   (original design, zero extra parameters).
    # True: amp comes from an independent gate_proj Linear(N,HIDDEN) instead --
    #   the slot a SwiGLU donor's gate_proj maps onto during --rebuild/loomcloner
    #   transplant. Adds HIDDEN*N parameters per layer; requires the extended
    #   paraplex CUDA kernel (gate_src argument) to keep the fused fast path.
    paraplex_gate_proj: bool = False
    # False (default, matches every checkpoint trained before this option
    # existed): head reads the last block's residual stream directly, no
    # final normalization -- LoomFormer's own long-standing design.
    # True: one RMSNorm right before head, stabilizing the scale that's
    # drifted across LAYERS blocks of pre-norm residual accumulation --
    # the slot a Llama-family donor's model.norm.weight maps onto during
    # --rebuild/loomcloner transplant (previously silently dropped).
    final_norm: bool = False
    # Fused linear + cross-entropy for the LM head (Liger-Kernel style chunked
    # projection, see `_FusedLinearCrossEntropy`): avoids ever materializing
    # the full [B*T, VOCAB] logits tensor. One-line opt-in; the model/train
    # call sites are unchanged (labels=None still returns full logits).
    fused_linear_ce: bool = False
    fused_linear_ce_chunk_size: int = 0  # 0 = auto (Liger memory-balancing formula); or a fixed row-chunk override
    use_cuda_tria: bool = False
    # Independent switches:
    #   compile -- wrap the model with torch.compile/Dynamo/Inductor;
    #   graph   -- register the pybind CUDA kernels as torch.library custom
    #              ops so Dynamo can trace through them without graph breaks.
    compile: bool = False
    graph: bool = False
    save_graph: bool = False  
    tria_temporal_enabled: bool = True
    tria_temporal_window: Optional[int] = None
    tria_temporal_window_min: int = 32
    tria_temporal_window_max: Optional[int] = None
    tria_temporal_calibration: Optional[str] = None 
    tria_temporal_auto: bool = True
    tria_carrier_alpha: float = 0.05
    tria_carrier_alpha_candidates: Optional[List[float]] = None
    tria_polarm_beta: float = 0.1
    tria_min_refeeds_per_sequence: int = 1
    tria_temporal_max_condition: float = 3.0
    tria_temporal_min_effective_rank: float = 2.70
    tria_temporal_population_pass_fraction: float = 0.90
    tria_temporal_calib_seeds: int = 3
    tria_temporal_calib_batch: int = 4
    tria_temporal_calib_tokens: Optional[int] = None
    tria_temporal_calib_device: str = "auto"
    tria_temporal_calib_parallel_sweep: int = 1
    tria_target_refeeds_per_sequence: int = 3
    tria_gamma_max: float = 0.25
    tria_raw_gamma_init: float = 0.0     
    warmup_steps: int = 100
    min_lr_frac: float = 0.1
    seed: int = 1
    log_every: int = 100
    eval_every: Optional[int] = None
    eval_batches: int = 4
    optimizer: str = "adamw"  # adamw | atom

    def summary(self) -> str:
        hd = self.head_dim if self.head_dim is not None else "auto"
        grp = self.gqa_group_size if self.gqa_group_size is not None else "auto"
        active = self.layers if self.attn_layers is None else len(self.attn_layers)
        return (
            f"LoomFormer [V={self.vocab} d_model={self.model_dim} qh={self.n_q_heads} "
            f"head_dim={hd} kvh={self.n_kv_heads} group={grp} "
            f"H={self.hidden} D={self.layers} attn={active}/{self.layers}"
            f"x{self.attn_token_stride} T={self.seq_len} B={self.batch_size}]"
        )

    @staticmethod
    def from_yaml(path: str) -> "Config":
        import yaml
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return Config.from_dict(d)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        """Build a config after rejecting unknown fields and coercing float strings."""
        import dataclasses
        field_map = {f.name: f for f in dataclasses.fields(Config)}
        unknown = sorted(set(d) - set(field_map))
        if unknown:
            raise ValueError(f"unknown config field(s): {unknown}")
        values = dict(d)
        float_fields = {name for name, f in field_map.items() if "float" in str(f.type)}
        for name in float_fields:
            value = values.get(name)
            if isinstance(value, str):
                values[name] = float(value)
        return Config(**values)

    @staticmethod
    def from_checkpoint_dict(d: dict) -> "Config":
        """Migrate supported legacy checkpoint fields and build a validated config."""
        values = dict(d)
        legacy_window = values.pop("tria_temporal_deadline", None)
        current_window = values.get("tria_temporal_window")
        if legacy_window is not None and current_window is not None and int(legacy_window) != int(current_window):
            raise ValueError(
                "checkpoint has conflicting tria_temporal_window/tria_temporal_deadline values: "
                f"{current_window!r} vs {legacy_window!r}"
            )
        if current_window is None and legacy_window is not None:
            values["tria_temporal_window"] = int(legacy_window)
        removed_baseline_hidden = values.pop("baseline_hidden", None)
        if removed_baseline_hidden is not None:
            raise ValueError("checkpoint requests removed GeluFFN baseline_hidden")
        removed_train_baseline = values.pop("train_baseline", False)
        if bool(removed_train_baseline):
            raise ValueError("checkpoint requests removed GeluFFN baseline training")
        return Config.from_dict(values)


def apply_temporal_tria_calibration(cfg: Config) -> None:
    path = getattr(cfg, "tria_temporal_calibration", None)
    if not path:
        return
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    src = blob.get("recommended", blob)
    required = ("tria_temporal_window",)
    missing = [k for k in required if k not in src]
    if missing:
        raise ValueError(f"{path!r} is missing temporal Tria calibration keys: {missing}")
    cfg.tria_temporal_window = int(src["tria_temporal_window"])
    if "tria_carrier_alpha" in src:
        cfg.tria_carrier_alpha = float(src["tria_carrier_alpha"])
    ddp_print(
        "[tria] temporal calibration loaded "
        f"{path}: W={cfg.tria_temporal_window} alpha={cfg.tria_carrier_alpha:g}"
    )


def _carrier_alpha_candidates(cfg: Config) -> List[float]:
    raw = getattr(cfg, "tria_carrier_alpha_candidates", None)
    vals = [0.025, 0.0375, 0.05] if raw is None else [float(x) for x in raw]
    vals.append(float(getattr(cfg, "tria_carrier_alpha", 0.05)))
    out = sorted({x for x in vals if math.isfinite(x) and x > 0.0})
    if not out:
        raise ValueError("tria_carrier_alpha_candidates must contain a positive finite value")
    return out


@torch.no_grad()
def calibrate_temporal_tria_from_init(cfg: Config) -> dict:
    device_pref = str(getattr(cfg, "tria_temporal_calib_device", "auto") or "auto").lower()
    if device_pref == "auto":
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
    else:
        device = torch.device(device_pref)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("tria_temporal_calib_device requests CUDA, but CUDA is unavailable")
    seeds = max(1, int(getattr(cfg, "tria_temporal_calib_seeds", 3)))
    batch = max(1, int(getattr(cfg, "tria_temporal_calib_batch", 4)))
    parallel_sweep = max(1, int(getattr(cfg, "tria_temporal_calib_parallel_sweep", 1)))

    target_refeeds = max(1, int(getattr(cfg, "tria_target_refeeds_per_sequence", 3)))
    min_refeeds = max(1, int(getattr(cfg, "tria_min_refeeds_per_sequence", 1)))
    target_refeeds = max(target_refeeds, min_refeeds)
    window_min = max(2, int(getattr(cfg, "tria_temporal_window_min", 8)))
    window_max_cfg = getattr(cfg, "tria_temporal_window_max", None)
    max_possible_refeeds = max(0, (SEQ_LEN - 1) // 2)  # W cannot be below 2.
    if max_possible_refeeds < min_refeeds:
        raise ValueError(
            f"seq_len={SEQ_LEN} cannot contain tria_min_refeeds_per_sequence={min_refeeds}; "
            "increase seq_len or lower the minimum")
    # A target of N refeeds reserves N+1 windows in the configured training
    # sequence. This intentionally prefers a real mid-sequence refeed over the
    # largest numerically stable W: T=512,N=3 -> W<=128.
    target_window = max(2, SEQ_LEN // (target_refeeds + 1))
    if window_max_cfg is not None:
        target_window = min(target_window, int(window_max_cfg))
    # Calibration only has to reach the W we may actually select. Running it
    # over the entire seq_len would multiply startup CPU work by N+1 for no
    # decision benefit. An explicit calib_tokens remains a user-controlled cap.
    requested_tokens = getattr(cfg, "tria_temporal_calib_tokens", None)
    T = int(target_window if requested_tokens is None else min(SEQ_LEN, int(requested_tokens)))
    if T <= 1:
        raise ValueError(f"temporal Tria calibration needs at least 2 tokens, got usable T={T}")
    target_window = min(T, target_window)
    if target_window < window_min:
        ddp_print(
            f"[tria] requested target refeeds cap W at {target_window}, below "
            f"tria_temporal_window_min={window_min}; preserving the refeed target")
    else:
        target_window = max(window_min, target_window)

    max_condition = float(getattr(cfg, "tria_temporal_max_condition", 3.0))
    min_effrank = float(getattr(cfg, "tria_temporal_min_effective_rank", 2.70))
    pass_fraction_req = float(getattr(cfg, "tria_temporal_population_pass_fraction", 0.90))
    if max_condition <= 1.0 or not (1.0 <= min_effrank <= 3.0):
        raise ValueError("bad Tria stability thresholds")
    if not (0.0 < pass_fraction_req <= 1.0):
        raise ValueError("tria_temporal_population_pass_fraction must be in (0,1]")

    carry_token_id = CARRY_TOKEN_ID
    alphas = _carrier_alpha_candidates(cfg)
    accum = {
        alpha: {
            "pass": torch.zeros(T, dtype=torch.float32, device=device),
            "cond": torch.zeros(T, dtype=torch.float32, device=device),
            "rank": torch.zeros(T, dtype=torch.float32, device=device),
            "count": 0,
        }
        for alpha in alphas
    }
    sweep_tasks = [(alpha, seed_idx) for alpha in alphas for seed_idx in range(seeds)]
    for task_start in range(0, len(sweep_tasks), parallel_sweep):
        sweep_batch = sweep_tasks[task_start:task_start + parallel_sweep]
        jobs = []
        for alpha, seed_idx in sweep_batch:
            stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
            stream_ctx = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
            with stream_ctx:
                torch.manual_seed(int(getattr(cfg, "seed", 1)) + 9176 + seed_idx)
                candidate_cfg = replace(
                    cfg, tria_carrier_alpha=float(alpha), tria_temporal_auto=False)
                model = Model(candidate_cfg).to(device).eval()
                model.capture_tria_depth_carry = True
                idx = torch.randint(0, VOCAB, (batch, T), device=device, dtype=torch.long)
                if carry_token_id is not None:
                    idx = torch.where(idx.eq(int(carry_token_id)), (idx + 1) % VOCAB, idx)
                position_ids = torch.arange(T, device=device, dtype=torch.long).view(1, T).expand(batch, T)
                # Flat path avoids self-reference through the current candidate W.
                model._forward_flat(idx, attn_mask=None, position_ids=position_ids)
                depth_carry = model.last_tria_depth_carry
                if depth_carry is None:
                    raise RuntimeError("temporal Tria auto-calibration requires tria_carry_enabled with paraplex")
                reset_mask = torch.zeros(batch, T, dtype=torch.bool, device=device)
                reset_mask[:, 0] = True
                document_carry = tria.temporal_carry(depth_carry.float(), reset_mask)
                B, Tc, H = document_carry.shape[:3]
                sv = torch.linalg.svdvals(document_carry.reshape(B * Tc, H, 3, 3))
                cond = (sv[..., 0] / sv[..., -1].clamp_min(1e-12)).reshape(B, Tc, H)
                prob = sv / sv.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                effrank = torch.exp(-(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)).reshape(B, Tc, H)
                ok = (cond <= max_condition) & (effrank >= min_effrank)
                jobs.append((alpha, stream, model, ok.float().sum(dim=(0, 2)),
                             cond.float().sum(dim=(0, 2)), effrank.float().sum(dim=(0, 2)), B * H))
        for alpha, stream, model, pass_part, cond_part, rank_part, count in jobs:
            if stream is not None:
                stream.synchronize()
            accum[alpha]["pass"] += pass_part
            accum[alpha]["cond"] += cond_part
            accum[alpha]["rank"] += rank_part
            accum[alpha]["count"] += count
            del model, pass_part, cond_part, rank_part
        jobs.clear()

    candidate_results = []
    for alpha in alphas:
        total_count = accum[alpha]["count"]
        pass_sum = accum[alpha]["pass"]
        cond_sum = accum[alpha]["cond"]
        rank_sum = accum[alpha]["rank"]
        pass_fraction = pass_sum / max(total_count, 1)
        cond_mean = cond_sum / max(total_count, 1)
        rank_mean = rank_sum / max(total_count, 1)
        failed = torch.nonzero(pass_fraction < pass_fraction_req, as_tuple=False)
        stable_horizon = T if failed.numel() == 0 else max(1, int(failed[0, 0].item()))
        candidate_results.append({
            "alpha": float(alpha),
            "stable_horizon": int(stable_horizon),
            "pass_fraction": pass_fraction.cpu().tolist(),
            "condition_mean": cond_mean.cpu().tolist(),
            "effective_rank_mean": rank_mean.cpu().tolist(),
        })

    eligible = [x for x in candidate_results if x["stable_horizon"] >= target_window]
    if eligible:
        selected = max(eligible, key=lambda x: x["alpha"])
        selected_window = target_window
        censored = selected["stable_horizon"] >= T
    else:
        selected = max(candidate_results, key=lambda x: (x["stable_horizon"], -x["alpha"]))
        selected_window = max(2, min(target_window, selected["stable_horizon"]))
        censored = False
    selected_window = max(2, selected_window)
    selected_alpha = float(selected["alpha"])
    selected_pass = float(selected["pass_fraction"][selected_window - 1])
    if selected_pass < pass_fraction_req:
        raise RuntimeError(
            "temporal Tria calibration found no valid window: "
            f"best alpha={selected_alpha:g}, W={selected_window}, "
            f"condition={selected['condition_mean'][selected_window - 1]:.3f}, "
            f"effective_rank={selected['effective_rank_mean'][selected_window - 1]:.3f}, "
            f"population_pass={selected_pass:.3f} < {pass_fraction_req:.3f}"
        )
    expected_refeeds = int((SEQ_LEN - 1) // selected_window)

    result = {
        "source": "forward_init_carrier",
        "tria_temporal_window": int(selected_window),
        "tria_carrier_alpha": selected_alpha,
        "target_window": int(target_window),
        "expected_refeeds_per_sequence": expected_refeeds,
        "target_refeeds_per_sequence": target_refeeds,
        "stable_horizon": int(selected["stable_horizon"]),
        "condition_at_window": float(selected["condition_mean"][selected_window - 1]),
        "effective_rank_at_window": float(selected["effective_rank_mean"][selected_window - 1]),
        "population_pass_at_window": float(selected["pass_fraction"][selected_window - 1]),
        "max_condition": max_condition,
        "min_effective_rank": min_effrank,
        "required_population_pass": pass_fraction_req,
        "calib_tokens": T,
        "calib_seeds": seeds,
        "calib_batch": batch,
        "calib_device": str(device),
        "calib_parallel_sweep": parallel_sweep,
        "censored": censored,
        "candidates": candidate_results,
    }
    ddp_print(
        "[tria] carrier auto-calibration: "
        f"device={device} H={cfg.hidden} L={cfg.layers} T={T} -> alpha={selected_alpha:g} W={selected_window} "
        f"refeeds={expected_refeeds} cond={result['condition_at_window']:.3f} "
        f"erank={result['effective_rank_at_window']:.3f} pass={result['population_pass_at_window']:.3f}"
    )
    return result


def apply_temporal_tria_auto_calibration(cfg: Config) -> None:
    if getattr(cfg, "tria_temporal_calibration", None):
        apply_temporal_tria_calibration(cfg)
        return
    if not bool(getattr(cfg, "tria_temporal_auto", True)):
        return
    result = calibrate_temporal_tria_from_init(cfg)
    cfg.tria_temporal_window = result["tria_temporal_window"]
    cfg.tria_carrier_alpha = result["tria_carrier_alpha"]
    cfg._tria_temporal_auto_result = result


def restore_temporal_tria_from_checkpoint(cfg: Config, path: Optional[str]) -> bool:
    """Report checkpoint Tria geometry without mutating the active config.

    Kept under its old name for callers outside this repository. The active
    YAML/Config is the SSOT on resume; checkpoint metadata is diagnostic only.
    """
    if not path:
        return False
    blob = torch.load(path, map_location="cpu", weights_only=True)
    saved_cfg = blob.get("cfg", {})
    keys = ("tria_temporal_window", "tria_carrier_alpha", "tria_polarm_beta")
    found = {key: saved_cfg.get(key) for key in keys if saved_cfg.get(key) is not None}
    if not found:
        return False
    active = {key: getattr(cfg, key) for key in keys}
    changed = {
        key: (saved, active[key])
        for key, saved in found.items()
        if active[key] is not None and float(saved) != float(active[key])
    }
    if changed:
        details = ", ".join(
            f"{key}: checkpoint={saved:g} config={current:g}"
            for key, (saved, current) in changed.items()
        )
        ddp_print(f"[resume] Tria geometry differs; keeping config SSOT ({details})")
    else:
        ddp_print("[resume] Tria geometry matches active config")
    return True


from loomformer_runtime.checkpoints import (
    assert_resume_attention_config,
    dataset_progress_key,
    resolve_resume_dataset_progress,
    checkpoint_tokens_seen,
)

# Shape globals used by the compact module definitions below. They are set from
# Config before any model is constructed.
N = 0
N_Q_HEADS = 0
N_KV_HEADS = 0
HIDDEN = 0
LAYERS = 0
VOCAB = 0
SEQ_LEN = 0
HEAD_DIM = 0
GQA_GROUP_SIZE = 0
TIED_EMBEDDINGS = True
KV_DIM = 0
HIDDEN_PER_Q_HEAD = 0
IMAG_IN = 0
PHASE_SECTORS = "head"
GRAD_CHECKPOINTING = False
TRIA_CARRY_ENABLED = False
PARAPLEX_GATE_PROJ = False  # False: amp = softplus(p_real), self-referential (original design)
                            # True: amp = softplus(gate_proj(u)), independent learned gate
                            # (donor-transplant path: gate_proj maps 1:1 onto a SwiGLU donor's
                            # gate_proj matrix -- see loomcloner.py mapping notes)
FINAL_NORM_ENABLED = False  # False: head reads the residual stream directly (original design)
                            # True: one RMSNorm before head -- see Config.final_norm
FUSED_LINEAR_CE = False  # see Config.fused_linear_ce / _FusedLinearCrossEntropy
FUSED_LINEAR_CE_CHUNK_SIZE = 0  # 0 = auto; see Config.fused_linear_ce_chunk_size
TRIA_GAMMA_MAX = 0.25
TRIA_RAW_GAMMA_INIT = 0.0
TRIA_TEMPORAL_ENABLED = True
CARRY_TOKEN_ID: Optional[int] = None
RESIDUAL_INIT = "beta"
ACTIVATION = "gelu"
POWLU_M = 3.0
PHASE_GRAD_FLOOR = 0.05
PHASE_GRAD_MODE = "floor"
USE_CUDA_PHASE_SIN = True
USE_CUDA_BETA_SPACE = True
USE_CUDA_PVPOWLU = True
USE_CUDA_DEPTH_ATTN = True
DEPTH_ATTN_READOUT = "shared"
DEPTH_ATTN_QKV_RMS = False
RESIDUAL_BRANCH_RMS_CAP: Optional[float] = None
GRAPH_MODE_ENABLED = False
_graph_pvpowlu_op = None
_graph_phase_sin_op = None
_graph_phase_sin_secant_op = None
_graph_depth_attn_op = None
_graph_beta_space_op = None
AMP_DTYPE = "fp32"
ATTN_IMPL = "auto"
ATTN_LAYERS: Tuple[int, ...] = ()
ATTN_SDPA_COMPUTE_DTYPE = "auto"
ATTN_SDPA_VALUE_FUSION = True
ATTN_SDPA_RECOMPUTE_BACKWARD = True
_REAL_STDOUT = sys.stdout
ROPE_THETA = 10000.0
ROPE_FACTOR = 4.0
ROPE_ORIGINAL_SEQ_LEN = 0
ROPE_BETA_FAST = 32.0
ROPE_BETA_SLOW = 1.0
ROPE_ATTENTION_FACTOR: Optional[float] = None
DEEPNORM_BETA = 1.0
FANIN_GAIN = 0.88

from loomformer_model import state as model_state

model_state.bind(sys.modules[__name__])


_checkpoint_anchor_override = model_state.checkpoint_anchor_override
_activation_checkpoint_recompute_context = (
    model_state.activation_checkpoint_recompute_context
)


def apply_config(cfg: Config) -> None:
    global N, N_Q_HEADS, N_KV_HEADS, HIDDEN, LAYERS, VOCAB, SEQ_LEN
    global HEAD_DIM, GQA_GROUP_SIZE, KV_DIM, HIDDEN_PER_Q_HEAD, IMAG_IN, PHASE_SECTORS, ATTN_IMPL, ATTN_LAYERS, ATTN_SDPA_COMPUTE_DTYPE, ATTN_SDPA_VALUE_FUSION, ATTN_SDPA_RECOMPUTE_BACKWARD, RESIDUAL_INIT, DEPTH_ATTN_READOUT, DEPTH_ATTN_QKV_RMS, RESIDUAL_BRANCH_RMS_CAP, ACTIVATION, POWLU_M, PHASE_GRAD_FLOOR, PHASE_GRAD_MODE, USE_CUDA_PHASE_SIN, USE_CUDA_BETA_SPACE, USE_CUDA_PVPOWLU, USE_CUDA_DEPTH_ATTN, AMP_DTYPE, GRAD_CHECKPOINTING, TRIA_CARRY_ENABLED, TRIA_GAMMA_MAX, TRIA_RAW_GAMMA_INIT, TRIA_TEMPORAL_ENABLED, TIED_EMBEDDINGS, PARAPLEX_GATE_PROJ, FINAL_NORM_ENABLED, FUSED_LINEAR_CE, FUSED_LINEAR_CE_CHUNK_SIZE
    global ROPE_THETA, ROPE_FACTOR, ROPE_ORIGINAL_SEQ_LEN, ROPE_BETA_FAST, ROPE_BETA_SLOW, ROPE_ATTENTION_FACTOR

    N_Q_HEADS = int(cfg.n_q_heads)
    LAYERS = int(cfg.layers)
    VOCAB = int(cfg.vocab)
    SEQ_LEN = int(cfg.seq_len)
    TIED_EMBEDDINGS = bool(getattr(cfg, "tied_embeddings", True))
    GRAD_CHECKPOINTING = bool(getattr(cfg, "grad_checkpointing", False))
    TRIA_CARRY_ENABLED = bool(getattr(cfg, "tria_carry_enabled", False))
    PARAPLEX_GATE_PROJ = bool(getattr(cfg, "paraplex_gate_proj", False))
    FINAL_NORM_ENABLED = bool(getattr(cfg, "final_norm", False))
    FUSED_LINEAR_CE = bool(getattr(cfg, "fused_linear_ce", False))
    FUSED_LINEAR_CE_CHUNK_SIZE = int(getattr(cfg, "fused_linear_ce_chunk_size", 0))  # 0 = auto
    TRIA_GAMMA_MAX = float(getattr(cfg, "tria_gamma_max", 0.25))
    TRIA_RAW_GAMMA_INIT = float(getattr(cfg, "tria_raw_gamma_init", 0.0))
    TRIA_TEMPORAL_ENABLED = bool(getattr(cfg, "tria_temporal_enabled", True))
    if TRIA_CARRY_ENABLED and not TRIA_TEMPORAL_ENABLED:
        raise ValueError("tria_carry_enabled requires tria_temporal_enabled")
    cfg.use_cuda_tria = bool(getattr(cfg, "use_cuda_tria", False))
    tria.set_cuda_tria_enabled(bool(cfg.use_cuda_tria))

    if N_Q_HEADS <= 0 or LAYERS <= 0 or VOCAB <= 0 or SEQ_LEN <= 0:
        raise ValueError("model/data dimensions must be positive")
    raw_attn_layers = getattr(cfg, "attn_layers", None)
    if raw_attn_layers is None:
        cfg.attn_layers = list(range(1, LAYERS + 1))
    else:
        layers = [int(layer) for layer in raw_attn_layers]
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("attn_layers must be a non-empty list without duplicates")
        if layers != sorted(layers) or layers[0] != 1 or layers[-1] > LAYERS:
            raise ValueError(f"attn_layers must be sorted, include 1, and stay within 1..{LAYERS}")
        cfg.attn_layers = layers
    ATTN_LAYERS = tuple(cfg.attn_layers)
    cfg.attn_token_stride = int(getattr(cfg, "attn_token_stride", 1))
    if cfg.attn_token_stride <= 0:
        raise ValueError("attn_token_stride must be positive")
    cfg.attn_token_schedule = str(
        getattr(cfg, "attn_token_schedule", "shared") or "shared").lower()
    if cfg.attn_token_schedule not in ("shared", "staggered"):
        raise ValueError("attn_token_schedule must be 'shared' or 'staggered'")
    if int(cfg.prefetch_batches) <= 0 or int(cfg.gpu_prefetch_batches) <= 0:
        raise ValueError("prefetch_batches and gpu_prefetch_batches must be positive")

    # d_model derivation from model_dim and/or head_dim.
    explicit_dim = cfg.model_dim
    explicit_head_dim = cfg.head_dim
    if explicit_head_dim is not None:
        HEAD_DIM = int(explicit_head_dim)
        if HEAD_DIM <= 0:
            raise ValueError("head_dim must be positive")
        N = N_Q_HEADS * HEAD_DIM
        if explicit_dim is not None and int(explicit_dim) != N:
            raise ValueError(
                f"inconsistent shape: model_dim={int(explicit_dim)} but "
                f"n_q_heads*head_dim={N_Q_HEADS}*{HEAD_DIM}={N}"
            )
    else:
        if explicit_dim is None:
            # Last-resort default: small-ish but not degenerate. For real LM configs, set either
            # model_dim or head_dim explicitly.
            HEAD_DIM = 32
            N = N_Q_HEADS * HEAD_DIM
        else:
            N = int(explicit_dim)
            if N <= 0:
                raise ValueError("model_dim must be positive")
            if N % N_Q_HEADS != 0:
                raise ValueError("model_dim must be divisible by n_q_heads, or set head_dim")
            HEAD_DIM = N // N_Q_HEADS

    if HEAD_DIM % 4 != 0:
        warnings.warn(
            f"HEAD_DIM={HEAD_DIM} is not divisible by 4; the optimized warp-per-row "
            "depth_attn CUDA kernel will be disabled and the slower block-per-row fallback "
            "will be used. Set head_dim to a multiple of 4 for the fast path.",
            RuntimeWarning,
            stacklevel=2,
        )

    # GQA derivation. group means how many query heads share one KV head.
    if cfg.gqa_group_size is not None and cfg.n_kv_heads is not None:
        GQA_GROUP_SIZE = int(cfg.gqa_group_size)
        N_KV_HEADS = int(cfg.n_kv_heads)
        if N_KV_HEADS * GQA_GROUP_SIZE != N_Q_HEADS:
            raise ValueError(
                f"inconsistent GQA: n_kv_heads*group={N_KV_HEADS}*{GQA_GROUP_SIZE} "
                f"!= n_q_heads={N_Q_HEADS}"
            )
    elif cfg.gqa_group_size is not None:
        GQA_GROUP_SIZE = int(cfg.gqa_group_size)
        if GQA_GROUP_SIZE <= 0 or N_Q_HEADS % GQA_GROUP_SIZE != 0:
            raise ValueError("gqa_group_size must divide n_q_heads")
        N_KV_HEADS = N_Q_HEADS // GQA_GROUP_SIZE
    elif cfg.n_kv_heads is not None:
        N_KV_HEADS = int(cfg.n_kv_heads)
        if N_KV_HEADS <= 0 or N_Q_HEADS % N_KV_HEADS != 0:
            raise ValueError("n_kv_heads must divide n_q_heads")
        GQA_GROUP_SIZE = N_Q_HEADS // N_KV_HEADS
    else:
        # Small default matching the common GQA pattern: several Q heads per KV head.
        GQA_GROUP_SIZE = 4 if N_Q_HEADS % 4 == 0 else (2 if N_Q_HEADS % 2 == 0 else 1)
        N_KV_HEADS = N_Q_HEADS // GQA_GROUP_SIZE

    if cfg.hidden is None:
        mult = 4.0 if cfg.hidden_mult is None else float(cfg.hidden_mult)
        raw_hidden = max(N_Q_HEADS, int(round(N * mult)))
        HIDDEN = ((raw_hidden + N_Q_HEADS - 1) // N_Q_HEADS) * N_Q_HEADS
    else:
        HIDDEN = int(cfg.hidden)
    if HIDDEN <= 0:
        raise ValueError("hidden must be positive")
    if HIDDEN % N_Q_HEADS != 0:
        raise ValueError("hidden must be divisible by n_q_heads")

    KV_DIM = N_KV_HEADS * HEAD_DIM
    HIDDEN_PER_Q_HEAD = HIDDEN // N_Q_HEADS
    if HEAD_DIM % 2 != 0:
        raise ValueError("head_dim must be even for rotary attention")
    raw_rope_theta = getattr(cfg, "rope_theta", 10000.0)
    raw_rope_factor = getattr(cfg, "rope_factor", 4.0)
    raw_rope_original = getattr(cfg, "rope_original_seq_len", None)
    raw_rope_beta_fast = getattr(cfg, "rope_beta_fast", 32.0)
    raw_rope_beta_slow = getattr(cfg, "rope_beta_slow", 1.0)
    raw_rope_attn = getattr(cfg, "rope_attention_factor", None)
    ROPE_THETA = 10000.0 if raw_rope_theta is None else float(raw_rope_theta)
    ROPE_FACTOR = 4.0 if raw_rope_factor is None else float(raw_rope_factor)
    ROPE_ORIGINAL_SEQ_LEN = SEQ_LEN if raw_rope_original is None else int(raw_rope_original)
    ROPE_BETA_FAST = 32.0 if raw_rope_beta_fast is None else float(raw_rope_beta_fast)
    ROPE_BETA_SLOW = 1.0 if raw_rope_beta_slow is None else float(raw_rope_beta_slow)
    ROPE_ATTENTION_FACTOR = None if raw_rope_attn is None else float(raw_rope_attn)
    if ROPE_THETA <= 0.0 or ROPE_FACTOR <= 0.0 or ROPE_ORIGINAL_SEQ_LEN <= 0:
        raise ValueError("rope_theta, rope_factor and rope_original_seq_len must be positive")
    if ROPE_BETA_FAST <= 0.0 or ROPE_BETA_SLOW <= 0.0:
        raise ValueError("rope_beta_fast and rope_beta_slow must be positive")
    cfg.rope_theta = ROPE_THETA
    cfg.rope_factor = ROPE_FACTOR
    cfg.rope_original_seq_len = ROPE_ORIGINAL_SEQ_LEN
    cfg.rope_beta_fast = ROPE_BETA_FAST
    cfg.rope_beta_slow = ROPE_BETA_SLOW
    cfg.rope_attention_factor = ROPE_ATTENTION_FACTOR
    RESIDUAL_INIT = str(getattr(cfg, "residual_init", "beta") or "beta").lower()
    if RESIDUAL_INIT not in ("beta", "fanin"):
        raise ValueError(f"residual_init must be 'beta' or 'fanin', got {RESIDUAL_INIT!r}")
    DEPTH_ATTN_READOUT = str(getattr(cfg, "depth_attn_readout", "shared") or "shared").lower()
    if DEPTH_ATTN_READOUT not in ("shared", "per-sublayer"):
        raise ValueError(
            "depth_attn_readout must be 'shared' or 'per-sublayer', "
            f"got {DEPTH_ATTN_READOUT!r}")
    DEPTH_ATTN_QKV_RMS = bool(getattr(cfg, "depth_attn_qkv_rms", False))
    raw_branch_cap = getattr(cfg, "residual_branch_rms_cap", None)
    RESIDUAL_BRANCH_RMS_CAP = None if raw_branch_cap is None else float(raw_branch_cap)
    if RESIDUAL_BRANCH_RMS_CAP is not None and RESIDUAL_BRANCH_RMS_CAP <= 0.0:
        raise ValueError("residual_branch_rms_cap must be positive or null")
    ACTIVATION = str(getattr(cfg, "activation", "gelu") or "gelu").lower()
    if ACTIVATION not in ("gelu", "powlu", "pvpowlu"):
        raise ValueError(f"activation must be 'gelu', 'powlu' or 'pvpowlu', got {ACTIVATION!r}")
    POWLU_M = float(getattr(cfg, "powlu_m", 3.0) or 3.0)
    PHASE_GRAD_FLOOR = float(getattr(cfg, "phase_grad_floor", 0.05) or 0.0)
    PHASE_GRAD_MODE = str(getattr(cfg, "phase_grad_mode", "floor") or "floor").lower()
    if PHASE_GRAD_MODE not in ("floor", "secant"):
        raise ValueError(f"phase_grad_mode must be 'floor' or 'secant', got {PHASE_GRAD_MODE!r}")
    USE_CUDA_PHASE_SIN = bool(getattr(cfg, "use_cuda_phase_sin", True))
    USE_CUDA_BETA_SPACE = bool(getattr(cfg, "use_cuda_beta_space", True))
    USE_CUDA_PVPOWLU = bool(getattr(cfg, "use_cuda_pvpowlu", True))
    USE_CUDA_DEPTH_ATTN = bool(getattr(cfg, "use_cuda_depth_attn", True))
    AMP_DTYPE = str(getattr(cfg, "amp_dtype", "fp32") or "fp32").lower()
    if AMP_DTYPE in ("float32",):
        AMP_DTYPE = "fp32"
    elif AMP_DTYPE in ("bfloat16",):
        AMP_DTYPE = "bf16"
    elif AMP_DTYPE in ("float16", "half"):
        AMP_DTYPE = "fp16"
    elif AMP_DTYPE in ("none", "false", "0", "no"):
        AMP_DTYPE = "off"
    if AMP_DTYPE not in ("bf16", "fp32", "fp16", "off"):
        raise ValueError(f"amp_dtype must be bf16, fp32/off, or fp16, got {AMP_DTYPE!r}")
    cfg.amp_dtype = AMP_DTYPE
    ATTN_IMPL = str(getattr(cfg, "attn_impl", "auto") or "auto").lower()
    if ATTN_IMPL not in ("auto", "flash", "sdpa", "manual"):
        raise ValueError(
            f"attn_impl must be 'auto', 'flash', 'sdpa' or 'manual', got {ATTN_IMPL!r}")
    ATTN_SDPA_COMPUTE_DTYPE = str(getattr(cfg, "attn_sdpa_compute_dtype", "auto") or "auto").lower()
    if ATTN_SDPA_COMPUTE_DTYPE in ("none", "native"):
        ATTN_SDPA_COMPUTE_DTYPE = "model"
    elif ATTN_SDPA_COMPUTE_DTYPE in ("float32",):
        ATTN_SDPA_COMPUTE_DTYPE = "fp32"
    elif ATTN_SDPA_COMPUTE_DTYPE in ("float16", "half"):
        ATTN_SDPA_COMPUTE_DTYPE = "fp16"
    elif ATTN_SDPA_COMPUTE_DTYPE in ("bfloat16",):
        ATTN_SDPA_COMPUTE_DTYPE = "bf16"
    if ATTN_SDPA_COMPUTE_DTYPE not in ("auto", "model", "fp32", "fp16", "bf16"):
        raise ValueError(
            "attn_sdpa_compute_dtype must be auto, model, fp32, fp16, or bf16; "
            f"got {ATTN_SDPA_COMPUTE_DTYPE!r}")
    cfg.attn_impl = ATTN_IMPL
    cfg.attn_sdpa_compute_dtype = ATTN_SDPA_COMPUTE_DTYPE
    ATTN_SDPA_VALUE_FUSION = bool(getattr(cfg, "attn_sdpa_value_fusion", True))
    ATTN_SDPA_RECOMPUTE_BACKWARD = bool(getattr(cfg, "attn_sdpa_recompute_backward", True))
    PHASE_SECTORS = str(getattr(cfg, "phase_sectors", "head") or "head").lower()
    if PHASE_SECTORS not in ("head", "open"):
        raise ValueError(f"phase_sectors must be 'head' or 'open', got {PHASE_SECTORS!r}")
    IMAG_IN = (N + 4 * HEAD_DIM) if PHASE_SECTORS == "head" else (4 * N + HEAD_DIM)

    global DEEPNORM_BETA
    n_sub = max(1, LAYERS)
    DEEPNORM_BETA = (8.0 * n_sub) ** -0.25

    cfg.model_dim = N
    cfg.head_dim = HEAD_DIM
    cfg.n_kv_heads = N_KV_HEADS
    cfg.gqa_group_size = GQA_GROUP_SIZE
    cfg.hidden = HIDDEN

    tria_alpha = float(cfg.tria_carrier_alpha)
    if not math.isfinite(tria_alpha) or tria_alpha <= 0.0:
        raise ValueError(
            f"tria_carrier_alpha must be finite and > 0, got {tria_alpha}")
    cfg.tria_carrier_alpha = tria_alpha
    tria_beta = float(cfg.tria_polarm_beta)
    if not math.isfinite(tria_beta) or tria_beta < 0.0 or tria_beta >= 1.0:
        raise ValueError(
            f"tria_polarm_beta must be finite and in [0, 1), got {tria_beta}")
    cfg.tria_polarm_beta = tria_beta

    if TRIA_CARRY_ENABLED and TRIA_TEMPORAL_ENABLED:
        apply_temporal_tria_auto_calibration(cfg)
    selected_window = getattr(cfg, "tria_temporal_window", None)
    if TRIA_CARRY_ENABLED and selected_window is None:
        raise ValueError(
            "tria_temporal_auto=false requires tria_temporal_window; "
            "auto mode selects it during startup calibration"
        )
    resolved_window = int(selected_window or SEQ_LEN)
    if resolved_window <= 0:
        raise ValueError("tria_temporal_window must be positive")
    cfg.tria_temporal_window = resolved_window
    warmup_cuda_kernels()

    if bool(getattr(cfg, "graph", False)):
        import graph_helper
        graph_helper.set_conditionally_required("phase_sin_secant", PHASE_GRAD_MODE == "secant")
        graph_helper.set_conditionally_required("phase_sin", PHASE_GRAD_MODE == "floor")
        # Chunked PT/SFT uses temporal_carry_endpoint(), not the full
        # temporal_carry() scan (the latter belongs to the flat/calibration
        # path). Do not claim that the active chunked graph should capture a
        # kernel it never calls.
        graph_helper.set_conditionally_required("temporal_carry", False)
        shadowed_by_fused = {"depth_attn"}
        if (
            ACTIVATION == "pvpowlu"
            and USE_CUDA_BETA_SPACE
            and USE_CUDA_PHASE_SIN
            and USE_CUDA_PVPOWLU
            and _try_load_cuda_beta_space() is not None
            and _try_load_cuda_paraplex() is not None
        ):
            # _ParaplexFused is the production PT/SFT path and subsumes these
            # decomposed fallback kernels in one forward/backward pair.
            shadowed_by_fused.update(
                {"phase_sin", "phase_sin_secant", "pvpowlu", "beta_space"}
            )
        graph_helper.set_shadowed_by_fused(shadowed_by_fused)
        # Depth replay is a Python/TLS side effect: record_depth_replay()
        # establishes ordering consumed later by Tria backward. Dynamo cannot
        # represent that hidden state and may otherwise run a registered Tria
        # op without its preceding tape record. Keep the replay-sensitive
        # temporal stack eager for both normal and checkpointed training.
        replay_stack_is_eager = bool(
            getattr(cfg, "compile", False)
            and TRIA_CARRY_ENABLED
            and TRIA_TEMPORAL_ENABLED
        )
        runtime_not_required = set()
        if not TRIA_CARRY_ENABLED:
            runtime_not_required.update(
                {
                    "tria_init", "tria_init_gate", "tria_step",
                    "tria_step_gate", "gate_slot_mix", "slot_attention_pool",
                    "final_ca_sparse", "temporal_carry",
                }
            )
        if replay_stack_is_eager:
            runtime_not_required.update({
                "tria_init", "tria_init_gate", "tria_step", "tria_step_gate",
                "gate_slot_mix", "temporal_carry", "phase_sin",
                "phase_sin_secant", "pvpowlu", "depth_attn", "beta_space",
            })
        graph_helper.set_runtime_not_required(runtime_not_required)
        graph_helper.install_capture_hooks(sys.modules[__name__], tria)


def warmup_cuda_kernels() -> None:
    if USE_CUDA_PHASE_SIN:
        _try_load_cuda_phase_sin()
    if USE_CUDA_PVPOWLU:
        _try_load_cuda_pvpowlu()
    if USE_CUDA_DEPTH_ATTN:
        _try_load_cuda_depth_attn()
    if USE_CUDA_BETA_SPACE:
        _try_load_cuda_beta_space()
    if tria.cuda_tria_enabled():
        tria._try_load_cuda_tria()
    # Keep every extension status line ahead of the architecture report.
    # Otherwise paraplex is first requested by the compile warmup and its
    # final "16/16" line appears later in the startup log.
    if USE_CUDA_BETA_SPACE:
        _try_load_cuda_paraplex()
    # Temporal packed attention needs this one-launch history packer before
    # calling the external varlen FA/TE backend. Build it during startup so
    # kernel progress reaches N/N and any compiler error is reported before
    # model/torch.compile warmup rather than lazily in the first train step.
    if (
        torch.cuda.is_available()
        and TRIA_CARRY_ENABLED
        and TRIA_TEMPORAL_ENABLED
    ):
        _try_load_cuda_packed_gather()

from loomformer_runtime.tokenization import (
    ByteTokenizer,
    BPETokenizerWrap,
    DEFAULT_SPECIAL_TOKENS,
    _tok_special_id,
    ChatTemplate as ChatTemplate,
)

def train_tokenizer(
    raw_dir: str,
    vocab_size: int,
    out: str,
    special_tokens: Optional[List[str]] = None,
    dataset_format: str = "auto",
    text_field: str = "text",
) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    fmt = str(dataset_format or "auto").lower()
    corpus = RawCorpus(raw_dir, fmt=fmt, text_field=text_field)

    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = list(special_tokens) if special_tokens is not None else list(DEFAULT_SPECIAL_TOKENS)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=specials,
    )
    tk.train_from_iterator(corpus.iter_texts(), trainer=trainer, length=len(corpus))
    tk.save(out)
    print(
        f"[train-tokenizer] vocab={tk.get_vocab_size()} format={corpus.fmt} "
        f"docs={len(corpus)} text_field={text_field!r} special_tokens={specials} -> {out}"
    )


def build_tokenizer(cfg: Config):
    global CARRY_TOKEN_ID
    if cfg.tokenizer:
        if cfg.tokenizer.endswith(".json") and os.path.exists(cfg.tokenizer):
            tok = BPETokenizerWrap.load(cfg.tokenizer)
            cfg.vocab = tok.vocab_size
            CARRY_TOKEN_ID = _tok_special_id(tok, "<CARRY>")
            return tok
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
        cfg.vocab = tok.vocab_size
        CARRY_TOKEN_ID = _tok_special_id(tok, "<CARRY>")
        return tok
    cfg.vocab = ByteTokenizer.vocab_size
    CARRY_TOKEN_ID = None
    return ByteTokenizer()

from loomformer_runtime.data import (
    TokenStream,
    RawCorpus,
    _split_count as _split_count,
    maybe_auto_val_split,
    _encode_batch_any as _encode_batch_any,
    ShardStream,
    is_sft_dataset,
)

def make_stream(path: str, cfg: Config, device: torch.device):
    fmt = str(getattr(cfg, "dataset_format", "auto") or "auto").lower()
    if fmt == "sft":
        import loomsft  # lazy: only SFT runs need the chat template/pyarrow path
        return loomsft.make_stream(path, cfg, device)
    if fmt == "bin" or (fmt == "auto" and os.path.isfile(path) and path.endswith(".bin")):
        bos_id = _tok_special_id(build_tokenizer(cfg), "<bos>")
        return TokenStream(path, cfg, device, bos_id=bos_id)
    return ShardStream(path, cfg, device, tokenizer=build_tokenizer(cfg))


IGNORE_INDEX = -100  # cross-entropy target for positions that must not be learned


from loomformer_runtime.layouts import (
    PackedAttentionLayout,
    PackedChunkLayout,
    packed_layout_from_segment_ids,
    _unpacked_attention_layout,
    build_packed_chunk_layout,
    temporal_chunk_stops,
)

def ensure_temporal_chunk_plans(
    idx: torch.Tensor,
    layout: Optional[PackedAttentionLayout],
    cfg: Optional[Config],
) -> Optional[PackedAttentionLayout]:
    """Build data-dependent gather metadata before entering compiled Model.

    The output lengths of ``nonzero``/``bincount`` in the plan builder depend
    on document boundaries. They are metadata, not model compute, and Inductor
    cannot safely specialize them across rank-local batches. SFT streams
    already supply these plans from their CPU prefetch path; PT and legacy
    callers are completed here, outside ``torch.compile``.
    """
    if cfg is None or not (
        bool(getattr(cfg, "tria_carry_enabled", False))
        and bool(getattr(cfg, "tria_temporal_enabled", True))
    ):
        return layout
    if layout is None:
        layout = _unpacked_attention_layout(
            int(idx.shape[0]), int(idx.shape[1]), idx.device
        )
    if layout.chunk_plans:
        return layout
    window = int(getattr(cfg, "tria_temporal_window", 0) or 0)
    if window <= 0:
        raise ValueError(
            "tria_temporal_window must be resolved before building chunk plans"
        )
    # This executes before Model/torch.compile, so preserve the exact eager
    # schedule including data-dependent <CARRY> boundaries.
    stops = temporal_chunk_stops(
        idx,
        window,
        True,
        CARRY_TOKEN_ID,
        compiling=False,
    )
    fire_positions = set(range(window - 1, int(idx.shape[1]) - 1, window))
    if CARRY_TOKEN_ID is not None:
        fire_positions.update(
            idx.eq(int(CARRY_TOKEN_ID))
            .any(dim=0)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
    ranges: List[Tuple[int, int]] = []
    plans: List[PackedChunkLayout] = []
    start = 0
    for stop in stops:
        end = min(int(stop) + 1, int(idx.shape[1]))
        if end <= start:
            continue
        ranges.append((start, end))
        plans.append(
            build_packed_chunk_layout(
                layout,
                start,
                end,
                tuple(ranges),
                ends_with_fire=(end - 1) in fire_positions,
            )
        )
        start = end
    if start != int(idx.shape[1]):
        raise RuntimeError(
            f"temporal chunk plan stopped at {start}, "
            f"expected {int(idx.shape[1])}"
        )
    return replace(layout, chunk_plans=tuple(plans))


def split_train_batch(
    batch,
    eos_id: Optional[int],
    cfg: Optional[Config] = None,
):
    """Return (x, y, position_ids, attention_layout) from a stream batch.

    Pretraining streams yield a plain [B, T+1] token tensor and every next token
    is a target. SFT streams yield (ids, loss_mask, packed_layout); the
    masked-out positions become IGNORE_INDEX, while CPU-produced cu-seqlens and
    chunk gather plans avoid per-step GPU synchronization/sorting. A quadratic
    block-diagonal mask is never materialized.
    """
    packed_max_seqlen = None
    supplied_layout = None
    if isinstance(batch, tuple):
        if len(batch) == 2:
            ids, loss_mask = batch
        elif len(batch) == 3:
            ids, loss_mask, metadata = batch
            if (
                isinstance(metadata, PackedAttentionLayout)
                or (
                    hasattr(metadata, "segment_ids")
                    and hasattr(metadata, "cu_seqlens")
                    and hasattr(metadata, "position_ids")
                )
            ):
                supplied_layout = metadata
            else:
                packed_max_seqlen = metadata
        else:
            raise ValueError(f"unexpected SFT batch tuple of length {len(batch)}")
        x, y = ids[:, :-1], ids[:, 1:]
        y = torch.where(loss_mask[:, 1:].bool(), y, torch.full_like(y, IGNORE_INDEX))
    else:
        x, y = batch[:, :-1], batch[:, 1:]
    if supplied_layout is not None:
        if supplied_layout.segment_ids.shape != x.shape:
            raise ValueError(
                f"supplied packed layout shape {tuple(supplied_layout.segment_ids.shape)} "
                f"does not match training input {tuple(x.shape)}")
        position_ids = supplied_layout.position_ids.to(dtype=torch.long)
        attn_mask = supplied_layout
    else:
        position_ids, attn_mask = build_doc_reset_state(
            x, eos_id, max_seqlen=packed_max_seqlen)
    attn_mask = ensure_temporal_chunk_plans(x, attn_mask, cfg)
    return x, y, position_ids, attn_mask


def build_doc_reset_state(
    x: torch.Tensor,
    eos_id: Optional[int],
    max_seqlen: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[PackedAttentionLayout]]:
    B, T = x.shape
    idx = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, T)
    if eos_id is None:
        return idx, None
    boundary = (x == eos_id).long()
    seg = torch.cumsum(boundary, dim=1) - boundary  # exclusive cumsum: eos stays in its own (old) segment
    new_seg = torch.ones_like(seg, dtype=torch.bool)
    new_seg[:, 1:] = seg[:, 1:] != seg[:, :-1]
    seg_start_idx = torch.cummax(torch.where(new_seg, idx, torch.zeros_like(idx)), dim=1).values
    position_ids = idx - seg_start_idx
    return position_ids, packed_layout_from_segment_ids(
        seg.to(torch.int32), max_seqlen=max_seqlen, position_ids=position_ids)


def prepare(raw_dir: str, cfg: Config, out: str) -> None:
    build_tokenizer(cfg)  # sets cfg.vocab as a side effect, same as before
    arr, bpt = _tokenize_raw_corpus_full(raw_dir, cfg)
    arr.tofile(out)
    total_bytes = round(bpt * len(arr))
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"tokens": len(arr), "bytes": total_bytes, "bytes_per_token": bpt, "vocab": cfg.vocab}, f)
    print(f"[prepare] {len(arr)} tokens, {total_bytes} bytes, {bpt:.3f} bytes/token -> {out} ({cfg.summary()})")


# ============================================================================
# paraplex/GQA model
# ============================================================================


from loomformer_model.paraplex import (
    _PhaseSinSecantCUDA as _PhaseSinSecantCUDA,
    _try_load_cuda_phase_sin,
    _PhaseSinFloorCUDA as _PhaseSinFloorCUDA,
    _try_load_cuda_pvpowlu,
    _PvPowluCUDA as _PvPowluCUDA,
    _try_load_cuda_beta_space,
    _try_load_cuda_paraplex,
    _BetaSpaceDirect as _BetaSpaceDirect,
    _ParaplexFused as _ParaplexFused,
    ParaplexFFN as ParaplexFFN,
)

from loomformer_model.primitives import (
    _fused_linear_cross_entropy_eager as _fused_linear_cross_entropy_eager,
)

from loomformer_model.attentions.depthattn import (
    _try_load_cuda_depth_attn,
    _DepthAttnOnlineFused as _DepthAttnOnlineFused,
    DepthAttn as DepthAttn,
)

from loomformer_model.types import (
    LayerCache as LayerCache,
    InferenceKVRuntime as InferenceKVRuntime,
)

from loomformer_model.attentions.attention_backends import (
    _flash_backend_cache as _flash_backend_cache,
    _probe_flash_value_fusion as _probe_flash_value_fusion,
    _probe_te_value_fusion as _probe_te_value_fusion,
    _te_backend_cache as _te_backend_cache,
    _try_load_cuda_packed_gather,
    _varlen_backend_failure_detail as _varlen_backend_failure_detail,
    _PackedGather as _PackedGather,
    _PackedGatherPair as _PackedGatherPair,
    _pack_selected_chunk_history as _pack_selected_chunk_history,
    _pack_selected_chunk_kv as _pack_selected_chunk_kv,
)

from loomformer_model.attentions.attention_dense import (
    GroupedQueryCausalSelfAttention as GroupedQueryCausalSelfAttention,
)

from loomformer_model.attentions.attention_sparse import (
    SelectedTokenLayout as SelectedTokenLayout,
    StridedChunkLayout as StridedChunkLayout,
    StridedGroupedQueryCausalSelfAttention as StridedGroupedQueryCausalSelfAttention,
    InheritedContextMixer as InheritedContextMixer,
)

from loomformer_model.attentions.attention_rope import (
    YaRNRotaryEmbedding as YaRNRotaryEmbedding,
)

from loomformer_model.block import Block as Block
from loomformer_model.model import Model as Model

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# train / eval / infer
# ============================================================================


def maybe_compile(model: nn.Module, device: torch.device, enabled: bool = True) -> nn.Module:
    if not enabled:
        return model
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability()
        if hasattr(torch, "compile") and major >= 7:
            # Internal PyTorch 2.5 deprecation, fixed upstream in newer
            # releases; unrelated to model code or graph correctness.
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\._prims_common\.check` is deprecated.*",
                category=FutureWarning,
                module=r"torch\._inductor\.lowering",
            )
            return torch.compile(model, dynamic=None, fullgraph=False)
    return model


async def eval_loss_async(model: nn.Module, stream: TokenStream, cfg: Config, device: torch.device,
                            eos_id: Optional[int] = None) -> float:
    model.eval()
    loop = asyncio.get_event_loop()
    n = max(1, cfg.eval_batches)
    raw_batches = await asyncio.gather(*[loop.run_in_executor(None, stream.sample_device_batch) for _ in range(n)])
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for b in raw_batches:
            x, y, position_ids, attn_mask = split_train_batch(
                b, eos_id, cfg
            )
            ntok = int((y != IGNORE_INDEX).sum().item())
            if ntok == 0:
                continue
            ddp_sync_mutable_buffers(model)
            with amp_autocast(device):
                loss = model(x, attn_mask=attn_mask, position_ids=position_ids, labels=y)
            total_nll += float(loss.item()) * ntok
            total_tokens += ntok
            raw = ddp_unwrap_model(model)
            raw.last_tria_depth_carry = None
            raw.last_tria_document_carry_stats = None
    model.train()
    return ddp_weighted_mean(total_nll, total_tokens, device)[0]


from loomformer_runtime.reporting import (
    lr_at,
    load_bytes_per_token,
    loss_to_bits,
    format_big_int,
    format_eta_hours_minutes,
    format_train_status,
    format_eval_status,
)

def _is_prepared_token_dataset(dataset: str, cfg: Optional["Config"] = None) -> bool:
    fmt = str(getattr(cfg, "dataset_format", "auto") or "auto").lower() if cfg is not None else "auto"
    return fmt == "bin" or (fmt == "auto" and os.path.isfile(dataset) and dataset.endswith(".bin"))


def compute_raw_bytes_per_token_meta(
    dataset: str,
    cfg: "Config",
    meta_path: Optional[str] = None,
    max_docs: Optional[int] = None,
) -> Tuple[float, str]:
    meta_path = meta_path or (dataset + ".meta.json")
    if _is_prepared_token_dataset(dataset, cfg):
        return 1.0, meta_path
    tok = build_tokenizer(cfg)
    corpus = RawCorpus(
        dataset,
        fmt=getattr(cfg, "dataset_format", "auto"),
        text_field=getattr(cfg, "text_field", "text"),
    )

    docs_src = corpus._docs
    exact = True
    sample_stride = 1
    if max_docs is not None and max_docs > 0 and len(docs_src) > max_docs:
        exact = False
        sample_stride = max(1, len(docs_src) // int(max_docs))
        docs_src = docs_src[::sample_stride][:int(max_docs)]

    total_bytes = 0
    total_tokens = 0
    docs = 0
    for text in corpus.iter_sampled_texts(docs_src):
        if not text:
            continue
        total_bytes += len(text.encode("utf-8"))
        total_tokens += len(tok.encode(text))
        docs += 1
    if total_tokens <= 0:
        raise ValueError(f"cannot compute bytes/token for empty corpus: {dataset}")
    bpt = total_bytes / max(1, total_tokens)
    payload = {
        "tokens": int(total_tokens),
        "bytes": int(total_bytes),
        "bytes_per_token": float(bpt),
        "vocab": int(cfg.vocab),
        "format": str(corpus.fmt),
        "text_field": str(getattr(cfg, "text_field", "text")),
        "docs": int(docs),
        "total_docs": int(len(corpus)),
        "total_chars": int(corpus.total_chars),
        "exact": bool(exact),
        "sample_stride": int(sample_stride),
        "source": "computed_from_raw_corpus" if exact else "estimated_from_raw_corpus_sample",
    }
    tmp = meta_path + f".tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, meta_path)
    return bpt, meta_path


def ensure_train_bytes_per_token_meta(dataset: str, cfg: "Config", device: torch.device) -> Tuple[float, str, bool]:
    bpt, meta_path, ok = load_bytes_per_token(dataset)
    if ok or _is_prepared_token_dataset(dataset, cfg):
        return bpt, meta_path, ok
    max_docs = 4096
    if ddp_is_main():
        ddp_print(
            f"[meta] {meta_path} missing; estimating train bytes/token from "
            f"{dataset} with active tokenizer (sample_docs={max_docs})..."
        )
        bpt, meta_path = compute_raw_bytes_per_token_meta(dataset, cfg, meta_path, max_docs=max_docs)
        try:
            with open(meta_path, encoding="utf-8") as _f:
                _exact = bool(json.load(_f).get("exact", False))
        except Exception:
            _exact = False
        ddp_print(f"[meta] wrote {meta_path} bytes/token={bpt:.6f} exact={str(_exact).lower()}")
    ddp_barrier(device)
    return load_bytes_per_token(dataset)


def ensure_eval_bytes_per_token_meta(dataset: str, cfg: "Config", device: torch.device) -> Tuple[float, str, bool]:
    bpt, meta_path, ok = load_bytes_per_token(dataset)
    if ok or _is_prepared_token_dataset(dataset, cfg):
        return bpt, meta_path, ok
    if ddp_is_main():
        ddp_print(f"[meta] {meta_path} missing; computing eval bytes/token from {dataset} with active tokenizer...")
        bpt, meta_path = compute_raw_bytes_per_token_meta(dataset, cfg, meta_path, max_docs=None)
        ddp_print(f"[meta] wrote {meta_path} bytes/token={bpt:.6f} exact=true")
    ddp_barrier(device)
    return load_bytes_per_token(dataset)


def _fast_parquet_token_count(dataset: str, cfg: "Config", tok, sample_docs: int = 50) -> Optional[int]:
    meta_path = dataset + ".meta.json"
    try:
        with open(meta_path, encoding="utf-8") as f:
            total_chars = int(json.load(f).get("total_chars", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if total_chars <= 0:
        return None

    import pyarrow.parquet as pq

    files = RawCorpus._resolve_files(dataset, "parquet")
    texts: List[str] = []
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=sample_docs - len(texts), columns=[cfg.text_field]):
            texts.extend(text for text in batch.column(0).to_pylist() if text)
            if len(texts) >= sample_docs:
                break
        if len(texts) >= sample_docs:
            break
    if not texts:
        return None

    sample_chars = sum(len(text) for text in texts)
    sample_tokens = sum(len(tok.encode(text)) for text in texts)
    if sample_chars <= 0 or sample_tokens <= 0:
        return None
    return int(total_chars * sample_tokens / sample_chars)


def dataset_token_count(dataset: str, cfg: Optional["Config"] = None) -> int:
    fmt = str(getattr(cfg, "dataset_format", "auto") or "auto").lower() if cfg is not None else "auto"
    if fmt == "sft":
        import loomsft  # already-built tokenized cache: an exact count, no re-read
        return loomsft.cached_token_count(dataset, cfg)
    is_bin = fmt == "bin" or (fmt == "auto" and os.path.isfile(dataset) and dataset.endswith(".bin"))
    if is_bin:
        nbytes = os.path.getsize(dataset)
        item = np.dtype(np.uint16).itemsize
        if nbytes % item != 0:
            raise ValueError(f"prepared dataset byte size is not uint16-aligned: {dataset} has {nbytes} bytes")
        return nbytes // item
    if cfg is None:
        raise ValueError("dataset_token_count needs cfg to estimate a raw-format corpus")
    tok = build_tokenizer(cfg)
    files = RawCorpus._resolve_files(dataset, fmt)
    if not files:
        raise ValueError(f"no corpus files found at {dataset!r} (format={fmt!r})")
    resolved_fmt = fmt if fmt != "auto" else RawCorpus._infer_format(files[0])
    if resolved_fmt == "parquet":
        fast_count = _fast_parquet_token_count(dataset, cfg, tok)
        if fast_count is not None:
            return fast_count
    corpus = RawCorpus(dataset, fmt=getattr(cfg, "dataset_format", "auto"),
                        text_field=getattr(cfg, "text_field", "text"))
    sample = corpus._docs[:: max(1, len(corpus._docs) // 50)][:50]
    sample_chars = sample_tokens = 0
    for text in corpus.iter_sampled_texts(sample):
        sample_chars += len(text)
        sample_tokens += len(tok.encode(text))
    ratio = sample_chars / max(1, sample_tokens)
    return int(corpus.total_chars / max(1e-6, ratio))



def print_training_budget(
    cfg: Config, dataset: str, start_step: int = 0, tokens_seen: int = 0,
) -> Tuple[int, int, int]:
    data_tokens = dataset_token_count(dataset, cfg)
    global_bs = int(getattr(cfg, "_global_batch_size", cfg.batch_size))
    accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    tokens_per_step = global_bs * int(cfg.seq_len) * accum_steps
    remaining_steps = max(0, int(cfg.steps) - int(start_step))
    run_tokens = remaining_steps * tokens_per_step
    cumulative_target_tokens = max(0, int(tokens_seen)) + run_tokens
    label = "budget" if int(start_step) == 0 else "remaining"
    if data_tokens > 0:
        run_epochs = run_tokens / data_tokens
        suffix = (
            f" ({run_epochs:.3f} epochs of "
            f"{format_big_int(data_tokens)})"
        )
    else:
        suffix = " (streamed SFT; epochs are not defined)"
    print(
        f"  {label:<10}{format_big_int(run_tokens)} tokens over "
        f"{remaining_steps:,} steps{suffix}"
    )
    if int(start_step) > 0:
        print(f"  schedule step {int(start_step):,} -> {int(cfg.steps):,} "
              f"(target is total steps, not additional steps)")
        print(f"  cumulative target {format_big_int(cumulative_target_tokens)} processed tokens")
    return run_tokens, data_tokens, cumulative_target_tokens


def print_training_scale(
    run_tokens: int, data_tokens: int, cumulative_target_tokens: int,
    model: nn.Module,
) -> None:
    params = count_params(model)
    tpp = run_tokens / max(1, params)
    cumulative_tpp = cumulative_target_tokens / max(1, params)
    epoch_tokens_per_param = data_tokens / max(1, params)
    scale = f"{tpp:.1f} tok/param"
    if cumulative_target_tokens != run_tokens:
        scale += f" remaining  ·  {cumulative_tpp:.1f} cumulative tok/param"
    data_scale = (
        f"  ·  {epoch_tokens_per_param:.1f} data-tok/param"
        if data_tokens > 0
        else ""
    )
    print(
        f"           paraplex: {format_big_int(params)} params  ·  "
        f"{scale}{data_scale}"
    )


def canonicalize_model_state_dict(state: dict) -> dict:
    state = dict(state)
    pairs = (
        ("_tria_reader.proj.weight", "tria_agg.reader.proj.weight"),
        ("_tria_reader.proj.bias", "tria_agg.reader.proj.bias"),
        ("_tria_reader.key_proj.weight", "tria_agg.reader.key_proj.weight"),
    )
    for legacy, canonical in pairs:
        legacy_value = state.pop(legacy, None)
        if legacy_value is None:
            continue
        canonical_value = state.get(canonical)
        if canonical_value is not None and not torch.equal(legacy_value, canonical_value):
            raise ValueError(f"checkpoint reader aliases disagree: {legacy!r} != {canonical!r}")
        if canonical_value is None:
            state[canonical] = legacy_value

    for layer in range(LAYERS):
        selector_key = f"blocks.{layer}.ffn.gate_selector.logits"
        selector = state.get(selector_key)
        if selector is not None:
            if selector.shape == (9,):
                state[selector_key] = selector.unsqueeze(0).repeat(N_Q_HEADS, 1)
            elif selector.shape != (N_Q_HEADS, 9):
                raise ValueError(
                    f"invalid selector shape at layer {layer}: {tuple(selector.shape)}")

        prefix = f"blocks.{layer}.attn."
        if layer + 1 not in ATTN_LAYERS:
            for key in tuple(state):
                if key.startswith(prefix):
                    state.pop(key)
            continue
        packed = prefix + "qkv_weight"
        old_keys = [prefix + "q.weight", prefix + "k.weight", prefix + "v.weight"]
        old_values = [state.get(key) for key in old_keys]
        if any(value is not None for value in old_values):
            if not all(value is not None for value in old_values):
                raise ValueError(f"incomplete QKV checkpoint group for layer {layer}")
            merged = torch.cat(old_values, dim=0)
            current = state.get(packed)
            if current is not None and not torch.equal(current, merged):
                raise ValueError(f"packed QKV disagrees with legacy weights at layer {layer}")
            state[packed] = merged
            for key in old_keys:
                state.pop(key)

    depth_keys = ["depth_attn.w_k.weight", "depth_attn.w_v.weight"]
    depth_values = [state.get(key) for key in depth_keys]
    if any(value is not None for value in depth_values):
        if not all(value is not None for value in depth_values):
            raise ValueError("incomplete depth K/V checkpoint group")
        merged = torch.cat(depth_values, dim=0)
        current = state.get("depth_attn.kv_weight")
        if current is not None and not torch.equal(current, merged):
            raise ValueError("packed depth K/V disagrees with legacy weights")
        state["depth_attn.kv_weight"] = merged
        for key in depth_keys:
            state.pop(key)
    return state


def load_model_blob_into(model: nn.Module, blob: dict, ablation: bool) -> None:
    if blob.get("model_kind") != "loomformer":
        raise ValueError(
            "checkpoint kind mismatch: expected model_kind='loomformer', "
            f"got {blob.get('model_kind')!r}"
        )
    if blob.get("ffn_type") != "paraplex":
        raise ValueError(f"checkpoint uses removed FFN type: {blob.get('ffn_type')!r}")
    if bool(blob.get("ablation", False)) != bool(ablation):
        raise ValueError(
            f"checkpoint ablation mismatch: checkpoint={blob.get('ablation', False)!r}, "
            f"model={bool(ablation)!r}"
        )
    model.load_state_dict(canonicalize_model_state_dict(blob["model"]), strict=True)


def load_model_checkpoint(model: nn.Module, path: str, ablation: bool, device: torch.device) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    blob = torch.load(path, map_location=device, weights_only=True)
    load_model_blob_into(model, blob, ablation=ablation)
    ddp_print(f"[resume] loaded paraplex weights from {path}")


def apply_train_lr_overrides(model: nn.Module, cfg: Config) -> Dict[int, float]:
    """Apply ``cfg.train_lr`` freeze/LR overrides and return parameter LR multipliers.

    Block indices are ignored when matching names, and the first matching entry wins.
    Returned values are each entry's absolute LR divided by ``cfg.lr``.
    """
    overrides = getattr(cfg, "train_lr", None)
    if not overrides:
        return {}
    strip_block_prefix = re.compile(r"^blocks\.\d+\.")
    mult_by_id: Dict[int, float] = {}
    matched_count = 0
    for name, param in model.named_parameters():
        canonical = strip_block_prefix.sub("", name)
        for entry in overrides:
            if entry.get("name") == canonical:
                param.requires_grad_(bool(entry.get("train", True)))
                if "lr" in entry:
                    mult_by_id[id(param)] = float(entry["lr"]) / float(cfg.lr)
                matched_count += 1
                break
    total = sum(1 for _ in model.parameters())
    ddp_print(f"[loomcloner] train_lr: {matched_count}/{total} parameters matched an override "
              f"({len(overrides)} entries, {sum(1 for e in overrides if not e.get('train', True))} frozen)")
    return mult_by_id


def optimizer_class_from_name(name: str):
    key = str(name or "adamw").strip().lower()
    if key == "adamw":
        return torch.optim.AdamW, "adamw"
    if key == "atom":
        try:
            from atom.atom import ATOM
        except ImportError:
            try:
                from atom import ATOM
            except ImportError as e:
                raise ImportError(
                    "optimizer: atom requested, but ATOM was not importable. "
                    "Put ViperLLM's atom/ package next to this script or install it on PYTHONPATH."
                ) from e
        return ATOM, "atom"
    raise ValueError(f"unknown optimizer={name!r}; expected 'adamw' or 'atom'")


def load_optimizer_checkpoint(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    path: str,
    optimizer_name: str,
    device: torch.device,
) -> bool:
    """Restore optimizer tensors while keeping active-config group options."""
    blob = torch.load(path, map_location=device, weights_only=True)
    saved_state = blob.get("optimizer")
    if saved_state is None:
        ddp_print(
            f"[resume] WARNING: {path!r} has no optimizer state "
            "(legacy checkpoint); optimizer starts fresh.")
        return False
    saved_name = str(blob.get("optimizer_name", optimizer_name)).strip().lower()
    active_name = str(optimizer_name).strip().lower()
    if saved_name != active_name:
        ddp_print(
            f"[resume] WARNING: checkpoint optimizer={saved_name!r}, "
            f"active config optimizer={active_name!r}; optimizer starts fresh.")
        return False

    selector_ids = {
        id(param) for name, param in model.named_parameters()
        if name.endswith(".gate_selector.logits")
    }
    saved_groups = saved_state.get("param_groups", [])
    if len(saved_groups) == len(optimizer.param_groups):
        for active_group, saved_group in zip(optimizer.param_groups, saved_groups):
            active_params = active_group["params"]
            saved_params = saved_group.get("params", [])
            if len(active_params) != len(saved_params):
                continue
            for param, saved_id in zip(active_params, saved_params):
                if id(param) not in selector_ids or tuple(param.shape) != (N_Q_HEADS, 9):
                    continue
                entry = saved_state.get("state", {}).get(saved_id, {})
                for key, value in tuple(entry.items()):
                    if torch.is_tensor(value) and tuple(value.shape) == (9,):
                        entry[key] = value.unsqueeze(0).repeat(N_Q_HEADS, 1)

    # Tensor history belongs to the checkpoint. LR/WD/lr_mult and other group
    # options belong to the active config and are restored after loading.
    active_group_options = [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]
    try:
        optimizer.load_state_dict(saved_state)
    except ValueError as error:
        raise ValueError(
            "optimizer checkpoint is incompatible with the active trainable "
            "parameter groups; keep train_lr/freeze settings unchanged or "
            "start without --resume"
        ) from error
    if len(optimizer.param_groups) != len(active_group_options):
        raise ValueError(
            "optimizer checkpoint changed the number of active parameter groups")
    for group, active_options in zip(optimizer.param_groups, active_group_options):
        group.update(active_options)
    ddp_print(f"[resume] restored {active_name} optimizer state from {path}")
    return True


from loomformer_runtime.run_control import _GracefulInterrupt, _RunpointWatcher

def _save_compiled_graph(cfg: Config, model_base: nn.Module, device: torch.device, tag: str) -> None:
    os.makedirs("graphs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join("graphs", f"{tag}_{ts}.pt2")

    was_training = model_base.training
    model_base.eval()
    try:
        example_x = torch.randint(0, VOCAB, (int(cfg.batch_size), SEQ_LEN), device=device, dtype=torch.long)
        with torch.no_grad():
            exported = torch.export.export(model_base, (example_x,))
            pkg_path = torch._inductor.aoti_compile_and_package(exported, package_path=out_path)
        ddp_print(f"[save-graph] wrote {pkg_path}")
        ddp_print(f"[save-graph] load with: torch._inductor.aoti_load_package({pkg_path!r})")
    except Exception as e:
        ddp_print(f"[save-graph] export failed ({type(e).__name__}: {e}); training continues without it.")
    finally:
        if was_training:
            model_base.train()


async def train_one_async(
    cfg: Config,
    dataset: str,
    device: torch.device,
    ablation: bool,
    ckpt_out: Optional[str],
    resume_in: Optional[str] = None,
    val_dataset: Optional[str] = None,
    resume_step: Optional[int] = None,
    resume_dataset_steps: Optional[int] = None,
    init_weights: Optional[str] = None,
) -> Dict[str, float]:
    rank = ddp_rank() if ddp_is_distributed() else 0
    set_seed(int(cfg.seed) + 1000003 * int(rank))
    stream = make_stream(dataset, cfg, device)
    start_step = 0
    tokens_seen_at_start = 0
    resume_blob: Dict[str, Any] = {}
    dataset_progress: Dict[str, Dict[str, int]] = {}
    current_dataset_key = dataset_progress_key(dataset)
    current_dataset_steps = 0
    current_dataset_draws = 0
    if resume_in:
        resume_blob = torch.load(resume_in, map_location="cpu", weights_only=True)
        if resume_step is not None:
            start_step = int(resume_step)
        else:
            _ckpt_step = resume_blob.get("step", None)
            if _ckpt_step is None:
                ddp_print(f"[resume] WARNING: {resume_in!r} has no saved 'step' (older checkpoint) -- "
                          f"defaulting to start_step=0. Pass --resume-step N to hard-set it.")
            start_step = int(_ckpt_step or 0)
        if start_step >= int(cfg.steps):
            ddp_print(f"[resume] start_step={start_step} >= cfg.steps={cfg.steps} -- nothing to do, exiting.")
            return {
                "final_eval_loss": float("nan"),
                "best_eval_loss": float("nan"),
                "full_eval_loss": float("nan"),
                "full_eval_bpb": float("nan"),
                "seconds": 0.0,
                "skipped": 1.0,
                "start_step": float(start_step),
            }
        tokens_seen_at_start, tokens_seen_exact = checkpoint_tokens_seen(
            resume_blob, start_step)
        if tokens_seen_exact:
            ddp_print(
                f"[resume] restored cumulative tokens_seen="
                f"{format_big_int(tokens_seen_at_start)}")
        else:
            ddp_print(
                f"[resume] checkpoint has no tokens_seen; estimated "
                f"{format_big_int(tokens_seen_at_start)} processed tokens "
                "from its saved step/batch/sequence config.")
        (
            dataset_progress,
            current_dataset_key,
            current_dataset_steps,
            current_dataset_draws,
            replay_reason,
        ) = resolve_resume_dataset_progress(
            cfg, dataset, resume_blob, start_step, resume_dataset_steps)
        ddp_print(f"[resume] data cursor policy: {replay_reason}.")
        if hasattr(stream, "fast_forward") and current_dataset_draws > 0:
            ddp_print(
                f"[resume] fast-forwarding the packing plan by "
                f"{current_dataset_draws} saved batch draws for "
                f"{current_dataset_key!r}...")
            stream.fast_forward(current_dataset_draws)
            ddp_print("[resume] fast-forward done.")
        elif isinstance(stream, ShardStream) and current_dataset_draws > 0:
            ddp_print(
                f"[resume] fast-forwarding ShardStream RNG by "
                f"{current_dataset_draws} saved batch draws for "
                f"{current_dataset_key!r}...")
            for _ in range(current_dataset_draws):
                stream._sample_batch()
            ddp_print("[resume] fast-forward done.")
        elif isinstance(stream, ShardStream):
            ddp_print("[resume] data stream starts at draw 0 (no fast-forward).")
        ddp_print(f"[resume] continuing from step {start_step}/{cfg.steps} -- "
                  f"LR schedule/log step numbering continue (unchanged if cfg.steps/warmup_steps match the original run).")
    else:
        dataset_progress[current_dataset_key] = {"steps": 0, "draws": 0}
    if hasattr(stream, "prime"):
        await stream.prime()
    eval_dataset = val_dataset or dataset
    eval_stream = stream if os.path.abspath(eval_dataset) == os.path.abspath(dataset) else make_stream(eval_dataset, cfg, device)
    train_eos_id = getattr(stream, "_eos_id", None) if bool(getattr(cfg, "doc_reset_attn", True)) else None
    eval_eos_id = getattr(eval_stream, "_eos_id", None) if bool(getattr(cfg, "doc_reset_attn", True)) else None
    if is_sft_dataset(cfg):
        # SFT rows are rendered chat, not raw corpus bytes: a bytes/token ratio
        # (and the bpb it feeds) has no meaning here, and estimating one would
        # re-read the dataset for nothing.
        train_bpt = eval_bpt = float("nan")
    else:
        train_bpt, _, _ = ensure_train_bytes_per_token_meta(dataset, cfg, device)
        if os.path.abspath(eval_dataset) == os.path.abspath(dataset):
            eval_bpt = train_bpt
        else:
            eval_bpt, _, _ = ensure_eval_bytes_per_token_meta(eval_dataset, cfg, device)

    ddp_barrier(device)
    if ddp_is_main():
        print("[data] all ranks ready", flush=True)
    apply_config(cfg)
    ddp_assert_config_consensus(cfg)
    if resume_in:
        assert_resume_attention_config(cfg, resume_blob.get("cfg", {}))
    print_architecture_report(cfg, device, ablation, dataset, val_dataset)
    budget = None
    if ddp_is_main():
        budget = print_training_budget(
            cfg, dataset, start_step=start_step,
            tokens_seen=tokens_seen_at_start)

    tag = "LoomFormer-ablation-s1" if ablation else "LoomFormer-paraplex"
    model_base = Model(cfg, ablation=ablation)
    if ddp_is_main():
        print_training_scale(*budget, model_base)
    ddp_print("=" * 64)
    model_base = model_base.to(device)
    if resume_in:
        load_model_checkpoint(model_base, resume_in, ablation=ablation, device=device)
    elif init_weights:
        # SFT/continued training: weights only. Optimizer state, step counter and
        # LR schedule all start fresh, unlike --resume.
        load_model_checkpoint(model_base, init_weights, ablation=ablation, device=device)
        ddp_print(f"[init] weights loaded from {init_weights} (fresh optimizer, step 0)")
    if bool(getattr(cfg, "fsdp_full_shard", False)):
        reshaped_scalars = 0
        with torch.no_grad():
            for parameter in model_base.parameters():
                if parameter.ndim == 0:
                    parameter.data = parameter.data.reshape(1)
                    reshaped_scalars += 1
        ddp_print(
            f"[fsdp] represented {reshaped_scalars} scalar parameters as "
            "length-1 tensors for FSDP flattening")
    if bool(getattr(cfg, "save_initial_checkpoint", False)) and resume_in:
        raise ValueError("save_initial_checkpoint requires a fresh run without resume")
    train_lr_by_id = apply_train_lr_overrides(model_base, cfg)

    # Capture and register custom ops before DDP installs reducer hooks on the
    # parameters. This warmup intentionally calls model_base directly. Doing
    # it after DDP construction bypasses DDP.forward/prepare_for_backward while
    # still firing the reducer's parameter hooks, poisoning static-graph
    # reducer state and hanging a later real backward.
    if bool(getattr(cfg, "graph", False)):
        import graph_helper
        _was_training = model_base.training
        model_base.train()
        _MAX_WARMUP_ATTEMPTS = 5
        _capture_bytes_released = 0
        for _attempt in range(1, _MAX_WARMUP_ATTEMPTS + 1):
            _warm_batch = torch.randint(
                0,
                VOCAB,
                (int(cfg.batch_size), SEQ_LEN + 1),
                device=device,
                dtype=torch.long,
            )
            _wx, _wy = _warm_batch[:, :-1], _warm_batch[:, 1:]
            _warm_pos, _warm_mask = build_doc_reset_state(
                _wx, train_eos_id
            )
            _warm_mask = ensure_temporal_chunk_plans(
                _wx, _warm_mask, cfg
            )
            with amp_autocast(device):
                _wloss = model_base(
                    _wx,
                    attn_mask=_warm_mask,
                    position_ids=_warm_pos,
                    labels=_wy,
                )
            _wloss.backward()
            model_base.zero_grad(set_to_none=True)
            _capture_before = graph_helper.captured_tensor_bytes()
            _registered_now = graph_helper.finalize_registration(
                sys.modules[__name__], tria
            )
            _capture_after = graph_helper.captured_tensor_bytes()
            _capture_bytes_released += max(
                0, _capture_before - _capture_after
            )
            del _warm_batch, _wx, _wy, _warm_pos, _warm_mask, _wloss
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if graph_helper.is_finalized():
                break
            if _registered_now == 0:
                # Repeating an identical full-size synthetic step cannot
                # discover a structurally inactive path.
                break
        if not _was_training:
            model_base.eval()
        _registered, _missing, _fallback_only = (
            graph_helper.registration_summary()
        )
        ddp_print(
            f"[custom-ops] registered after {_attempt} warmup attempt(s): "
            f"{', '.join(_registered) or '(none)'}; represented "
            f"{_capture_bytes_released / (1024 ** 2):.1f} MiB via metadata "
            "(0 MiB retained)"
        )
        if _fallback_only:
            ddp_print(
                "[custom-ops] inactive, fused-shadowed, or checkpoint-eager "
                "in this config: "
                f"{', '.join(_fallback_only)}"
            )
        if _missing:
            message = (
                f"[custom-ops] NOT registered after {_attempt} attempt(s): "
                f"{', '.join(_missing)}"
            )
            if os.environ.get("LOOM_STRICT_GRAPH_COVERAGE") == "1":
                raise RuntimeError(message)
            ddp_print(f"{message} (worth investigating)")
        if bool(getattr(cfg, "save_graph", False)):
            _save_compiled_graph(cfg, model_base, device, tag)

    # All ranks must finish independent eager custom-op capture before the
    # first DDP constructor collective.
    ddp_barrier(device)
    if ddp_is_distributed() and bool(getattr(cfg, "fsdp_full_shard", False)):
        if bool(getattr(cfg, "optimizer_zero_shard", False)):
            raise ValueError(
                "fsdp_full_shard and optimizer_zero_shard are mutually exclusive")
        if bool(getattr(cfg, "compile", False)):
            raise ValueError("fsdp_full_shard local mode requires compile=false")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        distributed_model = FSDP(
            model_base,
            device_id=device,
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=False,
        )
        ddp_print(
            f"[fsdp] FULL_SHARD across {ddp_world_size()} ranks "
            "use_orig_params=true limit_all_gathers=true")
    elif ddp_is_distributed():
        # static_graph reuses the first iteration's autograd trace, which is
        # incompatible with the no_sync() windows gradient accumulation needs
        # (the reducer asserts expect_autograd_hooks_ on the skipped backward).
        # It is also invalid for our compiled model: the depth-replay stack is
        # an intentional eager island inside the compiled outer model, so its
        # reducer-hook schedule is not one static compiled autograd graph.
        static_graph, static_graph_reason = ddp_static_graph_policy(cfg)
        distributed_model = DDP(
            model_base,
            device_ids=[ddp_local_rank()],
            output_device=ddp_local_rank(),
            broadcast_buffers=False,
            find_unused_parameters=False,
            bucket_cap_mb=64,
            gradient_as_bucket_view=True,
            static_graph=static_graph,
        )
        if static_graph:
            static_graph_note = ""
        else:
            static_graph_note = f" (disabled: {static_graph_reason})"
        ddp_print(
            "[ddp] buckets=64MiB gradient_as_bucket_view=true "
            "buffers=explicit-pre-forward "
            f"static_graph={str(static_graph).lower()}"
            f"{static_graph_note}"
        )
    else:
        distributed_model = model_base

    # DDP owns the AccumulateGrad/reducer hooks. Compile the DDP wrapper, not
    # the bare module: compiling the bare backward first and installing DDP
    # afterwards can cache a backward that never reaches the reducer hooks,
    # leaving the first real training iteration blocked forever.
    model = maybe_compile(
        distributed_model,
        device,
        enabled=bool(getattr(cfg, "compile", False)),
    )

    # Materialize Dynamo/Inductor through the exact wrapper used by training.
    # For DDP this is a complete reducer iteration, so its gradient hooks and
    # collectives are compiled/observed together on every rank.
    if (
        bool(getattr(cfg, "compile", False))
        and hasattr(torch, "compile")
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device)[0] >= 7
    ):
        checkpoint_note = (
            "; checkpointed Tria stack stays eager"
            if GRAD_CHECKPOINTING and TRIA_CARRY_ENABLED
            and TRIA_TEMPORAL_ENABLED
            else ""
        )
        ddp_print(
            "[compile] torch.compile warmup -- tracing tensor regions + "
            f"Inductor codegen{checkpoint_note} (this may take a while)..."
        )
        _warm_batch2 = torch.randint(
            0,
            VOCAB,
            (int(cfg.batch_size), SEQ_LEN + 1),
            device=device,
            dtype=torch.long,
        )
        _wx2, _wy2 = _warm_batch2[:, :-1], _warm_batch2[:, 1:]
        _warm_pos2, _warm_mask2 = build_doc_reset_state(
            _wx2, train_eos_id
        )
        _warm_mask2 = ensure_temporal_chunk_plans(
            _wx2, _warm_mask2, cfg
        )
        _compile_t0 = time.time()
        ddp_trace("compile_warmup_buffer_sync_begin", step=0)
        ddp_sync_mutable_buffers(model)
        ddp_trace("compile_warmup_buffer_sync_end", step=0)
        ddp_trace("compile_warmup_forward_begin", step=0)
        with amp_autocast(device):
            _wloss2 = model(
                _wx2,
                attn_mask=_warm_mask2,
                position_ids=_warm_pos2,
                labels=_wy2,
            )
        ddp_trace("compile_warmup_forward_end", step=0)
        ddp_trace("compile_warmup_backward_begin", step=0)
        _wloss2.backward()
        ddp_trace("compile_warmup_backward_end", step=0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _compile_s = time.time() - _compile_t0
        ddp_print(
            f"[compile] warmup done in {_compile_s:.1f}s -- "
            "compiled graph cached for subsequent steps"
        )
        model_base.zero_grad(set_to_none=True)
        del _warm_batch2, _wx2, _wy2, _warm_pos2, _warm_mask2, _wloss2
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Keep ranks aligned after local Inductor work before optimizer creation.
    ddp_barrier(device)

    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    OptimizerClass, optimizer_name = optimizer_class_from_name(cfg.optimizer)
    base_model = ddp_unwrap_model(model)
    tria_agg = getattr(base_model, "tria_agg", None)
    no_decay_param_ids = set()
    if tria_agg is not None:
        no_decay_param_ids.add(id(tria_agg.pool.logit_scale_raw))

    # One param_group per distinct (weight_decay, lr_mult) combination --
    # unifies the pre-existing no-decay case with train_lr_by_id's per-
    # parameter lr_mult overrides (loomcloner.py --clone's transplanted-vs-
    # fresh split). lr_mult (not an absolute lr) because the training loop's
    # own schedule does g["lr"] = lr_at(cfg, step) * g.get("lr_mult", 1.0)
    # every step -- setting an absolute lr here would just get overwritten
    # by that line on the very first step.
    groups: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}
    for p in params:
        wd = 0.0 if id(p) in no_decay_param_ids else cfg.weight_decay
        mult = train_lr_by_id.get(id(p), 1.0)
        groups.setdefault((wd, mult), []).append(p)
    if len(groups) > 1:
        summary = ", ".join(f"{len(ps)}@lr_mult={mult:g}/wd={wd:g}" for (wd, mult), ps in groups.items())
        ddp_print(f"[optimizer] {len(groups)} param groups: {summary}")
    optimizer_groups = [
        {"params": ps, "weight_decay": wd, "lr_mult": mult}
        for (wd, mult), ps in groups.items()
    ]
    if bool(getattr(cfg, "optimizer_zero_shard", False)):
        if not ddp_is_distributed():
            raise ValueError("optimizer_zero_shard=true requires DDP")
        if ckpt_out:
            raise ValueError(
                "optimizer_zero_shard currently requires checkpoint output to "
                "be disabled; consolidated optimizer checkpointing is not enabled")
        from torch.distributed.optim import ZeroRedundancyOptimizer
        opt = ZeroRedundancyOptimizer(
            optimizer_groups,
            optimizer_class=OptimizerClass,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        ddp_print(
            f"[optimizer] ZeroRedundancyOptimizer shards {optimizer_name} "
            f"state across {ddp_world_size()} ranks")
    else:
        opt = OptimizerClass(
            optimizer_groups,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    if resume_in:
        load_optimizer_checkpoint(opt, model, resume_in, optimizer_name, device)
    if bool(getattr(cfg, "save_initial_checkpoint", False)) and ckpt_out and ddp_is_main():
        root, ext = os.path.splitext(ckpt_out)
        init_path = f"{root}.init{ext or '.pt'}"
        saved_cfg = asdict(cfg)
        saved_cfg["batch_size"] = int(getattr(cfg, "_global_batch_size", cfg.batch_size))
        if saved_cfg.get("tria_temporal_window") is not None:
            saved_cfg["tria_temporal_auto"] = False
        torch.save(
            {"cfg": saved_cfg, "model_kind": "loomformer", "ffn_type": "paraplex",
             "ablation": ablation, "model": model_base.state_dict(),
             "optimizer_name": optimizer_name, "optimizer": opt.state_dict(),
             "step": 0, "tokens_seen": 0,
             "dataset_progress_version": 1,
             "dataset_progress": dataset_progress},
            init_path,
        )
        ddp_print(f"[train] saved initial {tag} with optimizer state -> {init_path}")
    n_params = count_params(ddp_unwrap_model(model))

    if is_sft_dataset(cfg):
        data_note = "SFT assistant-target loss"
    elif os.path.abspath(eval_dataset) == os.path.abspath(dataset):
        data_note = f"{train_bpt:.3f} bytes/token"
    else:
        data_note = f"{train_bpt:.3f} train bytes/token  ·  {eval_bpt:.3f} eval bytes/token"
    ddp_print(f"--- {tag}: {n_params:,} params  ·  optimizer={optimizer_name}  ·  {data_note} ---")
    t0 = time.time()
    final_eval = float("nan")
    best_eval = float("inf")
    full_eval_loss = float("nan")
    full_eval_bpb = float("nan")

    accum_steps = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    backward_scale = float(getattr(cfg, "backward_scale", 1.0) or 1.0)
    if not math.isfinite(backward_scale) or backward_scale < 1.0:
        raise ValueError(
            f"backward_scale must be finite and >= 1, got {backward_scale!r}")
    if backward_scale != 1.0:
        ddp_print(
            f"[precision] backward_scale={backward_scale:g} "
            "(loss downscaled during backward; gradients restored after clipping)")
    tokens_seen_global = int(tokens_seen_at_start)
    data_wait_s = 0.0 
    batch_iter = stream.batches((int(cfg.steps) - start_step) * accum_steps).__aiter__()

    def _tensor_stats(t: torch.Tensor) -> str:
        tf = t.detach().float()
        finite = torch.isfinite(tf)
        finite_count = int(finite.sum().item())
        total = tf.numel()
        if finite_count > 0:
            vals = tf[finite]
            amin = float(vals.amin().item())
            amax = float(vals.amax().item())
        else:
            amin = float("nan")
            amax = float("nan")
        nan_count = int(torch.isnan(tf).sum().item())
        inf_count = int(torch.isinf(tf).sum().item())
        return (
            f"shape={tuple(t.shape)} finite={finite_count}/{total} "
            f"nan={nan_count} inf={inf_count} min={amin:.9g} max={amax:.9g}"
        )

    def _first_nonfinite(which: str) -> Optional[str]:
        for name, p in named_params:
            t = p.grad if which == "grad" else p
            if t is None:
                continue
            if not torch.isfinite(t.detach()).all():
                grad_stats = _tensor_stats(p.grad) if p.grad is not None else "grad=None"
                return (
                    f"{name}: {which} {_tensor_stats(t)} | "
                    f"param {_tensor_stats(p)} | {grad_stats}"
                )
        return None

    def _collect_nonfinite(which: str, limit: int = 10) -> List[str]:
        out: List[str] = []
        for name, p in named_params:
            t = p.grad if which == "grad" else p
            if t is None:
                continue
            if not torch.isfinite(t.detach()).all():
                grad_stats = _tensor_stats(p.grad) if p.grad is not None else "grad=None"
                out.append(
                    f"{name}: {which} {_tensor_stats(t)} | "
                    f"param {_tensor_stats(p)} | {grad_stats}"
                )
                if len(out) >= limit:
                    break
        return out

    def _summarize_nonfinite(which: str) -> str:
        counts: Dict[str, int] = {}
        total = 0
        for name, p in named_params:
            t = p.grad if which == "grad" else p
            if t is None:
                continue
            if not torch.isfinite(t.detach()).all():
                total += 1
                prefix = name.split(".", 2)
                if len(prefix) >= 2 and prefix[0] == "blocks":
                    key = ".".join(prefix[:2])
                else:
                    key = prefix[0]
                counts[key] = counts.get(key, 0) + 1
        if total == 0:
            return "(none)"
        parts = [f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        return f"total={total} by_module=" + ", ".join(parts)

    def _save_checkpoint(path_override: Optional[str] = None,
                         step_override: Optional[int] = None) -> None:
        if ckpt_out and ddp_is_main():
            save_path = path_override or ckpt_out
            raw_model = ddp_unwrap_model(model)
            saved_cfg = asdict(cfg)
            # DDP mutates cfg.batch_size to the per-rank batch at startup. A
            # checkpoint is portable configuration, so persist the user-facing
            # global batch rather than silently halving it on the next launch.
            saved_cfg["batch_size"] = int(getattr(cfg, "_global_batch_size", cfg.batch_size))
            # Persist resolved geometry, not a request to calibrate it again.
            if saved_cfg.get("tria_temporal_window") is not None:
                saved_cfg["tria_temporal_auto"] = False
            saved_step = int(step if step_override is None else step_override)
            dataset_progress[current_dataset_key] = {
                "steps": int(current_dataset_steps),
                "draws": int(current_dataset_draws),
            }
            torch.save(
                {"cfg": saved_cfg, "model_kind": "loomformer", "ffn_type": "paraplex",
                 "ablation": ablation, "model": raw_model.state_dict(),
                 "optimizer_name": optimizer_name, "optimizer": opt.state_dict(),
                 "step": saved_step, "tokens_seen": int(tokens_seen_global),
                 "dataset_progress_version": 1,
                 "dataset_progress": dataset_progress},
                save_path,
            )
            print(f"[train] saved {tag} -> {save_path}")

    def _save_runpoint(current_step: int) -> None:
        if not (ckpt_out and ddp_is_main()):
            return
        base = ckpt_out
        root, ext = os.path.splitext(base)
        ext = ext or ".pt"
        fname = f"{os.path.basename(root)}.runpoint_step{current_step}{ext}"
        if cfg.runpoints_path:
            os.makedirs(cfg.runpoints_path, exist_ok=True)
            runpoint_path = os.path.join(cfg.runpoints_path, fname)
        else:
            runpoint_path = f"{root}.runpoint_step{current_step}{ext}"
        print(f"\n[runpoint] step {current_step}/{cfg.steps} -- saving, training continues.",
              file=_REAL_STDOUT, flush=True)
        _save_checkpoint(path_override=runpoint_path, step_override=current_step)

    def _handle_interrupt() -> None:
        completed_step = max(int(start_step), int(step) - 1)
        save_flag = 0
        if ddp_is_main():
            print(f"\n[interrupt] Ctrl-C after completed step "
                  f"{completed_step}/{cfg.steps} "
                  f"({tag}, {time.time() - t0:.0f}s elapsed).",
                  file=_REAL_STDOUT, flush=True)
            decision_path = os.environ.get(
                "LOOM_DDP_INTERRUPT_DECISION_FILE", ""
            )
            if decision_path:
                try:
                    with open(decision_path, encoding="ascii") as handle:
                        answer = "y" if handle.read().strip() == "1" else "n"
                except OSError:
                    answer = "n"
            else:
                runpoint.pause()  # keep the key watcher away from input()
                try:
                    print("[interrupt] save a checkpoint at this step before exiting? [y/N] ",
                          end="", file=_REAL_STDOUT, flush=True)
                    answer = input().strip().lower()
                except EOFError:
                    answer = "n"
                finally:
                    runpoint.resume()
            save_flag = 1 if answer in ("y", "yes") else 0

        if ddp_is_distributed():
            flag_t = torch.tensor([save_flag], dtype=torch.int32, device=device)
            dist.broadcast(flag_t, src=0)
            save_flag = int(flag_t.item())

        if save_flag:
            _save_checkpoint(step_override=completed_step)
            if ddp_is_main():
                print("[interrupt] saved.", file=_REAL_STDOUT, flush=True)
        elif ddp_is_main():
            print("[interrupt] not saving -- exiting without a checkpoint.", file=_REAL_STDOUT, flush=True)

        if ddp_is_distributed():
            ddp_barrier(device)
            dist.destroy_process_group()
        # A self-launching parent returns conventional status 130 after
        # torchrun observes clean rank exits. Direct/single-process runs own
        # their shell status themselves.
        raise SystemExit(
            0 if os.environ.get("LOOM_SELF_LAUNCHED_DDP") == "1" else 130
        )

    step = start_step
    refeeds_since_log = torch.zeros((), dtype=torch.long, device=device)
    tokens_per_second_ema: Optional[float] = None
    tps_ema_alpha = 2.0 / 11.0  # conventional EMA span of 10 completed steps
    raw_model_for_tria = ddp_unwrap_model(model)  # temporal Tria diagnostics
                                                    # doesn't proxy through DDP's
                                                    # wrapper -- same reason
                                                    # ddp_unwrap_model exists at all.
    ddp_trace("train_loop_ready", step=start_step)
    with _GracefulInterrupt() as interrupt, _RunpointWatcher() as runpoint:
        for step in range(start_step + 1, int(cfg.steps) + 1):
            ddp_trace("step_begin", step=step)
            if interrupt.requested:
                _handle_interrupt()
            if runpoint.consume():
                # The request was observed between iterations; the previous
                # optimizer update is the latest completed state.
                _save_runpoint(step - 1)
            _step_t0 = time.time()
            opt.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            train_tokens_step = 0
            for micro_idx in range(accum_steps):
                trace_micro = micro_idx + 1
                ddp_trace(
                    "batch_fetch_begin", step=step, micro=trace_micro
                )
                _wait_t0 = time.time()
                batch = await batch_iter.__anext__()
                ddp_trace(
                    "batch_fetch_end", step=step, micro=trace_micro
                )
                data_wait_s += time.time() - _wait_t0
                x, y, position_ids, attn_mask = split_train_batch(
                    batch, train_eos_id, cfg
                )
                sync_ctx = (
                    model.no_sync()
                    if ddp_is_distributed() and micro_idx + 1 < accum_steps
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    ddp_trace(
                        "buffer_sync_begin", step=step, micro=trace_micro
                    )
                    ddp_sync_mutable_buffers(model)
                    ddp_trace(
                        "buffer_sync_end", step=step, micro=trace_micro
                    )
                    ddp_trace(
                        "forward_begin", step=step, micro=trace_micro
                    )
                    with amp_autocast(device):
                        loss = model(x, attn_mask=attn_mask, position_ids=position_ids, labels=y)
                    ddp_trace(
                        "forward_end", step=step, micro=trace_micro
                    )
                    total_loss = loss
                    # Read before custom CUDA backward: if the same scalar changes
                    # afterwards, a backward kernel corrupted forward storage.
                    loss_before_backward = float(loss.detach().item())
                    if bool(getattr(cfg, "graph", False)) and loss_before_backward > 20.0:
                        anchors = [block.ffn.beta_anchor.detach().clone() for block in model_base.blocks]
                        with torch.no_grad(), amp_autocast(device):
                            eager_logits = model_base(x, attn_mask=attn_mask, position_ids=position_ids)
                        eager_loss = float(F.cross_entropy(
                            eager_logits.float().reshape(-1, VOCAB), y.reshape(-1)).item())
                        for block, anchor in zip(model_base.blocks, anchors):
                            block.ffn.beta_anchor.copy_(anchor)
                        raise RuntimeError(
                            "compiled/eager train-forward diagnostic: "
                            f"compiled_loss={loss_before_backward:.9g}, eager_loss={eager_loss:.9g}, "
                            f"eager_max_logit={float(eager_logits.detach().abs().amax().item()):.9g}"
                        )
                    if not math.isfinite(loss_before_backward):
                        raise RuntimeError(
                            "non-finite training loss before backward: "
                            f"loss={loss_before_backward:.9g} "
                            f"optimizer={optimizer_name} step={step} micro={micro_idx + 1}/{accum_steps}"
                        )
                    if raw_model_for_tria.last_tria_fire_mask is not None:
                        with torch.no_grad():
                            refeeds_since_log.add_(raw_model_for_tria.last_tria_fire_mask.detach().sum())
                    ddp_trace(
                        "backward_begin", step=step, micro=trace_micro
                    )
                    (total_loss / (float(accum_steps) * backward_scale)).backward()
                    ddp_trace(
                        "backward_end", step=step, micro=trace_micro
                    )
                    loss_after_backward = float(loss.detach().item())
                    if math.isfinite(loss_before_backward) and math.isfinite(loss_after_backward) and loss_after_backward != loss_before_backward:
                        raise RuntimeError(
                            "training loss tensor changed during backward: "
                            f"before={loss_before_backward:.9g}, after={loss_after_backward:.9g}; "
                            "a custom CUDA backward kernel wrote into forward storage"
                        )
                    if not math.isfinite(loss_after_backward):
                        bad_grad = _first_nonfinite("grad")
                        bad_grad_list = " || ".join(_collect_nonfinite("grad"))
                        bad_grad_summary = _summarize_nonfinite("grad")
                        raise RuntimeError(
                            "non-finite training loss after backward: "
                            f"before={loss_before_backward:.9g}, after={loss_after_backward:.9g}; "
                            f"summary={bad_grad_summary}; "
                            f"first_nonfinite_grad={bad_grad or '(none found)'}; "
                            f"examples={bad_grad_list or '(none found)'}"
                        )
                    raw_model_for_tria.last_tria_depth_carry = None
                    raw_model_for_tria.last_tria_document_carry_stats = None
                train_loss_sum += loss_before_backward
                train_tokens_step += int(y.numel())

            bad_grad = _first_nonfinite("grad")
            if bad_grad is not None:
                bad_grad_list = " || ".join(_collect_nonfinite("grad"))
                bad_grad_summary = _summarize_nonfinite("grad")
                raise RuntimeError(
                    "non-finite gradient detected before optimizer step: "
                    f"optimizer={optimizer_name} step={step} lr={lr_at(cfg, step - 1):.9g} "
                    f"backward_scale={backward_scale:g} "
                    f"summary={bad_grad_summary}; "
                    f"first={bad_grad}; "
                    f"examples={bad_grad_list}"
                )
            if cfg.grad_clip and cfg.grad_clip > 0:
                ddp_trace("grad_clip_begin", step=step)
                if backward_scale == 1.0:
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                else:
                    # Keep the norm reduction in scaled space.  Applying
                    # min(S, C / ||g_scaled||) restores S and clips to C in
                    # one multiply, without clip_grad_norm_'s fixed 1e-6
                    # denominator epsilon crushing deliberately tiny grads.
                    scaled_norm = torch.nn.utils.clip_grad_norm_(
                        params, float("inf"))
                    restore = (float(cfg.grad_clip) / scaled_norm).clamp(
                        max=backward_scale)
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(restore.to(device=p.grad.device))
                ddp_trace("grad_clip_end", step=step)
            elif backward_scale != 1.0:
                for p in params:
                    if p.grad is not None:
                        p.grad.mul_(backward_scale)
            if backward_scale != 1.0:
                bad_unscaled_grad = _first_nonfinite("grad")
                if bad_unscaled_grad is not None:
                    raise RuntimeError(
                        "non-finite gradient while restoring backward scale: "
                        f"optimizer={optimizer_name} step={step} "
                        f"backward_scale={backward_scale:g}; "
                        f"first={bad_unscaled_grad}"
                    )
            lr = lr_at(cfg, step - 1)
            for g in opt.param_groups:
                g["lr"] = lr * float(g.get("lr_mult", 1.0))
            ddp_trace("optimizer_begin", step=step)
            opt.step()
            ddp_trace("optimizer_end", step=step)
            bad_param = _first_nonfinite("param")
            if bad_param is not None:
                bad_param_list = " || ".join(_collect_nonfinite("param"))
                bad_param_summary = _summarize_nonfinite("param")
                raise RuntimeError(
                    "non-finite parameter detected after optimizer step: "
                    f"optimizer={optimizer_name} step={step} lr={lr:.9g} "
                    f"summary={bad_param_summary}; "
                    f"first={bad_param}; "
                    f"examples={bad_param_list}"
                )

            current_dataset_steps += 1
            current_dataset_draws += accum_steps
            dataset_progress[current_dataset_key] = {
                "steps": current_dataset_steps,
                "draws": current_dataset_draws,
            }
            train_loss_local = train_loss_sum / float(accum_steps)
            ddp_trace("metrics_allreduce_begin", step=step)
            train_loss_log = ddp_mean_float(train_loss_local, device)
            train_tokens_global = ddp_sum_int(train_tokens_step, device)
            ddp_trace("metrics_allreduce_end", step=step)
            tokens_seen_global += int(train_tokens_global)
            _step_seconds = max(time.time() - _step_t0, 1e-9)
            _step_tps = float(train_tokens_global) / _step_seconds
            tokens_per_second_ema = (
                _step_tps
                if tokens_per_second_ema is None
                else (
                    tps_ema_alpha * _step_tps
                    + (1.0 - tps_ema_alpha) * tokens_per_second_ema
                )
            )
            _remaining_tokens = (
                max(0, int(cfg.steps) - step) * int(train_tokens_global))
            _left = format_eta_hours_minutes(
                _remaining_tokens / max(tokens_per_second_ema, 1e-9))

            log_every = max(1, int(getattr(cfg, "log_every", 100)))
            eval_every_cfg = getattr(cfg, "eval_every", None)
            eval_every = log_every if eval_every_cfg is None else max(1, int(eval_every_cfg))
            if step == 1 or step % log_every == 0:
                refeeds_log_t = refeeds_since_log.detach().clone()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(refeeds_log_t, op=dist.ReduceOp.SUM)
                refeeds_log = int(refeeds_log_t.item())
                refeeds_since_log.zero_()
                if step == 1 or step % eval_every == 0:
                    eval_model = (
                        model
                        if bool(getattr(cfg, "fsdp_full_shard", False))
                        else model_base
                    )
                    final_eval_local = await eval_loss_async(
                        eval_model, eval_stream, cfg, device, eos_id=eval_eos_id)
                    final_eval = final_eval_local
                    best_eval = min(best_eval, final_eval)
                    bits_tok, bpb = loss_to_bits(final_eval, eval_bpt)
                    elapsed = time.time() - t0
                    ddp_print(format_train_status(
                        step, train_loss_log, refeeds_log, lr,
                        tokens_seen_global, data_wait_s, _left, elapsed))
                    ddp_print(format_eval_status(
                        step, final_eval, bits_tok, bpb))
                else:
                    elapsed = time.time() - t0
                    ddp_print(format_train_status(
                        step, train_loss_log, refeeds_log, lr,
                        tokens_seen_global, data_wait_s, _left, elapsed))
            if cfg.save_every and step % int(cfg.save_every) == 0:
                _save_runpoint(step)

    seconds = time.time() - t0
    _save_checkpoint()
    if val_dataset:
        # Training compiles the label/loss forward. Full evaluation calls the
        # logits-only forward with different shapes (including a short tail).
        # Reusing the compiled wrapper here forces a second, unrelated
        # Inductor trace and has triggered upstream post-grad matcher failures
        # on otherwise valid models. The eager module owns the same updated
        # parameters; retain the FSDP wrapper only when parameters are sharded.
        full_eval_model = (
            model
            if bool(getattr(cfg, "fsdp_full_shard", False))
            else model_base
        )
        full = eval_full_model(full_eval_model, cfg, val_dataset, device)
        if is_sft_dataset(cfg):
            full_eval_loss, full_eval_tokens = ddp_weighted_mean(
                float(full["total_nll"]), int(full["total_tokens"]), device)
        else:
            full_eval_loss = ddp_mean_float(float(full["loss_nats"]), device)
            full_eval_tokens = int(full["total_tokens"])
        full_eval_bpb = (
            full_eval_loss / math.log(2.0) / float(eval_bpt)
            if math.isfinite(float(full["bpb"]))
            else float("nan")
        )
        full_eval_fields = (
            f"[{tag}] full_eval {val_dataset}  "
            f"eval_loss {full_eval_loss:.4f}  "
            f"bits/tok {full_eval_loss / math.log(2.0):.4f}"
        )
        if math.isfinite(full_eval_bpb):
            full_eval_fields += f"  bpb {full_eval_bpb:.4f}"
        full_eval_fields += f"  tokens {full_eval_tokens}"
        ddp_print(full_eval_fields)
    return {
        "final_eval_loss": final_eval,
        "best_eval_loss": best_eval,
        "full_eval_loss": full_eval_loss,
        "full_eval_bpb": full_eval_bpb,
        "seconds": seconds,
    }


def print_architecture_report(cfg: Config, device: torch.device, ablation: bool,
                               dataset: str, val_dataset: Optional[str]) -> None:
    width = 64
    rule = "=" * width
    ddp_print(rule)
    ddp_print(f" LoomFormer  ·  {device}  ·  amp={AMP_DTYPE}  ·  ablation={ablation}")
    ddp_print(rule)
    grp = f"x{GQA_GROUP_SIZE}" if GQA_GROUP_SIZE else "x1"
    ddp_print(f"  shape    d_model={N}  heads={N_Q_HEADS}q/{N_KV_HEADS}kv({grp})  "
               f"head_dim={HEAD_DIM}  layers={LAYERS}")
    ddp_print(f"  ffn      hidden={HIDDEN}  phase={PHASE_SECTORS}  attn={ATTN_IMPL}")
    branch_cap = "off" if RESIDUAL_BRANCH_RMS_CAP is None else f"{RESIDUAL_BRANCH_RMS_CAP:g}"
    ddp_print(f"  depth    readout={DEPTH_ATTN_READOUT}  qkv_rms={DEPTH_ATTN_QKV_RMS}  "
               f"branch_rms_cap={branch_cap}")
    ddp_print(
        f"  memory   activation_checkpoint="
        f"{'temporal-chunk' if GRAD_CHECKPOINTING else 'off'}  "
        f"torch_compile={bool(getattr(cfg, 'compile', False))}  "
        f"custom_op_graph={bool(getattr(cfg, 'graph', False))}")
    ddp_print(f"  rope     yarn  theta={ROPE_THETA:g}  factor={ROPE_FACTOR:g}x  "
               f"orig_len={ROPE_ORIGINAL_SEQ_LEN}")
    if HEAD_DIM < 8:
        ddp_print(f"  WARNING: head_dim={HEAD_DIM} is extremely small for LM attention.")
    ddp_print(f"  data     {dataset}")
    ddp_print(f"  val      {val_dataset}" if val_dataset else "  val      (none -- training loss only)")


async def train_async(
    cfg: Config,
    dataset: str,
    device: torch.device,
    ckpt_out: Optional[str],
    ablation: bool,
    resume: Optional[str] = None,
    resume_step: Optional[int] = None,
    resume_dataset_steps: Optional[int] = None,
    init_weights: Optional[str] = None,
) -> None:
    set_seed(int(cfg.seed) + 1000003 * int(ddp_rank()))
    # Persist the effective path even when it came from CLI --dataset. This is
    # what makes resume_data_stream:auto reliable on the next launch.
    cfg.train_dataset = dataset
    build_tokenizer(cfg)
    restore_temporal_tria_from_checkpoint(cfg, resume or init_weights)

    if ddp_is_main():
        maybe_auto_val_split(cfg, dataset)
    ddp_barrier(device)
    if not ddp_is_main() and not cfg.val_dataset:
        maybe_auto_val_split(cfg, dataset)
    val_dataset = str(cfg.val_dataset).strip() if cfg.val_dataset else None

    results = {}
    results["paraplex"] = await train_one_async(
        cfg, dataset, device, ablation, ckpt_out, resume, val_dataset=val_dataset,
        resume_step=resume_step, resume_dataset_steps=resume_dataset_steps,
        init_weights=init_weights,
    )

    ddp_print("\nSummary:")
    for name, r in results.items():
        if r.get("skipped", 0.0):
            ddp_print(f"{name}: skipped -- checkpoint step {int(r['start_step'])} >= cfg.steps")
            continue
        full_note = ""
        if math.isfinite(float(r.get("full_eval_loss", float("nan")))):
            full_note = f" | full_eval_loss {r['full_eval_loss']:.4f} | full_bpb {r['full_eval_bpb']:.4f}"
        ddp_print(
            f"{name}: final_eval_loss {r['final_eval_loss']:.4f} | "
            f"best_eval_loss {r['best_eval_loss']:.4f}{full_note} | time {r['seconds']:.0f}s"
        )


@torch.no_grad()
def infer(cfg: Config, ckpt: str, prompt: str, max_new: int, device: torch.device) -> None:
    blob = torch.load(ckpt, map_location=device, weights_only=True)
    cfg = Config.from_checkpoint_dict(blob["cfg"])
    tok = build_tokenizer(cfg)
    apply_config(cfg)
    ablation = bool(blob.get("ablation", False))
    model = Model(cfg, ablation=ablation).to(device)
    load_model_blob_into(model, blob, ablation=ablation)
    model.eval()
    ids = tok.encode(prompt) or [0]
    states = None
    logits = None
    for pos, tid in enumerate(ids):
        x = torch.tensor([int(tid)], device=device, dtype=torch.long)
        logits, states = model.step(x, pos, states)
    out_ids = list(ids)
    for i in range(max_new):
        nxt = int(torch.argmax(logits, dim=-1).item())
        out_ids.append(nxt)
        x = torch.tensor([nxt], device=device, dtype=torch.long)
        logits, states = model.step(x, len(ids) + i, states)
    print(tok.decode(out_ids))


def export_aoti(cfg: Config, ckpt: str, out_path: str, device: torch.device, batch_size: int = 1) -> None:
    import graph_helper

    blob = torch.load(ckpt, map_location=device, weights_only=True)
    cfg = Config.from_checkpoint_dict(blob["cfg"])
    build_tokenizer(cfg)
    apply_config(cfg) 
    graph_helper.install_capture_hooks(sys.modules[__name__], tria)

    ablation = bool(blob.get("ablation", False))
    model = Model(cfg, ablation=ablation).to(device)
    load_model_blob_into(model, blob, ablation=ablation)
    model.eval()

    example_x = torch.randint(0, VOCAB, (batch_size, SEQ_LEN), device=device, dtype=torch.long)

    if not graph_helper.is_finalized():
        with torch.no_grad():
            model(example_x)
        graph_helper.finalize_registration(sys.modules[__name__], tria)

    with torch.no_grad():
        exported = torch.export.export(model, (example_x,))
        pkg_path = torch._inductor.aoti_compile_and_package(exported, package_path=out_path)

    print(f"[export-aoti] wrote {pkg_path}")
    print(f"[export-aoti] shape baked in: batch_size={batch_size}, seq_len={SEQ_LEN}, vocab={VOCAB}")
    print(f"[export-aoti] load with: torch._inductor.aoti_load_package({pkg_path!r})  -- no Python model code needed at inference time")


def _eval_full_batch_nll(model: nn.Module, batch: torch.Tensor, device: torch.device) -> Tuple[float, int]:
    x, y = batch[:, :-1], batch[:, 1:]
    with amp_autocast(device):
        logits = model(x)
    nll = F.cross_entropy(logits.float().reshape(-1, VOCAB), y.reshape(-1), reduction="sum")
    return float(nll.item()), int(y.numel())


@torch.no_grad()
def _eval_full_sft(model: nn.Module, cfg: Config, dataset: str, device: torch.device,
                    eval_batch_size: Optional[int] = None) -> Dict[str, float]:
    """One deterministic pass over a packed SFT split, loss on target tokens only.

    bpb is reported as NaN: SFT rows are rendered chat, so there is no
    bytes-per-token ratio to normalize bits against.
    """
    stream = make_stream(dataset, cfg, device)
    try:
        B = max(1, int(eval_batch_size if eval_batch_size is not None else cfg.batch_size))
        total_nll, total_tokens = 0.0, 0
        if hasattr(stream, "iter_eval_batches"):
            batches = stream.iter_eval_batches(B)
        else:
            def cached_batches():
                remaining = int(stream.n_rows)
                while remaining > 0:
                    rows = stream._take_rows(min(B, remaining))
                    remaining -= len(rows)
                    ids, mask, _ = stream._pack_batch_np(rows)
                    yield torch.from_numpy(ids), torch.from_numpy(mask), None
            batches = cached_batches()
        for ids, mask, packed_layout in batches:
            ids = ids.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            metadata = (
                packed_layout.to(device, non_blocking=True)
                if packed_layout is not None
                else None
            )
            batch = (ids, mask, metadata) if metadata is not None else (ids, mask)
            x, y, position_ids, attn_mask = split_train_batch(
                batch, stream._eos_id, cfg
            )
            ntok = int((y != IGNORE_INDEX).sum().item())
            if ntok == 0:
                continue
            with amp_autocast(device):
                loss = model(x, attn_mask=attn_mask, position_ids=position_ids, labels=y,
                             ignore_index=IGNORE_INDEX)
            total_nll += float(loss.item()) * ntok
            total_tokens += ntok
    finally:
        stream.close()
    loss_nats = total_nll / max(1, total_tokens)
    return {
        "total_tokens": float(total_tokens),
        "total_nll": float(total_nll),
        "loss_nats": float(loss_nats),
        "bits_tok": float(loss_nats / math.log(2.0)),
        "bpb": float("nan"),
    }


def _tokenize_raw_corpus_full(path: str, cfg: Config) -> Tuple[np.ndarray, float]:
    tok = build_tokenizer(cfg)
    corpus = RawCorpus(path, fmt=getattr(cfg, "dataset_format", "auto"),
                        text_field=getattr(cfg, "text_field", "text"))
    ids: List[int] = []
    total_bytes = 0
    for fi, key, length in corpus._docs:
        text = corpus._read_doc_text(fi, key, length)
        total_bytes += len(text.encode("utf-8"))
        ids.extend(tok.encode(text))
    assert cfg.vocab <= 65536, "uint16 storage requires vocab <= 65536"
    arr = np.array(ids, dtype=np.uint16)
    bpt = total_bytes / max(1, len(arr))
    return arr, bpt


@torch.no_grad()
def eval_full_model(
    model: nn.Module,
    cfg: Config,
    dataset: str,
    device: torch.device,
    eval_batch_size: Optional[int] = None,
    eval_data_cache: str = "ram",
) -> Dict[str, float]:
    model.eval()
    fmt = str(getattr(cfg, "dataset_format", "auto") or "auto").lower()
    if fmt == "sft":
        return _eval_full_sft(model, cfg, dataset, device, eval_batch_size)
    is_bin = fmt == "bin" or (fmt == "auto" and os.path.isfile(dataset) and dataset.endswith(".bin"))
    if is_bin:
        bpt, _, _ = load_bytes_per_token(dataset)
        mmap = np.memmap(dataset, dtype=np.uint16, mode="r")
    else:
        mmap, bpt = _tokenize_raw_corpus_full(dataset, cfg)
    if len(mmap) < 2:
        raise ValueError(f"dataset too short for eval: {dataset}")

    T = int(cfg.seq_len)
    B = max(1, int(eval_batch_size if eval_batch_size is not None else cfg.batch_size))
    cache = str(eval_data_cache or "ram").lower()
    if cache not in ("mmap", "ram", "gpu"):
        raise ValueError(f"eval_data_cache must be 'mmap', 'ram', or 'gpu', got {eval_data_cache!r}")
    if cache == "gpu" and device.type != "cuda":
        cache = "ram"

    n = int(len(mmap))
    n_full = (n - 1) // T
    tail_start = n_full * T
    total_tokens = 0
    total_nll = 0.0

    if cache == "gpu":
        data_t = torch.from_numpy(np.array(mmap, dtype=np.int64, copy=True)).to(device, non_blocking=True)
        if n_full > 0:
            windows = data_t.unfold(0, T + 1, T)
            for i in range(0, int(windows.shape[0]), B):
                batch = windows[i : i + B]
                nll, ntok = _eval_full_batch_nll(model, batch, device)
                total_nll += nll
                total_tokens += ntok
        if tail_start < n - 1:
            batch = data_t[tail_start:n].view(1, -1)
            nll, ntok = _eval_full_batch_nll(model, batch, device)
            total_nll += nll
            total_tokens += ntok
    else:
        data = mmap if cache == "mmap" else np.array(mmap, dtype=np.uint16, copy=True)
        if n_full > 0:
            stride = data.strides[0]
            windows = np.lib.stride_tricks.as_strided(
                data,
                shape=(n_full, T + 1),
                strides=(T * stride, stride),
                writeable=False,
            )
            for i in range(0, n_full, B):
                arr = np.asarray(windows[i : i + B], dtype=np.int64)
                batch = torch.from_numpy(arr)
                if device.type == "cuda":
                    batch = batch.pin_memory()
                batch = batch.to(device, non_blocking=True)
                nll, ntok = _eval_full_batch_nll(model, batch, device)
                total_nll += nll
                total_tokens += ntok
        if tail_start < n - 1:
            arr = np.asarray(data[tail_start:n], dtype=np.int64)[None, :]
            batch = torch.from_numpy(arr)
            if device.type == "cuda":
                batch = batch.pin_memory()
            batch = batch.to(device, non_blocking=True)
            nll, ntok = _eval_full_batch_nll(model, batch, device)
            total_nll += nll
            total_tokens += ntok

    loss_nats = total_nll / max(1, total_tokens)
    bits_tok = loss_nats / math.log(2.0)
    bpb = bits_tok / bpt
    return {
        "total_tokens": float(total_tokens),
        "total_nll": float(total_nll),
        "loss_nats": float(loss_nats),
        "bits_tok": float(bits_tok),
        "bpb": float(bpb),
    }


@torch.no_grad()
def eval_full(
    ckpt: str,
    dataset: Optional[str],
    device: torch.device,
    eval_batch_size: Optional[int] = None,
    eval_data_cache: str = "ram",
) -> Dict[str, float]:
    blob = torch.load(ckpt, map_location=device, weights_only=True)
    cfg = Config.from_checkpoint_dict(blob["cfg"])
    if dataset is None:
        dataset = cfg.val_dataset
    if not dataset:
        raise ValueError("--eval needs --dataset, or val_dataset in config/checkpoint")
    build_tokenizer(cfg)
    apply_config(cfg)

    ablation = bool(blob.get("ablation", False))
    model = Model(cfg, ablation=ablation).to(device)
    load_model_blob_into(model, blob, ablation=ablation)
    model.eval()
    model = maybe_compile(
        model, device, enabled=bool(getattr(cfg, "compile", False)))

    out = eval_full_model(model, cfg, dataset, device, eval_batch_size, eval_data_cache)
    print(f"dataset {dataset}")
    print(f"total_tokens {int(out['total_tokens'])}")
    print(f"total_nll {out['total_nll']:.6f}")
    print(f"eval_loss {out['loss_nats']:.6f}")
    print(f"bits/token {out['bits_tok']:.6f}")
    print(f"bpb {out['bpb']:.6f}")
    return out

def smoke_test() -> None:
    dev = device_auto()
    cfg = Config(vocab=64, model_dim=12, n_q_heads=6, n_kv_heads=3, hidden=66, layers=2, seq_len=16, batch_size=2, steps=2)
    apply_config(cfg)
    set_seed(cfg.seed)
    model = Model(cfg).to(dev)
    x = torch.randint(0, VOCAB, (cfg.batch_size, cfg.seq_len), device=dev)
    logits = model(x)
    assert logits.shape == (cfg.batch_size, cfg.seq_len, VOCAB)
    assert model.head.weight is model.emb.weight
    y = torch.randint(0, VOCAB, (cfg.batch_size, cfg.seq_len), device=dev)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), y.reshape(-1))
    loss.backward()
    logit_std = float(logits.float().std().item())
    emb_std = float(model.emb.weight.float().std().item())
    print(
        f"[smoke] forward/backward OK logits={tuple(logits.shape)} loss={loss.item():.4f} "
        f"logit_std={logit_std:.4f} emb_std={emb_std:.4f} tied={model.head.weight is model.emb.weight}"
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description="LoomFormer: GQA Transformer LM with Paraplex FFN")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--sft-dataset", type=str, default=None,
                    help="train on a chat/SFT dataset (JSONL/Arrow/Parquet with a 'messages' "
                         "column): implies --train and dataset_format=sft, loss is computed on "
                         "assistant tokens only")
    ap.add_argument("--init-checkpoint", type=str, default=None,
                    help="initialize weights from a checkpoint without its optimizer state or "
                         "step count (SFT from a pretrained model); use --resume to continue a run")
    ap.add_argument("--val-dataset", type=str, default=None,
                    help="held-out dataset for eval logs; overrides val_dataset in the config")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--prepare", type=str, default=None, metavar="RAWDIR")
    ap.add_argument("--train-tokenizer", type=str, default=None, metavar="RAWDIR")
    ap.add_argument("--tokenizer-out", type=str, default="tokenizer.json")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--amp-dtype", type=str, default=None, choices=("bf16", "fp32", "off", "fp16"),
                    help="override yaml amp_dtype; fp32/off disables autocast")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--eval", action="store_true", help="full sequential eval on --dataset, or config/checkpoint val_dataset")
    ap.add_argument("--eval-batch-size", type=int, default=None, help="batch size for --eval; higher is faster but uses more VRAM")
    ap.add_argument("--eval-data-cache", type=str, default="ram", choices=("mmap", "ram", "gpu"), help="dataset staging for --eval")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--export-aoti", type=str, default=None, metavar="OUT.pt2",
                    help="export --checkpoint's forward pass as a self-contained "
                         "AOTInductor package (torch.export + AOTInductor -- the modern "
                         "replacement for the deprecated torch.jit.script/trace+ONNX "
                         "route). Loadable/runnable with torch._inductor.aoti_load_package(...) "
                         "and no Python model code at inference time.")
    ap.add_argument("--export-batch-size", type=int, default=1,
                    help="batch dimension baked into the --export-aoti graph "
                         "(torch.export needs a concrete shape)")
    ap.add_argument("--resume", type=str, default=None, help="smart resume: load model and optimizer state, continue step count/LR schedule, and apply the configured dataset cursor policy")
    ap.add_argument("--resume-step", type=int, default=None, help="override/hard-set the step to resume from, for checkpoints saved before 'step' was recorded (or to force a specific value)")
    ap.add_argument(
        "--resume-dataset-steps", type=int, default=None,
        help="one-time override for completed optimizer steps on the current "
             "dataset; seeds per-dataset progress in legacy checkpoints")
    ap.add_argument(
        "--resume-data", type=str, default=None,
        choices=("auto", "continue", "restart"),
        help="resume dataset cursor: auto restores per-dataset progress (or "
             "starts a new dataset at draw 0); continue uses legacy global-step "
             "replay when no saved entry exists; restart always starts at draw 0")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--ablation", action="store_true", help="diagnostic: skip imag/phase and use s=1")
    ap.add_argument(
        "--color", action="store_true",
        help="force ANSI colors even when stdout is redirected to a file "
             "(tail -f displays them in a terminal)")
    ap.add_argument("--quiet", action="store_true",
                    help="--train only: relaunch training as a detached background "
                         "process (fresh interpreter, own CUDA context -- safe, unlike "
                         "fork() after CUDA init) with stdout/stderr going to a "
                         "timestamped log file. Prints one 'tail -f ...' hint, then this "
                         "invocation exits immediately -- the terminal is free right "
                         "away, no nohup/tmux/& needed. Skipped (falls back to plain "
                         "in-place log redirection, no detach) if WORLD_SIZE is already "
                         "set -- i.e. we're already one rank of an existing torchrun "
                         "launch, where self-detaching per rank would break rendezvous.")
    args = ap.parse_args()

    if args.color:
        os.environ.pop("NO_COLOR", None)
        os.environ["FORCE_COLOR"] = "1"

    if args.smoke_test:
        smoke_test()
        return

    if args.quiet and args.train and "WORLD_SIZE" not in os.environ:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = f"log_{ts}.log"
        child_argv = [a for a in sys.argv[1:] if a != "--quiet"]
        log_f = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)] + child_argv,
            stdout=log_f, stderr=log_f, stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from this terminal's session/process group --
        )                             # Ctrl-C here won't reach it; that's the whole point.
        time.sleep(3.0)  # catch instant startup failures (bad path/yaml) before we vanish
        if proc.poll() is not None and proc.returncode != 0:
            print(f"[quiet] background process exited immediately (code {proc.returncode}) -- "
                  f"see {log_path} for why:")
            print(open(log_path, encoding="utf-8").read()[-2000:])
            raise SystemExit(proc.returncode)
        with open(f"{log_path}.pid", "w") as f:
            f.write(str(proc.pid))
        print(f"[quiet] training running in background, pid={proc.pid}")
        print(f"[quiet] use: tail -f {log_path}   (kill with: kill {proc.pid})")
        return

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.amp_dtype is not None:
        cfg.amp_dtype = args.amp_dtype
    if args.resume_data is not None:
        cfg.resume_data_stream = args.resume_data
    if args.val_dataset is not None:
        cfg.val_dataset = args.val_dataset
    device_pref = args.device if args.device is not None else cfg.device
    dev, distributed, world_size, rank, local_rank = maybe_launch_or_init_ddp(
        device_pref, training=bool(args.train or args.sft_dataset))
    if dev.type == "cuda":
        # The distributed setup returns directly after NCCL initialization
        # instead of passing through device_auto(), so configure every rank.
        configure_cuda_math()
    if dev.type == "cuda" and not distributed:
        idx = 0 if dev.index is None else int(dev.index)
        n_cuda = torch.cuda.device_count()
        if idx < 0 or idx >= n_cuda:
            raise RuntimeError(
                f"Requested {dev}, but only {n_cuda} CUDA device(s) are visible. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
                "If you set CUDA_VISIBLE_DEVICES=2, use --device cuda:0 inside this process."
            )
        torch.cuda.set_device(idx)
        dev = torch.device(f"cuda:{idx}")
    if distributed:
        global_batch_size = int(cfg.batch_size)
        if global_batch_size < world_size:
            raise ValueError(f"batch_size={global_batch_size} is global in DDP mode and must be >= world_size={world_size}")
        if global_batch_size % world_size != 0:
            raise ValueError(f"batch_size={global_batch_size} must be divisible by world_size={world_size} in DDP mode")
        cfg._global_batch_size = global_batch_size
        cfg.batch_size = global_batch_size // world_size
        if ddp_is_main():
            print(f"[ddp] world_size={world_size} backend=nccl")
            print(f"[ddp] batch_size(global)={global_batch_size} -> batch_size(local)={cfg.batch_size}")
            print(f"[ddp] grad_accum_steps={cfg.grad_accum_steps} -> effective_global_batch={global_batch_size * int(cfg.grad_accum_steps)}")
    else:
        cfg._global_batch_size = int(cfg.batch_size)

    if args.quiet and args.train:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = f"log_{ts}_rank{rank}.log"
        if ddp_is_main():
            print(f"[quiet] rank {rank}: logging to {log_path}  ·  use: tail -f {log_path}")
        log_f = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = log_f
        sys.stderr = log_f
        print(f"[quiet] --- log started {ts} (rank {rank}/{world_size}) ---")

    if args.train_tokenizer:
        train_tokenizer(
            args.train_tokenizer, args.vocab, args.tokenizer_out,
            dataset_format=getattr(cfg, "dataset_format", "auto"),
            text_field=getattr(cfg, "text_field", "text"),
        )
        return
    if args.prepare:
        build_tokenizer(cfg)
        apply_config(cfg)
        prepare(args.prepare, cfg, args.output or "prep.bin")
        return
    if args.train or args.sft_dataset:
        train_dataset = args.sft_dataset or args.dataset or cfg.train_dataset
        assert train_dataset, "--train needs --dataset or train_dataset in config"
        resume_path = args.resume if args.resume is not None else cfg.resume
        init_checkpoint = (
            args.init_checkpoint
            if args.init_checkpoint is not None
            else cfg.init_checkpoint
        )
        if args.sft_dataset:
            cfg.dataset_format = "sft"
        if is_sft_dataset(cfg):
            assert init_checkpoint or resume_path, (
                "SFT needs init_checkpoint in config/--init-checkpoint "
                "(the pretrained model to fine-tune), or --resume")
        checkpoint_out = (
            args.checkpoint
            if args.checkpoint is not None
            else (
                cfg.checkpoint
                if cfg.checkpoint is not None
                else ("loomformer.pt" if cfg.save_final_checkpoint else None)
            )
        )
        asyncio.run(train_async(
            cfg, train_dataset, dev, checkpoint_out,
            args.ablation, resume_path, args.resume_step,
            args.resume_dataset_steps, init_weights=init_checkpoint))
        return
    if args.export_aoti:
        assert args.checkpoint, "--export-aoti needs --checkpoint"
        export_aoti(cfg, args.checkpoint, args.export_aoti, dev, args.export_batch_size)
        return
    if args.infer:
        assert args.checkpoint, "--infer needs --checkpoint"
        infer(cfg, args.checkpoint, args.prompt, args.max_new, dev)
        return
    if args.eval:
        assert args.checkpoint, "--eval needs --checkpoint"
        eval_dataset = args.dataset or cfg.val_dataset
        eval_full(args.checkpoint, eval_dataset, dev, args.eval_batch_size, args.eval_data_cache)
        return
    ap.print_help()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if dist.is_available() and dist.is_initialized():
            # A rank may fail while a peer is blocked in a different NCCL
            # collective. Calling destroy_process_group() here waits for that
            # peer and hides the original exception forever. Print it before
            # any teardown and terminate this worker immediately; torchrun
            # will then stop the remaining ranks.
            traceback.print_exc()
            with contextlib.suppress(Exception):
                sys.stdout.flush()
                sys.stderr.flush()
            os._exit(1)
        raise
    else:
        # Every torchrun rank owns its process group. Explicit teardown avoids
        # relying on interpreter destruction after a successful, aligned run.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
