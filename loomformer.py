#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import faulthandler
import gc
import signal
import contextlib
import glob
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP

import tria  # sibling module, same directory -- see tria.py's own docstring


def configure_cuda_math() -> None:
    """Use one CUDA matmul policy in single-GPU and every DDP rank."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_auto(pref: Optional[str] = None) -> torch.device:
    dev = torch.device(pref) if pref else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        configure_cuda_math()
    return dev


def _parse_cuda_device_list(pref: str) -> Optional[List[int]]:
    """Parse a comma-separated CUDA device list, or return ``None`` for a scalar selector."""
    raw = str(pref or "").strip().lower().replace(" ", "")
    if "," not in raw:
        return None
    out: List[int] = []
    for part in raw.split(","):
        if not part:
            raise ValueError(f"bad CUDA device list {pref!r}")
        if part.startswith("cuda:"):
            part = part[len("cuda:"):]
        if not part.isdigit():
            raise ValueError(f"bad CUDA device list {pref!r}; expected e.g. cuda:0,cuda:1")
        idx = int(part)
        if idx < 0:
            raise ValueError(f"bad CUDA device index {idx} in {pref!r}")
        out.append(idx)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate CUDA device in {pref!r}")
    return out


def _cuda_visible_devices_for_child(indices: List[int]) -> str:
    parent = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if parent:
        entries = [x.strip() for x in parent.split(",") if x.strip()]
        if entries and all(0 <= i < len(entries) for i in indices):
            return ",".join(entries[i] for i in indices)
    return ",".join(str(i) for i in indices)


def _auto_omp_threads(nproc: int) -> int:
    total = os.cpu_count() or 1
    if nproc <= 0:
        return 1
    return max(1, min(8, total // nproc))


def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_is_distributed() -> bool:
    return ddp_world_size() > 1


def ddp_is_main() -> bool:
    return ddp_rank() == 0


def ddp_print(*args, **kwargs) -> None:
    if ddp_is_main():
        print(*args, **kwargs)


def ddp_trace(stage: str, *, step: Optional[int] = None,
              micro: Optional[int] = None) -> None:
    """Emit rank-local progress only for the installation hang diagnostic."""
    if os.environ.get("LOOM_DDP_TRACE") != "1" or not ddp_is_distributed():
        return
    location = ""
    if step is not None:
        location += f" step={int(step)}"
    if micro is not None:
        location += f" micro={int(micro)}"
    print(
        f"[ddp-trace] rank={ddp_rank()}{location} stage={stage}",
        file=sys.stderr,
        flush=True,
    )


def ddp_barrier(device: Optional[torch.device] = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl":
        idx = ddp_local_rank() if device is None or device.type == "cuda" else 0
        dist.barrier(device_ids=[int(idx)])
    else:
        dist.barrier()


def ddp_mean_float(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def ddp_sum_int(value: int, device: torch.device) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    t = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def ddp_unwrap_model(model: nn.Module) -> nn.Module:
    raw = model
    seen: set[int] = set()
    while id(raw) not in seen:
        seen.add(id(raw))
        is_fsdp = raw.__class__.__name__ == "FullyShardedDataParallel"
        if isinstance(raw, DDP) or is_fsdp:
            raw = raw.module
            continue
        if hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
            continue
        break
    return raw


def ddp_sync_mutable_buffers(model: nn.Module) -> None:
    """Match DDP buffer semantics outside the compiled forward graph.

    ``ParaplexFFN.beta_anchor`` is the only mutable model buffer. DDP's
    built-in ``broadcast_buffers`` runs from its forward pre-hook; putting that
    hook inside ``torch.compile(DDP)`` lets rank-local guard misses execute an
    extra NCCL broadcast while a cache-hit rank has already entered gradient
    all-reduce. One explicit coalesced broadcast before every outer forward
    preserves rank-0 anchor semantics with a fixed collective order.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    raw = ddp_unwrap_model(model)
    anchors = [
        block.ffn.beta_anchor
        for block in getattr(raw, "blocks", ())
        if hasattr(getattr(block, "ffn", None), "beta_anchor")
    ]
    if not anchors:
        return
    with torch.no_grad():
        packed = torch.stack(
            [anchor.detach().reshape(()) for anchor in anchors]
        )
        dist.broadcast(packed, src=0)
        for anchor, value in zip(anchors, packed.unbind()):
            anchor.copy_(value)


def ddp_static_graph_policy(cfg: "Config") -> Tuple[bool, str]:
    """Return whether DDP may safely reuse one fixed reducer-hook trace."""
    accum_is_single = (
        max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1)) == 1
    )
    compiled_with_eager_islands = bool(
        getattr(cfg, "compile", False)
        and bool(getattr(cfg, "tria_carry_enabled", False))
        and bool(getattr(cfg, "tria_temporal_enabled", True))
    )
    if compiled_with_eager_islands:
        return False, "compiled depth-replay eager island"
    if not accum_is_single:
        return False, "grad_accum_steps > 1 needs no_sync"
    return True, ""


def maybe_launch_or_init_ddp(device_pref: Optional[str], training: bool) -> Tuple[torch.device, bool, int, int, int]:
    # Calling a venv interpreter by absolute path does not activate the venv
    # and therefore does not put sibling console tools (notably ninja) on
    # PATH. PyTorch's extension loader still resolves ninja through PATH even
    # when the compiled module is already cached. Make direct single-process
    # and self-launched torchrun invocations equivalent to an activated venv.
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if interpreter_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join(
            [interpreter_bin, *path_entries]
        )

    pref = str(device_pref or "").strip().lower()
    cuda_subset = _parse_cuda_device_list(pref)
    wants_ddp_launch = pref == "cudas" or cuda_subset is not None
    if wants_ddp_launch and training and "WORLD_SIZE" not in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {pref or 'cudas'} requested but CUDA is unavailable")
        env = os.environ.copy()
        if cuda_subset is None:
            n = torch.cuda.device_count()
            launch_note = "all visible CUDA devices"
        else:
            if len(cuda_subset) < 2:
                raise RuntimeError(f"--device {pref!r} selects fewer than 2 CUDA devices")
            visible = _cuda_visible_devices_for_child(cuda_subset)
            env["CUDA_VISIBLE_DEVICES"] = visible
            n = len(cuda_subset)
            launch_note = f"CUDA_VISIBLE_DEVICES={visible}"
        if n < 2:
            raise RuntimeError(f"--device {pref or 'cudas'} requested but fewer than 2 CUDA GPUs are visible")
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(n), __file__] + sys.argv[1:]
        if not env.get("OMP_NUM_THREADS"):
            env["OMP_NUM_THREADS"] = str(_auto_omp_threads(n))
        print(f"[ddp] launching ({launch_note}):", " ".join(cmd), flush=True)
        if "OMP_NUM_THREADS" in env and not os.environ.get("OMP_NUM_THREADS"):
            print(f"[ddp] auto OMP_NUM_THREADS={env['OMP_NUM_THREADS']}", flush=True)
        raise SystemExit(subprocess.call(cmd, env=env))

    world_size = ddp_world_size()
    rank = ddp_rank()
    local_rank = ddp_local_rank()
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP world detected but CUDA is unavailable")
        if not os.environ.get("OMP_NUM_THREADS"):
            os.environ["OMP_NUM_THREADS"] = str(_auto_omp_threads(world_size))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
        if (
            os.environ.get("LOOM_DDP_TRACE") == "1"
            and hasattr(signal, "SIGUSR1")
        ):
            # The matrix runner sends SIGUSR1 only to rank worker processes,
            # immediately before terminating a timed-out cell. This records
            # every Python thread's exact blocking frame in the retained log.
            faulthandler.register(
                signal.SIGUSR1,
                file=sys.stderr,
                all_threads=True,
                chain=False,
            )
        ddp_print(f"[ddp] rank={rank} local_rank={local_rank} device={dev}", flush=True)
        return dev, True, world_size, rank, local_rank
    if pref == "cudas":
        # Non-training actions do not self-launch. Treat cudas as cuda:0 there.
        pref = "cuda:0"
    elif cuda_subset is not None:
        # Non-training actions do not self-launch. Use the first requested local GPU.
        pref = f"cuda:{cuda_subset[0]}"
    dev = device_auto(pref or None)
    return dev, False, 1, 0, 0

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


def _profile_region(name: str):
    # record_function is useful in eager profiling but Dynamo explicitly
    # ignores it and emits a warning. Keep compiled graphs free of profiler
    # marker objects and retain the markers for normal profiler runs.
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    return torch.profiler.record_function(name)


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
        return (
            f"LoomFormer [V={self.vocab} d_model={self.model_dim} qh={self.n_q_heads} "
            f"head_dim={hd} kvh={self.n_kv_heads} group={grp} "
            f"H={self.hidden} D={self.layers} T={self.seq_len} B={self.batch_size}]"
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


def should_replay_resume_data(
    cfg: Config, dataset: str, saved_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """Resolve whether resume should advance the dataset RNG/cursor."""
    policy = str(getattr(cfg, "resume_data_stream", "auto") or "auto").lower()
    if policy not in ("auto", "continue", "restart"):
        raise ValueError(
            "resume_data_stream must be auto, continue, or restart; "
            f"got {policy!r}")
    if policy == "continue":
        return True, "forced by resume_data_stream=continue"
    if policy == "restart":
        return False, "forced by resume_data_stream=restart"
    saved_dataset = saved_cfg.get("train_dataset")
    if not saved_dataset:
        return True, (
            "checkpoint has no train_dataset metadata; auto keeps legacy replay "
            "(use resume_data_stream=restart to force a fresh stream)"
        )
    same = os.path.abspath(str(saved_dataset)) == os.path.abspath(dataset)
    if same:
        return True, "train_dataset matches checkpoint"
    return False, f"train_dataset changed ({saved_dataset!r} -> {dataset!r})"


def dataset_progress_key(dataset: str) -> str:
    """Portable, human-readable key for per-dataset checkpoint progress."""
    return os.path.normpath(str(dataset))


def normalize_dataset_progress(blob: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Load the versioned per-dataset progress map from a checkpoint."""
    raw = blob.get("dataset_progress", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for raw_key, raw_value in raw.items():
        key = dataset_progress_key(str(raw_key))
        if isinstance(raw_value, dict):
            steps = max(0, int(raw_value.get("steps", 0) or 0))
            draws = max(0, int(raw_value.get("draws", steps) or 0))
        else:
            # Accept an early/simple {dataset: steps} representation.
            steps = max(0, int(raw_value or 0))
            draws = steps
        out[key] = {"steps": steps, "draws": draws}
    return out


def find_dataset_progress(
    progress: Dict[str, Dict[str, int]], dataset: str,
) -> Tuple[str, Optional[Dict[str, int]]]:
    """Find progress across harmless relative/absolute path spelling changes."""
    key = dataset_progress_key(dataset)
    if key in progress:
        return key, progress[key]
    current_abs = os.path.abspath(str(dataset))
    for saved_key, entry in progress.items():
        if os.path.abspath(saved_key) == current_abs:
            return saved_key, entry
    return key, None


def resolve_resume_dataset_progress(
    cfg: "Config",
    dataset: str,
    blob: Dict[str, Any],
    global_step: int,
    override_steps: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, int]], str, int, int, str]:
    """Resolve the current dataset's cursor without conflating it with global step."""
    progress = normalize_dataset_progress(blob)
    current_key = dataset_progress_key(dataset)
    saved_key, saved_entry = find_dataset_progress(progress, dataset)
    accum = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    policy = str(getattr(cfg, "resume_data_stream", "auto") or "auto").lower()
    if policy not in ("auto", "continue", "restart"):
        raise ValueError(
            "resume_data_stream must be auto, continue, or restart; "
            f"got {policy!r}")

    if override_steps is not None:
        steps = int(override_steps)
        if steps < 0:
            raise ValueError(
                f"resume_dataset_steps must be >= 0, got {override_steps}")
        draws = steps * accum
        reason = (
            f"forced current-dataset progress: steps={steps}, draws={draws}")
    elif policy == "restart":
        steps = draws = 0
        reason = "forced by resume_data_stream=restart"
    elif saved_entry is not None:
        steps = int(saved_entry["steps"])
        draws = int(saved_entry["draws"])
        reason = (
            f"restored saved progress for {current_key!r}: "
            f"steps={steps}, draws={draws}")
    elif progress and policy == "auto":
        steps = draws = 0
        reason = f"no saved progress for new dataset {current_key!r}"
    else:
        # Old checkpoints have no per-dataset map. Preserve their existing
        # auto/continue/restart behavior as a one-time migration fallback.
        replay, legacy_reason = should_replay_resume_data(
            cfg, dataset, blob.get("cfg", {}))
        steps = max(0, int(global_step)) if replay else 0
        draws = steps * accum
        reason = (
            f"legacy checkpoint fallback ({legacy_reason}): "
            f"steps={steps}, draws={draws}")

    if saved_key != current_key:
        progress.pop(saved_key, None)
    progress[current_key] = {"steps": steps, "draws": draws}
    return progress, current_key, steps, draws, reason


def checkpoint_tokens_seen(blob: Dict[str, Any], completed_steps: int) -> Tuple[int, bool]:
    """Return cumulative processed tokens and whether the value was exact."""
    saved = blob.get("tokens_seen")
    if saved is not None:
        return max(0, int(saved)), True
    saved_cfg = blob.get("cfg", {})
    batch = int(saved_cfg.get("batch_size", 0) or 0)
    seq_len = int(saved_cfg.get("seq_len", 0) or 0)
    accum = max(1, int(saved_cfg.get("grad_accum_steps", 1) or 1))
    if batch <= 0 or seq_len <= 0:
        return 0, False
    return max(0, int(completed_steps)) * batch * seq_len * accum, False


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
ATTN_SDPA_COMPUTE_DTYPE = "auto"
ATTN_SDPA_VALUE_FUSION = True
ATTN_SDPA_RECOMPUTE_BACKWARD = True
_sdpa_bf16_efficient_cache: Dict[Tuple[int, int, int], bool] = {}
_flash_varlen_func = None
_flash_varlen_import_tried = False
_flash_varlen_import_error: Optional[str] = None
_flash_deterministic_kw_supported: Optional[bool] = None
_flash_backend_cache: Dict[Tuple[int, torch.dtype, int], bool] = {}
_flash_value_fusion_cache: Dict[Tuple[int, torch.dtype, int], bool] = {}
_flash_probe_errors: Dict[Tuple[int, torch.dtype, int], str] = {}
_te_backend_cache: Dict[Tuple[int, torch.dtype, int], bool] = {}
_te_value_fusion_cache: Dict[Tuple[int, torch.dtype, int], bool] = {}
_te_probe_errors: Dict[Tuple[int, torch.dtype, int], str] = {}
_te_dpa_modules: Dict[Tuple[int, int, int, int, int], nn.Module] = {}
_attention_backend_reported: set = set()
_REAL_STDOUT = sys.stdout
_activation_checkpoint_tls = threading.local()
ROPE_THETA = 10000.0
ROPE_FACTOR = 4.0
ROPE_ORIGINAL_SEQ_LEN = 0
ROPE_BETA_FAST = 32.0
ROPE_BETA_SLOW = 1.0
ROPE_ATTENTION_FACTOR: Optional[float] = None
DEEPNORM_BETA = 1.0
FANIN_GAIN = 0.88


def _checkpoint_anchor_override(module: nn.Module) -> Optional[torch.Tensor]:
    overrides = getattr(_activation_checkpoint_tls, "anchor_overrides", None)
    return None if overrides is None else overrides.get(id(module))


@contextlib.contextmanager
def _activation_checkpoint_recompute_context(holder: dict):
    previous = getattr(_activation_checkpoint_tls, "anchor_overrides", None)
    overrides = holder.get("anchor_overrides")
    if overrides is None:
        raise RuntimeError("activation-checkpoint anchor snapshots were not captured")
    _activation_checkpoint_tls.anchor_overrides = overrides
    try:
        yield
    finally:
        _activation_checkpoint_tls.anchor_overrides = previous


def fanin_std(fan_in: int, gain: float = FANIN_GAIN) -> float:
    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive, got {fan_in}")
    return gain / math.sqrt(float(fan_in))


def residual_std(fan_in: int, gain: float = FANIN_GAIN) -> float:
    beta = DEEPNORM_BETA if RESIDUAL_INIT == "beta" else 1.0
    return fanin_std(fan_in, gain) * beta


def fixed_rms(x: torch.Tensor, target: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Scale each last-dimension vector by ``target / sqrt(mean(x²) + eps)``."""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    scale = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps) * float(target)
    return x * scale.to(dtype=x.dtype)


def capped_rms(x: torch.Tensor, maximum: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS scaling up to ``maximum`` without amplifying the input."""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rms = torch.sqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    scale = (float(maximum) / rms).clamp(max=1.0)
    return x * scale.to(dtype=x.dtype)


def init_linear_fanin(m: nn.Linear, gain: float = FANIN_GAIN, zero_bias: bool = True) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=fanin_std(m.weight.shape[1], gain))
    if zero_bias and m.bias is not None:
        nn.init.zeros_(m.bias)


def init_linear_residual(m: nn.Linear, gain: float = FANIN_GAIN, zero_bias: bool = True) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=residual_std(m.weight.shape[1], gain))
    if zero_bias and m.bias is not None:
        nn.init.zeros_(m.bias)


def init_embedding_fanin(m: nn.Embedding, gain: float = FANIN_GAIN) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=fanin_std(m.embedding_dim, gain))


def apply_config(cfg: Config) -> None:
    global N, N_Q_HEADS, N_KV_HEADS, HIDDEN, LAYERS, VOCAB, SEQ_LEN
    global HEAD_DIM, GQA_GROUP_SIZE, KV_DIM, HIDDEN_PER_Q_HEAD, IMAG_IN, PHASE_SECTORS, ATTN_IMPL, ATTN_SDPA_COMPUTE_DTYPE, ATTN_SDPA_VALUE_FUSION, ATTN_SDPA_RECOMPUTE_BACKWARD, RESIDUAL_INIT, DEPTH_ATTN_READOUT, DEPTH_ATTN_QKV_RMS, RESIDUAL_BRANCH_RMS_CAP, ACTIVATION, POWLU_M, PHASE_GRAD_FLOOR, PHASE_GRAD_MODE, USE_CUDA_PHASE_SIN, USE_CUDA_BETA_SPACE, USE_CUDA_PVPOWLU, USE_CUDA_DEPTH_ATTN, AMP_DTYPE, GRAD_CHECKPOINTING, TRIA_CARRY_ENABLED, TRIA_GAMMA_MAX, TRIA_RAW_GAMMA_INIT, TRIA_TEMPORAL_ENABLED, TIED_EMBEDDINGS, PARAPLEX_GATE_PROJ, FINAL_NORM_ENABLED, FUSED_LINEAR_CE, FUSED_LINEAR_CE_CHUNK_SIZE
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
    ATTN_IMPL = str(getattr(cfg, "attn_impl", "auto") or "auto").lower()
    if ATTN_IMPL not in ("auto", "flash", "sdpa", "manual"):
        raise ValueError(
            f"attn_impl must be 'auto', 'flash', 'sdpa' or 'manual', got {ATTN_IMPL!r}")
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
        graph_helper.set_runtime_not_required(
            {
                "tria_init", "tria_init_gate", "tria_step", "tria_step_gate",
                "gate_slot_mix", "temporal_carry", "phase_sin",
                "phase_sin_secant", "pvpowlu", "depth_attn", "beta_space",
            }
            if replay_stack_is_eager
            else set()
        )
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

class ByteTokenizer:
    vocab_size = 256

    def encode(self, s: str) -> List[int]:
        return list(s.encode("utf-8"))

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [list(s.encode("utf-8")) for s in texts]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class BPETokenizerWrap:
    def __init__(self, tk):
        self.tk = tk
        self.vocab_size = tk.get_vocab_size()

    def encode(self, s: str) -> List[int]:
        return self.tk.encode(s).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self.tk.encode_batch(texts)]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def special_id(self, token: str) -> Optional[int]:
        return self.tk.token_to_id(token)

    @staticmethod
    def load(path: str) -> "BPETokenizerWrap":
        from tokenizers import Tokenizer
        return BPETokenizerWrap(Tokenizer.from_file(path))


DEFAULT_SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>",
    "<|im_start|>", "<|im_end|>",                      
    "<think>", "</think>",                            
    "<tool_call>", "</tool_call>",                     
    "<tool_response>", "</tool_response>",            
    "<CARRY>",                                          
]


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

class TokenStream:
    def __init__(self, path: str, cfg: Config, device: torch.device, bos_id: Optional[int] = None):
        cache = str(getattr(cfg, "dataset_cache", "mmap") or "mmap").lower()
        if cache not in ("mmap", "ram"):
            raise ValueError(f"dataset_cache must be 'mmap' or 'ram', got {cache!r}")
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        self.data = np.array(mm, dtype=np.uint16, copy=True) if cache == "ram" else mm
        self.cfg = cfg
        self.device = device
        self._bos_id = bos_id
        if len(self.data) <= cfg.seq_len + 1:
            raise ValueError(f"dataset too short: {len(self.data)} tokens for seq_len={cfg.seq_len}")

    def _sample_batch(self) -> torch.Tensor:
        B, T = self.cfg.batch_size, self.cfg.seq_len
        content_need = T + 1 - (1 if self._bos_id is not None else 0)
        ix = np.random.randint(0, len(self.data) - content_need - 1, size=B)
        rows = [self.data[i : i + content_need].astype(np.int64) for i in ix]
        if self._bos_id is not None:
            rows = [np.concatenate(([self._bos_id], r)) for r in rows]
        xb = np.stack(rows)
        return torch.from_numpy(xb)

    def sample_device_batch(self) -> torch.Tensor:
        b = self._sample_batch()
        if self.device.type == "cuda":
            b = b.pin_memory()
        return b.to(self.device, non_blocking=True)

    async def _produce(self, queue: "asyncio.Queue", n: int):
        loop = asyncio.get_event_loop()
        for _ in range(n):
            batch = await loop.run_in_executor(None, self._sample_batch)
            await queue.put(batch.pin_memory() if self.device.type == "cuda" else batch)
        await queue.put(None)

    async def batches(self, n: int):
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(getattr(self.cfg, "prefetch_batches", 256))))
        producer = asyncio.create_task(self._produce(queue, n))
        while True:
            batch = await queue.get()
            if batch is None:
                break
            yield batch.to(self.device, non_blocking=True)
        await producer


class RawCorpus:

    _EXTS = {"txt": (".txt",), "jsonl": (".jsonl", ".ndjson"),
             "parquet": (".parquet",), "arrow": (".arrow", ".feather")}

    def __init__(self, path: str, fmt: str = "auto", text_field: str = "text"):
        self.text_field = text_field
        files = self._resolve_files(path, fmt)
        if not files:
            raise ValueError(f"no corpus files found at {path!r} (format={fmt!r})")
        self.fmt = fmt if fmt != "auto" else self._infer_format(files[0])
        self._files = files
        self._cache: Dict[Tuple[str, int], object] = {}
        docs: List[Tuple[int, object, int]] = []  # (file_idx, row/offset key, char length)
        indexer = {"txt": self._index_txt, "jsonl": self._index_jsonl,
                   "parquet": self._index_parquet, "arrow": self._index_arrow}[self.fmt]
        for fi, p in enumerate(files):
            docs.extend(indexer(fi, p))
        if not docs:
            raise ValueError(f"corpus at {path!r} indexed to zero documents")
        self._docs = docs
        self._cum = np.cumsum([d[2] for d in docs])
        self.total_chars = int(self._cum[-1])

    def __len__(self) -> int:
        return len(self._docs)

    def iter_texts(self):
        for fi, key, length in self._docs:
            txt = self._read_doc_text(fi, key, length)
            if txt:
                yield txt

    @staticmethod
    def _resolve_files(path: str, fmt: str) -> List[str]:
        if os.path.isfile(path):
            return [path]
        exts = RawCorpus._EXTS.get(fmt, [e for v in RawCorpus._EXTS.values() for e in v])
        found = []
        for e in exts:
            found.extend(glob.glob(os.path.join(path, f"*{e}")))
        return sorted(set(found))

    @staticmethod
    def _infer_format(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        for fmt, exts in RawCorpus._EXTS.items():
            if ext in exts:
                return fmt
        raise ValueError(f"cannot infer corpus format from extension {ext!r} ({path!r})")

    def _index_txt(self, fi: int, p: str) -> List[Tuple[int, None, int]]:
        return [(fi, None, os.path.getsize(p))]

    def _index_jsonl(self, fi: int, p: str) -> List[Tuple[int, int, int]]:
        out = []
        with open(p, "rb") as f:
            offset = 0
            for line in f:
                s = line.decode("utf-8", errors="replace").strip()
                if s:
                    try:
                        txt = json.loads(s).get(self.text_field, "")
                        if txt:
                            out.append((fi, offset, len(txt)))
                    except Exception:
                        pass
                offset += len(line)
        return out

    def _index_parquet(self, fi: int, p: str) -> List[Tuple[int, int, int]]:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        pf = pq.ParquetFile(p)
        out = []
        row_base = 0
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=[self.text_field]).column(self.text_field)
            lens = pc.utf8_length(col).to_numpy(zero_copy_only=False)
            out.extend((fi, row_base + i, int(l)) for i, l in enumerate(lens) if l > 0)
            row_base += len(lens)
        return out

    def _index_arrow(self, fi: int, p: str) -> List[Tuple[int, int, int]]:
        import pyarrow.compute as pc
        # HuggingFace datasets usually store .arrow shards as Arrow IPC STREAMS,
        # while pyarrow/Feather-style files are Arrow IPC FILES with a footer.
        # The extension alone does not distinguish them, so use the shared reader
        # that tries open_file first and falls back to open_stream.
        table, _container = _read_arrow_table_with_container(p)
        col = table.column(self.text_field)
        lens = pc.utf8_length(col).to_numpy(zero_copy_only=False)
        return [(fi, i, int(l)) for i, l in enumerate(lens) if l > 0]

    def _read_doc_text(self, fi: int, key, length: int) -> str:
        p = self._files[fi]
        if self.fmt == "txt":
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        if self.fmt == "jsonl":
            with open(p, "rb") as f:
                f.seek(key)
                line = f.readline()
            return json.loads(line.decode("utf-8", errors="replace")).get(self.text_field, "")
        if self.fmt in ("parquet", "arrow"):
            ck = ("table", fi)
            table = self._cache.get(ck)
            if table is None:
                if self.fmt == "parquet":
                    import pyarrow.parquet as pq
                    table = pq.read_table(p, columns=[self.text_field])
                else:
                    table, _container = _read_arrow_table_with_container(p)
                self._cache[ck] = table
            return str(table.column(self.text_field)[key])
        raise ValueError(self.fmt)

    def iter_sampled_texts(self, docs):
        """Yield sampled texts, reading only the required Parquet row groups."""
        if self.fmt != "parquet":
            for fi, key, length in docs:
                yield self._read_doc_text(fi, key, length)
            return

        import pyarrow.parquet as pq

        by_file: Dict[int, List[int]] = {}
        for fi, key, _length in docs:
            by_file.setdefault(fi, []).append(int(key))

        for fi, row_indices in by_file.items():
            pf = pq.ParquetFile(self._files[fi])
            rows = iter(sorted(row_indices))
            wanted = next(rows, None)
            row_base = 0
            for rg in range(pf.num_row_groups):
                row_count = pf.metadata.row_group(rg).num_rows
                row_end = row_base + row_count
                if wanted is not None and wanted < row_end:
                    col = pf.read_row_group(rg, columns=[self.text_field]).column(self.text_field)
                    while wanted is not None and wanted < row_end:
                        yield str(col[wanted - row_base])
                        wanted = next(rows, None)
                row_base = row_end
                if wanted is None:
                    break

    def sample_window_spans(self, min_chars: int, rng: np.random.Generator) -> List[str]:
        pos = int(rng.integers(0, self.total_chars))
        doc_i = int(np.searchsorted(self._cum, pos, side="right"))
        doc_i = min(doc_i, len(self._docs) - 1)
        fi, key, length = self._docs[doc_i]
        prev_cum = self._cum[doc_i] - length
        start = max(0, pos - prev_cum)
        spans = [self._read_doc_text(fi, key, length)[start:]]
        total = len(spans[0])
        j = doc_i + 1
        while total < min_chars and j < len(self._docs):
            fi2, key2, length2 = self._docs[j]
            text2 = self._read_doc_text(fi2, key2, length2)
            spans.append(text2)
            total += len(text2)
            j += 1
        return spans


def _tok_special_id(tok, name: str) -> Optional[int]:
    fn = getattr(tok, "special_id", None)
    return fn(name) if fn is not None else None


class ChatTemplate:
    def __init__(self, tok, template_path: str = "chat_template.jinja"):
        import jinja2  # lazy: only chat-template users need this dependency at all
        self.tok = tok
        resolved = template_path
        if not os.path.isfile(resolved):
            # Fall back to a path next to this module, so callers don't need to be
            # run from the repo root for the default filename to resolve.
            candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), template_path)
            if os.path.isfile(candidate):
                resolved = candidate
        with open(resolved, "r", encoding="utf-8") as f:
            src = f.read()
        self._tpl = jinja2.Environment().from_string(src)

        im_start = _tok_special_id(tok, "<|im_start|>")
        im_end = _tok_special_id(tok, "<|im_end|>")
        if im_start is None or im_end is None:
            raise ValueError(
                "ChatTemplate needs <|im_start|>/<|im_end|> in the tokenizer's vocab "
                "(see loomformer.DEFAULT_SPECIAL_TOKENS) -- retrain it with those "
                "special tokens if this one predates them."
            )
        self.im_start_id, self.im_end_id = im_start, im_end
        self.bos_id = _tok_special_id(tok, "<bos>")
        self.bos_token = "<bos>" if self.bos_id is not None else ""
        eos_id = _tok_special_id(tok, "<eos>")
        self.stop_ids = {i for i in (im_end, eos_id) if i is not None}
        self._assistant_header_ids = [im_start] + tok.encode("assistant\n")

    def render_text(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                     add_generation_prompt: bool = False) -> str:
        kwargs = {"messages": messages, "add_generation_prompt": add_generation_prompt,
                  "bos_token": self.bos_token}
        if tools is not None:  # presence vs absence is meaningful -- see sft.md
            kwargs["tools"] = tools
        return self._tpl.render(**kwargs)

    def render_prompt_ids(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> List[int]:
        return self.tok.encode(self.render_text(messages, tools=tools, add_generation_prompt=True))

    @staticmethod
    def _find_all(haystack: List[int], needle: List[int]) -> List[int]:
        if not needle:
            return []
        out: List[int] = []
        n, m = len(haystack), len(needle)
        i = 0
        while i <= n - m:
            if haystack[i:i + m] == needle:
                out.append(i)
                i += m
            else:
                i += 1
        return out

    def render_training_ids(self, messages: List[Dict], tools: Optional[List[Dict]] = None
                             ) -> Tuple[List[int], List[int]]:
        ids = self.tok.encode(self.render_text(messages, tools=tools, add_generation_prompt=False))
        mask = [0] * len(ids)
        for p in self._find_all(ids, self._assistant_header_ids):
            start = p + len(self._assistant_header_ids)
            q = start
            while q < len(ids) and ids[q] != self.im_end_id:
                q += 1
            end = min(q, len(ids) - 1)  # include the closing <|im_end|> itself
            for k in range(start, end + 1):
                mask[k] = 1
        return ids, mask

    def parse_tool_calls(self, text: str) -> List[Dict]:
        calls: List[Dict] = []
        i = 0
        while True:
            s = text.find("<tool_call>", i)
            if s < 0:
                break
            e = text.find("</tool_call>", s)
            if e < 0:
                break
            payload = text[s + len("<tool_call>"):e].strip()
            i = e + len("</tool_call>")
            try:
                obj = json.loads(payload)
                calls.append({"id": f"call_{len(calls)}", "type": "function",
                              "function": {"name": obj.get("name"), "arguments": obj.get("arguments")}})
            except Exception:
                continue
        return calls


def _auto_val_split_pct(cfg: Config) -> float:
    return float(getattr(cfg, "auto_val_split_pct", 0.0) or 0.0)


def _split_count(n: int, pct: float) -> int:
    if n <= 1:
        return 0
    k = int(round(n * (pct / 100.0)))
    if k <= 0:
        k = 1
    return min(k, n - 1)


def _concat_arrow_tables(tables):
    import pyarrow as pa
    if not tables:
        raise ValueError("auto val split produced no validation rows")
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def _read_arrow_table_with_container(path: str):
    import pyarrow as pa
    ext = os.path.splitext(path)[1].lower()
    if ext == ".feather":
        import pyarrow.feather as feather
        return feather.read_table(path), "feather"
    with pa.memory_map(path, "rb") as src:
        try:
            return pa.ipc.open_file(src).read_all(), "file"
        except Exception:
            src.seek(0)
            return pa.ipc.open_stream(src).read_all(), "stream"


def _write_arrow_table_preserving_container(path: str, table, container: str) -> None:
    import pyarrow as pa
    tmp = path + ".tmp"
    if container == "feather":
        import pyarrow.feather as feather
        feather.write_feather(table, tmp)
    else:
        with pa.OSFile(tmp, "wb") as sink:
            writer_fn = pa.ipc.new_stream if container == "stream" else pa.ipc.new_file
            with writer_fn(sink, table.schema) as writer:
                writer.write_table(table)
    os.replace(tmp, path)


def _write_arrow_ipc_file(path: str, table) -> None:
    import pyarrow as pa
    tmp = path + ".tmp"
    with pa.OSFile(tmp, "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    os.replace(tmp, path)


def _auto_split_arrow(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    val_tables = []
    train_rows = val_rows = 0
    for path in files:
        table, container = _read_arrow_table_with_container(path)
        n = int(table.num_rows)
        k = _split_count(n, pct)
        if k <= 0:
            train_rows += n
            continue
        train = table.slice(0, n - k)
        val = table.slice(n - k, k)
        _write_arrow_table_preserving_container(path, train, container)
        val_tables.append(val)
        train_rows += int(train.num_rows)
        val_rows += int(val.num_rows)
    _write_arrow_ipc_file(val_path, _concat_arrow_tables(val_tables))
    return {"train_rows": train_rows, "val_rows": val_rows}


def _auto_split_parquet(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    train_rows = val_rows = 0
    val_tmp = val_path + ".tmp"
    train_tmps: List[Tuple[str, str]] = []
    val_writer = None
    val_schema = None
    try:
        for path in files:
            pf = pq.ParquetFile(path)
            schema = pf.schema_arrow
            if val_schema is None:
                val_schema = schema
                val_writer = pq.ParquetWriter(val_tmp, schema, compression="zstd")
            elif schema != val_schema:
                raise ValueError(
                    "auto_val_split_pct requires identical Parquet schemas; "
                    f"{path!r} differs from the first shard")

            n = int(pf.metadata.num_rows)
            k = _split_count(n, pct)
            cut = n - k
            tmp = path + ".tmp"
            train_tmps.append((tmp, path))
            train_writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            cursor = 0
            try:
                for batch in pf.iter_batches(batch_size=8192):
                    table = pa.Table.from_batches([batch])
                    end = cursor + int(table.num_rows)
                    train_count = max(0, min(end, cut) - cursor)
                    if train_count:
                        train_writer.write_table(table.slice(0, train_count))
                        train_rows += train_count
                    val_count = int(table.num_rows) - train_count
                    if val_count:
                        assert val_writer is not None
                        val_writer.write_table(table.slice(train_count, val_count))
                        val_rows += val_count
                    cursor = end
            finally:
                train_writer.close()
        if val_writer is None or val_rows <= 0:
            raise ValueError("auto val split produced no validation rows")
        val_writer.close()
        val_writer = None
        for tmp, path in train_tmps:
            os.replace(tmp, path)
        os.replace(val_tmp, val_path)
    except BaseException:
        if val_writer is not None:
            val_writer.close()
        for tmp, _path in train_tmps:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(val_tmp)
        raise
    return {"train_rows": train_rows, "val_rows": val_rows}


def _auto_split_jsonl(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    train_rows = val_rows = 0
    tmp_val = val_path + ".tmp"
    with open(tmp_val, "wb") as vf:
        for path in files:
            with open(path, "rb") as f:
                lines = f.readlines()
            n = len(lines)
            k = _split_count(n, pct)
            train_lines = lines[: n - k]
            val_lines = lines[n - k :] if k > 0 else []
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.writelines(train_lines)
            os.replace(tmp, path)
            vf.writelines(val_lines)
            train_rows += len(train_lines)
            val_rows += len(val_lines)
    os.replace(tmp_val, val_path)
    return {"train_rows": train_rows, "val_rows": val_rows}


def _auto_split_txt(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    train_bytes = val_bytes = 0
    tmp_val = val_path + ".tmp"
    with open(tmp_val, "wb") as vf:
        for path in files:
            with open(path, "rb") as f:
                data = f.read()
            n = len(data)
            if n <= 1:
                train = data
                val = b""
            else:
                k = max(1, int(round(n * (pct / 100.0))))
                k = min(k, n - 1)
                cut = n - k
                nl = data.rfind(b"\n", 0, cut)
                if nl > 0:
                    cut = nl + 1
                train, val = data[:cut], data[cut:]
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(train)
            os.replace(tmp, path)
            vf.write(val)
            train_bytes += len(train)
            val_bytes += len(val)
    os.replace(tmp_val, val_path)
    return {"train_bytes": train_bytes, "val_bytes": val_bytes}


def _auto_split_bin(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    train_tokens = val_tokens = 0
    vals = []
    for path in files:
        arr = np.fromfile(path, dtype=np.uint16)
        n = int(arr.shape[0])
        k = _split_count(n, pct)
        train = arr[: n - k]
        val = arr[n - k :] if k > 0 else arr[:0]
        train.tofile(path + ".tmp")
        os.replace(path + ".tmp", path)
        vals.append(val)
        train_tokens += int(train.shape[0])
        val_tokens += int(val.shape[0])
    np.concatenate(vals).astype(np.uint16, copy=False).tofile(val_path)
    return {"train_tokens": train_tokens, "val_tokens": val_tokens}


def maybe_auto_val_split(cfg: Config, dataset: str) -> Optional[str]:
    """Return a validation path, optionally cutting shard tails into a new split."""
    if cfg.val_dataset:
        return str(cfg.val_dataset)
    pct = _auto_val_split_pct(cfg)
    if pct <= 0.0:
        return None
    if not (0.0 < pct < 100.0):
        raise ValueError(f"auto_val_split_pct must be in (0,100), got {pct}")
    if not os.path.isdir(dataset):
        raise ValueError("auto_val_split_pct only works when the training dataset is a directory of shards")

    fmt = str(getattr(cfg, "dataset_format", "auto") or "auto").lower()
    if is_sft_dataset(cfg):
        # SFT directories are Parquet-only even if the Hub snapshot also ships
        # a JSONL export of the same examples. Never train on both duplicates.
        files = sorted(glob.glob(os.path.join(dataset, "*.parquet")))
        inferred = "parquet"
    else:
        files = RawCorpus._resolve_files(dataset, fmt)
        inferred = fmt if fmt != "auto" else (
            RawCorpus._infer_format(files[0]) if files else "")
    if not files:
        raise ValueError(f"auto_val_split_pct found no top-level corpus files in {dataset!r} (format={fmt!r})")
    for path in files:
        if RawCorpus._infer_format(path) != inferred:
            raise ValueError("auto_val_split_pct requires one dataset format per folder; found mixed extensions")

    ext = {"arrow": ".arrow", "parquet": ".parquet", "jsonl": ".jsonl", "txt": ".txt", "bin": ".bin"}.get(inferred)
    if ext is None:
        raise ValueError(f"auto_val_split_pct unsupported dataset_format={inferred!r}")
    val_dir = os.path.join(dataset, "val")
    os.makedirs(val_dir, exist_ok=True)
    val_path = os.path.join(val_dir, "val_split" + ext)
    manifest_path = val_path + ".manifest.json"
    if os.path.exists(val_path):
        ddp_print(f"[auto-val-split] existing {val_path}; using it and NOT splitting again")
        cfg.val_dataset = val_path
        return val_path

    ddp_print(f"[auto-val-split] destructive split: pct={pct:.6g}% format={inferred} files={len(files)} -> {val_path}")
    if inferred == "arrow":
        stats = _auto_split_arrow(files, val_path, pct)
    elif inferred == "parquet":
        stats = _auto_split_parquet(files, val_path, pct)
    elif inferred == "jsonl":
        stats = _auto_split_jsonl(files, val_path, pct)
    elif inferred == "txt":
        stats = _auto_split_txt(files, val_path, pct)
    elif inferred == "bin":
        stats = _auto_split_bin(files, val_path, pct)
    else:
        raise ValueError(f"auto_val_split_pct unsupported dataset_format={inferred!r}")
    manifest = {"dataset": os.path.abspath(dataset), "format": inferred, "pct": pct,
                "files": files, "val_dataset": val_path, "stats": stats}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    ddp_print(f"[auto-val-split] done: {stats} | manifest={manifest_path}")
    cfg.val_dataset = val_path
    return val_path


def _encode_batch_any(tok, texts: List[str]) -> List[List[int]]:
    if hasattr(tok, "encode_batch"):
        return tok.encode_batch(texts)
    return tok(texts)["input_ids"]


class ShardStream:
    _CHUNK_DOCS = 20000

    def __init__(self, path: str, cfg: Config, device: torch.device, tokenizer=None):
        self.cfg = cfg
        self.device = device
        self.tok = tokenizer if tokenizer is not None else build_tokenizer(cfg)
        self._bos_id = _tok_special_id(self.tok, "<bos>")
        self._eos_id = _tok_special_id(self.tok, "<eos>")
        fmt_cfg = str(getattr(cfg, "dataset_format", "auto") or "auto")
        self._files = RawCorpus._resolve_files(path, fmt_cfg)
        if not self._files:
            raise ValueError(f"no corpus files found at {path!r} (format={fmt_cfg!r})")
        self._fmt = fmt_cfg if fmt_cfg != "auto" else RawCorpus._infer_format(self._files[0])
        self._text_field = getattr(cfg, "text_field", "text")
        self._rank = ddp_rank() if ddp_is_distributed() else 0
        self._world_size = ddp_world_size() if ddp_is_distributed() else 1
        row_counts = [self._file_row_count(path) for path in self._files]
        total_rows = int(sum(row_counts))
        global_start = total_rows * self._rank // self._world_size
        global_end = total_rows * (self._rank + 1) // self._world_size
        self._row_plan: List[Tuple[int, int, int]] = []
        cursor = 0
        for file_index, nrows in enumerate(row_counts):
            a = max(global_start, cursor) - cursor
            b = min(global_end, cursor + nrows) - cursor
            if a < b:
                self._row_plan.append((file_index, int(a), int(b)))
            cursor += nrows
        self._assigned_rows = global_end - global_start
        if self._assigned_rows <= 0 or not self._row_plan:
            raise ValueError(f"rank {self._rank} received no rows")
        self._content_need = int(cfg.seq_len) + 1 - (1 if self._bos_id is not None else 0)
        self._ram_queue: queue.Queue = queue.Queue(maxsize=max(1, int(cfg.prefetch_batches)))
        self._stop = threading.Event()
        self._producer_error = None
        self._gpu_batches = None
        self._gpu_pos = 0
        self._producer = threading.Thread(target=self._produce_cpu_batches, daemon=True, name=f"data-rank-{self._rank}")
        self._producer.start()
        plan = ", ".join(f"{os.path.basename(self._files[i])}[{a}:{b}]" for i, a, b in self._row_plan)
        if ddp_is_distributed():
            plans = [None] * self._world_size
            dist.all_gather_object(plans, (self._rank, self._assigned_rows, plan))
            if ddp_is_main():
                for rank, rows, desc in sorted(plans):
                    print(f"[data] rank={rank} rows={rows:,} plan={desc}", flush=True)
        else:
            print(f"[data] rank=0 rows={self._assigned_rows:,} plan={plan}", flush=True)

    def _file_row_count(self, path: str) -> int:
        if self._fmt == "parquet":
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)
        if self._fmt == "arrow":
            table, _ = _read_arrow_table_with_container(path)
            n = int(table.num_rows)
            del table
            return n
        if self._fmt == "jsonl":
            with open(path, "rb") as f:
                return sum(1 for line in f if line.strip())
        if self._fmt == "txt":
            return 1
        raise ValueError(f"unsupported raw dataset format {self._fmt!r}")

    def _iter_text_chunks(self, path: str, row_start: int, row_end: int):
        if self._fmt in ("parquet", "arrow"):
            if self._fmt == "parquet":
                import pyarrow.parquet as pq
                table = pq.read_table(path, columns=[self._text_field])
            else:
                table, _ = _read_arrow_table_with_container(path)
            col = table.column(self._text_field).slice(row_start, row_end - row_start)
            for start in range(0, len(col), self._CHUNK_DOCS):
                yield col.slice(start, min(self._CHUNK_DOCS, len(col) - start)).to_pylist()
            return
        if self._fmt == "jsonl":
            chunk = []
            row = 0
            with open(path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if row >= row_end:
                        break
                    if row >= row_start:
                        chunk.append(json.loads(line.decode("utf-8", errors="replace")).get(self._text_field, ""))
                        if len(chunk) >= self._CHUNK_DOCS:
                            yield chunk
                            chunk = []
                    row += 1
            if chunk:
                yield chunk
            return
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield [f.read()]

    def _produce_cpu_batches(self) -> None:
        try:
            carry = np.zeros(0, dtype=np.int64)
            eos = np.array([self._eos_id], dtype=np.int64) if self._eos_id is not None else None
            first_doc = True
            while not self._stop.is_set():
                for file_index, row_start, row_end in self._row_plan:
                    for texts in self._iter_text_chunks(self._files[file_index], row_start, row_end):
                        encoded = _encode_batch_any(self.tok, texts)
                        pieces = []
                        for ids in encoded:
                            if not first_doc and eos is not None:
                                pieces.append(eos)
                            pieces.append(np.asarray(ids, dtype=np.int64))
                            first_doc = False
                        if pieces:
                            block = np.concatenate(pieces)
                            carry = np.concatenate((carry, block)) if carry.size else block
                        batch_tokens = int(self.cfg.batch_size) * self._content_need
                        while carry.size >= batch_tokens and not self._stop.is_set():
                            block = carry[:batch_tokens].reshape(int(self.cfg.batch_size), self._content_need)
                            carry = carry[batch_tokens:]
                            batch = torch.from_numpy(block.copy())
                            if self._bos_id is not None:
                                bos = torch.full((batch.shape[0], 1), int(self._bos_id), dtype=torch.int64)
                                batch = torch.cat((bos, batch), dim=1)
                            if self.device.type == "cuda":
                                batch = batch.pin_memory()
                            self._ram_queue.put(batch)
                first_doc = True
        except BaseException as exc:
            self._producer_error = exc
            try:
                self._ram_queue.put_nowait(None)
            except queue.Full:
                pass

    def _get_cpu_batch(self) -> torch.Tensor:
        batch = self._ram_queue.get()
        if batch is None:
            raise RuntimeError("data producer failed") from self._producer_error
        return batch

    async def _load_gpu_chunk(self, count: int) -> None:
        loop = asyncio.get_running_loop()
        batches = []
        for _ in range(count):
            batches.append(await loop.run_in_executor(None, self._get_cpu_batch))
        host = torch.stack(batches)
        self._gpu_batches = host.to(self.device, non_blocking=True)
        self._gpu_pos = 0

    def _gpu_chunk_size(self) -> int:
        return min(int(self.cfg.gpu_prefetch_batches), int(self.cfg.prefetch_batches))

    async def prime(self) -> None:
        count = self._gpu_chunk_size()
        await self._load_gpu_chunk(count)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if ddp_is_distributed():
            ready = [None] * self._world_size
            dist.all_gather_object(ready, (self._rank, count))
            if ddp_is_main():
                for rank, n in sorted(ready):
                    print(f"[data] rank={rank} ready: RAM={int(self.cfg.prefetch_batches)} batches, GPU={n} batches", flush=True)
        else:
            print(f"[data] rank=0 ready: RAM={int(self.cfg.prefetch_batches)} batches, GPU={count} batches", flush=True)

    def sample_device_batch(self) -> torch.Tensor:
        batch = self._get_cpu_batch()
        return batch.to(self.device, non_blocking=True)

    def _sample_batch(self) -> torch.Tensor:
        return self.sample_device_batch()

    async def batches(self, n: int):
        chunk_size = self._gpu_chunk_size()
        yielded = 0
        while yielded < n:
            if self._gpu_batches is None or self._gpu_pos >= self._gpu_batches.shape[0]:
                await self._load_gpu_chunk(min(chunk_size, n - yielded))
            while self._gpu_pos < self._gpu_batches.shape[0] and yielded < n:
                batch = self._gpu_batches[self._gpu_pos]
                self._gpu_pos += 1
                yielded += 1
                yield batch

    def close(self) -> None:
        self._stop.set()

def is_sft_dataset(cfg: Config) -> bool:
    return str(getattr(cfg, "dataset_format", "auto") or "auto").lower() == "sft"


def make_stream(path: str, cfg: Config, device: torch.device):
    fmt = str(getattr(cfg, "dataset_format", "auto") or "auto").lower()
    if fmt == "sft":
        import loomsft  # lazy: only SFT runs need the chat template/pyarrow path
        return loomsft.make_stream(path, cfg, device)
    if fmt == "bin" or (fmt == "auto" and os.path.isfile(path) and path.endswith(".bin")):
        bos_id = _tok_special_id(build_tokenizer(cfg), "<bos>")
        return TokenStream(path, cfg, device, bos_id=bos_id)
    return ShardStream(path, cfg, device)


IGNORE_INDEX = -100  # cross-entropy target for positions that must not be learned


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


class _PhaseSinFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, beta: torch.Tensor, eps: float):
        x = (math.pi / 2.0) * beta * torch.rsqrt(1.0 + beta * beta)
        cosx = torch.cos(x)
        dx_dbeta = (math.pi / 2.0) * (1.0 + beta * beta).pow(-1.5)
        ctx.save_for_backward(cosx, dx_dbeta)
        ctx.eps = eps
        return torch.sin(x)

    @staticmethod
    def backward(ctx, grad_output):
        cosx, dx_dbeta = ctx.saved_tensors
        grad_scale = torch.clamp(cosx, min=ctx.eps) * dx_dbeta
        return grad_output * grad_scale, None


class _PhaseSinSecant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, beta: torch.Tensor, anchor: torch.Tensor, near_eps: float = 1e-4):
        # `phase_sin()` passes an immutable snapshot of the EMA anchor as this
        # explicit tensor input. Saving the input itself keeps autograd versioning
        # safe and lets graph_helper derive the custom-op save recipe.
        x = (math.pi / 2.0) * beta * torch.rsqrt(1.0 + beta * beta)
        s = torch.sin(x)
        x_anchor = (math.pi / 2.0) * anchor * torch.rsqrt(1.0 + anchor * anchor)
        s_anchor = torch.sin(x_anchor)
        ctx.save_for_backward(beta, s_anchor, anchor)
        ctx.near_eps = near_eps
        return s

    @staticmethod
    def backward(ctx, grad_output):
        beta, s_anchor, anchor = ctx.saved_tensors
        x = (math.pi / 2.0) * beta * torch.rsqrt(1.0 + beta * beta)
        s = torch.sin(x)
        denom = beta - anchor
        near = denom.abs() < ctx.near_eps
        safe_denom = torch.where(near, torch.ones_like(denom), denom)
        secant = (s - s_anchor) / safe_denom
        dx_dbeta = (math.pi / 2.0) * (1.0 + beta * beta).pow(-1.5)
        true_local = torch.cos(x) * dx_dbeta
        grad_scale = torch.where(near, true_local, secant)
        return grad_output * grad_scale, None, None


class _PhaseSinSecantCUDA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, beta: torch.Tensor, anchor: torch.Tensor, near_eps: float = 1e-4):
        ext = _try_load_cuda_phase_sin()
        out = ext.phase_sin_forward_cuda(beta)
        # `anchor` is an explicit immutable snapshot tensor supplied by
        # phase_sin(). Save that input directly: graph_helper can represent it,
        # and later EMA updates cannot change its version counter.
        ctx.save_for_backward(beta, anchor)
        ctx.near_eps = near_eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        beta, anchor = ctx.saved_tensors
        ext = _try_load_cuda_phase_sin()
        anchor_f = float(anchor.item())
        x_anchor = (math.pi / 2.0) * anchor_f / math.sqrt(1.0 + anchor_f * anchor_f)
        s_anchor_f = math.sin(x_anchor)
        grad_beta = ext.phase_sin_secant_backward_cuda(
            beta, grad_output.contiguous(), anchor_f, s_anchor_f, ctx.near_eps)
        return grad_beta, None, None


_cuda_phase_sin_module = None
_cuda_phase_sin_tried = False


def _try_load_cuda_phase_sin():
    global _cuda_phase_sin_module, _cuda_phase_sin_tried
    if _cuda_phase_sin_tried:
        return _cuda_phase_sin_module
    _cuda_phase_sin_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_phase_sin_module = build_or_load(
            "loomformer_phase_sin",
            ["phase_sin/phase_sin_launcher.cu"],
            ptx_kernels={"phase_sin": "phase_sin/phase_sin_kernel.cu"},
        )
    except Exception as e:
        _cuda_phase_sin_module = None
        ddp_print(
            f"[loomformer] CUDA phase_sin failed ({type(e).__name__}: {e}); "
            "using SLOWER PyTorch fallback.")
    return _cuda_phase_sin_module


class _PhaseSinFloorCUDA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, beta: torch.Tensor, eps: float):
        ext = _try_load_cuda_phase_sin()
        out = ext.phase_sin_forward_cuda(beta)
        ctx.save_for_backward(beta)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (beta,) = ctx.saved_tensors
        ext = _try_load_cuda_phase_sin()
        grad_beta = ext.phase_sin_backward_cuda(beta, grad_output.contiguous(), ctx.eps)
        return grad_beta, None


def phase_sin(beta: torch.Tensor, anchor: Optional[torch.Tensor] = None) -> torch.Tensor:
    if PHASE_GRAD_MODE == "secant":
        if anchor is None:
            raise ValueError("phase_grad_mode='secant' needs an anchor tensor (see ParaplexFFN.beta_anchor)")
        # The module-level EMA buffer is updated in-place once per chunk. Snapshot
        # it before entering any custom autograd/custom-op path so each invocation
        # owns a stable explicit input through backward.
        anchor = anchor.detach().clone()
        if USE_CUDA_PHASE_SIN and beta.is_cuda and beta.dtype in (torch.float32, torch.float16, torch.bfloat16):
            ext = _try_load_cuda_phase_sin()
            if ext is not None:
                if GRAPH_MODE_ENABLED and _graph_phase_sin_secant_op is not None:
                    return _graph_phase_sin_secant_op(beta, anchor, 1e-4)
                return _PhaseSinSecantCUDA.apply(beta, anchor, 1e-4)
        return _PhaseSinSecant.apply(beta, anchor)
    eps = max(PHASE_GRAD_FLOOR, 0.0)
    if USE_CUDA_PHASE_SIN and beta.is_cuda and beta.dtype in (torch.float32, torch.float16, torch.bfloat16):
        ext = _try_load_cuda_phase_sin()
        if ext is not None:
            if GRAPH_MODE_ENABLED and _graph_phase_sin_op is not None:
                return _graph_phase_sin_op(beta, eps)
            return _PhaseSinFloorCUDA.apply(beta, eps)
    return _PhaseSinFloor.apply(beta, eps)


def phase_anchor_scale(beta: torch.Tensor, floor: float = 1e-4) -> torch.Tensor:
    """Return the detached FP32 RMS phase radius, clamped to ``floor``."""
    beta_f = beta.detach().float()
    return beta_f.square().mean().sqrt().clamp_min(float(floor))

_cuda_pvpowlu_module = None
_cuda_pvpowlu_tried = False


def _try_load_cuda_pvpowlu():
    global _cuda_pvpowlu_module, _cuda_pvpowlu_tried
    if _cuda_pvpowlu_tried:
        return _cuda_pvpowlu_module
    _cuda_pvpowlu_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_pvpowlu_module = build_or_load(
            "loomformer_pvpowlu",
            ["pvpowlu/pvpowlu_launcher.cu"],
            ptx_kernels={"pvpowlu": "pvpowlu/pvpowlu_kernel.cu"},
        )
    except Exception as e:
        _cuda_pvpowlu_module = None
        ddp_print(
            f"[loomformer] CUDA pvpowlu failed ({type(e).__name__}: {e}); "
            "using SLOWER PyTorch fallback.")
    return _cuda_pvpowlu_module


class _PvPowluCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, m: float):
        ext = _try_load_cuda_pvpowlu()
        if ext is None:
            raise RuntimeError("CUDA pvpowlu module is unavailable")
        out = ext.pvpowlu_forward_cuda(x1, x2, float(m))
        ctx.save_for_backward(x1, x2)
        ctx.m = float(m)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        ext = _try_load_cuda_pvpowlu()
        if ext is None:
            raise RuntimeError("CUDA pvpowlu module is unavailable")
        grad_x1, grad_x2 = ext.pvpowlu_backward_cuda(grad_output.contiguous(), x1, x2, ctx.m)
        return grad_x1, grad_x2, None


def pvpowlu_act(x1: torch.Tensor, x2: torch.Tensor, m: float = 3.0) -> torch.Tensor:
    if USE_CUDA_PVPOWLU and x1.is_cuda and x2.is_cuda and x1.dtype in (torch.float32, torch.float16, torch.bfloat16):
        if x2.dtype == x1.dtype:
            ext = _try_load_cuda_pvpowlu()
            if ext is not None:
                if GRAPH_MODE_ENABLED and _graph_pvpowlu_op is not None:
                    return _graph_pvpowlu_op(x1, x2, float(m))
                return _PvPowluCUDA.apply(x1, x2, float(m))
    return x1 * powlu_gate(x2, m)


def _depth_attn_online_tensor_pytorch(
    q: torch.Tensor, hist_k: torch.Tensor, hist_v: torch.Tensor
) -> torch.Tensor:
    out_dtype = q.dtype
    q = q.float()
    hist_k = hist_k.float()
    hist_v = hist_v.float()
    sqrt_d = math.sqrt(HEAD_DIM)
    m = None
    l = None
    d = None
    for s in range(hist_k.shape[2]):
        k_s = hist_k[:, :, s]
        v_s = hist_v[:, :, s]
        score_s = (q * k_s).sum(-1) / sqrt_d
        if m is None:
            m = score_s
            l = torch.exp(score_s - m)
            d = l.unsqueeze(-1) * v_s
        else:
            m_new = torch.maximum(m, score_s)
            exp_old = torch.exp(m - m_new)
            exp_new = torch.exp(score_s - m_new)
            l = l * exp_old + exp_new
            d = d * exp_old.unsqueeze(-1) + exp_new.unsqueeze(-1) * v_s
            m = m_new
    return (d / l.unsqueeze(-1)).to(out_dtype)


def _depth_attn_online_list_pytorch(q: torch.Tensor, hist_k, hist_v) -> torch.Tensor:
    out_dtype = q.dtype
    qf = q.float().view(1, 1, N_Q_HEADS, HEAD_DIM)
    sqrt_d = math.sqrt(HEAD_DIM)
    scores = torch.stack([(qf * k.float()).sum(-1) / sqrt_d for k in hist_k], dim=-1)
    weights = torch.softmax(scores, dim=-1)
    out = sum(weights[..., j, None] * v.float() for j, v in enumerate(hist_v))
    return out.to(out_dtype)


class _DepthAttnListFused(torch.autograd.Function):
    # FlashAttention-style depth-history attention (single query over S history
    # keys). The previous implementation kept the history as a Python list of S
    # separate tensors and saved the full softmax weight matrix `w` for backward.
    # That made every history key k_j a distinct autograd leaf reused by all
    # later layers, so autograd accumulated its gradient one aten::add_ at a time
    # -- an O(L^2) fan-out that dominated the step (tens of thousands of tiny
    # kernels). This version follows the canonical FA recipe:
    #   * forward stacks history into one [B,W,S,Hq,Dh] buffer, runs the
    #     attention once, and saves only O + LSE (log-sum-exp), NOT the weights.
    #   * backward recomputes P = exp(S - LSE) analytically (no re-softmax, no
    #     autograd graph) and forms dQ/dK/dV in a handful of batched matmuls via
    #     the standard softmax-attention gradient
    #         dV = P^T dO ; dP = dO V^T ; dS = P*(dP - rowsum(dO*O))/sqrt_d ;
    #         dQ = dS K ; dK = dS^T Q.
    # Same call signature (q, count, *history) and same return shape as before,
    # so callers are unchanged. A fused CUDA kernel is used when available and
    # this pure-PyTorch path is the exact-parity fallback.
    @staticmethod
    def forward(ctx, q, count, *history):
        n = int(count)
        ks = list(history[:n])
        vs = list(history[n:])
        K = torch.stack(ks, dim=2)            # [B,W,S,Hq,Dh] -- temporary, not saved
        V = torch.stack(vs, dim=2)
        B, W = K.shape[0], K.shape[1]
        sqrt_d = math.sqrt(HEAD_DIM)
        out_dtype = q.dtype
        ext = _try_load_cuda_depth_attn()
        if ext is not None and hasattr(ext, "depth_attn_stacked_forward"):
            d, lse = ext.depth_attn_stacked_forward(q.contiguous(), K.contiguous(),
                                                    V.contiguous())
        else:
            qf = q.view(1, 1, N_Q_HEADS, HEAD_DIM).to(torch.float32)
            qf = qf.expand(B, W, N_Q_HEADS, HEAD_DIM)
            s = torch.einsum('bwhd,bwshd->bwhs', qf, K.float()) / sqrt_d   # [B,W,Hq,S]
            m = s.amax(-1, keepdim=True)
            p = torch.exp(s - m)
            l = p.sum(-1, keepdim=True)
            d = torch.einsum('bwhs,bwshd->bwhd', p / l, V.float()).to(out_dtype)
            lse = (m.squeeze(-1) + torch.log(l.squeeze(-1)))               # [B,W,Hq]
        ctx.count = n
        ctx.out_dtype = out_dtype
        ctx.save_for_backward(q, lse, *history)
        return d

    @staticmethod
    def backward(ctx, grad_d):
        q, lse, *history = ctx.saved_tensors
        n = ctx.count
        ks = list(history[:n])
        vs = list(history[n:])
        K = torch.stack(ks, dim=2)            # temporary, freed after backward
        V = torch.stack(vs, dim=2)
        B, W = K.shape[0], K.shape[1]
        sqrt_d = math.sqrt(HEAD_DIM)
        ext = _try_load_cuda_depth_attn()
        if ext is not None and hasattr(ext, "depth_attn_stacked_backward"):
            gq, gK, gV = ext.depth_attn_stacked_backward(
                grad_d.contiguous(), q.contiguous(), K.contiguous(),
                V.contiguous(), lse.contiguous())
        else:
            qf = q.view(1, 1, N_Q_HEADS, HEAD_DIM).to(torch.float32)
            qf = qf.expand(B, W, N_Q_HEADS, HEAD_DIM)
            dO = grad_d.float()
            s = torch.einsum('bwhd,bwshd->bwhs', qf, K.float()) / sqrt_d
            p = torch.exp(s - lse.unsqueeze(-1).float())                   # [B,W,Hq,S]
            dP = torch.einsum('bwhd,bwshd->bwhs', dO, V.float())           # [B,W,Hq,S]
            D = (p * dP).sum(-1, keepdim=True)                             # [B,W,Hq,1]
            dS = p * (dP - D) / sqrt_d
            gV = torch.einsum('bwhs,bwhd->bwshd', p, dO)
            gq_full = torch.einsum('bwhs,bwshd->bwhd', dS, K.float())      # [B,W,Hq,Dh]
            gq = gq_full.sum((0, 1)).to(q.dtype)                           # reduce broadcast -> [Hq,Dh]
            gK = torch.einsum('bwhs,bwhd->bwshd', dS, qf).to(K.dtype)
            gV = gV.to(V.dtype)
        gk_list = [gK[:, :, j] for j in range(n)]
        gv_list = [gV[:, :, j] for j in range(n)]
        return (gq, None, *gk_list, *gv_list)


@torch._dynamo.disable
def depth_attn_online_list_cuda(q: torch.Tensor, hist_k, hist_v) -> Optional[torch.Tensor]:
    if not hist_k:
        return None
    if not USE_CUDA_DEPTH_ATTN or not q.is_cuda:
        return None
    if any(not k.is_cuda or not v.is_cuda for k, v in zip(hist_k, hist_v)):
        return None
    dtype = hist_k[0].dtype
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    if any(k.dtype != dtype or v.dtype != dtype for k, v in zip(hist_k, hist_v)):
        return None
    # _DepthAttnListFused now carries its own exact-parity PyTorch fallback and
    # only uses the CUDA kernel when it is present, so we no longer gate on the
    # kernel being loaded here.
    history = tuple(hist_k) + tuple(hist_v)
    return _DepthAttnListFused.apply(q, len(hist_k), *history)


_cuda_depth_attn_module = None
_cuda_depth_attn_tried = False


def _try_load_cuda_depth_attn():
    global _cuda_depth_attn_module, _cuda_depth_attn_tried
    if _cuda_depth_attn_tried:
        return _cuda_depth_attn_module
    _cuda_depth_attn_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_depth_attn_module = build_or_load(
            "loomformer_depth_attn_online",
            ["depth_attn/depth_attn_launcher.cu"],
            ptx_kernels={"depth_attn": "depth_attn/depth_attn_kernel.cu"},
        )
    except Exception as e:
        _cuda_depth_attn_module = None
        ddp_print(
            f"[loomformer] CUDA depth_attn failed ({type(e).__name__}: {e}); "
            "using SLOWER PyTorch fallback.")
    return _cuda_depth_attn_module


class _DepthAttnOnlineFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, hist_k, hist_v):
        ext = _try_load_cuda_depth_attn()
        if ext is None:
            raise RuntimeError("CUDA depth_attn module is unavailable")
        d, w = ext.depth_attn_forward(q.contiguous(), hist_k.contiguous(), hist_v.contiguous())
        ctx.save_for_backward(q, hist_k, hist_v, w)
        return d

    @staticmethod
    def backward(ctx, grad_d):
        ext = _try_load_cuda_depth_attn()
        if ext is None:
            raise RuntimeError("CUDA depth_attn module is unavailable")
        q, hist_k, hist_v, w = ctx.saved_tensors
        grad_q, grad_k, grad_v = ext.depth_attn_backward(
            grad_d.contiguous(), q, hist_k, hist_v, w)
        return grad_q, grad_k, grad_v


def depth_attn_online_cuda(q: torch.Tensor, hist_k: torch.Tensor, hist_v: torch.Tensor) -> Optional[torch.Tensor]:
    if not (USE_CUDA_DEPTH_ATTN and q.is_cuda and hist_k.is_cuda and hist_v.is_cuda):
        return None
    if hist_k.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    if not (q.dtype == hist_k.dtype and hist_v.dtype == hist_k.dtype):
        return None
    ext = _try_load_cuda_depth_attn()
    if ext is None:
        return None
    if GRAPH_MODE_ENABLED and _graph_depth_attn_op is not None:
        d, _w = _graph_depth_attn_op(q, hist_k, hist_v)
        return d
    return _DepthAttnOnlineFused.apply(q, hist_k, hist_v)

_cuda_beta_space_module = None
_cuda_beta_space_tried = False
_cuda_beta_space_active_printed = False
_cuda_beta_space_fallback_printed = False
_cuda_paraplex_module = None
_cuda_paraplex_tried = False


def _cuda_autocast_dtype_or_none():
    """Return the active CUDA autocast dtype, or ``None`` when disabled."""
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:
        # Older API shape.
        if torch.is_autocast_cuda_enabled():
            return torch.get_autocast_gpu_dtype()
    except Exception:
        return None
    return None


def _beta_space_fast_dtype(u: torch.Tensor, q_h: torch.Tensor, k_ctx_h: torch.Tensor,
                           k_ctx_h2: torch.Tensor, d_h: torch.Tensor,
                           w1_imag: torch.Tensor) -> Optional[torch.dtype]:
    ac_dtype = _cuda_autocast_dtype_or_none()
    if ac_dtype in (torch.float32, torch.bfloat16):
        return ac_dtype

    dt = u.dtype
    for t in (q_h, k_ctx_h, k_ctx_h2, d_h, w1_imag):
        dt = torch.promote_types(dt, t.dtype)
    return dt if dt in (torch.float32, torch.bfloat16) else None


def _try_load_cuda_beta_space():
    global _cuda_beta_space_module, _cuda_beta_space_tried
    if _cuda_beta_space_tried:
        return _cuda_beta_space_module
    _cuda_beta_space_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_beta_space_module = build_or_load(
            "loomformer_beta_space",
            ["beta_space/beta_space_launcher.cu"],
            ptx_kernels={"beta_space": "beta_space/beta_space_kernel.cu"},
        )
    except Exception as e:
        _cuda_beta_space_module = None
        ddp_print(
            f"[loomformer] CUDA beta_space failed ({type(e).__name__}: {e}); "
            "using SLOWER PyTorch fallback.")
    return _cuda_beta_space_module


def _try_load_cuda_paraplex():
    global _cuda_paraplex_module, _cuda_paraplex_tried
    if _cuda_paraplex_tried:
        return _cuda_paraplex_module
    _cuda_paraplex_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_paraplex_module = build_or_load(
            "loomformer_paraplex",
            ["paraplex/paraplex_launcher.cu"],
            ptx_kernels={"paraplex": "paraplex/paraplex_kernel.cu"},
        )
    except Exception as e:
        _cuda_paraplex_module = None
        ddp_print(f"[loomformer] CUDA paraplex unavailable ({type(e).__name__}: {e}); using composed ops.")
    return _cuda_paraplex_module


class _BetaSpaceDirect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
                hidden_per_q_head, head_dim, n_q_heads, open_sectors):
        ext = _try_load_cuda_beta_space()
        if ext is None:
            raise RuntimeError("CUDA beta_space module is unavailable")
        w_compute = (
            w1_imag_compact
            if w1_imag_compact.dtype == u.dtype
            else w1_imag_compact.to(dtype=u.dtype)
        )
        out, _r_pack, _w_contig = ext.beta_forward_cuda(
            u, q_h, k_ctx_h, c_h, d_h, w_compute,
            hidden_per_q_head, head_dim, n_q_heads, open_sectors)
        ctx.save_for_backward(u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact)
        ctx.shapes = (u.shape[0], u.shape[1], u.shape[2],
                      w1_imag_compact.shape[0], w1_imag_compact.shape[1])
        ctx.meta = (hidden_per_q_head, head_dim, n_q_heads, open_sectors)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ext = _try_load_cuda_beta_space()
        if ext is None:
            raise RuntimeError("CUDA beta_space module is unavailable")
        u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact = ctx.saved_tensors
        B, T, N, HIDDEN, IMAG_IN = ctx.shapes
        hidden_per_q_head, head_dim, n_q_heads, open_sectors = ctx.meta
        w_compute = (
            w1_imag_compact
            if w1_imag_compact.dtype == u.dtype
            else w1_imag_compact.to(dtype=u.dtype)
        )
        grads = ext.beta_backward_cuda_recompute(
            grad_out, u, q_h, k_ctx_h, c_h, d_h, w_compute,
            hidden_per_q_head, head_dim, n_q_heads, open_sectors)
        grad_u, grad_q, grad_k, grad_c, grad_d, grad_w = grads
        if grad_w.dtype != w1_imag_compact.dtype:
            grad_w = grad_w.to(dtype=w1_imag_compact.dtype)
        QH, HD = n_q_heads, head_dim
        return (grad_u,
                grad_q.view(B, T, QH, HD),
                grad_k.view(B, T, QH, HD),
                grad_c.view(B, T, QH, HD),
                grad_d.view(B, T, QH, HD),
                grad_w,
                None, None, None, None)


class _ParaplexFused(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, p_real, gate_src, gate_weight, gate_bias,
        recompute_intermediates,
        u, q_h, k_h, c_h, d_h, w_imag, bias, trace, trace_w,
        reset, anchor, hidden_per_q_head, head_dim, n_q_heads, open_sectors,
        phase_mode, update_anchor, anchor_decay, phase_floor, near_eps, powlu_m,
    ):
        beta_ext = _try_load_cuda_beta_space()
        core_ext = _try_load_cuda_paraplex()
        if beta_ext is None or core_ext is None:
            raise RuntimeError("CUDA paraplex dependencies are unavailable")
        w_compute = w_imag if w_imag.dtype == u.dtype else w_imag.to(dtype=u.dtype)
        beta, _r_pack, _w_contig = beta_ext.beta_forward_cuda(
            u, q_h, k_h, c_h, d_h, w_compute,
            hidden_per_q_head, head_dim, n_q_heads, open_sectors)
        act, s, next_trace, anchor_snapshot = core_ext.paraplex_forward(
            p_real, gate_src, beta, bias, trace, trace_w, reset, anchor,
            phase_mode, update_anchor, anchor_decay, powlu_m)
        ctx.mark_non_differentiable(anchor_snapshot)
        ctx.recompute_intermediates = bool(recompute_intermediates)
        if ctx.recompute_intermediates:
            ctx.save_for_backward(
                p_real, bias, trace, trace_w, reset, anchor_snapshot,
                u, q_h, k_h, c_h, d_h, w_imag, gate_weight, gate_bias)
        else:
            ctx.save_for_backward(
                p_real, gate_src, beta, bias, trace, trace_w, reset, anchor_snapshot,
                u, q_h, k_h, c_h, d_h, w_imag)
        ctx.has_gate_projection = gate_weight.numel() != 0
        ctx.shapes = (u.shape[0], u.shape[1], u.shape[2], w_imag.shape[0], w_imag.shape[1])
        ctx.meta = (
            hidden_per_q_head, head_dim, n_q_heads, open_sectors,
            phase_mode, phase_floor, near_eps, powlu_m,
        )
        return act, s, next_trace, anchor_snapshot

    @staticmethod
    def backward(ctx, grad_act, grad_s, grad_next, _grad_anchor):
        core_ext = _try_load_cuda_paraplex()
        beta_ext = _try_load_cuda_beta_space()
        if core_ext is None or beta_ext is None:
            raise RuntimeError("CUDA paraplex dependencies are unavailable")
        if ctx.recompute_intermediates:
            (p_real, bias, trace, trace_w, reset, anchor,
             u, q_h, k_h, c_h, d_h, w_imag, gate_weight, gate_bias) = ctx.saved_tensors
            gate_src = None
            beta = None
        else:
            (p_real, gate_src, beta, bias, trace, trace_w, reset, anchor,
             u, q_h, k_h, c_h, d_h, w_imag) = ctx.saved_tensors
            gate_weight = None
            gate_bias = None
        grad_act = torch.zeros_like(p_real) if grad_act is None else grad_act.to(dtype=p_real.dtype)
        grad_s = torch.zeros_like(p_real) if grad_s is None else grad_s.to(dtype=p_real.dtype)
        grad_next = torch.zeros_like(trace) if grad_next is None else grad_next.to(dtype=trace.dtype)
        hidden_per_q_head, head_dim, n_q_heads, open_sectors, mode, floor, near_eps, m = ctx.meta
        w_compute = w_imag if w_imag.dtype == u.dtype else w_imag.to(dtype=u.dtype)
        if ctx.recompute_intermediates:
            beta, _r_pack, _w_contig = beta_ext.beta_forward_cuda(
                u, q_h, k_h, c_h, d_h, w_compute,
                hidden_per_q_head, head_dim, n_q_heads, open_sectors)
            if ctx.has_gate_projection:
                gate_src = F.linear(
                    u,
                    gate_weight.to(dtype=u.dtype),
                    gate_bias.to(dtype=u.dtype),
                )
                if gate_src.dtype != p_real.dtype:
                    gate_src = gate_src.to(dtype=p_real.dtype)
            else:
                gate_src = p_real
        grad_p, grad_gate, grad_beta, grad_bias, grad_trace, grad_trace_w = core_ext.paraplex_backward(
            grad_act, grad_s, grad_next, p_real, gate_src, beta, bias, trace, trace_w,
            reset, anchor, mode, floor, near_eps, m)
        B, T, N_local, H_local, imag_in = ctx.shapes
        grad_u, grad_q, grad_k, grad_c, grad_d, grad_w = beta_ext.beta_backward_cuda_recompute(
            grad_beta, u, q_h, k_h, c_h, d_h, w_compute,
            hidden_per_q_head, head_dim, n_q_heads, open_sectors)
        if grad_w.dtype != w_imag.dtype:
            grad_w = grad_w.to(dtype=w_imag.dtype)
        QH, HD = n_q_heads, head_dim
        return (
            grad_p, grad_gate, None, None, None,
            grad_u, grad_q.view(B, T, QH, HD), grad_k.view(B, T, QH, HD),
            grad_c.view(B, T, QH, HD), grad_d.view(B, T, QH, HD), grad_w,
            grad_bias, grad_trace, grad_trace_w, None, None,
            None, None, None, None, None, None, None, None, None, None,
        )


def beta_space_cuda(u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
                    hidden_per_q_head, head_dim, n_q_heads, open_sectors):
    """Compute beta space with CUDA, or return ``None`` when unsupported."""
    if not (u.is_cuda and q_h.is_cuda and k_ctx_h.is_cuda and c_h.is_cuda and d_h.is_cuda and w1_imag_compact.is_cuda):
        return None
    if u.dtype not in (torch.float32, torch.bfloat16):
        return None
    if not (q_h.dtype == u.dtype and k_ctx_h.dtype == u.dtype and c_h.dtype == u.dtype and d_h.dtype == u.dtype):
        return None
    if w1_imag_compact.dtype not in (torch.float32, torch.bfloat16):
        return None
    N_local = u.shape[-1]
    IMAG_IN_local = w1_imag_compact.shape[-1]
    if (N_local % 4) != 0 or (head_dim % 4) != 0 or (IMAG_IN_local % 4) != 0:
        return None
    ext = _try_load_cuda_beta_space()
    if ext is None:
        return None
    if GRAPH_MODE_ENABLED and _graph_beta_space_op is not None:
        out = _graph_beta_space_op(
            u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
            hidden_per_q_head, head_dim, n_q_heads, open_sectors)
        return out
    return _BetaSpaceDirect.apply(
        u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
        hidden_per_q_head, head_dim, n_q_heads, open_sectors)


def sin_space_combine(s_base: torch.Tensor, trace_term: torch.Tensor) -> torch.Tensor:
    raw = s_base + trace_term
    return raw * torch.rsqrt(1.0 + raw * raw)


def prev_token_trace(s_base: torch.Tensor, initial_trace: Optional[torch.Tensor] = None,
                     reset_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # SAME neuron, previous token. s_base does not depend on the trace, so the shift
    # stays fully parallel over T — no slowdown vs the neighbor-shift version.
    B, T, H = s_base.shape
    trace = s_base.new_zeros(B, T, H)
    if initial_trace is not None:
        trace[:, 0, :] = initial_trace.to(device=s_base.device, dtype=s_base.dtype)
    if T > 1:
        trace[:, 1:, :] = s_base[:, :-1, :]
    if reset_mask is not None:
        trace = trace.masked_fill(reset_mask.to(device=s_base.device, dtype=torch.bool).unsqueeze(-1), 0)
    return trace


@dataclass
class LayerCache:
    k: Optional[torch.Tensor] = None            # [B,SEQ_LEN,N_KV_HEADS,HEAD_DIM] -- preallocated
    v: Optional[torch.Tensor] = None            # [B,SEQ_LEN,N_KV_HEADS,HEAD_DIM] -- preallocated
    phase_trace: Optional[torch.Tensor] = None  # [B,HIDDEN]
    cache_len: int = 0                          # сколько позиций реально заполнено


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
            shape = (batch, self.chunk_size, N_KV_HEADS, HEAD_DIM)
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


class ParaplexFFN(nn.Module):
    def __init__(self, ablation: bool = False) -> None:
        super().__init__()
        self.ablation = bool(ablation)
        self.w1_real = nn.Linear(N, HIDDEN)
        # Compact PARAMETER storage, dense COMPUTE path.
        # Only live phase weights are Parameters/optimizer state; forward expands them
        # into a transient dense matrix and keeps the fast single GEMM.
        self.w1_imag = nn.Parameter(torch.empty(HIDDEN, IMAG_IN))
        self.w1_imag_trace = nn.Parameter(torch.zeros(HIDDEN))
        self.w1_imag_bias = nn.Parameter(torch.zeros(HIDDEN))
        self.w2 = nn.Linear(HIDDEN, N)
        # phase_grad_mode: "secant" -- adaptive anchor for _PhaseSinSecant.
        # Persistent scalar buffer, NOT an nn.Parameter: updated by an EMA
        # tracking rule (see forward below), not by gradient descent -- same
        # role as an FP8 checkpoint's weight_scale/input_scale (a calibrated
        # scalar riding alongside the tensor it scales, not itself learned).
        # Only actually read/updated when phase_grad_mode=="secant"; harmless,
        # cheap dead weight otherwise (one scalar per layer).
        self.register_buffer("beta_anchor", torch.tensor(1.0), persistent=True)
        self.beta_anchor_decay = 0.99  # EMA smoothing for the FP32 RMS phase radius
        # tria.py §4: per-layer gate. Only constructed when TRIA_CARRY_ENABLED
        # -- when tria is off, p_in is always None and identity_gate would just
        # return wx+bias, so the module is pure overhead (10 params × LAYERS).
        if TRIA_CARRY_ENABLED:
            self.gate_selector = tria.GateSelector()
            self.identity_gate = tria.IdentityAnchoredGate()
        else:
            self.gate_selector = None
            self.identity_gate = None
        # Independent gate source (donor-transplant slot -- see PARAPLEX_GATE_PROJ
        # above). None in the default/original design: amp is self-referential,
        # derived from p_real with zero extra parameters.
        if PARAPLEX_GATE_PROJ:
            self.gate_proj = nn.Linear(N, HIDDEN)
        else:
            self.gate_proj = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear_fanin(self.w1_real)
        nn.init.normal_(self.w1_imag, mean=0.0, std=fanin_std(IMAG_IN))
        init_linear_residual(self.w2)
        # Start exactly on the non-recurrent phase path; trace learns in only if useful.
        nn.init.zeros_(self.w1_imag_trace)
        nn.init.zeros_(self.w1_imag_bias)
        if self.gate_proj is not None:
            init_linear_fanin(self.gate_proj)

    @staticmethod
    def _merge_query_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, G, Dh = x.shape
        if G != N_Q_HEADS or Dh != HEAD_DIM:
            raise ValueError("bad query-head tensor")
        return x.reshape(B, T, N)

    def _dense_imag_weight(self) -> torch.Tensor:
        # Expand head-local compact columns without a persistent dense zero buffer or
        # an HIDDEN*IMAG_IN int64 scatter index.  Those buffers cost ~2 GiB for alt6
        # and DDP tried to broadcast them before every forward.  The head selector is
        # tiny; multiplication materializes only the required dense head-local block,
        # and cat produces the single GEMM weight.
        def expand_head_local(x: torch.Tensor) -> torch.Tensor:
            x = x.view(N_Q_HEADS, HIDDEN_PER_Q_HEAD, HEAD_DIM)
            selector = torch.eye(
                N_Q_HEADS, dtype=x.dtype, device=x.device
            ).view(N_Q_HEADS, 1, N_Q_HEADS, 1)
            return (x.unsqueeze(2) * selector).reshape(HIDDEN, N)

        if PHASE_SECTORS == "head":
            q, k, c, u, d = torch.split(
                self.w1_imag, (HEAD_DIM, HEAD_DIM, HEAD_DIM, N, HEAD_DIM), dim=1
            )
            return torch.cat((
                expand_head_local(q),
                expand_head_local(k),
                expand_head_local(c),
                u,
                expand_head_local(d),
            ), dim=1)

        q, shared = torch.split(self.w1_imag, (HEAD_DIM, 4 * N), dim=1)
        return torch.cat((expand_head_local(q), shared), dim=1)

    def _beta_space(self, u: torch.Tensor, q_h: torch.Tensor, k_ctx_h: torch.Tensor,
                     c_h: torch.Tensor, d_h: torch.Tensor) -> torch.Tensor:
        global _cuda_beta_space_active_printed, _cuda_beta_space_fallback_printed

        if USE_CUDA_BETA_SPACE and u.is_cuda:
            fast_dtype = _beta_space_fast_dtype(u, q_h, k_ctx_h, c_h, d_h, self.w1_imag)
            if fast_dtype in (torch.float32, torch.bfloat16):
                # In the real training graph the five activation sources can be mixed dtype
                # under autocast. beta_space_cuda requires one dtype, so normalize explicitly.
                # Activation casts remain in the outer graph. The weight cast is transient
                # inside the custom autograd op, so its BF16 copy is not retained for backward.
                u_fast = u if u.dtype == fast_dtype else u.to(dtype=fast_dtype)
                q_fast = q_h if q_h.dtype == fast_dtype else q_h.to(dtype=fast_dtype)
                k_fast = k_ctx_h if k_ctx_h.dtype == fast_dtype else k_ctx_h.to(dtype=fast_dtype)
                c_fast = c_h if c_h.dtype == fast_dtype else c_h.to(dtype=fast_dtype)
                d_fast = d_h if d_h.dtype == fast_dtype else d_h.to(dtype=fast_dtype)
                out = beta_space_cuda(
                    u_fast, q_fast, k_fast, c_fast, d_fast, self.w1_imag,
                    HIDDEN_PER_Q_HEAD, HEAD_DIM, N_Q_HEADS, PHASE_SECTORS == "open")
                if out is not None:
                    if not _cuda_beta_space_active_printed:
                        dtypes = {u.dtype, q_h.dtype, k_ctx_h.dtype, c_h.dtype, d_h.dtype, self.w1_imag.dtype}
                        dtype_note = str(fast_dtype) if len(dtypes) == 1 else f"mixed:{sorted(str(d) for d in dtypes)}"
                        ddp_print(
                            f"[loomformer] CUDA beta_space active  dtype={dtype_note}  "
                            f"sectors={PHASE_SECTORS}  shape=B{u.shape[0]}xT{u.shape[1]}xN{N}xH{HIDDEN}"
                        )
                        _cuda_beta_space_active_printed = True
                    return out
            elif os.environ.get("LOOM_BETA_SPACE_DEBUG") == "1" and not _cuda_beta_space_fallback_printed:
                ddp_print(
                    "[loomformer] CUDA beta_space fallback: unsupported dtype mix "
                    f"u={u.dtype}, q={q_h.dtype}, k={k_ctx_h.dtype}, c={c_h.dtype}, "
                    f"d={d_h.dtype}, w={self.w1_imag.dtype}."
                )
                _cuda_beta_space_fallback_printed = True

        q_all = self._merge_query_heads(q_h)
        kctx_all = self._merge_query_heads(k_ctx_h)
        c_all = self._merge_query_heads(c_h)
        d_all = self._merge_query_heads(d_h)
        r_all = torch.cat((q_all, kctx_all, c_all, u, d_all), dim=-1)        # (B,T,5N)
        return F.linear(r_all, self._dense_imag_weight())                    # один dense GEMM

    @torch._dynamo.disable
    def _fused_paraplex(
        self, p_real: torch.Tensor, gate_src: torch.Tensor, u: torch.Tensor, q_h: torch.Tensor,
        k_ctx_h: torch.Tensor, c_h: torch.Tensor, d_h: torch.Tensor,
        trace: torch.Tensor, reset_mask: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.ablation or ACTIVATION != "pvpowlu" or not u.is_cuda:
            return None
        if not (USE_CUDA_BETA_SPACE and USE_CUDA_PHASE_SIN and USE_CUDA_PVPOWLU):
            return None
        fast_dtype = _beta_space_fast_dtype(u, q_h, k_ctx_h, c_h, d_h, self.w1_imag)
        if fast_dtype not in (torch.float32, torch.bfloat16):
            return None
        if _try_load_cuda_beta_space() is None or _try_load_cuda_paraplex() is None:
            return None

        def cast(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == fast_dtype else x.to(dtype=fast_dtype)

        reset = (
            torch.empty(0, dtype=torch.bool, device=u.device)
            if reset_mask is None
            else reset_mask.to(device=u.device, dtype=torch.bool).contiguous()
        )
        mode = 1 if PHASE_GRAD_MODE == "secant" else 0
        anchor_override = _checkpoint_anchor_override(self)
        anchor = self.beta_anchor.detach() if anchor_override is None else anchor_override
        update_anchor = bool(mode == 1 and self.training and anchor_override is None)
        gate_weight = (
            self.gate_proj.weight
            if self.gate_proj is not None
            else p_real.new_empty(0)
        )
        gate_bias = (
            self.gate_proj.bias
            if self.gate_proj is not None
            else p_real.new_empty(0)
        )
        recompute_intermediates = not (
            GRAD_CHECKPOINTING
            and self.training
            and TRIA_CARRY_ENABLED
            and TRIA_TEMPORAL_ENABLED
        )
        # Non-reentrant checkpoint already discards and regenerates saved
        # tensors, so saving beta/gate_src there costs no retained VRAM and
        # avoids recomputing both a second time in the custom backward.
        # Outside checkpoint, rebuild them from the already-saved inputs.
        act, s, next_trace, anchor_snapshot = _ParaplexFused.apply(
            cast(p_real), cast(gate_src), gate_weight, gate_bias,
            recompute_intermediates,
            cast(u), cast(q_h), cast(k_ctx_h), cast(c_h), cast(d_h),
            self.w1_imag, self.w1_imag_bias, cast(trace), self.w1_imag_trace,
            reset, anchor, HIDDEN_PER_Q_HEAD, HEAD_DIM, N_Q_HEADS,
            PHASE_SECTORS == "open", mode, update_anchor, self.beta_anchor_decay,
            max(PHASE_GRAD_FLOOR, 0.0), 1e-4, POWLU_M,
        )
        if update_anchor:
            with torch.no_grad():
                self.beta_anchor.copy_(anchor_snapshot)
        return act, s, next_trace

    def forward(
        self,
        u: torch.Tensor,
        q_h: torch.Tensor,
        k_ctx_h: torch.Tensor,
        c_h: torch.Tensor,
        d_h: torch.Tensor,
        phase_trace: Optional[torch.Tensor] = None,
        phase_reset_mask: Optional[torch.Tensor] = None,
        return_tria: bool = False,
        p_in: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = u.shape
        if D != N:
            raise ValueError(f"u last dim must be {N}, got {D}")
        trace = u.new_zeros(B, HIDDEN) if phase_trace is None else phase_trace.to(device=u.device, dtype=u.dtype)
        # spec §4: gate lives BETWEEN Wx and +b, never applied to Wx+b as one
        # blob -- see IdentityAnchoredGate's docstring for the exact bug that
        # guards against (gating the bias too, invisible until bias moves away
        # from its zero init during training). p_in=None (layer 1, or tria
        # disabled) makes this an exact no-op identical to self.w1_real(u).
        wx = F.linear(u, self.w1_real.weight, None)
        if self.identity_gate is not None:
            p_real = self.identity_gate(wx, self.w1_real.bias, p_in)
        else:
            p_real = wx + self.w1_real.bias
        # Independent gate source (donor-transplant path): PARAPLEX_GATE_PROJ off
        # (default) keeps the original, parameter-free self-referential design --
        # gate_src IS p_real, amp derives from the same tensor the value path uses.
        # PARAPLEX_GATE_PROJ on: an independent Linear gives amp its own signal
        # (the slot a SwiGLU donor's gate_proj maps onto under --rebuild).
        gate_src = self.gate_proj(u) if self.gate_proj is not None else p_real
        s = None  # only defined on the non-ablation path -- see return_tria note below
        fused = self._fused_paraplex(p_real, gate_src, u, q_h, k_ctx_h, c_h, d_h, trace, phase_reset_mask)
        if fused is not None:
            act_out, s, trace = fused
            ffn_out = self.w2(act_out)
            return (ffn_out, trace, (p_real, s, act_out)) if return_tria else (ffn_out, trace)

        amp = F.softplus(gate_src)
        if self.ablation:
            p = p_real + amp
            trace = torch.ones(B, HIDDEN, device=u.device, dtype=u.dtype)
        else:
            beta_base = self._beta_space(u, q_h, k_ctx_h, c_h, d_h) + self.w1_imag_bias
            anchor_override = _checkpoint_anchor_override(self)
            if PHASE_GRAD_MODE == "secant" and self.training and anchor_override is None:
                # FP32 RMS is a representative phase radius. A global amin over
                # B*T*H inevitably approaches zero as the model grows and turns
                # the adaptive secant anchor into a permanent zero anchor.
                with torch.no_grad():
                    batch_scale = phase_anchor_scale(beta_base)
                    self.beta_anchor.mul_(self.beta_anchor_decay).add_(
                        batch_scale.to(self.beta_anchor.dtype), alpha=1.0 - self.beta_anchor_decay)
            anchor = self.beta_anchor if anchor_override is None else anchor_override
            s_base = phase_sin(beta_base, anchor if PHASE_GRAD_MODE == "secant" else None)
            trace_mat = prev_token_trace(
                s_base, trace if phase_trace is not None else None, phase_reset_mask)
            s = sin_space_combine(s_base, trace_mat * self.w1_imag_trace.view(1, 1, HIDDEN))
            p = torch.addcmul(p_real, amp, s)
            # Return s_base, not s: this is exactly what the parallel forward feeds into
            # position t+1, so incremental step() computes the identical function.
            trace = s_base[:, -1, :]
        if ACTIVATION == "pvpowlu":
            act_out = pvpowlu_act(p, amp, POWLU_M)
        else:
            act_out = act_fn(p)
        ffn_out = self.w2(act_out)
        if return_tria:
            # r,i,o for tria.py (spec §1): r=p_real (pre-imag), i=s (post phase_sin,
            # NOT s_base/beta -- spec §1 fixed this explicitly: "уже обогащённое"),
            # o=act_out (the pre-w2 activated scalar). i is None under ablation --
            # there is no phase/imag path to draw it from, so tria is a no-op there
            # (caller must handle None, not synthesize a fake i).
            return ffn_out, trace, (p_real, s, act_out)
        return ffn_out, trace





def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _yarn_find_correction_dim(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))) / (2.0 * math.log(base))


def _yarn_find_correction_range(beta_fast: float, beta_slow: float, dim: int, base: float,
                                max_position_embeddings: int) -> Tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(beta_fast, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(beta_slow, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, *, device, dtype) -> torch.Tensor:
    if low == high:
        high += 0.001
    x = (torch.arange(dim, device=device, dtype=dtype) - low) / (high - low)
    return torch.clamp(x, 0.0, 1.0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class YaRNRotaryEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        inv_freq, attention_factor = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_factor = float(attention_factor)
        cos, sin = self._build_cos_sin_cache(SEQ_LEN, inv_freq.device)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    @staticmethod
    def _compute_inv_freq() -> Tuple[torch.Tensor, float]:
        rotary_dim = HEAD_DIM
        pos_freqs = ROPE_THETA ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (ROPE_FACTOR * pos_freqs)
        low, high = _yarn_find_correction_range(
            ROPE_BETA_FAST, ROPE_BETA_SLOW, rotary_dim, ROPE_THETA, ROPE_ORIGINAL_SEQ_LEN
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, rotary_dim // 2, device=pos_freqs.device, dtype=torch.float32
        )
        inv_freq = inv_freq_interpolation * (1.0 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        attention_factor = ROPE_ATTENTION_FACTOR
        if attention_factor is None:
            attention_factor = _yarn_get_mscale(ROPE_FACTOR)
        return inv_freq, float(attention_factor)

    def _build_cos_sin_cache(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().mul(self.attention_factor)
            sin = emb.sin().mul(self.attention_factor)
        return cos, sin

    def _ensure_cache_device(self, device: torch.device) -> None:
        if self.cos_cached.device == device and self.sin_cached.device == device:
            return
        self.cos_cached = self.cos_cached.to(device=device)
        self.sin_cached = self.sin_cached.to(device=device)

    def _cos_sin(self, position_ids: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_cache_device(position_ids.device)
        if position_ids.dim() == 1:
            cos = self.cos_cached[position_ids].to(dtype=dtype).unsqueeze(0).unsqueeze(2)
            sin = self.sin_cached[position_ids].to(dtype=dtype).unsqueeze(0).unsqueeze(2)
        elif position_ids.dim() == 2:
            cos = self.cos_cached[position_ids].to(dtype=dtype).unsqueeze(2)
            sin = self.sin_cached[position_ids].to(dtype=dtype).unsqueeze(2)
        else:
            raise ValueError(f"position_ids must be 1D or 2D, got shape {tuple(position_ids.shape)}")
        return cos, sin

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._cos_sin(position_ids, q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _bf16_efficient_sdpa_supported(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    major, minor = torch.cuda.get_device_capability(idx)
    key = (idx, major, minor)
    cached = _sdpa_bf16_efficient_cache.get(key)
    if cached is not None:
        return cached
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with torch.cuda.device(idx):
            q = torch.randn(1, 1, 16, HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, 1, 16, HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(1, 1, 16, 2 * HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
            torch.cuda.synchronize(device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    with torch.autocast(device_type="cuda", enabled=False):
                        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
                        y.sum().backward()
            torch.cuda.synchronize(device)
        ok = True
    except Exception:
        ok = False
    _sdpa_bf16_efficient_cache[key] = ok
    return ok


def _get_flash_varlen_func():
    global _flash_varlen_func, _flash_varlen_import_tried, _flash_varlen_import_error
    if _flash_varlen_import_tried:
        return _flash_varlen_func
    _flash_varlen_import_tried = True
    try:
        from flash_attn import flash_attn_varlen_func
    except (ImportError, OSError) as error:
        _flash_varlen_func = None
        _flash_varlen_import_error = f"{type(error).__name__}: {error}"
    else:
        _flash_varlen_func = flash_attn_varlen_func
        _flash_varlen_import_error = None
    return _flash_varlen_func


def _call_flash_varlen(func, q, k, v, *args, **kwargs):
    """Call both current and older FA2 varlen bindings without hiding errors."""
    global _flash_deterministic_kw_supported
    if _flash_deterministic_kw_supported is False:
        kwargs.pop("deterministic", None)
        return func(q, k, v, *args, **kwargs)
    try:
        out = func(q, k, v, *args, **kwargs)
    except TypeError as error:
        if (
            _flash_deterministic_kw_supported is None
            and "deterministic" in kwargs
            and "deterministic" in str(error)
        ):
            kwargs.pop("deterministic")
            out = func(q, k, v, *args, **kwargs)
            _flash_deterministic_kw_supported = False
            return out
        raise
    _flash_deterministic_kw_supported = True
    return out


def _flash_varlen_eligible(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda" or tensor.dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
    major, _minor = torch.cuda.get_device_capability(idx)
    key = (idx, tensor.dtype, HEAD_DIM)
    probed = _flash_backend_cache.get(key)
    if probed is not None:
        return probed
    return major >= 8 and HEAD_DIM <= 256 and _get_flash_varlen_func() is not None


def _attention_compute_dtype(device: torch.device, parameter_dtype: torch.dtype) -> torch.dtype:
    if device.type != "cuda":
        return parameter_dtype
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:  # PyTorch before the device-typed autocast query
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return parameter_dtype


def _probe_flash_value_fusion(device: torch.device, dtype: torch.dtype) -> bool:
    """Verify fused value=[K;V] forward *and backward once per device/shape."""
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, dtype, HEAD_DIM)
    cached = _flash_value_fusion_cache.get(key)
    if cached is not None:
        return cached
    func = _get_flash_varlen_func()
    major, _minor = torch.cuda.get_device_capability(idx)
    if func is None or major < 8 or HEAD_DIM > 256:
        _flash_backend_cache[key] = False
        _flash_value_fusion_cache[key] = False
        return False
    try:
        total = 4
        q = torch.randn(
            total, N_Q_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(
            total, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        v_base = torch.randn(
            total, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        cu = torch.tensor([0, total], dtype=torch.int32, device=device)
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            base_out = _call_flash_varlen(
                func,
                q, k, v_base, cu, cu, total, total,
                dropout_p=0.0, causal=True,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
            base_out.float().sum().backward()
            torch.cuda.synchronize(device)
        _flash_backend_cache[key] = True
        del q, k, v_base, base_out
    except Exception as error:
        _flash_backend_cache[key] = False
        _flash_value_fusion_cache[key] = False
        _flash_probe_errors[key] = f"{type(error).__name__}: {error}"
        return False
    if 2 * HEAD_DIM > 256:
        _flash_value_fusion_cache[key] = False
        return False
    try:
        q = torch.randn(
            total, N_Q_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        fused_kv = torch.randn(
            total, N_KV_HEADS, 2 * HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = fused_kv[..., :HEAD_DIM]
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            out = _call_flash_varlen(
                func,
                q, k, fused_kv, cu, cu, total, total,
                dropout_p=0.0, causal=True,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
            out.float().sum().backward()
            torch.cuda.synchronize(device)
        ok = out.shape[-1] == 2 * HEAD_DIM
        del q, k, fused_kv, cu, out
    except Exception as error:
        ok = False
        _flash_probe_errors[key] = (
            "fused-value probe only (base varlen works): "
            f"{type(error).__name__}: {error}")
    _flash_value_fusion_cache[key] = ok
    return ok


def _flash_attention_contexts(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Varlen causal attention in THD layout, returning (attention@K, attention@V)."""
    func = _get_flash_varlen_func()
    if func is None:
        raise RuntimeError(
            "FlashAttention was selected but flash_attn.flash_attn_varlen_func is unavailable")
    common = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        dropout_p=0.0,
        causal=True,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    with torch.autocast(device_type="cuda", enabled=False):
        k_ctx = _call_flash_varlen(func, q, k, k, **common)
        value_ctx = _call_flash_varlen(func, q, k, v, **common)
    return k_ctx, value_ctx


def _get_te_dpa(device: torch.device, value_dim: int):
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, int(value_dim))
    module = _te_dpa_modules.get(key)
    if module is not None:
        return module
    try:
        from transformer_engine.pytorch import DotProductAttention
        with torch.cuda.device(idx):
            module = DotProductAttention(
                num_attention_heads=N_Q_HEADS,
                kv_channels=(HEAD_DIM, int(value_dim)),
                num_gqa_groups=N_KV_HEADS,
                attention_dropout=0.0,
                qkv_format="thd",
                attn_mask_type="padding_causal_bottom_right",
            )
    except (ImportError, OSError, RuntimeError):
        return None
    _te_dpa_modules[key] = module
    return module


def _te_varlen_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    module = _get_te_dpa(q.device, v.shape[-1])
    if module is None:
        raise RuntimeError("Transformer Engine DotProductAttention is unavailable")
    return module(
        q, k, v,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_kv=int(max_seqlen_k),
        attn_mask_type="padding_causal_bottom_right",
    )


def _probe_te_value_fusion(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, dtype, HEAD_DIM)
    cached = _te_value_fusion_cache.get(key)
    if cached is not None:
        return cached
    major, _minor = torch.cuda.get_device_capability(idx)
    if major < 8:
        _te_backend_cache[key] = False
        _te_value_fusion_cache[key] = False
        return False
    total = 4
    cu = torch.tensor([0, total], dtype=torch.int32, device=device)
    try:
        q = torch.randn(
            total, N_Q_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(
            total, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(
            total, N_KV_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            out = _te_varlen_call(q, k, v, cu, cu, total, total)
            out.float().sum().backward()
            torch.cuda.synchronize(device)
        _te_backend_cache[key] = True
        del q, k, v, out
    except Exception as error:
        _te_backend_cache[key] = False
        _te_value_fusion_cache[key] = False
        _te_probe_errors[key] = f"{type(error).__name__}: {error}"
        return False
    if 2 * HEAD_DIM > 256:
        _te_value_fusion_cache[key] = False
        return False
    try:
        q = torch.randn(
            total, N_Q_HEADS, HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        fused_kv = torch.randn(
            total, N_KV_HEADS, 2 * HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = fused_kv[..., :HEAD_DIM]
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            out = _te_varlen_call(q, k, fused_kv, cu, cu, total, total)
            out.float().sum().backward()
            torch.cuda.synchronize(device)
        ok = out.shape[-1] == 2 * HEAD_DIM
        del q, k, fused_kv, out
    except Exception as error:
        ok = False
        _te_probe_errors[key] = (
            "fused-value probe only (base varlen works): "
            f"{type(error).__name__}: {error}")
    del cu
    _te_value_fusion_cache[key] = ok
    return ok


def _te_varlen_eligible(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda" or tensor.dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
    return _te_backend_cache.get((idx, tensor.dtype, HEAD_DIM), False)


def _te_attention_contexts(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", enabled=False):
        k_ctx = _te_varlen_call(
            q, k, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        value_ctx = _te_varlen_call(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    return k_ctx, value_ctx


def _varlen_backend(tensor: torch.Tensor) -> Optional[str]:
    if ATTN_IMPL not in ("auto", "flash"):
        return None
    backend = None
    if _flash_varlen_eligible(tensor):
        backend = "flash"
    elif _te_varlen_eligible(tensor):
        backend = "transformer_engine"
    if backend is not None:
        idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
        report_key = (idx, tensor.dtype, HEAD_DIM, backend)
        if report_key not in _attention_backend_reported:
            _attention_backend_reported.add(report_key)
            ddp_print(
                f"[attention] packed varlen backend={backend} device=cuda:{idx} "
                f"dtype={tensor.dtype} head_dim={HEAD_DIM}")
    return backend


def _varlen_backend_failure_detail(
    device: torch.device, dtype: torch.dtype
) -> str:
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, dtype, HEAD_DIM)
    if _flash_varlen_func is None:
        flash = (
            f"FlashAttention import failed ({_flash_varlen_import_error})"
            if _flash_varlen_import_error
            else "FlashAttention varlen API is unavailable"
        )
    else:
        flash = (
            f"FlashAttention probe failed ({_flash_probe_errors[key]})"
            if key in _flash_probe_errors
            else "FlashAttention is ineligible for this device/dtype/head_dim"
        )
    te = (
        f"Transformer Engine probe failed ({_te_probe_errors[key]})"
        if key in _te_probe_errors
        else "Transformer Engine varlen DPA is unavailable or ineligible"
    )
    return f"{flash}; {te}"


def _varlen_value_fusion_enabled(backend: str, tensor: torch.Tensor) -> bool:
    if not ATTN_SDPA_VALUE_FUSION:
        return False
    idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
    key = (idx, tensor.dtype, HEAD_DIM)
    if backend == "flash":
        return _flash_value_fusion_cache.get(key, False)
    if backend == "transformer_engine":
        return _te_value_fusion_cache.get(key, False)
    return False


def _varlen_attention_contexts(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    fused_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fused_value is not None:
        k_view = fused_value[..., :HEAD_DIM]
        if backend == "flash":
            func = _get_flash_varlen_func()
            with torch.autocast(device_type="cuda", enabled=False):
                out = _call_flash_varlen(
                    func,
                    q, k_view, fused_value,
                    cu_seqlens_q, cu_seqlens_k,
                    int(max_seqlen_q), int(max_seqlen_k),
                    dropout_p=0.0,
                    causal=True,
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                )
        elif backend == "transformer_engine":
            with torch.autocast(device_type="cuda", enabled=False):
                out = _te_varlen_call(
                    q, k_view, fused_value,
                    cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k)
        else:
            raise ValueError(f"unknown varlen attention backend {backend!r}")
        return out.split(HEAD_DIM, dim=-1)
    if backend == "flash":
        return _flash_attention_contexts(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    if backend == "transformer_engine":
        return _te_attention_contexts(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    raise ValueError(f"unknown varlen attention backend {backend!r}")


_cuda_packed_gather_module = None
_cuda_packed_gather_tried = False


def _try_load_cuda_packed_gather():
    """Load the one-launch THD history packer used before varlen attention."""
    global _cuda_packed_gather_module, _cuda_packed_gather_tried
    if _cuda_packed_gather_tried:
        return _cuda_packed_gather_module
    _cuda_packed_gather_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_packed_gather_module = build_or_load(
            "loomformer_packed_gather",
            ["packed_gather/packed_gather_launcher.cu"],
            ptx_kernels={"packed_gather": "packed_gather/packed_gather_kernel.cu"},
        )
    except Exception as e:
        _cuda_packed_gather_module = None
        ddp_print(
            f"[loomformer] CUDA packed_gather failed ({type(e).__name__}: {e}).")
    return _cuda_packed_gather_module


def _cuda_packed_gather_or_fallback(tensor: torch.Tensor):
    if tensor.device.type != "cuda":
        return None
    module = _try_load_cuda_packed_gather()
    if module is not None:
        return module
    major, _minor = torch.cuda.get_device_capability(tensor.device)
    if major >= 8 and ATTN_IMPL in ("auto", "flash"):
        raise RuntimeError(
            "packed varlen attention needs the CUDA packed_gather extension, "
            "but it failed to build/load; refusing the O(history_chunks) "
            "PyTorch packer on a production FlashAttention GPU")
    return None


class _PackedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selectors, destinations, piece_offsets, count, *chunks):
        module = _cuda_packed_gather_or_fallback(chunks[0])
        if module is None:
            raise RuntimeError("CUDA packed_gather extension is unavailable")
        count = int(count)
        if count != len(chunks):
            raise ValueError(f"packed_gather expected {count} chunks, got {len(chunks)}")
        first = chunks[0]
        ctx.save_for_backward(selectors, destinations, piece_offsets)
        ctx.chunk_lengths = tuple(int(chunk.shape[1]) for chunk in chunks)
        ctx.batch = int(first.shape[0])
        ctx.heads = int(first.shape[2])
        ctx.head_dim = int(first.shape[3])
        return module.forward(
            list(chunks), selectors, destinations, piece_offsets)

    @staticmethod
    def backward(ctx, grad_output):
        selectors, destinations, piece_offsets = ctx.saved_tensors
        module = _cuda_packed_gather_module
        if module is None:
            raise RuntimeError("CUDA packed_gather disappeared before backward")
        grads = module.backward(
            grad_output.contiguous(),
            selectors,
            destinations,
            piece_offsets,
            list(ctx.chunk_lengths),
            ctx.batch,
            ctx.heads,
            ctx.head_dim,
        )
        return (None, None, None, None, *grads)


class _PackedGatherPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selectors, destinations, piece_offsets, count, *chunks):
        count = int(count)
        if len(chunks) != 2 * count:
            raise ValueError(
                f"packed_gather_pair expected {2 * count} tensors, got {len(chunks)}")
        k_chunks = chunks[:count]
        v_chunks = chunks[count:]
        module = _cuda_packed_gather_or_fallback(k_chunks[0])
        if module is None:
            raise RuntimeError("CUDA packed_gather extension is unavailable")
        first = k_chunks[0]
        ctx.save_for_backward(selectors, destinations, piece_offsets)
        ctx.chunk_lengths = tuple(int(chunk.shape[1]) for chunk in k_chunks)
        ctx.batch = int(first.shape[0])
        ctx.heads = int(first.shape[2])
        ctx.head_dim = int(first.shape[3])
        return module.forward_pair(
            list(k_chunks), list(v_chunks),
            selectors, destinations, piece_offsets)

    @staticmethod
    def backward(ctx, grad_output):
        selectors, destinations, piece_offsets = ctx.saved_tensors
        module = _cuda_packed_gather_module
        if module is None:
            raise RuntimeError("CUDA packed_gather disappeared before backward")
        grads = module.backward_pair(
            grad_output.contiguous(),
            selectors,
            destinations,
            piece_offsets,
            list(ctx.chunk_lengths),
            ctx.batch,
            ctx.heads,
            ctx.head_dim,
        )
        return (None, None, None, None, *grads)


@torch._dynamo.disable
def _pack_selected_chunk_history(
    chunks: Tuple[torch.Tensor, ...],
    packed: PackedChunkLayout,
) -> torch.Tensor:
    """Gather live document prefixes without concatenating completed history."""
    if len(chunks) != len(packed.piece_sizes):
        raise ValueError(
            f"packed selector count {len(packed.piece_sizes)} != chunk count {len(chunks)}")
    total = packed.destinations.numel()
    if total == 0:
        raise RuntimeError("packed attention selected no history")
    first = chunks[0]
    if first.device.type == "cuda" and _cuda_packed_gather_or_fallback(first) is not None:
        return _PackedGather.apply(
            packed.selectors, packed.destinations, packed.piece_offsets,
            len(chunks), *chunks)
    document_major = first.new_empty(total, first.shape[-2], first.shape[-1])
    offset = 0
    for chunk, size in zip(chunks, packed.piece_sizes):
        selector = packed.selectors[offset:offset + size]
        destination = packed.destinations[offset:offset + size]
        offset += size
        if size == 0:
            continue
        selected = chunk.reshape(
            -1, chunk.shape[-2], chunk.shape[-1]).index_select(0, selector)
        # index_copy_ accepts only int64 indices. Keep persistent metadata
        # int32 and widen only the current small index vector.
        document_major.index_copy_(0, destination.to(torch.int64), selected)
    return document_major


@torch._dynamo.disable
def _pack_selected_chunk_kv(
    k_chunks: Tuple[torch.Tensor, ...],
    v_chunks: Tuple[torch.Tensor, ...],
    packed: PackedChunkLayout,
) -> torch.Tensor:
    """Pack [K|V] directly into one THD allocation for fused value attention."""
    if len(k_chunks) != len(v_chunks) or len(k_chunks) != len(packed.piece_sizes):
        raise ValueError("K/V chunk history does not match packed selectors")
    total = packed.destinations.numel()
    if total == 0:
        raise RuntimeError("packed attention selected no history")
    first = k_chunks[0]
    if first.device.type == "cuda" and _cuda_packed_gather_or_fallback(first) is not None:
        return _PackedGatherPair.apply(
            packed.selectors, packed.destinations, packed.piece_offsets,
            len(k_chunks), *k_chunks, *v_chunks)
    fused = first.new_empty(total, first.shape[-2], 2 * first.shape[-1])
    offset = 0
    for k_chunk, v_chunk, size in zip(k_chunks, v_chunks, packed.piece_sizes):
        selector = packed.selectors[offset:offset + size]
        destination = packed.destinations[offset:offset + size]
        offset += size
        if size == 0:
            continue
        shape = (-1, k_chunk.shape[-2], k_chunk.shape[-1])
        k_selected = k_chunk.reshape(shape).index_select(0, selector)
        v_selected = v_chunk.reshape(shape).index_select(0, selector)
        piece = torch.cat((k_selected, v_selected), dim=-1)
        fused.index_copy_(0, destination.to(torch.int64), piece)
    return fused


def _packed_chunk_mask(
    layout: Optional[PackedAttentionLayout],
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    q_pos = torch.arange(start, end, device=device, dtype=torch.long).view(1, 1, end - start, 1)
    k_pos = torch.arange(end, device=device, dtype=torch.long).view(1, 1, 1, end)
    allowed = k_pos <= q_pos
    if layout is not None:
        q_seg = layout.segment_ids[:, start:end].view(-1, 1, end - start, 1)
        k_seg = layout.segment_ids[:, :end].view(-1, 1, 1, end)
        allowed = allowed & (q_seg == k_seg)
    return allowed


def _sdpa_compute_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    src_dtype = q.dtype
    mode = ATTN_SDPA_COMPUTE_DTYPE
    target: Optional[torch.dtype]
    if mode == "model":
        target = None
    elif mode == "fp32":
        target = torch.float32
    elif mode == "fp16":
        target = torch.float16
    elif mode == "bf16":
        target = torch.bfloat16
    elif mode == "auto" and src_dtype == torch.bfloat16 and not _bf16_efficient_sdpa_supported(q.device):
        target = torch.float32
    else:
        target = None
    if target is None or src_dtype == target:
        return q, k, v, src_dtype
    return q.to(target), k.to(target), v.to(target), src_dtype


class _RecomputeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            elif is_causal:
                T, S = q.shape[-2], k.shape[-2]
                q_pos = torch.arange(S - T, S, device=q.device)[:, None]
                k_pos = torch.arange(S, device=q.device)[None, :]
                causal = (k_pos <= q_pos).view(1, 1, T, S)
                scores = scores.masked_fill(~causal, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out = torch.matmul(p, v)
        mask_to_save = attn_mask if attn_mask is not None else torch.empty(0, device=q.device, dtype=torch.bool)
        ctx.save_for_backward(q, k, v, mask_to_save)
        ctx.has_mask = attn_mask is not None
        ctx.is_causal = bool(is_causal)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, attn_mask = ctx.saved_tensors
        scale = ctx.scale
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if ctx.has_mask:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        elif ctx.is_causal:
            T, S = q.shape[-2], k.shape[-2]
            q_pos = torch.arange(S - T, S, device=q.device)[:, None]
            k_pos = torch.arange(S, device=q.device)[None, :]
            causal = (k_pos <= q_pos).view(1, 1, T, S)
            scores = scores.masked_fill(~causal, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        dv = torch.matmul(p.transpose(-1, -2), grad_out)
        dp = torch.matmul(grad_out, v.transpose(-1, -2))
        ds = (dp - (dp * p).sum(dim=-1, keepdim=True)) * p
        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-1, -2), q) * scale
        return dq, dk, dv, None, None


def _attention_contexts_sdpa(qg: torch.Tensor, kg: torch.Tensor, vg: torch.Tensor,
                             attn_mask: Optional[torch.Tensor], is_causal: bool,
                             cat_label: str) -> Tuple[torch.Tensor, torch.Tensor]:
    q_sdpa, k_sdpa, v_sdpa, out_dtype = _sdpa_compute_inputs(qg, kg, vg)
    ac = (
        torch.autocast(device_type="cuda", enabled=False)
        if q_sdpa.device.type == "cuda" and q_sdpa.dtype == torch.float32
        else contextlib.nullcontext()
    )
    if ATTN_SDPA_VALUE_FUSION:
        with _profile_region(cat_label):
            kv = torch.cat((k_sdpa, v_sdpa), dim=-1)
        with ac:
            if ATTN_SDPA_RECOMPUTE_BACKWARD:
                out = _RecomputeAttention.apply(q_sdpa, k_sdpa, kv, attn_mask, is_causal)
            else:
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, kv, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
        out = out.to(dtype=out_dtype)
        return out.split(HEAD_DIM, dim=-1)
    with ac:
        if ATTN_SDPA_RECOMPUTE_BACKWARD:
            c_g = _RecomputeAttention.apply(q_sdpa, k_sdpa, v_sdpa, attn_mask, is_causal)
            kctx_g = _RecomputeAttention.apply(q_sdpa, k_sdpa, k_sdpa, attn_mask, is_causal)
        else:
            c_g = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
            kctx_g = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, k_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
    return kctx_g.to(dtype=out_dtype), c_g.to(dtype=out_dtype)


def _attention_contexts_packed_sdpa(
    q: torch.Tensor,
    k_compact: torch.Tensor,
    v_compact: torch.Tensor,
    layout: PackedAttentionLayout,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact compact fallback: one causal SDPA per packed document.

    This path is primarily the Pascal/CPU correctness floor.  It never creates
    a block-diagonal T² mask and never lets one document enter another's SDPA
    call.  Production Ampere+ training takes the varlen FlashAttention path.
    """
    B, T, _hq, _d = q.shape
    q_flat = q.reshape(B * T, N_Q_HEADS, HEAD_DIM)
    k_flat = k_compact.reshape(B * T, N_KV_HEADS, HEAD_DIM)
    v_flat = v_compact.reshape(B * T, N_KV_HEADS, HEAD_DIM)
    # One host transfer per batch layout, not one per layer/document.
    offsets = layout.cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    kctx_parts: List[torch.Tensor] = []
    value_parts: List[torch.Tensor] = []
    for begin, finish in zip(offsets[:-1], offsets[1:]):
        q_doc = q_flat[begin:finish].transpose(0, 1).unsqueeze(0)
        k_doc = GroupedQueryCausalSelfAttention._kv_to_q_heads(
            k_flat[begin:finish].unsqueeze(0)).transpose(1, 2)
        v_doc = GroupedQueryCausalSelfAttention._kv_to_q_heads(
            v_flat[begin:finish].unsqueeze(0)).transpose(1, 2)
        kctx, value = _attention_contexts_sdpa(
            q_doc, k_doc, v_doc, attn_mask=None, is_causal=True,
            cat_label="loom.attn.cat_kv_value_packed_fallback")
        kctx_parts.append(kctx.squeeze(0).transpose(0, 1))
        value_parts.append(value.squeeze(0).transpose(0, 1))
    kctx_flat = torch.cat(kctx_parts, dim=0)
    value_flat = torch.cat(value_parts, dim=0)
    return (
        kctx_flat.view(B, T, N_Q_HEADS, HEAD_DIM),
        value_flat.view(B, T, N_Q_HEADS, HEAD_DIM),
    )


_cuda_chunk_attn_module = None
_cuda_chunk_attn_tried = False


def _try_load_cuda_chunk_attn():
    global _cuda_chunk_attn_module, _cuda_chunk_attn_tried
    if _cuda_chunk_attn_tried:
        return _cuda_chunk_attn_module
    _cuda_chunk_attn_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_chunk_attn_module = build_or_load(
            "loomformer_chunk_attn",
            ["chunk_attn/chunk_attn_launcher.cu"],
            ptx_kernels={"chunk_attn": "chunk_attn/chunk_attn_kernel.cu"},
        )
    except Exception as e:
        _cuda_chunk_attn_module = None
        ddp_print(
            f"[loomformer] CUDA chunk_attention failed ({type(e).__name__}: {e}); "
            "using SLOWER PyTorch fallback.")
    return _cuda_chunk_attn_module


def _chunk_attention_list_reference(q, k_chunks, v_chunks, mask):
    qg = q.transpose(1, 2).float()
    score_parts = []
    key_parts = []
    value_parts = []
    for k, v in zip(k_chunks, v_chunks):
        kq = k.repeat_interleave(GQA_GROUP_SIZE, dim=2).transpose(1, 2).float()
        vq = v.repeat_interleave(GQA_GROUP_SIZE, dim=2).transpose(1, 2).float()
        score_parts.append(torch.matmul(qg, kq.transpose(-1, -2)) / math.sqrt(HEAD_DIM))
        key_parts.append(kq)
        value_parts.append(vq)
    scores = torch.cat(score_parts, dim=-1)
    if mask is None:
        T, S = q.shape[1], scores.shape[-1]
        q_pos = torch.arange(S - T, S, device=q.device)[:, None]
        k_pos = torch.arange(S, device=q.device)[None, :]
        mask = (k_pos <= q_pos).view(1, 1, T, S)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    kctx = torch.zeros_like(qg)
    ctx = torch.zeros_like(qg)
    offset = 0
    for kq, vq in zip(key_parts, value_parts):
        length = kq.shape[2]
        w = weights[..., offset:offset + length]
        kctx = kctx + torch.matmul(w, kq)
        ctx = ctx + torch.matmul(w, vq)
        offset += length
    dtype = q.dtype
    return kctx.transpose(1, 2).to(dtype), ctx.transpose(1, 2).to(dtype)


class _ChunkAttentionListFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, mask, count, *history):
        ext = _try_load_cuda_chunk_attn()
        if ext is None:
            raise RuntimeError("CUDA chunk attention is unavailable")
        n = int(count)
        ks = list(history[:n])
        vs = list(history[n:])
        kctx, value_ctx, lse = ext.chunk_attn_forward(
            q.contiguous(), ks, vs, mask.contiguous(), HEAD_DIM ** -0.5)
        ctx.count = n
        ctx.save_for_backward(q, mask, kctx, value_ctx, lse, *history)
        return kctx, value_ctx

    @staticmethod
    def backward(ctx, grad_kctx, grad_value_ctx):
        ext = _try_load_cuda_chunk_attn()
        if ext is None:
            raise RuntimeError("CUDA chunk attention is unavailable")
        q, mask, kctx, value_ctx, lse, *history = ctx.saved_tensors
        n = ctx.count
        grads = ext.chunk_attn_backward(
            grad_kctx.contiguous(), grad_value_ctx.contiguous(), q,
            list(history[:n]), list(history[n:]), mask, kctx, value_ctx, lse, HEAD_DIM ** -0.5)
        return grads[0], None, None, *grads[1:]


def _chunk_attention_list(q, k_chunks, v_chunks, mask):
    if q.is_cuda and q.dtype in (torch.float32, torch.float16, torch.bfloat16):
        ext = _try_load_cuda_chunk_attn()
        if ext is not None:
            mask_arg = (
                torch.empty(0, dtype=torch.bool, device=q.device)
                if mask is None
                else mask.to(device=q.device, dtype=torch.bool)
            )
            history = tuple(k_chunks) + tuple(v_chunks)
            return _ChunkAttentionListFused.apply(q, mask_arg, len(k_chunks), *history)
    return _chunk_attention_list_reference(q, k_chunks, v_chunks, mask)


class GroupedQueryCausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_weight = nn.Parameter(torch.empty(N + 2 * KV_DIM, N))
        self.o = nn.Linear(N, N, bias=False)
        self.rope = YaRNRotaryEmbedding()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.qkv_weight[:N], mean=0.0, std=fanin_std(N))
        nn.init.normal_(self.qkv_weight[N:N + KV_DIM], mean=0.0, std=fanin_std(N))
        nn.init.normal_(self.qkv_weight[N + KV_DIM:], mean=0.0, std=residual_std(N))
        init_linear_residual(self.o)

    @staticmethod
    def _split_q_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if D != N:
            raise ValueError(f"expected q last dim {N}, got {D}")
        return x.view(B, T, N_Q_HEADS, HEAD_DIM)

    @staticmethod
    def _split_kv_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if D != KV_DIM:
            raise ValueError(f"expected kv last dim {KV_DIM}, got {D}")
        return x.view(B, T, N_KV_HEADS, HEAD_DIM)

    @staticmethod
    def _merge_q_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, G, Dh = x.shape
        if G != N_Q_HEADS or Dh != HEAD_DIM:
            raise ValueError("bad query-head shape")
        return x.reshape(B, T, N)

    @staticmethod
    def _kv_to_q_heads(x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(GQA_GROUP_SIZE, dim=2)

    def _qkv(self, z: torch.Tensor):
        qkv = F.linear(z, self.qkv_weight)
        return torch.split(qkv, (N, KV_DIM, KV_DIM), dim=-1)

    def forward(self, z: torch.Tensor, attn_mask: Optional[Any] = None,
                position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        if T > SEQ_LEN:
            raise ValueError(f"sequence length {T} exceeds configured seq_len {SEQ_LEN}")
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
        if ATTN_IMPL == "flash" and varlen_backend is None:
            raise RuntimeError(
                "attn_impl='flash' requires a validated varlen forward+backward "
                "backend (FlashAttention or Transformer Engine), fp16/bf16 CUDA "
                "tensors, and compact causal layout metadata")

        if varlen_backend is not None:
            with _profile_region(f"loom.attn.{varlen_backend}_varlen_flat"):
                q_flat = q.reshape(B * T, N_Q_HEADS, HEAD_DIM)
                k_flat = k_compact.reshape(B * T, N_KV_HEADS, HEAD_DIM)
                v_flat = v_compact.reshape(B * T, N_KV_HEADS, HEAD_DIM)
                fused_value = None
                if _varlen_value_fusion_enabled(varlen_backend, q):
                    fused_value = torch.cat(
                        (k_compact, v_compact), dim=-1
                    ).reshape(B * T, N_KV_HEADS, 2 * HEAD_DIM)
                kctx_flat, c_flat = _varlen_attention_contexts(
                    varlen_backend,
                    q_flat, k_flat, v_flat,
                    flash_layout.cu_seqlens, flash_layout.cu_seqlens,
                    flash_layout.max_seqlen, flash_layout.max_seqlen,
                    fused_value=fused_value)
                k_ctx = kctx_flat.view(B, T, N_Q_HEADS, HEAD_DIM)
                c = c_flat.view(B, T, N_Q_HEADS, HEAD_DIM)
        elif packed is not None and ATTN_IMPL != "manual":
            with _profile_region("loom.attn.sdpa_packed_fallback"):
                k_ctx, c = _attention_contexts_packed_sdpa(
                    q, k_compact, v_compact, packed)
        elif ATTN_IMPL in ("auto", "sdpa"):
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
            scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
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
        scores = torch.matmul(qg, kg.transpose(-1, -2)).float() / math.sqrt(HEAD_DIM)
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
            shape = (B, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
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
            scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
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
            shape = (B, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
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
        scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
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
    ):
        if cache_len >= SEQ_LEN:
            raise ValueError(f"generation exceeded seq_len={SEQ_LEN}; no wraparound support")
        B = z.shape[0]
        q_p, k_p, v_p = self._qkv(z)
        q = self._split_q_heads(q_p)
        k_new_q = self._kv_to_q_heads(self._split_kv_heads(k_p))
        pos = torch.tensor([int(position_id)], device=z.device, dtype=torch.long)
        q, k_new_q = self.rope(q, k_new_q, pos)
        k_new = k_new_q.view(B, 1, N_KV_HEADS, GQA_GROUP_SIZE, HEAD_DIM)[:, :, :, 0, :]
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
                k_cache = z.new_zeros(B, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
                v_cache = z.new_zeros(B, SEQ_LEN, N_KV_HEADS, HEAD_DIM)
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
            scores = torch.matmul(qg, kg.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
            a = torch.softmax(scores.float(), dim=-1).to(vg.dtype)
            c_g = torch.matmul(a, vg)
            kctx_g = torch.matmul(a, kg)
        c = c_g.transpose(1, 2).contiguous()
        k_ctx = kctx_g.transpose(1, 2).contiguous()
        # Возвращаем ПОЛНЫЙ буфер (не срез) -- вызывающий код хранит его в LayerCache и
        # переиспользует на следующем шаге; срез k_all был только для этого forward.
        return self.o(self._merge_q_heads(c)), q, k_ctx, c, k_cache, v_cache

    def forward_chunk(
        self,
        z: torch.Tensor,
        past_k_chunks: tuple,
        past_v_chunks: tuple,
        position_ids: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
    ):
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
        if ATTN_IMPL == "flash" and varlen_backend is None:
            raise RuntimeError(
                "attn_impl='flash' requires a validated FlashAttention or "
                "Transformer Engine varlen forward+backward backend")
        if varlen_backend is not None:
            with _profile_region(f"loom.attn.{varlen_backend}_varlen_chunk"):
                q_packed = q.reshape(B * T, N_Q_HEADS, HEAD_DIM)
                fused_value = None
                if _varlen_value_fusion_enabled(varlen_backend, q):
                    fused_value = _pack_selected_chunk_kv(
                        k_chunks, v_chunks, packed_chunk)
                    k_packed = fused_value[..., :HEAD_DIM]
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
                k_ctx = kctx.view(B, T, N_Q_HEADS, HEAD_DIM)
                c = value_ctx.view(B, T, N_Q_HEADS, HEAD_DIM)
        elif ATTN_IMPL in ("auto", "sdpa"):
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



class DepthAttn(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        n_sub = 2 * LAYERS
        self.kv_weight = nn.Parameter(torch.empty(2 * N, N))
        if DEPTH_ATTN_READOUT == "per-sublayer":
            self.w_o_weight = nn.Parameter(torch.empty(n_sub, N, N))
            self.w_o = None
        else:
            self.register_parameter("w_o_weight", None)
            self.w_o = nn.Linear(N, N, bias=False)
        self.q_params = nn.Parameter(torch.empty(n_sub, N_Q_HEADS, HEAD_DIM))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.kv_weight[:N], mean=0.0, std=fanin_std(N))
        nn.init.normal_(self.kv_weight[N:], mean=0.0, std=residual_std(N))
        if self.w_o_weight is not None:
            nn.init.normal_(self.w_o_weight, mean=0.0, std=residual_std(N))
        else:
            init_linear_residual(self.w_o)
        nn.init.normal_(self.q_params, mean=0.0, std=fanin_std(HEAD_DIM))

    def project(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = h.shape
        kv = F.linear(h, self.kv_weight)
        k, v = torch.split(kv, (N, N), dim=-1)
        k = k.view(B, T, N_Q_HEADS, HEAD_DIM)
        v = v.view(B, T, N_Q_HEADS, HEAD_DIM)
        if DEPTH_ATTN_QKV_RMS:
            # Match the expected initialization RMS, rather than forcing K/V to
            # unit scale and silently sharpening depth softmax at step zero.
            k = fixed_rms(k, FANIN_GAIN)
            v_beta = DEEPNORM_BETA if RESIDUAL_INIT == "beta" else 1.0
            v = fixed_rms(v, FANIN_GAIN * v_beta)
        return k, v

    def forward(self, sub_idx: int, hist_k, hist_v) -> Tuple[torch.Tensor, torch.Tensor]:
        q_row = self.q_params[sub_idx]
        if DEPTH_ATTN_QKV_RMS:
            q_row = fixed_rms(q_row, fanin_std(HEAD_DIM))
        if torch.is_tensor(hist_k):
            if q_row.dtype != hist_k.dtype:
                q_row = q_row.to(hist_k.dtype)
            q = q_row.view(1, 1, N_Q_HEADS, HEAD_DIM)
            d = depth_attn_online_cuda(q_row, hist_k, hist_v)
            if d is None:
                d = _depth_attn_online_tensor_pytorch(q, hist_k, hist_v)
            B, T = hist_k.shape[:2]
        else:
            if q_row.dtype != hist_k[0].dtype:
                q_row = q_row.to(hist_k[0].dtype)
            d = depth_attn_online_list_cuda(q_row, hist_k, hist_v)
            if d is None:
                d = _depth_attn_online_list_pytorch(q_row, hist_k, hist_v)
            B, T = hist_k[0].shape[:2]
        d_flat = d.reshape(B, T, N)
        skip = (
            F.linear(d_flat, self.w_o_weight[sub_idx])
            if self.w_o_weight is not None
            else self.w_o(d_flat)
        )
        return skip, d



def powlu(x: torch.Tensor, m: float = 3.0) -> torch.Tensor:
    x_pos_safe = x.clamp(min=1e-12)
    exponent = 1.0 + m / (torch.sqrt(x_pos_safe) + 1.0)
    pos_val = x_pos_safe.pow(exponent) * torch.sigmoid(x)
    neg_val = x.pow(2) * torch.sigmoid(x)
    return torch.where(x > 0, pos_val, neg_val)


def powlu_gate(x2: torch.Tensor, m: float = 3.0) -> torch.Tensor:
    """Apply the positive-input PowLU gate used by the GLU path."""
    exponent = m / (torch.sqrt(x2) + 1.0)
    return x2.pow(exponent) * torch.sigmoid(x2)


def act_fn(x: torch.Tensor) -> torch.Tensor:
    """Apply the configured GELU or PowLU activation."""
    return F.gelu(x) if ACTIVATION == "gelu" else powlu(x, POWLU_M)


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale and no bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).to(dtype) * self.weight


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """Chunked fused (LM-head projection + softmax cross-entropy).

    Direct port of the Liger-Kernel algorithm (https://github.com/linkedin/
    Liger-Kernel, src/liger_kernel/ops/fused_linear_cross_entropy.py), minus
    the Triton kernel (plain PyTorch ops instead of a fused Triton column
    block, since this repo has no Triton dependency): the [N, D] @ [D, V]^T
    head projection and the token-level cross-entropy are computed one
    row-chunk at a time, so only a [chunk_size, VOCAB] logits slice (plus its
    exact gradient) is ever alive instead of the full [N, VOCAB] tensor.
    Gradients are exact (softmax - onehot per chunk, algebraically identical
    to the unchunked computation) -- this is NOT activation-checkpoint
    recompute, there is no extra forward pass.

    Matching Liger's forward exactly: there is exactly ONE host sync for the
    non-ignored token count (needed for mean reduction), taken once before
    the chunk loop -- never per-chunk. An early `if chunk has no valid
    targets: skip` check would need its own per-chunk `.item()`/`bool()`
    sync, stalling the CUDA pipeline every single chunk for no algorithmic
    benefit (chunks are still masked correctly either way), so it is
    deliberately not done, exactly as upstream does not do it either.
    """

    @staticmethod
    def _prefers_fp32_gemm(hidden: torch.Tensor) -> bool:
        """On sm_61-class GPUs (Pascal: GTX 1080, Tesla P4 -- no bf16 tensor
        cores below sm_80) `autocast(bf16)` lowers `x @ W` to
        `magma_sgemmEx_kernel<float, __nv_bfloat16, ...>`, which does a
        serialized bf16->fp32 convert pass before the fp32 multiply-add --
        verified via torch.profiler to dominate this function's CUDA time
        (>90%) on such hardware. Forcing the GEMM operands to fp32 and
        disabling autocast for the call instead dispatches plain cuBLAS
        `sgemm`, which is substantially faster here. On sm_80+ (native bf16
        tensor cores) this would throw away real speedup, so it's gated on
        compute capability. Gated on the *active autocast state*, not
        `hidden.dtype`: whether the matmul actually lowers to bf16 depends
        on whether autocast is live right now, not on what dtype the
        incoming tensor happens to already be."""
        if not hidden.is_cuda:
            return False
        if _cuda_autocast_dtype_or_none() != torch.bfloat16:
            return False
        try:
            major, _minor = torch.cuda.get_device_capability(hidden.device)
        except Exception:
            return False
        return major < 8

    @staticmethod
    def _default_chunk_size(N: int, D: int, V: int) -> int:
        """Liger's memory-balancing formula: pick the chunk size so the
        transient [chunk, V] logits buffer is on the same order as the
        [N, D] hidden-state tensor, i.e. `inc_factor = ceil(V/D)`,
        `chunk_size = next_pow2(ceil(N / inc_factor))`."""
        if N <= 0:
            return 1
        inc_factor = -(-V // D)
        raw = -(-N // inc_factor)
        pow2 = 1 << max(raw - 1, 0).bit_length()
        return max(1, min(pow2, N))

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                ignore_index: int, chunk_size: int) -> torch.Tensor:
        N, D = hidden.shape
        V = weight.shape[0]
        step = int(chunk_size) if chunk_size else _FusedLinearCrossEntropy._default_chunk_size(N, D, V)
        step = max(1, min(step, max(N, 1)))

        valid_mask = targets.ne(ignore_index)
        denom = float(max(int(valid_mask.sum().item()), 1))

        want_weight_grad = weight.requires_grad
        grad_hidden = torch.zeros(N, D, dtype=torch.float32, device=hidden.device)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if want_weight_grad else None
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)

        # Cast once (not per chunk). Under autocast this is a no-op cost --
        # `hidden` already arrives in the autocast dtype (e.g. bf16) -- and it
        # also makes the manual backward matmuls below correct when autocast
        # is off (amp_dtype: fp32) where hidden/weight already match anyway.
        force_fp32_gemm = _FusedLinearCrossEntropy._prefers_fp32_gemm(hidden)
        compute_dtype = torch.float32 if force_fp32_gemm else hidden.dtype
        weight_c = weight if weight.dtype == compute_dtype else weight.to(compute_dtype)
        gemm_ctx = (
            torch.autocast(device_type="cuda", enabled=False)
            if force_fp32_gemm else contextlib.nullcontext()
        )

        with gemm_ctx:
            for s in range(0, N, step):
                e = min(s + step, N)
                h_chunk = hidden[s:e].to(compute_dtype) if force_fp32_gemm else hidden[s:e]
                t_chunk = targets[s:e]
                valid = valid_mask[s:e]
                logits_chunk = F.linear(h_chunk, weight_c).float()
                logp = F.log_softmax(logits_chunk, dim=-1)
                safe_t = t_chunk.clamp_min(0).unsqueeze(1)
                nll = -logp.gather(1, safe_t).squeeze(1)
                loss_sum += torch.where(valid, nll, nll.new_zeros(())).sum()
                grad_logits = logp.exp()
                grad_logits.scatter_add_(1, safe_t, -torch.ones_like(safe_t, dtype=grad_logits.dtype))
                grad_logits *= valid.unsqueeze(1)
                grad_logits = grad_logits.to(compute_dtype)
                grad_hidden[s:e] = (grad_logits @ weight_c).float()
                if want_weight_grad:
                    grad_weight += (grad_logits.t() @ h_chunk).float()
                del logits_chunk, logp, grad_logits

        grad_hidden = (grad_hidden / denom).to(hidden.dtype)
        ctx.want_weight_grad = want_weight_grad
        if want_weight_grad:
            ctx.save_for_backward(grad_hidden, (grad_weight / denom).to(weight.dtype))
        else:
            ctx.save_for_backward(grad_hidden)
        return loss_sum / denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.want_weight_grad:
            grad_hidden, grad_weight = ctx.saved_tensors
            return grad_hidden * grad_output, grad_weight * grad_output, None, None, None
        (grad_hidden,) = ctx.saved_tensors
        return grad_hidden * grad_output, None, None, None, None


@torch._dynamo.disable
def _fused_linear_cross_entropy_eager(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    chunk_size: int,
) -> torch.Tensor:
    """Explicit Dynamo boundary for the hand-fused CE implementation.

    Its one deliberate host sync computes the exact mean denominator and its
    Python chunk loop bounds peak logits memory. Letting Dynamo enter it only
    produces a noisy Tensor.item() graph break; compiling the surrounding
    model regions while executing this custom autograd op eager is the
    intended behavior.
    """
    return _FusedLinearCrossEntropy.apply(
        hidden, weight, targets, ignore_index, chunk_size)


class Block(nn.Module):
    def __init__(self, ablation: bool = False) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(N)
        self.attn = GroupedQueryCausalSelfAttention()
        self.ln_ffn = nn.LayerNorm(N)
        self.ffn = ParaplexFFN(ablation=ablation)


class Model(nn.Module):
    def __init__(self, cfg: Config, ablation: bool = False) -> None:
        super().__init__()
        # Runtime Tria geometry has exactly one owner. Keep the Config object
        # itself (also useful for intentional live overrides in loomchat) rather
        # than copying alpha/beta/W into module or process globals.
        self.cfg = cfg
        self.ablation = bool(ablation)
        self.emb = nn.Embedding(VOCAB, N)
        self.blocks = nn.ModuleList([Block(ablation=ablation) for _ in range(LAYERS)])
        self.depth_attn = DepthAttn()
        self.head = nn.Linear(N, VOCAB, bias=False)
        if TIED_EMBEDDINGS:
            self.head.weight = self.emb.weight
        self.ln_final = RMSNorm(N) if FINAL_NORM_ENABLED else None
        self.last_tria_depth_carry: Optional[torch.Tensor] = None
        self.last_tria_document_carry: Optional[torch.Tensor] = None
        self.capture_tria_depth_carry: bool = False
        if TRIA_CARRY_ENABLED:
            reader = tria.SharedTriaReader(k=32)
            self.tria_agg = tria.TriaAggregator(reader, N)
            self.tria_final_ca = tria.TriaFinalCrossAttention(
                N, gamma_max=TRIA_GAMMA_MAX, raw_gamma_init=TRIA_RAW_GAMMA_INIT)
        else:
            self.tria_agg = None
            self.tria_final_ca = None
        # SFT (see loomsft.py): flip to False so refeed fires ONLY on explicit
        # <CARRY>, never on the dense W-token deadline pretrain relies on --
        # the model has no fixed-grid dependency to begin with (deadline is a
        # SAFETY fallback for raw/undocumented pretrain streams, not a learned
        # requirement), so disabling it introduces no train/inference shift.
        self.tria_hard_fire_enabled = True
        self.last_tria_document_carry_stats: Optional[dict] = None
        self.last_tria_fire_mask: Optional[torch.Tensor] = None
        self._disable_structurally_unused_params()
        self.reset_parameters()

    def _disable_structurally_unused_params(self) -> None:
        if not TRIA_CARRY_ENABLED or not self.blocks:
            return
        # Block 0 never receives a tria carrier input, so this gate scalar is
        # outside the training graph on every step.
        first_gate = getattr(self.blocks[0].ffn, "identity_gate", None)
        if first_gate is not None:
            first_gate.raw_alpha.requires_grad_(False)
        # The last block never emits p_out to another block, so its selector is
        # also unreachable from the loss.
        last_selector = getattr(self.blocks[-1].ffn, "gate_selector", None)
        if last_selector is not None:
            last_selector.logits.requires_grad_(False)

    def _head_in(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the optional final norm and output head."""
        return self.head(self.ln_final(x) if self.ln_final is not None else x)

    def _head_or_loss(self, h: torch.Tensor, labels: Optional[torch.Tensor],
                       ignore_index: int) -> torch.Tensor:
        """Return logits (labels=None, unchanged path) or the LM loss.

        With `FUSED_LINEAR_CE` on (Config.fused_linear_ce), the loss is
        computed via `_FusedLinearCrossEntropy`, which never materializes the
        full [B*T, VOCAB] logits tensor -- see that class's docstring.
        """
        if labels is None:
            return self._head_in(h)
        if not FUSED_LINEAR_CE:
            logits = self._head_in(h)
            return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                    ignore_index=ignore_index)
        hidden = self.ln_final(h) if self.ln_final is not None else h
        return _fused_linear_cross_entropy_eager(
            hidden.reshape(-1, hidden.shape[-1]), self.head.weight, labels.reshape(-1),
            ignore_index, FUSED_LINEAR_CE_CHUNK_SIZE)

    def reset_parameters(self) -> None:
        init_embedding_fanin(self.emb)

    def _build_tria_document_reset_mask(
        self,
        idx: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        del idx  # unused now that <CARRY> no longer triggers a document reset
        reset = position_ids.eq(0)
        reset[:, 0] = True
        return reset

    def _block_core(
        self,
        block: "Block",
        h: torch.Tensor,
        hist_k: list,
        hist_v: list,
        sub_idx0: int,
        attn_out: torch.Tensor,
        q_h: torch.Tensor,
        k_ctx_h: torch.Tensor,
        c_h: torch.Tensor,
        position_ids: torch.Tensor,
        phase_trace: Optional[torch.Tensor],
        carry_prev: Optional[torch.Tensor],
        p_in: Optional[torch.Tensor],
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        is_first_block: bool,
        is_last_block: bool,
    ):
        sub_idx = sub_idx0
        tria_axis = (sub_idx0 // 2) % 3
        tria_alpha = float(self.cfg.tria_carrier_alpha)
        skip, _ = self.depth_attn(sub_idx, hist_k, hist_v)
        if RESIDUAL_BRANCH_RMS_CAP is not None:
            skip = capped_rms(skip, RESIDUAL_BRANCH_RMS_CAP)
            attn_out = capped_rms(attn_out, RESIDUAL_BRANCH_RMS_CAP)
        h = block.ln_attn(skip + attn_out)
        k_i, v_i = self.depth_attn.project(h)
        hist_k = [*hist_k, k_i]
        hist_v = [*hist_v, v_i]
        sub_idx += 1

        skip, d_h = self.depth_attn(sub_idx, hist_k, hist_v)
        want_tria = TRIA_CARRY_ENABLED and not self.ablation
        if want_tria:
            ffn_out, next_phase_trace, (r, i, o) = block.ffn(
                h, q_h, k_ctx_h, c_h, d_h,
                phase_trace=phase_trace,
                phase_reset_mask=position_ids.eq(0),
                return_tria=True,
                p_in=p_in,
            )
            tria.record_depth_replay(r, i, o, tria_axis)
            if is_first_block and accT_seed is not None:
                if seed_valid is None:
                    raise ValueError("seed_valid is required with accT_seed")
                if is_last_block:
                    carry_new = tria.tria_init_seed(
                        r, i, o, accT_seed, seed_valid, axis=tria_axis,
                        alpha=tria_alpha)
                    p_out = None
                else:
                    w = torch.softmax(block.ffn.gate_selector.logits, dim=0)
                    carry_new, p_out = tria.tria_init_seed_and_gate(
                        r, i, o, accT_seed, seed_valid, w, axis=tria_axis,
                        alpha=tria_alpha)
            elif is_last_block:
                carry_new = (
                    tria.tria_init(r, i, o, axis=tria_axis, alpha=tria_alpha)
                    if carry_prev is None
                    else tria.tria_step(
                        r, i, o, carry_prev, axis=tria_axis, alpha=tria_alpha)
                )
                p_out = None
            else:
                w = torch.softmax(block.ffn.gate_selector.logits, dim=0)
                carry_new, p_out = (
                    tria.tria_init_and_gate(
                        r, i, o, w, axis=tria_axis, alpha=tria_alpha)
                    if carry_prev is None
                    else tria.tria_step_and_gate(
                        r, i, o, carry_prev, w, axis=tria_axis, alpha=tria_alpha)
                )
        else:
            ffn_out, next_phase_trace = block.ffn(
                h, q_h, k_ctx_h, c_h, d_h,
                phase_trace=phase_trace,
                phase_reset_mask=position_ids.eq(0),
            )
            carry_new = carry_prev
            p_out = None

        if RESIDUAL_BRANCH_RMS_CAP is not None:
            skip = capped_rms(skip, RESIDUAL_BRANCH_RMS_CAP)
            ffn_out = capped_rms(ffn_out, RESIDUAL_BRANCH_RMS_CAP)
        h = block.ln_ffn(skip + ffn_out)
        if not is_last_block:
            k_i, v_i = self.depth_attn.project(h)
            hist_k = [*hist_k, k_i]
            hist_v = [*hist_v, v_i]
        return h, hist_k, hist_v, next_phase_trace, carry_new, p_out

    def _run_block(
        self,
        block: "Block",
        h: torch.Tensor,
        hist_k: list,
        hist_v: list,
        sub_idx0: int,
        attn_mask: Optional[Any],
        position_ids: torch.Tensor,
        carry_prev: Optional[torch.Tensor],
        p_in: Optional[torch.Tensor],
        is_last_block: bool = False,
    ):
        attn_out, q_h, k_ctx_h, c_h = block.attn(
            h, attn_mask=attn_mask, position_ids=position_ids)
        h, hist_k, hist_v, _, carry_new, p_out = self._block_core(
            block, h, hist_k, hist_v, sub_idx0,
            attn_out, q_h, k_ctx_h, c_h, position_ids,
            None, carry_prev, p_in, None, None, False, is_last_block)
        return h, hist_k, hist_v, carry_new, p_out

    def _run_block_chunk(
        self,
        block: "Block",
        h: torch.Tensor,
        hist_k: list,
        hist_v: list,
        sub_idx0: int,
        position_ids: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        past_k_chunks: tuple,
        past_v_chunks: tuple,
        phase_trace: Optional[torch.Tensor],
        carry_prev: Optional[torch.Tensor],
        p_in: Optional[torch.Tensor],
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        is_first_block: bool,
        is_last_block: bool,
    ):
        attn_out, q_h, k_ctx_h, c_h, k_new, v_new = block.attn.forward_chunk(
            h, past_k_chunks, past_v_chunks, position_ids, attention_layout, packed_chunk)
        h, hist_k, hist_v, next_phase_trace, carry_new, p_out = self._block_core(
            block, h, hist_k, hist_v, sub_idx0,
            attn_out, q_h, k_ctx_h, c_h, position_ids,
            phase_trace, carry_prev, p_in, accT_seed, seed_valid,
            is_first_block, is_last_block)
        return h, hist_k, hist_v, k_new, v_new, next_phase_trace, carry_new, p_out

    def _run_chunk_stack_impl(self, h_emb_chunk: torch.Tensor, position_ids_chunk: torch.Tensor,
                              attention_layout: Optional[PackedAttentionLayout],
                              packed_chunk: PackedChunkLayout, layer_states: list,
                              accT_seed: Optional[torch.Tensor], seed_valid: Optional[torch.Tensor],
                              endpoint_reset: torch.Tensor, want_endpoint: bool, want_tail: bool,
                              replay_tape):
        """Run the full block stack for one temporal chunk and return tensor state."""
        n_blocks = len(self.blocks)
        k0, v0 = self.depth_attn.project(h_emb_chunk)
        hist_k = [k0]
        hist_v = [v0]
        h = h_emb_chunk
        carry = None
        p = None
        k_new_out = []
        v_new_out = []
        phase_out = []
        with tria.depth_replay_scope(
            seed=accT_seed, seed_valid=seed_valid, tape=replay_tape
        ):
            for bi, block in enumerate(self.blocks):
                ls = layer_states[bi]
                h, hist_k, hist_v, k_new, v_new, next_phase_trace, carry, p = self._run_block_chunk(
                    block, h, hist_k, hist_v, 2 * bi, position_ids_chunk,
                    attention_layout, packed_chunk,
                    ls.k_chunks, ls.v_chunks, ls.phase_trace, carry, p,
                    accT_seed, seed_valid, bi == 0, bi == n_blocks - 1)
                k_new_out.append(k_new)
                v_new_out.append(v_new)
                phase_out.append(next_phase_trace)
            endpoint = (
                tria.temporal_carry_endpoint(
                    carry, endpoint_reset,
                    initial_state=accT_seed,
                )
                if want_endpoint else None
            )
            tail = carry[:, -1].detach().contiguous() if want_tail else None
        return (h, endpoint, tail, *k_new_out, *v_new_out, *phase_out)

    @torch._dynamo.disable
    def _run_chunk_stack_checkpointed(
        self,
        h_emb_chunk: torch.Tensor,
        position_ids_chunk: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        layer_states: list,
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        endpoint_reset: torch.Tensor,
        want_endpoint: bool,
        want_tail: bool,
    ):
        """Run only the activation-checkpointed stack behind a Dynamo boundary.

        Its non-reentrant checkpoint uses a per-call mutable replay tape plus
        different original/recompute Python contexts. PyTorch 2.5 Dynamo tries
        to turn checkpoint into a higher-order graph op, but does not support
        our nested ``context_fn`` and, more importantly, must not freeze the
        tape/TLS mutations into a compiled graph. Keep this correctness-
        sensitive region eager; torch.compile resumes on the tensor-only
        regions around it.
        """
        # Custom autograd ctx objects are created in the original forward,
        # while r/i/o are regenerated later. Keep the tape identity stable
        # across both passes so those ctx objects see the refilled entries.
        replay_tape = tria.new_depth_replay_tape()
        holder: dict = {}

        def context_fn():
            return (
                contextlib.nullcontext(),
                _activation_checkpoint_recompute_context(holder),
            )

        flat = torch.utils.checkpoint.checkpoint(
            self._run_chunk_stack_impl,
            h_emb_chunk,
            position_ids_chunk,
            attention_layout,
            packed_chunk,
            layer_states,
            accT_seed,
            seed_valid,
            endpoint_reset,
            want_endpoint,
            want_tail,
            replay_tape,
            use_reentrant=False,
            context_fn=context_fn,
        )
        # The original pass populated the stable tape identity, but its
        # tensors must not outlive the checkpointed region. Recompute resets
        # and refills the same tape before Tria backward uses it.
        replay_tape.release_inputs()
        # Forward updates the secant EMA once. Recompute must use exactly that
        # per-layer snapshot without updating the persistent buffer again.
        holder["anchor_overrides"] = {
            id(block.ffn): block.ffn.beta_anchor.detach().clone()
            for block in self.blocks
        }
        return flat

    @torch._dynamo.disable
    def _run_chunk_stack(
        self,
        h_emb_chunk: torch.Tensor,
        position_ids_chunk: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        layer_states: list,
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        endpoint_reset: torch.Tensor,
        want_endpoint: bool,
        want_tail: bool,
    ):
        """Run the replay-sensitive temporal stack as one eager island."""
        n_blocks = len(self.blocks)
        if GRAD_CHECKPOINTING and self.training:
            flat = self._run_chunk_stack_checkpointed(
                h_emb_chunk,
                position_ids_chunk,
                attention_layout,
                packed_chunk,
                layer_states,
                accT_seed,
                seed_valid,
                endpoint_reset,
                want_endpoint,
                want_tail,
            )
        else:
            replay_tape = tria.new_depth_replay_tape()
            flat = self._run_chunk_stack_impl(
                h_emb_chunk, position_ids_chunk, attention_layout, packed_chunk, layer_states,
                accT_seed, seed_valid, endpoint_reset, want_endpoint, want_tail,
                replay_tape)
        h = flat[0]
        endpoint = flat[1]
        tail = flat[2]
        k_new = flat[3:3 + n_blocks]
        v_new = flat[3 + n_blocks:3 + 2 * n_blocks]
        phase = flat[3 + 2 * n_blocks:3 + 3 * n_blocks]
        new_layer_states = [
            TrainChunkLayerState(
                k_chunks=layer_states[i].k_chunks + (k_new[i],),
                v_chunks=layer_states[i].v_chunks + (v_new[i],),
                phase_trace=phase[i],
            )
            for i in range(n_blocks)
        ]
        return h, endpoint, tail, new_layer_states

    def forward(self, idx: torch.Tensor, attn_mask: Optional[Any] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, ignore_index: int = -100) -> torch.Tensor:
        """Forward pass. With `labels=None` (default, all existing call sites
        unchanged) this returns full [B,T,VOCAB] logits exactly as before.
        Passing `labels` returns the scalar LM loss instead; when
        `Config.fused_linear_ce` is on, that loss is computed without ever
        materializing the full logits tensor (see `_head_or_loss`)."""
        B, T = idx.shape
        if T > SEQ_LEN:
            raise ValueError(f"input length {T} exceeds configured seq_len {SEQ_LEN}")
        # Backend probes perform a tiny backward and therefore belong outside
        # activation-checkpointed regions.  Results are cached per device/dtype.
        compute_dtype = _attention_compute_dtype(idx.device, self.emb.weight.dtype)
        if ATTN_IMPL in ("auto", "flash") and idx.device.type == "cuda":
            _probe_flash_value_fusion(idx.device, compute_dtype)
            idx_cuda = torch.cuda.current_device() if idx.device.index is None else int(idx.device.index)
            if not _flash_backend_cache.get((idx_cuda, compute_dtype, HEAD_DIM), False):
                _probe_te_value_fusion(idx.device, compute_dtype)
            major, _minor = torch.cuda.get_device_capability(idx_cuda)
            optimized = (
                _flash_backend_cache.get((idx_cuda, compute_dtype, HEAD_DIM), False)
                or _te_backend_cache.get((idx_cuda, compute_dtype, HEAD_DIM), False)
            )
            if (
                optimized
                and TRIA_CARRY_ENABLED
                and TRIA_TEMPORAL_ENABLED
                and not self.ablation
                and _try_load_cuda_packed_gather() is None
                and major >= 8
            ):
                raise RuntimeError(
                    "validated varlen attention is available, but the CUDA "
                    "packed_gather extension failed to build/load; refusing "
                    "the O(history_chunks) fallback in production mode")
            if (
                ATTN_IMPL == "auto"
                and major >= 8
                and compute_dtype in (torch.float16, torch.bfloat16)
                and not optimized
            ):
                raise RuntimeError(
                    "attn_impl='auto' found no validated varlen forward+backward "
                    f"backend on cuda:{idx_cuda} (SM{major}x, {compute_dtype}). "
                    f"{_varlen_backend_failure_detail(idx.device, compute_dtype)}. "
                    "Install a compatible flash-attn or Transformer Engine build; "
                    "set attn_impl='sdpa' explicitly only if the slow fallback is intentional.")
        if (
            ATTN_IMPL in ("auto", "sdpa")
            and ATTN_SDPA_COMPUTE_DTYPE == "auto"
            and idx.device.type == "cuda"
        ):
            _bf16_efficient_sdpa_supported(idx.device)
        if position_ids is None:
            position_ids = torch.arange(T, device=idx.device, dtype=torch.long).view(1, T).expand(B, T)
        else:
            position_ids = position_ids.to(device=idx.device, dtype=torch.long)
        want_chunked = TRIA_CARRY_ENABLED and TRIA_TEMPORAL_ENABLED and not self.ablation
        if not want_chunked:
            effective_attn = attn_mask
            if attn_mask is None and ATTN_IMPL in ("auto", "flash"):
                effective_attn = _unpacked_attention_layout(B, T, idx.device)
            return self._forward_flat(idx, attn_mask=effective_attn, position_ids=position_ids,
                                       labels=labels, ignore_index=ignore_index)
        return self._forward_chunked(idx, attn_mask=attn_mask, position_ids=position_ids,
                                      labels=labels, ignore_index=ignore_index)

    def _forward_chunked(self, idx: torch.Tensor, attn_mask: Optional[Any],
                          position_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                          ignore_index: int = -100) -> torch.Tensor:
        B, T = idx.shape
        W = int(self.cfg.tria_temporal_window)
        h_emb = self.emb(idx)
        document_reset = self._build_tria_document_reset_mask(idx, position_ids)
        if isinstance(attn_mask, PackedAttentionLayout):
            attention_layout = attn_mask
        elif attn_mask is None:
            attention_layout = _unpacked_attention_layout(B, T, idx.device)
        else:
            # Compatibility for old callers that still provide a dense
            # block-causal mask: recover its document layout from the already
            # authoritative reset positions, then never retain/slice the mask.
            seg = torch.cumsum(position_ids.eq(0).to(torch.int32), dim=1) - 1
            attention_layout = packed_layout_from_segment_ids(seg)
        layer_states = [TrainChunkLayerState() for _ in self.blocks]
        carry_token_id = CARRY_TOKEN_ID
        explicit_fire = (
            idx.eq(int(carry_token_id))
            if carry_token_id is not None
            else torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        )
        if self.tria_hard_fire_enabled:
            dense_pos = torch.arange(T, device=idx.device)
            grid_pos = (dense_pos + 1).remainder(W).eq(0)
            grid_pos[-1] = False
        else:
            grid_pos = torch.zeros(T, dtype=torch.bool, device=idx.device)
        next_is_reset = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        next_is_reset[:, :-1] = document_reset[:, 1:]
        hard_fire = grid_pos.view(1, T) & ~next_is_reset
        fire_mask = hard_fire | explicit_fire
        supplied_plans = tuple(attention_layout.chunk_plans)
        supplied_fire_metadata = (
            bool(supplied_plans)
            and all(plan.ends_with_fire is not None for plan in supplied_plans)
        )
        if supplied_fire_metadata:
            boundary_positions = [
                int(plan.end) - 1
                for plan in supplied_plans
                if bool(plan.ends_with_fire)
            ]
        elif torch.compiler.is_compiling():
            boundary_positions = list(range(W - 1, T, W)) if self.tria_hard_fire_enabled else []
        else:
            boundary_positions = (grid_pos | explicit_fire.any(dim=0)).nonzero().flatten().tolist()
        boundary_set = set(boundary_positions)

        h_chunks = []
        key_carries = []
        key_depth = []
        key_valid = []
        key_positions = []
        temporal_state = None
        s = 0
        chunk_ranges: List[Tuple[int, int]] = []
        stops = (
            [int(plan.end) - 1 for plan in supplied_plans]
            if supplied_plans
            else temporal_chunk_stops(
                idx, W, self.tria_hard_fire_enabled, carry_token_id,
                compiling=torch.compiler.is_compiling())
        )
        precomputed_plans = {
            (plan.start, plan.end): plan
            for plan in attention_layout.chunk_plans
        }
        for bp in stops:
            e = min(bp + 1, T)
            if e <= s:
                continue
            chunk_ranges.append((s, e))
            packed_chunk = precomputed_plans.get((s, e))
            if (
                packed_chunk is not None
                and len(packed_chunk.piece_sizes) != len(chunk_ranges)
            ):
                # A plan depends on the complete history partition, not only
                # its current (start,end) key. Reject stale/legacy metadata
                # rather than pairing N selectors with M K/V chunks.
                packed_chunk = None
            if packed_chunk is None:
                packed_chunk = build_packed_chunk_layout(
                    attention_layout, s, e, tuple(chunk_ranges))
            seed_valid = None
            temporal_seed = None
            if temporal_state is not None:
                seed_valid = fire_mask[:, s - 1] & ~document_reset[:, s]
                temporal_seed = temporal_state
            local_reset = document_reset[:, s:e].clone()
            if seed_valid is not None:
                local_reset[:, 0] |= seed_valid
            # The temporal endpoint is consumed only if it (a) seeds the next
            # chunk (there is a next iteration, e != T) or (b) is a fire
            # boundary collected into key_carries (e-1 in boundary_set). The
            # tail chunk of a sequence is neither: grid_pos[-1]=False means it
            # never fires and there is no next chunk. Computing its endpoint
            # created a dangling autograd node whose backward never runs, so
            # its ctx (and the depth-replay tape it pins, holding r/i/o and the
            # carry graph) was never released -- leaking VRAM that grew every
            # step. Structurally verified DEAD (no path to loss), so skipping
            # it changes neither loss nor any gradient.
            endpoint_consumed = (e != T) or ((e - 1) in boundary_set)
            h_chunk, temporal_endpoint, depth_tail, layer_states = self._run_chunk_stack(
                h_emb[:, s:e], position_ids[:, s:e],
                attention_layout, packed_chunk, layer_states,
                temporal_seed, seed_valid, local_reset, endpoint_consumed,
                self.capture_tria_depth_carry)
            if endpoint_consumed:
                temporal_state = temporal_endpoint
            h_chunks.append(h_chunk)
            if e - 1 in boundary_set:
                boundary_valid = fire_mask[:, e - 1]
                corrected_state = tria.polarm(
                    temporal_state, beta=float(self.cfg.tria_polarm_beta))
                temporal_state = torch.where(
                    boundary_valid[:, None, None, None], corrected_state, temporal_state)
                key_carries.append(temporal_state)
                if self.capture_tria_depth_carry:
                    key_depth.append(depth_tail)
                key_valid.append(boundary_valid)
                key_positions.append(e - 1)
            s = e
        if s != T:
            raise RuntimeError(f"chunk boundaries stopped at {s}, expected {T}")

        h_full = torch.cat(h_chunks, dim=1)
        self.last_tria_fire_mask = fire_mask
        self.last_tria_document_carry_stats = None
        if not key_carries:
            self.last_tria_depth_carry = None
            self.last_tria_document_carry = None
            return self._head_or_loss(h_full, labels, ignore_index)

        document_keys = torch.stack(key_carries, dim=1)
        valid_keys = torch.stack(key_valid, dim=1)
        positions = torch.tensor(key_positions, device=idx.device, dtype=torch.long)
        if self.capture_tria_depth_carry:
            self.last_tria_depth_carry = torch.stack(key_depth, dim=1).detach()
            self.last_tria_document_carry = document_keys.detach()
        else:
            self.last_tria_depth_carry = None
            self.last_tria_document_carry = None
        a_keys = self.tria_agg(document_keys)
        h_full = self.tria_final_ca(
            a_keys, h_full, attention_layout,
            carry_key_mask=valid_keys, key_positions=positions)
        return self._head_or_loss(h_full, labels, ignore_index)

    def _forward_flat(self, idx: torch.Tensor, attn_mask: Optional[Any] = None,
                       position_ids: Optional[torch.Tensor] = None,
                       labels: Optional[torch.Tensor] = None, ignore_index: int = -100) -> torch.Tensor:
        B, T = idx.shape
        h = self.emb(idx)
        k0, v0 = self.depth_attn.project(h)
        carry = None 
        p = None      
        hist_k = [k0]
        hist_v = [v0]
        n_blocks = len(self.blocks)
        with tria.depth_replay_scope():
            for bi, block in enumerate(self.blocks):
                h, hist_k, hist_v, carry, p = self._run_block(
                    block, h, hist_k, hist_v, 2 * bi,
                    attn_mask, position_ids, carry, p,
                    is_last_block=bi == n_blocks - 1)
        # `carry` is the final depth-composed Tria for each token/neuron. The
        # temporal path below composes those finished 3x3 matrices over T; it
        # does not recompute r/i/o or layer-local Tria.
        self.last_tria_depth_carry = None
        self.last_tria_document_carry_stats = None
        self.last_tria_fire_mask = None
        if carry is not None:
            depth_carry = carry
            if self.capture_tria_depth_carry:
                self.last_tria_depth_carry = depth_carry.detach()
            document_reset = self._build_tria_document_reset_mask(idx, position_ids)
            document_carry = (
                tria.temporal_carry(depth_carry, document_reset)
                if TRIA_TEMPORAL_ENABLED
                else depth_carry
            )
            W = int(self.cfg.tria_temporal_window)
            # §6.2: hard fire at the last position of each fixed-W chunk, only
            # when the NEXT position is still the same document (a fire whose
            # next position starts a new document is meaningless -- there is
            # nothing to refeed into).
            hard_fire = ((position_ids + 1) % W == 0)
            next_is_new_document = torch.zeros_like(hard_fire)
            if T > 1:
                next_is_new_document[:, :-1] = document_reset[:, 1:]
            carry_key_mask = hard_fire & (~next_is_new_document)
            a = self.tria_agg(document_carry)
            self.last_tria_fire_mask = carry_key_mask
            if torch.compiler.is_compiling():
                self.last_tria_document_carry_stats = None
            else:
                with torch.no_grad():
                    self.last_tria_document_carry_stats = {
                        "max_abs": float(document_carry.detach().abs().amax().item()),
                        "reset_count": int(document_reset.detach().sum().item()),
                        "fire_count": int(carry_key_mask.detach().sum().item()),
                    }
            h = self.tria_final_ca(a, h, attn_mask, carry_key_mask=carry_key_mask)
        return self._head_or_loss(h, labels, ignore_index)  # [B,T,VOCAB] or scalar loss

    @torch.no_grad()
    def step(
        self,
        idx_t: torch.Tensor,
        pos_t: int,
        states,
        kv_runtime: Optional[InferenceKVRuntime] = None,
    ):
        is_bos = bool(pos_t == 0)
        if states is None:
            caches, tria_ca_cache, tria_temporal_state = None, None, None
        else:
            if len(states) == 2:
                caches, tria_ca_cache = states
                tria_temporal_state = None
            else:
                caches, tria_ca_cache, tria_temporal_state = states
        if is_bos:
            caches, tria_ca_cache, tria_temporal_state = None, None, None
        if caches is None:
            caches = [LayerCache() for _ in range(LAYERS)]
        if tria_ca_cache is None:
            tria_ca_cache = tria.TriaCACache()
        if tria_temporal_state is None:
            tria_temporal_state = TriaTemporalState()

        h = self.emb(idx_t).view(idx_t.shape[0], 1, N)
        n_hist = 2 * LAYERS
        k0, v0 = self.depth_attn.project(h)
        hist_k = h.new_zeros(h.shape[0], h.shape[1], n_hist, N_Q_HEADS, HEAD_DIM)
        hist_v = h.new_zeros(h.shape[0], h.shape[1], n_hist, N_Q_HEADS, HEAD_DIM)
        hist_k[:, :, 0] = k0
        hist_v[:, :, 0] = v0
        sub_idx = 0
        new_caches = []
        carry = None  
        p = None      
        n_blocks = len(self.blocks)
        tria_alpha = float(self.cfg.tria_carrier_alpha)
        # spec §12.1: seed for the (single) current token's Tria L0.
        pending = (
            torch.zeros(idx_t.shape[0], dtype=torch.bool, device=idx_t.device)
            if tria_temporal_state.refeed_pending is None
            else tria_temporal_state.refeed_pending.to(device=idx_t.device)
        )
        seed_valid = pending & (not is_bos)
        accT_seed = (
            None
            if tria_temporal_state.carry is None
            else tria_temporal_state.carry
        )
        for bi, (block, cache) in enumerate(zip(self.blocks, caches)):
            is_last_block = bi == n_blocks - 1
            attn_out, q_h, k_ctx_h, c_h, k_all, v_all = block.attn.step(
                h, pos_t, cache.k, cache.v, cache.cache_len, kv_runtime=kv_runtime)
            skip, _ = self.depth_attn(sub_idx, hist_k[:, :, :sub_idx + 1], hist_v[:, :, :sub_idx + 1])
            if RESIDUAL_BRANCH_RMS_CAP is not None:
                skip = capped_rms(skip, RESIDUAL_BRANCH_RMS_CAP)
                attn_out = capped_rms(attn_out, RESIDUAL_BRANCH_RMS_CAP)
            h = block.ln_attn(skip + attn_out)
            k_i, v_i = self.depth_attn.project(h)
            sub_idx += 1
            hist_k[:, :, sub_idx] = k_i
            hist_v[:, :, sub_idx] = v_i

            skip, d_h = self.depth_attn(sub_idx, hist_k[:, :, :sub_idx + 1], hist_v[:, :, :sub_idx + 1])
            want_tria = TRIA_CARRY_ENABLED and not self.ablation
            if want_tria:
                ffn_out, next_phase_trace, (r, i, o) = block.ffn(
                    h, q_h, k_ctx_h, c_h, d_h, phase_trace=cache.phase_trace,
                    return_tria=True, p_in=p)
                if bi == 0 and accT_seed is not None:
                    accT_seed_b = accT_seed.to(device=r.device, dtype=r.dtype)
                    if is_last_block:
                        carry = tria.tria_init_seed(
                            r, i, o, accT_seed_b, seed_valid, axis=bi % 3,
                            alpha=tria_alpha)
                        p = None
                    else:
                        w = torch.softmax(block.ffn.gate_selector.logits, dim=0)
                        carry, p = tria.tria_init_seed_and_gate(
                            r, i, o, accT_seed_b, seed_valid, w, axis=bi % 3,
                            alpha=tria_alpha)
                elif is_last_block:
                    carry = (
                        tria.tria_init(r, i, o, axis=bi % 3, alpha=tria_alpha)
                        if carry is None
                        else tria.tria_step(
                            r, i, o, carry, axis=bi % 3, alpha=tria_alpha)
                    )
                    p = None
                else:
                    w = torch.softmax(block.ffn.gate_selector.logits, dim=0)
                    carry, p = (
                        tria.tria_init_and_gate(
                            r, i, o, w, axis=bi % 3, alpha=tria_alpha)
                        if carry is None
                        else tria.tria_step_and_gate(
                            r, i, o, carry, w, axis=bi % 3, alpha=tria_alpha)
                    )
            else:
                ffn_out, next_phase_trace = block.ffn(
                    h, q_h, k_ctx_h, c_h, d_h, phase_trace=cache.phase_trace)
                carry = None
            if RESIDUAL_BRANCH_RMS_CAP is not None:
                skip = capped_rms(skip, RESIDUAL_BRANCH_RMS_CAP)
                ffn_out = capped_rms(ffn_out, RESIDUAL_BRANCH_RMS_CAP)
            h = block.ln_ffn(skip + ffn_out)
            if not is_last_block:
                k_i, v_i = self.depth_attn.project(h)
                sub_idx += 1
                hist_k[:, :, sub_idx] = k_i
                hist_v[:, :, sub_idx] = v_i

            new_caches.append(LayerCache(k=k_all, v=v_all, phase_trace=next_phase_trace,
                                           cache_len=cache.cache_len + 1))

        self.last_tria_document_carry_stats = None
        if carry is not None:
            depth_carry_t = carry[:, 0]
            # spec §12.2: reset if this is the first ever step, a refeed was
            # pending (already consumed at L0 above), or this token is BOS.
            reset_now = pending | is_bos | (not TRIA_TEMPORAL_ENABLED)
            if tria_temporal_state.carry is None:
                document_carry_t = depth_carry_t
            else:
                prev_doc = tria_temporal_state.carry.to(device=h.device, dtype=depth_carry_t.dtype)
                continued = tria._local_normalize(torch.matmul(depth_carry_t, prev_doc))
                document_carry_t = torch.where(reset_now[:, None, None, None], depth_carry_t, continued)
            # spec §12.3: fire decision for the NEXT token.
            carry_token_id = CARRY_TOKEN_ID
            hard_fire_now = (
                self.tria_hard_fire_enabled
                and (pos_t + 1 < SEQ_LEN)
                and ((pos_t + 1) % int(self.cfg.tria_temporal_window) == 0)
            )
            if carry_token_id is not None:
                explicit_fire_now = idx_t.view(-1).eq(int(carry_token_id))
            else:
                explicit_fire_now = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
            fire_now = explicit_fire_now | hard_fire_now
            if hard_fire_now or (
                carry_token_id is not None and bool(explicit_fire_now.any().item())
            ):
                corrected_state = tria.polarm(
                    document_carry_t, beta=float(self.cfg.tria_polarm_beta))
                document_carry_t = torch.where(
                    fire_now[:, None, None, None], corrected_state, document_carry_t)
            document_carry = document_carry_t.unsqueeze(1)
            a_t = self.tria_agg(document_carry)
            h, tria_ca_cache = self.tria_final_ca.step(
                a_t, h, tria_ca_cache, pos_t, SEQ_LEN, carry_key_mask=fire_now[:, None])
            tria_temporal_state = TriaTemporalState(carry=document_carry_t, refeed_pending=fire_now)
        return self._head_in(h[:, -1, :]), (new_caches, tria_ca_cache, tria_temporal_state)



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
    losses = []
    with torch.no_grad():
        for b in raw_batches:
            x, y, position_ids, attn_mask = split_train_batch(
                b, eos_id, cfg
            )
            ddp_sync_mutable_buffers(model)
            with amp_autocast(device):
                loss = model(x, attn_mask=attn_mask, position_ids=position_ids, labels=y)
            losses.append(float(loss.item()))
            raw = ddp_unwrap_model(model)
            raw.last_tria_depth_carry = None
            raw.last_tria_document_carry_stats = None
    model.train()
    return sum(losses) / len(losses)


def lr_at(cfg: Config, step_zero_based: int) -> float:
    if step_zero_based < cfg.warmup_steps:
        return cfg.lr * (step_zero_based + 1) / max(1, cfg.warmup_steps)
    prog = (step_zero_based - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_frac + (1.0 - cfg.min_lr_frac) * cos)


def load_bytes_per_token(dataset: str) -> Tuple[float, str, bool]:
    meta_path = dataset + ".meta.json"
    if not os.path.exists(meta_path):
        return 1.0, meta_path, False
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    bpt = float(meta.get("bytes_per_token", 1.0))
    if not math.isfinite(bpt) or bpt <= 0.0:
        raise ValueError(f"bad bytes_per_token={bpt!r} in {meta_path}")
    return bpt, meta_path, True


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


def loss_to_bits(loss_nats: float, bytes_per_token: float) -> Tuple[float, float]:
    bits_tok = float(loss_nats) / math.log(2.0)
    bpb = bits_tok / bytes_per_token
    return bits_tok, bpb


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



def format_big_int(n: int) -> str:
    return f"{int(n):,}"


def format_eta_hours_minutes(seconds: float) -> str:
    """Compact non-negative ETA, rounded up to the next whole minute."""
    if not math.isfinite(seconds) or seconds < 0.0:
        return "?h ?min"
    total_minutes = int(math.ceil(seconds / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}min"


def _log_colors_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    forced = str(os.environ.get("FORCE_COLOR", "")).strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _log_color(text: str, color: int, bold: bool = False) -> str:
    if not _log_colors_enabled():
        return text
    weight = "1;" if bold else ""
    return f"\033[{weight}38;5;{int(color)}m{text}\033[0m"


def _log_line(parts: List[str]) -> str:
    separator = _log_color("|", 240)
    return f" {separator} ".join(parts)


def format_train_status(
    step: int,
    train_loss: float,
    refeeds: int,
    lr: float,
    tokens: int,
    data_wait_s: float,
    left: str,
    elapsed_s: float,
) -> str:
    # Orange/purple palette only. `tok` deliberately stays uncolored.
    return _log_line([
        _log_color("[LF]", 208, bold=True) + " " + _log_color(str(step), 141, bold=True),
        _log_color("tr.loss:", 208) + " " + _log_color(f"{train_loss:.4f}", 215, bold=True),
        _log_color("ref:", 135) + " " + _log_color(str(refeeds), 177),
        _log_color("lr:", 173) + " " + _log_color(f"{lr:.2e}", 215),
        f"tok: {format_big_int(tokens)}",
        _log_color("dw:", 135) + " " + _log_color(f"{data_wait_s:.0f}s", 177),
        _log_color("left:", 208) + " " + _log_color(left, 215, bold=True),
        _log_color("lst:", 135) + " " + _log_color(f"{elapsed_s:.0f}s", 177),
    ])


def format_eval_status(
    step: int,
    eval_loss: float,
    bits_tok: float,
    bpb: float,
) -> str:
    return _log_line([
        _log_color("[EVAL]", 135, bold=True) + " " + _log_color(str(step), 141, bold=True),
        _log_color("loss:", 208) + " " + _log_color(f"{eval_loss:.4f}", 215, bold=True),
        _log_color("bit/tok:", 135) + " " + _log_color(f"{bits_tok:.4f}", 183),
        _log_color("bpb:", 173) + " " + _log_color(f"{bpb:.4f}", 215),
    ])



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
    run_epochs = run_tokens / max(1, data_tokens)
    label = "budget" if int(start_step) == 0 else "remaining"
    print(f"  {label:<10}{format_big_int(run_tokens)} tokens over {remaining_steps:,} steps "
          f"({run_epochs:.3f} epochs of {format_big_int(data_tokens)})")
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
    print(f"           paraplex: {format_big_int(params)} params  ·  "
          f"{scale}  ·  {epoch_tokens_per_param:.1f} data-tok/param")


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
        prefix = f"blocks.{layer}.attn."
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


class _GracefulInterrupt:
    def __init__(self) -> None:
        self.requested = False
        self._count = 0
        self._original = None

    def _handler(self, signum, frame) -> None:
        self._count += 1
        if self._count >= 2 and self._original is not None:
            signal.signal(signal.SIGINT, self._original)
            self._original(signum, frame)
            return
        self.requested = True

    def __enter__(self) -> "_GracefulInterrupt":
        self._original = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handler)
        return self

    def __exit__(self, *exc) -> None:
        if self._original is not None:
            signal.signal(signal.SIGINT, self._original)


class _RunpointWatcher:

    def __init__(self) -> None:
        self.requested = threading.Event()
        self._old_termios = None
        self._stop = threading.Event()
        self._active = False
        self._thread: Optional[threading.Thread] = None

    def _reader_loop(self) -> None:
        import select
        while not self._stop.is_set():
            try:
                # Poll with a short timeout instead of a blocking read(1) --
                # a blocking read can only notice _stop on its NEXT keystroke,
                # which is exactly the race that let this thread steal the
                # first character of a y/N answer meant for _handle_interrupt's
                # input(). Polling means pause() actually stops this promptly
                # (~50ms), not "whenever the user happens to type again".
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
            except Exception:
                return
            if not ch:
                return
            if ch.lower() == "s":
                self.requested.set()

    def __enter__(self) -> "_RunpointWatcher":
        # Every local torchrun worker inherits the same controlling terminal.
        # Letting multiple ranks save/modify termios races on restore: a later
        # rank can save rank 0's cbreak/no-echo state as its "original" state
        # and leave the shell broken after an exception (notably CUDA OOM).
        if not ddp_is_main():
            return self
        try:
            import termios
            import tty
        except ImportError:
            return self  # e.g. Windows -- no-op, training is unaffected
        if not sys.stdin.isatty():
            return self  # quiet mode / piped / non-interactive -- no-op by design
        try:
            self._old_termios = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            no_echo = termios.tcgetattr(sys.stdin)
            no_echo[3] &= ~termios.ECHO   # cbreak alone still echoes keystrokes;
                                          # turn that off so pressing 's' doesn't
                                          # leave a stray "s" in the training log
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, no_echo)
        except Exception:
            self._old_termios = None
            return self  # any terminal weirdness at all -- no-op, never crash training
        self._active = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def pause(self) -> None:
        """Stop the key reader and restore the terminal's original settings."""
        if not self._active:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass

    def resume(self) -> None:
        """Restart the key reader in non-echoing cbreak mode after ``pause``."""
        if not self._active:
            return
        try:
            import termios
            import tty
            tty.setcbreak(sys.stdin.fileno())
            no_echo = termios.tcgetattr(sys.stdin)
            no_echo[3] &= ~termios.ECHO
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, no_echo)
        except Exception:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._old_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_termios)
            except Exception:
                pass
        self._active = False

    def consume(self) -> bool:
        if self.requested.is_set():
            self.requested.clear()
            return True
        return False


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
        train_bpt = eval_bpt = 1.0
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
        load_optimizer_checkpoint(opt, resume_in, optimizer_name, device)
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

    if os.path.abspath(eval_dataset) == os.path.abspath(dataset):
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
        save_flag = 0
        if ddp_is_main():
            print(f"\n[interrupt] Ctrl-C at step {step}/{cfg.steps} ({tag}, {time.time() - t0:.0f}s elapsed).",
                  file=_REAL_STDOUT, flush=True)
            runpoint.pause()  # stop the 's'-watcher thread and restore cooked/echo
                               # terminal mode -- without this, input() below
                               # competes with that thread for stdin (which
                               # was in cbreak+no-echo the whole time anyway)
                               # and silently loses keystrokes.
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
            _save_checkpoint()
            if ddp_is_main():
                print("[interrupt] saved.", file=_REAL_STDOUT, flush=True)
        elif ddp_is_main():
            print("[interrupt] not saving -- exiting without a checkpoint.", file=_REAL_STDOUT, flush=True)

        if ddp_is_distributed():
            ddp_barrier(device)
            dist.destroy_process_group()
        raise SystemExit(130)  # 128+SIGINT: conventional shell exit code for Ctrl-C

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
                            f"compiled_max_logit={float(logits.detach().abs().amax().item()):.9g}, "
                            f"eager_max_logit={float(eager_logits.detach().abs().amax().item()):.9g}"
                        )
                    if not math.isfinite(loss_before_backward):
                        raise RuntimeError(
                            "non-finite training loss before backward: "
                            f"loss={loss_before_backward:.9g} "
                            f"max_logit={float(logits.detach().abs().amax().item()):.9g} "
                            f"optimizer={optimizer_name} step={step} micro={micro_idx + 1}/{accum_steps}"
                        )
                    if raw_model_for_tria.last_tria_fire_mask is not None:
                        with torch.no_grad():
                            refeeds_since_log.add_(raw_model_for_tria.last_tria_fire_mask.detach().sum())
                    ddp_trace(
                        "backward_begin", step=step, micro=trace_micro
                    )
                    (total_loss / float(accum_steps)).backward()
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
                    f"summary={bad_grad_summary}; "
                    f"first={bad_grad}; "
                    f"examples={bad_grad_list}"
                )
            if cfg.grad_clip and cfg.grad_clip > 0:
                ddp_trace("grad_clip_begin", step=step)
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                ddp_trace("grad_clip_end", step=step)
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
                    final_eval = ddp_mean_float(final_eval_local, device)
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
        full_eval_loss = ddp_mean_float(float(full["loss_nats"]), device)
        full_eval_bpb = ddp_mean_float(float(full["bpb"]), device)
        ddp_print(
            f"[{tag}] full_eval {val_dataset}  "
            f"eval_loss {full_eval_loss:.4f}  bits/tok {full['bits_tok']:.4f}  "
            f"bpb {full_eval_bpb:.4f}  tokens {int(full['total_tokens'])}"
        )
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
                    ids, mask = stream._pack_batch_np(rows)
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
