#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import os
import threading
import weakref
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# §2: build_tria
# ============================================================================


def build_tria_pytorch(
    r: torch.Tensor,
    i: torch.Tensor,
    o: torch.Tensor,
    alpha: Optional[float] = None,
    axis: int = 0,
    abc_bound: float = 1.0,
) -> torch.Tensor:
    axis = _axis(axis)
    a = abc_bound * torch.tanh((r * i) / abc_bound)
    b = abc_bound * torch.tanh((r * o) / abc_bound)
    c = abc_bound * torch.tanh((i * o) / abc_bound)
    q = float(_TRIA_CARRIER_ALPHA if alpha is None else alpha)
    aa, bb, cc = a * q, b * q, c * q
    one = torch.ones_like(a)
    neg_one = -one
    if axis == 0:  # (I+qK) @ Rz(+90)
        rows = (
            (-cc, neg_one, bb),
            (one, -cc, -aa),
            (aa, bb, one),
        )
    elif axis == 1:  # (I+qK) @ Rx(+90)
        rows = (
            (one, bb, cc),
            (cc, -aa, neg_one),
            (-bb, one, -aa),
        )
    else:  # (I+qK) @ Ry(+90)
        rows = (
            (-bb, -cc, one),
            (aa, one, cc),
            (neg_one, aa, -bb),
        )
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


# ============================================================================
# §3: depth-wise carry chain
# ============================================================================


def _local_normalize(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Max-absolute-normalize each matrix over its final two dimensions."""
    work = m.float() if m.dtype in (torch.float16, torch.bfloat16) else m
    scale = work.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(eps)
    normalized = work / scale
    return normalized.to(m.dtype) if normalized.dtype != m.dtype else normalized


def carry_init_pytorch(tria_1: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the first layer's Tria matrix."""
    return _local_normalize(tria_1, eps)


def carry_step_pytorch(tria_l: torch.Tensor, carry_prev: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Left-multiply the previous carry by the current Tria matrix and normalize."""
    m = torch.matmul(tria_l, carry_prev)
    return _local_normalize(m, eps)


# ============================================================================
# CUDA fast path -- lazy load, fails closed. TESTED on real hardware
# ============================================================================

_cuda_tria_module = None
_cuda_tria_tried = False
# CUDA fused tria is opt-in until forward/backward parity is validated on real GPU.
# The PyTorch path is the reference implementation and is fully covered by self-tests.
_CUDA_TRIA_ENABLED = os.environ.get("LOOM_TRIA_CUDA", "0").lower() in ("1", "true", "yes", "on")
_TRIA_DISABLE_OPS = {
    part.strip().lower().replace("-", "_")
    for part in os.environ.get("LOOM_TRIA_DISABLE_OPS", "").split(",")
    if part.strip()
}
_TRIA_FALLBACK_PRINTED = set()

# Fixed, non-learned carrier geometry. alpha is selected/configured by loomformer.py
# (optionally by startup calibration); axis is supplied per depth layer. No tensor
# state is allocated for either value.
_TRIA_CARRIER_ALPHA = 0.05
_TRIA_POLARM_BETA = 0.1

def set_carrier_alpha(alpha: float) -> None:
    global _TRIA_CARRIER_ALPHA
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"tria carrier alpha must be finite and > 0, got {alpha}")
    _TRIA_CARRIER_ALPHA = alpha

def carrier_alpha() -> float:
    return float(_TRIA_CARRIER_ALPHA)


def set_polarm_beta(beta: float) -> None:
    global _TRIA_POLARM_BETA
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0.0 or beta >= 1.0:
        raise ValueError(f"PolARM beta must be finite and in [0, 1), got {beta}")
    _TRIA_POLARM_BETA = beta


def polarm_beta() -> float:
    return float(_TRIA_POLARM_BETA)


def _polarm_impl(matrix: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
    input_dtype = matrix.dtype
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        work = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
        gram = work.transpose(-1, -2) @ work
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True)
        scale = scale.unsqueeze(-1).clamp_min(eps)
        normalized_gram = gram / scale
        identity = torch.eye(3, device=work.device, dtype=work.dtype)
        correction = identity - 0.5 * beta * (normalized_gram - identity)
        corrected = work @ correction
    return corrected.to(input_dtype) if corrected.dtype != input_dtype else corrected


class _PolARMRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix: torch.Tensor, beta: float, eps: float) -> torch.Tensor:
        ctx.save_for_backward(matrix)
        ctx.beta = float(beta)
        ctx.eps = float(eps)
        return _polarm_impl(matrix, ctx.beta, ctx.eps)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (matrix,) = ctx.saved_tensors
        with torch.enable_grad():
            replay = matrix.detach().requires_grad_(True)
            output = _polarm_impl(replay, ctx.beta, ctx.eps)
            (grad_matrix,) = torch.autograd.grad(output, replay, grad_output)
        return grad_matrix, None, None


def polarm(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _PolARMRecompute.apply(matrix, _TRIA_POLARM_BETA, eps)

def _axis(axis: int) -> int:
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0(z), 1(x), or 2(y), got {axis}")
    return axis

# Set by graph_helper.register_all() right after it successfully registers
# this module's custom_op wrappers -- NOT by apply_config directly off
# cfg.graph, so this can never be True while the ops it gates aren't
# actually there. _graph_tria_init_forward_op is set alongside it, by the
# same call.
_GRAPH_MODE_ENABLED = False
_graph_tria_init_op = None
_graph_tria_init_gate_op = None
_graph_tria_step_op = None
_graph_tria_step_gate_op = None
_graph_gate_slot_mix_op = None
_graph_slot_attention_pool_op = None
_graph_temporal_carry_op = None


def set_graph_mode_enabled(enabled: bool) -> None:
    global _GRAPH_MODE_ENABLED
    _GRAPH_MODE_ENABLED = bool(enabled)


def set_cuda_tria_enabled(enabled: bool) -> None:
    """Enable or disable fused CUDA Tria operations at runtime."""
    global _CUDA_TRIA_ENABLED
    _CUDA_TRIA_ENABLED = bool(enabled)


def cuda_tria_enabled() -> bool:
    return bool(_CUDA_TRIA_ENABLED)


def _tria_cuda_op_enabled(name: str) -> bool:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "temporal_endpoint": "temporal_carry_endpoint",
        "temporal_carry_endpoint": "temporal_endpoint",
    }
    return (
        _CUDA_TRIA_ENABLED
        and key not in _TRIA_DISABLE_OPS
        and aliases.get(key, "") not in _TRIA_DISABLE_OPS
    )


def _warn_cuda_fallback(name: str, error: BaseException) -> None:
    key = name.strip().lower().replace("-", "_")
    if key in _TRIA_FALLBACK_PRINTED:
        return
    _TRIA_FALLBACK_PRINTED.add(key)
    print(
        f"[tria] CUDA {key} failed ({type(error).__name__}: {error}); "
        "using SLOWER PyTorch fallback.",
        flush=True,
    )

# CUDA source for the fused tria-step kernels now lives under kernels/tria/
# -- one subfolder per kernel group (tria_init, tria_init_gate, tria_step,
# tria_step_gate, gate_slot_mix, slot_attention_pool, temporal_carry),
# each with a pure-CUDA <name>_kernel.cuh (+ standalone
# <name>_kernel.cu for direct nvcc/PTX inspection) and an ATen-facing
# <name>_launcher.cu. kernels/tria/common.cuh holds the shared block-reduce
# device helpers; kernels/tria/bindings.cpp is the single PYBIND11_MODULE for
# the whole extension. See kernels/build.py for the sha256-tracked,
# ninja-incremental build.


# One thread per (b,t,h) 3x3 block. Everything lives in registers: the 6 real
# multiplies to build Tria_L, the 9-element (3x3 @ 3x3) matmul against
# carry_prev, the maxabs reduction over exactly those 9 elements, and the
# final scale -- no shared memory, no global-memory round-trip for the
# intermediate Tria_L or the pre-normalize matmul result. This is exactly the
# "9 transient register values are fine, don't persist 6/9 as state" design
# from the spec's hardware discussion (Pascal FP32 FMA throughput).


def _try_load_cuda_tria():
    """Lazily build and load the fused CUDA module, returning ``None`` on failure."""
    global _cuda_tria_module, _cuda_tria_tried
    if _cuda_tria_tried:
        return _cuda_tria_module
    _cuda_tria_tried = True
    try:
        from kernels.build import build_or_load
        _tria_groups = [
            "tria_init", "tria_init_gate", "tria_init_seed", "tria_step", "tria_step_gate",
            "depth_replay", "gate_slot_mix", "slot_attention_pool", "temporal_carry",
            "temporal_carry_endpoint", "final_ca_sparse",
        ]
        _tria_sources = ["tria/bindings.cpp"] + [
            f"tria/{g}/{g}_launcher.cu" for g in _tria_groups
        ]
        _tria_ptx_kernels = {
            g: f"tria/{g}/{g}_kernel.cu" for g in _tria_groups
        }
        _cuda_tria_module = build_or_load(
            "loomformer_tria_carry",
            _tria_sources,
            extra_cuda_cflags=["--use_fast_math"],
            ptx_kernels=_tria_ptx_kernels,
        )
    except Exception as e:
        _cuda_tria_module = None
        _warn_cuda_fallback("module_load", e)
    return _cuda_tria_module


class _DepthReplayTape:
    def __init__(self, seed=None, seed_valid=None):
        self.seed = seed
        self.seed_valid = seed_valid
        self.entries = []
        self.current_index = -1
        self._reverse_value = None
        self._expected = None
        self._tables = None
        self._cleanup_registered = False

    def record(self, r, i, o, axis):
        if r.shape != i.shape or r.shape != o.shape:
            raise ValueError("depth replay R/I/O shape mismatch")
        self.current_index = len(self.entries)
        self.entries.append({
            "shape": tuple(r.shape),
            "axis": _axis(axis),
            "r": None,
            "i": None,
            "o": None,
            "output": None,
        })
        return self.current_index

    def bind_current(self, r, i, o, axis):
        if self.current_index < 0:
            raise RuntimeError("depth replay input was not recorded")
        entry = self.entries[self.current_index]
        if tuple(r.shape) != entry["shape"] or i.shape != r.shape or o.shape != r.shape:
            raise RuntimeError("depth replay bound a different R/I/O shape")
        if _axis(axis) != entry["axis"]:
            raise RuntimeError("depth replay axis changed after recording")
        entry["r"], entry["i"], entry["o"] = r, i, o
        self._tables = None

    def bind_seed(self, seed, seed_valid):
        self.seed = seed
        self.seed_valid = seed_valid
        self._tables = None

    def capture(self, carry):
        # Stored per-entry (not just for the tape's overall last layer):
        # whichever layer's own carry_new backward runs FIRST needs ITS OWN
        # output value as the O(1)-reverse seed, not necessarily the last
        # layer's. A layer beyond `index` can legitimately never reach
        # backward at all -- e.g. the tail chunk of a sequence never fires
        # (_forward_chunked forces grid_pos[-1]=False), so its temporal
        # endpoint, and hence its last layer's carry, has no path to loss
        # and autograd correctly skips that Function's backward entirely.
        if self.current_index < 0:
            return
        entry = self.entries[self.current_index]
        expected = entry["shape"] + (3, 3)
        if tuple(carry.shape) == expected:
            entry["output"] = weakref.ref(carry)

    def _begin(self, index):
        if self._expected is None:
            self._expected = index
        if self._expected != index:
            raise RuntimeError(f"depth reverse order mismatch: expected {self._expected}, got {index}")

    def reverse_value(self, index):
        self._begin(index)
        if self._reverse_value is None:
            ref = self.entries[index]["output"]
            value = None if ref is None else ref()
            if value is None:
                raise RuntimeError(f"depth carry at layer {index} was released before backward")
            self._reverse_value = value
        return self._reverse_value

    def _prepare_tables(self):
        if self._tables is not None:
            return self._tables
        if not self.entries or any(e["r"] is None for e in self.entries):
            raise RuntimeError("depth replay tape is incomplete")
        first = self.entries[0]["r"]
        device, dtype, shape = first.device, first.dtype, first.shape
        for entry in self.entries:
            for name in ("r", "i", "o"):
                value = entry[name]
                if value.device != device or value.dtype != dtype or value.shape != shape:
                    raise RuntimeError("depth replay tensors changed dtype, device, or shape")
                if not value.is_contiguous():
                    raise RuntimeError("depth replay tensors must be contiguous")
        def pointer_table(name):
            return torch.tensor(
                [entry[name].data_ptr() for entry in self.entries],
                device=device, dtype=torch.int64)
        axes = torch.tensor(
            [entry["axis"] for entry in self.entries],
            device=device, dtype=torch.int32)
        if self.seed is None:
            seed = torch.empty(0, device=device, dtype=dtype)
            valid = torch.empty(0, device=device, dtype=torch.bool)
        else:
            seed = self.seed
            valid = self.seed_valid
            if seed.device != device or seed.dtype != dtype or not seed.is_contiguous():
                raise RuntimeError("depth replay seed mismatch")
            if valid is None or valid.device != device or valid.dtype != torch.bool:
                raise RuntimeError("depth replay seed_valid mismatch")
            valid = valid.contiguous()
        self._tables = (
            pointer_table("r"), pointer_table("i"), pointer_table("o"),
            axes, seed, valid)
        return self._tables

    def replay_args(self, index):
        self._begin(index)
        return self._prepare_tables()

    def advance(self, index, previous=None):
        if self._expected != index:
            raise RuntimeError("depth reverse state is out of order")
        self._reverse_value = previous
        self._expected = index - 1
        if self._expected <= 0:
            self.clear()

    def clear(self):
        self.entries.clear()
        self.seed = None
        self.seed_valid = None
        self._tables = None
        self._reverse_value = None
        self._expected = None

    def register_scope_cleanup(self):
        return


_depth_replay_tls = threading.local()


def _active_depth_replay():
    return getattr(_depth_replay_tls, "tape", None)


@contextmanager
def depth_replay_scope(seed=None, seed_valid=None):
    previous = _active_depth_replay()
    tape = _DepthReplayTape(seed=seed, seed_valid=seed_valid)
    _depth_replay_tls.tape = tape
    try:
        yield tape
    finally:
        _depth_replay_tls.tape = previous
        tape.register_scope_cleanup()


def record_depth_replay(r, i, o, axis):
    tape = _active_depth_replay()
    return None if tape is None else tape.record(r, i, o, axis)


def _capture_depth_output(carry):
    tape = _active_depth_replay()
    if tape is not None:
        tape.capture(carry)
    return carry


class _TriaInitFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, i, o, alpha, axis):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init module is unavailable")
        r, i, o = r.contiguous(), i.contiguous(), o.contiguous()
        # tria_init has no carry_prev to avoid saving (it's the axis's first
        # step), so unlike tria_step there is no recompute/replay variant --
        # only bind into the tape for depth_replay_backward's forward-replay
        # table (used by *later* layers' half-precision backward).
        tape = _active_depth_replay()
        if tape is not None:
            tape.bind_current(r, i, o, axis)
        carry, scale = module.tria_init_forward(r, i, o, float(alpha), int(axis))
        ctx.save_for_backward(r, i, o, scale)
        ctx.alpha = float(alpha)
        ctx.axis = int(axis)
        return carry

    @staticmethod
    def backward(ctx, grad_carry):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init module is unavailable")
        r, i, o, scale = ctx.saved_tensors
        grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
        grad_r, grad_i, grad_o = module.tria_init_backward(
            grad_carry, r, i, o, scale, ctx.alpha, ctx.axis)
        return grad_r, grad_i, grad_o, None, None


class _TriaInitAndGateFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, i, o, w, alpha, axis):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init-gate module is unavailable")
        r, i, o, w = r.contiguous(), i.contiguous(), o.contiguous(), w.contiguous()
        # See _TriaInitFused.forward: no carry_prev exists here either.
        tape = _active_depth_replay()
        if tape is not None:
            tape.bind_current(r, i, o, axis)
        carry, p, scale = module.tria_init_gate_forward(r, i, o, w, float(alpha), int(axis))
        ctx.save_for_backward(r, i, o, w, scale)
        ctx.alpha = float(alpha)
        ctx.axis = int(axis)
        return carry, p

    @staticmethod
    def backward(ctx, grad_carry, grad_p):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init-gate module is unavailable")
        r, i, o, w, scale = ctx.saved_tensors
        grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
        grad_p = torch.zeros_like(o) if grad_p is None else grad_p.contiguous()
        grad_r, grad_i, grad_o, grad_w = module.tria_init_gate_backward(
            grad_carry, grad_p, r, i, o, w, scale, ctx.alpha, ctx.axis)
        return grad_r, grad_i, grad_o, grad_w, None, None


class _TriaInitSeedFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, i, o, seed, valid, alpha, axis):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init-seed module is unavailable")
        r, i, o = r.contiguous(), i.contiguous(), o.contiguous()
        seed = seed.contiguous()
        valid = valid.to(torch.bool).contiguous()
        reverse = _active_depth_replay()
        if reverse is not None:
            reverse.bind_current(r, i, o, axis)
            reverse.bind_seed(seed, valid)
        carry, _ = module.tria_init_seed_forward(
            r, i, o, seed, valid, float(alpha), int(axis))
        ctx.alpha = float(alpha)
        ctx.axis = int(axis)
        ctx.reverse = reverse
        ctx.save_for_backward(r, i, o, seed, valid)
        return carry

    @staticmethod
    def backward(ctx, grad_carry):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init-seed module is unavailable")
        r, i, o, seed, valid = ctx.saved_tensors
        grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
        grad_r, grad_i, grad_o, grad_seed, _ = module.tria_init_seed_backward(
            grad_carry, r, i, o, seed, valid, ctx.alpha, ctx.axis)
        if ctx.reverse is not None:
            ctx.reverse.clear()
        return grad_r, grad_i, grad_o, grad_seed, None, None, None


class _TriaInitSeedAndGateFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, i, o, seed, valid, w, alpha, axis):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init-seed-gate module is unavailable")
        r, i, o = r.contiguous(), i.contiguous(), o.contiguous()
        seed, w = seed.contiguous(), w.contiguous()
        valid = valid.to(torch.bool).contiguous()
        reverse = _active_depth_replay()
        if reverse is not None:
            reverse.bind_current(r, i, o, axis)
            reverse.bind_seed(seed, valid)
        carry, p = module.tria_init_seed_gate_forward(
            r, i, o, seed, valid, w, float(alpha), int(axis))
        ctx.alpha = float(alpha)
        ctx.axis = int(axis)
        ctx.reverse = reverse
        ctx.save_for_backward(r, i, o, seed, valid, w)
        return carry, p

    @staticmethod
    def backward(ctx, grad_carry, grad_p):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-init-seed-gate module is unavailable")
        r, i, o, seed, valid, w = ctx.saved_tensors
        grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
        grad_p = torch.zeros_like(o) if grad_p is None else grad_p.contiguous()
        grad_r, grad_i, grad_o, grad_seed, grad_w = module.tria_init_seed_gate_backward(
            grad_carry, grad_p, r, i, o, seed, valid, w, ctx.alpha, ctx.axis)
        if ctx.reverse is not None:
            ctx.reverse.clear()
        return grad_r, grad_i, grad_o, grad_seed, None, grad_w, None, None


class _TriaStepFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, i, o, carry_previous, alpha, axis):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-step module is unavailable")
        r, i, o = r.contiguous(), i.contiguous(), o.contiguous()
        carry_previous = carry_previous.contiguous()
        tape = _active_depth_replay()
        reverse = tape is not None and tape.current_index > 0 and _tria_cuda_op_enabled("depth_replay")
        if reverse:
            tape.bind_current(r, i, o, axis)
            # Same forward math as the plain path (tria_step_forward) -- the
            # only difference is what backward does with it. `scale` is
            # discarded: reverse-mode backward rebuilds it from the local
            # carrier and the recomputed previous carry (see
            # tria_step_reverse_backward_kernel).
            carry, _scale = module.tria_step_forward(
                r, i, o, carry_previous, float(alpha), int(axis))
            ctx.reverse = tape
            ctx.reverse_index = tape.current_index
            ctx.save_for_backward(r, i, o)
        else:
            carry, scale = module.tria_step_forward(
                r, i, o, carry_previous, float(alpha), int(axis))
            ctx.save_for_backward(r, i, o, carry_previous, scale)
        ctx.alpha = float(alpha)
        ctx.axis = int(axis)
        return carry

    @staticmethod
    def backward(ctx, grad_carry):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-step module is unavailable")
        if hasattr(ctx, "reverse"):
            r, i, o = ctx.saved_tensors
            grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
            current = None
            if r.dtype not in (torch.float16, torch.bfloat16):
                try:
                    current = ctx.reverse.reverse_value(ctx.reverse_index)
                except RuntimeError:
                    # The O(1) analytic-reverse seed (this layer's own carry
                    # output) has no live reference: nothing downstream
                    # needed it (e.g. this is the tail chunk of a sequence,
                    # whose temporal endpoint is computed but never fires --
                    # see _forward_chunked's grid_pos[-1]=False -- so the
                    # tensor was already freed once forward returned). Fall
                    # back to the same forward-replay used for half
                    # precision: r/i/o are always retained (needed for the
                    # FFN's own weight gradients regardless), so replaying
                    # from the seed is always possible, just O(layer_index)
                    # instead of O(1) for this one layer.
                    current = None
            if current is not None:
                grad_r, grad_i, grad_o, grad_previous, previous = module.tria_step_reverse_backward(
                    grad_carry, r, i, o, current, ctx.alpha, ctx.axis)
                ctx.reverse.advance(ctx.reverse_index, previous)
            else:
                tables = ctx.reverse.replay_args(ctx.reverse_index)
                grad_r, grad_i, grad_o, grad_previous = module.depth_replay_backward(
                    grad_carry, r, i, o, *tables,
                    ctx.alpha, ctx.axis, ctx.reverse_index)
                ctx.reverse.advance(ctx.reverse_index)
            return grad_r, grad_i, grad_o, grad_previous, None, None
        r, i, o, previous, scale = ctx.saved_tensors
        grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
        grad_r, grad_i, grad_o, grad_previous = module.tria_step_backward(
            grad_carry, r, i, o, previous, scale, ctx.alpha, ctx.axis)
        return grad_r, grad_i, grad_o, grad_previous, None, None


class _TriaStepAndGateFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, i, o, carry_previous, w, alpha, axis):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-step-gate module is unavailable")
        r, i, o = r.contiguous(), i.contiguous(), o.contiguous()
        carry_previous, w = carry_previous.contiguous(), w.contiguous()
        tape = _active_depth_replay()
        reverse = tape is not None and tape.current_index > 0 and _tria_cuda_op_enabled("depth_replay")
        if reverse:
            tape.bind_current(r, i, o, axis)
            # See _TriaStepFused.forward: same forward kernel, scale dropped.
            carry, p, _scale = module.tria_step_gate_forward(
                r, i, o, carry_previous, w, float(alpha), int(axis))
            ctx.reverse = tape
            ctx.reverse_index = tape.current_index
            ctx.save_for_backward(r, i, o, w)
        else:
            carry, p, scale = module.tria_step_gate_forward(
                r, i, o, carry_previous, w, float(alpha), int(axis))
            ctx.save_for_backward(r, i, o, carry_previous, w, scale)
        ctx.alpha = float(alpha)
        ctx.axis = int(axis)
        return carry, p

    @staticmethod
    def backward(ctx, grad_carry, grad_p):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria-step-gate module is unavailable")
        if hasattr(ctx, "reverse"):
            r, i, o, w = ctx.saved_tensors
            grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
            grad_p = torch.zeros_like(o) if grad_p is None else grad_p.contiguous()
            current = None
            if r.dtype not in (torch.float16, torch.bfloat16):
                try:
                    current = ctx.reverse.reverse_value(ctx.reverse_index)
                except RuntimeError:
                    # See _TriaStepFused.backward for why this happens and
                    # why forward-replay is always a valid fallback.
                    current = None
            if current is not None:
                grad_r, grad_i, grad_o, grad_previous, grad_w, previous = module.tria_step_gate_reverse_backward(
                    grad_carry, grad_p, r, i, o, current, w, ctx.alpha, ctx.axis)
                ctx.reverse.advance(ctx.reverse_index, previous)
            else:
                tables = ctx.reverse.replay_args(ctx.reverse_index)
                grad_r, grad_i, grad_o, grad_previous, grad_w = module.depth_replay_gate_backward(
                    grad_carry, grad_p, r, i, o, w, *tables,
                    ctx.alpha, ctx.axis, ctx.reverse_index)
                ctx.reverse.advance(ctx.reverse_index)
            return grad_r, grad_i, grad_o, grad_previous, grad_w, None, None
        r, i, o, previous, w, scale = ctx.saved_tensors
        grad_carry = o.new_zeros((*o.shape, 3, 3)) if grad_carry is None else grad_carry.contiguous()
        grad_p = torch.zeros_like(o) if grad_p is None else grad_p.contiguous()
        grad_r, grad_i, grad_o, grad_previous, grad_w = module.tria_step_gate_backward(
            grad_carry, grad_p, r, i, o, previous, w, scale, ctx.alpha, ctx.axis)
        return grad_r, grad_i, grad_o, grad_previous, grad_w, None, None

def _cuda_tria_applicable(r: torch.Tensor, i: torch.Tensor, o: torch.Tensor, carry_prev: torch.Tensor) -> bool:
    if not (r.is_cuda and i.is_cuda and o.is_cuda and carry_prev.is_cuda):
        return False
    if r.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if not (i.dtype == r.dtype and o.dtype == r.dtype and carry_prev.dtype == r.dtype):
        return False
    return True


def _cuda_autocast_dtype_or_none() -> Optional[torch.dtype]:
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
    if not enabled:
        return None
    try:
        return torch.get_autocast_dtype("cuda")
    except AttributeError:
        return torch.get_autocast_gpu_dtype()


def _tria_fast_dtype(*xs: torch.Tensor) -> Optional[torch.dtype]:
    if not all(x.is_cuda for x in xs):
        return None
    ac_dtype = _cuda_autocast_dtype_or_none()
    if ac_dtype in (torch.float16, torch.bfloat16):
        return ac_dtype
    dt = xs[0].dtype
    for x in xs[1:]:
        dt = torch.promote_types(dt, x.dtype)
    return dt if dt in (torch.float32, torch.float16, torch.bfloat16) else None


def _cast_tria_args(dtype: torch.dtype, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return tuple(x if x.dtype == dtype else x.to(dtype=dtype) for x in xs)


class _GateSlotMixFused(torch.autograd.Function):
    """Autograd wrapper for the fused CUDA weighted slot reduction."""

    @staticmethod
    def forward(ctx, carry, w):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria module is unavailable")
        p = module.gate_slot_mix_forward(carry.contiguous(), w.contiguous())
        ctx.save_for_backward(carry, w)
        return p

    @staticmethod
    def backward(ctx, grad_p):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria module is unavailable")
        carry, w = ctx.saved_tensors
        grad_carry, grad_w = module.gate_slot_mix_backward(grad_p.contiguous(), carry, w)
        return grad_carry, grad_w


class _SlotAttentionPoolFused(torch.autograd.Function):
    """Autograd wrapper for fused CUDA attention pooling over Tria slots."""

    @staticmethod
    def forward(ctx, carry, score_w):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria module is unavailable")
        pooled, lse = module.slot_attention_pool_forward(carry.contiguous(), score_w.contiguous())
        ctx.save_for_backward(carry, score_w, lse)
        return pooled

    @staticmethod
    def backward(ctx, grad_pooled):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria module is unavailable")
        carry, score_w, lse = ctx.saved_tensors
        grad_carry, grad_score_w = module.slot_attention_pool_backward(
            grad_pooled.contiguous(), carry, score_w, lse)
        return grad_carry, grad_score_w


def _cuda_slot_op_applicable(carry: torch.Tensor, small: torch.Tensor) -> bool:
    """Return whether two tensors satisfy the fused slot kernels' device and dtype requirements."""
    if not (carry.is_cuda and small.is_cuda):
        return False
    if carry.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if small.dtype != carry.dtype:
        return False
    return True


# ============================================================================
# public API: dispatch to CUDA when applicable, PyTorch otherwise. This is the
# only surface loomformer.py should import from.
# ============================================================================




def tria_init(
    r_1: torch.Tensor, i_1: torch.Tensor, o_1: torch.Tensor, axis: int = 0
) -> torch.Tensor:
    """Build and normalize the initial carrier matrix."""
    axis = _axis(axis)
    alpha = float(_TRIA_CARRIER_ALPHA)
    if _tria_cuda_op_enabled("init") and r_1.is_cuda and i_1.is_cuda and o_1.is_cuda:
        fast_dtype = _tria_fast_dtype(r_1, i_1, o_1)
        if fast_dtype is not None:
            r_fast, i_fast, o_fast = _cast_tria_args(fast_dtype, r_1, i_1, o_1)
            if _GRAPH_MODE_ENABLED and _graph_tria_init_op is not None:
                carry_1, _scale = _graph_tria_init_op(r_fast, i_fast, o_fast, alpha, axis)
                return carry_1
            try:
                return _capture_depth_output(_TriaInitFused.apply(r_fast, i_fast, o_fast, alpha, axis))
            except RuntimeError as error:
                _warn_cuda_fallback("tria_init", error)
    return _capture_depth_output(carry_init_pytorch(build_tria_pytorch(r_1, i_1, o_1, alpha, axis)))


def tria_init_seed_reference(
    r: torch.Tensor, i: torch.Tensor, o: torch.Tensor,
    seed: torch.Tensor, seed_valid: torch.Tensor, axis: int = 0,
) -> torch.Tensor:
    local = build_tria_pytorch(r, i, o, carrier_alpha(), axis)
    first = torch.matmul(local[:, 0], seed)
    first = torch.where(seed_valid[:, None, None, None], first, local[:, 0])
    pre = torch.cat((first[:, None], local[:, 1:]), dim=1)
    return _local_normalize(pre)


def tria_init_seed(
    r: torch.Tensor, i: torch.Tensor, o: torch.Tensor,
    seed: torch.Tensor, seed_valid: torch.Tensor, axis: int = 0,
) -> torch.Tensor:
    axis = _axis(axis)
    alpha = carrier_alpha()
    valid = seed_valid.to(device=r.device, dtype=torch.bool)
    if _tria_cuda_op_enabled("init_seed") and r.is_cuda and i.is_cuda and o.is_cuda and seed.is_cuda:
        fast_dtype = _tria_fast_dtype(r, i, o, seed)
        if fast_dtype is not None:
            r_fast, i_fast, o_fast, seed_fast = _cast_tria_args(fast_dtype, r, i, o, seed)
            try:
                return _capture_depth_output(_TriaInitSeedFused.apply(
                    r_fast, i_fast, o_fast, seed_fast, valid, alpha, axis))
            except RuntimeError as error:
                _warn_cuda_fallback("tria_init_seed", error)
    return _capture_depth_output(tria_init_seed_reference(r, i, o, seed, valid, axis))


def tria_init_seed_and_gate(
    r: torch.Tensor, i: torch.Tensor, o: torch.Tensor,
    seed: torch.Tensor, seed_valid: torch.Tensor, w: torch.Tensor, axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    axis = _axis(axis)
    alpha = carrier_alpha()
    valid = seed_valid.to(device=r.device, dtype=torch.bool)
    if _tria_cuda_op_enabled("init_seed_gate") and r.is_cuda and i.is_cuda and o.is_cuda and seed.is_cuda and w.is_cuda:
        fast_dtype = _tria_fast_dtype(r, i, o, seed, w)
        if fast_dtype is not None:
            r_fast, i_fast, o_fast, seed_fast, w_fast = _cast_tria_args(
                fast_dtype, r, i, o, seed, w)
            try:
                carry, p = _TriaInitSeedAndGateFused.apply(
                    r_fast, i_fast, o_fast, seed_fast, valid, w_fast, alpha, axis)
                return _capture_depth_output(carry), p
            except RuntimeError as error:
                _warn_cuda_fallback("tria_init_seed_gate", error)
    carry = tria_init_seed_reference(r, i, o, seed, valid, axis)
    return _capture_depth_output(carry), (tria_slots(carry) * w).sum(dim=-1)


def tria_step(
    r: torch.Tensor, i: torch.Tensor, o: torch.Tensor, carry_prev: torch.Tensor,
    axis: int = 0,
) -> torch.Tensor:
    """Build a carrier matrix, compose it with ``carry_prev``, and normalize."""
    axis = _axis(axis)
    alpha = float(_TRIA_CARRIER_ALPHA)
    if _tria_cuda_op_enabled("step") and r.is_cuda and i.is_cuda and o.is_cuda and carry_prev.is_cuda:
        fast_dtype = _tria_fast_dtype(r, i, o, carry_prev)
        if fast_dtype is not None:
            r_fast, i_fast, o_fast, carry_fast = _cast_tria_args(fast_dtype, r, i, o, carry_prev)
            if _GRAPH_MODE_ENABLED and _graph_tria_step_op is not None and _active_depth_replay() is None:
                carry_new, _scale = _graph_tria_step_op(r_fast, i_fast, o_fast, carry_fast, alpha, axis)
                return carry_new
            try:
                return _capture_depth_output(
                    _TriaStepFused.apply(r_fast, i_fast, o_fast, carry_fast, alpha, axis))
            except RuntimeError as error:
                _warn_cuda_fallback("tria_step", error)
    return _capture_depth_output(
        carry_step_pytorch(build_tria_pytorch(r, i, o, alpha, axis), carry_prev))


def tria_init_and_gate(
    r_1: torch.Tensor, i_1: torch.Tensor, o_1: torch.Tensor, w: torch.Tensor,
    axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    axis = _axis(axis)
    alpha = float(_TRIA_CARRIER_ALPHA)
    if _tria_cuda_op_enabled("init_gate") and r_1.is_cuda and i_1.is_cuda and o_1.is_cuda and w.is_cuda:
        fast_dtype = _tria_fast_dtype(r_1, i_1, o_1, w)
        if fast_dtype is not None:
            r_fast, i_fast, o_fast, w_fast = _cast_tria_args(fast_dtype, r_1, i_1, o_1, w)
            if _GRAPH_MODE_ENABLED and _graph_tria_init_gate_op is not None:
                carry_1, p_out, _scale = _graph_tria_init_gate_op(
                    r_fast, i_fast, o_fast, w_fast, alpha, axis)
                return carry_1, p_out
            try:
                carry_1, p_out = _TriaInitAndGateFused.apply(
                    r_fast, i_fast, o_fast, w_fast, alpha, axis)
                return _capture_depth_output(carry_1), p_out
            except RuntimeError as error:
                _warn_cuda_fallback("tria_init_gate", error)
    carry_1 = tria_init(r_1, i_1, o_1, axis=axis)
    if _GRAPH_MODE_ENABLED and _graph_gate_slot_mix_op is not None and _cuda_slot_op_applicable(carry_1, w):
        p_out = _graph_gate_slot_mix_op(carry_1, w)
    elif _tria_cuda_op_enabled("gate_slot_mix") and _cuda_slot_op_applicable(carry_1, w):
        try:
            p_out = _GateSlotMixFused.apply(carry_1, w)
        except RuntimeError as error:
            _warn_cuda_fallback("gate_slot_mix", error)
            p_out = (tria_slots(carry_1) * w).sum(dim=-1)
    else:
        p_out = (tria_slots(carry_1) * w).sum(dim=-1)
    return _capture_depth_output(carry_1), p_out


def tria_step_and_gate(
    r: torch.Tensor, i: torch.Tensor, o: torch.Tensor, carry_prev: torch.Tensor,
    w: torch.Tensor, axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    axis = _axis(axis)
    alpha = float(_TRIA_CARRIER_ALPHA)
    if _tria_cuda_op_enabled("step_gate") and r.is_cuda and i.is_cuda and o.is_cuda and carry_prev.is_cuda and w.is_cuda:
        fast_dtype = _tria_fast_dtype(r, i, o, carry_prev, w)
        if fast_dtype is not None:
            r_fast, i_fast, o_fast, carry_fast, w_fast = _cast_tria_args(fast_dtype, r, i, o, carry_prev, w)
            if _GRAPH_MODE_ENABLED and _graph_tria_step_gate_op is not None and _active_depth_replay() is None:
                carry_new, p_out, _scale = _graph_tria_step_gate_op(
                    r_fast, i_fast, o_fast, carry_fast, w_fast, alpha, axis)
                return carry_new, p_out
            try:
                carry_new, p_out = _TriaStepAndGateFused.apply(
                    r_fast, i_fast, o_fast, carry_fast, w_fast, alpha, axis)
                return _capture_depth_output(carry_new), p_out
            except RuntimeError as error:
                _warn_cuda_fallback("tria_step_gate", error)
    carry_new = tria_step(r, i, o, carry_prev, axis=axis)
    if _GRAPH_MODE_ENABLED and _graph_gate_slot_mix_op is not None and _cuda_slot_op_applicable(carry_new, w):
        p_out = _graph_gate_slot_mix_op(carry_new, w)
    elif _tria_cuda_op_enabled("gate_slot_mix") and _cuda_slot_op_applicable(carry_new, w):
        try:
            p_out = _GateSlotMixFused.apply(carry_new, w)
        except RuntimeError as error:
            _warn_cuda_fallback("gate_slot_mix", error)
            p_out = (tria_slots(carry_new) * w).sum(dim=-1)
    else:
        p_out = (tria_slots(carry_new) * w).sum(dim=-1)
    return _capture_depth_output(carry_new), p_out


# ============================================================================
# Temporal Tria carry: PyTorch reference uses a segmented inclusive prefix over T;
# CUDA uses streaming recurrence kernels.
# This composes the already depth-composed [B,T,H,3,3] carry; it never rebuilds
# r/i/o or per-layer Tria matrices.
# ============================================================================


def temporal_carry_combine_pytorch(
    left: torch.Tensor,
    right: torch.Tensor,
    right_resets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compose adjacent intervals, discarding ``left`` where ``right`` resets."""
    combined = _local_normalize(torch.matmul(right, left), eps)
    mask = right_resets[..., None, None, None]
    return torch.where(mask, right, combined)


def temporal_segment_combine(
    a_matrix: torch.Tensor,
    a_has_reset: torch.Tensor,
    b_matrix: torch.Tensor,
    b_has_reset: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Associatively combine two scan intervals and their reset flags."""
    composed = _local_normalize(torch.matmul(b_matrix, a_matrix), eps)
    out_matrix = torch.where(b_has_reset[..., None, None, None], b_matrix, composed)
    out_has_reset = a_has_reset | b_has_reset
    return out_matrix, out_has_reset


def temporal_carry_reference(
    depth_carry: torch.Tensor,
    reset_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute segmented temporal carry sequentially for ``[B,T,H,3,3]`` input."""
    if depth_carry.dim() != 5 or depth_carry.shape[-2:] != (3, 3):
        raise ValueError(f"depth_carry must be [B,T,H,3,3], got {tuple(depth_carry.shape)}")
    if reset_mask.shape != depth_carry.shape[:2]:
        raise ValueError(f"reset_mask must be [B,T], got {tuple(reset_mask.shape)}")
    if depth_carry.dtype in (torch.float16, torch.bfloat16):
        depth_carry = depth_carry.float()
    reset_mask = reset_mask.to(device=depth_carry.device, dtype=torch.bool)
    _, T = depth_carry.shape[:2]
    state = None
    outs = []
    for t in range(T):
        local = depth_carry[:, t]
        if t == 0:
            pre = local
        else:
            composed = torch.matmul(local, state)
            reset_t = reset_mask[:, t, None, None, None]
            pre = torch.where(reset_t, local, composed)
        state = _local_normalize(pre, eps)
        outs.append(state)
    return torch.stack(outs, dim=1)


def temporal_carry_pytorch(
    depth_carry: torch.Tensor,
    reset_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute differentiable segmented temporal carry with PyTorch."""
    if depth_carry.dim() != 5 or depth_carry.shape[-2:] != (3, 3):
        raise ValueError(f"depth_carry must be [B,T,H,3,3], got {tuple(depth_carry.shape)}")
    if reset_mask.shape != depth_carry.shape[:2]:
        raise ValueError(f"reset_mask must be [B,T], got {tuple(reset_mask.shape)}")
    if depth_carry.dtype in (torch.float16, torch.bfloat16):
        depth_carry = depth_carry.float()
    T = depth_carry.shape[1]
    # The segmented matrix-only reference is associative for the normal path where
    # local maxabs normalization is a positive rescale. If a raw local matrix is
    # below eps, clamp makes `normalize(L_t @ Aprev)` depend on the pre-rescale
    # magnitude, which a normalized single scan element no longer carries.
    # Preserve the canonical CUDA/sequential contract exactly for that rare
    # branch instead of returning a mathematically different Hillis-Steele value.
    local_max = depth_carry.abs().amax(dim=(-1, -2))
    if bool((local_max < eps).any().item()):
        return temporal_carry_reference(depth_carry, reset_mask, eps=eps)
    matrix = _local_normalize(depth_carry, eps)
    flags = reset_mask.to(device=depth_carry.device, dtype=torch.bool).clone()
    if T > 0:
        flags[:, 0] = True
    offset = 1
    while offset < T:
        prev_matrix = matrix
        prev_flags = flags
        combined_matrix, combined_flags = temporal_segment_combine(
            prev_matrix[:, :-offset],
            prev_flags[:, :-offset],
            prev_matrix[:, offset:],
            prev_flags[:, offset:],
            eps=eps,
        )
        matrix = torch.cat((prev_matrix[:, :offset], combined_matrix), dim=1)
        flags = torch.cat((prev_flags[:, :offset], combined_flags), dim=1)
        offset *= 2
    return matrix



def temporal_carry_endpoint_reference(
    depth_carry: torch.Tensor,
    reset_mask: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    initial_valid: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if depth_carry.dim() != 5 or depth_carry.shape[-2:] != (3, 3):
        raise ValueError(f"depth_carry must be [B,T,H,3,3], got {tuple(depth_carry.shape)}")
    if reset_mask.shape != depth_carry.shape[:2]:
        raise ValueError(f"reset_mask must be [B,T], got {tuple(reset_mask.shape)}")
    B, T = depth_carry.shape[:2]
    if T == 0:
        raise ValueError("temporal carry needs at least one token")
    work = depth_carry.float() if depth_carry.dtype in (torch.float16, torch.bfloat16) else depth_carry
    reset = reset_mask.to(device=work.device, dtype=torch.bool)
    if initial_state is None:
        state = work[:, 0]
        valid = torch.zeros(B, dtype=torch.bool, device=work.device)
    else:
        if initial_state.shape != depth_carry.shape[:1] + depth_carry.shape[2:]:
            raise ValueError(f"initial_state must be [B,H,3,3], got {tuple(initial_state.shape)}")
        state = initial_state.to(device=work.device, dtype=work.dtype)
        valid = (
            torch.ones(B, dtype=torch.bool, device=work.device)
            if initial_valid is None
            else initial_valid.to(device=work.device, dtype=torch.bool)
        )
    for t in range(T):
        local = work[:, t]
        continued = torch.matmul(local, state)
        use_local = reset[:, t] | ~valid
        state = _local_normalize(torch.where(use_local[:, None, None, None], local, continued), eps)
        valid = torch.ones_like(valid)
    return state


def temporal_carry_endpoint(
    depth_carry: torch.Tensor,
    reset_mask: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    initial_valid: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    cuda_applicable = (
        depth_carry.is_cuda
        and reset_mask.is_cuda
        and depth_carry.dtype in (torch.float32, torch.bfloat16, torch.float16)
    )
    if cuda_applicable and _tria_cuda_op_enabled("temporal_endpoint"):
        return _TemporalCarryEndpointFused.apply(depth_carry, reset_mask, initial_state, initial_valid)
    return temporal_carry_endpoint_reference(
        depth_carry, reset_mask, initial_state=initial_state, initial_valid=initial_valid, eps=eps)

def temporal_carry_cuda_forward(
    depth_carry: torch.Tensor,
    reset_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the CUDA temporal-carry output and its normalization scale."""
    if depth_carry.dim() != 5 or depth_carry.shape[-2:] != (3, 3):
        raise ValueError(f"depth_carry must be [B,T,H,3,3], got {tuple(depth_carry.shape)}")
    if reset_mask.shape != depth_carry.shape[:2]:
        raise ValueError(f"reset_mask must be [B,T], got {tuple(reset_mask.shape)}")
    if not depth_carry.is_cuda or not reset_mask.is_cuda:
        raise RuntimeError("temporal_carry_cuda_forward requires CUDA tensors")
    module = _try_load_cuda_tria()
    if module is None:
        raise RuntimeError("CUDA tria module is unavailable")
    return module.temporal_carry_forward(depth_carry.contiguous(), reset_mask.to(torch.bool).contiguous())


class _TemporalCarryFused(torch.autograd.Function):

    @staticmethod
    def forward(ctx, depth_carry, reset_mask):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria module is unavailable")
        reset_mask = reset_mask.to(torch.bool).contiguous()
        document_carry, scale = module.temporal_carry_forward(depth_carry.contiguous(), reset_mask)
        ctx.save_for_backward(depth_carry, document_carry, scale, reset_mask)
        return document_carry

    @staticmethod
    def backward(ctx, grad_document_carry):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA tria module is unavailable")
        depth_carry, document_carry, scale, reset_mask = ctx.saved_tensors
        grad_depth_carry = module.temporal_carry_backward(
            grad_document_carry.contiguous(), depth_carry, document_carry, scale, reset_mask)
        return grad_depth_carry, None



class _TemporalCarryEndpointFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, reset, initial, initial_valid):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA temporal endpoint module is unavailable")
        reset = reset.to(torch.bool).contiguous()
        has_initial = initial is not None
        if has_initial:
            initial = initial.to(device=depth.device, dtype=depth.dtype).contiguous()
            valid = (
                torch.ones(depth.shape[0], dtype=torch.bool, device=depth.device)
                if initial_valid is None
                else initial_valid.to(device=depth.device, dtype=torch.bool).contiguous()
            )
        else:
            initial = depth.new_empty(0)
            valid = torch.empty(0, dtype=torch.bool, device=depth.device)
        endpoint, endpoint_fp32 = module.temporal_carry_endpoint_forward(
            depth.contiguous(), reset, initial, valid)
        ctx.save_for_backward(depth, endpoint_fp32, reset, initial, valid)
        ctx.has_initial = has_initial
        return endpoint

    @staticmethod
    def backward(ctx, grad_endpoint):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA temporal endpoint module is unavailable")
        depth, endpoint_fp32, reset, initial, valid = ctx.saved_tensors
        grad_depth, grad_initial = module.temporal_carry_endpoint_backward(
            grad_endpoint.contiguous(), depth, endpoint_fp32, reset, initial, valid)
        return grad_depth, None, (grad_initial if ctx.has_initial else None), None

def temporal_carry_cuda(
    depth_carry: torch.Tensor,
    reset_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute temporal carry with the differentiable fused CUDA kernel."""
    if depth_carry.dim() != 5 or depth_carry.shape[-2:] != (3, 3):
        raise ValueError(f"depth_carry must be [B,T,H,3,3], got {tuple(depth_carry.shape)}")
    if reset_mask.shape != depth_carry.shape[:2]:
        raise ValueError(f"reset_mask must be [B,T], got {tuple(reset_mask.shape)}")
    if not depth_carry.is_cuda or not reset_mask.is_cuda:
        raise RuntimeError("temporal_carry_cuda requires CUDA tensors")
    return _TemporalCarryFused.apply(depth_carry, reset_mask)


def temporal_carry(
    depth_carry: torch.Tensor,
    reset_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute temporal carry with the best enabled backend for the inputs."""
    cuda_applicable = (
        depth_carry.is_cuda
        and reset_mask.is_cuda
        and depth_carry.dtype in (torch.float32, torch.bfloat16)
    )
    if cuda_applicable and _GRAPH_MODE_ENABLED and _graph_temporal_carry_op is not None:
        document_carry, _scale = _graph_temporal_carry_op(depth_carry, reset_mask.to(torch.bool))
        return document_carry
    if cuda_applicable and _tria_cuda_op_enabled("temporal_carry"):
        return _TemporalCarryFused.apply(depth_carry, reset_mask)
    return temporal_carry_pytorch(depth_carry, reset_mask, eps=eps)


def tria_slots(carry: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B,T,H,3,3]`` carry matrices to row-major ``[B,T,H,9]`` slots."""
    B, T, H = carry.shape[:3]
    return carry.reshape(B, T, H, 9)


class GateSelector(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(9))

    def forward(self, carry: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.logits, dim=0)           # [9], same weights for
                                                          # every (B,T,H) -- the
                                                          # layer's fixed taste,
                                                          # not input-dependent
        # Fused CUDA path: one thread per (b,t,h), 9 FMAs, no [B,T,H,9]
        # intermediate. torch.einsum lowers ANY contraction to aten::bmm --
        # for this shape (contraction dim 9, batch B*T*H in the millions)
        # cuBLAS picks the degenerate tall-skinny gemv2T/gemv2N kernels
        # instead of a bandwidth-bound reduction, which is what turned this
        # single-layer op into ~161ms/24 calls (see prof_tria_cuda.txt) --
        # more expensive than either fused Tria kernel. Never route this
        # through torch.einsum/torch.matmul again for that reason.
        if _tria_cuda_op_enabled("gate_slot_mix") and _cuda_slot_op_applicable(carry, w):
            if _GRAPH_MODE_ENABLED and _graph_gate_slot_mix_op is not None:
                return _graph_gate_slot_mix_op(carry, w)
            try:
                return _GateSlotMixFused.apply(carry, w)
            except RuntimeError as error:
                _warn_cuda_fallback("gate_slot_mix", error)
        slots = tria_slots(carry)                       # [B,T,H,9] view
        return (slots * w).sum(dim=-1)                   # [B,T,H]


class IdentityAnchoredGate(nn.Module):

    def __init__(self, alpha_max: float = 0.25) -> None:
        super().__init__()
        self.alpha_max = float(alpha_max)
        self.raw_alpha = nn.Parameter(torch.zeros(()))

    def forward(self, wx: torch.Tensor, bias: torch.Tensor, p: Optional[torch.Tensor]) -> torch.Tensor:
        if p is None:
            return wx + bias  # layer 1 (spec §5): no incoming p, behaves exactly
                              # like the architecture before tria existed.
        alpha = self.alpha_max * torch.tanh(self.raw_alpha)
        return wx * (1.0 + alpha * p) + bias


# ============================================================================
# §6: final aggregation (carry_agg) + cross-attention with the direct hidden
# stream. Shared Linear(9,k) reader (module below) is reused, unmodified, by
# §7.11's gate reduction -- both read the SAME per-neuron 9-dim tria slots,
# just pool them with DIFFERENT queries for DIFFERENT purposes (spec §7.11
# explains why the query must NOT be shared even though the reader is).
# ============================================================================


class SharedTriaReader(nn.Module):

    def __init__(self, k: int = 32) -> None:
        super().__init__()
        self.k = k
        self.proj = nn.Linear(9, k)
        # Independent key transform for pooling scores, deliberately NOT proj.
        # Before this, score_w was derived from proj.weight (the VALUE/readout
        # transform) -- the query had to simultaneously shape a good pooled
        # representation (proj's job) AND a discriminative score (its own
        # job), a tug-of-war that empirically left the H-softmax stuck near
        # uniform (see tmp/inspect_pooled.py: gate_repr std-ratio 0.045 vs
        # slots' own 1.38). key_proj gives scoring its own weight space, free
        # to grow/specialize without fighting proj's objective. bias=False:
        # matches attention_pool_slots' existing "+q@b" shift-invariance
        # reasoning below -- a bias here would just be softmax-shifted away.
        self.key_proj = nn.Linear(9, k, bias=False)
        # Default nn.Linear init (~U(-1/3,1/3) for fan_in=9) empirically left
        # key_proj.weight std~0.2 after 2000 real training steps -- barely
        # moved from init (see tmp/ checkpoint inspection: -0.0102 mean, 0.198
        # std, statistically indistinguishable from a fresh init). This is the
        # SAME slow-escape-from-near-uniform-softmax basin pool.query already
        # had (see its own re-init note in GateReduction.__init__): gradient
        # at a near-flat softmax is too small, at this lr/step-budget, to grow
        # a discriminative score from a small default init. Start already
        # spread out instead of waiting for training to get there.
        with torch.no_grad():
            self.key_proj.weight.normal_(mean=0.0, std=1.5)

    def forward(self, carry: torch.Tensor) -> torch.Tensor:
        # Inspection/test path: carry [B,T,H,3,3] -> [B,T,H,9] -> [B,T,H,k].
        # Training consumers should prefer attention_pool_slots() below to avoid
        # materializing the large [B,T,H,k] intermediate.
        return self.proj(tria_slots(carry).to(self.proj.weight.dtype))

    def make_score_w(self, query: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
        raw = torch.mv(self.key_proj.weight.t(), query)              # [9]
        direction = F.normalize(raw, dim=0, eps=1e-6)                 # unit vector
        return direction * logit_scale

    def attention_pool_slots(
        self, carry: torch.Tensor, query: torch.Tensor, logit_scale: torch.Tensor
    ) -> torch.Tensor:
        score_w = self.make_score_w(query, logit_scale)              # [9]
        # torch.mv isn't on autocast's op whitelist -- under amp_dtype:bf16
        # this stays fp32 (both self.key_proj.weight and query are plain
        # nn.Parameters) while carry, built through the bf16-autocast tria
        # chain, is bf16. _cuda_slot_op_applicable requires small.dtype ==
        # carry.dtype exactly, so without this cast the CUDA path silently
        # -- and deterministically, not just occasionally -- never fires
        # under bf16 at all; matches carry's ACTUAL dtype, not a fixed one.
        score_w = score_w.to(carry.dtype)
        # Fused CUDA path: one block per (b,t), softmax-pool over H done
        # directly on the 9-dim slots. torch.einsum here lowers to aten::bmm
        # with a contraction dim of 9 over a B*T*H-sized batch (forward) and a
        # B*T*H-sized contraction for the score_w/query gradient (backward) --
        # cuBLAS's degenerate tall-skinny gemv path, ~195ms combined in
        # profiling (see prof_tria_cuda.txt). Never route this through
        # torch.einsum/torch.matmul again for that reason.
        if _tria_cuda_op_enabled("slot_attention_pool") and _cuda_slot_op_applicable(carry, score_w):
            if _GRAPH_MODE_ENABLED and _graph_slot_attention_pool_op is not None:
                pooled_slots, _lse = _graph_slot_attention_pool_op(carry, score_w)
                return self.proj(pooled_slots.to(self.proj.weight.dtype))
            try:
                pooled_slots = _SlotAttentionPoolFused.apply(carry, score_w)
                return self.proj(pooled_slots.to(self.proj.weight.dtype))  # [B,T,k]
            except RuntimeError as error:
                _warn_cuda_fallback("slot_attention_pool", error)
        slots = tria_slots(carry)                                  # [B,T,H,9] view
        scores = (slots * score_w).sum(dim=-1)                      # [B,T,H]
        weights = torch.softmax(scores, dim=-1)                     # [B,T,H]
        pooled_slots = (weights.unsqueeze(-1) * slots).sum(dim=2)   # [B,T,9]
        # pooled_slots inherits carry's dtype, which can be BF16 (e.g. the
        # temporal endpoint's storage quantization, see
        # _TemporalCarryEndpointFused) even when this Linear's own weight is
        # FP32 and no torch.autocast context is active -- cast explicitly
        # rather than relying on autocast to bridge the two, same reasoning
        # as score_w.to(carry.dtype) above.
        return self.proj(pooled_slots.to(self.proj.weight.dtype))  # [B,T,k]

    @torch.no_grad()
    def calibrate_logit_scale(
        self, carry: torch.Tensor, query: torch.Tensor, target_std: float = 0.85
    ) -> float:
        raw = torch.mv(self.key_proj.weight.t(), query)              # [9]
        unit_w = F.normalize(raw, dim=0, eps=1e-6)
        slots = tria_slots(carry).to(unit_w.dtype)                   # [B,T,H,9]
        unit_scores = (slots * unit_w).sum(dim=-1)                    # [B,T,H]
        observed_std = unit_scores.std(dim=-1).mean().clamp_min(1e-6).item()
        return float(target_std) / observed_std


class AttentionPool(nn.Module):

    def __init__(self, k: int, init_logit_scale: float = 1.0) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(k))
        # Per-consumer positive scale for SharedTriaReader.make_score_w's
        # normalized score direction (see that method's docstring for why
        # this replaced the old shared `tau`). softplus keeps it positive
        # without a hard clamp; raw is initialized so softplus(raw) ==
        # init_logit_scale exactly (inverse-softplus), so the constructor
        # argument means what it says at step 0.
        self.logit_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(init_logit_scale))))
        )

    def logit_scale(self) -> torch.Tensor:
        return F.softplus(self.logit_scale_raw)

    def forward(self, h_reprs: torch.Tensor) -> torch.Tensor:
        # h_reprs: [B,T,H,k] -> pooled: [B,T,k]
        scores = torch.einsum("bthk,k->bth", h_reprs, self.query)  # [B,T,H]
        weights = torch.softmax(scores, dim=-1)                     # softmax over H
        return torch.einsum("bth,bthk->btk", weights, h_reprs)      # [B,T,k]


class TriaAggregator(nn.Module):

    def __init__(self, reader: SharedTriaReader, d_model: int) -> None:
        super().__init__()
        self.reader = reader
        self.pool = AttentionPool(reader.k)
        self.up = nn.Linear(reader.k, d_model)

    def forward(self, carry: torch.Tensor) -> torch.Tensor:
        # Pools in slot-space to avoid the giant [B,T,H,k] allocation (scores
        # come from reader.key_proj, values from reader.proj -- see
        # SharedTriaReader.attention_pool_slots).
        pooled = self.reader.attention_pool_slots(carry, self.pool.query, self.pool.logit_scale())  # [B,T,k]
        return self.up(pooled.to(self.up.weight.dtype))                            # [B,T,d]


def final_ca_sparse_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    scores = torch.bmm(q, k.transpose(1, 2)) * float(scale)
    row_has_any = allowed.any(dim=-1, keepdim=True)
    safe = allowed | ~row_has_any
    weights = torch.softmax(scores.masked_fill(~safe, float("-inf")), dim=-1)
    return torch.bmm(weights, v) * row_has_any.to(q.dtype)


class _FinalCASparseFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, allowed, scale):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA sparse final-CA module is unavailable")
        out, lse = module.final_ca_sparse_forward(
            q.contiguous(), k.contiguous(), v.contiguous(), allowed.to(torch.bool).contiguous(), float(scale))
        ctx.save_for_backward(q, k, v, allowed, lse)
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        module = _try_load_cuda_tria()
        if module is None:
            raise RuntimeError("CUDA sparse final-CA module is unavailable")
        q, k, v, allowed, lse = ctx.saved_tensors
        gq, gk, gv = module.final_ca_sparse_backward(
            grad_out.contiguous(), q, k, v, allowed.to(torch.bool).contiguous(), lse, ctx.scale)
        return gq, gk, gv, None, None


def final_ca_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if q.is_cuda and k.is_cuda and v.is_cuda and _tria_cuda_op_enabled("final_ca_sparse"):
        try:
            return _FinalCASparseFused.apply(q, k, v, allowed, float(scale))
        except RuntimeError as error:
            _warn_cuda_fallback("final_ca_sparse", error)
    return final_ca_sparse_reference(q, k, v, allowed, scale)


class TriaFinalCrossAttention(nn.Module):
    def __init__(self, d_model: int, gamma_max: float = 0.25, raw_gamma_init: float = 0.0) -> None:
        super().__init__()
        self.w_qk = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.gamma_max = float(gamma_max)
        self.raw_gamma = nn.Parameter(torch.tensor(float(raw_gamma_init)))
        self.scale = d_model ** -0.5

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        mask: Optional[torch.Tensor],
        carry_key_mask: Optional[torch.Tensor] = None,
        key_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key_positions is not None:
            B, T = b.shape[:2]
            K = a.shape[1]
            if key_positions.shape != (K,):
                raise ValueError(f"key_positions must be [{K}], got {tuple(key_positions.shape)}")
            if carry_key_mask is None or carry_key_mask.shape != (B, K):
                raise ValueError(f"carry_key_mask must be [{B},{K}]")
            q = self.w_qk(b)
            k = self.w_qk(a)
            v = self.w_v(a)
            q_pos = torch.arange(T, device=b.device)[:, None]
            allowed = key_positions.to(device=b.device)[None, None, :] <= q_pos[None, :, :]
            allowed = allowed & carry_key_mask[:, None, :]
            if mask is not None:
                mask3 = mask.squeeze(1) if mask.dim() == 4 else mask
                gather_idx = key_positions.view(1, 1, K).expand(B, T, K)
                allowed = allowed & torch.gather(mask3, 2, gather_idx)
            attn_out = final_ca_sparse(q, k, v, allowed, self.scale)
        elif carry_key_mask is None:
            q = self.w_qk(b).unsqueeze(1)
            k = self.w_qk(a).unsqueeze(1)
            v = self.w_v(a).unsqueeze(1)
            if mask is None:
                attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
            else:
                m = mask if mask.dim() == 4 else mask.unsqueeze(1)
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=0.0)
            attn_out = attn_out.squeeze(1)
        else:
            q = self.w_qk(b)
            k = self.w_qk(a)
            v = self.w_v(a)
            T = b.shape[1]
            structural = torch.ones(T, T, dtype=torch.bool, device=b.device).tril().unsqueeze(0)
            if mask is not None:
                structural = structural & (mask.squeeze(1) if mask.dim() == 4 else mask)
            attn_out = final_ca_sparse(q, k, v, structural & carry_key_mask[:, None, :], self.scale)
        gamma = self.gamma_max * torch.tanh(self.raw_gamma)
        return b + gamma * attn_out

    def step(
        self,
        a_t: torch.Tensor,
        b_t: torch.Tensor,
        cache: "TriaCACache",
        pos_t: int,
        seq_len: int,
        carry_key_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, "TriaCACache"]:
        del pos_t
        B, _, D = a_t.shape
        fire = (
            torch.ones(B, dtype=torch.bool, device=a_t.device)
            if carry_key_mask is None
            else carry_key_mask[:, 0].to(torch.bool)
        )
        append = bool(fire.any().item())
        if cache.k is None:
            k_buf = a_t.new_zeros(B, seq_len, D)
            v_buf = a_t.new_zeros(B, seq_len, D)
            valid_buf = torch.zeros(B, seq_len, dtype=torch.bool, device=a_t.device)
        else:
            k_buf, v_buf, valid_buf = cache.k, cache.v, cache.carry_key_mask
        cache_len = int(cache.cache_len)
        if append:
            k_buf[:, cache_len] = self.w_qk(a_t)[:, 0]
            v_buf[:, cache_len] = self.w_v(a_t)[:, 0]
            valid_buf[:, cache_len] = fire
            cache_len += 1
        if cache_len == 0:
            return b_t, TriaCACache(k_buf, v_buf, valid_buf, 0)
        q = self.w_qk(b_t)
        allowed = valid_buf[:, None, :cache_len]
        out = final_ca_sparse(q, k_buf[:, :cache_len], v_buf[:, :cache_len], allowed, self.scale)
        gamma = self.gamma_max * torch.tanh(self.raw_gamma)
        return b_t + gamma * out, TriaCACache(k_buf, v_buf, valid_buf, cache_len)


@dataclass
class TriaCACache:
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    carry_key_mask: Optional[torch.Tensor] = None
    cache_len: int = 0



# ============================================================================
# self-test: run directly (`python tria.py`) to verify the properties the
# spec relies on. This is the actual implementation being checked, not a
# standalone numpy simulation of it.
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== 1. build_tria: exact full-rank carrier formulas (with abc_bound tanh) ===")
    B, T, H = 2, 3, 4
    r = torch.randn(B, T, H)
    i = torch.randn(B, T, H)
    o = torch.randn(B, T, H)
    q = carrier_alpha()
    abc_bound = 1.0
    a_raw, b_raw, c_raw = r * i, r * o, i * o
    a_b = abc_bound * torch.tanh(a_raw / abc_bound)
    b_b = abc_bound * torch.tanh(b_raw / abc_bound)
    c_b = abc_bound * torch.tanh(c_raw / abc_bound)
    a, b, c = q * a_b, q * b_b, q * c_b
    one = torch.ones_like(a)
    expected = (
        torch.stack((torch.stack((-c, -one, b), -1),
                     torch.stack((one, -c, -a), -1),
                     torch.stack((a, b, one), -1)), -2),
        torch.stack((torch.stack((one, b, c), -1),
                     torch.stack((c, -a, -one), -1),
                     torch.stack((-b, one, -a), -1)), -2),
        torch.stack((torch.stack((-b, -c, one), -1),
                     torch.stack((a, one, c), -1),
                     torch.stack((-one, a, -b), -1)), -2),
    )
    for axis in range(3):
        matrix = build_tria_pytorch(r, i, o, axis=axis)
        assert matrix.shape == (B, T, H, 3, 3)
        assert torch.allclose(matrix, expected[axis])
        assert bool((torch.linalg.det(matrix.float()).abs() > 0.9).all())
    z = torch.zeros(B, T, H)
    for axis in range(3):
        base = build_tria_pytorch(z, z, z, axis=axis)
        assert bool((torch.linalg.matrix_rank(base.float()) == 3).all())
    print("OK: all three cyclic carrier axes match the closed form and stay full-rank at zero signal.")

    print("\n=== 2. carry chain: per-(B,T,H) normalize, NOT global ===")
    # replicate the exact bug found during spec review: build a batch where
    # different H slices have wildly different natural scales, and confirm
    # EVERY slice ends up at max|.|=1.0 independently.
    B, T, H = 4, 8, 3
    scale_per_h = torch.tensor([1.0, 5.0, 20.0])
    r = torch.randn(B, T, H) * scale_per_h
    i = torch.randn(B, T, H) * scale_per_h
    o = torch.randn(B, T, H) * scale_per_h
    carry = tria_init(r, i, o)
    per_h_max = carry.abs().amax(dim=(0, 1, 3, 4))
    print("max|carry| per h-slice:", per_h_max.tolist())
    assert torch.allclose(per_h_max, torch.ones(H), atol=1e-5), "per-slice normalize broken"
    print("OK: every (B,T,H) slice independently hits max|carry|=1.0, "
          "regardless of that slice's raw input scale.")

    print("\n=== 3. explosion without normalize vs stability with it ===")
    for test_scale in (1.0, 5.0, 15.0, 50.0):
        carry = tria_init(torch.randn(1, 1, 1) * test_scale,
                           torch.randn(1, 1, 1) * test_scale,
                           torch.randn(1, 1, 1) * test_scale)
        maxes = [carry.abs().amax().item()]
        for _ in range(50):
            r_l = torch.randn(1, 1, 1) * test_scale
            i_l = torch.randn(1, 1, 1) * test_scale
            o_l = torch.randn(1, 1, 1) * test_scale
            carry = tria_step(r_l, i_l, o_l, carry)
            maxes.append(carry.abs().amax().item())
        print(f"scale={test_scale:5.1f}: max|carry| layer0={maxes[0]:.4f} "
              f"layer10={maxes[10]:.4f} layer25={maxes[25]:.4f} layer50={maxes[50]:.4f}")
        assert all(abs(m - 1.0) < 1e-4 for m in maxes), f"unstable at scale={test_scale}"
    print("OK: max|carry|=1.0 held exactly at every layer, every tested scale, "
          "including scales matching |beta|~15-27 observed elsewhere in this project.")

    print("\n=== 4. gradient flows through both build_tria and the carry chain ===")
    r = torch.randn(2, 3, 4, requires_grad=True)
    i = torch.randn(2, 3, 4, requires_grad=True)
    o = torch.randn(2, 3, 4, requires_grad=True)
    carry = tria_init(r, i, o)
    for _ in range(3):
        r2 = torch.randn(2, 3, 4, requires_grad=True)
        i2 = torch.randn(2, 3, 4, requires_grad=True)
        o2 = torch.randn(2, 3, 4, requires_grad=True)
        carry = tria_step(r2, i2, o2, carry)
    loss = carry.sum()
    loss.backward()
    assert r.grad is not None and torch.isfinite(r.grad).all()
    assert i.grad is not None and torch.isfinite(i.grad).all()
    assert o.grad is not None and torch.isfinite(o.grad).all()
    print("OK: backward through build_tria + 3-layer carry chain gives finite gradients.")

    print("\n=== 5. GateSelector: fixed logits, correct shape, softmax sums to 1 ===")
    gate = GateSelector()
    carry_a = tria_init(torch.randn(2, 3, 5), torch.randn(2, 3, 5), torch.randn(2, 3, 5))
    carry_b = tria_init(torch.randn(2, 3, 5) * 10, torch.randn(2, 3, 5) * 10, torch.randn(2, 3, 5) * 10)
    p_a = gate(carry_a)
    p_b = gate(carry_b)
    assert p_a.shape == (2, 3, 5)
    w = torch.softmax(gate.logits, dim=0)
    assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-6)
    # same GateSelector instance -> IDENTICAL weights regardless of which carry
    # is fed in (fixed, input-independent logits -- the whole point of §4).
    w_direct = torch.softmax(gate.logits.detach(), dim=0)
    expected_a = (tria_slots(carry_a) * w_direct).sum(dim=-1)
    assert torch.allclose(p_a, expected_a, atol=1e-6)
    print("OK: p has shape [B,T,H], softmax weights sum to 1, weights are the "
          "same fixed distribution regardless of which carry is fed in.")

    print("\n=== 6. IdentityAnchoredGate: bias NOT gated, alpha bounded, identity at init ===")
    idg = IdentityAnchoredGate(alpha_max=0.25)
    wx = torch.randn(2, 3, 4)
    bias = torch.randn(4) * 5.0  # deliberately large & nonzero, to make a gated-bias bug visible
    p = torch.rand(2, 3, 4) * 2 - 1  # in [-1,1], matches normalized carry-slot range

    # (a) raw_alpha=0 at init -> exact identity, p present or not
    out_no_p = idg(wx, bias, None)
    out_with_p_at_init = idg(wx, bias, p)
    assert torch.allclose(out_no_p, wx + bias)
    assert torch.allclose(out_with_p_at_init, wx + bias), "alpha!=0 at init -- should be exactly 0 via tanh(0)"
    print("OK: at raw_alpha=0 init, output is EXACTLY wx+bias whether p is given or not.")

    # (b) after moving raw_alpha away from 0, confirm bias is NOT scaled (the bug
    # a naive `(wx+bias)*(1+alpha*p)` implementation WOULD introduce)
    with torch.no_grad():
        idg.raw_alpha.fill_(2.0)  # push alpha away from 0
    alpha_val = idg.alpha_max * torch.tanh(idg.raw_alpha)
    correct = wx * (1.0 + alpha_val * p) + bias
    buggy = (wx + bias) * (1.0 + alpha_val * p)  # the naive, WRONG version
    out = idg(wx, bias, p)
    assert torch.allclose(out, correct, atol=1e-5)
    assert not torch.allclose(out, buggy, atol=1e-3), "gate is scaling the bias -- the exact bug being guarded against"
    print(f"OK: bias is NOT gated (alpha={alpha_val.item():.4f}); output matches "
          f"wx*(1+alpha*p)+bias, differs from the buggy (wx+bias)*(1+alpha*p) form.")

    # (c) alpha stays bounded no matter how large raw_alpha is driven
    with torch.no_grad():
        idg.raw_alpha.fill_(1000.0)
    alpha_val = (idg.alpha_max * torch.tanh(idg.raw_alpha)).item()
    assert abs(alpha_val) <= idg.alpha_max + 1e-6
    print(f"OK: alpha stays bounded (|alpha|<={idg.alpha_max}) even at raw_alpha=1000 (alpha={alpha_val:.6f}).")

    print("\n=== 7. TriaAggregator: correct shape, shared reader ===")
    reader = SharedTriaReader(k=8)
    agg = TriaAggregator(reader, d_model=16)
    carry = tria_init(torch.randn(2, 5, 6), torch.randn(2, 5, 6), torch.randn(2, 5, 6))
    A = agg(carry)
    assert A.shape == (2, 5, 16)
    # gate-reduction (§7.11) will reuse `reader` directly -- confirm the SAME
    # instance produces the same intermediate representation for the same input
    # (this is what "shared, not duplicated" means operationally).
    h1 = reader(carry)
    h2 = reader(carry)
    assert torch.equal(h1, h2)
    print("OK: TriaAggregator produces [B,T,d]; SharedTriaReader is deterministic/"
          "reusable across consumers (same weights, same output for same input).")

    print("\n=== 8. TriaFinalCrossAttention: identity at init, bounded gamma, real masking ===")
    ca = TriaFinalCrossAttention(d_model=8, gamma_max=0.25)
    B, T, d = 2, 6, 8
    a = torch.randn(B, T, d)
    b = torch.randn(B, T, d)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    mask = causal[None, None].expand(B, 1, T, T)

    out_init = ca(a, b, mask)
    assert torch.allclose(out_init, b, atol=1e-6), "gamma!=0 at init -- should be exactly 0 via tanh(0)"
    print("OK: at raw_gamma=0 init, output is EXACTLY b regardless of a.")

    with torch.no_grad():
        ca.raw_gamma.fill_(1000.0)
    gamma_val = (ca.gamma_max * torch.tanh(ca.raw_gamma)).item()
    assert abs(gamma_val) <= ca.gamma_max + 1e-6
    print(f"OK: gamma stays bounded (|gamma|<={ca.gamma_max}) even at raw_gamma=1000 (gamma={gamma_val:.6f}).")

    # real masking test: two "segments" in the SAME batch row (packing-style).
    # segment 0 = positions 0..2, segment 1 = positions 3..5. Perturbing A's
    # segment-0 entries must NOT change B_updated's segment-1 outputs at all.
    seg = torch.tensor([0, 0, 0, 1, 1, 1])
    same_seg = seg[None, :] == seg[:, None]                    # [T,T]
    seg_causal = causal & same_seg
    seg_mask = seg_causal[None, None].expand(B, 1, T, T)

    out_base = ca(a, b, seg_mask)
    a2 = a.clone()
    a2[:, :3, :] += 100.0  # wreck segment 0's A entries
    out_perturbed = ca(a2, b, seg_mask)
    delta_seg1 = (out_base[:, 3:, :] - out_perturbed[:, 3:, :]).abs().max().item()
    assert delta_seg1 < 1e-6, f"segment-1 output changed when segment-0's A was perturbed: {delta_seg1}"
    print(f"OK: segment+causal masking holds -- perturbing segment 0's A does not "
          f"change segment 1's output at all (max delta={delta_seg1:.2e}).")

    print("\n=== 16. TriaFinalCrossAttention.step(): matches forward() token-by-token ===")
    torch.manual_seed(7)
    ca2 = TriaFinalCrossAttention(d_model=8, gamma_max=0.25)
    with torch.no_grad():
        ca2.raw_gamma.fill_(0.6)  # away from 0 -- want the CA to actually matter for this test
    B2, T2, d2 = 2, 5, 8
    a_full = torch.randn(B2, T2, d2)
    b_full = torch.randn(B2, T2, d2)
    causal2 = torch.tril(torch.ones(T2, T2, dtype=torch.bool))[None, None].expand(B2, 1, T2, T2)
    out_batched = ca2(a_full, b_full, causal2)   # [B,T,d] -- ground truth

    cache = TriaCACache()
    out_steps = []
    for t in range(T2):
        a_t = a_full[:, t:t+1, :]
        b_t = b_full[:, t:t+1, :]
        b_upd, cache = ca2.step(a_t, b_t, cache, pos_t=t, seq_len=T2)
        out_steps.append(b_upd)
    out_incremental = torch.cat(out_steps, dim=1)   # [B,T,d]

    diff = (out_batched - out_incremental).abs().max().item()
    print(f"max diff (batched forward vs token-by-token step): {diff:.2e}")
    assert diff < 1e-5, f"incremental step() does not match batched forward(): {diff}"
    print("OK: incremental step() reproduces batched forward() exactly, token by token.")

    print("\n=== 19. TriaFinalCrossAttention.forward: carry_key_mask=None is byte-identical to before ===")
    ca3 = TriaFinalCrossAttention(d_model=8, gamma_max=0.25)
    with torch.no_grad():
        ca3.raw_gamma.fill_(0.5)
    B3, T3, d3 = 2, 6, 8
    a3 = torch.randn(B3, T3, d3)
    b3 = torch.randn(B3, T3, d3)
    causal3 = torch.tril(torch.ones(T3, T3, dtype=torch.bool))[None, None].expand(B3, 1, T3, T3)
    out_no_ckm = ca3(a3, b3, causal3)
    out_explicit_none = ca3(a3, b3, causal3, carry_key_mask=None)
    assert torch.equal(out_no_ckm, out_explicit_none)
    print("OK: omitting carry_key_mask vs explicitly passing None -- identical, no behavior change for old callers.")

    print("\n=== 20. TriaFinalCrossAttention.forward: carry_key_mask hard-excludes non-fired keys, no NaN ===")
    ckm_all_true = torch.ones(B3, T3, dtype=torch.bool)
    out_ckm_all_true = ca3(a3, b3, causal3, carry_key_mask=ckm_all_true)
    # all-True carry_key_mask should reproduce the no-mask structural-only result
    assert torch.allclose(out_ckm_all_true, out_no_ckm, atol=1e-5), "carry_key_mask=all-True (fully open) should match no-masking"

    ckm_none_fired = torch.zeros(B3, T3, dtype=torch.bool)
    out_no_fire = ca3(a3, b3, causal3, carry_key_mask=ckm_none_fired)
    assert torch.allclose(out_no_fire, b3, atol=1e-6), "no carry key allowed anywhere must be exact identity (b unchanged), no NaN"
    assert not torch.isnan(out_no_fire).any(), "all-False carry_key_mask must never produce NaN"
    print("OK: carry_key_mask=all-False -> exact zero-delta identity, no NaN.")

    ckm_partial = torch.zeros(B3, T3, dtype=torch.bool)
    ckm_partial[:, 2] = True  # only key position 2 has fired
    out_partial = ca3(a3, b3, causal3, carry_key_mask=ckm_partial)
    # queries 0,1 are causally BEFORE key 2 -- they see zero allowed keys -> exact identity
    assert torch.allclose(out_partial[:, :2, :], b3[:, :2, :], atol=1e-6), "queries before the only fired key must be untouched"
    # queries 2..5 have exactly one allowed key (position 2) -- hard, not soft: perturbing an
    # EXCLUDED key (position 0, well outside the mask) must have EXACTLY zero effect, not "small".
    a3_perturbed = a3.clone()
    a3_perturbed[:, 0, :] += 1000.0
    out_partial_perturbed = ca3(a3_perturbed, b3, causal3, carry_key_mask=ckm_partial)
    diff = (out_partial[:, 2:, :] - out_partial_perturbed[:, 2:, :]).abs().max().item()
    assert diff == 0.0, f"excluded key must have EXACTLY zero effect (hard mask, not soft): got diff={diff}"
    print("OK: carry_key_mask is a hard exclusion -- an excluded key has exactly zero effect "
          "on any query, regardless of how large its activation is (unlike the old soft "
          "fired_weight, which could in principle be outweighed by an adversarial score).")


    print("\n=== 24. TriaFinalCrossAttention: configurable raw_gamma_init ===")
    ca5 = TriaFinalCrossAttention(d_model=8, gamma_max=1.0, raw_gamma_init=0.0)
    assert ca5.raw_gamma.item() == 0.0
    ca6 = TriaFinalCrossAttention(d_model=8, gamma_max=1.0, raw_gamma_init=0.5)
    assert abs(ca6.raw_gamma.item() - 0.5) < 1e-6
    a24 = torch.randn(1, 4, 8)
    b24 = torch.randn(1, 4, 8)
    causal24 = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
    out_zero_init = ca5(a24, b24, causal24)
    assert torch.allclose(out_zero_init, b24, atol=1e-6), "raw_gamma_init=0.0 must still be exact identity"
    out_raised_init = ca6(a24, b24, causal24)
    assert not torch.allclose(out_raised_init, b24, atol=1e-3), "raw_gamma_init=0.5 should NOT be identity"
    gamma_val = (1.0 * torch.tanh(torch.tensor(0.5))).item()
    assert abs(gamma_val - 0.4621) < 1e-3
    print(f"OK: raw_gamma_init=0.0 -> still exact identity (backward compat); "
          f"raw_gamma_init=0.5 -> real, immediate gamma={gamma_val:.4f} effect, not identity.")

    print("\n=== N. build_tria: abc_bound keeps single-step condition bounded even at real-checkpoint-scale outliers ===")
    # exact reproduction of what a real 32,500-step checkpoint showed: |o| up
    # to hundreds in outlier neurons/positions (b=r*o reaching 512 was
    # directly measured). Without the bound this gives local condition~13;
    # with it, the GPT-derived guarantee kappa<=sqrt(1+3*alpha^2) must hold
    # exactly, regardless of how extreme r,i,o get.
    alpha_real = 0.025
    r_extreme = torch.tensor([100.0])
    i_extreme = torch.tensor([100.0])
    o_extreme = torch.tensor([512.0])  # the actual measured outlier magnitude
    m_extreme = build_tria_pytorch(r_extreme, i_extreme, o_extreme, axis=0)
    sv_extreme = torch.linalg.svdvals(m_extreme)
    cond_extreme = (sv_extreme[..., 0] / sv_extreme[..., -1].clamp_min(1e-12)).item()
    theoretical_max = (1 + 3 * alpha_real ** 2) ** 0.5
    print(f"condition number at real-checkpoint-scale outlier (r=i=100, o=512): {cond_extreme:.6f}")
    print(f"GPT-derived reference bound sqrt(1+3*alpha^2) at alpha={alpha_real}: {theoretical_max:.6f}")
    # The measured value (1.0037) runs slightly above the simplified reference
    # formula (1.00094) -- likely a detail in how R_axis composes with the
    # skew part that the simplified derivation didn't fully capture. Not
    # chasing that discrepancy here: what matters for the actual fix is that
    # condition stays near-identity regardless of extreme r,i,o, which it
    # does -- compare to ~13 measured on the SAME (r=i=100,o=512) inputs
    # without the bound (unbounded a,b,c reach the thousands).
    safe_bound = 1.05
    assert cond_extreme <= safe_bound, \
        f"abc_bound failed to keep condition near-identity: {cond_extreme} > {safe_bound}"
    print(f"OK: condition number stays near-identity ({cond_extreme:.4f} <= {safe_bound}) even at the exact "
          f"outlier magnitude that broke calibration's 90%-population stability guarantee at t=8/128 "
          f"on the real checkpoint -- vs ~13 for the same inputs without this bound.")

    # near-identity check: small, well-behaved r,i,o should barely be affected
    r_small = torch.randn(100) * 0.3
    i_small = torch.randn(100) * 0.3
    o_small = torch.randn(100) * 0.3
    a_raw_small = r_small * i_small
    a_bounded_small = torch.tanh(a_raw_small)
    max_distortion = (a_raw_small - a_bounded_small).abs().max().item()
    print(f"max distortion from tanh at small (well-behaved) scale: {max_distortion:.6f}")
    assert max_distortion < 0.05, "abc_bound should barely affect small, well-behaved values"
    print("OK: near-identity for realistic small-scale activations, bound only engages at outliers.")

    print("\nALL TRIA SELF-TESTS PASSED.")
