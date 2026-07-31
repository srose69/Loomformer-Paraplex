from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tria
from loomformer_runtime.distributed import ddp_print
from . import state as S
from .primitives import (
    act_fn,
    cuda_autocast_dtype_or_none,
    fanin_std,
    init_linear_fanin,
    init_linear_residual,
    powlu_gate,
)

_cuda_beta_space_module = None
_cuda_beta_space_tried = False
_cuda_beta_space_active_printed = False
_cuda_beta_space_fallback_printed = False
_cuda_paraplex_module = None
_cuda_paraplex_tried = False

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
    if S.PHASE_GRAD_MODE == "secant":
        if anchor is None:
            raise ValueError("phase_grad_mode='secant' needs an anchor tensor (see ParaplexFFN.beta_anchor)")
        # The module-level EMA buffer is updated in-place once per chunk. Snapshot
        # it before entering any custom autograd/custom-op path so each invocation
        # owns a stable explicit input through backward.
        anchor = anchor.detach().clone()
        if S.USE_CUDA_PHASE_SIN and beta.is_cuda and beta.dtype in (torch.float32, torch.float16, torch.bfloat16):
            ext = _try_load_cuda_phase_sin()
            if ext is not None:
                if S.GRAPH_MODE_ENABLED and S._graph_phase_sin_secant_op is not None:
                    return S._graph_phase_sin_secant_op(beta, anchor, 1e-4)
                return _PhaseSinSecantCUDA.apply(beta, anchor, 1e-4)
        return _PhaseSinSecant.apply(beta, anchor)
    eps = max(S.PHASE_GRAD_FLOOR, 0.0)
    if S.USE_CUDA_PHASE_SIN and beta.is_cuda and beta.dtype in (torch.float32, torch.float16, torch.bfloat16):
        ext = _try_load_cuda_phase_sin()
        if ext is not None:
            if S.GRAPH_MODE_ENABLED and S._graph_phase_sin_op is not None:
                return S._graph_phase_sin_op(beta, eps)
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
    if S.USE_CUDA_PVPOWLU and x1.is_cuda and x2.is_cuda and x1.dtype in (torch.float32, torch.float16, torch.bfloat16):
        if x2.dtype == x1.dtype:
            ext = _try_load_cuda_pvpowlu()
            if ext is not None:
                if S.GRAPH_MODE_ENABLED and S._graph_pvpowlu_op is not None:
                    return S._graph_pvpowlu_op(x1, x2, float(m))
                return _PvPowluCUDA.apply(x1, x2, float(m))
    return x1 * powlu_gate(x2, m)

def _beta_space_fast_dtype(u: torch.Tensor, q_h: torch.Tensor, k_ctx_h: torch.Tensor,
                           k_ctx_h2: torch.Tensor, d_h: torch.Tensor,
                           w1_imag: torch.Tensor) -> Optional[torch.dtype]:
    ac_dtype = cuda_autocast_dtype_or_none()
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
        B, T, _n, _hidden, _imag_in = ctx.shapes
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
    if S.GRAPH_MODE_ENABLED and S._graph_beta_space_op is not None:
        out = S._graph_beta_space_op(
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

class ParaplexFFN(nn.Module):
    def __init__(self, ablation: bool = False) -> None:
        super().__init__()
        self.ablation = bool(ablation)
        self.w1_real = nn.Linear(S.N, S.HIDDEN)
        # Compact PARAMETER storage, dense COMPUTE path.
        # Only live phase weights are Parameters/optimizer state; forward expands them
        # into a transient dense matrix and keeps the fast single GEMM.
        self.w1_imag = nn.Parameter(torch.empty(S.HIDDEN, S.IMAG_IN))
        self.w1_imag_trace = nn.Parameter(torch.zeros(S.HIDDEN))
        self.w1_imag_bias = nn.Parameter(torch.zeros(S.HIDDEN))
        self.w2 = nn.Linear(S.HIDDEN, S.N)
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
        if S.TRIA_CARRY_ENABLED:
            self.gate_selector = tria.GateSelector(S.N_Q_HEADS)
            self.identity_gate = tria.IdentityAnchoredGate()
        else:
            self.gate_selector = None
            self.identity_gate = None
        # Independent gate source (donor-transplant slot -- see PARAPLEX_GATE_PROJ
        # above). None in the default/original design: amp is self-referential,
        # derived from p_real with zero extra parameters.
        if S.PARAPLEX_GATE_PROJ:
            self.gate_proj = nn.Linear(S.N, S.HIDDEN)
        else:
            self.gate_proj = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_linear_fanin(self.w1_real)
        nn.init.normal_(self.w1_imag, mean=0.0, std=fanin_std(S.IMAG_IN))
        init_linear_residual(self.w2)
        # Start exactly on the non-recurrent phase path; trace learns in only if useful.
        nn.init.zeros_(self.w1_imag_trace)
        nn.init.zeros_(self.w1_imag_bias)
        if self.gate_proj is not None:
            init_linear_fanin(self.gate_proj)

    @staticmethod
    def _merge_query_heads(x: torch.Tensor) -> torch.Tensor:
        B, T, G, Dh = x.shape
        if G != S.N_Q_HEADS or Dh != S.HEAD_DIM:
            raise ValueError("bad query-head tensor")
        return x.reshape(B, T, S.N)

    def _dense_imag_weight(self) -> torch.Tensor:
        # Expand head-local compact columns without a persistent dense zero buffer or
        # an HIDDEN*IMAG_IN int64 scatter index.  Those buffers cost ~2 GiB for alt6
        # and DDP tried to broadcast them before every forward.  The head selector is
        # tiny; multiplication materializes only the required dense head-local block,
        # and cat produces the single GEMM weight.
        def expand_head_local(x: torch.Tensor) -> torch.Tensor:
            x = x.view(S.N_Q_HEADS, S.HIDDEN_PER_Q_HEAD, S.HEAD_DIM)
            selector = torch.eye(
                S.N_Q_HEADS, dtype=x.dtype, device=x.device
            ).view(S.N_Q_HEADS, 1, S.N_Q_HEADS, 1)
            return (x.unsqueeze(2) * selector).reshape(S.HIDDEN, S.N)

        if S.PHASE_SECTORS == "head":
            q, k, c, u, d = torch.split(
                self.w1_imag, (S.HEAD_DIM, S.HEAD_DIM, S.HEAD_DIM, S.N, S.HEAD_DIM), dim=1
            )
            return torch.cat((
                expand_head_local(q),
                expand_head_local(k),
                expand_head_local(c),
                u,
                expand_head_local(d),
            ), dim=1)

        q, shared = torch.split(self.w1_imag, (S.HEAD_DIM, 4 * S.N), dim=1)
        return torch.cat((expand_head_local(q), shared), dim=1)

    def _beta_space(self, u: torch.Tensor, q_h: torch.Tensor, k_ctx_h: torch.Tensor,
                     c_h: torch.Tensor, d_h: torch.Tensor) -> torch.Tensor:
        global _cuda_beta_space_active_printed, _cuda_beta_space_fallback_printed

        if S.USE_CUDA_BETA_SPACE and u.is_cuda:
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
                    S.HIDDEN_PER_Q_HEAD, S.HEAD_DIM, S.N_Q_HEADS, S.PHASE_SECTORS == "open")
                if out is not None:
                    if not _cuda_beta_space_active_printed:
                        dtypes = {u.dtype, q_h.dtype, k_ctx_h.dtype, c_h.dtype, d_h.dtype, self.w1_imag.dtype}
                        dtype_note = str(fast_dtype) if len(dtypes) == 1 else f"mixed:{sorted(str(d) for d in dtypes)}"
                        ddp_print(
                            f"[loomformer] CUDA beta_space active  dtype={dtype_note}  "
                            f"sectors={S.PHASE_SECTORS}  shape=B{u.shape[0]}xT{u.shape[1]}xN{S.N}xH{S.HIDDEN}"
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
        if self.ablation or S.ACTIVATION != "pvpowlu" or not u.is_cuda:
            return None
        if not (S.USE_CUDA_BETA_SPACE and S.USE_CUDA_PHASE_SIN and S.USE_CUDA_PVPOWLU):
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
        mode = 1 if S.PHASE_GRAD_MODE == "secant" else 0
        anchor_override = S.checkpoint_anchor_override(self)
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
            S.GRAD_CHECKPOINTING
            and self.training
            and S.TRIA_CARRY_ENABLED
            and S.TRIA_TEMPORAL_ENABLED
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
            reset, anchor, S.HIDDEN_PER_Q_HEAD, S.HEAD_DIM, S.N_Q_HEADS,
            S.PHASE_SECTORS == "open", mode, update_anchor, self.beta_anchor_decay,
            max(S.PHASE_GRAD_FLOOR, 0.0), 1e-4, S.POWLU_M,
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
        identity_alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = u.shape
        if D != S.N:
            raise ValueError(f"u last dim must be {S.N}, got {D}")
        trace = u.new_zeros(B, S.HIDDEN) if phase_trace is None else phase_trace.to(device=u.device, dtype=u.dtype)
        # spec §4: gate lives BETWEEN Wx and +b, never applied to Wx+b as one
        # blob -- see IdentityAnchoredGate's docstring for the exact bug that
        # guards against (gating the bias too, invisible until bias moves away
        # from its zero init during training). p_in=None (layer 1, or tria
        # disabled) makes this an exact no-op identical to self.w1_real(u).
        wx = F.linear(u, self.w1_real.weight, None)
        if self.identity_gate is not None:
            p_real = self.identity_gate(
                wx, self.w1_real.bias, p_in, alpha=identity_alpha)
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
            trace = torch.ones(B, S.HIDDEN, device=u.device, dtype=u.dtype)
        else:
            beta_base = self._beta_space(u, q_h, k_ctx_h, c_h, d_h) + self.w1_imag_bias
            anchor_override = S.checkpoint_anchor_override(self)
            if S.PHASE_GRAD_MODE == "secant" and self.training and anchor_override is None:
                # FP32 RMS is a representative phase radius. A global amin over
                # B*T*H inevitably approaches zero as the model grows and turns
                # the adaptive secant anchor into a permanent zero anchor.
                with torch.no_grad():
                    batch_scale = phase_anchor_scale(beta_base)
                    self.beta_anchor.mul_(self.beta_anchor_decay).add_(
                        batch_scale.to(self.beta_anchor.dtype), alpha=1.0 - self.beta_anchor_decay)
            anchor = self.beta_anchor if anchor_override is None else anchor_override
            s_base = phase_sin(beta_base, anchor if S.PHASE_GRAD_MODE == "secant" else None)
            trace_mat = prev_token_trace(
                s_base, trace if phase_trace is not None else None, phase_reset_mask)
            s = sin_space_combine(s_base, trace_mat * self.w1_imag_trace.view(1, 1, S.HIDDEN))
            p = torch.addcmul(p_real, amp, s)
            # Return s_base, not s: this is exactly what the parallel forward feeds into
            # position t+1, so incremental step() computes the identical function.
            trace = s_base[:, -1, :]
        if S.ACTIVATION == "pvpowlu":
            act_out = pvpowlu_act(p, amp, S.POWLU_M)
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

__all__ = ('_PhaseSinFloor', '_PhaseSinSecant', '_PhaseSinSecantCUDA', '_try_load_cuda_phase_sin', '_PhaseSinFloorCUDA', 'phase_sin', 'phase_anchor_scale', '_try_load_cuda_pvpowlu', '_PvPowluCUDA', 'pvpowlu_act', '_beta_space_fast_dtype', '_try_load_cuda_beta_space', '_try_load_cuda_paraplex', '_BetaSpaceDirect', '_ParaplexFused', 'beta_space_cuda', 'sin_space_combine', 'prev_token_trace', 'ParaplexFFN')
