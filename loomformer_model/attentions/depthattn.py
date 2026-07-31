from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from inline_kernels import depth_attention, fixed_rms as fixed_rms_inline
from loomformer_runtime.distributed import ddp_print
from .. import state as S
from ..primitives import fanin_std, fixed_rms, init_linear_residual, residual_std

def _depth_attn_online_tensor_pytorch(
    q: torch.Tensor, hist_k: torch.Tensor, hist_v: torch.Tensor
) -> torch.Tensor:
    out_dtype = q.dtype
    q = q.float()
    hist_k = hist_k.float()
    hist_v = hist_v.float()
    sqrt_d = math.sqrt(S.HEAD_DIM)
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
        d, lse = ext.depth_attn_stacked_forward(
            q.contiguous(), hist_k.contiguous(), hist_v.contiguous())
        ctx.save_for_backward(q, hist_k, hist_v, lse)
        return d

    @staticmethod
    def backward(ctx, grad_d):
        ext = _try_load_cuda_depth_attn()
        if ext is None:
            raise RuntimeError("CUDA depth_attn module is unavailable")
        q, hist_k, hist_v, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = ext.depth_attn_stacked_backward(
            grad_d.contiguous(), q, hist_k, hist_v, lse)
        return grad_q, grad_k, grad_v


def depth_attn_online_cuda(q: torch.Tensor, hist_k: torch.Tensor, hist_v: torch.Tensor) -> Optional[torch.Tensor]:
    if not (S.USE_CUDA_DEPTH_ATTN and q.is_cuda and hist_k.is_cuda and hist_v.is_cuda):
        return None
    if hist_k.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return None
    if not (q.dtype == hist_k.dtype and hist_v.dtype == hist_k.dtype):
        return None
    ext = _try_load_cuda_depth_attn()
    if ext is None:
        return None
    if S.GRAPH_MODE_ENABLED and S._graph_depth_attn_op is not None:
        d, _w = S._graph_depth_attn_op(q, hist_k, hist_v)
        return d
    return _DepthAttnOnlineFused.apply(q, hist_k, hist_v)

class DepthAttn(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        n_sub = 2 * S.LAYERS
        self.kv_weight = nn.Parameter(torch.empty(2 * S.N, S.N))
        if S.DEPTH_ATTN_READOUT == "per-sublayer":
            self.w_o_weight = nn.Parameter(torch.empty(n_sub, S.N, S.N))
            self.w_o = None
        else:
            self.register_parameter("w_o_weight", None)
            self.w_o = nn.Linear(S.N, S.N, bias=False)
        self.q_params = nn.Parameter(torch.empty(n_sub, S.N_Q_HEADS, S.HEAD_DIM))
        v_target = S.DEEPNORM_BETA if S.RESIDUAL_INIT == "beta" else 1.0
        self.register_buffer(
            "_kv_rms_targets",
            torch.tensor(
                (S.FANIN_GAIN, S.FANIN_GAIN * v_target),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.kv_weight[:S.N], mean=0.0, std=fanin_std(S.N))
        nn.init.normal_(self.kv_weight[S.N:], mean=0.0, std=residual_std(S.N))
        if self.w_o_weight is not None:
            nn.init.normal_(self.w_o_weight, mean=0.0, std=residual_std(S.N))
        else:
            init_linear_residual(self.w_o)
        nn.init.normal_(self.q_params, mean=0.0, std=fanin_std(S.HEAD_DIM))

    def project(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.project_paired(h).unbind(dim=2)

    def project_paired(self, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        kv = F.linear(h, self.kv_weight)
        kv = kv.view(B, T, 2, S.N_Q_HEADS, S.HEAD_DIM)
        if S.DEPTH_ATTN_QKV_RMS:
            kv = fixed_rms_inline(kv, self._kv_rms_targets)
        return kv

    def normalized_queries(self) -> torch.Tensor:
        if S.DEPTH_ATTN_QKV_RMS:
            return fixed_rms(self.q_params, fanin_std(S.HEAD_DIM))
        return self.q_params

    def forward(
        self,
        sub_idx: int,
        hist_k,
        hist_v,
        q_row: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_row = (
            self.normalized_queries()[sub_idx]
            if q_row is None
            else q_row
        )
        if hist_v is None:
            if q_row.dtype != hist_k.dtype:
                q_row = q_row.to(hist_k.dtype)
            d = depth_attention(q_row, hist_k)
            B, T = hist_k.shape[:2]
        else:
            if q_row.dtype != hist_k.dtype:
                q_row = q_row.to(hist_k.dtype)
            q = q_row.view(1, 1, S.N_Q_HEADS, S.HEAD_DIM)
            d = depth_attn_online_cuda(q_row, hist_k, hist_v)
            if d is None:
                d = _depth_attn_online_tensor_pytorch(q, hist_k, hist_v)
            B, T = hist_k.shape[:2]
        d_flat = d.reshape(B, T, S.N)
        skip = (
            F.linear(d_flat, self.w_o_weight[sub_idx])
            if self.w_o_weight is not None
            else self.w_o(d_flat)
        )
        return skip, d

__all__ = ('_depth_attn_online_tensor_pytorch', '_try_load_cuda_depth_attn', '_DepthAttnOnlineFused', 'depth_attn_online_cuda', 'DepthAttn')
