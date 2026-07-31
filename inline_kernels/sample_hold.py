from __future__ import annotations

import os
import threading
import warnings

import torch

_module = None
_failed = False
_lock = threading.Lock()

_CPP = r"""
#include <torch/extension.h>
std::vector<torch::Tensor> sample_hold_forward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor c, torch::Tensor residual,
    torch::Tensor inherited_q, torch::Tensor inherited_k, torch::Tensor inherited_c,
    torch::Tensor ranks);
std::vector<torch::Tensor> sample_hold_backward_cuda(
    torch::Tensor grad_q, torch::Tensor grad_k, torch::Tensor grad_c,
    torch::Tensor grad_residual, torch::Tensor ranks, int64_t selected);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sample_hold_forward_cuda);
    m.def("backward", &sample_hold_backward_cuda);
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sample_hold_fwd(
    const scalar_t* q, const scalar_t* k, const scalar_t* c, const scalar_t* residual,
    const scalar_t* iq, const scalar_t* ik, const scalar_t* ic, const int32_t* ranks,
    scalar_t* oq, scalar_t* ok, scalar_t* oc, scalar_t* ores,
    int64_t tokens, int64_t width) {
    int64_t work = tokens * width;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < work; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t token = i / width, col = i - token * width;
        int32_t rank = ranks[token];
        if (rank >= 0) {
            int64_t src = (int64_t)rank * width + col;
            oq[i] = q[src]; ok[i] = k[src]; oc[i] = c[src]; ores[i] = residual[src];
        } else {
            oq[i] = iq[i]; ok[i] = ik[i]; oc[i] = ic[i]; ores[i] = (scalar_t)0;
        }
    }
}

template <typename scalar_t>
__global__ void sample_hold_bwd(
    const scalar_t* gq, const scalar_t* gk, const scalar_t* gc, const scalar_t* gres,
    const int32_t* ranks, scalar_t* sq, scalar_t* sk, scalar_t* sc, scalar_t* sr,
    scalar_t* iq, scalar_t* ik, scalar_t* ic, int64_t tokens, int64_t width) {
    int64_t work = tokens * width;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < work; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t token = i / width, col = i - token * width;
        int32_t rank = ranks[token];
        if (rank >= 0) {
            int64_t dst = (int64_t)rank * width + col;
            sq[dst] = gq[i]; sk[dst] = gk[i]; sc[dst] = gc[i]; sr[dst] = gres[i];
            iq[i] = (scalar_t)0; ik[i] = (scalar_t)0; ic[i] = (scalar_t)0;
        } else {
            iq[i] = gq[i]; ik[i] = gk[i]; ic[i] = gc[i];
        }
    }
}

static void check(torch::Tensor q, torch::Tensor k, torch::Tensor c,
                  torch::Tensor residual, torch::Tensor iq, torch::Tensor ik,
                  torch::Tensor ic, torch::Tensor ranks) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && c.is_cuda() && residual.is_cuda()
                && iq.is_cuda() && ik.is_cuda() && ic.is_cuda() && ranks.is_cuda(),
                "sample_hold: CUDA tensors required");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == c.sizes(),
                "sample_hold: selected context mismatch");
    TORCH_CHECK(iq.sizes() == ik.sizes() && iq.sizes() == ic.sizes(),
                "sample_hold: inherited context mismatch");
    TORCH_CHECK(q.scalar_type() == iq.scalar_type()
                && q.scalar_type() == residual.scalar_type(),
                "sample_hold: dtype mismatch");
    TORCH_CHECK(ranks.scalar_type() == torch::kInt32 && ranks.is_contiguous(),
                "sample_hold: ranks must be contiguous int32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && c.is_contiguous()
                && residual.is_contiguous() && iq.is_contiguous()
                && ik.is_contiguous() && ic.is_contiguous(),
                "sample_hold: contiguous tensors required");
    TORCH_CHECK(q.numel() / q.size(0) == residual.numel() / residual.size(0),
                "sample_hold: context and residual widths differ");
    TORCH_CHECK(iq.numel() / ranks.numel() == q.numel() / q.size(0),
                "sample_hold: full context width mismatch");
}

std::vector<torch::Tensor> sample_hold_forward_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor c, torch::Tensor residual,
    torch::Tensor iq, torch::Tensor ik, torch::Tensor ic, torch::Tensor ranks) {
    check(q, k, c, residual, iq, ik, ic, ranks);
    c10::cuda::CUDAGuard guard(q.device());
    int64_t tokens = ranks.numel(), width = residual.numel() / q.size(0);
    auto oq = torch::empty_like(iq), ok = torch::empty_like(ik);
    auto oc = torch::empty_like(ic);
    std::vector<int64_t> residual_shape(ranks.sizes().begin(), ranks.sizes().end());
    residual_shape.push_back(width);
    auto ores = torch::empty(residual_shape, residual.options());
    int64_t blocks = std::min<int64_t>(65535, (tokens * width + 255) / 256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "sample_hold_forward", ([&] {
            sample_hold_fwd<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(), iq.data_ptr<scalar_t>(), ik.data_ptr<scalar_t>(),
                ic.data_ptr<scalar_t>(), ranks.data_ptr<int32_t>(), oq.data_ptr<scalar_t>(),
                ok.data_ptr<scalar_t>(), oc.data_ptr<scalar_t>(), ores.data_ptr<scalar_t>(),
                tokens, width);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {oq, ok, oc, ores};
}

std::vector<torch::Tensor> sample_hold_backward_cuda(
    torch::Tensor gq, torch::Tensor gk, torch::Tensor gc, torch::Tensor gres,
    torch::Tensor ranks, int64_t selected) {
    c10::cuda::CUDAGuard guard(gq.device());
    auto q = torch::empty({selected, gq.size(-2), gq.size(-1)}, gq.options());
    auto k = torch::empty_like(q), c = torch::empty_like(q);
    auto residual = torch::empty({selected, gq.size(-2) * gq.size(-1)}, gq.options());
    auto iq = torch::empty_like(gq), ik = torch::empty_like(gk), ic = torch::empty_like(gc);
    int64_t tokens = ranks.numel(), width = gq.numel() / tokens;
    int64_t blocks = std::min<int64_t>(65535, (tokens * width + 255) / 256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        gq.scalar_type(), "sample_hold_backward", ([&] {
            sample_hold_bwd<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                gq.data_ptr<scalar_t>(), gk.data_ptr<scalar_t>(), gc.data_ptr<scalar_t>(),
                gres.data_ptr<scalar_t>(), ranks.data_ptr<int32_t>(), q.data_ptr<scalar_t>(),
                k.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
                iq.data_ptr<scalar_t>(), ik.data_ptr<scalar_t>(), ic.data_ptr<scalar_t>(),
                tokens, width);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q, k, c, residual, iq, ik, ic};
}
"""


def _load():
    global _module, _failed
    if _module is not None or _failed:
        return _module
    with _lock:
        if _module is not None or _failed:
            return _module
        try:
            from torch.utils.cpp_extension import load_inline
            _module = load_inline(
                "loomformer_inline_sample_hold",
                cpp_sources=_CPP,
                cuda_sources=_CUDA,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
                verbose=False,
            )
        except Exception as error:
            _failed = True
            warnings.warn(
                f"sample_hold CUDA inline build failed; using torch fallback: {error}",
                RuntimeWarning,
            )
    return _module


class _SampleHold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, c, residual, iq, ik, ic, ranks):
        out = _load().forward(q, k, c, residual, iq, ik, ic, ranks)
        ctx.save_for_backward(ranks)
        ctx.selected = q.shape[0]
        return tuple(out)

    @staticmethod
    def backward(ctx, gq, gk, gc, gres):
        (ranks,) = ctx.saved_tensors
        return (*_load().backward(
            gq.contiguous(), gk.contiguous(), gc.contiguous(), gres.contiguous(),
            ranks, ctx.selected), None)


def sample_hold(q, k, c, residual, inherited, ranks):
    if q.shape[0] == 0:
        dependency = (q.sum() + k.sum() + c.sum() + residual.sum()) * 0
        return (
            inherited[0] + dependency,
            inherited[1] + dependency,
            inherited[2] + dependency,
            residual.new_zeros(*ranks.shape, residual.shape[-1]) + dependency,
        )
    module = (
        _load()
        if q.is_cuda and os.environ.get("LOOM_DISABLE_INLINE_KERNELS") != "1"
        else None
    )
    if (
        module is not None
        and all(x.is_contiguous() for x in (q, k, c, residual, *inherited))
        and ranks.dtype == torch.int32
        and ranks.is_contiguous()
    ):
        return _SampleHold.apply(q, k, c, residual, *inherited, ranks)
    selected = ranks.ge(0)
    safe = ranks.reshape(-1).clamp_min(0).to(torch.int64)
    shape = (*ranks.shape, q.shape[-2], q.shape[-1])
    own_q = q.index_select(0, safe).view(shape)
    own_k = k.index_select(0, safe).view(shape)
    own_c = c.index_select(0, safe).view(shape)
    own_r = residual.index_select(0, safe).view(*ranks.shape, residual.shape[-1])
    choose = selected.view(*ranks.shape, 1, 1)
    return (
        torch.where(choose, own_q, inherited[0]),
        torch.where(choose, own_k, inherited[1]),
        torch.where(choose, own_c, inherited[2]),
        torch.where(selected.unsqueeze(-1), own_r, torch.zeros_like(own_r)),
    )
