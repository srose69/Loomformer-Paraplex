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

std::vector<torch::Tensor> capped_residual_forward_cuda(
    torch::Tensor a, torch::Tensor b, double cap, double eps);
std::vector<torch::Tensor> capped_residual_backward_cuda(
    torch::Tensor grad, torch::Tensor a, torch::Tensor b, torch::Tensor stats);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &capped_residual_forward_cuda);
    m.def("backward", &capped_residual_backward_cuda);
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void capped_residual_fwd(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    float* __restrict__ stats,
    int64_t rows, int64_t width, float cap, float eps) {
    __shared__ float warp_a[8], warp_b[8], scale_a, scale_b, inv_a, inv_b;
    float sa = 0.0f, sb = 0.0f;
    const int64_t base = (int64_t)blockIdx.x * width;
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
        float av = (float)a[base + i], bv = (float)b[base + i];
        sa += av * av;
        sb += bv * bv;
    }
    for (int d = 16; d; d >>= 1) {
        sa += __shfl_down_sync(0xffffffffu, sa, d);
        sb += __shfl_down_sync(0xffffffffu, sb, d);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { warp_a[warp] = sa; warp_b[warp] = sb; }
    __syncthreads();
    if (warp == 0) {
        const int warps = blockDim.x >> 5;
        sa = lane < warps ? warp_a[lane] : 0.0f;
        sb = lane < warps ? warp_b[lane] : 0.0f;
        for (int d = 16; d; d >>= 1) {
            sa += __shfl_down_sync(0xffffffffu, sa, d);
            sb += __shfl_down_sync(0xffffffffu, sb, d);
        }
        if (lane == 0) {
            float ma = sa / (float)width + eps;
            float mb = sb / (float)width + eps;
            inv_a = 1.0f / ma;
            inv_b = 1.0f / mb;
            scale_a = fminf(1.0f, cap * rsqrtf(ma));
            scale_b = fminf(1.0f, cap * rsqrtf(mb));
            stats[(int64_t)blockIdx.x * 4 + 0] = scale_a;
            stats[(int64_t)blockIdx.x * 4 + 1] = scale_b;
            stats[(int64_t)blockIdx.x * 4 + 2] = inv_a;
            stats[(int64_t)blockIdx.x * 4 + 3] = inv_b;
        }
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x)
        out[base + i] = (scalar_t)((float)a[base + i] * scale_a
                                  + (float)b[base + i] * scale_b);
}

template <typename scalar_t>
__global__ void capped_residual_bwd(
    const scalar_t* __restrict__ grad,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const float* __restrict__ stats,
    scalar_t* __restrict__ grad_a,
    scalar_t* __restrict__ grad_b,
    int64_t rows, int64_t width) {
    __shared__ float warp_a[8], warp_b[8], dot_a, dot_b;
    float da = 0.0f, db = 0.0f;
    const int64_t base = (int64_t)blockIdx.x * width;
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
        float g = (float)grad[base + i];
        da += g * (float)a[base + i];
        db += g * (float)b[base + i];
    }
    for (int d = 16; d; d >>= 1) {
        da += __shfl_down_sync(0xffffffffu, da, d);
        db += __shfl_down_sync(0xffffffffu, db, d);
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) { warp_a[warp] = da; warp_b[warp] = db; }
    __syncthreads();
    if (warp == 0) {
        const int warps = blockDim.x >> 5;
        da = lane < warps ? warp_a[lane] : 0.0f;
        db = lane < warps ? warp_b[lane] : 0.0f;
        for (int d = 16; d; d >>= 1) {
            da += __shfl_down_sync(0xffffffffu, da, d);
            db += __shfl_down_sync(0xffffffffu, db, d);
        }
        if (lane == 0) { dot_a = da; dot_b = db; }
    }
    __syncthreads();
    const float* st = stats + (int64_t)blockIdx.x * 4;
    float scale_a = st[0], scale_b = st[1];
    float corr_a = scale_a < 1.0f ? dot_a * st[2] / (float)width : 0.0f;
    float corr_b = scale_b < 1.0f ? dot_b * st[3] / (float)width : 0.0f;
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
        float g = (float)grad[base + i];
        grad_a[base + i] = (scalar_t)(scale_a * (g - (float)a[base + i] * corr_a));
        grad_b[base + i] = (scalar_t)(scale_b * (g - (float)b[base + i] * corr_b));
    }
}

static void check(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "capped_residual: CUDA tensors required");
    TORCH_CHECK(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type(),
                "capped_residual: input mismatch");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && a.dim() >= 1,
                "capped_residual: contiguous tensors required");
}

std::vector<torch::Tensor> capped_residual_forward_cuda(
    torch::Tensor a, torch::Tensor b, double cap, double eps) {
    check(a, b);
    c10::cuda::CUDAGuard guard(a.device());
    int64_t width = a.size(-1), rows = a.numel() / width;
    auto out = torch::empty_like(a);
    auto stats = torch::empty({rows, 4}, a.options().dtype(torch::kFloat32));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        a.scalar_type(), "capped_residual_forward", ([&] {
            capped_residual_fwd<scalar_t><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                stats.data_ptr<float>(), rows, width, (float)cap, (float)eps);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, stats};
}

std::vector<torch::Tensor> capped_residual_backward_cuda(
    torch::Tensor grad, torch::Tensor a, torch::Tensor b, torch::Tensor stats) {
    check(a, b);
    TORCH_CHECK(grad.is_cuda() && grad.sizes() == a.sizes(),
                "capped_residual: bad output gradient");
    c10::cuda::CUDAGuard guard(a.device());
    auto g = grad.contiguous();
    int64_t width = a.size(-1), rows = a.numel() / width;
    auto ga = torch::empty_like(a), gb = torch::empty_like(b);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        a.scalar_type(), "capped_residual_backward", ([&] {
            capped_residual_bwd<scalar_t><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                g.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
                stats.data_ptr<float>(), ga.data_ptr<scalar_t>(), gb.data_ptr<scalar_t>(),
                rows, width);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {ga, gb};
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
                "loomformer_inline_capped_residual",
                cpp_sources=_CPP,
                cuda_sources=_CUDA,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                with_cuda=True,
                verbose=False,
            )
        except Exception as error:
            _failed = True
            warnings.warn(
                f"capped_residual CUDA inline build failed; using torch fallback: {error}",
                RuntimeWarning,
            )
    return _module


class _CappedResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, cap, eps):
        module = _load()
        out, stats = module.forward(a.contiguous(), b.contiguous(), cap, eps)
        ctx.save_for_backward(a, b, stats)
        return out

    @staticmethod
    def backward(ctx, grad):
        a, b, stats = ctx.saved_tensors
        ga, gb = _load().backward(grad, a, b, stats)
        return ga, gb, None, None


def capped_residual(a: torch.Tensor, b: torch.Tensor, cap: float, eps: float = 1e-6):
    if (
        os.environ.get("LOOM_DISABLE_INLINE_KERNELS") != "1"
        and
        a.is_cuda
        and b.is_cuda
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and _load() is not None
    ):
        return _CappedResidual.apply(a, b, float(cap), float(eps))
    return capped_residual_reference(a, b, cap, eps)


def capped_residual_reference(a, b, cap: float, eps: float = 1e-6):
    def one(x):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        rms = torch.sqrt(work.square().mean(dim=-1, keepdim=True) + eps)
        return x * (float(cap) / rms).clamp(max=1.0).to(x.dtype)
    return one(a) + one(b)
