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
std::vector<torch::Tensor> fixed_rms_forward_cuda(
    torch::Tensor x, torch::Tensor targets, int64_t group_span, double eps);
torch::Tensor fixed_rms_backward_cuda(
    torch::Tensor grad, torch::Tensor x, torch::Tensor stats);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fixed_rms_forward_cuda);
    m.def("backward", &fixed_rms_backward_cuda);
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fixed_rms_fwd(
    const scalar_t* x, const float* targets, scalar_t* out, float* stats,
    int64_t rows, int64_t width, int64_t groups, int64_t group_span, float eps) {
    __shared__ float warp_sum[8], scale, inv_mean;
    int64_t base = (int64_t)blockIdx.x * width;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x) {
        float v = (float)x[base + i];
        sum += v * v;
    }
    for (int d = 16; d; d >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, d);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int warps = blockDim.x >> 5;
        sum = lane < warps ? warp_sum[lane] : 0.0f;
        for (int d = 16; d; d >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (lane == 0) {
            inv_mean = 1.0f / (sum / (float)width + eps);
            scale = targets[((int64_t)blockIdx.x / group_span) % groups] * sqrtf(inv_mean);
            stats[(int64_t)blockIdx.x * 2] = scale;
            stats[(int64_t)blockIdx.x * 2 + 1] = inv_mean;
        }
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x)
        out[base + i] = (scalar_t)((float)x[base + i] * scale);
}

template <typename scalar_t>
__global__ void fixed_rms_bwd(
    const scalar_t* grad, const scalar_t* x, const float* stats, scalar_t* grad_x,
    int64_t rows, int64_t width) {
    __shared__ float warp_sum[8], dot;
    int64_t base = (int64_t)blockIdx.x * width;
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x)
        sum += (float)grad[base + i] * (float)x[base + i];
    for (int d = 16; d; d >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, d);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        int warps = blockDim.x >> 5;
        sum = lane < warps ? warp_sum[lane] : 0.0f;
        for (int d = 16; d; d >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (lane == 0) dot = sum;
    }
    __syncthreads();
    float scale = stats[(int64_t)blockIdx.x * 2];
    float corr = dot * stats[(int64_t)blockIdx.x * 2 + 1] / (float)width;
    for (int64_t i = threadIdx.x; i < width; i += blockDim.x)
        grad_x[base + i] = (scalar_t)(scale * ((float)grad[base + i]
                                              - (float)x[base + i] * corr));
}

std::vector<torch::Tensor> fixed_rms_forward_cuda(
    torch::Tensor x, torch::Tensor targets, int64_t group_span, double eps) {
    TORCH_CHECK(x.is_cuda() && targets.is_cuda() && x.is_contiguous()
                && targets.is_contiguous(), "fixed_rms: contiguous CUDA tensors required");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32 && targets.dim() == 1,
                "fixed_rms: targets must be FP32 vector");
    c10::cuda::CUDAGuard guard(x.device());
    int64_t width = x.size(-1), rows = x.numel() / width;
    TORCH_CHECK(group_span > 0 && rows % (targets.numel() * group_span) == 0,
                "fixed_rms: bad target grouping");
    auto out = torch::empty_like(x);
    auto stats = torch::empty({rows, 2}, x.options().dtype(torch::kFloat32));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fixed_rms_forward", ([&] {
            fixed_rms_fwd<scalar_t><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                x.data_ptr<scalar_t>(), targets.data_ptr<float>(), out.data_ptr<scalar_t>(),
                stats.data_ptr<float>(), rows, width, targets.numel(), group_span, (float)eps);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, stats};
}

torch::Tensor fixed_rms_backward_cuda(
    torch::Tensor grad, torch::Tensor x, torch::Tensor stats) {
    c10::cuda::CUDAGuard guard(x.device());
    auto g = grad.contiguous();
    int64_t width = x.size(-1), rows = x.numel() / width;
    auto grad_x = torch::empty_like(x);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fixed_rms_backward", ([&] {
            fixed_rms_bwd<scalar_t><<<rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                g.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), stats.data_ptr<float>(),
                grad_x.data_ptr<scalar_t>(), rows, width);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_x;
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
                "loomformer_inline_fixed_rms",
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
                f"fixed_rms CUDA inline build failed; using torch fallback: {error}",
                RuntimeWarning,
            )
    return _module


class _FixedRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, targets, eps):
        out, stats = _load().forward(x, targets, x.shape[-2], eps)
        ctx.save_for_backward(x, stats)
        return out

    @staticmethod
    def backward(ctx, grad):
        x, stats = ctx.saved_tensors
        return _load().backward(grad, x, stats), None, None


def fixed_rms(x: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    if (
        os.environ.get("LOOM_DISABLE_INLINE_KERNELS") != "1"
        and
        x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and targets.is_cuda
        and targets.dtype == torch.float32
        and targets.is_contiguous()
        and _load() is not None
    ):
        return _FixedRMS.apply(x, targets, float(eps))
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    target = targets.to(x.device).view(
        *((1,) * (x.dim() - 3)), targets.numel(), 1, 1)
    scale = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps) * target
    return x * scale.to(x.dtype)
