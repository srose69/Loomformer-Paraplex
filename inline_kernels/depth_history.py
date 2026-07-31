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
void depth_history_append_cuda(
    torch::Tensor state, torch::Tensor k, torch::Tensor v, int64_t slot);
void depth_history_append_pair_cuda(
    torch::Tensor state, torch::Tensor kv, int64_t slot);
std::vector<torch::Tensor> depth_history_forward_cuda(
    torch::Tensor q, torch::Tensor state);
std::vector<torch::Tensor> depth_history_backward_cuda(
    torch::Tensor grad, torch::Tensor q, torch::Tensor state, torch::Tensor lse);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("append", &depth_history_append_cuda);
    m.def("append_pair", &depth_history_append_pair_cuda);
    m.def("forward", &depth_history_forward_cuda);
    m.def("backward", &depth_history_backward_cuda);
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void append_kernel(
    scalar_t* state, const scalar_t* k, const scalar_t* v,
    int64_t batch, int64_t tokens, int64_t heads, int64_t dim,
    int64_t capacity, int64_t slot,
    int64_t s0, int64_t s1, int64_t s2) {
    int64_t inner = heads * dim;
    int64_t work = batch * tokens * inner;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < work; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t token = i / inner, col = i - token * inner;
        int64_t b = token / tokens, t = token - b * tokens;
        int64_t h = col / dim, d = col - h * dim;
        int64_t source = b * s0 + t * s1 + h * s2 + d;
        int64_t base = ((token * capacity + slot) * 2) * inner + col;
        state[base] = k[source];
        state[base + inner] = v[source];
    }
}

template <typename scalar_t>
__global__ void append_pair_kernel(
    scalar_t* state, const scalar_t* kv, int64_t work,
    int64_t capacity, int64_t slot, int64_t paired_inner) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < work; i += (int64_t)blockDim.x * gridDim.x) {
        int64_t token = i / paired_inner, col = i - token * paired_inner;
        state[(token * capacity + slot) * paired_inner + col] = kv[i];
    }
}

template <typename scalar_t>
__global__ void depth_fwd_kernel(
    const scalar_t* q, const scalar_t* state, scalar_t* out, float* lse,
    int64_t bt_count, int64_t capacity, int64_t length,
    int64_t heads, int64_t dim, float inv_sqrt) {
    extern __shared__ float smem[];
    float* score = smem;
    float* reduce = smem + length;
    __shared__ float max_value, sum_value;
    int64_t row = blockIdx.x, bt = row / heads, head = row - bt * heads;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int64_t s = 0; s < length; ++s) {
        float value = 0.0f;
        int64_t base = (((bt * capacity + s) * 2) * heads + head) * dim;
        for (int64_t d = threadIdx.x; d < dim; d += blockDim.x)
            value += (float)q[head * dim + d] * (float)state[base + d];
        for (int off = 16; off; off >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, off);
        if (lane == 0) reduce[warp] = value;
        __syncthreads();
        if (threadIdx.x == 0) {
            value = 0.0f;
            for (int w = 0; w < warps; ++w) value += reduce[w];
            score[s] = value * inv_sqrt;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float m = score[0];
        for (int64_t s = 1; s < length; ++s) m = fmaxf(m, score[s]);
        float sum = 0.0f;
        for (int64_t s = 0; s < length; ++s) {
            score[s] = expf(score[s] - m);
            sum += score[s];
        }
        max_value = m;
        sum_value = sum;
        lse[row] = m + logf(sum);
    }
    __syncthreads();
    int64_t out_base = (bt * heads + head) * dim;
    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0.0f;
        for (int64_t s = 0; s < length; ++s) {
            int64_t vbase = ((((bt * capacity + s) * 2 + 1) * heads + head) * dim);
            value += score[s] * (float)state[vbase + d];
        }
        out[out_base + d] = (scalar_t)(value / sum_value);
    }
}

template <typename scalar_t>
__global__ void depth_fwd_warp(
    const scalar_t* q, const scalar_t* state, scalar_t* out, float* lse,
    int64_t rows, int64_t capacity, int64_t length,
    int64_t heads, int64_t dim, float inv_sqrt) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int64_t row = (int64_t)blockIdx.x * 4 + warp;
    if (row >= rows) return;
    int64_t bt = row / heads, head = row - bt * heads;
    int64_t d0 = (int64_t)lane * 4;
    bool active = d0 < dim;
    float qv[4] = {0, 0, 0, 0}, acc[4] = {0, 0, 0, 0};
    if (active)
        for (int j = 0; j < 4; ++j) qv[j] = (float)q[head * dim + d0 + j];
    float m = -INFINITY, sum = 0.0f;
    for (int64_t s = 0; s < length; ++s) {
        int64_t kbase = (((bt * capacity + s) * 2) * heads + head) * dim;
        int64_t vbase = kbase + heads * dim;
        float score = 0.0f;
        if (active)
            for (int j = 0; j < 4; ++j)
                score += qv[j] * (float)state[kbase + d0 + j];
        for (int off = 16; off; off >>= 1)
            score += __shfl_down_sync(0xffffffffu, score, off);
        score = __shfl_sync(0xffffffffu, score, 0) * inv_sqrt;
        float next_m = fmaxf(m, score);
        float old_scale = expf(m - next_m), weight = expf(score - next_m);
        sum = sum * old_scale + weight;
        if (active)
            for (int j = 0; j < 4; ++j)
                acc[j] = acc[j] * old_scale + weight * (float)state[vbase + d0 + j];
        m = next_m;
    }
    if (active)
        for (int j = 0; j < 4; ++j)
            out[row * dim + d0 + j] = (scalar_t)(acc[j] / sum);
    if (lane == 0) lse[row] = m + logf(sum);
}

template <typename scalar_t>
__global__ void depth_bwd_kernel(
    const scalar_t* grad, const scalar_t* q, const scalar_t* state,
    const float* lse, scalar_t* grad_state, float* grad_q_partial,
    int64_t bt_count, int64_t capacity, int64_t length,
    int64_t heads, int64_t dim, float inv_sqrt) {
    extern __shared__ float smem[];
    float* prob = smem;
    float* dp = smem + length;
    float* reduce = smem + 2 * length;
    __shared__ float correction;
    int64_t row = blockIdx.x, bt = row / heads, head = row - bt * heads;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    int64_t grad_base = (bt * heads + head) * dim;
    for (int64_t s = 0; s < length; ++s) {
        int64_t kbase = (((bt * capacity + s) * 2) * heads + head) * dim;
        int64_t vbase = kbase + heads * dim;
        float score = 0.0f, value = 0.0f;
        for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
            score += (float)q[head * dim + d] * (float)state[kbase + d];
            value += (float)grad[grad_base + d] * (float)state[vbase + d];
        }
        for (int off = 16; off; off >>= 1) {
            score += __shfl_down_sync(0xffffffffu, score, off);
            value += __shfl_down_sync(0xffffffffu, value, off);
        }
        if (lane == 0) { reduce[warp] = score; reduce[warps + warp] = value; }
        __syncthreads();
        if (threadIdx.x == 0) {
            score = 0.0f; value = 0.0f;
            for (int w = 0; w < warps; ++w) {
                score += reduce[w];
                value += reduce[warps + w];
            }
            prob[s] = expf(score * inv_sqrt - lse[row]);
            dp[s] = value;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float value = 0.0f;
        for (int64_t s = 0; s < length; ++s) value += prob[s] * dp[s];
        correction = value;
    }
    __syncthreads();
    for (int64_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float g = (float)grad[grad_base + d], gq = 0.0f;
        float qv = (float)q[head * dim + d];
        for (int64_t s = 0; s < length; ++s) {
            int64_t kbase = (((bt * length + s) * 2) * heads + head) * dim;
            int64_t source_kbase = (((bt * capacity + s) * 2) * heads + head) * dim;
            int64_t vbase = kbase + heads * dim;
            float ds = prob[s] * (dp[s] - correction) * inv_sqrt;
            grad_state[kbase + d] = (scalar_t)(ds * qv);
            grad_state[vbase + d] = (scalar_t)(prob[s] * g);
            gq += ds * (float)state[source_kbase + d];
        }
        grad_q_partial[(head * dim + d) * bt_count + bt] = gq;
    }
}

template <typename scalar_t>
__global__ void depth_bwd_warp(
    const scalar_t* grad, const scalar_t* q, const scalar_t* state,
    const float* lse, scalar_t* grad_state, float* grad_q_partial,
    int64_t rows, int64_t bt_count, int64_t capacity, int64_t length,
    int64_t heads, int64_t dim, float inv_sqrt) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int64_t row = (int64_t)blockIdx.x * 4 + warp;
    if (row >= rows) return;
    int64_t bt = row / heads, head = row - bt * heads;
    int64_t d0 = (int64_t)lane * 4;
    bool active = d0 < dim;
    float qv[4] = {0, 0, 0, 0}, gv[4] = {0, 0, 0, 0};
    if (active)
        for (int j = 0; j < 4; ++j) {
            qv[j] = (float)q[head * dim + d0 + j];
            gv[j] = (float)grad[row * dim + d0 + j];
        }
    float correction = 0.0f;
    for (int64_t s = 0; s < length; ++s) {
        int64_t kbase = (((bt * capacity + s) * 2) * heads + head) * dim;
        int64_t vbase = kbase + heads * dim;
        float score = 0.0f, dp = 0.0f;
        if (active)
            for (int j = 0; j < 4; ++j) {
                score += qv[j] * (float)state[kbase + d0 + j];
                dp += gv[j] * (float)state[vbase + d0 + j];
            }
        for (int off = 16; off; off >>= 1) {
            score += __shfl_down_sync(0xffffffffu, score, off);
            dp += __shfl_down_sync(0xffffffffu, dp, off);
        }
        score = __shfl_sync(0xffffffffu, score, 0) * inv_sqrt;
        dp = __shfl_sync(0xffffffffu, dp, 0);
        correction += expf(score - lse[row]) * dp;
    }
    float gq[4] = {0, 0, 0, 0};
    for (int64_t s = 0; s < length; ++s) {
        int64_t source_k = (((bt * capacity + s) * 2) * heads + head) * dim;
        int64_t source_v = source_k + heads * dim;
        float score = 0.0f, dp = 0.0f;
        if (active)
            for (int j = 0; j < 4; ++j) {
                score += qv[j] * (float)state[source_k + d0 + j];
                dp += gv[j] * (float)state[source_v + d0 + j];
            }
        for (int off = 16; off; off >>= 1) {
            score += __shfl_down_sync(0xffffffffu, score, off);
            dp += __shfl_down_sync(0xffffffffu, dp, off);
        }
        score = __shfl_sync(0xffffffffu, score, 0) * inv_sqrt;
        dp = __shfl_sync(0xffffffffu, dp, 0);
        float p = expf(score - lse[row]);
        float ds = p * (dp - correction) * inv_sqrt;
        if (active) {
            int64_t target_k = (((bt * length + s) * 2) * heads + head) * dim;
            int64_t target_v = target_k + heads * dim;
            for (int j = 0; j < 4; ++j) {
                grad_state[target_k + d0 + j] = (scalar_t)(ds * qv[j]);
                grad_state[target_v + d0 + j] = (scalar_t)(p * gv[j]);
                gq[j] += ds * (float)state[source_k + d0 + j];
            }
        }
    }
    if (active)
        for (int j = 0; j < 4; ++j)
            grad_q_partial[((head * dim + d0 + j) * bt_count) + bt] = gq[j];
}

template <typename scalar_t>
__global__ void reduce_q_kernel(
    const float* partial, scalar_t* out, int64_t rows, int64_t width) {
    extern __shared__ float values[];
    int64_t col = blockIdx.x;
    float sum = 0.0f;
    for (int64_t row = threadIdx.x; row < rows; row += blockDim.x)
        sum += partial[col * rows + row];
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step >>= 1) {
        if (threadIdx.x < step) values[threadIdx.x] += values[threadIdx.x + step];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[col] = (scalar_t)values[0];
}

static void check_state(torch::Tensor state) {
    TORCH_CHECK(state.is_cuda() && state.dim() == 6
                && state.size(3) == 2, "depth_history: bad state");
    TORCH_CHECK(state.stride(5) == 1 && state.stride(4) == state.size(5)
                && state.stride(3) == state.size(4) * state.size(5)
                && state.stride(2) == 2 * state.size(4) * state.size(5),
                "depth_history: bad state strides");
}

void depth_history_append_cuda(
    torch::Tensor state, torch::Tensor k, torch::Tensor v, int64_t slot) {
    check_state(state);
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && k.sizes() == v.sizes()
                && k.scalar_type() == state.scalar_type()
                && k.strides() == v.strides() && k.dim() == 4
                && k.stride(3) == 1,
                "depth_history: bad append tensors");
    int64_t capacity = state.stride(1) / state.stride(2);
    int64_t batch = k.size(0), tokens = k.size(1);
    int64_t heads = k.size(2), dim = k.size(3);
    int64_t inner = heads * dim;
    TORCH_CHECK(slot >= 0 && slot < capacity, "depth_history: append overflow");
    c10::cuda::CUDAGuard guard(state.device());
    int64_t blocks = std::min<int64_t>(65535, (tokens * inner + 255) / 256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        state.scalar_type(), "depth_history_append", ([&] {
            append_kernel<scalar_t><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                state.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
                batch, tokens, heads, dim, capacity, slot,
                k.stride(0), k.stride(1), k.stride(2));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void depth_history_append_pair_cuda(
    torch::Tensor state, torch::Tensor kv, int64_t slot) {
    check_state(state);
    TORCH_CHECK(kv.is_cuda() && kv.is_contiguous() && kv.dim() == 5
                && kv.size(2) == 2 && kv.scalar_type() == state.scalar_type(),
                "depth_history: bad paired append tensor");
    int64_t capacity = state.stride(1) / state.stride(2);
    TORCH_CHECK(slot >= 0 && slot < capacity, "depth_history: append overflow");
    c10::cuda::CUDAGuard guard(state.device());
    int64_t work = kv.numel(), paired_inner = 2 * kv.size(3) * kv.size(4);
    int64_t blocks = std::min<int64_t>(65535, (work + 255) / 256);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        state.scalar_type(), "depth_history_append_pair", ([&] {
            append_pair_kernel<scalar_t><<<blocks, 256, 0,
                at::cuda::getCurrentCUDAStream()>>>(
                state.data_ptr<scalar_t>(), kv.data_ptr<scalar_t>(), work,
                capacity, slot, paired_inner);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> depth_history_forward_cuda(
    torch::Tensor q, torch::Tensor state) {
    check_state(state);
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 2,
                "depth_history: bad query");
    int64_t bt = state.size(0) * state.size(1), capacity = state.stride(1) / state.stride(2);
    int64_t length = state.size(2), heads = state.size(4), dim = state.size(5);
    TORCH_CHECK(length > 0 && q.size(0) == heads && q.size(1) == dim,
                "depth_history: shape mismatch");
    c10::cuda::CUDAGuard guard(state.device());
    auto out = torch::empty({state.size(0), state.size(1), heads, dim}, state.options());
    auto lse = torch::empty({bt, heads}, state.options().dtype(torch::kFloat32));
    bool use_warp = dim % 4 == 0;
    size_t shmem = (length + 8) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        state.scalar_type(), "depth_history_forward", ([&] {
            if (use_warp)
                depth_fwd_warp<scalar_t><<<(bt * heads + 3) / 4, 128, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    q.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(), lse.data_ptr<float>(), bt * heads,
                    capacity, length, heads, dim, 1.0f / sqrtf((float)dim));
            else
                depth_fwd_kernel<scalar_t><<<bt * heads, 256, shmem,
                    at::cuda::getCurrentCUDAStream()>>>(
                    q.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(), lse.data_ptr<float>(), bt,
                    capacity, length, heads, dim, 1.0f / sqrtf((float)dim));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}

std::vector<torch::Tensor> depth_history_backward_cuda(
    torch::Tensor grad, torch::Tensor q, torch::Tensor state, torch::Tensor lse) {
    check_state(state);
    c10::cuda::CUDAGuard guard(state.device());
    auto g = grad.contiguous();
    int64_t bt = state.size(0) * state.size(1), capacity = state.stride(1) / state.stride(2);
    int64_t length = state.size(2), heads = state.size(4), dim = state.size(5);
    auto grad_state = torch::empty(state.sizes(), state.options());
    auto partial = torch::empty({heads * dim, bt}, state.options().dtype(torch::kFloat32));
    auto grad_q = torch::empty({heads, dim}, q.options());
    bool use_warp = dim % 4 == 0;
    size_t shmem = (2 * length + 16) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        state.scalar_type(), "depth_history_backward", ([&] {
            if (use_warp)
                depth_bwd_warp<scalar_t><<<(bt * heads + 3) / 4, 128, 0,
                    at::cuda::getCurrentCUDAStream()>>>(
                    g.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                    lse.data_ptr<float>(), grad_state.data_ptr<scalar_t>(),
                    partial.data_ptr<float>(), bt * heads, bt, capacity, length,
                    heads, dim, 1.0f / sqrtf((float)dim));
            else
                depth_bwd_kernel<scalar_t><<<bt * heads, 256, shmem,
                    at::cuda::getCurrentCUDAStream()>>>(
                    g.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), state.data_ptr<scalar_t>(),
                    lse.data_ptr<float>(), grad_state.data_ptr<scalar_t>(),
                    partial.data_ptr<float>(), bt, capacity, length, heads, dim,
                    1.0f / sqrtf((float)dim));
            reduce_q_kernel<scalar_t><<<heads * dim, 256, 256 * sizeof(float),
                at::cuda::getCurrentCUDAStream()>>>(
                partial.data_ptr<float>(), grad_q.data_ptr<scalar_t>(), bt,
                heads * dim);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_q, grad_state};
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
                "loomformer_inline_depth_history",
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
                f"depth_history CUDA inline build failed; using packed fallback: {error}",
                RuntimeWarning,
            )
    return _module


class _Append(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, k, v):
        slot = state.shape[2]
        capacity = state.stride(1) // state.stride(2)
        _load().append(state, k, v, slot)
        ctx.slot = slot
        return state.as_strided(
            (*state.shape[:2], slot + 1, *state.shape[3:]), state.stride())

    @staticmethod
    def backward(ctx, grad):
        slot = ctx.slot
        return (
            grad[:, :, :slot],
            grad[:, :, slot, 0],
            grad[:, :, slot, 1],
        )


class _AppendPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, kv):
        slot = state.shape[2]
        _load().append_pair(state, kv, slot)
        ctx.slot = slot
        return state.as_strided(
            (*state.shape[:2], slot + 1, *state.shape[3:]), state.stride())

    @staticmethod
    def backward(ctx, grad):
        slot = ctx.slot
        return grad[:, :, :slot], grad[:, :, slot]


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, state):
        out, lse = _load().forward(q, state)
        ctx.save_for_backward(q, state, lse)
        return out

    @staticmethod
    def backward(ctx, grad):
        q, state, lse = ctx.saved_tensors
        return tuple(_load().backward(grad, q, state, lse))


def available(x: torch.Tensor) -> bool:
    return (
        os.environ.get("LOOM_DISABLE_INLINE_KERNELS") != "1"
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and _load() is not None
    )


def depth_history_init(k: torch.Tensor, v: torch.Tensor, capacity: int):
    if not available(k) or v.dtype != k.dtype or k.stride() != v.stride():
        return None
    arena = k.new_empty(*k.shape[:2], capacity, 2, *k.shape[2:])
    state = arena[:, :, :0]
    return _Append.apply(state, k, v)


def depth_history_init_pair(kv: torch.Tensor, capacity: int):
    if not available(kv) or not kv.is_contiguous() or kv.shape[2] != 2:
        return None
    arena = kv.new_empty(*kv.shape[:2], capacity, *kv.shape[2:])
    return _AppendPair.apply(arena[:, :, :0], kv)


def depth_history_append(state: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return _Append.apply(state, k, v)


def depth_history_append_pair(state: torch.Tensor, kv: torch.Tensor):
    return _AppendPair.apply(state, kv)


def depth_attention(q: torch.Tensor, state: torch.Tensor):
    return _Attention.apply(q, state)
