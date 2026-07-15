// kernels/phase_sin/phase_sin_kernel.cuh -- pure CUDA device kernels for
// phase_sin (the O(1)-memory 'no cache needed, recompute cos(x) from beta
// alone' backward). No torch includes: safe standalone nvcc/PTX target.
//
// Templated on scalar_t (float/at::Half/at::BFloat16) -- matches every
// other kernel group in this tree. sinf/cosf/rsqrtf have no native bf16/fp16
// device intrinsics, so every element is read as (float), all math (rsqrt,
// sin, cos, the dx/dbeta derivative) happens in fp32, and only the final
// store casts back down to scalar_t -- same accumulate-in-fp32 idiom
// pvpowlu/tria/etc. already use, not a new pattern.
#pragma once

#define PI_HALF 1.5707963267948966f

template <typename scalar_t>
__global__ void phase_sin_fwd_kernel(const scalar_t* __restrict__ beta,
                                      scalar_t* __restrict__ out,
                                      int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = (float)beta[i];
    float inv = rsqrtf(1.0f + b * b);
    float x = PI_HALF * b * inv;
    out[i] = (scalar_t)sinf(x);
}

template <typename scalar_t>
__global__ void phase_sin_bwd_kernel(const scalar_t* __restrict__ beta,
                                      const scalar_t* __restrict__ grad_out,
                                      scalar_t* __restrict__ grad_in,
                                      float eps,
                                      int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = (float)beta[i];
    float one_plus_b2 = 1.0f + b * b;
    float inv = rsqrtf(one_plus_b2);
    float x = PI_HALF * b * inv;
    // Пересчитываем cosf и dx/dbeta ЗАНОВО -- ничего не сохранялось из forward,
    // кроме самого beta (единственный тензор, который реально нужен).
    float cosx = cosf(x);
    float dx_dbeta = PI_HALF * inv * inv * inv;  // (1+b^2)^(-3/2) = inv^3
    float grad_scale = fmaxf(cosx, eps) * dx_dbeta;
    grad_in[i] = (scalar_t)((float)grad_out[i] * grad_scale);
}

// phase_grad_mode=="secant" -- see loomformer.py's _PhaseSinSecant (PyTorch
// reference) and README.md's Paraplex section for the full explanation.
// Same recompute-from-beta-alone structure as phase_sin_bwd_kernel above --
// only the grad_scale formula changes. anchor/s_anchor are PER-LAYER
// SCALARS (one sin(bound_phase(anchor)) computed once on the Python side,
// not per-element), passed as plain floats exactly like eps above -- no
// reason to recompute the same value in every thread.
template <typename scalar_t>
__global__ void phase_sin_secant_bwd_kernel(const scalar_t* __restrict__ beta,
                                             const scalar_t* __restrict__ grad_out,
                                             scalar_t* __restrict__ grad_in,
                                             float anchor,
                                             float s_anchor,
                                             float near_eps,
                                             int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float b = (float)beta[i];
    float one_plus_b2 = 1.0f + b * b;
    float inv = rsqrtf(one_plus_b2);
    float x = PI_HALF * b * inv;
    float s = sinf(x);
    float denom = b - anchor;
    float grad_scale;
    if (fabsf(denom) < near_eps) {
        // beta essentially at the anchor -- secant's 0/0 limit is the true
        // local derivative, same fallback as the PyTorch path.
        float cosx = cosf(x);
        float dx_dbeta = PI_HALF * inv * inv * inv;
        grad_scale = cosx * dx_dbeta;
    } else {
        grad_scale = (s - s_anchor) / denom;
    }
    grad_in[i] = (scalar_t)((float)grad_out[i] * grad_scale);
}

