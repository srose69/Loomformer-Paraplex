#pragma once
#include <math.h>
#include <stdint.h>

template <typename scalar_t>
__device__ __forceinline__ float pp_load(const scalar_t* p) { return (float)(*p); }

template <typename scalar_t>
__device__ __forceinline__ void pp_store(scalar_t* p, float v) { *p = (scalar_t)v; }

__device__ __forceinline__ float pp_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ __forceinline__ float pp_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float pp_phase(float beta) {
    constexpr float c = 1.57079632679489661923f;
    return sinf(c * beta * rsqrtf(1.0f + beta * beta));
}

__device__ __forceinline__ float pp_phase_grad(
    float beta, float anchor, int mode, float floor, float near_eps) {
    constexpr float c = 1.57079632679489661923f;
    const float inv = rsqrtf(1.0f + beta * beta);
    const float x = c * beta * inv;
    const float local = cosf(x) * c * inv * inv * inv;
    if (mode == 0) return fmaxf(cosf(x), floor) * c * inv * inv * inv;
    const float d = beta - anchor;
    if (fabsf(d) < near_eps) return local;
    return (sinf(x) - pp_phase(anchor)) / d;
}

__device__ __forceinline__ float pp_gate(float amp, float m, float& dlog) {
    const float safe = fmaxf(amp, 1.0e-12f);
    const float root = sqrtf(safe);
    const float denom = root + 1.0f;
    const float exponent = m / denom;
    const float sig = pp_sigmoid(amp);
    const float gate = powf(safe, exponent) * sig;
    const float exponent_prime = -m / (2.0f * root * denom * denom);
    dlog = exponent_prime * logf(safe) + exponent / safe + (1.0f - sig);
    return gate;
}

template <typename scalar_t>
__device__ __forceinline__ float pp_beta(
    const scalar_t* beta_linear, const float* bias, int64_t idx, int64_t h) {
    return pp_load(beta_linear + idx) + bias[h];
}

template <typename scalar_t>
__device__ __forceinline__ float pp_prev(
    const scalar_t* beta_linear, const float* bias, const scalar_t* trace,
    const bool* reset, int64_t b, int64_t t, int64_t h, int64_t T, int64_t H) {
    if (reset != nullptr && reset[b * T + t]) return 0.0f;
    if (t == 0) return pp_load(trace + b * H + h);
    return pp_phase(pp_beta(beta_linear, bias, ((b * T + t - 1) * H + h), h));
}

struct ParaplexTerms {
    float d_raw;
    float grad_r;
};

template <typename scalar_t, typename grad_t>
__device__ __forceinline__ ParaplexTerms pp_backward_terms(
    const grad_t* grad_act, const grad_t* grad_s,
    const scalar_t* p_real, const scalar_t* beta_linear,
    const float* bias, const scalar_t* trace, const float* trace_w,
    const bool* reset, int64_t b, int64_t t, int64_t h,
    int64_t T, int64_t H, float m) {
    const int64_t idx = (b * T + t) * H + h;
    const float r = pp_load(p_real + idx);
    const float beta = pp_beta(beta_linear, bias, idx, h);
    const float s0 = pp_phase(beta);
    const float prev = pp_prev(beta_linear, bias, trace, reset, b, t, h, T, H);
    const float raw = s0 + prev * trace_w[h];
    const float inv = rsqrtf(1.0f + raw * raw);
    const float s = raw * inv;
    const float amp = pp_softplus(r);
    const float p = r + amp * s;
    float dlog;
    const float gate = pp_gate(amp, m, dlog);
    const float ga = (float)grad_act[idx];
    const float gs = (float)grad_s[idx];
    const float d_p = ga * gate;
    const float d_amp = ga * p * gate * dlog + d_p * s;
    const float d_s = gs + d_p * amp;
    ParaplexTerms out;
    out.d_raw = d_s * inv * inv * inv;
    out.grad_r = d_p + d_amp * pp_sigmoid(r);
    return out;
}

template <typename scalar_t>
__global__ void paraplex_forward_kernel(
    const scalar_t* __restrict__ p_real,
    const scalar_t* __restrict__ beta_linear,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ trace,
    const float* __restrict__ trace_w,
    const bool* __restrict__ reset,
    scalar_t* __restrict__ act,
    scalar_t* __restrict__ s_out,
    scalar_t* __restrict__ next_trace,
    float m, int64_t B, int64_t T, int64_t H) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t n = B * T * H;
    if (idx >= n) return;
    const int64_t h = idx % H;
    const int64_t bt = idx / H;
    const int64_t t = bt % T;
    const int64_t b = bt / T;
    const float r = pp_load(p_real + idx);
    const float beta = pp_beta(beta_linear, bias, idx, h);
    const float s0 = pp_phase(beta);
    const float prev = pp_prev(beta_linear, bias, trace, reset, b, t, h, T, H);
    const float raw = s0 + prev * trace_w[h];
    const float s = raw * rsqrtf(1.0f + raw * raw);
    const float amp = pp_softplus(r);
    const float p = r + amp * s;
    float dlog;
    const float gate = pp_gate(amp, m, dlog);
    pp_store(s_out + idx, s);
    pp_store(act + idx, p * gate);
    if (t == T - 1) pp_store(next_trace + b * H + h, s0);
}

template <typename scalar_t, typename grad_t>
__device__ __forceinline__ float pp_grad_beta_at(
    const grad_t* grad_act, const grad_t* grad_s, const grad_t* grad_next,
    const scalar_t* p_real, const scalar_t* beta_linear,
    const float* bias, const scalar_t* trace, const float* trace_w,
    const bool* reset, const float* anchor, int mode, float floor,
    float near_eps, float m, int64_t b, int64_t t, int64_t h,
    int64_t T, int64_t H) {
    const ParaplexTerms here = pp_backward_terms(
        grad_act, grad_s, p_real, beta_linear, bias, trace, trace_w,
        reset, b, t, h, T, H, m);
    float grad_s0 = here.d_raw;
    if (t == T - 1) grad_s0 += (float)grad_next[b * H + h];
    if (t + 1 < T && (reset == nullptr || !reset[b * T + t + 1])) {
        const ParaplexTerms next = pp_backward_terms(
            grad_act, grad_s, p_real, beta_linear, bias, trace, trace_w,
            reset, b, t + 1, h, T, H, m);
        grad_s0 += next.d_raw * trace_w[h];
    }
    const int64_t idx = (b * T + t) * H + h;
    const float beta = pp_beta(beta_linear, bias, idx, h);
    return grad_s0 * pp_phase_grad(beta, anchor[0], mode, floor, near_eps);
}

template <typename scalar_t, typename grad_t>
__global__ void paraplex_backward_kernel(
    const grad_t* __restrict__ grad_act,
    const grad_t* __restrict__ grad_s,
    const grad_t* __restrict__ grad_next,
    const scalar_t* __restrict__ p_real,
    const scalar_t* __restrict__ beta_linear,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ trace,
    const float* __restrict__ trace_w,
    const bool* __restrict__ reset,
    const float* __restrict__ anchor,
    scalar_t* __restrict__ grad_p_real,
    scalar_t* __restrict__ grad_beta,
    scalar_t* __restrict__ grad_trace,
    int mode, float floor, float near_eps, float m,
    int64_t B, int64_t T, int64_t H) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t n = B * T * H;
    if (idx >= n) return;
    const int64_t h = idx % H;
    const int64_t bt = idx / H;
    const int64_t t = bt % T;
    const int64_t b = bt / T;
    const ParaplexTerms here = pp_backward_terms(
        grad_act, grad_s, p_real, beta_linear, bias, trace, trace_w,
        reset, b, t, h, T, H, m);
    pp_store(grad_p_real + idx, here.grad_r);
    pp_store(grad_beta + idx, pp_grad_beta_at(
        grad_act, grad_s, grad_next, p_real, beta_linear, bias, trace,
        trace_w, reset, anchor, mode, floor, near_eps, m, b, t, h, T, H));
    if (t == 0) {
        const float g = (reset != nullptr && reset[b * T]) ? 0.0f : here.d_raw * trace_w[h];
        pp_store(grad_trace + b * H + h, g);
    }
}

__device__ __forceinline__ float pp_warp_sum(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
    return x;
}

__device__ __forceinline__ float pp_block_sum(float x, float* shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    x = pp_warp_sum(x);
    if (lane == 0) shared[warp] = x;
    __syncthreads();
    x = threadIdx.x < (blockDim.x + 31) / 32 ? shared[lane] : 0.0f;
    if (warp == 0) x = pp_warp_sum(x);
    return x;
}

template <typename scalar_t>
__global__ void paraplex_anchor_sum_kernel(
    const scalar_t* __restrict__ beta_linear, const float* __restrict__ bias,
    float* __restrict__ sum, int64_t n, int64_t H) {
    extern __shared__ float shared[];
    float local = 0.0f;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < n; idx += (int64_t)blockDim.x * gridDim.x) {
        const float beta = pp_beta(beta_linear, bias, idx, idx % H);
        local = fmaf(beta, beta, local);
    }
    local = pp_block_sum(local, shared);
    if (threadIdx.x == 0) atomicAdd(sum, local);
}

__global__ void paraplex_anchor_update_kernel(
    const float* __restrict__ anchor, float* __restrict__ snapshot,
    const float* __restrict__ sum, int64_t n, float decay, bool update) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float value = anchor[0];
    if (update) {
        const float scale = fmaxf(sqrtf(sum[0] / (float)n), 1.0e-4f);
        value = decay * value + (1.0f - decay) * scale;
    }
    snapshot[0] = value;
}

template <typename scalar_t, typename grad_t>
__global__ void paraplex_reduce_kernel(
    const grad_t* __restrict__ grad_act,
    const grad_t* __restrict__ grad_s,
    const grad_t* __restrict__ grad_next,
    const scalar_t* __restrict__ p_real,
    const scalar_t* __restrict__ beta_linear,
    const float* __restrict__ bias,
    const scalar_t* __restrict__ trace,
    const float* __restrict__ trace_w,
    const bool* __restrict__ reset,
    const float* __restrict__ anchor,
    float* __restrict__ grad_bias,
    float* __restrict__ grad_trace_w,
    int mode, float floor, float near_eps, float m,
    int64_t B, int64_t T, int64_t H) {
    const int64_t h = blockIdx.x;
    extern __shared__ float shared[];
    float* sb = shared;
    float* sw = shared + (blockDim.x + 31) / 32;
    float gb = 0.0f, gw = 0.0f;
    const int64_t M = B * T;
    for (int64_t row = threadIdx.x; row < M; row += blockDim.x) {
        const int64_t b = row / T;
        const int64_t t = row - b * T;
        const int64_t idx = row * H + h;
        gb += pp_grad_beta_at(
            grad_act, grad_s, grad_next, p_real, beta_linear, bias, trace,
            trace_w, reset, anchor, mode, floor, near_eps, m, b, t, h, T, H);
        const ParaplexTerms terms = pp_backward_terms(
            grad_act, grad_s, p_real, beta_linear, bias, trace, trace_w,
            reset, b, t, h, T, H, m);
        const float prev = pp_prev(beta_linear, bias, trace, reset, b, t, h, T, H);
        gw = fmaf(terms.d_raw, prev, gw);
    }
    gb = pp_block_sum(gb, sb);
    __syncthreads();
    gw = pp_block_sum(gw, sw);
    if (threadIdx.x == 0) {
        grad_bias[h] = gb;
        grad_trace_w[h] = gw;
    }
}
