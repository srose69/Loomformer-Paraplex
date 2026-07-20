#pragma once
#include "../common.cuh"
#include "../carrier.cuh"

template <typename scalar_t>
__device__ __forceinline__ void tria_seed_load9(
    const scalar_t* __restrict__ seed, int64_t base, float out[9]) {
    #pragma unroll
    for (int k = 0; k < 9; ++k) out[k] = (float)seed[base + k];
}

template <typename scalar_t>
__global__ void tria_init_seed_forward_kernel(
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ seed, const bool* __restrict__ valid,
    scalar_t* __restrict__ carry_out, float* __restrict__ scale_out,
    float alpha, int axis, int64_t B, int64_t T, int64_t H) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t n = B * T * H;
    if (idx >= n) return;
    const int64_t h = idx % H;
    const int64_t bt = idx / H;
    const int64_t t = bt % T;
    const int64_t b = bt / T;
    float m[9], pre[9];
    tria_carrier_build9((float)r[idx], (float)i_[idx], (float)o[idx], alpha, axis, m);
    if (t == 0 && valid[b]) {
        float seed_vals[9];
        tria_seed_load9(seed, (b * H + h) * 9, seed_vals);
        tria_matmul9(m, seed_vals, pre);
    } else {
        #pragma unroll
        for (int k = 0; k < 9; ++k) pre[k] = m[k];
    }
    const float scale = tria_rms9(pre);
    const float inv = 1.0f / scale;
    scalar_t* out = carry_out == nullptr ? nullptr : (carry_out + idx * 9);
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        if (out != nullptr) out[k] = (scalar_t)(pre[k] * inv);
    }
    scale_out[idx] = scale;
}

template <typename scalar_t>
__global__ void tria_init_seed_backward_kernel(
    const scalar_t* __restrict__ grad_carry,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ seed, const bool* __restrict__ valid,
    const float* __restrict__ scale,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_seed,
    float alpha, int axis, int64_t B, int64_t T, int64_t H) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t n = B * T * H;
    if (idx >= n) return;
    const int64_t h = idx % H;
    const int64_t bt = idx / H;
    const int64_t t = bt % T;
    const int64_t b = bt / T;
    const float rv = (float)r[idx], iv = (float)i_[idx], ov = (float)o[idx];
    float m[9], pre[9], gout[9], gpre[9], gm[9];
    tria_carrier_build9(rv, iv, ov, alpha, axis, m);
    const scalar_t* g = grad_carry + idx * 9;
    #pragma unroll
    for (int k = 0; k < 9; ++k) gout[k] = (float)g[k];
    const bool use_seed = (t == 0 && valid[b]);
    if (use_seed) {
        float seed_vals[9], gseed[9];
        tria_seed_load9(seed, (b * H + h) * 9, seed_vals);
        tria_matmul9(m, seed_vals, pre);
        tria_rms_backward9(gout, pre, scale[idx], gpre);
        tria_matmul_right_transpose9(gpre, seed_vals, gm);
        tria_matmul_left_transpose9(m, gpre, gseed);
        scalar_t* gs = grad_seed + (b * H + h) * 9;
        #pragma unroll
        for (int k = 0; k < 9; ++k) gs[k] = (scalar_t)gseed[k];
    } else {
        #pragma unroll
        for (int k = 0; k < 9; ++k) pre[k] = m[k];
        tria_rms_backward9(gout, pre, scale[idx], gm);
    }
    float da, db, dc;
    tria_carrier_grad_abc(gm, alpha, axis, rv, iv, ov, da, db, dc);
    grad_r[idx] = (scalar_t)(da * iv + db * ov);
    grad_i[idx] = (scalar_t)(da * rv + dc * ov);
    grad_o[idx] = (scalar_t)(db * rv + dc * iv);
}

template <typename scalar_t>
__global__ void tria_init_seed_gate_forward_kernel(
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ seed, const bool* __restrict__ valid,
    const scalar_t* __restrict__ w, scalar_t* __restrict__ carry_out,
    scalar_t* __restrict__ p_out, float* __restrict__ scale_out,
    float alpha, int axis, int64_t B, int64_t T, int64_t H) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t n = B * T * H;
    if (idx >= n) return;
    const int64_t h = idx % H;
    const int64_t bt = idx / H;
    const int64_t t = bt % T;
    const int64_t b = bt / T;
    float m[9], pre[9];
    tria_carrier_build9((float)r[idx], (float)i_[idx], (float)o[idx], alpha, axis, m);
    if (t == 0 && valid[b]) {
        float seed_vals[9];
        tria_seed_load9(seed, (b * H + h) * 9, seed_vals);
        tria_matmul9(m, seed_vals, pre);
    } else {
        #pragma unroll
        for (int k = 0; k < 9; ++k) pre[k] = m[k];
    }
    const float scale = tria_rms9(pre);
    const float inv = 1.0f / scale;
    scalar_t* out = carry_out == nullptr ? nullptr : (carry_out + idx * 9);
    float p = 0.0f;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        const float cv = pre[k] * inv;
        if (out != nullptr) out[k] = (scalar_t)cv;
        p = fmaf(cv, (float)w[k], p);
    }
    if (p_out != nullptr) p_out[idx] = (scalar_t)p;
    scale_out[idx] = scale;
}

template <typename scalar_t>
__global__ void tria_init_seed_gate_backward_kernel(
    const scalar_t* __restrict__ grad_carry, const scalar_t* __restrict__ grad_p,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ seed, const bool* __restrict__ valid,
    const scalar_t* __restrict__ w, const float* __restrict__ scale,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_seed, float* __restrict__ grad_w_partial,
    float alpha, int axis, int64_t B, int64_t T, int64_t H) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t n = B * T * H;
    float local_w[9] = {0.0f};
    if (idx < n) {
        const int64_t h = idx % H;
        const int64_t bt = idx / H;
        const int64_t t = bt % T;
        const int64_t b = bt / T;
        const float rv = (float)r[idx], iv = (float)i_[idx], ov = (float)o[idx];
        float m[9], pre[9], gout[9], gpre[9], gm[9];
        tria_carrier_build9(rv, iv, ov, alpha, axis, m);
        const float gp = (float)grad_p[idx];
        const scalar_t* g = grad_carry + idx * 9;
        const bool use_seed = (t == 0 && valid[b]);
        if (use_seed) {
            float seed_vals[9], gseed[9];
            tria_seed_load9(seed, (b * H + h) * 9, seed_vals);
            tria_matmul9(m, seed_vals, pre);
            const float inv = 1.0f / scale[idx];
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                gout[k] = (float)g[k] + gp * (float)w[k];
                local_w[k] = gp * pre[k] * inv;
            }
            tria_rms_backward9(gout, pre, scale[idx], gpre);
            tria_matmul_right_transpose9(gpre, seed_vals, gm);
            tria_matmul_left_transpose9(m, gpre, gseed);
            scalar_t* gs = grad_seed + (b * H + h) * 9;
            #pragma unroll
            for (int k = 0; k < 9; ++k) gs[k] = (scalar_t)gseed[k];
        } else {
            #pragma unroll
            for (int k = 0; k < 9; ++k) pre[k] = m[k];
            const float inv = 1.0f / scale[idx];
            #pragma unroll
            for (int k = 0; k < 9; ++k) {
                gout[k] = (float)g[k] + gp * (float)w[k];
                local_w[k] = gp * pre[k] * inv;
            }
            tria_rms_backward9(gout, pre, scale[idx], gm);
        }
        float da, db, dc;
        tria_carrier_grad_abc(gm, alpha, axis, rv, iv, ov, da, db, dc);
        grad_r[idx] = (scalar_t)(da * iv + db * ov);
        grad_i[idx] = (scalar_t)(da * rv + dc * ov);
        grad_o[idx] = (scalar_t)(db * rv + dc * iv);
    }
    gate_mix_block_reduce9(local_w);
    if (threadIdx.x < 9) grad_w_partial[(int64_t)threadIdx.x * gridDim.x + blockIdx.x] = local_w[threadIdx.x];
}
