#pragma once
#include <stdint.h>
#include "../common.cuh"
#include "../carrier.cuh"

template <typename scalar_t>
__device__ __forceinline__ void depth_replay_quantize9(
    const float pre[9], float state[9]) {
    const float scale = fmaxf(tria_absmax9(pre), 1.0e-6f);
    const float inv = 1.0f / scale;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        const scalar_t stored = (scalar_t)(pre[k] * inv);
        state[k] = (float)stored;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void depth_replay_previous9(
    const scalar_t* const* __restrict__ r_ptrs,
    const scalar_t* const* __restrict__ i_ptrs,
    const scalar_t* const* __restrict__ o_ptrs,
    const int32_t* __restrict__ axes,
    const scalar_t* __restrict__ seed,
    const bool* __restrict__ seed_valid,
    float alpha, int layer_index,
    int64_t idx, int64_t B, int64_t T, int64_t H,
    float previous[9]) {
    const int64_t h = idx % H;
    const int64_t bt = idx / H;
    const int64_t t = bt % T;
    const int64_t b = bt / T;
    float m[9], pre[9];
    tria_carrier_build9(
        (float)r_ptrs[0][idx], (float)i_ptrs[0][idx],
        (float)o_ptrs[0][idx], alpha, axes[0], m);
    if (seed != nullptr && t == 0 && seed_valid[b]) {
        float initial[9];
        const scalar_t* src = seed + (b * H + h) * 9;
        #pragma unroll
        for (int k = 0; k < 9; ++k) initial[k] = (float)src[k];
        tria_matmul9(m, initial, pre);
    } else {
        #pragma unroll
        for (int k = 0; k < 9; ++k) pre[k] = m[k];
    }
    depth_replay_quantize9<scalar_t>(pre, previous);
    for (int layer = 1; layer < layer_index; ++layer) {
        tria_carrier_build9(
            (float)r_ptrs[layer][idx], (float)i_ptrs[layer][idx],
            (float)o_ptrs[layer][idx], alpha, axes[layer], m);
        tria_matmul9(m, previous, pre);
        depth_replay_quantize9<scalar_t>(pre, previous);
    }
}

template <typename scalar_t, bool GATED>
__global__ void depth_replay_backward_kernel(
    const scalar_t* __restrict__ grad_carry,
    const scalar_t* __restrict__ grad_p,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ i_,
    const scalar_t* __restrict__ o,
    const scalar_t* __restrict__ w,
    const scalar_t* const* __restrict__ r_ptrs,
    const scalar_t* const* __restrict__ i_ptrs,
    const scalar_t* const* __restrict__ o_ptrs,
    const int32_t* __restrict__ axes,
    const scalar_t* __restrict__ seed,
    const bool* __restrict__ seed_valid,
    scalar_t* __restrict__ grad_r,
    scalar_t* __restrict__ grad_i,
    scalar_t* __restrict__ grad_o,
    scalar_t* __restrict__ grad_previous,
    float* __restrict__ grad_w_partial,
    float alpha, int axis, int layer_index,
    int64_t B, int64_t T, int64_t H) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    const int64_t n = B * T * H;
    float local_w[9] = {0.0f};
    if (idx < n) {
        float previous[9];
        depth_replay_previous9(
            r_ptrs, i_ptrs, o_ptrs, axes, seed, seed_valid,
            alpha, layer_index, idx, B, T, H, previous);
        const float rv = (float)r[idx], iv = (float)i_[idx], ov = (float)o[idx];
        float m[9], pre[9], gout[9], gpre[9], gm[9], gprev[9];
        tria_carrier_build9(rv, iv, ov, alpha, axis, m);
        tria_matmul9(m, previous, pre);
        const float scale = fmaxf(tria_absmax9(pre), 1.0e-6f);
        const float inv = 1.0f / scale;
        const float gp = GATED ? (float)grad_p[idx] : 0.0f;
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            const scalar_t stored = (scalar_t)(pre[k] * inv);
            gout[k] = (float)grad_carry[idx * 9 + k] +
                (GATED ? gp * (float)w[k] : 0.0f);
            if (GATED) local_w[k] = gp * (float)stored;
        }
        tria_maxabs_backward9(gout, pre, scale, gpre);
        tria_matmul_right_transpose9(gpre, previous, gm);
        tria_matmul_left_transpose9(m, gpre, gprev);
        float da, db, dc;
        tria_carrier_grad_abc(gm, alpha, axis, rv, iv, ov, da, db, dc);
        grad_r[idx] = (scalar_t)(da * iv + db * ov);
        grad_i[idx] = (scalar_t)(da * rv + dc * ov);
        grad_o[idx] = (scalar_t)(db * rv + dc * iv);
        #pragma unroll
        for (int k = 0; k < 9; ++k)
            grad_previous[idx * 9 + k] = (scalar_t)gprev[k];
    }
    if (GATED) {
        gate_mix_block_reduce9(local_w);
        if (threadIdx.x < 9)
            grad_w_partial[(int64_t)threadIdx.x * gridDim.x + blockIdx.x] = local_w[threadIdx.x];
    }
}
