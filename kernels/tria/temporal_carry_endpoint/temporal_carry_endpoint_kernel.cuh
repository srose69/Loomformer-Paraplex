#pragma once
#include <stdint.h>
#include <math.h>
#include "../carrier.cuh"

constexpr int TCE_CHECKPOINT_INTERVAL = 32;

template <typename scalar_t>
__device__ __forceinline__ float tce_load(const scalar_t* p) { return (float)(*p); }

template <typename scalar_t>
__device__ __forceinline__ void tce_store(scalar_t* p, float v) { *p = (scalar_t)v; }

template <typename scalar_t>
__global__ void temporal_carry_endpoint_forward_kernel(
    const scalar_t* __restrict__ depth,
    const bool* __restrict__ reset,
    const scalar_t* __restrict__ initial,
    const bool* __restrict__ initial_valid,
    scalar_t* __restrict__ endpoint,
    float* __restrict__ endpoint_fp32,
    float* __restrict__ scales,
    float* __restrict__ checkpoints,
    int64_t B, int64_t T, int64_t H,
    int64_t C, bool has_initial) {
    const int64_t stream = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (stream >= B * H) return;
    const int64_t b = stream / H;
    const int64_t h = stream - b * H;
    const int64_t base = (b * T * H + h) * 9;
    const int64_t step = H * 9;
    float acc[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    bool have = has_initial && initial_valid[b];
    if (have) {
        const scalar_t* src = initial + stream * 9;
        #pragma unroll
        for (int k = 0; k < 9; ++k) acc[k] = tce_load(src + k);
    }
    for (int64_t t = 0; t < T; ++t) {
        const int64_t off = base + t * step;
        float local[9], pre[9];
        #pragma unroll
        for (int k = 0; k < 9; ++k) local[k] = tce_load(depth + off + k);
        const bool restart = reset[b * T + t] || !have;
        if (restart) {
            #pragma unroll
            for (int k = 0; k < 9; ++k) pre[k] = local[k];
        } else {
            tria_matmul9(local, acc, pre);
        }
        const float s = tria_rms9(pre);
        scales[(b * T + t) * H + h] = s;
        const float inv = 1.0f / s;
        #pragma unroll
        for (int k = 0; k < 9; ++k) acc[k] = pre[k] * inv;
        if ((t % TCE_CHECKPOINT_INTERVAL) == 0) {
            float* checkpoint = checkpoints + ((b * C + t / TCE_CHECKPOINT_INTERVAL) * H + h) * 9;
            #pragma unroll
            for (int k = 0; k < 9; ++k) checkpoint[k] = acc[k];
        }
        have = true;
    }
    scalar_t* out = endpoint + stream * 9;
    float* out32 = endpoint_fp32 + stream * 9;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        tce_store(out + k, acc[k]);
        out32[k] = acc[k];
    }
}

template <typename scalar_t, typename grad_t>
__global__ void temporal_carry_endpoint_backward_kernel(
    const grad_t* __restrict__ grad_endpoint,
    const scalar_t* __restrict__ depth,
    const float* __restrict__ endpoint_fp32,
    const bool* __restrict__ reset,
    const scalar_t* __restrict__ initial,
    const bool* __restrict__ initial_valid,
    const float* __restrict__ scales,
    const float* __restrict__ checkpoints,
    scalar_t* __restrict__ grad_depth,
    scalar_t* __restrict__ grad_initial,
    int64_t B, int64_t T, int64_t H,
    int64_t C, bool has_initial) {
    const int64_t stream = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (stream >= B * H) return;
    const int64_t b = stream / H;
    const int64_t h = stream - b * H;
    const int64_t base = (b * T * H + h) * 9;
    const int64_t step = H * 9;
    float cur[9], gcur[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        cur[k] = endpoint_fp32[stream * 9 + k];
        gcur[k] = (float)grad_endpoint[stream * 9 + k];
    }
    for (int64_t t = T - 1; t >= 0; --t) {
        const int64_t off = base + t * step;
        float local[9], pre[9], gpre[9];
        #pragma unroll
        for (int k = 0; k < 9; ++k) local[k] = tce_load(depth + off + k);
        const bool has_prev = (t > 0) || (has_initial && initial_valid[b]);
        const bool restart = reset[b * T + t] || !has_prev;
        if (restart) {
            #pragma unroll
            for (int k = 0; k < 9; ++k) pre[k] = local[k];
            const float s = scales[(b * T + t) * H + h];
            tria_rms_backward9(gcur, pre, s, gpre);
            #pragma unroll
            for (int k = 0; k < 9; ++k) tce_store(grad_depth + off + k, gpre[k]);
            break;
        }
        float prev[9], glocal[9], gprev[9];
        const float s = scales[(b * T + t) * H + h];
        if (t == 0 && has_initial && initial_valid[b]) {
            const scalar_t* src = initial + stream * 9;
            #pragma unroll
            for (int k = 0; k < 9; ++k) prev[k] = tce_load(src + k);
        } else if (((t - 1) % TCE_CHECKPOINT_INTERVAL) == 0) {
            const float* checkpoint = checkpoints +
                ((b * C + (t - 1) / TCE_CHECKPOINT_INTERVAL) * H + h) * 9;
            #pragma unroll
            for (int k = 0; k < 9; ++k) prev[k] = checkpoint[k];
        } else {
            // Reconstruct at most TCE_CHECKPOINT_INTERVAL-1 consecutive
            // states.  The old code inverted the entire T-long recurrence;
            // nearly aligned matrices amplified roundoff to 1e28 at T=1024.
            float inv[9], scaled_cur[9];
            tria_invert9(local, inv);
            #pragma unroll
            for (int k = 0; k < 9; ++k) scaled_cur[k] = cur[k] * s;
            tria_matmul9(inv, scaled_cur, prev);
        }
        #pragma unroll
        for (int k = 0; k < 9; ++k) pre[k] = cur[k] * s;
        tria_rms_backward9(gcur, pre, s, gpre);
        tria_matmul_right_transpose9(gpre, prev, glocal);
        tria_matmul_left_transpose9(local, gpre, gprev);
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            tce_store(grad_depth + off + k, glocal[k]);
            cur[k] = prev[k];
            gcur[k] = gprev[k];
        }
        if (t == 0 && has_initial && initial_valid[b]) {
            scalar_t* gi = grad_initial + stream * 9;
            #pragma unroll
            for (int k = 0; k < 9; ++k) tce_store(gi + k, gcur[k]);
        }
    }
}
