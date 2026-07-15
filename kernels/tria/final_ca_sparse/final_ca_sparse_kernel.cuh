#pragma once
#include <stdint.h>
#include <math.h>

constexpr int FINAL_CA_THREADS = 256;

template <typename scalar_t>
__device__ __forceinline__ float fca_load(const scalar_t* p) { return (float)(*p); }

template <typename scalar_t>
__device__ __forceinline__ void fca_store(scalar_t* p, float v) { *p = (scalar_t)v; }

__device__ __forceinline__ float fca_reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = FINAL_CA_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

template <typename scalar_t>
__global__ void final_ca_sparse_forward_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const bool* __restrict__ allowed,
    scalar_t* __restrict__ out,
    float* __restrict__ lse,
    int64_t B, int64_t T, int64_t K, int64_t D,
    float scale) {
    const int64_t row = blockIdx.x;
    if (row >= B * T) return;
    const int64_t b = row / T;
    const scalar_t* qrow = q + row * D;
    extern __shared__ float smem[];
    float* weights = smem;
    float* reduce = weights + K;
    bool any = false;

    for (int64_t j = 0; j < K; ++j) {
        float part = 0.0f;
        if (allowed[row * K + j]) {
            const scalar_t* krow = k + (b * K + j) * D;
            for (int64_t d = threadIdx.x; d < D; d += blockDim.x)
                part = fmaf(fca_load(qrow + d), fca_load(krow + d), part);
        }
        const float dot = fca_reduce_sum(part, reduce);
        if (threadIdx.x == 0) {
            const bool ok = allowed[row * K + j];
            weights[j] = ok ? dot * scale : -INFINITY;
            any |= ok;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float maxv = -INFINITY;
        for (int64_t j = 0; j < K; ++j) maxv = fmaxf(maxv, weights[j]);
        float sum = 0.0f;
        if (any) {
            for (int64_t j = 0; j < K; ++j) {
                const float w = isfinite(weights[j]) ? expf(weights[j] - maxv) : 0.0f;
                weights[j] = w;
                sum += w;
            }
            const float inv = 1.0f / sum;
            for (int64_t j = 0; j < K; ++j) weights[j] *= inv;
            lse[row] = maxv + logf(sum);
        } else {
            for (int64_t j = 0; j < K; ++j) weights[j] = 0.0f;
            lse[row] = -INFINITY;
        }
    }
    __syncthreads();

    scalar_t* orow = out + row * D;
    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float value = 0.0f;
        for (int64_t j = 0; j < K; ++j)
            value = fmaf(weights[j], fca_load(v + (b * K + j) * D + d), value);
        fca_store(orow + d, value);
    }
}

template <typename scalar_t, typename grad_t>
__global__ void final_ca_sparse_backward_rows_kernel(
    const grad_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const bool* __restrict__ allowed,
    const float* __restrict__ lse,
    scalar_t* __restrict__ grad_q,
    float* __restrict__ weights_out,
    float* __restrict__ dscore_out,
    int64_t B, int64_t T, int64_t K, int64_t D,
    float scale) {
    const int64_t row = blockIdx.x;
    if (row >= B * T) return;
    const int64_t b = row / T;
    const scalar_t* qrow = q + row * D;
    const grad_t* grow = grad_out + row * D;
    extern __shared__ float smem[];
    float* weights = smem;
    float* dweight = weights + K;
    float* dscore = dweight + K;
    float* reduce = dscore + K;
    const bool any = isfinite(lse[row]);

    for (int64_t j = 0; j < K; ++j) {
        float dot_qk = 0.0f;
        float dot_gv = 0.0f;
        if (any && allowed[row * K + j]) {
            const scalar_t* krow = k + (b * K + j) * D;
            const scalar_t* vrow = v + (b * K + j) * D;
            for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
                dot_qk = fmaf(fca_load(qrow + d), fca_load(krow + d), dot_qk);
                dot_gv = fmaf((float)grow[d], fca_load(vrow + d), dot_gv);
            }
        }
        const float sqk = fca_reduce_sum(dot_qk, reduce);
        const float sgv = fca_reduce_sum(dot_gv, reduce);
        if (threadIdx.x == 0) {
            const bool ok = any && allowed[row * K + j];
            weights[j] = ok ? expf(sqk * scale - lse[row]) : 0.0f;
            dweight[j] = ok ? sgv : 0.0f;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = 0.0f;
        for (int64_t j = 0; j < K; ++j) mean += weights[j] * dweight[j];
        for (int64_t j = 0; j < K; ++j) {
            dscore[j] = weights[j] * (dweight[j] - mean) * scale;
            weights_out[row * K + j] = weights[j];
            dscore_out[row * K + j] = dscore[j];
        }
    }
    __syncthreads();

    for (int64_t d = threadIdx.x; d < D; d += blockDim.x) {
        float gq = 0.0f;
        for (int64_t j = 0; j < K; ++j)
            gq = fmaf(dscore[j], fca_load(k + (b * K + j) * D + d), gq);
        fca_store(grad_q + row * D + d, gq);
    }
}

template <typename scalar_t, typename grad_t>
__global__ void final_ca_sparse_backward_keys_kernel(
    const grad_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const float* __restrict__ weights,
    const float* __restrict__ dscore,
    scalar_t* __restrict__ grad_k,
    scalar_t* __restrict__ grad_v,
    int64_t T, int64_t K, int64_t D) {
    const int64_t bk = blockIdx.x;
    const int64_t b = bk / K;
    const int64_t j = bk - b * K;
    const int64_t d0 = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
    if (d0 >= D) return;

    float gk = 0.0f;
    float gv = 0.0f;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t row = b * T + t;
        const int64_t wk = row * K + j;
        gk = fmaf(dscore[wk], fca_load(q + row * D + d0), gk);
        gv = fmaf(weights[wk], (float)grad_out[row * D + d0], gv);
    }
    const int64_t off = bk * D + d0;
    fca_store(grad_k + off, gk);
    fca_store(grad_v + off, gv);
}
