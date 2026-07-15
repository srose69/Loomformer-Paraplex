// kernels/pvpowlu/pvpowlu_kernel.cuh -- pure CUDA device kernels for PowLU-
// V (Jiang et al. 2026 PV variant). No torch includes.
#pragma once

template <typename scalar_t>
__global__ void pvpowlu_fwd_kernel(
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    scalar_t* __restrict__ out,
    float m,
    int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float p = (float)x1[i];
    float a = (float)x2[i];
    float safe = fmaxf(a, 1.0e-12f);
    float root = sqrtf(safe);
    float exponent = m / (root + 1.0f);
    float sig = 1.0f / (1.0f + expf(-a));
    float gate = powf(safe, exponent) * sig;
    out[i] = (scalar_t)(p * gate);
}

template <typename scalar_t>
__global__ void pvpowlu_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    scalar_t* __restrict__ grad_x1,
    scalar_t* __restrict__ grad_x2,
    float m,
    int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float go = (float)grad_out[i];
    float p = (float)x1[i];
    float a = (float)x2[i];
    float safe = fmaxf(a, 1.0e-12f);
    float root = sqrtf(safe);
    float denom = root + 1.0f;
    float exponent = m / denom;
    float sig = 1.0f / (1.0f + expf(-a));
    float gate = powf(safe, exponent) * sig;
    float exponent_prime = -m / (2.0f * root * denom * denom);
    float dlog_gate = exponent_prime * logf(safe) + exponent / safe + (1.0f - sig);
    grad_x1[i] = (scalar_t)(go * gate);
    grad_x2[i] = (scalar_t)(go * p * gate * dlog_gate);
}
