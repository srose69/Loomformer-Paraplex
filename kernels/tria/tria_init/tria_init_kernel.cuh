#pragma once
#include "../carrier.cuh"

template <typename scalar_t>
__global__ void tria_init_forward_kernel(
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    scalar_t* __restrict__ carry_1, float* __restrict__ scale_out,
    float alpha, int axis, int64_t n) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float vals[9];
    tria_carrier_build9((float)r[idx], (float)i_[idx], (float)o[idx], alpha, axis, vals);
    const float scale = fmaxf(tria_absmax9(vals), 1.0e-6f);
    const float inv = 1.0f / scale;
    scalar_t* out = carry_1 + idx * 9;
    #pragma unroll
    for (int k = 0; k < 9; ++k) out[k] = (scalar_t)(vals[k] * inv);
    scale_out[idx] = scale;
}

template <typename scalar_t>
__global__ void tria_init_backward_kernel(
    const scalar_t* __restrict__ grad_carry_1,
    const scalar_t* __restrict__ r, const scalar_t* __restrict__ i_, const scalar_t* __restrict__ o,
    const float* __restrict__ scale,
    scalar_t* __restrict__ grad_r, scalar_t* __restrict__ grad_i, scalar_t* __restrict__ grad_o,
    float alpha, int axis, int64_t n) {
    const int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float rv=(float)r[idx], iv=(float)i_[idx], ov=(float)o[idx];
    float vals[9], gout[9], gm[9];
    tria_carrier_build9(rv, iv, ov, alpha, axis, vals);
    const scalar_t* g = grad_carry_1 + idx * 9;
    #pragma unroll
    for (int k=0;k<9;++k) gout[k]=(float)g[k];
    tria_maxabs_backward9(gout, vals, scale[idx], gm);
    float da,db,dc;
    tria_carrier_grad_abc(gm, alpha, axis, rv, iv, ov, da, db, dc);
    grad_r[idx]=(scalar_t)(da*iv + db*ov);
    grad_i[idx]=(scalar_t)(da*rv + dc*ov);
    grad_o[idx]=(scalar_t)(db*rv + dc*iv);
}
