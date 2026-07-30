// kernels/tria/gate_slot_mix/gate_slot_mix_kernel.cuh -- pure CUDA device kernel(s)
// for the 'gate_slot_mix' tria op. No torch includes here on purpose: this is
// the file to open for reading/editing raw CUDA C or dropping in inline
// PTX. Extracted byte-for-byte from tria.py's old _CUDA_TRIA_CUDA_SRC
// inline string, including each kernel's original leading doc-comment
// where the source had one.
#pragma once

#include "../common.cuh"

template <typename scalar_t>
__global__ void gate_slot_mix_forward_kernel(
    const scalar_t* __restrict__ carry,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ p,
    int64_t n, int64_t hidden, int64_t hidden_per_head) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const scalar_t* c = carry + idx * 9;
    const scalar_t* wh = w + ((idx % hidden) / hidden_per_head) * 9;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 9; ++k) acc += (float)c[k] * (float)wh[k];
    p[idx] = (scalar_t)acc;
}

template <typename scalar_t>
__global__ void gate_slot_mix_backward_kernel(
    const scalar_t* __restrict__ grad_p,
    const scalar_t* __restrict__ carry,
    const scalar_t* __restrict__ w,
    scalar_t* __restrict__ grad_carry,
    float* __restrict__ grad_w_partial,
    int64_t n, int64_t hidden, int64_t hidden_per_head,
    int64_t heads, int64_t chunks_per_head, int64_t partials_per_head) {
    const int64_t chunk = blockIdx.x % chunks_per_head;
    const int64_t segment = blockIdx.x / chunks_per_head;
    const int64_t head = segment % heads;
    const int64_t bt = segment / heads;
    const int64_t h_local = chunk * blockDim.x + threadIdx.x;
    const int64_t idx = bt * hidden + head * hidden_per_head + h_local;
    const bool valid = h_local < hidden_per_head && idx < n;
    const scalar_t* wh = w + head * 9;
    float local[9] = {0.0f};
    if (valid) {
        float gp = (float)grad_p[idx];
        const scalar_t* c = carry + idx * 9;
        scalar_t* gc = grad_carry + idx * 9;
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            gc[k] = (scalar_t)(gp * (float)wh[k]);
            local[k] = gp * (float)c[k];
        }
    }
    gate_mix_block_reduce9(local);
    if (threadIdx.x < 9) {
        const int64_t partial = bt * chunks_per_head + chunk;
        grad_w_partial[(head * 9 + threadIdx.x) * partials_per_head + partial] =
            local[threadIdx.x];
    }
}
