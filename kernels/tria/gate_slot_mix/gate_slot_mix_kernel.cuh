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
    const scalar_t* __restrict__ carry,   // [N, 9]
    const scalar_t* __restrict__ w,       // [9]
    scalar_t* __restrict__ p,             // [N]
    int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const scalar_t* c = carry + idx * 9;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 9; ++k) acc += (float)c[k] * (float)w[k];
    p[idx] = (scalar_t)acc;
}

// Backward: grad_carry[n,k] = grad_p[n]*w[k] (pure broadcast, no reduction).
// grad_w[k] = Sum_n grad_p[n]*carry[n,k] -- the one real reduction here. Each
// thread accumulates its own local[9] in registers (n is 1:1 with threads
// here, so there is nothing to loop-accumulate first), then one block-wide
// tree reduction (gate_mix_block_reduce9) folds GATE_MIX_THREADS partials
// into block_partial[blocks,9] -- zero atomics anywhere. (Two earlier, both
// measured-and-rejected versions: (a) atomicAdd every thread straight into a
// single global grad_w[9] -- tens of thousands of blocks hammering 9
// addresses, catastrophic global contention; (b) atomicAdd into a 9-float
// SHARED array -- same contention pattern one level down, 256 threads/block
// serializing on 9 shared addresses, measured at 56ms/iter for N=5.24M,
// slower than the einsum/bmm baseline it was meant to replace. This tree-
// reduction form has no contention at all.)
template <typename scalar_t>
__global__ void gate_slot_mix_backward_kernel(
    const scalar_t* __restrict__ grad_p,     // [N]
    const scalar_t* __restrict__ carry,      // [N, 9]
    const scalar_t* __restrict__ w,          // [9]
    scalar_t* __restrict__ grad_carry,       // [N, 9]
    float* __restrict__ grad_w_partial,      // [9, gridDim.x]
    int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    float local[9] = {0.0f};
    if (idx < n) {
        float gp = (float)grad_p[idx];
        const scalar_t* c = carry + idx * 9;
        scalar_t* gc = grad_carry + idx * 9;
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            gc[k] = (scalar_t)(gp * (float)w[k]);
            local[k] = gp * (float)c[k];
        }
    }
    gate_mix_block_reduce9(local);
    if (threadIdx.x < 9)
        grad_w_partial[(int64_t)threadIdx.x * gridDim.x + blockIdx.x] = local[threadIdx.x];
}
