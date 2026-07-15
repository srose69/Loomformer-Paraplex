// kernels/tria/slot_attention_pool/slot_attention_pool_kernel.cuh -- pure CUDA device kernel(s)
// for the 'slot_attention_pool' tria op. No torch includes here on purpose: this is
// the file to open for reading/editing raw CUDA C or dropping in inline
// PTX. Extracted byte-for-byte from tria.py's old _CUDA_TRIA_CUDA_SRC
// inline string, including each kernel's original leading doc-comment
// where the source had one.
#pragma once

#include <cfloat>

#include "../common.cuh"

template <typename scalar_t>
__global__ void slot_attention_pool_forward_kernel(
    const scalar_t* __restrict__ carry,    // [BT, H, 9]
    const scalar_t* __restrict__ score_w,  // [9]
    scalar_t* __restrict__ pooled,         // [BT, 9]
    scalar_t* __restrict__ lse,            // [BT]
    int64_t H) {
    int64_t bt = blockIdx.x;
    const scalar_t* row = carry + bt * H * 9;

    float sw[9];
    #pragma unroll
    for (int s = 0; s < 9; ++s) sw[s] = (float)score_w[s];

    float local_max = -FLT_MAX;
    for (int64_t h = threadIdx.x; h < H; h += SLOT_POOL_THREADS) {
        const scalar_t* c = row + h * 9;
        float score = 0.0f;
        #pragma unroll
        for (int s = 0; s < 9; ++s) score += (float)c[s] * sw[s];
        local_max = fmaxf(local_max, score);
    }
    float row_max = slot_pool_block_reduce_max(local_max);

    float acc[10] = {0.0f};
    for (int64_t h = threadIdx.x; h < H; h += SLOT_POOL_THREADS) {
        const scalar_t* c = row + h * 9;
        float score = 0.0f;
        #pragma unroll
        for (int s = 0; s < 9; ++s) score += (float)c[s] * sw[s];
        float e = expf(score - row_max);
        acc[9] += e;
        #pragma unroll
        for (int s = 0; s < 9; ++s) acc[s] += e * (float)c[s];
    }
    slot_pool_block_reduce_sum10(acc);

    if (threadIdx.x == 0) {
        float row_sum = acc[9];
        float inv = 1.0f / row_sum;
        scalar_t* out = pooled + bt * 9;
        #pragma unroll
        for (int s = 0; s < 9; ++s) out[s] = (scalar_t)(acc[s] * inv);
        lse[bt] = (scalar_t)(logf(row_sum) + row_max);
    }
}

// Backward: standard softmax-weighted-pool backward, re-deriving weight[h]
// from the saved per-row lse (no [B,T,H] weights tensor ever persisted).
// Pass 1 gets dot_gw = Sum_h weight[h]*grad_weight[h] (needed by every h's
// grad_score); pass 2 writes grad_carry and this row's PRIVATE partial into
// block_partial[bt,9] (no cross-block atomics -- see gate_slot_mix_backward's
// comment for why direct atomicAdd-into-one-global-9 was a contention
// bottleneck, same fix applied here: torch::sum(dim=0) does the final Sum-
// over-BT-rows below).
template <typename scalar_t>
__global__ void slot_attention_pool_backward_kernel(
    const scalar_t* __restrict__ grad_pooled,  // [BT, 9]
    const scalar_t* __restrict__ carry,        // [BT, H, 9]
    const scalar_t* __restrict__ score_w,      // [9]
    const scalar_t* __restrict__ lse,          // [BT]
    scalar_t* __restrict__ grad_carry,         // [BT, H, 9]
    float* __restrict__ block_partial,         // [BT, 9]
    int64_t H) {
    int64_t bt = blockIdx.x;
    const scalar_t* row = carry + bt * H * 9;
    scalar_t* grow = grad_carry + bt * H * 9;
    float row_lse = (float)lse[bt];

    __shared__ float gp[9];
    if (threadIdx.x < 9) gp[threadIdx.x] = (float)grad_pooled[bt * 9 + threadIdx.x];
    float sw[9];
    #pragma unroll
    for (int s = 0; s < 9; ++s) sw[s] = (float)score_w[s];
    __syncthreads();

    float local_dot = 0.0f;
    for (int64_t h = threadIdx.x; h < H; h += SLOT_POOL_THREADS) {
        const scalar_t* c = row + h * 9;
        float score = 0.0f, gw = 0.0f;
        #pragma unroll
        for (int s = 0; s < 9; ++s) { score += (float)c[s] * sw[s]; gw += gp[s] * (float)c[s]; }
        float weight = expf(score - row_lse);
        local_dot += weight * gw;
    }
    float dot_gw = slot_pool_block_reduce_sum(local_dot);

    // Accumulate this thread's OWN grad_score_w contribution across all its
    // h-iterations in registers first (no cross-thread interaction at all),
    // then a single block-wide tree reduction at the end -- not an atomic
    // per h-iteration into a 9-float shared array, which serializes
    // GATE_MIX_THREADS-many threads on 9 addresses every iteration (measured
    // regression, see gate_slot_mix_backward_kernel's comment for the same
    // anti-pattern found and fixed there).
    float local_gsw[9] = {0.0f};
    for (int64_t h = threadIdx.x; h < H; h += SLOT_POOL_THREADS) {
        const scalar_t* c = row + h * 9;
        float score = 0.0f, gw = 0.0f;
        #pragma unroll
        for (int s = 0; s < 9; ++s) { score += (float)c[s] * sw[s]; gw += gp[s] * (float)c[s]; }
        float weight = expf(score - row_lse);
        float grad_score = weight * (gw - dot_gw);
        scalar_t* gc = grow + h * 9;
        #pragma unroll
        for (int s = 0; s < 9; ++s) {
            float val = grad_score * sw[s] + weight * gp[s];
            gc[s] = (scalar_t)val;
            local_gsw[s] += grad_score * (float)c[s];
        }
    }
    slot_pool_block_reduce_sum9(local_gsw);
    if (threadIdx.x < 9) block_partial[bt * 9 + threadIdx.x] = local_gsw[threadIdx.x];
}
