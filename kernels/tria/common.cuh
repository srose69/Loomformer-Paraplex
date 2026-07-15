// kernels/tria/common.cuh -- shared __device__ block-reduce helpers used by
// several tria kernel groups (gate_mix_block_reduce9 by tria_init_gate /
// tria_step_gate / gate_slot_mix; slot_pool_block_reduce_sum by
// slot_attention_pool). Extracted byte-for-byte from the
// original tria.py _CUDA_TRIA_CUDA_SRC inline string (including each
// helper's original leading doc-comment, where the source had one).
#pragma once

// GATE_MIX_THREADS: block size for the gate_mix_block_reduce9 tree reduction
// -- shared by tria_init_gate, tria_step_gate, and gate_slot_mix (both their
// __global__ kernels and their host launchers, which size <<<blocks,threads>>>
// with it directly).
constexpr int GATE_MIX_THREADS = 256;

// ============================================================================
// SharedTriaReader.attention_pool_slots: one query-attention pool over the H
// axis per (b,t), operating directly on the 9-dim slots (spec Sec6/Sec7.11's
// "pool in slot-space" identity: score_j = q.(W x_j + b) = (W^T q).x_j + q.b,
// the q.b term is a per-(b,t,h)-CONSTANT additive shift into a softmax over h
// -- softmax is exactly shift-invariant to that, so it contributes zero to
// both the forward output and every gradient; correctly omitted here, not a
// shortcut). One CUDA block per (b,t) row, threads stride over H, two passes
// (row-max, then exp-sum + weighted 9-dim accumulate) mirroring torch.softmax's
// own max-subtracted algorithm for numerical parity. Backward saves only the
// tiny per-row log-sum-exp (not the [B,T,H] weights) and recomputes score[h]/
// weight[h] from the already-alive carry tensor -- the "big sausage" (H=1280
// per row) never touches global memory as a persisted intermediate, only as
// the same carry tensor that was already going to live until backward anyway.
// ============================================================================
// SLOT_POOL_THREADS: block size for the slot_pool_block_reduce_* family --
// used by slot_attention_pool. The 256 cap is shared by the active slot reader --
// verified by grep, kept separate on purpose (no cross-group coupling if one
// changes independently later).
constexpr int SLOT_POOL_THREADS = 256;

// ============================================================================
// Spec Sec4's per-layer gate mix: p[b,t,h] = Sum_k w[k]*slot_k(carry[b,t,h]).
// One thread per (b,t,h), 9 FMAs in registers -- no [B,T,H,9] intermediate,
// no matmul/bmm of any kind (this is a fixed-length-9 weighted reduction, not
// a matrix product; routing it through cuBLAS via torch.einsum is exactly the
// degenerate-GEMV regression this kernel replaces). w = softmax(logits) is
// computed in PyTorch beforehand (9 elements, negligible cost, and letting
// autograd handle THAT tiny softmax's own backward is simpler and exactly as
// fast as doing it in-kernel).
// ============================================================================

// Block-wide tree reduction of 9 running accumulators (one per slot), no
// atomics: each thread contributes its own local[9], one shared-memory pass
// halves the live thread count log2(GATE_MIX_THREADS) times. Used only by
// the backward kernel below -- see its comment for why this replaced an
// atomics-based version.
__device__ __forceinline__ void gate_mix_block_reduce9(float* vals) {
    __shared__ float sdata[GATE_MIX_THREADS][9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) sdata[threadIdx.x][k] = vals[k];
    __syncthreads();
    #pragma unroll
    for (int stride = GATE_MIX_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int k = 0; k < 9; ++k) sdata[threadIdx.x][k] += sdata[threadIdx.x + stride][k];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < 9; ++k) vals[k] = sdata[0][k];
}

static __global__ void gate_mix_grad_w_reduce_kernel(
    const float* __restrict__ partial,  // [9, nblocks]
    float* __restrict__ grad_w,         // [9]
    int64_t nblocks) {
    extern __shared__ float smem[];

    const int k = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const float* row = partial + (int64_t)k * nblocks;

    float sum = 0.0f;
    int64_t b = tid;
    for (; b + (int64_t)blockDim.x * 3 < nblocks; b += (int64_t)blockDim.x * 4) {
        sum += row[b];
        sum += row[b + blockDim.x];
        sum += row[b + (int64_t)blockDim.x * 2];
        sum += row[b + (int64_t)blockDim.x * 3];
    }
    for (; b < nblocks; b += blockDim.x) sum += row[b];

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) smem[wid] = sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < nwarps; ++w) total += smem[w];
        grad_w[k] = total;
    }
}

__device__ __forceinline__ float slot_pool_block_reduce_max(float val) {
    __shared__ float sdata[SLOT_POOL_THREADS];
    sdata[threadIdx.x] = val;
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float result = sdata[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ float slot_pool_block_reduce_sum(float val) {
    __shared__ float sdata[SLOT_POOL_THREADS];
    sdata[threadIdx.x] = val;
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float result = sdata[0];
    __syncthreads();
    return result;
}

__device__ __forceinline__ void slot_pool_block_reduce_sum2(float* vals) {
    __shared__ float sdata[SLOT_POOL_THREADS][2];
    sdata[threadIdx.x][0] = vals[0];
    sdata[threadIdx.x][1] = vals[1];
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x][0] += sdata[threadIdx.x + stride][0];
            sdata[threadIdx.x][1] += sdata[threadIdx.x + stride][1];
        }
        __syncthreads();
    }
    vals[0] = sdata[0][0];
    vals[1] = sdata[0][1];
    __syncthreads();
}

// Reduces 10 running accumulators (9 weighted-slot sums + 1 sum_exp) at once,
// one shared-memory pass instead of ten -- forward's only per-row reduction.
__device__ __forceinline__ void slot_pool_block_reduce_sum10(float* vals) {
    __shared__ float sdata[SLOT_POOL_THREADS][10];
    #pragma unroll
    for (int k = 0; k < 10; ++k) sdata[threadIdx.x][k] = vals[k];
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int k = 0; k < 10; ++k) sdata[threadIdx.x][k] += sdata[threadIdx.x + stride][k];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < 10; ++k) vals[k] = sdata[0][k];
    __syncthreads();
}

// Same tree reduction, 9 accumulators -- backward's grad_score_w partial.
__device__ __forceinline__ void slot_pool_block_reduce_sum9(float* vals) {
    __shared__ float sdata[SLOT_POOL_THREADS][9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) sdata[threadIdx.x][k] = vals[k];
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int k = 0; k < 9; ++k) sdata[threadIdx.x][k] += sdata[threadIdx.x + stride][k];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < 9; ++k) vals[k] = sdata[0][k];
    __syncthreads();
}

__device__ __forceinline__ void slot_pool_block_reduce_max2(float* vals) {
    __shared__ float sdata[SLOT_POOL_THREADS][2];
    sdata[threadIdx.x][0] = vals[0];
    sdata[threadIdx.x][1] = vals[1];
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x][0] = fmaxf(sdata[threadIdx.x][0], sdata[threadIdx.x + stride][0]);
            sdata[threadIdx.x][1] = fmaxf(sdata[threadIdx.x][1], sdata[threadIdx.x + stride][1]);
        }
        __syncthreads();
    }
    vals[0] = sdata[0][0];
    vals[1] = sdata[0][1];
    __syncthreads();
}

__device__ __forceinline__ void slot_pool_block_reduce_sum20(float* vals) {
    __shared__ float sdata[SLOT_POOL_THREADS][20];
    #pragma unroll
    for (int k = 0; k < 20; ++k) sdata[threadIdx.x][k] = vals[k];
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int k = 0; k < 20; ++k) sdata[threadIdx.x][k] += sdata[threadIdx.x + stride][k];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < 20; ++k) vals[k] = sdata[0][k];
    __syncthreads();
}

__device__ __forceinline__ void slot_pool_block_reduce_sum18(float* vals) {
    __shared__ float sdata[SLOT_POOL_THREADS][18];
    #pragma unroll
    for (int k = 0; k < 18; ++k) sdata[threadIdx.x][k] = vals[k];
    __syncthreads();
    #pragma unroll
    for (int stride = SLOT_POOL_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            #pragma unroll
            for (int k = 0; k < 18; ++k) sdata[threadIdx.x][k] += sdata[threadIdx.x + stride][k];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int k = 0; k < 18; ++k) vals[k] = sdata[0][k];
    __syncthreads();
}
