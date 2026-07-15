// kernels/depth_attn/depth_attn_kernel.cuh -- pure CUDA device kernels for
// online-softmax depth attention.
#pragma once

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

// ------------------------------------------------------------------
// V4Ops: vectorized load/store of 4 contiguous scalar_t values.
// float  -> float4 (16 bytes)
// bf16   -> uint64_t (8 bytes = 4 lanes)
// fp16   -> uint64_t (8 bytes = 4 lanes)
// All arithmetic stays in fp32; V4Ops only packs/unpacks memory.
// ------------------------------------------------------------------
template <typename scalar_t>
struct V4Ops;

template <>
struct V4Ops<float> {
    using Vec = float4;
    static __device__ __forceinline__ Vec load(const float* p) {
        return *reinterpret_cast<const float4*>(p);
    }
    static __device__ __forceinline__ void store(float* p, Vec v) {
        *reinterpret_cast<float4*>(p) = v;
    }
    static __device__ __forceinline__ Vec zero() {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    static __device__ __forceinline__ void to_floats(Vec v, float& a, float& b, float& c, float& d) {
        a = v.x; b = v.y; c = v.z; d = v.w;
    }
    static __device__ __forceinline__ Vec from_floats(float a, float b, float c, float d) {
        return make_float4(a, b, c, d);
    }
};

static __device__ __forceinline__ float bf16_bits_to_float(uint16_t b) {
    union U { uint32_t u; float f; } x;
    x.u = static_cast<uint32_t>(b) << 16;
    return x.f;
}

static __device__ __forceinline__ uint16_t float_to_bf16_rn(float f) {
    union U { float f; uint32_t u; } x;
    x.f = f;
    uint32_t u = x.u;
    if ((u & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<uint16_t>((u >> 16) | 0x0040U);
    }
    uint32_t lsb = (u >> 16) & 1U;
    uint32_t bias = 0x7fffU + lsb;
    return static_cast<uint16_t>((u + bias) >> 16);
}

template <>
struct V4Ops<at::BFloat16> {
    using Vec = uint64_t;
    static __device__ __forceinline__ Vec load(const at::BFloat16* p) {
        return *reinterpret_cast<const uint64_t*>(p);
    }
    static __device__ __forceinline__ void store(at::BFloat16* p, Vec v) {
        *reinterpret_cast<uint64_t*>(p) = v;
    }
    static __device__ __forceinline__ Vec zero() {
        return 0ULL;
    }
    static __device__ __forceinline__ void to_floats(Vec v, float& a, float& b, float& c, float& d) {
        a = bf16_bits_to_float((uint16_t)(v & 0xffffULL));
        b = bf16_bits_to_float((uint16_t)((v >> 16) & 0xffffULL));
        c = bf16_bits_to_float((uint16_t)((v >> 32) & 0xffffULL));
        d = bf16_bits_to_float((uint16_t)((v >> 48) & 0xffffULL));
    }
    static __device__ __forceinline__ Vec from_floats(float a, float b, float c, float d) {
        uint64_t o = 0ULL;
        o |= (uint64_t)float_to_bf16_rn(a);
        o |= (uint64_t)float_to_bf16_rn(b) << 16;
        o |= (uint64_t)float_to_bf16_rn(c) << 32;
        o |= (uint64_t)float_to_bf16_rn(d) << 48;
        return o;
    }
};

template <>
struct V4Ops<at::Half> {
    using Vec = uint64_t;
    static __device__ __forceinline__ Vec load(const at::Half* p) {
        return *reinterpret_cast<const uint64_t*>(p);
    }
    static __device__ __forceinline__ void store(at::Half* p, Vec v) {
        *reinterpret_cast<uint64_t*>(p) = v;
    }
    static __device__ __forceinline__ Vec zero() {
        return 0ULL;
    }
    static __device__ __forceinline__ void to_floats(Vec v, float& a, float& b, float& c, float& d) {
        a = __half2float(__ushort_as_half((uint16_t)(v & 0xffffULL)));
        b = __half2float(__ushort_as_half((uint16_t)((v >> 16) & 0xffffULL)));
        c = __half2float(__ushort_as_half((uint16_t)((v >> 32) & 0xffffULL)));
        d = __half2float(__ushort_as_half((uint16_t)((v >> 48) & 0xffffULL)));
    }
    static __device__ __forceinline__ Vec from_floats(float a, float b, float c, float d) {
        uint64_t o = 0ULL;
        o |= (uint64_t)__half_as_ushort(__float2half_rn(a));
        o |= (uint64_t)__half_as_ushort(__float2half_rn(b)) << 16;
        o |= (uint64_t)__half_as_ushort(__float2half_rn(c)) << 32;
        o |= (uint64_t)__half_as_ushort(__float2half_rn(d)) << 48;
        return o;
    }
};

template <typename scalar_t>
__global__ void depth_attn_fwd_kernel(
    const scalar_t* __restrict__ q,       // [QH, HD]
    const scalar_t* __restrict__ hist_k,  // [B, T, S, QH, HD]
    const scalar_t* __restrict__ hist_v,  // [B, T, S, QH, HD]
    scalar_t* __restrict__ d_out,         // [B, T, QH, HD]
    scalar_t* __restrict__ w_out,         // [B, T, QH, S]
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd) {

    extern __shared__ float smem[];
    float* scores = smem;                          // [S]
    float* red = smem + S;                          // [nwarps] reduction scratch
    __shared__ float l_val;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const int64_t row = blockIdx.x;                 // 0 .. B*T*QH-1
    const int64_t qh = row % QH;
    const int64_t bt = row / QH;                    // combined (b*T + t)

    for (int64_t s = 0; s < S; ++s) {
        float partial = 0.0f;
        const int64_t base = ((bt * S + s) * QH + qh) * HD;
        for (int64_t d = tid; d < HD; d += blockDim.x)
            partial += (float)q[qh * HD + d] * (float)hist_k[base + d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, off);
        if (lane == 0) red[wid] = partial;
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int w = 0; w < nwarps; ++w) sum += red[w];
            scores[s] = sum * inv_sqrt_hd;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = scores[0];
        for (int64_t s = 1; s < S; ++s) mx = fmaxf(mx, scores[s]);
        float sum = 0.0f;
        for (int64_t s = 0; s < S; ++s) {
            float e = __expf(scores[s] - mx);
            scores[s] = e;
            sum += e;
        }
        l_val = sum;
    }
    __syncthreads();

    const float inv_l = 1.0f / l_val;
    const int64_t out_base = (bt * QH + qh) * HD;
    for (int64_t d = tid; d < HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int64_t s = 0; s < S; ++s)
            acc += scores[s] * (float)hist_v[((bt * S + s) * QH + qh) * HD + d];
        d_out[out_base + d] = (scalar_t)(acc * inv_l);
    }
    const int64_t w_base = (bt * QH + qh) * S;
    for (int64_t s = tid; s < S; s += blockDim.x)
        w_out[w_base + s] = (scalar_t)(scores[s] * inv_l);
}

template <typename scalar_t>
__global__ void depth_attn_bwd_kernel(
    const scalar_t* __restrict__ grad_d,  // [B, T, QH, HD]
    const scalar_t* __restrict__ q,       // [QH, HD]
    const scalar_t* __restrict__ hist_k,  // [B, T, S, QH, HD]
    const scalar_t* __restrict__ hist_v,  // [B, T, S, QH, HD]
    const scalar_t* __restrict__ w,       // [B, T, QH, S] -- saved softmax weights
    float* __restrict__ grad_q_partial,   // [QH, HD, B*T] -- fp32 per-row partials
    scalar_t* __restrict__ grad_k,        // [B, T, S, QH, HD]
    scalar_t* __restrict__ grad_v,        // [B, T, S, QH, HD]
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd) {

    extern __shared__ float smem[];
    __shared__ float dot_val;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    float* gw = smem;        // [S] -- grad w.r.t. w_s, then overwritten with grad_score_s
    float* red = smem + S;   // [nwarps] cross-warp reduction scratch
    float* gd_buf = red + nwarps;  // [HD] cached grad_d row
    float* q_buf = gd_buf + HD;    // [HD] cached q row
    const int64_t row = blockIdx.x;
    const int64_t qh = row % QH;
    const int64_t bt = row / QH;
    const int64_t BT = gridDim.x / QH;
    const int64_t gd_base = (bt * QH + qh) * HD;
    const int64_t w_base = (bt * QH + qh) * S;

    for (int64_t d = tid; d < HD; d += blockDim.x) {
        gd_buf[d] = (float)grad_d[gd_base + d];
        q_buf[d] = (float)q[qh * HD + d];
    }
    __syncthreads();

    // grad_w_s = dot(grad_d, v_s). One warp owns one s at a time, keeping the
    // reduction in registers and avoiding a block-wide sync/reduction per s.
    for (int64_t s = wid; s < S; s += nwarps) {
        float partial = 0.0f;
        const int64_t base = ((bt * S + s) * QH + qh) * HD;
        for (int64_t d = lane; d < HD; d += 32)
            partial += gd_buf[d] * (float)hist_v[base + d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, off);
        if (lane == 0) gw[s] = partial;
    }
    __syncthreads();

    float dot_part = 0.0f;
    for (int64_t s = tid; s < S; s += blockDim.x)
        dot_part += gw[s] * (float)w[w_base + s];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        dot_part += __shfl_down_sync(0xffffffffu, dot_part, off);
    if (lane == 0) red[wid] = dot_part;
    __syncthreads();

    if (wid == 0) {
        float block_part = (lane < nwarps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            block_part += __shfl_down_sync(0xffffffffu, block_part, off);
        if (lane == 0) dot_val = block_part;
    }
    __syncthreads();

    // softmax backward: grad_score_s = w_s * (grad_w_s - sum_s' w_s' * grad_w_s')
    for (int64_t s = tid; s < S; s += blockDim.x)
        gw[s] = (float)w[w_base + s] * (gw[s] - dot_val);
    __syncthreads();

    for (int64_t d = tid; d < HD; d += blockDim.x) {
        const float gdv = gd_buf[d];
        const float qv = q_buf[d];
        float gq_acc = 0.0f;
        for (int64_t s = 0; s < S; ++s) {
            const int64_t idx = ((bt * S + s) * QH + qh) * HD + d;
            const float wv = (float)w[w_base + s];
            grad_v[idx] = (scalar_t)(wv * gdv);
            const float gscore = gw[s];
            grad_k[idx] = (scalar_t)(gscore * qv * inv_sqrt_hd);
            gq_acc += gscore * (float)hist_k[idx] * inv_sqrt_hd;
        }
        grad_q_partial[(qh * HD + d) * BT + bt] = gq_acc;
    }
}

// -- legacy stacked (FA-style) kernels: forward outputs O+LSE, backward uses LSE --
// Block-per-row, shared memory. Used as fallback for S > 32 or HD % 4 != 0.

template <typename scalar_t>
__global__ void depth_attn_stacked_fwd_kernel(
    const scalar_t* __restrict__ q,       // [QH, HD]
    const scalar_t* __restrict__ hist_k,  // [B, T, S, QH, HD]
    const scalar_t* __restrict__ hist_v,  // [B, T, S, QH, HD]
    scalar_t* __restrict__ d_out,         // [B, T, QH, HD]
    float* __restrict__ lse_out,          // [B, T, QH] -- fp32 always
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd) {

    extern __shared__ float smem[];
    float* scores = smem;                          // [S]
    float* red = smem + S;                          // [nwarps] reduction scratch
    __shared__ float l_val;
    __shared__ float m_val;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    const int64_t row = blockIdx.x;                 // 0 .. B*T*QH-1
    const int64_t qh = row % QH;
    const int64_t bt = row / QH;                    // combined (b*T + t)

    for (int64_t s = 0; s < S; ++s) {
        float partial = 0.0f;
        const int64_t base = ((bt * S + s) * QH + qh) * HD;
        for (int64_t d = tid; d < HD; d += blockDim.x)
            partial += (float)q[qh * HD + d] * (float)hist_k[base + d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, off);
        if (lane == 0) red[wid] = partial;
        __syncthreads();
        if (tid == 0) {
            float sum = 0.0f;
            for (int w = 0; w < nwarps; ++w) sum += red[w];
            scores[s] = sum * inv_sqrt_hd;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = scores[0];
        for (int64_t s = 1; s < S; ++s) mx = fmaxf(mx, scores[s]);
        float sum = 0.0f;
        for (int64_t s = 0; s < S; ++s) {
            float e = expf(scores[s] - mx);
            scores[s] = e;
            sum += e;
        }
        l_val = sum;
        m_val = mx;
    }
    __syncthreads();

    const float inv_l = 1.0f / l_val;
    const int64_t out_base = (bt * QH + qh) * HD;
    for (int64_t d = tid; d < HD; d += blockDim.x) {
        float acc = 0.0f;
        for (int64_t s = 0; s < S; ++s)
            acc += scores[s] * (float)hist_v[((bt * S + s) * QH + qh) * HD + d];
        d_out[out_base + d] = (scalar_t)(acc * inv_l);
    }
    if (tid == 0)
        lse_out[bt * QH + qh] = m_val + logf(l_val);
}

template <typename scalar_t>
__global__ void depth_attn_stacked_bwd_kernel(
    const scalar_t* __restrict__ grad_d,  // [B, T, QH, HD]
    const scalar_t* __restrict__ q,       // [QH, HD]
    const scalar_t* __restrict__ hist_k,  // [B, T, S, QH, HD]
    const scalar_t* __restrict__ hist_v,  // [B, T, S, QH, HD]
    const float* __restrict__ lse,        // [B, T, QH]
    float* __restrict__ grad_q_partial,   // [QH, HD, B*T] -- fp32 per-row partials
    scalar_t* __restrict__ grad_k,        // [B, T, S, QH, HD]
    scalar_t* __restrict__ grad_v,        // [B, T, S, QH, HD]
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd) {

    extern __shared__ float smem[];
    __shared__ float D_val;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarps = (blockDim.x + 31) >> 5;
    float* scores = smem;           // [S] -- s_j, then p_j
    float* dp = scores + S;         // [S] -- dP_j
    float* red = dp + S;            // [nwarps] reduction scratch
    float* gd_buf = red + nwarps;   // [HD] cached grad_d row
    float* q_buf = gd_buf + HD;     // [HD] cached q row
    const int64_t row = blockIdx.x;
    const int64_t qh = row % QH;
    const int64_t bt = row / QH;
    const int64_t BT = gridDim.x / QH;
    const int64_t gd_base = (bt * QH + qh) * HD;

    for (int64_t d = tid; d < HD; d += blockDim.x) {
        gd_buf[d] = (float)grad_d[gd_base + d];
        q_buf[d] = (float)q[qh * HD + d];
    }
    __syncthreads();

    // Phase 1: compute s_j = dot(q, K[j])/sqrt_d and dP_j = dot(dO, V[j])
    const float lse_val = lse[bt * QH + qh];
    for (int64_t s = wid; s < S; s += nwarps) {
        float s_partial = 0.0f;
        const int64_t k_base = ((bt * S + s) * QH + qh) * HD;
        for (int64_t d = lane; d < HD; d += 32)
            s_partial += q_buf[d] * (float)hist_k[k_base + d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s_partial += __shfl_down_sync(0xffffffffu, s_partial, off);
        if (lane == 0) scores[s] = s_partial * inv_sqrt_hd;

        float dp_partial = 0.0f;
        const int64_t v_base = ((bt * S + s) * QH + qh) * HD;
        for (int64_t d = lane; d < HD; d += 32)
            dp_partial += gd_buf[d] * (float)hist_v[v_base + d];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dp_partial += __shfl_down_sync(0xffffffffu, dp_partial, off);
        if (lane == 0) dp[s] = dp_partial;
    }
    __syncthreads();

    // Phase 2: p_j = exp(s_j - lse), then D = Sigma_j p_j * dP_j
    for (int64_t s = tid; s < S; s += blockDim.x)
        scores[s] = expf(scores[s] - lse_val);
    __syncthreads();

    {
        float partial = 0.0f;
        for (int64_t s = tid; s < S; s += blockDim.x)
            partial += scores[s] * dp[s];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xffffffffu, partial, off);
        if (lane == 0) red[wid] = partial;
        __syncthreads();
        if (wid == 0) {
            float block_part = (lane < nwarps) ? red[lane] : 0.0f;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                block_part += __shfl_down_sync(0xffffffffu, block_part, off);
            if (lane == 0) D_val = block_part;
        }
        __syncthreads();
    }

    // Phase 3: dS_j = p_j * (dP_j - D) / sqrt_d ; write gV, gK, accumulate gq
    for (int64_t d = tid; d < HD; d += blockDim.x) {
        const float gdv = gd_buf[d];
        const float qv = q_buf[d];
        float gq_acc = 0.0f;
        for (int64_t s = 0; s < S; ++s) {
            const float p_j = scores[s];
            const float dS_j = p_j * (dp[s] - D_val) * inv_sqrt_hd;
            const int64_t idx = ((bt * S + s) * QH + qh) * HD + d;
            grad_v[idx] = (scalar_t)(p_j * gdv);
            grad_k[idx] = (scalar_t)(dS_j * qv);
            gq_acc += dS_j * (float)hist_k[idx];
        }
        grad_q_partial[(qh * HD + d) * BT + bt] = gq_acc;
    }
}

// -- new stacked (FA-style) kernels: forward outputs O+LSE, backward uses LSE --
// Warp-per-row, vectorized-by-4, zero shared memory, zero block sync.
// Templated on S so the history loop is fully unrolled.
// HD must be divisible by 4; S must be <= 32 (common "layers" case).
// Fallback block-per-row kernels above are used for S > 32 or HD % 4 != 0.

template <typename scalar_t, int S, int NWARPS>
__global__ void depth_attn_stacked_fwd_warp_kernel(
    const scalar_t* __restrict__ q,       // [QH, HD]
    const scalar_t* __restrict__ hist_k,  // [B, T, S, QH, HD]
    const scalar_t* __restrict__ hist_v,  // [B, T, S, QH, HD]
    scalar_t* __restrict__ d_out,         // [B, T, QH, HD]
    float* __restrict__ lse_out,          // [B, T, QH]
    int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd) {

    using V4 = V4Ops<scalar_t>;
    using Vec = typename V4::Vec;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int d_start = lane << 2;                    // each lane owns 4 contiguous d's
    const bool active = d_start < HD;
    const int64_t warp_id = (int64_t)blockIdx.x * NWARPS + wid;
    if (warp_id >= (int64_t)BT * QH) return;
    const int64_t qh = warp_id % QH;
    const int64_t bt = warp_id / QH;

    const scalar_t* q_row = q + (int64_t)qh * HD;
    Vec qv = active ? V4::load(q_row + d_start) : V4::zero();
    float q0, q1, q2, q3;
    V4::to_floats(qv, q0, q1, q2, q3);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float m = __int_as_float(0xff800000);  // -inf
    float l = 0.0f;

    #pragma unroll
    for (int s = 0; s < S; ++s) {
        const scalar_t* k_row = hist_k + (((bt * S + s) * QH + qh) * HD);
        const scalar_t* v_row = hist_v + (((bt * S + s) * QH + qh) * HD);

        Vec kv = active ? V4::load(k_row + d_start) : V4::zero();
        float k0, k1, k2, k3;
        V4::to_floats(kv, k0, k1, k2, k3);

        float s_dot = 0.0f;
        if (active) s_dot += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            s_dot += __shfl_down_sync(0xffffffffu, s_dot, off);
        s_dot = __shfl_sync(0xffffffffu, s_dot, 0) * inv_sqrt_hd;

        const float new_m = fmaxf(m, s_dot);
        const float scale = expf(m - new_m);
        const float p_j = expf(s_dot - new_m);
        l = l * scale + p_j;

        Vec vv = active ? V4::load(v_row + d_start) : V4::zero();
        float v0, v1, v2, v3;
        V4::to_floats(vv, v0, v1, v2, v3);
        if (active) {
            acc0 = acc0 * scale + p_j * v0;
            acc1 = acc1 * scale + p_j * v1;
            acc2 = acc2 * scale + p_j * v2;
            acc3 = acc3 * scale + p_j * v3;
        }
        m = new_m;
    }

    const float inv_l = 1.0f / l;
    if (active) {
        Vec ov = V4::from_floats(acc0 * inv_l, acc1 * inv_l, acc2 * inv_l, acc3 * inv_l);
        V4::store(d_out + ((bt * QH + qh) * HD + d_start), ov);
    }
    if (lane == 0)
        lse_out[bt * QH + qh] = m + logf(l);
}

template <typename scalar_t, int S, int NWARPS>
__global__ void depth_attn_stacked_bwd_warp_kernel(
    const scalar_t* __restrict__ grad_d,  // [B, T, QH, HD]
    const scalar_t* __restrict__ q,       // [QH, HD]
    const scalar_t* __restrict__ hist_k,  // [B, T, S, QH, HD]
    const scalar_t* __restrict__ hist_v,  // [B, T, S, QH, HD]
    const float* __restrict__ lse,        // [B, T, QH]
    float* __restrict__ grad_q_partial,   // [QH, HD, B*T]
    scalar_t* __restrict__ grad_k,        // [B, T, S, QH, HD]
    scalar_t* __restrict__ grad_v,        // [B, T, S, QH, HD]
    int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd) {

    using V4 = V4Ops<scalar_t>;
    using Vec = typename V4::Vec;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int d_start = lane << 2;
    const bool active = d_start < HD;
    const int64_t warp_id = (int64_t)blockIdx.x * NWARPS + wid;
    if (warp_id >= (int64_t)BT * QH) return;
    const int64_t qh = warp_id % QH;
    const int64_t bt = warp_id / QH;

    const float lse_val = lse[bt * QH + qh];

    const scalar_t* q_row = q + (int64_t)qh * HD;
    const scalar_t* gd_row = grad_d + ((bt * QH + qh) * HD);
    Vec qv = active ? V4::load(q_row + d_start) : V4::zero();
    Vec gdv = active ? V4::load(gd_row + d_start) : V4::zero();
    float q0, q1, q2, q3, gd0, gd1, gd2, gd3;
    V4::to_floats(qv, q0, q1, q2, q3);
    V4::to_floats(gdv, gd0, gd1, gd2, gd3);

    // Pass 1: each lane j (j < S) stores s_j and dP_j in its own registers.
    float s_reg = 0.0f, dp_reg = 0.0f;
    #pragma unroll
    for (int j = 0; j < S; ++j) {
        const scalar_t* k_row = hist_k + (((bt * S + j) * QH + qh) * HD);
        const scalar_t* v_row = hist_v + (((bt * S + j) * QH + qh) * HD);

        Vec kv = active ? V4::load(k_row + d_start) : V4::zero();
        Vec vv = active ? V4::load(v_row + d_start) : V4::zero();
        float k0, k1, k2, k3, v0, v1, v2, v3;
        V4::to_floats(kv, k0, k1, k2, k3);
        V4::to_floats(vv, v0, v1, v2, v3);

        float s_dot = 0.0f, dp_dot = 0.0f;
        if (active) {
            s_dot += q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
            dp_dot += gd0 * v0 + gd1 * v1 + gd2 * v2 + gd3 * v3;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s_dot += __shfl_down_sync(0xffffffffu, s_dot, off);
            dp_dot += __shfl_down_sync(0xffffffffu, dp_dot, off);
        }
        s_dot = __shfl_sync(0xffffffffu, s_dot, 0) * inv_sqrt_hd;
        dp_dot = __shfl_sync(0xffffffffu, dp_dot, 0);

        if (lane == j) {
            s_reg = s_dot;
            dp_reg = dp_dot;
        }
    }

    // D = Sigma_j p_j * dP_j via warp reduction.
    float pdp = 0.0f;
    if (lane < S) {
        const float p_j = expf(s_reg - lse_val);
        pdp = p_j * dp_reg;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        pdp += __shfl_down_sync(0xffffffffu, pdp, off);
    const float D = __shfl_sync(0xffffffffu, pdp, 0);

    // Pass 2: write gV, gK, accumulate gq.
    float gq0 = 0.0f, gq1 = 0.0f, gq2 = 0.0f, gq3 = 0.0f;
    #pragma unroll
    for (int j = 0; j < S; ++j) {
        const float s_j = __shfl_sync(0xffffffffu, s_reg, j);
        const float dp_j = __shfl_sync(0xffffffffu, dp_reg, j);
        const float p_j = expf(s_j - lse_val);
        const float dS_j = p_j * (dp_j - D) * inv_sqrt_hd;

        const scalar_t* k_row = hist_k + (((bt * S + j) * QH + qh) * HD);
        Vec kv = active ? V4::load(k_row + d_start) : V4::zero();
        float k0, k1, k2, k3;
        V4::to_floats(kv, k0, k1, k2, k3);

        if (active) {
            Vec gvv = V4::from_floats(p_j * gd0, p_j * gd1, p_j * gd2, p_j * gd3);
            V4::store(grad_v + (((bt * S + j) * QH + qh) * HD + d_start), gvv);

            Vec gkv = V4::from_floats(dS_j * q0, dS_j * q1, dS_j * q2, dS_j * q3);
            V4::store(grad_k + (((bt * S + j) * QH + qh) * HD + d_start), gkv);

            gq0 += dS_j * k0;
            gq1 += dS_j * k1;
            gq2 += dS_j * k2;
            gq3 += dS_j * k3;
        }
    }

    if (active) {
        const int64_t base = ((int64_t)qh * HD + d_start) * BT + bt;
        grad_q_partial[base] = gq0;
        grad_q_partial[base + BT] = gq1;
        grad_q_partial[base + 2 * BT] = gq2;
        grad_q_partial[base + 3 * BT] = gq3;
    }
}

// Warp-per-row reduction of grad_q_partial over BT.
// One warp per (qh, d); 4 warps per block.
__global__ void depth_attn_grad_q_reduce_kernel(
    const float* __restrict__ grad_q_partial,  // [QH, HD, B*T]
    float* __restrict__ grad_q,                // [QH, HD]
    int64_t BT, int64_t QH, int64_t HD) {

    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int NWARPS = blockDim.x >> 5;
    const int64_t col = (int64_t)blockIdx.x * NWARPS + wid;
    if (col >= (int64_t)QH * HD) return;
    const int64_t qh = col / HD;
    const int64_t d = col - qh * HD;

    const float* row = grad_q_partial + ((qh * HD + d) * BT);
    float sum = 0.0f;
    for (int64_t bt = lane; bt < BT; bt += 32)
        sum += row[bt];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) grad_q[col] = sum;
}
