// kernels/beta_space/beta_space_kernel.cuh -- pure device code for the
// maskpack beta_space op: BF16 bit-twiddling helpers (bf16_bits_to_float /
// float_to_bf16_rn), the vectorized-load/store V4Ops<scalar_t> template (+
// float and at::BFloat16 specializations), and the 4 pack/unpack __global__
// kernels. Kept as ONE file (not split further into per-kernel trios like
// tria's groups): V4Ops and the bf16 helpers are genuinely shared,
// tightly-coupled infrastructure for all 4 kernels below, and splitting
// them apart would risk breaking that coupling for no real benefit.
//
// Needs c10/util/BFloat16.h for the at::BFloat16 type used in the V4Ops
// specialization -- that's the one ATen dependency that can't be avoided
// even in the device-only file (at::BFloat16 is a lightweight, device-
// usable type, NOT the heavy torch::Tensor/pybind machinery).
#pragma once

#include <cuda_bf16.h>
#include <c10/util/BFloat16.h>
#include <cstdint>

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
    static __device__ __forceinline__ Vec add(Vec a, Vec b) {
        a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
        return a;
    }
};

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
    static __device__ __forceinline__ Vec add(Vec a, Vec b) {
        uint64_t out = 0ULL;
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            uint16_t av = static_cast<uint16_t>((a >> (lane * 16)) & 0xffffULL);
            uint16_t bv = static_cast<uint16_t>((b >> (lane * 16)) & 0xffffULL);
            uint16_t ov = float_to_bf16_rn(bf16_bits_to_float(av) + bf16_bits_to_float(bv));
            out |= (static_cast<uint64_t>(ov) << (lane * 16));
        }
        return out;
    }
};

// r_pack[g,row,k] — sector-mask pack, vectorized by 4 logical elements.
// For float32 that is float4/16 bytes; for BF16 it is four raw 16-bit lanes/8 bytes.
template <typename scalar_t>
__global__ void pack_r_mask4_head_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ q_h,
    const scalar_t* __restrict__ k_ctx_h,
    const scalar_t* __restrict__ c_h,
    const scalar_t* __restrict__ d_h,
    scalar_t* __restrict__ r_pack,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads) {

    int64_t K4 = K >> 2;
    int64_t HD4 = head_dim >> 2;
    int64_t N4 = N >> 2;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = n_q_heads * M * K4;
    if (idx >= total) return;

    int64_t k4 = idx % K4;
    int64_t row = (idx / K4) % M;
    int64_t g = idx / (M * K4);
    int64_t base = row * N;
    int64_t c0 = g * head_dim;

    const scalar_t* src;
    int64_t dst4;
    if (k4 < HD4) {
        src = q_h + base + c0 + (k4 << 2);
        dst4 = k4;
    } else if (k4 < 2 * HD4) {
        int64_t t = k4 - HD4;
        src = k_ctx_h + base + c0 + (t << 2);
        dst4 = HD4 + t;
    } else if (k4 < 3 * HD4) {
        int64_t t = k4 - 2 * HD4;
        src = c_h + base + c0 + (t << 2);
        dst4 = 2 * HD4 + t;
    } else if (k4 < 3 * HD4 + N4) {
        int64_t t = k4 - 3 * HD4;
        src = u + base + (t << 2);
        dst4 = 3 * HD4 + t;
    } else {
        int64_t t = k4 - 3 * HD4 - N4;
        src = d_h + base + c0 + (t << 2);
        dst4 = 3 * HD4 + N4 + t;
    }

    scalar_t* dst = r_pack + (g * M + row) * K + (dst4 << 2);
    V4Ops<scalar_t>::store(dst, V4Ops<scalar_t>::load(src));
}

template <typename scalar_t>
__global__ void pack_r_mask4_open_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ q_h,
    const scalar_t* __restrict__ k_ctx_h,
    const scalar_t* __restrict__ c_h,
    const scalar_t* __restrict__ d_h,
    scalar_t* __restrict__ r_pack,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads) {

    int64_t K4 = K >> 2;
    int64_t HD4 = head_dim >> 2;
    int64_t N4 = N >> 2;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = n_q_heads * M * K4;
    if (idx >= total) return;

    int64_t k4 = idx % K4;
    int64_t row = (idx / K4) % M;
    int64_t g = idx / (M * K4);
    int64_t base = row * N;
    int64_t c0 = g * head_dim;

    const scalar_t* src;
    int64_t dst4;
    if (k4 < HD4) {
        src = q_h + base + c0 + (k4 << 2);
        dst4 = k4;
    } else if (k4 < HD4 + N4) {
        int64_t t = k4 - HD4;
        src = k_ctx_h + base + (t << 2);
        dst4 = HD4 + t;
    } else if (k4 < HD4 + 2 * N4) {
        int64_t t = k4 - HD4 - N4;
        src = c_h + base + (t << 2);
        dst4 = HD4 + N4 + t;
    } else if (k4 < HD4 + 3 * N4) {
        int64_t t = k4 - HD4 - 2 * N4;
        src = u + base + (t << 2);
        dst4 = HD4 + 2 * N4 + t;
    } else {
        int64_t t = k4 - HD4 - 3 * N4;
        src = d_h + base + (t << 2);
        dst4 = HD4 + 3 * N4 + t;
    }

    scalar_t* dst = r_pack + (g * M + row) * K + (dst4 << 2);
    V4Ops<scalar_t>::store(dst, V4Ops<scalar_t>::load(src));
}

// grad_r_pack -> five input gradients, vectorized by 4 logical elements.
// One thread owns one output vector in the original N-wide row, so the shared U/open
// accumulations across heads are deterministic and need no atomics.
template <typename scalar_t>
__global__ void unpack_grad_r_mask4_head_kernel(
    const scalar_t* __restrict__ grad_r_pack,
    scalar_t* __restrict__ grad_u,
    scalar_t* __restrict__ grad_q,
    scalar_t* __restrict__ grad_k,
    scalar_t* __restrict__ grad_c,
    scalar_t* __restrict__ grad_d,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads) {

    int64_t N4 = N >> 2;
    int64_t HD4 = head_dim >> 2;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = M * N4;
    if (idx >= total) return;

    int64_t off4 = idx % N4;
    int64_t row = idx / N4;
    int64_t g = off4 / HD4;
    int64_t local4 = off4 - g * HD4;
    int64_t off = off4 << 2;
    int64_t local = local4 << 2;

    const scalar_t* gr = grad_r_pack + (g * M + row) * K;
    V4Ops<scalar_t>::store(grad_q + row * N + off, V4Ops<scalar_t>::load(gr + local));
    V4Ops<scalar_t>::store(grad_k + row * N + off, V4Ops<scalar_t>::load(gr + head_dim + local));
    V4Ops<scalar_t>::store(grad_c + row * N + off, V4Ops<scalar_t>::load(gr + 2 * head_dim + local));
    V4Ops<scalar_t>::store(grad_d + row * N + off, V4Ops<scalar_t>::load(gr + 3 * head_dim + N + local));

    typename V4Ops<scalar_t>::Vec su = V4Ops<scalar_t>::zero();
    for (int64_t gg = 0; gg < n_q_heads; ++gg) {
        const scalar_t* gru = grad_r_pack + (gg * M + row) * K + 3 * head_dim + off;
        su = V4Ops<scalar_t>::add(su, V4Ops<scalar_t>::load(gru));
    }
    V4Ops<scalar_t>::store(grad_u + row * N + off, su);
}

template <typename scalar_t>
__global__ void unpack_grad_r_mask4_open_kernel(
    const scalar_t* __restrict__ grad_r_pack,
    scalar_t* __restrict__ grad_u,
    scalar_t* __restrict__ grad_q,
    scalar_t* __restrict__ grad_k,
    scalar_t* __restrict__ grad_c,
    scalar_t* __restrict__ grad_d,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads) {

    int64_t N4 = N >> 2;
    int64_t HD4 = head_dim >> 2;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = M * N4;
    if (idx >= total) return;

    int64_t off4 = idx % N4;
    int64_t row = idx / N4;
    int64_t g = off4 / HD4;
    int64_t local4 = off4 - g * HD4;
    int64_t off = off4 << 2;
    int64_t local = local4 << 2;

    const scalar_t* grq = grad_r_pack + (g * M + row) * K;
    V4Ops<scalar_t>::store(grad_q + row * N + off, V4Ops<scalar_t>::load(grq + local));

    typename V4Ops<scalar_t>::Vec sk = V4Ops<scalar_t>::zero();
    typename V4Ops<scalar_t>::Vec sc = V4Ops<scalar_t>::zero();
    typename V4Ops<scalar_t>::Vec su = V4Ops<scalar_t>::zero();
    typename V4Ops<scalar_t>::Vec sd = V4Ops<scalar_t>::zero();
    for (int64_t gg = 0; gg < n_q_heads; ++gg) {
        const scalar_t* gr = grad_r_pack + (gg * M + row) * K;
        sk = V4Ops<scalar_t>::add(sk, V4Ops<scalar_t>::load(gr + head_dim + off));
        sc = V4Ops<scalar_t>::add(sc, V4Ops<scalar_t>::load(gr + head_dim + N + off));
        su = V4Ops<scalar_t>::add(su, V4Ops<scalar_t>::load(gr + head_dim + 2 * N + off));
        sd = V4Ops<scalar_t>::add(sd, V4Ops<scalar_t>::load(gr + head_dim + 3 * N + off));
    }
    V4Ops<scalar_t>::store(grad_k + row * N + off, sk);
    V4Ops<scalar_t>::store(grad_c + row * N + off, sc);
    V4Ops<scalar_t>::store(grad_u + row * N + off, su);
    V4Ops<scalar_t>::store(grad_d + row * N + off, sd);
}

