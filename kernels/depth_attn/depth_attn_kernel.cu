// kernels/depth_attn/depth_attn_kernel.cu -- standalone compilable TU.
#include "depth_attn_kernel.cuh"

// S dispatch table for the templated warp-per-row kernels.
#define DEPTH_ATTN_S_CASES(M) \
    M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) \
    M(11) M(12) M(13) M(14) M(15) M(16) M(17) M(18) M(19) M(20) \
    M(21) M(22) M(23) M(24) M(25) M(26) M(27) M(28) M(29) M(30) \
    M(31) M(32)

// Legacy block-per-row kernels (fallback for S > 32 or HD % 4 != 0).
template __global__ void depth_attn_fwd_kernel<float>(const float* __restrict__ q,
    const float* __restrict__ hist_k,
    const float* __restrict__ hist_v,
    float* __restrict__ d_out,
    float* __restrict__ w_out,
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd);

template __global__ void depth_attn_bwd_kernel<float>(const float* __restrict__ grad_d,
    const float* __restrict__ q,
    const float* __restrict__ hist_k,
    const float* __restrict__ hist_v,
    const float* __restrict__ w,
    float* __restrict__ grad_q_partial,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd);

template __global__ void depth_attn_stacked_fwd_kernel<float>(const float* __restrict__ q,
    const float* __restrict__ hist_k,
    const float* __restrict__ hist_v,
    float* __restrict__ d_out,
    float* __restrict__ lse_out,
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd);

template __global__ void depth_attn_stacked_bwd_kernel<float>(const float* __restrict__ grad_d,
    const float* __restrict__ q,
    const float* __restrict__ hist_k,
    const float* __restrict__ hist_v,
    const float* __restrict__ lse,
    float* __restrict__ grad_q_partial,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    int64_t S, int64_t QH, int64_t HD, float inv_sqrt_hd);

// Warp-per-row stacked kernels: float, S=1..32, NWARPS=4.
#define INSTANTIATE_FWD_WARP_FLOAT(S) \
    template __global__ void depth_attn_stacked_fwd_warp_kernel<float, S, 4>( \
        const float* __restrict__ q, const float* __restrict__ hist_k, \
        const float* __restrict__ hist_v, float* __restrict__ d_out, \
        float* __restrict__ lse_out, int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd);
DEPTH_ATTN_S_CASES(INSTANTIATE_FWD_WARP_FLOAT)
#undef INSTANTIATE_FWD_WARP_FLOAT

#define INSTANTIATE_BWD_WARP_FLOAT(S) \
    template __global__ void depth_attn_stacked_bwd_warp_kernel<float, S, 4>( \
        const float* __restrict__ grad_d, const float* __restrict__ q, \
        const float* __restrict__ hist_k, const float* __restrict__ hist_v, \
        const float* __restrict__ lse, float* __restrict__ grad_q_partial, \
        float* __restrict__ grad_k, float* __restrict__ grad_v, \
        int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd);
DEPTH_ATTN_S_CASES(INSTANTIATE_BWD_WARP_FLOAT)
#undef INSTANTIATE_BWD_WARP_FLOAT

// Warp-per-row stacked kernels: bfloat16, S=1..32, NWARPS=4.
#define INSTANTIATE_FWD_WARP_BF16(S) \
    template __global__ void depth_attn_stacked_fwd_warp_kernel<at::BFloat16, S, 4>( \
        const at::BFloat16* __restrict__ q, const at::BFloat16* __restrict__ hist_k, \
        const at::BFloat16* __restrict__ hist_v, at::BFloat16* __restrict__ d_out, \
        float* __restrict__ lse_out, int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd);
DEPTH_ATTN_S_CASES(INSTANTIATE_FWD_WARP_BF16)
#undef INSTANTIATE_FWD_WARP_BF16

#define INSTANTIATE_BWD_WARP_BF16(S) \
    template __global__ void depth_attn_stacked_bwd_warp_kernel<at::BFloat16, S, 4>( \
        const at::BFloat16* __restrict__ grad_d, const at::BFloat16* __restrict__ q, \
        const at::BFloat16* __restrict__ hist_k, const at::BFloat16* __restrict__ hist_v, \
        const float* __restrict__ lse, float* __restrict__ grad_q_partial, \
        at::BFloat16* __restrict__ grad_k, at::BFloat16* __restrict__ grad_v, \
        int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd);
DEPTH_ATTN_S_CASES(INSTANTIATE_BWD_WARP_BF16)
#undef INSTANTIATE_BWD_WARP_BF16

// Warp-per-row stacked kernels: float16, S=1..32, NWARPS=4.
#define INSTANTIATE_FWD_WARP_FP16(S) \
    template __global__ void depth_attn_stacked_fwd_warp_kernel<at::Half, S, 4>( \
        const at::Half* __restrict__ q, const at::Half* __restrict__ hist_k, \
        const at::Half* __restrict__ hist_v, at::Half* __restrict__ d_out, \
        float* __restrict__ lse_out, int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd);
DEPTH_ATTN_S_CASES(INSTANTIATE_FWD_WARP_FP16)
#undef INSTANTIATE_FWD_WARP_FP16

#define INSTANTIATE_BWD_WARP_FP16(S) \
    template __global__ void depth_attn_stacked_bwd_warp_kernel<at::Half, S, 4>( \
        const at::Half* __restrict__ grad_d, const at::Half* __restrict__ q, \
        const at::Half* __restrict__ hist_k, const at::Half* __restrict__ hist_v, \
        const float* __restrict__ lse, float* __restrict__ grad_q_partial, \
        at::Half* __restrict__ grad_k, at::Half* __restrict__ grad_v, \
        int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd);
DEPTH_ATTN_S_CASES(INSTANTIATE_BWD_WARP_FP16)
#undef INSTANTIATE_BWD_WARP_FP16

#undef DEPTH_ATTN_S_CASES
