// kernels/beta_space/beta_space_kernel.cu -- standalone compilable TU.
// Still needs c10/util/BFloat16.h on the include path (see .cuh comment).
#include "beta_space_kernel.cuh"

// Explicit instantiation for float -- WITHOUT this, nvcc --ptx on a
// translation unit that only #includes a template definition emits NO
// device code at all (just PTX header boilerplate, ~200 bytes, no
// .visible .entry) because nothing ever asked for a concrete
// specialization. float is instantiated here because it's the actual
// production dtype for this codebase's target GPUs (see the profiling
// history this kernels/ split came out of); add more `template __global__
// void ...<at::Half>(...)`/`<at::BFloat16>(...)` lines here if you need
// PTX for those paths too (note: at::Half/at::BFloat16 need ATen headers,
// which most of these kernel.cuh files deliberately don't include --
// add the include here in kernel.cu, not in the .cuh, to keep the .cuh
// torch-free).
template __global__ void pack_r_mask4_head_kernel<float>(const float* __restrict__ u,
    const float* __restrict__ q_h,
    const float* __restrict__ k_ctx_h,
    const float* __restrict__ c_h,
    const float* __restrict__ d_h,
    float* __restrict__ r_pack,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads);
template __global__ void pack_r_mask4_open_kernel<float>(const float* __restrict__ u,
    const float* __restrict__ q_h,
    const float* __restrict__ k_ctx_h,
    const float* __restrict__ c_h,
    const float* __restrict__ d_h,
    float* __restrict__ r_pack,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads);
template __global__ void unpack_grad_r_mask4_head_kernel<float>(const float* __restrict__ grad_r_pack,
    float* __restrict__ grad_u,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_c,
    float* __restrict__ grad_d,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads);
template __global__ void unpack_grad_r_mask4_open_kernel<float>(const float* __restrict__ grad_r_pack,
    float* __restrict__ grad_u,
    float* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_c,
    float* __restrict__ grad_d,
    int64_t M, int64_t N, int64_t K,
    int64_t head_dim, int64_t n_q_heads);
