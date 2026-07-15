// kernels/phase_sin/phase_sin_kernel.cu -- standalone compilable TU, e.g.:
//   nvcc -arch=sm_61 --ptx phase_sin_kernel.cu -o phase_sin_kernel.ptx
#include "phase_sin_kernel.cuh"

// Explicit instantiation -- WITHOUT this, nvcc --ptx on a translation unit
// that only #includes a template definition emits NO device code at all
// (see kernels/build.py's module docstring / every other kernel.cu in this
// tree for the full explanation). float is always instantiated (matches
// the historical fp32 production path); at::BFloat16 is ALSO instantiated
// here now that phase_sin's CUDA path accepts bf16 (see phase_sin_launcher.cu) --
// L40S/Ada bf16 throughput is the whole point of that change, so its PTX
// should be inspectable same as float's, not just assumed correct.
#include <ATen/core/TensorBase.h>  // pulls in at::BFloat16's device-usable operators

template __global__ void phase_sin_fwd_kernel<float>(
    const float* __restrict__ beta, float* __restrict__ out, int64_t n);
template __global__ void phase_sin_bwd_kernel<float>(
    const float* __restrict__ beta, const float* __restrict__ grad_out,
    float* __restrict__ grad_in, float eps, int64_t n);
template __global__ void phase_sin_secant_bwd_kernel<float>(
    const float* __restrict__ beta, const float* __restrict__ grad_out,
    float* __restrict__ grad_in, float anchor, float s_anchor, float near_eps, int64_t n);

template __global__ void phase_sin_fwd_kernel<at::BFloat16>(
    const at::BFloat16* __restrict__ beta, at::BFloat16* __restrict__ out, int64_t n);
template __global__ void phase_sin_bwd_kernel<at::BFloat16>(
    const at::BFloat16* __restrict__ beta, const at::BFloat16* __restrict__ grad_out,
    at::BFloat16* __restrict__ grad_in, float eps, int64_t n);
template __global__ void phase_sin_secant_bwd_kernel<at::BFloat16>(
    const at::BFloat16* __restrict__ beta, const at::BFloat16* __restrict__ grad_out,
    at::BFloat16* __restrict__ grad_in, float anchor, float s_anchor, float near_eps, int64_t n);
