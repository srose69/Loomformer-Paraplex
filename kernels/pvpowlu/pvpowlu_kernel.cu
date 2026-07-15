// kernels/pvpowlu/pvpowlu_kernel.cu -- standalone compilable TU, e.g.:
//   nvcc -arch=sm_61 --ptx pvpowlu_kernel.cu -o pvpowlu_kernel.ptx
#include "pvpowlu_kernel.cuh"

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
template __global__ void pvpowlu_fwd_kernel<float>(const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ out,
    float m,
    int64_t n);
template __global__ void pvpowlu_bwd_kernel<float>(const float* __restrict__ grad_out,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ grad_x1,
    float* __restrict__ grad_x2,
    float m,
    int64_t n);
