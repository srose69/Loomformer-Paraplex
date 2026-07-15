// kernels/tria/tria_step_gate/tria_step_gate_kernel.cu -- standalone compilable TU
// wrapping just the device kernel(s) above. No torch dependency, so this
// compiles alone with plain nvcc for quick PTX/SASS inspection, e.g.:
//   nvcc -arch=sm_61 --ptx tria_step_gate_kernel.cu -o tria_step_gate_kernel.ptx
#include "tria_step_gate_kernel.cuh"

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
template __global__ void tria_step_gate_forward_kernel<float>(const float* __restrict__ r, const float* __restrict__ i_, const float* __restrict__ o,
    const float* __restrict__ carry_prev,
    const float* __restrict__ w,
    float* __restrict__ carry_new,
    float* __restrict__ p_out,
    float* __restrict__ scale_out,
    float alpha, int axis, int64_t n);
template __global__ void tria_step_gate_backward_kernel<float>(const float* __restrict__ grad_carry_new,
    const float* __restrict__ grad_p_out,
    const float* __restrict__ r, const float* __restrict__ i_, const float* __restrict__ o,
    const float* __restrict__ carry_prev,
    const float* __restrict__ w,
    const float* __restrict__ scale,
    float* __restrict__ grad_r, float* __restrict__ grad_i, float* __restrict__ grad_o,
    float* __restrict__ grad_carry_prev,
    float* __restrict__ grad_w_acc,
    float alpha, int axis, int64_t n);
