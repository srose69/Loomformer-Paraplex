// kernels/tria/gate_slot_mix/gate_slot_mix_kernel.cu -- standalone compilable TU
// wrapping just the device kernel(s) above. No torch dependency, so this
// compiles alone with plain nvcc for quick PTX/SASS inspection, e.g.:
//   nvcc -arch=sm_61 --ptx gate_slot_mix_kernel.cu -o gate_slot_mix_kernel.ptx
#include "gate_slot_mix_kernel.cuh"

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
template __global__ void gate_slot_mix_forward_kernel<float>(const float* __restrict__ carry,   
    const float* __restrict__ w,       
    float* __restrict__ p,             
    int64_t n);
template __global__ void gate_slot_mix_backward_kernel<float>(const float* __restrict__ grad_p,     
    const float* __restrict__ carry,      
    const float* __restrict__ w,          
    float* __restrict__ grad_carry,       
    float* __restrict__ grad_w_acc,          
    int64_t n);
