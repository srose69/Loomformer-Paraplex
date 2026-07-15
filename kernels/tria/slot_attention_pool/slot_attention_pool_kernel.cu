// kernels/tria/slot_attention_pool/slot_attention_pool_kernel.cu -- standalone compilable TU
// wrapping just the device kernel(s) above. No torch dependency, so this
// compiles alone with plain nvcc for quick PTX/SASS inspection, e.g.:
//   nvcc -arch=sm_61 --ptx slot_attention_pool_kernel.cu -o slot_attention_pool_kernel.ptx
#include "slot_attention_pool_kernel.cuh"

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
template __global__ void slot_attention_pool_forward_kernel<float>(const float* __restrict__ carry,    
    const float* __restrict__ score_w,  
    float* __restrict__ pooled,         
    float* __restrict__ lse,            
    int64_t H);
template __global__ void slot_attention_pool_backward_kernel<float>(const float* __restrict__ grad_pooled,  
    const float* __restrict__ carry,        
    const float* __restrict__ score_w,      
    const float* __restrict__ lse,          
    float* __restrict__ grad_carry,         
    float* __restrict__ block_partial,         
    int64_t H);
