#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "temporal_carry_kernel.cuh"

template __global__ void temporal_carry_forward_kernel<float>(
    const float*, const bool*, float*, float*, int64_t, int64_t, int64_t);
template __global__ void temporal_carry_backward_kernel<float>(
    const float*, const float*, const float*, const float*, const bool*, float*,
    int64_t, int64_t, int64_t);
