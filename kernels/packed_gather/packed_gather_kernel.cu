#include <ATen/ATen.h>
#include "packed_gather_kernel.cuh"

template __global__ void packed_gather_forward_kernel<float>(
    const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    float*, int64_t, int64_t, int64_t);
template __global__ void packed_gather_forward_kernel<at::Half>(
    const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    at::Half*, int64_t, int64_t, int64_t);
template __global__ void packed_gather_forward_kernel<at::BFloat16>(
    const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    at::BFloat16*, int64_t, int64_t, int64_t);

template __global__ void packed_gather_backward_kernel<float>(
    const float*, const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    int64_t, int64_t, int64_t);
template __global__ void packed_gather_backward_kernel<at::Half>(
    const at::Half*, const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    int64_t, int64_t, int64_t);
template __global__ void packed_gather_backward_kernel<at::BFloat16>(
    const at::BFloat16*, const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    int64_t, int64_t, int64_t);

template __global__ void packed_gather_pair_forward_kernel<float>(
    const int64_t*, const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    float*, int64_t, int64_t, int64_t, int64_t);
template __global__ void packed_gather_pair_forward_kernel<at::Half>(
    const int64_t*, const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    at::Half*, int64_t, int64_t, int64_t, int64_t);
template __global__ void packed_gather_pair_forward_kernel<at::BFloat16>(
    const int64_t*, const int64_t*, const int32_t*, const int32_t*, const int32_t*,
    at::BFloat16*, int64_t, int64_t, int64_t, int64_t);

template __global__ void packed_gather_pair_backward_kernel<float>(
    const float*, const int64_t*, const int64_t*, const int32_t*, const int32_t*,
    const int32_t*, int64_t, int64_t, int64_t, int64_t);
template __global__ void packed_gather_pair_backward_kernel<at::Half>(
    const at::Half*, const int64_t*, const int64_t*, const int32_t*, const int32_t*,
    const int32_t*, int64_t, int64_t, int64_t, int64_t);
template __global__ void packed_gather_pair_backward_kernel<at::BFloat16>(
    const at::BFloat16*, const int64_t*, const int64_t*, const int32_t*, const int32_t*,
    const int32_t*, int64_t, int64_t, int64_t, int64_t);
