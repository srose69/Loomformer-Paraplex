#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename scalar_t>
__global__ void packed_gather_forward_kernel(
    const int64_t* __restrict__ input_ptrs,
    const int32_t* __restrict__ selectors,
    const int32_t* __restrict__ destinations,
    const int32_t* __restrict__ piece_offsets,
    scalar_t* __restrict__ output,
    int64_t source_tokens,
    int64_t pieces,
    int64_t inner) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = source_tokens * inner;
    if (linear >= total) return;
    int64_t source_token = linear / inner;
    int64_t feature = linear - source_token * inner;

    int lo = 0;
    int hi = (int)pieces;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((int64_t)piece_offsets[mid] <= source_token) lo = mid;
        else hi = mid;
    }
    const scalar_t* input = reinterpret_cast<const scalar_t*>(
        static_cast<uintptr_t>(input_ptrs[lo]));
    int64_t source = (int64_t)selectors[source_token] * inner + feature;
    int64_t destination = (int64_t)destinations[source_token] * inner + feature;
    output[destination] = input[source];
}

template <typename scalar_t>
__global__ void packed_gather_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const int64_t* __restrict__ grad_input_ptrs,
    const int32_t* __restrict__ selectors,
    const int32_t* __restrict__ destinations,
    const int32_t* __restrict__ piece_offsets,
    int64_t source_tokens,
    int64_t pieces,
    int64_t inner) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = source_tokens * inner;
    if (linear >= total) return;
    int64_t source_token = linear / inner;
    int64_t feature = linear - source_token * inner;

    int lo = 0;
    int hi = (int)pieces;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((int64_t)piece_offsets[mid] <= source_token) lo = mid;
        else hi = mid;
    }
    scalar_t* grad_input = reinterpret_cast<scalar_t*>(
        static_cast<uintptr_t>(grad_input_ptrs[lo]));
    int64_t source = (int64_t)selectors[source_token] * inner + feature;
    int64_t destination = (int64_t)destinations[source_token] * inner + feature;
    grad_input[source] = grad_output[destination];
}

template <typename scalar_t>
__global__ void packed_gather_pair_forward_kernel(
    const int64_t* __restrict__ k_ptrs,
    const int64_t* __restrict__ v_ptrs,
    const int32_t* __restrict__ selectors,
    const int32_t* __restrict__ destinations,
    const int32_t* __restrict__ piece_offsets,
    scalar_t* __restrict__ output,
    int64_t source_tokens,
    int64_t pieces,
    int64_t inner,
    int64_t head_dim) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = source_tokens * inner;
    if (linear >= total) return;
    int64_t source_token = linear / inner;
    int64_t feature = linear - source_token * inner;
    int lo = 0, hi = (int)pieces;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((int64_t)piece_offsets[mid] <= source_token) lo = mid;
        else hi = mid;
    }
    const scalar_t* k = reinterpret_cast<const scalar_t*>(
        static_cast<uintptr_t>(k_ptrs[lo]));
    const scalar_t* v = reinterpret_cast<const scalar_t*>(
        static_cast<uintptr_t>(v_ptrs[lo]));
    int64_t source = (int64_t)selectors[source_token] * inner + feature;
    int64_t head = feature / head_dim;
    int64_t feature_in_head = feature - head * head_dim;
    int64_t destination = (int64_t)destinations[source_token] * (2 * inner)
        + head * (2 * head_dim) + feature_in_head;
    output[destination] = k[source];
    output[destination + head_dim] = v[source];
}

template <typename scalar_t>
__global__ void packed_gather_pair_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const int64_t* __restrict__ grad_k_ptrs,
    const int64_t* __restrict__ grad_v_ptrs,
    const int32_t* __restrict__ selectors,
    const int32_t* __restrict__ destinations,
    const int32_t* __restrict__ piece_offsets,
    int64_t source_tokens,
    int64_t pieces,
    int64_t inner,
    int64_t head_dim) {
    int64_t linear = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = source_tokens * inner;
    if (linear >= total) return;
    int64_t source_token = linear / inner;
    int64_t feature = linear - source_token * inner;
    int lo = 0, hi = (int)pieces;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((int64_t)piece_offsets[mid] <= source_token) lo = mid;
        else hi = mid;
    }
    scalar_t* grad_k = reinterpret_cast<scalar_t*>(
        static_cast<uintptr_t>(grad_k_ptrs[lo]));
    scalar_t* grad_v = reinterpret_cast<scalar_t*>(
        static_cast<uintptr_t>(grad_v_ptrs[lo]));
    int64_t source = (int64_t)selectors[source_token] * inner + feature;
    int64_t head = feature / head_dim;
    int64_t feature_in_head = feature - head * head_dim;
    int64_t destination = (int64_t)destinations[source_token] * (2 * inner)
        + head * (2 * head_dim) + feature_in_head;
    grad_k[source] = grad_output[destination];
    grad_v[source] = grad_output[destination + head_dim];
}
