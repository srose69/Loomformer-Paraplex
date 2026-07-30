#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

#include "packed_gather_kernel.cuh"

static torch::Tensor device_pointer_table(
    const std::vector<torch::Tensor>& tensors,
    const torch::Device& device) {
    auto host = torch::empty(
        {(int64_t)tensors.size()},
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true));
    auto ptr = host.data_ptr<int64_t>();
    for (size_t i = 0; i < tensors.size(); ++i) {
        ptr[i] = reinterpret_cast<int64_t>(tensors[i].data_ptr());
    }
    auto gpu = torch::empty(
        {(int64_t)tensors.size()},
        torch::TensorOptions().dtype(torch::kInt64).device(device));
    gpu.copy_(host, true);
    return gpu;
}

static void check_metadata(
    const torch::Tensor& selectors,
    const torch::Tensor& destinations,
    const torch::Tensor& piece_offsets,
    int64_t pieces,
    const torch::Device& device) {
    TORCH_CHECK(selectors.is_cuda() && destinations.is_cuda() && piece_offsets.is_cuda(),
                "packed_gather metadata must be CUDA tensors");
    TORCH_CHECK(selectors.device() == device && destinations.device() == device
                && piece_offsets.device() == device,
                "packed_gather metadata and chunks must share a device");
    TORCH_CHECK(selectors.scalar_type() == torch::kInt32
                && destinations.scalar_type() == torch::kInt32
                && piece_offsets.scalar_type() == torch::kInt32,
                "packed_gather metadata must be int32");
    TORCH_CHECK(selectors.is_contiguous() && destinations.is_contiguous()
                && piece_offsets.is_contiguous(),
                "packed_gather metadata must be contiguous");
    TORCH_CHECK(selectors.numel() == destinations.numel(),
                "selector/destination length mismatch");
    TORCH_CHECK(piece_offsets.numel() == pieces + 1,
                "piece_offsets must have number_of_chunks+1 entries");
}

torch::Tensor packed_gather_forward_cuda(
    std::vector<torch::Tensor> chunks,
    torch::Tensor selectors,
    torch::Tensor destinations,
    torch::Tensor piece_offsets) {
    TORCH_CHECK(!chunks.empty(), "packed_gather needs at least one chunk");
    auto device = chunks[0].device();
    TORCH_CHECK(device.is_cuda(), "packed_gather chunks must be CUDA tensors");
    c10::cuda::CUDAGuard guard(device);
    auto dtype = chunks[0].scalar_type();
    TORCH_CHECK(chunks[0].dim() == 4, "packed_gather chunks must be [B,T,H,D]");
    int64_t B = chunks[0].size(0), H = chunks[0].size(2), D = chunks[0].size(3);
    for (const auto& chunk : chunks) {
        TORCH_CHECK(chunk.is_cuda() && chunk.device() == device,
                    "all packed_gather chunks must share one CUDA device");
        TORCH_CHECK(chunk.scalar_type() == dtype && chunk.dim() == 4,
                    "all packed_gather chunks must share dtype/rank");
        TORCH_CHECK(chunk.size(0) == B && chunk.size(2) == H && chunk.size(3) == D,
                    "packed_gather chunk shape mismatch");
        TORCH_CHECK(chunk.is_contiguous(), "packed_gather chunks must be contiguous");
    }
    check_metadata(selectors, destinations, piece_offsets, chunks.size(), device);
    auto ptrs = device_pointer_table(chunks, device);
    auto output = torch::empty({selectors.numel(), H, D}, chunks[0].options());
    int64_t inner = H * D;
    int64_t work = selectors.numel() * inner;
    int threads = 256;
    int64_t blocks = (work + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, dtype,
        "packed_gather_forward_cuda", ([&] {
            packed_gather_forward_kernel<scalar_t><<<
                blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                ptrs.data_ptr<int64_t>(),
                selectors.data_ptr<int32_t>(),
                destinations.data_ptr<int32_t>(),
                piece_offsets.data_ptr<int32_t>(),
                output.data_ptr<scalar_t>(),
                selectors.numel(), chunks.size(), inner);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> packed_gather_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor selectors,
    torch::Tensor destinations,
    torch::Tensor piece_offsets,
    std::vector<int64_t> chunk_lengths,
    int64_t batch,
    int64_t heads,
    int64_t head_dim) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "packed_gather grad_output must be contiguous CUDA");
    auto device = grad_output.device();
    c10::cuda::CUDAGuard guard(device);
    check_metadata(selectors, destinations, piece_offsets, chunk_lengths.size(), device);
    TORCH_CHECK(grad_output.dim() == 3
                && grad_output.size(0) == selectors.numel()
                && grad_output.size(1) == heads
                && grad_output.size(2) == head_dim,
                "packed_gather grad_output shape mismatch");

    int64_t total_elements = 0;
    for (int64_t length : chunk_lengths) {
        TORCH_CHECK(length >= 0, "packed_gather chunk lengths must be non-negative");
        total_elements += batch * length * heads * head_dim;
    }
    TORCH_CHECK(total_elements > 0,
                "packed_gather cannot backpropagate into an all-empty history");
    auto storage = torch::zeros({total_elements}, grad_output.options());
    std::vector<torch::Tensor> grads;
    grads.reserve(chunk_lengths.size());
    int64_t offset = 0;
    for (int64_t length : chunk_lengths) {
        int64_t elements = batch * length * heads * head_dim;
        grads.push_back(storage.narrow(0, offset, elements).view(
            {batch, length, heads, head_dim}));
        offset += elements;
    }
    auto ptrs = device_pointer_table(grads, device);
    int64_t inner = heads * head_dim;
    int64_t work = selectors.numel() * inner;
    int threads = 256;
    int64_t blocks = (work + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, grad_output.scalar_type(),
        "packed_gather_backward_cuda", ([&] {
            packed_gather_backward_kernel<scalar_t><<<
                blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output.data_ptr<scalar_t>(),
                ptrs.data_ptr<int64_t>(),
                selectors.data_ptr<int32_t>(),
                destinations.data_ptr<int32_t>(),
                piece_offsets.data_ptr<int32_t>(),
                selectors.numel(), chunk_lengths.size(), inner);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grads;
}

torch::Tensor packed_gather_pair_forward_cuda(
    std::vector<torch::Tensor> k_chunks,
    std::vector<torch::Tensor> v_chunks,
    torch::Tensor selectors,
    torch::Tensor destinations,
    torch::Tensor piece_offsets) {
    TORCH_CHECK(!k_chunks.empty() && k_chunks.size() == v_chunks.size(),
                "packed_gather_pair needs matching K/V chunks");
    auto device = k_chunks[0].device();
    TORCH_CHECK(device.is_cuda(), "packed_gather_pair chunks must be CUDA");
    c10::cuda::CUDAGuard guard(device);
    auto dtype = k_chunks[0].scalar_type();
    TORCH_CHECK(k_chunks[0].dim() == 4, "packed_gather_pair chunks must be [B,T,H,D]");
    int64_t B = k_chunks[0].size(0), H = k_chunks[0].size(2), D = k_chunks[0].size(3);
    for (size_t i = 0; i < k_chunks.size(); ++i) {
        const auto& k = k_chunks[i];
        const auto& v = v_chunks[i];
        TORCH_CHECK(k.is_cuda() && v.is_cuda() && k.device() == device && v.device() == device,
                    "packed_gather_pair chunks must share a CUDA device");
        TORCH_CHECK(k.scalar_type() == dtype && v.scalar_type() == dtype
                    && k.sizes() == v.sizes(),
                    "packed_gather_pair K/V dtype or shape mismatch");
        TORCH_CHECK(k.size(0) == B && k.size(2) == H && k.size(3) == D,
                    "packed_gather_pair chunk shape mismatch");
        TORCH_CHECK(k.is_contiguous() && v.is_contiguous(),
                    "packed_gather_pair chunks must be contiguous");
    }
    check_metadata(selectors, destinations, piece_offsets, k_chunks.size(), device);
    auto k_ptrs = device_pointer_table(k_chunks, device);
    auto v_ptrs = device_pointer_table(v_chunks, device);
    auto output = torch::empty({selectors.numel(), H, 2 * D}, k_chunks[0].options());
    int64_t inner = H * D;
    int64_t work = selectors.numel() * inner;
    int threads = 256;
    int64_t blocks = (work + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, dtype,
        "packed_gather_pair_forward_cuda", ([&] {
            packed_gather_pair_forward_kernel<scalar_t><<<
                blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                k_ptrs.data_ptr<int64_t>(), v_ptrs.data_ptr<int64_t>(),
                selectors.data_ptr<int32_t>(), destinations.data_ptr<int32_t>(),
                piece_offsets.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
                selectors.numel(), k_chunks.size(), inner, D);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<torch::Tensor> packed_gather_pair_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor selectors,
    torch::Tensor destinations,
    torch::Tensor piece_offsets,
    std::vector<int64_t> chunk_lengths,
    int64_t batch,
    int64_t heads,
    int64_t head_dim) {
    TORCH_CHECK(grad_output.is_cuda() && grad_output.is_contiguous(),
                "packed_gather_pair grad_output must be contiguous CUDA");
    auto device = grad_output.device();
    c10::cuda::CUDAGuard guard(device);
    check_metadata(selectors, destinations, piece_offsets, chunk_lengths.size(), device);
    TORCH_CHECK(grad_output.dim() == 3
                && grad_output.size(0) == selectors.numel()
                && grad_output.size(1) == heads
                && grad_output.size(2) == 2 * head_dim,
                "packed_gather_pair grad_output shape mismatch");

    int64_t total_elements = 0;
    for (int64_t length : chunk_lengths) {
        TORCH_CHECK(length >= 0,
                    "packed_gather_pair chunk lengths must be non-negative");
        total_elements += batch * length * heads * head_dim;
    }
    TORCH_CHECK(total_elements > 0,
                "packed_gather_pair cannot backpropagate into an all-empty history");
    auto grad_k_storage = torch::zeros({total_elements}, grad_output.options());
    auto grad_v_storage = torch::zeros({total_elements}, grad_output.options());
    std::vector<torch::Tensor> grad_k, grad_v;
    grad_k.reserve(chunk_lengths.size());
    grad_v.reserve(chunk_lengths.size());
    int64_t offset = 0;
    for (int64_t length : chunk_lengths) {
        int64_t elements = batch * length * heads * head_dim;
        grad_k.push_back(grad_k_storage.narrow(0, offset, elements).view(
            {batch, length, heads, head_dim}));
        grad_v.push_back(grad_v_storage.narrow(0, offset, elements).view(
            {batch, length, heads, head_dim}));
        offset += elements;
    }
    auto grad_k_ptrs = device_pointer_table(grad_k, device);
    auto grad_v_ptrs = device_pointer_table(grad_v, device);
    int64_t inner = heads * head_dim;
    int64_t work = selectors.numel() * inner;
    int threads = 256;
    int64_t blocks = (work + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, grad_output.scalar_type(),
        "packed_gather_pair_backward_cuda", ([&] {
            packed_gather_pair_backward_kernel<scalar_t><<<
                blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_output.data_ptr<scalar_t>(),
                grad_k_ptrs.data_ptr<int64_t>(), grad_v_ptrs.data_ptr<int64_t>(),
                selectors.data_ptr<int32_t>(), destinations.data_ptr<int32_t>(),
                piece_offsets.data_ptr<int32_t>(), selectors.numel(),
                chunk_lengths.size(), inner, head_dim);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::vector<torch::Tensor> grads;
    grads.reserve(2 * chunk_lengths.size());
    grads.insert(grads.end(), grad_k.begin(), grad_k.end());
    grads.insert(grads.end(), grad_v.begin(), grad_v.end());
    return grads;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &packed_gather_forward_cuda);
    m.def("backward", &packed_gather_backward_cuda);
    m.def("forward_pair", &packed_gather_pair_forward_cuda);
    m.def("backward_pair", &packed_gather_pair_backward_cuda);
}
