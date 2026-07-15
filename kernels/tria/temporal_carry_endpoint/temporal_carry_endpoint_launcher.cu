#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "temporal_carry_endpoint_kernel.cuh"

std::vector<torch::Tensor> temporal_carry_endpoint_forward_cuda(
    torch::Tensor depth, torch::Tensor reset, torch::Tensor initial, torch::Tensor initial_valid) {
    TORCH_CHECK(depth.is_cuda() && reset.is_cuda(), "temporal_carry_endpoint_forward_cuda: CUDA inputs required");
    TORCH_CHECK(depth.dim() == 5 && depth.size(3) == 3 && depth.size(4) == 3,
                "temporal_carry_endpoint_forward_cuda: depth must be [B,T,H,3,3]");
    TORCH_CHECK(reset.scalar_type() == torch::kBool && reset.sizes() == depth.sizes().slice(0, 2),
                "temporal_carry_endpoint_forward_cuda: reset must be [B,T] bool");
    const bool has_initial = initial.numel() != 0;
    if (has_initial) {
        TORCH_CHECK(initial.is_cuda() && initial_valid.is_cuda(), "initial state must be CUDA");
        TORCH_CHECK(initial.dim() == 4 && initial.size(0) == depth.size(0) && initial.size(1) == depth.size(2) &&
                    initial.size(2) == 3 && initial.size(3) == 3,
                    "initial must be [B,H,3,3]");
        TORCH_CHECK(initial_valid.dim() == 1 && initial_valid.size(0) == depth.size(0) && initial_valid.scalar_type() == torch::kBool,
                    "initial_valid must be [B] bool");
    }
    c10::cuda::CUDAGuard guard(depth.device());
    auto depth_c = depth.contiguous();
    auto reset_c = reset.contiguous();
    auto init_c = has_initial ? initial.contiguous() : torch::empty({0}, depth.options());
    auto valid_c = has_initial ? initial_valid.contiguous() : torch::empty({0}, depth.options().dtype(torch::kBool));
    const int64_t B = depth_c.size(0), T = depth_c.size(1), H = depth_c.size(2);
    auto endpoint = torch::empty({B, H, 3, 3}, depth.options().dtype(torch::kBFloat16));
    auto endpoint_fp32 = torch::empty({B, H, 3, 3}, depth.options().dtype(torch::kFloat32));
    const int threads = 256;
    const int64_t blocks = (B * H + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, depth_c.scalar_type(),
        "temporal_carry_endpoint_forward_cuda", ([&] {
            temporal_carry_endpoint_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                depth_c.data_ptr<scalar_t>(), reset_c.data_ptr<bool>(),
                has_initial ? init_c.data_ptr<scalar_t>() : nullptr,
                has_initial ? valid_c.data_ptr<bool>() : nullptr,
                reinterpret_cast<__nv_bfloat16*>(endpoint.data_ptr<at::BFloat16>()),
                endpoint_fp32.data_ptr<float>(), B, T, H, has_initial);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {endpoint, endpoint_fp32};
}

std::vector<torch::Tensor> temporal_carry_endpoint_backward_cuda(
    torch::Tensor grad_endpoint, torch::Tensor depth, torch::Tensor endpoint_fp32,
    torch::Tensor reset, torch::Tensor initial, torch::Tensor initial_valid) {
    TORCH_CHECK(grad_endpoint.is_cuda() && depth.is_cuda() && endpoint_fp32.is_cuda() && reset.is_cuda(),
                "temporal_carry_endpoint_backward_cuda: CUDA inputs required");
    const bool has_initial = initial.numel() != 0;
    c10::cuda::CUDAGuard guard(depth.device());
    auto grad_c = grad_endpoint.contiguous();
    auto depth_c = depth.contiguous();
    auto endpoint_c = endpoint_fp32.contiguous();
    auto reset_c = reset.contiguous();
    auto init_c = has_initial ? initial.contiguous() : torch::empty({0}, depth.options());
    auto valid_c = has_initial ? initial_valid.contiguous() : torch::empty({0}, depth.options().dtype(torch::kBool));
    const int64_t B = depth_c.size(0), T = depth_c.size(1), H = depth_c.size(2);
    auto grad_depth = torch::empty_like(depth_c);
    auto grad_initial = has_initial ? torch::empty_like(init_c) : torch::empty({0}, depth.options());
    const int threads = 256;
    const int64_t blocks = (B * H + threads - 1) / threads;
    // Nested AT_DISPATCH: both macros bind a local alias literally named
    // `scalar_t`, so the inner one shadows the outer -- capture the outer
    // type under its own name (state_t) BEFORE nesting, otherwise every
    // depth_c/init_c/grad_depth data_ptr<scalar_t>() below would silently
    // reinterpret them as grad_c's dtype instead of depth_c's.
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, depth_c.scalar_type(),
        "temporal_carry_endpoint_backward_input", ([&] {
            using state_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, grad_c.scalar_type(),
                "temporal_carry_endpoint_backward_grad", ([&] {
                    using grad_t = scalar_t;
                    temporal_carry_endpoint_backward_kernel<state_t, grad_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        grad_c.data_ptr<grad_t>(), depth_c.data_ptr<state_t>(), endpoint_c.data_ptr<float>(),
                        reset_c.data_ptr<bool>(),
                        has_initial ? init_c.data_ptr<state_t>() : nullptr,
                        has_initial ? valid_c.data_ptr<bool>() : nullptr,
                        grad_depth.data_ptr<state_t>(),
                        has_initial ? grad_initial.data_ptr<state_t>() : nullptr,
                        B, T, H, has_initial);
                }));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_depth, grad_initial};
}
