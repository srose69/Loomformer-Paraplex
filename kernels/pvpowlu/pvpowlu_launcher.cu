// kernels/pvpowlu/pvpowlu_launcher.cu -- ATen host wrappers + pybind.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "pvpowlu_kernel.cuh"

torch::Tensor pvpowlu_forward_cuda(torch::Tensor x1, torch::Tensor x2, double m) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda(), "pvpowlu_forward_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(x1.device() == x2.device(), "pvpowlu_forward_cuda: device mismatch");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "pvpowlu_forward_cuda: dtype mismatch");
    TORCH_CHECK(x1.numel() == x2.numel(), "pvpowlu_forward_cuda: shape mismatch");
    c10::cuda::CUDAGuard device_guard(x1.device());
    auto x1_c = x1.contiguous();
    auto x2_c = x2.contiguous();
    auto out = torch::empty_like(x1_c);
    int64_t n = x1_c.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x1_c.scalar_type(),
        "pvpowlu_forward_cuda", ([&] {
            pvpowlu_fwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                x1_c.data_ptr<scalar_t>(), x2_c.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), (float)m, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> pvpowlu_backward_cuda(
    torch::Tensor grad_out, torch::Tensor x1, torch::Tensor x2, double m) {
    TORCH_CHECK(grad_out.is_cuda() && x1.is_cuda() && x2.is_cuda(), "pvpowlu_backward_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(grad_out.device() == x1.device() && x2.device() == x1.device(), "pvpowlu_backward_cuda: device mismatch");
    TORCH_CHECK(grad_out.scalar_type() == x1.scalar_type() && x2.scalar_type() == x1.scalar_type(),
                "pvpowlu_backward_cuda: dtype mismatch");
    TORCH_CHECK(grad_out.numel() == x1.numel() && x2.numel() == x1.numel(), "pvpowlu_backward_cuda: shape mismatch");
    c10::cuda::CUDAGuard device_guard(x1.device());
    auto go_c = grad_out.contiguous();
    auto x1_c = x1.contiguous();
    auto x2_c = x2.contiguous();
    auto grad_x1 = torch::empty_like(x1_c);
    auto grad_x2 = torch::empty_like(x2_c);
    int64_t n = x1_c.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, x1_c.scalar_type(),
        "pvpowlu_backward_cuda", ([&] {
            pvpowlu_bwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                go_c.data_ptr<scalar_t>(), x1_c.data_ptr<scalar_t>(), x2_c.data_ptr<scalar_t>(),
                grad_x1.data_ptr<scalar_t>(), grad_x2.data_ptr<scalar_t>(), (float)m, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_x1, grad_x2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pvpowlu_forward_cuda", &pvpowlu_forward_cuda, "pvpowlu_forward_cuda");
    m.def("pvpowlu_backward_cuda", &pvpowlu_backward_cuda, "pvpowlu_backward_cuda");
}
