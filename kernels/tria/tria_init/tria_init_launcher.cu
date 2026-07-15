// kernels/tria/tria_init/tria_init_launcher.cu -- ATen-facing host wrappers
// for 'tria_init': arg checks, dtype dispatch, kernel launch. This is the
// torch-integration layer; the actual math lives in tria_init_kernel.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "tria_init_kernel.cuh"

std::vector<torch::Tensor> tria_init_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, double alpha, int64_t axis) {
    TORCH_CHECK(r.is_cuda() && i.is_cuda() && o.is_cuda(),
                "tria_init_forward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(i.device() == r.device() && o.device() == r.device(),
                "tria_init_forward_cuda: all inputs must be on the same CUDA device");
    TORCH_CHECK(i.scalar_type() == r.scalar_type() && o.scalar_type() == r.scalar_type(),
                "tria_init_forward_cuda: all inputs must have the same dtype");
    TORCH_CHECK(i.numel() == r.numel() && o.numel() == r.numel(),
                "tria_init_forward_cuda: r, i, o must have the same numel");
    c10::cuda::CUDAGuard device_guard(r.device());

    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto n = r_c.numel();
    auto carry_1_flat = torch::empty({n, 9}, r_c.options());
    auto scale = torch::empty({n}, r_c.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_forward_cuda", ([&] {
            tria_init_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                carry_1_flat.data_ptr<scalar_t>(), scale.data_ptr<float>(), (float)alpha, (int)axis, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {carry_1_flat.view({r_c.size(0), r_c.size(1), r_c.size(2), 3, 3}), scale};
}

std::vector<torch::Tensor> tria_init_backward_cuda(
    torch::Tensor grad_carry_1, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor scale, double alpha, int64_t axis) {
    TORCH_CHECK(grad_carry_1.is_cuda() && r.is_cuda() && i.is_cuda() && o.is_cuda() && scale.is_cuda(),
                "tria_init_backward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(r.device() == grad_carry_1.device() && i.device() == grad_carry_1.device() &&
                o.device() == grad_carry_1.device() && scale.device() == grad_carry_1.device(),
                "tria_init_backward_cuda: all inputs must be on the same CUDA device");
    TORCH_CHECK(r.scalar_type() == grad_carry_1.scalar_type() && i.scalar_type() == grad_carry_1.scalar_type() &&
                o.scalar_type() == grad_carry_1.scalar_type(),
                "tria_init_backward_cuda: activation inputs must have the same dtype");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "tria_init_backward_cuda: scale must be float32");
    TORCH_CHECK(i.numel() == r.numel() && o.numel() == r.numel() && scale.numel() == r.numel(),
                "tria_init_backward_cuda: r, i, o, scale must have the same numel");
    TORCH_CHECK(grad_carry_1.numel() == r.numel() * 9,
                "tria_init_backward_cuda: grad_carry_1 must have 9 values per r/i/o element");
    c10::cuda::CUDAGuard device_guard(grad_carry_1.device());

    auto go_c = grad_carry_1.contiguous();
    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto scale_c = scale.contiguous();
    auto n = r_c.numel();
    auto grad_r = torch::empty_like(r_c);
    auto grad_i = torch::empty_like(i_c);
    auto grad_o = torch::empty_like(o_c);

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_backward_cuda", ([&] {
            tria_init_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                go_c.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                scale_c.data_ptr<float>(),
                grad_r.data_ptr<scalar_t>(), grad_i.data_ptr<scalar_t>(), grad_o.data_ptr<scalar_t>(),
                (float)alpha, (int)axis, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_r, grad_i, grad_o};
}
