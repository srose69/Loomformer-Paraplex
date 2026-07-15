// kernels/tria/tria_step/tria_step_launcher.cu -- ATen-facing host wrappers
// for 'tria_step': arg checks, dtype dispatch, kernel launch. This is the
// torch-integration layer; the actual math lives in tria_step_kernel.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "tria_step_kernel.cuh"

std::vector<torch::Tensor> tria_step_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor carry_prev,
    double alpha, int64_t axis) {
    TORCH_CHECK(r.is_cuda() && i.is_cuda() && o.is_cuda() && carry_prev.is_cuda(),
                "tria_step_forward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(i.device() == r.device() && o.device() == r.device() && carry_prev.device() == r.device(),
                "tria_step_forward_cuda: all inputs must be on the same CUDA device");
    TORCH_CHECK(i.scalar_type() == r.scalar_type() && o.scalar_type() == r.scalar_type() &&
                carry_prev.scalar_type() == r.scalar_type(),
                "tria_step_forward_cuda: all inputs must have the same dtype");
    TORCH_CHECK(i.numel() == r.numel() && o.numel() == r.numel(),
                "tria_step_forward_cuda: r, i, o must have the same numel");
    TORCH_CHECK(carry_prev.numel() == r.numel() * 9,
                "tria_step_forward_cuda: carry_prev must have 9 values per r/i/o element");
    c10::cuda::CUDAGuard device_guard(r.device());

    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto cp_c = carry_prev.contiguous();
    auto n = r_c.numel();
    auto carry_new = torch::empty_like(cp_c);
    auto scale = torch::empty({n}, r_c.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_step_forward_cuda", ([&] {
            tria_step_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                cp_c.data_ptr<scalar_t>(), carry_new.data_ptr<scalar_t>(),
                scale.data_ptr<float>(), (float)alpha, (int)axis, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {carry_new, scale};
}

std::vector<torch::Tensor> tria_step_backward_cuda(
    torch::Tensor grad_carry_new, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor carry_prev, torch::Tensor scale, double alpha, int64_t axis) {
    TORCH_CHECK(grad_carry_new.is_cuda() && r.is_cuda() && i.is_cuda() && o.is_cuda() &&
                carry_prev.is_cuda() && scale.is_cuda(),
                "tria_step_backward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(r.device() == grad_carry_new.device() && i.device() == grad_carry_new.device() &&
                o.device() == grad_carry_new.device() && carry_prev.device() == grad_carry_new.device() &&
                scale.device() == grad_carry_new.device(),
                "tria_step_backward_cuda: all inputs must be on the same CUDA device");
    TORCH_CHECK(r.scalar_type() == grad_carry_new.scalar_type() && i.scalar_type() == grad_carry_new.scalar_type() &&
                o.scalar_type() == grad_carry_new.scalar_type() && carry_prev.scalar_type() == grad_carry_new.scalar_type(),
                "tria_step_backward_cuda: activation inputs must have the same dtype");
    TORCH_CHECK(scale.scalar_type() == torch::kFloat32, "tria_step_backward_cuda: scale must be float32");
    TORCH_CHECK(i.numel() == r.numel() && o.numel() == r.numel() && scale.numel() == r.numel(),
                "tria_step_backward_cuda: r, i, o, scale must have the same numel");
    TORCH_CHECK(carry_prev.numel() == r.numel() * 9 && grad_carry_new.numel() == r.numel() * 9,
                "tria_step_backward_cuda: carry tensors must have 9 values per r/i/o element");
    c10::cuda::CUDAGuard device_guard(grad_carry_new.device());

    auto go_c = grad_carry_new.contiguous();
    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto cp_c = carry_prev.contiguous();
    auto scale_c = scale.contiguous();
    auto n = r_c.numel();
    auto grad_r = torch::empty_like(r_c);
    auto grad_i = torch::empty_like(i_c);
    auto grad_o = torch::empty_like(o_c);
    auto grad_carry_prev = torch::empty_like(cp_c);

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_step_backward_cuda", ([&] {
            tria_step_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                go_c.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                cp_c.data_ptr<scalar_t>(), scale_c.data_ptr<float>(),
                grad_r.data_ptr<scalar_t>(), grad_i.data_ptr<scalar_t>(), grad_o.data_ptr<scalar_t>(),
                grad_carry_prev.data_ptr<scalar_t>(), (float)alpha, (int)axis, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_r, grad_i, grad_o, grad_carry_prev};
}

std::vector<torch::Tensor> tria_step_reverse_backward_cuda(
    torch::Tensor grad_carry_new, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor current, double alpha, int64_t axis) {
    TORCH_CHECK(grad_carry_new.is_cuda() && r.is_cuda() && i.is_cuda() && o.is_cuda() && current.is_cuda(),
                "tria_step_reverse_backward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(r.scalar_type() == grad_carry_new.scalar_type() && i.scalar_type() == grad_carry_new.scalar_type() &&
                o.scalar_type() == grad_carry_new.scalar_type() && current.scalar_type() == grad_carry_new.scalar_type(),
                "tria_step_reverse_backward_cuda: activation inputs must have the same dtype");
    TORCH_CHECK(i.numel() == r.numel() && o.numel() == r.numel(),
                "tria_step_reverse_backward_cuda: r, i, o must have the same numel");
    TORCH_CHECK(current.numel() == r.numel() * 9 && grad_carry_new.numel() == r.numel() * 9,
                "tria_step_reverse_backward_cuda: carry tensors must have 9 values per r/i/o element");
    c10::cuda::CUDAGuard device_guard(grad_carry_new.device());

    auto go_c = grad_carry_new.contiguous();
    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto cur_c = current.contiguous();
    auto n = r_c.numel();
    auto grad_r = torch::empty_like(r_c);
    auto grad_i = torch::empty_like(i_c);
    auto grad_o = torch::empty_like(o_c);
    auto grad_previous = torch::empty_like(cur_c);
    auto previous = torch::empty_like(cur_c);

    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(r_c.scalar_type(),
        "tria_step_reverse_backward_cuda", ([&] {
            tria_step_reverse_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                go_c.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                cur_c.data_ptr<scalar_t>(),
                grad_r.data_ptr<scalar_t>(), grad_i.data_ptr<scalar_t>(), grad_o.data_ptr<scalar_t>(),
                grad_previous.data_ptr<scalar_t>(), previous.data_ptr<scalar_t>(),
                (float)alpha, (int)axis, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_r, grad_i, grad_o, grad_previous, previous};
}
