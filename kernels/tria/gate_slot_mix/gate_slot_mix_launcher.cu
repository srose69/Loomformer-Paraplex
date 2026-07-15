// kernels/tria/gate_slot_mix/gate_slot_mix_launcher.cu -- ATen-facing host wrappers
// for 'gate_slot_mix': arg checks, dtype dispatch, kernel launch. This is the
// torch-integration layer; the actual math lives in gate_slot_mix_kernel.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "gate_slot_mix_kernel.cuh"

torch::Tensor gate_slot_mix_forward_cuda(torch::Tensor carry, torch::Tensor w) {
    TORCH_CHECK(carry.is_cuda() && w.is_cuda(), "gate_slot_mix_forward_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(carry.device() == w.device(), "gate_slot_mix_forward_cuda: inputs must be on the same device");
    TORCH_CHECK(carry.scalar_type() == w.scalar_type(), "gate_slot_mix_forward_cuda: dtype mismatch");
    TORCH_CHECK(carry.dim() >= 2 && carry.size(-1) == 3 && carry.size(-2) == 3,
                "gate_slot_mix_forward_cuda: carry must end in [...,3,3]");
    TORCH_CHECK(w.numel() == 9, "gate_slot_mix_forward_cuda: w must have exactly 9 elements");
    c10::cuda::CUDAGuard device_guard(carry.device());

    auto carry_c = carry.contiguous();
    auto w_c = w.contiguous();
    auto out_shape = carry_c.sizes().vec();
    out_shape.pop_back();
    out_shape.pop_back();
    int64_t n = carry_c.numel() / 9;
    auto p = torch::empty(out_shape, carry_c.options());

    const int threads = GATE_MIX_THREADS;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, carry_c.scalar_type(),
        "gate_slot_mix_forward_cuda", ([&] {
            gate_slot_mix_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                carry_c.data_ptr<scalar_t>(), w_c.data_ptr<scalar_t>(), p.data_ptr<scalar_t>(), n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return p;
}

std::vector<torch::Tensor> gate_slot_mix_backward_cuda(
    torch::Tensor grad_p, torch::Tensor carry, torch::Tensor w) {
    TORCH_CHECK(grad_p.is_cuda() && carry.is_cuda() && w.is_cuda(),
                "gate_slot_mix_backward_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(carry.scalar_type() == w.scalar_type() && grad_p.scalar_type() == w.scalar_type(),
                "gate_slot_mix_backward_cuda: dtype mismatch");
    TORCH_CHECK(w.numel() == 9, "gate_slot_mix_backward_cuda: w must have exactly 9 elements");
    c10::cuda::CUDAGuard device_guard(carry.device());

    auto gp_c = grad_p.contiguous();
    auto carry_c = carry.contiguous();
    auto w_c = w.contiguous();
    int64_t n = carry_c.numel() / 9;
    TORCH_CHECK(gp_c.numel() == n, "gate_slot_mix_backward_cuda: grad_p/carry shape mismatch");
    auto grad_carry = torch::empty_like(carry_c);

    const int threads = GATE_MIX_THREADS;
    const int64_t blocks = (n + threads - 1) / threads;
    auto grad_w_partial = torch::empty({9, blocks}, carry_c.options().dtype(torch::kFloat32));
    auto grad_w_acc = torch::empty({9}, carry_c.options().dtype(torch::kFloat32));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, carry_c.scalar_type(),
        "gate_slot_mix_backward_cuda", ([&] {
            gate_slot_mix_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                gp_c.data_ptr<scalar_t>(), carry_c.data_ptr<scalar_t>(), w_c.data_ptr<scalar_t>(),
                grad_carry.data_ptr<scalar_t>(), grad_w_partial.data_ptr<float>(), n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int reduce_threads = 256;
    const int reduce_nwarps = (reduce_threads + 31) / 32;
    gate_mix_grad_w_reduce_kernel<<<9, reduce_threads, reduce_nwarps * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        grad_w_partial.data_ptr<float>(), grad_w_acc.data_ptr<float>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto grad_w = grad_w_acc.to(w_c.scalar_type());
    return {grad_carry, grad_w};
}
