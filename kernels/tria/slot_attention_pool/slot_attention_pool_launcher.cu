// kernels/tria/slot_attention_pool/slot_attention_pool_launcher.cu -- ATen-facing host wrappers
// for 'slot_attention_pool': arg checks, dtype dispatch, kernel launch. This is the
// torch-integration layer; the actual math lives in slot_attention_pool_kernel.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "slot_attention_pool_kernel.cuh"

std::vector<torch::Tensor> slot_attention_pool_forward_cuda(torch::Tensor carry, torch::Tensor score_w) {
    TORCH_CHECK(carry.is_cuda() && score_w.is_cuda(),
                "slot_attention_pool_forward_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(carry.scalar_type() == score_w.scalar_type(),
                "slot_attention_pool_forward_cuda: dtype mismatch");
    TORCH_CHECK(carry.dim() == 5 && carry.size(-1) == 3 && carry.size(-2) == 3,
                "slot_attention_pool_forward_cuda: carry must be [B,T,H,3,3]");
    TORCH_CHECK(score_w.numel() == 9, "slot_attention_pool_forward_cuda: score_w must have exactly 9 elements");
    c10::cuda::CUDAGuard device_guard(carry.device());

    auto carry_c = carry.contiguous();
    auto sw_c = score_w.contiguous();
    int64_t B = carry_c.size(0), T = carry_c.size(1), H = carry_c.size(2);
    int64_t BT = B * T;
    auto pooled = torch::empty({B, T, 9}, carry_c.options());
    auto lse = torch::empty({B, T}, carry_c.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, carry_c.scalar_type(),
        "slot_attention_pool_forward_cuda", ([&] {
            slot_attention_pool_forward_kernel<scalar_t><<<BT, SLOT_POOL_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                carry_c.data_ptr<scalar_t>(), sw_c.data_ptr<scalar_t>(),
                pooled.data_ptr<scalar_t>(), lse.data_ptr<scalar_t>(), H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {pooled, lse};
}

std::vector<torch::Tensor> slot_attention_pool_backward_cuda(
    torch::Tensor grad_pooled, torch::Tensor carry, torch::Tensor score_w, torch::Tensor lse) {
    TORCH_CHECK(grad_pooled.is_cuda() && carry.is_cuda() && score_w.is_cuda() && lse.is_cuda(),
                "slot_attention_pool_backward_cuda: inputs must be CUDA tensors");
    TORCH_CHECK(carry.scalar_type() == score_w.scalar_type() && carry.scalar_type() == grad_pooled.scalar_type() &&
                carry.scalar_type() == lse.scalar_type(),
                "slot_attention_pool_backward_cuda: dtype mismatch");
    TORCH_CHECK(carry.dim() == 5 && carry.size(-1) == 3 && carry.size(-2) == 3,
                "slot_attention_pool_backward_cuda: carry must be [B,T,H,3,3]");
    TORCH_CHECK(score_w.numel() == 9, "slot_attention_pool_backward_cuda: score_w must have exactly 9 elements");
    c10::cuda::CUDAGuard device_guard(carry.device());

    auto gpooled_c = grad_pooled.contiguous();
    auto carry_c = carry.contiguous();
    auto sw_c = score_w.contiguous();
    auto lse_c = lse.contiguous();
    int64_t B = carry_c.size(0), T = carry_c.size(1), H = carry_c.size(2);
    int64_t BT = B * T;
    auto grad_carry = torch::empty_like(carry_c);
    auto block_partial = torch::empty({BT, 9}, carry_c.options().dtype(torch::kFloat32));

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, carry_c.scalar_type(),
        "slot_attention_pool_backward_cuda", ([&] {
            slot_attention_pool_backward_kernel<scalar_t><<<BT, SLOT_POOL_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                gpooled_c.data_ptr<scalar_t>(), carry_c.data_ptr<scalar_t>(), sw_c.data_ptr<scalar_t>(),
                lse_c.data_ptr<scalar_t>(), grad_carry.data_ptr<scalar_t>(),
                block_partial.data_ptr<float>(), H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto grad_score_w = block_partial.sum(0).to(sw_c.scalar_type());
    return {grad_carry, grad_score_w};
}
