#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

#include "tria_init_seed_kernel.cuh"

std::vector<torch::Tensor> tria_init_seed_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor seed, torch::Tensor valid, double alpha, int64_t axis) {
    TORCH_CHECK(r.is_cuda() && i.is_cuda() && o.is_cuda() && seed.is_cuda() && valid.is_cuda(),
                "tria_init_seed_forward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(r.dim() == 3 && i.sizes() == r.sizes() && o.sizes() == r.sizes(),
                "tria_init_seed_forward_cuda: r, i, o must be [B,T,H]");
    TORCH_CHECK(seed.dim() == 4 && seed.size(1) == r.size(2) && seed.size(2) == 3 && seed.size(3) == 3,
                "tria_init_seed_forward_cuda: seed must be [B,H,3,3]");
    TORCH_CHECK(seed.size(0) == r.size(0), "tria_init_seed_forward_cuda: batch mismatch");
    TORCH_CHECK(valid.dim() == 1 && valid.size(0) == r.size(0) && valid.scalar_type() == torch::kBool,
                "tria_init_seed_forward_cuda: valid must be [B] bool");
    TORCH_CHECK(seed.scalar_type() == r.scalar_type() && i.scalar_type() == r.scalar_type() && o.scalar_type() == r.scalar_type(),
                "tria_init_seed_forward_cuda: activation/seed dtype mismatch");
    c10::cuda::CUDAGuard guard(r.device());

    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto seed_c = seed.contiguous();
    auto valid_c = valid.contiguous();
    const int64_t B = r_c.size(0), T = r_c.size(1), H = r_c.size(2);
    const int64_t n = r_c.numel();
    auto carry = torch::empty({n, 9}, r_c.options());
    auto scale = torch::empty({n}, r_c.options().dtype(torch::kFloat32));
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_seed_forward_cuda", ([&] {
            tria_init_seed_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                seed_c.data_ptr<scalar_t>(), valid_c.data_ptr<bool>(),
                carry.data_ptr<scalar_t>(), scale.data_ptr<float>(),
                (float)alpha, (int)axis, B, T, H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {carry.view({B, T, H, 3, 3}), scale};
}

std::vector<torch::Tensor> tria_init_seed_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor seed, torch::Tensor valid, double alpha, int64_t axis) {
    TORCH_CHECK(grad_carry.is_cuda() && r.is_cuda() && i.is_cuda() && o.is_cuda() && seed.is_cuda() && valid.is_cuda(),
                "tria_init_seed_backward_cuda: all inputs must be CUDA tensors");
    c10::cuda::CUDAGuard guard(r.device());
    auto go = grad_carry.contiguous();
    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto seed_c = seed.contiguous();
    auto valid_c = valid.contiguous();
    const int64_t B = r_c.size(0), T = r_c.size(1), H = r_c.size(2);
    const int64_t n = r_c.numel();
    auto grad_r = torch::empty_like(r_c);
    auto grad_i = torch::empty_like(i_c);
    auto grad_o = torch::empty_like(o_c);
    auto grad_seed = torch::zeros_like(seed_c);
    auto unused = torch::empty({0}, r_c.options().dtype(torch::kBool));
    auto scale = torch::empty({n}, r_c.options().dtype(torch::kFloat32));
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_seed_backward_cuda", ([&] {
            tria_init_seed_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                seed_c.data_ptr<scalar_t>(), valid_c.data_ptr<bool>(),
                static_cast<scalar_t*>(nullptr), scale.data_ptr<float>(),
                (float)alpha, (int)axis, B, T, H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_seed_backward_cuda", ([&] {
            tria_init_seed_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                go.data_ptr<scalar_t>(), r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                seed_c.data_ptr<scalar_t>(), valid_c.data_ptr<bool>(), scale.data_ptr<float>(),
                grad_r.data_ptr<scalar_t>(), grad_i.data_ptr<scalar_t>(), grad_o.data_ptr<scalar_t>(),
                grad_seed.data_ptr<scalar_t>(), (float)alpha, (int)axis, B, T, H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_r, grad_i, grad_o, grad_seed, unused};
}

std::vector<torch::Tensor> tria_init_seed_gate_forward_cuda(
    torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor seed, torch::Tensor valid, torch::Tensor w, double alpha, int64_t axis) {
    TORCH_CHECK(w.numel() == 9, "tria_init_seed_gate_forward_cuda: w must have 9 elements");
    c10::cuda::CUDAGuard guard(r.device());
    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto seed_c = seed.contiguous();
    auto valid_c = valid.contiguous();
    auto w_c = w.contiguous();
    const int64_t B = r_c.size(0), T = r_c.size(1), H = r_c.size(2);
    const int64_t n = r_c.numel();
    auto carry = torch::empty({n, 9}, r_c.options());
    auto p_out = torch::empty_like(r_c);
    auto scale = torch::empty({n}, r_c.options().dtype(torch::kFloat32));
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_seed_gate_forward_cuda", ([&] {
            tria_init_seed_gate_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                seed_c.data_ptr<scalar_t>(), valid_c.data_ptr<bool>(), w_c.data_ptr<scalar_t>(),
                carry.data_ptr<scalar_t>(), p_out.data_ptr<scalar_t>(), scale.data_ptr<float>(),
                (float)alpha, (int)axis, B, T, H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {carry.view({B, T, H, 3, 3}), p_out};
}

std::vector<torch::Tensor> tria_init_seed_gate_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor grad_p,
    torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor seed, torch::Tensor valid, torch::Tensor w, double alpha, int64_t axis) {
    c10::cuda::CUDAGuard guard(r.device());
    auto go = grad_carry.contiguous();
    auto gp = grad_p.contiguous();
    auto r_c = r.contiguous();
    auto i_c = i.contiguous();
    auto o_c = o.contiguous();
    auto seed_c = seed.contiguous();
    auto valid_c = valid.contiguous();
    auto w_c = w.contiguous();
    const int64_t B = r_c.size(0), T = r_c.size(1), H = r_c.size(2);
    const int64_t n = r_c.numel();
    auto grad_r = torch::empty_like(r_c);
    auto grad_i = torch::empty_like(i_c);
    auto grad_o = torch::empty_like(o_c);
    auto grad_seed = torch::zeros_like(seed_c);
    const int threads = GATE_MIX_THREADS;
    const int64_t blocks = (n + threads - 1) / threads;
    auto grad_w_partial = torch::empty({9, blocks}, r_c.options().dtype(torch::kFloat32));
    auto grad_w_acc = torch::empty({9}, r_c.options().dtype(torch::kFloat32));
    auto scale = torch::empty({n}, r_c.options().dtype(torch::kFloat32));
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_seed_gate_backward_cuda", ([&] {
            tria_init_seed_gate_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                seed_c.data_ptr<scalar_t>(), valid_c.data_ptr<bool>(), w_c.data_ptr<scalar_t>(),
                static_cast<scalar_t*>(nullptr), static_cast<scalar_t*>(nullptr), scale.data_ptr<float>(),
                (float)alpha, (int)axis, B, T, H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, r_c.scalar_type(),
        "tria_init_seed_gate_backward_cuda", ([&] {
            tria_init_seed_gate_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                go.data_ptr<scalar_t>(), gp.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(), i_c.data_ptr<scalar_t>(), o_c.data_ptr<scalar_t>(),
                seed_c.data_ptr<scalar_t>(), valid_c.data_ptr<bool>(), w_c.data_ptr<scalar_t>(),
                scale.data_ptr<float>(), grad_r.data_ptr<scalar_t>(), grad_i.data_ptr<scalar_t>(), grad_o.data_ptr<scalar_t>(),
                grad_seed.data_ptr<scalar_t>(), grad_w_partial.data_ptr<float>(),
                (float)alpha, (int)axis, B, T, H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int reduce_threads = 256;
    const int reduce_nwarps = (reduce_threads + 31) / 32;
    gate_mix_grad_w_reduce_kernel<<<9, reduce_threads, reduce_nwarps * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        grad_w_partial.data_ptr<float>(), grad_w_acc.data_ptr<float>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto grad_w = grad_w_acc.to(w_c.scalar_type());
    return {grad_r, grad_i, grad_o, grad_seed, grad_w};
}
