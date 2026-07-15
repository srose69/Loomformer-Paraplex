#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <vector>
#include "paraplex_kernel.cuh"

std::vector<torch::Tensor> paraplex_forward_cuda(
    torch::Tensor p_real, torch::Tensor beta_linear, torch::Tensor bias,
    torch::Tensor trace, torch::Tensor trace_w, torch::Tensor reset,
    torch::Tensor anchor, int64_t mode, bool update_anchor, double decay, double m);
std::vector<torch::Tensor> paraplex_backward_cuda(
    torch::Tensor grad_act, torch::Tensor grad_s, torch::Tensor grad_next,
    torch::Tensor p_real, torch::Tensor beta_linear, torch::Tensor bias,
    torch::Tensor trace, torch::Tensor trace_w, torch::Tensor reset,
    torch::Tensor anchor, int64_t mode, double floor, double near_eps, double m);

static void paraplex_check(
    const torch::Tensor& p, const torch::Tensor& beta, const torch::Tensor& bias,
    const torch::Tensor& trace, const torch::Tensor& trace_w,
    const torch::Tensor& reset, const torch::Tensor& anchor) {
    TORCH_CHECK(p.is_cuda() && beta.is_cuda() && bias.is_cuda() && trace.is_cuda() &&
                trace_w.is_cuda() && anchor.is_cuda(), "paraplex: CUDA tensors required");
    TORCH_CHECK(beta.device() == p.device() && bias.device() == p.device() && trace.device() == p.device() &&
                trace_w.device() == p.device() && anchor.device() == p.device(), "paraplex: device mismatch");
    TORCH_CHECK(p.dim() == 3 && beta.sizes() == p.sizes(), "p/beta must be [B,T,H]");
    TORCH_CHECK(beta.scalar_type() == p.scalar_type() && trace.scalar_type() == p.scalar_type(),
                "p/beta/trace dtype mismatch");
    TORCH_CHECK(trace.dim() == 2 && trace.size(0) == p.size(0) && trace.size(1) == p.size(2),
                "trace must be [B,H]");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32 && trace_w.scalar_type() == torch::kFloat32 &&
                anchor.scalar_type() == torch::kFloat32, "bias/trace_w/anchor must be float32");
    TORCH_CHECK(bias.numel() == p.size(2) && trace_w.numel() == p.size(2) && anchor.numel() == 1,
                "parameter shape mismatch");
    TORCH_CHECK(reset.numel() == 0 || (reset.is_cuda() && reset.scalar_type() == torch::kBool &&
                reset.dim() == 2 && reset.size(0) == p.size(0) && reset.size(1) == p.size(1)),
                "reset must be empty or bool [B,T]");
    TORCH_CHECK(reset.numel() == 0 || reset.device() == p.device(), "reset device mismatch");
    TORCH_CHECK(p.numel() > 0, "paraplex: empty tensors are unsupported");
}

std::vector<torch::Tensor> paraplex_forward_cuda(
    torch::Tensor p_real, torch::Tensor beta_linear, torch::Tensor bias,
    torch::Tensor trace, torch::Tensor trace_w, torch::Tensor reset,
    torch::Tensor anchor, int64_t mode, bool update_anchor, double decay, double m) {
    paraplex_check(p_real, beta_linear, bias, trace, trace_w, reset, anchor);
    c10::cuda::CUDAGuard guard(p_real.device());
    auto pc = p_real.contiguous(), bc = beta_linear.contiguous();
    auto bic = bias.contiguous(), tc = trace.contiguous(), twc = trace_w.contiguous();
    auto rc = reset.numel() ? reset.contiguous() : torch::empty({0}, p_real.options().dtype(torch::kBool));
    auto ac = anchor.contiguous();
    const int64_t B = p_real.size(0), T = p_real.size(1), H = p_real.size(2), n = p_real.numel();
    // Match anchor's own shape (ParaplexFFN.beta_anchor is a 0-dim scalar
    // buffer, not a [1]-shaped tensor) -- Tensor.copy_ can't broadcast a
    // [1] source down into a [] destination.
    auto snapshot = torch::empty(anchor.sizes(), anchor.options());
    const int threads = 256;
    torch::Tensor sum;
    const float* sum_ptr = nullptr;
    if (update_anchor) {
        sum = torch::zeros(anchor.sizes(), anchor.options());
        const int warps = (threads + 31) / 32;
        const int64_t blocks = std::min<int64_t>((n + threads - 1) / threads, 4096);
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pc.scalar_type(),
            "paraplex_anchor_sum", ([&] {
                paraplex_anchor_sum_kernel<scalar_t><<<blocks, threads, warps * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
                    bc.data_ptr<scalar_t>(), bic.data_ptr<float>(), sum.data_ptr<float>(), n, H);
            }));
        sum_ptr = sum.data_ptr<float>();
    }
    paraplex_anchor_update_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        ac.data_ptr<float>(), snapshot.data_ptr<float>(), sum_ptr, n, (float)decay, update_anchor);
    auto act = torch::empty_like(pc), s = torch::empty_like(pc);
    auto next = torch::empty({B, H}, pc.options());
    const int64_t elem_blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pc.scalar_type(),
        "paraplex_forward_cuda", ([&] {
            paraplex_forward_kernel<scalar_t><<<elem_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                pc.data_ptr<scalar_t>(), bc.data_ptr<scalar_t>(), bic.data_ptr<float>(), tc.data_ptr<scalar_t>(),
                twc.data_ptr<float>(), rc.numel() ? rc.data_ptr<bool>() : nullptr,
                act.data_ptr<scalar_t>(), s.data_ptr<scalar_t>(), next.data_ptr<scalar_t>(),
                (float)m, B, T, H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {act, s, next, snapshot};
}

std::vector<torch::Tensor> paraplex_backward_cuda(
    torch::Tensor grad_act, torch::Tensor grad_s, torch::Tensor grad_next,
    torch::Tensor p_real, torch::Tensor beta_linear, torch::Tensor bias,
    torch::Tensor trace, torch::Tensor trace_w, torch::Tensor reset,
    torch::Tensor anchor, int64_t mode, double floor, double near_eps, double m) {
    paraplex_check(p_real, beta_linear, bias, trace, trace_w, reset, anchor);
    c10::cuda::CUDAGuard guard(p_real.device());
    auto ga = grad_act.contiguous(), gs = grad_s.contiguous(), gn = grad_next.contiguous();
    auto pc = p_real.contiguous(), bc = beta_linear.contiguous();
    auto bic = bias.contiguous(), tc = trace.contiguous(), twc = trace_w.contiguous();
    auto rc = reset.numel() ? reset.contiguous() : torch::empty({0}, p_real.options().dtype(torch::kBool));
    auto ac = anchor.contiguous();
    const int64_t B = p_real.size(0), T = p_real.size(1), H = p_real.size(2), n = p_real.numel();
    auto gp = torch::empty_like(pc), gbeta = torch::empty_like(bc), gtrace = torch::empty_like(tc);
    auto gbias = torch::empty_like(bic), gtw = torch::empty_like(twc);
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, pc.scalar_type(),
        "paraplex_backward_input", ([&] {
            using input_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, ga.scalar_type(),
                "paraplex_backward_grad", ([&] {
                    using grad_t = scalar_t;
                    paraplex_backward_kernel<input_t, grad_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                        ga.data_ptr<grad_t>(), gs.data_ptr<grad_t>(), gn.data_ptr<grad_t>(), pc.data_ptr<input_t>(),
                        bc.data_ptr<input_t>(), bic.data_ptr<float>(), tc.data_ptr<input_t>(), twc.data_ptr<float>(),
                        rc.numel() ? rc.data_ptr<bool>() : nullptr, ac.data_ptr<float>(), gp.data_ptr<input_t>(),
                        gbeta.data_ptr<input_t>(), gtrace.data_ptr<input_t>(), (int)mode, (float)floor,
                        (float)near_eps, (float)m, B, T, H);
                    const int warps = (threads + 31) / 32;
                    paraplex_reduce_kernel<input_t, grad_t><<<H, threads, 2 * warps * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
                        ga.data_ptr<grad_t>(), gs.data_ptr<grad_t>(), gn.data_ptr<grad_t>(), pc.data_ptr<input_t>(),
                        bc.data_ptr<input_t>(), bic.data_ptr<float>(), tc.data_ptr<input_t>(), twc.data_ptr<float>(),
                        rc.numel() ? rc.data_ptr<bool>() : nullptr, ac.data_ptr<float>(),
                        gbias.data_ptr<float>(), gtw.data_ptr<float>(), (int)mode, (float)floor,
                        (float)near_eps, (float)m, B, T, H);
                }));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gp, gbeta, gbias, gtrace, gtw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paraplex_forward", &paraplex_forward_cuda);
    m.def("paraplex_backward", &paraplex_backward_cuda);
}
