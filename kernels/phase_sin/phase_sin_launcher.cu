// kernels/phase_sin/phase_sin_launcher.cu -- ATen host wrappers + pybind.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "phase_sin_kernel.cuh"

torch::Tensor phase_sin_forward_cuda(torch::Tensor beta) {
    TORCH_CHECK(beta.is_cuda(), "beta must be a CUDA tensor");
    // Was float32-only; now float/half/bfloat16 via AT_DISPATCH, matching
    // every other kernel group in this tree (pvpowlu, tria, depth_attn,
    // beta_space). sinf/cosf/rsqrtf have no native half/bf16 device
    // intrinsics, so the kernel itself reads each element as (float), does
    // all math in fp32, and only the final store casts back to scalar_t --
    // see phase_sin_kernel.cuh. Numerically identical to the old fp32-only
    // path for fp32 inputs; for bf16 inputs it's the same math with a
    // narrower storage format, not a different formula.
    c10::cuda::CUDAGuard device_guard(beta.device());
    auto beta_c = beta.contiguous();
    auto out = torch::empty_like(beta_c);
    int64_t n = beta_c.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, beta_c.scalar_type(),
        "phase_sin_forward_cuda", ([&] {
            phase_sin_fwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                beta_c.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor phase_sin_backward_cuda(torch::Tensor beta, torch::Tensor grad_out, double eps) {
    TORCH_CHECK(beta.is_cuda() && grad_out.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(grad_out.device() == beta.device(), "grad_out device mismatch");
    TORCH_CHECK(grad_out.scalar_type() == beta.scalar_type(), "phase_sin_backward_cuda: dtype mismatch");
    c10::cuda::CUDAGuard device_guard(beta.device());
    auto beta_c = beta.contiguous();
    auto grad_out_c = grad_out.contiguous();
    auto grad_in = torch::empty_like(beta_c);
    int64_t n = beta_c.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, beta_c.scalar_type(),
        "phase_sin_backward_cuda", ([&] {
            phase_sin_bwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                beta_c.data_ptr<scalar_t>(), grad_out_c.data_ptr<scalar_t>(),
                grad_in.data_ptr<scalar_t>(), (float)eps, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_in;
}

// phase_grad_mode=="secant" -- see phase_sin_kernel.cuh's phase_sin_secant_bwd_kernel
// and loomformer.py's _PhaseSinSecant for the PyTorch reference this mirrors.
// anchor/s_anchor are per-layer scalars (one sin(bound_phase(anchor)) computed
// once on the Python side), passed as plain doubles exactly like eps above --
// not tensors, no per-element anchor.
torch::Tensor phase_sin_secant_backward_cuda(torch::Tensor beta, torch::Tensor grad_out,
                                              double anchor, double s_anchor, double near_eps) {
    TORCH_CHECK(beta.is_cuda() && grad_out.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(grad_out.device() == beta.device(), "grad_out device mismatch");
    TORCH_CHECK(grad_out.scalar_type() == beta.scalar_type(), "phase_sin_secant_backward_cuda: dtype mismatch");
    c10::cuda::CUDAGuard device_guard(beta.device());
    auto beta_c = beta.contiguous();
    auto grad_out_c = grad_out.contiguous();
    auto grad_in = torch::empty_like(beta_c);
    int64_t n = beta_c.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, beta_c.scalar_type(),
        "phase_sin_secant_backward_cuda", ([&] {
            phase_sin_secant_bwd_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                beta_c.data_ptr<scalar_t>(), grad_out_c.data_ptr<scalar_t>(),
                grad_in.data_ptr<scalar_t>(), (float)anchor, (float)s_anchor, (float)near_eps, n);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grad_in;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("phase_sin_forward_cuda", &phase_sin_forward_cuda, "phase_sin_forward_cuda");
    m.def("phase_sin_backward_cuda", &phase_sin_backward_cuda, "phase_sin_backward_cuda");
    m.def("phase_sin_secant_backward_cuda", &phase_sin_secant_backward_cuda, "phase_sin_secant_backward_cuda");
}
