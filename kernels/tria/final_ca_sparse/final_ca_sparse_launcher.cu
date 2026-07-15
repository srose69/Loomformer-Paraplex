#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "final_ca_sparse_kernel.cuh"

static void final_ca_check(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& allowed) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && allowed.is_cuda(),
                "final CA: CUDA inputs required");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3,
                "final CA: q/k/v must be rank 3");
    TORCH_CHECK(q.size(0) == k.size(0) && k.sizes() == v.sizes() && q.size(2) == k.size(2),
                "final CA: q/k/v shape mismatch");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(),
                "final CA: q/k/v dtype mismatch");
    TORCH_CHECK(allowed.scalar_type() == torch::kBool && allowed.dim() == 3 &&
                allowed.size(0) == q.size(0) && allowed.size(1) == q.size(1) &&
                allowed.size(2) == k.size(1),
                "final CA: allowed must be bool [B,T,K]");
    TORCH_CHECK(q.device() == k.device() && k.device() == v.device() &&
                v.device() == allowed.device(), "final CA: device mismatch");
}

std::vector<torch::Tensor> final_ca_sparse_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor allowed,
    double scale) {
    final_ca_check(q, k, v, allowed);
    c10::cuda::CUDAGuard guard(q.device());
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto vc = v.contiguous();
    auto ac = allowed.contiguous();
    const int64_t B = qc.size(0);
    const int64_t T = qc.size(1);
    const int64_t K = kc.size(1);
    const int64_t D = qc.size(2);
    TORCH_CHECK(K > 0 && K <= 256, "final CA: expected 1..256 sparse keys");

    auto out = torch::empty_like(qc);
    auto lse = torch::empty({B, T}, qc.options().dtype(torch::kFloat32));
    const size_t smem = (K + FINAL_CA_THREADS) * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
        "final_ca_sparse_forward_cuda", ([&] {
            final_ca_sparse_forward_kernel<scalar_t>
                <<<B * T, FINAL_CA_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
                    qc.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(),
                    ac.data_ptr<bool>(), out.data_ptr<scalar_t>(), lse.data_ptr<float>(),
                    B, T, K, D, (float)scale);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, lse};
}

std::vector<torch::Tensor> final_ca_sparse_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor allowed,
    torch::Tensor lse,
    double scale) {
    final_ca_check(q, k, v, allowed);
    TORCH_CHECK(grad_out.is_cuda() && grad_out.sizes() == q.sizes(),
                "final CA: grad_out shape mismatch");
    TORCH_CHECK(lse.is_cuda() && lse.scalar_type() == torch::kFloat32 && lse.dim() == 2 &&
                lse.size(0) == q.size(0) && lse.size(1) == q.size(1),
                "final CA: lse must be float [B,T]");
    TORCH_CHECK(grad_out.device() == q.device() && lse.device() == q.device(),
                "final CA: backward device mismatch");

    c10::cuda::CUDAGuard guard(q.device());
    auto go = grad_out.contiguous();
    auto qc = q.contiguous();
    auto kc = k.contiguous();
    auto vc = v.contiguous();
    auto ac = allowed.contiguous();
    auto lc = lse.contiguous();
    const int64_t B = qc.size(0);
    const int64_t T = qc.size(1);
    const int64_t K = kc.size(1);
    const int64_t D = qc.size(2);

    auto gq = torch::empty_like(qc);
    auto gk = torch::empty_like(kc);
    auto gv = torch::empty_like(vc);
    auto weights = torch::empty({B, T, K}, qc.options().dtype(torch::kFloat32));
    auto dscore = torch::empty_like(weights);
    const size_t smem = (3 * K + FINAL_CA_THREADS) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, qc.scalar_type(),
        "final_ca_sparse_backward_qkv", ([&] {
            using q_t = scalar_t;
            AT_DISPATCH_FLOATING_TYPES_AND2(
                at::ScalarType::Half, at::ScalarType::BFloat16, go.scalar_type(),
                "final_ca_sparse_backward_grad", ([&] {
                    using g_t = scalar_t;
                    final_ca_sparse_backward_rows_kernel<q_t, g_t>
                        <<<B * T, FINAL_CA_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
                            go.data_ptr<g_t>(), qc.data_ptr<q_t>(), kc.data_ptr<q_t>(),
                            vc.data_ptr<q_t>(), ac.data_ptr<bool>(), lc.data_ptr<float>(),
                            gq.data_ptr<q_t>(), weights.data_ptr<float>(), dscore.data_ptr<float>(),
                            B, T, K, D, (float)scale);
                    const dim3 grid((unsigned)(B * K), (unsigned)((D + FINAL_CA_THREADS - 1) / FINAL_CA_THREADS));
                    final_ca_sparse_backward_keys_kernel<q_t, g_t>
                        <<<grid, FINAL_CA_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                            go.data_ptr<g_t>(), qc.data_ptr<q_t>(), weights.data_ptr<float>(),
                            dscore.data_ptr<float>(), gk.data_ptr<q_t>(), gv.data_ptr<q_t>(),
                            T, K, D);
                }));
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gq, gk, gv};
}
