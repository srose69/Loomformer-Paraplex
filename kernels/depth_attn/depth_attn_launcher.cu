// kernels/depth_attn/depth_attn_launcher.cu -- ATen host wrappers + pybind.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include <type_traits>

#include "depth_attn_kernel.cuh"

// Runtime dispatch table for the templated warp-per-row kernels.
// S is the history length (number of layers in the depth network).
#define DEPTH_ATTN_S_DISPATCH(M) \
    M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9) M(10) \
    M(11) M(12) M(13) M(14) M(15) M(16) M(17) M(18) M(19) M(20) \
    M(21) M(22) M(23) M(24) M(25) M(26) M(27) M(28) M(29) M(30) \
    M(31) M(32)

// Out-of-line dispatch helpers so the switch is not inside the AT_DISPATCH macro argument.
#define FWD_CASE(S) \
    case S: depth_attn_stacked_fwd_warp_kernel<scalar_t, S, 4><<<blocks, threads, shmem, stream>>>(q, k, v, d_out, lse_out, BT, QH, HD, inv_sqrt_hd); break;

template <typename scalar_t>
static inline void launch_stacked_fwd_warp(
    int64_t S, const scalar_t* q, const scalar_t* k, const scalar_t* v,
    scalar_t* d_out, float* lse_out, int64_t BT, int64_t QH, int64_t HD,
    float inv_sqrt_hd, int64_t blocks, int threads, size_t shmem, cudaStream_t stream) {
    switch ((int)S) {
        DEPTH_ATTN_S_DISPATCH(FWD_CASE)
        default: TORCH_CHECK(false, "depth_attn_stacked_forward_cuda: S out of templated range"); break;
    }
}
#undef FWD_CASE

#define BWD_CASE(S) \
    case S: depth_attn_stacked_bwd_warp_kernel<scalar_t, S, 4><<<blocks, threads, shmem, stream>>>(gd, q, k, v, lse, gq, gk, gv, BT, QH, HD, inv_sqrt_hd); break;

template <typename scalar_t>
static inline void launch_stacked_bwd_warp(
    int64_t S, const scalar_t* gd, const scalar_t* q, const scalar_t* k, const scalar_t* v,
    const float* lse, float* gq, scalar_t* gk, scalar_t* gv,
    int64_t BT, int64_t QH, int64_t HD, float inv_sqrt_hd,
    int64_t blocks, int threads, size_t shmem, cudaStream_t stream) {
    switch ((int)S) {
        DEPTH_ATTN_S_DISPATCH(BWD_CASE)
        default: TORCH_CHECK(false, "depth_attn_stacked_backward_cuda: S out of templated range"); break;
    }
}
#undef BWD_CASE

#undef DEPTH_ATTN_S_DISPATCH

std::vector<torch::Tensor> depth_attn_forward_cuda(
    torch::Tensor q, torch::Tensor hist_k, torch::Tensor hist_v) {
    TORCH_CHECK(q.is_cuda() && hist_k.is_cuda() && hist_v.is_cuda(),
                "depth_attn_forward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(hist_k.dim() == 5, "hist_k must be [B,T,S,QH,HD]");
    TORCH_CHECK(hist_v.sizes() == hist_k.sizes(), "hist_v shape mismatch");
    TORCH_CHECK(q.dim() == 2, "q must be [QH,HD]");
    c10::cuda::CUDAGuard device_guard(q.device());

    auto q_c = q.contiguous();
    auto k_c = hist_k.contiguous();
    auto v_c = hist_v.contiguous();
    int64_t B = k_c.size(0), T = k_c.size(1), S = k_c.size(2), QH = k_c.size(3), HD = k_c.size(4);
    TORCH_CHECK(q_c.size(0) == QH && q_c.size(1) == HD, "q shape mismatch");

    auto d_out = torch::empty({B, T, QH, HD}, k_c.options());
    auto w_out = torch::empty({B, T, QH, S}, k_c.options());

    const int threads = 128;
    const int64_t blocks = B * T * QH;
    const int nwarps = (threads + 31) / 32;
    const size_t shmem = (size_t)(S + nwarps) * sizeof(float);
    const float inv_sqrt_hd = 1.0f / sqrtf((float)HD);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, k_c.scalar_type(),
        "depth_attn_forward_cuda", ([&] {
            depth_attn_fwd_kernel<scalar_t><<<blocks, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
                q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(), v_c.data_ptr<scalar_t>(),
                d_out.data_ptr<scalar_t>(), w_out.data_ptr<scalar_t>(), S, QH, HD, inv_sqrt_hd);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {d_out, w_out};
}

std::vector<torch::Tensor> depth_attn_backward_cuda(
    torch::Tensor grad_d, torch::Tensor q, torch::Tensor hist_k,
    torch::Tensor hist_v, torch::Tensor w) {
    TORCH_CHECK(grad_d.is_cuda() && q.is_cuda() && hist_k.is_cuda() && hist_v.is_cuda() && w.is_cuda(),
                "depth_attn_backward_cuda: all inputs must be CUDA tensors");
    c10::cuda::CUDAGuard device_guard(q.device());

    auto gd = grad_d.contiguous();
    auto q_c = q.contiguous();
    auto k_c = hist_k.contiguous();
    auto v_c = hist_v.contiguous();
    auto w_c = w.contiguous();
    int64_t B = k_c.size(0), T = k_c.size(1), S = k_c.size(2), QH = k_c.size(3), HD = k_c.size(4);

    auto grad_q_partial = torch::empty({QH, HD, B * T}, q_c.options().dtype(torch::kFloat32));
    auto grad_q_acc = torch::empty({QH, HD}, q_c.options().dtype(torch::kFloat32));
    auto grad_k = torch::empty_like(k_c);
    auto grad_v = torch::empty_like(v_c);

    const int threads = 128;
    const int64_t blocks = B * T * QH;
    const int nwarps = (threads + 31) / 32;
    const size_t shmem = (size_t)(S + nwarps + 2 * HD) * sizeof(float);
    const float inv_sqrt_hd = 1.0f / sqrtf((float)HD);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, k_c.scalar_type(),
        "depth_attn_backward_cuda", ([&] {
            depth_attn_bwd_kernel<scalar_t><<<blocks, threads, shmem, at::cuda::getCurrentCUDAStream()>>>(
                gd.data_ptr<scalar_t>(), q_c.data_ptr<scalar_t>(), k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(), w_c.data_ptr<scalar_t>(),
                grad_q_partial.data_ptr<float>(), grad_k.data_ptr<scalar_t>(), grad_v.data_ptr<scalar_t>(),
                S, QH, HD, inv_sqrt_hd);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int reduce_threads = 256;
    const int reduce_nwarps = (reduce_threads + 31) / 32;
    const size_t reduce_shmem = (size_t)reduce_nwarps * sizeof(float);
    depth_attn_grad_q_reduce_kernel<<<QH * HD, reduce_threads, reduce_shmem, at::cuda::getCurrentCUDAStream()>>>(
        grad_q_partial.data_ptr<float>(), grad_q_acc.data_ptr<float>(), B * T, QH, HD);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto grad_q = grad_q_acc.to(q_c.scalar_type());
    return {grad_q, grad_k, grad_v};
}

std::tuple<torch::Tensor, torch::Tensor> depth_attn_stacked_forward_cuda(
    torch::Tensor q, torch::Tensor hist_k, torch::Tensor hist_v) {
    TORCH_CHECK(q.is_cuda() && hist_k.is_cuda() && hist_v.is_cuda(),
                "depth_attn_stacked_forward_cuda: all inputs must be CUDA tensors");
    TORCH_CHECK(hist_k.dim() == 5, "hist_k must be [B,T,S,QH,HD]");
    TORCH_CHECK(hist_v.sizes() == hist_k.sizes(), "hist_v shape mismatch");
    TORCH_CHECK(q.dim() == 2, "q must be [QH,HD]");
    c10::cuda::CUDAGuard device_guard(q.device());

    auto q_c = q.contiguous();
    auto k_c = hist_k.contiguous();
    auto v_c = hist_v.contiguous();
    int64_t B = k_c.size(0), T = k_c.size(1), S = k_c.size(2), QH = k_c.size(3), HD = k_c.size(4);
    TORCH_CHECK(q_c.size(0) == QH && q_c.size(1) == HD, "q shape mismatch");

    auto d_out = torch::empty({B, T, QH, HD}, k_c.options());
    auto lse_out = torch::empty({B, T, QH}, k_c.options().dtype(torch::kFloat32));

    const int64_t BT = B * T;
    const float inv_sqrt_hd = 1.0f / sqrtf((float)HD);
    const bool use_warp = (S <= 32) && (HD % 4 == 0);
    const int NWARPS = 4;
    const int threads = 32 * NWARPS;
    const int64_t blocks = use_warp ? (BT * QH + NWARPS - 1) / NWARPS : BT * QH;
    const size_t shmem = 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, k_c.scalar_type(),
        "depth_attn_stacked_forward_cuda", ([&] {
            auto q_ptr = q_c.data_ptr<scalar_t>();
            auto k_ptr = k_c.data_ptr<scalar_t>();
            auto v_ptr = v_c.data_ptr<scalar_t>();
            auto d_ptr = d_out.data_ptr<scalar_t>();
            auto l_ptr = lse_out.data_ptr<float>();
            if constexpr (std::is_same<scalar_t, float>::value || std::is_same<scalar_t, at::Half>::value || std::is_same<scalar_t, at::BFloat16>::value) {
                if (use_warp) {
                    launch_stacked_fwd_warp<scalar_t>(S, q_ptr, k_ptr, v_ptr, d_ptr, l_ptr, BT, QH, HD, inv_sqrt_hd, blocks, threads, shmem, stream);
                } else {
                    const int legacy_threads = 128;
                    const int legacy_nwarps = (legacy_threads + 31) / 32;
                    const size_t legacy_shmem = (size_t)(S + legacy_nwarps) * sizeof(float);
                    depth_attn_stacked_fwd_kernel<scalar_t><<<BT * QH, legacy_threads, legacy_shmem, stream>>>(
                        q_ptr, k_ptr, v_ptr, d_ptr, l_ptr, S, QH, HD, inv_sqrt_hd);
                }
            } else {
                const int legacy_threads = 128;
                const int legacy_nwarps = (legacy_threads + 31) / 32;
                const size_t legacy_shmem = (size_t)(S + legacy_nwarps) * sizeof(float);
                depth_attn_stacked_fwd_kernel<scalar_t><<<BT * QH, legacy_threads, legacy_shmem, stream>>>(
                    q_ptr, k_ptr, v_ptr, d_ptr, l_ptr, S, QH, HD, inv_sqrt_hd);
            }
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(d_out, lse_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> depth_attn_stacked_backward_cuda(
    torch::Tensor grad_d, torch::Tensor q, torch::Tensor hist_k,
    torch::Tensor hist_v, torch::Tensor lse) {
    TORCH_CHECK(grad_d.is_cuda() && q.is_cuda() && hist_k.is_cuda() && hist_v.is_cuda()
                && lse.is_cuda(),
                "depth_attn_stacked_backward_cuda: all inputs must be CUDA tensors");
    c10::cuda::CUDAGuard device_guard(q.device());

    auto gd = grad_d.contiguous();
    auto q_c = q.contiguous();
    auto k_c = hist_k.contiguous();
    auto v_c = hist_v.contiguous();
    auto lse_c = lse.contiguous();
    int64_t B = k_c.size(0), T = k_c.size(1), S = k_c.size(2), QH = k_c.size(3), HD = k_c.size(4);

    auto grad_q_partial = torch::empty({QH, HD, B * T}, q_c.options().dtype(torch::kFloat32));
    auto grad_q_acc = torch::empty({QH, HD}, q_c.options().dtype(torch::kFloat32));
    auto grad_k = torch::empty_like(k_c);
    auto grad_v = torch::empty_like(v_c);

    const int64_t BT = B * T;
    const float inv_sqrt_hd = 1.0f / sqrtf((float)HD);
    const bool use_warp = (S <= 32) && (HD % 4 == 0);
    const int NWARPS = 4;
    const int threads = 32 * NWARPS;
    const int64_t blocks = use_warp ? (BT * QH + NWARPS - 1) / NWARPS : BT * QH;
    const size_t shmem = 0;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, k_c.scalar_type(),
        "depth_attn_stacked_backward_cuda", ([&] {
            auto gd_ptr = gd.data_ptr<scalar_t>();
            auto q_ptr = q_c.data_ptr<scalar_t>();
            auto k_ptr = k_c.data_ptr<scalar_t>();
            auto v_ptr = v_c.data_ptr<scalar_t>();
            auto l_ptr = lse_c.data_ptr<float>();
            auto gq_ptr = grad_q_partial.data_ptr<float>();
            auto gk_ptr = grad_k.data_ptr<scalar_t>();
            auto gv_ptr = grad_v.data_ptr<scalar_t>();
            if constexpr (std::is_same<scalar_t, float>::value || std::is_same<scalar_t, at::Half>::value || std::is_same<scalar_t, at::BFloat16>::value) {
                if (use_warp) {
                    launch_stacked_bwd_warp<scalar_t>(S, gd_ptr, q_ptr, k_ptr, v_ptr, l_ptr, gq_ptr, gk_ptr, gv_ptr, BT, QH, HD, inv_sqrt_hd, blocks, threads, shmem, stream);
                } else {
                    const int legacy_threads = 128;
                    const int legacy_nwarps = (legacy_threads + 31) / 32;
                    const size_t legacy_shmem = (size_t)(2 * S + legacy_nwarps + 2 * HD) * sizeof(float);
                    depth_attn_stacked_bwd_kernel<scalar_t><<<BT * QH, legacy_threads, legacy_shmem, stream>>>(
                        gd_ptr, q_ptr, k_ptr, v_ptr, l_ptr, gq_ptr, gk_ptr, gv_ptr,
                        S, QH, HD, inv_sqrt_hd);
                }
            } else {
                const int legacy_threads = 128;
                const int legacy_nwarps = (legacy_threads + 31) / 32;
                const size_t legacy_shmem = (size_t)(2 * S + legacy_nwarps + 2 * HD) * sizeof(float);
                depth_attn_stacked_bwd_kernel<scalar_t><<<BT * QH, legacy_threads, legacy_shmem, stream>>>(
                    gd_ptr, q_ptr, k_ptr, v_ptr, l_ptr, gq_ptr, gk_ptr, gv_ptr,
                    S, QH, HD, inv_sqrt_hd);
            }
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int reduce_threads = 128;
    const int reduce_nwarps = reduce_threads >> 5;
    const int64_t reduce_blocks = (QH * HD + reduce_nwarps - 1) / reduce_nwarps;
    depth_attn_grad_q_reduce_kernel<<<reduce_blocks, reduce_threads, 0, stream>>>(
        grad_q_partial.data_ptr<float>(), grad_q_acc.data_ptr<float>(), BT, QH, HD);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto grad_q = grad_q_acc.to(q_c.scalar_type());
    return std::make_tuple(grad_q, grad_k, grad_v);
}

std::vector<torch::Tensor> depth_attn_list_forward_cuda(
    torch::Tensor q, std::vector<torch::Tensor> hist_k, std::vector<torch::Tensor> hist_v) {
    TORCH_CHECK(!hist_k.empty(), "depth_attn_list_forward_cuda: history must be non-empty");
    TORCH_CHECK(hist_k.size() == hist_v.size(), "depth_attn_list_forward_cuda: K/V history length mismatch");
    auto k = torch::stack(hist_k, 2);
    auto v = torch::stack(hist_v, 2);
    return depth_attn_forward_cuda(q, k, v);
}

std::vector<torch::Tensor> depth_attn_list_backward_cuda(
    torch::Tensor grad_d, torch::Tensor q, std::vector<torch::Tensor> hist_k,
    std::vector<torch::Tensor> hist_v, torch::Tensor w) {
    TORCH_CHECK(!hist_k.empty(), "depth_attn_list_backward_cuda: history must be non-empty");
    TORCH_CHECK(hist_k.size() == hist_v.size(), "depth_attn_list_backward_cuda: K/V history length mismatch");
    auto k = torch::stack(hist_k, 2);
    auto v = torch::stack(hist_v, 2);
    auto grads = depth_attn_backward_cuda(grad_d, q, k, v, w);
    std::vector<torch::Tensor> out;
    out.push_back(grads[0]);
    const int64_t s = (int64_t)hist_k.size();
    for (int64_t j = 0; j < s; ++j) out.push_back(grads[1].select(2, j));
    for (int64_t j = 0; j < s; ++j) out.push_back(grads[2].select(2, j));
    return out;
}



std::tuple<torch::Tensor, torch::Tensor> depth_attn_forward_op(
    torch::Tensor q, torch::Tensor hist_k, torch::Tensor hist_v) {
    auto out = depth_attn_forward_cuda(q, hist_k, hist_v);
    return std::make_tuple(out[0], out[1]);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> depth_attn_backward_op(
    torch::Tensor grad_d, torch::Tensor q, torch::Tensor hist_k,
    torch::Tensor hist_v, torch::Tensor w) {
    auto out = depth_attn_backward_cuda(grad_d, q, hist_k, hist_v, w);
    return std::make_tuple(out[0], out[1], out[2]);
}

TORCH_LIBRARY(loomformer_depth_attn, m) {
    m.def("forward(Tensor q, Tensor hist_k, Tensor hist_v) -> (Tensor, Tensor)");
    m.def("backward(Tensor grad_d, Tensor q, Tensor hist_k, Tensor hist_v, Tensor w) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(loomformer_depth_attn, CUDA, m) {
    m.impl("forward", TORCH_FN(depth_attn_forward_op));
    m.impl("backward", TORCH_FN(depth_attn_backward_op));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depth_attn_forward", &depth_attn_forward_cuda, "depth_attn_forward");
    m.def("depth_attn_backward", &depth_attn_backward_cuda, "depth_attn_backward");
    m.def("depth_attn_list_forward", &depth_attn_list_forward_cuda, "depth_attn_list_forward");
    m.def("depth_attn_list_backward", &depth_attn_list_backward_cuda, "depth_attn_list_backward");
    m.def("depth_attn_stacked_forward", &depth_attn_stacked_forward_cuda, "depth_attn_stacked_forward");
    m.def("depth_attn_stacked_backward", &depth_attn_stacked_backward_cuda, "depth_attn_stacked_backward");
}
