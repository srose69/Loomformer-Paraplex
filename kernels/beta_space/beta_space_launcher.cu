// kernels/beta_space/beta_space_launcher.cu -- ATen + cuBLAS host wrappers
// (row_gemm_batched / row_grad_a_batched / row_grad_w_batched, the
// CHECK_*/CUBLAS_CHECK macros, and beta_forward_cuda/beta_backward_cuda)
// + pybind. All cublas/TORCH_CHECK-dependent code lives here, not in the
// device-only .cuh, since none of it can run on-device anyway.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <limits>
#include <vector>

#include "beta_space_kernel.cuh"

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT_OR_BF16(x) TORCH_CHECK(((x).scalar_type() == at::kFloat || (x).scalar_type() == at::kBFloat16), #x " must be float32 or bfloat16")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CUBLAS_CHECK(expr) do { \
    cublasStatus_t _status = (expr); \
    TORCH_CHECK(_status == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", static_cast<int>(_status)); \
} while (0)

static inline int i32(int64_t x, const char* name) {
    TORCH_CHECK(x >= 0 && x <= std::numeric_limits<int>::max(), name, " exceeds int32");
    return static_cast<int>(x);
}
static inline long long ll(int64_t x) { return static_cast<long long>(x); }

static inline void check_same_dtype(const torch::Tensor& ref, const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.scalar_type() == ref.scalar_type(), name, " dtype mismatch");
}

static inline void check_same_device(const torch::Tensor& ref, const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.device() == ref.device(), name, " device mismatch");
}

static inline cudaDataType_t cuda_dtype_from_scalar(at::ScalarType dtype) {
    if (dtype == at::kFloat) return CUDA_R_32F;
    if (dtype == at::kBFloat16) return CUDA_R_16BF;
    TORCH_CHECK(false, "unsupported dtype");
}

// Row-major strided-batched GEMM:
//   C_b[M,NOUT] = beta*C_b + A_b[M,K] @ W_b[NOUT,K].T
// A/W/C are row-major views with arbitrary row stride and batch stride.
// float32 path keeps cublasSgemm. bfloat16 path uses cublasGemmStridedBatchedEx
// with FP32 accumulation and BF16 output, matching CUDA/PyTorch's usual BF16 GEMM contract.
static inline void row_gemm_batched(
    cublasHandle_t handle, at::ScalarType dtype,
    const void* A, int64_t a_rs, int64_t strideA,
    const void* W, int64_t w_rs, int64_t strideW,
    void* C, int64_t c_rs, int64_t strideC,
    int64_t M, int64_t NOUT, int64_t K, int64_t batch, float beta) {
    const float alpha = 1.0f;
    if (dtype == at::kFloat) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            i32(NOUT, "NOUT"), i32(M, "M"), i32(K, "K"),
            &alpha,
            static_cast<const float*>(W), i32(w_rs, "w_rs"), ll(strideW),
            static_cast<const float*>(A), i32(a_rs, "a_rs"), ll(strideA),
            &beta,
            static_cast<float*>(C), i32(c_rs, "c_rs"), ll(strideC),
            i32(batch, "batch")));
    } else {
        const cudaDataType_t dt = CUDA_R_16BF;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            i32(NOUT, "NOUT"), i32(M, "M"), i32(K, "K"),
            &alpha,
            W, dt, i32(w_rs, "w_rs"), ll(strideW),
            A, dt, i32(a_rs, "a_rs"), ll(strideA),
            &beta,
            C, dt, i32(c_rs, "c_rs"), ll(strideC),
            i32(batch, "batch"), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
}

// dA_b[M,K] = beta*dA_b + dY_b[M,NOUT] @ W_b[NOUT,K].
static inline void row_grad_a_batched(
    cublasHandle_t handle, at::ScalarType dtype,
    const void* dY, int64_t dy_rs, int64_t strideDY,
    const void* W, int64_t w_rs, int64_t strideW,
    void* dA, int64_t da_rs, int64_t strideDA,
    int64_t M, int64_t NOUT, int64_t K, int64_t batch, float beta) {
    const float alpha = 1.0f;
    if (dtype == at::kFloat) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            i32(K, "K"), i32(M, "M"), i32(NOUT, "NOUT"),
            &alpha,
            static_cast<const float*>(W), i32(w_rs, "w_rs"), ll(strideW),
            static_cast<const float*>(dY), i32(dy_rs, "dy_rs"), ll(strideDY),
            &beta,
            static_cast<float*>(dA), i32(da_rs, "da_rs"), ll(strideDA),
            i32(batch, "batch")));
    } else {
        const cudaDataType_t dt = CUDA_R_16BF;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            i32(K, "K"), i32(M, "M"), i32(NOUT, "NOUT"),
            &alpha,
            W, dt, i32(w_rs, "w_rs"), ll(strideW),
            dY, dt, i32(dy_rs, "dy_rs"), ll(strideDY),
            &beta,
            dA, dt, i32(da_rs, "da_rs"), ll(strideDA),
            i32(batch, "batch"), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
}

// dW_b[NOUT,K] = dY_b[M,NOUT].T @ A_b[M,K].
static inline void row_grad_w_batched(
    cublasHandle_t handle, at::ScalarType dtype,
    const void* dY, int64_t dy_rs, int64_t strideDY,
    const void* A, int64_t a_rs, int64_t strideA,
    void* dW, int64_t dw_rs, int64_t strideDW,
    int64_t M, int64_t NOUT, int64_t K, int64_t batch) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (dtype == at::kFloat) {
        CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            i32(K, "K"), i32(NOUT, "NOUT"), i32(M, "M"),
            &alpha,
            static_cast<const float*>(A), i32(a_rs, "a_rs"), ll(strideA),
            static_cast<const float*>(dY), i32(dy_rs, "dy_rs"), ll(strideDY),
            &beta,
            static_cast<float*>(dW), i32(dw_rs, "dw_rs"), ll(strideDW),
            i32(batch, "batch")));
    } else {
        const cudaDataType_t dt = CUDA_R_16BF;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T,
            i32(K, "K"), i32(NOUT, "NOUT"), i32(M, "M"),
            &alpha,
            A, dt, i32(a_rs, "a_rs"), ll(strideA),
            dY, dt, i32(dy_rs, "dy_rs"), ll(strideDY),
            &beta,
            dW, dt, i32(dw_rs, "dw_rs"), ll(strideDW),
            i32(batch, "batch"), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
}

static inline void check_common_inputs(
    const torch::Tensor& u, const torch::Tensor& q_h, const torch::Tensor& k_ctx_h,
    const torch::Tensor& c_h, const torch::Tensor& d_h,
    const torch::Tensor& w1_imag_compact,
    int64_t hidden_per_q_head, int64_t head_dim, int64_t n_q_heads,
    bool open_sectors) {
    CHECK_CUDA(u); CHECK_CUDA(q_h); CHECK_CUDA(k_ctx_h); CHECK_CUDA(c_h); CHECK_CUDA(d_h); CHECK_CUDA(w1_imag_compact);
    CHECK_FLOAT_OR_BF16(u); CHECK_FLOAT_OR_BF16(q_h); CHECK_FLOAT_OR_BF16(k_ctx_h); CHECK_FLOAT_OR_BF16(c_h); CHECK_FLOAT_OR_BF16(d_h); CHECK_FLOAT_OR_BF16(w1_imag_compact);
    check_same_device(u, q_h, "q_h");
    check_same_device(u, k_ctx_h, "k_ctx_h");
    check_same_device(u, c_h, "c_h");
    check_same_device(u, d_h, "d_h");
    check_same_device(u, w1_imag_compact, "w1_imag_compact");
    check_same_dtype(u, q_h, "q_h");
    check_same_dtype(u, k_ctx_h, "k_ctx_h");
    check_same_dtype(u, c_h, "c_h");
    check_same_dtype(u, d_h, "d_h");
    check_same_dtype(u, w1_imag_compact, "w1_imag_compact");
    TORCH_CHECK(u.dim() == 3, "u must have shape (B,T,N)");
    TORCH_CHECK(q_h.dim() == 4, "q_h must have shape (B,T,QH,HD)");
    TORCH_CHECK(k_ctx_h.sizes() == q_h.sizes(), "k_ctx_h shape mismatch");
    TORCH_CHECK(c_h.sizes() == q_h.sizes(), "c_h shape mismatch");
    TORCH_CHECK(d_h.sizes() == q_h.sizes(), "d_h shape mismatch");
    TORCH_CHECK(q_h.size(0) == u.size(0) && q_h.size(1) == u.size(1), "B/T mismatch");
    TORCH_CHECK(q_h.size(2) == n_q_heads && q_h.size(3) == head_dim, "QH/HD mismatch");
    TORCH_CHECK(u.size(2) == n_q_heads * head_dim, "N must equal QH*HD");
    TORCH_CHECK(w1_imag_compact.dim() == 2, "w1_imag_compact must have shape (HIDDEN,IMAG_IN)");
    TORCH_CHECK(w1_imag_compact.size(0) == hidden_per_q_head * n_q_heads, "HIDDEN mismatch");
    int64_t expected_imag_in = open_sectors ? (head_dim + 4 * u.size(2)) : (u.size(2) + 4 * head_dim);
    TORCH_CHECK(w1_imag_compact.size(1) == expected_imag_in, "IMAG_IN mismatch");
}

static torch::Tensor pack_beta_inputs(
    const torch::Tensor& u, const torch::Tensor& q_h, const torch::Tensor& k_ctx_h,
    const torch::Tensor& c_h, const torch::Tensor& d_h,
    int64_t hidden_per_q_head, int64_t head_dim, int64_t n_q_heads,
    int64_t imag_in, bool open_sectors) {
    const int64_t M = u.size(0) * u.size(1);
    const int64_t N = u.size(2);
    auto r_pack = torch::empty({n_q_heads, M, imag_in}, u.options());
    const int threads = 256;
    const int64_t pack_total4 = n_q_heads * M * (imag_in / 4);
    if (u.scalar_type() == at::kFloat) {
        if (!open_sectors) {
            pack_r_mask4_head_kernel<float><<<(pack_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                u.data_ptr<float>(), q_h.data_ptr<float>(), k_ctx_h.data_ptr<float>(),
                c_h.data_ptr<float>(), d_h.data_ptr<float>(), r_pack.data_ptr<float>(),
                M, N, imag_in, head_dim, n_q_heads);
        } else {
            pack_r_mask4_open_kernel<float><<<(pack_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                u.data_ptr<float>(), q_h.data_ptr<float>(), k_ctx_h.data_ptr<float>(),
                c_h.data_ptr<float>(), d_h.data_ptr<float>(), r_pack.data_ptr<float>(),
                M, N, imag_in, head_dim, n_q_heads);
        }
    } else {
        if (!open_sectors) {
            pack_r_mask4_head_kernel<at::BFloat16><<<(pack_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                u.data_ptr<at::BFloat16>(), q_h.data_ptr<at::BFloat16>(), k_ctx_h.data_ptr<at::BFloat16>(),
                c_h.data_ptr<at::BFloat16>(), d_h.data_ptr<at::BFloat16>(), r_pack.data_ptr<at::BFloat16>(),
                M, N, imag_in, head_dim, n_q_heads);
        } else {
            pack_r_mask4_open_kernel<at::BFloat16><<<(pack_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                u.data_ptr<at::BFloat16>(), q_h.data_ptr<at::BFloat16>(), k_ctx_h.data_ptr<at::BFloat16>(),
                c_h.data_ptr<at::BFloat16>(), d_h.data_ptr<at::BFloat16>(), r_pack.data_ptr<at::BFloat16>(),
                M, N, imag_in, head_dim, n_q_heads);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return r_pack;
}


std::vector<torch::Tensor> beta_forward_cuda(
    torch::Tensor u, torch::Tensor q_h, torch::Tensor k_ctx_h,
    torch::Tensor c_h, torch::Tensor d_h, torch::Tensor w1_imag_compact,
    int64_t hidden_per_q_head, int64_t head_dim, int64_t n_q_heads,
    bool open_sectors) {

    check_common_inputs(u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
                        hidden_per_q_head, head_dim, n_q_heads, open_sectors);
    c10::cuda::CUDAGuard device_guard(u.device());

    auto u_c = u.contiguous();
    auto q_c = q_h.contiguous();
    auto k_c = k_ctx_h.contiguous();
    auto c_c = c_h.contiguous();
    auto d_c = d_h.contiguous();
    auto w_c = w1_imag_compact.contiguous();
    at::ScalarType dtype = u_c.scalar_type();

    int64_t B = u_c.size(0);
    int64_t T = u_c.size(1);
    int64_t N = u_c.size(2);
    int64_t M = B * T;
    int64_t HIDDEN = w_c.size(0);
    int64_t K = w_c.size(1);
    TORCH_CHECK((N % 4) == 0 && (head_dim % 4) == 0 && (K % 4) == 0, "maskpack path requires N, head_dim, IMAG_IN divisible by 4");

    auto r_pack = pack_beta_inputs(
        u_c, q_c, k_c, c_c, d_c,
        hidden_per_q_head, head_dim, n_q_heads, K, open_sectors);
    auto out2d = torch::empty({M, HIDDEN}, u_c.options());

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    row_gemm_batched(handle, dtype,
        r_pack.data_ptr(), K, M * K,
        w_c.data_ptr(), K, hidden_per_q_head * K,
        out2d.data_ptr(), HIDDEN, hidden_per_q_head,
        M, hidden_per_q_head, K, n_q_heads, 0.0f);

    return {out2d.view({B, T, HIDDEN}), r_pack, w_c};
}

std::vector<torch::Tensor> beta_backward_cuda(
    torch::Tensor grad_out, torch::Tensor r_pack, torch::Tensor w1_imag_compact,
    int64_t B, int64_t T, int64_t N, int64_t HIDDEN, int64_t IMAG_IN,
    int64_t hidden_per_q_head, int64_t head_dim, int64_t n_q_heads,
    bool open_sectors) {

    CHECK_CUDA(grad_out); CHECK_CUDA(r_pack); CHECK_CUDA(w1_imag_compact);
    CHECK_FLOAT_OR_BF16(grad_out); CHECK_FLOAT_OR_BF16(r_pack); CHECK_FLOAT_OR_BF16(w1_imag_compact);
    check_same_device(grad_out, r_pack, "r_pack");
    check_same_device(grad_out, w1_imag_compact, "w1_imag_compact");
    check_same_dtype(grad_out, r_pack, "r_pack");
    check_same_dtype(grad_out, w1_imag_compact, "w1_imag_compact");
    c10::cuda::CUDAGuard device_guard(grad_out.device());

    auto go = grad_out.contiguous().view({B * T, HIDDEN});
    auto rp = r_pack.contiguous();
    auto w_c = w1_imag_compact.contiguous();
    at::ScalarType dtype = go.scalar_type();

    int64_t M = B * T;
    int64_t K = IMAG_IN;

    auto grad_u = torch::empty({B, T, N}, go.options());
    auto grad_q = torch::empty({B, T, N}, go.options());
    auto grad_k = torch::empty({B, T, N}, go.options());
    auto grad_c = torch::empty({B, T, N}, go.options());
    auto grad_d = torch::empty({B, T, N}, go.options());
    auto grad_w = torch::empty_like(w_c);
    auto grad_r_pack = torch::empty_like(rp);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    row_grad_w_batched(handle, dtype,
        go.data_ptr(), HIDDEN, hidden_per_q_head,
        rp.data_ptr(), K, M * K,
        grad_w.data_ptr(), K, hidden_per_q_head * K,
        M, hidden_per_q_head, K, n_q_heads);

    row_grad_a_batched(handle, dtype,
        go.data_ptr(), HIDDEN, hidden_per_q_head,
        w_c.data_ptr(), K, hidden_per_q_head * K,
        grad_r_pack.data_ptr(), K, M * K,
        M, hidden_per_q_head, K, n_q_heads, 0.0f);

    TORCH_CHECK((N % 4) == 0 && (head_dim % 4) == 0 && (K % 4) == 0, "maskpack path requires N, head_dim, IMAG_IN divisible by 4");
    const int threads = 256;
    int64_t in_total4 = M * (N / 4);
    if (dtype == at::kFloat) {
        if (!open_sectors) {
            unpack_grad_r_mask4_head_kernel<float><<<(in_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_r_pack.data_ptr<float>(), grad_u.data_ptr<float>(), grad_q.data_ptr<float>(),
                grad_k.data_ptr<float>(), grad_c.data_ptr<float>(), grad_d.data_ptr<float>(),
                M, N, K, head_dim, n_q_heads);
        } else {
            unpack_grad_r_mask4_open_kernel<float><<<(in_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_r_pack.data_ptr<float>(), grad_u.data_ptr<float>(), grad_q.data_ptr<float>(),
                grad_k.data_ptr<float>(), grad_c.data_ptr<float>(), grad_d.data_ptr<float>(),
                M, N, K, head_dim, n_q_heads);
        }
    } else {
        if (!open_sectors) {
            unpack_grad_r_mask4_head_kernel<at::BFloat16><<<(in_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_r_pack.data_ptr<at::BFloat16>(), grad_u.data_ptr<at::BFloat16>(), grad_q.data_ptr<at::BFloat16>(),
                grad_k.data_ptr<at::BFloat16>(), grad_c.data_ptr<at::BFloat16>(), grad_d.data_ptr<at::BFloat16>(),
                M, N, K, head_dim, n_q_heads);
        } else {
            unpack_grad_r_mask4_open_kernel<at::BFloat16><<<(in_total4 + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                grad_r_pack.data_ptr<at::BFloat16>(), grad_u.data_ptr<at::BFloat16>(), grad_q.data_ptr<at::BFloat16>(),
                grad_k.data_ptr<at::BFloat16>(), grad_c.data_ptr<at::BFloat16>(), grad_d.data_ptr<at::BFloat16>(),
                M, N, K, head_dim, n_q_heads);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // BF16 correctness fix for shared sectors.
    // The compact backward above computes one BF16 partial gradient per q-head and then
    // sums those rounded partials in the unpack kernel. PyTorch's dense BF16 backward
    // computes the shared input columns in one GEMM reduction over the whole HIDDEN axis
    // and rounds only the final value. For shared sectors that matters numerically
    // (notably U in head mode). Recompute/overwrite those shared gradients with the
    // dense-equivalent GEMM shape while keeping the fast compact path for everything else.
    if (dtype == at::kBFloat16) {
        const int64_t elem_size = 2;
        const char* w_base = static_cast<const char*>(w_c.data_ptr());
        if (!open_sectors) {
            const void* w_u = static_cast<const void*>(w_base + (3 * head_dim) * elem_size);
            row_grad_a_batched(handle, dtype,
                go.data_ptr(), HIDDEN, HIDDEN * M,
                w_u, K, HIDDEN * K,
                grad_u.data_ptr(), N, M * N,
                M, HIDDEN, N, 1, 0.0f);
        } else {
            const void* w_k = static_cast<const void*>(w_base + head_dim * elem_size);
            const void* w_cptr = static_cast<const void*>(w_base + (head_dim + N) * elem_size);
            const void* w_u = static_cast<const void*>(w_base + (head_dim + 2 * N) * elem_size);
            const void* w_d = static_cast<const void*>(w_base + (head_dim + 3 * N) * elem_size);
            row_grad_a_batched(handle, dtype,
                go.data_ptr(), HIDDEN, HIDDEN * M,
                w_k, K, HIDDEN * K,
                grad_k.data_ptr(), N, M * N,
                M, HIDDEN, N, 1, 0.0f);
            row_grad_a_batched(handle, dtype,
                go.data_ptr(), HIDDEN, HIDDEN * M,
                w_cptr, K, HIDDEN * K,
                grad_c.data_ptr(), N, M * N,
                M, HIDDEN, N, 1, 0.0f);
            row_grad_a_batched(handle, dtype,
                go.data_ptr(), HIDDEN, HIDDEN * M,
                w_u, K, HIDDEN * K,
                grad_u.data_ptr(), N, M * N,
                M, HIDDEN, N, 1, 0.0f);
            row_grad_a_batched(handle, dtype,
                go.data_ptr(), HIDDEN, HIDDEN * M,
                w_d, K, HIDDEN * K,
                grad_d.data_ptr(), N, M * N,
                M, HIDDEN, N, 1, 0.0f);
        }
    }

    return {grad_u, grad_q, grad_k, grad_c, grad_d, grad_w};
}

std::vector<torch::Tensor> beta_backward_cuda_recompute(
    torch::Tensor grad_out,
    torch::Tensor u, torch::Tensor q_h, torch::Tensor k_ctx_h,
    torch::Tensor c_h, torch::Tensor d_h, torch::Tensor w1_imag_compact,
    int64_t hidden_per_q_head, int64_t head_dim, int64_t n_q_heads,
    bool open_sectors) {
    check_common_inputs(u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
                        hidden_per_q_head, head_dim, n_q_heads, open_sectors);
    c10::cuda::CUDAGuard device_guard(u.device());
    auto u_c = u.contiguous();
    auto q_c = q_h.contiguous();
    auto k_c = k_ctx_h.contiguous();
    auto c_c = c_h.contiguous();
    auto d_c = d_h.contiguous();
    auto w_c = w1_imag_compact.contiguous();
    const int64_t B = u_c.size(0);
    const int64_t T = u_c.size(1);
    const int64_t N = u_c.size(2);
    const int64_t HIDDEN = w_c.size(0);
    const int64_t K = w_c.size(1);
    TORCH_CHECK((N % 4) == 0 && (head_dim % 4) == 0 && (K % 4) == 0,
                "maskpack path requires N, head_dim, IMAG_IN divisible by 4");
    auto r_pack = pack_beta_inputs(
        u_c, q_c, k_c, c_c, d_c,
        hidden_per_q_head, head_dim, n_q_heads, K, open_sectors);
    return beta_backward_cuda(
        grad_out, r_pack, w_c, B, T, N, HIDDEN, K,
        hidden_per_q_head, head_dim, n_q_heads, open_sectors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("beta_forward_cuda", &beta_forward_cuda, "beta_forward_cuda");
    m.def("beta_backward_cuda", &beta_backward_cuda, "beta_backward_cuda");
    m.def("beta_backward_cuda_recompute", &beta_backward_cuda_recompute, "beta_backward_cuda_recompute");
}
