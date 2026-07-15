#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "temporal_carry_kernel.cuh"

std::vector<torch::Tensor> temporal_carry_forward_cuda(
    torch::Tensor depth_carry, torch::Tensor reset_mask) {
    TORCH_CHECK(depth_carry.is_cuda() && reset_mask.is_cuda(), "temporal_carry: CUDA inputs required");
    TORCH_CHECK(depth_carry.dim()==5 && depth_carry.size(3)==3 && depth_carry.size(4)==3,
                "depth_carry must be [B,T,H,3,3]");
    TORCH_CHECK(reset_mask.sizes()==depth_carry.sizes().slice(0,2), "reset_mask must be [B,T]");
    TORCH_CHECK(reset_mask.scalar_type()==torch::kBool, "reset_mask must be bool");
    c10::cuda::CUDAGuard guard(depth_carry.device());
    auto dc=depth_carry.contiguous(); auto rm=reset_mask.contiguous();
    const int64_t B=dc.size(0),T=dc.size(1),H=dc.size(2);
    auto out=torch::empty(dc.sizes(),dc.options().dtype(torch::kFloat32));
    auto scale=torch::empty({B,T,H},dc.options().dtype(torch::kFloat32));
    const int threads=256; const int64_t blocks=(B*H+threads-1)/threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,dc.scalar_type(),
        "temporal_carry_forward_cuda",([&]{
            temporal_carry_forward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                dc.data_ptr<scalar_t>(),rm.data_ptr<bool>(),out.data_ptr<float>(),
                scale.data_ptr<float>(),B,T,H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out,scale};
}

torch::Tensor temporal_carry_backward_cuda(
    torch::Tensor grad_document_carry, torch::Tensor depth_carry,
    torch::Tensor document_carry, torch::Tensor scale, torch::Tensor reset_mask) {
    TORCH_CHECK(grad_document_carry.is_cuda() && depth_carry.is_cuda() && document_carry.is_cuda() &&
                scale.is_cuda() && reset_mask.is_cuda(), "temporal_carry backward: CUDA inputs required");
    TORCH_CHECK(grad_document_carry.scalar_type()==torch::kFloat32 &&
                document_carry.scalar_type()==torch::kFloat32 && scale.scalar_type()==torch::kFloat32,
                "temporal_carry backward: document carry, its gradient and scale must be float32");
    c10::cuda::CUDAGuard guard(depth_carry.device());
    auto go=grad_document_carry.contiguous(); auto dc=depth_carry.contiguous();
    auto doc=document_carry.contiguous(); auto sc=scale.contiguous(); auto rm=reset_mask.contiguous();
    const int64_t B=dc.size(0),T=dc.size(1),H=dc.size(2);
    auto gd=torch::empty_like(dc);
    const int threads=256; const int64_t blocks=(B*H+threads-1)/threads;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,dc.scalar_type(),
        "temporal_carry_backward_cuda",([&]{
            temporal_carry_backward_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                go.data_ptr<float>(),dc.data_ptr<scalar_t>(),doc.data_ptr<float>(),
                sc.data_ptr<float>(),rm.data_ptr<bool>(),gd.data_ptr<scalar_t>(),B,T,H);
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gd;
}
