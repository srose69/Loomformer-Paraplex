#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include "depth_replay_kernel.cuh"

static void depth_replay_check(
    torch::Tensor grad_carry, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor r_ptrs, torch::Tensor i_ptrs, torch::Tensor o_ptrs,
    torch::Tensor axes, torch::Tensor seed, torch::Tensor seed_valid,
    int64_t layer_index) {
    TORCH_CHECK(grad_carry.is_cuda() && r.is_cuda() && i.is_cuda() && o.is_cuda(),
                "depth replay requires CUDA tensors");
    TORCH_CHECK(r.dim() == 3 && i.sizes() == r.sizes() && o.sizes() == r.sizes(),
                "R/I/O must be matching [B,T,H]");
    TORCH_CHECK(grad_carry.numel() == r.numel() * 9,
                "grad_carry must contain nine values per R/I/O element");
    TORCH_CHECK(r.scalar_type() == i.scalar_type() && r.scalar_type() == o.scalar_type() &&
                r.scalar_type() == grad_carry.scalar_type(), "activation dtype mismatch");
    TORCH_CHECK(r.device() == i.device() && r.device() == o.device() &&
                r.device() == grad_carry.device(), "activation device mismatch");
    TORCH_CHECK(r_ptrs.is_cuda() && i_ptrs.is_cuda() && o_ptrs.is_cuda() && axes.is_cuda(),
                "replay tables must be CUDA tensors");
    TORCH_CHECK(r_ptrs.scalar_type() == torch::kInt64 && i_ptrs.scalar_type() == torch::kInt64 &&
                o_ptrs.scalar_type() == torch::kInt64 && axes.scalar_type() == torch::kInt32,
                "replay pointer tables must be int64 and axes int32");
    TORCH_CHECK(r_ptrs.dim() == 1 && i_ptrs.sizes() == r_ptrs.sizes() &&
                o_ptrs.sizes() == r_ptrs.sizes() && axes.sizes() == r_ptrs.sizes(),
                "replay tables must have matching rank-one shapes");
    TORCH_CHECK(layer_index > 0 && layer_index < r_ptrs.numel(),
                "layer_index must select a non-initial replay layer");
    TORCH_CHECK(r_ptrs.device() == r.device() && i_ptrs.device() == r.device() &&
                o_ptrs.device() == r.device() && axes.device() == r.device(),
                "replay tables must be on the activation device");
    if (seed.numel() != 0) {
        TORCH_CHECK(seed.is_cuda() && seed.device() == r.device() &&
                    seed.scalar_type() == r.scalar_type(), "seed dtype/device mismatch");
        TORCH_CHECK(seed.dim() == 4 && seed.size(0) == r.size(0) &&
                    seed.size(1) == r.size(2) && seed.size(2) == 3 && seed.size(3) == 3,
                    "seed must be [B,H,3,3]");
        TORCH_CHECK(seed_valid.is_cuda() && seed_valid.device() == r.device() &&
                    seed_valid.scalar_type() == torch::kBool && seed_valid.numel() == r.size(0),
                    "seed_valid must be bool [B]");
    } else {
        TORCH_CHECK(seed_valid.numel() == 0, "seed_valid must be empty without seed");
    }
}

static std::vector<torch::Tensor> depth_replay_launch(
    torch::Tensor grad_carry, torch::Tensor grad_p,
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor w,
    torch::Tensor r_ptrs, torch::Tensor i_ptrs, torch::Tensor o_ptrs,
    torch::Tensor axes, torch::Tensor seed, torch::Tensor seed_valid,
    double alpha, int64_t axis, int64_t layer_index, bool gated) {
    depth_replay_check(
        grad_carry, r, i, o, r_ptrs, i_ptrs, o_ptrs,
        axes, seed, seed_valid, layer_index);
    if (gated) {
        TORCH_CHECK(grad_p.is_cuda() && grad_p.device() == r.device() &&
                    grad_p.scalar_type() == r.scalar_type() && grad_p.sizes() == r.sizes(),
                    "grad_p must match R/I/O");
        TORCH_CHECK(w.is_cuda() && w.device() == r.device() &&
                    w.scalar_type() == r.scalar_type() && w.numel() == 9,
                    "w must contain nine activation-dtype values");
    }
    c10::cuda::CUDAGuard guard(r.device());
    auto gc = grad_carry.contiguous();
    auto gp = gated ? grad_p.contiguous() : torch::empty({0}, r.options());
    auto rc = r.contiguous(), ic = i.contiguous(), oc = o.contiguous();
    auto wc = gated ? w.contiguous() : torch::empty({0}, r.options());
    auto rp = r_ptrs.contiguous(), ip = i_ptrs.contiguous(), op = o_ptrs.contiguous();
    auto ax = axes.contiguous(), sc = seed.contiguous(), sv = seed_valid.contiguous();
    auto gr = torch::empty_like(rc), gi = torch::empty_like(ic), go = torch::empty_like(oc);
    auto gprev = torch::empty_like(gc);
    const int64_t B = rc.size(0), T = rc.size(1), H = rc.size(2), n = rc.numel();
    const int threads = GATE_MIX_THREADS;
    const int64_t blocks = (n + threads - 1) / threads;
    auto partial = gated
        ? torch::empty({9, blocks}, rc.options().dtype(torch::kFloat32))
        : torch::empty({0}, rc.options().dtype(torch::kFloat32));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, rc.scalar_type(),
        "depth_replay_backward_cuda", ([&] {
            auto rpp = reinterpret_cast<const scalar_t* const*>(rp.data_ptr<int64_t>());
            auto ipp = reinterpret_cast<const scalar_t* const*>(ip.data_ptr<int64_t>());
            auto opp = reinterpret_cast<const scalar_t* const*>(op.data_ptr<int64_t>());
            if (gated) {
                depth_replay_backward_kernel<scalar_t, true><<<
                    blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    gc.data_ptr<scalar_t>(), gp.data_ptr<scalar_t>(), rc.data_ptr<scalar_t>(),
                    ic.data_ptr<scalar_t>(), oc.data_ptr<scalar_t>(), wc.data_ptr<scalar_t>(),
                    rpp, ipp, opp, ax.data_ptr<int32_t>(),
                    sc.numel() ? sc.data_ptr<scalar_t>() : nullptr,
                    sv.numel() ? sv.data_ptr<bool>() : nullptr,
                    gr.data_ptr<scalar_t>(), gi.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
                    gprev.data_ptr<scalar_t>(), partial.data_ptr<float>(), (float)alpha,
                    (int)axis, (int)layer_index, B, T, H);
            } else {
                depth_replay_backward_kernel<scalar_t, false><<<
                    blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    gc.data_ptr<scalar_t>(), nullptr, rc.data_ptr<scalar_t>(),
                    ic.data_ptr<scalar_t>(), oc.data_ptr<scalar_t>(), nullptr,
                    rpp, ipp, opp, ax.data_ptr<int32_t>(),
                    sc.numel() ? sc.data_ptr<scalar_t>() : nullptr,
                    sv.numel() ? sv.data_ptr<bool>() : nullptr,
                    gr.data_ptr<scalar_t>(), gi.data_ptr<scalar_t>(), go.data_ptr<scalar_t>(),
                    gprev.data_ptr<scalar_t>(), nullptr, (float)alpha,
                    (int)axis, (int)layer_index, B, T, H);
            }
        }));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!gated) return {gr, gi, go, gprev};
    auto gw32 = torch::empty({9}, rc.options().dtype(torch::kFloat32));
    const int reduce_threads = 256;
    const int reduce_warps = reduce_threads / 32;
    gate_mix_grad_w_reduce_kernel<<<
        9, reduce_threads, reduce_warps * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        partial.data_ptr<float>(), gw32.data_ptr<float>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gr, gi, go, gprev, gw32.to(wc.scalar_type())};
}

std::vector<torch::Tensor> depth_replay_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor r, torch::Tensor i, torch::Tensor o,
    torch::Tensor r_ptrs, torch::Tensor i_ptrs, torch::Tensor o_ptrs,
    torch::Tensor axes, torch::Tensor seed, torch::Tensor seed_valid,
    double alpha, int64_t axis, int64_t layer_index) {
    return depth_replay_launch(
        grad_carry, torch::Tensor(), r, i, o, torch::Tensor(),
        r_ptrs, i_ptrs, o_ptrs, axes, seed, seed_valid,
        alpha, axis, layer_index, false);
}

std::vector<torch::Tensor> depth_replay_gate_backward_cuda(
    torch::Tensor grad_carry, torch::Tensor grad_p,
    torch::Tensor r, torch::Tensor i, torch::Tensor o, torch::Tensor w,
    torch::Tensor r_ptrs, torch::Tensor i_ptrs, torch::Tensor o_ptrs,
    torch::Tensor axes, torch::Tensor seed, torch::Tensor seed_valid,
    double alpha, int64_t axis, int64_t layer_index) {
    return depth_replay_launch(
        grad_carry, grad_p, r, i, o, w,
        r_ptrs, i_ptrs, o_ptrs, axes, seed, seed_valid,
        alpha, axis, layer_index, true);
}
