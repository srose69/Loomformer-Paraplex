# (c) 2026 Ivan K (srose69, SimpleRose). Licensed under AGPL-3.0-only.
# Viper-LLM: Volumetric Language Model with Triangle Cross-Scan State Modelling.
# See TRADEMARKS.md for project naming and origin policy.

from __future__ import annotations

import os
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline


_EXT_MOD = None


def _maybe_set_cuda_arch_list() -> None:
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    if not torch.cuda.is_available():
        return
    try:
        caps = sorted({torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
    except Exception:
        return
    if not caps:
        return
    archs = [f"{major}.{minor}" for major, minor in caps]
    archs[-1] = archs[-1] + "+PTX"
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(archs)


def _get_ext():
    global _EXT_MOD
    if _EXT_MOD is not None:
        return _EXT_MOD

    _maybe_set_cuda_arch_list()

    name = "atom_cuda_ext_v7"
    cpp_src = r"""
#include <torch/extension.h>
#include <vector>
torch::Tensor atom_step_cuda(
    torch::Tensor p,
    torch::Tensor g,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor gprev_blk,
    torch::Tensor vmax,
    torch::Tensor trust_blk,
    torch::Tensor pnode_blk,
    torch::Tensor out,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double wd,
    int64_t step_i,
    bool maximize,
    bool amsgrad,
    int64_t block_size);
torch::Tensor atom_step_cuda_mixed_grad(
    torch::Tensor p,
    torch::Tensor g,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor gprev_blk,
    torch::Tensor vmax,
    torch::Tensor trust_blk,
    torch::Tensor pnode_blk,
    torch::Tensor out,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double wd,
    int64_t step_i,
    bool maximize,
    bool amsgrad,
    int64_t block_size);
torch::Tensor atom_step_sparse_cuda(
    torch::Tensor p,
    torch::Tensor idx1d,
    torch::Tensor gvals,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor gprev_blk,
    torch::Tensor vmax,
    torch::Tensor trust_blk,
    torch::Tensor pnode_blk,
    torch::Tensor out,
    double lr,
    double beta1,
    double beta2,
    double eps,
    double wd,
    int64_t step_i,
    bool maximize,
    bool amsgrad,
    int64_t block_size);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("atom_step_cuda", &atom_step_cuda, "ATOM CUDA step");
  m.def("atom_step_cuda_mixed_grad", &atom_step_cuda_mixed_grad, "ATOM CUDA step mixed grad dtype");
  m.def("atom_step_sparse_cuda", &atom_step_sparse_cuda, "ATOM CUDA sparse step");
}
"""
    cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Fused 4-way block reduction (sum) along threadIdx.x.
// One __syncthreads per call; safe to chain without interleaved syncs.
// Requires blockDim.x <= 1024 and all threads in the block to reach here
// (i.e. no early return above -- threads past `valid` must contribute 0).
__device__ __forceinline__ void block_reduce4_sum(
    float& a, float& b, float& c, float& d) {
  __shared__ float sa[32], sb[32], sc[32], sd[32];
  int lane = threadIdx.x & 31;
  int wid  = threadIdx.x >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    a += __shfl_down_sync(0xFFFFFFFF, a, off);
    b += __shfl_down_sync(0xFFFFFFFF, b, off);
    c += __shfl_down_sync(0xFFFFFFFF, c, off);
    d += __shfl_down_sync(0xFFFFFFFF, d, off);
  }
  if (lane == 0) { sa[wid] = a; sb[wid] = b; sc[wid] = c; sd[wid] = d; }
  __syncthreads();
  int nwarps = (blockDim.x + 31) >> 5;
  a = (threadIdx.x < nwarps) ? sa[lane] : 0.0f;
  b = (threadIdx.x < nwarps) ? sb[lane] : 0.0f;
  c = (threadIdx.x < nwarps) ? sc[lane] : 0.0f;
  d = (threadIdx.x < nwarps) ? sd[lane] : 0.0f;
  if (wid == 0) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      a += __shfl_down_sync(0xFFFFFFFF, a, off);
      b += __shfl_down_sync(0xFFFFFFFF, b, off);
      c += __shfl_down_sync(0xFFFFFFFF, c, off);
      d += __shfl_down_sync(0xFFFFFFFF, d, off);
    }
  }
}

// Single-tensor ATOM step kernel.
//
// Layout contract:
//   blockDim.x == block_size; gridDim.x == nblk; each CUDA block owns
//   exactly one parameter-block (one slot in gprev_blk/trust_blk/pnode_blk).
//   Threads past n (last block tail) set valid=false but DO participate
//   in __syncthreads -- no early return, no UB on partial blocks.
template <typename T>
__global__ void atom_step_kernel(
    T* __restrict__ p,
    const T* __restrict__ g,
    T* __restrict__ m,
    T* __restrict__ v,
    T* __restrict__ gprev_blk,
    T* __restrict__ vmax,
    const float* __restrict__ trust_blk,
    float* __restrict__ pnode_blk,
    float lr,
    float beta1, float beta2, float eps, float wd,
    int64_t step_i,
    int maximize, int amsgrad,
    int64_t n, int64_t block_size, int64_t nblk,
    float* __restrict__ block_out) {
  int64_t i  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t bi = (int64_t)blockIdx.x;  // block_size == blockDim.x => bi == blockIdx.x
  bool valid = (i < n);

  // Load per-block scalars once into shared memory.
  __shared__ float s_trust, s_gp_prev, s_pnode;
  if (threadIdx.x == 0) {
    s_trust   = (bi < nblk) ? trust_blk[bi] : 1.25f;
    s_gp_prev = (bi < nblk) ? (float)gprev_blk[bi] : 0.0f;
    s_pnode   = (bi < nblk) ? pnode_blk[bi] : 0.65f;
  }
  __syncthreads();

  float gg = 0.0f, mm = 0.0f, vv = 0.0f, delta = 0.0f;
  int   commit = 0;

  if (valid) {
    gg = (float)g[i];
    if (maximize) gg = -gg;

    float mm_prev = (float)m[i];
    float vv_prev = (float)v[i];
    float vm_prev = (float)vmax[i];
    float p_old   = (float)p[i];

    float dm = (1.0f - beta1) * (gg - mm_prev);
    float dv = (1.0f - beta2) * (gg * gg - vv_prev);
    mm = mm_prev + dm;
    vv = vv_prev + dv;

    float bc1 = 1.0f - powf(beta1, (float)step_i);
    float bc2 = 1.0f - powf(beta2, (float)step_i);
    float v_use = vv;
    if (amsgrad) {
      v_use = fmaxf(vm_prev, vv);
      vmax[i] = (T)v_use;
    }
    float m_hat = mm / fmaxf(bc1, 1e-12f);
    float v_hat = v_use / fmaxf(bc2, 1e-12f);
    float base  = m_hat / (sqrtf(v_hat) + eps);

    delta = -lr * base * (1.0f + s_trust);
    float gpred = gg + 0.35f * (gg - s_gp_prev);
    float pred  = -(gpred * delta) - 0.5f * v_hat * delta * delta;
    commit = (pred > 0.0f && s_pnode >= 0.5f) ? 1 : 0;
    if (!commit) delta = 0.0f;
    if (wd != 0.0f) delta += -lr * wd * p_old;

    p[i] = (T)(p_old + delta);
    m[i] = (T)mm;
    v[i] = (T)vv;
  }

  // Per-parameter-block reductions: gsum for the gprev EMA, csum/cntsum for
  // commit_frac, proxysum for global stats. Inactive threads contribute 0.
  float gsum     = valid ? gg                  : 0.0f;
  float csum     = valid ? (float)commit       : 0.0f;
  float cntsum   = valid ? 1.0f                : 0.0f;
  float proxysum = valid ? -(gg * delta)       : 0.0f;
  block_reduce4_sum(gsum, csum, cntsum, proxysum);

  if (threadIdx.x == 0 && bi < nblk) {
    float inv_cnt = 1.0f / fmaxf(cntsum, 1.0f);
    gprev_blk[bi] = (T)(gsum * inv_cnt);
    float commit_frac_blk = csum * inv_cnt;  // NOW counts actual threads, not warps.
    float pnode_new = fmaf(0.95f, s_pnode,
                           0.05f * (0.5f + 0.5f * commit_frac_blk));
    pnode_new = fmaxf(0.01f, fminf(0.99f, pnode_new));
    pnode_blk[bi] = pnode_new;

    int idx_base = blockIdx.x * 3;
    block_out[idx_base + 0] = proxysum;
    block_out[idx_base + 1] = csum;
    block_out[idx_base + 2] = cntsum;
  }
}

template <typename PT, typename GT>
__global__ void atom_step_kernel_mixed_grad(
    PT* __restrict__ p,
    const GT* __restrict__ g,
    PT* __restrict__ m,
    PT* __restrict__ v,
    PT* __restrict__ gprev_blk,
    PT* __restrict__ vmax,
    const float* __restrict__ trust_blk,
    float* __restrict__ pnode_blk,
    float lr,
    float beta1, float beta2, float eps, float wd,
    int64_t step_i,
    int maximize, int amsgrad,
    int64_t n, int64_t block_size, int64_t nblk,
    float* __restrict__ block_out) {
  int64_t i  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t bi = (int64_t)blockIdx.x;
  bool valid = (i < n);
  __shared__ float s_trust, s_gp_prev, s_pnode;
  if (threadIdx.x == 0) {
    s_trust   = (bi < nblk) ? trust_blk[bi] : 1.25f;
    s_gp_prev = (bi < nblk) ? (float)gprev_blk[bi] : 0.0f;
    s_pnode   = (bi < nblk) ? pnode_blk[bi] : 0.65f;
  }
  __syncthreads();
  float gg = 0.0f, mm = 0.0f, vv = 0.0f, delta = 0.0f;
  int commit = 0;
  if (valid) {
    gg = (float)g[i];
    if (maximize) gg = -gg;
    float mm_prev = (float)m[i];
    float vv_prev = (float)v[i];
    float vm_prev = (float)vmax[i];
    float p_old   = (float)p[i];
    float dm = (1.0f - beta1) * (gg - mm_prev);
    float dv = (1.0f - beta2) * (gg * gg - vv_prev);
    mm = mm_prev + dm;
    vv = vv_prev + dv;
    float bc1 = 1.0f - powf(beta1, (float)step_i);
    float bc2 = 1.0f - powf(beta2, (float)step_i);
    float v_use = vv;
    if (amsgrad) { v_use = fmaxf(vm_prev, vv); vmax[i] = (PT)v_use; }
    float m_hat = mm / fmaxf(bc1, 1e-12f);
    float v_hat = v_use / fmaxf(bc2, 1e-12f);
    float base  = m_hat / (sqrtf(v_hat) + eps);
    delta = -lr * base * (1.0f + s_trust);
    float gpred = gg + 0.35f * (gg - s_gp_prev);
    float pred  = -(gpred * delta) - 0.5f * v_hat * delta * delta;
    commit = (pred > 0.0f && s_pnode >= 0.5f) ? 1 : 0;
    if (!commit) delta = 0.0f;
    if (wd != 0.0f) delta += -lr * wd * p_old;
    p[i] = (PT)(p_old + delta);
    m[i] = (PT)mm;
    v[i] = (PT)vv;
  }
  float gsum = valid ? gg : 0.0f, csum = valid ? (float)commit : 0.0f, cntsum = valid ? 1.0f : 0.0f, proxysum = valid ? -(gg * delta) : 0.0f;
  block_reduce4_sum(gsum, csum, cntsum, proxysum);
  if (threadIdx.x == 0 && bi < nblk) {
    float inv_cnt = 1.0f / fmaxf(cntsum, 1.0f);
    gprev_blk[bi] = (PT)(gsum * inv_cnt);
    float commit_frac_blk = csum * inv_cnt;
    float pnode_new = fmaf(0.95f, s_pnode, 0.05f * (0.5f + 0.5f * commit_frac_blk));
    pnode_new = fmaxf(0.01f, fminf(0.99f, pnode_new));
    pnode_blk[bi] = pnode_new;
    int idx_base = blockIdx.x * 3;
    block_out[idx_base + 0] = proxysum;
    block_out[idx_base + 1] = csum;
    block_out[idx_base + 2] = cntsum;
  }
}

// Sum per-block [proxy, commits, count] triples into 3 global scalars.
__global__ void sum_reduce_kernel(
    const float* __restrict__ in, float* __restrict__ out, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int out_idx = (int)(i % 3);
    atomicAdd(out + out_idx, in[i]);
  }
}

torch::Tensor atom_step_cuda(
    torch::Tensor p,
    torch::Tensor g,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor gprev_blk,
    torch::Tensor vmax,
    torch::Tensor trust_blk,
    torch::Tensor pnode_blk,
    torch::Tensor out,
    double lr, double beta1, double beta2, double eps, double wd,
    int64_t step_i, bool maximize, bool amsgrad, int64_t block_size) {
  TORCH_CHECK(p.is_cuda() && g.is_cuda(), "p/g must be CUDA");
  TORCH_CHECK(m.is_cuda() && v.is_cuda() && gprev_blk.is_cuda() && vmax.is_cuda(),
              "state must be CUDA");
  TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous()
              && v.is_contiguous() && gprev_blk.is_contiguous() && vmax.is_contiguous(),
              "p/g/state must be contiguous");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.numel() == 3,
              "out must be fp32 CUDA[3]");
  TORCH_CHECK(trust_blk.is_cuda() && trust_blk.is_contiguous()
              && pnode_blk.is_cuda() && pnode_blk.is_contiguous(),
              "trust_blk/pnode_blk must be contiguous CUDA");
  TORCH_CHECK(block_size > 0 && block_size <= 1024 && (block_size % 32 == 0),
              "block_size must be in (0, 1024] and a multiple of 32");
  const at::cuda::CUDAGuard device_guard(p.device());

  int64_t n = p.numel();
  int64_t nblk = pnode_blk.numel();
  if (n == 0) return out;

  const int threads = (int)block_size;
  const int blocks  = (int)((n + threads - 1) / threads);
  TORCH_CHECK((int64_t)blocks == nblk,
              "grid blocks must equal nblk: blocks=", blocks, " nblk=", nblk);

  auto block_out = torch::zeros({blocks, 3},
                                torch::dtype(torch::kFloat32).device(p.device()));
  auto stream = at::cuda::getCurrentCUDAStream(p.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, p.scalar_type(),
    "atom_step_cuda", ([&] {
      atom_step_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
          p.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(),
          m.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
          gprev_blk.data_ptr<scalar_t>(), vmax.data_ptr<scalar_t>(),
          trust_blk.data_ptr<float>(), pnode_blk.data_ptr<float>(),
          (float)lr, (float)beta1, (float)beta2, (float)eps, (float)wd,
          (int64_t)step_i, maximize ? 1 : 0, amsgrad ? 1 : 0,
          n, block_size, nblk,
          block_out.data_ptr<float>());
      int64_t total_block_vals = blocks * 3;
      int sum_threads = 256;
      int sum_blocks = (int)((total_block_vals + sum_threads - 1) / sum_threads);
      sum_reduce_kernel<<<sum_blocks, sum_threads, 0, stream.stream()>>>(
          block_out.data_ptr<float>(), out.data_ptr<float>(), total_block_vals);
    }));
  return out;
}

torch::Tensor atom_step_cuda_mixed_grad(
    torch::Tensor p,
    torch::Tensor g,
    torch::Tensor m,
    torch::Tensor v,
    torch::Tensor gprev_blk,
    torch::Tensor vmax,
    torch::Tensor trust_blk,
    torch::Tensor pnode_blk,
    torch::Tensor out,
    double lr, double beta1, double beta2, double eps, double wd,
    int64_t step_i, bool maximize, bool amsgrad, int64_t block_size) {
  TORCH_CHECK(p.is_cuda() && g.is_cuda(), "p/g must be CUDA");
  TORCH_CHECK(p.scalar_type() == torch::kFloat32, "mixed grad path requires fp32 params");
  TORCH_CHECK(g.scalar_type() == torch::kHalf || g.scalar_type() == torch::kBFloat16,
              "mixed grad path requires fp16/bf16 grads");
  TORCH_CHECK(p.is_contiguous() && g.is_contiguous() && m.is_contiguous()
              && v.is_contiguous() && gprev_blk.is_contiguous() && vmax.is_contiguous(),
              "p/g/state must be contiguous");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.numel() == 3,
              "out must be fp32 CUDA[3]");
  int64_t n = p.numel();
  int64_t nblk = pnode_blk.numel();
  if (n == 0) return out;
  const at::cuda::CUDAGuard device_guard(p.device());
  const int threads = (int)block_size;
  const int blocks  = (int)((n + threads - 1) / threads);
  TORCH_CHECK((int64_t)blocks == nblk, "grid blocks must equal nblk");
  auto block_out = torch::zeros({blocks, 3}, torch::dtype(torch::kFloat32).device(p.device()));
  auto stream = at::cuda::getCurrentCUDAStream(p.get_device());
  if (g.scalar_type() == torch::kHalf) {
    atom_step_kernel_mixed_grad<float, at::Half><<<blocks, threads, 0, stream.stream()>>>(
      p.data_ptr<float>(), g.data_ptr<at::Half>(), m.data_ptr<float>(), v.data_ptr<float>(),
      gprev_blk.data_ptr<float>(), vmax.data_ptr<float>(),
      trust_blk.data_ptr<float>(), pnode_blk.data_ptr<float>(),
      (float)lr, (float)beta1, (float)beta2, (float)eps, (float)wd,
      (int64_t)step_i, maximize ? 1 : 0, amsgrad ? 1 : 0,
      n, block_size, nblk, block_out.data_ptr<float>());
  } else {
    atom_step_kernel_mixed_grad<float, at::BFloat16><<<blocks, threads, 0, stream.stream()>>>(
      p.data_ptr<float>(), g.data_ptr<at::BFloat16>(), m.data_ptr<float>(), v.data_ptr<float>(),
      gprev_blk.data_ptr<float>(), vmax.data_ptr<float>(),
      trust_blk.data_ptr<float>(), pnode_blk.data_ptr<float>(),
      (float)lr, (float)beta1, (float)beta2, (float)eps, (float)wd,
      (int64_t)step_i, maximize ? 1 : 0, amsgrad ? 1 : 0,
      n, block_size, nblk, block_out.data_ptr<float>());
  }
  int64_t total_block_vals = blocks * 3;
  int sum_threads = 256;
  int sum_blocks = (int)((total_block_vals + sum_threads - 1) / sum_threads);
  sum_reduce_kernel<<<sum_blocks, sum_threads, 0, stream.stream()>>>(
      block_out.data_ptr<float>(), out.data_ptr<float>(), total_block_vals);
  return out;
}

// Sparse step: COO gradient indices. Per-block reductions are NOT updated
// here (sparse updates touch arbitrary indices, so scattered pnode/gprev
// updates would need atomics per block with unclear correctness). Kept
// simple -- sparse is only used by embedding sparse=True paths, rare here.
template <typename T>
__global__ void atom_step_sparse_kernel(
    T* __restrict__ p,
    const int64_t* __restrict__ idx1d,
    const T* __restrict__ gv,
    T* __restrict__ m, T* __restrict__ v,
    T* __restrict__ gprev_blk,
    T* __restrict__ vmax,
    const float* __restrict__ trust_blk,
    const float* __restrict__ pnode_blk,
    float lr, float beta1, float beta2, float eps, float wd,
    int64_t step_i, int maximize, int amsgrad,
    int64_t nnz, int64_t block_size, int64_t nblk,
    float* __restrict__ block_out) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  bool valid = (i < nnz);

  float gg = 0.0f, delta = 0.0f;
  int   commit = 0;

  if (valid) {
    int64_t j = idx1d[i];
    gg = (float)gv[i];
    if (maximize) gg = -gg;
    float mm_prev = (float)m[j];
    float vv_prev = (float)v[j];
    int64_t bi = j / block_size;
    float gp_prev = (bi < nblk) ? (float)gprev_blk[bi] : 0.0f;
    float vm_prev = (float)vmax[j];

    float dm = (1.0f - beta1) * (gg - mm_prev);
    float dv = (1.0f - beta2) * (gg * gg - vv_prev);
    float mm = mm_prev + dm;
    float vv = vv_prev + dv;

    float bc1 = 1.0f - powf(beta1, (float)step_i);
    float bc2 = 1.0f - powf(beta2, (float)step_i);
    float v_use = vv;
    if (amsgrad) {
      v_use = fmaxf(vm_prev, vv);
      vmax[j] = (T)v_use;
    }
    float m_hat = mm / fmaxf(bc1, 1e-12f);
    float v_hat = v_use / fmaxf(bc2, 1e-12f);
    float base  = m_hat / (sqrtf(v_hat) + eps);
    float trust = (bi < nblk) ? trust_blk[bi] : 1.25f;
    delta = -lr * base * (1.0f + trust);
    float gpred = gg + 0.35f * (gg - gp_prev);
    float pred  = -(gpred * delta) - 0.5f * v_hat * delta * delta;
    float pnode = (bi < nblk) ? pnode_blk[bi] : 0.65f;
    commit = (pred > 0.0f && pnode >= 0.5f) ? 1 : 0;
    if (!commit) delta = 0.0f;
    if (wd != 0.0f) delta += -lr * wd * (float)p[j];

    p[j] = (T)((float)p[j] + delta);
    m[j] = (T)mm;
    v[j] = (T)vv;
  }

  // Global stats only -- warp-level shuffle, atomic into block_out triple.
  float proxy = valid ? -(gg * delta) : 0.0f;
  float cm    = valid ? (float)commit : 0.0f;
  float cnt   = valid ? 1.0f          : 0.0f;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    proxy += __shfl_down_sync(0xFFFFFFFF, proxy, off);
    cm    += __shfl_down_sync(0xFFFFFFFF, cm,    off);
    cnt   += __shfl_down_sync(0xFFFFFFFF, cnt,   off);
  }
  if ((threadIdx.x & 31) == 0) {
    int idx_base = blockIdx.x * 3;
    atomicAdd(block_out + idx_base + 0, proxy);
    atomicAdd(block_out + idx_base + 1, cm);
    atomicAdd(block_out + idx_base + 2, cnt);
  }
}

torch::Tensor atom_step_sparse_cuda(
    torch::Tensor p,
    torch::Tensor idx1d,
    torch::Tensor gvals,
    torch::Tensor m, torch::Tensor v,
    torch::Tensor gprev_blk,
    torch::Tensor vmax,
    torch::Tensor trust_blk,
    torch::Tensor pnode_blk,
    torch::Tensor out,
    double lr, double beta1, double beta2, double eps, double wd,
    int64_t step_i, bool maximize, bool amsgrad, int64_t block_size) {
  TORCH_CHECK(p.is_cuda() && idx1d.is_cuda() && gvals.is_cuda(),
              "p/idx/gvals must be CUDA");
  TORCH_CHECK(m.is_cuda() && v.is_cuda() && gprev_blk.is_cuda() && vmax.is_cuda(),
              "state must be CUDA");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == torch::kFloat32 && out.numel() == 3,
              "out must be fp32 CUDA[3]");
  TORCH_CHECK(trust_blk.is_cuda() && trust_blk.is_contiguous()
              && pnode_blk.is_cuda() && pnode_blk.is_contiguous(),
              "trust_blk/pnode_blk must be contiguous CUDA");

  int64_t nnz = idx1d.numel();
  int64_t nblk = pnode_blk.numel();
  if (nnz == 0) return out;
  const at::cuda::CUDAGuard device_guard(p.device());
  const int threads = 256;
  const int blocks  = (int)((nnz + threads - 1) / threads);

  auto block_out = torch::zeros({blocks, 3},
                                torch::dtype(torch::kFloat32).device(p.device()));
  auto stream = at::cuda::getCurrentCUDAStream(p.get_device());
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, p.scalar_type(),
    "atom_step_sparse_cuda", ([&] {
      atom_step_sparse_kernel<scalar_t><<<blocks, threads, 0, stream.stream()>>>(
          p.data_ptr<scalar_t>(),
          idx1d.data_ptr<int64_t>(),
          gvals.data_ptr<scalar_t>(),
          m.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
          gprev_blk.data_ptr<scalar_t>(), vmax.data_ptr<scalar_t>(),
          trust_blk.data_ptr<float>(), pnode_blk.data_ptr<float>(),
          (float)lr, (float)beta1, (float)beta2, (float)eps, (float)wd,
          (int64_t)step_i, maximize ? 1 : 0, amsgrad ? 1 : 0,
          nnz, block_size, nblk,
          block_out.data_ptr<float>());
      int64_t total_block_vals = blocks * 3;
      int sum_threads = 256;
      int sum_blocks = (int)((total_block_vals + sum_threads - 1) / sum_threads);
      sum_reduce_kernel<<<sum_blocks, sum_threads, 0, stream.stream()>>>(
          block_out.data_ptr<float>(), out.data_ptr<float>(), total_block_vals);
    }));
  return out;
}
"""
    _EXT_MOD = load_inline(
        name=name,
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        extra_cflags=["-O3"],
        with_cuda=True,
        verbose=os.getenv("ATOM_BUILD_VERBOSE", "0") == "1",
    )
    return _EXT_MOD


class ATOM(torch.optim.Optimizer):
    """Drop-in AdamW API, ATOM backend on raw CUDA C++ kernels."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        maximize: bool = False,
        amsgrad: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        **kwargs,
    ):
        if kwargs:
            pass
        defaults = dict(
            lr=float(lr),
            betas=tuple(betas),
            eps=float(eps),
            weight_decay=float(weight_decay),
            maximize=bool(maximize),
            amsgrad=bool(amsgrad),
            foreach=foreach,
            capturable=bool(capturable),
            differentiable=bool(differentiable),
            fused=fused,
        )
        super().__init__(params, defaults)
        # Meta-scale controls optimizer complexity (higher = more granular analysis)
        self._meta_scale = int(os.getenv("ATOM_META_SCALE", "1"))
        self._meta_scale = max(1, min(8, self._meta_scale))  # Clamp to 1..8 (int levels)
        self._step = 0
        self._trust_min = 0.90
        self._trust_max = 1.90
        self._bootstrap_steps = 20 * self._meta_scale
        self._bootstrap_gain = 2.4
        self._proxy_ema = 0.0
        self._pnode = 0.65
        self.last_stats: Dict[str, float] = {}
        self._stats_buf: Dict[Tuple[int, int], torch.Tensor] = {}
        self._ext_grads: Dict[torch.nn.Parameter, torch.Tensor] = {}
        # Inverse scale: higher meta_scale = smaller blocks = more detail
        base_block = 256
        self._gprev_block = max(32, min(512, base_block // self._meta_scale))

    @torch.no_grad()
    def _get_stats_buf(self, device: torch.device) -> torch.Tensor:
        key = (device.index if device.index is not None else -1, int(device.type == "cuda"))
        t = self._stats_buf.get(key)
        if t is None or t.device != device:
            t = torch.zeros(3, device=device, dtype=torch.float32)
            self._stats_buf[key] = t
        else:
            t.zero_()
        return t

    @torch.no_grad()
    def _init_state(self, p: torch.Tensor) -> None:
        st = self.state[p]
        st_dtype = p.dtype if p.is_cuda else torch.float32
        nblk = (p.numel() + self._gprev_block - 1) // self._gprev_block

        def _reset_like_param(key: str, dtype: torch.dtype) -> None:
            t = st.get(key)
            if t is None or tuple(t.shape) != tuple(p.shape) or t.device != p.device or t.dtype != dtype:
                st[key] = torch.zeros_like(p, dtype=dtype, memory_format=torch.preserve_format)

        def _reset_like_blocks(key: str, fill: float, dtype: torch.dtype) -> None:
            t = st.get(key)
            if t is None or t.numel() != nblk or t.device != p.device or t.dtype != dtype:
                st[key] = torch.full((nblk,), fill, device=p.device, dtype=dtype)

        # Always ensure required keys exist (handles partial init from AdamW checkpoints)
        if "step" not in st:
            st["step"] = torch.zeros((), dtype=torch.int64, device=p.device)
        _reset_like_param("m", st_dtype)
        _reset_like_param("v", st_dtype)
        _reset_like_blocks("gprev_blk", 0.0, st_dtype)
        _reset_like_blocks("pnode_blk", self._pnode, torch.float32)
        _reset_like_blocks("trust_blk", 1.25, torch.float32)
        if "vmax" in st:
            _reset_like_param("vmax", st_dtype)

    @torch.no_grad()
    def _ensure_vmax(self, p: torch.Tensor, st: Dict) -> None:
        if "vmax" in st:
            return
        st_dtype = st["v"].dtype if "v" in st else (p.dtype if p.is_cuda else torch.float32)
        st["vmax"] = torch.zeros_like(p, dtype=st_dtype, memory_format=torch.preserve_format)

    @torch.no_grad()
    def _ensure_pnode_blk(self, p: torch.Tensor, st: Dict) -> None:
        """Ensure pnode_blk exists for legacy state dict compatibility."""
        nblk = (p.numel() + self._gprev_block - 1) // self._gprev_block
        t = st.get("pnode_blk")
        if t is not None and t.numel() == nblk and t.device == p.device and t.dtype == torch.float32:
            return
        st["pnode_blk"] = torch.full((nblk,), self._pnode, device=p.device, dtype=torch.float32)

    @torch.no_grad()
    def _ensure_trust_blk(self, p: torch.Tensor, st: Dict) -> None:
        """Ensure trust_blk exists for legacy state dict compatibility."""
        nblk = (p.numel() + self._gprev_block - 1) // self._gprev_block
        t = st.get("trust_blk")
        if t is not None and t.numel() == nblk and t.device == p.device and t.dtype == torch.float32:
            return
        st["trust_blk"] = torch.full((nblk,), 1.25, device=p.device, dtype=torch.float32)

    @torch.no_grad()
    def _refresh_trust_blk(self, st: Dict) -> None:
        """Map each block's pnode into its own trust value.

        This makes trust genuinely per-block adaptive: blocks with higher
        historical accept rates get higher trust on the next step, while
        blocks with lower accept rates become more conservative.
        """
        trust_blk = st["trust_blk"]
        pnode_blk = st["pnode_blk"]
        # Keep update fully in-place on the already-allocated trust buffer.
        # This avoids an explicit copy_ from a temporary tensor each step.
        trust_blk.zero_()
        trust_blk.add_(pnode_blk)
        trust_blk.mul_(self._trust_max - self._trust_min)
        trust_blk.add_(self._trust_min)
        if self._step <= self._bootstrap_steps:
            trust_blk.clamp_(min=self._bootstrap_gain)

    @staticmethod
    def _sparse_to_flat(p: torch.Tensor, gs: torch.Tensor):
        idx = gs.indices()  # [sparse_dim, nnz]
        vals = gs.values()  # [nnz, *dense_dims] or [nnz]
        sparse_dim = idx.shape[0]
        strides = p.stride()
        base = torch.zeros((idx.shape[1],), device=idx.device, dtype=idx.dtype)
        for d in range(sparse_dim):
            base = base + idx[d] * int(strides[d])
        if vals.dim() == 1:
            return base.contiguous(), vals.contiguous()
        tail_shape = vals.shape[1:]
        k = int(vals[0].numel())
        dev = idx.device
        off = torch.arange(k, device=dev, dtype=idx.dtype)
        tail_multi = []
        for sz in reversed(tail_shape):
            tail_multi.append(off % sz)
            off = off // sz
        tail_multi = list(reversed(tail_multi))
        tail_off = torch.zeros((k,), device=dev, dtype=idx.dtype)
        for j, ax in enumerate(tail_multi):
            tail_off = tail_off + ax * int(strides[sparse_dim + j])
        idx1d = base.unsqueeze(1) + tail_off.unsqueeze(0)
        return idx1d.reshape(-1).contiguous(), vals.reshape(-1).contiguous()

    @torch.no_grad()
    def ingest_grads(self, grad_accum) -> None:
        self._ext_grads.clear()
        for group in self.param_groups:
            for p in group["params"]:
                buf = grad_accum._buf.get(id(p))
                if buf is None:
                    continue
                if buf.device == p.device:
                    self._ext_grads[p] = buf
                else:
                    self._ext_grads[p] = buf.to(device=p.device)
                grad_accum._buf[id(p)] = None

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float):
        if max_norm is None or max_norm <= 0:
            return None
        total = None
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad if p.grad is not None else self._ext_grads.get(p)
                if g is None:
                    continue
                gsq = g.detach().float().pow(2).sum()
                total = gsq if total is None else (total + gsq)
        if total is None:
            return None
        norm = total.sqrt()
        coef = float(max_norm) / float(norm.item() + 1e-6)
        if coef < 1.0:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.mul_(coef)
                    else:
                        g = self._ext_grads.get(p)
                        if g is not None:
                            g.mul_(coef)
        return norm

    @torch.no_grad()
    def step(self, closure=None):
        self._step += 1
        ext = None
        total_proxy = 0.0
        total_commit = 0.0
        total_count = 0.0
        total_trust_weighted = 0.0
        total_trust_blocks = 0.0
        total_proxy_dev = None
        total_commit_dev = None
        total_count_dev = None
        total_trust_dev = None
        tensors = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            maximize = bool(group.get("maximize", False))
            amsgrad = bool(group.get("amsgrad", False))
            cuda_group_out = None
            for p in group["params"]:
                g = p.grad if p.grad is not None else self._ext_grads.get(p)
                if g is None:
                    continue
                self._init_state(p)
                st = self.state[p]
                # Avoid per-tensor device->host sync via st["step"].item().
                # One global optimizer step index is enough for bias correction.
                st["step"].fill_(self._step)
                step_i = self._step
                if amsgrad:
                    self._ensure_vmax(p, st)
                vmax_t = st["vmax"] if amsgrad else st["v"]

                if p.is_cuda:
                    if ext is None:
                        ext = _get_ext()
                    if cuda_group_out is None:
                        cuda_group_out = self._get_stats_buf(p.device)
                    self._ensure_pnode_blk(p, st)
                    self._ensure_trust_blk(p, st)
                    self._refresh_trust_blk(st)
                    # Defer host sync: accumulate trust_blk sum on-device, pull once at
                    # the end of step() instead of one .item() sync per parameter tensor.
                    trust_sum_t = st["trust_blk"].sum()
                    total_trust_dev = trust_sum_t if total_trust_dev is None else total_trust_dev + trust_sum_t
                    total_trust_blocks += float(st["trust_blk"].numel())
                    if g.is_sparse:
                        gs = g.coalesce()
                        idx1d, vals = self._sparse_to_flat(p, gs)
                        out = ext.atom_step_sparse_cuda(
                            p,
                            idx1d,
                            vals,
                            st["m"],
                            st["v"],
                            st["gprev_blk"],
                            vmax_t,
                            st["trust_blk"],
                            st["pnode_blk"],
                            cuda_group_out,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            wd,
                            step_i,
                            maximize,
                            amsgrad,
                            self._gprev_block,
                        )
                    else:
                        if p.dtype == torch.float32 and g.dtype in (torch.float16, torch.bfloat16):
                            out = ext.atom_step_cuda_mixed_grad(
                                p,
                                g,
                                st["m"],
                                st["v"],
                                st["gprev_blk"],
                                vmax_t,
                                st["trust_blk"],
                                st["pnode_blk"],
                                cuda_group_out,
                                lr,
                                beta1,
                                beta2,
                                eps,
                                wd,
                                step_i,
                                maximize,
                                amsgrad,
                                self._gprev_block,
                            )
                        else:
                            out = ext.atom_step_cuda(
                                p,
                                g,
                                st["m"],
                                st["v"],
                                st["gprev_blk"],
                                vmax_t,
                                st["trust_blk"],
                                st["pnode_blk"],
                                cuda_group_out,
                                lr,
                                beta1,
                                beta2,
                                eps,
                                wd,
                                step_i,
                                maximize,
                                amsgrad,
                                self._gprev_block,
                            )
                    # defer host sync to end of group
                else:
                    # CPU fallback with same formula.
                    self._ensure_pnode_blk(p, st)
                    self._ensure_trust_blk(p, st)
                    self._refresh_trust_blk(st)
                    total_trust_weighted += float(st["trust_blk"].sum().item())
                    total_trust_blocks += float(st["trust_blk"].numel())
                    if g.is_sparse:
                        gs = g.coalesce()
                        idx1d, vals = self._sparse_to_flat(p, gs)
                        vals = vals.float()
                        gg = torch.zeros_like(p, dtype=torch.float32).view(-1)
                        gg[idx1d] = (-vals if maximize else vals)
                        gg = gg.view_as(p)
                    else:
                        gg = -g.float() if maximize else g.float()
                    mm = st["m"]
                    vv = st["v"]
                    gp_blk = st["gprev_blk"]
                    pnode_blk = st["pnode_blk"]
                    trust_blk = st["trust_blk"]
                    vm = vmax_t
                    dm = (gg - mm) * (1.0 - beta1)
                    dv = (gg * gg - vv) * (1.0 - beta2)
                    mm.add_(dm)
                    vv.add_(dv)
                    bc1 = 1.0 - (beta1 ** step_i)
                    bc2 = 1.0 - (beta2 ** step_i)
                    if amsgrad:
                        torch.maximum(vm, vv, out=vm)
                        v_use = vm
                    else:
                        v_use = vv
                    base = (mm / max(bc1, 1e-12)) / ((v_use / max(bc2, 1e-12)).sqrt() + eps)
                    # Per-block trust
                    flat_n = gg.numel()
                    bi_cpu = torch.arange(flat_n, device=trust_blk.device) // self._gprev_block
                    trust_per_elem = trust_blk[bi_cpu.clamp_max(trust_blk.numel() - 1)]
                    delta = (-lr * base * (1.0 + trust_per_elem.view_as(gg))).view(-1)
                    flat_n = gg.numel()
                    gp = gp_blk.repeat_interleave(self._gprev_block)[:flat_n].view_as(gg)
                    gpred = gg + 0.35 * (gg - gp)
                    pred = -(gpred.view(-1) * delta) - 0.5 * (v_use.view(-1) / max(bc2, 1e-12)) * delta.pow(2)
                    # Per-block pnode for commit decision
                    bi = torch.arange(flat_n, device=pnode_blk.device) // self._gprev_block
                    pnode_per_elem = pnode_blk[bi.clamp_max(pnode_blk.numel() - 1)]
                    cm = ((pred > 0) & (pnode_per_elem >= 0.5)).to(delta.dtype)
                    delta.mul_(cm)
                    if wd != 0.0:
                        delta.add_(p.float().view(-1), alpha=-lr * wd)
                    d = delta.view_as(p)
                    if d.dtype == p.dtype:
                        p.add_(d)
                    else:
                        p.add_(d.to(dtype=p.dtype))
                    blk = gg.view(-1).split(self._gprev_block)
                    blk_mean = torch.stack([b.mean() for b in blk])
                    if blk_mean.dtype != gp_blk.dtype:
                        blk_mean = blk_mean.to(gp_blk.dtype)
                    gp_blk.copy_(blk_mean)
                    total_proxy += float((-(gg * d).sum()).item())
                    total_commit += float(cm.sum().item())
                    total_count += float(cm.numel())
                tensors += 1
            if cuda_group_out is not None:
                # Defer host sync: accumulate on device, pull scalars once at end.
                if total_proxy_dev is None:
                    total_proxy_dev = cuda_group_out[0].detach()
                    total_commit_dev = cuda_group_out[1].detach()
                    total_count_dev = cuda_group_out[2].detach()
                else:
                    total_proxy_dev.add_(cuda_group_out[0])
                    total_commit_dev.add_(cuda_group_out[1])
                    total_count_dev.add_(cuda_group_out[2])

        if total_proxy_dev is not None:
            total_proxy += float(total_proxy_dev.item())
            total_commit += float(total_commit_dev.item())
            total_count += float(total_count_dev.item())
        if total_trust_dev is not None:
            total_trust_weighted += float(total_trust_dev.item())

        if total_count > 0:
            proxy_mean = total_proxy / total_count
            commit_frac = total_commit / total_count
            trust_mean = (total_trust_weighted / total_trust_blocks) if total_trust_blocks > 0 else 0.0
            # Adaptive EMA: higher meta_scale = longer memory (slower decay)
            ema_decay = 1.0 - 0.02 / self._meta_scale  # scale=1: 0.98, scale=8: 0.9975
            self._proxy_ema = ema_decay * self._proxy_ema + (1 - ema_decay) * proxy_mean
            # Adaptive pnode: higher meta_scale = slower adaptation (more trust to history)
            pnode_decay = 1.0 - 0.05 / self._meta_scale  # scale=1: 0.95, scale=8: 0.99375
            adapt_rate = 0.05 / self._meta_scale
            self._pnode = float(min(0.99, max(0.01, pnode_decay * self._pnode + adapt_rate * (0.5 + 0.5 * commit_frac))))
            # Note: per-block pnode_blk is now updated independently inside CUDA kernels
            self.last_stats = {
                "params_tensors": float(tensors),
                "trust_mean": float(trust_mean),
                "commit_frac": float(commit_frac),
                "impact_proxy_mean": float(proxy_mean),
                "pnode": float(self._pnode),
            }
        else:
            self.last_stats = {"params_tensors": 0.0, "trust_mean": 0.0, "commit_frac": 0.0, "impact_proxy_mean": 0.0, "pnode": self._pnode}
        self._ext_grads.clear()
        return None

    def state_dict(self):
        sd = super().state_dict()
        sd["atom_meta"] = {
            "step": self._step,
            "proxy_ema": self._proxy_ema,
            "pnode": self._pnode,
            "last_stats": self.last_stats,
            "meta_scale": int(self._meta_scale),
        }
        return sd

    def load_state_dict(self, state_dict):
        meta = state_dict.pop("atom_meta", None)
        # Convert AdamW-style state dict to Atom format
        for key in list(state_dict.get("state", {}).keys()):
            st = state_dict["state"][key]
            if isinstance(st, dict):
                # Rename AdamW keys to Atom keys
                if "exp_avg" in st and "m" not in st:
                    st["m"] = st.pop("exp_avg")
                if "exp_avg_sq" in st and "v" not in st:
                    st["v"] = st.pop("exp_avg_sq")
                # Initialize Atom-specific fields if missing
                if "gprev_blk" not in st or "pnode_blk" not in st or "trust_blk" not in st:
                    # Need to get param reference to initialize blocks
                    # This will be handled lazily in step() via _init_state and _ensure_* methods
                    pass
        out = super().load_state_dict(state_dict)
        if meta is not None:
            self._step = int(meta.get("step", self._step))
            self._proxy_ema = float(meta.get("proxy_ema", self._proxy_ema))
            self._pnode = float(meta.get("pnode", self._pnode))
            self.last_stats = dict(meta.get("last_stats", self.last_stats))
            # Use meta_scale from checkpoint for consistency (block sizes must match)
            loaded_scale = meta.get("meta_scale")
            if loaded_scale is not None:
                self._meta_scale = int(loaded_scale)
                # Re-derive dependent values
                self._bootstrap_steps = 20 * self._meta_scale
                base_block = 256
                self._gprev_block = max(32, min(512, base_block // self._meta_scale))
        return out

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        self._ext_grads.clear()


__all__ = ["ATOM"]
