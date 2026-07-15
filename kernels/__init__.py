"""kernels/ -- CUDA module sources + build cache for LoomFormer's fused
kernels (tria, phase_sin, pvpowlu, depth_attn, beta_space). See build.py.

Layout per kernel group:
  kernels/<name>/<name>_kernel.cuh   -- pure CUDA device code (__global__
                                         kernels + any shared __device__
                                         helpers), no torch includes.
  kernels/<name>/<name>_kernel.cu    -- `#include "<name>_kernel.cuh"` only;
                                         a standalone TU for direct
                                         `nvcc --ptx` inspection.
  kernels/<name>/<name>_launcher.cu  -- ATen-facing host wrappers (arg
                                         checks, dtype dispatch, kernel
                                         launch) + pybind.

tria/ has 7 built kernel-group subfolders (tria_init, tria_init_gate,
tria_step, tria_step_gate, gate_slot_mix, slot_attention_pool,
temporal_carry) sharing one module/.so, built from all 7 launcher files
and tria/bindings.cpp (the single PYBIND11_MODULE).
"""
