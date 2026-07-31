from __future__ import annotations

import contextlib
import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from loomformer_runtime.distributed import ddp_print
from loomformer_runtime.layouts import PackedAttentionLayout, PackedChunkLayout
from loomformer_runtime.profiling import profile_region as _profile_region
from .. import state as S

_sdpa_bf16_efficient_cache = {}
_flash_varlen_func = None
_flash_varlen_import_tried = False
_flash_varlen_import_error = None
_flash_deterministic_kw_supported = None
_flash_backend_cache = {}
_flash_value_fusion_cache = {}
_flash_probe_errors = {}
_te_backend_cache = {}
_te_value_fusion_cache = {}
_te_probe_errors = {}
_te_dpa_modules = {}
_attention_backend_reported = set()

def _kv_to_q_heads(x: torch.Tensor) -> torch.Tensor:
    return x.repeat_interleave(S.GQA_GROUP_SIZE, dim=2)

def _bf16_efficient_sdpa_supported(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return True
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    major, minor = torch.cuda.get_device_capability(idx)
    key = (idx, major, minor)
    cached = _sdpa_bf16_efficient_cache.get(key)
    if cached is not None:
        return cached
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with torch.cuda.device(idx):
            q = torch.randn(1, 1, 16, S.HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn(1, 1, 16, S.HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
            v = torch.randn(1, 1, 16, 2 * S.HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
            torch.cuda.synchronize(device)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                    with torch.autocast(device_type="cuda", enabled=False):
                        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
                        y.sum().backward()
            torch.cuda.synchronize(device)
        ok = True
    except Exception:
        ok = False
    _sdpa_bf16_efficient_cache[key] = ok
    return ok


def _get_flash_varlen_func():
    global _flash_varlen_func, _flash_varlen_import_tried, _flash_varlen_import_error
    if _flash_varlen_import_tried:
        return _flash_varlen_func
    _flash_varlen_import_tried = True
    try:
        from flash_attn import flash_attn_varlen_func
    except (ImportError, OSError) as error:
        _flash_varlen_func = None
        _flash_varlen_import_error = f"{type(error).__name__}: {error}"
    else:
        _flash_varlen_func = flash_attn_varlen_func
        _flash_varlen_import_error = None
    return _flash_varlen_func


def _call_flash_varlen(func, q, k, v, *args, **kwargs):
    """Call both current and older FA2 varlen bindings without hiding errors."""
    global _flash_deterministic_kw_supported
    if _flash_deterministic_kw_supported is False:
        kwargs.pop("deterministic", None)
        return func(q, k, v, *args, **kwargs)
    try:
        out = func(q, k, v, *args, **kwargs)
    except TypeError as error:
        if (
            _flash_deterministic_kw_supported is None
            and "deterministic" in kwargs
            and "deterministic" in str(error)
        ):
            kwargs.pop("deterministic")
            out = func(q, k, v, *args, **kwargs)
            _flash_deterministic_kw_supported = False
            return out
        raise
    _flash_deterministic_kw_supported = True
    return out


def _flash_varlen_eligible(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda" or tensor.dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
    major, _minor = torch.cuda.get_device_capability(idx)
    key = (idx, tensor.dtype, S.HEAD_DIM)
    probed = _flash_backend_cache.get(key)
    if probed is not None:
        return probed
    return major >= 8 and S.HEAD_DIM <= 256 and _get_flash_varlen_func() is not None


def _attention_compute_dtype(device: torch.device, parameter_dtype: torch.dtype) -> torch.dtype:
    if device.type != "cuda":
        return parameter_dtype
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:  # PyTorch before the device-typed autocast query
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return parameter_dtype


def _probe_flash_value_fusion(device: torch.device, dtype: torch.dtype) -> bool:
    """Verify fused value=[K;V] forward *and backward once per device/shape."""
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, dtype, S.HEAD_DIM)
    cached = _flash_value_fusion_cache.get(key)
    if cached is not None:
        return cached
    func = _get_flash_varlen_func()
    major, _minor = torch.cuda.get_device_capability(idx)
    if func is None or major < 8 or S.HEAD_DIM > 256:
        _flash_backend_cache[key] = False
        _flash_value_fusion_cache[key] = False
        return False
    try:
        total = 4
        q = torch.randn(
            total, S.N_Q_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(
            total, S.N_KV_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        v_base = torch.randn(
            total, S.N_KV_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        cu = torch.tensor([0, total], dtype=torch.int32, device=device)
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            base_out = _call_flash_varlen(
                func,
                q, k, v_base, cu, cu, total, total,
                dropout_p=0.0, causal=True,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
            base_out.float().sum().backward()
            torch.cuda.synchronize(device)
        _flash_backend_cache[key] = True
        del q, k, v_base, base_out
    except Exception as error:
        _flash_backend_cache[key] = False
        _flash_value_fusion_cache[key] = False
        _flash_probe_errors[key] = f"{type(error).__name__}: {error}"
        return False
    if 2 * S.HEAD_DIM > 256:
        _flash_value_fusion_cache[key] = False
        return False
    try:
        q = torch.randn(
            total, S.N_Q_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        fused_kv = torch.randn(
            total, S.N_KV_HEADS, 2 * S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = fused_kv[..., :S.HEAD_DIM]
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            out = _call_flash_varlen(
                func,
                q, k, fused_kv, cu, cu, total, total,
                dropout_p=0.0, causal=True,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            )
            out.float().sum().backward()
            torch.cuda.synchronize(device)
        ok = out.shape[-1] == 2 * S.HEAD_DIM
        del q, k, fused_kv, cu, out
    except Exception as error:
        ok = False
        _flash_probe_errors[key] = (
            "fused-value probe only (base varlen works): "
            f"{type(error).__name__}: {error}")
    _flash_value_fusion_cache[key] = ok
    return ok


def _flash_attention_contexts(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Varlen causal attention in THD layout, returning (attention@K, attention@V)."""
    func = _get_flash_varlen_func()
    if func is None:
        raise RuntimeError(
            "FlashAttention was selected but flash_attn.flash_attn_varlen_func is unavailable")
    common = dict(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        dropout_p=0.0,
        causal=True,
        deterministic=torch.are_deterministic_algorithms_enabled(),
    )
    with torch.autocast(device_type="cuda", enabled=False):
        k_ctx = _call_flash_varlen(func, q, k, k, **common)
        value_ctx = _call_flash_varlen(func, q, k, v, **common)
    return k_ctx, value_ctx


def _get_te_dpa(device: torch.device, value_dim: int):
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, S.N_Q_HEADS, S.N_KV_HEADS, S.HEAD_DIM, int(value_dim))
    module = _te_dpa_modules.get(key)
    if module is not None:
        return module
    try:
        from transformer_engine.pytorch import DotProductAttention
        with torch.cuda.device(idx):
            module = DotProductAttention(
                num_attention_heads=S.N_Q_HEADS,
                kv_channels=(S.HEAD_DIM, int(value_dim)),
                num_gqa_groups=S.N_KV_HEADS,
                attention_dropout=0.0,
                qkv_format="thd",
                attn_mask_type="padding_causal_bottom_right",
            )
    except (ImportError, OSError, RuntimeError):
        return None
    _te_dpa_modules[key] = module
    return module


def _te_varlen_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> torch.Tensor:
    module = _get_te_dpa(q.device, v.shape[-1])
    if module is None:
        raise RuntimeError("Transformer Engine DotProductAttention is unavailable")
    return module(
        q, k, v,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_kv=int(max_seqlen_k),
        attn_mask_type="padding_causal_bottom_right",
    )


def _probe_te_value_fusion(device: torch.device, dtype: torch.dtype) -> bool:
    if device.type != "cuda" or dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, dtype, S.HEAD_DIM)
    cached = _te_value_fusion_cache.get(key)
    if cached is not None:
        return cached
    major, _minor = torch.cuda.get_device_capability(idx)
    if major < 8:
        _te_backend_cache[key] = False
        _te_value_fusion_cache[key] = False
        return False
    total = 4
    cu = torch.tensor([0, total], dtype=torch.int32, device=device)
    try:
        q = torch.randn(
            total, S.N_Q_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(
            total, S.N_KV_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(
            total, S.N_KV_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            out = _te_varlen_call(q, k, v, cu, cu, total, total)
            out.float().sum().backward()
            torch.cuda.synchronize(device)
        _te_backend_cache[key] = True
        del q, k, v, out
    except Exception as error:
        _te_backend_cache[key] = False
        _te_value_fusion_cache[key] = False
        _te_probe_errors[key] = f"{type(error).__name__}: {error}"
        return False
    if 2 * S.HEAD_DIM > 256:
        _te_value_fusion_cache[key] = False
        return False
    try:
        q = torch.randn(
            total, S.N_Q_HEADS, S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        fused_kv = torch.randn(
            total, S.N_KV_HEADS, 2 * S.HEAD_DIM, device=device, dtype=dtype, requires_grad=True)
        k = fused_kv[..., :S.HEAD_DIM]
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(
            device_type="cuda", enabled=False
        ):
            out = _te_varlen_call(q, k, fused_kv, cu, cu, total, total)
            out.float().sum().backward()
            torch.cuda.synchronize(device)
        ok = out.shape[-1] == 2 * S.HEAD_DIM
        del q, k, fused_kv, out
    except Exception as error:
        ok = False
        _te_probe_errors[key] = (
            "fused-value probe only (base varlen works): "
            f"{type(error).__name__}: {error}")
    del cu
    _te_value_fusion_cache[key] = ok
    return ok


def _te_varlen_eligible(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda" or tensor.dtype not in (torch.float16, torch.bfloat16):
        return False
    idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
    return _te_backend_cache.get((idx, tensor.dtype, S.HEAD_DIM), False)


def _te_attention_contexts(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", enabled=False):
        k_ctx = _te_varlen_call(
            q, k, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
        value_ctx = _te_varlen_call(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    return k_ctx, value_ctx


def _varlen_backend(tensor: torch.Tensor) -> Optional[str]:
    if S.ATTN_IMPL not in ("auto", "flash"):
        return None
    backend = None
    if _flash_varlen_eligible(tensor):
        backend = "flash"
    elif _te_varlen_eligible(tensor):
        backend = "transformer_engine"
    if backend is not None:
        idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
        report_key = (idx, tensor.dtype, S.HEAD_DIM, backend)
        if report_key not in _attention_backend_reported:
            _attention_backend_reported.add(report_key)
            ddp_print(
                f"[attention] packed varlen backend={backend} device=cuda:{idx} "
                f"dtype={tensor.dtype} head_dim={S.HEAD_DIM}")
    return backend


def _varlen_backend_failure_detail(
    device: torch.device, dtype: torch.dtype
) -> str:
    idx = torch.cuda.current_device() if device.index is None else int(device.index)
    key = (idx, dtype, S.HEAD_DIM)
    if _flash_varlen_func is None:
        flash = (
            f"FlashAttention import failed ({_flash_varlen_import_error})"
            if _flash_varlen_import_error
            else "FlashAttention varlen API is unavailable"
        )
    else:
        flash = (
            f"FlashAttention probe failed ({_flash_probe_errors[key]})"
            if key in _flash_probe_errors
            else "FlashAttention is ineligible for this device/dtype/head_dim"
        )
    te = (
        f"Transformer Engine probe failed ({_te_probe_errors[key]})"
        if key in _te_probe_errors
        else "Transformer Engine varlen DPA is unavailable or ineligible"
    )
    return f"{flash}; {te}"


def _varlen_value_fusion_enabled(backend: str, tensor: torch.Tensor) -> bool:
    if not S.ATTN_SDPA_VALUE_FUSION:
        return False
    idx = torch.cuda.current_device() if tensor.device.index is None else int(tensor.device.index)
    key = (idx, tensor.dtype, S.HEAD_DIM)
    if backend == "flash":
        return _flash_value_fusion_cache.get(key, False)
    if backend == "transformer_engine":
        return _te_value_fusion_cache.get(key, False)
    return False


def _varlen_attention_contexts(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    fused_value: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fused_value is not None:
        k_view = fused_value[..., :S.HEAD_DIM]
        if backend == "flash":
            func = _get_flash_varlen_func()
            with torch.autocast(device_type="cuda", enabled=False):
                out = _call_flash_varlen(
                    func,
                    q, k_view, fused_value,
                    cu_seqlens_q, cu_seqlens_k,
                    int(max_seqlen_q), int(max_seqlen_k),
                    dropout_p=0.0,
                    causal=True,
                    deterministic=torch.are_deterministic_algorithms_enabled(),
                )
        elif backend == "transformer_engine":
            with torch.autocast(device_type="cuda", enabled=False):
                out = _te_varlen_call(
                    q, k_view, fused_value,
                    cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q, max_seqlen_k)
        else:
            raise ValueError(f"unknown varlen attention backend {backend!r}")
        return out.split(S.HEAD_DIM, dim=-1)
    if backend == "flash":
        return _flash_attention_contexts(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    if backend == "transformer_engine":
        return _te_attention_contexts(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    raise ValueError(f"unknown varlen attention backend {backend!r}")


_cuda_packed_gather_module = None
_cuda_packed_gather_tried = False


def _try_load_cuda_packed_gather():
    """Load the one-launch THD history packer used before varlen attention."""
    global _cuda_packed_gather_module, _cuda_packed_gather_tried
    if _cuda_packed_gather_tried:
        return _cuda_packed_gather_module
    _cuda_packed_gather_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_packed_gather_module = build_or_load(
            "loomformer_packed_gather",
            ["packed_gather/packed_gather_launcher.cu"],
            ptx_kernels={"packed_gather": "packed_gather/packed_gather_kernel.cu"},
        )
    except Exception as e:
        _cuda_packed_gather_module = None
        ddp_print(
            f"[loomformer] CUDA packed_gather failed ({type(e).__name__}: {e}).")
    return _cuda_packed_gather_module


def _cuda_packed_gather_or_fallback(tensor: torch.Tensor):
    if tensor.device.type != "cuda":
        return None
    module = _try_load_cuda_packed_gather()
    if module is not None:
        return module
    major, _minor = torch.cuda.get_device_capability(tensor.device)
    if major >= 8 and S.ATTN_IMPL in ("auto", "flash"):
        raise RuntimeError(
            "packed varlen attention needs the CUDA packed_gather extension, "
            "but it failed to build/load; refusing the O(history_chunks) "
            "PyTorch packer on a production FlashAttention GPU")
    return None


class _PackedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selectors, destinations, piece_offsets, count, *chunks):
        module = _cuda_packed_gather_or_fallback(chunks[0])
        if module is None:
            raise RuntimeError("CUDA packed_gather extension is unavailable")
        count = int(count)
        if count != len(chunks):
            raise ValueError(f"packed_gather expected {count} chunks, got {len(chunks)}")
        first = chunks[0]
        ctx.save_for_backward(selectors, destinations, piece_offsets)
        ctx.chunk_lengths = tuple(int(chunk.shape[1]) for chunk in chunks)
        ctx.batch = int(first.shape[0])
        ctx.heads = int(first.shape[2])
        ctx.head_dim = int(first.shape[3])
        return module.forward(
            list(chunks), selectors, destinations, piece_offsets)

    @staticmethod
    def backward(ctx, grad_output):
        selectors, destinations, piece_offsets = ctx.saved_tensors
        module = _cuda_packed_gather_module
        if module is None:
            raise RuntimeError("CUDA packed_gather disappeared before backward")
        grads = module.backward(
            grad_output.contiguous(),
            selectors,
            destinations,
            piece_offsets,
            list(ctx.chunk_lengths),
            ctx.batch,
            ctx.heads,
            ctx.head_dim,
        )
        return (None, None, None, None, *grads)


class _PackedGatherPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selectors, destinations, piece_offsets, count, *chunks):
        count = int(count)
        if len(chunks) != 2 * count:
            raise ValueError(
                f"packed_gather_pair expected {2 * count} tensors, got {len(chunks)}")
        k_chunks = chunks[:count]
        v_chunks = chunks[count:]
        module = _cuda_packed_gather_or_fallback(k_chunks[0])
        if module is None:
            raise RuntimeError("CUDA packed_gather extension is unavailable")
        first = k_chunks[0]
        ctx.save_for_backward(selectors, destinations, piece_offsets)
        ctx.chunk_lengths = tuple(int(chunk.shape[1]) for chunk in k_chunks)
        ctx.batch = int(first.shape[0])
        ctx.heads = int(first.shape[2])
        ctx.head_dim = int(first.shape[3])
        return module.forward_pair(
            list(k_chunks), list(v_chunks),
            selectors, destinations, piece_offsets)

    @staticmethod
    def backward(ctx, grad_output):
        selectors, destinations, piece_offsets = ctx.saved_tensors
        module = _cuda_packed_gather_module
        if module is None:
            raise RuntimeError("CUDA packed_gather disappeared before backward")
        grads = module.backward_pair(
            grad_output.contiguous(),
            selectors,
            destinations,
            piece_offsets,
            list(ctx.chunk_lengths),
            ctx.batch,
            ctx.heads,
            ctx.head_dim,
        )
        return (None, None, None, None, *grads)


@torch._dynamo.disable
def _pack_selected_chunk_history(
    chunks: Tuple[torch.Tensor, ...],
    packed: PackedChunkLayout,
) -> torch.Tensor:
    """Gather live document prefixes without concatenating completed history."""
    if len(chunks) != len(packed.piece_sizes):
        raise ValueError(
            f"packed selector count {len(packed.piece_sizes)} != chunk count {len(chunks)}")
    total = packed.destinations.numel()
    if total == 0:
        raise RuntimeError("packed attention selected no history")
    first = chunks[0]
    if first.device.type == "cuda" and _cuda_packed_gather_or_fallback(first) is not None:
        return _PackedGather.apply(
            packed.selectors, packed.destinations, packed.piece_offsets,
            len(chunks), *chunks)
    document_major = first.new_empty(total, first.shape[-2], first.shape[-1])
    offset = 0
    for chunk, size in zip(chunks, packed.piece_sizes):
        selector = packed.selectors[offset:offset + size]
        destination = packed.destinations[offset:offset + size]
        offset += size
        if size == 0:
            continue
        selected = chunk.reshape(
            -1, chunk.shape[-2], chunk.shape[-1]).index_select(0, selector)
        # index_copy_ accepts only int64 indices. Keep persistent metadata
        # int32 and widen only the current small index vector.
        document_major.index_copy_(0, destination.to(torch.int64), selected)
    return document_major


@torch._dynamo.disable
def _pack_selected_chunk_kv(
    k_chunks: Tuple[torch.Tensor, ...],
    v_chunks: Tuple[torch.Tensor, ...],
    packed: PackedChunkLayout,
) -> torch.Tensor:
    """Pack [K|V] directly into one THD allocation for fused value attention."""
    if len(k_chunks) != len(v_chunks) or len(k_chunks) != len(packed.piece_sizes):
        raise ValueError("K/V chunk history does not match packed selectors")
    total = packed.destinations.numel()
    if total == 0:
        raise RuntimeError("packed attention selected no history")
    first = k_chunks[0]
    if first.device.type == "cuda" and _cuda_packed_gather_or_fallback(first) is not None:
        return _PackedGatherPair.apply(
            packed.selectors, packed.destinations, packed.piece_offsets,
            len(k_chunks), *k_chunks, *v_chunks)
    fused = first.new_empty(total, first.shape[-2], 2 * first.shape[-1])
    offset = 0
    for k_chunk, v_chunk, size in zip(k_chunks, v_chunks, packed.piece_sizes):
        selector = packed.selectors[offset:offset + size]
        destination = packed.destinations[offset:offset + size]
        offset += size
        if size == 0:
            continue
        shape = (-1, k_chunk.shape[-2], k_chunk.shape[-1])
        k_selected = k_chunk.reshape(shape).index_select(0, selector)
        v_selected = v_chunk.reshape(shape).index_select(0, selector)
        piece = torch.cat((k_selected, v_selected), dim=-1)
        fused.index_copy_(0, destination.to(torch.int64), piece)
    return fused


def _packed_chunk_mask(
    layout: Optional[PackedAttentionLayout],
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    q_pos = torch.arange(start, end, device=device, dtype=torch.long).view(1, 1, end - start, 1)
    k_pos = torch.arange(end, device=device, dtype=torch.long).view(1, 1, 1, end)
    allowed = k_pos <= q_pos
    if layout is not None:
        q_seg = layout.segment_ids[:, start:end].view(-1, 1, end - start, 1)
        k_seg = layout.segment_ids[:, :end].view(-1, 1, 1, end)
        allowed = allowed & (q_seg == k_seg)
    return allowed


def _sdpa_compute_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    src_dtype = q.dtype
    mode = S.ATTN_SDPA_COMPUTE_DTYPE
    target: Optional[torch.dtype]
    if mode == "model":
        target = None
    elif mode == "fp32":
        target = torch.float32
    elif mode == "fp16":
        target = torch.float16
    elif mode == "bf16":
        target = torch.bfloat16
    elif mode == "auto" and src_dtype == torch.bfloat16 and not _bf16_efficient_sdpa_supported(q.device):
        target = torch.float32
    else:
        target = None
    if target is None or src_dtype == target:
        return q, k, v, src_dtype
    return q.to(target), k.to(target), v.to(target), src_dtype


class _RecomputeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attn_mask: Optional[torch.Tensor], is_causal: bool) -> torch.Tensor:
        scale = 1.0 / math.sqrt(q.shape[-1])
        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-1, -2)) * scale
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            elif is_causal:
                T, S = q.shape[-2], k.shape[-2]
                q_pos = torch.arange(S - T, S, device=q.device)[:, None]
                k_pos = torch.arange(S, device=q.device)[None, :]
                causal = (k_pos <= q_pos).view(1, 1, T, S)
                scores = scores.masked_fill(~causal, float("-inf"))
            p = torch.softmax(scores, dim=-1)
            out = torch.matmul(p, v)
        mask_to_save = attn_mask if attn_mask is not None else torch.empty(0, device=q.device, dtype=torch.bool)
        ctx.save_for_backward(q, k, v, mask_to_save)
        ctx.has_mask = attn_mask is not None
        ctx.is_causal = bool(is_causal)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, attn_mask = ctx.saved_tensors
        scale = ctx.scale
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if ctx.has_mask:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        elif ctx.is_causal:
            T, S = q.shape[-2], k.shape[-2]
            q_pos = torch.arange(S - T, S, device=q.device)[:, None]
            k_pos = torch.arange(S, device=q.device)[None, :]
            causal = (k_pos <= q_pos).view(1, 1, T, S)
            scores = scores.masked_fill(~causal, float("-inf"))
        p = torch.softmax(scores, dim=-1)
        dv = torch.matmul(p.transpose(-1, -2), grad_out)
        dp = torch.matmul(grad_out, v.transpose(-1, -2))
        ds = (dp - (dp * p).sum(dim=-1, keepdim=True)) * p
        dq = torch.matmul(ds, k) * scale
        dk = torch.matmul(ds.transpose(-1, -2), q) * scale
        return dq, dk, dv, None, None


def _attention_contexts_sdpa(qg: torch.Tensor, kg: torch.Tensor, vg: torch.Tensor,
                             attn_mask: Optional[torch.Tensor], is_causal: bool,
                             cat_label: str) -> Tuple[torch.Tensor, torch.Tensor]:
    q_sdpa, k_sdpa, v_sdpa, out_dtype = _sdpa_compute_inputs(qg, kg, vg)
    ac = (
        torch.autocast(device_type="cuda", enabled=False)
        if q_sdpa.device.type == "cuda" and q_sdpa.dtype == torch.float32
        else contextlib.nullcontext()
    )
    if S.ATTN_SDPA_VALUE_FUSION:
        with _profile_region(cat_label):
            kv = torch.cat((k_sdpa, v_sdpa), dim=-1)
        with ac:
            if S.ATTN_SDPA_RECOMPUTE_BACKWARD:
                out = _RecomputeAttention.apply(q_sdpa, k_sdpa, kv, attn_mask, is_causal)
            else:
                out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, kv, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
        out = out.to(dtype=out_dtype)
        return out.split(S.HEAD_DIM, dim=-1)
    with ac:
        if S.ATTN_SDPA_RECOMPUTE_BACKWARD:
            c_g = _RecomputeAttention.apply(q_sdpa, k_sdpa, v_sdpa, attn_mask, is_causal)
            kctx_g = _RecomputeAttention.apply(q_sdpa, k_sdpa, k_sdpa, attn_mask, is_causal)
        else:
            c_g = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
            kctx_g = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, k_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal)
    return kctx_g.to(dtype=out_dtype), c_g.to(dtype=out_dtype)


def _attention_contexts_packed_sdpa(
    q: torch.Tensor,
    k_compact: torch.Tensor,
    v_compact: torch.Tensor,
    layout: PackedAttentionLayout,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact compact fallback: one causal SDPA per packed document.

    This path is primarily the Pascal/CPU correctness floor.  It never creates
    a block-diagonal T² mask and never lets one document enter another's SDPA
    call.  Production Ampere+ training takes the varlen FlashAttention path.
    """
    B, T, _hq, _d = q.shape
    q_flat = q.reshape(B * T, S.N_Q_HEADS, S.HEAD_DIM)
    k_flat = k_compact.reshape(B * T, S.N_KV_HEADS, S.HEAD_DIM)
    v_flat = v_compact.reshape(B * T, S.N_KV_HEADS, S.HEAD_DIM)
    # One host transfer per batch layout, not one per layer/document.
    offsets = layout.cu_seqlens.detach().to(device="cpu", dtype=torch.int64).tolist()
    kctx_parts: List[torch.Tensor] = []
    value_parts: List[torch.Tensor] = []
    for begin, finish in zip(offsets[:-1], offsets[1:]):
        q_doc = q_flat[begin:finish].transpose(0, 1).unsqueeze(0)
        k_doc = _kv_to_q_heads(
            k_flat[begin:finish].unsqueeze(0)).transpose(1, 2)
        v_doc = _kv_to_q_heads(
            v_flat[begin:finish].unsqueeze(0)).transpose(1, 2)
        kctx, value = _attention_contexts_sdpa(
            q_doc, k_doc, v_doc, attn_mask=None, is_causal=True,
            cat_label="loom.attn.cat_kv_value_packed_fallback")
        kctx_parts.append(kctx.squeeze(0).transpose(0, 1))
        value_parts.append(value.squeeze(0).transpose(0, 1))
    kctx_flat = torch.cat(kctx_parts, dim=0)
    value_flat = torch.cat(value_parts, dim=0)
    return (
        kctx_flat.view(B, T, S.N_Q_HEADS, S.HEAD_DIM),
        value_flat.view(B, T, S.N_Q_HEADS, S.HEAD_DIM),
    )


_cuda_chunk_attn_module = None
_cuda_chunk_attn_tried = False


def _try_load_cuda_chunk_attn():
    global _cuda_chunk_attn_module, _cuda_chunk_attn_tried
    if _cuda_chunk_attn_tried:
        return _cuda_chunk_attn_module
    _cuda_chunk_attn_tried = True
    try:
        from kernels.build import build_or_load
        _cuda_chunk_attn_module = build_or_load(
            "loomformer_chunk_attn",
            ["chunk_attn/chunk_attn_launcher.cu"],
            ptx_kernels={"chunk_attn": "chunk_attn/chunk_attn_kernel.cu"},
        )
    except Exception as e:
        _cuda_chunk_attn_module = None
        ddp_print(
            f"[loomformer] CUDA chunk_attention failed ({type(e).__name__}: {e}); "
            "using SLOWER PyTorch fallback.")
    return _cuda_chunk_attn_module


def _chunk_attention_list_reference(q, k_chunks, v_chunks, mask):
    qg = q.transpose(1, 2).float()
    score_parts = []
    key_parts = []
    value_parts = []
    for k, v in zip(k_chunks, v_chunks):
        kq = k.repeat_interleave(S.GQA_GROUP_SIZE, dim=2).transpose(1, 2).float()
        vq = v.repeat_interleave(S.GQA_GROUP_SIZE, dim=2).transpose(1, 2).float()
        score_parts.append(torch.matmul(qg, kq.transpose(-1, -2)) / math.sqrt(S.HEAD_DIM))
        key_parts.append(kq)
        value_parts.append(vq)
    scores = torch.cat(score_parts, dim=-1)
    if mask is None:
        T, source_len = q.shape[1], scores.shape[-1]
        q_pos = torch.arange(
            source_len - T, source_len, device=q.device)[:, None]
        k_pos = torch.arange(source_len, device=q.device)[None, :]
        mask = (k_pos <= q_pos).view(1, 1, T, source_len)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    kctx = torch.zeros_like(qg)
    ctx = torch.zeros_like(qg)
    offset = 0
    for kq, vq in zip(key_parts, value_parts):
        length = kq.shape[2]
        w = weights[..., offset:offset + length]
        kctx = kctx + torch.matmul(w, kq)
        ctx = ctx + torch.matmul(w, vq)
        offset += length
    dtype = q.dtype
    return kctx.transpose(1, 2).to(dtype), ctx.transpose(1, 2).to(dtype)


class _ChunkAttentionListFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, mask, count, *history):
        ext = _try_load_cuda_chunk_attn()
        if ext is None:
            raise RuntimeError("CUDA chunk attention is unavailable")
        n = int(count)
        ks = list(history[:n])
        vs = list(history[n:])
        kctx, value_ctx, lse = ext.chunk_attn_forward(
            q.contiguous(), ks, vs, mask.contiguous(), S.HEAD_DIM ** -0.5)
        ctx.count = n
        ctx.save_for_backward(q, mask, kctx, value_ctx, lse, *history)
        return kctx, value_ctx

    @staticmethod
    def backward(ctx, grad_kctx, grad_value_ctx):
        ext = _try_load_cuda_chunk_attn()
        if ext is None:
            raise RuntimeError("CUDA chunk attention is unavailable")
        q, mask, kctx, value_ctx, lse, *history = ctx.saved_tensors
        n = ctx.count
        grads = ext.chunk_attn_backward(
            grad_kctx.contiguous(), grad_value_ctx.contiguous(), q,
            list(history[:n]), list(history[n:]), mask, kctx, value_ctx, lse, S.HEAD_DIM ** -0.5)
        return grads[0], None, None, *grads[1:]


def _chunk_attention_list(q, k_chunks, v_chunks, mask):
    if q.is_cuda and q.dtype in (torch.float32, torch.float16, torch.bfloat16):
        ext = _try_load_cuda_chunk_attn()
        if ext is not None:
            mask_arg = (
                torch.empty(0, dtype=torch.bool, device=q.device)
                if mask is None
                else mask.to(device=q.device, dtype=torch.bool)
            )
            history = tuple(k_chunks) + tuple(v_chunks)
            return _ChunkAttentionListFused.apply(q, mask_arg, len(k_chunks), *history)
    return _chunk_attention_list_reference(q, k_chunks, v_chunks, mask)

__all__ = ('_bf16_efficient_sdpa_supported', '_get_flash_varlen_func', '_call_flash_varlen', '_flash_varlen_eligible', '_attention_compute_dtype', '_probe_flash_value_fusion', '_flash_attention_contexts', '_get_te_dpa', '_te_varlen_call', '_probe_te_value_fusion', '_te_varlen_eligible', '_te_attention_contexts', '_varlen_backend', '_varlen_backend_failure_detail', '_varlen_value_fusion_enabled', '_varlen_attention_contexts', '_try_load_cuda_packed_gather', '_cuda_packed_gather_or_fallback', '_PackedGather', '_PackedGatherPair', '_pack_selected_chunk_history', '_pack_selected_chunk_kv', '_packed_chunk_mask', '_sdpa_compute_inputs', '_RecomputeAttention', '_attention_contexts_sdpa', '_attention_contexts_packed_sdpa', '_try_load_cuda_chunk_attn', '_chunk_attention_list_reference', '_ChunkAttentionListFused', '_chunk_attention_list')
