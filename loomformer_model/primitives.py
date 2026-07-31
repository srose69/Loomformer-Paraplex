from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import state as S

def cuda_autocast_dtype_or_none():
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:
        if torch.is_autocast_cuda_enabled():
            return torch.get_autocast_gpu_dtype()
    except Exception:
        return None
    return None

def fanin_std(fan_in: int, gain: float = S.FANIN_GAIN) -> float:
    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive, got {fan_in}")
    return gain / math.sqrt(float(fan_in))


def residual_std(fan_in: int, gain: float = S.FANIN_GAIN) -> float:
    beta = S.DEEPNORM_BETA if S.RESIDUAL_INIT == "beta" else 1.0
    return fanin_std(fan_in, gain) * beta


def fixed_rms(x: torch.Tensor, target: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Scale each last-dimension vector by ``target / sqrt(mean(x²) + eps)``."""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    scale = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps) * float(target)
    return x * scale.to(dtype=x.dtype)


def capped_rms(x: torch.Tensor, maximum: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS scaling up to ``maximum`` without amplifying the input."""
    work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    rms = torch.sqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    scale = (float(maximum) / rms).clamp(max=1.0)
    return x * scale.to(dtype=x.dtype)


def init_linear_fanin(m: nn.Linear, gain: float = S.FANIN_GAIN, zero_bias: bool = True) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=fanin_std(m.weight.shape[1], gain))
    if zero_bias and m.bias is not None:
        nn.init.zeros_(m.bias)


def init_linear_residual(m: nn.Linear, gain: float = S.FANIN_GAIN, zero_bias: bool = True) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=residual_std(m.weight.shape[1], gain))
    if zero_bias and m.bias is not None:
        nn.init.zeros_(m.bias)


def init_embedding_fanin(m: nn.Embedding, gain: float = S.FANIN_GAIN) -> None:
    nn.init.normal_(m.weight, mean=0.0, std=fanin_std(m.embedding_dim, gain))

def powlu(x: torch.Tensor, m: float = 3.0) -> torch.Tensor:
    x_pos_safe = x.clamp(min=1e-12)
    exponent = 1.0 + m / (torch.sqrt(x_pos_safe) + 1.0)
    pos_val = x_pos_safe.pow(exponent) * torch.sigmoid(x)
    neg_val = x.pow(2) * torch.sigmoid(x)
    return torch.where(x > 0, pos_val, neg_val)


def powlu_gate(x2: torch.Tensor, m: float = 3.0) -> torch.Tensor:
    """Apply the positive-input PowLU gate used by the GLU path."""
    exponent = m / (torch.sqrt(x2) + 1.0)
    return x2.pow(exponent) * torch.sigmoid(x2)


def act_fn(x: torch.Tensor) -> torch.Tensor:
    """Apply the configured GELU or PowLU activation."""
    return F.gelu(x) if S.ACTIVATION == "gelu" else powlu(x, S.POWLU_M)


class RMSNorm(nn.Module):
    """RMS normalization with a learned scale and no bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms).to(dtype) * self.weight


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """Chunked fused (LM-head projection + softmax cross-entropy).

    Direct port of the Liger-Kernel algorithm (https://github.com/linkedin/
    Liger-Kernel, src/liger_kernel/ops/fused_linear_cross_entropy.py), minus
    the Triton kernel (plain PyTorch ops instead of a fused Triton column
    block, since this repo has no Triton dependency): the [N, D] @ [D, V]^T
    head projection and the token-level cross-entropy are computed one
    row-chunk at a time, so only a [chunk_size, VOCAB] logits slice (plus its
    exact gradient) is ever alive instead of the full [N, VOCAB] tensor.
    Gradients are exact (softmax - onehot per chunk, algebraically identical
    to the unchunked computation) -- this is NOT activation-checkpoint
    recompute, there is no extra forward pass.

    Matching Liger's forward exactly: there is exactly ONE host sync for the
    non-ignored token count (needed for mean reduction), taken once before
    the chunk loop -- never per-chunk. An early `if chunk has no valid
    targets: skip` check would need its own per-chunk `.item()`/`bool()`
    sync, stalling the CUDA pipeline every single chunk for no algorithmic
    benefit (chunks are still masked correctly either way), so it is
    deliberately not done, exactly as upstream does not do it either.
    """

    @staticmethod
    def _prefers_fp32_gemm(hidden: torch.Tensor) -> bool:
        """On sm_61-class GPUs (Pascal: GTX 1080, Tesla P4 -- no bf16 tensor
        cores below sm_80) `autocast(bf16)` lowers `x @ W` to
        `magma_sgemmEx_kernel<float, __nv_bfloat16, ...>`, which does a
        serialized bf16->fp32 convert pass before the fp32 multiply-add --
        verified via torch.profiler to dominate this function's CUDA time
        (>90%) on such hardware. Forcing the GEMM operands to fp32 and
        disabling autocast for the call instead dispatches plain cuBLAS
        `sgemm`, which is substantially faster here. On sm_80+ (native bf16
        tensor cores) this would throw away real speedup, so it's gated on
        compute capability. Gated on the *active autocast state*, not
        `hidden.dtype`: whether the matmul actually lowers to bf16 depends
        on whether autocast is live right now, not on what dtype the
        incoming tensor happens to already be."""
        if not hidden.is_cuda:
            return False
        if cuda_autocast_dtype_or_none() != torch.bfloat16:
            return False
        try:
            major, _minor = torch.cuda.get_device_capability(hidden.device)
        except Exception:
            return False
        return major < 8

    @staticmethod
    def _default_chunk_size(N: int, D: int, V: int) -> int:
        """Liger's memory-balancing formula: pick the chunk size so the
        transient [chunk, V] logits buffer is on the same order as the
        [N, D] hidden-state tensor, i.e. `inc_factor = ceil(V/D)`,
        `chunk_size = next_pow2(ceil(N / inc_factor))`."""
        if N <= 0:
            return 1
        inc_factor = -(-V // D)
        raw = -(-N // inc_factor)
        pow2 = 1 << max(raw - 1, 0).bit_length()
        return max(1, min(pow2, N))

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor,
                ignore_index: int, chunk_size: int) -> torch.Tensor:
        N, D = hidden.shape
        V = weight.shape[0]
        step = int(chunk_size) if chunk_size else _FusedLinearCrossEntropy._default_chunk_size(N, D, V)
        step = max(1, min(step, max(N, 1)))

        valid_mask = targets.ne(ignore_index)
        denom = float(max(int(valid_mask.sum().item()), 1))

        want_weight_grad = weight.requires_grad
        grad_hidden = torch.zeros(N, D, dtype=torch.float32, device=hidden.device)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if want_weight_grad else None
        loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)

        # Cast once (not per chunk). Under autocast this is a no-op cost --
        # `hidden` already arrives in the autocast dtype (e.g. bf16) -- and it
        # also makes the manual backward matmuls below correct when autocast
        # is off (amp_dtype: fp32) where hidden/weight already match anyway.
        force_fp32_gemm = _FusedLinearCrossEntropy._prefers_fp32_gemm(hidden)
        compute_dtype = torch.float32 if force_fp32_gemm else hidden.dtype
        weight_c = weight if weight.dtype == compute_dtype else weight.to(compute_dtype)
        gemm_ctx = (
            torch.autocast(device_type="cuda", enabled=False)
            if force_fp32_gemm else contextlib.nullcontext()
        )

        with gemm_ctx:
            for s in range(0, N, step):
                e = min(s + step, N)
                h_chunk = hidden[s:e].to(compute_dtype) if force_fp32_gemm else hidden[s:e]
                t_chunk = targets[s:e]
                valid = valid_mask[s:e]
                logits_chunk = F.linear(h_chunk, weight_c).float()
                logp = F.log_softmax(logits_chunk, dim=-1)
                safe_t = t_chunk.clamp_min(0).unsqueeze(1)
                nll = -logp.gather(1, safe_t).squeeze(1)
                loss_sum += torch.where(valid, nll, nll.new_zeros(())).sum()
                grad_logits = logp.exp()
                grad_logits.scatter_add_(1, safe_t, -torch.ones_like(safe_t, dtype=grad_logits.dtype))
                grad_logits *= valid.unsqueeze(1)
                grad_logits = grad_logits.to(compute_dtype)
                grad_hidden[s:e] = (grad_logits @ weight_c).float()
                if want_weight_grad:
                    grad_weight += (grad_logits.t() @ h_chunk).float()
                del logits_chunk, logp, grad_logits

        grad_hidden = (grad_hidden / denom).to(hidden.dtype)
        ctx.want_weight_grad = want_weight_grad
        if want_weight_grad:
            ctx.save_for_backward(grad_hidden, (grad_weight / denom).to(weight.dtype))
        else:
            ctx.save_for_backward(grad_hidden)
        return loss_sum / denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.want_weight_grad:
            grad_hidden, grad_weight = ctx.saved_tensors
            return grad_hidden * grad_output, grad_weight * grad_output, None, None, None
        (grad_hidden,) = ctx.saved_tensors
        return grad_hidden * grad_output, None, None, None, None


@torch._dynamo.disable
def _fused_linear_cross_entropy_eager(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    chunk_size: int,
) -> torch.Tensor:
    """Explicit Dynamo boundary for the hand-fused CE implementation.

    Its one deliberate host sync computes the exact mean denominator and its
    Python chunk loop bounds peak logits memory. Letting Dynamo enter it only
    produces a noisy Tensor.item() graph break; compiling the surrounding
    model regions while executing this custom autograd op eager is the
    intended behavior.
    """
    return _FusedLinearCrossEntropy.apply(
        hidden, weight, targets, ignore_index, chunk_size)

__all__ = ('fanin_std', 'residual_std', 'fixed_rms', 'capped_rms', 'init_linear_fanin', 'init_linear_residual', 'init_embedding_fanin', 'powlu', 'powlu_gate', 'act_fn', 'RMSNorm', '_FusedLinearCrossEntropy', '_fused_linear_cross_entropy_eager')
