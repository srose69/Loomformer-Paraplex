#!/usr/bin/env python3
"""Numerical GPU parity checks for LoomFormer attention and checkpointing.

This is intentionally a standalone process. ``loomformer.apply_config`` sets
architecture globals, and isolating each CUDA device makes backend selection
and its probe caches unambiguous.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loomformer as lf
import tria


def _config(
    *,
    device: str,
    attn_impl: str,
    value_fusion: bool,
    recompute: bool = False,
    checkpointing: bool = False,
) -> lf.Config:
    return lf.Config(
        vocab=64,
        seq_len=16,
        batch_size=2,
        model_dim=32,
        n_q_heads=4,
        head_dim=8,
        n_kv_heads=2,
        hidden=64,
        # Two blocks miss the middle step+gate path entirely. Four cover
        # init+gate, repeated step+gate, terminal step and all Tria axes.
        layers=4,
        device=device,
        amp_dtype="bf16",
        attn_impl=attn_impl,
        attn_sdpa_compute_dtype="auto",
        attn_sdpa_value_fusion=value_fusion,
        attn_sdpa_recompute_backward=recompute,
        grad_checkpointing=checkpointing,
        fused_linear_ce=False,
        phase_sectors="open",
        activation="pvpowlu",
        phase_grad_mode="secant",
        tria_carry_enabled=True,
        tria_temporal_enabled=True,
        tria_temporal_auto=False,
        tria_temporal_window=8,
        tria_temporal_window_min=8,
        tria_temporal_window_max=8,
        tria_temporal_calib_device="cpu",
        use_cuda_tria=True,
        use_cuda_phase_sin=True,
        use_cuda_beta_space=True,
        use_cuda_pvpowlu=True,
        use_cuda_depth_attn=True,
    )


def _layout(device: torch.device) -> Tuple[lf.PackedAttentionLayout, torch.Tensor]:
    segment_ids = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3],
            [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4],
        ],
        dtype=torch.int32,
        device=device,
    )
    positions = torch.empty_like(segment_ids, dtype=torch.long)
    for row in range(segment_ids.shape[0]):
        start = 0
        for col in range(segment_ids.shape[1]):
            if col == 0 or segment_ids[row, col] != segment_ids[row, col - 1]:
                start = col
            positions[row, col] = col - start
    return lf.packed_layout_from_segment_ids(segment_ids), positions


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    atol: float = 3e-2,
    rtol: float = 8e-2,
) -> None:
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{label}: actual tensor contains non-finite values")
    if not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: reference tensor contains non-finite values")
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as error:
        delta = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{label}: max_abs={delta.max().item():.6g}, "
            f"mean_abs={delta.mean().item():.6g}"
        ) from error


def _assert_gradient_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
    max_relative_l2: float = 1.2e-1,
    min_cosine: float = 0.99,
) -> None:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: gradient contains non-finite values")
    delta = actual - expected
    expected_norm = torch.linalg.vector_norm(expected)
    relative_l2 = (
        torch.linalg.vector_norm(delta) / expected_norm.clamp_min(1e-12)
    ).item()
    if expected_norm.item() <= 1e-12:
        if torch.linalg.vector_norm(actual).item() > 1e-10:
            raise AssertionError(f"{label}: nonzero gradient against zero reference")
        return
    cosine = torch.nn.functional.cosine_similarity(
        actual.unsqueeze(0), expected.unsqueeze(0), dim=1
    ).item()
    if relative_l2 > max_relative_l2 or cosine < min_cosine:
        raise AssertionError(
            f"{label}: relative_l2={relative_l2:.6g} "
            f"(limit {max_relative_l2}), cosine={cosine:.6g} "
            f"(limit {min_cosine})"
        )


def _attention_run(
    cfg: lf.Config,
    state: Dict[str, torch.Tensor],
    source: torch.Tensor,
    layout: lf.PackedAttentionLayout,
    positions: torch.Tensor,
    output_grads: Tuple[torch.Tensor, ...],
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, Dict[str, torch.Tensor]]:
    lf.apply_config(cfg)
    module = lf.GroupedQueryCausalSelfAttention().to(source.device)
    module.load_state_dict(state)
    value = source.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = module(value, layout, positions)
        loss = sum(
            (output.float() * grad).sum()
            for output, grad in zip(outputs, output_grads)
        )
    loss.backward()
    parameter_grads = {
        name: parameter.grad.detach().float().cpu()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }
    return (
        tuple(output.detach().float().cpu() for output in outputs),
        value.grad.detach().float().cpu(),
        parameter_grads,
    )


def _compare_attention_runs(label: str, actual, reference) -> None:
    for index, (got, expected) in enumerate(zip(actual[0], reference[0])):
        _assert_close(got, expected, label=f"{label} output[{index}]")
    _assert_gradient_close(
        actual[1], reference[1], label=f"{label} input gradient")
    if actual[2].keys() != reference[2].keys():
        raise AssertionError(f"{label}: parameter-gradient key mismatch")
    for name in actual[2]:
        _assert_gradient_close(
            actual[2][name],
            reference[2][name],
            label=f"{label} parameter gradient {name}",
        )


def _attention_matrix(device: torch.device, modern: bool) -> None:
    torch.manual_seed(101)
    base_cfg = _config(
        device=str(device),
        attn_impl="manual",
        value_fusion=False,
    )
    lf.apply_config(base_cfg)
    seed_module = lf.GroupedQueryCausalSelfAttention().to(device)
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_module.state_dict().items()
    }
    source = torch.randn(2, 16, 32, device=device)
    layout, positions = _layout(device)
    output_grads = tuple(
        torch.randn_like(output, dtype=torch.float32)
        for output in seed_module(source, layout, positions)
    )
    del seed_module

    reference = _attention_run(
        base_cfg, state, source, layout, positions, output_grads)
    candidates = (
        ("SDPA separate K/V", replace(base_cfg, attn_impl="sdpa")),
        (
            "SDPA fused [K;V]",
            replace(base_cfg, attn_impl="sdpa", attn_sdpa_value_fusion=True),
        ),
        (
            "SDPA recompute backward",
            replace(
                base_cfg,
                attn_impl="sdpa",
                attn_sdpa_value_fusion=True,
                attn_sdpa_recompute_backward=True,
            ),
        ),
    )
    for label, cfg in candidates:
        actual = _attention_run(
            cfg, state, source, layout, positions, output_grads)
        _compare_attention_runs(label, actual, reference)
        print(f"[gpu-parity] PASS {label}", flush=True)

    if not modern:
        print("[gpu-parity] SM<8: varlen backend parity is inapplicable", flush=True)
        return

    # Both probes include a real backward. FlashAttention is preferred; TE is
    # accepted as the production varlen backend when FA is unavailable.
    lf.apply_config(replace(base_cfg, attn_impl="flash"))
    flash_fused = lf._probe_flash_value_fusion(device, torch.bfloat16)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (int(index), torch.bfloat16, lf.HEAD_DIM)
    if not lf._flash_backend_cache.get(key, False):
        lf._probe_te_value_fusion(device, torch.bfloat16)
    if not (
        lf._flash_backend_cache.get(key, False)
        or lf._te_backend_cache.get(key, False)
    ):
        detail = lf._varlen_backend_failure_detail(device, torch.bfloat16)
        raise RuntimeError(f"no validated bf16 varlen backend: {detail}")

    for fusion in (False, True):
        cfg = replace(
            base_cfg,
            attn_impl="flash",
            attn_sdpa_value_fusion=fusion,
        )
        actual = _attention_run(
            cfg, state, source, layout, positions, output_grads)
        label = f"varlen {'fused [K;V]' if fusion else 'separate K/V'}"
        _compare_attention_runs(label, actual, reference)
        print(
            f"[gpu-parity] PASS {label} "
            f"(fused probe supported={flash_fused})",
            flush=True,
        )


def _model_run(
    cfg: lf.Config,
    state: Dict[str, torch.Tensor],
    tokens: torch.Tensor,
    labels: torch.Tensor,
    layout: lf.PackedAttentionLayout,
    positions: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    lf.apply_config(cfg)
    model = lf.Model(cfg).to(tokens.device).train()
    model.load_state_dict(state)
    runner = lf.maybe_compile(model, tokens.device, enabled=bool(cfg.compile))
    torch.manual_seed(303)
    checkpoint_calls = 0
    original_checkpoint = torch.utils.checkpoint.checkpoint

    def counted_checkpoint(*args, **kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return original_checkpoint(*args, **kwargs)

    if cfg.grad_checkpointing:
        torch.utils.checkpoint.checkpoint = counted_checkpoint
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = runner(
                tokens,
                attn_mask=layout,
                position_ids=positions,
                labels=labels,
            )
        loss.backward()
    finally:
        torch.utils.checkpoint.checkpoint = original_checkpoint
    if cfg.grad_checkpointing and checkpoint_calls == 0:
        raise AssertionError(
            "grad_checkpointing=True did not invoke activation checkpointing"
        )
    grads = {
        name: parameter.grad.detach().float().cpu()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach().float().cpu(), grads


def _compare_model_runs(
    label: str,
    actual: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    reference: Tuple[torch.Tensor, Dict[str, torch.Tensor]],
) -> None:
    _assert_close(
        actual[0], reference[0], label=f"{label} loss",
        atol=2e-2, rtol=2e-2,
    )
    if actual[1].keys() != reference[1].keys():
        missing = reference[1].keys() ^ actual[1].keys()
        raise AssertionError(
            f"{label}: gradient key mismatch: {sorted(missing)}"
        )
    for name in reference[1]:
        _assert_gradient_close(
            actual[1][name],
            reference[1][name],
            label=f"{label} gradient {name}",
        )


def _run_with_production_kernel_coverage(run, *, modern: bool):
    """Require the real chunked PT fused paths to execute fwd and bwd."""
    classes = {
        "paraplex": lf._ParaplexFused,
        "depth_attn_list": lf._DepthAttnListFused,
        "tria_init_gate": tria._TriaInitAndGateFused,
        "tria_init_seed_gate": tria._TriaInitSeedAndGateFused,
        "tria_step_gate": tria._TriaStepAndGateFused,
        "tria_terminal_step": tria._TriaStepFused,
        "temporal_endpoint": tria._TemporalCarryEndpointFused,
        "slot_attention_pool": tria._SlotAttentionPoolFused,
        "final_ca_sparse": tria._FinalCASparseFused,
    }
    if modern:
        classes["packed_gather_pair"] = lf._PackedGatherPair
    counts = {
        name: {"forward": 0, "backward": 0}
        for name in classes
    }
    originals = {}
    for name, cls in classes.items():
        original_forward = cls.forward
        original_backward = cls.backward
        originals[cls] = (original_forward, original_backward)

        def forward(*args, _name=name, _original=original_forward):
            counts[_name]["forward"] += 1
            return _original(*args)

        def backward(*args, _name=name, _original=original_backward):
            counts[_name]["backward"] += 1
            return _original(*args)

        cls.forward = staticmethod(forward)
        cls.backward = staticmethod(backward)
    try:
        result = run()
    finally:
        for cls, (forward, backward) in originals.items():
            cls.forward = staticmethod(forward)
            cls.backward = staticmethod(backward)
    missing = [
        f"{name}.{direction}"
        for name, directions in counts.items()
        for direction, count in directions.items()
        if count == 0
    ]
    if missing:
        raise AssertionError(
            "production fused-kernel coverage missing: "
            + ", ".join(missing)
        )
    print(
        "[gpu-parity] PASS production fused-kernel fwd/bwd coverage: "
        + ", ".join(classes),
        flush=True,
    )
    return result


def _checkpoint_matrix(
    device: torch.device,
    modern: bool,
    compile_supported: bool,
) -> None:
    base_cfg = _config(
        device=str(device),
        attn_impl="sdpa",
        value_fusion=True,
        checkpointing=False,
    )
    lf.apply_config(base_cfg)
    torch.manual_seed(202)
    seed_model = lf.Model(base_cfg).to(device).train()
    state = {
        name: tensor.detach().clone()
        for name, tensor in seed_model.state_dict().items()
    }
    tokens = torch.randint(0, base_cfg.vocab, (2, 16), device=device)
    labels = tokens.roll(-1, dims=1)
    # Synthetic assistant-only supervision: every document contains ignored
    # prompt tokens and trained response tokens.
    labels[:, ::3] = -100
    layout, positions = _layout(device)
    del seed_model

    eager = _run_with_production_kernel_coverage(
        lambda: _model_run(
            base_cfg, state, tokens, labels, layout, positions
        ),
        modern=modern,
    )
    checkpointed = _model_run(
        replace(base_cfg, grad_checkpointing=True),
        state,
        tokens,
        labels,
        layout,
        positions,
    )
    _compare_model_runs("eager checkpoint=on", checkpointed, eager)
    print(
        "[gpu-parity] PASS eager checkpoint=off/on forward/backward",
        flush=True,
    )

    if compile_supported:
        compiled = _model_run(
            replace(base_cfg, compile=True),
            state,
            tokens,
            labels,
            layout,
            positions,
        )
        compiled_checkpointed = _model_run(
            replace(base_cfg, compile=True, grad_checkpointing=True),
            state,
            tokens,
            labels,
            layout,
            positions,
        )
        _compare_model_runs("compiled checkpoint=off", compiled, eager)
        _compare_model_runs(
            "compiled checkpoint=on", compiled_checkpointed, eager
        )
        _compare_model_runs(
            "compiled checkpoint off/on",
            compiled_checkpointed,
            compiled,
        )
        print(
            "[gpu-parity] PASS compile checkpoint=off/on "
            "forward/backward and eager parity",
            flush=True,
        )
    else:
        print(
            "[gpu-parity] SKIP compile checkpoint matrix (SM<7)",
            flush=True,
        )

    if modern:
        varlen = _model_run(
            replace(base_cfg, attn_impl="flash"),
            state,
            tokens,
            labels,
            layout,
            positions,
        )
        _assert_close(
            varlen[0], eager[0], label="varlen model loss",
            atol=4e-2, rtol=4e-2)
        if varlen[1].keys() != eager[1].keys():
            raise AssertionError("varlen model gradient key mismatch")
        for name in eager[1]:
            _assert_gradient_close(
                varlen[1][name],
                eager[1][name],
                label=f"varlen model gradient {name}",
                max_relative_l2=2e-1,
                min_cosine=0.98,
            )
        print("[gpu-parity] PASS full-model SDPA/varlen parity", flush=True)


@torch.no_grad()
def _incremental_model_parity(device: torch.device) -> None:
    cfg = replace(
        _config(device=str(device), attn_impl="sdpa", value_fusion=True),
        # Force the cap to engage on random-init branches; cap=1 can be a
        # numerical no-op here and would not detect a missing step() call.
        residual_branch_rms_cap=0.05,
    )
    lf.apply_config(cfg)
    torch.manual_seed(707)
    model = lf.Model(cfg).to(device).eval()
    tokens = torch.randint(0, cfg.vocab, (2, 7), device=device)
    positions = torch.arange(tokens.shape[1], device=device).view(1, -1).expand_as(tokens)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        full = model(tokens, position_ids=positions)
        states = None
        pieces = []
        for pos in range(tokens.shape[1]):
            logits, states = model.step(tokens[:, pos:pos + 1], pos, states)
            pieces.append(logits[:, None, :])
        incremental = torch.cat(pieces, dim=1)
    _assert_close(
        incremental.float(), full.float(),
        label="full/incremental model with residual RMS cap",
        atol=5e-2, rtol=8e-2,
    )
    print("[gpu-parity] PASS full/incremental model with residual RMS cap", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="one CUDA device, e.g. cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU parity requires CUDA")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must select a CUDA device")
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    print(
        f"[gpu-parity] {device} {torch.cuda.get_device_name(device)} "
        f"SM{major}.{minor}",
        flush=True,
    )
    _attention_matrix(device, major >= 8)
    _checkpoint_matrix(device, major >= 8, major >= 7)
    _incremental_model_parity(device)
    torch.cuda.synchronize(device)
    print(f"[gpu-parity] ALL CASES PASSED on {device}", flush=True)


if __name__ == "__main__":
    main()
