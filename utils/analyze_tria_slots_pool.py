#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import loomformer as lf
import tria
from analyze_tria_geometry import build_rows, tensor_quantiles


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokens", type=int, default=1536)
    parser.add_argument("--window", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--polarm-beta", type=float)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="tria_slots_pool.json")
    return parser.parse_args()


def pairwise_context_cosines(values):
    values = F.normalize(values.float(), dim=-1, eps=1e-12)
    pairs = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            pairs.append((values[left] * values[right]).sum(dim=-1))
    if not pairs:
        return values.new_empty(0)
    return torch.stack(pairs).flatten()


def covariance_ranks(slots):
    samples = slots.float().reshape(-1, 9)
    centered = samples - samples.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(samples.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    probability = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum()
    )
    participation_rank = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)
    return eigenvalues, effective_rank, participation_rank


def chunk_report(carry, model, chunk_index):
    slots = carry.reshape(carry.shape[0], carry.shape[1], 9).float()
    eigenvalues, effective_rank, participation_rank = covariance_ranks(slots)
    context_cosine = pairwise_context_cosines(slots)

    reader = model.tria_agg.reader
    pool = model.tria_agg.pool
    score_w = reader.make_score_w(pool.query, pool.logit_scale()).float()
    scores = (slots * score_w).sum(dim=-1)
    weights = torch.softmax(scores, dim=-1)
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    pooled_slots = (weights.unsqueeze(-1) * slots).sum(dim=1)
    reader_output = F.linear(
        pooled_slots,
        reader.proj.weight.float(),
        None if reader.proj.bias is None else reader.proj.bias.float(),
    )
    aggregated = F.linear(
        reader_output,
        model.tria_agg.up.weight.float(),
        None if model.tria_agg.up.bias is None else model.tria_agg.up.bias.float(),
    )
    aggregated_context_cosine = pairwise_context_cosines(aggregated.unsqueeze(1))

    return {
        "chunk_index": chunk_index,
        "slot_population": {
            "covariance_eigenvalues": [float(value) for value in eigenvalues],
            "covariance_effective_rank": float(effective_rank),
            "covariance_participation_rank": float(participation_rank),
            "context_cosine": tensor_quantiles(context_cosine),
            "context_abs_cosine": tensor_quantiles(context_cosine.abs()),
            "slot_norm": tensor_quantiles(slots.norm(dim=-1)),
        },
        "pool": {
            "normalized_entropy": tensor_quantiles(entropy / math.log(slots.shape[1])),
            "effective_streams": tensor_quantiles(entropy.exp()),
            "max_weight": tensor_quantiles(weights.amax(dim=-1)),
        },
        "aggregated_key": {
            "norm": tensor_quantiles(aggregated.norm(dim=-1)),
            "context_cosine": tensor_quantiles(aggregated_context_cosine),
        },
    }, slots, aggregated


def main():
    args = parse_args()
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = lf.Config.from_checkpoint_dict(dict(blob["cfg"]))
    if args.tokenizer is not None:
        cfg.tokenizer = args.tokenizer
    token_count = int(args.tokens)
    if token_count <= 0:
        raise ValueError("tokens must be positive")
    cfg.seq_len = token_count
    if args.window is not None:
        cfg.tria_temporal_window = int(args.window)
    if args.alpha is not None:
        cfg.tria_carrier_alpha = float(args.alpha)
    if args.polarm_beta is not None:
        cfg.tria_polarm_beta = float(args.polarm_beta)
    cfg.grad_checkpointing = False
    cfg.device = None
    lf.apply_config(cfg)
    tria.set_cuda_tria_enabled(bool(cfg.use_cuda_tria))
    tria.set_carrier_alpha(float(cfg.tria_carrier_alpha))

    device = torch.device(args.device)
    window = int(cfg.tria_temporal_window)
    if token_count % window:
        raise ValueError("tokens must be divisible by the temporal window")
    rows = build_rows(cfg, args.data, token_count, int(args.sequences))

    model = lf.Model()
    lf.load_model_blob_into(model, blob, ablation=False)
    model.to(device).eval().requires_grad_(False)
    model.head = torch.nn.Identity()
    position_ids = torch.arange(token_count, device=device).view(1, token_count)
    endpoints_by_sequence = []

    with torch.inference_mode():
        for sequence_index, row in enumerate(rows, 1):
            captured_depth = []
            original_run_chunk_stack = model._run_chunk_stack

            def capture_run_chunk_stack(*call_args, **call_kwargs):
                hidden, carry, states = original_run_chunk_stack(*call_args, **call_kwargs)
                captured_depth.append(carry.detach())
                return hidden, carry, states

            model._run_chunk_stack = capture_run_chunk_stack
            try:
                tokens = row.view(1, token_count).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(tokens, position_ids=position_ids)
            finally:
                model._run_chunk_stack = original_run_chunk_stack

            if len(captured_depth) != token_count // window:
                raise RuntimeError(
                    f"captured {len(captured_depth)} chunks, expected {token_count // window}"
                )
            endpoints = []
            for depth_chunk in captured_depth:
                reset = torch.zeros(1, window, dtype=torch.bool, device=device)
                reset[:, 0] = True
                endpoints.append(tria.temporal_carry_endpoint(depth_chunk, reset).detach())
            endpoints_by_sequence.append(torch.stack(endpoints, dim=1).cpu())
            print(f"sequence {sequence_index}/{len(rows)}", flush=True)

    del original_run_chunk_stack, capture_run_chunk_stack
    del tokens, captured_depth, endpoints, depth_chunk, reset
    endpoints = torch.cat(endpoints_by_sequence, dim=0).float()
    chunks = []
    slot_chunks = []
    aggregated_chunks = []
    model.cpu()
    for chunk_index in range(endpoints.shape[1]):
        report, slots, aggregated = chunk_report(endpoints[:, chunk_index], model, chunk_index)
        chunks.append(report)
        slot_chunks.append(slots)
        aggregated_chunks.append(aggregated)

    adjacent_slots = []
    adjacent_keys = []
    for chunk_index in range(1, len(slot_chunks)):
        left = F.normalize(slot_chunks[chunk_index - 1], dim=-1, eps=1e-12)
        right = F.normalize(slot_chunks[chunk_index], dim=-1, eps=1e-12)
        adjacent_slots.append((left * right).sum(dim=-1).flatten())
        left_key = F.normalize(aggregated_chunks[chunk_index - 1], dim=-1, eps=1e-12)
        right_key = F.normalize(aggregated_chunks[chunk_index], dim=-1, eps=1e-12)
        adjacent_keys.append((left_key * right_key).sum(dim=-1).flatten())
    adjacent_slots = torch.cat(adjacent_slots)
    adjacent_keys = torch.cat(adjacent_keys)

    report = {
        "checkpoint": args.checkpoint,
        "step": int(blob.get("step", 0) or 0),
        "dataset": args.data,
        "window": window,
        "tokens_analyzed": token_count,
        "sequences": len(rows),
        "polarm_beta": float(cfg.tria_polarm_beta),
        "rope": {
            "original_seq_len": int(cfg.rope_original_seq_len),
            "factor": float(cfg.rope_factor),
            "attention_factor": (
                float(cfg.rope_attention_factor)
                if cfg.rope_attention_factor is not None
                else float(lf._yarn_get_mscale(cfg.rope_factor))
            ),
            "cache_seq_len": token_count,
        },
        "chunks": chunks,
        "adjacent_boundary_slot_cosine": tensor_quantiles(adjacent_slots),
        "adjacent_boundary_slot_abs_cosine": tensor_quantiles(adjacent_slots.abs()),
        "adjacent_aggregated_key_cosine": tensor_quantiles(adjacent_keys),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"JSON saved to {args.output}")


if __name__ == "__main__":
    main()
