#!/usr/bin/env python3

import argparse
import gc
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

import loomformer as lf
import tria


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--init")
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokens", type=int, default=768)
    parser.add_argument("--raw-tokens", type=int, default=768)
    parser.add_argument("--window", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--polarm-beta", type=float)
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="tria_geometry_analysis.json")
    parser.add_argument("--max-condition", type=float, default=3.0)
    parser.add_argument("--min-effective-rank", type=float, default=2.70)
    parser.add_argument("--population-pass", type=float, default=0.90)
    return parser.parse_args()


def tensor_quantiles(values):
    values = values.float().flatten()
    return {
        "mean": float(values.mean()),
        "p10": float(values.quantile(0.10)),
        "p50": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
    }


def normalized_singular_values(singular_values):
    return singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def singular_metrics(singular_values, max_condition, min_effective_rank):
    condition = singular_values[..., 0] / singular_values[..., -1].clamp_min(1e-12)
    probability = normalized_singular_values(singular_values)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    )
    passed = (condition <= max_condition) & (effective_rank >= min_effective_rank)
    return probability, condition, effective_rank, passed


def proper_polar_rotation(u, vh):
    rotation = u @ vh
    determinant = torch.linalg.det(rotation)
    correction = torch.where(
        determinant < 0,
        -torch.ones_like(determinant),
        torch.ones_like(determinant),
    )
    corrected_u = u.clone()
    corrected_u[..., :, -1] *= correction[..., None]
    proper_rotation = corrected_u @ vh
    return proper_rotation, determinant < 0


def rotation_angle(rotation):
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cosine)


def rotation_axis(rotation):
    r00 = rotation[..., 0, 0]
    r11 = rotation[..., 1, 1]
    r22 = rotation[..., 2, 2]
    qw = 0.5 * torch.sqrt((1.0 + r00 + r11 + r22).clamp_min(0.0))
    qx = 0.5 * torch.sqrt((1.0 + r00 - r11 - r22).clamp_min(0.0))
    qy = 0.5 * torch.sqrt((1.0 - r00 + r11 - r22).clamp_min(0.0))
    qz = 0.5 * torch.sqrt((1.0 - r00 - r11 + r22).clamp_min(0.0))
    qx = torch.copysign(qx, rotation[..., 2, 1] - rotation[..., 1, 2])
    qy = torch.copysign(qy, rotation[..., 0, 2] - rotation[..., 2, 0])
    qz = torch.copysign(qz, rotation[..., 1, 0] - rotation[..., 0, 1])
    quaternion = torch.stack((qw, qx, qy, qz), dim=-1)
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-12)
    vector = quaternion[..., 1:]
    vector_norm = vector.norm(dim=-1, keepdim=True)
    axis = vector / vector_norm.clamp_min(1e-12)
    valid = vector_norm[..., 0] > 1e-5
    return axis, valid


def projector_concentration(vectors, valid=None):
    vectors = vectors.float().reshape(-1, 3)
    if valid is not None:
        valid = valid.reshape(-1)
        vectors = vectors[valid]
    if vectors.numel() == 0:
        return {
            "valid_count": 0,
            "projector_eigenvalues": [float("nan")] * 3,
            "concentration": float("nan"),
        }
    moment = torch.einsum("ni,nj->ij", vectors, vectors) / vectors.shape[0]
    eigenvalues = torch.linalg.eigvalsh(moment).flip(0)
    return {
        "valid_count": int(vectors.shape[0]),
        "projector_eigenvalues": [float(value) for value in eigenvalues],
        "concentration": float(eigenvalues[0]),
    }


def context_frame_coherence(vectors):
    vectors = vectors.float()
    sequence_count = vectors.shape[0]
    if sequence_count < 2:
        return None
    overlaps = []
    for left in range(sequence_count):
        for right in range(left + 1, sequence_count):
            overlaps.append((vectors[left] * vectors[right]).sum(dim=-1).square())
    return tensor_quantiles(torch.stack(overlaps).mean(dim=0))


def first_failure_horizon(passed):
    sequence_count, token_count, hidden = passed.shape
    failed = ~passed
    has_failed = failed.any(dim=1)
    first = failed.float().argmax(dim=1) + 1
    return torch.where(
        has_failed,
        first,
        torch.full_like(first, token_count + 1),
    ).reshape(sequence_count, hidden)


def regime_fractions(condition, effective_rank, max_condition, min_effective_rank):
    stable = (condition <= max_condition) & (effective_rank >= min_effective_rank)
    directional_full_rank = (~stable) & (effective_rank >= min_effective_rank)
    plane_like = (effective_rank < min_effective_rank) & (effective_rank >= 1.8)
    line_like = effective_rank < 1.8
    return {
        "stable": float(stable.float().mean()),
        "directional_full_rank": float(directional_full_rank.float().mean()),
        "plane_like": float(plane_like.float().mean()),
        "line_like": float(line_like.float().mean()),
    }


def build_rows(cfg, data_path, token_count, sequence_count):
    tokenizer = lf.build_tokenizer(cfg)
    bos = lf._tok_special_id(tokenizer, "<bos>")
    eos = lf._tok_special_id(tokenizer, "<eos>")
    body_length = token_count - (1 if bos is not None else 0)
    required = sequence_count * body_length
    stream = []
    first = True
    corpus = lf.RawCorpus(data_path, fmt="auto", text_field=cfg.text_field)
    for text in corpus.iter_texts():
        if not first and eos is not None:
            stream.append(eos)
        stream.extend(tokenizer.encode(text))
        first = False
        if len(stream) >= required:
            break
    if len(stream) < required:
        raise RuntimeError(f"need {required} tokens, got {len(stream)}")
    rows = []
    pointer = 0
    for _ in range(sequence_count):
        row = stream[pointer:pointer + body_length]
        pointer += body_length
        if bos is not None:
            row = [bos] + row
        rows.append(torch.tensor(row, dtype=torch.long))
    return rows


class GeometryAccumulator:
    def __init__(
        self,
        token_count,
        horizons,
        window,
        max_condition,
        min_effective_rank,
        population_pass,
    ):
        self.token_count = token_count
        self.horizons = horizons
        self.window = window
        self.max_condition = max_condition
        self.min_effective_rank = min_effective_rank
        self.population_pass = population_pass
        self.condition_sum = torch.zeros(token_count, dtype=torch.float64)
        self.effective_rank_sum = torch.zeros(token_count, dtype=torch.float64)
        self.pass_sum = torch.zeros(token_count, dtype=torch.float64)
        self.rotation_angle_sum = torch.zeros(token_count, dtype=torch.float64)
        self.incremental_angle_sum = torch.zeros(token_count, dtype=torch.float64)
        self.path_length_sum = torch.zeros(token_count, dtype=torch.float64)
        self.lyapunov_gap_12_sum = torch.zeros(token_count, dtype=torch.float64)
        self.lyapunov_gap_23_sum = torch.zeros(token_count, dtype=torch.float64)
        self.reflection_sum = torch.zeros(token_count, dtype=torch.float64)
        self.count = 0
        self.selected = {
            str(horizon): {
                "condition": [],
                "effective_rank": [],
                "singular_probability": [],
                "rotation_angle": [],
                "rotation_axis": [],
                "rotation_axis_valid": [],
                "dominant_input": [],
                "weak_input": [],
                "dominant_output": [],
                "weak_output": [],
                "rotation": [],
                "path_length": [],
            }
            for horizon in horizons
        }
        self.first_failure = []
        self.weak_window_concentration = []
        self.dominant_window_concentration = []
        self.weak_step_overlap = []
        self.dominant_step_overlap = []

    def add(self, matrix):
        u, singular_values, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
        singular_probability, condition, effective_rank, passed = singular_metrics(
            singular_values,
            self.max_condition,
            self.min_effective_rank,
        )
        rotation, reflected = proper_polar_rotation(u, vh)
        angle = rotation_angle(rotation)
        axis, axis_valid = rotation_axis(rotation)

        identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        previous = torch.cat(
            (
                identity.view(1, 1, 1, 3, 3).expand(rotation.shape[0], 1, rotation.shape[2], 3, 3),
                rotation[:, :-1],
            ),
            dim=1,
        )
        incremental = rotation @ previous.transpose(-1, -2)
        incremental_angle = rotation_angle(incremental)
        path_length = incremental_angle.cumsum(dim=1)

        horizon = torch.arange(
            1,
            self.token_count + 1,
            device=matrix.device,
            dtype=torch.float32,
        ).view(1, self.token_count, 1)
        lyapunov_gap_12 = (singular_values[..., 0] / singular_values[..., 1].clamp_min(1e-12)).log() / horizon
        lyapunov_gap_23 = (singular_values[..., 1] / singular_values[..., 2].clamp_min(1e-12)).log() / horizon

        batch, _, hidden = condition.shape
        population = batch * hidden
        self.condition_sum += condition.sum(dim=(0, 2)).double().cpu()
        self.effective_rank_sum += effective_rank.sum(dim=(0, 2)).double().cpu()
        self.pass_sum += passed.sum(dim=(0, 2)).double().cpu()
        self.rotation_angle_sum += angle.sum(dim=(0, 2)).double().cpu()
        self.incremental_angle_sum += incremental_angle.sum(dim=(0, 2)).double().cpu()
        self.path_length_sum += path_length.sum(dim=(0, 2)).double().cpu()
        self.lyapunov_gap_12_sum += lyapunov_gap_12.sum(dim=(0, 2)).double().cpu()
        self.lyapunov_gap_23_sum += lyapunov_gap_23.sum(dim=(0, 2)).double().cpu()
        self.reflection_sum += reflected.sum(dim=(0, 2)).double().cpu()
        self.count += population
        self.first_failure.append(first_failure_horizon(passed).cpu())

        dominant_input = vh[..., 0, :]
        weak_input = vh[..., -1, :]
        dominant_output = u[..., :, 0]
        weak_output = u[..., :, -1]

        dominant_overlap = (dominant_input[:, 1:] * dominant_input[:, :-1]).sum(dim=-1).square()
        weak_overlap = (weak_input[:, 1:] * weak_input[:, :-1]).sum(dim=-1).square()
        self.dominant_step_overlap.append(dominant_overlap.mean(dim=1).cpu())
        self.weak_step_overlap.append(weak_overlap.mean(dim=1).cpu())

        for start in range(0, self.token_count, self.window):
            end = min(start + self.window, self.token_count)
            for vectors, destination in (
                (dominant_input[:, start:end], self.dominant_window_concentration),
                (weak_input[:, start:end], self.weak_window_concentration),
            ):
                moment = torch.einsum("bthi,bthj->bhij", vectors, vectors) / (end - start)
                concentration = torch.linalg.eigvalsh(moment)[..., -1]
                destination.append({
                    "start": start,
                    "end": end,
                    "values": concentration.cpu(),
                })

        for selected_horizon in self.horizons:
            index = selected_horizon - 1
            destination = self.selected[str(selected_horizon)]
            destination["condition"].append(condition[:, index].flatten().cpu())
            destination["effective_rank"].append(effective_rank[:, index].flatten().cpu())
            destination["singular_probability"].append(
                singular_probability[:, index].reshape(-1, 3).cpu()
            )
            destination["rotation_angle"].append(angle[:, index].flatten().cpu())
            destination["rotation_axis"].append(axis[:, index].reshape(-1, 3).cpu())
            destination["rotation_axis_valid"].append(axis_valid[:, index].flatten().cpu())
            destination["dominant_input"].append(dominant_input[:, index].reshape(-1, hidden, 3).cpu())
            destination["weak_input"].append(weak_input[:, index].reshape(-1, hidden, 3).cpu())
            destination["dominant_output"].append(dominant_output[:, index].reshape(-1, hidden, 3).cpu())
            destination["weak_output"].append(weak_output[:, index].reshape(-1, hidden, 3).cpu())
            destination["rotation"].append(rotation[:, index].reshape(-1, hidden, 3, 3).cpu())
            destination["path_length"].append(path_length[:, index].flatten().cpu())

        return {
            "rotation": {str(h): rotation[:, h - 1].float().cpu() for h in self.horizons},
            "dominant_input": {str(h): dominant_input[:, h - 1].float().cpu() for h in self.horizons},
            "weak_input": {str(h): weak_input[:, h - 1].float().cpu() for h in self.horizons},
            "singular_probability": {
                str(h): singular_probability[:, h - 1].float().cpu() for h in self.horizons
            },
        }

    def _window_concentration_report(self, entries):
        grouped = {}
        for entry in entries:
            key = f"{entry['start']}:{entry['end']}"
            grouped.setdefault(key, []).append(entry["values"])
        return {
            key: tensor_quantiles(torch.cat(values).flatten())
            for key, values in grouped.items()
        }

    def finish(self):
        condition_mean = self.condition_sum / self.count
        effective_rank_mean = self.effective_rank_sum / self.count
        pass_fraction = self.pass_sum / self.count
        failed = torch.nonzero(pass_fraction < self.population_pass, as_tuple=False)
        stable_horizon = (
            self.token_count
            if failed.numel() == 0
            else max(1, int(failed[0, 0]))
        )
        selected_report = {}
        selected_tensors = {}
        for horizon in self.horizons:
            source = self.selected[str(horizon)]
            condition = torch.cat(source["condition"])
            effective_rank = torch.cat(source["effective_rank"])
            singular_probability = torch.cat(source["singular_probability"])
            rotation_angle_values = torch.cat(source["rotation_angle"])
            rotation_axis_values = torch.cat(source["rotation_axis"])
            rotation_axis_valid = torch.cat(source["rotation_axis_valid"])
            dominant_input = torch.cat(source["dominant_input"], dim=0)
            weak_input = torch.cat(source["weak_input"], dim=0)
            dominant_output = torch.cat(source["dominant_output"], dim=0)
            weak_output = torch.cat(source["weak_output"], dim=0)
            rotations = torch.cat(source["rotation"], dim=0)
            path_length = torch.cat(source["path_length"])
            selected_report[str(horizon)] = {
                "condition": tensor_quantiles(condition),
                "effective_rank": tensor_quantiles(effective_rank),
                "population_pass": float(
                    ((condition <= self.max_condition) & (effective_rank >= self.min_effective_rank))
                    .float()
                    .mean()
                ),
                "regimes": regime_fractions(
                    condition,
                    effective_rank,
                    self.max_condition,
                    self.min_effective_rank,
                ),
                "normalized_singular_values_mean": [
                    float(value) for value in singular_probability.mean(dim=0)
                ],
                "rotation_angle": tensor_quantiles(rotation_angle_values),
                "rotation_axis_global": projector_concentration(
                    rotation_axis_values,
                    rotation_axis_valid,
                ),
                "rotation_path_length": tensor_quantiles(path_length),
                "dominant_input_global": projector_concentration(dominant_input),
                "weak_input_global": projector_concentration(weak_input),
                "dominant_output_global": projector_concentration(dominant_output),
                "weak_output_global": projector_concentration(weak_output),
                "dominant_input_context_coherence": context_frame_coherence(dominant_input),
                "weak_input_context_coherence": context_frame_coherence(weak_input),
            }
            selected_tensors[str(horizon)] = {
                "rotation": rotations,
                "dominant_input": dominant_input,
                "weak_input": weak_input,
                "singular_probability": singular_probability,
            }

        failure = torch.cat(self.first_failure, dim=0)
        channel_mean_horizon = failure.float().mean(dim=0)
        channel_context_std = failure.float().std(dim=0, unbiased=False)
        always_stable = failure > self.token_count
        failed_by_window = failure <= self.window
        return {
            "report": {
                "stable_horizon": stable_horizon,
                "condition_mean": condition_mean.tolist(),
                "effective_rank_mean": effective_rank_mean.tolist(),
                "population_pass": pass_fraction.tolist(),
                "rotation_angle_mean": (self.rotation_angle_sum / self.count).tolist(),
                "incremental_rotation_angle_mean": (
                    self.incremental_angle_sum / self.count
                ).tolist(),
                "rotation_path_length_mean": (self.path_length_sum / self.count).tolist(),
                "finite_time_lyapunov_gap_12_mean": (
                    self.lyapunov_gap_12_sum / self.count
                ).tolist(),
                "finite_time_lyapunov_gap_23_mean": (
                    self.lyapunov_gap_23_sum / self.count
                ).tolist(),
                "polar_reflection_fraction": (self.reflection_sum / self.count).tolist(),
                "selected_horizons": selected_report,
                "per_stream_first_failure": {
                    "all": tensor_quantiles(failure.float()),
                    "channel_mean": tensor_quantiles(channel_mean_horizon),
                    "channel_context_std": tensor_quantiles(channel_context_std),
                    "fraction_always_stable": float(always_stable.float().mean()),
                    "fraction_failed_by_window": float(failed_by_window.float().mean()),
                    "channels_failed_by_window_all_contexts": float(
                        failed_by_window.all(dim=0).float().mean()
                    ),
                    "channels_failed_by_window_some_context": float(
                        failed_by_window.any(dim=0).float().mean()
                    ),
                },
                "singular_frame_dynamics": {
                    "dominant_step_overlap": tensor_quantiles(
                        torch.cat(self.dominant_step_overlap).flatten()
                    ),
                    "weak_step_overlap": tensor_quantiles(
                        torch.cat(self.weak_step_overlap).flatten()
                    ),
                    "dominant_window_projector_concentration": self._window_concentration_report(
                        self.dominant_window_concentration
                    ),
                    "weak_window_projector_concentration": self._window_concentration_report(
                        self.weak_window_concentration
                    ),
                },
            },
            "selected_tensors": selected_tensors,
            "first_failure": failure,
        }


def analyze_checkpoint(
    label,
    path,
    blob,
    rows,
    cfg,
    device,
    token_count,
    raw_token_count,
    horizons,
    max_condition,
    min_effective_rank,
    population_pass,
):
    print(f"loading {label}: {path}", flush=True)
    model = lf.Model()
    lf.load_model_blob_into(model, blob, ablation=False)
    model.to(device).eval().requires_grad_(False)
    model.head = torch.nn.Identity()
    model.capture_tria_depth_carry = True
    window = int(model.tria_temporal_window)
    raw_horizons = [horizon for horizon in horizons if horizon <= raw_token_count]
    depth_accumulator = GeometryAccumulator(
        raw_token_count,
        raw_horizons,
        window,
        max_condition,
        min_effective_rank,
        population_pass,
    )
    temporal_accumulator = GeometryAccumulator(
        raw_token_count,
        raw_horizons,
        window,
        max_condition,
        min_effective_rank,
        population_pass,
    )
    operational_horizons = [horizon for horizon in horizons if horizon <= window]
    if window not in operational_horizons:
        operational_horizons.append(window)
        operational_horizons.sort()
    operational_depth_accumulator = GeometryAccumulator(
        window,
        operational_horizons,
        window,
        max_condition,
        min_effective_rank,
        population_pass,
    )
    operational_temporal_accumulator = GeometryAccumulator(
        window,
        operational_horizons,
        window,
        max_condition,
        min_effective_rank,
        population_pass,
    )
    complete_chunks = token_count // window
    operational_depth_by_chunk = [
        GeometryAccumulator(
            window,
            operational_horizons,
            window,
            max_condition,
            min_effective_rank,
            population_pass,
        )
        for _ in range(complete_chunks)
    ]
    operational_temporal_by_chunk = [
        GeometryAccumulator(
            window,
            operational_horizons,
            window,
            max_condition,
            min_effective_rank,
            population_pass,
        )
        for _ in range(complete_chunks)
    ]
    position_ids = torch.arange(token_count, device=device).view(1, token_count)
    raw_position_ids = position_ids[:, :raw_token_count]
    paired = []
    with torch.inference_mode():
        for sequence_index, row in enumerate(rows, 1):
            tokens = row[:raw_token_count].view(1, raw_token_count).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model._forward_flat(
                    tokens,
                    attn_mask=None,
                    position_ids=raw_position_ids,
                )
            depth = model.last_tria_depth_carry
            if depth is None:
                raise RuntimeError("Tria depth carry capture failed")
            reset = torch.zeros(1, raw_token_count, dtype=torch.bool, device=device)
            reset[:, 0] = True
            temporal = tria.temporal_carry(depth.float(), reset)
            depth_accumulator.add(depth)
            paired.append(temporal_accumulator.add(temporal))
            print(f"  {label}: sequence {sequence_index}/{len(rows)}", flush=True)
            del tokens, logits, depth, temporal

        original_run_chunk_stack = model._run_chunk_stack
        for sequence_index, row in enumerate(rows, 1):
            captured_depth = []

            def capture_run_chunk_stack(*call_args, **call_kwargs):
                h, carry, states = original_run_chunk_stack(*call_args, **call_kwargs)
                captured_depth.append(carry.detach())
                return h, carry, states

            model._run_chunk_stack = capture_run_chunk_stack
            tokens = row.view(1, token_count).to(device)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(tokens, position_ids=position_ids)
            finally:
                model._run_chunk_stack = original_run_chunk_stack
            chunked_depth = torch.cat(captured_depth, dim=1)
            if chunked_depth.shape[1] != token_count:
                raise RuntimeError(
                    f"captured {chunked_depth.shape[1]} chunked tokens, expected {token_count}"
                )
            for chunk_index, start in enumerate(range(0, token_count, window)):
                end = start + window
                if end > token_count:
                    break
                depth_chunk = chunked_depth[:, start:end]
                reset = torch.zeros(1, window, dtype=torch.bool, device=device)
                reset[:, 0] = True
                temporal_chunk = tria.temporal_carry(depth_chunk.float(), reset)
                operational_depth_accumulator.add(depth_chunk)
                operational_temporal_accumulator.add(temporal_chunk)
                operational_depth_by_chunk[chunk_index].add(depth_chunk)
                operational_temporal_by_chunk[chunk_index].add(temporal_chunk)
                del depth_chunk, temporal_chunk, reset
            print(
                f"  {label}: operational sequence {sequence_index}/{len(rows)} ",
                f"chunks={token_count // window}",
                flush=True,
            )
            del tokens, logits, chunked_depth, captured_depth
    depth_result = depth_accumulator.finish()
    temporal_result = temporal_accumulator.finish()
    operational_depth_result = operational_depth_accumulator.finish()
    operational_temporal_result = operational_temporal_accumulator.finish()
    operational_depth_chunk_results = [accumulator.finish() for accumulator in operational_depth_by_chunk]
    operational_temporal_chunk_results = [accumulator.finish() for accumulator in operational_temporal_by_chunk]
    del original_run_chunk_stack, capture_run_chunk_stack
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "step": int(blob.get("step", 0) or 0),
        "depth": depth_result["report"],
        "temporal": temporal_result["report"],
        "operational": {
            "window": window,
            "chunks": (token_count // window) * len(rows),
            "depth": operational_depth_result["report"],
            "temporal": operational_temporal_result["report"],
            "by_chunk": [
                {
                    "chunk_index": chunk_index,
                    "start": chunk_index * window,
                    "end": (chunk_index + 1) * window,
                    "receives_refeed": chunk_index > 0,
                    "depth": operational_depth_chunk_results[chunk_index]["report"],
                    "temporal": operational_temporal_chunk_results[chunk_index]["report"],
                }
                for chunk_index in range(complete_chunks)
            ],
        },
    }, temporal_result, operational_temporal_result, paired


def paired_checkpoint_report(init_result, trained_result, horizons, token_count):
    report = {}
    for horizon in horizons:
        init_selected = init_result["selected_tensors"][str(horizon)]
        trained_selected = trained_result["selected_tensors"][str(horizon)]
        init_rotation = init_selected["rotation"]
        trained_rotation = trained_selected["rotation"]
        relative_rotation = trained_rotation @ init_rotation.transpose(-1, -2)
        relative_angle = rotation_angle(relative_rotation)
        dominant_overlap = (
            init_selected["dominant_input"] * trained_selected["dominant_input"]
        ).sum(dim=-1).square()
        weak_overlap = (
            init_selected["weak_input"] * trained_selected["weak_input"]
        ).sum(dim=-1).square()
        stretch_delta = (
            init_selected["singular_probability"]
            - trained_selected["singular_probability"]
        ).norm(dim=-1)
        report[str(horizon)] = {
            "relative_polar_rotation_angle": tensor_quantiles(relative_angle),
            "dominant_input_frame_overlap_squared": tensor_quantiles(dominant_overlap),
            "weak_input_frame_overlap_squared": tensor_quantiles(weak_overlap),
            "normalized_stretch_delta_l2": tensor_quantiles(stretch_delta),
        }
    init_failure = init_result["first_failure"].float()
    trained_failure = trained_result["first_failure"].float()
    clipped_init = init_failure.clamp_max(token_count + 1)
    clipped_trained = trained_failure.clamp_max(token_count + 1)
    delta = clipped_trained - clipped_init
    centered_init = clipped_init.flatten() - clipped_init.float().mean()
    centered_trained = clipped_trained.flatten() - clipped_trained.float().mean()
    correlation = float(
        (centered_init * centered_trained).mean()
        / (
            centered_init.square().mean().sqrt()
            * centered_trained.square().mean().sqrt()
        ).clamp_min(1e-12)
    )
    return {
        "selected_horizons": report,
        "first_failure_delta_trained_minus_init": tensor_quantiles(delta),
        "fraction_streams_shortened": float((delta < 0).float().mean()),
        "fraction_streams_lengthened": float((delta > 0).float().mean()),
        "first_failure_correlation": correlation,
    }


def print_summary(report, horizons):
    print("\n=== Rotation-aware summary ===")
    for label in ("init", "trained"):
        section = report[label]["temporal"]
        print(
            f"{label}: temporal stable horizon={section['stable_horizon']} "
            f"depth stable horizon={report[label]['depth']['stable_horizon']}"
        )
        for horizon in horizons:
            metrics = section["selected_horizons"][str(horizon)]
            singular_values = "/".join(
                f"{value:.3f}" for value in metrics["normalized_singular_values_mean"]
            )
            print(
                f"  h={horizon:4d} pass={metrics['population_pass']:.4f} "
                f"cond={metrics['condition']['mean']:.3f} "
                f"rank={metrics['effective_rank']['mean']:.3f} "
                f"rot={math.degrees(metrics['rotation_angle']['mean']):6.1f}deg "
                f"path={metrics['rotation_path_length']['mean']:.1f}rad "
                f"sv={singular_values}"
            )
        failure = section["per_stream_first_failure"]
        dynamics = section["singular_frame_dynamics"]
        print(
            f"  first-failure median={failure['all']['p50']:.1f} "
            f"p10={failure['all']['p10']:.1f} p90={failure['all']['p90']:.1f} "
            f"failed_by_W={failure['fraction_failed_by_window']:.4f}"
        )
        print(
            f"  weak-frame lag1 overlap={dynamics['weak_step_overlap']['mean']:.4f} "
            f"dominant-frame lag1 overlap={dynamics['dominant_step_overlap']['mean']:.4f}"
        )
        operational = report[label]["operational"]["temporal"]
        boundary = operational["selected_horizons"][str(report["window"])]
        operational_failure = operational["per_stream_first_failure"]
        print(
            f"  operational W={report['window']}: pass={boundary['population_pass']:.4f} "
            f"cond={boundary['condition']['mean']:.3f} "
            f"rank={boundary['effective_rank']['mean']:.3f} "
            f"failed_by_W={operational_failure['fraction_failed_by_window']:.4f}"
        )
        chunk_line = []
        for chunk in report[label]["operational"]["by_chunk"]:
            chunk_boundary = chunk["temporal"]["selected_horizons"][str(report["window"])]
            chunk_line.append(
                f"c{chunk['chunk_index']}:{chunk_boundary['population_pass']:.3f}/"
                f"{chunk_boundary['condition']['mean']:.2f}/"
                f"{chunk_boundary['effective_rank']['mean']:.2f}"
            )
        print("  operational chunks pass/cond/rank: " + " ".join(chunk_line))
    print("\n=== Learned change ===")
    paired = report["trained_vs_init"]
    print(
        f"streams shortened={paired['fraction_streams_shortened']:.4f} "
        f"lengthened={paired['fraction_streams_lengthened']:.4f} "
        f"horizon correlation={paired['first_failure_correlation']:.4f}"
    )
    for horizon in horizons:
        metrics = paired["selected_horizons"][str(horizon)]
        print(
            f"  h={horizon:4d} relative rotation="
            f"{math.degrees(metrics['relative_polar_rotation_angle']['mean']):6.1f}deg "
            f"dominant overlap={metrics['dominant_input_frame_overlap_squared']['mean']:.3f} "
            f"weak overlap={metrics['weak_input_frame_overlap_squared']['mean']:.3f} "
            f"stretch delta={metrics['normalized_stretch_delta_l2']['mean']:.3f}"
        )


def print_single_summary(label, checkpoint_report, window, horizons):
    section = checkpoint_report["temporal"]
    print("\n=== Rotation-aware summary ===")
    print(
        f"{label}: temporal stable horizon={section['stable_horizon']} "
        f"depth stable horizon={checkpoint_report['depth']['stable_horizon']}"
    )
    for horizon in horizons:
        metrics = section["selected_horizons"][str(horizon)]
        singular_values = "/".join(
            f"{value:.3f}" for value in metrics["normalized_singular_values_mean"]
        )
        print(
            f"  h={horizon:4d} pass={metrics['population_pass']:.4f} "
            f"cond={metrics['condition']['mean']:.3f} "
            f"rank={metrics['effective_rank']['mean']:.3f} "
            f"rot={math.degrees(metrics['rotation_angle']['mean']):6.1f}deg "
            f"path={metrics['rotation_path_length']['mean']:.1f}rad "
            f"sv={singular_values}"
        )
    failure = section["per_stream_first_failure"]
    dynamics = section["singular_frame_dynamics"]
    print(
        f"  first-failure median={failure['all']['p50']:.1f} "
        f"p10={failure['all']['p10']:.1f} p90={failure['all']['p90']:.1f} "
        f"failed_by_W={failure['fraction_failed_by_window']:.4f}"
    )
    print(
        f"  weak-frame lag1 overlap={dynamics['weak_step_overlap']['mean']:.4f} "
        f"dominant-frame lag1 overlap={dynamics['dominant_step_overlap']['mean']:.4f}"
    )
    operational = checkpoint_report["operational"]["temporal"]
    boundary = operational["selected_horizons"][str(window)]
    operational_failure = operational["per_stream_first_failure"]
    print(
        f"  operational W={window}: pass={boundary['population_pass']:.4f} "
        f"cond={boundary['condition']['mean']:.3f} "
        f"rank={boundary['effective_rank']['mean']:.3f} "
        f"failed_by_W={operational_failure['fraction_failed_by_window']:.4f}"
    )
    chunk_line = []
    for chunk in checkpoint_report["operational"]["by_chunk"]:
        chunk_boundary = chunk["temporal"]["selected_horizons"][str(window)]
        chunk_line.append(
            f"c{chunk['chunk_index']}:{chunk_boundary['population_pass']:.3f}/"
            f"{chunk_boundary['condition']['mean']:.2f}/"
            f"{chunk_boundary['effective_rank']['mean']:.2f}"
        )
    print("  operational chunks pass/cond/rank: " + " ".join(chunk_line))


def main():
    args = parse_args()
    trained_blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    init_blob = (
        torch.load(args.init, map_location="cpu", weights_only=False)
        if args.init is not None
        else None
    )
    cfg = lf.Config.from_checkpoint_dict(dict(trained_blob["cfg"]))
    if args.tokenizer is not None:
        cfg.tokenizer = args.tokenizer
    requested_tokens = int(args.tokens)
    if requested_tokens <= 0:
        raise ValueError("tokens must be positive")
    cfg.seq_len = requested_tokens
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
    if device.type != "cuda":
        raise ValueError("this audit currently requires CUDA autocast")
    token_count = requested_tokens
    raw_token_count = min(int(args.raw_tokens), token_count)
    window = int(cfg.tria_temporal_window)
    if token_count % window:
        raise ValueError("tokens must be divisible by the temporal window")
    horizons = sorted({
        horizon
        for horizon in (
            1,
            8,
            16,
            32,
            64,
            96,
            128,
            window,
            2 * window,
            3 * window,
            4 * window,
            token_count,
        )
        if 1 <= horizon <= token_count
    })
    raw_horizons = [horizon for horizon in horizons if horizon <= raw_token_count]
    rope_attention_factor = (
        float(cfg.rope_attention_factor)
        if cfg.rope_attention_factor is not None
        else float(lf._yarn_get_mscale(cfg.rope_factor))
    )
    rope_report = {
        "original_seq_len": int(cfg.rope_original_seq_len),
        "factor": float(cfg.rope_factor),
        "attention_factor": rope_attention_factor,
        "cache_seq_len": token_count,
    }
    rows = build_rows(cfg, args.data, token_count, args.sequences)
    print(
        f"T={token_count} W={window} alpha={cfg.tria_carrier_alpha:g} "
        f"sequences={len(rows)} horizons={horizons} "
        f"rope_original={cfg.rope_original_seq_len} rope_factor={cfg.rope_factor:g}"
    )
    if init_blob is None:
        checkpoint_report, _, _, _ = analyze_checkpoint(
            "checkpoint",
            args.checkpoint,
            trained_blob,
            rows,
            cfg,
            device,
            token_count,
            raw_token_count,
            horizons,
            args.max_condition,
            args.min_effective_rank,
            args.population_pass,
        )
        report = {
            "checkpoint": args.checkpoint,
            "dataset": args.data,
            "native_alpha": float(cfg.tria_carrier_alpha),
            "polarm_beta": float(cfg.tria_polarm_beta),
            "window": window,
            "tokens_analyzed": token_count,
            "raw_tokens_analyzed": raw_token_count,
            "sequences": len(rows),
            "rope": rope_report,
            "thresholds": {
                "max_condition": args.max_condition,
                "min_effective_rank": args.min_effective_rank,
                "population_pass": args.population_pass,
            },
            "analysis": checkpoint_report,
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print_single_summary("checkpoint", checkpoint_report, window, raw_horizons)
        print(f"\nJSON saved to {output_path}")
        return
    init_report, init_internal, init_operational_internal, _ = analyze_checkpoint(
        "init",
        args.init,
        init_blob,
        rows,
        cfg,
        device,
        token_count,
        raw_token_count,
        horizons,
        args.max_condition,
        args.min_effective_rank,
        args.population_pass,
    )
    trained_report, trained_internal, trained_operational_internal, _ = analyze_checkpoint(
        "trained",
        args.checkpoint,
        trained_blob,
        rows,
        cfg,
        device,
        token_count,
        raw_token_count,
        horizons,
        args.max_condition,
        args.min_effective_rank,
        args.population_pass,
    )
    report = {
        "checkpoint": args.checkpoint,
        "init_checkpoint": args.init,
        "dataset": args.data,
        "native_alpha": float(cfg.tria_carrier_alpha),
        "polarm_beta": float(cfg.tria_polarm_beta),
        "window": window,
        "tokens_analyzed": token_count,
        "raw_tokens_analyzed": raw_token_count,
        "sequences": len(rows),
        "rope": rope_report,
        "thresholds": {
            "max_condition": args.max_condition,
            "min_effective_rank": args.min_effective_rank,
            "population_pass": args.population_pass,
        },
        "init": init_report,
        "trained": trained_report,
        "trained_vs_init": paired_checkpoint_report(
            init_internal,
            trained_internal,
            raw_horizons,
            raw_token_count,
        ),
        "trained_vs_init_operational": paired_checkpoint_report(
            init_operational_internal,
            trained_operational_internal,
            [horizon for horizon in raw_horizons if horizon <= window],
            window,
        ),
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report, raw_horizons)
    print(f"\nJSON saved to {output_path}")


if __name__ == "__main__":
    main()
