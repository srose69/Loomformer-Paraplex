#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loomcloner.py -- transplant a pretrained donor transformer's attention/FFN
weights into a freshly-shaped LoomFormer checkpoint ("--rebuild" from the
design discussion this came out of).

Two stages, run separately on purpose (scan is cheap and donor-only; clone
needs the donor's actual weight files and does the real work):

  1. --scan   reads the donor's own config.json, fills a LoomFormer YAML
              1:1 from it (every field resolved, nothing left as a guess),
              ending with `cloned: true` and a per-parameter train/lr
              override list (LoomFormer-side names, so the training script
              can match them directly against model.named_parameters() --
              no donor-name translation needed at train time).

  2. --clone  loads the donor's real weights, remaps names per the chosen
              mappings/*.json, builds a LoomFormer Model from the YAML
              --scan produced, loads everything that has a mapped source
              (strict=False -- Tria/Paraplex-specific parameters have no
              donor equivalent and keep their own identity-anchored init),
              saves a normal LoomFormer checkpoint.

Mapping files (mappings/*.json) describe ONE donor architecture family each
(name translation + config-field translation + train/lr policy) -- kept out
of this file's own code on purpose, so a new donor family is a JSON file, not
a code change.

CLI:
  loomcloner.py --scan  <donor_dir> --mapping mappings/llama.json --out chk.yaml
                [--steps N] [--lr F] [--paraplex-gate-proj] [--final-norm]
  loomcloner.py --clone <config.yaml> --donor <donor_dir> --mapping mappings/llama.json
                --out cloned.pt
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import loomformer as lf


# ============================================================================
# donor introspection (config.json + safetensors header -- no torch needed
# for --scan at all, it never touches the actual weight bytes)
# ============================================================================

def load_donor_config(donor_dir: str) -> Dict[str, Any]:
    path = os.path.join(donor_dir, "config.json")
    with open(path) as f:
        return json.load(f)


def load_mapping(mapping_path: str) -> Dict[str, Any]:
    with open(mapping_path) as f:
        return json.load(f)


def read_safetensors_header(path: str) -> Dict[str, Any]:
    """Reads just the JSON header (key -> {shape, dtype}) without loading any
    tensor data -- this is all --scan needs, and --clone uses it too before
    deciding what to actually read off disk."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def find_safetensors_files(donor_dir: str) -> List[str]:
    single = os.path.join(donor_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    shards = sorted(Path(donor_dir).glob("model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model.safetensors or sharded safetensors files found in {donor_dir}")
    return [str(p) for p in shards]


def detect_mapping(donor_cfg: Dict[str, Any], mappings_dir: str) -> str:
    """Auto-pick a mapping file by donor_cfg['architectures'], used when the
    caller didn't pass --mapping explicitly. Falls back to raising with a
    clear message (never guesses silently)."""
    archs = set(donor_cfg.get("architectures", []))
    for p in sorted(Path(mappings_dir).glob("*.json")):
        m = load_mapping(str(p))
        candidates = set(m.get("detect", {}).get("any_of", []))
        if archs & candidates:
            return str(p)
    raise ValueError(
        f"no mapping in {mappings_dir} declares support for architectures={sorted(archs)}; "
        f"pass --mapping explicitly or add a mappings/*.json for this family"
    )


# ============================================================================
# --scan: donor config.json -> LoomFormer YAML
# ============================================================================

def _resolve_config_fields(donor_cfg: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for donor_key, loom_key in mapping.get("config_map", {}).items():
        if donor_key in donor_cfg:
            out[loom_key] = donor_cfg[donor_key]

    hf = mapping.get("hidden_from")
    if hf:
        num = donor_cfg[hf["numerator"]]
        den = donor_cfg[hf["denominator"]]
        out[hf["target_field"]] = num / den  # NOT rounded -- see mapping's config_notes

    hd = mapping.get("head_dim_from")
    if hd:
        num = donor_cfg[hd["numerator"]]
        den = donor_cfg[hd["denominator"]]
        if num % den != 0:
            raise ValueError(f"head_dim would be non-integer: {num}/{den} -- donor shape assumption violated")
        out["head_dim"] = num // den

    return out


def _hidden_round_up(model_dim: int, hidden_mult: float, n_q_heads: int) -> int:
    raw = model_dim * hidden_mult
    return int(-(-round(raw) // n_q_heads) * n_q_heads)


def build_yaml_text(donor_cfg: Dict[str, Any], mapping: Dict[str, Any], donor_dir: str,
                     tokenizer_path: str, dataset_placeholder: str,
                     steps: int, lr: float, paraplex_gate_proj: bool, final_norm: bool) -> str:
    fields = _resolve_config_fields(donor_cfg, mapping)
    model_dim = int(fields["model_dim"])
    n_q_heads = int(fields["n_q_heads"])
    n_kv_heads = int(fields.get("n_kv_heads", n_q_heads))
    head_dim = int(fields["head_dim"])
    hidden_mult = float(fields["hidden_mult"])
    layers = int(fields["layers"])
    vocab = int(fields["vocab"])
    rope_theta = float(fields.get("rope_theta", 10000.0))
    rope_orig = int(fields.get("rope_original_seq_len", 2048))
    tied = bool(fields.get("tied_embeddings", False))

    hidden = _hidden_round_up(model_dim, hidden_mult, n_q_heads)
    expected_intermediate = donor_cfg.get(mapping.get("hidden_from", {}).get("numerator", ""), None)
    hidden_mismatch_note = ""
    if expected_intermediate is not None and hidden != expected_intermediate:
        # hidden MUST stay divisible by n_q_heads (HIDDEN_PER_Q_HEAD =
        # HIDDEN // N_Q_HEADS feeds the per-head phase/beta_space slicing --
        # this is a real structural requirement, not a style choice, so we
        # can't just take the donor's raw intermediate_size when it doesn't
        # divide evenly, the way an earlier version of this script did).
        # --clone pads the donor's up_proj/gate_proj/down_proj with zero
        # rows/columns to bridge the gap instead -- padding with zeros is a
        # safe no-op (those extra channels start at zero, contribute nothing
        # until trained), unlike truncating, which would discard real,
        # trained donor parameters.
        hidden_mismatch_note = (
            f"\n# NOTE: donor intermediate_size={expected_intermediate} is not divisible by "
            f"n_q_heads={n_q_heads}\n"
            f"# (LoomFormer requires this -- HIDDEN_PER_Q_HEAD=HIDDEN//N_Q_HEADS feeds per-head\n"
            f"# phase slicing). hidden is rounded up to {hidden}; --clone zero-pads the donor's\n"
            f"# up_proj/gate_proj/down_proj by {hidden - expected_intermediate} channel(s) to match.\n"
        )

    donor_name = donor_cfg.get("_name_or_path") or os.path.basename(os.path.normpath(donor_dir))
    lr_overrides = mapping.get("train_lr_overrides", [])
    override_lines = []
    for ov in lr_overrides:
        lit_lr = lr * float(ov.get("lr_scale", 1.0))
        override_lines.append(
            f"  - name: {ov['pattern']}\n"
            f"    train: {'yes' if ov.get('train', True) else 'no'}\n"
            f"    lr: {lit_lr:.2e}"
            + (f"  # {ov['note']}" if ov.get("note") else "")
        )

    text = f"""\
# {donor_name} -- clone
# ===AUTO GENERATED=== by loomcloner.py --scan, from {os.path.join(donor_dir, 'config.json')}
# mapping: {mapping.get('family', '?')}
optimizer: atom
steps: {steps}
lr: {lr:g}
weight_decay: 0.01
grad_clip: 1.55
warmup_steps: {max(1, steps // 30)}
min_lr_frac: 0.1
seed: 1
log_every: 10
eval_every: 50
eval_batches: 4
device: cuda:0
grad_accum_steps: 1
grad_checkpointing: false
graph: false
save_graph: false
amp_dtype: bf16
save_every: 500
runpoints_path: null

seq_len: {rope_orig}
batch_size: 8
prefetch_batches: 256

rope_theta: {rope_theta:g}
rope_original_seq_len: {rope_orig}
rope_factor: 1.0
rope_beta_fast: 32.0
rope_beta_slow: 1.0
rope_attention_factor: null

tria_carry_enabled: true
use_cuda_tria: true
tria_gamma_max: 0.80
tria_raw_gamma_init: 0.25
tria_temporal_enabled: true
tria_temporal_auto: false
tria_temporal_window: 128
tria_temporal_window_min: 32
tria_target_refeeds_per_sequence: 7
tria_min_refeeds_per_sequence: 1
tria_carrier_alpha: 0.0375
tria_carrier_alpha_candidates: [0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075]
tria_temporal_max_condition: 3.0
tria_temporal_min_effective_rank: 2.70
tria_temporal_population_pass_fraction: 0.90
tria_temporal_calib_seeds: 1
tria_temporal_calib_batch: 1
tria_temporal_calib_tokens: 512
tria_temporal_calib_device: auto
tria_temporal_calib_parallel_sweep: 2

# tokenizer: the DONOR's own -- see loomcloner design notes, using LoomFormer's
# own tokenizer instead would make emb.weight/head.weight untransplantable.
tokenizer: {tokenizer_path}
vocab: {vocab}
tied_embeddings: {"true" if tied else "false"}

train_dataset: {dataset_placeholder}   # placeholder -- point this at a real corpus before training
val_dataset: null
auto_val_split_pct: 1.0

dataset_format: arrow
text_field: text
dataset_cache: null

model_dim: {model_dim}

n_q_heads: {n_q_heads}
head_dim: {head_dim}                    # d_model = {n_q_heads} * {head_dim} = {n_q_heads * head_dim}

n_kv_heads: {n_kv_heads}
gqa_group_size: null

hidden: {hidden}                        {"# round_up(model_dim*hidden_mult, n_q_heads) -- see zero-pad note above" if hidden_mismatch_note else "# round_up(model_dim*hidden_mult, n_q_heads)"}
hidden_mult: {hidden_mult:.6f}          # = donor intermediate_size / hidden_size, NOT rounded -- see mapping notes
layers: {layers}
{hidden_mismatch_note}
phase_sectors: head
residual_init: beta
activation: pvpowlu
powlu_m: 3.0
phase_grad_floor: 0.05
phase_grad_mode: secant
attn_impl: sdpa

paraplex_gate_proj: {"true" if paraplex_gate_proj else "false"}   # {"gate_proj transplanted from donor's SwiGLU gate_proj" if paraplex_gate_proj else "donor's gate_proj will be DROPPED -- see mapping notes"}
final_norm: {"true" if final_norm else "false"}   # {"model.norm.weight transplanted to ln_final -- exact RMSNorm match" if final_norm else "donor's model.norm.weight will be DROPPED -- LoomFormer has no destination without this flag"}

use_cuda_beta_space: true
use_cuda_phase_sin: true
use_cuda_pvpowlu: true
use_cuda_depth_attn: true

# ===AUTO GENERATED=== cloned-model bookkeeping. Consumed by the training
# script to set per-parameter requires_grad/lr; LoomFormer-side names (not
# donor names) so they match model.named_parameters() directly. `train_lr`
# entries match by SUFFIX against every 'blocks.{{i}}.<pattern>' or exact
# global name (emb.weight, head.weight) -- one entry covers all layers.
cloned: true
cloned_from: {donor_name}
cloned_mapping: {mapping.get('family', '?')}
train_lr:
{chr(10).join(override_lines)}
"""
    return text


# ============================================================================
# --clone: actually load + remap + transplant the donor's real weights
# ============================================================================

def _load_donor_tensors(donor_dir: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    tensors: Dict[str, torch.Tensor] = {}
    for shard in find_safetensors_files(donor_dir):
        tensors.update(load_file(shard))
    return tensors


def _pad_hidden_dim(tensor: torch.Tensor, target_hidden: int, axis: int) -> torch.Tensor:
    """Zero-pads `tensor` along `axis` up to target_hidden -- never truncates
    (that would discard real, trained donor parameters). A no-op if the
    donor's width already matches."""
    current = tensor.shape[axis]
    if current == target_hidden:
        return tensor
    if current > target_hidden:
        raise ValueError(
            f"donor hidden width {current} > target {target_hidden} on axis {axis} -- "
            f"this function only pads up, never truncates; check the --scan-computed 'hidden' value"
        )
    pad_shape = list(tensor.shape)
    pad_shape[axis] = target_hidden - current
    zeros = torch.zeros(pad_shape, dtype=tensor.dtype)
    return torch.cat([tensor, zeros], dim=axis)


# LoomFormer key -> which axis carries the HIDDEN dimension, for padding.
_FFN_HIDDEN_AXIS = {
    "ffn.w1_real.weight": 0,   # [HIDDEN, N]
    "ffn.gate_proj.weight": 0,  # [HIDDEN, N]
    "ffn.w2.weight": 1,        # [N, HIDDEN]
}


def remap_donor_to_loomformer(donor_tensors: Dict[str, torch.Tensor], mapping: Dict[str, Any],
                               n_layers: int, cfg: "lf.Config") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    dropped: List[str] = []
    qkv_parts: Dict[int, Dict[str, torch.Tensor]] = {}
    target_hidden = int(cfg.hidden) if cfg.hidden else None

    def donor_key(layer_prefix: str, suffix: str) -> Optional[str]:
        # Llama-family layout: model.layers.{i}.<suffix>
        return f"model.layers.{layer_prefix}.{suffix}"

    for i in range(n_layers):
        for suffix, rule in mapping.get("per_layer", {}).items():
            dkey = donor_key(i, suffix)
            if dkey not in donor_tensors:
                continue
            req = rule.get("requires_config")
            if req and not all(bool(getattr(cfg, k, False)) == v for k, v in req.items()):
                dropped.append(dkey)
                continue
            tensor = donor_tensors[dkey]
            op = rule["op"]
            if op == "copy":
                to = rule["to"]
                loom_key = f"blocks.{i}.{to}"
                if to in _FFN_HIDDEN_AXIS and target_hidden is not None:
                    tensor = _pad_hidden_dim(tensor, target_hidden, _FFN_HIDDEN_AXIS[to])
                out[loom_key] = tensor
            elif op == "qkv_concat":
                qkv_parts.setdefault(i, {})[rule["part"]] = tensor
            elif op == "drop":
                dropped.append(dkey)
            else:
                raise ValueError(f"unknown op {op!r} for {dkey}")

    for i, parts in qkv_parts.items():
        if not all(p in parts for p in ("q", "k", "v")):
            raise ValueError(f"layer {i}: incomplete qkv parts collected: {sorted(parts)}")
        out[f"blocks.{i}.attn.qkv_weight"] = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)

    for dkey, rule in mapping.get("global", {}).items():
        if dkey not in donor_tensors:
            if not rule.get("optional", False) and rule.get("op") != "drop":
                print(f"[loomcloner] WARNING: global key {dkey!r} not found in donor, and not marked optional")
            continue
        req = rule.get("requires_config")
        if req and not all(bool(getattr(cfg, k, False)) == v for k, v in req.items()):
            dropped.append(dkey)
            continue
        if rule["op"] == "drop":
            dropped.append(dkey)
            continue
        out[rule["to"]] = donor_tensors[dkey]

    if dropped:
        print(f"[loomcloner] dropped {len(dropped)} donor tensor(s) with no LoomFormer destination:")
        for d in dropped[:10]:
            print(f"  - {d}")
        if len(dropped) > 10:
            print(f"  ... and {len(dropped) - 10} more")

    return out


def _upsert_resume_field(yaml_path: str, resume_value: str) -> None:
    """Text-level edit, not yaml.dump -- round-tripping through PyYAML would
    strip every comment in the --scan-generated file. Replaces an existing
    top-level `resume:` line if present, otherwise inserts one right before
    the first real config line (`optimizer:`, which --scan always emits)."""
    with open(yaml_path, encoding="utf-8") as f:
        lines = f.readlines()
    new_line = f"resume: {resume_value}\n"
    for i, line in enumerate(lines):
        if line.startswith("resume:"):
            lines[i] = new_line
            break
    else:
        for i, line in enumerate(lines):
            if line.startswith("optimizer:"):
                lines.insert(i, new_line)
                break
        else:
            lines.append(new_line)  # no 'optimizer:' anchor found -- just append, still correct
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def clone_checkpoint(config_path: str, donor_dir: str, mapping_path: str, out_path: str) -> None:
    with open(config_path) as f:
        import yaml
        cfg_dict = yaml.safe_load(f)
    cfg = lf.Config(**{k: v for k, v in cfg_dict.items()
                        if k not in ("cloned", "cloned_from", "cloned_mapping", "train_lr")})
    lf.apply_config(cfg)

    mapping = load_mapping(mapping_path)
    donor_tensors = _load_donor_tensors(donor_dir)
    donor_dtype = next(iter(donor_tensors.values())).dtype
    remapped = remap_donor_to_loomformer(donor_tensors, mapping, int(cfg.layers), cfg)
    del donor_tensors
    import gc
    gc.collect()

    # Build directly in the donor's own dtype (bf16 here, not fp32) -- there's
    # no information to gain from upcasting a transplant, and on a memory-
    # constrained box this alone can be the difference between fitting and
    # not: fp32 params are 2x the bytes of bf16 for no benefit when every
    # transplanted tensor is bf16 to begin with. torch.set_default_dtype
    # makes Model()'s own nn.Linear/nn.Parameter construction allocate in
    # this dtype directly, rather than building fp32 and casting after.
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(donor_dtype)
    try:
        model = lf.Model()
    finally:
        torch.set_default_dtype(prev_default)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[loomcloner] built model in {donor_dtype} (matching donor -- not upcast to fp32)")
    print(f"[loomcloner] loaded {len(remapped) - len(unexpected)} tensors from donor, "
          f"{len(missing)} LoomFormer-only params kept at their own init, "
          f"{len(unexpected)} unexpected (ignored)")
    if unexpected:
        print(f"[loomcloner] unexpected keys (not in Model, check the mapping): {unexpected[:10]}")

    torch.save(
        {"cfg": cfg_dict, "model_kind": "loomformer", "ffn_type": "paraplex",
         "ablation": False, "model": model.state_dict(), "step": 0, "cloned": True,
         "cloned_from": cfg_dict.get("cloned_from"), "cloned_mapping": cfg_dict.get("cloned_mapping")},
        out_path,
    )
    print(f"[loomcloner] saved -> {out_path}")

    _upsert_resume_field(config_path, out_path)
    print(f"[loomcloner] {config_path} updated with resume: {out_path} "
          f"-- `--train --config {config_path}` alone now resumes it, no --resume needed")


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="loomcloner: donor-transplant tool for LoomFormer")
    ap.add_argument("--scan", type=str, default=None, metavar="DONOR_DIR")
    ap.add_argument("--clone", type=str, default=None, metavar="CONFIG_YAML")
    ap.add_argument("--donor", type=str, default=None, help="donor dir (required with --clone)")
    ap.add_argument("--mapping", type=str, default=None, help="mappings/*.json; auto-detected if omitted")
    ap.add_argument("--mappings-dir", type=str, default=os.path.join(os.path.dirname(__file__), "mappings"))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--paraplex-gate-proj", action="store_true")
    ap.add_argument("--final-norm", action="store_true", help="transplant donor's model.norm.weight onto ln_final (requires loomformer.py's final_norm support)")
    ap.add_argument("--dataset", type=str, default="./datasets/REPLACE_ME")
    args = ap.parse_args()

    if bool(args.scan) == bool(args.clone):
        raise SystemExit("pass exactly one of --scan DONOR_DIR or --clone CONFIG_YAML")

    if args.scan:
        donor_cfg = load_donor_config(args.scan)
        mapping_path = args.mapping or detect_mapping(donor_cfg, args.mappings_dir)
        mapping = load_mapping(mapping_path)
        tok_candidates = ["tokenizer.json"]
        tok_path = next((os.path.join(args.scan, t) for t in tok_candidates
                          if os.path.exists(os.path.join(args.scan, t))), os.path.join(args.scan, "tokenizer.json"))
        text = build_yaml_text(donor_cfg, mapping, args.scan, tok_path, args.dataset,
                                args.steps, args.lr, args.paraplex_gate_proj, args.final_norm)
        with open(args.out, "w") as f:
            f.write(text)
        print(f"[loomcloner] wrote {args.out} (mapping={mapping.get('family')})")
        return

    if not args.donor:
        raise SystemExit("--clone requires --donor DONOR_DIR")
    donor_cfg = load_donor_config(args.donor)
    mapping_path = args.mapping or detect_mapping(donor_cfg, args.mappings_dir)
    clone_checkpoint(args.clone, args.donor, mapping_path, args.out)


if __name__ == "__main__":
    main()
