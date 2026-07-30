import json
from pathlib import Path

import yaml


MATRIX_PATH = Path(__file__).with_name("attention_matrix.yaml")


def attention_cases():
    with MATRIX_PATH.open(encoding="utf-8") as handle:
        matrix = yaml.safe_load(handle) or {}
    layer_modes = matrix.get("layer_modes")
    token_modes = matrix.get("token_modes")
    checkpointing = matrix.get("checkpointing")
    if not isinstance(layer_modes, dict) or not layer_modes:
        raise ValueError(f"{MATRIX_PATH}: layer_modes must be a non-empty mapping")
    if not isinstance(token_modes, dict) or not token_modes:
        raise ValueError(f"{MATRIX_PATH}: token_modes must be a non-empty mapping")
    if checkpointing != [False, True]:
        raise ValueError(f"{MATRIX_PATH}: checkpointing must be [false, true]")
    cases = []
    signatures = set()
    for layer_name, layers in layer_modes.items():
        if layers is not None and (
            not isinstance(layers, list)
            or not layers
            or any(not isinstance(layer, int) for layer in layers)
        ):
            raise ValueError(f"{MATRIX_PATH}: invalid layer mode {layer_name!r}")
        for token_name, token in token_modes.items():
            if not isinstance(token, dict):
                raise ValueError(f"{MATRIX_PATH}: invalid token mode {token_name!r}")
            stride = int(token.get("attn_token_stride", 0))
            schedule = str(token.get("attn_token_schedule", ""))
            if stride <= 0 or schedule not in ("shared", "staggered"):
                raise ValueError(f"{MATRIX_PATH}: invalid token mode {token_name!r}")
            if stride == 1 and schedule != "shared":
                raise ValueError(
                    f"{MATRIX_PATH}: stride-1 mode must use shared schedule")
            for checkpoint in checkpointing:
                overrides = {
                    "attn_layers": layers,
                    "attn_token_stride": stride,
                    "attn_token_schedule": schedule,
                    "grad_checkpointing": checkpoint,
                }
                signature = json.dumps(overrides, sort_keys=True)
                if signature in signatures:
                    raise ValueError(f"{MATRIX_PATH}: duplicate case {signature}")
                signatures.add(signature)
                cases.append({
                    "name": (
                        f"{layer_name}_{token_name}_"
                        f"ckpt_{'on' if checkpoint else 'off'}"
                    ),
                    "overrides": overrides,
                })
    return cases
