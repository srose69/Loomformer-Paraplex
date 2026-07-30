#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import torch

FORMAT_NAME = "loom.aio"
FORMAT_VERSION = 1
DEFAULT_MIN_QUANT_ELEMENTS = 65536

_DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float64: "fp64",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_CRITICAL_MARKERS = (
    "layernorm",
    "layer_norm",
    ".ln_",
    ".norm",
    "beta_anchor",
    "raw_gamma",
    "raw_alpha",
    "logit_scale",
    "gate_selector",
    "identity_gate",
    "tria_agg.reader",
    "tria_agg.pool",
)


def _dtype_name(dtype: torch.dtype) -> str:
    return _DTYPE_NAMES.get(dtype, str(dtype).replace("torch.", ""))


def _quant_dtype(name: str) -> Optional[torch.dtype]:
    key = str(name).strip().lower()
    if key in ("none", "off", "keep"):
        return None
    if key in ("fp32", "float32"):
        return torch.float32
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"unsupported --quant {name!r}; use none, fp32, bf16, or fp16")


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _storage_identity(tensor: torch.Tensor) -> Optional[Tuple[Any, ...]]:
    if tensor.device.type != "cpu" or tensor.layout != torch.strided:
        return None
    try:
        storage = tensor.untyped_storage()
        return (
            int(storage.data_ptr()),
            int(tensor.storage_offset()),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
        )
    except Exception:
        return None


def _keep_fp32(name: str, tensor: torch.Tensor, min_quant_elements: int) -> Tuple[bool, str]:
    lname = name.lower()
    if not tensor.is_floating_point():
        return True, "non-floating"
    if tensor.dtype == torch.float64:
        return True, "fp64"
    if tensor.ndim < 2:
        return True, "scalar/vector"
    if tensor.numel() < min_quant_elements:
        return True, "small"
    if lname.endswith(".bias") or lname == "bias":
        return True, "bias"
    if any(marker in lname for marker in _CRITICAL_MARKERS):
        return True, "critical"
    return False, "matrix"


def _convert_state_dict(
    state: Mapping[str, Any],
    target_dtype: Optional[torch.dtype],
    min_quant_elements: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    converted: Dict[str, Any] = {}
    aliases: Dict[Tuple[Any, ...], torch.Tensor] = {}
    decisions = Counter()
    source_bytes = 0
    packed_bytes = 0
    changed_tensors = 0
    kept_tensors = 0

    for name, value in state.items():
        if not torch.is_tensor(value):
            converted[name] = value
            continue

        tensor = value.detach().cpu()
        source_bytes += _tensor_bytes(tensor)
        identity = _storage_identity(tensor)
        if identity is not None and identity in aliases:
            out = aliases[identity]
            converted[name] = out
            packed_bytes += _tensor_bytes(out)
            decisions["alias"] += 1
            continue

        keep, reason = _keep_fp32(name, tensor, min_quant_elements)
        if target_dtype is None:
            out = tensor.contiguous()
            reason = "unchanged"
        elif keep:
            out = tensor.float().contiguous() if tensor.is_floating_point() and tensor.dtype != torch.float64 else tensor.contiguous()
            kept_tensors += 1
        else:
            out = tensor.to(dtype=target_dtype).contiguous()
            changed_tensors += int(out.dtype != tensor.dtype)

        converted[name] = out
        packed_bytes += _tensor_bytes(out)
        decisions[reason] += 1
        if identity is not None:
            aliases[identity] = out

    report = {
        "target_dtype": "none" if target_dtype is None else _dtype_name(target_dtype),
        "min_quant_elements": int(min_quant_elements),
        "source_bytes": int(source_bytes),
        "packed_tensor_bytes": int(packed_bytes),
        "changed_tensors": int(changed_tensors),
        "kept_fp32_tensors": int(kept_tensors),
        "decisions": dict(sorted(decisions.items())),
    }
    return converted, report


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_checkpoint(path: str) -> MutableMapping[str, Any]:
    if path.lower().endswith(".safetensors"):
        return _load_safetensors_checkpoint(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, MutableMapping):
        raise TypeError("checkpoint must be a mapping")
    if "model" not in blob or not isinstance(blob["model"], Mapping):
        raise ValueError("checkpoint must contain a model state dict under key 'model'")
    if "cfg" not in blob or not isinstance(blob["cfg"], Mapping):
        raise ValueError("checkpoint must contain configuration under key 'cfg'")
    return blob


def _load_safetensors_checkpoint(path: str) -> Dict[str, Any]:
    from safetensors.torch import load_file as _load_safetensors

    base = os.path.splitext(path)[0]
    config_path = base + ".json"
    if not os.path.isfile(config_path):
        config_path = os.path.join(os.path.dirname(path), "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"safetensors checkpoint needs a sibling config JSON "
            f"(tried {base + '.json'} and config.json in the same directory)"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "cfg" not in meta or not isinstance(meta["cfg"], Mapping):
        raise ValueError("config JSON must contain 'cfg' mapping")
    model_state = _load_safetensors(path)
    blob: Dict[str, Any] = {
        "model": model_state,
        "cfg": meta["cfg"],
        "model_kind": meta.get("model_kind", "loomformer"),
        "ffn_type": meta.get("ffn_type", "paraplex"),
        "ablation": meta.get("ablation", False),
    }
    for key in ("step", "tokens_seen", "train_loss", "eval_loss"):
        if key in meta:
            blob[key] = meta[key]
    return blob


def _inference_checkpoint(blob: Mapping[str, Any], model_state: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "model_kind": blob.get("model_kind", "loomformer"),
        "ffn_type": blob.get("ffn_type", "paraplex"),
        "cfg": dict(blob["cfg"]),
        "ablation": bool(blob.get("ablation", False)),
        "model": dict(model_state),
    }
    for key in ("step", "tokens_seen", "train_loss", "eval_loss"):
        value = blob.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if key in blob:
                out[key] = value
    return out


def build_package(
    model_path: str,
    tokenizer_path: str,
    template_path: str,
    output_path: str,
    quant: str,
    min_quant_elements: int,
) -> Dict[str, Any]:
    if min_quant_elements < 1:
        raise ValueError("--min-quant-elements must be positive")

    blob = _load_checkpoint(model_path)
    target_dtype = _quant_dtype(quant)
    model_state, quant_report = _convert_state_dict(
        blob["model"], target_dtype, min_quant_elements
    )
    checkpoint = _inference_checkpoint(blob, model_state)
    tokenizer = _read_bytes(tokenizer_path)
    template = _read_bytes(template_path)

    cfg = checkpoint["cfg"]
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "created_unix": int(time.time()),
        "model_kind": checkpoint["model_kind"],
        "ffn_type": checkpoint["ffn_type"],
        "ablation": checkpoint["ablation"],
        "quantization": quant_report,
        "model": {
            "source_name": os.path.basename(model_path),
            "state_keys": len(model_state),
            "parameters": int(sum(v.numel() for v in model_state.values() if torch.is_tensor(v))),
            "step": checkpoint.get("step"),
            "config": {
                key: cfg.get(key)
                for key in (
                    "vocab", "model_dim", "n_q_heads", "n_kv_heads", "head_dim",
                    "hidden", "layers", "seq_len", "tied_embeddings", "activation",
                    "phase_sectors", "tria_carry_enabled", "tria_temporal_window",
                    "tria_carrier_alpha", "tria_polarm_beta",
                )
                if key in cfg
            },
        },
        "assets": {
            "tokenizer": {
                "name": os.path.basename(tokenizer_path),
                "bytes": len(tokenizer),
                "sha256": _sha256(tokenizer),
            },
            "chat_template": {
                "name": os.path.basename(template_path),
                "bytes": len(template),
                "sha256": _sha256(template),
            },
        },
    }

    package = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "tokenizer_json": tokenizer,
        "chat_template_jinja": template,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    torch.save(package, temp)
    os.replace(temp, output)
    manifest["archive_bytes"] = output.stat().st_size
    return manifest


def load_aio(path: str, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    package = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(package, dict):
        raise TypeError("AIO package is not a mapping")
    if package.get("format") != FORMAT_NAME:
        raise ValueError(f"not a {FORMAT_NAME} package")
    if int(package.get("version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported AIO version {package.get('version')!r}")
    return package


def inspect_package(path: str, show_keys: bool = False) -> None:
    package = load_aio(path)
    manifest = package["manifest"]
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if show_keys:
        state = package["checkpoint"]["model"]
        for name, value in state.items():
            if torch.is_tensor(value):
                print(f"{name:72s} {str(tuple(value.shape)):24s} {_dtype_name(value.dtype)}")
            else:
                print(f"{name:72s} {type(value).__name__}")


def extract_package(path: str, directory: str) -> None:
    package = load_aio(path)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(package["checkpoint"], out / "model.pt")
    (out / "tokenizer.json").write_bytes(package["tokenizer_json"])
    (out / "chat_template.jinja").write_bytes(package["chat_template_jinja"])
    (out / "manifest.json").write_text(
        json.dumps(package["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_safetensors(
    model_path: str,
    output_path: str,
    quant: str,
    min_quant_elements: int,
) -> Dict[str, Any]:
    if min_quant_elements < 1:
        raise ValueError("--min-quant-elements must be positive")

    blob = _load_checkpoint(model_path)
    target_dtype = _quant_dtype(quant)
    model_state, quant_report = _convert_state_dict(
        blob["model"], target_dtype, min_quant_elements
    )

    from safetensors.torch import save_file

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sd = {k: v.contiguous() for k, v in model_state.items()}

    shared = set()
    seen_ptrs: Dict[int, str] = {}
    for k, v in sd.items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            shared.add(seen_ptrs[ptr])
            shared.add(k)
        else:
            seen_ptrs[ptr] = k
    for k in shared:
        sd[k] = sd[k].clone()

    save_file(sd, str(out))

    meta = {
        "cfg": dict(blob["cfg"]),
        "model_kind": blob.get("model_kind", "loomformer"),
        "ffn_type": blob.get("ffn_type", "paraplex"),
        "ablation": bool(blob.get("ablation", False)),
    }
    for key in ("step", "tokens_seen", "train_loss", "eval_loss"):
        if key in blob:
            meta[key] = blob[key]

    config_path = out.with_suffix(".json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    quant_report["archive_bytes"] = out.stat().st_size
    quant_report["config_path"] = str(config_path)
    return quant_report


def _size_text(value: int) -> str:
    n = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{value} B"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loompack.py")
    sub = parser.add_subparsers(dest="command", required=True)

    pack = sub.add_parser("pack")
    pack.add_argument("model", help="LoomFormer .pt or .safetensors checkpoint")
    pack.add_argument("--tokenizer", required=True, help="tokenizer JSON")
    pack.add_argument("--template", required=True, help="chat template Jinja")
    pack.add_argument("-o", "--output", required=True, help="output .aio")
    pack.add_argument("--quant", default="none", help="none, fp32, bf16, or fp16")
    pack.add_argument(
        "--min-quant-elements",
        type=int,
        default=DEFAULT_MIN_QUANT_ELEMENTS,
        help=f"keep smaller tensors in fp32 (default: {DEFAULT_MIN_QUANT_ELEMENTS})",
    )

    inspect = sub.add_parser("inspect")
    inspect.add_argument("archive")
    inspect.add_argument("--keys", action="store_true")

    extract = sub.add_parser("extract")
    extract.add_argument("archive")
    extract.add_argument("-d", "--directory", required=True)

    export = sub.add_parser("export-safetensors")
    export.add_argument("model", help="LoomFormer .pt or .safetensors checkpoint")
    export.add_argument("-o", "--output", required=True, help="output .safetensors")
    export.add_argument("--quant", default="none", help="none, fp32, bf16, or fp16")
    export.add_argument(
        "--min-quant-elements",
        type=int,
        default=DEFAULT_MIN_QUANT_ELEMENTS,
        help=f"keep smaller tensors in fp32 (default: {DEFAULT_MIN_QUANT_ELEMENTS})",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "pack":
        manifest = build_package(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            template_path=args.template,
            output_path=args.output,
            quant=args.quant,
            min_quant_elements=args.min_quant_elements,
        )
        q = manifest["quantization"]
        print(
            f"[loompack] {args.output}: {manifest['model']['parameters']:,} params, "
            f"quant={q['target_dtype']}, tensors={q['changed_tensors']} converted, "
            f"archive={_size_text(manifest['archive_bytes'])}"
        )
        print(
            f"[loompack] tensor bytes: {_size_text(q['source_bytes'])} -> "
            f"{_size_text(q['packed_tensor_bytes'])}"
        )
        return 0
    if args.command == "inspect":
        inspect_package(args.archive, show_keys=args.keys)
        return 0
    if args.command == "extract":
        extract_package(args.archive, args.directory)
        print(f"[loompack] extracted to {args.directory}")
        return 0
    if args.command == "export-safetensors":
        report = export_safetensors(
            model_path=args.model,
            output_path=args.output,
            quant=args.quant,
            min_quant_elements=args.min_quant_elements,
        )
        print(
            f"[loompack] {args.output}: quant={report['target_dtype']}, "
            f"tensors={report['changed_tensors']} converted, "
            f"archive={_size_text(report['archive_bytes'])}"
        )
        print(
            f"[loompack] tensor bytes: {_size_text(report['source_bytes'])} -> "
            f"{_size_text(report['packed_tensor_bytes'])}"
        )
        print(f"[loompack] config: {report['config_path']}")
        return 0
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"loompack: {exc}", file=sys.stderr)
        raise SystemExit(1)
