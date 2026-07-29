#!/usr/bin/env python3
"""Self-contained LoomFormer installation/integration validation.

The runner never reads project checkpoints or datasets. It creates a temporary
tokenizer plus tiny PT and SFT corpora, runs the focused unittest suite, then
executes real optimizer steps through the same CLI used in production.

Coverage:
  * config parsing and focused algebra/mask/OTF unit tests;
  * all fused CUDA extension modules (when CUDA is visible);
  * per-GPU forward/backward parity across manual, SDPA, recompute and
    FlashAttention/Transformer Engine varlen paths;
  * PT bin train, checkpoint, runpoint, resume, eval and inference;
  * PT Parquet/OTF train and destructive automatic validation split;
  * SFT Parquet/OTF packing, assistant-only loss, PT initialization,
    activation checkpointing, checkpoint and resume;
  * torch.compile + custom-op graph on supported modern CUDA;
  * PT and SFT DDP on every visible GPU when at least two are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
PT_TEMPLATE = TESTS / "test_pt.yaml"
SFT_TEMPLATE = TESTS / "test_sft.yaml"


def _banner(label: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n[matrix] {label}\n{bar}", flush=True)


def _run(label: str, argv: Iterable[object], *, timeout: int = 1800) -> None:
    command = [str(item) for item in argv]
    _banner(label)
    print("[matrix] $ " + " ".join(command), flush=True)
    started = time.monotonic()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("NO_COLOR", "1")
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode} after {elapsed:.1f}s"
        )
    print(f"[matrix] PASS {label} ({elapsed:.1f}s)", flush=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a YAML mapping")
    return value


def _write_config(
    template: Path,
    output: Path,
    overrides: Dict[str, Any],
) -> Path:
    config = _load_yaml(template)
    config.update(overrides)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return output


def _make_tokenizer(work: Path) -> tuple[Path, int]:
    raw = work / "tokenizer_corpus"
    raw.mkdir()
    text = (
        "LoomFormer synthetic corpus. The assistant answers carefully. "
        "Tokens cross document boundaries and exercise causal packing. "
        "<think>reason</think> <tool_call>call</tool_call> "
    )
    (raw / "corpus.txt").write_text((text + "\n") * 128, encoding="utf-8")
    tokenizer = work / "tokenizer.json"

    # Exercise the repository tokenizer builder, including all chat/control
    # special tokens needed by SFT.
    code = (
        "import loomformer as lf; "
        f"lf.train_tokenizer({str(raw)!r}, 320, {str(tokenizer)!r})"
    )
    _run("synthetic tokenizer", [sys.executable, "-c", code])

    from tokenizers import Tokenizer

    vocab = Tokenizer.from_file(str(tokenizer)).get_vocab_size()
    if vocab < 256:
        raise AssertionError(f"synthetic tokenizer unexpectedly small: {vocab}")
    return tokenizer, int(vocab)


def _make_pt_bin(work: Path, tokenizer_path: Path) -> Path:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    sample = tokenizer.encode(
        "synthetic pretraining document with repeated causal language tokens "
    ).ids
    if not sample:
        raise AssertionError("tokenizer produced no PT tokens")
    tokens = np.asarray((sample * 128)[:2048], dtype=np.uint16)
    path = work / "pt_train.bin"
    tokens.tofile(path)
    return path


def _make_pt_parquet(work: Path) -> Path:
    directory = work / "pt_parquet"
    directory.mkdir()
    rows = [
        {
            "text": (
                f"document {index}: synthetic parquet pretraining text with "
                "enough repeated material for packing. " * 4
            )
        }
        for index in range(30)
    ]
    pq.write_table(pa.Table.from_pylist(rows), directory / "part-000.parquet")
    return directory


def _make_sft_parquet(work: Path) -> Path:
    directory = work / "sft_parquet"
    directory.mkdir()
    rows = []
    for index in range(40):
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": f"Item {index}?"},
                    {
                        "role": "assistant",
                        "content": f"Item {index}.",
                    },
                ]
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), directory / "part-000.parquet")
    return directory


def _checkpoint_digest(blob: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(blob["model"].items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _check_checkpoint(
    path: Path,
    *,
    step: int,
    optimizer: str,
    previous_digest: Optional[str] = None,
) -> str:
    if not path.is_file():
        raise AssertionError(f"checkpoint was not created: {path}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("model_kind") != "loomformer":
        raise AssertionError(f"{path}: wrong model_kind")
    if blob.get("ffn_type") != "paraplex":
        raise AssertionError(f"{path}: wrong ffn_type")
    if int(blob.get("step", -1)) != step:
        raise AssertionError(f"{path}: step={blob.get('step')}, expected {step}")
    if blob.get("optimizer_name") != optimizer:
        raise AssertionError(
            f"{path}: optimizer={blob.get('optimizer_name')}, expected {optimizer}"
        )
    if not isinstance(blob.get("optimizer"), dict) or not blob["optimizer"]:
        raise AssertionError(f"{path}: optimizer state is absent")
    if int(blob.get("tokens_seen", 0)) <= 0:
        raise AssertionError(f"{path}: tokens_seen was not advanced")
    progress = blob.get("dataset_progress")
    if not isinstance(progress, dict) or not progress:
        raise AssertionError(f"{path}: dataset cursor state is absent")
    for name, tensor in blob["model"].items():
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{path}: non-finite model tensor {name}")
    digest = _checkpoint_digest(blob)
    if previous_digest is not None and digest == previous_digest:
        raise AssertionError(f"{path}: model weights did not change after resume")
    return digest


def _device_profile(requested: str) -> Dict[str, Any]:
    if requested == "cpu" or not torch.cuda.is_available():
        return {
            "device": "cpu",
            "amp_dtype": "fp32",
            "modern": False,
            "all_modern": False,
            "compile": False,
            "gpu_count": 0,
            "devices": [],
        }
    device = "cuda:0" if requested == "auto" else requested
    index = 0
    if device.startswith("cuda:") and "," not in device:
        index = int(device.split(":", 1)[1])
    major, _minor = torch.cuda.get_device_capability(index)
    devices = [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
        }
        for i in range(torch.cuda.device_count())
    ]
    return {
        "device": device,
        "amp_dtype": "bf16",
        "modern": major >= 8,
        "all_modern": all(item["capability"][0] >= 8 for item in devices),
        "compile": major >= 7,
        "gpu_count": torch.cuda.device_count(),
        "devices": devices,
    }


def _run_gpu_parity(profile: Dict[str, Any]) -> None:
    for item in profile["devices"]:
        index = int(item["index"])
        _run(
            f"GPU backend parity on cuda:{index}",
            [sys.executable, "tests/gpu_parity.py", "--device", f"cuda:{index}"],
        )


def _probe_cuda_extensions(config_path: Path, modern: bool) -> None:
    if not torch.cuda.is_available():
        print("[matrix] CUDA unavailable: fused-extension probe skipped", flush=True)
        return

    # Keep this probe in its own interpreter: apply_config mutates module-level
    # geometry by design.
    probe = r'''
import sys
import torch
import loomformer as lf
import tria

cfg = lf.Config.from_yaml(sys.argv[1])
cfg.device = "cuda:0"
cfg.amp_dtype = "bf16"
cfg.attn_impl = "auto" if int(sys.argv[2]) else "sdpa"
lf.apply_config(cfg)

modules = {
    "phase_sin": lf._try_load_cuda_phase_sin(),
    "pvpowlu": lf._try_load_cuda_pvpowlu(),
    "depth_attn": lf._try_load_cuda_depth_attn(),
    "beta_space": lf._try_load_cuda_beta_space(),
    "paraplex": lf._try_load_cuda_paraplex(),
    "tria_carry": tria._try_load_cuda_tria(),
    "packed_gather": lf._try_load_cuda_packed_gather(),
}
missing = [name for name, module in modules.items() if module is None]
if missing:
    raise RuntimeError(f"fused CUDA extension(s) failed to build/load: {missing}")

if int(sys.argv[2]):
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        if major < 8:
            continue
        device = torch.device(f"cuda:{index}")
        fused = lf._probe_flash_value_fusion(device, torch.bfloat16)
        key = (index, torch.bfloat16, lf.HEAD_DIM)
        if not lf._flash_backend_cache.get(key, False):
            detail = lf._flash_probe_errors.get(key, "no detail")
            raise RuntimeError(
                f"FlashAttention varlen forward/backward failed on "
                f"cuda:{index} SM{major}.{minor}: {detail}"
            )
        print(
            f"[matrix] cuda:{index} FlashAttention varlen OK "
            f"(fused-value={fused})"
        )
print("[matrix] all fused CUDA extension modules loaded")
'''
    _run(
        "CUDA extensions and attention backend",
        [sys.executable, "-c", probe, config_path, int(modern)],
    )


def _assert_split(directory: Path, original_rows: int) -> None:
    train_file = directory / "part-000.parquet"
    val_file = directory / "val" / "val_split.parquet"
    manifest = Path(str(val_file) + ".manifest.json")
    if not val_file.is_file() or not manifest.is_file():
        raise AssertionError(f"automatic validation split missing under {directory}")
    train_rows = pq.ParquetFile(train_file).metadata.num_rows
    val_rows = pq.ParquetFile(val_file).metadata.num_rows
    if train_rows + val_rows != original_rows:
        raise AssertionError(
            f"{directory}: destructive split lost rows: "
            f"{train_rows}+{val_rows}!={original_rows}"
        )
    if train_rows <= 0 or val_rows <= 0:
        raise AssertionError(f"{directory}: empty train/validation split")


def run_matrix(args: argparse.Namespace) -> None:
    profile = _device_profile(args.device)
    _banner("host profile")
    print(json.dumps(profile, indent=2), flush=True)
    if args.setup and profile["device"] == "cpu":
        raise RuntimeError(
            "setup validation requires at least one visible CUDA GPU; refusing "
            "to report a successful installation without exercising GPU "
            "forward/backward, fused kernels and optimizer steps")

    if args.keep_temp:
        work = Path(tempfile.mkdtemp(prefix="loomformer-matrix."))
        cleanup = False
    else:
        context = tempfile.TemporaryDirectory(prefix="loomformer-matrix.")
        work = Path(context.name)
        cleanup = True
    print(f"[matrix] workspace: {work}", flush=True)

    try:
        _run(
            "focused unittest suite",
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-p",
                "test_*.py",
                "-v",
            ],
        )

        tokenizer, vocab = _make_tokenizer(work)
        pt_bin = _make_pt_bin(work, tokenizer)
        pt_parquet = _make_pt_parquet(work)
        sft_parquet = _make_sft_parquet(work)

        common = {
            "device": profile["device"],
            "amp_dtype": profile["amp_dtype"],
            "tokenizer": str(tokenizer),
            "vocab": vocab,
        }
        pt_checkpoint = work / "pt.pt"
        pt_runpoints = work / "pt_runpoints"
        pt_config = _write_config(
            PT_TEMPLATE,
            work / "pt.yaml",
            {
                **common,
                "train_dataset": str(pt_bin),
                "dataset_format": "bin",
                "dataset_cache": "mmap",
                "checkpoint": str(pt_checkpoint),
                "runpoints_path": str(pt_runpoints),
            },
        )
        _probe_cuda_extensions(pt_config, bool(profile["modern"]))
        _run_gpu_parity(profile)

        if profile["device"] == "cpu":
            _banner("CPU-ONLY CHECKS PASSED")
            print(
                "[matrix] CUDA was not selected; optimizer-step integration, "
                "backend parity and DDP were intentionally skipped.",
                flush=True,
            )
            return

        _run(
            "PT bin: two optimizer steps",
            [sys.executable, "loomformer.py", "--train", "--config", pt_config],
        )
        pt_digest = _check_checkpoint(
            pt_checkpoint, step=2, optimizer="adamw"
        )
        init_checkpoint = work / "pt.init.pt"
        if not init_checkpoint.is_file():
            raise AssertionError("save_initial_checkpoint did not create pt.init.pt")
        runpoints = sorted(pt_runpoints.glob("*.runpoint_step*.pt"))
        if len(runpoints) < 2:
            raise AssertionError("periodic runpoint checkpoints were not created")

        pt_resume_config = _write_config(
            PT_TEMPLATE,
            work / "pt_resume.yaml",
            {
                **common,
                "steps": 3,
                "train_dataset": str(pt_bin),
                "dataset_format": "bin",
                "dataset_cache": "mmap",
                "checkpoint": str(pt_checkpoint),
                "runpoints_path": str(pt_runpoints),
                "save_initial_checkpoint": False,
                "resume": str(pt_checkpoint),
            },
        )
        _run(
            "PT checkpoint resume and dataset cursor restore",
            [
                sys.executable,
                "loomformer.py",
                "--train",
                "--config",
                pt_resume_config,
            ],
        )
        pt_digest = _check_checkpoint(
            pt_checkpoint,
            step=3,
            optimizer="adamw",
            previous_digest=pt_digest,
        )

        _run(
            "PT full sequential evaluation",
            [
                sys.executable,
                "loomformer.py",
                "--eval",
                "--checkpoint",
                pt_checkpoint,
                "--dataset",
                pt_bin,
                "--device",
                profile["device"],
                "--eval-batch-size",
                "2",
            ],
        )
        _run(
            "checkpointed autoregressive inference",
            [
                sys.executable,
                "loomformer.py",
                "--infer",
                "--config",
                pt_config,
                "--checkpoint",
                pt_checkpoint,
                "--device",
                profile["device"],
                "--prompt",
                "synthetic",
                "--max-new",
                "2",
            ],
        )

        otf_checkpoint = work / "pt_otf.pt"
        otf_config = _write_config(
            PT_TEMPLATE,
            work / "pt_otf.yaml",
            {
                **common,
                "steps": 1,
                "grad_accum_steps": 1,
                "save_every": 0,
                "save_initial_checkpoint": False,
                "train_dataset": str(pt_parquet),
                "dataset_format": "parquet",
                "dataset_cache": "ram",
                "auto_val_split_pct": 20.0,
                "checkpoint": str(otf_checkpoint),
                "runpoints_path": None,
                # Exercise Dynamo/custom-op integration where the architecture
                # supports torch.compile; Pascal intentionally stays eager.
                "compile": bool(profile["compile"]),
                "graph": bool(profile["compile"]),
                "attn_impl": "auto" if profile["modern"] else "sdpa",
            },
        )
        _run(
            "PT Parquet OTF + auto-val + compile/custom-op graph",
            [sys.executable, "loomformer.py", "--train", "--config", otf_config],
        )
        _check_checkpoint(otf_checkpoint, step=1, optimizer="adamw")
        _assert_split(pt_parquet, 30)

        sft_checkpoint = work / "sft.pt"
        sft_config = _write_config(
            SFT_TEMPLATE,
            work / "sft.yaml",
            {
                **common,
                "train_dataset": str(sft_parquet),
                "init_checkpoint": str(pt_checkpoint),
                "checkpoint": str(sft_checkpoint),
                "attn_impl": "auto" if profile["modern"] else "sdpa",
                "compile": bool(profile["compile"]),
                "graph": bool(profile["compile"]),
            },
        )
        _run(
            "SFT Parquet OTF: PT init, packed masks and activation checkpointing",
            [sys.executable, "loomformer.py", "--train", "--config", sft_config],
        )
        sft_digest = _check_checkpoint(
            sft_checkpoint, step=2, optimizer="atom"
        )
        if sft_digest == pt_digest:
            raise AssertionError("SFT did not change pretrained model weights")
        _assert_split(sft_parquet, 40)

        _run(
            "SFT checkpoint resume",
            [
                sys.executable,
                "loomformer.py",
                "--train",
                "--config",
                sft_config,
                "--resume",
                sft_checkpoint,
                "--steps",
                "3",
            ],
        )
        _check_checkpoint(
            sft_checkpoint,
            step=3,
            optimizer="atom",
            previous_digest=sft_digest,
        )

        gpu_count = int(profile["gpu_count"])
        if gpu_count >= 2 and not args.no_ddp:
            ddp_device = "cudas"
            ddp_pt_checkpoint = work / "pt_ddp.pt"
            ddp_pt_config = _write_config(
                PT_TEMPLATE,
                work / "pt_ddp.yaml",
                {
                    **common,
                    "device": ddp_device,
                    "batch_size": gpu_count,
                    "steps": 1,
                    "grad_accum_steps": 1,
                    "save_every": 0,
                    "save_initial_checkpoint": False,
                    "checkpoint": str(ddp_pt_checkpoint),
                    "train_dataset": str(pt_bin),
                    "runpoints_path": None,
                },
            )
            _run(
                f"PT DDP across all {gpu_count} visible GPUs",
                [sys.executable, "loomformer.py", "--train", "--config", ddp_pt_config],
            )
            _check_checkpoint(ddp_pt_checkpoint, step=1, optimizer="adamw")

            ddp_sft_checkpoint = work / "sft_ddp.pt"
            ddp_sft_config = _write_config(
                SFT_TEMPLATE,
                work / "sft_ddp.yaml",
                {
                    **common,
                    "device": ddp_device,
                    "batch_size": gpu_count,
                    "steps": 1,
                    "train_dataset": str(sft_parquet),
                    "auto_val_split_pct": 20.0,
                    "init_checkpoint": str(pt_checkpoint),
                    "checkpoint": str(ddp_sft_checkpoint),
                    "attn_impl": "auto" if profile["all_modern"] else "sdpa",
                },
            )
            _run(
                f"SFT DDP across all {gpu_count} visible GPUs",
                [sys.executable, "loomformer.py", "--train", "--config", ddp_sft_config],
            )
            _check_checkpoint(ddp_sft_checkpoint, step=1, optimizer="atom")
        else:
            print("[matrix] DDP cases skipped (fewer than two GPUs or --no-ddp)", flush=True)

        _banner("ALL MATRIX CASES PASSED")
        print(f"[matrix] synthetic workspace: {work}", flush=True)
    except BaseException:
        print(f"[matrix] FAILED; synthetic workspace was {work}", file=sys.stderr, flush=True)
        raise
    finally:
        if cleanup:
            context.cleanup()
        else:
            print(f"[matrix] retained workspace: {work}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA when available), cpu, or a single CUDA device",
    )
    parser.add_argument(
        "--no-ddp",
        action="store_true",
        help="skip multi-GPU PT/SFT subprocesses",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="retain generated tokenizer, datasets, configs and checkpoints",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="setup.sh mode: require CUDA and run the complete matrix",
    )
    run_matrix(parser.parse_args())


if __name__ == "__main__":
    main()
