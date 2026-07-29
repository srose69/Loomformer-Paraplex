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
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from synthetic_tokenizer import build_synthetic_bpe


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
PT_TEMPLATE = TESTS / "test_pt.yaml"
SFT_TEMPLATE = TESTS / "test_sft.yaml"
_RUN_LOG_DIR: Optional[Path] = None
_RUN_INDEX = 0
_LIVE_ACTIVE = False
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ
_COLORS = {
    "RUN": "\033[1;36m",
    "PASS": "\033[1;32m",
    "SKIP": "\033[1;33m",
    "WARN": "\033[1;33m",
    "FAIL": "\033[1;31m",
    "INFO": "\033[0;36m",
}
_RESET = "\033[0m"


class MatrixFailure(RuntimeError):
    pass


def _paint(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}" if _COLOR else text


def _clear_live() -> None:
    global _LIVE_ACTIVE
    if _LIVE_ACTIVE and sys.stdout.isatty():
        print("\r\033[2K", end="", flush=True)
    _LIVE_ACTIVE = False


def _status(kind: str, message: str) -> None:
    _clear_live()
    marker = _paint(f"[{kind}]", _COLORS[kind])
    print(f"{marker} {message}", flush=True)


def _live(message: str) -> None:
    global _LIVE_ACTIVE
    if sys.stdout.isatty():
        print(f"\r\033[2K{_paint('[....]', _COLORS['INFO'])} {message}",
              end="", flush=True)
        _LIVE_ACTIVE = True
    else:
        print(f"[....] {message}", flush=True)


def _banner(label: str) -> None:
    _clear_live()
    rule = "━" * max(8, 72 - len(label))
    print(f"\n{_paint(f'━━━ {label} {rule}', _COLORS['INFO'])}", flush=True)


def _clean_line(line: str) -> str:
    return _ANSI_RE.sub("", line).strip()


def _emit_child_progress(label: str, line: str) -> None:
    clean = _clean_line(line)
    if not clean:
        return
    if clean.startswith("[kernels]") and (
        " compiling " in clean or " compiled " in clean
    ):
        _live(clean)
    elif clean.startswith("[compile]"):
        _live(clean)
    elif clean.startswith("[LF]") or clean.startswith("[EVAL]"):
        _live(f"{label}: {clean}")
    elif (
        "worth investigating" in clean
        or (
            clean.startswith("[loomformer] CUDA")
            and (" failed " in clean or " unavailable " in clean)
        )
    ):
        _status("WARN", clean)


def _failure_summary(lines: Iterable[str]) -> list[str]:
    markers = (
        "AssertionError:",
        "ChildFailedError:",
        "CUDA error:",
        "Error:",
        "Exception:",
        "InductorError:",
        "OSError:",
        "RuntimeError:",
        "TypeError:",
        "ValueError:",
        "FAILED (",
    )
    selected: list[str] = []
    for raw in lines:
        line = _clean_line(raw)
        if line and any(marker in line for marker in markers):
            if line not in selected:
                selected.append(line)
    if selected:
        return selected[-8:]
    fallback = [_clean_line(line) for line in lines if _clean_line(line)]
    return fallback[-5:]


def _run(
    label: str,
    argv: Iterable[object],
    *,
    timeout: int = 1800,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    global _RUN_INDEX
    command = [str(item) for item in argv]
    _status("RUN", label)
    started = time.monotonic()
    env = os.environ.copy()
    # setup.sh invokes the venv interpreter by absolute path without
    # activating it. Keep sibling console tools such as ninja visible to
    # PyTorch's C++/CUDA extension loader in every matrix subprocess.
    interpreter_bin = str(Path(sys.executable).absolute().parent)
    env["PATH"] = interpreter_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("NO_COLOR", "1")
    if extra_env:
        env.update(extra_env)

    _RUN_INDEX += 1
    log_dir = _RUN_LOG_DIR or Path(tempfile.gettempdir())
    log_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "case"
    log_path = log_dir / f"{_RUN_INDEX:02d}-{slug}.log"
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    output: queue.Queue[Optional[str]] = queue.Queue()

    def _reader() -> None:
        try:
            for child_line in process.stdout:
                output.put(child_line)
        finally:
            output.put(None)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    tail: deque[str] = deque(maxlen=400)
    reader_done = False
    timed_out = False

    def _signal_group(sig: signal.Signals) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    try:
        with log_path.open("w", encoding="utf-8") as log:
            while not reader_done or process.poll() is None:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0 and process.poll() is None:
                    timed_out = True
                    _signal_group(signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        _signal_group(signal.SIGKILL)
                    continue
                try:
                    item = output.get(
                        timeout=max(0.05, min(0.25, remaining))
                    )
                except queue.Empty:
                    continue
                if item is None:
                    reader_done = True
                    continue
                log.write(item)
                log.flush()
                for part in re.split(r"[\r\n]+", item):
                    if not part:
                        continue
                    tail.append(part)
                    _emit_child_progress(label, part)
    except BaseException:
        _signal_group(signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _signal_group(signal.SIGKILL)
        raise
    reader.join(timeout=1)
    returncode = process.wait()
    elapsed = time.monotonic() - started
    if timed_out or returncode:
        reason = f"timeout after {elapsed:.1f}s" if timed_out else (
            f"exit {returncode} after {elapsed:.1f}s"
        )
        _status("FAIL", f"{label} ({reason})")
        for line in _failure_summary(tail):
            print(f"       {_paint(line, _COLORS['FAIL'])}", flush=True)
        print(f"       log: {log_path}", flush=True)
        raise MatrixFailure(f"{label} failed")
    log_path.unlink(missing_ok=True)
    _status("PASS", f"{label} ({elapsed:.1f}s)")


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
    tokenizer = work / "tokenizer.json"
    vocab = build_synthetic_bpe(tokenizer, vocab_size=256)
    _status("PASS", f"synthetic BPE tokenizer ({vocab} tokens)")
    return tokenizer, vocab


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


def _probe_cuda_extensions(
    config_path: Path,
    modern: bool,
    report_path: Path,
) -> list[Dict[str, Any]]:
    if not torch.cuda.is_available():
        _status("SKIP", "CUDA extension probe (CUDA unavailable)")
        return []

    # Keep this probe in its own interpreter: apply_config mutates module-level
    # geometry by design.
    probe = r'''
import json
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

backends = []
for index in range(torch.cuda.device_count()):
    major, minor = torch.cuda.get_device_capability(index)
    device = torch.device(f"cuda:{index}")
    dtype = torch.bfloat16
    flash_fused = False
    te_fused = False
    if int(sys.argv[2]) and major >= 8:
        flash_fused = lf._probe_flash_value_fusion(device, dtype)
        te_fused = lf._probe_te_value_fusion(device, dtype)
    key = (index, dtype, lf.HEAD_DIM)
    flash_ok = bool(lf._flash_backend_cache.get(key, False))
    te_ok = bool(lf._te_backend_cache.get(key, False))
    if int(sys.argv[2]) and major >= 8 and not (flash_ok or te_ok):
        raise RuntimeError(
            f"no validated varlen forward/backward backend on cuda:{index}: "
            f"{lf._varlen_backend_failure_detail(device, dtype)}"
        )
    selected = (
        "flash-attn" if flash_ok
        else "transformer-engine" if te_ok
        else "sdpa"
    )
    backends.append(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": [major, minor],
            "dtype": str(dtype).removeprefix("torch."),
            "head_dim": int(lf.HEAD_DIM),
            "selected": selected,
            "flash_attn": {
                "forward_backward": flash_ok,
                "fused_value": bool(flash_fused),
            },
            "transformer_engine": {
                "forward_backward": te_ok,
                "fused_value": bool(te_fused),
            },
            "sdpa": {"forward_backward": False},
        }
    )
with open(sys.argv[3], "w", encoding="utf-8") as handle:
    json.dump(backends, handle, indent=2)
print("[matrix] all fused CUDA extension modules loaded")
'''
    _run(
        "CUDA extensions and attention backend",
        [
            sys.executable,
            "-c",
            probe,
            config_path,
            int(modern),
            report_path,
        ],
    )
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, list):
        raise AssertionError("attention backend probe returned a non-list report")
    return report


def _write_validation_report(
    output: Path,
    profile: Dict[str, Any],
    attention: list[Dict[str, Any]],
) -> None:
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "profile": {
            "device": profile["device"],
            "amp_dtype": profile["amp_dtype"],
            "compile": bool(profile["compile"]),
            "gpu_count": int(profile["gpu_count"]),
        },
        "attention": attention,
    }
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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
    global _RUN_LOG_DIR, _RUN_INDEX
    profile = _device_profile(args.device)
    _banner("LoomFormer installation matrix")
    devices = profile["devices"]
    if devices:
        summary = ", ".join(
            f"cuda:{item['index']} {item['name']} "
            f"SM{item['capability'][0]}.{item['capability'][1]}"
            for item in devices
        )
        _status(
            "INFO",
            f"{summary} · amp={profile['amp_dtype']} · "
            f"compile={profile['compile']}",
        )
    else:
        _status("WARN", "CUDA unavailable; CPU-only validation profile")
    if args.setup and profile["device"] == "cpu":
        _status("FAIL", "setup validation requires a visible CUDA GPU")
        raise MatrixFailure(
            "setup validation requires at least one visible CUDA GPU; refusing "
            "to report a successful installation without exercising GPU "
            "forward/backward, fused kernels and optimizer steps")

    work = Path(tempfile.mkdtemp(prefix="loomformer-matrix."))
    cleanup = not args.keep_temp
    completed = False
    _RUN_INDEX = 0
    _RUN_LOG_DIR = work / "logs"

    try:
        tokenizer, vocab = _make_tokenizer(work)

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
            extra_env={
                "LOOM_TEST_TOKENIZER": str(tokenizer),
                "LOOM_TEST_VOCAB": str(vocab),
            },
        )

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
        attention_report = _probe_cuda_extensions(
            pt_config,
            bool(profile["modern"]),
            work / "attention-backends.json",
        )
        _run_gpu_parity(profile)
        for backend in attention_report:
            backend["sdpa"]["forward_backward"] = True

        if profile["device"] == "cpu":
            _status(
                "SKIP",
                "CUDA optimizer steps, backend parity and DDP "
                "(CPU-only profile)",
            )
            completed = True
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
            _status("SKIP", "DDP cases (fewer than two GPUs or --no-ddp)")

        if args.report:
            _write_validation_report(
                Path(args.report),
                profile,
                attention_report,
            )
        completed = True
        _banner("Validation complete")
        _status("PASS", "all installation matrix cases")
    except BaseException:
        cleanup = False
        _status("FAIL", f"matrix artifacts retained at {work}")
        raise
    finally:
        _RUN_LOG_DIR = None
        if cleanup and completed:
            shutil.rmtree(work)
        elif completed:
            _status("INFO", f"matrix artifacts retained at {work}")


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
    parser.add_argument(
        "--report",
        help="write an atomic JSON validation report after every matrix case passes",
    )
    try:
        run_matrix(parser.parse_args())
    except MatrixFailure:
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        _status("FAIL", "matrix interrupted")
        raise SystemExit(130) from None
    except BaseException as exc:
        _status("FAIL", f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
