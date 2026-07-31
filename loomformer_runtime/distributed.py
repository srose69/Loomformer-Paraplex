from __future__ import annotations

import contextlib
import faulthandler
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

if TYPE_CHECKING:
    from loomformer import Config

def configure_cuda_math() -> None:
    """Use one CUDA matmul policy in single-GPU and every DDP rank."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_auto(pref: Optional[str] = None) -> torch.device:
    dev = torch.device(pref) if pref else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        configure_cuda_math()
    return dev


def _parse_cuda_device_list(pref: str) -> Optional[List[int]]:
    """Parse a comma-separated CUDA device list, or return ``None`` for a scalar selector."""
    raw = str(pref or "").strip().lower().replace(" ", "")
    if "," not in raw:
        return None
    out: List[int] = []
    for part in raw.split(","):
        if not part:
            raise ValueError(f"bad CUDA device list {pref!r}")
        if part.startswith("cuda:"):
            part = part[len("cuda:"):]
        if not part.isdigit():
            raise ValueError(f"bad CUDA device list {pref!r}; expected e.g. cuda:0,cuda:1")
        idx = int(part)
        if idx < 0:
            raise ValueError(f"bad CUDA device index {idx} in {pref!r}")
        out.append(idx)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate CUDA device in {pref!r}")
    return out


def _cuda_visible_devices_for_child(indices: List[int]) -> str:
    parent = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if parent:
        entries = [x.strip() for x in parent.split(",") if x.strip()]
        if entries and all(0 <= i < len(entries) for i in indices):
            return ",".join(entries[i] for i in indices)
    return ",".join(str(i) for i in indices)


def _auto_omp_threads(nproc: int) -> int:
    total = os.cpu_count() or 1
    if nproc <= 0:
        return 1
    return max(1, min(8, total // nproc))


def _linux_process_children(pid: int) -> List[int]:
    """Return direct child PIDs for self-launched torchrun diagnostics/control."""
    try:
        raw = open(
            f"/proc/{int(pid)}/task/{int(pid)}/children",
            encoding="ascii",
        ).read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [int(value) for value in raw.split() if value.isdigit()]


def _torchrun_rank_workers(root_pid: int) -> List[int]:
    """Find only LoomFormer rank processes, excluding Inductor subprocesses."""
    pending = [int(root_pid)]
    seen = {int(root_pid)}
    workers: List[int] = []
    while pending:
        parent = pending.pop()
        for pid in _linux_process_children(parent):
            if pid in seen:
                continue
            seen.add(pid)
            pending.append(pid)
            try:
                environ = open(
                    f"/proc/{pid}/environ", "rb"
                ).read().split(b"\0")
                cmdline = open(
                    f"/proc/{pid}/cmdline", "rb"
                ).read().split(b"\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            has_rank = any(item.startswith(b"LOCAL_RANK=") for item in environ)
            is_loomformer = any(
                item.endswith(b"/loomformer.py") or item == b"loomformer.py"
                for item in cmdline
            )
            if has_rank and is_loomformer:
                workers.append(pid)
    return sorted(workers)


def ddp_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def ddp_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def ddp_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def ddp_is_distributed() -> bool:
    return ddp_world_size() > 1


def ddp_is_main() -> bool:
    return ddp_rank() == 0


def ddp_print(*args, **kwargs) -> None:
    if ddp_is_main():
        print(*args, **kwargs)


def ddp_trace(stage: str, *, step: Optional[int] = None,
              micro: Optional[int] = None) -> None:
    """Emit rank-local progress only for the installation hang diagnostic."""
    if os.environ.get("LOOM_DDP_TRACE") != "1" or not ddp_is_distributed():
        return
    location = ""
    if step is not None:
        location += f" step={int(step)}"
    if micro is not None:
        location += f" micro={int(micro)}"
    print(
        f"[ddp-trace] rank={ddp_rank()}{location} stage={stage}",
        file=sys.stderr,
        flush=True,
    )


def ddp_barrier(device: Optional[torch.device] = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl":
        idx = ddp_local_rank() if device is None or device.type == "cuda" else 0
        dist.barrier(device_ids=[int(idx)])
    else:
        dist.barrier()


def ddp_mean_float(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(value)
    t = torch.tensor([float(value)], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return float(t.item())


def ddp_sum_int(value: int, device: torch.device) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    t = torch.tensor([int(value)], dtype=torch.long, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def ddp_weighted_mean(total: float, count: int, device: torch.device) -> Tuple[float, int]:
    values = torch.tensor(
        [float(total), float(count)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    global_count = int(values[1].item())
    return float(values[0].item()) / max(1, global_count), global_count


def ddp_unwrap_model(model: nn.Module) -> nn.Module:
    raw = model
    seen: set[int] = set()
    while id(raw) not in seen:
        seen.add(id(raw))
        is_fsdp = raw.__class__.__name__ == "FullyShardedDataParallel"
        if isinstance(raw, DDP) or is_fsdp:
            raw = raw.module
            continue
        if hasattr(raw, "_orig_mod"):
            raw = raw._orig_mod
            continue
        break
    return raw


def ddp_sync_mutable_buffers(model: nn.Module) -> None:
    """Match DDP buffer semantics outside the compiled forward graph.

    ``ParaplexFFN.beta_anchor`` is the only mutable model buffer. DDP's
    built-in ``broadcast_buffers`` runs from its forward pre-hook; putting that
    hook inside ``torch.compile(DDP)`` lets rank-local guard misses execute an
    extra NCCL broadcast while a cache-hit rank has already entered gradient
    all-reduce. One explicit coalesced broadcast before every outer forward
    preserves rank-0 anchor semantics with a fixed collective order.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    raw = ddp_unwrap_model(model)
    anchors = [
        block.ffn.beta_anchor
        for block in getattr(raw, "blocks", ())
        if hasattr(getattr(block, "ffn", None), "beta_anchor")
    ]
    if not anchors:
        return
    with torch.no_grad():
        packed = torch.stack(
            [anchor.detach().reshape(()) for anchor in anchors]
        )
        dist.broadcast(packed, src=0)
        for anchor, value in zip(anchors, packed.unbind()):
            anchor.copy_(value)


def ddp_assert_config_consensus(cfg: "Config") -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    local = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    gathered: List[Optional[str]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    if any(value != gathered[0] for value in gathered[1:]):
        decoded = [json.loads(value) for value in gathered]
        keys = sorted({
            key
            for item in decoded
            for key in item
            if any(other.get(key) != item.get(key) for other in decoded)
        })
        raise RuntimeError(
            "DDP ranks resolved different configs: " + ", ".join(keys))


def ddp_static_graph_policy(cfg: "Config") -> Tuple[bool, str]:
    """Return whether DDP may safely reuse one fixed reducer-hook trace."""
    accum_is_single = (
        max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1)) == 1
    )
    compiled_with_eager_islands = bool(
        getattr(cfg, "compile", False)
        and bool(getattr(cfg, "tria_carry_enabled", False))
        and bool(getattr(cfg, "tria_temporal_enabled", True))
    )
    if compiled_with_eager_islands:
        return False, "compiled depth-replay eager island"
    if not accum_is_single:
        return False, "grad_accum_steps > 1 needs no_sync"
    return True, ""


def maybe_launch_or_init_ddp(device_pref: Optional[str], training: bool) -> Tuple[torch.device, bool, int, int, int]:
    # Calling a venv interpreter by absolute path does not activate the venv
    # and therefore does not put sibling console tools (notably ninja) on
    # PATH. PyTorch's extension loader still resolves ninja through PATH even
    # when the compiled module is already cached. Make direct single-process
    # and self-launched torchrun invocations equivalent to an activated venv.
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if interpreter_bin not in path_entries:
        os.environ["PATH"] = os.pathsep.join(
            [interpreter_bin, *path_entries]
        )

    pref = str(device_pref or "").strip().lower()
    cuda_subset = _parse_cuda_device_list(pref)
    wants_ddp_launch = pref == "cudas" or cuda_subset is not None
    if wants_ddp_launch and training and "WORLD_SIZE" not in os.environ:
        if not torch.cuda.is_available():
            raise RuntimeError(f"--device {pref or 'cudas'} requested but CUDA is unavailable")
        env = os.environ.copy()
        if cuda_subset is None:
            n = torch.cuda.device_count()
            launch_note = "all visible CUDA devices"
        else:
            if len(cuda_subset) < 2:
                raise RuntimeError(f"--device {pref!r} selects fewer than 2 CUDA devices")
            visible = _cuda_visible_devices_for_child(cuda_subset)
            env["CUDA_VISIBLE_DEVICES"] = visible
            n = len(cuda_subset)
            launch_note = f"CUDA_VISIBLE_DEVICES={visible}"
        if n < 2:
            raise RuntimeError(f"--device {pref or 'cudas'} requested but fewer than 2 CUDA GPUs are visible")
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(n), os.path.abspath(sys.argv[0])] + sys.argv[1:]
        if not env.get("OMP_NUM_THREADS"):
            env["OMP_NUM_THREADS"] = str(_auto_omp_threads(n))
        print(f"[ddp] launching ({launch_note}):", " ".join(cmd), flush=True)
        if "OMP_NUM_THREADS" in env and not os.environ.get("OMP_NUM_THREADS"):
            print(f"[ddp] auto OMP_NUM_THREADS={env['OMP_NUM_THREADS']}", flush=True)
        decision_path = os.path.join(
            tempfile.gettempdir(),
            f"loomformer-ddp-interrupt.{os.getpid()}",
        )
        env["LOOM_SELF_LAUNCHED_DDP"] = "1"
        env["LOOM_DDP_INTERRUPT_DECISION_FILE"] = decision_path
        with contextlib.suppress(FileNotFoundError):
            os.unlink(decision_path)
        child = subprocess.Popen(
            cmd,
            env=env,
            start_new_session=True,
        )
        interrupted = False
        try:
            while True:
                try:
                    returncode = child.wait()
                    break
                except KeyboardInterrupt:
                    if interrupted:
                        print(
                            "\n[interrupt] second Ctrl-C; terminating DDP job.",
                            flush=True,
                        )
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGTERM)
                        returncode = child.wait()
                        break
                    interrupted = True
                    try:
                        print(
                            "\n[interrupt] save a checkpoint after the active "
                            "optimizer step before exiting? [y/N] ",
                            end="",
                            flush=True,
                        )
                        answer = input().strip().lower()
                    except EOFError:
                        answer = "n"
                    save = answer in ("y", "yes")
                    with open(decision_path, "w", encoding="ascii") as handle:
                        handle.write("1\n" if save else "0\n")
                    workers: List[int] = []
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline and not workers:
                        workers = _torchrun_rank_workers(child.pid)
                        if not workers:
                            time.sleep(0.05)
                    if workers:
                        print(
                            f"[interrupt] graceful stop requested on "
                            f"{len(workers)} DDP rank(s); waiting for the "
                            "active step to finish...",
                            flush=True,
                        )
                        for pid in workers:
                            with contextlib.suppress(ProcessLookupError):
                                os.kill(pid, signal.SIGINT)
                    else:
                        print(
                            "[interrupt] rank workers were not discoverable; "
                            "stopping torchrun directly.",
                            flush=True,
                        )
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGINT)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(decision_path)
        raise SystemExit(130 if interrupted else returncode)

    world_size = ddp_world_size()
    rank = ddp_rank()
    local_rank = ddp_local_rank()
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP world detected but CUDA is unavailable")
        if not os.environ.get("OMP_NUM_THREADS"):
            os.environ["OMP_NUM_THREADS"] = str(_auto_omp_threads(world_size))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
        if (
            os.environ.get("LOOM_DDP_TRACE") == "1"
            and hasattr(signal, "SIGUSR1")
        ):
            # The matrix runner sends SIGUSR1 only to rank worker processes,
            # immediately before terminating a timed-out cell. This records
            # every Python thread's exact blocking frame in the retained log.
            faulthandler.register(
                signal.SIGUSR1,
                file=sys.stderr,
                all_threads=True,
                chain=False,
            )
        ddp_print(f"[ddp] rank={rank} local_rank={local_rank} device={dev}", flush=True)
        return dev, True, world_size, rank, local_rank
    if pref == "cudas":
        # Non-training actions do not self-launch. Treat cudas as cuda:0 there.
        pref = "cuda:0"
    elif cuda_subset is not None:
        # Non-training actions do not self-launch. Use the first requested local GPU.
        pref = f"cuda:{cuda_subset[0]}"
    dev = device_auto(pref or None)
    return dev, False, 1, 0, 0

__all__ = ('configure_cuda_math', 'set_seed', 'device_auto', '_parse_cuda_device_list', '_cuda_visible_devices_for_child', '_auto_omp_threads', '_linux_process_children', '_torchrun_rank_workers', 'ddp_world_size', 'ddp_rank', 'ddp_local_rank', 'ddp_is_distributed', 'ddp_is_main', 'ddp_print', 'ddp_trace', 'ddp_barrier', 'ddp_mean_float', 'ddp_sum_int', 'ddp_weighted_mean', 'ddp_unwrap_model', 'ddp_sync_mutable_buffers', 'ddp_assert_config_consensus', 'ddp_static_graph_policy', 'maybe_launch_or_init_ddp')
