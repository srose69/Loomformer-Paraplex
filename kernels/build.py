"""
kernels/build.py -- shared loader for the CUDA modules that used to live as
inline r\"\"\"...\"\"\" strings inside tria.py/loomformer.py.

Why this exists: load_inline() recompiles from an in-memory string every time
its content-hash changes, which makes it clumsy to (a) diff kernel changes,
(b) point clangd/nsight at real files, (c) hand-inspect generated PTX/SASS.
Real .cu/.cuh/.cpp files under kernels/<name>/ fix all three -- each kernel
group is split into a torch-free <name>_kernel.cuh (the actual device code,
safe to nvcc --ptx standalone) and an ATen-facing <name>_launcher.cu (arg
checks, dispatch, kernel launch, pybind).

Rebuild mechanics: torch.utils.cpp_extension.load() already drives ninja
under build_directory, and ninja already does its own incremental/skip-if-
unchanged compilation. We do NOT try to out-guess ninja's own change
detection -- we compute+store sha256 per source file purely as a visible,
committable record (".hashes.json") and as a fast pre-check to decide
whether a real rebuild happened (only then is there anything worth printing).
ninja/nvcc's own --verbose spam is always suppressed (verbose=False
unconditionally) in favor of one compact, self-overwriting "[kernels]"
status line per build step -- see _status().

Progress counter: "[kernels] i/N compiling ..." where N is the total number
of kernel GROUPS across the whole kernels/ tree, discovered by counting
*_kernel.cu files on disk -- never hardcoded, so adding a new kernel group
later just grows N automatically. A module that bundles several groups
into one .so (tria's 8) advances the counter by all of them at once.

PTX: standalone dumps are disabled during normal builds. Set
`KERNELS_DUMP_PTX=1` to compile each requested <name>_kernel.cu with
`nvcc --ptx` after a changed extension build.

Every build_or_load() call fails closed: on any exception during the main
.so build, raises -- callers are expected to wrap this in their own
try/except and fall back to PyTorch on failure, exactly as the old
load_inline() call sites did.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Dict, List, Optional, Sequence

_KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_KERNELS_DIR, "build")
_HASH_FILE = os.path.join(_KERNELS_DIR, ".hashes.json")
_STATE_LOCK = threading.Lock()
_EXT_LOCKS: Dict[str, threading.Lock] = {}


def _extension_lock(name: str) -> threading.Lock:
    with _STATE_LOCK:
        return _EXT_LOCKS.setdefault(name, threading.Lock())

# ============================================================================
# progress counter -- lazily discovers the total kernel-group count from disk
# (counts *_kernel.cu files under kernels/, excluding kernels/build/ output),
# so it never needs to be hand-maintained as groups get added/removed.
# ============================================================================
_progress_total: Optional[int] = None
_progress_done = 0


def _discover_total_kernel_groups() -> int:
    total = 0
    for root, dirs, files in os.walk(_KERNELS_DIR):
        dirs[:] = [d for d in dirs if d not in ("build", "ptx", "__pycache__")]
        total += sum(1 for f in files if f.endswith("_kernel.cu"))
    return total


def _progress_total_count() -> int:
    global _progress_total
    if _progress_total is None:
        _progress_total = max(1, _discover_total_kernel_groups())
    return _progress_total


# ============================================================================
# status line: single self-overwriting "[kernels] ..." line per build step.
# ============================================================================

def _status(msg: str, done: bool = False) -> None:
    pad = " " * 8
    line = f"[kernels] {msg}"
    if done:
        sys.stdout.write(f"\r{line}{pad}\n")
    else:
        sys.stdout.write(f"\r{line}{pad}")
    sys.stdout.flush()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_LOCAL_INCLUDE_RE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)


def _source_dependencies(sources: Sequence[str]) -> List[str]:
    """Return sources plus recursively included project-local headers."""
    found = set()
    pending = list(sources)
    while pending:
        path = os.path.abspath(pending.pop())
        if path in found or not os.path.isfile(path):
            continue
        found.add(path)
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                includes = _LOCAL_INCLUDE_RE.findall(f.read())
        except OSError:
            continue
        for include in includes:
            candidate = os.path.abspath(os.path.join(os.path.dirname(path), include))
            if os.path.commonpath((candidate, _KERNELS_DIR)) == _KERNELS_DIR:
                pending.append(candidate)
    return sorted(found)


def _ptx_dump_enabled() -> bool:
    return os.environ.get("KERNELS_DUMP_PTX", "0").lower() in ("1", "true", "yes", "on")


def _load_hash_db() -> Dict[str, Dict[str, str]]:
    if not os.path.exists(_HASH_FILE):
        return {}
    try:
        with open(_HASH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}  # corrupt/partial hash file -> treat as "nothing recorded", not fatal


def _save_hash_db(db: Dict[str, Dict[str, str]]) -> None:
    tmp = _HASH_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, sort_keys=True)
    os.replace(tmp, _HASH_FILE)  # atomic on POSIX -- no half-written .hashes.json


# ============================================================================
# PTX generation -- fully silent, best-effort, never affects the main build.
# ============================================================================

def _visible_cuda_arch_list() -> Optional[str]:
    """Return unique compute capabilities for the CUDA devices visible here.

    The format matches TORCH_CUDA_ARCH_LIST, for example ``"8.0;8.6"``.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        arches = {
            f"{major}.{minor}"
            for major, minor in (
                torch.cuda.get_device_capability(i)
                for i in range(torch.cuda.device_count())
            )
        }
        return ";".join(sorted(arches, key=lambda x: tuple(map(int, x.split("."))))) or None
    except Exception:
        return None


def _configure_visible_cuda_arches() -> None:
    """Limit extension builds to architectures present in visible GPUs.

    An explicitly supplied TORCH_CUDA_ARCH_LIST remains an override.
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return
    arch_list = _visible_cuda_arch_list()
    if arch_list:
        os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list


def _nvcc_arch_flag() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"sm_{major}{minor}"
    except Exception:
        pass
    return "sm_61"  # this repo's baseline Pascal target


def _dump_ptx_silent(kernel_cu_rel: str) -> None:
    """Best-effort `nvcc --ptx` dump into a ptx/ subfolder next to the
    source. Deliberately prints nothing, ever, on any path (success,
    missing nvcc, compile error) -- inspect kernels/<group>/ptx/*.ptx
    directly if/when you need it."""
    try:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            return
        src = os.path.join(_KERNELS_DIR, kernel_cu_rel)
        if not os.path.exists(src):
            return
        out_dir = os.path.join(os.path.dirname(src), "ptx")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(src))[0]
        out_path = os.path.join(out_dir, f"{stem}.ptx")

        try:
            from torch.utils.cpp_extension import include_paths
            inc_flags = [f"-I{p}" for p in include_paths(cuda=True)]
        except Exception:
            inc_flags = []

        cmd = [
            nvcc, f"-arch={_nvcc_arch_flag()}", "--ptx",
            "-std=c++17", "--expt-relaxed-constexpr",
            *inc_flags, src, "-o", out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        pass  # silent, on purpose -- see docstring


# ============================================================================
# public entry point
# ============================================================================

def build_or_load(
    ext_name: str,
    sources: Sequence[str],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    ptx_kernels: Optional[Dict[str, str]] = None,
):
    """Compute sha256 for each file in `sources` (paths relative to kernels/,
    e.g. "tria/bindings.cpp" or "tria/tria_init/tria_init_launcher.cu"),
    recursively include project-local quoted headers in that hash, compare
    against the record for `ext_name` in kernels/.hashes.json, then
    hand off to torch.utils.cpp_extension.load() (which drives ninja for the
    real incremental build) under kernels/build/<ext_name>/. Note: only the
    *_launcher.cu / bindings.cpp translation units are ever passed here as
    `sources` -- the *_kernel.cuh headers they #include are automatically
    picked up by ninja's own header dependency scan, so a kernel.cuh-only
    edit still triggers a real rebuild even though it's never listed here.

    ptx_kernels: optional {label: "<group>/<name>_kernel.cu"} map (paths
    relative to kernels/) -- one entry per kernel GROUP this module
    bundles (7 for tria, 1 for everything else). Drives both the progress
    counter (advances by len(ptx_kernels) at once) and the silent per-group
    PTX dump described in the module docstring.

    Returns the loaded module, or raises -- callers are expected to wrap this
    in their own try/except and fall back to PyTorch on failure, exactly as
    the old load_inline() call sites did.
    """
    global _progress_done
    _configure_visible_cuda_arches()
    from torch.utils.cpp_extension import load as _cpp_load

    abs_sources = [s if os.path.isabs(s) else os.path.join(_KERNELS_DIR, s) for s in sources]
    for s in abs_sources:
        if not os.path.exists(s):
            raise FileNotFoundError(f"kernels.build_or_load({ext_name!r}): missing source {s}")

    group_count = max(1, len(ptx_kernels) if ptx_kernels else 1)

    dependency_sources = _source_dependencies(abs_sources)
    current = {
        os.path.relpath(s, _KERNELS_DIR): _sha256_file(s)
        for s in dependency_sources
    }
    ext_lock = _extension_lock(ext_name)
    with ext_lock:
        with _STATE_LOCK:
            db = _load_hash_db()
            changed = db.get(ext_name) != current
            total = _progress_total_count()
            start = min(_progress_done + 1, total)
            end = min(_progress_done + group_count, total)
            _progress_done = end
            counter = f"{start}/{total}" if start == end else f"{start}-{end}/{total}"

        build_dir = os.path.join(_BUILD_DIR, ext_name)
        os.makedirs(build_dir, exist_ok=True)

        groups_label = ""
        if ptx_kernels and len(ptx_kernels) > 1:
            groups_label = " [" + ", ".join(ptx_kernels.keys()) + "]"

        verb = "compiling" if changed else "loading (cached)"
        _status(f"{counter} {verb} {ext_name}{groups_label}...")
        module = _cpp_load(
            name=ext_name,
            sources=abs_sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            build_directory=build_dir,
            verbose=bool(os.environ.get("KERNELS_VERBOSE")),
        )
        verb_done = "compiled" if changed else "loaded (cached)"
        _status(f"{counter} {verb_done} {ext_name}{groups_label}", done=True)

        with _STATE_LOCK:
            db = _load_hash_db()
            db[ext_name] = current
            _save_hash_db(db)

        if changed and ptx_kernels and _ptx_dump_enabled():
            for kernel_cu_rel in ptx_kernels.values():
                _dump_ptx_silent(kernel_cu_rel)

    return module
