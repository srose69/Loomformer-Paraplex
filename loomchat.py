#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive chat and inference for packaged LoomFormer models."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, fields, replace
from typing import Deque, Dict, List, Optional, Tuple

import torch

import loomformer as lf

# ============================================================================
# terminal color: NO_COLOR / non-tty / TERM=dumb all fall back to plain text,
# quietly -- no error, no warning, just no escape codes.
# ============================================================================


def _color_supported() -> bool:
    if os.environ.get("NO_COLOR") is not None:  # https://no-color.org/
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("TERM", "") in ("", "dumb"):
        return False
    return True


class _Colors:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def dim(self, text: str) -> str: return self._wrap("2", text)
    def bold(self, text: str) -> str: return self._wrap("1", text)
    def cyan(self, text: str) -> str: return self._wrap("36", text)
    def green(self, text: str) -> str: return self._wrap("32", text)
    def yellow(self, text: str) -> str: return self._wrap("33", text)
    def magenta(self, text: str) -> str: return self._wrap("35", text)
    def red(self, text: str) -> str: return self._wrap("31", text)
    def gray(self, text: str) -> str: return self._wrap("90", text)


COLOR = _Colors(_color_supported())

BANNER = """+------------------------------+
| #       ###    ###   #   #   |
| #      ##  #  ##  #  ## ##   |
| #      # # #  # # #  # # #   |
| #   #  #  ##  #  ##  #   #   |
| #####   ###    ###   #   #   |
+------------------------------+"""


# ============================================================================
# session settings -- the single source of truth for every runtime knob.
# `/settings` prints this; individual `/word value` commands mutate one field.
# ============================================================================

@dataclass
class Settings:
    device: str
    dtype: str          # "bf16" | "fp16" | "fp32"
    temperature: float
    top_k: int
    top_p: float
    max_new: int
    window: int          # Tria temporal refeed window (model.cfg.tria_temporal_window)
    alpha: float          # Tria carrier write-strength (model.cfg.tria_carrier_alpha)
    beta: float           # PolARM correction strength (model.cfg.tria_polarm_beta)
    kvstorage: str        # where prefix-KV snapshots live: same | cpu | cuda:N

    def torch_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype]


def _dtype_default_for(device: torch.device) -> str:
    # bf16 on a pre-Ampere/CPU device silently falls back to fp32 math anyway
    # (no bf16 tensor cores) -- default to fp32 there instead of paying autocast
    # overhead for nothing. See the Pascal/GTX-1080 note this codebase already
    # carries elsewhere.
    if device.type != "cuda":
        return "fp32"
    major, _ = torch.cuda.get_device_capability(device)
    return "bf16" if major >= 8 else "fp32"


# ============================================================================
# Esc-to-interrupt: a background thread doing raw single-key reads on POSIX
# terminals. Falls back to nothing (Ctrl-C/KeyboardInterrupt still works
# everywhere) when stdin isn't a real tty or termios isn't available (e.g.
# piped input, Windows) -- same "degrade quietly, never crash" policy as color
# support above.
# ============================================================================

class EscWatcher:
    def __init__(self) -> None:
        self.requested = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._enabled = sys.stdin.isatty() and os.name == "posix"

    def __enter__(self) -> "EscWatcher":
        if self._enabled:
            self.requested.clear()
            self._stop.clear()
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.2)

    def _watch(self) -> None:
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while not self._stop.is_set():
                    r, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if r and sys.stdin.read(1) == "\x1b":  # ESC
                        self.requested.set()
                        return
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            pass  # any raw-tty failure just means Esc-interrupt is unavailable this session


# ============================================================================
# sampling
# ============================================================================

def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """Sample from 1-D logits, using greedy selection at non-positive temperature."""
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits.float() / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        kth = torch.topk(logits, k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    if top_p and 0 < top_p < 1:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        cutoff = int(torch.searchsorted(cum, torch.tensor(top_p, device=cum.device)).item()) + 1
        sp = sp[:cutoff]
        si = si[:cutoff]
        sp = sp / sp.sum()
        choice = torch.multinomial(sp, 1)
        return int(si[choice].item())
    return int(torch.multinomial(probs, 1).item())


# ============================================================================
# streaming display (unchanged logic from the previous version -- this part
# was already solid: redecode-whole-turn-so-far, print only the new suffix,
# <think> dim, <tool_call> buffered until closed)
# ============================================================================

class StreamRenderer:
    def __init__(self) -> None:
        self.shown = ""
        self.in_think = False
        self.tc_buffer: Optional[str] = None

    def feed(self, full_raw: str) -> None:
        chunk = full_raw[len(self.shown):]
        self.shown = full_raw
        if not chunk:
            return
        if self.tc_buffer is not None:
            self.tc_buffer += chunk
            if "</tool_call>" in self.tc_buffer:
                payload, _, rest = self.tc_buffer.partition("</tool_call>")
                print(COLOR.yellow(f"\n  \u2192 tool_call: {payload.strip()}"), flush=True)
                self.tc_buffer = None
                self.feed_text(rest)
            return
        self.feed_text(chunk)

    def feed_text(self, chunk: str) -> None:
        while chunk:
            if not self.in_think and "<think>" in chunk:
                before, _, chunk = chunk.partition("<think>")
                self._print(before, dim=False)
                self.in_think = True
            elif self.in_think and "</think>" in chunk:
                before, _, chunk = chunk.partition("</think>")
                self._print(before, dim=True)
                self.in_think = False
            elif "<tool_call>" in chunk:
                before, _, chunk = chunk.partition("<tool_call>")
                self._print(before, dim=self.in_think)
                self.tc_buffer = ""
                if chunk:
                    self.feed(chunk)
                return
            else:
                self._print(chunk, dim=self.in_think)
                chunk = ""

    @staticmethod
    def _print(text: str, dim: bool) -> None:
        if not text:
            return
        print(COLOR.dim(text) if dim else text, end="", flush=True)


# ============================================================================
# AIO loading
# ============================================================================

AIO_FORMAT = "loom.aio"
AIO_VERSION = 1


def _special_id(tok, token: str) -> Optional[int]:
    fn = getattr(tok, "special_id", None)
    return fn(token) if fn is not None else None


class AIOChatTemplate:
    def __init__(self, tok, source: str) -> None:
        import jinja2
        self.tok = tok
        self._tpl = jinja2.Environment().from_string(source)
        im_start = _special_id(tok, "<|im_start|>")
        im_end = _special_id(tok, "<|im_end|>")
        if im_start is None or im_end is None:
            raise ValueError("AIO tokenizer lacks <|im_start|>/<|im_end|>")
        self.im_start_id = im_start
        self.im_end_id = im_end
        self.bos_id = _special_id(tok, "<bos>")
        self.bos_token = "<bos>" if self.bos_id is not None else ""
        eos_id = _special_id(tok, "<eos>")
        self.stop_ids = {i for i in (im_end, eos_id) if i is not None}
        self._assistant_header_ids = [im_start] + tok.encode("assistant\n")

    def render_text(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                    add_generation_prompt: bool = False) -> str:
        kwargs = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self.bos_token,
        }
        if tools is not None:
            kwargs["tools"] = tools
        return self._tpl.render(**kwargs)

    def render_prompt_ids(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> List[int]:
        return self.tok.encode(self.render_text(messages, tools=tools, add_generation_prompt=True))

    def parse_tool_calls(self, text: str) -> List[Dict]:
        return lf.ChatTemplate.parse_tool_calls(self, text)


def _archive_dtype(package: Dict) -> str:
    target = str(package.get("manifest", {}).get("quantization", {}).get("target_dtype", "none"))
    return target if target in ("bf16", "fp16", "fp32") else "fp32"


def load_aio(path: str, device: torch.device):
    if not str(path).lower().endswith(".aio"):
        raise ValueError("loomchat accepts only .aio archives")
    package = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(package, dict) or package.get("format") != AIO_FORMAT:
        raise ValueError(f"not a {AIO_FORMAT} archive")
    if int(package.get("version", -1)) != AIO_VERSION:
        raise ValueError(f"unsupported AIO version {package.get('version')!r}")
    checkpoint = package.get("checkpoint")
    tokenizer_json = package.get("tokenizer_json")
    template_jinja = package.get("chat_template_jinja")
    if not isinstance(checkpoint, dict):
        raise ValueError("AIO archive has no checkpoint")
    if not isinstance(tokenizer_json, (bytes, bytearray)):
        raise ValueError("AIO archive has no tokenizer JSON")
    if not isinstance(template_jinja, (bytes, bytearray)):
        raise ValueError("AIO archive has no chat template")

    from tokenizers import Tokenizer
    cfg = lf.Config.from_checkpoint_dict(checkpoint["cfg"])
    lf.apply_config(cfg)
    tok = lf.BPETokenizerWrap(Tokenizer.from_str(bytes(tokenizer_json).decode("utf-8")))
    cfg.vocab = tok.vocab_size
    lf.CARRY_TOKEN_ID = _special_id(tok, "<CARRY>")

    ablation = bool(checkpoint.get("ablation", False))
    model = lf.Model(cfg, ablation=ablation)
    if checkpoint.get("model_kind") != "loomformer":
        raise ValueError("AIO checkpoint is not a LoomFormer model")
    if checkpoint.get("ffn_type") != "paraplex":
        raise ValueError("AIO checkpoint is not a Paraplex model")
    state = lf.canonicalize_model_state_dict(checkpoint["model"])
    model.load_state_dict(state, strict=True, assign=True)
    if bool(getattr(cfg, "tied_embeddings", True)):
        model.head.weight = model.emb.weight
    model.to(device=device)
    model.eval()
    chat = AIOChatTemplate(tok, bytes(template_jinja).decode("utf-8"))
    return model, tok, chat, cfg, package.get("manifest", {}), _archive_dtype(package)


def move_model(model: torch.nn.Module, device: torch.device,
               dtype: Optional[torch.dtype] = None) -> torch.nn.Module:
    model = model.to(device=device) if dtype is None else model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _autocast(settings: Settings):
    device = torch.device(settings.device)
    if device.type != "cuda" or settings.dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if settings.dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@dataclass
class KVCalibration:
    prefill_tps: float
    decode_tps: float
    chunk_size: int
    preload_chunks: int
    transfer_gbps: float
    peer_access: Optional[bool] = None
    copy_us: float = 0.0
    consume_us: float = 0.0


def _median_cuda_us(device: torch.device, fn, repeats: int = 7) -> float:
    values: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        values.append((time.perf_counter() - start) * 1e6)
    values.sort()
    return values[len(values) // 2]


def calibrate_kv_runtime(
    model,
    settings: Settings,
    storage_device: torch.device,
) -> KVCalibration:
    """Calibrate the real model and the selected KV transport.

    The model measurements deliberately use a full 2048-token forward (or the
    configured context when shorter) plus incremental ``step`` TPS. CPU KV then
    benchmarks the exact streamed attention implementation for candidate chunk
    sizes. The measured no-starvation preload is padded by two chunks.
    """
    compute = torch.device(settings.device)
    calib_len = min(2048, int(lf.SEQ_LEN))
    if compute.type != "cuda":
        return KVCalibration(0.0, 0.0, calib_len, 1, 0.0, None)

    probe = torch.zeros((1, calib_len), dtype=torch.long, device=compute)
    warm_len = min(64, calib_len)
    with torch.inference_mode(), _autocast(settings):
        _ = model(probe[:, :warm_len])
    torch.cuda.synchronize(compute)
    start = time.perf_counter()
    with torch.inference_mode(), _autocast(settings):
        full_out = model(probe)
    torch.cuda.synchronize(compute)
    prefill_seconds = max(time.perf_counter() - start, 1e-9)
    del full_out
    prefill_tps = calib_len / prefill_seconds

    # Measure actual Python/model.step decode throughput, not a FLOP estimate.
    step_count = min(32, calib_len)
    state = None
    torch.cuda.synchronize(compute)
    start = time.perf_counter()
    with torch.inference_mode(), _autocast(settings):
        for pos in range(step_count):
            _, state = model.step(probe[:, pos], pos, state)
    torch.cuda.synchronize(compute)
    decode_tps = step_count / max(time.perf_counter() - start, 1e-9)
    del state

    if storage_device.type == "cuda":
        src_idx = compute.index if compute.index is not None else torch.cuda.current_device()
        dst_idx = storage_device.index if storage_device.index is not None else 0
        peer = bool(torch.cuda.can_device_access_peer(src_idx, dst_idx)) if src_idx != dst_idx else True
        if storage_device == compute:
            return KVCalibration(prefill_tps, decode_tps, calib_len, 1, 0.0, peer)

        # End-to-end activation shuttle (Q/Knew/Vnew out, c/kctx back).
        outgoing = torch.empty(
            (1, 1, lf.N + 2 * lf.KV_DIM), dtype=settings.torch_dtype(), device=compute)
        def remote_roundtrip() -> None:
            remote = outgoing.to(storage_device, non_blocking=True)
            _ = remote[:, :, :2 * lf.N].to(compute, non_blocking=True)
        shuttle_us = _median_cuda_us(compute, remote_roundtrip)
        shuttle_bytes = outgoing.numel() * outgoing.element_size() + 2 * lf.N * outgoing.element_size()
        gbps = shuttle_bytes / max(shuttle_us, 1e-9) / 1e3
        return KVCalibration(prefill_tps, decode_tps, calib_len, 1, gbps, peer)

    # CPU storage: use pinned memory and benchmark both transport and the exact
    # online-attention pipeline. Candidate sizes are geometry-independent.
    attn = model.blocks[0].attn
    stride = int(getattr(attn, "token_stride", 1))
    physical_len = max(1, math.ceil(calib_len / stride))
    candidates = [
        c for c in (128, 256, 512, 1024, 2048, 4096)
        if c <= physical_len
    ]
    if physical_len not in candidates:
        candidates.append(physical_len)
    candidates = sorted(set(candidates))
    dtype = settings.torch_dtype()
    best: Optional[Tuple[float, int, int, float, float, float]] = None

    for chunk in candidates:
        n_chunks = math.ceil(physical_len / chunk)
        host_k = torch.zeros(
            (1, physical_len, lf.N_KV_HEADS, lf.HEAD_DIM),
            dtype=dtype, pin_memory=True)
        host_v = torch.zeros(
            host_k.shape, dtype=dtype, pin_memory=True)
        gpu_k = torch.empty(
            (1, chunk, lf.N_KV_HEADS, lf.HEAD_DIM), dtype=dtype, device=compute)
        gpu_v = torch.empty_like(gpu_k)

        copy_us = _median_cuda_us(
            compute,
            lambda: (
                gpu_k.copy_(host_k[:, :chunk], non_blocking=True),
                gpu_v.copy_(host_v[:, :chunk], non_blocking=True),
            ),
            repeats=5,
        )
        q = torch.randn(
            1, 1, lf.N_Q_HEADS, lf.HEAD_DIM, dtype=dtype, device=compute)
        qg = q.transpose(1, 2)
        compute_us = _median_cuda_us(
            compute,
            lambda: attn._online_chunk(qg, gpu_k, gpu_v, None),
            repeats=5,
        )
        # Bound the synthetic-kernel result by both measured whole-model rates.
        # Taking the minimum is intentionally conservative: calibration must
        # never claim more overlap budget than either real execution mode
        # demonstrated on this machine.
        active_layers = len(getattr(model.cfg, "attn_layers", ())) or int(lf.LAYERS)
        decode_layer_us = 1e6 / max(decode_tps * active_layers, 1e-9)
        prefill_layer_us = (
            chunk * stride / max(prefill_tps * active_layers, 1e-9) * 1e6)
        overlap_us = min(compute_us, decode_layer_us, prefill_layer_us)
        if copy_us <= overlap_us:
            required = 1
        else:
            required = math.ceil(
                n_chunks - (n_chunks - 1) * overlap_us / copy_us)
        preload = min(n_chunks, max(1, required + 2))

        runtime = lf.InferenceKVRuntime(
            storage_device,
            compute,
            chunk_size=chunk,
            preload_chunks=preload,
            copy_us=copy_us,
            consume_us=overlap_us,
        )
        k_new = torch.zeros(
            (1, 1, lf.N_KV_HEADS, lf.HEAD_DIM), dtype=dtype, device=compute)
        v_new = torch.zeros_like(k_new)
        pipeline_us = _median_cuda_us(
            compute,
            lambda: attn._step_cpu_kv(
                q, k_new, v_new, host_k, host_v, physical_len - 1, runtime),
            repeats=3,
        )
        transfer_bytes = 2 * chunk * lf.N_KV_HEADS * lf.HEAD_DIM * torch.empty(
            (), dtype=dtype).element_size()
        gbps = transfer_bytes / max(copy_us, 1e-9) / 1e3
        candidate = (pipeline_us, chunk, preload, gbps, copy_us, overlap_us)
        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None
    _, chunk, preload, gbps, copy_us, overlap_us = best
    return KVCalibration(
        prefill_tps, decode_tps, chunk, preload, gbps, None,
        copy_us=copy_us, consume_us=overlap_us)


# ============================================================================
# prefix KV cache
#
# Every turn re-renders the whole conversation, so prefilling it from scratch
# each time is O(turns^2) work over tokens whose KV was already computed one
# turn ago -- the dominant cost of an agentic tool loop. A snapshot is keyed by
# the exact token ids it consumed, so a turn only pays for its new suffix.
#
# Snapshots are trimmed to their used length (the model preallocates every KV
# buffer at SEQ_LEN) and can be parked on another device via --kvstorage, which
# keeps the compute GPU's VRAM free of conversation history it isn't reading.
# ============================================================================

@dataclass
class _KVSnapshot:
    ids: List[int]           # exact prefix this state consumed
    states: Tuple            # (caches, tria_ca_cache, tria_temporal_state), trimmed
    nbytes: int


def resolve_kv_device(spec: Optional[str], compute: torch.device) -> torch.device:
    """Resolve --kvstorage: 'same'/'auto' keeps snapshots on the compute device."""
    s = str(spec or "same").strip().lower()
    if s in ("", "same", "auto", "model"):
        return compute
    try:
        dev = torch.device(s)
    except RuntimeError as e:
        # torch.device reports malformed strings as RuntimeError, while callers
        # (notably the interactive /kvstorage command) use ValueError for
        # user-facing validation failures.
        raise ValueError(f"invalid kvstorage device {spec!r}: {e}") from e
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, cannot store KV there")
        idx = 0 if dev.index is None else int(dev.index)
        if idx >= torch.cuda.device_count():
            raise ValueError(f"cuda:{idx} does not exist ({torch.cuda.device_count()} visible)")
        return torch.device(f"cuda:{idx}")
    if dev.type != "cpu":
        raise ValueError(f"kvstorage must be cpu, cuda[:N] or same, got {spec!r}")
    return dev


def _copy_to(t: Optional[torch.Tensor], device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    out = t.to(device, copy=True)
    # see _trim_to: only pin for CUDA DMA when a driver is present.
    if device.type == "cpu" and torch.cuda.is_available() and not out.is_pinned():
        return out.pin_memory()
    return out


def _trim_to(t: Optional[torch.Tensor], used: int, device: torch.device) -> Optional[torch.Tensor]:
    if t is None:
        return None
    out = t[:, :int(used)].to(device, copy=True)
    # pin_memory() stages the tensor for CUDA DMA; it requires a working CUDA
    # driver and is pointless without one, so only pin when CUDA is actually
    # available (otherwise a CPU-only host raises "found no NVIDIA driver").
    if device.type == "cpu" and torch.cuda.is_available() and not out.is_pinned():
        return out.pin_memory()
    return out


def _expand_to(t: Optional[torch.Tensor], seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
    """Re-inflate a trimmed buffer: step() writes in place at index cache_len."""
    if t is None:
        return None
    full = torch.zeros(
        (t.shape[0], seq_len, *t.shape[2:]),
        dtype=t.dtype,
        device=device,
        pin_memory=device.type == "cpu",
    )
    if t.shape[1]:
        full[:, :t.shape[1]] = t.to(device)
    return full


def _state_nbytes(states: Tuple) -> int:
    total = 0
    for group in states:
        for obj in (group if isinstance(group, list) else [group]):
            if obj is None:
                continue
            for f in fields(obj):
                t = getattr(obj, f.name)
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
                elif isinstance(t, tuple):
                    total += sum(
                        x.numel() * x.element_size()
                        for x in t if isinstance(x, torch.Tensor))
    return total


def snapshot_states(states: Tuple, device: torch.device) -> Tuple:
    caches, ca_cache, temporal = states
    snap_caches = [
        lf.LayerCache(
            k=_trim_to(c.k, c.cache_pos, device),
            v=_trim_to(c.v, c.cache_pos, device),
            phase_trace=_copy_to(c.phase_trace, device),
            cache_pos=int(c.cache_pos),
            cache_capacity=int(c.cache_capacity),
            attn_context=(
                None if c.attn_context is None
                else tuple(_copy_to(x, device) for x in c.attn_context)
            ),
        )
        for c in caches
    ]
    snap_ca = None if ca_cache is None else replace(
        ca_cache,
        k=_trim_to(ca_cache.k, ca_cache.cache_len, device),
        v=_trim_to(ca_cache.v, ca_cache.cache_len, device),
        carry_key_mask=_trim_to(ca_cache.carry_key_mask, ca_cache.cache_len, device),
    )
    snap_temporal = None if temporal is None else replace(
        temporal,
        carry=_copy_to(temporal.carry, device),
        refeed_pending=_copy_to(temporal.refeed_pending, device),
    )
    return snap_caches, snap_ca, snap_temporal


def restore_states(
    states: Tuple,
    device: torch.device,
    seq_len: int,
    kv_device: Optional[torch.device] = None,
) -> Tuple:
    caches, ca_cache, temporal = states
    kv_device = device if kv_device is None else torch.device(kv_device)
    live_caches = [
        lf.LayerCache(
            k=_expand_to(c.k, c.cache_capacity or seq_len, kv_device),
            v=_expand_to(c.v, c.cache_capacity or seq_len, kv_device),
            phase_trace=_copy_to(c.phase_trace, device),
            cache_pos=int(c.cache_pos),
            cache_capacity=int(c.cache_capacity),
            attn_context=(
                None if c.attn_context is None
                else tuple(_copy_to(x, device) for x in c.attn_context)
            ),
        )
        for c in caches
    ]
    live_ca = None if ca_cache is None else replace(
        ca_cache,
        k=_expand_to(ca_cache.k, seq_len, device),
        v=_expand_to(ca_cache.v, seq_len, device),
        carry_key_mask=_expand_to(ca_cache.carry_key_mask, seq_len, device),
    )
    live_temporal = None if temporal is None else replace(
        temporal,
        carry=_copy_to(temporal.carry, device),
        refeed_pending=_copy_to(temporal.refeed_pending, device),
    )
    return live_caches, live_ca, live_temporal


class PrefixKVCache:
    """Token-id-keyed prefix cache over Model.step states.

    Two snapshots are kept per conversation: the one taken right after a turn's
    prefill and the one after its generation. The second covers the common case
    (the next turn strictly extends it); the first is the fallback for when the
    assistant's own text does not re-tokenize identically once the template
    folds it back into the history.
    """

    def __init__(
        self,
        storage: Optional[str],
        compute: torch.device,
        max_entries: int = 2,
        chunk_size: int = 1024,
        preload_chunks: int = 1,
        copy_us: float = 0.0,
        consume_us: float = 0.0,
    ) -> None:
        self.compute = compute
        self.device = resolve_kv_device(storage, compute)
        self.chunk_size = int(chunk_size)
        self.preload_chunks = int(preload_chunks)
        self.copy_us = float(copy_us)
        self.consume_us = float(consume_us)
        self.runtime = lf.InferenceKVRuntime(
            self.device, self.compute, self.chunk_size, self.preload_chunks,
            self.copy_us, self.consume_us)
        self._snaps: Deque[_KVSnapshot] = deque(maxlen=max_entries)

    def clear(self) -> None:
        self._snaps.clear()

    def retarget(self, storage: Optional[str], compute: torch.device) -> None:
        """Point at a new storage/compute device. Cached states are dropped:
        they were produced by the previous device/dtype/Tria settings."""
        self.compute = compute
        self.device = resolve_kv_device(storage, compute)
        self.runtime = lf.InferenceKVRuntime(
            self.device, self.compute, self.chunk_size, self.preload_chunks,
            self.copy_us, self.consume_us)
        self.clear()

    def configure_streaming(
        self,
        chunk_size: int,
        preload_chunks: int,
        copy_us: float = 0.0,
        consume_us: float = 0.0,
    ) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.preload_chunks = max(1, int(preload_chunks))
        self.copy_us = max(0.0, float(copy_us))
        self.consume_us = max(0.0, float(consume_us))
        self.runtime = lf.InferenceKVRuntime(
            self.device, self.compute, self.chunk_size, self.preload_chunks,
            self.copy_us, self.consume_us)
        self.clear()

    def nbytes(self) -> int:
        return sum(s.nbytes for s in self._snaps)

    def reuse(self, ids: List[int]) -> Tuple[Optional[Tuple], int]:
        """Longest cached prefix of ``ids``, restored onto the compute device.

        A snapshot covering all of ``ids`` is rejected: at least one token must
        be fed for the caller to get logits, and states cannot be rewound.
        """
        best: Optional[_KVSnapshot] = None
        for snap in self._snaps:
            n = len(snap.ids)
            if 0 < n < len(ids) and snap.ids == ids[:n] and (best is None or n > len(best.ids)):
                best = snap
        if best is None:
            return None, 0
        return (
            restore_states(best.states, self.compute, lf.SEQ_LEN, self.device),
            len(best.ids),
        )

    def store(self, ids: List[int], states: Tuple) -> None:
        snap = snapshot_states(states, self.device)
        self._snaps.append(_KVSnapshot(ids=list(ids), states=snap, nbytes=_state_nbytes(snap)))


# ============================================================================
# generation
# ============================================================================

def generate_turn(model, tok, chat: AIOChatTemplate, messages: List[Dict],
                   settings: Settings, esc: Optional[EscWatcher] = None,
                   kv: Optional[PrefixKVCache] = None) -> Tuple[Dict, bool]:
    """Stream one assistant turn and return its message and interruption status."""
    device = torch.device(settings.device)
    ids = chat.render_prompt_ids(messages)
    room = lf.SEQ_LEN - len(ids)
    if room <= 0:
        raise RuntimeError(
            f"conversation ({len(ids)} tokens) already fills the model's context "
            f"(seq_len={lf.SEQ_LEN}); the incremental cache has no wraparound. /reset."
        )
    max_new = min(settings.max_new, room)
    states, start = kv.reuse(ids) if kv is not None else (None, 0)
    if start:
        print(COLOR.gray(f"[kv {start}/{len(ids)} reused from {kv.device}] "), end="", flush=True)
    logits = None
    with torch.inference_mode(), _autocast(settings):
        for pos in range(start, len(ids)):
            x = torch.tensor([int(ids[pos])], device=device, dtype=torch.long)
            logits, states = model.step(
                x, pos, states, kv_runtime=None if kv is None else kv.runtime)
        if kv is not None:
            kv.store(ids, states)

        renderer = StreamRenderer()
        gen_ids: List[int] = []
        interrupted = False
        for i in range(max_new):
            if esc is not None and esc.requested.is_set():
                interrupted = True
                break
            nxt = sample_next(logits[0], settings.temperature, settings.top_k, settings.top_p)
            if nxt in chat.stop_ids:
                break
            gen_ids.append(nxt)
            renderer.feed(tok.decode(gen_ids, skip_special_tokens=False))
            x = torch.tensor([nxt], device=device, dtype=torch.long)
            logits, states = model.step(
                x, len(ids) + i, states, kv_runtime=None if kv is None else kv.runtime)
        if kv is not None and gen_ids:
            kv.store(ids + gen_ids, states)
    print()
    if interrupted:
        print(COLOR.gray("  (interrupted -- Esc)"))

    raw_text = tok.decode(gen_ids, skip_special_tokens=False)
    tool_calls = chat.parse_tool_calls(raw_text)
    if tool_calls:
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}, interrupted
    clean_text = tok.decode(gen_ids, skip_special_tokens=True)
    return {"role": "assistant", "content": clean_text}, interrupted


# ============================================================================
# banner / settings display
# ============================================================================

def print_banner(aio_path: str, n_params: int, settings: Settings, manifest: Dict, cfg) -> None:
    print(COLOR.cyan(BANNER))
    quant = manifest.get("quantization", {}).get("target_dtype", "none")
    print(COLOR.dim(f"  archive: {aio_path}  ({n_params:,} params, packed={quant})"))
    print(COLOR.dim(f"  device={settings.device}  dtype={settings.dtype}  "
                     f"window={settings.window}  alpha={settings.alpha:g}  beta={settings.beta:g}"))
    print(COLOR.dim(
        f"  kvstorage={settings.kvstorage}  "
        "(persistent KV placement and prefix reuse across turns)"))
    active_layers = len(getattr(cfg, "attn_layers", ())) or int(cfg.layers)
    stride = int(getattr(cfg, "attn_token_stride", 1))
    physical = math.ceil(int(cfg.seq_len) / stride)
    print(COLOR.dim(
        f"  attention={active_layers}/{int(cfg.layers)} layers  "
        f"logical={int(cfg.seq_len)}  physical_kv<={physical}  stride={stride}"))
    print(COLOR.dim("  /help for commands · type / to browse them · Esc interrupts a reply\n"))


COMMANDS = {
    "/help":      "show this list",
    "/settings":  "show every current setting",
    "/device":    "/device <cpu|cuda:0|cuda:1|...> -- move the model, reload nothing else",
    "/dtype":     "/dtype <bf16|fp16|fp32>",
    "/window":    "/window <int> -- Tria temporal refeed window (config SSOT)",
    "/alpha":     "/alpha <float> -- Tria carrier write-strength (config SSOT)",
    "/beta":      "/beta <float 0..1) -- PolARM correction strength (config SSOT)",
    "/temperature": "/temperature <float>  (0 = greedy)",
    "/top-k":     "/top-k <int>  (0 = disabled)",
    "/top-p":     "/top-p <float 0..1>",
    "/max-new":   "/max-new <int> -- cap on tokens generated per turn",
    "/kvstorage": "/kvstorage <same|cpu|cuda:N> -- persistent KV/attention placement",
    "/system":    "/system <text> -- set/replace the system prompt",
    "/reset":     "clear conversation history (keeps the system prompt)",
    "/reload":    "/reload <model.aio> -- swap archives without restarting",
    "/exit":      "leave (also /quit)",
}


def print_help() -> None:
    width = max(len(k) for k in COMMANDS)
    for cmd, desc in COMMANDS.items():
        print(f"  {COLOR.cyan(cmd.ljust(width))}  {COLOR.dim(desc)}")


def print_command_menu(prefix: str) -> None:
    matches = [c for c in COMMANDS if c.startswith(prefix)]
    if not matches:
        return
    width = max(len(c) for c in matches)
    print(COLOR.gray("  " + "   ".join(c.ljust(width) for c in matches)))


def print_settings(settings: Settings) -> None:
    for f in fields(settings):
        print(f"  {COLOR.cyan(f.name.ljust(12))} {getattr(settings, f.name)}")


# ============================================================================
# command dispatch
# ============================================================================

# Settings whose change makes every cached state stale: the states were
# produced on a specific device/dtype and under specific Tria parameters.
_KV_INVALIDATING = frozenset({"device", "dtype", "window", "alpha", "beta", "kvstorage"})


def apply_setting(name: str, value: str, settings: Settings, model,
                   kv: Optional[PrefixKVCache] = None) -> Optional[str]:
    """Apply a runtime setting and return a validation error or ``None``."""
    try:
        if name == "device":
            new_device = torch.device(value)
            if new_device.type == "cuda" and not torch.cuda.is_available():
                return "CUDA is not available in this environment"
            move_model(model, new_device)
            settings.device = value
        elif name == "kvstorage":
            resolve_kv_device(value, torch.device(settings.device))  # validate before committing
            settings.kvstorage = value
        elif name == "dtype":
            if value not in ("bf16", "fp16", "fp32"):
                return "dtype must be one of: bf16, fp16, fp32"
            new_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[value]
            move_model(model, torch.device(settings.device), new_dtype)
            settings.dtype = value
        elif name == "window":
            new_window = int(value)
            if new_window <= 0:
                return "window must be > 0"
            settings.window = new_window
            model.cfg.tria_temporal_window = new_window
        elif name == "alpha":
            new_alpha = float(value)
            if not math.isfinite(new_alpha) or new_alpha <= 0.0:
                return "alpha must be finite and > 0"
            settings.alpha = new_alpha
            model.cfg.tria_carrier_alpha = new_alpha
        elif name == "beta":
            new_beta = float(value)
            if not math.isfinite(new_beta) or new_beta < 0.0 or new_beta >= 1.0:
                return "beta must be finite and in [0, 1)"
            settings.beta = new_beta
            model.cfg.tria_polarm_beta = new_beta
        elif name == "temperature":
            settings.temperature = float(value)
        elif name == "top_k":
            settings.top_k = int(value)
        elif name == "top_p":
            settings.top_p = float(value)
        elif name == "max_new":
            settings.max_new = int(value)
        else:
            return f"unknown setting: {name}"
    except ValueError as e:
        return str(e) if name == "kvstorage" else f"couldn't parse {value!r} for {name}"
    if kv is not None and name in _KV_INVALIDATING:
        kv.retarget(settings.kvstorage, torch.device(settings.device))
        if name in {"device", "dtype", "kvstorage"} and kv.device != kv.compute:
            print(COLOR.dim("  (recalibrating KV transport...)"))
            calibration = calibrate_kv_runtime(model, settings, kv.device)
            kv.configure_streaming(
                calibration.chunk_size, calibration.preload_chunks,
                calibration.copy_us, calibration.consume_us)
    return None


# ============================================================================
# chat loop
# ============================================================================

def run_chat(aio_path: str, settings: Settings, system: Optional[str],
             dtype_override: Optional[str]) -> None:
    device = torch.device(settings.device)
    model, tok, chat, cfg, manifest, packed_dtype = load_aio(aio_path, device)
    dtype_forced = dtype_override is not None
    if dtype_forced:
        model = move_model(model, device, settings.torch_dtype())
    else:
        settings.dtype = packed_dtype
    settings.window = settings.window or int(cfg.tria_temporal_window)
    cfg.tria_temporal_window = settings.window
    settings.alpha = float(getattr(cfg, "tria_carrier_alpha", settings.alpha))
    cfg.tria_carrier_alpha = settings.alpha
    settings.beta = float(getattr(cfg, "tria_polarm_beta", settings.beta))
    cfg.tria_polarm_beta = settings.beta

    kv_device = resolve_kv_device(settings.kvstorage, device)
    if kv_device != device:
        print(COLOR.dim(
            f"  calibrating KV backend ({device} compute, {kv_device} storage; "
            f"{min(2048, int(lf.SEQ_LEN))}-token model forward)..."),
            flush=True,
        )
        calibration = calibrate_kv_runtime(model, settings, kv_device)
    else:
        calibration = KVCalibration(0.0, 0.0, int(lf.SEQ_LEN), 1, 0.0, True)
    kv = PrefixKVCache(
        settings.kvstorage,
        device,
        chunk_size=calibration.chunk_size,
        preload_chunks=calibration.preload_chunks,
        copy_us=calibration.copy_us,
        consume_us=calibration.consume_us,
    )

    print_banner(aio_path, lf.count_params(model), settings, manifest, cfg)
    if kv_device != device:
        peer_text = (
            "" if calibration.peer_access is None
            else f"  peer={'yes' if calibration.peer_access else 'host-staged'}"
        )
        print(COLOR.dim(
            f"  KV auto: prefill={calibration.prefill_tps:,.1f} tok/s  "
            f"decode={calibration.decode_tps:,.1f} tok/s  "
            f"chunk={calibration.chunk_size}  "
            f"preload@2048={calibration.preload_chunks} (dynamic,+2)  "
            f"transport={calibration.transfer_gbps:.2f} GB/s{peer_text}"
        ))
    messages: List[Dict] = []
    if system:
        messages.append({"role": "system", "content": system})

    while True:
        try:
            user_text = input(COLOR.bold(COLOR.green("you> ")))
        except (EOFError, KeyboardInterrupt):
            print()
            break
        user_text = user_text.strip()
        if not user_text:
            continue

        if user_text == "/":
            print_command_menu("")
            continue
        if user_text.startswith("/") and " " not in user_text and user_text not in (
                "/help", "/settings", "/reset", "/exit", "/quit"):
            matches = [c for c in COMMANDS if c.startswith(user_text)]
            if len(matches) != 1:
                print_command_menu(user_text)
                continue
            user_text = matches[0]

        if user_text in ("/exit", "/quit"):
            break
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/settings":
            print_settings(settings)
            continue
        if user_text == "/reset":
            messages = [messages[0]] if messages and messages[0]["role"] == "system" else []
            kv.clear()
            print(COLOR.dim("(history cleared)"))
            continue
        if user_text.startswith("/system "):
            new_sys = user_text[len("/system "):]
            if messages and messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": new_sys}
            else:
                messages.insert(0, {"role": "system", "content": new_sys})
            kv.clear()  # the system prompt is the very head of every prefix
            print(COLOR.dim("(system prompt set)"))
            continue
        if user_text.startswith("/reload "):
            new_aio = user_text[len("/reload "):].strip()
            try:
                new_model, new_tok, new_chat, new_cfg, new_manifest, new_packed_dtype = load_aio(
                    new_aio, torch.device(settings.device))
                if dtype_forced:
                    new_model = move_model(new_model, torch.device(settings.device), settings.torch_dtype())
                else:
                    settings.dtype = new_packed_dtype
                model, tok, chat, cfg, manifest = new_model, new_tok, new_chat, new_cfg, new_manifest
                settings.window = int(cfg.tria_temporal_window)
                settings.alpha = float(getattr(cfg, "tria_carrier_alpha", settings.alpha))
                settings.beta = float(getattr(cfg, "tria_polarm_beta", settings.beta))
                kv.retarget(settings.kvstorage, torch.device(settings.device))
                if kv.device != kv.compute:
                    print(COLOR.dim("  (recalibrating KV transport...)"))
                    calibration = calibrate_kv_runtime(model, settings, kv.device)
                    kv.configure_streaming(
                        calibration.chunk_size, calibration.preload_chunks,
                        calibration.copy_us, calibration.consume_us)
                aio_path = new_aio
                print(COLOR.dim(f"(reloaded {new_aio})"))
            except Exception as e:
                print(COLOR.red(f"  ! reload failed: {e}"))
            continue

        handled_setting = False
        for key, attr in (("/device", "device"), ("/dtype", "dtype"), ("/window", "window"),
                          ("/alpha", "alpha"), ("/beta", "beta"), ("/temperature", "temperature"),
                          ("/top-k", "top_k"), ("/top-p", "top_p"), ("/max-new", "max_new"),
                          ("/kvstorage", "kvstorage")):
            if user_text.startswith(key + " "):
                err = apply_setting(attr, user_text[len(key) + 1:].strip(), settings, model, kv)
                if not err and attr == "dtype":
                    dtype_forced = True
                print(COLOR.red(f"  ! {err}") if err else COLOR.dim(f"({attr}={getattr(settings, attr)})"))
                handled_setting = True
                break
        if handled_setting:
            continue

        if user_text.startswith("/"):
            print(COLOR.red(f"  ! unknown command: {user_text}  (try /help)"))
            continue

        messages.append({"role": "user", "content": user_text})
        print(COLOR.bold(COLOR.magenta("loom> ")), end="", flush=True)
        try:
            with EscWatcher() as esc:
                reply, _ = generate_turn(model, tok, chat, messages, settings, esc, kv)
        except RuntimeError as e:
            print(COLOR.red(f"\n  ! {e}"))
            messages.pop()
            continue
        messages.append(reply)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="loomchat: interactive chat for LoomFormer .aio archives")
    ap.add_argument("archive", type=str, help="model.aio produced by loompack.py")
    ap.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0 | cuda:1 | ...")
    ap.add_argument("--dtype", type=str, default=None, choices=["bf16", "fp16", "fp32"],
                    help="override packed mixed precision and cast the whole model")
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--window", type=int, default=0, help="0 -> use the archive's Tria window")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--kvstorage", type=str, default="same", metavar="TARGET",
                    help="where KV remains during inference: same (compute device) | "
                         "cpu (pinned streaming) | cuda:N (attention executes there)")
    args = ap.parse_args()

    dev = lf.device_auto(args.device)
    resolve_kv_device(args.kvstorage, dev)  # fail at startup, not mid-conversation
    settings = Settings(
        device=str(dev),
        dtype=args.dtype or "fp32",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new=args.max_new,
        window=args.window,
        alpha=args.alpha,
        beta=args.beta,
        kvstorage=args.kvstorage,
    )
    run_chat(args.archive, settings, args.system, args.dtype)


if __name__ == "__main__":
    main()
