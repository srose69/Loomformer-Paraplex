#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interactive chat and inference for packaged LoomFormer models."""

from __future__ import annotations

import argparse
import contextlib
import os
import select
import sys
import threading
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

import torch

import loomformer as lf
import tria as tria_mod

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
    window: int          # Tria temporal refeed window (model.tria_temporal_window)
    alpha: float          # Tria carrier write-strength (tria.set_carrier_alpha)

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
    model = lf.Model(ablation=ablation)
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


# ============================================================================
# generation
# ============================================================================

def generate_turn(model, tok, chat: AIOChatTemplate, messages: List[Dict],
                   settings: Settings, esc: Optional[EscWatcher] = None) -> Tuple[Dict, bool]:
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
    states = None
    logits = None
    with torch.inference_mode(), _autocast(settings):
        for pos, tid in enumerate(ids):
            x = torch.tensor([int(tid)], device=device, dtype=torch.long)
            logits, states = model.step(x, pos, states)

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
            logits, states = model.step(x, len(ids) + i, states)
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

def print_banner(aio_path: str, n_params: int, settings: Settings, manifest: Dict) -> None:
    print(COLOR.cyan(BANNER))
    quant = manifest.get("quantization", {}).get("target_dtype", "none")
    print(COLOR.dim(f"  archive: {aio_path}  ({n_params:,} params, packed={quant})"))
    print(COLOR.dim(f"  device={settings.device}  dtype={settings.dtype}  "
                     f"window={settings.window}  alpha={settings.alpha:g}"))
    print(COLOR.dim("  /help for commands · type / to browse them · Esc interrupts a reply\n"))


COMMANDS = {
    "/help":      "show this list",
    "/settings":  "show every current setting",
    "/device":    "/device <cpu|cuda:0|cuda:1|...> -- move the model, reload nothing else",
    "/dtype":     "/dtype <bf16|fp16|fp32>",
    "/window":    "/window <int> -- Tria temporal refeed window (model.tria_temporal_window)",
    "/alpha":     "/alpha <float> -- Tria carrier write-strength (tria.set_carrier_alpha)",
    "/temperature": "/temperature <float>  (0 = greedy)",
    "/top-k":     "/top-k <int>  (0 = disabled)",
    "/top-p":     "/top-p <float 0..1>",
    "/max-new":   "/max-new <int> -- cap on tokens generated per turn",
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

def apply_setting(name: str, value: str, settings: Settings, model) -> Optional[str]:
    """Apply a runtime setting and return a validation error or ``None``."""
    try:
        if name == "device":
            new_device = torch.device(value)
            if new_device.type == "cuda" and not torch.cuda.is_available():
                return "CUDA is not available in this environment"
            move_model(model, new_device)
            settings.device = value
        elif name == "dtype":
            if value not in ("bf16", "fp16", "fp32"):
                return "dtype must be one of: bf16, fp16, fp32"
            new_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[value]
            move_model(model, torch.device(settings.device), new_dtype)
            settings.dtype = value
        elif name == "window":
            settings.window = int(value)
            model.tria_temporal_window = settings.window
        elif name == "alpha":
            settings.alpha = float(value)
            tria_mod.set_carrier_alpha(settings.alpha)
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
    except ValueError:
        return f"couldn't parse {value!r} for {name}"
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
    settings.window = settings.window or model.tria_temporal_window
    model.tria_temporal_window = settings.window
    settings.alpha = float(getattr(cfg, "tria_carrier_alpha", settings.alpha))
    tria_mod.set_carrier_alpha(settings.alpha)

    print_banner(aio_path, lf.count_params(model), settings, manifest)

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
            print(COLOR.dim("(history cleared)"))
            continue
        if user_text.startswith("/system "):
            new_sys = user_text[len("/system "):]
            if messages and messages[0]["role"] == "system":
                messages[0] = {"role": "system", "content": new_sys}
            else:
                messages.insert(0, {"role": "system", "content": new_sys})
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
                settings.window = int(model.tria_temporal_window)
                settings.alpha = float(getattr(cfg, "tria_carrier_alpha", settings.alpha))
                tria_mod.set_carrier_alpha(settings.alpha)
                aio_path = new_aio
                print(COLOR.dim(f"(reloaded {new_aio})"))
            except Exception as e:
                print(COLOR.red(f"  ! reload failed: {e}"))
            continue

        handled_setting = False
        for key, attr in (("/device", "device"), ("/dtype", "dtype"), ("/window", "window"),
                          ("/alpha", "alpha"), ("/temperature", "temperature"),
                          ("/top-k", "top_k"), ("/top-p", "top_p"), ("/max-new", "max_new")):
            if user_text.startswith(key + " "):
                err = apply_setting(attr, user_text[len(key) + 1:].strip(), settings, model)
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
                reply, _ = generate_turn(model, tok, chat, messages, settings, esc)
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
    args = ap.parse_args()

    dev = lf.device_auto(args.device)
    settings = Settings(
        device=str(dev),
        dtype=args.dtype or "fp32",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new=args.max_new,
        window=args.window,
        alpha=args.alpha,
    )
    run_chat(args.archive, settings, args.system, args.dtype)


if __name__ == "__main__":
    main()
