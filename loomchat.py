#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loomchat.py -- interactive chat / inference for LoomFormer checkpoints.

Split out from loomformer.py on purpose: loomformer.py is the model plus everything
close to training (streams, packing helpers, the trainer); loomchat.py is everything
about TALKING to an already-trained checkpoint (chat template application, streaming
generation, sampling, the terminal UI). Imports Model/ChatTemplate/build_tokenizer
from loomformer -- does not duplicate them (same reasoning as loomsft.py: one source
of truth for the architecture and for "how a turn becomes text", see
loomformer.ChatTemplate / chat_template.jinja / sft.md).

loomformer.py's own `infer()` stays as a low-frills raw-completion debug tool for
checkpoints that never saw chat data at all (a pure pretrain run) -- running THOSE
through a chat template would be a category error, not a missing feature. This file
is for checkpoints that did see chat/SFT data.

CLI:
  loomchat.py --checkpoint sft.pt [--system "..."] [--temperature 0.7] [--top-p 0.9]
              [--top-k 0] [--max-new 512] [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import torch

import loomformer as lf

# ============================================================================
# terminal color: NO_COLOR / non-tty / TERM=dumb all fall back to plain text,
# quietly -- no error, no warning, just no escape codes.
# ============================================================================


def _color_supported() -> bool:
    if os.environ.get("NO_COLOR") is not None:  # https://no-color.org/ -- a real,
        return False                             # respected convention, not invented here
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


COLOR = _Colors(_color_supported())

BANNER = """+------------------------------+
| #       ###    ###   #   #   |
| #      ##  #  ##  #  ## ##   |
| #      # # #  # # #  # # #   |
| #   #  #  ##  #  ##  #   #   |
| #####   ###    ###   #   #   |
+------------------------------+"""


def print_banner(ckpt_path: str, n_params: int) -> None:
    print(COLOR.cyan(BANNER))
    print(COLOR.dim(f"  checkpoint: {ckpt_path}  ({n_params:,} params)"))
    print(COLOR.dim("  /reset clears history · /system <text> sets the system prompt · /exit quits\n"))


# ============================================================================
# sampling
# ============================================================================

def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> int:
    """logits: 1D tensor (VOCAB,). temperature<=0 -> greedy argmax."""
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
# streaming display: redecode-the-whole-turn-so-far and print only the NEW
# suffix each step (robust to ByteLevel BPE's leading-space joining quirks --
# decoding one freshly generated token in isolation is NOT reliably the same
# text as it contributes when decoded as part of the full sequence). <think>
# spans print dim; <tool_call> spans are buffered (not streamed raw mid-JSON)
# and shown as a distinct formatted block once closed.
# ============================================================================

class StreamRenderer:
    def __init__(self) -> None:
        self.shown = ""       # raw text (special tokens visible) already handled
        self.in_think = False
        self.tc_buffer: Optional[str] = None  # None, or text accumulated since <tool_call>

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
                    self.feed(chunk)  # re-enter through feed() to keep buffering tool_call text
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
# chat loop
# ============================================================================

def load_checkpoint(ckpt_path: str, device: torch.device):
    blob = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = lf.Config.from_checkpoint_dict(blob["cfg"])
    tok = lf.build_tokenizer(cfg)
    lf.apply_config(cfg)
    ablation = bool(blob.get("ablation", False))
    model = lf.Model(ablation=ablation).to(device)
    lf.load_model_blob_into(model, blob, ablation=ablation)
    model.eval()
    return model, tok, cfg


def generate_turn(model, tok, chat: "lf.ChatTemplate", messages: List[Dict], device: torch.device,
                   max_new: int, temperature: float, top_k: int, top_p: float) -> Dict:
    """Runs one assistant turn to completion (streaming to stdout as it goes) and
    returns the new message dict ({"role": "assistant", "content": ..., "tool_calls"?:
    ...}) to append to the running conversation."""
    ids = chat.render_prompt_ids(messages)
    room = lf.SEQ_LEN - len(ids)
    if room <= 0:
        raise RuntimeError(
            f"conversation ({len(ids)} tokens) already fills the model's context "
            f"(seq_len={lf.SEQ_LEN}); the incremental cache has no wraparound. /reset."
        )
    max_new = min(max_new, room)
    states = None
    logits = None
    for pos, tid in enumerate(ids):
        x = torch.tensor([int(tid)], device=device, dtype=torch.long)
        logits, states = model.step(x, pos, states)

    renderer = StreamRenderer()
    gen_ids: List[int] = []
    for i in range(max_new):
        nxt = sample_next(logits[0], temperature, top_k, top_p)
        if nxt in chat.stop_ids:
            break
        gen_ids.append(nxt)
        renderer.feed(tok.decode(gen_ids, skip_special_tokens=False))
        x = torch.tensor([nxt], device=device, dtype=torch.long)
        logits, states = model.step(x, len(ids) + i, states)
    print()

    raw_text = tok.decode(gen_ids, skip_special_tokens=False)
    tool_calls = chat.parse_tool_calls(raw_text)
    if tool_calls:
        return {"role": "assistant", "content": None, "tool_calls": tool_calls}
    clean_text = tok.decode(gen_ids, skip_special_tokens=True)
    return {"role": "assistant", "content": clean_text}


def run_chat(ckpt_path: str, device: torch.device, system: Optional[str],
             temperature: float, top_k: int, top_p: float, max_new: int) -> None:
    model, tok, cfg = load_checkpoint(ckpt_path, device)
    chat = lf.ChatTemplate(tok)
    print_banner(ckpt_path, lf.count_params(model))

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
        if user_text in ("/exit", "/quit"):
            break
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

        messages.append({"role": "user", "content": user_text})
        print(COLOR.bold(COLOR.magenta("loom> ")), end="", flush=True)
        try:
            reply = generate_turn(model, tok, chat, messages, device, max_new, temperature, top_k, top_p)
        except RuntimeError as e:
            print(COLOR.red(f"\n  ! {e}"))
            messages.pop()  # the user turn that couldn't be answered isn't part of history
            continue
        messages.append(reply)


# ============================================================================
# CLI
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="loomchat: interactive chat for LoomFormer checkpoints")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()
    dev = lf.device_auto(args.device)
    run_chat(args.checkpoint, dev, args.system, args.temperature, args.top_k, args.top_p, args.max_new)


if __name__ == "__main__":
    main()
