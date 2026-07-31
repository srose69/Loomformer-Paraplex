from __future__ import annotations

import json
import math
import os
import sys
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from loomformer import Config

def lr_at(cfg: Config, step_zero_based: int) -> float:
    if step_zero_based < cfg.warmup_steps:
        return cfg.lr * (step_zero_based + 1) / max(1, cfg.warmup_steps)
    prog = (step_zero_based - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, prog))
    cos = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_frac + (1.0 - cfg.min_lr_frac) * cos)


def load_bytes_per_token(dataset: str) -> Tuple[float, str, bool]:
    meta_path = dataset + ".meta.json"
    if not os.path.exists(meta_path):
        return 1.0, meta_path, False
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    bpt = float(meta.get("bytes_per_token", 1.0))
    if not math.isfinite(bpt) or bpt <= 0.0:
        raise ValueError(f"bad bytes_per_token={bpt!r} in {meta_path}")
    return bpt, meta_path, True

def loss_to_bits(loss_nats: float, bytes_per_token: float) -> Tuple[float, float]:
    bits_tok = float(loss_nats) / math.log(2.0)
    bpb = bits_tok / bytes_per_token
    return bits_tok, bpb

def format_big_int(n: int) -> str:
    return f"{int(n):,}"


def format_eta_hours_minutes(seconds: float) -> str:
    """Compact non-negative ETA, rounded up to the next whole minute."""
    if not math.isfinite(seconds) or seconds < 0.0:
        return "?h ?min"
    total_minutes = int(math.ceil(seconds / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes:02d}min"


def _log_colors_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    forced = str(os.environ.get("FORCE_COLOR", "")).strip().lower()
    if forced in ("1", "true", "yes", "on"):
        return True
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _log_color(text: str, color: int, bold: bool = False) -> str:
    if not _log_colors_enabled():
        return text
    weight = "1;" if bold else ""
    return f"\033[{weight}38;5;{int(color)}m{text}\033[0m"


def _log_line(parts: List[str]) -> str:
    separator = _log_color("|", 240)
    return f" {separator} ".join(parts)


def format_train_status(
    step: int,
    train_loss: float,
    refeeds: int,
    lr: float,
    tokens: int,
    data_wait_s: float,
    left: str,
    elapsed_s: float,
) -> str:
    # Orange/purple palette only. `tok` deliberately stays uncolored.
    return _log_line([
        _log_color("[LF]", 208, bold=True) + " " + _log_color(str(step), 141, bold=True),
        _log_color("tr.loss:", 208) + " " + _log_color(f"{train_loss:.4f}", 215, bold=True),
        _log_color("ref:", 135) + " " + _log_color(str(refeeds), 177),
        _log_color("lr:", 173) + " " + _log_color(f"{lr:.2e}", 215),
        f"tok: {format_big_int(tokens)}",
        _log_color("dw:", 135) + " " + _log_color(f"{data_wait_s:.0f}s", 177),
        _log_color("left:", 208) + " " + _log_color(left, 215, bold=True),
        _log_color("lst:", 135) + " " + _log_color(f"{elapsed_s:.0f}s", 177),
    ])


def format_eval_status(
    step: int,
    eval_loss: float,
    bits_tok: float,
    bpb: float,
) -> str:
    parts = [
        _log_color("[EVAL]", 135, bold=True) + " " + _log_color(str(step), 141, bold=True),
        _log_color("loss:", 208) + " " + _log_color(f"{eval_loss:.4f}", 215, bold=True),
        _log_color("bit/tok:", 135) + " " + _log_color(f"{bits_tok:.4f}", 183),
    ]
    if math.isfinite(bpb):
        parts.append(
            _log_color("bpb:", 173)
            + " "
            + _log_color(f"{bpb:.4f}", 215)
        )
    return _log_line(parts)

__all__ = ('lr_at', 'load_bytes_per_token', 'loss_to_bits', 'format_big_int', 'format_eta_hours_minutes', '_log_colors_enabled', '_log_color', '_log_line', 'format_train_status', 'format_eval_status')
