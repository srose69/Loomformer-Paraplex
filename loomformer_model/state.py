from __future__ import annotations

from types import ModuleType
from typing import Any
import contextlib
import threading

import torch.nn as nn

_source: ModuleType | None = None
_checkpoint_tls = threading.local()


def bind(source: ModuleType) -> None:
    global _source
    if _source is not None and _source is not source:
        raise RuntimeError("LoomFormer model state is already bound")
    _source = source


def __getattr__(name: str) -> Any:
    if _source is None:
        raise RuntimeError("LoomFormer model state is not bound")
    return getattr(_source, name)


def checkpoint_anchor_override(module: nn.Module):
    overrides = getattr(_checkpoint_tls, "anchor_overrides", None)
    return None if overrides is None else overrides.get(id(module))


@contextlib.contextmanager
def activation_checkpoint_recompute_context(holder: dict):
    previous = getattr(_checkpoint_tls, "anchor_overrides", None)
    overrides = holder.get("anchor_overrides")
    if overrides is None:
        raise RuntimeError("activation-checkpoint anchor snapshots were not captured")
    _checkpoint_tls.anchor_overrides = overrides
    try:
        yield
    finally:
        _checkpoint_tls.anchor_overrides = previous
