from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from loomformer import Config

def assert_resume_attention_config(cfg: Config, saved_cfg: Dict[str, Any]) -> None:
    saved_layers = saved_cfg.get("attn_layers")
    if saved_layers is None:
        saved_layers = list(range(1, int(saved_cfg.get("layers", cfg.layers)) + 1))
    saved = {
        "attn_layers": [int(layer) for layer in saved_layers],
        "attn_token_stride": int(saved_cfg.get("attn_token_stride", 1)),
        "attn_token_schedule": str(
            saved_cfg.get("attn_token_schedule", "shared") or "shared").lower(),
    }
    active = {
        "attn_layers": list(cfg.attn_layers),
        "attn_token_stride": int(cfg.attn_token_stride),
        "attn_token_schedule": str(cfg.attn_token_schedule),
    }
    changed = [
        key for key in active
        if active[key] != saved[key]
        and not (
            key == "attn_token_schedule"
            and active["attn_token_stride"] == saved["attn_token_stride"] == 1
        )
    ]
    if changed:
        details = ", ".join(
            f"{key}: checkpoint={saved[key]!r} config={active[key]!r}"
            for key in changed)
        raise ValueError(
            "resume cannot change the attention architecture; use "
            f"init_checkpoint for an intentional conversion ({details})")


def should_replay_resume_data(
    cfg: Config, dataset: str, saved_cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    """Resolve whether resume should advance the dataset RNG/cursor."""
    policy = str(getattr(cfg, "resume_data_stream", "auto") or "auto").lower()
    if policy not in ("auto", "continue", "restart"):
        raise ValueError(
            "resume_data_stream must be auto, continue, or restart; "
            f"got {policy!r}")
    if policy == "continue":
        return True, "forced by resume_data_stream=continue"
    if policy == "restart":
        return False, "forced by resume_data_stream=restart"
    saved_dataset = saved_cfg.get("train_dataset")
    if not saved_dataset:
        return True, (
            "checkpoint has no train_dataset metadata; auto keeps legacy replay "
            "(use resume_data_stream=restart to force a fresh stream)"
        )
    same = os.path.abspath(str(saved_dataset)) == os.path.abspath(dataset)
    if same:
        return True, "train_dataset matches checkpoint"
    return False, f"train_dataset changed ({saved_dataset!r} -> {dataset!r})"


def dataset_progress_key(dataset: str) -> str:
    """Portable, human-readable key for per-dataset checkpoint progress."""
    return os.path.normpath(str(dataset))


def normalize_dataset_progress(blob: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Load the versioned per-dataset progress map from a checkpoint."""
    raw = blob.get("dataset_progress", {})
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for raw_key, raw_value in raw.items():
        key = dataset_progress_key(str(raw_key))
        if isinstance(raw_value, dict):
            steps = max(0, int(raw_value.get("steps", 0) or 0))
            draws = max(0, int(raw_value.get("draws", steps) or 0))
        else:
            # Accept an early/simple {dataset: steps} representation.
            steps = max(0, int(raw_value or 0))
            draws = steps
        out[key] = {"steps": steps, "draws": draws}
    return out


def find_dataset_progress(
    progress: Dict[str, Dict[str, int]], dataset: str,
) -> Tuple[str, Optional[Dict[str, int]]]:
    """Find progress across harmless relative/absolute path spelling changes."""
    key = dataset_progress_key(dataset)
    if key in progress:
        return key, progress[key]
    current_abs = os.path.abspath(str(dataset))
    for saved_key, entry in progress.items():
        if os.path.abspath(saved_key) == current_abs:
            return saved_key, entry
    return key, None


def resolve_resume_dataset_progress(
    cfg: "Config",
    dataset: str,
    blob: Dict[str, Any],
    global_step: int,
    override_steps: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, int]], str, int, int, str]:
    """Resolve the current dataset's cursor without conflating it with global step."""
    progress = normalize_dataset_progress(blob)
    current_key = dataset_progress_key(dataset)
    saved_key, saved_entry = find_dataset_progress(progress, dataset)
    accum = max(1, int(getattr(cfg, "grad_accum_steps", 1) or 1))
    policy = str(getattr(cfg, "resume_data_stream", "auto") or "auto").lower()
    if policy not in ("auto", "continue", "restart"):
        raise ValueError(
            "resume_data_stream must be auto, continue, or restart; "
            f"got {policy!r}")

    if override_steps is not None:
        steps = int(override_steps)
        if steps < 0:
            raise ValueError(
                f"resume_dataset_steps must be >= 0, got {override_steps}")
        draws = steps * accum
        reason = (
            f"forced current-dataset progress: steps={steps}, draws={draws}")
    elif policy == "restart":
        steps = draws = 0
        reason = "forced by resume_data_stream=restart"
    elif saved_entry is not None:
        steps = int(saved_entry["steps"])
        draws = int(saved_entry["draws"])
        reason = (
            f"restored saved progress for {current_key!r}: "
            f"steps={steps}, draws={draws}")
    elif progress and policy == "auto":
        steps = draws = 0
        reason = f"no saved progress for new dataset {current_key!r}"
    else:
        # Old checkpoints have no per-dataset map. Preserve their existing
        # auto/continue/restart behavior as a one-time migration fallback.
        replay, legacy_reason = should_replay_resume_data(
            cfg, dataset, blob.get("cfg", {}))
        steps = max(0, int(global_step)) if replay else 0
        draws = steps * accum
        reason = (
            f"legacy checkpoint fallback ({legacy_reason}): "
            f"steps={steps}, draws={draws}")

    if saved_key != current_key:
        progress.pop(saved_key, None)
    progress[current_key] = {"steps": steps, "draws": draws}
    return progress, current_key, steps, draws, reason


def checkpoint_tokens_seen(blob: Dict[str, Any], completed_steps: int) -> Tuple[int, bool]:
    """Return cumulative processed tokens and whether the value was exact."""
    saved = blob.get("tokens_seen")
    if saved is not None:
        return max(0, int(saved)), True
    saved_cfg = blob.get("cfg", {})
    batch = int(saved_cfg.get("batch_size", 0) or 0)
    seq_len = int(saved_cfg.get("seq_len", 0) or 0)
    accum = max(1, int(saved_cfg.get("grad_accum_steps", 1) or 1))
    if batch <= 0 or seq_len <= 0:
        return 0, False
    return max(0, int(completed_steps)) * batch * seq_len * accum, False

__all__ = ('assert_resume_attention_config', 'should_replay_resume_data', 'dataset_progress_key', 'normalize_dataset_progress', 'find_dataset_progress', 'resolve_resume_dataset_progress', 'checkpoint_tokens_seen')
