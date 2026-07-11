"""Per-Gaussian utilization telemetry and conservative pruning decisions."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def init_utilization(num_gaussians: int) -> dict[str, mx.array]:
    n = int(num_gaussians)
    return {
        "ema": mx.zeros((n,), dtype=mx.float32),
        "observations": mx.zeros((n,), dtype=mx.uint32),
        "age": mx.zeros((n,), dtype=mx.uint32),
        "consecutive_low": mx.zeros((n,), dtype=mx.uint32),
    }


def update_utilization(
    state: dict[str, mx.array],
    opacity_raw_grad: mx.array,
    opacities_raw: mx.array,
    active_view_count: mx.array,
    batch: int,
    height: int,
    width: int,
    ema_decay: float = 0.95,
    epsilon: float = 1e-8,
) -> dict[str, mx.array]:
    """Update the normalized net-gradient proxy entirely in the MLX graph."""
    alpha = mx.sigmoid(opacities_raw)
    g_alpha = opacity_raw_grad / mx.maximum(alpha * (1.0 - alpha), epsilon)
    observed = active_view_count > 0
    u_step = float(batch * height * width) * mx.abs(g_alpha) / mx.maximum(active_view_count, 1.0)
    ema = mx.where(observed, ema_decay * state["ema"] + (1.0 - ema_decay) * u_step, state["ema"])
    return {
        **state,
        "ema": ema,
        "observations": state["observations"] + observed.astype(mx.uint32),
        "age": state["age"] + mx.ones_like(state["age"]),
    }


def pruning_window(
    state: dict[str, mx.array],
    threshold: float,
    warmup_steps: int,
    minimum_observations: int,
    grace_steps: int,
    repeated_windows: int,
) -> tuple[np.ndarray, dict[str, mx.array]]:
    mx.eval(*state.values())
    arrays = {k: np.asarray(v) for k, v in state.items()}
    eligible = (arrays["age"] >= max(int(warmup_steps), int(grace_steps))) & (
        arrays["observations"] >= int(minimum_observations)
    )
    low = eligible & (arrays["ema"] < float(threshold))
    consecutive = np.where(low, arrays["consecutive_low"] + 1, 0).astype(np.uint32)
    updated = {**state, "consecutive_low": mx.array(consecutive)}
    return consecutive >= int(repeated_windows), updated


def remap_utilization(
    state: dict[str, mx.array], idx_keep: np.ndarray, new_parent_indices: np.ndarray
) -> dict[str, mx.array]:
    """Survivors retain state; children inherit EMA but restart lifecycle."""
    mx.eval(*state.values())
    p = {k: np.asarray(v) for k, v in state.items()}
    idx_keep = np.asarray(idx_keep, dtype=np.int64)
    parents = np.asarray(new_parent_indices, dtype=np.int64)
    n_new = len(parents)
    return {
        "ema": mx.array(np.concatenate([p["ema"][idx_keep], p["ema"][parents]]).astype(np.float32)),
        "observations": mx.array(np.concatenate([p["observations"][idx_keep], np.zeros(n_new, dtype=np.uint32)])),
        "age": mx.array(np.concatenate([p["age"][idx_keep], np.zeros(n_new, dtype=np.uint32)])),
        "consecutive_low": mx.array(np.concatenate([p["consecutive_low"][idx_keep], np.zeros(n_new, dtype=np.uint32)])),
    }


def telemetry_correlations(
    state: dict[str, mx.array], params: dict[str, mx.array], tile_activity: mx.array | None = None
) -> dict[str, float]:
    mx.eval(*state.values(), *params.values())
    ema = np.asarray(state["ema"])
    features = {
        "opacity": 1.0 / (1.0 + np.exp(-np.asarray(params["opacities_raw"]))),
        "max_scale": np.exp(np.asarray(params["log_scales"])).max(axis=1),
    }
    if tile_activity is not None:
        mx.eval(tile_activity)
        features["tile_activity"] = np.asarray(tile_activity, dtype=np.float64)
    out = {}
    for name, values in features.items():
        out[name] = float(np.corrcoef(ema, values)[0, 1]) if len(ema) > 1 else float("nan")
    return out
