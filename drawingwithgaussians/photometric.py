"""Optional per-camera affine exposure and white-balance correction."""

from __future__ import annotations

import mlx.core as mx


def init_photometric(num_cameras: int) -> dict[str, mx.array]:
    return {
        "log_gain": mx.zeros((int(num_cameras), 3), dtype=mx.float32),
        "bias": mx.zeros((int(num_cameras), 3), dtype=mx.float32),
    }


def apply_photometric(colors: mx.array, params: dict[str, mx.array], camera_indices: mx.array) -> mx.array:
    """Apply correction after SH evaluation and before rasterization."""
    gains = mx.exp(mx.take(params["log_gain"], camera_indices, axis=0))
    biases = mx.take(params["bias"], camera_indices, axis=0)
    if colors.ndim == 2:
        return colors * gains[0] + biases[0]
    return colors * gains[:, None, :] + biases[:, None, :]


def photometric_identity_regularizer(params: dict[str, mx.array]) -> mx.array:
    return mx.mean(params["log_gain"] ** 2) + mx.mean(params["bias"] ** 2)
