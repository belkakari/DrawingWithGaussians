"""LPIPS-Alex inference implemented with MLX operations."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_WEIGHTS = Path(__file__).with_name("assets") / "lpips_alex.npz"


class LPIPSAlex:
    """TorchMetrics-compatible LPIPS-Alex for RGB images in ``[0, 1]``."""

    def __init__(self, weights_path: str | Path = _WEIGHTS):
        arrays = np.load(weights_path)
        self.weights = {name: mx.array(arrays[name]) for name in arrays.files}
        mx.eval(self.weights)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def _conv(self, x, layer, stride=1, padding=0):
        weight = self.weights[f"conv{layer}.weight"]
        bias = self.weights[f"conv{layer}.bias"]
        return mx.maximum(mx.conv2d(x, weight, stride=stride, padding=padding) + bias, 0.0)

    def _features(self, x):
        first = self._conv(x, 0, stride=4, padding=2)
        x = self.pool(first)
        second = self._conv(x, 3, padding=2)
        x = self.pool(second)
        third = self._conv(x, 6, padding=1)
        fourth = self._conv(third, 8, padding=1)
        fifth = self._conv(fourth, 10, padding=1)
        return first, second, third, fourth, fifth

    def __call__(self, prediction, target):
        pred = mx.array(np.asarray(prediction, dtype=np.float32))
        truth = mx.array(np.asarray(target, dtype=np.float32))
        if pred.ndim == 3:
            pred, truth = pred[None], truth[None]
        if pred.shape != truth.shape or pred.ndim != 4 or pred.shape[-1] != 3:
            raise ValueError(f"expected matching NHWC RGB inputs, got {pred.shape} and {truth.shape}")
        shift = mx.array([-0.030, -0.088, -0.188]).reshape(1, 1, 1, 3)
        scale = mx.array([0.458, 0.448, 0.450]).reshape(1, 1, 1, 3)
        pred = (2.0 * mx.clip(pred, 0.0, 1.0) - 1.0 - shift) / scale
        truth = (2.0 * mx.clip(truth, 0.0, 1.0) - 1.0 - shift) / scale
        result = mx.zeros((pred.shape[0],), dtype=mx.float32)
        for index, (a, b) in enumerate(zip(self._features(pred), self._features(truth), strict=True)):
            a = a / mx.sqrt(mx.sum(a * a, axis=-1, keepdims=True) + 1e-8)
            b = b / mx.sqrt(mx.sum(b * b, axis=-1, keepdims=True) + 1e-8)
            weighted = mx.sum((a - b) ** 2 * self.weights[f"lin{index}"], axis=-1)
            result = result + mx.mean(weighted, axis=(1, 2))
        return result


_DEFAULT_MODEL: LPIPSAlex | None = None


def lpips_alex(predictions, targets, batch_size: int = 4) -> list[float]:
    global _DEFAULT_MODEL
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = LPIPSAlex()
    scores = []
    for start in range(0, len(predictions), batch_size):
        values = _DEFAULT_MODEL(
            np.stack(predictions[start : start + batch_size]),
            np.stack(targets[start : start + batch_size]),
        )
        mx.eval(values)
        scores.extend(np.asarray(values).astype(float).tolist())
    return scores
