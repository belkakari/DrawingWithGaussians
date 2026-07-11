"""Structured RGB evaluation and reproducibility manifests.

Predictions are clamped only for metrics.  Raw renderer output remains
available to training and its out-of-range fractions are reported explicitly.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ViewMetrics:
    image_id: str
    psnr: float
    ssim: float
    lpips_alex: float | None
    raw_underflow_fraction: float
    raw_overflow_fraction: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def dependency_versions(names: Iterable[str] = ()) -> dict[str, str]:
    default = ("numpy", "mlx", "torch", "pycolmap", "scipy", "plyfile")
    versions = {}
    for name in tuple(names) or default:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def write_run_manifest(
    out_dir: Path,
    resolved_config: Any,
    seed: int,
    train_image_ids: Sequence[str | int],
    validation_image_ids: Sequence[str | int],
    camera_batches: Sequence[Sequence[int]],
    repo_root: Path,
) -> Path:
    payload = {
        "resolved_config": _jsonable(resolved_config),
        "seed": int(seed),
        "train_image_ids": [str(v) for v in train_image_ids],
        "validation_image_ids": [str(v) for v in validation_image_ids],
        "camera_batches": [[int(i) for i in batch] for batch in camera_batches],
        "git": git_revision(repo_root),
        "dependencies": dependency_versions(),
        "platform": {"python": platform.python_version(), "platform": platform.platform()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


class RGBMetricSuite:
    """MLX PSNR/SSIM/LPIPS-Alex wrapper with per-view output."""

    def __init__(self, enable_lpips: bool = True):
        from .lpips_mlx import LPIPSAlex

        self.lpips = LPIPSAlex() if enable_lpips else None

    def view(self, image_id: str | int, prediction: np.ndarray, target: np.ndarray) -> ViewMetrics:
        raw = np.asarray(prediction, dtype=np.float32)
        gt = np.asarray(target, dtype=np.float32)
        if raw.shape != gt.shape or raw.ndim != 3 or raw.shape[-1] != 3:
            raise ValueError(f"expected matching HWC RGB arrays, got {raw.shape} and {gt.shape}")
        pred = np.clip(raw, 0.0, 1.0)
        target_clamped = np.clip(gt, 0.0, 1.0)
        import mlx.core as mx

        from .losses import ssim as mlx_ssim

        mse = float(np.mean((pred - target_clamped) ** 2))
        ssim_value = mlx_ssim(mx.array(pred), mx.array(target_clamped))
        lpips_value = self.lpips(pred, target_clamped) if self.lpips is not None else None
        mx.eval(ssim_value, *(() if lpips_value is None else (lpips_value,)))
        return ViewMetrics(
            image_id=str(image_id),
            psnr=10.0 * math.log10(1.0 / max(mse, 1e-12)),
            ssim=float(ssim_value),
            lpips_alex=float(lpips_value[0]) if lpips_value is not None else None,
            raw_underflow_fraction=float(np.mean(raw < 0.0)),
            raw_overflow_fraction=float(np.mean(raw > 1.0)),
        )


def aggregate_view_metrics(per_view: Sequence[ViewMetrics]) -> dict[str, Any]:
    if not per_view:
        raise ValueError("cannot aggregate an empty evaluation")
    keys = ("psnr", "ssim", "lpips_alex", "raw_underflow_fraction", "raw_overflow_fraction")
    aggregate: dict[str, float | None] = {}
    for key in keys:
        values = [getattr(v, key) for v in per_view if getattr(v, key) is not None]
        aggregate[key] = float(np.mean(values)) if values else None
    return {"per_view": [asdict(v) for v in per_view], "aggregate": aggregate}


def lpips_alex_mlx(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> list[float]:
    """Compute LPIPS-Alex in one batched MLX graph."""
    from .lpips_mlx import lpips_alex

    return lpips_alex(predictions, targets)


def write_metrics(
    out_dir: Path, step: int, per_view: Sequence[ViewMetrics], final: bool = False, label: str | None = None
) -> Path:
    payload = {"step": int(step), **aggregate_view_metrics(per_view)}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "metrics" if label is None else f"metrics_{label}"
    path = out_dir / (f"{stem}_final.json" if final else f"{stem}_step_{int(step)}.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
