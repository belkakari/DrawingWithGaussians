"""Structured final evaluation shared by the standalone image fitters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

from .evaluation import RGBMetricSuite, write_metrics, write_run_manifest


def finalize_single_image_run(
    *,
    out_dir: Path,
    resolved_config: Any,
    seed: int,
    image_path: str | Path,
    prediction: mx.array,
    target: mx.array,
    final_loss: mx.array,
    final_gaussian_count: int,
    total_steps: int,
    wall_seconds: float,
    enable_lpips: bool = False,
    save_render: bool = True,
) -> tuple[Path, Path]:
    """Write exact raw-render metrics, reproducibility data, and timing."""
    mx.eval(prediction, target, final_loss)
    pred_np = np.asarray(prediction, dtype=np.float32)
    target_np = np.asarray(target, dtype=np.float32)
    suite = RGBMetricSuite(enable_lpips=enable_lpips)
    view = suite.view(Path(image_path).name, pred_np, target_np)
    metrics_path = write_metrics(out_dir, total_steps, [view], final=True)
    write_run_manifest(
        out_dir,
        resolved_config,
        seed,
        train_image_ids=[str(image_path)],
        validation_image_ids=[],
        camera_batches=[],
        repo_root=Path(__file__).resolve().parents[1],
    )
    summary = {
        "end_to_end_wall_seconds": float(wall_seconds),
        "peak_mlx_allocator_bytes": int(mx.get_peak_memory()),
        "active_mlx_allocator_bytes": int(mx.get_active_memory()),
        "cached_mlx_allocator_bytes": int(mx.get_cache_memory()),
        "final_gaussian_count": int(final_gaussian_count),
        "final_loss": float(final_loss),
        "total_steps": int(total_steps),
    }
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if save_render:
        rgb = (np.clip(pred_np, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(out_dir / "final_render.png")
    return metrics_path, summary_path
