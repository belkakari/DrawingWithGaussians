"""Train 3D Gaussians on a COLMAP/Mip-NeRF 360 scene with MLX.

Lightweight trainer for datasets such as
``~/360_extra_scenes/{flowers,treehill}``.
It reads COLMAP cameras/images/points with pycolmap, trains with this repo's
MLX 3D rasterizer/loss/densification, and exports a standard PLY.

Run with:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python train_colmap3d.py --config-name train_colmap3d.yaml

Override params with Hydra, e.g.:
    uv run python train_colmap3d.py --config-name train_colmap3d.yaml \
      data.dir=~/Downloads/360_extra_scenes/flowers \
      data.factor=8 gaussians.max_init_points=1000 optim.num_steps=100 data.max_side=256
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# pycolmap may load a second OpenMP runtime on macOS. Set before importing it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import hydra  # type: ignore[import-not-found]
import mlx.core as mx
import mlx.optimizers as mlx_optim
import numpy as np
from omegaconf import DictConfig, OmegaConf  # type: ignore[import-not-found]
from PIL import Image

from drawingwithgaussians.evaluation import (
    RGBMetricSuite,
    ViewMetrics,
    lpips_alex_mlx,
    write_metrics,
    write_run_manifest,
)
from drawingwithgaussians.gaussian3d import (
    carry_optimizer_state_3d,
    densify_masks,
    get_opt_step,
    reset_opacities_3d,
    set_up_optimizer_3d,
    split_n_prune_3d,
    zero_param_moments,
)
from drawingwithgaussians.losses import pixel_loss_2dgs, pixel_loss_3d
from drawingwithgaussians.photometric import apply_photometric, init_photometric, photometric_identity_regularizer
from drawingwithgaussians.rendering2dgs import project_gaussians_2dgs  # type: ignore[import-not-found]
from drawingwithgaussians.rendering2dgs_fused import (  # type: ignore[import-not-found]
    _count_bbox_intersections,
    rasterize2dgs_fused,
)
from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE, project_gaussians
from drawingwithgaussians.rendering3d_fused import _count_tile_intersections, _num_tiles, rasterize3d_fused
from drawingwithgaussians.schedule import MomentumBudget, resolution_segments
from drawingwithgaussians.selective_adam import SelectiveAdam
from drawingwithgaussians.sh import rgb_to_sh0, sh_degree_for_step, view_dependent_colors
from drawingwithgaussians.splat_export import export_ply_3d
from drawingwithgaussians.utilization import (
    init_utilization,
    pruning_window,
    remap_utilization,
    telemetry_correlations,
    update_utilization,
)


@dataclass
class ColmapScene:
    image_paths: list[Path]
    camtoworlds: np.ndarray
    Ks: list[np.ndarray]
    points: np.ndarray
    points_rgb: np.ndarray
    scene_scale: float
    normalization_center: np.ndarray
    normalization_scale: float


def _as_dict(map_like: Any) -> dict[int, Any]:
    return {int(k): v for k, v in map_like.items()}


def _image_w2c(image: Any) -> np.ndarray:
    cam_from_world = image.cam_from_world
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = np.asarray(cam_from_world.matrix(), dtype=np.float64)  # type: ignore[attr-defined]
    return w2c


def _get_rel_paths(path_dir: Path) -> list[str]:
    paths = []
    for root, _, files in os.walk(path_dir):
        for file in files:
            if file.startswith("."):
                continue
            paths.append(os.path.relpath(Path(root) / file, path_dir))
    return sorted(paths)


def _load_colmap_scene(
    data_dir: Path, factor: int, normalize: bool, test_every: int
) -> tuple[ColmapScene, list[int], list[int]]:
    import pycolmap  # type: ignore[import-not-found]

    sparse_dir = data_dir / "sparse/0"
    if not sparse_dir.exists():
        sparse_dir = data_dir / "sparse"
    if not sparse_dir.exists():
        raise FileNotFoundError(f"COLMAP sparse directory not found under {data_dir}")

    recon = pycolmap.Reconstruction(str(sparse_dir))
    cameras = _as_dict(recon.cameras)
    images = _as_dict(recon.images)
    image_ids = [int(i) for i in recon.reg_image_ids()]
    imdata = {image_id: images[image_id] for image_id in image_ids}
    if not imdata:
        raise ValueError(f"No registered COLMAP images in {sparse_dir}")

    rows = []
    for im in imdata.values():
        cam = cameras[int(im.camera_id)]
        K = np.asarray(cam.calibration_matrix(), dtype=np.float32)
        K[:2, :] /= factor
        rows.append((im.name, _image_w2c(im), int(im.camera_id), K))
    rows.sort(key=lambda r: r[0])

    image_dir = data_dir / (f"images_{factor}" if factor > 1 else "images")
    if not image_dir.exists():
        image_dir = data_dir / "images"
    colmap_files = _get_rel_paths(data_dir / "images")
    image_files = _get_rel_paths(image_dir)
    colmap_to_image = dict(zip(colmap_files, image_files, strict=False))

    image_paths = [image_dir / colmap_to_image[name] for name, *_ in rows]
    w2cs = np.stack([r[1] for r in rows], axis=0)
    camtoworlds = np.linalg.inv(w2cs).astype(np.float32)
    Ks = [r[3].copy() for r in rows]

    points3d = _as_dict(recon.points3D)
    point_ids = sorted(points3d)
    points = np.array([points3d[pid].xyz for pid in point_ids], dtype=np.float32).reshape(-1, 3)
    points_rgb = np.array([points3d[pid].color for pid in point_ids], dtype=np.float32).reshape(-1, 3)

    center = np.zeros(3, dtype=np.float32)
    scale = 1.0
    if normalize:
        center = np.mean(camtoworlds[:, :3, 3], axis=0)
        cam_dists = np.linalg.norm(camtoworlds[:, :3, 3] - center, axis=1)
        scale = float(np.max(cam_dists)) if len(cam_dists) else 1.0
        scale = max(scale, 1e-6)
        camtoworlds[:, :3, 3] = (camtoworlds[:, :3, 3] - center) / scale
        points = (points - center[None, :]) / scale
    camera_locations = camtoworlds[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    scene_scale = float(np.max(np.linalg.norm(camera_locations - scene_center, axis=1)))

    # Correct intrinsics if actual downsampled image dimensions differ from COLMAP/factor rounding.
    with Image.open(image_paths[0]) as img:
        actual_w, actual_h = img.size
    # pycolmap camera size for first image, after /factor.
    first_cam = cameras[rows[0][2]]
    expected_w, expected_h = (
        int(first_cam.width // factor),
        int(first_cam.height // factor),
    )
    if expected_w > 0 and expected_h > 0:
        sx, sy = actual_w / expected_w, actual_h / expected_h
        for K in Ks:
            K[0, :] *= sx
            K[1, :] *= sy

    indices = np.arange(len(image_paths))
    train_indices = indices[indices % test_every != 0].tolist()
    val_indices = indices[indices % test_every == 0].tolist()
    return (
        ColmapScene(image_paths, camtoworlds, Ks, points, points_rgb, scene_scale, center, scale),
        train_indices,
        val_indices,
    )


def _logit(x: np.ndarray | float, eps: float = 1e-5):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _resize_image_np(image: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None or max(image.shape[:2]) <= max_side:
        return image
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _load_item(scene: ColmapScene, index: int, max_side: int | None):
    with Image.open(scene.image_paths[index]) as img:
        image = np.array(img.convert("RGB"), dtype=np.float32)
    old_hw = image.shape[:2]
    image = _resize_image_np(image, max_side)
    image = np.clip(image / 255.0, 0.0, 1.0).astype(np.float32)

    K = scene.Ks[index].copy().astype(np.float32)
    if image.shape[:2] != old_hw:
        old_h, old_w = old_hw
        new_h, new_w = image.shape[:2]
        K[0, :] *= new_w / old_w
        K[1, :] *= new_h / old_h

    viewmat = np.linalg.inv(scene.camtoworlds[index]).astype(np.float32)
    return mx.array(image), mx.array(viewmat), mx.array(K)


def _preload_views(
    scene: ColmapScene,
    indices: list[int],
    max_side: int | None,
    log,
    target_hw: tuple[int, int] | None = None,
):
    """Decode/resize every train view once into resident MLX buffers.

    The old path ran PIL JPEG decode + cv2 resize + fp32 normalize + the
    numpy->MLX buffer copy *inside* the training loop — several ms of
    synchronous host work per step (x batch with camera batching), stalling
    MLX stream submission. (Memory is unified on Apple Silicon, so the copy itself
    is a cheap same-DRAM memcpy; the decode/resize/normalize is the cost.)
    Images are stored stacked as uint8 (~a quarter of the fp32 footprint;
    e.g. ~90 MB for 150 views at 512x384) and normalized to float on the
    active MLX stream inside the compiled step. All views are resized to the
    first view's (H, W) so they stack; intrinsics are rescaled to match.

    ``target_hw`` forces every view to a caller-chosen (H, W) — the val split
    must match the train split's shape instead of deriving its own from its
    first image.

    Returns ``(targets_u8 (M, H, W, 3), viewmats (M, 4, 4), Ks (M, 3, 3))``
    mx arrays, ordered like ``indices``.
    """
    t0 = time.perf_counter()
    imgs, viewmats, Ks = [], [], []
    ref_hw = tuple(target_hw) if target_hw is not None else None
    for index in indices:
        with Image.open(scene.image_paths[index]) as img:
            image = np.array(img.convert("RGB"), dtype=np.uint8)
        old_h, old_w = image.shape[:2]
        image = _resize_image_np(image.astype(np.float32), max_side)
        if ref_hw is None:
            ref_hw = image.shape[:2]
        elif image.shape[:2] != ref_hw:
            image = cv2.resize(image, (ref_hw[1], ref_hw[0]), interpolation=cv2.INTER_AREA)
        K = scene.Ks[index].copy().astype(np.float32)
        K[0, :] *= ref_hw[1] / old_w
        K[1, :] *= ref_hw[0] / old_h
        imgs.append(np.clip(image, 0.0, 255.0).astype(np.uint8))
        viewmats.append(np.linalg.inv(scene.camtoworlds[index]).astype(np.float32))
        Ks.append(K)
    if ref_hw is None:
        raise ValueError("no views to preload")
    if target_hw is not None and imgs and imgs[0].shape[:2] != tuple(target_hw):
        raise ValueError(f"preloaded views have shape {imgs[0].shape[:2]}, expected {tuple(target_hw)}")
    targets_u8 = mx.array(np.stack(imgs, axis=0))
    viewmats_mx = mx.array(np.stack(viewmats, axis=0))
    Ks_mx = mx.array(np.stack(Ks, axis=0))
    mx.eval(targets_u8, viewmats_mx, Ks_mx)
    log.info(
        "preloaded %d views (%dx%d) in %.1fs, %.0f MB resident",
        len(indices),
        ref_hw[1],
        ref_hw[0],
        time.perf_counter() - t0,
        targets_u8.nbytes / 1e6,
    )
    return targets_u8, viewmats_mx, Ks_mx


def _camera_matrices_at_resolution(scene: ColmapScene, indices: list[int], max_side: int | None):
    """Load only camera matrices at a requested render resolution."""
    sizes = []
    for index in indices:
        with Image.open(scene.image_paths[index]) as image:
            sizes.append(image.size)
    first_width, first_height = sizes[0]
    if max_side is not None and max(first_width, first_height) > max_side:
        factor = max_side / max(first_width, first_height)
        width, height = max(1, round(first_width * factor)), max(1, round(first_height * factor))
    else:
        width, height = first_width, first_height
    viewmats, Ks = [], []
    for index, (source_width, source_height) in zip(indices, sizes, strict=True):
        K = scene.Ks[index].copy().astype(np.float32)
        K[0, :] *= width / source_width
        K[1, :] *= height / source_height
        Ks.append(K)
        viewmats.append(np.linalg.inv(scene.camtoworlds[index]).astype(np.float32))
    return mx.array(np.stack(viewmats)), mx.array(np.stack(Ks)), width, height


def _knn_init_scales(
    points: np.ndarray,
    fallback: float,
    k: int,
    multiplier: float,
    min_scale: float,
    max_scale: float | None,
) -> np.ndarray:
    """Per-point 3D scales from sparse-cloud nearest-neighbor spacing."""
    n = len(points)
    if n <= 1 or k <= 0:
        scale = np.full((n,), fallback, dtype=np.float32)
    else:
        from scipy.spatial import cKDTree  # type: ignore[import-not-found]

        kk = min(k + 1, n)
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=kk)
        dists = np.asarray(dists, dtype=np.float32)
        if dists.ndim == 1:
            dists = dists[:, None]
        nn = dists[:, 1:] if dists.shape[1] > 1 else dists
        nn = np.where(np.isfinite(nn) & (nn > 0.0), nn, np.nan)
        scale = np.nanmean(nn, axis=1) * float(multiplier)
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, fallback).astype(np.float32)
    hi = np.inf if max_scale is None or max_scale <= 0.0 else float(max_scale)
    return np.clip(scale, float(min_scale), hi).astype(np.float32)


def _sample_init_points(
    scene: ColmapScene,
    max_points: int,
    seed: int,
    init_opacity: float,
    init_scale: float,
    init_scale_mode: str,
    init_knn_k: int,
    init_scale_mult: float,
    init_scale_min: float,
    init_scale_max: float | None,
):
    points = scene.points.astype(np.float32)
    if len(points) == 0:
        raise ValueError("COLMAP scene has no sparse points")
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        rgb = scene.points_rgb[idx]
    else:
        rgb = scene.points_rgb[: len(points)]
    if rgb.max(initial=0.0) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.02, 0.98)
    n = len(points)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    if init_scale_mode.lower() == "knn":
        scale_1d = _knn_init_scales(
            points,
            init_scale,
            init_knn_k,
            init_scale_mult,
            init_scale_min,
            init_scale_max,
        )
        scales = np.repeat(scale_1d[:, None], 3, axis=1)
    elif init_scale_mode.lower() == "constant":
        scales = np.full((n, 3), init_scale, dtype=np.float32)
    else:
        raise ValueError(f"unknown init_scale_mode: {init_scale_mode!r}")
    opac = np.full((n,), init_opacity, dtype=np.float32)
    return {
        "means3d": mx.array(points),
        "log_scales": mx.array(np.log(scales).astype(np.float32)),
        "quats": mx.array(quats),
        "opacities_raw": mx.array(_logit(opac)),
        "sh0": rgb_to_sh0(mx.array(rgb)),
        "shN": mx.zeros((n, 15, 3), dtype=mx.float32),
    }


def _intersection_counts(params, viewmats, Ks, width, height, splat_mode: str):
    """Exact per-(view, gaussian) tile-intersection counts, lazily.

    Projects the current gaussians for the given cameras and runs the exact
    opacity/tile-culling count kernels. Returns the lazy ``(V, N)`` counts
    array (callers reduce and ``mx.eval`` as needed).
    """
    if splat_mode == "2dgs":
        radii, means2d, depths, ray, _normals = project_gaussians_2dgs(
            params["means3d"],
            params["log_scales"],
            params["quats"],
            viewmats,
            Ks,
            width,
            height,
        )
        opacities = mx.where(
            (depths > NEAR_PLANE) & (depths < FAR_PLANE),
            mx.sigmoid(params["opacities_raw"])[None, :],
            0.0,
        )
        return _count_bbox_intersections(means2d, ray, opacities, radii, width, height)
    means2d, conics, depths = project_gaussians(
        params["means3d"],
        params["log_scales"],
        params["quats"],
        viewmats,
        Ks,
        width,
        height,
    )
    opacities = mx.where(
        (depths > NEAR_PLANE) & (depths < FAR_PLANE),
        mx.sigmoid(params["opacities_raw"])[None, :],
        0.0,
    )
    return _count_tile_intersections(means2d, conics, opacities, width, height)


def _choose_bins(
    params,
    viewmats,
    Ks,
    width,
    height,
    mode: str,
    min_pad: int,
    margin: float,
    camera_batch: int,
    capacity_stat: str,
    splat_mode: str,
):
    """Pick compact-bin settings for the epoch.

    ``auto`` counts exact tile intersections for all train cameras and sizes a
    static INVALID-padded compact buffer from mean/p95/worst per-view counts.
    Integer values keep the historical per-gaussian ``bin_pad`` sort length,
    but still route through the compact builder.
    """
    mode_l = mode.lower()
    if mode_l in {"none", "exact"}:
        return None, None, "exact"
    batch = max(1, int(camera_batch))
    n = int(params["means3d"].shape[0])
    ntiles = _num_tiles(width, height)
    exact_capacity = ntiles * n * batch
    if mode_l != "auto":
        pad = int(mode)
        capacity = max(1, min(exact_capacity, n * batch * pad))
        return None, capacity, f"capacity={capacity} (pad={pad})"

    counts = _intersection_counts(params, viewmats, Ks, width, height, splat_mode)
    mx.eval(counts)
    counts_np = np.asarray(counts)
    per_view = counts_np.sum(axis=1)
    n_views = int(per_view.shape[0])
    if n_views == 0:
        expected = 0.0
    elif capacity_stat == "mean":
        expected = float(per_view.mean()) * batch
    elif capacity_stat == "p95":
        expected = float(np.percentile(per_view, 95)) * batch
    elif capacity_stat == "worst":
        expected = float(per_view.max()) * batch if batch > n_views else float(np.sort(per_view)[-batch:].sum())
    else:
        raise ValueError(f"unknown bin capacity stat: {capacity_stat!r}")

    min_capacity = int(min_pad) * n * batch
    capacity = max(1, min(exact_capacity, max(min_capacity, math.ceil(expected * float(margin)))))
    utilization = expected / capacity if capacity else 0.0
    per_gaussian = counts_np.reshape(-1)
    return (
        None,
        capacity,
        f"capacity={capacity} {capacity_stat}≈{expected:.0f} ({utilization:.0%} used, margin {margin}x, "
        f"tiles/G mean={per_gaussian.mean():.1f} p99={np.percentile(per_gaussian, 99):.0f} "
        f"max={per_gaussian.max(initial=0)})",
    )


def _count_batch_intersections(params, viewmats, Ks, width, height, splat_mode: str) -> int:
    """Exact compact-bin intersection count for the current sampled batch."""
    total = mx.sum(_intersection_counts(params, viewmats, Ks, width, height, splat_mode))
    mx.eval(total)
    return int(total)


def _capacity_for_count(
    real_count: int,
    n: int,
    batch: int,
    width: int,
    height: int,
    min_pad: int,
    margin: float,
) -> int:
    exact_capacity = _num_tiles(width, height) * n * batch
    min_capacity = int(min_pad) * n * batch
    return max(1, min(exact_capacity, max(min_capacity, math.ceil(real_count * float(margin)))))


def _depth_panel(depth: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    """Depth preview panel in RGB (INFERNO colormap), invalid pixels black.

    A perceptual colormap makes the depth structure legible where the old
    grayscale panel read as near-black once normalized against outlier depths.
    """
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)
    denom = max(depth_max - depth_min, 1e-6)
    scaled = np.clip((depth - depth_min) / denom, 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    # applyColorMap returns BGR; flip to RGB so the writer's final ::-1 (RGB->BGR)
    # lands correct, matching the rgb/target panels.
    panel = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)[:, :, ::-1].copy()
    panel[~valid] = 0
    return panel


def _render_view(
    params,
    viewmat,
    K,
    width,
    height,
    splat_mode: str,
    bin_pad,
    bin_capacity,
    return_depth: bool = False,
    active_sh_degree: int = 0,
    photometric_params: dict[str, mx.array] | None = None,
    camera_index: int | None = None,
):
    """Render one view with the current params (no absgrad, lazy result).

    Shared by the video preview (train bins, cosmetic) and eval (compact-exact
    bins sized by :func:`eval_capacity`).
    """
    colors = view_dependent_colors(params, viewmat, active_sh_degree)
    if photometric_params is not None:
        if camera_index is None:
            raise ValueError("camera_index is required with photometric_params")
        colors = apply_photometric(colors, photometric_params, mx.array([camera_index], dtype=mx.int32))
    if splat_mode == "2dgs":
        radii, means2d, depths, ray, _normals = project_gaussians_2dgs(
            params["means3d"],
            params["log_scales"],
            params["quats"],
            viewmat,
            K,
            width,
            height,
        )
        return rasterize2dgs_fused(
            means2d,
            ray,
            mx.sigmoid(params["opacities_raw"]),
            colors,
            mx.zeros((3,), dtype=mx.float32),
            depths,
            radii,
            height,
            width,
            absgrad_sink=None,
            bin_pad=bin_pad,
            bin_capacity=bin_capacity,
            return_aux=return_depth,
        )
    means2d, conics, depths = project_gaussians(
        params["means3d"],
        params["log_scales"],
        params["quats"],
        viewmat,
        K,
        width,
        height,
    )
    return rasterize3d_fused(
        means2d,
        conics,
        mx.sigmoid(params["opacities_raw"]),
        colors,
        mx.zeros((3,), dtype=mx.float32),
        depths,
        height,
        width,
        absgrad_sink=None,
        bin_pad=bin_pad,
        bin_capacity=bin_capacity,
        return_aux=return_depth,
    )


def eval_capacity(max_count: int, n: int, ntiles: int) -> int:
    """Compact-exact eval bin capacity: geometric 1.25x bucket, capped at exact.

    Bucketing keeps capacity-driven shape changes rare across epochs without
    power-of-two's up-to-2x invalid-key sort waste. Invariants: capacity >= 1,
    capacity <= n * ntiles (always sufficient), capacity >= max_count unless
    capped at exact.
    """
    assert n > 0 and ntiles > 0, "eval with empty params/tiles"
    exact = n * ntiles
    bucketed = math.ceil(1.25 ** math.ceil(math.log(max(max_count, 1)) / math.log(1.25)))
    return max(1, min(max(bucketed, max_count), exact))


def _evaluate(
    params,
    targets_u8,
    viewmats,
    Ks,
    width,
    height,
    splat_mode: str,
    count_batch: int,
    eval_step: int,
    log,
    save_dir: Path | None = None,
    smoke_run: bool = False,
    active_sh_degree: int = 0,
    image_ids: list[str | int] | None = None,
    metrics_out_dir: Path | None = None,
    metric_suite: RGBMetricSuite | None = None,
    final: bool = False,
    photometric_params: dict[str, mx.array] | None = None,
    photometric_camera_indices: list[int] | None = None,
    enable_lpips: bool = False,
):
    """Held-out evaluation at a training-step checkpoint: mean per-image PSNR/SSIM.

    Bins are compact-exact with one bucketed capacity for the whole pass:
    the count pass runs in chunks of ``count_batch`` views (2DGS projected
    state is heavy) and reduces each chunk to per-view totals immediately —
    the (chunk, N) counts are never retained.
    """
    from drawingwithgaussians.losses import ssim as _ssim

    t_start = time.perf_counter()
    n_views = int(targets_u8.shape[0])
    per_view_totals = []
    for i in range(0, n_views, max(1, count_batch)):
        counts = _intersection_counts(
            params,
            viewmats[i : i + count_batch],
            Ks[i : i + count_batch],
            width,
            height,
            splat_mode,
        )
        totals = mx.sum(counts, axis=1)
        mx.eval(totals)
        per_view_totals.append(np.asarray(totals))
    per_view = np.concatenate(per_view_totals)
    max_count = int(per_view.max(initial=0))
    n = int(params["means3d"].shape[0])
    capacity = eval_capacity(max_count, n, _num_tiles(width, height))
    assert capacity >= max_count or capacity == n * _num_tiles(width, height)
    t_counts = time.perf_counter() - t_start

    psnrs, ssims, view_times = [], [], []
    structured: list[ViewMetrics] = []
    corrected_structured: list[ViewMetrics] = []
    raw_predictions: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    corrected_predictions: list[np.ndarray] = []
    for v in range(n_views):
        t0 = time.perf_counter()
        rendered = _render_view(
            params,
            viewmats[v],
            Ks[v],
            width,
            height,
            splat_mode,
            None,
            capacity,
            active_sh_degree=active_sh_degree,
        )
        target = targets_u8[v].astype(mx.float32) / 255.0
        rendered_clamped = mx.clip(rendered, 0.0, 1.0)
        mse = mx.mean(mx.square(rendered_clamped - target))
        ssim_v = _ssim(rendered_clamped, target)
        mx.eval(rendered, rendered_clamped, mse, ssim_v)
        view_times.append(time.perf_counter() - t0)
        psnrs.append(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))
        ssims.append(float(ssim_v))
        raw_np = np.asarray(rendered)
        target_np = np.asarray(target)
        raw_predictions.append(raw_np)
        raw_targets.append(target_np)
        if metric_suite is not None:
            structured.append(metric_suite.view((image_ids or list(range(n_views)))[v], raw_np, target_np))
        else:
            structured.append(
                ViewMetrics(
                    image_id=str((image_ids or list(range(n_views)))[v]),
                    psnr=psnrs[-1],
                    ssim=ssims[-1],
                    lpips_alex=None,
                    raw_underflow_fraction=float(np.mean(raw_np < 0.0)),
                    raw_overflow_fraction=float(np.mean(raw_np > 1.0)),
                )
            )
        if photometric_params is not None:
            corrected = _render_view(
                params,
                viewmats[v],
                Ks[v],
                width,
                height,
                splat_mode,
                None,
                capacity,
                active_sh_degree=active_sh_degree,
                photometric_params=photometric_params,
                camera_index=(photometric_camera_indices or list(range(n_views)))[v],
            )
            mx.eval(corrected)
            corrected_np = np.asarray(corrected)
            corrected_predictions.append(corrected_np)
            if metric_suite is not None:
                corrected_structured.append(
                    metric_suite.view((image_ids or list(range(n_views)))[v], corrected_np, target_np)
                )
            else:
                corrected_clamped = np.clip(corrected_np, 0, 1)
                corrected_mse = float(np.mean((corrected_clamped - target_np) ** 2))
                corrected_ssim = _ssim(mx.array(corrected_clamped), target)
                mx.eval(corrected_ssim)
                corrected_structured.append(
                    ViewMetrics(
                        image_id=str((image_ids or list(range(n_views)))[v]),
                        psnr=10.0 * math.log10(1.0 / max(corrected_mse, 1e-12)),
                        ssim=float(corrected_ssim),
                        lpips_alex=None,
                        raw_underflow_fraction=float(np.mean(corrected_np < 0)),
                        raw_overflow_fraction=float(np.mean(corrected_np > 1)),
                    )
                )
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            img = (np.clip(np.array(rendered), 0.0, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"val_{v:04d}_step{eval_step:06d}.png"), img[:, :, ::-1])

    if enable_lpips:
        scores = lpips_alex_mlx(raw_predictions, raw_targets)
        structured = [replace(v, lpips_alex=score) for v, score in zip(structured, scores, strict=True)]
        if corrected_predictions:
            corrected_scores = lpips_alex_mlx(corrected_predictions, raw_targets)
            corrected_structured = [
                replace(v, lpips_alex=score) for v, score in zip(corrected_structured, corrected_scores, strict=True)
            ]
    wall = time.perf_counter() - t_start
    steady = float(np.mean(view_times[1:])) if len(view_times) > 1 else view_times[0]
    log.info(
        "eval step %d%s: %d views, PSNR %.2f dB (min %.2f), SSIM %.4f (min %.4f) "
        "[mean per-image] | capacity=%d max_count=%d waste=%.0f%% | "
        "count pass %.2fs, first view %.3fs, steady/view %.3fs, wall %.2fs",
        eval_step,
        " (smoke-length run; eval-share budget does not apply)" if smoke_run else "",
        n_views,
        float(np.mean(psnrs)),
        float(np.min(psnrs)),
        float(np.mean(ssims)),
        float(np.min(ssims)),
        capacity,
        max_count,
        100.0 * (1.0 - max_count / capacity) if capacity else 0.0,
        t_counts,
        view_times[0],
        steady,
        wall,
    )
    if metrics_out_dir is not None:
        write_metrics(metrics_out_dir, eval_step, structured, final=final)
        if corrected_structured:
            write_metrics(metrics_out_dir, eval_step, corrected_structured, final=final, label="corrected")
    return float(np.mean(psnrs)), float(np.mean(ssims)), wall


def _export_dtu_renders(
    params,
    image_paths,
    viewmats,
    Ks,
    width,
    height,
    active_sh_degree,
    count_batch,
    out_dir,
    log,
):
    """Render canonical 2DGS RGB, alpha, and median depth for DTU fusion."""
    view_numbers = []
    for path in image_paths:
        match = re.search(r"clean_(\d{3})_", path.name)
        if match is None:
            raise ValueError(f"DTU render export requires clean_###_* image names, got {path.name!r}")
        view_numbers.append(int(match.group(1)))

    totals = []
    for start in range(0, len(view_numbers), max(1, count_batch)):
        counts = _intersection_counts(
            params,
            viewmats[start : start + count_batch],
            Ks[start : start + count_batch],
            width,
            height,
            "2dgs",
        )
        batch_totals = mx.sum(counts, axis=1)
        mx.eval(batch_totals)
        totals.append(np.asarray(batch_totals))
    max_count = int(np.concatenate(totals).max(initial=0))
    capacity = eval_capacity(max_count, int(params["means3d"].shape[0]), _num_tiles(width, height))

    out_dir.mkdir(parents=True, exist_ok=True)
    for view, viewmat, K in zip(view_numbers, viewmats, Ks, strict=True):
        rgb, aux = _render_view(
            params,
            viewmat,
            K,
            width,
            height,
            "2dgs",
            None,
            capacity,
            return_depth=True,
            active_sh_degree=active_sh_degree,
        )
        mx.eval(rgb, aux["alpha"], aux["median_depth"])
        np.save(out_dir / f"rgb_{view:03d}.npy", np.asarray(rgb, dtype=np.float32))
        np.save(out_dir / f"alpha_{view:03d}.npy", np.asarray(aux["alpha"][..., 0], dtype=np.float32))
        np.save(out_dir / f"K_{view:03d}.npy", np.asarray(K, dtype=np.float32))
        np.save(
            out_dir / f"median_depth_{view:03d}.npy",
            np.asarray(aux["median_depth"][..., 0], dtype=np.float32),
        )
    log.info(
        "saved %d DTU fusion views at %dx%d to %s",
        len(view_numbers),
        width,
        height,
        out_dir,
    )


def _log_shadow_densification(log, epoch, params, old_sig, new_sig, vis, args, scene_scale, grid):
    """Legacy/normalized signal telemetry using the live densify mask helper."""
    p_np = {k: np.asarray(v) for k, v in params.items()}
    old_np, new_np, vis_np = np.asarray(old_sig), np.asarray(new_sig), np.asarray(vis)

    def pct(a):
        return tuple(float(np.percentile(a, q)) for q in (50, 90, 99))

    log.info(
        "  signal comparison epoch %d: live=%s N=%d "
        "vis/G p50/p90/max=%.0f/%.0f/%.0f | "
        "old p50/p90/p99=%.3g/%.3g/%.3g | new p50/p90/p99=%.3g/%.3g/%.3g",
        epoch,
        args.densify_signal,
        len(new_np),
        float(np.percentile(vis_np, 50)),
        float(np.percentile(vis_np, 90)),
        float(vis_np.max(initial=0)),
        *pct(old_np),
        *pct(new_np),
    )
    live_np = new_np if args.densify_signal == "normalized" else old_np
    live_d, live_s, _live_e, *_ = densify_masks(
        p_np,
        live_np,
        args.grad_thr,
        args.grow_scale,
        scene_scale,
        args.prune_opa,
        args.prune_scale3d,
    )
    live_grow = live_d | live_s
    for thr in grid:
        nd, ns, erase, *_ = densify_masks(
            p_np,
            new_np,
            thr,
            args.grow_scale,
            scene_scale,
            args.prune_opa,
            args.prune_scale3d,
        )
        new_grow = nd | ns
        high_erased = erase & (new_np > thr)
        inter = int((new_grow & live_grow).sum())
        union = int((new_grow | live_grow).sum())
        log.info(
            "    thr=%.0e: dupli=%d split=%d grow=%d erase=%d " "(high-erased=%d, live-grow=%d, IoU=%.2f)",
            thr,
            int(nd.sum()),
            int(ns.sum()),
            int(new_grow.sum()),
            int(erase.sum()),
            int(high_erased.sum()),
            int(live_grow.sum()),
            (inter / union) if union else 1.0,
        )


def _build_optimizer(params, args, total_steps, scene_scale, segment_start, restart_period, old_opt, decay_from_step=0):
    """Set up the optimizer for one split segment, applying the Stage 2b LR ladder knobs.

    L2 (``means_lr_scene_scale``): means LR scaled by ``scene_scale`` (gsplat).
    L3 (``means_only_schedule``): ``means_mode`` schedule restricted to means.
    L4a (``global_schedule``): the LR schedule spans the whole run instead of
    restarting each split segment. Off by default so SGDR warm-restarts survive
    (the load-bearing behavior from EXPERIMENTS Exp 8). The step offset follows the
    pinned carry rule: ``carried_step = old_step if carrying else 0``.

    ``decay_from_step`` (DashGaussian LR delay) holds the schedule at its initial
    value until that global step; only meaningful with ``global_schedule`` (a
    global step notion) and ``means_mode != const``, else inert.
    """
    lr = args.lr
    if isinstance(lr, dict) and args.means_lr_scene_scale:
        lr = {**lr, "means": lr["means"] * float(scene_scale)}
    if args.global_schedule:
        carried_step = int(get_opt_step(old_opt)) if (args.carry_optimizer_state and old_opt is not None) else 0
        step_offset = int(segment_start) - carried_step
    else:
        step_offset = 0
    return set_up_optimizer_3d(
        params,
        lr=lr,
        max_steps=total_steps,
        mode=args.means_mode,
        restart_period=max(1, int(restart_period)),
        step_offset=step_offset,
        means_only_schedule=(isinstance(lr, dict) and args.means_only_schedule),
        decay_from_step=decay_from_step,
        selective=bool(getattr(args, "selective_adam", False)),
    )


def _path_from_config(path: str | Path) -> Path:
    return Path(hydra.utils.to_absolute_path(os.path.expanduser(str(path))))


def _none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_lr(value: Any) -> float | dict[str, float]:
    """``optim.lr`` may be a scalar (single Adam) or a group-keyed mapping
    (per-group Adams via MultiOptimizer). Returns a plain float or a plain
    ``{group: float}`` dict (OmegaConf containers flattened)."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "items"):
        return {str(k): float(v) for k, v in value.items()}
    return float(value)


def _parse_overflow_mode(value):
    """Normalize the overflow policy, retaining bool config compatibility."""
    if isinstance(value, bool):
        return "preflight" if value else "off"
    mode = str(value).lower()
    aliases = {"true": "preflight", "false": "off"}
    mode = aliases.get(mode, mode)
    if mode not in {"preflight", "lazy", "off"}:
        raise ValueError("gaussians.bin_check_overflow must be preflight, lazy, or off")
    return mode


class _ViewSampler:
    """Deterministic fixed-size random or shuffled camera batches."""

    def __init__(self, n_views, batch_size, seed, mode="random"):
        self.n_views = int(n_views)
        self.batch_size = int(batch_size)
        self.mode = str(mode).lower()
        if self.n_views <= 0 or self.batch_size <= 0:
            raise ValueError("view sampler requires positive view and batch counts")
        if self.mode not in {"random", "shuffle"}:
            raise ValueError("train.view_sampling must be 'random' or 'shuffle'")
        self.rng = np.random.default_rng(int(seed))
        self._remaining = np.empty((0,), dtype=np.int64)

    def _next_permutation(self):
        self._remaining = self.rng.permutation(self.n_views)

    def sample(self):
        if self.mode == "random":
            return self.rng.choice(
                self.n_views,
                size=self.batch_size,
                replace=self.n_views < self.batch_size,
            )

        selected: list[int] = []
        while len(selected) < self.batch_size:
            if self._remaining.size == 0:
                self._next_permutation()
            need = self.batch_size - len(selected)
            if self.n_views < self.batch_size:
                take = min(need, int(self._remaining.size))
                selected.extend(self._remaining[:take].tolist())
                self._remaining = self._remaining[take:]
                continue

            if self._remaining.size >= need:
                selected.extend(self._remaining[:need].tolist())
                self._remaining = self._remaining[need:]
                continue

            # Wrap a short tail into the next permutation without repeating
            # any of those tail views inside this batch. Elements skipped for
            # the fill remain in the next permutation stream.
            tail = self._remaining.tolist()
            selected.extend(tail)
            need = self.batch_size - len(selected)
            next_perm = self.rng.permutation(self.n_views)
            tail_set = set(tail)
            fill_mask = np.array([value not in tail_set for value in next_perm], dtype=bool)
            fill = next_perm[fill_mask][:need]
            selected.extend(fill.tolist())
            used = set(fill.tolist())
            self._remaining = np.array(
                [value for value in next_perm if value not in used],
                dtype=np.int64,
            )

        return np.asarray(selected, dtype=np.int64)


def _regularizer_weight(base_weight, start_frac, epoch_start_step, total_steps):
    """Split-segment-constant 2DGS regularizer weight with fractional warm-up."""
    if float(base_weight) <= 0.0:
        return 0.0
    threshold = float(start_frac) * int(total_steps)
    return float(base_weight) if int(epoch_start_step) >= threshold else 0.0


def _split_steps(split_iters, total_steps):
    """Sanitize the explicit list of completed-step counts where split/prune fires.

    Splits happen after the optimizer has completed that many steps, so only
    events with ``0 < step < total_steps`` are valid; the result is deduplicated
    and sorted. An empty list disables densification.
    """
    total_steps = int(total_steps)
    return sorted({int(s) for s in split_iters if 0 < int(s) < total_steps})


def _split_index_by_step(split_steps):
    """Map a refinement step to its one-based refinement-event index.

    Resolution, SH, and regularizer transitions also create trainer segments;
    they must not alter refinement RNG keys or opacity-reset cadence.
    """
    return {int(step): index for index, step in enumerate(split_steps, start=1)}


@hydra.main(version_base=None, config_path="./configs")
def train_colmap3d(cfg: DictConfig):
    run_started_at = time.perf_counter()
    mx.reset_peak_memory()
    log = logging.getLogger(__name__)
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    if cfg.optim.loss.name != "pixel":
        raise NotImplementedError(f"loss {cfg.optim.loss.name!r} is not supported; only 'pixel'.")

    mode = str(cfg.gaussians.get("mode", "3dgs")).lower()
    if mode not in {"3dgs", "2dgs"}:
        raise ValueError(f"gaussians.mode must be '3dgs' or '2dgs', got {mode!r}")

    bin_capacity_stat = str(cfg.gaussians.get("bin_capacity_stat", "mean")).lower()
    if bin_capacity_stat not in {"mean", "p95", "worst"}:
        raise ValueError(
            "gaussians.bin_capacity_stat must be one of 'mean', 'p95', or 'worst', " f"got {bin_capacity_stat!r}"
        )

    args = SimpleNamespace(
        data_dir=_path_from_config(cfg.data.dir),
        out_dir=Path(hydra_cfg["runtime"]["output_dir"]),
        data_factor=int(cfg.data.factor),
        max_side=_none_or_int(cfg.data.get("max_side", None)),
        test_every=int(cfg.data.test_every),
        max_init_points=int(cfg.gaussians.max_init_points),
        mode=mode,
        normalize=bool(cfg.data.normalize),
        seed=int(cfg.optim.seed),
        steps=int(cfg.optim.num_steps),
        lr=_parse_lr(cfg.optim.lr),
        means_lr_scene_scale=bool(cfg.optim.get("means_lr_scene_scale", True)),
        means_only_schedule=bool(cfg.optim.get("means_only_schedule", True)),
        global_schedule=bool(cfg.optim.get("global_schedule", False)),
        means_mode=str(cfg.optim.get("means_mode", "const")),
        ssim_weight=float(cfg.optim.loss.ssim_weight),
        normal_weight=float(cfg.optim.loss.get("normal_weight", 0.0)),
        distortion_weight=float(cfg.optim.loss.get("distortion_weight", 0.0)),
        normal_start_frac=float(cfg.optim.loss.get("normal_start_frac", 0.0)),
        distortion_start_frac=float(cfg.optim.loss.get("distortion_start_frac", 0.0)),
        normal_depth_mode=str(cfg.optim.loss.get("normal_depth_mode", "expected")),
        init_opacity=float(cfg.gaussians.init_opacity),
        init_scale=float(cfg.gaussians.init_scale),
        init_scale_mode=str(cfg.gaussians.init_scale_mode),
        init_knn_k=int(cfg.gaussians.init_knn_k),
        init_scale_mult=float(cfg.gaussians.init_scale_mult),
        init_scale_min=float(cfg.gaussians.init_scale_min),
        init_scale_max=_none_or_float(cfg.gaussians.get("init_scale_max", None)),
        densify_signal=str(cfg.gaussians.get("densify_signal", "normalized")).lower(),
        grad_thr=float(cfg.gaussians.grad_thr),
        split_iters=[int(s) for s in cfg.gaussians.get("split_iters", [])],
        shadow_signal=bool(cfg.gaussians.get("shadow_signal", False)),
        grow_scale=float(cfg.gaussians.grow_scale),
        prune_opa=float(cfg.gaussians.prune_opa),
        prune_scale3d=_none_or_float(cfg.gaussians.get("prune_scale3d", None)),
        reset_opacity_every=int(cfg.gaussians.get("reset_opacity_every", 0)),
        carry_optimizer_state=bool(cfg.gaussians.get("carry_optimizer_state", False)),
        selective_adam=bool(cfg.gaussians.get("selective_adam", False)),
        utilization_telemetry=bool(cfg.gaussians.get("utilization_telemetry", True)),
        utilization_pruning=bool(cfg.gaussians.get("utilization_pruning", False)),
        utilization_ema_decay=float(cfg.gaussians.get("utilization_ema_decay", 0.95)),
        utilization_threshold=float(cfg.gaussians.get("utilization_threshold", 0.0)),
        utilization_warmup=int(cfg.gaussians.get("utilization_warmup", 1000)),
        utilization_min_observations=int(cfg.gaussians.get("utilization_min_observations", 100)),
        utilization_grace=int(cfg.gaussians.get("utilization_grace", 500)),
        utilization_low_windows=int(cfg.gaussians.get("utilization_low_windows", 3)),
        photometric_correction=bool(cfg.optim.get("photometric_correction", False)),
        photometric_lr=float(cfg.optim.get("photometric_lr", 1e-3)),
        photometric_regularizer=float(cfg.optim.get("photometric_regularizer", 1e-3)),
        camera_batch=int(cfg.train.camera_batch),
        view_sampling=str(cfg.train.get("view_sampling", "random")).lower(),
        bin_pad=cfg.gaussians.get("bin_pad", "auto"),
        bin_pad_min=int(cfg.gaussians.get("bin_pad_min", 16)),
        bin_pad_margin=float(cfg.gaussians.get("bin_pad_margin", 2.0)),
        bin_overflow_margin=float(cfg.gaussians.get("bin_overflow_margin", 1.25)),
        bin_capacity_stat=bin_capacity_stat,
        bin_check_overflow=_parse_overflow_mode(cfg.gaussians.get("bin_check_overflow", "preflight")),
        log_every=int(cfg.train.log_frequency),
        eval_every=int(cfg.train.get("eval_every", 1)),
        eval_count_batch_size=int(cfg.train.get("eval_count_batch_size", 8)),
        eval_save_renders=bool(cfg.train.get("eval_save_renders", False)),
        eval_lpips=bool(cfg.train.get("eval_lpips", True)),
        save_video=bool(cfg.train.get("save_video", False)),
        save_depth=bool(cfg.train.get("save_depth", False)),
        save_dtu_renders=bool(cfg.train.get("save_dtu_renders", False)),
        dtu_render_max_side=_none_or_int(cfg.train.get("dtu_render_max_side", None)),
        video_every=int(cfg.train.get("video_every", 50)),
        video_index=int(cfg.train.get("video_index", 0)),
        # --- DashGaussian scheduling (arXiv:2503.18402); defaults preserve behavior ---
        resolution_mode=str(cfg.gaussians.get("resolution_mode", "const")).lower(),
        start_significance_factor=float(cfg.gaussians.get("start_significance_factor", 4.0)),
        increase_reso_frac=float(cfg.gaussians.get("increase_reso_frac", 0.5)),
        densify_mode=str(cfg.gaussians.get("densify_mode", "free")).lower(),
        max_n_gaussian=int(cfg.gaussians.get("max_n_gaussian", -1)),
        budget_gamma=float(cfg.gaussians.get("budget_gamma", 0.98)),
        budget_eta=float(cfg.gaussians.get("budget_eta", 1.0)),
        max_densify_rate_per_step=float(cfg.gaussians.get("max_densify_rate_per_step", 0.2)),
        budget_grad_percentile=float(cfg.gaussians.get("budget_grad_percentile", 50.0)),
        lr_decay_from_full_res=bool(cfg.gaussians.get("lr_decay_from_full_res", False)),
        sh_degree=int(cfg.gaussians.get("sh_degree", 3)),
        sh_interval=int(cfg.gaussians.get("sh_interval", 1000)),
    )
    if args.resolution_mode not in {"const", "freq"}:
        raise ValueError("gaussians.resolution_mode must be 'const' or 'freq'")
    if args.densify_mode not in {"free", "budget"}:
        raise ValueError("gaussians.densify_mode must be 'free' or 'budget'")
    if args.densify_signal not in {"legacy", "normalized"}:
        raise ValueError("gaussians.densify_signal must be 'legacy' or 'normalized'")
    if args.view_sampling not in {"random", "shuffle"}:
        raise ValueError("train.view_sampling must be 'random' or 'shuffle'")
    if args.steps <= 0:
        raise ValueError("optim.num_steps must be positive")
    if args.utilization_pruning and args.utilization_threshold <= 0:
        raise ValueError("utilization_pruning requires a positive, pre-tuned utilization_threshold")
    if not 0 <= args.sh_degree <= 3:
        raise ValueError("gaussians.sh_degree must be in [0, 3]")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scene, train_indices, val_indices = _load_colmap_scene(
        args.data_dir, args.data_factor, args.normalize, args.test_every
    )
    log.info(
        "loaded %s: %d train / %d val images, %d sfm points, scene_scale=%.3f",
        args.data_dir,
        len(train_indices),
        len(val_indices),
        len(scene.points),
        scene.scene_scale,
    )
    if not train_indices:
        raise ValueError("train split is empty")
    scale = float(scene.normalization_scale)
    center = np.asarray(scene.normalization_center, dtype=np.float64)
    original_to_normalized = np.eye(4, dtype=np.float64)
    original_to_normalized[:3, :3] /= scale
    original_to_normalized[:3, 3] = -center / scale
    (args.out_dir / "normalization.json").write_text(
        json.dumps(
            {
                "center": center.tolist(),
                "scale": scale,
                "original_to_normalized": original_to_normalized.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    targets_u8, viewmats_all, Ks_all = _preload_views(scene, train_indices, args.max_side, log)
    height, width = int(targets_u8.shape[1]), int(targets_u8.shape[2])
    eval_enabled = args.eval_every > 0 and len(val_indices) > 0
    val_targets_u8 = val_viewmats = val_Ks = None
    if args.eval_every > 0 and not val_indices:
        log.info("val split is empty; eval disabled")
    if eval_enabled:
        val_targets_u8, val_viewmats, val_Ks = _preload_views(
            scene, val_indices, args.max_side, log, target_hw=(height, width)
        )
    # PSNR, SSIM, and LPIPS-Alex are evaluated with MLX operations.
    metric_suite = None
    manifest_sampler = _ViewSampler(len(train_indices), args.camera_batch, args.seed, args.view_sampling)
    camera_batches = [manifest_sampler.sample().tolist() for _ in range(args.steps)]
    write_run_manifest(
        args.out_dir,
        OmegaConf.to_container(cfg, resolve=True),
        args.seed,
        [scene.image_paths[i].name for i in train_indices],
        [scene.image_paths[i].name for i in val_indices],
        camera_batches,
        Path(__file__).resolve().parent,
    )
    video_pos = max(0, min(len(train_indices) - 1, args.video_index))
    video_viewmat, video_K = viewmats_all[video_pos], Ks_all[video_pos]
    params = _sample_init_points(
        scene,
        args.max_init_points,
        args.seed,
        args.init_opacity,
        args.init_scale,
        args.init_scale_mode,
        args.init_knn_k,
        args.init_scale_mult,
        args.init_scale_min,
        args.init_scale_max,
    )
    init_scales = np.exp(np.asarray(params["log_scales"]))
    log.info(
        "init scales: mode=%s p50=%.4g p95=%.4g max=%.4g",
        args.init_scale_mode,
        float(np.percentile(init_scales, 50)),
        float(np.percentile(init_scales, 95)),
        float(np.max(init_scales)),
    )
    total_steps = args.steps
    split_steps = _split_steps(args.split_iters, total_steps)
    split_index_by_step = _split_index_by_step(split_steps)

    # --- DashGaussian resolution schedule (freq): coarse->fine render res -----
    # Resolution transitions are independent of the densification boundaries and
    # ramp over the first `increase_reso_frac` of training (the long final split
    # segment would otherwise pin the whole tail to a coarse res). Extra segment
    # boundaries are merged in so the compiled step recompiles at each new res.
    increase_reso_until = int(args.increase_reso_frac * total_steps)
    if args.resolution_mode == "freq":
        n_spec = min(16, targets_u8.shape[0])
        spec_idx = np.linspace(0, targets_u8.shape[0] - 1, n_spec).astype(int)
        reso_sample = np.asarray(targets_u8[mx.array(spec_idx)]).astype(np.float32) / 255.0
        reso_segs = resolution_segments(reso_sample, increase_reso_until, args.start_significance_factor)
    else:
        reso_segs = [(0, 1)]
    reso_boundary_steps = [s for s, f in reso_segs if 0 < s < total_steps]
    log.info("resolution schedule (mode=%s): %s", args.resolution_mode, reso_segs)

    def factor_for_step(step):
        f = 1
        for s, fac in reso_segs:
            if s <= step:
                f = fac
            else:
                break
        return f

    # --- DashGaussian primitive-count budget (only used in densify_mode=budget) ---
    budget = MomentumBudget(
        params["means3d"].shape[0],
        gamma=args.budget_gamma,
        eta=args.budget_eta,
        max_n_gaussian=args.max_n_gaussian,
    )
    lr_decay_from_step = increase_reso_until if (args.lr_decay_from_full_res and args.resolution_mode == "freq") else 0

    sh_boundary_steps = (
        [
            s
            for s in range(args.sh_interval, total_steps, args.sh_interval)
            if sh_degree_for_step(s, args.sh_interval, args.sh_degree)
            != sh_degree_for_step(s - 1, args.sh_interval, args.sh_degree)
        ]
        if args.sh_interval > 0
        else []
    )
    geometry_boundary_steps = set()
    if args.mode == "2dgs":
        for weight, fraction in (
            (args.normal_weight, args.normal_start_frac),
            (args.distortion_weight, args.distortion_start_frac),
        ):
            boundary = int(math.ceil(float(fraction) * total_steps))
            if weight > 0 and 0 < boundary < total_steps:
                geometry_boundary_steps.add(boundary)
    segment_ends = sorted(
        set(split_steps) | set(reso_boundary_steps) | set(sh_boundary_steps) | geometry_boundary_steps | {total_steps}
    )
    opt = _build_optimizer(
        params,
        args,
        total_steps,
        scene.scene_scale,
        segment_start=0,
        restart_period=segment_ends[0],
        old_opt=None,
        decay_from_step=lr_decay_from_step,
    )
    mx.eval(*params.values())
    utilization = init_utilization(params["means3d"].shape[0])

    frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    photometric_params = init_photometric(len(scene.image_paths))
    photometric_opt = mlx_optim.Adam(learning_rate=args.photometric_lr, bias_correction=True)
    photometric_opt.init(photometric_params)

    def make_step(
        bin_pad, bin_capacity, normal_weight, distortion_weight, width_e, height_e, active_sh_degree
    ) -> tuple[Any, list[Any]]:
        need_counts = (
            args.selective_adam
            or args.utilization_telemetry
            or args.densify_signal == "normalized"
            or args.shadow_signal
            or args.bin_check_overflow == "lazy"
        )

        def loss_fn(params, targets_u8, viewmats, Ks, offset_zeros, absgrad_zeros, photo_params, camera_indices):
            # One batched render for the whole camera batch (gsplat's
            # [..., C, N] convention): batched projection broadcasts the
            # camera entries, the rasterizer launches once with grid z = B,
            # and the shared offset/absgrad sinks sum their cotangents over
            # views. Targets arrive uint8 and are normalized on the active MLX
            # stream. Loss is the mean over all views.
            targets = targets_u8.astype(mx.float32) / 255.0
            colors = view_dependent_colors(params, viewmats, active_sh_degree)
            if args.photometric_correction:
                colors = apply_photometric(colors, photo_params, camera_indices)
            loss_fn_impl = pixel_loss_2dgs if args.mode == "2dgs" else pixel_loss_3d
            kwargs = {
                "bin_pad": bin_pad,
                "bin_capacity": bin_capacity,
                "return_counts": need_counts,
            }
            if args.mode == "2dgs":
                kwargs.update(
                    {
                        "normal_weight": normal_weight,
                        "distortion_weight": distortion_weight,
                        "normal_depth_mode": args.normal_depth_mode,
                        "return_components": True,
                    }
                )
            result = loss_fn_impl(
                params["means3d"],
                params["log_scales"],
                params["quats"],
                params["opacities_raw"],
                colors,
                targets,
                viewmats,
                Ks,
                ssim_weight=args.ssim_weight,
                means2d_offset=offset_zeros,
                means2d_absgrad_sink=absgrad_zeros,
                **kwargs,
            )
            if need_counts:
                if args.mode == "2dgs":
                    loss, _rendered, counts, components = result
                else:
                    loss, _rendered, counts = result
                    components = {
                        "photometric": loss,
                        "normal_consistency": mx.zeros((), dtype=loss.dtype),
                        "distortion": mx.zeros((), dtype=loss.dtype),
                    }
                if args.photometric_correction:
                    loss = loss + args.photometric_regularizer * photometric_identity_regularizer(photo_params)
                return loss, (counts, components)
            if args.mode == "2dgs":
                loss, _rendered, components = result
            else:
                loss, _rendered = result
                components = {
                    "photometric": loss,
                    "normal_consistency": mx.zeros((), dtype=loss.dtype),
                    "distortion": mx.zeros((), dtype=loss.dtype),
                }
            if args.photometric_correction:
                loss = loss + args.photometric_regularizer * photometric_identity_regularizer(photo_params)
            return loss, components

        loss_and_grad = mx.value_and_grad(loss_fn, argnums=[0, 4, 5, 6])
        state = [opt.state, photometric_opt.state]

        # Screen-space scaling of the shared absgrad sink. The sink sums over
        # B views while the loss is their mean, so multiplying by
        # B*(width/2, height/2) makes the signal approximately independent of
        # resolution and camera-batch size. Uses this segment's render dims so
        # the signal stays comparable across a coarse->fine resolution schedule.
        b = float(args.camera_batch)
        sig_scale = mx.array([width_e * 0.5 * b, height_e * 0.5 * b], dtype=mx.float32)

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_step(
            params,
            targets_u8,
            viewmats,
            Ks,
            offset_zeros,
            absgrad_zeros,
            grad_accum,
            sig_accum,
            vis_accum,
            util_ema,
            util_observations,
            util_age,
            util_consecutive_low,
            photo_params,
            camera_indices,
        ):
            result, (grads, _offset_grad, absgrad_grad, photo_grads) = loss_and_grad(
                params, targets_u8, viewmats, Ks, offset_zeros, absgrad_zeros, photo_params, camera_indices
            )
            if need_counts:
                loss, (counts, components) = result
            else:
                loss, components = result
            grad_accum = grad_accum + mx.sqrt(mx.sum(absgrad_grad * absgrad_grad, axis=1))
            if need_counts:
                sig_accum = sig_accum + mx.sqrt(mx.sum((absgrad_grad * sig_scale) ** 2, axis=1))
                active_view_count = mx.sum((counts > 0).astype(mx.float32), axis=0)
                vis_accum = vis_accum + active_view_count
            if args.utilization_telemetry:
                util = update_utilization(
                    {
                        "ema": util_ema,
                        "observations": util_observations,
                        "age": util_age,
                        "consecutive_low": util_consecutive_low,
                    },
                    grads["opacities_raw"],
                    params["opacities_raw"],
                    active_view_count,
                    args.camera_batch,
                    height_e,
                    width_e,
                    args.utilization_ema_decay,
                )
                util_ema, util_observations, util_age, util_consecutive_low = util.values()
            real_isects = (
                mx.sum(counts.astype(mx.int64)) if args.bin_check_overflow == "lazy" else mx.array(0, dtype=mx.int64)
            )
            if isinstance(opt, SelectiveAdam):
                tile_visible_mask = mx.any(counts > 0, axis=0)
                params = opt.apply_gradients(grads, params, tile_visible_mask)
            else:
                params = opt.apply_gradients(grads, params)
            if args.photometric_correction:
                photo_params = photometric_opt.apply_gradients(photo_grads, photo_params)
            return (
                loss,
                params,
                grad_accum,
                sig_accum,
                vis_accum,
                real_isects,
                util_ema,
                util_observations,
                util_age,
                util_consecutive_low,
                photo_params,
                components["photometric"],
                components["normal_consistency"],
                components["distortion"],
            )

        return compiled_step, state

    def segment_resolution(r):
        """Downsampled train targets + intrinsics for a segment downscale ``r``.
        Intrinsics are rebuilt from the resized dims (fx,cx by w'/w and fy,cy by
        h'/h) so the principal point stays consistent. ``r <= 1`` returns the
        full-res arrays unchanged (const-mode no-op)."""
        if r <= 1:
            return width, height, targets_u8, Ks_all
        w_e = max(1, int(round(width / r)))
        h_e = max(1, int(round(height / r)))
        tgt_np = np.asarray(targets_u8)  # (N, H, W, 3) uint8
        resized = np.stack([cv2.resize(v, (w_e, h_e), interpolation=cv2.INTER_AREA) for v in tgt_np], axis=0)
        targets_seg = mx.array(resized.astype(np.uint8))
        sx, sy = w_e / width, h_e / height
        Ks_np = np.asarray(Ks_all).copy()
        Ks_np[:, 0, :] *= sx
        Ks_np[:, 1, :] *= sy
        Ks_seg = mx.array(Ks_np.astype(np.float32))
        mx.eval(targets_seg, Ks_seg)
        return w_e, h_e, targets_seg, Ks_seg

    step_global = 0
    ts = time.perf_counter()
    n_views = len(train_indices)
    view_sampler = _ViewSampler(n_views, args.camera_batch, args.seed, args.view_sampling)
    segment_start = 0
    for segment_idx, segment_end in enumerate(segment_ends):
        segment_steps = segment_end - segment_start
        r = factor_for_step(segment_start)
        active_sh_degree = sh_degree_for_step(segment_start, args.sh_interval, args.sh_degree)
        width_e, height_e, targets_seg, Ks_seg = segment_resolution(r)
        normal_weight = (
            _regularizer_weight(
                args.normal_weight,
                args.normal_start_frac,
                segment_start,
                total_steps,
            )
            if args.mode == "2dgs"
            else 0.0
        )
        distortion_weight = (
            _regularizer_weight(
                args.distortion_weight,
                args.distortion_start_frac,
                segment_start,
                total_steps,
            )
            if args.mode == "2dgs"
            else 0.0
        )
        bin_pad, bin_capacity, bin_label = _choose_bins(
            params,
            viewmats_all,
            Ks_seg,
            width_e,
            height_e,
            str(args.bin_pad),
            args.bin_pad_min,
            args.bin_pad_margin,
            args.camera_batch,
            args.bin_capacity_stat,
            args.mode,
        )
        log.info(
            "segment %d/%d: steps=[%d,%d) N=%d res=%dx%d(r=%d) bins=%s camera_batch=%d densify_signal=%s "
            "sh=%d normal_w=%.3g distortion_w=%.3g sampling=%s overflow=%s",
            segment_idx,
            len(segment_ends),
            segment_start,
            segment_end,
            params["means3d"].shape[0],
            width_e,
            height_e,
            r,
            bin_label,
            args.camera_batch,
            args.densify_signal,
            active_sh_degree,
            normal_weight,
            distortion_weight,
            args.view_sampling,
            args.bin_check_overflow,
        )
        compiled_step, state = make_step(
            bin_pad, bin_capacity, normal_weight, distortion_weight, width_e, height_e, active_sh_degree
        )
        n = params["means3d"].shape[0]
        offset_zeros = mx.zeros((n, 2), dtype=mx.float32)
        absgrad_zeros = mx.zeros((n, 2), dtype=mx.float32)
        grad_accum = mx.zeros((n,), dtype=mx.float32)
        sig_accum = mx.zeros((n,), dtype=mx.float32)  # Stage 3a shadow signal
        vis_accum = mx.zeros((n,), dtype=mx.float32)  # Stage 3a visibility counts
        real_isects_epoch: list[int] = []
        overflow_events = 0

        for _ in range(segment_steps):
            sel = view_sampler.sample()
            idx = mx.array(sel.astype(np.int32))
            camera_indices = mx.take(mx.array(train_indices, dtype=mx.int32), idx, axis=0)
            batch_targets = mx.take(targets_seg, idx, axis=0)
            batch_viewmats = mx.take(viewmats_all, idx, axis=0)
            batch_Ks = mx.take(Ks_seg, idx, axis=0)
            if args.bin_check_overflow == "preflight" and bin_capacity is not None:
                real_isects_preflight = _count_batch_intersections(
                    params,
                    batch_viewmats,
                    batch_Ks,
                    width_e,
                    height_e,
                    args.mode,
                )
                real_isects_epoch.append(real_isects_preflight)
                if real_isects_preflight > bin_capacity:
                    overflow_events += 1
                    old_capacity = bin_capacity
                    bin_capacity = _capacity_for_count(
                        real_isects_preflight,
                        n,
                        len(sel),
                        width_e,
                        height_e,
                        args.bin_pad_min,
                        max(1.01, args.bin_overflow_margin),
                    )
                    log.warning(
                        "bin capacity overflow before step %d: real=%d > capacity=%d; recompiling with capacity=%d",
                        step_global,
                        real_isects_preflight,
                        old_capacity,
                        bin_capacity,
                    )
                    compiled_step, state = make_step(
                        bin_pad,
                        bin_capacity,
                        normal_weight,
                        distortion_weight,
                        width_e,
                        height_e,
                        active_sh_degree,
                    )
            (
                loss,
                params,
                grad_accum,
                sig_accum,
                vis_accum,
                real_isects_step,
                utilization["ema"],
                utilization["observations"],
                utilization["age"],
                utilization["consecutive_low"],
                photometric_params,
                loss_photometric,
                loss_normal,
                loss_distortion,
            ) = compiled_step(
                params,
                batch_targets,
                batch_viewmats,
                batch_Ks,
                offset_zeros,
                absgrad_zeros,
                grad_accum,
                sig_accum,
                vis_accum,
                utilization["ema"],
                utilization["observations"],
                utilization["age"],
                utilization["consecutive_low"],
                photometric_params,
                camera_indices,
            )
            mx.eval(
                loss,
                grad_accum,
                sig_accum,
                vis_accum,
                real_isects_step,
                *utilization.values(),
                *photometric_params.values(),
                loss_photometric,
                loss_normal,
                loss_distortion,
                *params.values(),
                *state,
            )
            component_values = np.array(
                [float(loss), float(loss_photometric), float(loss_normal), float(loss_distortion)]
            )
            if not np.isfinite(component_values).all():
                raise FloatingPointError(
                    f"non-finite loss at step {step_global}: total/rgb/normal/distortion={component_values.tolist()}"
                )
            if float(loss_normal) < -1e-6 or float(loss_distortion) < -1e-6 or float(loss) < -1e-6:
                raise FloatingPointError(
                    f"negative non-negative loss at step {step_global}: "
                    f"total/rgb/normal/distortion={component_values.tolist()}"
                )
            if args.bin_check_overflow == "lazy" and bin_capacity is not None:
                real_isects_lazy = int(real_isects_step)
                real_isects_epoch.append(real_isects_lazy)
                if real_isects_lazy > bin_capacity:
                    overflow_events += 1
                    old_capacity = bin_capacity
                    bin_capacity = _capacity_for_count(
                        real_isects_lazy,
                        n,
                        len(sel),
                        width_e,
                        height_e,
                        args.bin_pad_min,
                        max(1.01, args.bin_overflow_margin),
                    )
                    log.warning(
                        "bin capacity overflow after lazy step %d: real=%d > "
                        "capacity=%d; recompiling with capacity=%d",
                        step_global,
                        real_isects_lazy,
                        old_capacity,
                        bin_capacity,
                    )
                    compiled_step, state = make_step(
                        bin_pad,
                        bin_capacity,
                        normal_weight,
                        distortion_weight,
                        width_e,
                        height_e,
                        active_sh_degree,
                    )
            if step_global % args.log_every == 0:
                dt = (time.perf_counter() - ts) / max(1, args.log_every if step_global else 1)
                log.info(
                    "step %d/%d loss=%.5f rgb=%.5f normal=%.5f distortion=%.5f " "N=%d views=%s time/step=%.4f",
                    step_global,
                    total_steps,
                    float(loss),
                    float(loss_photometric),
                    float(loss_normal),
                    float(loss_distortion),
                    n,
                    sel.tolist(),
                    dt,
                )
                ts = time.perf_counter()
            if args.save_video and step_global % args.video_every == 0:
                # Render at this segment's resolution (the bin capacity is sized
                # for it), then upsample to full res so the video strip never
                # shape-mismatches the full-res target.
                video_K_seg = mx.take(Ks_seg, mx.array([video_pos]), axis=0)[0]
                preview = _render_view(
                    params,
                    video_viewmat,
                    video_K_seg,
                    width_e,
                    height_e,
                    args.mode,
                    bin_pad,
                    bin_capacity,
                    return_depth=args.save_depth,
                    active_sh_degree=active_sh_degree,
                )

                def _to_full(a):
                    a = np.array(a)
                    if a.shape[:2] != (height, width):
                        a = cv2.resize(a, (width, height), interpolation=cv2.INTER_NEAREST)
                    return a

                if args.save_depth:
                    preview_rgb, preview_aux = preview
                    mx.eval(preview_rgb, preview_aux["depth"])
                    frames.append(_to_full(preview_rgb))
                    depth_frames.append(_to_full(preview_aux["depth"][..., 0]))
                else:
                    mx.eval(preview)
                    frames.append(_to_full(preview))
            step_global += 1
            if eval_enabled and args.eval_every > 0 and step_global % args.eval_every == 0:
                assert val_targets_u8 is not None and val_viewmats is not None and val_Ks is not None
                _evaluate(
                    params,
                    val_targets_u8,
                    val_viewmats,
                    val_Ks,
                    width,
                    height,
                    args.mode,
                    args.eval_count_batch_size,
                    step_global,
                    log,
                    save_dir=(args.out_dir / "val") if args.eval_save_renders else None,
                    smoke_run=total_steps < 500,
                    active_sh_degree=sh_degree_for_step(step_global, args.sh_interval, args.sh_degree),
                    image_ids=[scene.image_paths[i].name for i in val_indices],
                    metrics_out_dir=args.out_dir,
                    metric_suite=metric_suite,
                    final=step_global == total_steps,
                    photometric_params=photometric_params if args.photometric_correction else None,
                    photometric_camera_indices=val_indices,
                    enable_lpips=args.eval_lpips,
                )
                ts = time.perf_counter()  # don't attribute eval time to the next train steps

        if real_isects_epoch:
            real_np = np.asarray(real_isects_epoch)
            log.info(
                "overflow telemetry segment %d [%d,%d): mode=%s events=%d/%d "
                "real intersections p50/p95/max=%.0f/%.0f/%d",
                segment_idx,
                segment_start,
                segment_end,
                args.bin_check_overflow,
                overflow_events,
                len(real_isects_epoch),
                float(np.percentile(real_np, 50)),
                float(np.percentile(real_np, 95)),
                int(real_np.max()),
            )

        split_idx = split_index_by_step.get(segment_end)
        at_split_boundary = split_idx is not None

        if at_split_boundary:
            assert split_idx is not None
            old_opt = opt
            normalized_signal = sig_accum / mx.maximum(vis_accum, 1.0)
            utilization_prune_mask = None
            if args.utilization_telemetry:
                utilization_prune_mask, utilization = pruning_window(
                    utilization,
                    args.utilization_threshold,
                    args.utilization_warmup,
                    args.utilization_min_observations,
                    args.utilization_grace,
                    args.utilization_low_windows,
                )
                correlations = telemetry_correlations(utilization, params, vis_accum)
                log.info(
                    "utilization telemetry: ema p10/p50/p90=%.3g/%.3g/%.3g "
                    "corr(opacity/scale/tile)=%.3f/%.3f/%.3f low_candidates=%d",
                    *np.percentile(np.asarray(utilization["ema"]), [10, 50, 90]),
                    correlations["opacity"],
                    correlations["max_scale"],
                    correlations["tile_activity"],
                    int(utilization_prune_mask.sum()),
                )
                if not args.utilization_pruning:
                    utilization_prune_mask = None
            if args.shadow_signal:
                _log_shadow_densification(
                    log,
                    split_idx,
                    params,
                    grad_accum / float(segment_steps),
                    normalized_signal,
                    vis_accum,
                    args,
                    float(scene.scene_scale),
                    grid=(1e-5, 2e-5, 4e-5, 8e-5, 1e-4, 2e-4, 4e-4, 8e-4),
                )
            refine_signal = (
                normalized_signal if args.densify_signal == "normalized" else grad_accum / float(segment_steps)
            )
            # Budget mode: the new gaussians train at the *next* segment's
            # resolution, so the count target uses that downscale and the
            # boundary global step (DashGaussian Eq. 4).
            if args.densify_mode == "budget":
                target_count = budget.target_count(factor_for_step(segment_end), segment_end, total_steps)
            else:
                target_count = None
            params, refine_info = split_n_prune_3d(
                params,
                refine_signal,
                mx.random.key(args.seed + split_idx),
                grad_thr=args.grad_thr,
                grow_scale=args.grow_scale,
                scene_scale=float(scene.scene_scale),
                prune_opa=args.prune_opa,
                prune_scale3d=args.prune_scale3d,
                densify_mode=args.densify_mode,
                target_count=target_count,
                grad_percentile=args.budget_grad_percentile,
                max_densify_rate=args.max_densify_rate_per_step,
                extra_prune_mask=utilization_prune_mask,
            )
            if args.utilization_telemetry:
                utilization = remap_utilization(utilization, refine_info["idx_keep"], refine_info["idx_new_parent"])
            if args.densify_mode == "budget":
                budget.update(refine_info["n_densified"])
                log.info(
                    "budget: target=%s densified k=%d P_fin=%d",
                    target_count,
                    refine_info["n_densified"],
                    budget.p_fin,
                )
            log.info(
                "refine: %d duplicated, %d split, %d pruned (opa=%d scale=%d utilization=%d) -> %d",
                refine_info["n_dupli"],
                refine_info["n_split"],
                refine_info["n_prune"],
                refine_info.get("n_prune_opa", 0),
                refine_info.get("n_prune_scale3d", 0),
                refine_info.get("n_prune_utilization", 0),
                params["means3d"].shape[0],
            )
            reset_opacity = args.reset_opacity_every > 0 and split_idx % args.reset_opacity_every == 0
            if reset_opacity:
                params = reset_opacities_3d(params, args.prune_opa)
                log.info(
                    "reset opacities at split boundary %d (step %d): max opacity=%.3g",
                    split_idx,
                    segment_end,
                    min(2.0 * args.prune_opa, 1.0 - 1e-6),
                )
            next_segment_end = segment_ends[segment_idx + 1]
            opt = _build_optimizer(
                params,
                args,
                total_steps,
                scene.scene_scale,
                segment_start=segment_end,
                restart_period=next_segment_end - segment_end,
                old_opt=old_opt,
                decay_from_step=lr_decay_from_step,
            )
            if args.carry_optimizer_state or args.selective_adam:
                carry_optimizer_state_3d(
                    old_opt,
                    opt,
                    params,
                    refine_info["idx_keep"],
                    refine_info["num_new"],
                )
            if reset_opacity:
                zero_param_moments(opt, "opacities_raw")
        segment_start = segment_end

    # Emit a final metrics file when the regular cadence did not land exactly
    # on the last step. The in-loop evaluation marks that checkpoint final
    # otherwise, avoiding a duplicate render/LPIPS pass.
    if eval_enabled and total_steps % args.eval_every != 0:
        assert val_targets_u8 is not None and val_viewmats is not None and val_Ks is not None
        _evaluate(
            params,
            val_targets_u8,
            val_viewmats,
            val_Ks,
            width,
            height,
            args.mode,
            args.eval_count_batch_size,
            total_steps,
            log,
            save_dir=(args.out_dir / "val") if args.eval_save_renders else None,
            smoke_run=total_steps < 500,
            active_sh_degree=sh_degree_for_step(total_steps, args.sh_interval, args.sh_degree),
            image_ids=[scene.image_paths[i].name for i in val_indices],
            metrics_out_dir=args.out_dir,
            metric_suite=metric_suite,
            final=True,
            photometric_params=photometric_params if args.photometric_correction else None,
            photometric_camera_indices=val_indices,
            enable_lpips=args.eval_lpips,
        )

    if args.save_dtu_renders:
        if args.mode != "2dgs":
            raise ValueError("train.save_dtu_renders requires gaussians.mode=2dgs")
        dtu_viewmats, dtu_Ks, dtu_width, dtu_height = _camera_matrices_at_resolution(
            scene, train_indices, args.dtu_render_max_side
        )
        _export_dtu_renders(
            params,
            [scene.image_paths[i] for i in train_indices],
            dtu_viewmats,
            dtu_Ks,
            dtu_width,
            dtu_height,
            sh_degree_for_step(total_steps, args.sh_interval, args.sh_degree),
            args.eval_count_batch_size,
            args.out_dir / "dtu_renders",
            log,
        )

    ply_path = export_ply_3d(params, args.out_dir / "final.ply")
    log.info("saved %s", ply_path)

    if args.save_video and frames:
        target_np = np.array(targets_u8[video_pos].astype(mx.float32) / 255.0)
        out_path = args.out_dir / "train_preview.avi"
        if args.save_depth:
            valid_depth = [d[np.isfinite(d) & (d > 0.0)] for d in depth_frames]
            valid_depth = [d for d in valid_depth if d.size > 0]
            if valid_depth:
                # Robust 2nd/98th percentiles over the pooled valid depths: a
                # single far outlier (early-training garbage or a stray splat)
                # otherwise blows up the max and squashes the whole scene to
                # near-black. Percentiles keep the panel readable and stable.
                pooled = np.concatenate(valid_depth)
                depth_min = float(np.percentile(pooled, 2))
                depth_max = float(np.percentile(pooled, 98))
            else:
                depth_min, depth_max = 0.0, 1.0
            frame_width = width * 3
        else:
            depth_min = depth_max = 0.0
            frame_width = width * 2
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter.fourcc("M", "J", "P", "G"),
            24,
            (frame_width, height),
        )
        for i, frame in enumerate(frames):
            g = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            t = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
            if args.save_depth:
                d = _depth_panel(depth_frames[i], depth_min, depth_max)
                canvas = np.hstack([d, g, t])
            else:
                canvas = np.hstack([g, t])
            writer.write(canvas[:, :, ::-1])
        writer.release()
        log.info(
            "saved %s (preview view = train image %d%s)",
            out_path,
            train_indices[video_pos],
            ", panels=depth|rgb|target" if args.save_depth else ", panels=rgb|target",
        )

    run_summary = {
        "end_to_end_wall_seconds": time.perf_counter() - run_started_at,
        "peak_mlx_allocator_bytes": int(mx.get_peak_memory()),
        "active_mlx_allocator_bytes": int(mx.get_active_memory()),
        "cached_mlx_allocator_bytes": int(mx.get_cache_memory()),
        "final_gaussian_count": int(params["means3d"].shape[0]),
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n")
    log.info(
        "run summary: wall=%.2fs peak MLX allocation=%.1f MiB final N=%d",
        run_summary["end_to_end_wall_seconds"],
        run_summary["peak_mlx_allocator_bytes"] / 2**20,
        run_summary["final_gaussian_count"],
    )


if __name__ == "__main__":
    train_colmap3d()  # type: ignore[call-arg]
