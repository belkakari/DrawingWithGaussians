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

import logging
import math
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# pycolmap may load a second OpenMP runtime on macOS. Set before importing it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import hydra  # type: ignore[import-not-found]
import mlx.core as mx
import numpy as np
from omegaconf import DictConfig, OmegaConf  # type: ignore[import-not-found]
from PIL import Image

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
from drawingwithgaussians.rendering2dgs import project_gaussians_2dgs  # type: ignore[import-not-found]
from drawingwithgaussians.rendering2dgs_fused import (  # type: ignore[import-not-found]
    _count_bbox_intersections,
    rasterize2dgs_fused,
)
from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE, project_gaussians
from drawingwithgaussians.rendering3d_fused import _count_tile_intersections, _num_tiles, rasterize3d_fused
from drawingwithgaussians.splat_export import export_ply_3d


@dataclass
class ColmapScene:
    image_paths: list[Path]
    camtoworlds: np.ndarray
    Ks: list[np.ndarray]
    points: np.ndarray
    points_rgb: np.ndarray
    scene_scale: float


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
        ColmapScene(image_paths, camtoworlds, Ks, points, points_rgb, scene_scale),
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
    synchronous CPU work per step (x batch with camera batching), stalling
    GPU submission. (Memory is unified on Apple Silicon, so the copy itself
    is a cheap same-DRAM memcpy; the decode/resize/normalize is the cost.)
    Images are stored stacked as uint8 (~a quarter of the fp32 footprint;
    e.g. ~90 MB for 150 views at 512x384) and normalized to float on the
    GPU stream inside the compiled step. All views are resized to the
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
        "colors_raw": mx.array(_logit(rgb)),
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
    """Depth preview panel in RGB, with invalid pixels black."""
    valid = np.isfinite(depth) & (depth > 0.0)
    panel = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return panel
    denom = max(depth_max - depth_min, 1e-6)
    scaled = np.clip((depth - depth_min) / denom, 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    panel[valid] = gray[valid, None]
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
):
    """Render one view with the current params (no absgrad, lazy result).

    Shared by the video preview (train bins, cosmetic) and eval (compact-exact
    bins sized by :func:`eval_capacity`).
    """
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
            mx.sigmoid(params["colors_raw"]),
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
        mx.sigmoid(params["colors_raw"]),
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
    for v in range(n_views):
        t0 = time.perf_counter()
        rendered = _render_view(params, viewmats[v], Ks[v], width, height, splat_mode, None, capacity)
        target = targets_u8[v].astype(mx.float32) / 255.0
        mse = mx.mean(mx.square(rendered - target))
        ssim_v = _ssim(rendered, target)
        mx.eval(rendered, mse, ssim_v)
        view_times.append(time.perf_counter() - t0)
        psnrs.append(10.0 * math.log10(1.0 / max(float(mse), 1e-12)))
        ssims.append(float(ssim_v))
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            img = (np.clip(np.array(rendered), 0.0, 1.0) * 255).astype(np.uint8)
            cv2.imwrite(str(save_dir / f"val_{v:04d}_step{eval_step:06d}.png"), img[:, :, ::-1])

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
    return float(np.mean(psnrs)), float(np.mean(ssims)), wall


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


def _build_optimizer(params, args, total_steps, scene_scale, segment_start, restart_period, old_opt):
    """Set up the optimizer for one split segment, applying the Stage 2b LR ladder knobs.

    L2 (``means_lr_scene_scale``): means LR scaled by ``scene_scale`` (gsplat).
    L3 (``means_only_schedule``): ``means_mode`` schedule restricted to means.
    L4a (``global_schedule``): the LR schedule spans the whole run instead of
    restarting each split segment. Off by default so SGDR warm-restarts survive
    (the load-bearing behavior from EXPERIMENTS Exp 8). The step offset follows the
    pinned carry rule: ``carried_step = old_step if carrying else 0``.
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


def _split_steps(total_steps, split_start_iter, split_end_iter, split_every):
    """Completed-step counts where split/prune fires.

    Splits happen after the optimizer has completed that many steps, so valid
    events satisfy ``0 < step < total_steps``.
    """
    total_steps = int(total_steps)
    split_every = int(split_every)
    if total_steps <= 1 or split_every <= 0:
        return []
    step = int(split_start_iter)
    if step <= 0:
        step = split_every
    end = min(int(split_end_iter), total_steps - 1)
    if step > end:
        return []
    events = []
    while step <= end:
        events.append(step)
        step += split_every
    return events


@hydra.main(version_base=None, config_path="./configs")
def train_colmap3d(cfg: DictConfig):
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
        split_start_iter=int(cfg.gaussians.get("split_start_iter", 500)),
        split_end_iter=_none_or_int(cfg.gaussians.get("split_end_iter", None)),
        split_every=int(cfg.gaussians.get("split_every", 500)),
        shadow_signal=bool(cfg.gaussians.get("shadow_signal", False)),
        grow_scale=float(cfg.gaussians.grow_scale),
        prune_opa=float(cfg.gaussians.prune_opa),
        prune_scale3d=_none_or_float(cfg.gaussians.get("prune_scale3d", None)),
        reset_opacity_every=int(cfg.gaussians.get("reset_opacity_every", 0)),
        carry_optimizer_state=bool(cfg.gaussians.get("carry_optimizer_state", False)),
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
        save_video=bool(cfg.train.get("save_video", False)),
        save_depth=bool(cfg.train.get("save_depth", False)),
        video_every=int(cfg.train.get("video_every", 50)),
        video_index=int(cfg.train.get("video_index", 0)),
    )
    if args.densify_signal not in {"legacy", "normalized"}:
        raise ValueError("gaussians.densify_signal must be 'legacy' or 'normalized'")
    if args.view_sampling not in {"random", "shuffle"}:
        raise ValueError("train.view_sampling must be 'random' or 'shuffle'")
    if args.steps <= 0:
        raise ValueError("optim.num_steps must be positive")
    if args.split_end_iter is None:
        args.split_end_iter = max(0, args.steps - max(1, args.split_every))
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
    split_steps = _split_steps(total_steps, args.split_start_iter, args.split_end_iter, args.split_every)
    segment_ends = split_steps + [total_steps]
    opt = _build_optimizer(
        params,
        args,
        total_steps,
        scene.scene_scale,
        segment_start=0,
        restart_period=segment_ends[0],
        old_opt=None,
    )
    mx.eval(*params.values())

    frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []

    def make_step(bin_pad, bin_capacity, normal_weight, distortion_weight) -> tuple[Any, list[Any]]:
        need_counts = args.densify_signal == "normalized" or args.shadow_signal or args.bin_check_overflow == "lazy"

        def loss_fn(params, targets_u8, viewmats, Ks, offset_zeros, absgrad_zeros):
            # One batched render for the whole camera batch (gsplat's
            # [..., C, N] convention): batched projection broadcasts the
            # camera entries, the rasterizer launches once with grid z = B,
            # and the shared offset/absgrad sinks sum their cotangents over
            # views. Targets arrive uint8 and are normalized on the GPU
            # stream. Loss is the mean over all views.
            targets = targets_u8.astype(mx.float32) / 255.0
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
                    }
                )
            result = loss_fn_impl(
                params["means3d"],
                params["log_scales"],
                params["quats"],
                params["opacities_raw"],
                params["colors_raw"],
                targets,
                viewmats,
                Ks,
                ssim_weight=args.ssim_weight,
                means2d_offset=offset_zeros,
                means2d_absgrad_sink=absgrad_zeros,
                **kwargs,
            )
            if need_counts:
                loss, _rendered, counts = result
                return loss, counts
            loss, _rendered = result
            return loss

        loss_and_grad = mx.value_and_grad(loss_fn, argnums=[0, 4, 5])
        state = [opt.state]

        # Screen-space scaling of the shared absgrad sink. The sink sums over
        # B views while the loss is their mean, so multiplying by
        # B*(width/2, height/2) makes the signal approximately independent of
        # resolution and camera-batch size.
        b = float(args.camera_batch)
        sig_scale = mx.array([width * 0.5 * b, height * 0.5 * b], dtype=mx.float32)

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_step(
            params, targets_u8, viewmats, Ks, offset_zeros, absgrad_zeros, grad_accum, sig_accum, vis_accum
        ):
            result, (grads, _offset_grad, absgrad_grad) = loss_and_grad(
                params, targets_u8, viewmats, Ks, offset_zeros, absgrad_zeros
            )
            if need_counts:
                loss, counts = result
            else:
                loss = result
            grad_accum = grad_accum + mx.sqrt(mx.sum(absgrad_grad * absgrad_grad, axis=1))
            if need_counts:
                sig_accum = sig_accum + mx.sqrt(mx.sum((absgrad_grad * sig_scale) ** 2, axis=1))
                vis_accum = vis_accum + mx.sum((counts > 0).astype(mx.float32), axis=0)
            real_isects = (
                mx.sum(counts.astype(mx.int64)) if args.bin_check_overflow == "lazy" else mx.array(0, dtype=mx.int64)
            )
            params = opt.apply_gradients(grads, params)
            return loss, params, grad_accum, sig_accum, vis_accum, real_isects

        return compiled_step, state

    step_global = 0
    ts = time.perf_counter()
    n_views = len(train_indices)
    view_sampler = _ViewSampler(n_views, args.camera_batch, args.seed, args.view_sampling)
    segment_start = 0
    for segment_idx, segment_end in enumerate(segment_ends):
        segment_steps = segment_end - segment_start
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
            Ks_all,
            width,
            height,
            str(args.bin_pad),
            args.bin_pad_min,
            args.bin_pad_margin,
            args.camera_batch,
            args.bin_capacity_stat,
            args.mode,
        )
        log.info(
            "segment %d/%d: steps=[%d,%d) N=%d bins=%s camera_batch=%d densify_signal=%s "
            "normal_w=%.3g distortion_w=%.3g sampling=%s overflow=%s",
            segment_idx,
            len(segment_ends),
            segment_start,
            segment_end,
            params["means3d"].shape[0],
            bin_label,
            args.camera_batch,
            args.densify_signal,
            normal_weight,
            distortion_weight,
            args.view_sampling,
            args.bin_check_overflow,
        )
        compiled_step, state = make_step(bin_pad, bin_capacity, normal_weight, distortion_weight)
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
            batch_targets = mx.take(targets_u8, idx, axis=0)
            batch_viewmats = mx.take(viewmats_all, idx, axis=0)
            batch_Ks = mx.take(Ks_all, idx, axis=0)
            if args.bin_check_overflow == "preflight" and bin_capacity is not None:
                real_isects_preflight = _count_batch_intersections(
                    params,
                    batch_viewmats,
                    batch_Ks,
                    width,
                    height,
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
                        width,
                        height,
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
                    )
            (
                loss,
                params,
                grad_accum,
                sig_accum,
                vis_accum,
                real_isects_step,
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
            )
            mx.eval(
                loss,
                grad_accum,
                sig_accum,
                vis_accum,
                real_isects_step,
                *params.values(),
                *state,
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
                        width,
                        height,
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
                    )
            if step_global % args.log_every == 0:
                dt = (time.perf_counter() - ts) / max(1, args.log_every if step_global else 1)
                log.info(
                    "step %d/%d loss=%.5f N=%d views=%s time/step=%.4f",
                    step_global,
                    total_steps,
                    float(loss),
                    n,
                    sel.tolist(),
                    dt,
                )
                ts = time.perf_counter()
            if args.save_video and step_global % args.video_every == 0:
                preview = _render_view(
                    params,
                    video_viewmat,
                    video_K,
                    width,
                    height,
                    args.mode,
                    bin_pad,
                    bin_capacity,
                    return_depth=args.save_depth,
                )
                if args.save_depth:
                    preview_rgb, preview_aux = preview
                    mx.eval(preview_rgb, preview_aux["depth"])
                    frames.append(np.array(preview_rgb))
                    depth_frames.append(np.array(preview_aux["depth"][..., 0]))
                else:
                    mx.eval(preview)
                    frames.append(np.array(preview))
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

        at_split_boundary = segment_end in split_steps
        split_idx = segment_idx + 1

        if at_split_boundary:
            old_opt = opt
            normalized_signal = sig_accum / mx.maximum(vis_accum, 1.0)
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
            params, refine_info = split_n_prune_3d(
                params,
                refine_signal,
                mx.random.key(args.seed + split_idx),
                grad_thr=args.grad_thr,
                grow_scale=args.grow_scale,
                scene_scale=float(scene.scene_scale),
                prune_opa=args.prune_opa,
                prune_scale3d=args.prune_scale3d,
            )
            log.info(
                "refine: %d duplicated, %d split, %d pruned (opa=%d scale=%d) -> %d",
                refine_info["n_dupli"],
                refine_info["n_split"],
                refine_info["n_prune"],
                refine_info.get("n_prune_opa", 0),
                refine_info.get("n_prune_scale3d", 0),
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
            )
            if args.carry_optimizer_state:
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

    if eval_enabled and (args.eval_every <= 0 or total_steps % args.eval_every != 0):
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
                depth_min = float(min(d.min(initial=np.inf) for d in valid_depth))
                depth_max = float(max(d.max(initial=0.0) for d in valid_depth))
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


if __name__ == "__main__":
    train_colmap3d()  # type: ignore[call-arg]
