"""Train 3D Gaussians on a COLMAP/Mip-NeRF 360 scene with MLX.

Lightweight trainer for datasets such as
``/Users/glebsterkin/Downloads/360_extra_scenes/{flowers,treehill}``.
It reads COLMAP cameras/images/points with pycolmap, trains with this repo's
MLX 3D rasterizer/loss/densification, and exports a standard PLY.

Example smoke run:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python train_colmap3d.py \
      --data-dir /Users/glebsterkin/Downloads/360_extra_scenes/flowers \
      --data-factor 8 --max-init-points 1000 --steps 100 --max-side 256
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

# pycolmap may load a second OpenMP runtime on macOS. Set before importing it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import mlx.core as mx
import numpy as np
from PIL import Image

from drawingwithgaussians.gaussian3d import (
    carry_optimizer_state_3d,
    set_up_optimizer_3d,
    split_n_prune_3d,
)
from drawingwithgaussians.losses import pixel_loss_3d
from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE, project_gaussians
from drawingwithgaussians.rendering3d_fused import (
    _TILE,
    _bounding_radii,
    rasterize3d_fused,
)
from drawingwithgaussians.splat_export import export_ply_3d

DEFAULT_DATA_DIR = Path("/Users/glebsterkin/Downloads/360_extra_scenes/flowers")


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
    points = np.array(
        [points3d[pid].xyz for pid in point_ids], dtype=np.float32
    ).reshape(-1, 3)
    points_rgb = np.array(
        [points3d[pid].color for pid in point_ids], dtype=np.float32
    ).reshape(-1, 3)

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


def _sample_init_points(
    scene: ColmapScene,
    max_points: int,
    seed: int,
    init_opacity: float,
    init_scale: float,
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
    scales = np.full((n, 3), init_scale, dtype=np.float32)
    opac = np.full((n,), init_opacity, dtype=np.float32)
    return {
        "means3d": mx.array(points),
        "log_scales": mx.array(np.log(scales).astype(np.float32)),
        "quats": mx.array(quats),
        "opacities_raw": mx.array(_logit(opac)),
        "colors_raw": mx.array(_logit(rgb)),
    }


def _choose_bin_pad(
    params, viewmat, K, width, height, mode: str, min_pad: int, margin: float
):
    if mode.lower() in {"none", "exact"}:
        return None
    if mode.lower() != "auto":
        return int(mode)
    means2d, conics, depths = project_gaussians(
        params["means3d"],
        params["log_scales"],
        params["quats"],
        viewmat,
        K,
        width,
        height,
    )
    opacities = mx.where(
        (depths > NEAR_PLANE) & (depths < FAR_PLANE),
        mx.sigmoid(params["opacities_raw"]),
        0.0,
    )
    radii = _bounding_radii(conics, opacities)
    tw = (width + _TILE - 1) // _TILE
    th = (height + _TILE - 1) // _TILE
    ntiles = tw * th
    mxs, mys = means2d[:, 0], means2d[:, 1]
    rx, ry = radii[:, 0], radii[:, 1]
    valid = rx > 0
    tx0 = mx.clip(mx.floor((mxs - rx) / _TILE), 0, tw - 1).astype(mx.int32)
    tx1 = mx.clip(mx.floor((mxs + rx) / _TILE), 0, tw - 1).astype(mx.int32)
    ty0 = mx.clip(mx.floor((mys - ry) / _TILE), 0, th - 1).astype(mx.int32)
    ty1 = mx.clip(mx.floor((mys + ry) / _TILE), 0, th - 1).astype(mx.int32)
    area = mx.where(valid, (tx1 - tx0 + 1) * (ty1 - ty0 + 1), 0)
    mx.eval(area)
    max_area = int(mx.max(area)) if area.size > 0 else 0
    return min(ntiles, max(1, max(min_pad, math.ceil(max_area * margin))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/colmap3d"))
    p.add_argument("--data-factor", type=int, default=8)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--test-every", type=int, default=8)
    p.add_argument("--max-init-points", type=int, default=20_000)
    p.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=2_000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.6e-4)
    p.add_argument("--ssim-weight", type=float, default=0.2)
    p.add_argument("--init-opacity", type=float, default=0.1)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--grad-thr", type=float, default=1e-5)
    p.add_argument("--grow-scale", type=float, default=0.05)
    p.add_argument("--prune-opa", type=float, default=0.005)
    p.add_argument("--bin-pad", default="auto", help="auto | exact | integer")
    p.add_argument("--bin-pad-min", type=int, default=16)
    p.add_argument("--bin-pad-margin", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-video", action="store_true")
    p.add_argument(
        "--video-every",
        type=int,
        default=50,
        help="Render preview frame every N steps from a fixed view",
    )
    p.add_argument(
        "--video-index",
        type=int,
        default=0,
        help="Train-split image index used as the fixed preview viewpoint",
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s"
    )
    log = logging.getLogger("train_colmap3d")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scene, train_indices, _ = _load_colmap_scene(
        args.data_dir, args.data_factor, args.normalize, args.test_every
    )
    log.info(
        "loaded %s: %d train images, %d sfm points, scene_scale=%.3f",
        args.data_dir,
        len(train_indices),
        len(scene.points),
        scene.scene_scale,
    )
    if not train_indices:
        raise ValueError("train split is empty")

    video_index = train_indices[max(0, min(len(train_indices) - 1, args.video_index))]
    video_target, video_viewmat, video_K = _load_item(scene, video_index, args.max_side)
    first_target, first_viewmat, first_K = (
        (video_target, video_viewmat, video_K)
        if args.save_video
        else _load_item(scene, train_indices[0], args.max_side)
    )
    height, width = first_target.shape[:2]
    params = _sample_init_points(
        scene, args.max_init_points, args.seed, args.init_opacity, args.init_scale
    )
    opt = set_up_optimizer_3d(
        params, lr=args.lr, max_steps=args.steps * args.epochs, mode="const"
    )
    mx.eval(*params.values(), first_target, first_viewmat, first_K)

    rng = np.random.default_rng(args.seed)
    frames = []
    total_steps = args.steps * args.epochs

    def make_step(bin_pad) -> tuple[Any, list[Any]]:
        def loss_fn(params, target_image, viewmat, K, offset_zeros, absgrad_zeros):
            return pixel_loss_3d(
                params["means3d"],
                params["log_scales"],
                params["quats"],
                params["opacities_raw"],
                params["colors_raw"],
                target_image,
                viewmat,
                K,
                ssim_weight=args.ssim_weight,
                means2d_offset=offset_zeros,
                means2d_absgrad_sink=absgrad_zeros,
                bin_pad=bin_pad,
            )

        loss_and_grad = mx.value_and_grad(loss_fn, argnums=[0, 4, 5])
        state = [opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_step(
            params, target_image, viewmat, K, offset_zeros, absgrad_zeros, grad_accum
        ):
            (loss, rendered), (grads, _offset_grad, absgrad_grad) = loss_and_grad(
                params, target_image, viewmat, K, offset_zeros, absgrad_zeros
            )
            grad_accum = grad_accum + mx.sqrt(
                mx.sum(absgrad_grad * absgrad_grad, axis=1)
            )
            params = opt.apply_gradients(grads, params)
            return loss, rendered, params, grad_accum

        return compiled_step, state

    step_global = 0
    ts = time.perf_counter()
    for epoch in range(args.epochs):
        bin_pad = _choose_bin_pad(
            params,
            first_viewmat,
            first_K,
            width,
            height,
            str(args.bin_pad),
            args.bin_pad_min,
            args.bin_pad_margin,
        )
        log.info(
            "epoch %d/%d: N=%d bin_pad=%s",
            epoch,
            args.epochs,
            params["means3d"].shape[0],
            bin_pad or "exact",
        )
        compiled_step, state = make_step(bin_pad)
        n = params["means3d"].shape[0]
        offset_zeros = mx.zeros((n, 2), dtype=mx.float32)
        absgrad_zeros = mx.zeros((n, 2), dtype=mx.float32)
        grad_accum = mx.zeros((n,), dtype=mx.float32)

        for _ in range(args.steps):
            item_idx = int(train_indices[int(rng.integers(0, len(train_indices)))])
            target, viewmat, K = _load_item(scene, item_idx, args.max_side)
            loss, rendered, params, grad_accum = compiled_step(
                params, target, viewmat, K, offset_zeros, absgrad_zeros, grad_accum
            )
            mx.eval(loss, rendered, grad_accum, *params.values(), *state)
            if step_global % args.log_every == 0:
                dt = (time.perf_counter() - ts) / max(
                    1, args.log_every if step_global else 1
                )
                log.info(
                    "step %d/%d loss=%.5f N=%d image=%d time/step=%.4f",
                    step_global,
                    total_steps,
                    float(loss),
                    n,
                    item_idx,
                    dt,
                )
                ts = time.perf_counter()
            if args.save_video and step_global % args.video_every == 0:
                means2d_v, conics_v, depths_v = project_gaussians(
                    params["means3d"],
                    params["log_scales"],
                    params["quats"],
                    video_viewmat,
                    video_K,
                    width,
                    height,
                )
                preview = rasterize3d_fused(
                    mx.take(means2d_v, mx.argsort(depths_v), axis=0),
                    mx.take(conics_v, mx.argsort(depths_v), axis=0),
                    mx.take(
                        mx.sigmoid(params["opacities_raw"]),
                        mx.argsort(depths_v),
                        axis=0,
                    ),
                    mx.take(
                        mx.sigmoid(params["colors_raw"]), mx.argsort(depths_v), axis=0
                    ),
                    mx.zeros((3,), dtype=mx.float32),
                    mx.take(depths_v, mx.argsort(depths_v), axis=0),
                    height,
                    width,
                    absgrad_sink=None,
                    bin_pad=bin_pad,
                )
                mx.eval(preview)
                frames.append(np.array(preview))
            step_global += 1

        if epoch != args.epochs - 1:
            old_opt = opt
            params, refine_info = split_n_prune_3d(
                params,
                grad_accum / float(args.steps),
                mx.random.key(args.seed + epoch + 1),
                grad_thr=args.grad_thr,
                grow_scale=args.grow_scale,
                scene_scale=float(scene.scene_scale),
                prune_opa=args.prune_opa,
            )
            log.info(
                "refine: %d duplicated, %d split, %d pruned -> %d",
                refine_info["n_dupli"],
                refine_info["n_split"],
                refine_info["n_prune"],
                params["means3d"].shape[0],
            )
            opt = set_up_optimizer_3d(
                params, lr=args.lr, max_steps=total_steps, mode="const"
            )
            carry_optimizer_state_3d(
                old_opt, opt, params, refine_info["idx_keep"], refine_info["num_new"]
            )

    ply_path = export_ply_3d(params, args.out_dir / "final.ply")
    log.info("saved %s", ply_path)

    if args.save_video and frames:
        target_np = np.array(first_target)
        out_path = args.out_dir / "train_preview.avi"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter.fourcc("M", "J", "P", "G"),
            24,
            (width * 2, height),
        )
        for frame in frames:
            g = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            t = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
            writer.write(np.hstack([g, t])[:, :, ::-1])
        writer.release()
        log.info("saved %s (preview view = train image %d)", out_path, video_index)


if __name__ == "__main__":
    main()
