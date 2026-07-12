"""Decompose the 2DGS training step to choose the next speedup.

This mirrors ``scripts/profile_3d_step.py`` but exercises the surfel/2DGS
projection and rasterizer. It times projection, compact-bin construction,
projected rasterization, projected raster backward, and full pixel losses. The
full fwd+bwd timings include the same means2d offset and absgrad sink used by
``train_colmap3d.py`` densification.

Usage:
    uv run python scripts/profile_2dgs_step.py --n 20000 --width 512 --height 338 --iters 30 --ssim-weight 0.2
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawingwithgaussians.losses import pixel_loss_2dgs  # noqa: E402
from drawingwithgaussians.rendering2dgs import project_gaussians_2dgs  # noqa: E402
from drawingwithgaussians.rendering2dgs_fused import (  # noqa: E402
    _build_bins,
    _count_bbox_intersections,
    estimate_bin_capacity_2dgs,
    rasterize2dgs_fused,
)
from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE  # noqa: E402


def timeit(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def scene_2dgs(n, width, height, rng, spread=False):
    if spread:
        # Fill the 90-degree horizontal frustum. With fx == fy, the vertical
        # extent scales by height / width for a rectangular image.
        z = rng.uniform(7.0, 9.0, size=(n, 1))
        x = rng.uniform(-0.9, 0.9, size=(n, 1)) * z
        y = rng.uniform(-0.9 * height / width, 0.9 * height / width, size=(n, 1)) * z
        means3d = mx.array(np.concatenate([x, y, z - 8.0], axis=1).astype(np.float32))
    else:
        means3d = mx.array((2.0 * (rng.random((n, 3)) - 0.5)).astype(np.float32))
    log_scales = mx.array(np.log(rng.uniform(0.02, 0.3, size=(n, 3))).astype(np.float32))
    quats = mx.array(rng.normal(size=(n, 4)).astype(np.float32))
    opac_raw = mx.array((rng.normal(size=(n,)) - 1.0).astype(np.float32))
    col_raw = mx.array(rng.normal(size=(n, 3)).astype(np.float32))
    target = mx.array(rng.random((height, width, 3)).astype(np.float32))
    focal = 0.5 * width / math.tan(0.25 * math.pi)
    K = mx.array(
        np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            dtype=np.float32,
        )
    )
    viewmat = mx.array(
        np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 8.0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
    )
    params = (means3d, log_scales, quats, opac_raw, col_raw)
    mx.eval(*params, target, K, viewmat)
    return params, target, K, viewmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument(
        "--size",
        type=int,
        default=None,
        help="square image size; overridden by --width/--height",
    )
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--capacity-margin", type=float, default=2.0)
    ap.add_argument("--min-per-gaussian", type=int, default=16)
    ap.add_argument(
        "--bin-pad",
        type=int,
        default=None,
        help="use fixed compact capacity N * bin_pad",
    )
    ap.add_argument(
        "--exact",
        action="store_true",
        help="use exact padded bins instead of compact capacity",
    )
    args = ap.parse_args()

    if args.size is not None:
        width = height = args.size
    else:
        width = args.width
        height = args.height if args.height is not None else args.width

    rng = np.random.default_rng(0)
    params, target, K, viewmat = scene_2dgs(args.n, width, height, rng, spread=args.spread)
    means3d, log_scales, quats, opac_raw, col_raw = params
    opacities = mx.sigmoid(opac_raw)
    colors = mx.sigmoid(col_raw)
    bg = mx.zeros((3,), dtype=mx.float32)
    absgrad_zeros = mx.zeros((args.n, 2), dtype=mx.float32)

    proj = mx.compile(lambda m, ls, q: project_gaussians_2dgs(m, ls, q, viewmat, K, width, height))
    radii, means2d, depths, ray_transforms, _normals = proj(means3d, log_scales, quats)
    visible_opac = mx.where((depths > NEAR_PLANE) & (depths < FAR_PLANE), opacities, 0.0)
    counts = _count_bbox_intersections(means2d, ray_transforms, visible_opac, radii, width, height)
    mx.eval(radii, means2d, depths, ray_transforms, visible_opac, colors, counts)

    real_intersections = int(mx.sum(counts)) if counts.size > 0 else 0
    if args.exact:
        capacity = None
        capacity_label = "exact"
    elif args.bin_pad is not None:
        capacity = args.n * int(args.bin_pad)
        capacity_label = f"capacity={capacity} (pad={args.bin_pad})"
    else:
        capacity = estimate_bin_capacity_2dgs(
            means2d,
            ray_transforms,
            visible_opac,
            radii,
            width,
            height,
            margin=args.capacity_margin,
            min_per_gaussian=args.min_per_gaussian,
        )
        capacity_label = f"capacity={capacity} (auto)"

    def run_projection():
        rad, m2, dep, ray, normals = proj(means3d, log_scales, quats)
        mx.eval(rad, m2, dep, ray, normals)

    def bins_only(m2d, ray, dep, rad, opac):
        order = mx.argsort(dep, axis=-1)
        m = mx.take_along_axis(m2d, mx.broadcast_to(order[:, None], m2d.shape), axis=0)
        r = mx.take_along_axis(ray, mx.broadcast_to(order[:, None, None], ray.shape), axis=0)
        d = mx.take_along_axis(dep, order, axis=-1)
        ra = mx.take_along_axis(rad, mx.broadcast_to(order[:, None], rad.shape), axis=0)
        o = mx.take(opac, order)
        o = mx.where((d > NEAR_PLANE) & (d < FAR_PLANE), o, 0.0)
        return _build_bins(m, r, o, ra, width, height, capacity=capacity)

    bins_compiled = mx.compile(bins_only)

    def run_bins():
        ids, bounds, counts_out = bins_compiled(means2d, ray_transforms, depths, radii, opacities)
        mx.eval(ids, bounds, counts_out)

    raster = mx.compile(
        lambda m2d, ray, dep, rad, op, col, abs_sink: rasterize2dgs_fused(
            m2d,
            ray,
            op,
            col,
            bg,
            dep,
            rad,
            height,
            width,
            absgrad_sink=abs_sink,
            bin_capacity=capacity,
        )
    )

    def run_projected_raster():
        img = raster(means2d, ray_transforms, depths, radii, opacities, colors, absgrad_zeros)
        mx.eval(img)

    def projected_loss_l1(m2d, ray, dep, rad, op, col, abs_sink):
        img = rasterize2dgs_fused(
            m2d,
            ray,
            op,
            col,
            bg,
            dep,
            rad,
            height,
            width,
            absgrad_sink=abs_sink,
            bin_capacity=capacity,
        )
        return mx.mean(mx.abs(img - target))

    projected_fb = mx.compile(mx.value_and_grad(projected_loss_l1, argnums=[0, 1, 4, 5, 6]))

    def run_projected_fb():
        loss, grads = projected_fb(means2d, ray_transforms, depths, radii, opacities, colors, absgrad_zeros)
        mx.eval(loss, *grads)

    def loss_l1(m, ls, q, o, c, abs_sink):
        return pixel_loss_2dgs(
            m,
            ls,
            q,
            o,
            c,
            target,
            viewmat,
            K,
            ssim_weight=0.0,
            means2d_absgrad_sink=abs_sink,
            bin_capacity=capacity,
        )[0]

    def loss_ssim(m, ls, q, o, c, abs_sink):
        return pixel_loss_2dgs(
            m,
            ls,
            q,
            o,
            c,
            target,
            viewmat,
            K,
            ssim_weight=args.ssim_weight,
            means2d_absgrad_sink=abs_sink,
            bin_capacity=capacity,
        )[0]

    fwd_l1 = mx.compile(loss_l1)
    fb_l1 = mx.compile(mx.value_and_grad(loss_l1, argnums=[0, 1, 2, 3, 4, 5]))
    fwd_ssim = mx.compile(loss_ssim)
    fb_ssim = mx.compile(mx.value_and_grad(loss_ssim, argnums=[0, 1, 2, 3, 4, 5]))

    def run_fwd_l1():
        loss = fwd_l1(*params, absgrad_zeros)
        mx.eval(loss)

    def run_fb_l1():
        loss, grads = fb_l1(*params, absgrad_zeros)
        mx.eval(loss, *grads)

    def run_fwd_ssim():
        loss = fwd_ssim(*params, absgrad_zeros)
        mx.eval(loss)

    def run_fb_ssim():
        loss, grads = fb_ssim(*params, absgrad_zeros)
        mx.eval(loss, *grads)

    counts_np = np.asarray(counts).reshape(-1)
    utilization = real_intersections / capacity if capacity else 1.0
    print(
        f"2DGS step profile: N={args.n} image={width}x{height} spread={args.spread} "
        f"bins={capacity_label} real={real_intersections} ({utilization:.0%} used)"
    )
    print(
        f"tile stats: mean={counts_np.mean():.1f} p95={np.percentile(counts_np, 95):.0f} "
        f"p99={np.percentile(counts_np, 99):.0f} max={counts_np.max(initial=0)}"
    )
    print(f"projection only:           {timeit(run_projection, args.iters):8.2f} ms")
    print(f"compact bins only:        {timeit(run_bins, args.iters):8.2f} ms")
    print(f"projected raster fwd:     {timeit(run_projected_raster, args.iters):8.2f} ms")
    print(f"projected raster fwd+bwd: {timeit(run_projected_fb, args.iters):8.2f} ms")
    print(f"full fwd L1:              {timeit(run_fwd_l1, args.iters):8.2f} ms")
    print(f"full fwd+bwd L1:          {timeit(run_fb_l1, args.iters):8.2f} ms")
    if args.ssim_weight > 0.0:
        print(f"full fwd SSIM:            {timeit(run_fwd_ssim, args.iters):8.2f} ms")
        print(f"full fwd+bwd SSIM:        {timeit(run_fb_ssim, args.iters):8.2f} ms")


if __name__ == "__main__":
    main()
