"""Decompose the 3D training step to choose the next speedup.

This is intentionally a small profiling harness, not a benchmark suite. It
breaks the current MLX path into projection, compact-bin construction,
projected rasterization, and full fwd/fwd+bwd losses, optionally comparing
pure L1 against the SSIM-blended loss used by training.

Usage:
    uv run python scripts/profile_3d_step.py --n 50000 --size 512 --iters 30 --ssim-weight 0.2
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawingwithgaussians.losses import pixel_loss_3d  # noqa: E402
from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE, project_gaussians  # noqa: E402
from drawingwithgaussians.rendering3d_fused import (  # noqa: E402
    _bounding_radii,
    _build_bins,
    estimate_bin_capacity,
    rasterize3d_fused,
)
from scripts.bench_render import scene_3d  # noqa: E402


def timeit(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--spread", action="store_true")
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--capacity-margin", type=float, default=2.0)
    ap.add_argument("--min-per-gaussian", type=int, default=16)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    params, target, K, viewmat = scene_3d(args.n, args.size, rng, spread=args.spread)
    means3d, log_scales, quats, opac_raw, col_raw = params
    opacities = mx.sigmoid(opac_raw)
    colors = mx.sigmoid(col_raw)
    bg = mx.zeros((3,), dtype=mx.float32)

    proj = mx.compile(
        lambda m, ls, q: project_gaussians(m, ls, q, viewmat, K, args.size, args.size)
    )
    means2d, conics, depths = proj(means3d, log_scales, quats)
    visible_opac = mx.where(
        (depths > NEAR_PLANE) & (depths < FAR_PLANE), opacities, 0.0
    )
    capacity = estimate_bin_capacity(
        means2d,
        conics,
        visible_opac,
        args.size,
        args.size,
        margin=args.capacity_margin,
        min_per_gaussian=args.min_per_gaussian,
    )
    mx.eval(means2d, conics, depths, visible_opac, colors)

    def run_projection():
        m2, co, de = proj(means3d, log_scales, quats)
        mx.eval(m2, co, de)

    def bins_only(m2d, con, dep, opac):
        order = mx.argsort(dep, axis=-1)
        m = mx.take_along_axis(
            m2d, mx.broadcast_to(order[..., None], m2d.shape), axis=0
        )
        c = mx.take_along_axis(
            con, mx.broadcast_to(order[..., None], con.shape), axis=0
        )
        d = mx.take_along_axis(dep, order, axis=-1)
        o = mx.take(opac, order)
        o = mx.where((d > NEAR_PLANE) & (d < FAR_PLANE), o, 0.0)
        radii = _bounding_radii(c, o)
        return _build_bins(m, c, o, radii, args.size, args.size, capacity=capacity)

    bins_compiled = mx.compile(bins_only)

    def run_bins():
        ids, bounds, counts = bins_compiled(means2d, conics, depths, opacities)
        mx.eval(ids, bounds, counts)

    raster = mx.compile(
        lambda m2d, con, dep, op, col: rasterize3d_fused(
            m2d, con, op, col, bg, dep, args.size, args.size, bin_capacity=capacity
        )
    )

    def run_projected_raster():
        img = raster(means2d, conics, depths, opacities, colors)
        mx.eval(img)

    def loss_l1(m, ls, q, o, c):
        return pixel_loss_3d(
            m, ls, q, o, c, target, viewmat, K, ssim_weight=0.0, bin_capacity=capacity
        )[0]

    def loss_ssim(m, ls, q, o, c):
        return pixel_loss_3d(
            m,
            ls,
            q,
            o,
            c,
            target,
            viewmat,
            K,
            ssim_weight=args.ssim_weight,
            bin_capacity=capacity,
        )[0]

    fwd_l1 = mx.compile(loss_l1)
    fb_l1 = mx.compile(mx.value_and_grad(loss_l1, argnums=[0, 1, 2, 3, 4]))
    fwd_ssim = mx.compile(loss_ssim)
    fb_ssim = mx.compile(mx.value_and_grad(loss_ssim, argnums=[0, 1, 2, 3, 4]))

    def run_fwd_l1():
        loss = fwd_l1(*params)
        mx.eval(loss)

    def run_fb_l1():
        loss, grads = fb_l1(*params)
        mx.eval(loss, *grads)

    def run_fwd_ssim():
        loss = fwd_ssim(*params)
        mx.eval(loss)

    def run_fb_ssim():
        loss, grads = fb_ssim(*params)
        mx.eval(loss, *grads)

    print(
        f"3D step profile: N={args.n} size={args.size} spread={args.spread} capacity={capacity}"
    )
    print(f"projection only:       {timeit(run_projection, args.iters):8.2f} ms")
    print(f"compact bins only:    {timeit(run_bins, args.iters):8.2f} ms")
    print(f"projected raster fwd: {timeit(run_projected_raster, args.iters):8.2f} ms")
    print(f"full fwd L1:          {timeit(run_fwd_l1, args.iters):8.2f} ms")
    print(f"full fwd+bwd L1:      {timeit(run_fb_l1, args.iters):8.2f} ms")
    if args.ssim_weight > 0.0:
        print(f"full fwd SSIM:        {timeit(run_fwd_ssim, args.iters):8.2f} ms")
        print(f"full fwd+bwd SSIM:    {timeit(run_fb_ssim, args.iters):8.2f} ms")


if __name__ == "__main__":
    main()
