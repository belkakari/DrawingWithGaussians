"""Rendering benchmark: forward-only and forward+backward times vs N.

Usage:
    uv run python scripts/bench_render.py [--size 128] [--iters 100]

Times the fused rasterizers (2D and 3D) on seeded synthetic scenes. Used to
track the numbers in EXPERIMENTS.md; run before/after touching kernels.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawingwithgaussians.losses import pixel_loss, pixel_loss_3d  # noqa: E402


def timeit(fn, iters, warmup=15):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def scene_2d(n, size, rng):
    means = mx.array(rng.uniform(0, size, size=(n, 2)).astype(np.float32))
    log_diag = mx.array(rng.uniform(np.log(0.3), np.log(12.0), size=(n, 2)).astype(np.float32))
    offdiag = mx.array(rng.normal(0, 1.5, size=(n,)).astype(np.float32))
    colors = mx.array(rng.uniform(0, 0.4, size=(n, 3)).astype(np.float32))
    bg = mx.array(rng.uniform(0, 1, size=(1, 1, 3)).astype(np.float32))
    target = mx.array(rng.random((size, size, 3)).astype(np.float32))
    args = (means, log_diag, offdiag, colors, bg, target)
    mx.eval(*args)
    return args


def scene_3d(n, size, rng):
    means3d = mx.array((2.0 * (rng.random((n, 3)) - 0.5)).astype(np.float32))
    log_scales = mx.array(np.log(rng.uniform(0.02, 0.3, size=(n, 3))).astype(np.float32))
    quats = mx.array(rng.normal(size=(n, 4)).astype(np.float32))
    opac_raw = mx.array((rng.normal(size=(n,)) - 1.0).astype(np.float32))
    col_raw = mx.array(rng.normal(size=(n, 3)).astype(np.float32))
    target = mx.array(rng.random((size, size, 3)).astype(np.float32))
    focal = 0.5 * size / math.tan(0.25 * math.pi)
    K = mx.array(np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float32))
    viewmat = mx.array(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 8.0], [0, 0, 0, 1]], dtype=np.float32))
    args = (means3d, log_scales, quats, opac_raw, col_raw)
    mx.eval(*args, target, K, viewmat)
    return args, target, K, viewmat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--ns", type=int, nargs="+", default=[1000, 5000, 20000, 50000])
    args_cli = ap.parse_args()
    size, iters = args_cli.size, args_cli.iters
    rng = np.random.default_rng(0)

    print(f"image {size}x{size}, {iters} iters, MLX {mx.__version__}")
    print(f"{'path':6s} {'N':>7s} {'fwd ms':>9s} {'fwd+bwd ms':>11s}")

    for n in args_cli.ns:
        a2 = scene_2d(n, size, rng)

        def fwd2():
            loss, _ = pixel_loss(*a2, ssim_weight=0.0)
            mx.eval(loss)

        vg2 = mx.compile(mx.value_and_grad(lambda *a: pixel_loss(*a, ssim_weight=0.0), argnums=[0, 1, 2, 3, 4]))

        def fb2():
            (loss, _), g = vg2(*a2)
            mx.eval(loss, *g)

        f2c = mx.compile(lambda *a: pixel_loss(*a, ssim_weight=0.0))

        def fwd2c():
            loss, _ = f2c(*a2)
            mx.eval(loss)

        print(f"{'2D':6s} {n:7d} {timeit(fwd2c, iters):9.2f} {timeit(fb2, iters):11.2f}")

    for n in args_cli.ns:
        a3, target, K, viewmat = scene_3d(n, size, rng)

        def loss3(*p):
            return pixel_loss_3d(*p, target, viewmat, K, ssim_weight=0.0)

        f3c = mx.compile(loss3)

        def fwd3c():
            loss, _ = f3c(*a3)
            mx.eval(loss)

        vg3 = mx.compile(mx.value_and_grad(loss3, argnums=[0, 1, 2, 3, 4]))

        def fb3():
            (loss, _), g = vg3(*a3)
            mx.eval(loss, *g)

        print(f"{'3D':6s} {n:7d} {timeit(fwd3c, iters):9.2f} {timeit(fb3, iters):11.2f}")


if __name__ == "__main__":
    main()
