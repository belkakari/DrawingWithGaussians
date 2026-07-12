#!/usr/bin/env python3
"""Apple-Silicon dense-vs-selective Adam correctness and timing benchmark."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from drawingwithgaussians.selective_adam import SelectiveAdam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=int, default=100_000)
    parser.add_argument("--width", type=int, default=48, help="floats per Gaussian across the synthetic table")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--visible-fraction", type=float, default=0.25)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    p0 = mx.array(rng.normal(size=(args.gaussians, args.width)).astype(np.float32))
    grad = mx.array(rng.normal(size=p0.shape).astype(np.float32))
    visible = mx.array(rng.random(args.gaussians) < args.visible_fraction)

    def run_dense():
        opt = optim.Adam(learning_rate=1e-3, bias_correction=True)
        opt.init({"p": p0})
        p = p0
        mx.eval(p, opt.state)
        start = time.perf_counter()
        for _ in range(args.steps):
            p = opt.apply_gradients({"p": grad}, {"p": p})["p"]
            mx.eval(p, opt.state)
        return time.perf_counter() - start, p

    def run_selective(mask):
        opt = SelectiveAdam({"p": 1e-3})
        opt.init({"p": p0})
        p = p0
        mx.eval(p, opt.state)
        start = time.perf_counter()
        for _ in range(args.steps):
            p = opt.apply_gradients({"p": grad}, {"p": p}, mask)["p"]
            mx.eval(p, opt.state)
        return time.perf_counter() - start, p

    dense_time, dense = run_dense()
    all_time, all_visible = run_selective(mx.ones((args.gaussians,), dtype=mx.bool_))
    sparse_time, _ = run_selective(visible)
    error = float(mx.max(mx.abs(dense - all_visible)))
    print(f"dense:              {1e3 * dense_time / args.steps:.3f} ms/step")
    print(f"selective all rows: {1e3 * all_time / args.steps:.3f} ms/step (max error {error:.3e})")
    print(f"selective {args.visible_fraction:.0%}:   {1e3 * sparse_time / args.steps:.3f} ms/step")


if __name__ == "__main__":
    main()
