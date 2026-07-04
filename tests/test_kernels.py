"""Pytest regression tests for the fused Metal kernels and SSIM loss.

Methodology borrowed from gsplat-mlx: frozen seeded inputs, expected values
from independent implementations (the dense MLX reference paths, fp64 numpy
compositing, scipy-free numpy SSIM), plus golden npz fixtures capturing the
fused kernels' own outputs so silent drift is caught even when both live
paths move together.

Run:
    uv run pytest tests/test_kernels.py
    uv run pytest tests/test_kernels.py --write-goldens  # (re)write fixtures
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
import pytest

from drawingwithgaussians.gaussian import build_L
from drawingwithgaussians.losses import pixel_loss, pixel_loss_2dgs, pixel_loss_3d, ssim
from drawingwithgaussians.rendering2d import rasterize
from drawingwithgaussians.rendering2dgs import (
    project_gaussians_2dgs,
    rasterize2dgs_dense,
)
from drawingwithgaussians.rendering3d import project_gaussians, rasterize3d_dense

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
H = W = 64  # small grid keeps the dense (N, P) reference cheap


def _assert_close(name: str, err: float, tol: float) -> None:
    if err > tol:
        pytest.fail(f"{name}: err={err:.3e} > tol={tol:.1e}")


def _check_or_write_fixture(
    path: Path,
    payload: dict[str, np.ndarray],
    tolerances: dict[str, float],
    write: bool,
) -> None:
    if write:
        path.parent.mkdir(exist_ok=True)
        np.savez(str(path), **cast(dict[str, Any], payload))
        return

    with np.load(path) as ref:
        for key, value in payload.items():
            tol = tolerances.get(key, tolerances["default"])
            _assert_close(f"golden {key}", float(np.abs(value - ref[key]).max()), tol)


# ---------------------------------------------------------------------------
# 2D scene
# ---------------------------------------------------------------------------


def scene_2d():
    rng = np.random.default_rng(7)
    n = 200
    means = rng.uniform(0, H, size=(n, 2)).astype(np.float32)
    # include tiny (0.3) and large sigmas plus nonzero off-diagonals
    log_diag = rng.uniform(np.log(0.3), np.log(12.0), size=(n, 2)).astype(np.float32)
    offdiag = rng.normal(0, 1.5, size=(n,)).astype(np.float32)
    colors = rng.uniform(0, 0.4, size=(n, 3)).astype(np.float32)
    bg = rng.uniform(0, 1, size=(1, 1, 3)).astype(np.float32)
    target = rng.uniform(0, 1, size=(H, W, 3)).astype(np.float32)
    return means, log_diag, offdiag, colors, bg, target


def dense_loss_2d(m, ld, od, c, b, t):
    L = build_L(ld, od)
    covs = L @ mx.transpose(L, (0, 2, 1))
    background = mx.broadcast_to(b, (H, W, 3))
    rendered, _, _ = rasterize(m, covs, c, background, H, W)
    return mx.mean(mx.abs(rendered - t)), rendered


def fused_loss_2d(m, ld, od, c, b, t):
    return pixel_loss(m, ld, od, c, b, t, ssim_weight=0.0)


def truth_2d_forward_fp64(means, log_diag, offdiag, colors, bg):
    diag = np.minimum(np.exp(log_diag.astype(np.float64)), 20.0)
    n = len(means)
    L = np.zeros((n, 2, 2))
    L[:, 0, 0] = diag[:, 0]
    L[:, 1, 0] = offdiag
    L[:, 1, 1] = diag[:, 1]
    cov = L @ np.transpose(L, (0, 2, 1))
    det = np.maximum(cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] * cov[:, 1, 0], 1e-8)
    p00 = cov[:, 1, 1] / det
    p11 = cov[:, 0, 0] / det
    cross = (-cov[:, 1, 0] - cov[:, 0, 1]) / det
    xg = np.repeat(np.arange(H), W).astype(np.float64)
    yg = np.tile(np.arange(W), H).astype(np.float64)
    dx = xg[None, :] - means.astype(np.float64)[:, 0:1]
    dy = yg[None, :] - means.astype(np.float64)[:, 1:2]
    pdf = 0.5 * (
        p00[:, None] * dx * dx + cross[:, None] * dx * dy + p11[:, None] * dy * dy
    )
    y = np.exp(-(pdf - pdf.min(axis=1)[:, None]))
    rendered = bg.reshape(1, 3).astype(np.float64) + y.T @ colors.astype(np.float64)
    return rendered.reshape(H, W, 3)


def test_2d_fused_rasterizer(write_goldens: bool):
    means, log_diag, offdiag, colors, bg, target = scene_2d()
    args = [mx.array(a) for a in (means, log_diag, offdiag, colors, bg, target)]

    vg_dense = mx.value_and_grad(dense_loss_2d, argnums=[0, 1, 2, 3, 4])
    vg_fused = mx.value_and_grad(fused_loss_2d, argnums=[0, 1, 2, 3, 4])
    (ld_, rd), gd = vg_dense(*args)
    (lf, rf), gf = vg_fused(*args)
    mx.eval(ld_, lf, rd, rf, *gd, *gf)

    rend64 = truth_2d_forward_fp64(means, log_diag, offdiag, colors, bg)

    # Fused must stay at least as close to fp64 truth as the dense path.
    err_dense = np.abs(np.array(rd, dtype=np.float64) - rend64).max()
    err_fused = np.abs(np.array(rf, dtype=np.float64) - rend64).max()
    _assert_close("forward vs fp64 truth", err_fused, max(2e-3, 2 * err_dense))
    _assert_close("loss fused-vs-dense", abs(lf.item() - ld_.item()), 5e-3)

    names = ["dmeans", "dlog_diag", "doffdiag", "dcolors", "dbg"]
    # dbg flows through sign(rendered - target): pixels at the L1 sign
    # boundary flip between implementations, each worth 2/(3*H*W) ~ 1.6e-4.
    tols = [1e-4, 5e-4, 1e-4, 5e-4, 2e-3]
    for name, dense_grad, fused_grad, tol in zip(names, gd, gf, tols, strict=True):
        _assert_close(
            f"grad {name} fused-vs-dense",
            float(mx.abs(dense_grad - fused_grad).max()),
            tol,
        )

    # Golden fixture: the fused outputs themselves (2D kernels are
    # deterministic, so the tolerance is tight).
    payload = {"loss": np.array(lf.item()), "rendered": np.array(rf)}
    payload.update({f"g{i}": np.array(g) for i, g in enumerate(gf)})
    _check_or_write_fixture(
        FIXTURE_DIR / "fused2d.npz", payload, {"default": 1e-6}, write_goldens
    )


# ---------------------------------------------------------------------------
# 3D scene
# ---------------------------------------------------------------------------


def scene_3d():
    rng = np.random.default_rng(11)
    n = 300
    means3d = (2.0 * (rng.random((n, 3)) - 0.5)).astype(np.float32)
    log_scales = np.log(rng.uniform(0.05, 0.6, size=(n, 3))).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    opac_raw = (rng.normal(size=(n,)) - 1.0).astype(np.float32)
    col_raw = rng.normal(size=(n, 3)).astype(np.float32)
    target = rng.random((H, W, 3)).astype(np.float32)
    focal = 0.5 * W / math.tan(0.25 * math.pi)
    K = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float32)
    viewmat = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 8.0], [0, 0, 0, 1]], dtype=np.float32
    )
    return means3d, log_scales, quats, opac_raw, col_raw, target, K, viewmat


def test_3d_fused_rasterizer(write_goldens: bool):
    means3d, log_scales, quats, opac_raw, col_raw, target, K, viewmat = scene_3d()
    args = [mx.array(a) for a in (means3d, log_scales, quats, opac_raw, col_raw)]
    target_mx, K_mx, view_mx = mx.array(target), mx.array(K), mx.array(viewmat)
    bg = mx.zeros((3,), dtype=mx.float32)

    def dense_loss(m, ls, q, o, c):
        m2d, con, dep = project_gaussians(m, ls, q, view_mx, K_mx, W, H)
        img = rasterize3d_dense(m2d, con, mx.sigmoid(o), mx.sigmoid(c), bg, dep, H, W)
        return mx.mean(mx.abs(img - target_mx)), img

    def fused_loss(m, ls, q, o, c):
        return pixel_loss_3d(m, ls, q, o, c, target_mx, view_mx, K_mx, ssim_weight=0.0)

    (ld_, rd), gd = mx.value_and_grad(dense_loss, argnums=[0, 1, 2, 3, 4])(*args)
    (lf, rf), gf = mx.value_and_grad(fused_loss, argnums=[0, 1, 2, 3, 4])(*args)
    mx.eval(ld_, lf, rd, rf, *gd, *gf)

    _assert_close("image fused-vs-dense", float(mx.abs(rd - rf).max()), 2e-3)
    _assert_close("loss fused-vs-dense", abs(lf.item() - ld_.item()), 1e-4)
    names = ["dmeans3d", "dlog_scales", "dquats", "dopac", "dcolors"]
    for name, dense_grad, fused_grad in zip(names, gd, gf, strict=True):
        # bounded by early-termination + fp32 differences of the *dense* path
        _assert_close(
            f"grad {name} fused-vs-dense",
            float(mx.abs(dense_grad - fused_grad).max()),
            5e-5,
        )

    # Absgrad densification path: the ignored sink receives the fused
    # rasterizer's per-pixel absolute means2d-gradient accumulation.
    sink = mx.zeros((means3d.shape[0], 2), dtype=mx.float32)

    def fused_loss_abs(s):
        return pixel_loss_3d(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            target_mx,
            view_mx,
            K_mx,
            ssim_weight=0.0,
            means2d_absgrad_sink=s,
        )[0]

    absgrad = mx.grad(fused_loss_abs)(sink)
    mx.eval(absgrad)
    if not bool(mx.all(mx.isfinite(absgrad))):
        pytest.fail("absgrad contains non-finite values")
    if float(mx.max(mx.abs(absgrad))) <= 0.0:
        pytest.fail("absgrad is all zero")

    # Golden fixture with a loose tolerance on grads: the 3D backward uses
    # atomic adds, so results are non-deterministic at the ulp level.
    payload = {"loss": np.array(lf.item()), "rendered": np.array(rf)}
    payload.update({f"g{i}": np.array(g) for i, g in enumerate(gf)})
    _check_or_write_fixture(
        FIXTURE_DIR / "fused3d.npz", payload, {"default": 1e-5}, write_goldens
    )


# ---------------------------------------------------------------------------
# SSIM vs numpy reference
# ---------------------------------------------------------------------------


def ssim_reference_np(img1, img2, window=11, sigma=1.5, c1=0.01**2, c2=0.03**2):
    """Direct (non-separable) fp64 SSIM with 'same' zero padding, matching
    the original 3DGS torch implementation's structure."""
    half = window // 2
    g1 = np.exp(-((np.arange(window) - half) ** 2) / (2 * sigma**2))
    g1 /= g1.sum()
    kernel = np.outer(g1, g1)

    def blur(x):
        out = np.zeros_like(x)
        xp = np.pad(x, ((half, half), (half, half), (0, 0)))
        for i in range(window):
            for j in range(window):
                out += kernel[i, j] * xp[i : i + x.shape[0], j : j + x.shape[1]]
        return out

    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    mu1, mu2 = blur(i1), blur(i2)
    s1 = blur(i1 * i1) - mu1**2
    s2 = blur(i2 * i2) - mu2**2
    s12 = blur(i1 * i2) - mu1 * mu2
    num = (2 * mu1 * mu2 + c1) * (2 * s12 + c2)
    den = (mu1**2 + mu2**2 + c1) * (s1 + s2 + c2)
    return (num / den).mean()


def test_2dgs_fused_rasterizer():
    means3d, log_scales, quats, opac_raw, col_raw, target, K, viewmat = scene_3d()
    # Keep the dense reference cheap while still exercising nontrivial surfels.
    means3d, log_scales, quats, opac_raw, col_raw = (
        means3d[:150],
        log_scales[:150],
        quats[:150],
        opac_raw[:150],
        col_raw[:150],
    )
    args = [mx.array(a) for a in (means3d, log_scales, quats, opac_raw, col_raw)]
    target_mx, K_mx, view_mx = mx.array(target), mx.array(K), mx.array(viewmat)
    bg = mx.zeros((3,), dtype=mx.float32)

    def dense_loss(m, ls, q, o, c):
        radii, m2d, dep, ray, _ = project_gaussians_2dgs(m, ls, q, view_mx, K_mx, W, H)
        img = rasterize2dgs_dense(
            m2d, ray, mx.sigmoid(o), mx.sigmoid(c), bg, dep, H, W, radii
        )
        return mx.mean(mx.abs(img - target_mx)), img

    def fused_loss(m, ls, q, o, c):
        return pixel_loss_2dgs(
            m, ls, q, o, c, target_mx, view_mx, K_mx, ssim_weight=0.0, bin_pad=16
        )

    (ld_, rd), gd = mx.value_and_grad(dense_loss, argnums=[0, 1, 2, 3, 4])(*args)
    (lf, rf), gf = mx.value_and_grad(fused_loss, argnums=[0, 1, 2, 3, 4])(*args)
    mx.eval(ld_, lf, rd, rf, *gd, *gf)

    _assert_close("2dgs image fused-vs-dense", float(mx.abs(rd - rf).max()), 2e-3)
    _assert_close("2dgs loss fused-vs-dense", abs(lf.item() - ld_.item()), 1e-4)
    names = ["dmeans3d", "dlog_scales", "dquats", "dopac", "dcolors"]
    for name, dense_grad, fused_grad in zip(names, gd, gf, strict=True):
        _assert_close(
            f"2dgs grad {name} fused-vs-dense",
            float(mx.abs(dense_grad - fused_grad).max()),
            1e-4,
        )


def test_ssim_matches_numpy_reference():
    rng = np.random.default_rng(3)
    a = rng.random((H, W, 3)).astype(np.float32)
    b = np.clip(a + 0.1 * rng.random((H, W, 3)).astype(np.float32), 0, 1)

    got_same = float(ssim(mx.array(a), mx.array(a)))
    _assert_close("ssim(x, x) == 1", abs(got_same - 1.0), 1e-5)

    got = float(ssim(mx.array(a), mx.array(b)))
    want = ssim_reference_np(a, b)
    _assert_close("ssim vs fp64 reference", abs(got - want), 1e-4)

    # Gradient flows and is finite.
    grad = mx.grad(lambda x: ssim(x, mx.array(b)))(mx.array(a))
    mx.eval(grad)
    if not bool(mx.all(mx.isfinite(grad))):
        pytest.fail("SSIM gradient contains non-finite values")


def test_batched_3d_matches_per_view_loop():
    """Camera-batched rendering must equal the per-view loop.

    This checks losses and gradients for every parameter, including the shared
    means2d_offset (net grad) and absgrad sinks whose cotangents sum over
    views.
    """
    means3d, log_scales, quats, opac_raw, col_raw, _, K, viewmat = scene_3d()
    m3, ls, q, o, c = (
        mx.array(a) for a in (means3d, log_scales, quats, opac_raw, col_raw)
    )
    rng = np.random.default_rng(5)
    ncams = 3
    Ks, vms, targets = [], [], []
    for i in range(ncams):
        vm = viewmat.copy()
        vm[2, 3] += 0.5 * i
        vm[0, 3] += 0.2 * i
        Ks.append(K)
        vms.append(vm)
        targets.append(rng.random((H, W, 3)).astype(np.float32))
    KB, VB, TB = (
        mx.array(np.stack(Ks)),
        mx.array(np.stack(vms)),
        mx.array(np.stack(targets)),
    )
    n = means3d.shape[0]
    offset = mx.zeros((n, 2))
    sink = mx.zeros((n, 2))

    def batched(m3, ls, q, o, c, offset, sink):
        loss, _ = pixel_loss_3d(
            m3,
            ls,
            q,
            o,
            c,
            TB,
            VB,
            KB,
            ssim_weight=0.2,
            means2d_offset=offset,
            means2d_absgrad_sink=sink,
        )
        return loss

    def loop(m3, ls, q, o, c, offset, sink):
        total = 0.0
        for i in range(ncams):
            li, _ = pixel_loss_3d(
                m3,
                ls,
                q,
                o,
                c,
                TB[i],
                VB[i],
                KB[i],
                ssim_weight=0.2,
                means2d_offset=offset,
                means2d_absgrad_sink=sink,
            )
            total = total + li
        return total / ncams

    argnums = [0, 1, 2, 3, 4, 5, 6]
    lb, gb = mx.value_and_grad(batched, argnums=argnums)(m3, ls, q, o, c, offset, sink)
    ll, gl = mx.value_and_grad(loop, argnums=argnums)(m3, ls, q, o, c, offset, sink)
    mx.eval(lb, ll, *gb, *gl)

    _assert_close("loss batched-vs-loop", abs(float(lb) - float(ll)), 1e-6)
    names = [
        "dmeans3d",
        "dlog_scales",
        "dquats",
        "dopac",
        "dcolors",
        "doffset",
        "dabsgrad",
    ]
    for name, batched_grad, loop_grad in zip(names, gb, gl, strict=True):
        _assert_close(
            f"grad {name} batched-vs-loop",
            float(mx.abs(batched_grad - loop_grad).max()),
            1e-6,
        )
