"""Regression tests for the fused Metal kernels and the SSIM loss.

Methodology borrowed from gsplat-mlx: frozen seeded inputs, expected values
from independent implementations (the dense MLX reference paths, fp64 numpy
compositing, scipy-free numpy SSIM), plus golden npz fixtures capturing the
fused kernels' own outputs so silent drift is caught even when both live
paths move together.

Run:
    poetry run python tests/test_kernels.py            # check
    poetry run python tests/test_kernels.py --write    # (re)write golden fixtures

No pytest dependency — plain asserts, exits non-zero on failure.
"""

import math
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawingwithgaussians.gaussian import build_L  # noqa: E402
from drawingwithgaussians.losses import pixel_loss, pixel_loss_3d, ssim  # noqa: E402
from drawingwithgaussians.rendering2d import rasterize  # noqa: E402
from drawingwithgaussians.rendering3d import project_gaussians, rasterize3d_dense  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
H = W = 64  # small grid keeps the dense (N, P) reference cheap


def _report(name, err, tol):
    status = "ok " if err <= tol else "FAIL"
    print(f"  [{status}] {name}: err={err:.3e} tol={tol:.1e}")
    return err <= tol


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
    pdf = 0.5 * (p00[:, None] * dx * dx + cross[:, None] * dx * dy + p11[:, None] * dy * dy)
    y = np.exp(-(pdf - pdf.min(axis=1)[:, None]))
    rendered = bg.reshape(1, 3).astype(np.float64) + y.T @ colors.astype(np.float64)
    return rendered.reshape(H, W, 3)


def test_2d(write):
    print("2D fused rasterizer:")
    means, log_diag, offdiag, colors, bg, target = scene_2d()
    args = [mx.array(a) for a in (means, log_diag, offdiag, colors, bg, target)]

    vg_dense = mx.value_and_grad(dense_loss_2d, argnums=[0, 1, 2, 3, 4])
    vg_fused = mx.value_and_grad(fused_loss_2d, argnums=[0, 1, 2, 3, 4])
    (ld_, rd), gd = vg_dense(*args)
    (lf, rf), gf = vg_fused(*args)
    mx.eval(ld_, lf, rd, rf, *gd, *gf)

    rend64 = truth_2d_forward_fp64(means, log_diag, offdiag, colors, bg)

    ok = True
    # fused must stay at least as close to fp64 truth as the dense path
    err_dense = np.abs(np.array(rd, dtype=np.float64) - rend64).max()
    err_fused = np.abs(np.array(rf, dtype=np.float64) - rend64).max()
    ok &= _report("forward vs fp64 truth", err_fused, max(2e-3, 2 * err_dense))
    ok &= _report("loss fused-vs-dense", abs(lf.item() - ld_.item()), 5e-3)
    names = ["dmeans", "dlog_diag", "doffdiag", "dcolors", "dbg"]
    # dbg flows through sign(rendered - target): pixels at the L1 sign
    # boundary flip between implementations, each worth 2/(3*H*W) ~ 1.6e-4.
    tols = [1e-4, 5e-4, 1e-4, 5e-4, 2e-3]
    for name, a, b, tol in zip(names, gd, gf, tols):
        ok &= _report(f"grad {name} fused-vs-dense", float(mx.abs(a - b).max()), tol)

    # golden fixture: the fused outputs themselves (2D kernels are
    # deterministic, so the tolerance is tight)
    fix = FIXTURE_DIR / "fused2d.npz"
    payload = {"loss": np.array(lf.item()), "rendered": np.array(rf)}
    payload.update({f"g{i}": np.array(g) for i, g in enumerate(gf)})
    if write:
        np.savez(fix, **payload)
        print(f"  wrote {fix}")
    else:
        ref = np.load(fix)
        for k, v in payload.items():
            ok &= _report(f"golden {k}", float(np.abs(v - ref[k]).max()), 1e-6)
    return ok


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
    viewmat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 8.0], [0, 0, 0, 1]], dtype=np.float32)
    return means3d, log_scales, quats, opac_raw, col_raw, target, K, viewmat


def test_3d(write):
    print("3D fused rasterizer:")
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

    ok = True
    ok &= _report("image fused-vs-dense", float(mx.abs(rd - rf).max()), 2e-3)
    ok &= _report("loss fused-vs-dense", abs(lf.item() - ld_.item()), 1e-4)
    names = ["dmeans3d", "dlog_scales", "dquats", "dopac", "dcolors"]
    for name, a, b in zip(names, gd, gf):
        # bounded by early-termination + fp32 differences of the *dense* path
        ok &= _report(f"grad {name} fused-vs-dense", float(mx.abs(a - b).max()), 5e-5)

    # golden fixture with a loose tolerance on grads: the 3D backward uses
    # atomic adds, so results are non-deterministic at the ulp level.
    fix = FIXTURE_DIR / "fused3d.npz"
    payload = {"loss": np.array(lf.item()), "rendered": np.array(rf)}
    payload.update({f"g{i}": np.array(g) for i, g in enumerate(gf)})
    if write:
        np.savez(fix, **payload)
        print(f"  wrote {fix}")
    else:
        ref = np.load(fix)
        ok &= _report("golden loss", float(np.abs(payload["loss"] - ref["loss"])), 1e-5)
        ok &= _report("golden rendered", float(np.abs(payload["rendered"] - ref["rendered"]).max()), 1e-5)
        for i in range(5):
            ok &= _report(f"golden g{i}", float(np.abs(payload[f"g{i}"] - ref[f"g{i}"]).max()), 1e-5)
    return ok


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


def test_ssim():
    print("SSIM:")
    rng = np.random.default_rng(3)
    a = rng.random((H, W, 3)).astype(np.float32)
    b = np.clip(a + 0.1 * rng.random((H, W, 3)).astype(np.float32), 0, 1)
    ok = True
    got_same = float(ssim(mx.array(a), mx.array(a)))
    ok &= _report("ssim(x, x) == 1", abs(got_same - 1.0), 1e-5)
    got = float(ssim(mx.array(a), mx.array(b)))
    want = ssim_reference_np(a, b)
    ok &= _report("ssim vs fp64 reference", abs(got - want), 1e-4)
    # gradient flows and is finite
    g = mx.grad(lambda x: ssim(x, mx.array(b)))(mx.array(a))
    mx.eval(g)
    ok &= _report("ssim grad finite", 0.0 if bool(mx.all(mx.isfinite(g))) else 1.0, 0.5)
    return ok


if __name__ == "__main__":
    write = "--write" in sys.argv
    if write:
        FIXTURE_DIR.mkdir(exist_ok=True)
    passed = True
    passed &= test_2d(write)
    passed &= test_3d(write)
    passed &= test_ssim()
    print("PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)
