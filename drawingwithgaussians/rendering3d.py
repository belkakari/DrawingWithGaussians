"""MLX 3D gaussian splatting: projection + dense reference rasterizer.

Port of gsplat's 3DGS pipeline (reference math from
``gsplat/cuda/_torch_impl.py``) for fitting a single image through a fixed
pinhole camera, like ``gsplat/examples/image_fitting.py``:

* ``project_gaussians``: quat/scale -> 3D covariance, world -> camera, EWA
  perspective projection with the clamped Jacobian, +0.3 px low-pass on the
  2D covariance, conic (inverse 2D covariance) and depth. All in regular MLX
  autodiff ops (everything is N-sized).
* ``rasterize3d_dense``: alpha compositing in depth order, materializing the
  (N, P) alpha matrix with ``mx.cumprod`` for transmittance. Reference
  implementation for the fused Metal kernels in ``rendering3d_fused.py``
  (same role as ``rendering2d.rasterize`` for the 2D path).

Conventions follow gsplat: ``sigma = 0.5 (A dx^2 + C dy^2) + B dx dy`` with
conic (A, B, C), pixel centers at +0.5, ``alpha = min(MAX_ALPHA,
opacity * exp(-sigma))``, contributions with ``alpha < 1/255`` skipped.
There is no tile culling: at one 128x128 image per step the pixel loop over
all gaussians is already fast, and skipping culling keeps the math of the
dense and fused paths identical.
"""

import mlx.core as mx

# gsplat constants (gsplat/cuda/_constants.py).
MAX_ALPHA = 0.99
ALPHA_THRESHOLD = 1.0 / 255.0
TRANSMITTANCE_THRESHOLD = 1e-4  # fused path only: early termination, exclusive
EPS2D = 0.3  # low-pass filter added to the projected 2D covariance
NEAR_PLANE = 0.01
FAR_PLANE = 1e10


def project_gaussians(means3d, log_scales, quats, viewmat, K, width, height):
    """EWA-project 3D gaussians through a pinhole camera (gsplat pinhole path).

    Args:
        means3d: (N, 3) world-space centers.
        log_scales: (N, 3) log of the per-axis scales (this repo's log-space
            convention; gsplat passes linear scales).
        quats: (N, 4) wxyz quaternions, normalized in-graph.
        viewmat: (4, 4) world-to-camera matrix, or a (B, 4, 4) batch of them.
        K: (3, 3) intrinsics, or (B, 3, 3) matching ``viewmat``.
        width, height: image size in pixels (shared across the batch).

    Returns:
        ``(means2d, conics, depths)``: (N, 2) pixel-space centers, (N, 3)
        conics (A, B, C) of the inverse 2D covariance (after the +EPS2D
        low-pass), (N,) camera-space depths. With batched cameras the shapes
        gain a leading B: (B, N, 2), (B, N, 3), (B, N) — the scalarized math
        broadcasts (B, 1) camera entries against (N,) gaussian entries, so
        the whole batch is one fused elementwise chain (no loop, no vmap).
    """
    # Fully scalarized (gsplat's CUDA projection structure): every step is an
    # elementwise expression over (N,) arrays, so mx.compile fuses the whole
    # chain into a handful of kernels. The batched-matmul formulation
    # (R@S, M@M^T, Rcw@Sigma@Rcw^T, J@Sigma@J^T on (N, 3, 3)) dispatched GEMM
    # kernels that dominated the entire training step at large N: 11.2 ms of
    # a 12.5 ms forward at N=50k, roughly doubled again in their VJPs, while
    # the rasterization kernels cost 1.7 ms fwd+bwd (see EXPERIMENTS.md).
    # Same math; only fp reassociation differs (goldens regenerated).

    # Rotation matrix entries from the normalized quaternion (wxyz).
    qn = quats / mx.linalg.norm(quats, axis=-1, keepdims=True)
    w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    # cov3d = R diag(s^2) R^T, six unique entries.
    s2 = mx.exp(2.0 * log_scales)
    s0, s1, s2_ = s2[:, 0], s2[:, 1], s2[:, 2]
    v00 = r00 * r00 * s0 + r01 * r01 * s1 + r02 * r02 * s2_
    v11 = r10 * r10 * s0 + r11 * r11 * s1 + r12 * r12 * s2_
    v22 = r20 * r20 * s0 + r21 * r21 * s1 + r22 * r22 * s2_
    v01 = r00 * r10 * s0 + r01 * r11 * s1 + r02 * r12 * s2_
    v02 = r00 * r20 * s0 + r01 * r21 * s1 + r02 * r22 * s2_
    v12 = r10 * r20 * s0 + r11 * r21 * s1 + r12 * r22 * s2_

    # World -> camera. Camera entries are indexed with an ellipsis so a
    # (C, 4, 4) batch broadcasts as (C, 1) against (N,) gaussians — gsplat's
    # [..., C, N] convention (_fully_fused_projection) without loop or vmap.
    def _vm(i, j):
        return viewmat[..., i, j][..., None]

    a00, a01, a02 = _vm(0, 0), _vm(0, 1), _vm(0, 2)
    a10, a11, a12 = _vm(1, 0), _vm(1, 1), _vm(1, 2)
    a20, a21, a22 = _vm(2, 0), _vm(2, 1), _vm(2, 2)
    px, py, pz = means3d[:, 0], means3d[:, 1], means3d[:, 2]
    tx = a00 * px + a01 * py + a02 * pz + _vm(0, 3)
    ty = a10 * px + a11 * py + a12 * pz + _vm(1, 3)
    tz = a20 * px + a21 * py + a22 * pz + _vm(2, 3)

    # cov_c = Rcw cov3d Rcw^T: rows of (Rcw @ cov3d) first, then contract.
    b00 = a00 * v00 + a01 * v01 + a02 * v02
    b01 = a00 * v01 + a01 * v11 + a02 * v12
    b02 = a00 * v02 + a01 * v12 + a02 * v22
    b10 = a10 * v00 + a11 * v01 + a12 * v02
    b11 = a10 * v01 + a11 * v11 + a12 * v12
    b12 = a10 * v02 + a11 * v12 + a12 * v22
    b20 = a20 * v00 + a21 * v01 + a22 * v02
    b21 = a20 * v01 + a21 * v11 + a22 * v12
    b22 = a20 * v02 + a21 * v12 + a22 * v22
    c00_ = b00 * a00 + b01 * a01 + b02 * a02
    c01_ = b00 * a10 + b01 * a11 + b02 * a12
    c02_ = b00 * a20 + b01 * a21 + b02 * a22
    c11_ = b10 * a10 + b11 * a11 + b12 * a12
    c12_ = b10 * a20 + b11 * a21 + b12 * a22
    c22_ = b20 * a20 + b21 * a21 + b22 * a22

    fx = K[..., 0, 0][..., None]
    fy = K[..., 1, 1][..., None]
    cx = K[..., 0, 2][..., None]
    cy = K[..., 1, 2][..., None]

    # Clamp the Jacobian evaluation point into (a margin around) the view
    # frustum, as gsplat does, so off-screen gaussians don't get absurd
    # linearizations.
    tan_fovx = 0.5 * width / fx
    tan_fovy = 0.5 * height / fy
    lim_x_pos = (width - cx) / fx + 0.3 * tan_fovx
    lim_x_neg = cx / fx + 0.3 * tan_fovx
    lim_y_pos = (height - cy) / fy + 0.3 * tan_fovy
    lim_y_neg = cy / fy + 0.3 * tan_fovy
    tx_c = tz * mx.clip(tx / tz, -lim_x_neg, lim_x_pos)
    ty_c = tz * mx.clip(ty / tz, -lim_y_neg, lim_y_pos)

    # cov2d = J cov_c J^T with J rows (ja, 0, jb) and (0, jd, je).
    tz2 = tz * tz
    ja = fx / tz
    jb = -fx * tx_c / tz2
    jd = fy / tz
    je = -fy * ty_c / tz2
    c2_00 = ja * ja * c00_ + 2.0 * ja * jb * c02_ + jb * jb * c22_ + EPS2D
    c2_01 = ja * jd * c01_ + ja * je * c02_ + jb * jd * c12_ + jb * je * c22_
    c2_11 = jd * jd * c11_ + 2.0 * jd * je * c12_ + je * je * c22_ + EPS2D

    det = mx.maximum(c2_00 * c2_11 - c2_01 * c2_01, 1e-10)
    conics = mx.stack([c2_11 / det, -c2_01 / det, c2_00 / det], axis=-1)
    means2d = mx.stack([fx * tx / tz + cx, fy * ty / tz + cy], axis=-1)
    return means2d, conics, tz


def rasterize3d_dense(means2d, conics, opacities, colors, background, depths, height, width):
    """Dense reference alpha compositing (materializes the (N, P) matrix).

    Args:
        means2d, conics, depths: from :func:`project_gaussians`.
        opacities: (N,) in [0, 1].
        colors: (N, 3) in [0, 1].
        background: (3,) composited behind the splats with the final
            transmittance.

    Returns:
        (H, W, 3) image.
    """
    order = mx.argsort(depths)
    m = mx.take(means2d, order, axis=0)
    con = mx.take(conics, order, axis=0)
    opac = mx.take(opacities, order, axis=0)
    col = mx.take(colors, order, axis=0)
    dep = mx.take(depths, order, axis=0)
    # Hard-cull gaussians outside the depth range (gsplat sets their radius
    # to 0): zero opacity removes them from compositing.
    opac = mx.where((dep > NEAR_PLANE) & (dep < FAR_PLANE), opac, 0.0)

    # Pixel centers, x = column.
    px = (mx.arange(width, dtype=mx.float32) + 0.5)[None, :]
    py = (mx.arange(height, dtype=mx.float32) + 0.5)[:, None]
    px = mx.broadcast_to(px, (height, width)).reshape(-1)
    py = mx.broadcast_to(py, (height, width)).reshape(-1)

    dx = px[None, :] - m[:, 0:1]  # (N, P)
    dy = py[None, :] - m[:, 1:2]
    a, b, c = con[:, 0:1], con[:, 1:2], con[:, 2:3]
    sigma = 0.5 * (a * dx * dx + c * dy * dy) + b * dx * dy
    alpha = mx.minimum(opac[:, None] * mx.exp(-sigma), MAX_ALPHA)
    # Match the kernels: negative sigma is invalid, tiny alphas are skipped.
    alpha = mx.where((sigma >= 0) & (alpha >= ALPHA_THRESHOLD), alpha, 0.0)

    one_minus = 1.0 - alpha  # (N, P)
    trans = mx.cumprod(one_minus, axis=0)  # T after each gaussian
    t_before = mx.concatenate([mx.ones((1, alpha.shape[1])), trans[:-1]], axis=0)
    weights = alpha * t_before  # (N, P)
    acc = weights.T @ col  # (P, 3)
    t_final = trans[-1]  # (P,)
    out = acc + t_final[:, None] * background[None, :]
    return out.reshape(height, width, 3)
