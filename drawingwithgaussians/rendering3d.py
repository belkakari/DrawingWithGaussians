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


def quats_to_rotmats(quats):
    """(N, 4) wxyz quaternions (not necessarily normalized) -> (N, 3, 3)."""
    q = quats / mx.linalg.norm(quats, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return mx.stack(
        [
            mx.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                axis=-1,
            ),
            mx.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                axis=-1,
            ),
            mx.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def project_gaussians(means3d, log_scales, quats, viewmat, K, width, height):
    """EWA-project 3D gaussians through a pinhole camera (gsplat pinhole path).

    Args:
        means3d: (N, 3) world-space centers.
        log_scales: (N, 3) log of the per-axis scales (this repo's log-space
            convention; gsplat passes linear scales).
        quats: (N, 4) wxyz quaternions, normalized in-graph.
        viewmat: (4, 4) world-to-camera matrix.
        K: (3, 3) intrinsics.
        width, height: image size in pixels.

    Returns:
        ``(means2d, conics, depths)``: (N, 2) pixel-space centers, (N, 3)
        conics (A, B, C) of the inverse 2D covariance (after the +EPS2D
        low-pass), (N,) camera-space depths.
    """
    R = quats_to_rotmats(quats)  # (N, 3, 3)
    M = R * mx.exp(log_scales)[:, None, :]  # R @ diag(s)
    cov3d = M @ mx.transpose(M, (0, 2, 1))

    Rcw = viewmat[:3, :3]
    tcw = viewmat[:3, 3]
    means_c = means3d @ Rcw.T + tcw  # (N, 3)
    cov_c = Rcw @ cov3d @ Rcw.T  # (N, 3, 3) (broadcasted matmuls)

    tx, ty, tz = means_c[:, 0], means_c[:, 1], means_c[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

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

    tz2 = tz * tz
    zeros = mx.zeros_like(tz)
    J = mx.stack(
        [fx / tz, zeros, -fx * tx_c / tz2, zeros, fy / tz, -fy * ty_c / tz2],
        axis=-1,
    ).reshape(-1, 2, 3)

    cov2d = J @ cov_c @ mx.transpose(J, (0, 2, 1))  # (N, 2, 2)
    c00 = cov2d[:, 0, 0] + EPS2D
    c11 = cov2d[:, 1, 1] + EPS2D
    c01 = cov2d[:, 0, 1]
    c10 = cov2d[:, 1, 0]
    det = mx.maximum(c00 * c11 - c01 * c10, 1e-10)
    conics = mx.stack([c11 / det, -(c01 + c10) / 2.0 / det, c00 / det], axis=-1)

    means2d = (means_c @ K[:2, :3].T) / tz[:, None]  # (N, 2), x = column
    return means2d, conics, tz


def rasterize3d_dense(
    means2d, conics, opacities, colors, background, depths, height, width
):
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
