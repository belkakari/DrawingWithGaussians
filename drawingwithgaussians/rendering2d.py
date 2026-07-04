"""MLX 2D-gaussian rasterizer.

Equivalent of the JAX ``rasterize`` in the original project, but in MLX. The
math is identical: per-gaussian peak-normalized exp(-mahalanobis), then a
single matmul ``intensity.T @ colors`` collapses the gaussian axis. Background
is broadcast-added.

The function signature mirrors the JAX port (``covariances`` not ``L`` is the
input) so losses.py and other callers stay the same shape.
"""

import mlx.core as mx

# Floor on the determinant of the rotated covariance used when inverting.
# Prevents the precision matrix from overflowing when the optimizer drives
# a variance very close to zero. Tiny (1e-8) so it doesn't distort normal
# rendering. The actual guard against degenerate gaussians is the variance
# pruning in ``split_n_prune``.
_DET_EPS = 1e-8

# Pixel-grid cache keyed by (height, width, dtype). The grid is constant for
# a given image size, so rebuilding it lazily on every rasterize call just
# adds graph nodes to every forward/backward. Cached arrays are evaluated
# once and reused as constants (also keeps them out of mx.compile retracing).
_GRID_CACHE = {}


def _pixel_grid(height, width, dtype):
    key = (height, width, dtype)
    grid = _GRID_CACHE.get(key)
    if grid is None:
        xg = mx.broadcast_to(mx.arange(height, dtype=dtype)[:, None], (height, width)).reshape(-1)
        yg = mx.broadcast_to(mx.arange(width, dtype=dtype)[None, :], (height, width)).reshape(-1)
        mx.eval(xg, yg)
        grid = _GRID_CACHE[key] = (xg, yg)
    return grid


def rasterize(
    means: mx.array,
    covariances: mx.array,
    colors: mx.array,
    background: mx.array,
    height: int,
    width: int,
):
    """Rasterize Gaussians into an (H, W, 3) image.

    Args:
        means: (N, 2) gaussian centers in pixel coords.
        covariances: (N, 2, 2) per-gaussian covariance matrices. Orientation
            is fully encoded here (the cholesky off-diagonal gives the
            cross-correlation), so no separate rotation matrix is needed.
        colors: (N, 3) RGB values per gaussian. Only the first 3 channels are used.
        background: (H, W, 3) background image.
        height: image height in pixels.
        width: image width in pixels.

    Returns:
        (color, None, None): ``color`` is (H, W, 3) RGB image. The trailing
        ``(opacities, partitioning)`` are ``None`` placeholders to mirror the
        alpha-composer's three-tuple signature used in the JAX port.
    """
    assert means.shape[0] == covariances.shape[0] == colors.shape[0]

    # Pixel grid (P,) flattened — same layout as jnp.mgrid[0:H, 0:W].reshape(-1).
    xg, yg = _pixel_grid(height, width, means.dtype)

    # Precision matrix = inv(covariances). Orientation already lives in the
    # covariance (built from the full cholesky factor L = build_L(log_diag,
    # offdiag), so cov = L @ L^T spans every 2x2 SPD matrix), so there is no
    # separate rotation to apply — gsplat likewise keeps orientation in a
    # single parameterization (R @ diag(s^2) @ R^T from a normalized
    # quaternion). Only the symmetric part contributes to ``delta^T P
    # delta``, so we invert via a 2x2 closed form (faster than mx.linalg.inv
    # on (N, 2, 2)).
    M = covariances  # (N, 2, 2)
    m00, m01 = M[:, 0, 0], M[:, 0, 1]
    m10, m11 = M[:, 1, 0], M[:, 1, 1]
    # Add an ``_DET_EPS`` floor to the determinant so a near-singular
    # covariance (e.g., one of the variances was driven to ~0 by the
    # optimizer) doesn't produce precision-matrix entries large enough to
    # overflow the per-pixel exp. The actual gaussians with tiny variance
    # are pruned during ``split_n_prune``; this eps is just a safety net.
    det = mx.maximum(m00 * m11 - m01 * m10, _DET_EPS)
    # precision entries: p00 = m11/det, p11 = m00/det, p01 = -m10/det, p10 = -m01/det
    # Only (p00 + p11) and (p01 + p10) appear in delta^T P delta, so we keep the
    # combined cross term rather than the full 4-element inverse.
    p00 = (m11 / det)[:, None]
    p11 = (m00 / det)[:, None]
    cross = ((-m10 - m01) / det)[:, None]

    dx = xg[None, :] - means[:, 0:1]  # (N, P)
    dy = yg[None, :] - means[:, 1:2]
    pdf = 0.5 * (p00 * dx * dx + cross * dx * dy + p11 * dy * dy)  # (N, P)

    # Per-gaussian peak-normalize over pixels, then exp. Same effect as the old
    # ``jax_stable_exp(-pdf, axis=1)``.
    z = -pdf
    z = z - mx.max(z, axis=1, keepdims=True)
    intensity = mx.exp(z)  # (N, P)

    color = background + (intensity.T @ colors[:, :3]).reshape(height, width, 3)
    return color, None, None
