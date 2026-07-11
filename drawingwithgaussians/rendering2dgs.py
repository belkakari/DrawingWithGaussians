"""MLX 2D Gaussian Splatting (surfel) projection and dense reference.

2DGS uses the same alpha compositing as 3DGS after per-pixel alpha is known,
but the primitive is a camera-visible disk/surfel. Projection produces a
per-splat ray transform ``M``; rasterization intersects each pixel ray with the
surfel and evaluates ``sigma = 0.5 * min(u^2 + v^2, 2 * ||pixel-mean2d||^2)``.

This file is the differentiable MLX reference. The fused Metal kernels live in
``rendering2dgs_fused.py``.
"""

import mlx.core as mx

from .rendering3d import ALPHA_THRESHOLD, FAR_PLANE, MAX_ALPHA, NEAR_PLANE

_TILE = 8


def project_gaussians_2dgs(means3d, log_scales, quats, viewmat, K, width, height, eps=0.0):
    """Project disk/surfel Gaussians for 2DGS.

    Args mirror :func:`drawingwithgaussians.rendering3d.project_gaussians`.
    ``log_scales`` may be ``(N, 3)``; the first two axes span the disk and the
    third column defines the normal orientation as in gsplat's 2DGS path.

    Returns ``(radii, means2d, depths, ray_transforms, normals)`` with an
    optional leading camera-batch dimension.
    """
    qn = quats / mx.linalg.norm(quats, axis=-1, keepdims=True)
    qw, qx, qy, qz = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qw * qz)
    r02 = 2 * (qx * qz + qw * qy)
    r10 = 2 * (qx * qy + qw * qz)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qw * qx)
    r20 = 2 * (qx * qz - qw * qy)
    r21 = 2 * (qy * qz + qw * qx)
    r22 = 1 - 2 * (qx * qx + qy * qy)

    scales = mx.exp(log_scales)
    s0, s1, s2 = scales[:, 0], scales[:, 1], scales[:, 2]
    # Columns of R_wl @ diag(scales).
    w0x, w0y, w0z = r00 * s0, r10 * s0, r20 * s0
    w1x, w1y, w1z = r01 * s1, r11 * s1, r21 * s1
    w2x, w2y, w2z = r02 * s2, r12 * s2, r22 * s2

    def _vm(i, j):
        return viewmat[..., i, j][..., None]

    a00, a01, a02 = _vm(0, 0), _vm(0, 1), _vm(0, 2)
    a10, a11, a12 = _vm(1, 0), _vm(1, 1), _vm(1, 2)
    a20, a21, a22 = _vm(2, 0), _vm(2, 1), _vm(2, 2)
    px, py, pz = means3d[:, 0], means3d[:, 1], means3d[:, 2]
    tx = a00 * px + a01 * py + a02 * pz + _vm(0, 3)
    ty = a10 * px + a11 * py + a12 * pz + _vm(1, 3)
    tz = a20 * px + a21 * py + a22 * pz + _vm(2, 3)

    # Camera-space columns of the scaled local frame.
    c0x = a00 * w0x + a01 * w0y + a02 * w0z
    c0y = a10 * w0x + a11 * w0y + a12 * w0z
    c0z = a20 * w0x + a21 * w0y + a22 * w0z
    c1x = a00 * w1x + a01 * w1y + a02 * w1z
    c1y = a10 * w1x + a11 * w1y + a12 * w1z
    c1z = a20 * w1x + a21 * w1y + a22 * w1z
    nx = a00 * w2x + a01 * w2y + a02 * w2z
    ny = a10 * w2x + a11 * w2y + a12 * w2z
    nz = a20 * w2x + a21 * w2y + a22 * w2z
    normal_sign = mx.where(-(nx * tx + ny * ty + nz * tz) > 0.0, 1.0, -1.0)
    normals = mx.stack([nx * normal_sign, ny * normal_sign, nz * normal_sign], axis=-1)
    # The third scale controls surfel thickness/shared 3D scale pruning, not
    # normal-consistency magnitude. Match gsplat 2DGS by rasterizing unit
    # orientation normals.
    normals = normals / mx.maximum(mx.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)

    fx = K[..., 0, 0][..., None]
    fy = K[..., 1, 1][..., None]
    cx = K[..., 0, 2][..., None]
    cy = K[..., 1, 2][..., None]

    # T_sl = K @ [RS_cl[:, :2], mean_c]. This is the ray transform returned to rasterization.
    m00 = fx * c0x + cx * c0z
    m10 = fy * c0y + cy * c0z
    m20 = c0z
    m01 = fx * c1x + cx * c1z
    m11 = fy * c1y + cy * c1z
    m21 = c1z
    m02 = fx * tx + cx * tz
    m12 = fy * ty + cy * tz
    m22 = tz
    ray_transforms = mx.stack(
        [
            mx.stack([m00, m01, m02], axis=-1),
            mx.stack([m10, m11, m12], axis=-1),
            mx.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    )

    # gsplat AABB from M = (T_sl)^T (gsplat.cuda._torch_impl_2dgs): their
    # ``M[..., 2]`` is M's column 2, i.e. T_sl's third ROW (m20, m21, m22).
    # In T_sl row/col notation: d = t20^2 + t21^2 - t22^2 and
    # mean_j = (t_j0 t20 + t_j1 t21 - t_j2 t22) / d. Sanity anchor: a
    # fronto-parallel on-axis disk (c0z = c1z = 0) gives d = -tz^2 and
    # mean = (cx, cy) exactly.
    d = m20 * m20 + m21 * m21 - m22 * m22
    valid_d = mx.abs(d) > eps
    inv_d = mx.where(valid_d, 1.0 / d, 0.0)
    mean_x = (m00 * m20 + m01 * m21 - m02 * m22) * inv_d
    mean_y = (m10 * m20 + m11 * m21 - m12 * m22) * inv_d
    means2d = mx.stack([mean_x, mean_y], axis=-1)
    ex2 = mx.maximum(mean_x * mean_x - (m00 * m00 + m01 * m01 - m02 * m02) * inv_d, 1e-4)
    ey2 = mx.maximum(mean_y * mean_y - (m10 * m10 + m11 * m11 - m12 * m12) * inv_d, 1e-4)
    radii = mx.ceil(3.33 * mx.stack([mx.sqrt(ex2), mx.sqrt(ey2)], axis=-1))

    valid = valid_d & (tz > NEAR_PLANE) & (tz < FAR_PLANE)
    inside = (
        (mean_x + radii[..., 0] > 0)
        & (mean_x - radii[..., 0] < width)
        & (mean_y + radii[..., 1] > 0)
        & (mean_y - radii[..., 1] < height)
    )
    radii = mx.where((valid & inside)[..., None], radii, 0.0)
    return radii, means2d, tz, ray_transforms, normals


def _sigma_2dgs(means2d, ray_transforms, px, py):
    dx = px[None, :] - means2d[:, 0:1]
    dy = py[None, :] - means2d[:, 1:2]
    M = ray_transforms
    hu0 = -M[:, 0:1, 0] + M[:, 2:3, 0] * px[None, :]
    hu1 = -M[:, 0:1, 1] + M[:, 2:3, 1] * px[None, :]
    hu2 = -M[:, 0:1, 2] + M[:, 2:3, 2] * px[None, :]
    hv0 = -M[:, 1:2, 0] + M[:, 2:3, 0] * py[None, :]
    hv1 = -M[:, 1:2, 1] + M[:, 2:3, 1] * py[None, :]
    hv2 = -M[:, 1:2, 2] + M[:, 2:3, 2] * py[None, :]
    tx = hu1 * hv2 - hu2 * hv1
    ty = hu2 * hv0 - hu0 * hv2
    tw = hu0 * hv1 - hu1 * hv0
    valid = mx.abs(tw) > 1e-8
    inv_w = mx.where(valid, 1.0 / tw, 0.0)
    u = tx * inv_w
    v = ty * inv_w
    sigma3d = u * u + v * v
    sigma2d = 2.0 * (dx * dx + dy * dy)
    sigma = 0.5 * mx.minimum(sigma3d, sigma2d)
    return sigma, valid


def rasterize2dgs_dense(
    means2d,
    ray_transforms,
    opacities,
    colors,
    background,
    depths,
    height,
    width,
    radii=None,
):
    """Dense RGB-only 2DGS alpha compositing reference for a single camera.

    When ``radii`` is provided, it mirrors gsplat's raster path by evaluating
    only pixels inside each projected surfel AABB. The AABB is a visibility
    gate only and carries no gradients in the fused path.
    """
    order = mx.argsort(depths)
    m = mx.take(means2d, order, axis=0)
    M = mx.take(ray_transforms, order, axis=0)
    opac = mx.take(opacities, order, axis=0)
    col = mx.take(colors, order, axis=0)
    dep = mx.take(depths, order, axis=0)
    rad = mx.take(radii, order, axis=0) if radii is not None else None
    opac = mx.where((dep > NEAR_PLANE) & (dep < FAR_PLANE), opac, 0.0)

    px = (mx.arange(width, dtype=mx.float32) + 0.5)[None, :]
    py = (mx.arange(height, dtype=mx.float32) + 0.5)[:, None]
    px = mx.broadcast_to(px, (height, width)).reshape(-1)
    py = mx.broadcast_to(py, (height, width)).reshape(-1)

    sigma, valid = _sigma_2dgs(m, M, px, py)
    alpha = mx.minimum(opac[:, None] * mx.exp(-sigma), MAX_ALPHA)
    if rad is not None:
        tile_x = mx.floor((px - 0.5) / _TILE).astype(mx.int32)
        tile_y = mx.floor((py - 0.5) / _TILE).astype(mx.int32)
        tx0 = mx.clip(
            mx.floor((m[:, 0:1] - rad[:, 0:1]) / _TILE),
            0,
            (width + _TILE - 1) // _TILE - 1,
        ).astype(mx.int32)
        tx1 = mx.clip(
            mx.floor((m[:, 0:1] + rad[:, 0:1]) / _TILE),
            0,
            (width + _TILE - 1) // _TILE - 1,
        ).astype(mx.int32)
        ty0 = mx.clip(
            mx.floor((m[:, 1:2] - rad[:, 1:2]) / _TILE),
            0,
            (height + _TILE - 1) // _TILE - 1,
        ).astype(mx.int32)
        ty1 = mx.clip(
            mx.floor((m[:, 1:2] + rad[:, 1:2]) / _TILE),
            0,
            (height + _TILE - 1) // _TILE - 1,
        ).astype(mx.int32)
        in_tile_bbox = (
            (tile_x[None, :] >= tx0) & (tile_x[None, :] <= tx1) & (tile_y[None, :] >= ty0) & (tile_y[None, :] <= ty1)
        )
        valid = valid & in_tile_bbox
    alpha = mx.where(valid & (alpha >= ALPHA_THRESHOLD), alpha, 0.0)
    one_minus = 1.0 - alpha
    trans = mx.cumprod(one_minus, axis=0)
    t_before = mx.concatenate([mx.ones((1, alpha.shape[1])), trans[:-1]], axis=0)
    weights = alpha * t_before
    acc = weights.T @ col
    t_final = trans[-1]
    out = acc + t_final[:, None] * background[None, :]
    return out.reshape(height, width, 3)
