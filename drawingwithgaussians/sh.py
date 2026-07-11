"""Real spherical harmonics used by 3DGS (degrees zero through three).

The coefficient layout is deliberately the standard 3DGS layout:
``sh0`` is ``(N, 1, 3)`` and ``shN`` is ``(N, 15, 3)``.  Evaluation accepts
either a single camera (directions ``(N, 3)``) or a camera batch
(``(B, N, 3)``); MLX broadcasting keeps the latter in one graph.

The polynomial and constants match the reference implementation in mlx3D.
"""

from __future__ import annotations

import mlx.core as mx

C0 = 0.28209479177387814
_C1 = 0.4886025119029199
_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def num_sh_bases(degree: int) -> int:
    if not 0 <= int(degree) <= 3:
        raise ValueError("SH degree must be in [0, 3]")
    return (int(degree) + 1) ** 2


def rgb_to_sh0(rgb: mx.array) -> mx.array:
    """Convert activated RGB to the degree-zero coefficient tensor."""
    if rgb.shape[-1] != 3:
        raise ValueError("RGB must have three channels")
    return ((rgb - 0.5) / C0)[:, None, :]


def sh0_to_rgb(sh0: mx.array) -> mx.array:
    return sh0[..., 0, :] * C0 + 0.5


def eval_sh(degree: int, sh0: mx.array, shN: mx.array, dirs: mx.array) -> mx.array:
    """Evaluate standard real SH and return ``max(0, 0.5 + SH)`` colors.

    ``dirs`` is the world-space direction from camera center to Gaussian.
    Direction gradients are intentionally stopped: appearance fitting should
    not move geometry merely by changing the angular basis.
    """
    degree = int(degree)
    num_sh_bases(degree)
    if sh0.ndim != 3 or sh0.shape[1:] != (1, 3):
        raise ValueError(f"sh0 must have shape (N,1,3), got {sh0.shape}")
    if shN.ndim != 3 or shN.shape[1:] != (15, 3):
        raise ValueError(f"shN must have shape (N,15,3), got {shN.shape}")
    if dirs.shape[-1] != 3 or dirs.shape[-2] != sh0.shape[0]:
        raise ValueError("directions must end in (N,3) and match SH rows")

    batched = dirs.ndim == 3
    coeff0 = sh0[None] if batched else sh0
    coeffN = shN[None] if batched else shN
    dirs = mx.stop_gradient(dirs)
    result = C0 * coeff0[..., 0, :]
    if batched:
        result = mx.broadcast_to(result, dirs.shape[:-1] + (3,))
    if degree > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = result - _C1 * y * coeffN[..., 0, :] + _C1 * z * coeffN[..., 1, :] - _C1 * x * coeffN[..., 2, :]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + _C2[0] * xy * coeffN[..., 3, :]
                + _C2[1] * yz * coeffN[..., 4, :]
                + _C2[2] * (2.0 * zz - xx - yy) * coeffN[..., 5, :]
                + _C2[3] * xz * coeffN[..., 6, :]
                + _C2[4] * (xx - yy) * coeffN[..., 7, :]
            )
            if degree > 2:
                result = (
                    result
                    + _C3[0] * y * (3.0 * xx - yy) * coeffN[..., 8, :]
                    + _C3[1] * xy * z * coeffN[..., 9, :]
                    + _C3[2] * y * (4.0 * zz - xx - yy) * coeffN[..., 10, :]
                    + _C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * coeffN[..., 11, :]
                    + _C3[4] * x * (4.0 * zz - xx - yy) * coeffN[..., 12, :]
                    + _C3[5] * z * (xx - yy) * coeffN[..., 13, :]
                    + _C3[6] * x * (xx - 3.0 * yy) * coeffN[..., 14, :]
                )
    return mx.maximum(0.0, result + 0.5)


def camera_centers_from_viewmats(viewmats: mx.array) -> mx.array:
    """World-space camera centers from OpenCV/COLMAP world-to-camera matrices."""
    squeeze = viewmats.ndim == 2
    mats = viewmats[None] if squeeze else viewmats
    R, t = mats[:, :3, :3], mats[:, :3, 3]
    centers = -mx.matmul(mx.transpose(R, (0, 2, 1)), t[..., None])[..., 0]
    return centers[0] if squeeze else centers


def view_dependent_colors(params: dict[str, mx.array], viewmats: mx.array, active_degree: int) -> mx.array:
    """Evaluate a parameter dictionary's canonical SH appearance per camera."""
    centers = camera_centers_from_viewmats(viewmats)
    if centers.ndim == 1:
        dirs = params["means3d"] - centers
    else:
        dirs = params["means3d"][None, :, :] - centers[:, None, :]
    dirs = dirs / mx.maximum(mx.linalg.norm(dirs, axis=-1, keepdims=True), 1e-8)
    return eval_sh(active_degree, params["sh0"], params["shN"], dirs)


def sh_degree_for_step(global_step: int, interval: int = 1000, max_degree: int = 3) -> int:
    if interval <= 0:
        return int(max_degree)
    return min(max(int(global_step) // int(interval), 0), int(max_degree))
