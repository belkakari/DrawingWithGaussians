"""Export trained 3D Gaussians to SuperSplat-compatible PLY.

This mirrors the standard uncompressed PLY path in
``gsplat/gsplat/exporter.py`` without a PyTorch dependency. The current
trainer has only view-independent RGB logits, so colors are exported as SH
degree-0 coefficients (``f_dc_*``) with no ``f_rest_*`` columns.
"""

from io import BytesIO
from pathlib import Path

import mlx.core as mx
import numpy as np

_SH_C0 = 0.28209479177387814


def _sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def _params_to_arrays(params):
    mx.eval(*params.values())
    means = np.asarray(params["means3d"], dtype=np.float32)
    log_scales = np.asarray(params["log_scales"], dtype=np.float32)
    quats = np.asarray(params["quats"], dtype=np.float32)
    opacities = np.asarray(params["opacities_raw"], dtype=np.float32)
    rgb = _sigmoid_np(np.asarray(params["colors_raw"], dtype=np.float32)).astype(
        np.float32
    )

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = (quats / np.maximum(norms, 1e-12)).astype(np.float32)

    valid = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(log_scales).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(opacities)
        & np.isfinite(rgb).all(axis=1)
    )
    return means[valid], log_scales[valid], quats[valid], opacities[valid], rgb[valid]


def ply_bytes_3d(params):
    """Return standard uncompressed 3DGS binary PLY bytes.

    The current trainer has only view-independent RGB logits, so we export SH
    degree 0 only: ``f_dc = (rgb - 0.5) / C0`` and no ``f_rest_*`` columns.
    """
    means, log_scales, quats, opacities, rgb = _params_to_arrays(params)
    sh0 = ((rgb - 0.5) / _SH_C0).astype(np.float32)

    buffer = BytesIO()
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element vertex {means.shape[0]}\n".encode())
    for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
        buffer.write(f"property float {name}\n".encode())
    for i in range(3):
        buffer.write(f"property float scale_{i}\n".encode())
    for i in range(4):
        buffer.write(f"property float rot_{i}\n".encode())
    buffer.write(b"end_header\n")

    data = np.concatenate(
        [means, sh0, opacities[:, None], log_scales, quats], axis=1
    ).astype("<f4", copy=False)
    buffer.write(data.tobytes())
    return buffer.getvalue()


def export_ply_3d(params, path):
    """Write standard 3DGS ``.ply`` and return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ply_bytes_3d(params))
    return path
