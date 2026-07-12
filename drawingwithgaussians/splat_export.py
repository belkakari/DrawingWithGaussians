"""Export trained 3D Gaussians to standard 3DGS/SuperSplat PLY.

This mirrors the standard uncompressed PLY path in
``gsplat/gsplat/exporter.py`` without a PyTorch dependency. Higher-order
coefficients use canonical channel-major ``f_rest_*`` ordering.
"""

from io import BytesIO
from pathlib import Path

import mlx.core as mx
import numpy as np
from plyfile import PlyData


def _params_to_arrays(params):
    mx.eval(*params.values())
    means = np.asarray(params["means3d"], dtype=np.float32)
    log_scales = np.asarray(params["log_scales"], dtype=np.float32)
    quats = np.asarray(params["quats"], dtype=np.float32)
    opacities = np.asarray(params["opacities_raw"], dtype=np.float32)
    sh0 = np.asarray(params["sh0"], dtype=np.float32)
    shN = np.asarray(params["shN"], dtype=np.float32)

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = (quats / np.maximum(norms, 1e-12)).astype(np.float32)

    valid = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(log_scales).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(opacities)
        & np.isfinite(sh0).all(axis=(1, 2))
        & np.isfinite(shN).all(axis=(1, 2))
    )
    return means[valid], log_scales[valid], quats[valid], opacities[valid], sh0[valid], shN[valid]


def ply_bytes_3d(params):
    """Return standard uncompressed 3DGS binary PLY bytes.

    ``f_rest_*`` is flattened after ``(N,15,3) -> (N,3,15)`` so all red
    coefficients precede green, then blue, matching the canonical format.
    """
    means, log_scales, quats, opacities, sh0, shN = _params_to_arrays(params)
    dc = sh0[:, 0, :]
    rest = shN.transpose(0, 2, 1).reshape(len(means), -1)

    buffer = BytesIO()
    buffer.write(b"ply\n")
    buffer.write(b"format binary_little_endian 1.0\n")
    buffer.write(f"element vertex {means.shape[0]}\n".encode())
    for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"):
        buffer.write(f"property float {name}\n".encode())
    for i in range(rest.shape[1]):
        buffer.write(f"property float f_rest_{i}\n".encode())
    buffer.write(b"property float opacity\n")
    for i in range(3):
        buffer.write(f"property float scale_{i}\n".encode())
    for i in range(4):
        buffer.write(f"property float rot_{i}\n".encode())
    buffer.write(b"end_header\n")

    data = np.concatenate([means, dc, rest, opacities[:, None], log_scales, quats], axis=1).astype("<f4", copy=False)
    buffer.write(data.tobytes())
    return buffer.getvalue()


def export_ply_3d(params, path):
    """Write standard 3DGS ``.ply`` and return the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ply_bytes_3d(params))
    return path


def load_ply_3d(path):
    """Load this pipeline's canonical degree-3 3DGS PLY into MLX tensors."""
    vertex = PlyData.read(path)["vertex"]
    names = set(vertex.data.dtype.names or ())
    required = {
        *("x", "y", "z"),
        *(f"f_dc_{i}" for i in range(3)),
        *(f"f_rest_{i}" for i in range(45)),
        "opacity",
        *(f"scale_{i}" for i in range(3)),
        *(f"rot_{i}" for i in range(4)),
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"degree-3 Gaussian PLY is missing properties: {missing}")

    def columns(prefix, count):
        return np.column_stack([vertex[f"{prefix}_{i}"] for i in range(count)]).astype(np.float32)

    rest = columns("f_rest", 45).reshape(len(vertex), 3, 15).transpose(0, 2, 1)
    return {
        "means3d": mx.array(np.column_stack([vertex[axis] for axis in "xyz"]).astype(np.float32)),
        "sh0": mx.array(columns("f_dc", 3)[:, None, :]),
        "shN": mx.array(rest),
        "opacities_raw": mx.array(np.asarray(vertex["opacity"], dtype=np.float32)),
        "log_scales": mx.array(columns("scale", 3)),
        "quats": mx.array(columns("rot", 4)),
    }
