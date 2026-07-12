"""DTU SampleSet adapter, calibration, TSDF fusion, and official-style metrics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.linalg import rq
from scipy.spatial import cKDTree

SUPPORTED_SCANS = (1, 6)


@dataclass(frozen=True)
class Projection:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    w2c: np.ndarray
    camera_center: np.ndarray
    reconstruction_relative_error: float


@dataclass(frozen=True)
class SceneNormalization:
    center: np.ndarray
    scale: float

    @property
    def original_to_normalized(self) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] /= self.scale
        T[:3, 3] = -self.center / self.scale
        return T

    @property
    def normalized_to_original(self) -> np.ndarray:
        return np.linalg.inv(self.original_to_normalized)

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) - self.center) / self.scale

    def denormalize_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points) * self.scale + self.center

    def normalize_w2c(self, w2c_original: np.ndarray) -> np.ndarray:
        source = np.asarray(w2c_original, dtype=np.float64)
        normalized = source.copy()
        normalized[:3, 3] = (source[:3, :3] @ self.center + source[:3, 3]) / self.scale
        return normalized

    def to_json(self) -> dict:
        return {"center": self.center.tolist(), "scale": float(self.scale)}


def factor_projection_matrix(P: np.ndarray, tolerance: float = 1e-8) -> Projection:
    """Factor a DTU ``P`` into positive-diagonal ``K`` and proper ``[R|t]``.

    Projection matrices are defined only up to scale.  Choosing the sign that
    makes ``det(P[:,:3])`` positive makes positive-diagonal RQ yield a proper
    rotation without sacrificing the intrinsic sign convention.
    """
    source = np.asarray(P, dtype=np.float64)
    if source.shape != (3, 4) or not np.isfinite(source).all():
        raise ValueError("P must be a finite 3x4 projection matrix")
    work = source.copy()
    if np.linalg.det(work[:, :3]) < 0:
        work = -work
    K, R = rq(work[:, :3])
    signs = np.sign(np.diag(K))
    signs[signs == 0] = 1.0
    D = np.diag(signs)
    K, R = K @ D, D @ R
    if np.linalg.det(R) < 0:  # numerical/degenerate guard
        raise ValueError("projection factorization did not produce det(R)=+1")
    intrinsic_scale = float(K[2, 2])
    K = K / intrinsic_scale
    t = np.linalg.solve(K, work[:, 3] / intrinsic_scale)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3], w2c[:3, 3] = R, t
    reconstructed = K @ np.column_stack([R, t])
    scale = float(np.vdot(source, reconstructed) / np.vdot(reconstructed, reconstructed))
    rel = float(np.linalg.norm(source - scale * reconstructed) / np.linalg.norm(source))
    if rel > tolerance:
        raise ValueError(f"projection reconstruction relative error {rel:.3e} exceeds {tolerance:.3e}")
    return Projection(K, R, t, w2c, -R.T @ t, rel)


def scale_intrinsics(K: np.ndarray, sx: float, sy: float | None = None) -> np.ndarray:
    out = np.asarray(K, dtype=np.float64).copy()
    sy = sx if sy is None else sy
    out[0, :] *= float(sx)
    out[1, :] *= float(sy)
    return out


def scene_normalization(camera_centers: np.ndarray) -> SceneNormalization:
    centers = np.asarray(camera_centers, dtype=np.float64)
    center = centers.mean(axis=0)
    scale = max(float(np.linalg.norm(centers - center, axis=1).max(initial=0.0)), 1e-8)
    return SceneNormalization(center, scale)


class DTUSampleSet:
    """Concrete adapter for scans 1 and 6 of the downloadable DTU SampleSet."""

    def __init__(self, root: str | Path, scan: int, lighting: str | int = 3, image_scale: float = 1.0):
        self.root = Path(root).expanduser().resolve()
        self.scan = int(scan)
        if self.scan not in SUPPORTED_SCANS:
            raise ValueError(f"SampleSet adapter supports scans {SUPPORTED_SCANS}, got {scan}")
        self.lighting = str(lighting).lower()
        if self.lighting != "max" and int(self.lighting) not in range(7):
            raise ValueError("lighting must be 0..6 or 'max'")
        self.image_scale = float(image_scale)
        if not 0 < self.image_scale <= 1:
            raise ValueError("image_scale must be in (0, 1]")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

    @property
    def views(self) -> tuple[int, ...]:
        return tuple(range(1, 50))

    @property
    def validation_views(self) -> tuple[int, ...]:
        return self.views[::8]

    @property
    def train_views(self) -> tuple[int, ...]:
        held_out = set(self.validation_views)
        return tuple(v for v in self.views if v not in held_out)

    def image_path(self, view: int) -> Path:
        suffix = "max" if self.lighting == "max" else f"{int(self.lighting)}_r5000"
        path = self.root / "Cleaned" / f"scan{self.scan}" / f"clean_{int(view):03d}_{suffix}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def projection_path(self, view: int) -> Path:
        return self.root / "Calibration" / "cal18" / f"pos_{int(view):03d}.txt"

    def projection(self, view: int) -> Projection:
        result = factor_projection_matrix(np.loadtxt(self.projection_path(view), dtype=np.float64))
        if self.image_scale == 1:
            return result
        K = scale_intrinsics(result.K, self.image_scale)
        return Projection(K, result.R, result.t, result.w2c, result.camera_center, result.reconstruction_relative_error)

    @property
    def ground_truth_points_path(self) -> Path:
        return self.root / "Points" / "stl" / f"stl{self.scan:03d}_total.ply"

    @property
    def observation_mask_path(self) -> Path:
        return self.root / "ObsMask" / f"ObsMask{self.scan}_10.mat"

    @property
    def plane_path(self) -> Path:
        return self.root / "ObsMask" / f"Plane{self.scan}.mat"

    def cache_key(self, feature_settings: dict, matcher_settings: dict, filter_settings: dict | None = None) -> str:
        payload = {
            "scan": self.scan,
            "lighting": self.lighting,
            "image_scale": self.image_scale,
            "train_views": self.train_views,
            "feature": feature_settings,
            "matcher": matcher_settings,
            "filter": filter_settings or {},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def validate_layout(self) -> None:
        for view in self.views:
            self.image_path(view)
            if not self.projection_path(view).is_file():
                raise FileNotFoundError(self.projection_path(view))
        for path in (self.ground_truth_points_path, self.observation_mask_path, self.plane_path):
            if not path.is_file():
                raise FileNotFoundError(path)


def project_points(P: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=np.float64)
    q = np.column_stack([p, np.ones(len(p))]) @ np.asarray(P, dtype=np.float64).T
    return q[:, :2] / q[:, 2:3], q[:, 2]


def deterministic_min_distance(points: np.ndarray, min_distance: float = 0.2, seed: int = 0) -> np.ndarray:
    """Deterministic equivalent of DTU's stochastic ``reducePts_haa``."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts.copy()
    order = np.random.default_rng(seed).permutation(len(pts))
    cell_size = float(min_distance)
    grid: dict[tuple[int, int, int], list[int]] = {}
    kept: list[int] = []
    for idx in order:
        cell = tuple(np.floor(pts[idx] / cell_size).astype(np.int64))
        accept = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for other in grid.get((cell[0] + dx, cell[1] + dy, cell[2] + dz), ()):
                        if np.linalg.norm(pts[idx] - pts[other]) < min_distance:
                            accept = False
                            break
                    if not accept:
                        break
                if not accept:
                    break
            if not accept:
                break
        if accept:
            kept.append(int(idx))
            grid.setdefault(cell, []).append(int(idx))
    return pts[np.sort(kept)]


def sample_mesh_surface(vertices: np.ndarray, triangles: np.ndarray, spacing: float = 0.2, seed: int = 0) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    tri = vertices[triangles]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    valid = areas > 0
    tri, areas = tri[valid], areas[valid]
    if not len(tri):
        return np.empty((0, 3), dtype=np.float64)
    count = max(1, int(np.ceil(2.0 * areas.sum() / (spacing * spacing))))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(tri), count, p=areas / areas.sum())
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    sampled = (
        tri[chosen, 0] + uv[:, 0:1] * (tri[chosen, 1] - tri[chosen, 0]) + uv[:, 1:2] * (tri[chosen, 2] - tri[chosen, 0])
    )
    return deterministic_min_distance(sampled, spacing, seed)


def observation_mask_filter(points: np.ndarray, mask_file: str | Path) -> np.ndarray:
    data = loadmat(mask_file)
    mask = np.asarray(data["ObsMask"], dtype=bool)
    bb = np.asarray(data["BB"], dtype=np.float64)
    res = float(np.asarray(data["Res"]).item())
    # MATLAB: round((Qdata-BB(1,:))/Res+1), then 1-based indexing.
    idx = np.rint((np.asarray(points) - bb[0]) / res).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.asarray(mask.shape)), axis=1)
    keep = np.zeros(len(idx), dtype=bool)
    valid = idx[inside]
    keep[inside] = mask[valid[:, 0], valid[:, 1], valid[:, 2]]
    return keep


def above_plane_filter(points: np.ndarray, plane_file: str | Path) -> np.ndarray:
    plane = np.asarray(loadmat(plane_file)["P"], dtype=np.float64).reshape(4)
    return np.column_stack([points, np.ones(len(points))]) @ plane > 0


def _load_ply_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"]
    return np.column_stack([vertex[axis] for axis in "xyz"]).astype(np.float64)


def evaluate_dtu_points(
    predicted_points: np.ndarray,
    dataset: DTUSampleSet,
    min_distance: float = 0.2,
    max_distance: float = 20.0,
    seed: int = 0,
) -> dict[str, float | int]:
    """Official MATLAB protocol in Python, with deterministic downsampling."""
    pred = deterministic_min_distance(predicted_points, min_distance, seed)
    gt = _load_ply_points(dataset.ground_truth_points_path)
    if not len(pred) or not len(gt):
        raise ValueError("DTU evaluation requires non-empty prediction and ground truth")
    d_pred = cKDTree(gt).query(pred, workers=-1)[0]
    d_gt = cKDTree(pred).query(gt, workers=-1)[0]
    pred_use = observation_mask_filter(pred, dataset.observation_mask_path)
    pred_use &= above_plane_filter(pred, dataset.plane_path)
    gt_use = above_plane_filter(gt, dataset.plane_path)
    accuracy_dist = d_pred[pred_use]
    completeness_dist = d_gt[gt_use]
    accuracy = accuracy_dist[accuracy_dist < max_distance]
    completeness = completeness_dist[completeness_dist < max_distance]
    if not len(accuracy) or not len(completeness):
        raise ValueError("DTU masks/outlier threshold removed every distance")

    def fscore(threshold: float) -> float:
        precision = float(np.mean(accuracy_dist < threshold))
        recall = float(np.mean(completeness_dist < threshold))
        return 2 * precision * recall / max(precision + recall, 1e-12)

    acc_mean, comp_mean = float(accuracy.mean()), float(completeness.mean())
    return {
        "accuracy_mm": acc_mean,
        "completeness_mm": comp_mean,
        "overall_mm": 0.5 * (acc_mean + comp_mean),
        "fscore_1mm": fscore(1.0),
        "fscore_2mm": fscore(2.0),
        "predicted_samples": int(len(pred)),
        "accuracy_samples": int(len(accuracy)),
        "completeness_samples": int(len(completeness)),
    }


def fuse_tsdf(
    depths: Sequence[np.ndarray],
    alphas: Sequence[np.ndarray],
    rgbs: Sequence[np.ndarray],
    Ks: Sequence[np.ndarray],
    w2cs_normalized: Sequence[np.ndarray],
    normalization: SceneNormalization,
    output_mesh: str | Path,
    voxel_size: float = 0.004,
    truncation: float = 0.02,
    alpha_threshold: float = 0.5,
) -> Path:
    """Fuse full-resolution median z-depths in normalized space with MLX."""
    from .tsdf_mlx import fuse_tsdf_mlx

    output = Path(output_mesh)
    fuse_tsdf_mlx(
        depths,
        alphas,
        rgbs,
        Ks,
        w2cs_normalized,
        normalization.normalized_to_original,
        output,
        voxel_size,
        truncation,
        alpha_threshold,
    )
    output.with_suffix(".normalization.json").write_text(json.dumps(normalization.to_json(), indent=2) + "\n")
    return output
