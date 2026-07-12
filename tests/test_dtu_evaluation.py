from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.io import savemat

from drawingwithgaussians.dtu import (
    DTUSampleSet,
    SceneNormalization,
    deterministic_min_distance,
    evaluate_dtu_points,
    factor_projection_matrix,
    observation_mask_filter,
    project_points,
    scene_normalization,
)
from drawingwithgaussians.evaluation import ViewMetrics, aggregate_view_metrics, write_metrics


def test_projection_factorization_and_reprojection():
    theta = 0.31
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    K = np.array([[1200.0, 2.0, 800.0], [0, 1180.0, 600.0], [0, 0, 1]])
    t = np.array([20.0, -10.0, 700.0])
    P = -3.7 * K @ np.column_stack([R, t])  # exercise arbitrary negative scale
    result = factor_projection_matrix(P)
    assert np.all(np.diag(result.K) > 0)
    np.testing.assert_allclose(np.linalg.det(result.R), 1.0, atol=1e-10)
    points = np.array([[1, 2, 3], [-4, 2, 8], [10, -3, 2]], dtype=float)
    uv0, _ = project_points(P, points)
    uv1, _ = project_points(result.K @ np.column_stack([result.R, result.t]), points)
    np.testing.assert_allclose(uv0, uv1, atol=1e-9)


def test_normalization_reversal_and_pose():
    cameras = np.array([[10, 0, 0], [-10, 0, 0], [0, 6, 0]], dtype=float)
    norm = scene_normalization(cameras)
    points = np.array([[4, 5, 6], [-1, 2, 3]], dtype=float)
    np.testing.assert_allclose(norm.denormalize_points(norm.normalize_points(points)), points)
    w2c = np.eye(4)
    w2c[:3, 3] = [1, 2, 3]
    normalized_w2c = norm.normalize_w2c(w2c)
    p_h = np.column_stack([points, np.ones(len(points))])
    pn_h = np.column_stack([norm.normalize_points(points), np.ones(len(points))])
    np.testing.assert_allclose((p_h @ w2c.T)[:, :3] / norm.scale, (pn_h @ normalized_w2c.T)[:, :3])


def test_mask_indexing_fixture(tmp_path: Path):
    mask = np.zeros((3, 4, 5), dtype=np.uint8)
    mask[1, 2, 3] = 1
    path = tmp_path / "mask.mat"
    savemat(path, {"ObsMask": mask, "BB": np.array([[10, 20, 30], [12, 23, 34]]), "Res": [[1.0]]})
    points = np.array([[11, 22, 33], [10, 20, 30], [50, 50, 50]], dtype=float)
    np.testing.assert_array_equal(observation_mask_filter(points, path), [True, False, False])


def test_deterministic_min_distance():
    rng = np.random.default_rng(9)
    points = rng.normal(size=(1000, 3))
    a = deterministic_min_distance(points, 0.2, seed=7)
    b = deterministic_min_distance(points, 0.2, seed=7)
    np.testing.assert_array_equal(a, b)


def _write_points(path: Path, points: np.ndarray):
    vertex = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i, axis in enumerate("xyz"):
        vertex[axis] = points[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def test_dtu_metric_known_fixture(tmp_path: Path):
    root = tmp_path / "MVS Data"
    (root / "ObsMask").mkdir(parents=True)
    gt = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1]], dtype=float)
    pred = gt + np.array([0, 0, 0.5])
    _write_points(root / "Points/stl/stl001_total.ply", gt)
    mask = np.ones((8, 8, 8), dtype=np.uint8)
    savemat(root / "ObsMask/ObsMask1_10.mat", {"ObsMask": mask, "BB": [[-2, -2, -2], [5, 5, 5]], "Res": [[1.0]]})
    savemat(root / "ObsMask/Plane1.mat", {"P": np.array([[0], [0], [1], [0]], dtype=float)})
    dataset = DTUSampleSet(root, scan=1)
    result = evaluate_dtu_points(pred, dataset, min_distance=0.01, seed=2)
    np.testing.assert_allclose(result["accuracy_mm"], 0.5, atol=1e-7)
    np.testing.assert_allclose(result["completeness_mm"], 0.5, atol=1e-7)
    np.testing.assert_allclose(result["overall_mm"], 0.5, atol=1e-7)
    assert result["fscore_1mm"] == 1.0


def test_structured_metric_output(tmp_path: Path):
    views = [ViewMetrics("a", 20, 0.8, 0.2, 0.1, 0.05), ViewMetrics("b", 22, 0.9, 0.1, 0, 0)]
    aggregate = aggregate_view_metrics(views)
    assert aggregate["aggregate"]["psnr"] == 21
    path = write_metrics(tmp_path, 12, views, final=True)
    assert path.name == "metrics_final.json"
    assert json.loads(path.read_text())["step"] == 12


def test_local_sampleset_protocol_if_available():
    root = Path(__file__).resolve().parents[1] / "inputs" / "dtu"
    if not root.is_dir():
        return
    dataset = DTUSampleSet(root, scan=1, lighting=3)
    assert len(dataset.train_views) == 42
    assert len(dataset.validation_views) == 7
    assert dataset.validation_views == (1, 9, 17, 25, 33, 41, 49)
    dataset.validate_layout()
    projection = dataset.projection(1)
    assert projection.reconstruction_relative_error < 1e-8


def test_tsdf_extraction_in_isolated_process(tmp_path: Path):
    output = tmp_path / "plane.ply"
    code = """
import numpy as np, sys
from drawingwithgaussians.dtu import SceneNormalization, fuse_tsdf
h=w=32; K=np.array([[30,0,w/2],[0,30,h/2],[0,0,1]],float)
d=np.ones((h,w),np.float32); a=np.ones((h,w),np.float32); rgb=np.full((h,w,3),.5,np.float32)
fuse_tsdf([d],[a],[rgb],[K],[np.eye(4)],SceneNormalization(np.zeros(3),1.0),sys.argv[1],voxel_size=.04,truncation=.12)
"""
    subprocess.run([sys.executable, "-c", code, str(output)], check=True)
    mesh = PlyData.read(output)
    assert len(mesh["vertex"]) > 0 and len(mesh["face"]) > 0
    np.testing.assert_allclose(np.mean(mesh["vertex"]["z"]), 1.0, atol=1e-3)
    assert output.with_suffix(".normalization.json").is_file()
