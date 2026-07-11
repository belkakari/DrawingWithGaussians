#!/usr/bin/env python3
"""Fuse DTU 2DGS depth renders and score the resulting surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData

from drawingwithgaussians.dtu import (
    DTUSampleSet,
    SceneNormalization,
    above_plane_filter,
    evaluate_dtu_points,
    fuse_tsdf,
    observation_mask_filter,
    sample_mesh_surface,
)

DEFAULT_DTU_ROOT = Path(__file__).resolve().parents[1] / "inputs" / "dtu"


def _mesh_points(path: Path, spacing: float, seed: int, dataset: DTUSampleSet) -> np.ndarray:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    vertices = np.column_stack([vertex[a] for a in "xyz"])
    if "face" not in ply or len(ply["face"]) == 0:
        return vertices
    triangles = np.stack(ply["face"].data["vertex_indices"]).astype(np.int64)
    centroids = vertices[triangles].mean(axis=1)
    keep = observation_mask_filter(centroids, dataset.observation_mask_path)
    keep &= above_plane_filter(centroids, dataset.plane_path)
    triangles = triangles[keep]
    return sample_mesh_surface(vertices, triangles, spacing, seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_DTU_ROOT)
    parser.add_argument("--scan", type=int, choices=(1, 6), required=True)
    parser.add_argument("--lighting", default="3")
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--output-mesh", type=Path, default=Path("dtu_tsdf.ply"))
    parser.add_argument("--output-json", type=Path, default=Path("dtu_metrics.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    dataset = DTUSampleSet(args.root, args.scan, args.lighting)
    dataset.validate_layout()

    mesh = args.mesh
    if args.render_dir is not None:
        if args.normalization is None:
            parser.error("--normalization is required with --render-dir")
        payload = json.loads(args.normalization.read_text())
        normalization = SceneNormalization(np.asarray(payload["center"], dtype=float), float(payload["scale"]))
        depths, alphas, rgbs, Ks, w2cs = [], [], [], [], []
        for view in dataset.train_views:
            depths.append(np.load(args.render_dir / f"median_depth_{view:03d}.npy"))
            alphas.append(np.load(args.render_dir / f"alpha_{view:03d}.npy"))
            rgbs.append(np.load(args.render_dir / f"rgb_{view:03d}.npy"))
            projection = dataset.projection(view)
            Ks.append(np.load(args.render_dir / f"K_{view:03d}.npy"))
            w2cs.append(normalization.normalize_w2c(projection.w2c))
        mesh = fuse_tsdf(
            depths,
            alphas,
            rgbs,
            Ks,
            w2cs,
            normalization,
            args.output_mesh,
            voxel_size=0.004,
            truncation=0.02,
            alpha_threshold=0.5,
        )
    if mesh is None:
        parser.error("provide --mesh or --render-dir")
    points = _mesh_points(mesh, spacing=0.2, seed=args.seed, dataset=dataset)
    metrics = evaluate_dtu_points(points, dataset, min_distance=0.2, seed=args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
