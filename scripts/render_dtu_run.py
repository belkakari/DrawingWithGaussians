#!/usr/bin/env python3
"""Render DTU fusion inputs from an existing completed 2DGS run."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from drawingwithgaussians.splat_export import load_ply_3d
from train_colmap3d import _camera_matrices_at_resolution, _export_dtu_renders, _load_colmap_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-side", type=int, help="omit for native DTU resolution")
    parser.add_argument("--count-batch", type=int, default=8)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    config = manifest["resolved_config"]
    scene, train_indices, _ = _load_colmap_scene(
        Path(config["data"]["dir"]),
        int(config["data"]["factor"]),
        bool(config["data"]["normalize"]),
        int(config["data"]["test_every"]),
    )
    stored = json.loads((run_dir / "normalization.json").read_text())
    np.testing.assert_allclose(scene.normalization_center, stored["center"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(scene.normalization_scale, stored["scale"], rtol=0, atol=1e-5)

    params = load_ply_3d(run_dir / "final.ply")
    viewmats, Ks, width, height = _camera_matrices_at_resolution(scene, train_indices, args.max_side)
    output = args.output_dir or (run_dir / "dtu_renders")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _export_dtu_renders(
        params,
        [scene.image_paths[i] for i in train_indices],
        viewmats,
        Ks,
        width,
        height,
        active_sh_degree=int(config["gaussians"]["sh_degree"]),
        count_batch=args.count_batch,
        out_dir=output,
        log=logging.getLogger(__name__),
    )


if __name__ == "__main__":
    main()
