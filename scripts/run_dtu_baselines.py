#!/usr/bin/env python3
"""Run and aggregate three-seed DTU 2DGS RGB and geometry baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from baseline_common import experiment_sha256, matching_run_stamp, source_sha256, write_run_stamp


def _command(command, root: Path, log_path: Path) -> None:
    environment = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "HYDRA_FULL_ERROR": "1"}
    with log_path.open("a") as log:
        result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def _complete_training(output: Path, seed: int, config_sha256: str, source_hash: str) -> bool:
    return (
        (output / "metrics_final.json").is_file()
        and (output / "final.ply").is_file()
        and matching_run_stamp(
            output,
            mode="2dgs",
            seed=seed,
            config_sha256=config_sha256,
            source_sha256=source_hash,
        )
    )


def _train(
    root: Path,
    scene: Path,
    output: Path,
    seed: int,
    config_sha256: str,
    source_hash: str,
) -> None:
    if _complete_training(output, seed, config_sha256, source_hash):
        print(f"reuse scan6 seed {seed} training: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    print(f"train scan6 seed {seed}: {output}", flush=True)
    _command(
        [
            sys.executable,
            str(root / "train_colmap3d.py"),
            "--config-name",
            "train_colmap3d.yaml",
            f"data.dir={scene}",
            "gaussians.mode=2dgs",
            f"optim.seed={seed}",
            "train.eval_save_renders=false",
            "train.eval_lpips=true",
            "train.save_video=false",
            "train.save_depth=false",
            "train.save_dtu_renders=false",
            f"hydra.run.dir={output}",
        ],
        root,
        output / "driver.log",
    )
    write_run_stamp(
        output,
        mode="2dgs",
        seed=seed,
        config_sha256=config_sha256,
        source_sha256=source_hash,
    )


def _geometry(root: Path, output: Path, scan: int) -> None:
    if (output / "dtu_metrics.json").is_file() and (output / "dtu_tsdf.ply").is_file():
        print(f"reuse scan{scan} geometry: {output}", flush=True)
        return
    renders = output / "dtu_renders"
    if not renders.is_dir():
        print(f"render scan{scan} native fusion views: {output}", flush=True)
        _command(
            [sys.executable, str(root / "scripts" / "render_dtu_run.py"), "--run-dir", str(output)],
            root,
            output / "geometry_driver.log",
        )
    print(f"fuse/score scan{scan}: {output}", flush=True)
    _command(
        [
            sys.executable,
            str(root / "scripts" / "evaluate_dtu.py"),
            "--root",
            str(root / "inputs" / "dtu"),
            "--scan",
            str(scan),
            "--render-dir",
            str(renders),
            "--normalization",
            str(output / "normalization.json"),
            "--output-mesh",
            str(output / "dtu_tsdf.ply"),
            "--output-json",
            str(output / "dtu_metrics.json"),
        ],
        root,
        output / "geometry_driver.log",
    )


def _summarize(
    output_root: Path,
    runs: dict[int, Path],
    config_sha256: str,
    source_hash: str,
    scan: int,
) -> Path:
    rows = []
    for seed, run in sorted(runs.items()):
        rgb = json.loads((run / "metrics_final.json").read_text())["aggregate"]
        geometry = json.loads((run / "dtu_metrics.json").read_text())
        timing = json.loads((run / "run_summary.json").read_text())
        rows.append({"seed": seed, "run": str(run), "rgb": rgb, "geometry": geometry, "training": timing})
    aggregate = {}
    for group, keys in {
        "rgb": ("psnr", "ssim", "lpips_alex"),
        "geometry": ("accuracy_mm", "completeness_mm", "overall_mm", "fscore_1mm", "fscore_2mm"),
        "training": ("end_to_end_wall_seconds", "peak_mlx_allocator_bytes", "final_gaussian_count"),
    }.items():
        aggregate[group] = {}
        for key in keys:
            values = np.asarray([row[group][key] for row in rows], dtype=float)
            aggregate[group][key] = {"mean": float(values.mean()), "std": float(values.std())}
    payload = {
        "dataset": f"dtu_scan{scan}",
        "config_sha256": config_sha256,
        "source_sha256": source_hash,
        "runs": rows,
        "aggregate": aggregate,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "baseline_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--scan", type=int, choices=(1, 6), default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--reuse-seed-1", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_bytes = (root / "configs" / "train_colmap3d.yaml").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    source_hash = source_sha256(root)
    experiment_hash = experiment_sha256(config_sha256, source_hash)
    requested = args.output_root or Path("outputs/baselines") / f"dtu_scan{args.scan}_{experiment_hash[:12]}"
    output_root = (root / requested).resolve() if not requested.is_absolute() else requested
    scene = args.scene.resolve()
    runs = {}
    for seed in args.seeds:
        reusable = args.reuse_seed_1.resolve() if seed == 1 and args.reuse_seed_1 is not None else None
        if reusable is not None and _complete_training(reusable, seed, config_sha256, source_hash):
            output = reusable
        else:
            output = output_root / f"seed_{seed}"
            _train(root, scene, output, seed, config_sha256, source_hash)
        _geometry(root, output, args.scan)
        runs[seed] = output
    print(_summarize(output_root, runs, config_sha256, source_hash, args.scan))


if __name__ == "__main__":
    main()
