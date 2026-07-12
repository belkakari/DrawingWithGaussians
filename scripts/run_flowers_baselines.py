#!/usr/bin/env python3
"""Run and aggregate the frozen three-seed Flowers 2DGS/3DGS baselines."""

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


def _complete_run(path: Path, mode: str, seed: int, config_sha256: str, source_hash: str) -> bool:
    metrics = path / "metrics_final.json"
    manifest = path / "run_manifest.json"
    if not metrics.is_file() or not manifest.is_file():
        return False
    config = json.loads(manifest.read_text())["resolved_config"]
    return (
        config["gaussians"]["mode"] == mode
        and int(config["optim"]["seed"]) == seed
        and matching_run_stamp(
            path,
            mode=mode,
            seed=seed,
            config_sha256=config_sha256,
            source_sha256=source_hash,
        )
    )


def _run(
    root: Path,
    output: Path,
    mode: str,
    seed: int,
    force: bool,
    config_sha256: str,
    source_hash: str,
) -> None:
    if _complete_run(output, mode, seed, config_sha256, source_hash) and not force:
        print(f"reuse {mode} seed {seed}: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "train_colmap3d.py"),
        "--config-name",
        "train_colmap3d.yaml",
        f"data.dir={root / 'inputs' / 'flowers'}",
        f"gaussians.mode={mode}",
        f"optim.seed={seed}",
        "train.eval_save_renders=false",
        "train.eval_lpips=true",
        "train.save_video=false",
        "train.save_depth=false",
        "train.save_dtu_renders=false",
        f"hydra.run.dir={output}",
    ]
    if mode == "3dgs":
        command.extend(("optim.loss.normal_weight=0", "optim.loss.distortion_weight=0"))
    environment = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "HYDRA_FULL_ERROR": "1"}
    print(f"run {mode} seed {seed}: {output}", flush=True)
    with (output / "driver.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    write_run_stamp(
        output,
        mode=mode,
        seed=seed,
        config_sha256=config_sha256,
        source_sha256=source_hash,
    )


def _summarize(root: Path, modes: list[str], seeds: list[int], config_sha256: str, source_hash: str) -> Path:
    summary: dict[str, object] = {
        "dataset": "flowers",
        "config_sha256": config_sha256,
        "source_sha256": source_hash,
        "seeds": seeds,
        "modes": {},
    }
    for mode in modes:
        rows = []
        for seed in seeds:
            run = root / mode / f"seed_{seed}"
            metrics = json.loads((run / "metrics_final.json").read_text())["aggregate"]
            timing = json.loads((run / "run_summary.json").read_text())
            rows.append(
                {
                    "seed": seed,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "lpips_alex": metrics["lpips_alex"],
                    "raw_underflow_fraction": metrics["raw_underflow_fraction"],
                    "raw_overflow_fraction": metrics["raw_overflow_fraction"],
                    **timing,
                }
            )
        aggregates = {}
        for key in (
            "psnr",
            "ssim",
            "lpips_alex",
            "raw_underflow_fraction",
            "raw_overflow_fraction",
            "end_to_end_wall_seconds",
            "peak_mlx_allocator_bytes",
            "final_gaussian_count",
        ):
            values = np.asarray([row[key] for row in rows], dtype=float)
            aggregates[key] = {"mean": float(values.mean()), "std": float(values.std())}
        summary["modes"][mode] = {"runs": rows, "aggregate": aggregates}  # type: ignore[index]
    path = root / "baseline_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--modes", nargs="+", choices=("2dgs", "3dgs"), default=["2dgs", "3dgs"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_bytes = (root / "configs" / "train_colmap3d.yaml").read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    source_hash = source_sha256(root)
    experiment_hash = experiment_sha256(config_sha256, source_hash)
    requested_root = args.output_root or Path("outputs/baselines") / f"flowers_{experiment_hash[:12]}"
    output_root = (root / requested_root).resolve() if not requested_root.is_absolute() else requested_root
    for mode in args.modes:
        for seed in args.seeds:
            _run(
                root,
                output_root / mode / f"seed_{seed}",
                mode,
                seed,
                args.force,
                config_sha256,
                source_hash,
            )
    print(_summarize(output_root, args.modes, args.seeds, config_sha256, source_hash))


if __name__ == "__main__":
    main()
