#!/usr/bin/env python3
"""Convert TorchVision AlexNet + LPIPS-Alex weights to an MLX NPZ asset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alexnet", type=Path, required=True)
    parser.add_argument("--lpips", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    alex = torch.load(args.alexnet, map_location="cpu", weights_only=True)
    linear = torch.load(args.lpips, map_location="cpu", weights_only=True)
    weights = {}
    for layer in (0, 3, 6, 8, 10):
        weight = alex[f"features.{layer}.weight"].detach().numpy()
        weights[f"conv{layer}.weight"] = weight.transpose(0, 2, 3, 1).astype(np.float32)
        weights[f"conv{layer}.bias"] = alex[f"features.{layer}.bias"].detach().numpy().astype(np.float32)
    for layer in range(5):
        weights[f"lin{layer}"] = linear[f"lin{layer}.model.1.weight"].detach().numpy()[0, :, 0, 0].astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **weights)


if __name__ == "__main__":
    main()
