"""P2a gate: cost of building per-tile intersection bins in MLX ops.

Usage:
    uv run python scripts/bench_binning.py [--size 128] [--spread] [--pads 4 8 16]

Builds the (tile, gaussian) intersection lists the tiled rasterizer would
consume — per-gaussian tile bbox from the bounding radii, fixed per-gaussian
slot budget `pad`, INVALID-padded keys, one global `mx.argsort`, per-tile
counts/offsets via scatter-add — and reports, per (N, pad):

    avg tiles/gaussian, max tiles/gaussian, fallback count (bbox area > pad),
    bin-build ms, and the current fused fwd / fwd+bwd ms for context.

Decision rule (see plan): proceed with binning integration only if the
build cost is clearly below the streaming waste it would remove.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drawingwithgaussians.rendering3d import NEAR_PLANE, project_gaussians  # noqa: E402
from drawingwithgaussians.rendering3d_fused import _TILE, _bounding_radii  # noqa: E402
from scripts.bench_render import scene_3d, timeit  # noqa: E402

INVALID = mx.array(0xFFFFFFFF, dtype=mx.uint32)


def build_bins(means2d, radii, width, height, pad):
    """Return (sorted_ids, offsets, stats). Keys are tile * N + rank so one
    argsort yields depth-ordered per-tile runs (inputs are depth-sorted)."""
    n = means2d.shape[0]
    tw, th = (width + _TILE - 1) // _TILE, (height + _TILE - 1) // _TILE
    ntiles = tw * th

    mxs, mys = means2d[:, 0], means2d[:, 1]
    rx, ry = radii[:, 0], radii[:, 1]
    valid = rx > 0
    tx0 = mx.clip(mx.floor((mxs - rx) / _TILE), 0, tw - 1).astype(mx.int32)
    tx1 = mx.clip(mx.floor((mxs + rx) / _TILE), 0, tw - 1).astype(mx.int32)
    ty0 = mx.clip(mx.floor((mys - ry) / _TILE), 0, th - 1).astype(mx.int32)
    ty1 = mx.clip(mx.floor((mys + ry) / _TILE), 0, th - 1).astype(mx.int32)
    bw = tx1 - tx0 + 1
    bh = ty1 - ty0 + 1
    area = mx.where(valid, bw * bh, 0)
    binned = valid & (area <= pad)  # oversized ones go to the fallback stream

    k = mx.arange(pad, dtype=mx.int32)[None, :]  # (1, pad)
    slot_ok = binned[:, None] & (k < area[:, None])
    tx = tx0[:, None] + k % mx.maximum(bw, 1)[:, None]
    ty = ty0[:, None] + k // mx.maximum(bw, 1)[:, None]
    tile = (ty * tw + tx).astype(mx.uint32)
    rank = mx.arange(n, dtype=mx.uint32)[:, None]
    keys = mx.where(slot_ok, tile * n + rank, INVALID).reshape(-1)

    order = mx.argsort(keys)
    sorted_keys = mx.take(keys, order)
    sorted_ids = (sorted_keys % n).astype(mx.uint32)  # gaussian ids, per-tile depth-ordered
    sorted_tiles = mx.minimum(sorted_keys // n, ntiles).astype(mx.uint32)

    counts = mx.zeros((ntiles + 1,), dtype=mx.int32).at[sorted_tiles].add(1)
    offsets = mx.cumsum(counts[:-1]) - counts[:-1]  # exclusive prefix over real tiles

    stats = (area, binned, valid)
    return sorted_ids, offsets, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--ns", type=int, nargs="+", default=[5000, 20000, 50000])
    ap.add_argument("--pads", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--spread", action="store_true")
    args = ap.parse_args()
    size = args.size
    rng = np.random.default_rng(0)

    focal = 0.5 * size / math.tan(0.25 * math.pi)
    scene_tag = "spread" if args.spread else "clustered"
    print(f"image {size}x{size} ({scene_tag}), tile {_TILE}, MLX {mx.__version__}")
    print(f"{'N':>7s} {'pad':>4s} {'avg t/g':>8s} {'max t/g':>8s} {'fallback':>9s} {'build ms':>9s}")

    for n in args.ns:
        (means3d, log_scales, quats, opac_raw, col_raw), target, K, viewmat = scene_3d(
            n, size, rng, spread=args.spread
        )
        means2d, conics, depths = project_gaussians(means3d, log_scales, quats, viewmat, K, size, size)
        order = mx.argsort(depths)
        m = mx.take(means2d, order, axis=0)
        con = mx.take(conics, order, axis=0)
        opac = mx.where(mx.take(depths, order) > NEAR_PLANE, mx.sigmoid(mx.take(opac_raw, order)), 0.0)
        radii = _bounding_radii(con, opac)
        mx.eval(m, radii)

        for pad in args.pads:
            _, _, (area, binned, valid) = build_bins(m, radii, size, size, pad)
            mx.eval(area, binned, valid)
            nvalid = max(int(mx.sum(valid)), 1)
            avg_t = float(mx.sum(area)) / nvalid
            max_t = int(mx.max(area))
            fallback = int(mx.sum(valid & ~binned))

            fn = mx.compile(lambda mm, rr, p=pad: build_bins(mm, rr, size, size, p)[:2])

            def run():
                ids, offs = fn(m, radii)
                mx.eval(ids, offs)

            ms = timeit(run, args.iters)
            print(f"{n:7d} {pad:4d} {avg_t:8.2f} {max_t:8d} {fallback:9d} {ms:9.3f}")


if __name__ == "__main__":
    main()
