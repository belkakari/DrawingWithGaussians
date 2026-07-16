"""Exact 2DGS per-tile append and tile-local depth sort.

This prototypes the useful part of msplat's bin architecture without its AABB
false positives or fixed, silently truncating 2048-entry tile lists. One Metal
thread evaluates the exact rational-quadratic predicate once per candidate
tile, atomically appends a full-depth key to that tile, and records the exact
per-Gaussian count. A second kernel bitonic-sorts each tile's actual members.

The tile capacity is an explicit power-of-two bucket. ``tile_overflow`` is a
GPU scalar consumed by the trainer's transactional commit/rollback policy.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

from .rendering2dgs_fused import (
    _BBOX_TG,
    _HEADER_2DGS,
    _as_camera_batch,
    _core,
    _num_tiles,
    _pad,
    _squeeze_aux,
    _take_sorted_features,
)

_KERNEL_CACHE: dict[int, tuple[Any, Any, Any]] = {}

_SCATTER_PREALLOC_SRC = r"""
    uint gid = thread_position_in_grid.x;
    uint N = (uint)sizes[0];
    uint W = (uint)sizes[1];
    uint H = (uint)sizes[2];
    uint n_per_view = (uint)sizes[3];
    uint tile_capacity = (uint)sizes[4];
    uint num_tiles = (uint)sizes[5];
    if (gid >= N) return;
    uint TW = (W + TILE - 1) / TILE;
    uint TH = (H + TILE - 1) / TILE;
    uint tiles_per_view = TW * TH;
    uint cam = gid / n_per_view;
    float mx = means2d[2 * gid];
    float my = means2d[2 * gid + 1];
    float rx = radii[2 * gid];
    float ry = radii[2 * gid + 1];
    float opacity = opac[gid];
    uint written = 0u;
    if (!(rx > 0.0f && ry > 0.0f && opacity > ALPHA_THRESHOLD)) {
        gaussian_counts[gid] = 0;
        return;
    }

    float m00 = ray_transforms[9 * gid], m01 = ray_transforms[9 * gid + 1], m02 = ray_transforms[9 * gid + 2];
    float m10 = ray_transforms[9 * gid + 3], m11 = ray_transforms[9 * gid + 4], m12 = ray_transforms[9 * gid + 5];
    float m20 = ray_transforms[9 * gid + 6], m21 = ray_transforms[9 * gid + 7], m22 = ray_transforms[9 * gid + 8];
    float max_sigma = metal::log(255.0f * opacity);
    float tux = m22 * m11 - m21 * m12;
    float tuy = m02 * m21 - m01 * m22;
    float tuc = m01 * m12 - m02 * m11;
    float tvx = m20 * m12 - m22 * m10;
    float tvy = m00 * m22 - m02 * m20;
    float tvc = m02 * m10 - m00 * m12;
    float twx = m21 * m10 - m20 * m11;
    float twy = m01 * m20 - m00 * m21;
    float twc = m00 * m11 - m01 * m10;
    float r2 = 2.0f * max_sigma;
    float qxx = tux * tux + tvx * tvx - r2 * twx * twx;
    float qxy = 2.0f * (tux * tuy + tvx * tvy - r2 * twx * twy);
    float qyy = tuy * tuy + tvy * tvy - r2 * twy * twy;
    float qx = 2.0f * (tux * tuc + tvx * tvc - r2 * twx * twc);
    float qy = 2.0f * (tuy * tuc + tvy * tvc - r2 * twy * twc);
    float q0 = tuc * tuc + tvc * tvc - r2 * twc * twc;
    int tx0 = (int)metal::clamp(metal::floor((mx - rx) / (float)TILE), 0.0f, (float)(TW - 1));
    int tx1 = (int)metal::clamp(metal::floor((mx + rx) / (float)TILE), 0.0f, (float)(TW - 1));
    int ty0 = (int)metal::clamp(metal::floor((my - ry) / (float)TILE), 0.0f, (float)(TH - 1));
    int ty1 = (int)metal::clamp(metal::floor((my + ry) / (float)TILE), 0.0f, (float)(TH - 1));

    // Classified scanline span for the bounded rational-quadratic cases. The
    // Hessian is [[A,B],[B,C]]. Positive definiteness gives one bounded
    // ellipse; weak determinants and a tw=0 crossing retain the full exact
    // rectangle path. The span is only a candidate reduction: every retained
    // tile still runs tile_contributes_2dgs_precomputed below.
    float A = qxx, Bq = 0.5f * qxy, C = qyy;
    float hscale = metal::max(metal::max(metal::abs(A), metal::abs(Bq)), metal::max(metal::abs(C), 1e-20f));
    float det = A * C - Bq * Bq;
    bool span_safe = A > 1e-6f * hscale && C > 1e-6f * hscale && det > 1e-6f * hscale * hscale;
    float ellipse_cx = (Bq * qy - C * qx) / (2.0f * det);
    float ellipse_cy = (Bq * qx - A * qy) / (2.0f * det);
    float ellipse_level = 1e-4f - eval_quad(qxx, qxy, qyy, qx, qy, q0, ellipse_cx, ellipse_cy);
    float ellipse_x_extent = metal::sqrt(metal::max(ellipse_level * C / det, 0.0f));
    float ellipse_y_extent = metal::sqrt(metal::max(ellipse_level * A / det, 0.0f));

    float bbox_x0 = (float)(tx0 * (int)TILE) + 0.5f;
    float bbox_x1 = (float)metal::min((uint)(tx1 + 1) * TILE, W) - 0.5f;
    float bbox_y0 = (float)(ty0 * (int)TILE) + 0.5f;
    float bbox_y1 = (float)metal::min((uint)(ty1 + 1) * TILE, H) - 0.5f;
    float tw00 = twx * bbox_x0 + twy * bbox_y0 + twc;
    float tw01 = twx * bbox_x0 + twy * bbox_y1 + twc;
    float tw10 = twx * bbox_x1 + twy * bbox_y0 + twc;
    float tw11 = twx * bbox_x1 + twy * bbox_y1 + twc;
    float tw_min = metal::min(metal::min(tw00, tw01), metal::min(tw10, tw11));
    float tw_max = metal::max(metal::max(tw00, tw01), metal::max(tw10, tw11));
    span_safe = span_safe && (tw_min > 1e-6f || tw_max < -1e-6f);
    bool ellipse_active = span_safe && ellipse_level >= 0.0f;

    uint depth_bits = as_type<uint>(depths[gid]);
    device atomic_uint* counters = reinterpret_cast<device atomic_uint*>(storage + num_tiles * tile_capacity);
    device atomic_uint* overflow = counters + num_tiles;
    for (int ty = ty0; ty <= ty1; ++ty) {
        int row_tx0 = tx0;
        int row_tx1 = tx1;
        if (span_safe) {
            float y0 = (float)(ty * (int)TILE) + 0.5f;
            float y1 = (float)metal::min((uint)(ty + 1) * TILE, H) - 0.5f;
            bool have_span = false;
            float span_min = 0.0f, span_max = 0.0f;

            // Screen-space circle: distance to the row slab gives its exact
            // projected x interval (including the predicate's 1e-5 slack).
            float circle_y = metal::clamp(my, y0, y1);
            float circle_dy = circle_y - my;
            float circle_rem = max_sigma + 1e-5f - circle_dy * circle_dy;
            if (circle_rem >= 0.0f) {
                float circle_x = metal::sqrt(circle_rem);
                span_min = mx - circle_x;
                span_max = mx + circle_x;
                have_span = true;
            }

            if (ellipse_active) {
                float ellipse_y0 = ellipse_cy - ellipse_y_extent;
                float ellipse_y1 = ellipse_cy + ellipse_y_extent;
                if (y1 >= ellipse_y0 && y0 <= ellipse_y1) {
                    float ya = metal::clamp(y0, ellipse_y0, ellipse_y1);
                    float yb = metal::clamp(y1, ellipse_y0, ellipse_y1);
                    float dya = ya - ellipse_cy;
                    float dyb = yb - ellipse_cy;
                    float sa = metal::sqrt(metal::max(ellipse_level * A - det * dya * dya, 0.0f));
                    float sb = metal::sqrt(metal::max(ellipse_level * A - det * dyb * dyb, 0.0f));
                    float xa0 = ellipse_cx + (-Bq * dya - sa) / A;
                    float xa1 = ellipse_cx + (-Bq * dya + sa) / A;
                    float xb0 = ellipse_cx + (-Bq * dyb - sb) / A;
                    float xb1 = ellipse_cx + (-Bq * dyb + sb) / A;
                    float ellipse_min = metal::min(xa0, xb0);
                    float ellipse_max = metal::max(xa1, xb1);
                    float xmin_arg_y = ellipse_cy + Bq * ellipse_x_extent / C;
                    float xmax_arg_y = ellipse_cy - Bq * ellipse_x_extent / C;
                    if (xmin_arg_y >= y0 && xmin_arg_y <= y1) ellipse_min = ellipse_cx - ellipse_x_extent;
                    if (xmax_arg_y >= y0 && xmax_arg_y <= y1) ellipse_max = ellipse_cx + ellipse_x_extent;
                    if (have_span) {
                        span_min = metal::min(span_min, ellipse_min);
                        span_max = metal::max(span_max, ellipse_max);
                    } else {
                        span_min = ellipse_min;
                        span_max = ellipse_max;
                        have_span = true;
                    }
                }
            }
            if (!have_span) continue;
            // Expand by one tile before the exact predicate so fp32 boundary
            // rounding can only create extra candidates, never false negatives.
            row_tx0 = metal::max(tx0, (int)metal::floor(span_min / (float)TILE) - 1);
            row_tx1 = metal::min(tx1, (int)metal::floor(span_max / (float)TILE) + 1);
        }
        for (int tx = row_tx0; tx <= row_tx1; ++tx) {
            if (!tile_contributes_2dgs_precomputed(
                mx, my, max_sigma, qxx, qxy, qyy, qx, qy, q0, (uint)tx, (uint)ty, W, H
            )) continue;
            uint tile = cam * tiles_per_view + (uint)ty * TW + (uint)tx;
            uint pos = atomic_fetch_add_explicit(&counters[tile], 1u, metal::memory_order_relaxed);
            if (pos < tile_capacity) {
                storage[(ulong)tile * tile_capacity + pos] = ((ulong)depth_bits << 32) | (ulong)gid;
            } else {
                atomic_store_explicit(overflow, 1u, metal::memory_order_relaxed);
            }
            ++written;
        }
    }
    gaussian_counts[gid] = (int)written;
"""

_EXTRACT_COUNTERS_SRC = r"""
    uint idx = thread_position_in_grid.x;
    uint num_tiles = (uint)sizes[0];
    uint tile_capacity = (uint)sizes[1];
    if (idx > num_tiles) return;
    uint counter_idx = idx;
    ulong word = storage[(ulong)num_tiles * tile_capacity + counter_idx / 2u];
    uint value = (counter_idx & 1u) ? (uint)(word >> 32) : (uint)word;
    if (idx < num_tiles) tile_counts[idx] = (int)value;
    else overflow[0] = value;
"""

_BITONIC_SORT_SRC_TEMPLATE = r"""
    uint tile = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    uint num_tiles = (uint)sizes[0];
    uint global_capacity = (uint)sizes[1];
    if (tile >= num_tiles) return;
    uint count = metal::min((uint)tile_counts[tile], (uint)TILE_CAPACITY);
    uint start = (uint)bounds[tile];
    threadgroup ulong data[TILE_CAPACITY];
    uint n = 1u;
    while (n < count) n <<= 1u;
    ulong base = (ulong)tile * (ulong)TILE_CAPACITY;
    for (uint i = tid; i < n; i += 256u) {
        data[i] = i < count ? storage[base + i] : 0xfffffffffffffffful;
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (uint k = 2u; k <= n; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            for (uint i = tid; i < (n >> 1u); i += 256u) {
                uint pos = 2u * i - (i & (j - 1u));
                uint partner = pos ^ j;
                bool ascending = (pos & k) == 0u;
                ulong a = data[pos];
                ulong b = data[partner];
                if ((a > b) == ascending) {
                    data[pos] = b;
                    data[partner] = a;
                }
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
    }
    for (uint i = tid; i < count; i += 256u) {
        uint output_idx = start + i;
        if (output_idx < global_capacity) bin_ids[output_idx] = (uint)data[i];
    }
"""


def _kernels(tile_capacity: int):
    if tile_capacity not in {256, 512, 1024, 2048}:
        raise ValueError("tile_capacity must be one of 256, 512, 1024, or 2048")
    cached = _KERNEL_CACHE.get(tile_capacity)
    if cached is not None:
        return cached
    scatter = mx.fast.metal_kernel(
        name=f"gauss2dgs_scatter_tile_local_{tile_capacity}",
        input_names=["means2d", "ray_transforms", "opac", "radii", "depths", "sizes"],
        output_names=["storage", "gaussian_counts"],
        header=_HEADER_2DGS,
        source=_SCATTER_PREALLOC_SRC,
    )
    extract = mx.fast.metal_kernel(
        name=f"gauss2dgs_extract_tile_counts_{tile_capacity}",
        input_names=["storage", "sizes"],
        output_names=["tile_counts", "overflow"],
        source=_EXTRACT_COUNTERS_SRC,
    )
    sort = mx.fast.metal_kernel(
        name=f"gauss2dgs_bitonic_tile_local_{tile_capacity}",
        input_names=["storage", "tile_counts", "bounds", "sizes"],
        output_names=["bin_ids"],
        header=f"constant uint TILE_CAPACITY = {tile_capacity};\n",
        source=_BITONIC_SORT_SRC_TEMPLATE,
    )
    _KERNEL_CACHE[tile_capacity] = scatter, extract, sort
    return scatter, extract, sort


def build_bins_tile_local(
    means2d,
    ray_transforms,
    opacities,
    depths,
    radii,
    width: int,
    height: int,
    *,
    capacity: int,
    tile_capacity: int,
):
    """Build exact bins with one predicate pass and per-tile depth sorting."""
    means2d, ray_transforms, opacities, depths, radii = _as_camera_batch(
        means2d, ray_transforms, opacities, depths, radii
    )
    ncams, n = means2d.shape[:2]
    flat_n = ncams * n
    num_tiles = ncams * _num_tiles(width, height)
    capacity = max(1, int(capacity))
    scatter, extract, sort = _kernels(tile_capacity)
    # Counters and the overflow flag occupy packed uint32 words after the key
    # storage. Padding to uint64 preserves atomic alignment.
    counter_words = (num_tiles + 2) // 2
    storage_size = num_tiles * tile_capacity + counter_words
    sizes = mx.array([flat_n, width, height, n, tile_capacity, num_tiles], dtype=mx.int32)
    storage, counts = scatter(  # type: ignore[operator]
        inputs=[
            means2d.reshape(flat_n, 2),
            ray_transforms.reshape(flat_n, 3, 3),
            opacities.reshape(flat_n),
            radii.reshape(flat_n, 2),
            depths.reshape(flat_n),
            sizes,
        ],
        grid=(_pad(flat_n, _BBOX_TG), 1, 1),
        threadgroup=(_BBOX_TG, 1, 1),
        output_shapes=[(storage_size,), (flat_n,)],
        output_dtypes=[mx.uint64, mx.int32],
        init_value=0,
    )
    counter_sizes = mx.array([num_tiles, tile_capacity], dtype=mx.int32)
    tile_counts, overflow = extract(  # type: ignore[operator]
        inputs=[storage, counter_sizes],
        grid=(_pad(num_tiles + 1, _BBOX_TG), 1, 1),
        threadgroup=(_BBOX_TG, 1, 1),
        output_shapes=[(num_tiles,), (1,)],
        output_dtypes=[mx.int32, mx.uint32],
    )
    cumulative = mx.minimum(mx.cumsum(tile_counts), capacity)
    bounds = mx.concatenate([mx.zeros((1,), dtype=mx.int32), cumulative]).astype(mx.int32)
    sort_sizes = mx.array([num_tiles, capacity], dtype=mx.int32)
    bin_ids = sort(  # type: ignore[operator]
        inputs=[storage, tile_counts, bounds, sort_sizes],
        grid=(num_tiles * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(capacity,)],
        output_dtypes=[mx.uint32],
        init_value=0xFFFFFFFF,
    )[0]
    return tuple(
        mx.stop_gradient(value) for value in (bin_ids, bounds, counts.reshape(ncams, n), overflow[0], tile_counts)
    )


def rasterize2dgs_tile_local(
    means2d,
    ray_transforms,
    opacities,
    colors,
    background,
    depths,
    radii,
    height: int,
    width: int,
    *,
    capacity: int,
    tile_capacity: int,
    densify_sink=None,
    normals=None,
    return_aux=False,
    return_counts=False,
    return_status=False,
):
    """Rasterizer using exact tile append/sort with depth-sorted records.

    Keeping the record array in global depth order is load-bearing for dense
    gradient parity: changing the atomic-gradient address layout while keeping
    identical per-tile lists measurably changes cancellation in the VJP.
    """
    batched = means2d.ndim == 3
    if not batched:
        means2d, ray_transforms, depths, radii = (
            means2d[None],
            ray_transforms[None],
            depths[None],
            radii[None],
        )
    ncams, n = means2d.shape[:2]
    order = mx.argsort(depths, axis=-1)
    means2d = mx.take_along_axis(means2d, mx.broadcast_to(order[..., None], means2d.shape), axis=1)
    ray_transforms = mx.take_along_axis(
        ray_transforms, mx.broadcast_to(order[..., None, None], ray_transforms.shape), axis=1
    )
    depths = mx.take_along_axis(depths, order, axis=1)
    radii = mx.take_along_axis(radii, mx.broadcast_to(order[..., None], radii.shape), axis=1)
    if opacities.ndim != 1 or opacities.shape != (n,):
        raise ValueError("opacities must have shape N")
    opac = mx.take(opacities, order)
    opac = mx.where((depths > 0.01) & (depths < 1e10), opac, 0.0)
    colors = _take_sorted_features(colors, order)
    if normals is None:
        normals = mx.zeros((n, 3), dtype=means2d.dtype)
    normals = _take_sorted_features(normals, order)
    if densify_sink is None:
        densify_sink = mx.zeros((ncams, n, 2), dtype=means2d.dtype)
    elif not batched and densify_sink.shape == (n, 2):
        densify_sink = densify_sink[None]
    elif densify_sink.shape != (ncams, n, 2):
        raise ValueError(f"densify_sink must have shape {(ncams, n, 2)}")
    densify_sink = _take_sorted_features(densify_sink, order)

    bin_ids, bounds, builder_counts, tile_overflow, tile_counts = build_bins_tile_local(
        means2d,
        ray_transforms,
        opac,
        depths,
        radii,
        width,
        height,
        capacity=capacity,
        tile_capacity=tile_capacity,
    )
    inverse_order = mx.argsort(order, axis=-1)
    counts = mx.take_along_axis(builder_counts, inverse_order, axis=1)
    flat_n = ncams * n
    acc, acc_depth, acc_normals, distort, median, tfinal, _, _ = _core(height, width, ncams)(
        means2d.reshape(flat_n, 2),
        ray_transforms.reshape(flat_n, 3, 3),
        opac.reshape(flat_n),
        colors.reshape(flat_n, 3),
        depths.reshape(flat_n),
        normals.reshape(flat_n, 3),
        bin_ids,
        bounds,
        densify_sink.reshape(flat_n, 2),
    )
    out = (acc + tfinal[:, None] * background[None, :]).reshape(ncams, height, width, 3)
    status = {"tile_overflow": tile_overflow, "tile_counts": tile_counts}
    extras = []
    if return_counts:
        extras.append(counts if batched else counts[0])
    if return_status:
        extras.append(status)
    if not return_aux:
        rgb = out if batched else out[0]
        return (rgb, *extras) if extras else rgb

    alpha = (1.0 - tfinal).reshape(ncams, height, width, 1)
    depth_accum = acc_depth.reshape(ncams, height, width, 1)
    aux = {
        "alpha": alpha,
        "depth_accum": depth_accum,
        "depth": depth_accum / mx.maximum(alpha, 1e-10),
        "normals": acc_normals.reshape(ncams, height, width, 3),
        "distortion": distort.reshape(ncams, height, width, 1),
        "median_depth": median.reshape(ncams, height, width, 1),
    }
    result = (out, _squeeze_aux(aux, batched)) if batched else (out[0], _squeeze_aux(aux, batched))
    return (*result, *extras)
