"""Fused RGB-only 2DGS surfel rasterizer.

Phase-1 2DGS support: projection/ray transforms come from ``rendering2dgs``;
this kernel keeps the existing 3DGS tile-bin/compositing structure but swaps
the conic power for gsplat's ray-splat intersection power.
"""

import math
from typing import Any

import mlx.core as mx

from .rendering3d_fused import _HEADER, _TILE, _build_bins_padded, _pad

_TG_N = _TILE * _TILE

_FORWARD_SRC = """
    uint lid = thread_index_in_threadgroup;
    uint3 tg3 = threadgroup_position_in_grid;
    uint N = (uint)sizes[0];
    uint W = (uint)sizes[1];
    uint H = (uint)sizes[2];
    uint px_i = tg3.x * TILE + (lid % TILE);
    uint py_i = tg3.y * TILE + (lid / TILE);
    bool active = (px_i < W) && (py_i < H);
    float px = (float)px_i + 0.5f;
    float py = (float)py_i + 0.5f;

    threadgroup float sh[TG_N * 15];
    threadgroup uint shid[TG_N];
    threadgroup metal::atomic_uint ndone;
    if (lid == 0) atomic_store_explicit(&ndone, 0u, metal::memory_order_relaxed);

    uint TW = (W + TILE - 1) / TILE;
    uint TH = (H + TILE - 1) / TILE;
    uint tile = tg3.z * TW * TH + tg3.y * TW + tg3.x;
    uint lo = (uint)bounds[tile];
    uint hi = (uint)bounds[tile + 1];

    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    uint contribs = 0;
    bool done = !active;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (done) atomic_fetch_add_explicit(&ndone, 1u, metal::memory_order_relaxed);

    uint nchunks = (hi - lo + TG_N - 1) / TG_N;
    for (uint c = 0; c < nchunks; ++c) {
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&ndone, metal::memory_order_relaxed) == TG_N) break;
        uint idx = lo + c * TG_N + lid;
        uint total = metal::min(hi - lo - c * TG_N, TG_N);
        if (idx < hi) {
            uint i = bin_ids[idx];
            sh[15 * lid + 0] = means2d[2 * i];
            sh[15 * lid + 1] = means2d[2 * i + 1];
            for (uint k = 0; k < 9; ++k) sh[15 * lid + 2 + k] = ray_transforms[9 * i + k];
            sh[15 * lid + 11] = opac[i];
            sh[15 * lid + 12] = colors[3 * i];
            sh[15 * lid + 13] = colors[3 * i + 1];
            sh[15 * lid + 14] = colors[3 * i + 2];
            shid[lid] = i;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (!done) {
            for (uint j = 0; j < total; ++j) {
                float mx = sh[15 * j], my = sh[15 * j + 1];
                float m00 = sh[15 * j + 2], m01 = sh[15 * j + 3], m02 = sh[15 * j + 4];
                float m10 = sh[15 * j + 5], m11 = sh[15 * j + 6], m12 = sh[15 * j + 7];
                float m20 = sh[15 * j + 8], m21 = sh[15 * j + 9], m22 = sh[15 * j + 10];
                float hu0 = -m00 + m20 * px, hu1 = -m01 + m21 * px, hu2 = -m02 + m22 * px;
                float hv0 = -m10 + m20 * py, hv1 = -m11 + m21 * py, hv2 = -m12 + m22 * py;
                float tu = hu1 * hv2 - hu2 * hv1;
                float tv = hu2 * hv0 - hu0 * hv2;
                float tw = hu0 * hv1 - hu1 * hv0;
                if (metal::abs(tw) <= 1e-8f) continue;
                float u = tu / tw, v = tv / tw;
                float dx = px - mx, dy = py - my;
                float sigma3 = u * u + v * v;
                float sigma2 = 2.0f * (dx * dx + dy * dy);
                float sigma = 0.5f * metal::min(sigma3, sigma2);
                float alpha = metal::min(MAX_ALPHA, sh[15 * j + 11] * metal::precise::exp(-sigma));
                if (alpha < ALPHA_THRESHOLD) continue;
                float next_T = T * (1.0f - alpha);
                if (next_T <= TRANSMITTANCE_THRESHOLD) {
                    done = true;
                    atomic_fetch_add_explicit(&ndone, 1u, metal::memory_order_relaxed);
                    break;
                }
                float fac = alpha * T;
                r += fac * sh[15 * j + 12];
                g += fac * sh[15 * j + 13];
                b += fac * sh[15 * j + 14];
                T = next_T;
                contribs = shid[j] + 1;
            }
        }
    }
    if (active) {
        uint p = tg3.z * W * H + py_i * W + px_i;
        acc[3 * p] = r;
        acc[3 * p + 1] = g;
        acc[3 * p + 2] = b;
        tfinal[p] = T;
        last[p] = contribs;
    }
"""

_BACKWARD_SRC = """
    uint lid = thread_index_in_threadgroup;
    uint3 tg3 = threadgroup_position_in_grid;
    uint N = (uint)sizes[0];
    uint W = (uint)sizes[1];
    uint H = (uint)sizes[2];
    uint px_i = tg3.x * TILE + (lid % TILE);
    uint py_i = tg3.y * TILE + (lid / TILE);
    bool active = (px_i < W) && (py_i < H);
    float px = (float)px_i + 0.5f;
    float py = (float)py_i + 0.5f;

    threadgroup float sh[TG_N * 15];
    threadgroup uint shid[TG_N];
    threadgroup metal::atomic_uint tile_last;
    if (lid == 0) atomic_store_explicit(&tile_last, 0u, metal::memory_order_relaxed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint TW = (W + TILE - 1) / TILE;
    uint TH = (H + TILE - 1) / TILE;
    uint tile = tg3.z * TW * TH + tg3.y * TW + tg3.x;
    uint lo = (uint)bounds[tile];
    uint hi = (uint)bounds[tile + 1];

    uint p = tg3.z * W * H + py_i * W + px_i;
    uint mylast = 0;
    float T = 0.0f, Tfin = 0.0f;
    float vr0 = 0.0f, vr1 = 0.0f, vr2 = 0.0f, cT = 0.0f;
    if (active) {
        mylast = last[p];
        Tfin = tfinal[p];
        T = Tfin;
        vr0 = dacc[3 * p];
        vr1 = dacc[3 * p + 1];
        vr2 = dacc[3 * p + 2];
        cT = dt[p];
    }
    atomic_fetch_max_explicit(&tile_last, mylast, metal::memory_order_relaxed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint maxlast = atomic_load_explicit(&tile_last, metal::memory_order_relaxed);
    if (maxlast == 0 || lo == hi) return;

    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;
    uint nchunks = (hi - lo + TG_N - 1) / TG_N;
    for (uint cc = nchunks; cc-- > 0;) {
        if (bin_ids[lo + cc * TG_N] >= maxlast) continue;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        uint idx = lo + cc * TG_N + lid;
        uint total = metal::min(hi - lo - cc * TG_N, TG_N);
        if (idx < hi) {
            uint i = bin_ids[idx];
            sh[15 * lid + 0] = means2d[2 * i];
            sh[15 * lid + 1] = means2d[2 * i + 1];
            for (uint k = 0; k < 9; ++k) sh[15 * lid + 2 + k] = ray_transforms[9 * i + k];
            sh[15 * lid + 11] = opac[i];
            sh[15 * lid + 12] = colors[3 * i];
            sh[15 * lid + 13] = colors[3 * i + 1];
            sh[15 * lid + 14] = colors[3 * i + 2];
            shid[lid] = i;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint jj = total; jj-- > 0;) {
            uint gid = shid[jj];
            float mx = sh[15 * jj], my = sh[15 * jj + 1];
            float m00 = sh[15 * jj + 2], m01 = sh[15 * jj + 3], m02 = sh[15 * jj + 4];
            float m10 = sh[15 * jj + 5], m11 = sh[15 * jj + 6], m12 = sh[15 * jj + 7];
            float m20 = sh[15 * jj + 8], m21 = sh[15 * jj + 9], m22 = sh[15 * jj + 10];
            float hu0 = -m00 + m20 * px, hu1 = -m01 + m21 * px, hu2 = -m02 + m22 * px;
            float hv0 = -m10 + m20 * py, hv1 = -m11 + m21 * py, hv2 = -m12 + m22 * py;
            float tu = hu1 * hv2 - hu2 * hv1;
            float tv = hu2 * hv0 - hu0 * hv2;
            float tw = hu0 * hv1 - hu1 * hv0;
            bool valid = metal::abs(tw) > 1e-8f;
            float invw = valid ? 1.0f / tw : 0.0f;
            float u = tu * invw, v = tv * invw;
            float dx = px - mx, dy = py - my;
            float sigma3 = u * u + v * v;
            float sigma2 = 2.0f * (dx * dx + dy * dy);
            bool use3d = sigma3 <= sigma2;
            float sigma = 0.5f * (use3d ? sigma3 : sigma2);
            float vis = metal::precise::exp(-sigma);
            float alpha_raw = sh[15 * jj + 11] * vis;
            float alpha = metal::min(MAX_ALPHA, alpha_raw);
            bool contrib = active && valid && (gid < mylast) && (alpha >= ALPHA_THRESHOLD);
            float ra = contrib ? 1.0f / (1.0f - alpha) : 1.0f;
            if (contrib) T *= ra;
            float fac = contrib ? alpha * T : 0.0f;

            float c0 = sh[15 * jj + 12], c1 = sh[15 * jj + 13], c2 = sh[15 * jj + 14];
            float v_alpha = 0.0f;
            if (contrib) {
                v_alpha = (c0 * T - S0 * ra) * vr0 + (c1 * T - S1 * ra) * vr1 + (c2 * T - S2 * ra) * vr2;
                v_alpha += cT * (-Tfin * ra);
            }
            bool grad_gate = contrib && (alpha_raw <= MAX_ALPHA);
            float g_opac = grad_gate ? vis * v_alpha : 0.0f;

            if (metal::simd_any(contrib)) {
                float v_sigma = grad_gate ? -alpha_raw * v_alpha : 0.0f;
                float gmx = 0.0f, gmy = 0.0f;
                float gm0 = 0.0f, gm1 = 0.0f, gm2 = 0.0f, gm3 = 0.0f, gm4 = 0.0f;
                float gm5 = 0.0f, gm6 = 0.0f, gm7 = 0.0f, gm8 = 0.0f;
                if (use3d) {
                    float gu = v_sigma * u;
                    float gv = v_sigma * v;
                    float ga = gu * invw;
                    float gb = gv * invw;
                    float gc = -(gu * tu + gv * tv) * invw * invw;
                    float dhu0 = -gb * hv2 + gc * hv1;
                    float dhu1 = ga * hv2 - gc * hv0;
                    float dhu2 = -ga * hv1 + gb * hv0;
                    float dhv0 = gb * hu2 - gc * hu1;
                    float dhv1 = -ga * hu2 + gc * hu0;
                    float dhv2 = ga * hu1 - gb * hu0;
                    gm0 = -dhu0; gm1 = -dhu1; gm2 = -dhu2;
                    gm3 = -dhv0; gm4 = -dhv1; gm5 = -dhv2;
                    gm6 = px * dhu0 + py * dhv0;
                    gm7 = px * dhu1 + py * dhv1;
                    gm8 = px * dhu2 + py * dhv2;
                } else {
                    gmx = -2.0f * dx * v_sigma;
                    gmy = -2.0f * dy * v_sigma;
                }

                float t0 = fac * vr0, t1 = fac * vr1, t2 = fac * vr2, t3 = g_opac;
                if (simd_reduce_add4(t0, t1, t2, t3, lid)) {
                    atomic_fetch_add_explicit(&dcolors[3 * gid], t0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dcolors[3 * gid + 1], t1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dcolors[3 * gid + 2], t2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dopac[gid], t3, metal::memory_order_relaxed);
                }
                float w0 = gmx, w1 = gmy, w2 = metal::abs(gmx), w3 = metal::abs(gmy);
                if (simd_reduce_add4(w0, w1, w2, w3, lid)) {
                    atomic_fetch_add_explicit(&dmeans2d[2 * gid], w0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d[2 * gid + 1], w1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid], w2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid + 1], w3, metal::memory_order_relaxed);
                }
                float a0 = gm0, a1 = gm1, a2 = gm2, a3 = gm3;
                if (simd_reduce_add4(a0, a1, a2, a3, lid)) {
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid], a0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 1], a1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 2], a2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 3], a3, metal::memory_order_relaxed);
                }
                float a4 = gm4, a5 = gm5, a6 = gm6, a7 = gm7;
                if (simd_reduce_add4(a4, a5, a6, a7, lid)) {
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 4], a4, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 5], a5, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 6], a6, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 7], a7, metal::memory_order_relaxed);
                }
                float a8 = gm8, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
                if (simd_reduce_add4(a8, z1, z2, z3, lid)) {
                    atomic_fetch_add_explicit(&dray_transforms[9 * gid + 8], a8, metal::memory_order_relaxed);
                }
            }

            if (contrib) {
                S0 += c0 * fac;
                S1 += c1 * fac;
                S2 += c2 * fac;
            }
        }
    }
"""

_COUNT_BBOX_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = (uint)sizes[0];
    uint W = (uint)sizes[1];
    uint H = (uint)sizes[2];
    if (gid >= N) return;
    uint TW = (W + TILE - 1) / TILE;
    uint TH = (H + TILE - 1) / TILE;
    float mx = means2d[2 * gid];
    float my = means2d[2 * gid + 1];
    float rx = radii[2 * gid];
    float ry = radii[2 * gid + 1];
    int count = 0;
    if (rx > 0.0f && ry > 0.0f) {
        int tx0 = (int)metal::clamp(metal::floor((mx - rx) / (float)TILE), 0.0f, (float)(TW - 1));
        int tx1 = (int)metal::clamp(metal::floor((mx + rx) / (float)TILE), 0.0f, (float)(TW - 1));
        int ty0 = (int)metal::clamp(metal::floor((my - ry) / (float)TILE), 0.0f, (float)(TH - 1));
        int ty1 = (int)metal::clamp(metal::floor((my + ry) / (float)TILE), 0.0f, (float)(TH - 1));
        count = (tx1 - tx0 + 1) * (ty1 - ty0 + 1);
    }
    counts[gid] = count;
"""

_SCATTER_BBOX_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = (uint)sizes[0];
    uint W = (uint)sizes[1];
    uint H = (uint)sizes[2];
    uint n_per_view = (uint)sizes[3];
    uint capacity = (uint)sizes[4];
    if (gid >= N) return;
    uint TW = (W + TILE - 1) / TILE;
    uint TH = (H + TILE - 1) / TILE;
    uint ntiles = TW * TH;
    uint cam = gid / n_per_view;
    float mx = means2d[2 * gid];
    float my = means2d[2 * gid + 1];
    float rx = radii[2 * gid];
    float ry = radii[2 * gid + 1];
    if (!(rx > 0.0f && ry > 0.0f)) return;
    int tx0 = (int)metal::clamp(metal::floor((mx - rx) / (float)TILE), 0.0f, (float)(TW - 1));
    int tx1 = (int)metal::clamp(metal::floor((mx + rx) / (float)TILE), 0.0f, (float)(TW - 1));
    int ty0 = (int)metal::clamp(metal::floor((my - ry) / (float)TILE), 0.0f, (float)(TH - 1));
    int ty1 = (int)metal::clamp(metal::floor((my + ry) / (float)TILE), 0.0f, (float)(TH - 1));
    uint base = (uint)offsets[gid];
    uint written = 0;
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            uint pos = base + written;
            if (pos < capacity) {
                uint tile = cam * ntiles + (uint)ty * TW + (uint)tx;
                keys[pos] = tile * N + gid;
            }
            ++written;
        }
    }
"""

_k_fwd = mx.fast.metal_kernel(
    name="gauss2dgs_forward",
    input_names=[
        "means2d",
        "ray_transforms",
        "opac",
        "colors",
        "bin_ids",
        "bounds",
        "sizes",
    ],
    output_names=["acc", "tfinal", "last"],
    header=_HEADER,
    source=_FORWARD_SRC,
)
_k_bwd = mx.fast.metal_kernel(
    name="gauss2dgs_backward",
    input_names=[
        "means2d",
        "ray_transforms",
        "opac",
        "colors",
        "bin_ids",
        "bounds",
        "tfinal",
        "last",
        "dacc",
        "dt",
        "sizes",
    ],
    output_names=["dmeans2d", "dray_transforms", "dopac", "dcolors", "dmeans2d_abs"],
    header=_HEADER,
    source=_BACKWARD_SRC,
    atomic_outputs=True,
)
_k_count_bbox = mx.fast.metal_kernel(
    name="gauss2dgs_count_bbox_bins",
    input_names=["means2d", "radii", "sizes"],
    output_names=["counts"],
    header=_HEADER,
    source=_COUNT_BBOX_SRC,
)
_k_scatter_bbox = mx.fast.metal_kernel(
    name="gauss2dgs_scatter_bbox_bins",
    input_names=["means2d", "radii", "offsets", "sizes"],
    output_names=["keys"],
    header=_HEADER,
    source=_SCATTER_BBOX_SRC,
)

_BBOX_TG = 256
_INVALID_KEY = mx.array(0xFFFFFFFF, dtype=mx.uint32)

_CORE_CACHE: dict[tuple[int, int, int], Any] = {}


def _core(height, width, ncams=1) -> Any:
    key = (height, width, ncams)
    cached = _CORE_CACHE.get(key)
    if cached is not None:
        return cached
    num_pixels = ncams * height * width
    grid = (_pad(width, _TILE), _pad(height, _TILE), ncams)
    tg = (_TILE, _TILE, 1)

    @mx.custom_function
    def core(means2d, ray_transforms, opacities, colors, bin_ids, bounds, absgrad_sink):
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        acc, tfinal, last = _k_fwd(  # type: ignore[operator]
            inputs=[means2d, ray_transforms, opacities, colors, bin_ids, bounds, sizes],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(num_pixels, 3), (num_pixels,), (num_pixels,)],
            output_dtypes=[mx.float32, mx.float32, mx.uint32],
        )
        return acc, tfinal, last

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        means2d, ray_transforms, opacities, colors, bin_ids, bounds, absgrad_sink = (
            primals
        )
        dacc, dt = cotangents[0], cotangents[1]
        _, tfinal, last = outputs
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        dmeans2d, dray, dopac, dcolors, dabs = _k_bwd(  # type: ignore[operator]
            inputs=[
                means2d,
                ray_transforms,
                opacities,
                colors,
                bin_ids,
                bounds,
                tfinal,
                last,
                dacc,
                dt,
                sizes,
            ],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(n, 2), (n, 3, 3), (n,), (n, 3), (n, 2)],
            output_dtypes=[mx.float32] * 5,
            init_value=0,
        )
        return (
            dmeans2d,
            dray,
            dopac,
            dcolors,
            mx.zeros_like(bin_ids),
            mx.zeros_like(bounds),
            dabs + mx.zeros_like(absgrad_sink),
        )

    _CORE_CACHE[key] = core
    return core


def _as_camera_batch(*arrays):
    if arrays[0].ndim == 2:
        return tuple(a[None] if a.ndim in {1, 2, 3} else a for a in arrays)
    return arrays


def _num_tiles(width, height):
    return ((width + _TILE - 1) // _TILE) * ((height + _TILE - 1) // _TILE)


def _count_bbox_intersections(means2d, radii, width, height):
    means2d, radii = _as_camera_batch(means2d, radii)
    ncams, n = means2d.shape[0], means2d.shape[1]
    flat_n = ncams * n
    sizes = mx.array([flat_n, width, height], dtype=mx.int32)
    counts = _k_count_bbox(  # type: ignore[operator]
        inputs=[means2d.reshape(flat_n, 2), radii.reshape(flat_n, 2), sizes],
        grid=(_pad(flat_n, _BBOX_TG), 1, 1),
        threadgroup=(_BBOX_TG, 1, 1),
        output_shapes=[(flat_n,)],
        output_dtypes=[mx.int32],
    )[0]
    return mx.stop_gradient(counts.reshape(ncams, n))


def estimate_bin_capacity_2dgs(
    means2d, radii, width, height, *, margin=2.0, min_per_gaussian=16
):
    counts = _count_bbox_intersections(means2d, radii, width, height)
    mx.eval(counts)
    n = counts.shape[-1]
    total = int(mx.sum(counts)) if counts.size > 0 else 0
    min_capacity = int(min_per_gaussian) * int(n) * int(counts.shape[0])
    return max(
        1,
        min(
            max(min_capacity, math.ceil(total * float(margin))),
            counts.shape[0] * n * _num_tiles(width, height),
        ),
    )


def _build_bins_compact(means2d, radii, width, height, capacity):
    means2d, radii = _as_camera_batch(means2d, radii)
    ncams, n = means2d.shape[0], means2d.shape[1]
    flat_n = ncams * n
    ntiles = _num_tiles(width, height)
    capacity = int(max(1, capacity))
    counts = _count_bbox_intersections(means2d, radii, width, height).reshape(flat_n)
    offsets = mx.cumsum(counts) - counts
    sizes = mx.array([flat_n, width, height, n, capacity], dtype=mx.int32)
    keys = _k_scatter_bbox(  # type: ignore[operator]
        inputs=[means2d.reshape(flat_n, 2), radii.reshape(flat_n, 2), offsets, sizes],
        grid=(_pad(flat_n, _BBOX_TG), 1, 1),
        threadgroup=(_BBOX_TG, 1, 1),
        output_shapes=[(capacity,)],
        output_dtypes=[mx.uint32],
        init_value=0xFFFFFFFF,
    )[0]
    sorted_keys = mx.sort(keys)
    bin_ids = (sorted_keys % flat_n).astype(mx.uint32)
    sorted_tiles = mx.minimum(sorted_keys // flat_n, ncams * ntiles).astype(mx.uint32)
    per_tile = mx.zeros((ncams * ntiles + 1,), dtype=mx.int32).at[sorted_tiles].add(1)
    cum = mx.cumsum(per_tile[:-1])
    bounds = mx.concatenate([mx.zeros((1,), dtype=mx.int32), cum]).astype(mx.int32)
    return (
        mx.stop_gradient(bin_ids),
        mx.stop_gradient(bounds),
        mx.stop_gradient(counts.reshape(ncams, n)),
    )


def _build_bins(means2d, radii, width, height, pad=None, capacity=None):
    if capacity is not None:
        return _build_bins_compact(means2d, radii, width, height, capacity)
    if pad is not None:
        means2d_b = means2d[None] if means2d.ndim == 2 else means2d
        capacity = means2d_b.shape[0] * means2d_b.shape[1] * int(pad)
        return _build_bins_compact(means2d, radii, width, height, capacity)
    return _build_bins_padded(means2d, radii, width, height, None)


def rasterize2dgs_fused(
    means2d,
    ray_transforms,
    opacities,
    colors,
    background,
    depths,
    radii,
    height,
    width,
    absgrad_sink=None,
    bin_pad=None,
    bin_capacity=None,
):
    """RGB-only 2DGS fused rasterizer, single camera or camera batch."""
    batched = means2d.ndim == 3
    if not batched:
        means2d, ray_transforms, depths, radii = (
            means2d[None],
            ray_transforms[None],
            depths[None],
            radii[None],
        )
    ncams, n = means2d.shape[0], means2d.shape[1]
    order = mx.argsort(depths, axis=-1)
    m = mx.take_along_axis(
        means2d, mx.broadcast_to(order[..., None], means2d.shape), axis=1
    )
    ray = mx.take_along_axis(
        ray_transforms,
        mx.broadcast_to(order[..., None, None], ray_transforms.shape),
        axis=1,
    )
    dep = mx.take_along_axis(depths, order, axis=-1)
    rad = mx.take_along_axis(
        radii, mx.broadcast_to(order[..., None], radii.shape), axis=1
    )
    opac = mx.take(opacities, order)
    col = mx.take(colors, order, axis=0)
    opac = mx.where((dep > 0.01) & (dep < 1e10), opac, 0.0)
    bin_ids, bounds, _ = _build_bins(
        m, rad, width, height, pad=bin_pad, capacity=bin_capacity
    )

    if absgrad_sink is None:
        absgrad_sink = mx.zeros((n, 2), dtype=means2d.dtype)
    abs_sink = mx.take(absgrad_sink, order, axis=0)
    flat_n = ncams * n
    acc, tfinal, _ = _core(height, width, ncams)(
        m.reshape(flat_n, 2),
        ray.reshape(flat_n, 3, 3),
        opac.reshape(flat_n),
        col.reshape(flat_n, 3),
        bin_ids,
        bounds,
        abs_sink.reshape(flat_n, 2),
    )
    out = acc + tfinal[:, None] * background[None, :]
    out = out.reshape(ncams, height, width, 3)
    return out if batched else out[0]
