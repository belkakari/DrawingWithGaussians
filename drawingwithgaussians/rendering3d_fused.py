"""Fused Metal-kernel 3D gaussian splatting rasterizer (gsplat-style, tiled).

Alpha compositing of depth-sorted projected gaussians, same math as
:func:`drawingwithgaussians.rendering3d.rasterize3d_dense` but with
hand-written tiled kernels following gsplat/msplat's structure:

* Each 16x16 threadgroup owns one image tile; its 256 threads each own one
  pixel *and* cooperatively stream the depth-sorted gaussians through
  threadgroup memory in chunks of 256: every thread loads one gaussian,
  tests its 3.33-sigma bounding box against the tile, survivors are
  compacted with a simdgroup prefix sum, then all threads composite the
  shared list. This keeps compositing order identical to the untiled kernel
  while cutting per-pixel work from O(N) to O(N / 256 + hits).
* The 3.33-sigma box provably contains every pixel with
  ``alpha >= ALPHA_THRESHOLD`` (alpha = opac * exp(-sigma) >= 1/255 implies
  sigma <= ln(255 * 0.99), i.e. |dx| <= 3.326 * sqrt(cov_xx)), and the
  untiled kernel already skipped sub-threshold contributions — so tiling is
  numerically lossless, not an approximation.
* Forward: gsplat's exact thresholds (alpha clamp 0.99, 1/255 skip,
  exclusive 1e-4 transmittance early termination, whole-tile early exit).
  Outputs accumulated color, final transmittance (background compositing
  happens outside, in autodiff-land) and per-pixel contributor count.
* Backward: back-to-front replay per pixel with the same tile streaming,
  iterating chunks in reverse starting from the tile's deepest contributor;
  per-gaussian gradients are scattered with ``atomic_fetch_add``
  (compositing order couples every pixel to every gaussian, so a
  gaussian-parallel backward would be O(N^2 P)). The cotangent on the final
  transmittance carries the background path.

The depth sort (``mx.argsort`` + ``mx.take``), the projection, and the
per-gaussian bounding radii (from the conics, gradient-stopped) stay
outside the custom function in regular MLX ops. Atomic adds make the
backward non-deterministic at the fp32-rounding level across runs (same as
gsplat).
"""

import mlx.core as mx

from .rendering3d import (
    FAR_PLANE,
    NEAR_PLANE,
)

_TILE = 16
_TG_N = _TILE * _TILE  # threads per group == gaussians per chunk == pixels per tile

_HEADER = """
constant float MAX_ALPHA = 0.99f;
constant float ALPHA_THRESHOLD = 1.0f / 255.0f;
constant float TRANSMITTANCE_THRESHOLD = 1e-4f;
constant uint TILE = 16;
constant uint TG_N = 256;

// Compact the per-thread predicate across the 256-thread group; returns this
// thread's write slot in `pre_out` and the group total. Uses a simdgroup
// prefix sum plus an 8-entry cross-simd scan in threadgroup memory.
inline uint tg_compact(uint pred, uint lid, threadgroup uint *simd_totals,
                       thread uint &pre_out) {
    uint lane = lid & 31u;
    uint sgid = lid >> 5u;
    uint pre = metal::simd_prefix_exclusive_sum(pred);
    if (lane == 31u) simd_totals[sgid] = pre + pred;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint base = 0;
    for (uint k = 0; k < sgid; ++k) base += simd_totals[k];
    uint total = 0;
    for (uint k = 0; k < 8; ++k) total += simd_totals[k];
    pre_out = base + pre;
    return total;
}
"""

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
    float x0 = (float)(tg3.x * TILE), x1 = x0 + (float)TILE;
    float y0 = (float)(tg3.y * TILE), y1 = y0 + (float)TILE;

    threadgroup float sh[TG_N * 9];
    threadgroup uint shid[TG_N];
    threadgroup uint simd_totals[8];
    threadgroup metal::atomic_uint ndone;
    if (lid == 0) atomic_store_explicit(&ndone, 0u, metal::memory_order_relaxed);

    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    uint contribs = 0;
    bool done = !active;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    if (done) atomic_fetch_add_explicit(&ndone, 1u, metal::memory_order_relaxed);

    uint nchunks = (N + TG_N - 1) / TG_N;
    for (uint c = 0; c < nchunks; ++c) {
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (atomic_load_explicit(&ndone, metal::memory_order_relaxed) == TG_N) break;
        uint i = c * TG_N + lid;
        uint pred = 0;
        float m0 = 0.0f, m1 = 0.0f;
        if (i < N) {
            m0 = means2d[2 * i];
            m1 = means2d[2 * i + 1];
            float rx = radii[2 * i], ry = radii[2 * i + 1];
            pred = (rx > 0.0f && m0 + rx >= x0 && m0 - rx <= x1 && m1 + ry >= y0 && m1 - ry <= y1) ? 1u : 0u;
        }
        uint slot;
        uint total = tg_compact(pred, lid, simd_totals, slot);
        if (pred) {
            sh[9 * slot + 0] = m0;
            sh[9 * slot + 1] = m1;
            sh[9 * slot + 2] = conics[3 * i];
            sh[9 * slot + 3] = conics[3 * i + 1];
            sh[9 * slot + 4] = conics[3 * i + 2];
            sh[9 * slot + 5] = opac[i];
            sh[9 * slot + 6] = colors[3 * i];
            sh[9 * slot + 7] = colors[3 * i + 1];
            sh[9 * slot + 8] = colors[3 * i + 2];
            shid[slot] = i;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (!done) {
            for (uint j = 0; j < total; ++j) {
                float dx = px - sh[9 * j];
                float dy = py - sh[9 * j + 1];
                float A = sh[9 * j + 2], B = sh[9 * j + 3], C = sh[9 * j + 4];
                float sigma = 0.5f * (A * dx * dx + C * dy * dy) + B * dx * dy;
                if (sigma < 0.0f) continue;
                float alpha = metal::min(MAX_ALPHA, sh[9 * j + 5] * metal::precise::exp(-sigma));
                if (alpha < ALPHA_THRESHOLD) continue;
                float next_T = T * (1.0f - alpha);
                if (next_T <= TRANSMITTANCE_THRESHOLD) {  // exclusive, like gsplat
                    done = true;
                    atomic_fetch_add_explicit(&ndone, 1u, metal::memory_order_relaxed);
                    break;
                }
                float fac = alpha * T;
                r += fac * sh[9 * j + 6];
                g += fac * sh[9 * j + 7];
                b += fac * sh[9 * j + 8];
                T = next_T;
                contribs = shid[j] + 1;
            }
        }
    }
    if (active) {
        uint p = py_i * W + px_i;
        acc[3 * p] = r;
        acc[3 * p + 1] = g;
        acc[3 * p + 2] = b;
        tfinal[p] = T;
        last[p] = contribs;
    }
"""

# Back-to-front replay with the same tile streaming, starting from the
# chunk containing the tile's deepest contributor. `dacc` is the cotangent
# on `acc`, `dt` the cotangent on `tfinal` (background path).
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
    float x0 = (float)(tg3.x * TILE), x1 = x0 + (float)TILE;
    float y0 = (float)(tg3.y * TILE), y1 = y0 + (float)TILE;

    threadgroup float sh[TG_N * 9];
    threadgroup uint shid[TG_N];
    threadgroup uint simd_totals[8];
    threadgroup metal::atomic_uint tile_last;
    if (lid == 0) atomic_store_explicit(&tile_last, 0u, metal::memory_order_relaxed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint p = py_i * W + px_i;
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
    if (maxlast == 0) return;  // uniform across the group: no contributor in this tile

    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;
    uint nchunks = (maxlast + TG_N - 1) / TG_N;
    for (uint cc = nchunks; cc-- > 0;) {
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        uint i = cc * TG_N + lid;
        uint pred = 0;
        float m0 = 0.0f, m1 = 0.0f;
        if (i < N) {
            m0 = means2d[2 * i];
            m1 = means2d[2 * i + 1];
            float rx = radii[2 * i], ry = radii[2 * i + 1];
            pred = (rx > 0.0f && m0 + rx >= x0 && m0 - rx <= x1 && m1 + ry >= y0 && m1 - ry <= y1) ? 1u : 0u;
        }
        uint slot;
        uint total = tg_compact(pred, lid, simd_totals, slot);
        if (pred) {
            sh[9 * slot + 0] = m0;
            sh[9 * slot + 1] = m1;
            sh[9 * slot + 2] = conics[3 * i];
            sh[9 * slot + 3] = conics[3 * i + 1];
            sh[9 * slot + 4] = conics[3 * i + 2];
            sh[9 * slot + 5] = opac[i];
            sh[9 * slot + 6] = colors[3 * i];
            sh[9 * slot + 7] = colors[3 * i + 1];
            sh[9 * slot + 8] = colors[3 * i + 2];
            shid[slot] = i;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint jj = total; jj-- > 0;) {
            uint gid = shid[jj];
            if (gid >= mylast) continue;  // this pixel never composited it
            float dx = px - sh[9 * jj];
            float dy = py - sh[9 * jj + 1];
            float A = sh[9 * jj + 2], B = sh[9 * jj + 3], C = sh[9 * jj + 4];
            float sigma = 0.5f * (A * dx * dx + C * dy * dy) + B * dx * dy;
            if (sigma < 0.0f) continue;
            float vis = metal::precise::exp(-sigma);
            float alpha_raw = sh[9 * jj + 5] * vis;
            float alpha = metal::min(MAX_ALPHA, alpha_raw);
            if (alpha < ALPHA_THRESHOLD) continue;
            float ra = 1.0f / (1.0f - alpha);
            T *= ra;  // transmittance in front of this gaussian
            float fac = alpha * T;

            float c0 = sh[9 * jj + 6], c1 = sh[9 * jj + 7], c2 = sh[9 * jj + 8];
            atomic_fetch_add_explicit(&dcolors[3 * gid], fac * vr0, metal::memory_order_relaxed);
            atomic_fetch_add_explicit(&dcolors[3 * gid + 1], fac * vr1, metal::memory_order_relaxed);
            atomic_fetch_add_explicit(&dcolors[3 * gid + 2], fac * vr2, metal::memory_order_relaxed);

            float v_alpha =
                (c0 * T - S0 * ra) * vr0 + (c1 * T - S1 * ra) * vr1 + (c2 * T - S2 * ra) * vr2;
            v_alpha += cT * (-Tfin * ra);

            if (alpha_raw <= MAX_ALPHA) {  // min() gate: no gradient when clamped
                float v_sigma = -alpha_raw * v_alpha;
                atomic_fetch_add_explicit(&dconics[3 * gid], 0.5f * v_sigma * dx * dx, metal::memory_order_relaxed);
                atomic_fetch_add_explicit(&dconics[3 * gid + 1], v_sigma * dx * dy, metal::memory_order_relaxed);
                atomic_fetch_add_explicit(&dconics[3 * gid + 2], 0.5f * v_sigma * dy * dy, metal::memory_order_relaxed);
                atomic_fetch_add_explicit(&dmeans2d[2 * gid], -v_sigma * (A * dx + B * dy), metal::memory_order_relaxed);
                atomic_fetch_add_explicit(&dmeans2d[2 * gid + 1], -v_sigma * (C * dy + B * dx), metal::memory_order_relaxed);
                atomic_fetch_add_explicit(&dopac[gid], vis * v_alpha, metal::memory_order_relaxed);
            }

            S0 += c0 * fac;
            S1 += c1 * fac;
            S2 += c2 * fac;
        }
    }
"""

_k_fwd3d = mx.fast.metal_kernel(
    name="gauss3d_tiled_forward",
    input_names=["means2d", "conics", "opac", "colors", "radii", "sizes"],
    output_names=["acc", "tfinal", "last"],
    header=_HEADER,
    source=_FORWARD_SRC,
)
_k_bwd3d = mx.fast.metal_kernel(
    name="gauss3d_tiled_backward",
    input_names=[
        "means2d",
        "conics",
        "opac",
        "colors",
        "radii",
        "tfinal",
        "last",
        "dacc",
        "dt",
        "sizes",
    ],
    output_names=["dmeans2d", "dconics", "dopac", "dcolors"],
    header=_HEADER,
    source=_BACKWARD_SRC,
    atomic_outputs=True,
)


def _pad(n, m):
    return (n + m - 1) // m * m


_CORE_CACHE = {}


def _fused_core3d(height, width):
    key = (height, width)
    core = _CORE_CACHE.get(key)
    if core is not None:
        return core

    num_pixels = height * width
    grid = (_pad(width, _TILE), _pad(height, _TILE), 1)
    tg = (_TILE, _TILE, 1)

    @mx.custom_function
    def core(means2d, conics, opacities, colors, radii):
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        acc, tfinal, last = _k_fwd3d(
            inputs=[means2d, conics, opacities, colors, radii, sizes],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(num_pixels, 3), (num_pixels,), (num_pixels,)],
            output_dtypes=[mx.float32, mx.float32, mx.uint32],
        )
        return acc, tfinal, last

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        means2d, conics, opacities, colors, radii = primals
        dacc, dt = cotangents[0], cotangents[1]  # no cotangent on `last`
        _, tfinal, last = outputs
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        dmeans2d, dconics, dopac, dcolors = _k_bwd3d(
            inputs=[
                means2d,
                conics,
                opacities,
                colors,
                radii,
                tfinal,
                last,
                dacc,
                dt,
                sizes,
            ],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(n, 2), (n, 3), (n,), (n, 3)],
            output_dtypes=[mx.float32] * 4,
            init_value=0,
        )
        # radii gate visibility only (their boundary sits below the 1/255
        # contribution threshold); like gsplat, no gradient flows through them.
        return dmeans2d, dconics, dopac, dcolors, mx.zeros_like(radii)

    _CORE_CACHE[key] = core
    return core


def _bounding_radii(conics, opacities):
    """Per-gaussian 3.33-sigma axis-aligned half-extents in pixels, from the
    conic (inverse 2D covariance): cov_xx = C / det, cov_yy = A / det. The
    marginal std bounds the alpha >= 1/255 ellipse exactly (see module
    docstring). Gradient-stopped; culled gaussians (opacity 0) get radius -1
    so the tile test drops them."""
    a, b, c = conics[:, 0], conics[:, 1], conics[:, 2]
    det = mx.maximum(a * c - b * b, 1e-12)
    rx = 3.33 * mx.sqrt(mx.maximum(c / det, 0.0))
    ry = 3.33 * mx.sqrt(mx.maximum(a / det, 0.0))
    radii = mx.stack([rx, ry], axis=1)
    radii = mx.where(opacities[:, None] > 0.0, radii, -1.0)
    return mx.stop_gradient(radii)


def rasterize3d_fused(
    means2d, conics, opacities, colors, background, depths, height, width
):
    """Drop-in replacement for :func:`rendering3d.rasterize3d_dense`."""
    order = mx.argsort(depths)
    m = mx.take(means2d, order, axis=0)
    con = mx.take(conics, order, axis=0)
    opac = mx.take(opacities, order, axis=0)
    col = mx.take(colors, order, axis=0)
    dep = mx.take(depths, order, axis=0)
    opac = mx.where((dep > NEAR_PLANE) & (dep < FAR_PLANE), opac, 0.0)
    radii = _bounding_radii(con, opac)

    core = _fused_core3d(height, width)
    acc, tfinal, _ = core(m, con, opac, col, radii)
    out = acc + tfinal[:, None] * background[None, :]
    return out.reshape(height, width, 3)
