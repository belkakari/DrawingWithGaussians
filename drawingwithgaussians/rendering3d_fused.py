"""Fused Metal-kernel 3D gaussian splatting rasterizer (gsplat-style, tiled).

Alpha compositing of depth-sorted projected gaussians, same math as
:func:`drawingwithgaussians.rendering3d.rasterize3d_dense` but with
hand-written tiled kernels following gsplat/msplat's structure:

* Each 16x16 threadgroup owns one image tile; its 256 threads each own one
  pixel and cooperatively stream that tile's precomputed depth-ordered
  gaussian bin through threadgroup memory in chunks of 256. The bins are
  built in MLX ops from 3.33-sigma bounding boxes before entering the custom
  function, so the kernels avoid the old all-N tile scan and in-kernel
  compaction while preserving front-to-back compositing order.
* The 3.33-sigma box provably contains every pixel with
  ``alpha >= ALPHA_THRESHOLD`` (alpha = opac * exp(-sigma) >= 1/255 implies
  sigma <= ln(255 * 0.99), i.e. |dx| <= 3.326 * sqrt(cov_xx)), and the
  untiled kernel already skipped sub-threshold contributions — so tiling is
  numerically lossless, not an approximation.
* Forward: gsplat's exact thresholds (alpha clamp 0.99, 1/255 skip,
  exclusive 1e-4 transmittance early termination, whole-tile early exit).
  Outputs accumulated color, final transmittance (background compositing
  happens outside, in autodiff-land) and per-pixel contributor count.
* Backward: back-to-front replay per pixel over the same tile bins, iterating
  chunks in reverse from the tile's deepest contributor. Per-gaussian
  gradient scatters are simdgroup-reduced before ``atomic_fetch_add`` to cut
  atomic traffic by up to 32x without threadgroup barriers. The cotangent on
  final transmittance carries the background path.

The depth sort (``mx.argsort`` + ``mx.take``), the projection, and the
per-gaussian bounding radii and tile bins (gradient-stopped) stay outside
the custom function in regular MLX ops. Atomic adds make the
backward non-deterministic at the fp32-rounding level across runs (same as
gsplat).
"""

from typing import Any

import mlx.core as mx

from .rendering3d import (
    FAR_PLANE,
    NEAR_PLANE,
)

_TILE = 8
_TG_N = _TILE * _TILE  # threads per group == gaussians per chunk == pixels per tile

_HEADER = """
constant float MAX_ALPHA = 0.99f;
constant float ALPHA_THRESHOLD = 1.0f / 255.0f;
constant float TRANSMITTANCE_THRESHOLD = 1e-4f;
constant uint TILE = 8;
constant uint TG_N = 64;

// Reduce four per-thread values across the 32-lane simdgroup into lane 0
// (gsplat's warpSum pattern). Callers issue one atomic per component per
// simdgroup instead of per thread - a 32x cut in atomic traffic with ZERO
// barriers (simdgroups execute in lockstep, so simd_sum needs no
// synchronization). A first attempt reduced across the whole 256-thread
// group via threadgroup memory; the two barriers per gaussian made it
// SLOWER than raw atomics on Apple Silicon (see EXPERIMENTS.md).
inline bool simd_reduce_add4(thread float &v0, thread float &v1, thread float &v2, thread float &v3, uint lid) {
    v0 = metal::simd_sum(v0);
    v1 = metal::simd_sum(v1);
    v2 = metal::simd_sum(v2);
    v3 = metal::simd_sum(v3);
    return (lid & 31u) == 0;
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

    threadgroup float sh[TG_N * 9];
    threadgroup uint shid[TG_N];
    threadgroup metal::atomic_uint ndone;
    if (lid == 0) atomic_store_explicit(&ndone, 0u, metal::memory_order_relaxed);

    // This tile's precomputed bin: depth-ordered gaussian ids that overlap
    // it (built in MLX ops, see _build_bins). No bbox testing or compaction
    // in the kernel - every staged entry is a hit.
    uint TW = (W + TILE - 1) / TILE;
    uint tile = tg3.y * TW + tg3.x;
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
            sh[9 * lid + 0] = means2d[2 * i];
            sh[9 * lid + 1] = means2d[2 * i + 1];
            sh[9 * lid + 2] = conics[3 * i];
            sh[9 * lid + 3] = conics[3 * i + 1];
            sh[9 * lid + 4] = conics[3 * i + 2];
            sh[9 * lid + 5] = opac[i];
            sh[9 * lid + 6] = colors[3 * i];
            sh[9 * lid + 7] = colors[3 * i + 1];
            sh[9 * lid + 8] = colors[3 * i + 2];
            shid[lid] = i;
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

    threadgroup float sh[TG_N * 9];
    threadgroup uint shid[TG_N];
    threadgroup metal::atomic_uint tile_last;
    if (lid == 0) atomic_store_explicit(&tile_last, 0u, metal::memory_order_relaxed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint TW = (W + TILE - 1) / TILE;
    uint tile = tg3.y * TW + tg3.x;
    uint lo = (uint)bounds[tile];
    uint hi = (uint)bounds[tile + 1];

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
    if (maxlast == 0 || lo == hi) return;  // uniform: no contributor in this tile

    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;
    uint nchunks = (hi - lo + TG_N - 1) / TG_N;
    for (uint cc = nchunks; cc-- > 0;) {
        // Bin ids ascend within the tile, so chunks whose first id is past
        // the tile's deepest contributor have nothing to replay (uniform test).
        if (bin_ids[lo + cc * TG_N] >= maxlast) continue;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        uint idx = lo + cc * TG_N + lid;
        uint total = metal::min(hi - lo - cc * TG_N, TG_N);
        if (idx < hi) {
            uint i = bin_ids[idx];
            sh[9 * lid + 0] = means2d[2 * i];
            sh[9 * lid + 1] = means2d[2 * i + 1];
            sh[9 * lid + 2] = conics[3 * i];
            sh[9 * lid + 3] = conics[3 * i + 1];
            sh[9 * lid + 4] = conics[3 * i + 2];
            sh[9 * lid + 5] = opac[i];
            sh[9 * lid + 6] = colors[3 * i];
            sh[9 * lid + 7] = colors[3 * i + 1];
            sh[9 * lid + 8] = colors[3 * i + 2];
            shid[lid] = i;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        // Non-divergent replay: every skip condition becomes a predicate so
        // all 256 threads reach the reductions/barriers on every iteration
        // (gid and the shared data are uniform across the group).
        for (uint jj = total; jj-- > 0;) {
            uint gid = shid[jj];
            float dx = px - sh[9 * jj];
            float dy = py - sh[9 * jj + 1];
            float A = sh[9 * jj + 2], B = sh[9 * jj + 3], C = sh[9 * jj + 4];
            float sigma = 0.5f * (A * dx * dx + C * dy * dy) + B * dx * dy;
            float vis = metal::precise::exp(-sigma);
            float alpha_raw = sh[9 * jj + 5] * vis;
            float alpha = metal::min(MAX_ALPHA, alpha_raw);
            bool contrib = active && (gid < mylast) && (sigma >= 0.0f) && (alpha >= ALPHA_THRESHOLD);
            float ra = contrib ? 1.0f / (1.0f - alpha) : 1.0f;
            if (contrib) T *= ra;  // transmittance in front of this gaussian
            float fac = contrib ? alpha * T : 0.0f;

            float c0 = sh[9 * jj + 6], c1 = sh[9 * jj + 7], c2 = sh[9 * jj + 8];
            float v_alpha = 0.0f;
            if (contrib) {
                v_alpha = (c0 * T - S0 * ra) * vr0 + (c1 * T - S1 * ra) * vr1 + (c2 * T - S2 * ra) * vr2;
                v_alpha += cT * (-Tfin * ra);
            }
            // min() gate: no gradient through alpha when clamped
            bool grad_gate = contrib && (alpha_raw <= MAX_ALPHA);
            float g_opac = grad_gate ? vis * v_alpha : 0.0f;

            // All 11 gradient components simd-reduced (warpSum): one atomic
            // per component per simdgroup per gaussian; skipped entirely
            // when the whole simdgroup is inactive (grad_gate implies
            // contrib, so one vote covers both paths).
            if (metal::simd_any(contrib)) {
                float v_sigma = grad_gate ? -alpha_raw * v_alpha : 0.0f;
                float gmx = -v_sigma * (A * dx + B * dy);
                float gmy = -v_sigma * (C * dy + B * dx);
                float t0 = fac * vr0, t1 = fac * vr1, t2 = fac * vr2, t3 = g_opac;
                if (simd_reduce_add4(t0, t1, t2, t3, lid)) {
                    atomic_fetch_add_explicit(&dcolors[3 * gid], t0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dcolors[3 * gid + 1], t1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dcolors[3 * gid + 2], t2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dopac[gid], t3, metal::memory_order_relaxed);
                }
                float u0 = 0.5f * v_sigma * dx * dx, u1 = v_sigma * dx * dy, u2 = 0.5f * v_sigma * dy * dy;
                float u3 = gmx;
                if (simd_reduce_add4(u0, u1, u2, u3, lid)) {
                    atomic_fetch_add_explicit(&dconics[3 * gid], u0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dconics[3 * gid + 1], u1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dconics[3 * gid + 2], u2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d[2 * gid], u3, metal::memory_order_relaxed);
                }
                float w0 = gmy, w1 = metal::abs(gmx), w2 = metal::abs(gmy), w3 = 0.0f;
                if (simd_reduce_add4(w0, w1, w2, w3, lid)) {
                    atomic_fetch_add_explicit(&dmeans2d[2 * gid + 1], w0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid], w1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid + 1], w2, metal::memory_order_relaxed);
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

_k_fwd3d = mx.fast.metal_kernel(
    name="gauss3d_binned_forward",
    input_names=["means2d", "conics", "opac", "colors", "bin_ids", "bounds", "sizes"],
    output_names=["acc", "tfinal", "last"],
    header=_HEADER,
    source=_FORWARD_SRC,
)
_k_bwd3d = mx.fast.metal_kernel(
    name="gauss3d_binned_backward",
    input_names=[
        "means2d",
        "conics",
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
    output_names=["dmeans2d", "dconics", "dopac", "dcolors", "dmeans2d_abs"],
    header=_HEADER,
    source=_BACKWARD_SRC,
    atomic_outputs=True,
)

_BACKWARD_SRC_NO_ABS = _BACKWARD_SRC.replace(
    """                float w0 = gmy, w1 = metal::abs(gmx), w2 = metal::abs(gmy), w3 = 0.0f;\n                if (simd_reduce_add4(w0, w1, w2, w3, lid)) {\n                    atomic_fetch_add_explicit(&dmeans2d[2 * gid + 1], w0, metal::memory_order_relaxed);\n                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid], w1, metal::memory_order_relaxed);\n                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid + 1], w2, metal::memory_order_relaxed);\n                }\n""",
    """                float w0 = gmy, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;\n                if (simd_reduce_add4(w0, w1, w2, w3, lid)) {\n                    atomic_fetch_add_explicit(&dmeans2d[2 * gid + 1], w0, metal::memory_order_relaxed);\n                }\n""",
)
_k_bwd3d_no_abs = mx.fast.metal_kernel(
    name="gauss3d_binned_backward_no_abs",
    input_names=[
        "means2d",
        "conics",
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
    output_names=["dmeans2d", "dconics", "dopac", "dcolors"],
    header=_HEADER,
    source=_BACKWARD_SRC_NO_ABS,
    atomic_outputs=True,
)


def _pad(n, m):
    return (n + m - 1) // m * m


_CORE_CACHE: dict[tuple[int, int, bool], Any] = {}


def _fused_core3d(height, width, compute_absgrad=True) -> Any:
    key = (height, width, compute_absgrad)
    cached = _CORE_CACHE.get(key)  # type: ignore[assignment]
    if cached is not None:
        return cached

    num_pixels = height * width
    grid = (_pad(width, _TILE), _pad(height, _TILE), 1)
    tg = (_TILE, _TILE, 1)

    @mx.custom_function
    def core(means2d, conics, opacities, colors, bin_ids, bounds, absgrad_sink):
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        acc, tfinal, last = _k_fwd3d(  # type: ignore[operator]
            inputs=[means2d, conics, opacities, colors, bin_ids, bounds, sizes],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(num_pixels, 3), (num_pixels,), (num_pixels,)],
            output_dtypes=[mx.float32, mx.float32, mx.uint32],
        )
        return acc, tfinal, last

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        means2d, conics, opacities, colors, bin_ids, bounds, absgrad_sink = primals
        dacc, dt = cotangents[0], cotangents[1]  # no cotangent on `last`
        _, tfinal, last = outputs
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        if compute_absgrad:
            dmeans2d, dconics, dopac, dcolors, dmeans2d_abs = _k_bwd3d(  # type: ignore[operator]
                inputs=[
                    means2d,
                    conics,
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
                output_shapes=[(n, 2), (n, 3), (n,), (n, 3), (n, 2)],
                output_dtypes=[mx.float32] * 5,
                init_value=0,
            )
            dabs = dmeans2d_abs + mx.zeros_like(absgrad_sink)
        else:
            dmeans2d, dconics, dopac, dcolors = _k_bwd3d_no_abs(  # type: ignore[operator]
                inputs=[
                    means2d,
                    conics,
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
                output_shapes=[(n, 2), (n, 3), (n,), (n, 3)],
                output_dtypes=[mx.float32] * 4,
                init_value=0,
            )
            dabs = mx.zeros_like(absgrad_sink)
        # The bins gate visibility only (their boundary sits below the 1/255
        # contribution threshold); like gsplat's radii, no gradient flows
        # through them.
        return (
            dmeans2d,
            dconics,
            dopac,
            dcolors,
            mx.zeros_like(bin_ids),
            mx.zeros_like(bounds),
            dabs,
        )

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


_INVALID_KEY = mx.array(0xFFFFFFFF, dtype=mx.uint32)


def _build_bins(means2d, radii, width, height, pad):
    """Per-tile intersection lists, built in MLX ops (gradient-free).

    Each gaussian emits up to ``pad`` (tile, rank) keys covering its bbox;
    one global ``mx.argsort`` then yields depth-ordered runs per tile
    (inputs are depth-sorted, so rank order == depth order). Returns
    ``(bin_ids, bounds)``: ``bin_ids[bounds[t]:bounds[t+1]]`` are the
    gaussian ids overlapping tile ``t``, front to back. Build cost is
    ~0.3-0.9 ms at N=50k (see scripts/bench_binning.py — the P2a gate).

    ``pad`` caps the tile-bbox area per gaussian; a gaussian whose bbox
    exceeds ``pad`` would be SILENTLY TRUNCATED to its first ``pad`` tiles
    (wrong rendering and wrong gradients — this bit at 512x512 where init
    gaussians span ~196 tiles). Pass ``pad=None`` (default) for the exact
    setting ``pad = ntiles``, which can never truncate; pass a tuned pad
    only when the scene's max bbox area is known (fit3d recomputes it per
    refine with a 2x margin and logs it).
    """
    n = means2d.shape[0]
    tw = (width + _TILE - 1) // _TILE
    th = (height + _TILE - 1) // _TILE
    ntiles = tw * th
    if pad is None:
        pad = ntiles

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

    k = mx.arange(pad, dtype=mx.int32)[None, :]
    slot_ok = valid[:, None] & (k < mx.minimum(area, pad)[:, None])
    tx = tx0[:, None] + k % mx.maximum(bw, 1)[:, None]
    ty = ty0[:, None] + k // mx.maximum(bw, 1)[:, None]
    tile = (ty * tw + tx).astype(mx.uint32)
    rank = mx.arange(n, dtype=mx.uint32)[:, None]
    keys = mx.where(slot_ok, tile * n + rank, _INVALID_KEY).reshape(-1)

    sorted_keys = mx.sort(keys)
    bin_ids = (sorted_keys % n).astype(mx.uint32)
    sorted_tiles = mx.minimum(sorted_keys // n, ntiles).astype(mx.uint32)
    counts = mx.zeros((ntiles + 1,), dtype=mx.int32).at[sorted_tiles].add(1)
    cum = mx.cumsum(counts[:-1])
    bounds = mx.concatenate([mx.zeros((1,), dtype=mx.int32), cum]).astype(mx.int32)
    return mx.stop_gradient(bin_ids), mx.stop_gradient(bounds), mx.stop_gradient(area)


def rasterize3d_fused(
    means2d,
    conics,
    opacities,
    colors,
    background,
    depths,
    height,
    width,
    absgrad_sink=None,
    bin_pad=None,
):
    """Drop-in replacement for :func:`rendering3d.rasterize3d_dense`.

    ``bin_pad=None`` uses the exact-but-larger setting ``pad = num_tiles`` so
    no Gaussian can be silently truncated from its tile list. Pass a tuned
    integer only when the scene's maximum tile-bbox area is known."""
    order = mx.argsort(depths)
    m = mx.take(means2d, order, axis=0)
    con = mx.take(conics, order, axis=0)
    opac = mx.take(opacities, order, axis=0)
    col = mx.take(colors, order, axis=0)
    dep = mx.take(depths, order, axis=0)
    opac = mx.where((dep > NEAR_PLANE) & (dep < FAR_PLANE), opac, 0.0)
    radii = _bounding_radii(con, opac)
    bin_ids, bounds, _ = _build_bins(m, radii, width, height, bin_pad)

    compute_absgrad = absgrad_sink is not None
    if absgrad_sink is None:
        absgrad_sink = mx.zeros_like(means2d)
    abs_sink = mx.take(absgrad_sink, order, axis=0)

    core = _fused_core3d(height, width, compute_absgrad)
    acc, tfinal, _ = core(m, con, opac, col, bin_ids, bounds, abs_sink)  # type: ignore[misc]
    out = acc + tfinal[:, None] * background[None, :]
    return out.reshape(height, width, 3)
