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

import math
from typing import Any

import mlx.core as mx

from .rendering3d import FAR_PLANE, NEAR_PLANE

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

inline float sigma_at(float A, float B, float C, float dx, float dy) {
    return 0.5f * (A * dx * dx + C * dy * dy) + B * dx * dy;
}

// Exact minimum of the positive-definite 2D gaussian power over an
// axis-aligned rectangle of pixel centers, in coordinates relative to the
// gaussian mean. The unconstrained optimum is (0, 0); if that is outside the
// box, the convex minimum lies on one of the four edges.
inline float min_sigma_rect(float A, float B, float C, float dx0, float dx1, float dy0, float dy1) {
    if (dx0 <= 0.0f && dx1 >= 0.0f && dy0 <= 0.0f && dy1 >= 0.0f) {
        return 0.0f;
    }
    float best = 3.402823466e38f;
    float y = metal::clamp(-B * dx0 / C, dy0, dy1);
    best = metal::min(best, sigma_at(A, B, C, dx0, y));
    y = metal::clamp(-B * dx1 / C, dy0, dy1);
    best = metal::min(best, sigma_at(A, B, C, dx1, y));
    float x = metal::clamp(-B * dy0 / A, dx0, dx1);
    best = metal::min(best, sigma_at(A, B, C, x, dy0));
    x = metal::clamp(-B * dy1 / A, dx0, dx1);
    best = metal::min(best, sigma_at(A, B, C, x, dy1));
    return metal::max(best, 0.0f);
}

inline bool tile_contributes(
    float mx,
    float my,
    float A,
    float B,
    float C,
    float max_sigma,
    uint tx,
    uint ty,
    uint W,
    uint H
) {
    float x0 = (float)(tx * TILE) + 0.5f - mx;
    float x1 = (float)metal::min((tx + 1u) * TILE, W) - 0.5f - mx;
    float y0 = (float)(ty * TILE) + 0.5f - my;
    float y1 = (float)metal::min((ty + 1u) * TILE, H) - 0.5f - my;
    if (x0 > x1 || y0 > y1) return false;
    return min_sigma_rect(A, B, C, x0, x1, y0, y1) <= max_sigma;
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
    uint TH = (H + TILE - 1) / TILE;
    // Camera batch: grid z selects the view; tiles and pixels are laid out
    // view-major so one launch rasterizes the whole batch.
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
        uint p = tg3.z * W * H + py_i * W + px_i;
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
    uint TH = (H + TILE - 1) / TILE;
    // Camera batch: grid z selects the view; tiles and pixels are laid out
    // view-major so one launch rasterizes the whole batch.
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

_COUNT_ISECTS_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = (uint)sizes[0];
    uint W = (uint)sizes[1];
    uint H = (uint)sizes[2];
    if (gid >= N) return;

    float mx = means2d[2 * gid];
    float my = means2d[2 * gid + 1];
    float A = conics[3 * gid];
    float B = conics[3 * gid + 1];
    float C = conics[3 * gid + 2];
    float opacity = opac[gid];
    uint count = 0;

    if (opacity > ALPHA_THRESHOLD && A > 0.0f && C > 0.0f) {
        float det = A * C - B * B;
        if (det > 1e-12f) {
            float max_sigma = metal::log(255.0f * opacity);
            float rscale = metal::sqrt(2.0f * max_sigma);
            float rx = rscale * metal::sqrt(C / det);
            float ry = rscale * metal::sqrt(A / det);
            float xmin = mx - rx, xmax = mx + rx;
            float ymin = my - ry, ymax = my + ry;
            if (xmax >= 0.5f && xmin <= (float)W - 0.5f && ymax >= 0.5f && ymin <= (float)H - 0.5f) {
                uint TW = (W + TILE - 1) / TILE;
                uint TH = (H + TILE - 1) / TILE;
                int tx0 = (int)metal::clamp(metal::floor(xmin / (float)TILE), 0.0f, (float)(TW - 1));
                int tx1 = (int)metal::clamp(metal::floor(xmax / (float)TILE), 0.0f, (float)(TW - 1));
                int ty0 = (int)metal::clamp(metal::floor(ymin / (float)TILE), 0.0f, (float)(TH - 1));
                int ty1 = (int)metal::clamp(metal::floor(ymax / (float)TILE), 0.0f, (float)(TH - 1));
                for (int ty = ty0; ty <= ty1; ++ty) {
                    for (int tx = tx0; tx <= tx1; ++tx) {
                        if (tile_contributes(mx, my, A, B, C, max_sigma, (uint)tx, (uint)ty, W, H)) {
                            ++count;
                        }
                    }
                }
            }
        }
    }
    counts[gid] = (int)count;
"""

_SCATTER_ISECTS_SRC = """
    uint gid = thread_position_in_grid.x;
    uint N = (uint)sizes[0];
    uint W = (uint)sizes[1];
    uint H = (uint)sizes[2];
    uint n_per_view = (uint)sizes[3];
    uint capacity = (uint)sizes[4];
    if (gid >= N) return;

    float mx = means2d[2 * gid];
    float my = means2d[2 * gid + 1];
    float A = conics[3 * gid];
    float B = conics[3 * gid + 1];
    float C = conics[3 * gid + 2];
    float opacity = opac[gid];
    uint written = 0;
    uint base = (uint)offsets[gid];

    if (opacity > ALPHA_THRESHOLD && A > 0.0f && C > 0.0f) {
        float det = A * C - B * B;
        if (det > 1e-12f) {
            float max_sigma = metal::log(255.0f * opacity);
            float rscale = metal::sqrt(2.0f * max_sigma);
            float rx = rscale * metal::sqrt(C / det);
            float ry = rscale * metal::sqrt(A / det);
            float xmin = mx - rx, xmax = mx + rx;
            float ymin = my - ry, ymax = my + ry;
            if (xmax >= 0.5f && xmin <= (float)W - 0.5f && ymax >= 0.5f && ymin <= (float)H - 0.5f) {
                uint TW = (W + TILE - 1) / TILE;
                uint TH = (H + TILE - 1) / TILE;
                uint ntiles = TW * TH;
                uint cam = gid / n_per_view;
                int tx0 = (int)metal::clamp(metal::floor(xmin / (float)TILE), 0.0f, (float)(TW - 1));
                int tx1 = (int)metal::clamp(metal::floor(xmax / (float)TILE), 0.0f, (float)(TW - 1));
                int ty0 = (int)metal::clamp(metal::floor(ymin / (float)TILE), 0.0f, (float)(TH - 1));
                int ty1 = (int)metal::clamp(metal::floor(ymax / (float)TILE), 0.0f, (float)(TH - 1));
                for (int ty = ty0; ty <= ty1; ++ty) {
                    for (int tx = tx0; tx <= tx1; ++tx) {
                        if (tile_contributes(mx, my, A, B, C, max_sigma, (uint)tx, (uint)ty, W, H)) {
                            uint pos = base + written;
                            if (pos < capacity) {
                                uint tile = cam * ntiles + (uint)ty * TW + (uint)tx;
                                keys[pos] = tile * N + gid;
                            }
                            ++written;
                        }
                    }
                }
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
_k_count_isects = mx.fast.metal_kernel(
    name="gauss3d_count_intersections",
    input_names=["means2d", "conics", "opac", "sizes"],
    output_names=["counts"],
    header=_HEADER,
    source=_COUNT_ISECTS_SRC,
)
_k_scatter_isects = mx.fast.metal_kernel(
    name="gauss3d_scatter_intersections",
    input_names=["means2d", "conics", "opac", "offsets", "sizes"],
    output_names=["keys"],
    header=_HEADER,
    source=_SCATTER_ISECTS_SRC,
)


def _pad(n, m):
    return (n + m - 1) // m * m


_CORE_CACHE: dict[tuple[int, int, int, bool], Any] = {}


def _fused_core3d(height, width, ncams=1, compute_absgrad=True) -> Any:
    key = (height, width, ncams, compute_absgrad)
    cached = _CORE_CACHE.get(key)  # type: ignore[assignment]
    if cached is not None:
        return cached

    # Camera batch = grid z: one launch rasterizes all views, with tiles,
    # pixels and gaussian rows laid out view-major (see _build_bins).
    num_pixels = ncams * height * width
    grid = (_pad(width, _TILE), _pad(height, _TILE), ncams)
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


def _as_camera_batch(*arrays):
    """Normalize single-view arrays to the view-major ``(C, N, ...)`` layout."""
    if arrays[0].ndim == 2:
        return tuple(a[None] if a.ndim in {1, 2} else a for a in arrays)
    return arrays


def _bounding_radii(conics, opacities):
    """Opacity-scaled axis-aligned half-extents for the alpha >= 1/255 region.

    For ``alpha = opacity * exp(-sigma)`` and the kernel's skip threshold,
    contributing pixels satisfy ``sigma <= log(255 * opacity)``. The largest
    x/y displacement of that ellipse is ``sqrt(2 * sigma_max)`` times the
    marginal standard deviation. Transparent floaters therefore get much
    smaller bins than the old fixed 3.33-sigma bound; gaussians below the
    alpha threshold get radius -1 so binning drops them entirely.
    """
    a, b, c = conics[..., 0], conics[..., 1], conics[..., 2]
    det = mx.maximum(a * c - b * b, 1e-12)
    sigma_max = mx.maximum(mx.log(mx.maximum(opacities * 255.0, 1.0)), 0.0)
    scale = mx.sqrt(2.0 * sigma_max)
    rx = scale * mx.sqrt(mx.maximum(c / det, 0.0))
    ry = scale * mx.sqrt(mx.maximum(a / det, 0.0))
    radii = mx.stack([rx, ry], axis=-1)
    radii = mx.where(opacities[..., None] > float(1.0 / 255.0), radii, -1.0)
    return mx.stop_gradient(radii)


_INVALID_KEY = mx.array(0xFFFFFFFF, dtype=mx.uint32)
_ISECT_TG = 256


def _count_tile_intersections(means2d, conics, opacities, width, height):
    """Count exact tile intersections per sorted gaussian/view row.

    The Metal kernel first applies the opacity-aware cutoff radius, then an
    exact convex ellipse-vs-tile-center-rectangle test. Counts are
    gradient-stopped because bins only gate visibility below the kernel's
    alpha threshold.
    """
    means2d, conics, opacities = _as_camera_batch(means2d, conics, opacities)
    ncams, n = means2d.shape[0], means2d.shape[1]
    flat_n = ncams * n
    sizes = mx.array([flat_n, width, height], dtype=mx.int32)
    counts = _k_count_isects(  # type: ignore[operator]
        inputs=[
            means2d.reshape(flat_n, 2),
            conics.reshape(flat_n, 3),
            opacities.reshape(flat_n),
            sizes,
        ],
        grid=(_pad(flat_n, _ISECT_TG), 1, 1),
        threadgroup=(_ISECT_TG, 1, 1),
        output_shapes=[(flat_n,)],
        output_dtypes=[mx.int32],
    )[0]
    return mx.stop_gradient(counts.reshape(ncams, n))


def estimate_bin_capacity(means2d, conics, opacities, width, height, *, margin=2.0, min_per_gaussian=16):
    """Host-side helper for epoch/batch-specialized compact-bin capacity.

    Returns a Python integer capacity with an INVALID-padded tail. ``None`` is
    never returned here; callers that want the old exact full-tile capacity can
    pass ``bin_capacity=None`` to :func:`rasterize3d_fused`.
    """
    counts = _count_tile_intersections(means2d, conics, opacities, width, height)
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


def _num_tiles(width, height):
    return ((width + _TILE - 1) // _TILE) * ((height + _TILE - 1) // _TILE)


def _build_bins_padded(means2d, radii, width, height, pad):
    """Compatibility path: fixed per-gaussian bbox slots plus INVALID tail."""
    means2d, radii = _as_camera_batch(means2d, radii)
    ncams, n = means2d.shape[0], means2d.shape[1]
    tw = (width + _TILE - 1) // _TILE
    th = (height + _TILE - 1) // _TILE
    ntiles = tw * th
    if pad is None:
        pad = ntiles
    flat_n = ncams * n

    mxs, mys = means2d[..., 0], means2d[..., 1]
    rx, ry = radii[..., 0], radii[..., 1]
    valid = rx > 0
    tx0 = mx.clip(mx.floor((mxs - rx) / _TILE), 0, tw - 1).astype(mx.int32)
    tx1 = mx.clip(mx.floor((mxs + rx) / _TILE), 0, tw - 1).astype(mx.int32)
    ty0 = mx.clip(mx.floor((mys - ry) / _TILE), 0, th - 1).astype(mx.int32)
    ty1 = mx.clip(mx.floor((mys + ry) / _TILE), 0, th - 1).astype(mx.int32)
    bw = tx1 - tx0 + 1
    bh = ty1 - ty0 + 1
    area = mx.where(valid, bw * bh, 0)

    max_key = (ncams * ntiles) * flat_n + flat_n
    if max_key < 2**32 - 1:
        key_dtype, invalid = mx.uint32, _INVALID_KEY
    else:
        key_dtype, invalid = mx.int64, mx.array(max_key + 1, dtype=mx.int64)

    k = mx.arange(pad, dtype=mx.int32)[None, None, :]
    slot_ok = valid[..., None] & (k < mx.minimum(area, pad)[..., None])
    tx = tx0[..., None] + k % mx.maximum(bw, 1)[..., None]
    ty = ty0[..., None] + k // mx.maximum(bw, 1)[..., None]
    view_off = (mx.arange(ncams, dtype=mx.int32) * ntiles)[:, None, None]
    tile = (view_off + ty * tw + tx).astype(key_dtype)
    rank = (mx.arange(ncams, dtype=mx.int32)[:, None] * n + mx.arange(n, dtype=mx.int32)[None, :]).astype(key_dtype)
    keys = mx.where(slot_ok, tile * flat_n + rank[..., None], invalid).reshape(-1)

    sorted_keys = mx.sort(keys)
    bin_ids = (sorted_keys % flat_n).astype(mx.uint32)
    sorted_tiles = mx.minimum(sorted_keys // flat_n, ncams * ntiles).astype(mx.uint32)
    counts = mx.zeros((ncams * ntiles + 1,), dtype=mx.int32).at[sorted_tiles].add(1)
    cum = mx.cumsum(counts[:-1])
    bounds = mx.concatenate([mx.zeros((1,), dtype=mx.int32), cum]).astype(mx.int32)
    return mx.stop_gradient(bin_ids), mx.stop_gradient(bounds), mx.stop_gradient(area)


def _build_bins_compact(means2d, conics, opacities, width, height, capacity):
    """Compact count -> prefix -> scatter bin builder.

    ``capacity`` is the static sort length used inside ``mx.compile``. The
    scatter kernel writes only real exact tile intersections and leaves the
    INVALID-padded tail for the global sort, so a single giant gaussian no
    longer forces every row to expand to its footprint.
    """
    means2d, conics, opacities = _as_camera_batch(means2d, conics, opacities)
    ncams, n = means2d.shape[0], means2d.shape[1]
    flat_n = ncams * n
    ntiles = _num_tiles(width, height)
    max_key = (ncams * ntiles) * flat_n + flat_n
    if max_key >= 2**32 - 1:
        # The compact Metal scatter writes uint32 keys. Keep the old int64
        # path for very large camera/N/tile products rather than risking key
        # overflow.
        radii = _bounding_radii(conics, opacities)
        return _build_bins_padded(means2d, radii, width, height, None)

    capacity = int(max(1, capacity))
    counts = _count_tile_intersections(means2d, conics, opacities, width, height).reshape(flat_n)
    offsets = mx.cumsum(counts) - counts
    sizes = mx.array([flat_n, width, height, n, capacity], dtype=mx.int32)
    keys = _k_scatter_isects(  # type: ignore[operator]
        inputs=[
            means2d.reshape(flat_n, 2),
            conics.reshape(flat_n, 3),
            opacities.reshape(flat_n),
            offsets,
            sizes,
        ],
        grid=(_pad(flat_n, _ISECT_TG), 1, 1),
        threadgroup=(_ISECT_TG, 1, 1),
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


def _build_bins(means2d, conics, opacities, radii, width, height, pad=None, capacity=None):
    """Build depth-ordered per-tile bins.

    With ``capacity`` (or integer ``pad``) this uses compact exact
    intersections. ``capacity=None`` and ``pad=None`` remains the fully exact
    compatibility path with one slot per tile per gaussian; it is safe but can
    be very slow at large images.
    """
    if capacity is not None:
        return _build_bins_compact(means2d, conics, opacities, width, height, capacity)
    if pad is not None:
        means2d_b = means2d[None] if means2d.ndim == 2 else means2d
        capacity = means2d_b.shape[0] * means2d_b.shape[1] * int(pad)
        return _build_bins_compact(means2d, conics, opacities, width, height, capacity)
    return _build_bins_padded(means2d, radii, width, height, None)


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
    bin_capacity=None,
):
    """Drop-in replacement for :func:`rendering3d.rasterize3d_dense`.

    Accepts a single camera (``means2d`` (N, 2), ``conics`` (N, 3),
    ``depths`` (N,) -> (H, W, 3)) or a camera batch in gsplat's
    ``[..., C, N]`` convention (``means2d`` (C, N, 2), ``conics`` (C, N, 3),
    ``depths`` (C, N) -> (C, H, W, 3)). ``opacities`` (N,), ``colors``
    (N, 3), ``background`` (3,) and ``absgrad_sink`` (N, 2) are shared
    across the batch (same gaussians seen from C cameras); their gradients
    sum over views through the gathers. The batch renders in ONE kernel
    launch per pass (grid z = C) over one jointly sorted bin list — no
    Python loop.

    ``bin_capacity`` selects the compact count/prefix/scatter builder and is
    the static sort length used inside ``mx.compile``. For backwards
    compatibility, an integer ``bin_pad`` becomes ``C * N * bin_pad`` compact
    capacity; ``bin_pad=None``/``bin_capacity=None`` keeps the old fully exact
    padded path with one slot per tile per gaussian."""
    batched = means2d.ndim == 3
    if not batched:
        means2d, conics, depths = means2d[None], conics[None], depths[None]
    ncams, n = means2d.shape[0], means2d.shape[1]

    # Per-view depth order; gathers of the shared (N, ...) params scatter-add
    # their gradients over the batch in the VJP.
    order = mx.argsort(depths, axis=-1)  # (C, N)
    m = mx.take_along_axis(means2d, mx.broadcast_to(order[..., None], means2d.shape), axis=1)
    con = mx.take_along_axis(conics, mx.broadcast_to(order[..., None], conics.shape), axis=1)
    dep = mx.take_along_axis(depths, order, axis=-1)
    opac = mx.take(opacities, order)  # (C, N)
    col = mx.take(colors, order, axis=0)  # (C, N, 3)
    opac = mx.where((dep > NEAR_PLANE) & (dep < FAR_PLANE), opac, 0.0)
    radii = _bounding_radii(con, opac)
    bin_ids, bounds, _ = _build_bins(m, con, opac, radii, width, height, pad=bin_pad, capacity=bin_capacity)

    compute_absgrad = absgrad_sink is not None
    if absgrad_sink is None:
        absgrad_sink = mx.zeros((n, 2), dtype=means2d.dtype)
    abs_sink = mx.take(absgrad_sink, order, axis=0)  # (C, N, 2)

    flat_n = ncams * n
    core = _fused_core3d(height, width, ncams, compute_absgrad)
    acc, tfinal, _ = core(  # type: ignore[misc]
        m.reshape(flat_n, 2),
        con.reshape(flat_n, 3),
        opac.reshape(flat_n),
        col.reshape(flat_n, 3),
        bin_ids,
        bounds,
        abs_sink.reshape(flat_n, 2),
    )
    out = acc + tfinal[:, None] * background[None, :]
    out = out.reshape(ncams, height, width, 3)
    return out if batched else out[0]
