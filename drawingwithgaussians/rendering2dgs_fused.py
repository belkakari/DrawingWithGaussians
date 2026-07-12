"""Fused RGB-only 2DGS surfel rasterizer.

Phase-1 2DGS support: projection/ray transforms come from ``rendering2dgs``;
this kernel keeps the existing 3DGS tile-bin/compositing structure but swaps
the conic power for gsplat's ray-splat intersection power.
"""

import math
from typing import Any

import mlx.core as mx

from .rendering3d_fused import _HEADER, _TILE, _build_bins_padded, _compact_keys_fit_uint32, _pad

_TG_N = _TILE * _TILE

_HEADER_2DGS = (
    _HEADER
    + """
inline float eval_quad(
    float qxx,
    float qxy,
    float qyy,
    float qx,
    float qy,
    float q0,
    float x,
    float y
) {
    return qxx * x * x + qxy * x * y + qyy * y * y + qx * x + qy * y + q0;
}

// Exact minimum of a general quadratic over an axis-aligned rectangle.
// The interior candidate plus the four edge minima cover all extrema; corners
// are included by the clamped edge/corner evaluations. This is used for the
// rational 2DGS ray-splat power test: tu/tw and tv/tw are affine-over-affine,
// so tu^2 + tv^2 <= r^2 tw^2 is a quadratic inequality in pixel coordinates.
inline float min_quad_rect(
    float qxx,
    float qxy,
    float qyy,
    float qx,
    float qy,
    float q0,
    float x0,
    float x1,
    float y0,
    float y1
) {
    float best = eval_quad(qxx, qxy, qyy, qx, qy, q0, x0, y0);
    best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, x0, y1));
    best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, x1, y0));
    best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, x1, y1));

    float det = 4.0f * qxx * qyy - qxy * qxy;
    if (metal::abs(det) > 1e-12f) {
        float xs = (qxy * qy - 2.0f * qyy * qx) / det;
        float ys = (qxy * qx - 2.0f * qxx * qy) / det;
        if (xs >= x0 && xs <= x1 && ys >= y0 && ys <= y1) {
            best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, xs, ys));
        }
    }

    if (qyy > 1e-12f) {
        float y = metal::clamp(-(qxy * x0 + qy) / (2.0f * qyy), y0, y1);
        best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, x0, y));
        y = metal::clamp(-(qxy * x1 + qy) / (2.0f * qyy), y0, y1);
        best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, x1, y));
    }
    if (qxx > 1e-12f) {
        float x = metal::clamp(-(qxy * y0 + qx) / (2.0f * qxx), x0, x1);
        best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, x, y0));
        x = metal::clamp(-(qxy * y1 + qx) / (2.0f * qxx), x0, x1);
        best = metal::min(best, eval_quad(qxx, qxy, qyy, qx, qy, q0, x, y1));
    }
    return best;
}

inline bool tile_contributes_2dgs(
    float mx,
    float my,
    float m00,
    float m01,
    float m02,
    float m10,
    float m11,
    float m12,
    float m20,
    float m21,
    float m22,
    float opacity,
    uint tx,
    uint ty,
    uint W,
    uint H
) {
    if (!(opacity > ALPHA_THRESHOLD)) return false;
    float x0 = (float)(tx * TILE) + 0.5f;
    float x1 = (float)metal::min((tx + 1u) * TILE, W) - 0.5f;
    float y0 = (float)(ty * TILE) + 0.5f;
    float y1 = (float)metal::min((ty + 1u) * TILE, H) - 0.5f;
    if (x0 > x1 || y0 > y1) return false;

    float max_sigma = metal::log(255.0f * opacity);

    // Screen-space fallback in the 2DGS kernel: sigma = ||pixel - mean||^2.
    float cx = metal::clamp(mx, x0, x1);
    float cy = metal::clamp(my, y0, y1);
    float dx = cx - mx;
    float dy = cy - my;
    if (dx * dx + dy * dy <= max_sigma + 1e-5f) return true;

    // Ray-splat branch: tu, tv and tw are affine in (x, y); test whether
    // tu^2 + tv^2 - (2 max_sigma) tw^2 can be non-positive over the tile.
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
    return min_quad_rect(qxx, qxy, qyy, qx, qy, q0, x0, x1, y0, y1) <= 1e-4f;
}

"""
)

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

    threadgroup float4 sh[TG_N * 4];
    threadgroup float4 shn[TG_N];
    threadgroup uint shid[TG_N];
    threadgroup metal::atomic_uint ndone;
    if (lid == 0) atomic_store_explicit(&ndone, 0u, metal::memory_order_relaxed);

    uint TW = (W + TILE - 1) / TILE;
    uint TH = (H + TILE - 1) / TILE;
    uint tile = tg3.z * TW * TH + tg3.y * TW + tg3.x;
    uint lo = (uint)bounds[tile];
    uint hi = (uint)bounds[tile + 1];

    device const float4* params4 = reinterpret_cast<device const float4*>(params);

    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    float depth_acc = 0.0f;
    float nr = 0.0f, ng = 0.0f, nb = 0.0f;
    float distortion = 0.0f, accum_vis_depth = 0.0f;
    float median_depth = 0.0f;
    uint median_contrib = 0;
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
            sh[4 * lid] = params4[4 * i];
            sh[4 * lid + 1] = params4[4 * i + 1];
            sh[4 * lid + 2] = params4[4 * i + 2];
            sh[4 * lid + 3] = params4[4 * i + 3];
            shn[lid] = float4(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2], 0.0f);
            shid[lid] = i;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (!done) {
            for (uint j = 0; j < total; ++j) {
                float4 q0 = sh[4 * j];
                float4 q1 = sh[4 * j + 1];
                float4 q2 = sh[4 * j + 2];
                float4 q3 = sh[4 * j + 3];
                float mx = q0.x, my = q0.y;
                float m00 = q1.x, m01 = q1.y, m02 = q1.z;
                float m10 = q2.x, m11 = q2.y, m12 = q2.z;
                float m20 = q3.x, m21 = q3.y, m22 = q3.z;
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
                float alpha = metal::min(MAX_ALPHA, q0.z * metal::fast::exp(-sigma));
                if (alpha < ALPHA_THRESHOLD) continue;
                float next_T = T * (1.0f - alpha);
                if (next_T <= TRANSMITTANCE_THRESHOLD) {
                    done = true;
                    atomic_fetch_add_explicit(&ndone, 1u, metal::memory_order_relaxed);
                    break;
                }
                float fac = alpha * T;
                r += fac * q0.w;
                g += fac * q1.w;
                b += fac * q2.w;
                // Camera-space depth at the ray-splat intersection. q3.w is
                // the projected Gaussian-center depth and is only valid for
                // a fronto-parallel splat.
                float d = u * m20 + v * m21 + m22;
                depth_acc += fac * d;
                float4 nn = shn[j];
                nr += fac * nn.x;
                ng += fac * nn.y;
                nb += fac * nn.z;
                distortion += 2.0f * (fac * d * (1.0f - T) - fac * accum_vis_depth);
                accum_vis_depth += fac * d;
                if (T > 0.5f) {
                    median_depth = d;
                    median_contrib = shid[j] + 1;
                }
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
        acc_depth[p] = depth_acc;
        acc_normals[3 * p] = nr;
        acc_normals[3 * p + 1] = ng;
        acc_normals[3 * p + 2] = nb;
        distort[p] = distortion;
        median[p] = median_depth;
        tfinal[p] = T;
        last[p] = contribs;
        median_id[p] = median_contrib;
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

    threadgroup float4 sh[TG_N * 4];
    threadgroup float4 shn[TG_N];
    threadgroup uint shid[TG_N];
    threadgroup metal::atomic_uint tile_last;
    if (lid == 0) atomic_store_explicit(&tile_last, 0u, metal::memory_order_relaxed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    uint TW = (W + TILE - 1) / TILE;
    uint TH = (H + TILE - 1) / TILE;
    uint tile = tg3.z * TW * TH + tg3.y * TW + tg3.x;
    uint lo = (uint)bounds[tile];
    uint hi = (uint)bounds[tile + 1];

    device const float4* params4 = reinterpret_cast<device const float4*>(params);

    uint p = tg3.z * W * H + py_i * W + px_i;
    uint mylast = 0;
    float T = 0.0f, Tfin = 0.0f;
    float vr0 = 0.0f, vr1 = 0.0f, vr2 = 0.0f, cT = 0.0f;
    float vd = 0.0f, vn0 = 0.0f, vn1 = 0.0f, vn2 = 0.0f, vdist = 0.0f, vmed = 0.0f;
    uint mymedian = 0;
    float accum_d = 0.0f, accum_w = 0.0f, accum_d_buf = 0.0f, accum_w_buf = 0.0f, distort_buf = 0.0f;
    if (active) {
        mylast = last[p];
        mymedian = median_id[p];
        Tfin = tfinal[p];
        T = Tfin;
        vr0 = dacc[3 * p];
        vr1 = dacc[3 * p + 1];
        vr2 = dacc[3 * p + 2];
        vd = dacc_depth[p];
        vn0 = dacc_normals[3 * p];
        vn1 = dacc_normals[3 * p + 1];
        vn2 = dacc_normals[3 * p + 2];
        vdist = ddistort[p];
        vmed = dmedian[p];
        cT = dt[p];
        accum_d = acc_depth[p];
        accum_w = 1.0f - Tfin;
        accum_d_buf = accum_d;
        accum_w_buf = accum_w;
    }
    atomic_fetch_max_explicit(&tile_last, mylast, metal::memory_order_relaxed);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    uint maxlast = atomic_load_explicit(&tile_last, metal::memory_order_relaxed);
    if (maxlast == 0 || lo == hi) return;

    float S0 = 0.0f, S1 = 0.0f, S2 = 0.0f;
    float Sd = 0.0f, Sn0 = 0.0f, Sn1 = 0.0f, Sn2 = 0.0f;
    uint nchunks = (hi - lo + TG_N - 1) / TG_N;
    for (uint cc = nchunks; cc-- > 0;) {
        if (bin_ids[lo + cc * TG_N] >= maxlast) continue;
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        uint idx = lo + cc * TG_N + lid;
        uint total = metal::min(hi - lo - cc * TG_N, TG_N);
        if (idx < hi) {
            uint i = bin_ids[idx];
            sh[4 * lid] = params4[4 * i];
            sh[4 * lid + 1] = params4[4 * i + 1];
            sh[4 * lid + 2] = params4[4 * i + 2];
            sh[4 * lid + 3] = params4[4 * i + 3];
            shn[lid] = float4(normals[3 * i], normals[3 * i + 1], normals[3 * i + 2], 0.0f);
            shid[lid] = i;
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        for (uint jj = total; jj-- > 0;) {
            uint gid = shid[jj];
            float4 q0 = sh[4 * jj];
            float4 q1 = sh[4 * jj + 1];
            float4 q2 = sh[4 * jj + 2];
            float4 q3 = sh[4 * jj + 3];
            float mx = q0.x, my = q0.y;
            float m00 = q1.x, m01 = q1.y, m02 = q1.z;
            float m10 = q2.x, m11 = q2.y, m12 = q2.z;
            float m20 = q3.x, m21 = q3.y, m22 = q3.z;
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
            float vis = metal::fast::exp(-sigma);
            float alpha_raw = q0.z * vis;
            float alpha = metal::min(MAX_ALPHA, alpha_raw);
            bool contrib = active && valid && (gid < mylast) && (alpha >= ALPHA_THRESHOLD);
            float ra = contrib ? 1.0f / (1.0f - alpha) : 1.0f;
            if (contrib) T *= ra;
            float fac = contrib ? alpha * T : 0.0f;

            float c0 = q0.w, c1 = q1.w, c2 = q2.w;
            float depth = u * m20 + v * m21 + m22;
            float4 nn = shn[jj];
            float v_depth = 0.0f, v_normal0 = 0.0f, v_normal1 = 0.0f, v_normal2 = 0.0f;
            float v_alpha = 0.0f;
            if (contrib) {
                v_alpha = (c0 * T - S0 * ra) * vr0 + (c1 * T - S1 * ra) * vr1 + (c2 * T - S2 * ra) * vr2;
                v_alpha += (depth * T - Sd * ra) * vd;
                v_alpha += (nn.x * T - Sn0 * ra) * vn0 + (nn.y * T - Sn1 * ra) * vn1 + (nn.z * T - Sn2 * ra) * vn2;
                v_alpha += cT * (-Tfin * ra);
                v_depth = fac * vd;
                v_normal0 = fac * vn0;
                v_normal1 = fac * vn1;
                v_normal2 = fac * vn2;
                if (gid + 1u == mymedian) {
                    v_depth += vmed;
                }
                float dl_dw = 2.0f * (2.0f * (depth * accum_w_buf - accum_d_buf) + (accum_d - depth * accum_w));
                v_alpha += (dl_dw * T - distort_buf * ra) * vdist;
                accum_d_buf -= fac * depth;
                accum_w_buf -= fac;
                distort_buf += dl_dw * fac;
                v_depth += 2.0f * fac * (2.0f - 2.0f * T - accum_w + fac) * vdist;
            }
            bool grad_gate = contrib && (alpha_raw <= MAX_ALPHA);
            float g_opac = grad_gate ? vis * v_alpha : 0.0f;

            if (metal::simd_any(contrib)) {
                float v_sigma = grad_gate ? -alpha_raw * v_alpha : 0.0f;
                float gmx = 0.0f, gmy = 0.0f;
                float gm0 = 0.0f, gm1 = 0.0f, gm2 = 0.0f, gm3 = 0.0f, gm4 = 0.0f;
                float gm5 = 0.0f, gm6 = 0.0f, gm7 = 0.0f, gm8 = 0.0f;
                // u and v affect both the ray-splat Gaussian power and the
                // intersection depth. The depth path remains active when the
                // screen-space fallback supplies the Gaussian power.
                float gu = v_depth * m20;
                float gv = v_depth * m21;
                if (use3d) {
                    gu += v_sigma * u;
                    gv += v_sigma * v;
                } else {
                    gmx = -2.0f * dx * v_sigma;
                    gmy = -2.0f * dy * v_sigma;
                }
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
                gm6 = px * dhu0 + py * dhv0 + v_depth * u;
                gm7 = px * dhu1 + py * dhv1 + v_depth * v;
                gm8 = px * dhu2 + py * dhv2 + v_depth;

                float t0 = fac * vr0, t1 = fac * vr1, t2 = fac * vr2, t3 = g_opac;
                if (simd_reduce_add4(t0, t1, t2, t3, lid)) {
                    atomic_fetch_add_explicit(&dparams[16 * gid + 3], t0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 7], t1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 11], t2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 2], t3, metal::memory_order_relaxed);
                }
                float tn0 = v_normal0, tn1 = v_normal1, tn2 = v_normal2, nz = 0.0f;
                if (simd_reduce_add4(tn0, tn1, tn2, nz, lid)) {
                    atomic_fetch_add_explicit(&dnormals[3 * gid], tn0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dnormals[3 * gid + 1], tn1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dnormals[3 * gid + 2], tn2, metal::memory_order_relaxed);
                }
                float w0 = gmx, w1 = gmy, w2 = metal::abs(gmx), w3 = metal::abs(gmy);
                if (simd_reduce_add4(w0, w1, w2, w3, lid)) {
                    atomic_fetch_add_explicit(&dparams[16 * gid], w0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 1], w1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid], w2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dmeans2d_abs[2 * gid + 1], w3, metal::memory_order_relaxed);
                }
                float a0 = gm0, a1 = gm1, a2 = gm2, a3 = gm3;
                if (simd_reduce_add4(a0, a1, a2, a3, lid)) {
                    atomic_fetch_add_explicit(&dparams[16 * gid + 4], a0, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 5], a1, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 6], a2, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 8], a3, metal::memory_order_relaxed);
                }
                float a4 = gm4, a5 = gm5, a6 = gm6, a7 = gm7;
                if (simd_reduce_add4(a4, a5, a6, a7, lid)) {
                    atomic_fetch_add_explicit(&dparams[16 * gid + 9], a4, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 10], a5, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 12], a6, metal::memory_order_relaxed);
                    atomic_fetch_add_explicit(&dparams[16 * gid + 13], a7, metal::memory_order_relaxed);
                }
                float a8 = gm8, z1 = 0.0f, z2 = 0.0f, z3 = 0.0f;
                if (simd_reduce_add4(a8, z1, z2, z3, lid)) {
                    atomic_fetch_add_explicit(&dparams[16 * gid + 14], a8, metal::memory_order_relaxed);
                }
            }

            if (contrib) {
                S0 += c0 * fac;
                S1 += c1 * fac;
                S2 += c2 * fac;
                Sd += depth * fac;
                Sn0 += nn.x * fac;
                Sn1 += nn.y * fac;
                Sn2 += nn.z * fac;
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
    float opacity = opac[gid];
    float m00 = ray_transforms[9 * gid], m01 = ray_transforms[9 * gid + 1], m02 = ray_transforms[9 * gid + 2];
    float m10 = ray_transforms[9 * gid + 3], m11 = ray_transforms[9 * gid + 4], m12 = ray_transforms[9 * gid + 5];
    float m20 = ray_transforms[9 * gid + 6], m21 = ray_transforms[9 * gid + 7], m22 = ray_transforms[9 * gid + 8];
    int count = 0;
    if (rx > 0.0f && ry > 0.0f && opacity > ALPHA_THRESHOLD) {
        int tx0 = (int)metal::clamp(metal::floor((mx - rx) / (float)TILE), 0.0f, (float)(TW - 1));
        int tx1 = (int)metal::clamp(metal::floor((mx + rx) / (float)TILE), 0.0f, (float)(TW - 1));
        int ty0 = (int)metal::clamp(metal::floor((my - ry) / (float)TILE), 0.0f, (float)(TH - 1));
        int ty1 = (int)metal::clamp(metal::floor((my + ry) / (float)TILE), 0.0f, (float)(TH - 1));
        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                if (tile_contributes_2dgs(
                    mx, my, m00, m01, m02, m10, m11, m12, m20, m21, m22, opacity, (uint)tx, (uint)ty, W, H
                )) {
                    ++count;
                }
            }
        }
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
    float opacity = opac[gid];
    if (!(rx > 0.0f && ry > 0.0f && opacity > ALPHA_THRESHOLD)) return;
    float m00 = ray_transforms[9 * gid], m01 = ray_transforms[9 * gid + 1], m02 = ray_transforms[9 * gid + 2];
    float m10 = ray_transforms[9 * gid + 3], m11 = ray_transforms[9 * gid + 4], m12 = ray_transforms[9 * gid + 5];
    float m20 = ray_transforms[9 * gid + 6], m21 = ray_transforms[9 * gid + 7], m22 = ray_transforms[9 * gid + 8];
    int tx0 = (int)metal::clamp(metal::floor((mx - rx) / (float)TILE), 0.0f, (float)(TW - 1));
    int tx1 = (int)metal::clamp(metal::floor((mx + rx) / (float)TILE), 0.0f, (float)(TW - 1));
    int ty0 = (int)metal::clamp(metal::floor((my - ry) / (float)TILE), 0.0f, (float)(TH - 1));
    int ty1 = (int)metal::clamp(metal::floor((my + ry) / (float)TILE), 0.0f, (float)(TH - 1));
    uint base = (uint)offsets[gid];
    uint written = 0;
    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            if (tile_contributes_2dgs(
                mx, my, m00, m01, m02, m10, m11, m12, m20, m21, m22, opacity, (uint)tx, (uint)ty, W, H
            )) {
                uint pos = base + written;
                if (pos < capacity) {
                    uint tile = cam * ntiles + (uint)ty * TW + (uint)tx;
                    keys[pos] = tile * N + gid;
                }
                ++written;
            }
        }
    }
"""

_k_fwd = mx.fast.metal_kernel(
    name="gauss2dgs_forward",
    input_names=["params", "normals", "bin_ids", "bounds", "sizes"],
    output_names=[
        "acc",
        "acc_depth",
        "acc_normals",
        "distort",
        "median",
        "tfinal",
        "last",
        "median_id",
    ],
    header=_HEADER_2DGS,
    source=_FORWARD_SRC,
)
_k_bwd = mx.fast.metal_kernel(
    name="gauss2dgs_backward",
    input_names=[
        "params",
        "normals",
        "bin_ids",
        "bounds",
        "acc_depth",
        "tfinal",
        "last",
        "median_id",
        "dacc",
        "dacc_depth",
        "dacc_normals",
        "ddistort",
        "dmedian",
        "dt",
        "sizes",
    ],
    output_names=["dparams", "dnormals", "dmeans2d_abs"],
    header=_HEADER_2DGS,
    source=_BACKWARD_SRC,
    atomic_outputs=True,
)
_k_count_bbox = mx.fast.metal_kernel(
    name="gauss2dgs_count_bbox_bins",
    input_names=["means2d", "ray_transforms", "opac", "radii", "sizes"],
    output_names=["counts"],
    header=_HEADER_2DGS,
    source=_COUNT_BBOX_SRC,
)
_k_scatter_bbox = mx.fast.metal_kernel(
    name="gauss2dgs_scatter_bbox_bins",
    input_names=["means2d", "ray_transforms", "opac", "radii", "offsets", "sizes"],
    output_names=["keys"],
    header=_HEADER_2DGS,
    source=_SCATTER_BBOX_SRC,
)

_BBOX_TG = 256
_INVALID_KEY = mx.array(0xFFFFFFFF, dtype=mx.uint32)

_CORE_CACHE: dict[tuple[int, int, int], Any] = {}


def _pack_params(means2d, ray_transforms, opacities, colors, depths):
    """Pack 2DGS per-splat state into four float4 records.

    Layout per row:
      [mx, my, opacity, cr,
       m00, m01, m02, cg,
       m10, m11, m12, cb,
       m20, m21, m22, center_depth]

    ``center_depth`` is retained for the sorting/visibility input's custom-VJP
    slot but is not used to render auxiliary depth. Rendered depth comes from
    the ray-splat intersection encoded by the transform.
    """
    return mx.concatenate(
        [
            means2d,
            opacities[:, None],
            colors[:, 0:1],
            ray_transforms[:, 0, :],
            colors[:, 1:2],
            ray_transforms[:, 1, :],
            colors[:, 2:3],
            ray_transforms[:, 2, :],
            depths[:, None],
        ],
        axis=1,
    )


def _unpack_param_grads(dparams):
    dmeans2d = dparams[:, 0:2]
    dopac = dparams[:, 2]
    dcolors = mx.stack([dparams[:, 3], dparams[:, 7], dparams[:, 11]], axis=-1)
    ddepths = dparams[:, 15]
    dray = mx.stack(
        [
            mx.stack([dparams[:, 4], dparams[:, 5], dparams[:, 6]], axis=-1),
            mx.stack([dparams[:, 8], dparams[:, 9], dparams[:, 10]], axis=-1),
            mx.stack([dparams[:, 12], dparams[:, 13], dparams[:, 14]], axis=-1),
        ],
        axis=-2,
    )
    return dmeans2d, dray, dopac, dcolors, ddepths


def _core(height, width, ncams=1) -> Any:
    key = (height, width, ncams)
    cached = _CORE_CACHE.get(key)
    if cached is not None:
        return cached
    num_pixels = ncams * height * width
    grid = (_pad(width, _TILE), _pad(height, _TILE), ncams)
    tg = (_TILE, _TILE, 1)

    @mx.custom_function
    def core(
        means2d,
        ray_transforms,
        opacities,
        colors,
        depths,
        normals,
        bin_ids,
        bounds,
        absgrad_sink,
    ):
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        params = _pack_params(means2d, ray_transforms, opacities, colors, depths)
        acc, acc_depth, acc_normals, distort, median, tfinal, last, median_id = _k_fwd(  # type: ignore[operator]
            inputs=[params, normals, bin_ids, bounds, sizes],
            grid=grid,
            threadgroup=tg,
            output_shapes=[
                (num_pixels, 3),
                (num_pixels,),
                (num_pixels, 3),
                (num_pixels,),
                (num_pixels,),
                (num_pixels,),
                (num_pixels,),
                (num_pixels,),
            ],
            output_dtypes=[
                mx.float32,
                mx.float32,
                mx.float32,
                mx.float32,
                mx.float32,
                mx.float32,
                mx.uint32,
                mx.uint32,
            ],
        )
        return acc, acc_depth, acc_normals, distort, median, tfinal, last, median_id

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        (
            means2d,
            ray_transforms,
            opacities,
            colors,
            depths,
            normals,
            bin_ids,
            bounds,
            absgrad_sink,
        ) = primals
        dacc, dacc_depth, dacc_normals, ddistort, dmedian, dt = (
            cotangents[0],
            cotangents[1],
            cotangents[2],
            cotangents[3],
            cotangents[4],
            cotangents[5],
        )
        _, acc_depth, _, _, _, tfinal, last, median_id = outputs
        n = means2d.shape[0]
        sizes = mx.array([n, width, height], dtype=mx.int32)
        params = _pack_params(means2d, ray_transforms, opacities, colors, depths)
        dparams, dnormals, dabs = _k_bwd(  # type: ignore[operator]
            inputs=[
                params,
                normals,
                bin_ids,
                bounds,
                acc_depth,
                tfinal,
                last,
                median_id,
                dacc,
                dacc_depth,
                dacc_normals,
                ddistort,
                dmedian,
                dt,
                sizes,
            ],
            grid=grid,
            threadgroup=tg,
            output_shapes=[(n, 16), (n, 3), (n, 2)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
            init_value=0,
        )
        dmeans2d, dray, dopac, dcolors, ddepths = _unpack_param_grads(dparams)
        return (
            dmeans2d,
            dray,
            dopac,
            dcolors,
            ddepths,
            dnormals,
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


def _count_bbox_intersections(means2d, ray_transforms, opacities, radii, width, height):
    means2d, ray_transforms, opacities, radii = _as_camera_batch(means2d, ray_transforms, opacities, radii)
    ncams, n = means2d.shape[0], means2d.shape[1]
    flat_n = ncams * n
    sizes = mx.array([flat_n, width, height], dtype=mx.int32)
    counts = _k_count_bbox(  # type: ignore[operator]
        inputs=[
            means2d.reshape(flat_n, 2),
            ray_transforms.reshape(flat_n, 3, 3),
            opacities.reshape(flat_n),
            radii.reshape(flat_n, 2),
            sizes,
        ],
        grid=(_pad(flat_n, _BBOX_TG), 1, 1),
        threadgroup=(_BBOX_TG, 1, 1),
        output_shapes=[(flat_n,)],
        output_dtypes=[mx.int32],
    )[0]
    return mx.stop_gradient(counts.reshape(ncams, n))


def estimate_bin_capacity_2dgs(
    means2d,
    ray_transforms,
    opacities,
    radii,
    width,
    height,
    *,
    margin=2.0,
    min_per_gaussian=16,
):
    counts = _count_bbox_intersections(means2d, ray_transforms, opacities, radii, width, height)
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


def _build_bins_compact(
    means2d,
    ray_transforms,
    opacities,
    radii,
    width,
    height,
    capacity,
    return_counts=False,
):
    means2d, ray_transforms, opacities, radii = _as_camera_batch(means2d, ray_transforms, opacities, radii)
    ncams, n = means2d.shape[0], means2d.shape[1]
    flat_n = ncams * n
    ntiles = _num_tiles(width, height)
    # The scatter kernel writes uint32 keys (tile * flat_n + gid); fall back
    # to the exact int64 padded path when the key range would overflow
    # (large camera-batch x N x tile products) — same guard as the 3DGS
    # builder. Silent overflow would corrupt tile assignments.
    if not _compact_keys_fit_uint32(ncams, n, ntiles):
        bin_ids, bounds, area = _build_bins_padded(means2d, radii, width, height, None)
        if return_counts:
            area = _count_bbox_intersections(
                means2d,
                ray_transforms,
                opacities,
                radii,
                width,
                height,
            )
        return bin_ids, bounds, area
    capacity = int(max(1, capacity))
    counts = _count_bbox_intersections(means2d, ray_transforms, opacities, radii, width, height).reshape(flat_n)
    offsets = mx.cumsum(counts) - counts
    sizes = mx.array([flat_n, width, height, n, capacity], dtype=mx.int32)
    keys = _k_scatter_bbox(  # type: ignore[operator]
        inputs=[
            means2d.reshape(flat_n, 2),
            ray_transforms.reshape(flat_n, 3, 3),
            opacities.reshape(flat_n),
            radii.reshape(flat_n, 2),
            offsets,
            sizes,
        ],
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


def _build_bins(
    means2d,
    ray_transforms,
    opacities,
    radii,
    width,
    height,
    capacity=None,
    return_counts=False,
):
    if capacity is not None:
        return _build_bins_compact(
            means2d,
            ray_transforms,
            opacities,
            radii,
            width,
            height,
            capacity,
            return_counts=return_counts,
        )
    bin_ids, bounds, area = _build_bins_padded(means2d, radii, width, height, None)
    if return_counts:
        area = _count_bbox_intersections(
            means2d,
            ray_transforms,
            opacities,
            radii,
            width,
            height,
        )
    return bin_ids, bounds, area


def _take_sorted_features(features, order):
    """Gather shared ``(N, D)`` or per-view ``(C, N, D)`` features by depth order."""
    if features.ndim == 3:
        return mx.take_along_axis(features, mx.broadcast_to(order[..., None], features.shape), axis=1)
    return mx.take(features, order, axis=0)


def _squeeze_aux(aux, batched):
    if batched:
        return aux
    return {k: v[0] for k, v in aux.items()}


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
    bin_capacity=None,
    normals=None,
    return_aux=False,
    return_counts=False,
):
    """2DGS fused rasterizer, single camera or camera batch.

    By default returns RGB only. With ``return_aux=True`` returns
    ``(rgb, aux)`` where ``aux`` contains alpha, accumulated/expected depth,
    accumulated normals, distortion, and median depth.

    With ``return_counts=True``, exact tile-intersection counts are appended
    to the return tuple in original parameter order. Compact bins reuse their
    builder counts; exact/padded and uint32-fallback modes use the standalone
    exact count kernel rather than exposing padded bbox slot counts.
    """
    batched = means2d.ndim == 3
    if not batched:
        means2d, ray_transforms, depths, radii = (
            means2d[None],
            ray_transforms[None],
            depths[None],
            radii[None],
        )
        if colors.ndim == 3:
            colors = colors[0]
        if normals is not None and normals.ndim == 3:
            normals = normals[0]
    ncams, n = means2d.shape[0], means2d.shape[1]
    order = mx.argsort(depths, axis=-1)
    m = mx.take_along_axis(means2d, mx.broadcast_to(order[..., None], means2d.shape), axis=1)
    ray = mx.take_along_axis(
        ray_transforms,
        mx.broadcast_to(order[..., None, None], ray_transforms.shape),
        axis=1,
    )
    dep = mx.take_along_axis(depths, order, axis=-1)
    rad = mx.take_along_axis(radii, mx.broadcast_to(order[..., None], radii.shape), axis=1)
    opac = mx.take(opacities, order)
    col = _take_sorted_features(colors, order)
    if normals is None:
        normals = mx.zeros((n, 3), dtype=means2d.dtype)
    nrm = _take_sorted_features(normals, order)
    opac = mx.where((dep > 0.01) & (dep < 1e10), opac, 0.0)
    bin_ids, bounds, builder_counts = _build_bins(
        m,
        ray,
        opac,
        rad,
        width,
        height,
        capacity=bin_capacity,
        return_counts=return_counts,
    )
    counts_out = None
    if return_counts:
        inverse_order = mx.argsort(order, axis=-1)
        counts_param = mx.take_along_axis(builder_counts, inverse_order, axis=1)
        counts_out = counts_param if batched else counts_param[0]

    if absgrad_sink is None:
        absgrad_sink = mx.zeros((n, 2), dtype=means2d.dtype)
    abs_sink = mx.take(absgrad_sink, order, axis=0)
    flat_n = ncams * n
    acc, acc_depth, acc_normals, distort, median, tfinal, _, _ = _core(height, width, ncams)(
        m.reshape(flat_n, 2),
        ray.reshape(flat_n, 3, 3),
        opac.reshape(flat_n),
        col.reshape(flat_n, 3),
        dep.reshape(flat_n),
        nrm.reshape(flat_n, 3),
        bin_ids,
        bounds,
        abs_sink.reshape(flat_n, 2),
    )
    out = acc + tfinal[:, None] * background[None, :]
    out = out.reshape(ncams, height, width, 3)
    if not return_aux:
        out = out if batched else out[0]
        return (out, counts_out) if return_counts else out

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
    return (*result, counts_out) if return_counts else result
