"""Fused Metal-kernel 2D-gaussian rasterizer (gsplat-style).

Same math as :func:`drawingwithgaussians.rendering2d.rasterize` — per-pixel
``dx = x - mean`` (no quadratic expansion), ``pdf = 0.5 * (p00 dx^2 +
cross dx dy + p11 dy^2)``, per-gaussian peak normalization
``exp(-(pdf - min_p pdf))``, colors accumulated additively — but implemented
with hand-written kernels via ``mx.fast.metal_kernel`` following gsplat's
CUDA rasterizer structure: the (N, P) gaussian-pixel matrix is never
materialized, so the step is compute-bound instead of bandwidth-bound.

Three kernels; the per-gaussian ones are additionally chunked over
``_N_CHUNKS`` pixel blocks (grid ``(chunks, N)``) because N alone (~1e3)
underutilizes the Metal device, with the tiny cross-chunk reductions done in regular
MLX ops:

* pass A (gaussian x chunk): per-gaussian ``min_p pdf`` + argmin, needed by
  the peak normalization (gsplat has no such reduction; it's this renderer's
  convention). ``minpdf``/``argmin`` are extra *outputs* of the custom
  function so the backward reuses them instead of recomputing the pass.
* pass B (pixel-parallel): forward color accumulation, one thread per pixel
  looping over gaussians.
* pass C (gaussian x chunk): analytic backward — partial gradients w.r.t.
  means, precision entries (p00, p11, cross) and colors; no atomics, one
  thread owns one (gaussian, pixel-block) pair (gsplat's backward needs
  atomicAdd because pixels are tiled). The gradient path through the argmin
  pixel of the normalization is added afterwards in N-sized MLX ops. Wired
  into autodiff with ``mx.custom_function``; the small chains (L -> cov ->
  precision, loss) stay in regular MLX autodiff.

Numerics: validated against an fp64 numpy ground truth — the fused path is
*closer* to fp64 than the dense MLX path on loss, rendered image and
gradients (see EXPERIMENTS.md). Accumulation loops use Kahan compensated
summation; ``metal::precise::exp`` avoids fast-math exp error.
"""

import mlx.core as mx

from .rendering2d import _DET_EPS, _pixel_grid

_HEADER = """
inline float pdf_at(float px, float py, float mx0, float mx1,
                    float p00, float p11, float cr) {
    float dx = px - mx0;
    float dy = py - mx1;
    return 0.5f * (p00 * dx * dx + cr * dx * dy + p11 * dy * dy);
}

// Kahan compensated accumulator: the serial fp32 loops here sum a few
// hundred to a few thousand terms; plain accumulation drifts up to ~1e-3
// relative in the worst case, Kahan keeps the sums at ~1 ulp.
struct KahanAcc {
    float s = 0.0f;
    float c = 0.0f;
    inline void add(float v) {
        float y = v - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
};
"""

# Pass A: per-(gaussian, pixel-block) min pdf + argmin. Thread (cx, g) scans
# pixels [cx*chunk, (cx+1)*chunk); the (N, C) partials are reduced with
# mx.argmin afterwards. min is order-independent; ties resolve to the lowest
# pixel index within a chunk and the lowest chunk across chunks.
_MINPDF_SRC = """
    uint cx = thread_position_in_grid.x;
    uint g = thread_position_in_grid.y;
    uint N = (uint)sizes[0];
    uint P = (uint)sizes[1];
    uint C = (uint)sizes[2];
    if (g >= N || cx >= C) return;
    uint chunk = (P + C - 1) / C;
    uint p0 = cx * chunk;
    uint p1 = metal::min(p0 + chunk, P);
    float mx0 = means[2 * g];
    float mx1 = means[2 * g + 1];
    float a = p00[g]; float b = p11[g]; float c = cross_[g];
    float best = INFINITY;
    uint besti = 0;
    for (uint p = p0; p < p1; ++p) {
        float v = pdf_at(xg[p], yg[p], mx0, mx1, a, b, c);
        if (v < best) { best = v; besti = p; }
    }
    minpdf_part[g * C + cx] = best;
    argmin_part[g * C + cx] = besti;
"""

# Pass B: pixel-parallel forward accumulation. grid = (P,)
_FORWARD_SRC = """
    uint p = thread_position_in_grid.x;
    uint N = (uint)sizes[0];
    uint P = (uint)sizes[1];
    if (p >= P) return;
    float px = xg[p]; float py = yg[p];
    KahanAcc r, gch, bch;
    for (uint g = 0; g < N; ++g) {
        float v = pdf_at(px, py, means[2 * g], means[2 * g + 1],
                         p00[g], p11[g], cross_[g]);
        float y = metal::precise::exp(-(v - minpdf[g]));
        r.add(y * colors[3 * g]);
        gch.add(y * colors[3 * g + 1]);
        bch.add(y * colors[3 * g + 2]);
    }
    acc[3 * p] = r.s;
    acc[3 * p + 1] = gch.s;
    acc[3 * p + 2] = bch.s;
"""

# Pass C: per-(gaussian, pixel-block) backward partials. With intensity
# y = exp(-(pdf - minpdf)): d pdf_p = -y_p * gy_p for every pixel, where
# gy_p = dot(colors[g], dacc[p]). S = sum_p y_p gy_p feeds the gradient of
# the min (+S into pdf at the argmin pixel) — accumulated here per block,
# applied in MLX ops after the cross-block reduction. Partials are laid out
# (C, N, k) so mx.sum(axis=0) reduces them.
_BACKWARD_SRC = """
    uint cx = thread_position_in_grid.x;
    uint g = thread_position_in_grid.y;
    uint N = (uint)sizes[0];
    uint P = (uint)sizes[1];
    uint C = (uint)sizes[2];
    if (g >= N || cx >= C) return;
    uint chunk = (P + C - 1) / C;
    uint p0 = cx * chunk;
    uint p1 = metal::min(p0 + chunk, P);
    float mx0 = means[2 * g];
    float mx1 = means[2 * g + 1];
    float a = p00[g]; float b = p11[g]; float c = cross_[g];
    float mp = minpdf[g];
    float c0 = colors[3 * g], c1 = colors[3 * g + 1], c2 = colors[3 * g + 2];

    KahanAcc S, dm0, dm1, da, db, dc, dc0, dc1, dc2;
    for (uint p = p0; p < p1; ++p) {
        float dx = xg[p] - mx0;
        float dy = yg[p] - mx1;
        float v = 0.5f * (a * dx * dx + c * dx * dy + b * dy * dy);
        float y = metal::precise::exp(-(v - mp));
        float g0 = dacc[3 * p], g1 = dacc[3 * p + 1], g2 = dacc[3 * p + 2];
        float gy = c0 * g0 + c1 * g1 + c2 * g2;
        float yg_ = y * gy;
        S.add(yg_);
        float dpdf = -yg_;
        dm0.add(dpdf * (-(a * dx + 0.5f * c * dy)));
        dm1.add(dpdf * (-(b * dy + 0.5f * c * dx)));
        da.add(dpdf * 0.5f * dx * dx);
        db.add(dpdf * 0.5f * dy * dy);
        dc.add(dpdf * 0.5f * dx * dy);
        dc0.add(y * g0); dc1.add(y * g1); dc2.add(y * g2);
    }
    uint base = cx * N + g;
    S_part[base] = S.s;
    dmeans_part[2 * base] = dm0.s;
    dmeans_part[2 * base + 1] = dm1.s;
    dp00_part[base] = da.s;
    dp11_part[base] = db.s;
    dcross_part[base] = dc.s;
    dcolors_part[3 * base] = dc0.s;
    dcolors_part[3 * base + 1] = dc1.s;
    dcolors_part[3 * base + 2] = dc2.s;
"""

_k_minpdf = mx.fast.metal_kernel(
    name="gauss2d_minpdf",
    input_names=["means", "p00", "p11", "cross_", "xg", "yg", "sizes"],
    output_names=["minpdf_part", "argmin_part"],
    header=_HEADER,
    source=_MINPDF_SRC,
)
_k_forward = mx.fast.metal_kernel(
    name="gauss2d_forward",
    input_names=["means", "p00", "p11", "cross_", "colors", "minpdf", "xg", "yg", "sizes"],
    output_names=["acc"],
    header=_HEADER,
    source=_FORWARD_SRC,
)
_k_backward = mx.fast.metal_kernel(
    name="gauss2d_backward",
    input_names=["means", "p00", "p11", "cross_", "colors", "minpdf", "dacc", "xg", "yg", "sizes"],
    output_names=["S_part", "dmeans_part", "dp00_part", "dp11_part", "dcross_part", "dcolors_part"],
    header=_HEADER,
    source=_BACKWARD_SRC,
)

_N_CHUNKS = 64  # pixel blocks per gaussian in passes A/C (parallelism knob)
_TG_PIX = 256  # threadgroup width for the pixel-parallel pass B
_TG_G = 8  # gaussians per threadgroup in the (chunk, gaussian) passes


def _pad(n, m):
    return (n + m - 1) // m * m


# One custom_function per image size: the pixel grid is captured in the
# closure (mx.custom_function tracks only explicit array arguments, and the
# vjp must return one cotangent per primal, so constants stay out of the
# primals). N still varies freely call to call.
_CORE_CACHE = {}


def _fused_core(height, width):
    key = (height, width)
    core = _CORE_CACHE.get(key)
    if core is not None:
        return core

    num_pixels = height * width
    xg, yg = _pixel_grid(height, width, mx.float32)

    @mx.custom_function
    def core(means, p00, p11, cross, colors):
        n = means.shape[0]
        sizes = mx.array([n, num_pixels, _N_CHUNKS], dtype=mx.int32)
        minpdf_part, argmin_part = _k_minpdf(
            inputs=[means, p00, p11, cross, xg, yg, sizes],
            grid=(_N_CHUNKS, _pad(n, _TG_G), 1),
            threadgroup=(_N_CHUNKS, _TG_G, 1),
            output_shapes=[(n, _N_CHUNKS), (n, _N_CHUNKS)],
            output_dtypes=[mx.float32, mx.uint32],
        )
        j = mx.argmin(minpdf_part, axis=1, keepdims=True)  # (n, 1)
        minpdf = mx.take_along_axis(minpdf_part, j, axis=1)[:, 0]
        argmin = mx.take_along_axis(argmin_part, j, axis=1)[:, 0]
        (acc,) = _k_forward(
            inputs=[means, p00, p11, cross, colors, minpdf, xg, yg, sizes],
            grid=(_pad(num_pixels, _TG_PIX), 1, 1),
            threadgroup=(_TG_PIX, 1, 1),
            output_shapes=[(num_pixels, 3)],
            output_dtypes=[mx.float32],
        )
        return acc, minpdf, argmin

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        means, p00, p11, cross, colors = primals
        dacc = cotangents[0]  # cotangents on minpdf/argmin are zero: they
        # only leave rasterize_fused through acc.
        _, minpdf, argmin = outputs
        n = means.shape[0]
        sizes = mx.array([n, num_pixels, _N_CHUNKS], dtype=mx.int32)
        S_part, dmeans_part, dp00_part, dp11_part, dcross_part, dcolors_part = _k_backward(
            inputs=[means, p00, p11, cross, colors, minpdf, dacc, xg, yg, sizes],
            grid=(_N_CHUNKS, _pad(n, _TG_G), 1),
            threadgroup=(_N_CHUNKS, _TG_G, 1),
            output_shapes=[
                (_N_CHUNKS, n),
                (_N_CHUNKS, n, 2),
                (_N_CHUNKS, n),
                (_N_CHUNKS, n),
                (_N_CHUNKS, n),
                (_N_CHUNKS, n, 3),
            ],
            output_dtypes=[mx.float32] * 6,
        )
        S = mx.sum(S_part, axis=0)
        dmeans = mx.sum(dmeans_part, axis=0)
        dp00 = mx.sum(dp00_part, axis=0)
        dp11 = mx.sum(dp11_part, axis=0)
        dcross = mx.sum(dcross_part, axis=0)
        dcolors = mx.sum(dcolors_part, axis=0)

        # min-path: d minpdf = +S flows into pdf at the argmin pixel; these
        # are N-sized ops, done here instead of in the kernel.
        dxs = mx.take(xg, argmin) - means[:, 0]
        dys = mx.take(yg, argmin) - means[:, 1]
        dmeans = dmeans + S[:, None] * mx.stack(
            [-(p00 * dxs + 0.5 * cross * dys), -(p11 * dys + 0.5 * cross * dxs)], axis=1
        )
        dp00 = dp00 + S * 0.5 * dxs * dxs
        dp11 = dp11 + S * 0.5 * dys * dys
        dcross = dcross + S * 0.5 * dxs * dys
        return dmeans, dp00, dp11, dcross, dcolors

    _CORE_CACHE[key] = core
    return core


def rasterize_fused(
    means: mx.array,
    covariances: mx.array,
    colors: mx.array,
    background: mx.array,
    height: int,
    width: int,
):
    """Drop-in replacement for :func:`rendering2d.rasterize` (same signature,
    same ``(color, None, None)`` return), backed by the fused kernels."""
    assert means.shape[0] == covariances.shape[0] == colors.shape[0]
    assert means.dtype == mx.float32

    # cov -> precision entries in regular MLX so autodiff covers this chain
    # (identical to the dense rasterizer, including the determinant floor).
    m00, m01 = covariances[:, 0, 0], covariances[:, 0, 1]
    m10, m11 = covariances[:, 1, 0], covariances[:, 1, 1]
    det = mx.maximum(m00 * m11 - m01 * m10, _DET_EPS)
    p00 = m11 / det
    p11 = m00 / det
    cross = (-m10 - m01) / det

    return _rasterize_precision(means, p00, p11, cross, colors, background, height, width)


def rasterize_fused_cholesky(
    means: mx.array,
    log_diag: mx.array,
    offdiag: mx.array,
    colors: mx.array,
    background: mx.array,
    height: int,
    width: int,
):
    """Rasterize directly from the lower-triangular covariance factor.

    For ``L = [[a, 0], [b, c]]``, form the three unique entries of
    ``L @ L.T`` elementwise and convert them to the precision coefficients
    consumed by the fused kernel. This avoids dispatching a batched 2x2
    matrix multiplication while preserving the covariance path's determinant
    floor and regular-MLX autodiff.
    """
    assert means.shape[0] == log_diag.shape[0] == offdiag.shape[0] == colors.shape[0]
    assert log_diag.shape[1] == 2
    assert means.dtype == log_diag.dtype == offdiag.dtype == mx.float32

    diag = mx.exp(log_diag)
    a, c = diag[:, 0], diag[:, 1]
    m00 = a * a
    m01 = a * offdiag
    m11 = offdiag * offdiag + c * c
    # det(L L.T) = det(L)^2 = (a * c)^2. Computing it from the
    # Cholesky diagonal avoids cancellation between m00*m11 and m01^2 when
    # the off-diagonal is large, and is closer to the fp64 reference.
    ac = a * c
    det = mx.maximum(ac * ac, _DET_EPS)
    p00 = m11 / det
    p11 = m00 / det
    cross = -2.0 * m01 / det

    return _rasterize_precision(means, p00, p11, cross, colors, background, height, width)


def _rasterize_precision(means, p00, p11, cross, colors, background, height, width):
    """Run the fused kernels from precomputed symmetric precision entries."""

    core = _fused_core(height, width)
    acc, _, _ = core(means, p00, p11, cross, colors[:, :3])
    color = background + acc.reshape(height, width, 3)
    return color, None, None
