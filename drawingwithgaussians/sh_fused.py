"""Fused Metal degree-0..3 view-dependent spherical harmonics.

The public SH contract intentionally stops direction gradients. That makes a
compact custom primitive possible: forward fuses camera-center recovery,
direction normalization, basis evaluation, and ReLU; backward assigns one
thread per Gaussian and reduces shared coefficient gradients across cameras.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_TG = 256
_CACHE: dict[tuple[int, int, int], Any] = {}

_HEADER = r"""
constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[5] = {
    1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
    -1.0925484305920792f, 0.5462742152960396f
};
constant float SH_C3[7] = {
    -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
    0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
    -0.5900435899266435f
};

inline float3 sh_direction(const device float* means, const device float* viewmats, uint gid, uint cam) {
    uint v = cam * 16u;
    float tx = viewmats[v + 3u], ty = viewmats[v + 7u], tz = viewmats[v + 11u];
    float3 center = -float3(
        viewmats[v] * tx + viewmats[v + 4u] * ty + viewmats[v + 8u] * tz,
        viewmats[v + 1u] * tx + viewmats[v + 5u] * ty + viewmats[v + 9u] * tz,
        viewmats[v + 2u] * tx + viewmats[v + 6u] * ty + viewmats[v + 10u] * tz
    );
    float3 dir = float3(means[3u * gid], means[3u * gid + 1u], means[3u * gid + 2u]) - center;
    return dir / metal::max(metal::length(dir), 1e-8f);
}

inline void sh_bases(float3 dir, uint degree, thread float* bases) {
    float x = dir.x, y = dir.y, z = dir.z;
    bases[0] = SH_C0;
    if (degree == 0u) return;
    bases[1] = -SH_C1 * y;
    bases[2] = SH_C1 * z;
    bases[3] = -SH_C1 * x;
    if (degree == 1u) return;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    bases[4] = SH_C2[0] * xy;
    bases[5] = SH_C2[1] * yz;
    bases[6] = SH_C2[2] * (2.0f * zz - xx - yy);
    bases[7] = SH_C2[3] * xz;
    bases[8] = SH_C2[4] * (xx - yy);
    if (degree == 2u) return;
    bases[9] = SH_C3[0] * y * (3.0f * xx - yy);
    bases[10] = SH_C3[1] * xy * z;
    bases[11] = SH_C3[2] * y * (4.0f * zz - xx - yy);
    bases[12] = SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    bases[13] = SH_C3[4] * x * (4.0f * zz - xx - yy);
    bases[14] = SH_C3[5] * z * (xx - yy);
    bases[15] = SH_C3[6] * x * (xx - 3.0f * yy);
}
"""

_FORWARD = r"""
    uint idx = thread_position_in_grid.x;
    uint B = (uint)sizes[0], N = (uint)sizes[1], degree = (uint)sizes[2];
    if (idx >= B * N) return;
    uint cam = idx / N, gid = idx % N;
    float bases[16];
    sh_bases(sh_direction(means, viewmats, gid, cam), degree, bases);
    uint n_bases = (degree + 1u) * (degree + 1u);
    for (uint channel = 0u; channel < 3u; ++channel) {
        float value = 0.5f + bases[0] * sh0[3u * gid + channel];
        for (uint basis = 1u; basis < n_bases; ++basis) {
            value += bases[basis] * shN[45u * gid + 3u * (basis - 1u) + channel];
        }
        colors[3u * idx + channel] = metal::max(value, 0.0f);
    }
"""

_BACKWARD = r"""
    uint gid = thread_position_in_grid.x;
    uint B = (uint)sizes[0], N = (uint)sizes[1], degree = (uint)sizes[2];
    if (gid >= N) return;
    uint n_bases = (degree + 1u) * (degree + 1u);
    float accum[48];
    for (uint i = 0u; i < 48u; ++i) accum[i] = 0.0f;
    for (uint cam = 0u; cam < B; ++cam) {
        uint idx = cam * N + gid;
        float bases[16];
        sh_bases(sh_direction(means, viewmats, gid, cam), degree, bases);
        for (uint channel = 0u; channel < 3u; ++channel) {
            float chain = colors[3u * idx + channel] > 0.0f ? dcolors[3u * idx + channel] : 0.0f;
            accum[channel] += bases[0] * chain;
            for (uint basis = 1u; basis < n_bases; ++basis) {
                accum[3u * basis + channel] += bases[basis] * chain;
            }
        }
    }
    for (uint channel = 0u; channel < 3u; ++channel) dsh0[3u * gid + channel] = accum[channel];
    for (uint basis = 1u; basis < n_bases; ++basis) {
        for (uint channel = 0u; channel < 3u; ++channel) {
            dshN[45u * gid + 3u * (basis - 1u) + channel] = accum[3u * basis + channel];
        }
    }
"""

_K_FORWARD = mx.fast.metal_kernel(
    name="view_dependent_sh_forward",
    input_names=["means", "sh0", "shN", "viewmats", "sizes"],
    output_names=["colors"],
    header=_HEADER,
    source=_FORWARD,
)

_K_BACKWARD = mx.fast.metal_kernel(
    name="view_dependent_sh_backward",
    input_names=["means", "viewmats", "colors", "dcolors", "sizes"],
    output_names=["dsh0", "dshN"],
    header=_HEADER,
    source=_BACKWARD,
)


def _pad(value: int) -> int:
    return (value + _TG - 1) // _TG * _TG


def _core(batch: int, n: int, degree: int):
    key = (batch, n, degree)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    @mx.custom_function
    def core(means, sh0, shN, viewmats):
        sizes = mx.array([batch, n, degree], dtype=mx.uint32)
        return _K_FORWARD(  # type: ignore[operator]
            inputs=[means, sh0, shN, viewmats, sizes],
            grid=(_pad(batch * n), 1, 1),
            threadgroup=(_TG, 1, 1),
            output_shapes=[(batch, n, 3)],
            output_dtypes=[mx.float32],
        )[0]

    @core.vjp
    def core_vjp(primals, cotangents, outputs):
        means, sh0, shN, viewmats = primals
        sizes = mx.array([batch, n, degree], dtype=mx.uint32)
        dsh0, dshN = _K_BACKWARD(  # type: ignore[operator]
            inputs=[means, viewmats, outputs, cotangents, sizes],
            grid=(_pad(n), 1, 1),
            threadgroup=(_TG, 1, 1),
            output_shapes=[sh0.shape, shN.shape],
            output_dtypes=[mx.float32, mx.float32],
            init_value=0,
        )
        return mx.zeros_like(means), dsh0, dshN, mx.zeros_like(viewmats)

    _CACHE[key] = core
    return core


def view_dependent_colors_fused(means, sh0, shN, viewmats, degree: int):
    """Evaluate stopped-direction SH colors for one or more cameras."""
    degree = int(degree)
    if not 0 <= degree <= 3:
        raise ValueError("SH degree must be in [0, 3]")
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError("means must have shape Nx3")
    n = means.shape[0]
    if sh0.shape != (n, 1, 3) or shN.shape != (n, 15, 3):
        raise ValueError("SH coefficients must have shapes Nx1x3 and Nx15x3")
    batched = viewmats.ndim == 3
    if not batched:
        viewmats = viewmats[None]
    if viewmats.ndim != 3 or viewmats.shape[-2:] != (4, 4):
        raise ValueError("viewmats must have shape 4x4 or Bx4x4")
    colors = _core(viewmats.shape[0], n, degree)(means, sh0, shN, viewmats)
    return colors if batched else colors[0]
