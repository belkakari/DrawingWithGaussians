"""Fused 2DGS normal-consistency loss for MLX/Metal.

The dense reference builds a full camera-space point image, finite-differences
it, materializes a surface-normal image, multiplies by stopped alpha, then
reduces ``1 - dot(rendered_normal, surface_normal)``. This custom primitive
keeps the same centered stencil and analytical VJP in two Metal kernels without
materializing points or surface normals.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

_BLOCK_X = 16
_BLOCK_Y = 16
_CORE_CACHE: dict[tuple[int, int, int], Any] = {}

_HEADER = r"""
inline uint pixel_index(uint b, uint y, uint x, uint H, uint W) {
    return (b * H + y) * W + x;
}

inline float3 camera_direction(float x, float y, float fx, float fy, float cx, float cy) {
    return float3((x - cx) / fx, (y - cy) / fy, 1.0f);
}
"""

_FORWARD_SRC = r"""
    uint3 pos = thread_position_in_grid;
    uint x = pos.x;
    uint y = pos.y;
    uint b = pos.z;
    uint B = (uint)sizes[0];
    uint H = (uint)sizes[1];
    uint W = (uint)sizes[2];
    if (b >= B || y >= H || x >= W) return;

    uint p = pixel_index(b, y, x, H, W);
    if (x == 0u || x + 1u >= W || y == 0u || y + 1u >= H) {
        loss_map[p] = 1.0f;
        return;
    }

    uint k = b * 9u;
    float fx = intrinsics[k];
    float fy = intrinsics[k + 4u];
    float cx = intrinsics[k + 2u];
    float cy = intrinsics[k + 5u];
    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;

    uint pu = pixel_index(b, y - 1u, x, H, W);
    uint pd = pixel_index(b, y + 1u, x, H, W);
    uint pl = pixel_index(b, y, x - 1u, H, W);
    uint pr = pixel_index(b, y, x + 1u, H, W);
    float3 dir_u = camera_direction(px, py - 1.0f, fx, fy, cx, cy);
    float3 dir_d = camera_direction(px, py + 1.0f, fx, fy, cx, cy);
    float3 dir_l = camera_direction(px - 1.0f, py, fx, fy, cx, cy);
    float3 dir_r = camera_direction(px + 1.0f, py, fx, fy, cx, cy);
    float3 dx = depths[pd] * dir_d - depths[pu] * dir_u;
    float3 dy = depths[pr] * dir_r - depths[pl] * dir_l;
    float3 raw = metal::cross(dx, dy);
    float inv_len = 1.0f / metal::sqrt(metal::dot(raw, raw) + 1e-20f);
    float3 surface = raw * inv_len;
    float3 rendered = float3(
        rendered_normals[3u * p],
        rendered_normals[3u * p + 1u],
        rendered_normals[3u * p + 2u]
    );
    loss_map[p] = 1.0f - alpha[p] * metal::dot(rendered, surface);
"""

_BACKWARD_SRC = r"""
    uint3 pos = thread_position_in_grid;
    uint x = pos.x;
    uint y = pos.y;
    uint b = pos.z;
    uint B = (uint)sizes[0];
    uint H = (uint)sizes[1];
    uint W = (uint)sizes[2];
    if (b >= B || y >= H || x >= W) return;
    if (x == 0u || x + 1u >= W || y == 0u || y + 1u >= H) return;

    uint p = pixel_index(b, y, x, H, W);
    uint k = b * 9u;
    float fx = intrinsics[k];
    float fy = intrinsics[k + 4u];
    float cx = intrinsics[k + 2u];
    float cy = intrinsics[k + 5u];
    float px = (float)x + 0.5f;
    float py = (float)y + 0.5f;

    uint pu = pixel_index(b, y - 1u, x, H, W);
    uint pd = pixel_index(b, y + 1u, x, H, W);
    uint pl = pixel_index(b, y, x - 1u, H, W);
    uint pr = pixel_index(b, y, x + 1u, H, W);
    float3 dir_u = camera_direction(px, py - 1.0f, fx, fy, cx, cy);
    float3 dir_d = camera_direction(px, py + 1.0f, fx, fy, cx, cy);
    float3 dir_l = camera_direction(px - 1.0f, py, fx, fy, cx, cy);
    float3 dir_r = camera_direction(px + 1.0f, py, fx, fy, cx, cy);
    float3 dx = depths[pd] * dir_d - depths[pu] * dir_u;
    float3 dy = depths[pr] * dir_r - depths[pl] * dir_l;
    float3 raw = metal::cross(dx, dy);
    float len = metal::sqrt(metal::dot(raw, raw) + 1e-20f);
    float3 surface = raw / len;
    float3 rendered = float3(
        rendered_normals[3u * p],
        rendered_normals[3u * p + 1u],
        rendered_normals[3u * p + 2u]
    );
    float chain = dloss_map[p];
    float stopped_alpha = alpha[p];

    float3 v_surface = -chain * stopped_alpha * rendered;
    float3 v_raw = (v_surface - surface * metal::dot(surface, v_surface)) / len;
    float3 v_dx = metal::cross(dy, v_raw);
    float3 v_dy = metal::cross(v_raw, dx);

    atomic_fetch_add_explicit(&ddepths[pu], -metal::dot(v_dx, dir_u), metal::memory_order_relaxed);
    atomic_fetch_add_explicit(&ddepths[pd], metal::dot(v_dx, dir_d), metal::memory_order_relaxed);
    atomic_fetch_add_explicit(&ddepths[pl], -metal::dot(v_dy, dir_l), metal::memory_order_relaxed);
    atomic_fetch_add_explicit(&ddepths[pr], metal::dot(v_dy, dir_r), metal::memory_order_relaxed);

    float3 v_rendered = -chain * stopped_alpha * surface;
    atomic_fetch_add_explicit(&drendered_normals[3u * p], v_rendered.x, metal::memory_order_relaxed);
    atomic_fetch_add_explicit(&drendered_normals[3u * p + 1u], v_rendered.y, metal::memory_order_relaxed);
    atomic_fetch_add_explicit(&drendered_normals[3u * p + 2u], v_rendered.z, metal::memory_order_relaxed);
"""

_K_FORWARD = mx.fast.metal_kernel(
    name="normal_consistency_forward",
    input_names=["depths", "rendered_normals", "alpha", "intrinsics", "sizes"],
    output_names=["loss_map"],
    header=_HEADER,
    source=_FORWARD_SRC,
)

_K_BACKWARD = mx.fast.metal_kernel(
    name="normal_consistency_backward",
    input_names=["depths", "rendered_normals", "alpha", "intrinsics", "dloss_map", "sizes"],
    output_names=["ddepths", "drendered_normals"],
    header=_HEADER,
    source=_BACKWARD_SRC,
    atomic_outputs=True,
)


def _pad(value: int, block: int) -> int:
    return (value + block - 1) // block * block


def _core(batch: int, height: int, width: int):
    key = (batch, height, width)
    cached = _CORE_CACHE.get(key)
    if cached is not None:
        return cached
    grid = (_pad(width, _BLOCK_X), _pad(height, _BLOCK_Y), batch)
    threadgroup = (_BLOCK_X, _BLOCK_Y, 1)
    depth_shape = (batch, height, width, 1)
    normal_shape = (batch, height, width, 3)
    map_shape = (batch, height, width)

    @mx.custom_function
    def core(depths, rendered_normals, alpha, intrinsics):
        sizes = mx.array([batch, height, width], dtype=mx.uint32)
        return _K_FORWARD(  # type: ignore[operator]
            inputs=[depths, rendered_normals, alpha, intrinsics, sizes],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[map_shape],
            output_dtypes=[mx.float32],
        )[0]

    @core.vjp
    def core_vjp(primals, cotangents, _outputs):
        depths, rendered_normals, alpha, intrinsics = primals
        sizes = mx.array([batch, height, width], dtype=mx.uint32)
        ddepths, drendered_normals = _K_BACKWARD(  # type: ignore[operator]
            inputs=[depths, rendered_normals, alpha, intrinsics, cotangents, sizes],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[depth_shape, normal_shape],
            output_dtypes=[mx.float32, mx.float32],
            init_value=0,
        )
        return ddepths, drendered_normals, mx.zeros_like(alpha), mx.zeros_like(intrinsics)

    _CORE_CACHE[key] = core
    return core


def normal_consistency_loss_fused(depths, rendered_normals, alpha, intrinsics):
    """Return the mean 2DGS normal-consistency loss with an analytical VJP."""
    if depths.ndim == 3:
        depths = depths[None]
        rendered_normals = rendered_normals[None]
        alpha = alpha[None]
        intrinsics = intrinsics[None] if intrinsics.ndim == 2 else intrinsics
    if depths.ndim != 4 or depths.shape[-1] != 1:
        raise ValueError(f"depths must have shape HxWx1 or BxHxWx1, got {depths.shape}")
    batch, height, width, _ = depths.shape
    if rendered_normals.shape != (batch, height, width, 3):
        raise ValueError("rendered_normals must match depths with three channels")
    if alpha.shape != depths.shape:
        raise ValueError("alpha must match depths")
    if intrinsics.shape != (batch, 3, 3):
        raise ValueError("intrinsics must have shape 3x3 or Bx3x3 and match the batch")
    if height < 3 or width < 3:
        return mx.ones((), dtype=depths.dtype)
    loss_map = _core(batch, height, width)(depths, rendered_normals, mx.stop_gradient(alpha), intrinsics)
    return mx.mean(loss_map)
