"""Sparse TSDF fusion with MLX-streamed voxel evaluation."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import numpy as np
from plyfile import PlyData, PlyElement

from .spatial_hash_metal import unique_int3_metal

_CORNERS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int32)
_TETS = ((0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7))
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _surface_blocks(depth, alpha, K, w2c, voxel_size, truncation, block_resolution, alpha_threshold):
    valid = np.isfinite(depth) & (depth > 0) & (np.asarray(alpha).squeeze() >= alpha_threshold)
    y, x = np.nonzero(valid)
    if not len(x):
        return np.empty((0, 3), dtype=np.int32)
    d = mx.array(np.asarray(depth, dtype=np.float32)[y, x])
    K = np.asarray(K, dtype=np.float32)
    camera = mx.stack(
        [
            (mx.array(x, dtype=mx.float32) + 0.5 - K[0, 2]) * d / K[0, 0],
            (mx.array(y, dtype=mx.float32) + 0.5 - K[1, 2]) * d / K[1, 1],
            d,
        ],
        axis=1,
    )
    w2c = np.asarray(w2c, dtype=np.float32)
    world = (camera - mx.array(w2c[:3, 3])) @ mx.array(w2c[:3, :3])
    blocks = mx.floor(world / float(voxel_size * block_resolution)).astype(mx.int32)
    blocks = unique_int3_metal(blocks)
    radius = max(1, int(np.ceil(truncation / (voxel_size * block_resolution))))
    neighbors = np.array(list(itertools.product(range(-radius, radius + 1), repeat=3)), dtype=np.int32)
    expanded = (blocks[:, None, :] + neighbors[None, :, :]).reshape(-1, 3)
    return unique_int3_metal(expanded)


def _evaluate_blocks(
    block_coords,
    depths,
    alphas,
    rgbs,
    Ks,
    w2cs,
    voxel_size,
    truncation,
    alpha_threshold,
    block_resolution,
    block_batch,
):
    local = mx.array(np.array(list(itertools.product(range(block_resolution), repeat=3)), dtype=np.float32))
    blocks = {}
    for start in range(0, len(block_coords), block_batch):
        coords_np = block_coords[start : start + block_batch]
        coords = mx.array(coords_np.astype(np.float32))
        voxel_indices = coords[:, None, :] * block_resolution + local[None, :, :]
        points = (voxel_indices.reshape(-1, 3) + 0.5) * voxel_size
        tsdf_sum = mx.zeros((len(points),), dtype=mx.float32)
        weight = mx.zeros_like(tsdf_sum)
        color_sum = mx.zeros((len(points), 3), dtype=mx.float32)
        for depth_np, alpha_np, rgb_np, K_np, w2c_np in zip(depths, alphas, rgbs, Ks, w2cs, strict=True):
            depth = mx.array(np.asarray(depth_np, dtype=np.float32))
            alpha = mx.array(np.asarray(alpha_np, dtype=np.float32).squeeze())
            rgb = mx.array(np.asarray(rgb_np, dtype=np.float32))
            K = np.asarray(K_np, dtype=np.float32)
            w2c = np.asarray(w2c_np, dtype=np.float32)
            camera = points @ mx.array(w2c[:3, :3]).T + mx.array(w2c[:3, 3])
            z = camera[:, 2]
            safe_z = mx.maximum(z, 1e-8)
            u = mx.floor(K[0, 0] * camera[:, 0] / safe_z + K[0, 2]).astype(mx.int32)
            v = mx.floor(K[1, 1] * camera[:, 1] / safe_z + K[1, 2]).astype(mx.int32)
            height, width = depth.shape
            inside = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            flat_index = mx.clip(v, 0, height - 1) * width + mx.clip(u, 0, width - 1)
            observed = mx.take(depth.reshape(-1), flat_index)
            observed_alpha = mx.take(alpha.reshape(-1), flat_index)
            observed_color = mx.take(rgb.reshape(-1, 3), flat_index, axis=0)
            sdf = observed - z
            valid = inside & mx.isfinite(observed) & (observed > 0) & (observed_alpha >= alpha_threshold)
            valid = valid & (sdf >= -truncation)
            contribution = valid.astype(mx.float32)
            tsdf_sum = tsdf_sum + contribution * mx.clip(sdf / truncation, -1.0, 1.0)
            weight = weight + contribution
            color_sum = color_sum + contribution[:, None] * observed_color
        tsdf = tsdf_sum / mx.maximum(weight, 1.0)
        color = color_sum / mx.maximum(weight[:, None], 1.0)
        mx.eval(tsdf, weight, color)
        shape = (len(coords_np), block_resolution, block_resolution, block_resolution)
        tsdf_np = np.asarray(tsdf).reshape(shape)
        weight_np = np.asarray(weight).reshape(shape)
        color_np = np.asarray(color).reshape(shape + (3,))
        for index, coord in enumerate(coords_np):
            blocks[tuple(int(v) for v in coord)] = (tsdf_np[index], weight_np[index], color_np[index])
    return blocks


def _extract_tetrahedra(blocks, voxel_size, block_resolution):
    def sample(index):
        index = np.asarray(index, dtype=np.int32)
        block = tuple(np.floor_divide(index, block_resolution).tolist())
        payload = blocks.get(block)
        if payload is None:
            return None
        local = tuple(np.mod(index, block_resolution).tolist())
        return payload[0][local], payload[1][local], payload[2][local]

    vertices, colors, faces, edge_vertices = [], [], [], {}

    def vertex_for(a_index, b_index, a_value, b_value, a_color, b_color):
        a_key, b_key = tuple(a_index.tolist()), tuple(b_index.tolist())
        edge = tuple(sorted((a_key, b_key)))
        if edge in edge_vertices:
            return edge_vertices[edge]
        fraction = float(a_value / (a_value - b_value))
        position = (a_index + 0.5 + fraction * (b_index - a_index)) * voxel_size
        color = a_color + fraction * (b_color - a_color)
        edge_vertices[edge] = len(vertices)
        vertices.append(position)
        colors.append(color)
        return edge_vertices[edge]

    for block_coord, payload in blocks.items():
        if not np.any(payload[1] > 0):
            continue
        base = np.asarray(block_coord, dtype=np.int32) * block_resolution
        for local in itertools.product(range(block_resolution), repeat=3):
            cell = base + np.asarray(local, dtype=np.int32)
            corner_indices = cell[None, :] + _CORNERS
            samples = [sample(index) for index in corner_indices]
            if any(value is None or value[1] <= 0 for value in samples):
                continue
            values = np.array([value[0] for value in samples])
            if values.min() > 0 or values.max() < 0:
                continue
            corner_colors = np.stack([value[2] for value in samples])
            for tet in _TETS:
                crossings = []
                for a, b in _TET_EDGES:
                    ia, ib = tet[a], tet[b]
                    if (values[ia] < 0) == (values[ib] < 0):
                        continue
                    crossings.append(
                        vertex_for(
                            corner_indices[ia],
                            corner_indices[ib],
                            values[ia],
                            values[ib],
                            corner_colors[ia],
                            corner_colors[ib],
                        )
                    )
                if len(crossings) == 3:
                    faces.append(crossings)
                elif len(crossings) == 4:
                    faces.extend((crossings[:3], (crossings[0], crossings[2], crossings[3])))
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )


def fuse_tsdf_mlx(
    depths: Sequence[np.ndarray],
    alphas: Sequence[np.ndarray],
    rgbs: Sequence[np.ndarray],
    Ks: Sequence[np.ndarray],
    w2cs: Sequence[np.ndarray],
    normalized_to_original: np.ndarray,
    output_mesh: str | Path,
    voxel_size: float = 0.004,
    truncation: float = 0.02,
    alpha_threshold: float = 0.5,
    block_resolution: int = 8,
    block_batch: int = 128,
) -> Path:
    active = []
    for args in zip(depths, alphas, Ks, w2cs, strict=True):
        active.append(_surface_blocks(*args, voxel_size, truncation, block_resolution, alpha_threshold))
    block_coords = unique_int3_metal(np.concatenate(active, axis=0))
    blocks = _evaluate_blocks(
        block_coords,
        depths,
        alphas,
        rgbs,
        Ks,
        w2cs,
        voxel_size,
        truncation,
        alpha_threshold,
        block_resolution,
        block_batch,
    )
    vertices, colors, faces = _extract_tetrahedra(blocks, voxel_size, block_resolution)
    if not len(vertices) or not len(faces):
        raise RuntimeError("MLX TSDF fusion produced an empty mesh")
    transform = np.asarray(normalized_to_original, dtype=np.float64)
    vertices = np.column_stack([vertices, np.ones(len(vertices))]) @ transform.T
    vertices = vertices[:, :3].astype(np.float32)
    vertex = np.empty(
        len(vertices),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    for index, axis in enumerate("xyz"):
        vertex[axis] = vertices[:, index]
    color_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    for index, channel in enumerate(("red", "green", "blue")):
        vertex[channel] = color_u8[:, index]
    face = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face["vertex_indices"] = faces
    output = Path(output_mesh)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")], text=False).write(output)
    return output
