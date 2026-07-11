from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from plyfile import PlyData

from drawingwithgaussians.gaussian3d import densify_masks
from drawingwithgaussians.losses import distortion_l1_loss
from drawingwithgaussians.lpips_mlx import LPIPSAlex
from drawingwithgaussians.photometric import apply_photometric, init_photometric, photometric_identity_regularizer
from drawingwithgaussians.rendering2dgs import project_gaussians_2dgs
from drawingwithgaussians.selective_adam import SelectiveAdam, selective_adam_update
from drawingwithgaussians.sh import eval_sh, rgb_to_sh0, view_dependent_colors
from drawingwithgaussians.spatial_hash_metal import HashTableOverflow, unique_int3_metal
from drawingwithgaussians.splat_export import load_ply_3d, ply_bytes_3d
from drawingwithgaussians.utilization import init_utilization, pruning_window, remap_utilization, update_utilization


def test_degree_zero_conversion_forward_parity_and_gradient():
    rng = np.random.default_rng(3)
    rgb = mx.array(rng.uniform(0.01, 0.99, (8, 3)).astype(np.float32))
    sh0 = rgb_to_sh0(rgb)
    shN = mx.zeros((8, 15, 3))
    dirs = mx.array(rng.normal(size=(8, 3)).astype(np.float32))
    dirs = dirs / mx.linalg.norm(dirs, axis=-1, keepdims=True)
    out = eval_sh(0, sh0, shN, dirs)
    np.testing.assert_allclose(np.asarray(out), np.asarray(rgb), atol=2e-7)

    def objective(dc, rest):
        return mx.sum(eval_sh(3, dc, rest, dirs))

    grads = mx.grad(objective, argnums=(0, 1))(sh0, shN)
    mx.eval(*grads)
    assert bool(mx.all(mx.isfinite(grads[0]))) and bool(mx.all(mx.isfinite(grads[1])))
    assert float(mx.max(mx.abs(grads[1]))) > 0


def test_lpips_alex_torchmetrics_parity_fixture():
    rng = np.random.default_rng(0)
    a = rng.random((2, 64, 64, 3), dtype=np.float32)
    b = rng.random((2, 64, 64, 3), dtype=np.float32)
    values = LPIPSAlex()(a, b)
    mx.eval(values)
    np.testing.assert_allclose(np.asarray(values), [0.14474462, 0.11879930], atol=2e-7, rtol=0)


def test_metal_int3_hash_set_duplicates_collisions_and_signed_values():
    rng = np.random.default_rng(21)
    source = rng.integers(-1000, 1000, size=(2000, 3), dtype=np.int32)
    source = np.concatenate([source, source[:500], np.zeros((20, 3), dtype=np.int32)])
    expected = {tuple(row) for row in source.tolist()}
    first = unique_int3_metal(source)
    second = unique_int3_metal(source)
    assert {tuple(row) for row in first.tolist()} == expected
    np.testing.assert_array_equal(first, second)


def test_metal_int3_hash_set_overflow_telemetry():
    coords = np.arange(48, dtype=np.int32).reshape(16, 3)
    with np.testing.assert_raises(HashTableOverflow):
        unique_int3_metal(coords, capacity=8)


def test_sh_batched_views_and_world_frame_invariance():
    rng = np.random.default_rng(4)
    n = 7
    params = {
        "means3d": mx.array(rng.normal(size=(n, 3)).astype(np.float32)),
        "sh0": mx.array(rng.normal(size=(n, 1, 3)).astype(np.float32)),
        "shN": mx.array(rng.normal(size=(n, 15, 3)).astype(np.float32) * 0.05),
    }
    viewmats = mx.array(np.stack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)]))
    colors = view_dependent_colors(params, viewmats, 3)
    assert colors.shape == (2, n, 3)
    np.testing.assert_allclose(np.asarray(colors[0]), np.asarray(colors[1]), atol=1e-6)


def test_ply_sh_channel_major_ordering():
    n = 2
    rest = np.arange(n * 15 * 3, dtype=np.float32).reshape(n, 15, 3)
    params = {
        "means3d": mx.zeros((n, 3)),
        "log_scales": mx.zeros((n, 3)),
        "quats": mx.array(np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)),
        "opacities_raw": mx.zeros((n,)),
        "sh0": mx.zeros((n, 1, 3)),
        "shN": mx.array(rest),
    }
    vertex = PlyData.read(BytesIO(ply_bytes_3d(params)))["vertex"]
    loaded = np.column_stack([vertex[f"f_rest_{i}"] for i in range(45)])
    np.testing.assert_array_equal(loaded, rest.transpose(0, 2, 1).reshape(n, 45))


def test_ply_roundtrip_through_local_mlx3d(tmp_path: Path):
    local = Path("/Users/glebsterkin/repos/mlx3D/src")
    if not local.is_dir():
        return
    n = 3
    rng = np.random.default_rng(15)
    params = {
        "means3d": mx.array(rng.normal(size=(n, 3)).astype(np.float32)),
        "log_scales": mx.array(rng.normal(size=(n, 3)).astype(np.float32)),
        "quats": mx.array(rng.normal(size=(n, 4)).astype(np.float32)),
        "opacities_raw": mx.array(rng.normal(size=(n,)).astype(np.float32)),
        "sh0": mx.array(rng.normal(size=(n, 1, 3)).astype(np.float32)),
        "shN": mx.array(rng.normal(size=(n, 15, 3)).astype(np.float32)),
    }
    path = tmp_path / "model.ply"
    path.write_bytes(ply_bytes_3d(params))
    sys.path.insert(0, str(local))
    try:
        from mlx3d.splatting.model import GaussianModel

        loaded = GaussianModel.load_ply(str(path), sh_degree=3)
        mx.eval(loaded.params)
        np.testing.assert_allclose(np.asarray(loaded.params["sh_dc"]), np.asarray(params["sh0"]))
        np.testing.assert_allclose(np.asarray(loaded.params["sh_rest"]), np.asarray(params["shN"]))
    finally:
        sys.path.remove(str(local))


def test_ply_roundtrip_through_local_pipeline(tmp_path: Path):
    rng = np.random.default_rng(16)
    n = 4
    params = {
        "means3d": mx.array(rng.normal(size=(n, 3)).astype(np.float32)),
        "log_scales": mx.array(rng.normal(size=(n, 3)).astype(np.float32)),
        "quats": mx.array(rng.normal(size=(n, 4)).astype(np.float32)),
        "opacities_raw": mx.array(rng.normal(size=n).astype(np.float32)),
        "sh0": mx.array(rng.normal(size=(n, 1, 3)).astype(np.float32)),
        "shN": mx.array(rng.normal(size=(n, 15, 3)).astype(np.float32)),
    }
    path = tmp_path / "model.ply"
    path.write_bytes(ply_bytes_3d(params))
    loaded = load_ply_3d(path)
    mx.eval(loaded)
    for name in ("means3d", "log_scales", "opacities_raw", "sh0", "shN"):
        np.testing.assert_allclose(np.asarray(loaded[name]), np.asarray(params[name]), atol=1e-6)
    expected_quats = np.asarray(params["quats"])
    expected_quats /= np.linalg.norm(expected_quats, axis=1, keepdims=True)
    np.testing.assert_allclose(np.asarray(loaded["quats"]), expected_quats, atol=1e-6)


def test_selective_adam_all_visible_matches_mlx_adam():
    rng = np.random.default_rng(5)
    p0 = mx.array(rng.normal(size=(10, 3)).astype(np.float32))
    dense_p, selective_p = p0, p0
    dense = optim.Adam(learning_rate=2e-3, bias_correction=True)
    dense.init({"p": dense_p})
    selective = SelectiveAdam({"p": 2e-3})
    selective.init({"p": selective_p})
    visible = mx.ones((10,), dtype=mx.bool_)
    for _ in range(12):
        grad = mx.array(rng.normal(size=(10, 3)).astype(np.float32))
        dense_p = dense.apply_gradients({"p": grad}, {"p": dense_p})["p"]
        selective_p = selective.apply_gradients({"p": grad}, {"p": selective_p}, visible)["p"]
        mx.eval(dense_p, selective_p)
    np.testing.assert_allclose(np.asarray(selective_p), np.asarray(dense_p), rtol=2e-5, atol=2e-6)


def test_selective_adam_hidden_rows_and_remap():
    p = mx.arange(12, dtype=mx.float32).reshape(4, 3)
    g = mx.ones_like(p)
    m = mx.zeros_like(p)
    v = mx.zeros_like(p)
    counters = mx.zeros((4,), dtype=mx.uint32)
    mask = mx.array([True, False, True, False])
    out, m1, v1, c1 = selective_adam_update(p, g, m, v, counters, mask, 1e-2)
    mx.eval(out, m1, v1, c1)
    np.testing.assert_array_equal(np.asarray(out)[[1, 3]], np.asarray(p)[[1, 3]])
    np.testing.assert_array_equal(np.asarray(m1)[[1, 3]], 0)
    np.testing.assert_array_equal(np.asarray(c1), [1, 0, 1, 0])

    opt = SelectiveAdam({"p": 1e-2})
    opt.init({"p": p})
    opt.apply_gradients({"p": g}, {"p": p}, mask)
    opt.remap(np.array([0, 2]), num_new=1)
    mx.eval(opt.state)
    np.testing.assert_array_equal(np.asarray(opt.state["counters"]), [1, 1, 0])


def test_utilization_proxy_lifecycle_and_child_inheritance():
    state = init_utilization(3)
    state = update_utilization(
        state,
        mx.array([0.1, 0.0, 0.2]),
        mx.zeros((3,)),
        mx.array([1.0, 0.0, 2.0]),
        batch=2,
        height=10,
        width=10,
        ema_decay=0.0,
    )
    mask, state = pruning_window(
        state, threshold=1e9, warmup_steps=1, minimum_observations=1, grace_steps=1, repeated_windows=1
    )
    np.testing.assert_array_equal(mask, [True, False, True])
    remapped = remap_utilization(state, np.array([0, 1]), np.array([2]))
    mx.eval(remapped)
    assert float(remapped["ema"][2]) == float(state["ema"][2])
    assert int(remapped["age"][2]) == 0
    assert int(remapped["consecutive_low"][2]) == 0


def test_photometric_identity_and_camera_selection():
    params = init_photometric(3)
    colors = mx.array(np.full((2, 4, 3), 0.25, np.float32))
    corrected = apply_photometric(colors, params, mx.array([0, 2]))
    np.testing.assert_array_equal(np.asarray(corrected), np.asarray(colors))
    assert float(photometric_identity_regularizer(params)) == 0.0
    params["log_gain"] = params["log_gain"].at[2, 0].add(np.log(2.0))
    params["bias"] = params["bias"].at[2, 1].add(0.1)
    corrected = apply_photometric(colors, params, mx.array([0, 2]))
    np.testing.assert_allclose(np.asarray(corrected[0]), 0.25)
    np.testing.assert_allclose(np.asarray(corrected[1, :, 0]), 0.5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(corrected[1, :, 1]), 0.35, atol=1e-6)


def test_2dgs_third_scale_normal_invariance_and_shared_pruning():
    means = mx.array([[0.0, 0.0, 2.0]], dtype=mx.float32)
    quats = mx.array([[1.0, 0.0, 0.0, 0.0]], dtype=mx.float32)
    scales_a = mx.log(mx.array([[0.2, 0.3, 0.01]], dtype=mx.float32))
    scales_b = mx.log(mx.array([[0.2, 0.3, 2.0]], dtype=mx.float32))
    view = mx.eye(4)
    K = mx.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1]], dtype=mx.float32)
    a = project_gaussians_2dgs(means, scales_a, quats, view, K, 64, 64)
    b = project_gaussians_2dgs(means, scales_b, quats, view, K, 64, 64)
    mx.eval(*a, *b)
    for idx in (0, 1, 2, 3, 4):
        np.testing.assert_allclose(np.asarray(a[idx]), np.asarray(b[idx]), atol=1e-6)

    base = {
        "means3d": np.zeros((1, 3), np.float32),
        "quats": np.array([[1, 0, 0, 0]], np.float32),
        "opacities_raw": np.array([5.0], np.float32),
        "sh0": np.zeros((1, 1, 3), np.float32),
        "shN": np.zeros((1, 15, 3), np.float32),
    }
    p_a, p_b = {**base, "log_scales": np.asarray(scales_a)}, {**base, "log_scales": np.asarray(scales_b)}
    assert not densify_masks(p_a, np.zeros(1), 1, 0.01, 1, 0.01, 0.5)[4][0]
    assert densify_masks(p_b, np.zeros(1), 1, 0.01, 1, 0.01, 0.5)[4][0]


def test_2dgs_distortion_l1_rejects_signed_crossing_residuals():
    residuals = mx.array([-2.0, -0.01, 3.0, 4.0], dtype=mx.float32)
    loss = distortion_l1_loss(residuals)
    grad = mx.grad(distortion_l1_loss)(residuals)
    mx.eval(loss, grad)
    np.testing.assert_allclose(float(loss), 2.2525)
    np.testing.assert_array_equal(np.asarray(grad), [-0.25, -0.25, 0.25, 0.25])
