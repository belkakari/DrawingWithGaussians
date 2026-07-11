"""Unit tests for train_colmap3d.py trainer helpers (Stage 1: eval path).

Run:
    uv run pytest tests/test_trainer_units.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import drawingwithgaussians.rendering2dgs_fused as fused2dgs
import drawingwithgaussians.rendering3d_fused as fused3d
from drawingwithgaussians.gaussian3d import (
    carry_optimizer_state_3d,
    get_opt_step,
    get_param_state,
    reset_opacities_3d,
    set_up_optimizer_3d,
    zero_param_moments,
)
from drawingwithgaussians.losses import pixel_loss_2dgs, pixel_loss_3d
from drawingwithgaussians.sh import rgb_to_sh0, view_dependent_colors
from train_colmap3d import (
    ColmapScene,
    _camera_matrices_at_resolution,
    _capacity_for_count,
    _evaluate,
    _intersection_counts,
    _parse_overflow_mode,
    _regularizer_weight,
    _render_view,
    _ViewSampler,
    eval_capacity,
)

H, W = 64, 96


def test_camera_matrices_at_independent_render_resolution(tmp_path: Path):
    paths = [tmp_path / f"clean_{view:03d}_3_r5000.png" for view in (2, 3)]
    for path in paths:
        Image.new("RGB", (80, 60)).save(path)
    K = np.array([[100.0, 0, 40], [0, 120.0, 30], [0, 0, 1]], dtype=np.float32)
    scene = ColmapScene(
        paths,
        np.stack([np.eye(4, dtype=np.float32)] * 2),
        [K.copy(), K.copy()],
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.float32),
        1.0,
        np.zeros(3, np.float32),
        1.0,
    )
    views, intrinsics, width, height = _camera_matrices_at_resolution(scene, [0, 1], 40)
    mx.eval(views, intrinsics)
    assert (width, height) == (40, 30)
    np.testing.assert_allclose(np.asarray(intrinsics[0]), K * np.array([[0.5], [0.5], [1.0]]))


def _synthetic_scene(n: int = 64, seed: int = 7, views: int = 2):
    rng = np.random.default_rng(seed)
    params = {
        "means3d": mx.array(rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)),
        "log_scales": mx.array(np.log(rng.uniform(0.05, 0.3, (n, 3))).astype(np.float32)),
        "quats": mx.array(
            (lambda q: q / np.linalg.norm(q, axis=1, keepdims=True))(rng.normal(size=(n, 4))).astype(np.float32)
        ),
        "opacities_raw": mx.array(rng.uniform(-1.0, 2.0, (n,)).astype(np.float32)),
    }
    legacy_rgb = 1.0 / (1.0 + np.exp(-rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)))
    params["sh0"] = rgb_to_sh0(mx.array(legacy_rgb))
    params["shN"] = mx.zeros((n, 15, 3), dtype=mx.float32)
    f = 0.7 * W
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float32)
    viewmats = []
    for v in range(views):
        w2c = np.eye(4, dtype=np.float32)
        w2c[0, 3] = 0.2 * v  # slight lateral shift per view
        w2c[2, 3] = 8.0  # camera 8 units in front of the cloud
        viewmats.append(w2c)
    viewmats_mx = mx.array(np.stack(viewmats))
    Ks_mx = mx.array(np.stack([K] * views))
    mx.eval(*params.values(), viewmats_mx, Ks_mx)
    return params, viewmats_mx, Ks_mx


def test_eval_capacity_invariants():
    for max_count, n, ntiles in [(0, 10, 12), (1, 10, 12), (37, 100, 48), (10_000, 500, 96), (10**7, 200, 48)]:
        cap = eval_capacity(max_count, n, ntiles)
        exact = n * ntiles
        assert cap >= 1
        assert cap <= exact
        assert cap >= max_count or cap == exact
    # geometric bucket: growing max_count within one bucket keeps capacity stable
    caps = {eval_capacity(mc, 10**6, 10**3) for mc in range(1000, 1100)}
    assert len(caps) <= 2  # at most one 1.25x boundary inside a 10% span
    with pytest.raises(AssertionError):
        eval_capacity(5, 0, 12)


@pytest.mark.parametrize("mode", ["3dgs", "2dgs"])
def test_compact_exact_matches_padded_exact(mode):
    """Eval's compact-exact capacity must reproduce the padded-exact render bit-for-bit."""
    params, viewmats, Ks = _synthetic_scene()
    counts = _intersection_counts(params, viewmats[:1], Ks[:1], W, H, mode)
    total = int(mx.sum(counts))
    assert total > 0, "synthetic scene projects to zero intersections"
    img_compact = _render_view(params, viewmats[0], Ks[0], W, H, mode, None, total)
    img_padded = _render_view(params, viewmats[0], Ks[0], W, H, mode, None, None)
    mx.eval(img_compact, img_padded)
    diff = float(mx.max(mx.abs(img_compact - img_padded)))
    assert diff == 0.0, f"compact-exact vs padded-exact max|diff|={diff}"


def test_evaluate_golden_psnr():
    """Fixed-checkpoint eval: catches eval-path drift independently of training."""
    params, viewmats, Ks = _synthetic_scene()
    rng = np.random.default_rng(3)
    targets = mx.array(rng.integers(0, 256, (2, H, W, 3), dtype=np.uint8))
    mx.eval(targets)
    psnr, ssim_val, _wall = _evaluate(
        params, targets, viewmats, Ks, W, H, "3dgs", count_batch=1, eval_step=0, log=logging.getLogger("test")
    )
    assert math.isfinite(psnr) and math.isfinite(ssim_val)
    assert -1.0 <= ssim_val <= 1.0
    golden_psnr = 4.9539571549  # 3 identical trials on Apple Silicon, MLX 0.31.2 (forward path is deterministic)
    if golden_psnr is not None:
        assert abs(psnr - golden_psnr) < 1e-3, f"PSNR drifted: {psnr} vs golden {golden_psnr}"


# ---------------------------------------------------------------------------
# Stage 2a: per-group optimizer plumbing (must be behavior-preserving)
# ---------------------------------------------------------------------------

PARAM_NAMES = ["means3d", "log_scales", "quats", "opacities_raw", "sh0", "shN"]
GROUPS = ["means", "scales", "quats", "opacities", "sh0", "shN"]


def _opt_params(n=6, seed=1):
    rng = np.random.default_rng(seed)
    shapes = {
        "means3d": (n, 3),
        "log_scales": (n, 3),
        "quats": (n, 4),
        "opacities_raw": (n,),
        "sh0": (n, 1, 3),
        "shN": (n, 15, 3),
    }
    p = {k: mx.array(rng.normal(size=s).astype(np.float32)) for k, s in shapes.items()}
    mx.eval(*p.values())
    return p


def _run_steps(opt, params, n_steps, seed=2):
    rng = np.random.default_rng(seed)
    params = dict(params)
    for _ in range(n_steps):
        # Fixed key order: MultiOptimizer reorders the returned dict, so drawing
        # grads by params.items() would desync the two RNG streams under test.
        grads = {k: mx.array(rng.normal(size=params[k].shape).astype(np.float32) * 0.1) for k in PARAM_NAMES}
        params = opt.apply_gradients(grads, params)
        mx.eval(*params.values())
    return params


@pytest.mark.parametrize("mode", ["const", "cos", "cos_restart"])
@pytest.mark.parametrize("scene_scale", [1.0])
def test_dict_equal_lr_matches_scalar(mode, scene_scale):
    """Dict LR with all groups equal must match the single-Adam scalar path
    elementwise in params AND optimizer state (m, v, step)."""
    base_lr = 1.6e-4
    max_steps, period = 40, 20
    p0 = _opt_params()
    opt_scalar = set_up_optimizer_3d(p0, base_lr, max_steps, mode, period, step_offset=0)
    opt_dict = set_up_optimizer_3d(p0, {g: base_lr for g in GROUPS}, max_steps, mode, period, step_offset=0)
    p_scalar = _run_steps(opt_scalar, p0, 25, seed=5)
    p_dict = _run_steps(opt_dict, p0, 25, seed=5)
    for k in PARAM_NAMES:
        d = float(mx.max(mx.abs(p_scalar[k] - p_dict[k])))
        assert d < 1e-7, f"[{mode}] param {k} scalar-vs-dict max|diff|={d}"
        for moment in ("m", "v"):
            md = float(mx.max(mx.abs(get_param_state(opt_scalar, k)[moment] - get_param_state(opt_dict, k)[moment])))
            assert md < 1e-9, f"[{mode}] {k}.{moment} scalar-vs-dict max|diff|={md}"
    assert int(get_opt_step(opt_scalar)) == int(get_opt_step(opt_dict))


def test_resolve_group_lrs_validation():
    from drawingwithgaussians.gaussian3d import _resolve_group_lrs

    p = _opt_params()
    assert _resolve_group_lrs(p, {g: 1.0 for g in GROUPS}).keys() == set(PARAM_NAMES) | set()  # full coverage ok
    with pytest.raises(ValueError, match="unknown LR group"):
        _resolve_group_lrs(p, {**{g: 1.0 for g in GROUPS}, "bogus": 1.0})
    with pytest.raises(ValueError, match="missing groups"):
        _resolve_group_lrs(p, {"means": 1.0})  # other params uncovered


def test_cos_restart_matches_across_epochs_fresh_optimizer():
    """cos_restart + fresh optimizer per epoch, step_offset=0: LR restarts each
    epoch (today's SGDR behavior preserved). A fresh optimizer at step 0 must
    produce the same first-step LR every epoch."""
    period = 20
    p = _opt_params()
    lrs = []
    for _epoch in range(2):
        opt = set_up_optimizer_3d(p, 1e-3, max_steps=40, mode="cos_restart", restart_period=period, step_offset=0)
        p2 = _run_steps(opt, p, 1, seed=9)
        # LR at step 0 of cos_restart is the peak (1e-3); recover it from the state
        lrs.append(float(opt.state["learning_rate"]))
    assert abs(lrs[0] - lrs[1]) < 1e-12, f"fresh cos_restart LR differs across epochs: {lrs}"
    assert abs(lrs[0] - 1e-3) < 1e-9, f"cos_restart step-0 LR should be peak: {lrs[0]}"


@pytest.mark.parametrize("as_dict", [False, True])
def test_carry_optimizer_state_roundtrip(as_dict):
    """carry_optimizer_state_3d preserves kept-row moments and zeros new rows,
    for both single-Adam and MultiOptimizer; step is carried."""
    p = _opt_params(n=6, seed=3)
    lr = {g: 1e-3 for g in GROUPS} if as_dict else 1e-3
    old_opt = set_up_optimizer_3d(p, lr, 40, "const", 20)
    p = _run_steps(old_opt, p, 5, seed=4)
    # keep rows [0,2,4], add 2 new rows -> new N=5
    idx_keep = np.array([0, 2, 4])
    num_new = 2
    new_p = {k: mx.concatenate([v[mx.array(idx_keep)], mx.zeros((num_new,) + v.shape[1:])]) for k, v in p.items()}
    mx.eval(*new_p.values())
    new_opt = set_up_optimizer_3d(new_p, lr, 40, "const", 20)
    carry_optimizer_state_3d(old_opt, new_opt, new_p, idx_keep, num_new)
    for k in PARAM_NAMES:
        old_m = np.array(get_param_state(old_opt, k)["m"])
        new_m = np.array(get_param_state(new_opt, k)["m"])
        assert np.allclose(new_m[:3], old_m[idx_keep]), f"{k}: kept moment rows not preserved"
        assert np.all(new_m[3:] == 0.0), f"{k}: new moment rows not zeroed"
    assert int(get_opt_step(new_opt)) == int(get_opt_step(old_opt))


# ---------------------------------------------------------------------------
# Stage 4: periodic opacity reset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("as_dict", [False, True])
def test_opacity_reset_and_zero_moments(as_dict):
    p = _opt_params(n=6, seed=31)
    raw = np.array([-5.0, -2.0, 0.0, 2.0, 5.0, 10.0], dtype=np.float32)
    p["opacities_raw"] = mx.array(raw)
    reset = reset_opacities_3d(p, prune_opa=0.05)
    cap = math.log(0.1 / 0.9)
    assert np.allclose(np.asarray(reset["opacities_raw"]), np.minimum(raw, cap))
    for name in PARAM_NAMES:
        if name != "opacities_raw":
            assert reset[name] is p[name]

    lr = {g: 1e-3 for g in GROUPS} if as_dict else 1e-3
    opt = set_up_optimizer_3d(reset, lr, 20, "const", 10)
    _run_steps(opt, reset, 2, seed=32)
    assert np.any(np.asarray(get_param_state(opt, "opacities_raw")["m"]) != 0)
    zero_param_moments(opt, "opacities_raw")
    state = get_param_state(opt, "opacities_raw")
    assert np.all(np.asarray(state["m"]) == 0)
    assert np.all(np.asarray(state["v"]) == 0)


# ---------------------------------------------------------------------------
# Stage 5: epoch-boundary 2DGS regularizer warm-up
# ---------------------------------------------------------------------------


def test_regularizer_warmup_activates_on_first_eligible_epoch_boundary():
    total_steps = 2500
    starts = [0, 500, 1000, 1500, 2000]
    normal = [_regularizer_weight(1e-5, 0.233, step, total_steps) for step in starts]
    distortion = [_regularizer_weight(1e-2, 0.1, step, total_steps) for step in starts]
    assert normal == [0.0, 0.0, 1e-5, 1e-5, 1e-5]
    assert distortion == [0.0, 1e-2, 1e-2, 1e-2, 1e-2]
    assert _regularizer_weight(0.0, 0.0, 0, total_steps) == 0.0


# ---------------------------------------------------------------------------
# Stage 6a: overflow modes and uncapped builder-count detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "preflight"),
        (False, "off"),
        ("true", "preflight"),
        ("false", "off"),
        ("preflight", "preflight"),
        ("lazy", "lazy"),
        ("off", "off"),
    ],
)
def test_parse_overflow_mode(value, expected):
    assert _parse_overflow_mode(value) == expected
    with pytest.raises(ValueError):
        _parse_overflow_mode("invalid")


def test_uncapped_count_detects_compact_overflow():
    from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE, project_gaussians

    params, viewmats, Ks = _synthetic_scene(n=48, seed=41, views=1)
    means2d, conics, depths = project_gaussians(
        params["means3d"],
        params["log_scales"],
        params["quats"],
        viewmats,
        Ks,
        W,
        H,
    )
    order = mx.argsort(depths, axis=-1)
    means_sorted = mx.take_along_axis(means2d, mx.broadcast_to(order[..., None], means2d.shape), axis=1)
    conics_sorted = mx.take_along_axis(conics, mx.broadcast_to(order[..., None], conics.shape), axis=1)
    depths_sorted = mx.take_along_axis(depths, order, axis=-1)
    opacities = mx.take(mx.sigmoid(params["opacities_raw"]), order)
    opacities = mx.where(
        (depths_sorted > NEAR_PLANE) & (depths_sorted < FAR_PLANE),
        opacities,
        0.0,
    )
    radii = fused3d._bounding_radii(conics_sorted, opacities)
    exact = fused3d._count_tile_intersections(means_sorted, conics_sorted, opacities, W, H)
    total = int(mx.sum(exact))
    assert total > 1
    capacity = total - 1
    _ids, bounds, counts = fused3d._build_bins(
        means_sorted,
        conics_sorted,
        opacities,
        radii,
        W,
        H,
        capacity=capacity,
        return_counts=True,
    )
    mx.eval(bounds, counts)
    assert int(mx.sum(counts)) == total
    assert int(bounds[-1]) == capacity
    bumped = _capacity_for_count(
        total,
        n=48,
        batch=1,
        width=W,
        height=H,
        min_pad=0,
        margin=1.25,
    )
    assert bumped >= total and bumped > capacity


# ---------------------------------------------------------------------------
# Stage 6b: shuffled view sampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_views", "batch_size"),
    [(4, 4), (5, 4), (3, 4), (11, 4)],
)
def test_shuffle_sampler_static_unique_balanced_and_deterministic(n_views, batch_size):
    first = _ViewSampler(n_views, batch_size, seed=17, mode="shuffle")
    second = _ViewSampler(n_views, batch_size, seed=17, mode="shuffle")
    batches = [first.sample() for _ in range(50)]
    matching = [second.sample() for _ in range(50)]
    for batch, other in zip(batches, matching, strict=True):
        assert batch.shape == (batch_size,)
        assert np.array_equal(batch, other)
        if n_views >= batch_size:
            assert len(np.unique(batch)) == batch_size
    counts = np.bincount(np.concatenate(batches), minlength=n_views)
    assert int(counts.max() - counts.min()) <= 1


def test_random_sampler_matches_historical_rng_choice():
    sampler = _ViewSampler(7, 4, seed=9, mode="random")
    reference = np.random.default_rng(9)
    for _ in range(10):
        expected = reference.choice(7, size=4, replace=False)
        assert np.array_equal(sampler.sample(), expected)


# ---------------------------------------------------------------------------
# Stage 2b: LR ladder — L4a global schedule offset + carry ordering
# ---------------------------------------------------------------------------


def _ladder_args(**over):
    from types import SimpleNamespace

    base = dict(
        lr={g: 1e-3 for g in GROUPS},
        means_lr_scene_scale=False,
        means_only_schedule=False,
        global_schedule=True,
        means_mode="exp",
        split_every=20,
        carry_optimizer_state=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("carry", [False, True])
def test_global_schedule_offset_gives_correct_epoch_lr(carry):
    """L4a: with global_schedule, epoch e's first-step LR equals the whole-run
    exp schedule evaluated at global step e*steps, in BOTH carry modes."""
    from train_colmap3d import _build_optimizer

    steps, epochs = 20, 3
    total = steps * epochs
    args = _ladder_args(split_every=steps, carry_optimizer_state=carry)
    p = _opt_params()
    # reference: single exp schedule over the whole run, sampled at epoch starts
    import mlx.optimizers as optim

    ref = optim.exponential_decay(1e-3, 0.01 ** (1.0 / total))

    old_opt = None
    for epoch in range(epochs):
        opt = _build_optimizer(
            p,
            args,
            total,
            scene_scale=1.0,
            segment_start=epoch * steps,
            restart_period=steps,
            old_opt=old_opt,
        )
        if carry and old_opt is not None:
            carry_optimizer_state_3d(old_opt, opt, p, np.arange(len(p["means3d"])), 0)
        # LR is set from the schedule at each apply using the pre-increment step,
        # so after exactly one step the state holds this epoch's first-step LR.
        p = _run_steps(opt, p, 1, seed=epoch)
        used = float(opt.state["states"][0]["learning_rate"])
        expected = float(ref(mx.array(epoch * steps)))
        assert abs(used - expected) < 1e-9, f"carry={carry} epoch {epoch}: LR {used} != global-schedule {expected}"
        p = _run_steps(opt, p, steps - 1, seed=epoch + 100)  # finish the epoch
        old_opt = opt


def test_local_schedule_restarts_each_epoch():
    """global_schedule=false (default): exp schedule restarts each epoch, so
    every fresh epoch's first-step LR is the peak (1e-3)."""
    from train_colmap3d import _build_optimizer

    args = _ladder_args(global_schedule=False, carry_optimizer_state=False)
    p = _opt_params()
    total = args.split_every * 3
    for epoch in range(3):
        opt = _build_optimizer(
            p,
            args,
            total,
            scene_scale=1.0,
            segment_start=epoch * args.split_every,
            restart_period=args.split_every,
            old_opt=None,
        )
        p = _run_steps(opt, p, 1, seed=epoch)  # first-step LR
        used = float(opt.state["states"][0]["learning_rate"])
        assert abs(used - 1e-3) < 1e-9, f"epoch {epoch}: local exp did not restart at peak (got {used})"
        p = _run_steps(opt, p, args.split_every - 1, seed=epoch + 100)


def test_means_lr_scene_scale_and_means_only_schedule():
    """L2 scales only the means group's LR by scene_scale; L3 keeps non-means
    groups constant while means follows the schedule."""
    from train_colmap3d import _build_optimizer

    args = _ladder_args(means_lr_scene_scale=True, means_only_schedule=True, global_schedule=False, means_mode="const")
    p = _opt_params()
    opt = _build_optimizer(p, args, 60, scene_scale=4.0, segment_start=0, restart_period=20, old_opt=None)
    # means group LR is base(1e-3) * scene_scale(4) = 4e-3; others unchanged 1e-3
    lrs = [float(s["learning_rate"]) for s in opt.state["states"]]
    assert abs(lrs[0] - 4e-3) < 1e-9, f"means LR not scaled by scene_scale: {lrs[0]}"
    assert all(abs(x - 1e-3) < 1e-9 for x in lrs[1:]), f"non-means LRs changed: {lrs[1:]}"


# ---------------------------------------------------------------------------
# Stage 3a: densification signal (visibility counts + screen-space normalization)
# ---------------------------------------------------------------------------

from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE  # noqa: E402


def _normalized_signal(params, viewmats, Ks, w, h, mode):
    """Mirror compiled_step's shadow-signal math for a given camera set: the
    screen-space-scaled absgrad-sink norm accumulated over one step, divided by
    the per-(view,gaussian) visibility count."""
    n = params["means3d"].shape[0]
    b = float(viewmats.shape[0])
    targets = mx.zeros((viewmats.shape[0], h, w, 3), dtype=mx.float32)
    absgrad = mx.zeros((n, 2), dtype=mx.float32)
    offset = mx.zeros((n, 2), dtype=mx.float32)
    loss_impl = pixel_loss_2dgs if mode == "2dgs" else pixel_loss_3d

    def loss_fn(absgrad_sink):
        loss, _, counts = loss_impl(
            params["means3d"],
            params["log_scales"],
            params["quats"],
            params["opacities_raw"],
            view_dependent_colors(params, viewmats, 0),
            targets,
            viewmats,
            Ks,
            ssim_weight=0.0,
            means2d_offset=offset,
            means2d_absgrad_sink=absgrad_sink,
            return_counts=True,
        )
        return loss, counts

    (_loss, counts), absgrad_grad = mx.value_and_grad(loss_fn)(absgrad)
    sig_scale = mx.array([w * 0.5 * b, h * 0.5 * b], dtype=mx.float32)
    sig = mx.sqrt(mx.sum((absgrad_grad * sig_scale) ** 2, axis=1))
    vis = mx.sum((counts > 0).astype(mx.float32), axis=0)
    mx.eval(sig, vis)
    return np.asarray(sig), np.asarray(vis)


def _loss_counts(params, viewmats, Ks, mode, bin_capacity):
    targets = mx.zeros((viewmats.shape[0], H, W, 3), dtype=mx.float32)
    loss_impl = pixel_loss_2dgs if mode == "2dgs" else pixel_loss_3d
    _loss, _rendered, counts = loss_impl(
        params["means3d"],
        params["log_scales"],
        params["quats"],
        params["opacities_raw"],
        view_dependent_colors(params, viewmats, 0),
        targets,
        viewmats,
        Ks,
        ssim_weight=0.0,
        bin_capacity=bin_capacity,
        return_counts=True,
    )
    mx.eval(counts)
    return np.asarray(counts)


@pytest.mark.parametrize("mode", ["3dgs", "2dgs"])
@pytest.mark.parametrize("compact", [False, True], ids=["padded-exact", "compact"])
def test_return_counts_are_exact_and_in_param_order(mode, compact):
    """The fused return path must match standalone exact counts after the
    rasterizer's per-view depth sort, including the padded/exact fallback."""
    params, viewmats, Ks = _synthetic_scene(n=40, seed=11)
    expected = np.asarray(_intersection_counts(params, viewmats, Ks, W, H, mode))
    capacity = max(1, int(expected.sum())) if compact else None
    got = _loss_counts(params, viewmats, Ks, mode, capacity)
    assert np.array_equal(got, expected)


@pytest.mark.parametrize(
    ("mode", "module"),
    [("3dgs", fused3d), ("2dgs", fused2dgs)],
)
def test_return_counts_uint32_fallback_stays_exact(monkeypatch, mode, module):
    """Force the compact uint32-key fallback and reject padded bbox/area
    counts as a visibility substitute."""
    params, viewmats, Ks = _synthetic_scene(n=32, seed=19)
    expected = np.asarray(_intersection_counts(params, viewmats, Ks, W, H, mode))
    monkeypatch.setattr(module, "_compact_keys_fit_uint32", lambda *_args: False)
    got = _loss_counts(params, viewmats, Ks, mode, max(1, int(expected.sum())))
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("mode", ["3dgs", "2dgs"])
def test_visibility_zero_behind_camera_and_low_opacity(mode):
    params, viewmats, Ks = _synthetic_scene(n=20, seed=5)
    p = {k: mx.array(np.array(v)) for k, v in params.items()}
    # gaussian 0 behind the camera (negative depth in cam space -> depth < NEAR)
    m = np.array(p["means3d"])
    m[0] = [0.0, 0.0, -20.0]
    p["means3d"] = mx.array(m)
    # gaussian 1 essentially transparent
    o = np.array(p["opacities_raw"])
    o[1] = -20.0  # sigmoid ~ 2e-9 << 1/255
    p["opacities_raw"] = mx.array(o)
    counts = np.asarray(_intersection_counts(p, viewmats[:1], Ks[:1], W, H, mode))[0]
    assert counts[0] == 0, "behind-camera gaussian should have zero visibility"
    assert counts[1] == 0, "transparent gaussian should have zero visibility"


@pytest.mark.parametrize("mode", ["3dgs", "2dgs"])
def test_signal_batch_invariance(mode):
    """Duplicating one camera B times must leave the normalized signal
    unchanged vs B=1: the explicit *B in the numerator cancels the B-fold
    visibility in the denominator (the load-bearing batch factor)."""
    params, viewmats, Ks = _synthetic_scene(n=30, seed=13, views=1)
    sig1, vis1 = _normalized_signal(params, viewmats, Ks, W, H, mode)
    dup_v = mx.broadcast_to(viewmats[0], (4, 4, 4))
    dup_k = mx.broadcast_to(Ks[0], (4, 3, 3))
    sig4, vis4 = _normalized_signal(params, dup_v, dup_k, W, H, mode)
    norm1 = sig1 / np.maximum(vis1, 1.0)
    norm4 = sig4 / np.maximum(vis4, 1.0)
    vis_seen = vis1 > 0
    assert np.all(vis4[vis_seen] == 4 * vis1[vis_seen]), "duplicated views should give 4x visibility"
    rel = np.abs(norm4[vis_seen] - norm1[vis_seen]) / (np.abs(norm1[vis_seen]) + 1e-12)
    assert np.max(rel) < 1e-4, f"normalized signal not batch-invariant: max rel {np.max(rel):.2e}"


def test_signal_uses_partial_visibility_denominator():
    """With B=4 and three cameras looking away, gaussians visible in only the
    first camera divide by one observation, not by camera batch size."""
    params, viewmats, Ks = _synthetic_scene(n=30, seed=23, views=4)
    vm = np.asarray(viewmats).copy()
    vm[1:, 2, 3] = -8.0
    sig, vis = _normalized_signal(params, mx.array(vm), Ks, W, H, "3dgs")
    partial = (vis == 1) & (sig > 0)
    assert partial.any(), "synthetic scene produced no partially visible gaussian"
    normalized = sig / np.maximum(vis, 1.0)
    assert np.array_equal(normalized[partial], sig[partial])


# --- DashGaussian schedulers (schedule.py) + budget densification -----------


def test_resolution_schedule_non_increasing_last_full_res():
    """A broadband target yields a coarse->fine, non-increasing per-epoch
    downscale schedule whose final entry is full resolution."""
    from drawingwithgaussians.schedule import resolution_schedule

    rng = np.random.default_rng(0)
    target = rng.uniform(0.0, 1.0, (128, 128, 3)).astype(np.float32)  # flat (broadband) spectrum
    factors = resolution_schedule(target, num_epochs=10, num_steps=100, start_significance_factor=4.0)
    assert len(factors) == 10
    assert factors[-1] == 1
    assert all(f >= 1 for f in factors)
    assert all(factors[i] >= factors[i + 1] for i in range(len(factors) - 1)), factors
    assert factors[0] >= 2, f"broadband target should start downscaled: {factors}"


def test_resolution_schedule_degenerate_cases():
    from drawingwithgaussians.schedule import resolution_schedule

    # Single epoch => no schedule.
    assert resolution_schedule(np.zeros((32, 32, 3), np.float32), 1, 100) == [1]
    # Constant image (all energy at DC) => no headroom to downscale.
    const = np.full((64, 64, 3), 0.5, np.float32)
    assert resolution_schedule(const, 5, 100) == [1, 1, 1, 1, 1]


def test_momentum_budget_monotone_and_fixed_point():
    from drawingwithgaussians.schedule import MomentumBudget

    budget = MomentumBudget(p_init=100, gamma=0.98, eta=1.0)
    assert budget.p_fin == 600  # p_init + 5 * p_init
    prev = budget.p_fin
    for _ in range(5000):
        pf = budget.update(20)  # realized k = 20
        assert pf >= prev, "P_fin must be monotone non-decreasing"
        prev = pf
    # Ideal continuous fixed point is p_init + eta*k/(1-gamma) = 1100; the faithful
    # int() truncation (DashGaussian's) stalls momentum ~1/(1-gamma) below it.
    ideal = 100 + 20 / (1 - 0.98)
    assert ideal - 1 / (1 - 0.98) - 1 <= budget.p_fin <= ideal, budget.p_fin
    assert budget.update(20) == budget.p_fin, "must be at a stable fixed point"


def test_momentum_budget_target_count_and_fixed_budget():
    from drawingwithgaussians.schedule import MomentumBudget

    budget = MomentumBudget(p_init=500, gamma=0.98, eta=1.0)
    full = budget.target_count(1, 0, 1000)
    coarse = budget.target_count(2, 0, 1000)
    assert full == budget.p_fin  # r == 1 -> full budget
    assert coarse < full, "coarse resolution must suppress the count target"

    fixed = MomentumBudget(p_init=500, max_n_gaussian=5000)
    assert fixed.is_fixed
    assert fixed.update(999) == 5000  # updates are no-ops
    assert fixed.target_count(1, 0, 1000) == 5000


def _budget_scene(n, g_norm):
    """Small all-small, all-opaque scene so every high-signal gaussian is a
    duplicate candidate (no split, no prune) — isolates the top-k budget."""
    params = {
        "means3d": mx.array(np.zeros((n, 3), np.float32)),
        "log_scales": mx.array(np.full((n, 3), np.log(0.01), np.float32)),  # < grow_scale*scene_scale
        "quats": mx.array(np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))),
        "opacities_raw": mx.array(np.full((n,), 5.0, np.float32)),  # sigmoid ~1, never pruned
        "sh0": mx.array(np.zeros((n, 1, 3), np.float32)),
        "shN": mx.array(np.zeros((n, 15, 3), np.float32)),
    }
    return params, mx.array(np.asarray(g_norm, np.float32))


def test_split_n_prune_3d_budget_topk():
    from drawingwithgaussians.gaussian3d import split_n_prune_3d

    n = 10
    params, g_norm = _budget_scene(n, np.arange(n))  # signal 0..9
    # percentile 50 keeps signal > 4.5 (indices 5..9); target 13, n_kept 10,
    # rate 1.0 => k = clamp(13-10, 0, 10) = 3 -> top-3 by signal = {7,8,9}.
    new_params, info = split_n_prune_3d(
        params,
        g_norm,
        mx.random.key(0),
        grow_scale=0.01,
        scene_scale=2.0,
        prune_opa=0.005,
        prune_scale3d=None,
        densify_mode="budget",
        target_count=13,
        grad_percentile=50.0,
        max_densify_rate=1.0,
    )
    assert info["n_densified"] == 3
    assert info["n_dupli"] == 3 and info["n_split"] == 0
    assert new_params["means3d"].shape[0] == 13  # n_kept (10) + k (3) == target


def test_split_n_prune_3d_budget_zero_growth():
    from drawingwithgaussians.gaussian3d import split_n_prune_3d

    n = 10
    params, g_norm = _budget_scene(n, np.arange(n))
    # target == n_kept => k = 0 => no densification.
    new_params, info = split_n_prune_3d(
        params,
        g_norm,
        mx.random.key(0),
        grow_scale=0.01,
        scene_scale=2.0,
        prune_opa=0.005,
        prune_scale3d=None,
        densify_mode="budget",
        target_count=10,
        grad_percentile=50.0,
        max_densify_rate=1.0,
    )
    assert info["n_densified"] == 0
    assert new_params["means3d"].shape[0] == 10
