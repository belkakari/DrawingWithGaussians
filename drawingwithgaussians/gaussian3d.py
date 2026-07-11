"""3D gaussian parameter init / optimization / densification.

The 3D analog of ``gaussian.py``: gsplat's DefaultStrategy (duplicate small
gradient-high gaussians, split large ones, prune transparent ones) adapted
per the 2D A/B findings in EXPERIMENTS.md — fresh optimizer state per refine
by default, warm-restart LR available, refinement driven by the mean
per-step *screen-space* gradient norm accumulated in the train step (fit3d
gets it through a zero ``means2d_offset`` parameter, the MLX equivalent of
gsplat's ``retain_grad`` on means2d).

Split children take gsplat's ``revised_opacity`` correction
(``a_child = 1 - sqrt(1 - a)``, arXiv:2404.06109): alpha compositing
double-counts a straight opacity copy the same way the 2D additive renderer
double-counts colors.

Like the 2D path, the refine op materializes through numpy (MLX 0.31 has no
boolean indexing); it runs once per epoch so the overhead is negligible.
"""

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from .schedule import clamp_densify_count
from .selective_adam import SelectiveAdam
from .sh import rgb_to_sh0


def init_gaussians_3d(num_points, key):
    """gsplat image_fitting init: means in [-1, 1]^3, uniform scales,
    random rotations, opacity logits at 1 (sigmoid -> 0.73), color logits
    uniform. Scales are stored in log-space (this repo's convention)."""
    keys = mx.random.split(key, 4)
    means3d = 2.0 * (mx.random.uniform(shape=(num_points, 3), key=keys[0]) - 0.5)
    log_scales = mx.log(mx.random.uniform(low=1e-3, high=1.0, shape=(num_points, 3), key=keys[1]))
    quats = mx.random.normal(shape=(num_points, 4), key=keys[2])
    opacities_raw = mx.ones((num_points,))
    rgb = mx.sigmoid(mx.random.uniform(shape=(num_points, 3), key=keys[3]))
    return {
        "means3d": means3d,
        "log_scales": log_scales,
        "quats": quats,
        "opacities_raw": opacities_raw,
        "sh0": rgb_to_sh0(rgb),
        "shN": mx.zeros((num_points, 15, 3), dtype=mx.float32),
    }


# Config LR-group names -> actual parameter-dict keys. The dict LR path is
# keyed by the left column; every param must be covered by exactly one group.
_LR_GROUP_TO_PARAM = {
    "means": "means3d",
    "scales": "log_scales",
    "quats": "quats",
    "opacities": "opacities_raw",
    "sh0": "sh0",
    "shN": "shN",
}


def _lr_schedule(base_lr, mode, max_steps, restart_period, step_offset, decay_from_step=0):
    """Build one group's LR schedule. ``step_offset`` shifts the schedule's
    step so it can survive per-epoch optimizer rebuilds (global schedule); with
    ``step_offset == 0`` every mode reproduces the pre-per-group behavior
    exactly (``const`` -> constant float, ``cos`` -> ``cosine_decay``,
    ``cos_restart`` -> ``step % period`` SGDR).

    ``decay_from_step`` (DashGaussian LR delay) holds the LR at ``base_lr``
    until that global step, then runs the decay as if starting from 0. Only
    meaningful with the frequency resolution schedule (decay begins at the
    first full-resolution epoch); ``0`` disables it (historical behavior)."""
    if mode == "const":
        return base_lr  # constant ignores both offsets
    if mode == "cos":
        decay = optim.cosine_decay(base_lr, max_steps)
        base = decay if step_offset == 0 else (lambda s: decay(s + step_offset))
    elif mode == "cos_restart":
        if restart_period is None:
            raise ValueError("mode='cos_restart' requires restart_period")

        def base(step):
            s = (step + step_offset) % restart_period
            return base_lr * 0.5 * (1 + mx.cos(np.pi * s / restart_period))

    elif mode == "exp":
        # gsplat means schedule: exponential decay to 1% of base over max_steps.
        decay = optim.exponential_decay(base_lr, 0.01 ** (1.0 / max(1, max_steps)))
        base = decay if step_offset == 0 else (lambda s: decay(s + step_offset))
    else:
        raise ValueError(f"unknown mode: {mode}")

    if decay_from_step <= 0:
        return base

    def delayed(step):
        shifted = mx.maximum(step - decay_from_step, 0)
        return mx.where(step < decay_from_step, mx.array(base_lr, dtype=mx.float32), base(shifted))

    return delayed


def _resolve_group_lrs(params, lr_dict):
    """Map a group-keyed LR dict to param-keyed floats, validating full,
    non-overlapping coverage of ``params`` (raises on unknown group, a group
    whose param is absent, or any param left without a group)."""
    resolved = {}
    for group, value in lr_dict.items():
        if group not in _LR_GROUP_TO_PARAM:
            raise ValueError(f"unknown LR group {group!r}; expected {list(_LR_GROUP_TO_PARAM)}")
        pname = _LR_GROUP_TO_PARAM[group]
        if pname not in params:
            raise ValueError(f"LR group {group!r} -> {pname!r} not in params {list(params)}")
        resolved[pname] = float(value)
    missing = set(params) - set(resolved)
    if missing:
        raise ValueError(f"LR dict is missing groups for params: {sorted(missing)}")
    return resolved


def set_up_optimizer_3d(
    params,
    lr,
    max_steps,
    mode="const",
    restart_period=None,
    step_offset=0,
    means_only_schedule=False,
    decay_from_step=0,
    selective=False,
):
    """Adam optimizer(s) over the parameter dict.

    ``lr`` may be a scalar (single Adam over all groups, gsplat image_fitting
    style — behaviorally identical to the original) or a group-keyed dict (one
    Adam per group wrapped in :class:`optim.MultiOptimizer`). ``mode`` mirrors
    the 2D ``means_mode``: ``const`` (gsplat default), ``cos``, ``cos_restart``
    (SGDR), or ``exp``. ``step_offset`` makes the schedule global across
    per-epoch rebuilds (0 keeps schedules local, the historical behavior).
    ``means_only_schedule`` restricts the schedule to the means group (other
    groups constant) — the gsplat COLMAP convention (Stage 2b)."""
    if selective:
        if mode != "const":
            raise ValueError("selective Adam currently requires means_mode='const'")
        group_lrs = (
            {name: float(lr) for name in params} if isinstance(lr, (int, float)) else _resolve_group_lrs(params, lr)
        )
        opt = SelectiveAdam(group_lrs)
        opt.init(params)
        return opt
    if isinstance(lr, (int, float)):
        lr_sched = _lr_schedule(float(lr), mode, max_steps, restart_period, step_offset, decay_from_step)
        opt = optim.Adam(learning_rate=lr_sched, bias_correction=True)
        opt.init(params)
        return opt

    group_lrs = _resolve_group_lrs(params, lr)
    names = list(params.keys())
    optimizers, filters = [], []
    for i, pname in enumerate(names):
        grp_mode = mode if (pname == "means3d" or not means_only_schedule) else "const"
        sched = _lr_schedule(group_lrs[pname], grp_mode, max_steps, restart_period, step_offset, decay_from_step)
        optimizers.append(optim.Adam(learning_rate=sched, bias_correction=True))
        if i < len(names) - 1:  # last optimizer is MultiOptimizer's fallback
            filters.append(lambda path, val, n=pname: path == n)
    opt = optim.MultiOptimizer(optimizers, filters)
    opt.init(params)
    return opt


def _param_substate(opt, name):
    """The state dict holding Adam moments for ``name`` (single Adam or the
    owning MultiOptimizer sub-optimizer). Mutating the returned dict mutates
    the live optimizer state."""
    st = opt.state
    if "states" in st:  # MultiOptimizer
        for sub in st["states"]:
            if name in sub:
                return sub
        raise KeyError(f"{name!r} not owned by any sub-optimizer")
    return st


def get_param_state(opt, name):
    """Adam moment dict ``{"m", "v"}`` for parameter ``name``."""
    return _param_substate(opt, name)[name]


def set_param_state(opt, name, state):
    """Replace the Adam moment dict for parameter ``name``."""
    _param_substate(opt, name)[name] = state


def zero_param_moments(opt, name):
    """Zero the Adam m/v moments for parameter ``name`` in place."""
    if isinstance(opt, SelectiveAdam):
        for moment in ("m", "v"):
            opt.state["params"][name][moment] = mx.zeros_like(opt.state["params"][name][moment])
        return
    ps = get_param_state(opt, name)
    for moment in ("m", "v"):
        ps[moment] = mx.zeros_like(ps[moment])


def get_opt_step(opt):
    """The Adam step counter (sub-optimizers step in lockstep, so the first)."""
    if isinstance(opt, SelectiveAdam):
        mx.eval(opt.state["counters"])
        return mx.max(opt.state["counters"])
    st = opt.state
    return st["states"][0]["step"] if "states" in st else st["step"]


def set_opt_step(opt, step):
    """Set the Adam step counter across single-Adam and MultiOptimizer."""
    st = opt.state
    if "states" in st:
        for sub in st["states"]:
            sub["step"] = step
    else:
        st["step"] = step


def _quats_to_rotmats_np(quats):
    q = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def densify_masks(p, g_norm, grad_thr, grow_scale, scene_scale, prune_opa, prune_scale3d, grad_percentile=None):
    """The duplicate/split/erase decision masks, shared by the live refine
    (:func:`split_n_prune_3d`) and shadow-mode densification telemetry so the
    two can never drift. ``p`` is a numpy param dict, ``g_norm`` the (N,) refine
    signal. Returns ``(mask_dupli, mask_split, mask_erase, mask_prune_opa,
    mask_prune_scale)`` as numpy bool arrays.

    ``grad_percentile`` (budget mode) replaces the absolute ``grad_thr`` gate
    with a relative one: candidates are those above the ``grad_percentile``-th
    percentile of ``g_norm``. This sidesteps the resolution-dilation of the
    screen-space signal (EXPERIMENTS.md Exp 12) that a fixed ``grad_thr`` would
    suffer under coarse-resolution epochs. ``None`` keeps the absolute gate
    (free mode, byte-identical to the historical behavior)."""
    opacity = 1.0 / (1.0 + np.exp(-p["opacities_raw"]))
    max_scale = np.exp(p["log_scales"]).max(axis=1)
    mask_prune_opa = opacity < prune_opa
    if prune_scale3d is None or float(prune_scale3d) <= 0.0:
        mask_prune_scale = np.zeros_like(mask_prune_opa)
    else:
        mask_prune_scale = max_scale > float(prune_scale3d) * scene_scale
    mask_erase = mask_prune_opa | mask_prune_scale
    if grad_percentile is None:
        mask_grad_high = g_norm > grad_thr
    else:
        mask_grad_high = g_norm > np.percentile(g_norm, float(grad_percentile))
    mask_small = max_scale <= grow_scale * scene_scale
    mask_dupli = mask_grad_high & mask_small & ~mask_erase
    mask_split = mask_grad_high & ~mask_small & ~mask_erase
    return mask_dupli, mask_split, mask_erase, mask_prune_opa, mask_prune_scale


def reset_opacities_3d(params, prune_opa):
    """Cap opacity at ``2 * prune_opa`` after a densification boundary.

    This is the standard 3DGS periodic-opacity reset expressed in this repo's
    logit parameterization. A new dict is returned; arrays other than
    ``opacities_raw`` are shared unchanged.
    """
    reset_opacity = min(max(2.0 * float(prune_opa), 1e-6), 1.0 - 1e-6)
    reset_logit = np.float32(np.log(reset_opacity / (1.0 - reset_opacity)))
    return {
        **params,
        "opacities_raw": mx.minimum(params["opacities_raw"], mx.array(reset_logit, dtype=mx.float32)),
    }


def split_n_prune_3d(
    params,
    avg_grad_norms,
    key,
    grad_thr=1e-6,
    grow_scale=0.01,
    scene_scale=2.0,
    prune_opa=0.005,
    prune_scale3d=None,
    densify_mode="free",
    target_count=None,
    grad_percentile=50.0,
    max_densify_rate=0.2,
    extra_prune_mask=None,
):
    """Densify (duplicate/split) and prune the 3D gaussians.

    Args:
        params: dict from :func:`init_gaussians_3d`.
        avg_grad_norms: (N,) mean per-step screen-space means2d-gradient norm
            over the epoch.
        key: MLX key; seeds the numpy Generator (reproducible splits).
        grad_thr: threshold on ``avg_grad_norms`` for duplicate/split.
        grow_scale: size threshold as a fraction of ``scene_scale``
            (gsplat's grow_scale3d): grad-high gaussians with
            ``max(exp(log_scales)) <= grow_scale * scene_scale`` duplicate,
            larger ones split.
        scene_scale: world extent of the scene (means init in [-1, 1] -> 2).
        prune_opa: prune gaussians with ``sigmoid(opacity) < prune_opa``.
        prune_scale3d: optional gsplat-style too-big prune threshold as a
            fraction of ``scene_scale``; gaussians whose largest 3D scale
            exceeds ``prune_scale3d * scene_scale`` are removed instead of
            being split into more giant children.
        densify_mode: ``"free"`` (default, historical unbounded threshold
            growth) or ``"budget"`` (DashGaussian count budget). In ``budget``
            mode the ``grad_thr`` gate is replaced by a ``grad_percentile``
            relative gate and only the **top-k by ``g_norm``** candidates
            densify, where ``k = clamp(target_count - n_kept, 0,
            max_densify_rate * n_kept)`` (union of dupli+split, branch
            assignment preserved).
        target_count: budget-mode target primitive count for this refine
            (from :meth:`schedule.MomentumBudget.target_count`); required when
            ``densify_mode == "budget"``.
        grad_percentile: budget-mode relative gate percentile of ``g_norm``.
        max_densify_rate: budget-mode per-refine growth cap fraction.

    Returns:
        ``(params, info)`` — new parameter dict and the same ``info`` dict
        shape as the 2D :func:`gaussian.split_n_prune` (``idx_keep``,
        ``num_new``, per-branch counts), plus ``n_densified`` (realized
        post-clamp count ``k``, to feed back into the momentum budget).
    """
    mx.eval(*params.values(), avg_grad_norms)
    p = {k: np.array(v) for k, v in params.items()}
    g_norm = np.array(avg_grad_norms)
    rng = np.random.default_rng(np.array(key))

    budget = densify_mode == "budget"
    mask_dupli, mask_split, mask_erase, mask_prune_opa, mask_prune_scale = densify_masks(
        p,
        g_norm,
        grad_thr,
        grow_scale,
        scene_scale,
        prune_opa,
        prune_scale3d,
        grad_percentile=grad_percentile if budget else None,
    )
    mask_prune_utilization = (
        np.zeros_like(mask_erase) if extra_prune_mask is None else np.asarray(extra_prune_mask, dtype=bool)
    )
    if mask_prune_utilization.shape != mask_erase.shape:
        raise ValueError("extra_prune_mask must have one entry per Gaussian")
    mask_erase = mask_erase | mask_prune_utilization
    mask_dupli = mask_dupli & ~mask_erase
    mask_split = mask_split & ~mask_erase
    if budget:
        if target_count is None:
            raise ValueError("densify_mode='budget' requires target_count")
        n_kept = int((~mask_erase).sum())
        k = clamp_densify_count(target_count, n_kept, max_densify_rate)
        cand_idx = np.where(mask_dupli | mask_split)[0]
        if len(cand_idx) > k:
            # Rank the dupli+split union by the refine signal; keep the top k,
            # preserving each survivor's branch (small->dupli, large->split).
            order = cand_idx[np.argsort(g_norm[cand_idx], kind="stable")[::-1]]
            keep_cand = np.zeros(len(g_norm), dtype=bool)
            if k > 0:
                keep_cand[order[:k]] = True
            mask_dupli = mask_dupli & keep_cand
            mask_split = mask_split & keep_cand
    mask_keep = ~(mask_split | mask_erase)

    idx_split = np.where(mask_split)[0]
    idx_dupli = np.where(mask_dupli)[0]
    idx_keep = np.where(mask_keep)[0]

    kept = {k: v[idx_keep] for k, v in p.items()}
    dupli = {k: v[idx_dupli] for k, v in p.items()}

    n_split = len(idx_split)
    if n_split > 0:
        R = _quats_to_rotmats_np(p["quats"][idx_split])
        scales = np.exp(p["log_scales"][idx_split])
        z = rng.standard_normal((2, n_split, 3)).astype(np.float32)
        # children means = mean + R @ (scales * z)   (gsplat's split op)
        offsets = np.einsum("nij,bnj->bni", R, scales[None] * z)
        s_means = (p["means3d"][idx_split][None] + offsets).reshape(-1, 3)
        s_log_scales = np.tile(p["log_scales"][idx_split] - np.log(1.6, dtype=np.float32), (2, 1))
        s_quats = np.tile(p["quats"][idx_split], (2, 1))
        # revised opacity: a_child = 1 - sqrt(1 - a), back to logits.
        a = 1.0 / (1.0 + np.exp(-p["opacities_raw"][idx_split]))
        a_child = np.clip(1.0 - np.sqrt(1.0 - np.clip(a, 0.0, 0.9999)), 1e-6, 1.0 - 1e-6)
        s_opac = np.log(a_child / (1.0 - a_child)).astype(np.float32)
        s_opac = np.tile(s_opac, 2)
        # Every row-shaped auxiliary/appearance tensor follows its parent.
        # Geometry and revised opacity are then replaced with their special
        # split rules. This automatically covers sh0/shN and future telemetry.
        split_rows = {k: np.tile(v[idx_split], (2,) + (1,) * (v.ndim - 1)) for k, v in p.items()}
        split_rows.update(
            {
                "means3d": s_means,
                "log_scales": s_log_scales,
                "quats": s_quats,
                "opacities_raw": s_opac,
            }
        )
    else:
        split_rows = {k: np.zeros((0,) + v.shape[1:], dtype=v.dtype) for k, v in p.items()}

    new_params = {k: mx.array(np.concatenate([kept[k], dupli[k], split_rows[k]], axis=0).astype(np.float32)) for k in p}
    info = {
        "idx_keep": idx_keep,
        "idx_new_parent": np.concatenate([idx_dupli, np.tile(idx_split, 2)]).astype(np.int64),
        "num_new": len(idx_dupli) + 2 * n_split,
        "n_dupli": len(idx_dupli),
        "n_split": n_split,
        # Realized post-clamp densification count k = net growth in gaussians
        # (each dupli/split parent contributes +1). Feed this to the momentum
        # budget so P_fin tracks real demand, not the candidate count.
        "n_densified": len(idx_dupli) + n_split,
        "n_prune": int(mask_erase.sum()),
        "n_prune_opa": int(mask_prune_opa.sum()),
        "n_prune_scale3d": int(mask_prune_scale.sum()),
        "n_prune_utilization": int(mask_prune_utilization.sum()),
    }
    return new_params, info


def carry_optimizer_state_3d(old_opt, new_opt, params, idx_keep, num_new):
    """Optional gsplat-orthodox state carry (off by default — see the 2D A/B
    finding in EXPERIMENTS.md). Remaps the moment rows of every parameter
    (kept rows preserved, new rows zeroed) and carries the step counter.
    Works for both the single Adam and the per-group MultiOptimizer via the
    state helpers, so optimizer internals never leak into the trainer."""
    if isinstance(old_opt, SelectiveAdam) or isinstance(new_opt, SelectiveAdam):
        if not isinstance(old_opt, SelectiveAdam) or not isinstance(new_opt, SelectiveAdam):
            raise TypeError("cannot carry state between dense and selective Adam")
        new_opt.carry_from(old_opt, idx_keep, num_new)
        return
    for name in params:
        old = get_param_state(old_opt, name)
        merged = {}
        for moment in ("m", "v"):
            arr = np.array(old[moment])
            new_rows = np.zeros((num_new,) + arr.shape[1:], dtype=arr.dtype)
            merged[moment] = mx.array(np.concatenate([arr[idx_keep], new_rows], axis=0))
        set_param_state(new_opt, name, merged)
    set_opt_step(new_opt, get_opt_step(old_opt))
