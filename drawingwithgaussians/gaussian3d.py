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

import numpy as np

import mlx.core as mx
import mlx.optimizers as optim


def init_gaussians_3d(num_points, key):
    """gsplat image_fitting init: means in [-1, 1]^3, uniform scales,
    random rotations, opacity logits at 1 (sigmoid -> 0.73), color logits
    uniform. Scales are stored in log-space (this repo's convention)."""
    keys = mx.random.split(key, 4)
    means3d = 2.0 * (mx.random.uniform(shape=(num_points, 3), key=keys[0]) - 0.5)
    log_scales = mx.log(mx.random.uniform(low=1e-3, high=1.0, shape=(num_points, 3), key=keys[1]))
    quats = mx.random.normal(shape=(num_points, 4), key=keys[2])
    opacities_raw = mx.ones((num_points,))
    colors_raw = mx.random.uniform(shape=(num_points, 3), key=keys[3])
    return {
        "means3d": means3d,
        "log_scales": log_scales,
        "quats": quats,
        "opacities_raw": opacities_raw,
        "colors_raw": colors_raw,
    }


def set_up_optimizer_3d(params, lr, max_steps, mode="const", restart_period=None):
    """Single Adam over the whole parameter dict (gsplat image_fitting uses
    one optimizer for all groups). ``mode`` mirrors the 2D ``means_mode``:
    ``const`` (gsplat default), ``cos``, or ``cos_restart`` (SGDR)."""
    if mode == "const":
        lr_sched = lr
    elif mode == "cos":
        lr_sched = optim.cosine_decay(lr, max_steps)
    elif mode == "cos_restart":
        if restart_period is None:
            raise ValueError("mode='cos_restart' requires restart_period")

        def lr_sched(step):
            s = step % restart_period
            return lr * 0.5 * (1 + mx.cos(np.pi * s / restart_period))

    else:
        raise ValueError(f"unknown mode: {mode}")
    opt = optim.Adam(learning_rate=lr_sched, bias_correction=True)
    opt.init(params)
    return opt


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


def split_n_prune_3d(
    params,
    avg_grad_norms,
    key,
    grad_thr=1e-6,
    grow_scale=0.01,
    scene_scale=2.0,
    prune_opa=0.005,
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
            (gsplat's too-big prune is deliberately not ported: it is tied
            to their opacity-reset schedule and scene scaling.)

    Returns:
        ``(params, info)`` — new parameter dict and the same ``info`` dict
        shape as the 2D :func:`gaussian.split_n_prune` (``idx_keep``,
        ``num_new``, per-branch counts).
    """
    mx.eval(*params.values(), avg_grad_norms)
    p = {k: np.array(v) for k, v in params.items()}
    g_norm = np.array(avg_grad_norms)
    rng = np.random.default_rng(np.array(key))
    n = len(g_norm)

    mask_erase = 1.0 / (1.0 + np.exp(-p["opacities_raw"])) < prune_opa
    mask_grad_high = g_norm > grad_thr
    mask_small = np.exp(p["log_scales"]).max(axis=1) <= grow_scale * scene_scale
    mask_dupli = mask_grad_high & mask_small & ~mask_erase
    mask_split = mask_grad_high & ~mask_small & ~mask_erase
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
        s_colors = np.tile(p["colors_raw"][idx_split], (2, 1))
        # revised opacity: a_child = 1 - sqrt(1 - a), back to logits.
        a = 1.0 / (1.0 + np.exp(-p["opacities_raw"][idx_split]))
        a_child = np.clip(1.0 - np.sqrt(1.0 - np.clip(a, 0.0, 0.9999)), 1e-6, 1.0 - 1e-6)
        s_opac = np.log(a_child / (1.0 - a_child)).astype(np.float32)
        s_opac = np.tile(s_opac, 2)
        split_rows = {
            "means3d": s_means,
            "log_scales": s_log_scales,
            "quats": s_quats,
            "opacities_raw": s_opac,
            "colors_raw": s_colors,
        }
    else:
        split_rows = {k: np.zeros((0,) + v.shape[1:], dtype=v.dtype) for k, v in p.items()}

    new_params = {
        k: mx.array(np.concatenate([kept[k], dupli[k], split_rows[k]], axis=0).astype(np.float32)) for k in p
    }
    info = {
        "idx_keep": idx_keep,
        "num_new": len(idx_dupli) + 2 * n_split,
        "n_dupli": len(idx_dupli),
        "n_split": n_split,
        "n_prune": int(mask_erase.sum()),
    }
    return new_params, info


def carry_optimizer_state_3d(old_opt, new_opt, params, idx_keep, num_new):
    """Optional gsplat-orthodox state carry for the single 3D Adam (off by
    default — see the 2D A/B finding in EXPERIMENTS.md). Remaps the moment
    rows of every parameter and carries the step counter."""
    for name in params:
        for moment in ("m", "v"):
            old = np.array(old_opt.state[name][moment])
            new_rows = np.zeros((num_new,) + old.shape[1:], dtype=old.dtype)
            new_opt.state[name][moment] = mx.array(np.concatenate([old[idx_keep], new_rows], axis=0))
    new_opt.state["step"] = old_opt.state["step"]
