"""MLX Gaussian parameter init / optimization / splitting.

Mirrors the JAX port: ``init_gaussians`` returns the gaussian parameters,
``set_up_optimizers`` returns five :class:`mlx.optimizers.Adam` instances,
``update`` applies gradients and returns the new params; ``split_n_prune`` /
``split_gaussian`` replicate gaussians with large gradients.

L is parameterized as ``(log_diag, offdiag)`` following the gsplat convention:
the diagonal of L is stored in log-space (``L[i,i] = exp(log_diag[i])``) so
the diagonal is *always* positive by construction. The optimizer lives in
unconstrained space, which makes the effective Adam step scale-invariant and
stops training from pushing variances to zero.

There is deliberately no upper cap on L's diagonal: the optimizer may grow
large splats when useful. Degenerate tiny splats are still pruned during
``split_n_prune`` because near-singular covariances can destabilize the
precision-matrix math.

State (optimizer momentum / variance) lives inside each optimizer object
rather than in separate optax-style tuples, which is the MLX idiom.
"""

import math

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

# No upper cap on L's diagonal. ``split_n_prune`` only prunes collapsed
# gaussians with variance below this threshold in either direction: a 2D
# gaussian with one tiny axis contributes little but risks precision-matrix
# blowups.
_MIN_VARIANCE_FOR_KEEP = 0.05  # variance threshold below which a gaussian is pruned


def build_L(log_diag, offdiag):
    """Reconstruct the (N, 2, 2) lower-triangular cholesky factor from its
    unconstrained parameterization. The diagonal is positive by construction
    and is not upper-capped.

    Args:
        log_diag: (N, 2) — log of L's diagonal entries. ``L[i,i] = exp(log_diag[i])``.
        offdiag: (N,) — L's off-diagonal entry ``L[i, 1, 0]``. Unconstrained.
    """
    diag = mx.exp(log_diag)
    zeros = mx.zeros_like(diag[:, 0])
    row0 = mx.stack([diag[:, 0], zeros], axis=-1)
    row1 = mx.stack([offdiag, diag[:, 1]], axis=-1)
    return mx.stack([row0, row1], axis=-2)


def init_gaussians(num_gaussians, target_image, key, optimize_background=True):
    """Initialize gaussian parameters.

    Args:
        num_gaussians: number of Gaussians to allocate.
        target_image: (H, W, 3) target image. Only the height/width are used.
        key: an ``mx.array`` uint32 key (``mx.random.key(seed)``). A Python
            seed int or a 2-element tuple/list is also accepted.
        optimize_background: whether to allocate a learnable background.
            When ``False`` the background is initialized to zero.

    Returns:
        ``(means, log_diag, offdiag, colors, background_color)``.
        ``log_diag`` is the (N, 2) log of L's diagonal; ``offdiag`` is the
        (N,) L[1, 0] entry. Use :func:`build_L` to reconstruct L. Orientation
        is fully captured by L (via ``offdiag``); there is no separate
        rotation parameter.
    """
    if not isinstance(key, mx.array):
        if isinstance(key, (tuple, list)):
            seed = int(key[0])
        else:
            seed = int(key)
        key = mx.random.key(seed)
    key = mx.random.split(key, 1)[0]

    height = int(target_image.shape[0])

    # Split off keys for each random call so reusing the input key doesn't
    # collapse them into the same sample.
    keys = mx.random.split(key, 5)

    background_color = mx.random.uniform(shape=(1, 1, 3), dtype=mx.float32, key=keys[0])
    if not optimize_background:
        background_color = background_color * 0.0

    means = mx.random.uniform(
        low=0.0,
        high=float(height),
        shape=(num_gaussians, 2),
        dtype=mx.float32,
        key=keys[1],
    )
    sigmas = mx.random.uniform(
        low=1.0,
        high=float(height) / 8.0,
        shape=(num_gaussians, 2),
        dtype=mx.float32,
        key=keys[2],
    )
    # log_diag: cholesky(diag(sigma^2)) = diag(sigma), so initial L is
    # diagonal with entries equal to sigma. Store log(sigma) so the
    # exp-parameterization matches.
    log_diag = mx.log(sigmas)
    # No initial cross-correlation.
    offdiag = mx.zeros((num_gaussians,), dtype=mx.float32)

    colors = mx.random.uniform(shape=(num_gaussians, 3), dtype=mx.float32, key=keys[3])

    return means, log_diag, offdiag, colors, background_color


def set_up_optimizers(
    means,
    log_diag,
    offdiag,
    colors,
    background_color,
    lr,
    max_steps,
    means_mode="cos",
    optimize_background=True,
    restart_period=None,
):
    """Build the Adam optimizers used by the inner training step.

    Args:
        means, log_diag, offdiag, colors, background_color: arrays
            returned by :func:`init_gaussians` (or the current params after
            a split/prune).
        lr: base learning rate.
        max_steps: length of the means LR schedule in steps. Pass the *total*
            training steps (num_epochs * steps_per_epoch): the step counter
            is carried across refines by :func:`carry_optimizer_state`, so
            the cosine decays once over the whole run (gsplat-style), not
            per epoch.
        means_mode: ``"cos"`` for one cosine decay over ``max_steps``,
            ``"cos_restart"`` for SGDR-style warm restarts every
            ``restart_period`` steps, ``"const"`` for a constant LR.
        restart_period: warm-restart period in steps (``"cos_restart"`` only;
            pass the refine cadence, i.e. steps per epoch).
        optimize_background: when ``True`` an Adam is also created for the
            background color.

    Returns:
        ``(opt_means, opt_log_diag, opt_offdiag, opt_colors, opt_bg)``:
        :class:`mlx.optimizers.Adam` instances, each initialized with its
        parameter's state. ``opt_bg`` is ``None`` when ``optimize_background``
        is ``False``.
    """
    if means_mode == "cos":
        lr_means = optim.cosine_decay(lr, max_steps)
    elif means_mode == "cos_restart":
        # SGDR-style warm restarts: cosine over each refine period. The step
        # counter is carried across refines (carry_optimizer_state), so the
        # restart comes from the modulo, not from resetting the optimizer.
        # Empirically load-bearing for this fit (see EXPERIMENTS.md): each
        # refine adds fresh gaussians that need a high LR to settle.
        if restart_period is None:
            raise ValueError("means_mode='cos_restart' requires restart_period")

        def lr_means(step):
            s = step % restart_period
            return lr * 0.5 * (1 + mx.cos(math.pi * s / restart_period))

    elif means_mode == "const":
        lr_means = lr
    else:
        raise ValueError(f"unknown means_mode: {means_mode}")

    opt_means = optim.Adam(learning_rate=lr_means, bias_correction=True)
    opt_means.init({"means": means})
    opt_log_diag = optim.Adam(learning_rate=lr, bias_correction=True)
    opt_log_diag.init({"log_diag": log_diag})
    opt_offdiag = optim.Adam(learning_rate=lr, bias_correction=True)
    opt_offdiag.init({"offdiag": offdiag})
    opt_colors = optim.Adam(learning_rate=lr, bias_correction=True)
    opt_colors.init({"colors": colors})

    if optimize_background:
        opt_bg = optim.Adam(learning_rate=lr, bias_correction=True)
        opt_bg.init({"background_color": background_color})
    else:
        opt_bg = None

    return opt_means, opt_log_diag, opt_offdiag, opt_colors, opt_bg


def update(
    means,
    log_diag,
    offdiag,
    colors,
    background_color,
    optimizers,
    gradients,
):
    """Apply Adam updates to the gaussian parameters.

    Args:
        means, log_diag, offdiag, colors, background_color: current
            parameters.
        optimizers: 5-tuple returned by :func:`set_up_optimizers`. The last
            entry may be ``None`` when background optimization is disabled.
        gradients: gradient tuple ``(g_means, g_log_diag, g_offdiag, g_colors
            [, g_bg])`` matching the parameter order.

    Returns:
        ``(means, log_diag, offdiag, colors, background_color, optimizers)``
        with optimizer state updated in-place.
    """
    opt_means, opt_log_diag, opt_offdiag, opt_colors, opt_bg = optimizers

    new_means = opt_means.apply_gradients({"means": gradients[0]}, {"means": means})["means"]
    new_log_diag = opt_log_diag.apply_gradients({"log_diag": gradients[1]}, {"log_diag": log_diag})["log_diag"]
    new_offdiag = opt_offdiag.apply_gradients({"offdiag": gradients[2]}, {"offdiag": offdiag})["offdiag"]
    new_colors = opt_colors.apply_gradients({"colors": gradients[3]}, {"colors": colors})["colors"]

    if opt_bg is not None:
        new_bg = opt_bg.apply_gradients({"background_color": gradients[4]}, {"background_color": background_color})[
            "background_color"
        ]
    else:
        new_bg = background_color

    return (
        new_means,
        new_log_diag,
        new_offdiag,
        new_colors,
        new_bg,
        optimizers,
    )


def split_n_prune(
    means,
    log_diag,
    offdiag,
    colors,
    background_color,
    avg_grad_norms,
    key,
    grad_thr=1e-6,
    color_demp_coeff=0.1,
    grow_scale_px=1.3,
    do_reset=False,
    child_color_coeff=0.1,
    bg_demp_coeff=0.1,
):
    """Densify (duplicate/split) and prune, following gsplat's DefaultStrategy.

    gsplat-style refinement adapted to the 2D additive renderer:

    * **signal**: ``avg_grad_norms`` is the per-gaussian means-gradient norm
      *averaged over every step of the epoch* (accumulated in the train
      step), not a one-shot end-of-epoch probe.
    * **duplicate vs split**: gradient-high gaussians that are *small*
      (``max(exp(log_diag)) <= grow_scale_px``) are duplicated in place
      (parent kept + identical copy); large ones are split into two children
      sampled from the parent (children cov = parent / 1.6, i.e. child
      ``L = L / sqrt(1.6)`` exactly — no cholesky round-trip needed).
    * **child soft start**: the renderer is additive (color plays opacity's
      role), so a naive duplicate/split doubles the contribution. New rows
      (split children, duplicate pairs) get their colors scaled by
      ``child_color_coeff`` (default 0.1 — the original strategy's value,
      which won the A/B against contribution-preserving 0.5, see
      EXPERIMENTS.md): newborns start nearly invisible and re-earn their
      contribution at high post-refine LR. The background is damped by
      ``bg_demp_coeff`` every refine for the same reason — it forces a
      global re-fit that levels the field for newborns. The full-population
      color damp (``x color_demp_coeff``) only runs when ``do_reset`` is
      set (gsplat decouples opacity resets from refinement the same way).
    * **prune**: low color norm or collapsed variance (unchanged).

    MLX 0.31 has no boolean indexing or ``nonzero``, so this op materializes
    everything eagerly through numpy. Split/prune is a per-epoch op, so the
    overhead of the materialization is negligible.

    Args:
        means, log_diag, offdiag, colors, background_color: current
            parameters (in the same form as :func:`init_gaussians` returns).
        avg_grad_norms: (N,) mean per-step means-gradient norm over the epoch.
        key: MLX key (uint32 array); seeds the numpy Generator for the split
            samples so refinement is reproducible.
        grad_thr: threshold on ``avg_grad_norms`` above which a gaussian is
            duplicated/split.
        color_demp_coeff: damping multiplier applied to all colors and the
            background when ``do_reset`` is set.
        grow_scale_px: size threshold (on ``max(exp(log_diag))``, in pixels)
            separating duplicate (small) from split (large).
        do_reset: apply the global color/background damping this round.

    Returns:
        ``(means, log_diag, offdiag, colors, background_color, info)`` with
        the gaussian count changed. ``info`` is a dict with ``idx_keep``
        (old-row indices of the surviving gaussians, in output order),
        ``num_new`` (rows appended after them: duplicates then split
        children — used to remap optimizer state), and ``n_dupli`` /
        ``n_split`` / ``n_prune`` counts for logging.
    """
    # Force evaluation of all inputs so we can do the work in numpy. ``mx.eval``
    # is side-effecting and returns ``None``; call it as a separate step.
    mx.eval(means, log_diag, offdiag, colors, background_color, avg_grad_norms)
    means_np = np.array(means)
    log_diag_np = np.array(log_diag)
    offdiag_np = np.array(offdiag)
    colors_np = np.array(colors)
    bg_np = np.array(background_color)
    g_norm = np.array(avg_grad_norms)
    rng = np.random.default_rng(np.array(key))

    mask_grad_high = g_norm > grad_thr
    # Erase low-color gaussians and gaussians whose variance has collapsed
    # below ``_MIN_VARIANCE_FOR_KEEP`` in either direction. A 2D gaussian
    # with very small variance along one axis is essentially a line — it
    # contributes nothing to a 2D image but produces precision-matrix
    # entries large enough to cause late-epoch NaNs. Erase takes priority
    # over refine: a gaussian flagged for both is degenerate and its
    # near-singular covariance makes bad split children.
    mask_low_color = np.linalg.norm(colors_np, axis=1) < 0.05
    min_log_diag_for_keep = 0.5 * np.log(_MIN_VARIANCE_FOR_KEEP)
    mask_low_variance = (log_diag_np[:, 0] < min_log_diag_for_keep) | (log_diag_np[:, 1] < min_log_diag_for_keep)
    mask_to_erase = mask_low_color | mask_low_variance

    # Duplicate small gradient-high gaussians, split large ones (gsplat's
    # grow_scale3d branch).
    mask_small = np.exp(log_diag_np).max(axis=1) <= grow_scale_px
    mask_to_dupli = mask_grad_high & mask_small & ~mask_to_erase
    mask_to_split = mask_grad_high & ~mask_small & ~mask_to_erase
    # Duplicated parents stay in the population; split parents are replaced
    # by their children.
    mask_keep = ~(mask_to_split | mask_to_erase)

    idx_split = np.where(mask_to_split)[0]
    idx_dupli = np.where(mask_to_dupli)[0]
    idx_keep = np.where(mask_keep)[0]

    # Scale the colors of duplicated parents *before* slicing kept rows so
    # parent and copy together stay close to what the parent contributed.
    colors_np = colors_np.copy()
    colors_np[idx_dupli] *= child_color_coeff

    kept_means = means_np[idx_keep]
    kept_log_diag = log_diag_np[idx_keep]
    kept_offdiag = offdiag_np[idx_keep]
    kept_colors = colors_np[idx_keep]

    # Duplicates: exact copies (colors already halved above).
    d_means = means_np[idx_dupli]
    d_log_diag = log_diag_np[idx_dupli]
    d_offdiag = offdiag_np[idx_dupli]
    d_colors = colors_np[idx_dupli]

    # Split children: two per parent, means sampled from the parent
    # (``mean + L @ z``, z ~ N(0, I) — same as gsplat's rotmat*scale*randn),
    # covariance = parent / 1.6, i.e. ``log_diag - 0.5*log(1.6)`` and
    # ``offdiag / sqrt(1.6)`` exactly. Colors halved to preserve the total
    # additive contribution.
    n_split = len(idx_split)
    if n_split > 0:
        Lp = np.zeros((n_split, 2, 2), dtype=np.float32)
        Lp[:, 0, 0] = np.exp(log_diag_np[idx_split, 0])
        Lp[:, 1, 1] = np.exp(log_diag_np[idx_split, 1])
        Lp[:, 1, 0] = offdiag_np[idx_split]
        z = rng.standard_normal((2, n_split, 2)).astype(np.float32)
        s_means = (means_np[idx_split][None] + np.einsum("nij,bnj->bni", Lp, z)).reshape(-1, 2)
        s_log_diag = np.tile(log_diag_np[idx_split] - 0.5 * np.log(1.6, dtype=np.float32), (2, 1))
        s_offdiag = np.tile(offdiag_np[idx_split] / np.sqrt(np.float32(1.6)), 2)
        s_colors = np.tile(colors_np[idx_split] * child_color_coeff, (2, 1))
    else:
        s_means = np.zeros((0, 2), dtype=np.float32)
        s_log_diag = np.zeros((0, 2), dtype=np.float32)
        s_offdiag = np.zeros((0,), dtype=np.float32)
        s_colors = np.zeros((0, 3), dtype=np.float32)

    new_means_np = np.concatenate([kept_means, d_means, s_means], axis=0).astype(np.float32)
    new_log_diag_np = np.concatenate([kept_log_diag, d_log_diag, s_log_diag], axis=0).astype(np.float32)
    new_offdiag_np = np.concatenate([kept_offdiag, d_offdiag, s_offdiag], axis=0).astype(np.float32)
    new_colors_np = np.concatenate([kept_colors, d_colors, s_colors], axis=0).astype(np.float32)

    # Periodic global reset (gsplat's reset_opa analog): damp all colors and
    # the background so accumulated over-bright gaussians have to re-earn
    # their contribution. Decoupled from the refine cadence via ``do_reset``.
    new_bg_np = bg_np * bg_demp_coeff
    if do_reset:
        new_colors_np = new_colors_np * color_demp_coeff
        new_bg_np = new_bg_np * color_demp_coeff

    info = {
        "idx_keep": idx_keep,
        "num_new": len(idx_dupli) + 2 * n_split,
        "n_dupli": len(idx_dupli),
        "n_split": n_split,
        "n_prune": int(mask_to_erase.sum()),
        "n_prune_color": int(mask_low_color.sum()),
        "n_prune_var": int(mask_low_variance.sum()),
    }
    return (
        mx.array(new_means_np),
        mx.array(new_log_diag_np),
        mx.array(new_offdiag_np),
        mx.array(new_colors_np),
        mx.array(new_bg_np),
        info,
    )


def carry_optimizer_state(old_optimizers, new_optimizers, idx_keep, num_new, optimize_background=True):
    """Preserve Adam state across a :func:`split_n_prune` (gsplat's
    ``_update_param_with_optimizer``): surviving gaussians keep their first
    and second moments (rows remapped by ``idx_keep``), new rows (duplicates
    and split children) start with zeroed moments, and the step counter —
    which drives both bias correction and the means LR schedule — carries
    over instead of restarting every epoch.

    NOTE: this is **off by default** (``gaussians.carry_optimizer_state:
    false``). The A/B in EXPERIMENTS.md shows carried moments lose badly on
    this workload: sustained Adam shrink momentum collapses variances en
    masse, the collapsed gaussians hit the variance prune, and the
    population stops growing. Kept for experimentation since it is the
    gsplat-orthodox behavior.

    Args:
        old_optimizers, new_optimizers: 5-tuples from :func:`set_up_optimizers`
            (new ones freshly built for the post-refine parameter shapes).
        idx_keep: old-row indices of surviving gaussians, in new-row order.
        num_new: number of appended rows (zeroed moments).
        optimize_background: whether the background optimizer exists.
    """
    # Moments are carried for means and colors only. Carrying the log_diag /
    # offdiag moments lets Adam's shrink momentum run uninterrupted across
    # refines, which collapses variances en masse (they then hit the
    # variance prune and the population stops growing — see EXPERIMENTS.md).
    # The old full-reset behavior interrupted that pressure every epoch;
    # resetting just the scale moments keeps the useful part.
    names = ["means", "log_diag", "offdiag", "colors"]
    carry_moments = ("means", "colors")
    for name, old_opt, new_opt in zip(names, old_optimizers[:4], new_optimizers[:4], strict=True):
        if name in carry_moments:
            for moment in ("m", "v"):
                old = np.array(old_opt.state[name][moment])
                new_rows = np.zeros((num_new,) + old.shape[1:], dtype=old.dtype)
                new_opt.state[name][moment] = mx.array(np.concatenate([old[idx_keep], new_rows], axis=0))
        new_opt.state["step"] = old_opt.state["step"]
    if optimize_background and old_optimizers[4] is not None:
        old_opt, new_opt = old_optimizers[4], new_optimizers[4]
        for moment in ("m", "v"):
            new_opt.state["background_color"][moment] = old_opt.state["background_color"][moment]
        new_opt.state["step"] = old_opt.state["step"]


def split_gaussian(mean, covariance, color, key, cov_scale=1.6):
    """Sample two child Gaussians from one parent.

    Each child inherits the parent's covariance scaled by ``cov_scale`` and
    the parent's color; the means are sampled from a multivariate normal
    centered on the parent.

    Note: ``mx.random.multivariate_normal`` is CPU-only (uses SVD), so the
    sample is taken on the CPU stream.
    """
    children_means = mx.random.multivariate_normal(
        mean=mean,
        cov=covariance,
        shape=(2,),
        dtype=mx.float32,
        key=key,
        stream=mx.Device(mx.cpu),
    )
    children_covs = mx.concatenate([covariance, covariance]) / cov_scale
    children_colors = mx.concatenate([color, color])
    return children_means, children_covs, children_colors
