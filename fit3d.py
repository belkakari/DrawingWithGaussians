"""Fit 3D Gaussians (gaussian splatting) to an image with MLX.

MLX port of gsplat's ``examples/image_fitting.py`` (3dgs mode) plus the
densification strategy from the 2D path (see gaussian3d.py and the A/B in
EXPERIMENTS.md): training runs ``num_epochs`` epochs of ``num_steps`` steps,
refining (duplicate/split/prune) at every epoch boundary except the last.
``optim.num_epochs: 1`` disables densification entirely (fixed-N training,
the original gsplat image_fitting behavior).

The densification signal is the screen-space means2d gradient. By default it
uses gsplat-style ``absgrad``: the fused rasterizer accumulates per-pixel
absolute means2d-gradient contributions into a dummy zero parameter's VJP.
Projection is regular MLX autodiff (rendering3d.py); rasterization is the
fused Metal alpha-compositing kernels (rendering3d_fused.py). The whole train
step is one ``mx.compile`` region, rebuilt per epoch.

Run with:
    uv run python fit3d.py --config-name fit_to_image_3d.yaml
"""

import logging
import math
import time
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import hydra  # type: ignore[import-not-found]
import mlx.core as mx
import numpy as np
from omegaconf import DictConfig, OmegaConf  # type: ignore[import-not-found]
from PIL import Image

from drawingwithgaussians.gaussian3d import (
    carry_optimizer_state_3d,
    init_gaussians_3d,
    set_up_optimizer_3d,
    split_n_prune_3d,
)
from drawingwithgaussians.losses import pixel_loss_3d
from drawingwithgaussians.rendering3d import FAR_PLANE, NEAR_PLANE, project_gaussians
from drawingwithgaussians.rendering3d_fused import _num_tiles, estimate_bin_capacity
from drawingwithgaussians.schedule import MomentumBudget, resolution_schedule
from drawingwithgaussians.sh import view_dependent_colors
from drawingwithgaussians.splat_export import export_ply_3d


@hydra.main(version_base=None, config_path="./configs")
def fit3d(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = Path(hydra_cfg["runtime"]["output_dir"])

    if cfg.optim.loss.name != "pixel":
        raise NotImplementedError(f"loss {cfg.optim.loss.name!r} is not supported; only 'pixel'.")

    height = int(cfg.image.height)
    width = int(cfg.image.width)
    num_epochs = int(cfg.optim.num_epochs)
    max_steps = int(cfg.optim.num_steps)
    if min(height, width, num_epochs, max_steps) <= 0:
        raise ValueError("image dimensions, num_epochs, and num_steps must be positive")
    total_steps = num_epochs * max_steps
    ssim_weight = cfg.optim.loss.ssim_weight

    with Image.open(cfg.image.path) as img:
        # PIL takes (width, height). Explicit RGB conversion also handles
        # grayscale, paletted, and RGBA inputs consistently.
        target_np = np.asarray(img.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
    target_image = mx.array(target_np)

    # Fixed pinhole camera (gsplat image_fitting setup).
    fov_x = math.radians(cfg.camera.fov_x_deg)
    focal = 0.5 * width / math.tan(0.5 * fov_x)
    K = mx.array(
        np.array(
            [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
            dtype=np.float32,
        )
    )
    viewmat = mx.array(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, float(cfg.camera.camera_z)],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )

    initial_n = int(cfg.gaussians.initial_num_gaussians)
    prune_scale3d = float(cfg.gaussians.get("prune_scale3d", 0.0))
    scene_scale = float(cfg.gaussians.scene_scale)
    configured_scale_max = cfg.gaussians.get("init_scale_max", None)
    if configured_scale_max is None:
        # Do not initialize rows already beyond the active too-large prune
        # threshold. This matters especially for deliberate low-N starts.
        init_scale_max = min(1.0, 0.95 * prune_scale3d * scene_scale) if prune_scale3d > 0 else 1.0
    else:
        init_scale_max = float(configured_scale_max)
    params = init_gaussians_3d(
        initial_n,
        mx.random.key(cfg.optim.seed),
        scale_min=float(cfg.gaussians.get("init_scale_min", 1e-3)),
        scale_max=init_scale_max,
    )
    mx.eval(*params.values(), target_image, K, viewmat)

    # --- DashGaussian schedulers (arXiv:2503.18402), all defaulting off ------
    resolution_mode = cfg.gaussians.get("resolution_mode", "const")
    densify_mode = cfg.gaussians.get("densify_mode", "free")
    lr_decay_from_full_res = bool(cfg.gaussians.get("lr_decay_from_full_res", False))

    # Per-epoch integer downscale factors (1:1 epoch -> factor, last == 1). For
    # resolution_mode="const" this is all ones (no behavior change).
    if resolution_mode == "freq":
        res_factors = resolution_schedule(
            np.array(target_image),
            num_epochs,
            max_steps,
            start_significance_factor=float(cfg.gaussians.get("start_significance_factor", 4.0)),
        )
    else:
        res_factors = [1] * num_epochs
    log.info(f"Resolution schedule (per-epoch downscale, mode={resolution_mode}): {res_factors}")

    # Momentum primitive-count budget (only consulted in densify_mode="budget").
    budget = MomentumBudget(
        cfg.gaussians.initial_num_gaussians,
        gamma=float(cfg.gaussians.get("budget_gamma", 0.98)),
        eta=float(cfg.gaussians.get("budget_eta", 1.0)),
        max_n_gaussian=int(cfg.gaussians.get("max_n_gaussian", -1)),
    )
    budget_grad_percentile = float(cfg.gaussians.get("budget_grad_percentile", 50.0))
    max_densify_rate = float(cfg.gaussians.get("max_densify_rate_per_step", 0.2))
    min_n_gaussian = int(cfg.gaussians.get("min_n_gaussian", -1))
    if min_n_gaussian <= 0:
        min_n_gaussian = initial_n
    # LR decay begins at the first full-resolution epoch (DashGaussian LR delay).
    full_res_epoch = next((e for e, f in enumerate(res_factors) if f == 1), num_epochs - 1)

    def epoch_resolution(r):
        """Downscaled target + intrinsics for a per-epoch downscale factor ``r``.
        ``K`` is rebuilt from the resized dims (not scaled by 1/r) so the
        principal point stays centered even when ``r`` doesn't divide evenly."""
        if r <= 1:
            return target_image, K, height, width
        h_r = max(1, int(round(height / r)))
        w_r = max(1, int(round(width / r)))
        tgt_r = cv2.resize(np.array(target_image), (w_r, h_r), interpolation=cv2.INTER_AREA)
        target_r = mx.array(tgt_r.astype(np.float32))
        focal_r = 0.5 * w_r / math.tan(0.5 * fov_x)
        K_r = mx.array(np.array([[focal_r, 0, w_r / 2], [0, focal_r, h_r / 2], [0, 0, 1]], dtype=np.float32))
        mx.eval(target_r, K_r)
        return target_r, K_r, h_r, w_r

    def make_optimizer(params, epoch=0):
        if lr_decay_from_full_res and resolution_mode == "freq":
            step_offset = epoch * max_steps
            decay_from_step = full_res_epoch * max_steps
        else:
            step_offset = 0
            decay_from_step = 0
        return set_up_optimizer_3d(
            params,
            lr=cfg.optim.lr,
            max_steps=total_steps,
            mode=cfg.optim.means_mode,
            restart_period=max_steps,
            step_offset=step_offset,
            decay_from_step=decay_from_step,
        )

    opt = make_optimizer(params, epoch=0)

    use_absgrad = bool(cfg.gaussians.get("absgrad", False))

    def choose_bins(params, K_e, w_e, h_e):
        """Pick compact-bin settings for this epoch, at the epoch's render
        resolution ``(w_e, h_e)`` and intrinsics ``K_e``.

        ``exact``/``None`` keeps the old all-tiles capacity. Integer values keep
        the historical per-gaussian ``bin_pad`` meaning. ``auto`` estimates the
        exact intersection count with the same Metal counter used by the
        compact builder, applies a margin, and retraces this epoch's compiled
        step with that static capacity.
        """
        mode = cfg.gaussians.get("bin_pad", "auto")
        if mode is None or str(mode).lower() in {"none", "exact"}:
            return None, None, "exact"
        if str(mode).lower() != "auto":
            pad = int(mode)
            capacity = max(
                1,
                min(
                    _num_tiles(w_e, h_e) * params["means3d"].shape[0],
                    params["means3d"].shape[0] * pad,
                ),
            )
            return None, capacity, f"capacity={capacity} (pad={pad})"

        means2d, conics, depths = project_gaussians(
            params["means3d"],
            params["log_scales"],
            params["quats"],
            viewmat,
            K_e,
            w_e,
            h_e,
        )
        opacities = mx.where(
            (depths > NEAR_PLANE) & (depths < FAR_PLANE),
            mx.sigmoid(params["opacities_raw"]),
            0.0,
        )
        capacity = estimate_bin_capacity(
            means2d,
            conics,
            opacities,
            w_e,
            h_e,
            margin=float(cfg.gaussians.get("bin_pad_margin", 2.0)),
            min_per_gaussian=int(cfg.gaussians.get("bin_pad_min", 16)),
        )
        return None, capacity, f"capacity={capacity}"

    def count_current_intersections(params, K_e, w_e, h_e) -> int:
        means2d, conics, depths = project_gaussians(
            params["means3d"],
            params["log_scales"],
            params["quats"],
            viewmat,
            K_e,
            w_e,
            h_e,
        )
        opacities = mx.where(
            (depths > NEAR_PLANE) & (depths < FAR_PLANE),
            mx.sigmoid(params["opacities_raw"]),
            0.0,
        )
        return estimate_bin_capacity(means2d, conics, opacities, w_e, h_e, margin=1.0, min_per_gaussian=0)

    def make_step(bin_pad, bin_capacity, target_e, K_e) -> tuple[Any, list[Any]]:
        def loss_fn(params, means2d_offset, means2d_absgrad_sink):
            # A single fixed camera underdetermines higher SH bands, so fit3d
            # intentionally stays at degree zero while sharing the schema.
            colors = view_dependent_colors(params, viewmat, active_degree=0)
            return pixel_loss_3d(
                params["means3d"],
                params["log_scales"],
                params["quats"],
                params["opacities_raw"],
                colors,
                target_e,
                viewmat,
                K_e,
                ssim_weight=ssim_weight,
                means2d_offset=means2d_offset,
                means2d_absgrad_sink=means2d_absgrad_sink,
                bin_pad=bin_pad,
                bin_capacity=bin_capacity,
            )

        # Gradient w.r.t. the params dict, the zero screen-space offset (net
        # densification signal), and the ignored absgrad sink (absolute signal).
        loss_and_grad = mx.value_and_grad(loss_fn, argnums=[0, 1, 2])
        state = [opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_step(params, offset_zeros, absgrad_zeros, grad_accum):
            (loss, rendered), (grads, offset_grad, absgrad_grad) = loss_and_grad(params, offset_zeros, absgrad_zeros)
            signal_grad = absgrad_grad if use_absgrad else offset_grad
            grad_accum = grad_accum + mx.sqrt(mx.sum(signal_grad * signal_grad, axis=1))
            params = opt.apply_gradients(grads, params)
            return loss, rendered, params, grad_accum

        return compiled_step, state

    frames = []
    ts = time.perf_counter()
    for num_epoch in range(num_epochs):
        # This epoch's render resolution (coarse->fine under resolution_mode=freq).
        r = res_factors[num_epoch]
        target_e, K_e, h_e, w_e = epoch_resolution(r)
        bin_pad, bin_capacity, bin_label = choose_bins(params, K_e, w_e, h_e)
        log.info(f"Using 3D raster bins {bin_label} at epoch {num_epoch} (downscale r={r}, {w_e}x{h_e})")
        compiled_step, state = make_step(bin_pad, bin_capacity, target_e, K_e)
        n = params["means3d"].shape[0]
        offset_zeros = mx.zeros((n, 2), dtype=mx.float32)
        absgrad_zeros = mx.zeros((n, 2), dtype=mx.float32)
        grad_accum = mx.zeros((n,), dtype=mx.float32)
        for step_idx in range(max_steps):
            if bin_capacity is not None:
                real_isects = count_current_intersections(params, K_e, w_e, h_e)
                if real_isects > bin_capacity:
                    old_capacity = bin_capacity
                    exact_capacity = _num_tiles(w_e, h_e) * n
                    min_capacity = int(cfg.gaussians.get("bin_pad_min", 16)) * n
                    overflow_margin = max(1.01, float(cfg.gaussians.get("bin_overflow_margin", 1.25)))
                    bumped = max(min_capacity, math.ceil(real_isects * overflow_margin))
                    bin_capacity = max(1, min(exact_capacity, bumped))
                    log.warning(
                        "bin capacity overflow before step %d: real=%d > capacity=%d; recompiling with capacity=%d",
                        step_idx,
                        real_isects,
                        old_capacity,
                        bin_capacity,
                    )
                    compiled_step, state = make_step(bin_pad, bin_capacity, target_e, K_e)
            loss, rendered, params, grad_accum = compiled_step(params, offset_zeros, absgrad_zeros, grad_accum)
            mx.eval(loss, rendered, grad_accum, *params.values(), *state)

            if not math.isfinite(loss.item()):
                log.error("Loss became non-finite, stopping.")
                break

            if step_idx % cfg.train.log_frequency == 0:
                log.info(
                    f"Loss: {float(loss):.5f}, step: {step_idx}, at epoch {num_epoch} / "
                    f"{num_epochs}, num gaussians: {n}, "
                    f"time per step: {(time.perf_counter() - ts) / cfg.train.log_frequency:.4f}"
                )
                ts = time.perf_counter()
                # Store frames at full resolution so the video strip never
                # shape-mismatches the full-res target under a resolution schedule.
                frame_np = np.clip(np.array(rendered), 0, 1).astype(np.float32)
                if frame_np.shape[:2] != (height, width):
                    frame_np = cv2.resize(frame_np, (width, height), interpolation=cv2.INTER_NEAREST)
                frames.append(frame_np)

        # End-of-epoch refinement; skipped after the final epoch (and thus
        # entirely when num_epochs == 1 — fixed-N training).
        if num_epoch == num_epochs - 1:
            break
        avg_grad_norms = grad_accum / float(max_steps)
        signal_np = np.asarray(avg_grad_norms)
        log.info(
            "Refine signal after epoch %d: p50=%.3g p95=%.3g max=%.3g threshold=%.3g",
            num_epoch,
            float(np.percentile(signal_np, 50)),
            float(np.percentile(signal_np, 95)),
            float(signal_np.max()),
            float(cfg.gaussians.grad_thr),
        )
        # Budget mode: the new gaussians train at the *next* epoch's resolution,
        # so the count target uses that epoch's downscale and start step (Eq. 4).
        if densify_mode == "budget":
            next_epoch = num_epoch + 1
            target_count = budget.target_count(res_factors[next_epoch], next_epoch * max_steps, total_steps)
        else:
            target_count = None
        params, refine_info = split_n_prune_3d(
            params,
            avg_grad_norms,
            mx.random.key(cfg.optim.seed + num_epoch + 1),
            grad_thr=cfg.gaussians.grad_thr,
            grow_scale=cfg.gaussians.grow_scale,
            scene_scale=cfg.gaussians.scene_scale,
            prune_opa=cfg.gaussians.prune_opa,
            prune_scale3d=cfg.gaussians.get("prune_scale3d", None),
            densify_mode=densify_mode,
            target_count=target_count,
            grad_percentile=budget_grad_percentile,
            max_densify_rate=max_densify_rate,
            min_n_gaussian=min_n_gaussian,
            split_oversized_high_grad=bool(cfg.gaussians.get("split_oversized_high_grad", True)),
        )
        if densify_mode == "budget":
            budget.update(refine_info["n_densified"])
            log.info(
                f"Budget after epoch {num_epoch}: target={target_count}, "
                f"densified k={refine_info['n_densified']}, P_fin={budget.p_fin}"
            )
        log.info(
            f"Refine after epoch {num_epoch}: {refine_info['n_dupli']} duplicated, "
            f"{refine_info['n_split']} split, {refine_info['n_prune']} pruned "
            f"(opa={refine_info.get('n_prune_opa', 0)}, scale={refine_info.get('n_prune_scale3d', 0)}) "
            f"-> {params['means3d'].shape[0]} gaussians"
        )
        old_opt = opt
        opt = make_optimizer(params, epoch=num_epoch + 1)
        if bool(cfg.gaussians.get("carry_optimizer_state", False)):
            carry_optimizer_state_3d(old_opt, opt, params, refine_info["idx_keep"], refine_info["num_new"])

    ply_path = export_ply_3d(params, out_dir / "final.ply")
    log.info(f"Saved SuperSplat export: {ply_path}")

    width_out = width * 2
    out = cv2.VideoWriter(
        str(out_dir / "outpy.avi"),
        cv2.VideoWriter.fourcc("M", "J", "P", "G"),
        24,
        (width_out, height),
    )
    target_np = np.array(target_image)
    for frame in frames:
        g = (np.clip(np.array(frame), 0, 1) * 255).astype(np.uint8)
        i = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
        out.write(np.hstack([g, i])[:, :, ::-1])
    out.release()


if __name__ == "__main__":
    fit3d()  # type: ignore[call-arg]
