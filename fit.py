"""Fit 2D Gaussians to an image with MLX.

MLX-only port of the original JAX fit.py. Supports the pixel-loss path only;
diffusion guidance was intentionally not ported. Follows the MLX examples
pattern of:

* flat params dict with per-parameter Adam optimizers (one cosine-decayed
  for means, constant for the rest);
* per-step ``mx.eval`` to realize the GPU work;
* ``mx.isnan`` for NaN guarding.

The rasterizer runs as fused Metal kernels (rendering2d_fused.py, gsplat
kernel structure) and the whole train step — loss/grad plus the five Adam
updates — is one ``mx.compile`` region with optimizer state threaded via
``inputs=``/``outputs=`` (the standard MLX pattern). The step is rebuilt per
epoch because ``split_n_prune`` changes N and rebuilds the optimizers; the
retrace cost is negligible. See EXPERIMENTS.md for measurements.

Run with:
    uv run python fit.py --config-name fit_to_image.yaml
"""

import logging
import math
import time
from functools import partial
from pathlib import Path

import cv2
import hydra
import mlx.core as mx
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from drawingwithgaussians.gaussian import carry_optimizer_state, init_gaussians, set_up_optimizers, split_n_prune
from drawingwithgaussians.losses import pixel_loss


@hydra.main(version_base=None, config_path="./configs")
def fit(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = Path(hydra_cfg["runtime"]["output_dir"])

    if cfg.optim.loss.name != "pixel":
        raise NotImplementedError(f"loss {cfg.optim.loss.name!r} is not ported to MLX; only 'pixel' is supported.")

    height = cfg.image.height
    width = cfg.image.width
    num_epochs = cfg.optim.num_epochs
    max_steps = cfg.optim.num_steps
    ssim_weight = cfg.optim.loss.ssim_weight
    optimize_bg = cfg.optim.optimize_background

    # Load and resize the target image.
    img = Image.open(cfg.image.path)
    target_image = mx.array(np.array(img.resize((height, width)), dtype=np.float32)[:, :, :3] / 255)

    # Initialize gaussian parameters and optimizers. L is parameterized as
    # ``(log_diag, offdiag)`` — see drawingwithgaussians.gaussian.
    means, log_diag, offdiag, colors, background_color = init_gaussians(
        num_gaussians=cfg.gaussians.initial_num_gaussians,
        target_image=target_image,
        key=mx.random.key(cfg.optim.seed),
        optimize_background=optimize_bg,
    )
    # means LR schedule: "cos_restart" (default) warm-restarts every epoch
    # via step % max_steps; "cos" decays once over the whole run (needs
    # carry_optimizer_state so the step counter survives refines).
    total_steps = num_epochs * max_steps
    optimizers = list(
        set_up_optimizers(
            means,
            log_diag,
            offdiag,
            colors,
            background_color,
            lr=cfg.optim.lr,
            max_steps=total_steps,
            means_mode=cfg.optim.means_mode,
            optimize_background=optimize_bg,
            restart_period=max_steps,
        )
    )

    # Densification knobs (gsplat DefaultStrategy analogs, see gaussian.py).
    grow_scale_px = float(cfg.gaussians.get("grow_scale", 0.01)) * height
    reset_every_epochs = int(cfg.gaussians.get("reset_every_epochs", 0))

    frames = []

    def loss_fn(m, ld, od, c, b, t):
        return pixel_loss(m, ld, od, c, b, t, ssim_weight=ssim_weight)

    # Gradient w.r.t. (means, log_diag, offdiag, colors[, background]).
    # Compiled: fuses the (N, P) elementwise chains in the rasterizer.
    # Gradients are bit-identical to the uncompiled version; the per-epoch
    # shape change from split_n_prune only costs one retrace per epoch.
    loss_and_grad = mx.compile(
        mx.value_and_grad(
            loss_fn,
            argnums=[0, 1, 2, 3] + ([4] if optimize_bg else []),
        )
    )

    def make_step():
        """Build the compiled train step for the *current* optimizers.

        The whole step — loss/grad plus the five Adam updates — is one
        ``mx.compile`` region (the standard MLX pattern: optimizer state is
        threaded through ``inputs=``/``outputs=`` so the in-place state
        mutation is captured). This fuses the ~50 small optimizer kernels;
        must be rebuilt whenever the optimizers are rebuilt (per epoch).
        """
        (
            opt_means,
            opt_log_diag,
            opt_offdiag,
            opt_colors,
            opt_bg,
        ) = optimizers
        assert opt_means is not None and opt_log_diag is not None and opt_offdiag is not None and opt_colors is not None
        state = [
            opt_means.state,
            opt_log_diag.state,
            opt_offdiag.state,
            opt_colors.state,
        ] + ([opt_bg.state] if optimize_bg else [])

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_step(means, log_diag, offdiag, colors, background_color, grad_accum):
            bg = background_color if optimize_bg else mx.zeros((1, 1, 3), mx.float32)
            (loss, rendered), grads = loss_and_grad(means, log_diag, offdiag, colors, bg, target_image)
            # Densification signal: running sum of per-step means-grad norms
            # (gsplat's grad2d state; every gaussian is "visible" every step
            # here, so the count is just the step count).
            grad_accum = grad_accum + mx.sqrt(mx.sum(grads[0] * grads[0], axis=1))
            means = opt_means.apply_gradients({"means": grads[0]}, {"means": means})["means"]
            log_diag = opt_log_diag.apply_gradients({"log_diag": grads[1]}, {"log_diag": log_diag})["log_diag"]
            offdiag = opt_offdiag.apply_gradients({"offdiag": grads[2]}, {"offdiag": offdiag})["offdiag"]
            colors = opt_colors.apply_gradients({"colors": grads[3]}, {"colors": colors})["colors"]
            if optimize_bg:
                background_color = opt_bg.apply_gradients(
                    {"background_color": grads[4]},
                    {"background_color": background_color},
                )["background_color"]
            return (
                loss,
                rendered,
                means,
                log_diag,
                offdiag,
                colors,
                background_color,
                grad_accum,
            )

        def step():
            nonlocal means, log_diag, offdiag, colors, background_color, grad_accum
            (
                loss,
                rendered,
                means,
                log_diag,
                offdiag,
                colors,
                background_color,
                grad_accum,
            ) = compiled_step(means, log_diag, offdiag, colors, background_color, grad_accum)
            # Realize the GPU work; cheap since arrays are small.
            mx.eval(
                means,
                log_diag,
                offdiag,
                colors,
                background_color,
                grad_accum,
                loss,
                *state,
            )
            return loss, rendered

        return step

    ts = time.perf_counter()
    for num_epoch in range(num_epochs):
        step = make_step()
        # Fresh densification-signal accumulator each epoch (gsplat zeroes
        # its grad2d/count running state after every refine).
        grad_accum = mx.zeros((means.shape[0],), dtype=mx.float32)
        for step_idx in range(max_steps):
            loss, rendered = step()

            if math.isnan(loss.item()):
                log.error("Loss became NaN, stopping.")
                break

            if step_idx % cfg.train.log_frequency == 0:
                # Track L's diagonal in the *actual* (post-exp) scale — that's
                # the scale the precision matrix sees. With log-diag
                # parameterization the diagonal is always positive; we use
                # the min as a stress test for "vanishingly small gaussian".
                actual_diag = mx.exp(log_diag)
                mx.eval(actual_diag)
                d_min = float(actual_diag.min())
                d_max = float(actual_diag.max())
                d_mean = float(actual_diag.mean())
                log.info(
                    f"Loss: {float(loss):.5f}, step: {step_idx}, at epoch {num_epoch} / "
                    f"{num_epochs}, num gaussians: {means.shape[0]}, "
                    f"L_diag min/mean/max: {d_min:.4f}/{d_mean:.4f}/{d_max:.4f}, "
                    f"time per step: {(time.perf_counter() - ts) / cfg.train.log_frequency:.4f}"
                )
                ts = time.perf_counter()
                frames.append(rendered)

        # End-of-epoch refinement (gsplat DefaultStrategy analog). Skip after
        # the final epoch — gsplat likewise stops refining before training
        # ends so the population converges (refine_stop_iter).
        if num_epoch == num_epochs - 1:
            break

        # Densification signal: mean per-step means-grad norm over the epoch.
        avg_grad_norms = grad_accum / float(max_steps)

        # Global color reset decoupled from the refine cadence (gsplat's
        # reset_every); 0 disables it. Never fires right before the end.
        do_reset = reset_every_epochs > 0 and (num_epoch + 1) % reset_every_epochs == 0

        # Split/prune is implemented eagerly in numpy because MLX 0.31 has no
        # boolean indexing / nonzero. It mutates gaussian count and shape so
        # we rebuild the optimizers and carry their state over.
        (
            means,
            log_diag,
            offdiag,
            colors,
            background_color,
            refine_info,
        ) = split_n_prune(
            means,
            log_diag,
            offdiag,
            colors,
            background_color,
            avg_grad_norms,
            mx.random.key(cfg.optim.seed + num_epoch + 1),
            grad_thr=cfg.gaussians.grad_thr,
            color_demp_coeff=cfg.gaussians.color_demp_coeff,
            grow_scale_px=grow_scale_px,
            do_reset=do_reset,
            child_color_coeff=float(cfg.gaussians.get("child_color_coeff", 0.5)),
            bg_demp_coeff=float(cfg.gaussians.get("bg_demp_coeff", 1.0)),
        )
        log.info(
            f"Refine after epoch {num_epoch}: {refine_info['n_dupli']} duplicated, "
            f"{refine_info['n_split']} split, {refine_info['n_prune']} pruned "
            f"(color {refine_info['n_prune_color']} / var {refine_info['n_prune_var']}), "
            f"reset={do_reset} -> {means.shape[0]} gaussians"
        )

        # Rebuild optimizers for the new gaussian count, then carry over the
        # Adam moments of surviving gaussians and the step counter (new rows
        # start with zeroed moments) — gsplat's _update_param_with_optimizer.
        old_optimizers = optimizers
        optimizers = list(
            set_up_optimizers(
                means,
                log_diag,
                offdiag,
                colors,
                background_color,
                lr=cfg.optim.lr,
                max_steps=total_steps,
                means_mode=cfg.optim.means_mode,
                optimize_background=optimize_bg,
                restart_period=max_steps,
            )
        )
        if bool(cfg.gaussians.get("carry_optimizer_state", True)):
            carry_optimizer_state(
                old_optimizers,
                optimizers,
                refine_info["idx_keep"],
                refine_info["num_new"],
                optimize_background=optimize_bg,
            )

    # Video output.
    width_out = width * 2
    out = cv2.VideoWriter(
        str(out_dir / "outpy.avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        24,
        (width_out, height),
    )
    target_np = np.array(target_image)
    for frame in frames:
        g = (np.clip(np.array(frame), 0, 1) * 255).astype(np.uint8)
        i = (np.clip(target_np, 0, 1) * 255).astype(np.uint8)
        processed = np.hstack([g, i])
        out.write(processed[:, :, ::-1])
    out.release()


if __name__ == "__main__":
    fit()
