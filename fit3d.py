"""Fit 3D Gaussians (gaussian splatting) to an image with MLX.

MLX port of gsplat's ``examples/image_fitting.py`` (3dgs mode) plus the
densification strategy from the 2D path (see gaussian3d.py and the A/B in
EXPERIMENTS.md): training runs ``num_epochs`` epochs of ``num_steps`` steps,
refining (duplicate/split/prune) at every epoch boundary except the last.
``optim.num_epochs: 1`` disables densification entirely (fixed-N training,
the original gsplat image_fitting behavior).

The densification signal is the screen-space means2d gradient, obtained by
adding a zero ``means2d_offset`` parameter to the projected means (the MLX
equivalent of gsplat's ``retain_grad``). Projection is regular MLX autodiff
(rendering3d.py); rasterization is the fused Metal alpha-compositing kernels
(rendering3d_fused.py). The whole train step is one ``mx.compile`` region,
rebuilt per epoch.

Run with:
    poetry run python fit3d.py --config-name fit_to_image_3d.yaml
"""

import logging
import math
import time
from functools import partial
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import mlx.core as mx
from drawingwithgaussians.gaussian3d import (
    carry_optimizer_state_3d,
    init_gaussians_3d,
    set_up_optimizer_3d,
    split_n_prune_3d,
)
from drawingwithgaussians.losses import pixel_loss_3d


@hydra.main(version_base=None, config_path="./configs")
def fit3d(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(f"Running with config:\n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = Path(hydra_cfg["runtime"]["output_dir"])

    if cfg.optim.loss.name != "pixel":
        raise NotImplementedError(f"loss {cfg.optim.loss.name!r} is not supported; only 'pixel'.")

    height = cfg.image.height
    width = cfg.image.width
    num_epochs = cfg.optim.num_epochs
    max_steps = cfg.optim.num_steps
    total_steps = num_epochs * max_steps
    ssim_weight = cfg.optim.loss.ssim_weight

    img = Image.open(cfg.image.path)
    target_image = mx.array(np.array(img.resize((height, width)), dtype=np.float32)[:, :, :3] / 255)

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
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, float(cfg.camera.camera_z)], [0, 0, 0, 1]],
            dtype=np.float32,
        )
    )

    params = init_gaussians_3d(cfg.gaussians.initial_num_gaussians, mx.random.key(cfg.optim.seed))
    mx.eval(*params.values(), target_image, K, viewmat)

    def make_optimizer(params):
        return set_up_optimizer_3d(
            params,
            lr=cfg.optim.lr,
            max_steps=total_steps,
            mode=cfg.optim.means_mode,
            restart_period=max_steps,
        )

    opt = make_optimizer(params)

    def loss_fn(params, means2d_offset):
        return pixel_loss_3d(
            params["means3d"],
            params["log_scales"],
            params["quats"],
            params["opacities_raw"],
            params["colors_raw"],
            target_image,
            viewmat,
            K,
            ssim_weight=ssim_weight,
            means2d_offset=means2d_offset,
        )

    # Gradient w.r.t. the params dict and the zero screen-space offset (the
    # densification signal).
    loss_and_grad = mx.value_and_grad(loss_fn, argnums=[0, 1])

    def make_step():
        state = [opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def compiled_step(params, offset_zeros, grad_accum):
            (loss, rendered), (grads, offset_grad) = loss_and_grad(params, offset_zeros)
            grad_accum = grad_accum + mx.sqrt(mx.sum(offset_grad * offset_grad, axis=1))
            params = opt.apply_gradients(grads, params)
            return loss, rendered, params, grad_accum

        return compiled_step, state

    frames = []
    ts = time.perf_counter()
    for num_epoch in range(num_epochs):
        compiled_step, state = make_step()
        n = params["means3d"].shape[0]
        offset_zeros = mx.zeros((n, 2), dtype=mx.float32)
        grad_accum = mx.zeros((n,), dtype=mx.float32)
        for step_idx in range(max_steps):
            loss, rendered, params, grad_accum = compiled_step(params, offset_zeros, grad_accum)
            mx.eval(loss, rendered, grad_accum, *params.values(), *state)

            if math.isnan(loss.item()):
                log.error("Loss became NaN, stopping.")
                break

            if step_idx % cfg.train.log_frequency == 0:
                log.info(
                    f"Loss: {float(loss):.5f}, step: {step_idx}, at epoch {num_epoch} / "
                    f"{num_epochs}, num gaussians: {n}, "
                    f"time per step: {(time.perf_counter() - ts) / cfg.train.log_frequency:.4f}"
                )
                ts = time.perf_counter()
                frames.append(rendered)

        # End-of-epoch refinement; skipped after the final epoch (and thus
        # entirely when num_epochs == 1 — fixed-N training).
        if num_epoch == num_epochs - 1:
            break
        avg_grad_norms = grad_accum / float(max_steps)
        params, refine_info = split_n_prune_3d(
            params,
            avg_grad_norms,
            mx.random.key(cfg.optim.seed + num_epoch + 1),
            grad_thr=cfg.gaussians.grad_thr,
            grow_scale=cfg.gaussians.grow_scale,
            scene_scale=cfg.gaussians.scene_scale,
            prune_opa=cfg.gaussians.prune_opa,
        )
        log.info(
            f"Refine after epoch {num_epoch}: {refine_info['n_dupli']} duplicated, "
            f"{refine_info['n_split']} split, {refine_info['n_prune']} pruned "
            f"-> {params['means3d'].shape[0]} gaussians"
        )
        old_opt = opt
        opt = make_optimizer(params)
        if bool(cfg.gaussians.get("carry_optimizer_state", False)):
            carry_optimizer_state_3d(old_opt, opt, params, refine_info["idx_keep"], refine_info["num_new"])

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
        out.write(np.hstack([g, i])[:, :, ::-1])
    out.release()


if __name__ == "__main__":
    fit3d()
