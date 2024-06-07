import logging
from pathlib import Path

import cv2
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from drawingwithgaussians.twod.gaussian import init_gaussians, set_up_optimizers, split_n_prune, update
from drawingwithgaussians.twod.losses import diffusion_guidance, pixel_loss
from drawingwithgaussians.twod.sds_pipeline import FlaxStableDiffusionImg2ImgPipeline


@hydra.main(version_base=None, config_path="./configs")
def fit(cfg: DictConfig):
    log = logging.getLogger(__name__)
    log.info(f"Running with config: \n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = Path(hydra_cfg["runtime"]["output_dir"])

    key = jax.random.key(cfg.optim.seed)

    img = Image.open(cfg.image.path)

    height = cfg.image.height
    width = cfg.image.width
    num_epochs = cfg.optim.num_epochs
    max_steps = cfg.optim.num_steps

    target_image = jnp.array(img.resize((height, width)), dtype=jnp.float32)[:, :, :3] / 255

    means, L, colors, rotmats, background_color = init_gaussians(
        num_gaussians=cfg.gaussians.initial_num_gaussians,
        target_image=target_image,
        key=key,
        optimize_background=cfg.optim.optimize_background,
    )
    optimizers = set_up_optimizers(
        means,
        L,
        colors,
        rotmats,
        background_color,
        lr=cfg.optim.lr,
        max_steps=cfg.optim.num_steps,
        means_mode=cfg.optim.means_mode,
        optimize_background=cfg.optim.optimize_background,
    )

    if cfg.optim.loss.name == "diffusion_guidance":
        dtype = jnp.bfloat16
        pipeline, params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="bf16",
            dtype=dtype,
        )

    prev_stats = []
    frames = []

    if cfg.optim.optimize_background:
        grad_argnums = [0, 1, 2, 3, 4]
    else:
        grad_argnums = [0, 1, 2, 3]

    if cfg.optim.loss.name == "diffusion_guidance":
        loss_grad = jax.value_and_grad(diffusion_guidance, argnums=grad_argnums, has_aux=True)
    elif cfg.optim.loss.name == "pixel":
        loss_grad = jax.value_and_grad(pixel_loss, argnums=grad_argnums, has_aux=True)

    for num_epoch in range(num_epochs):
        for step in range(max_steps):
            if cfg.optim.loss.name == "diffusion_guidance":
                strength = cfg.optim.loss.strength
                if cfg.optim.loss.strength_annealing:
                    strength = strength * ((num_epochs * max_steps - num_epoch * step) / (num_epochs * max_steps))

                if step % cfg.optim.loss.img2img_freq == 0:
                    target_image = None
                else:
                    target_image = jnp.copy(diffusion_image)
                (loss, (renderred_gaussians, diffusion_image)), gradients = loss_grad(
                    means,
                    L,
                    colors,
                    rotmats,
                    background_color,
                    prompt=cfg.optim.loss.prompt,
                    key=key,
                    shape=(height, width, 3),
                    diffusion_shape=(cfg.optim.loss.height, cfg.optim.loss.height, 3),
                    num_steps=cfg.optim.loss.num_steps,
                    strength=strength,
                    pipeline=pipeline,
                    params=params,
                    dtype=dtype,
                    cfg_scale=cfg.optim.loss.cfg_scale,
                    target_image=target_image,
                )
            if cfg.optim.loss.name == "pixel":
                (loss, renderred_gaussians), gradients = loss_grad(
                    means,
                    L,
                    colors,
                    rotmats,
                    background_color,
                    target_image,
                    ssim_weight=cfg.optim.loss.ssim_weight,
                )

            means, L, colors, rotmats, background_color, optimizers = update(
                means, L, colors, rotmats, background_color, optimizers, gradients
            )

            if jnp.isnan(loss):
                log.error("Loss became NaN")
                log.debug(prev_stats)
                log.debug(f"{loss}, {[(jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients]}")
                break
            if step % cfg.train.log_frequency == 0:
                log.info(
                    f"Loss: {loss:.5f}, step: {step}, at epoch {num_epoch} / {num_epochs}, num gaussians: {means.shape[0]}"
                )
                if cfg.optim.loss.name == "diffusion_guidance":
                    frames.append((renderred_gaussians, diffusion_image))
                elif cfg.optim.loss.name == "pixel":
                    frames.append((renderred_gaussians))
            prev_stats = [(jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients]

        means, L, colors, rotmats, background_color = split_n_prune(
            means,
            L,
            colors,
            rotmats,
            background_color,
            gradients,
            key,
            grad_thr=cfg.gaussians.grad_thr,
            color_demp_coeff=cfg.gaussians.color_demp_coeff,
        )
        optimizers = set_up_optimizers(
            means,
            L,
            colors,
            rotmats,
            background_color,
            lr=cfg.optim.lr,
            max_steps=cfg.optim.num_steps,
            means_mode=cfg.optim.means_mode,
            optimize_background=cfg.optim.optimize_background,
        )

    width = width * 2
    out = cv2.VideoWriter(
        str(out_dir / "outpy.avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        24,
        (width, height),
    )
    for frame in frames:
        if cfg.optim.loss.name == "diffusion_guidance":
            g = (np.clip(np.array(jnp.array(frame[0].block_until_ready())), 0, 1) * 255).astype(np.uint8)
            i = (np.clip(np.array(jnp.array(frame[1].block_until_ready())), 0, 1) * 255).astype(np.uint8)
            processed = np.hstack([g, i])
        elif cfg.optim.loss.name == "pixel":
            g = (np.clip(np.array(jnp.array(frame.block_until_ready())), 0, 1) * 255).astype(np.uint8)
            i = (np.clip(np.array(jnp.array(target_image.block_until_ready())), 0, 1) * 255).astype(np.uint8)
            processed = np.hstack([g, i])
        out.write(processed[:, :, ::-1])
    out.release()


if __name__ == "__main__":
    fit()
