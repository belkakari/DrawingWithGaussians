import logging
from pathlib import Path

import cv2
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from drawingwithgaussians.gaussian import init_gaussians, set_up_optimizers, split_n_prune, update
from drawingwithgaussians.losses import diffusion_guidance, pixel_loss
from drawingwithgaussians.sds_pipeline import FlaxStableDiffusionImg2ImgPipeline


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
    for num_epoch in range(num_epochs):
        for step in range(max_steps):
            if cfg.optim.loss.name == "diffusion_guidance":
                loss_grad = jax.value_and_grad(diffusion_guidance, argnums=[0, 1, 2, 3, 4], has_aux=True)
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
                    strength=cfg.optim.loss.strength,
                    pipeline=pipeline,
                    params=params,
                )
            if cfg.optim.loss.name == "pixel":
                loss_grad = jax.value_and_grad(pixel_loss, argnums=[0, 1, 2, 3, 4], has_aux=True)
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
                    g = (np.clip(np.array(jnp.array(renderred_gaussians.block_until_ready())), 0, 1) * 255).astype(
                        np.uint8
                    )
                    i = (np.clip(np.array(jnp.array(diffusion_image.block_until_ready())), 0, 1) * 255).astype(np.uint8)
                    Image.fromarray(np.hstack([g, i])).save(str(out_dir / f"frames_{step}.jpg"))
                frames.append(np.array(renderred_gaussians))
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
        )

    out = cv2.VideoWriter(
        str(out_dir / "outpy.avi"),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        24,
        (width, height),
    )
    for frame in frames:
        out.write((np.clip(frame[:, :, ::-1], 0, 1) * 255).astype(np.uint8))
    out.release()


if __name__ == "__main__":
    fit()
