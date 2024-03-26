import logging
import os
from pathlib import Path

import cv2
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.scipy.spatial.transform import Rotation as R
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from drawingwithgaussians.losses import pixel_loss
from drawingwithgaussians.rendering2d import rasterize, split_gaussian


@hydra.main(version_base=None, config_path="./configs", config_name="fit_to_image.yaml")
def fit(cfg: DictConfig):
    log = logging.getLogger(__name__)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = Path(hydra_cfg["runtime"]["output_dir"])

    key = jax.random.key(cfg.optim.seed)

    img = Image.open(cfg.image.path)

    height = cfg.image.height
    width = cfg.image.width
    angle = 0.0
    num_gaussians = cfg.gaussians.num_gaussians
    lr = cfg.optim.lr
    num_epochs = cfg.optim.num_epochs
    max_steps = cfg.optim.num_steps
    learning_rate_schedule = optax.cosine_decay_schedule(lr, max_steps)

    target_image = (
        jnp.array(img.resize((height, width)), dtype=jnp.float32)[:, :, :3] / 255
    )
    background_color = target_image.mean(0).mean(0)[None, None, :] * 0.0
    means = jax.random.uniform(
        key, (num_gaussians, 2), minval=0, maxval=height, dtype=jnp.float32
    )
    sigmas = jax.random.uniform(
        key, (num_gaussians, 2), minval=1, maxval=height / 8, dtype=jnp.float32
    )
    covariances = jnp.stack([jnp.diag(sigma**2) for sigma in sigmas])
    L = jax.lax.linalg.cholesky(covariances)
    colors = jax.random.uniform(key, (num_gaussians, 4), jnp.float32, 0, 1)
    colors = colors.at[:, 3].set(colors[:, 3] / colors[:, 3].sum())
    r = R.from_euler(
        "x",
        [
            angle,
        ],
    )
    rotmats = jnp.repeat(r.as_matrix()[1:, 1:][None], num_gaussians, axis=0)

    optimize_means = optax.adam(learning_rate_schedule)
    optimize_cov = optax.adam(lr)
    optimize_colors = optax.adam(lr)
    optimize_rotmats = optax.adam(lr)
    optimize_background = optax.adam(lr)

    opt_state_means = optimize_means.init(means)
    opt_state_cov = optimize_cov.init(L)
    opt_state_colors = optimize_colors.init(colors)
    opt_state_rotmats = optimize_rotmats.init(rotmats)
    opt_state_background = optimize_background.init(background_color)

    prev_stats = []
    frames = []
    for num_epoch in range(num_epochs):
        for step in range(max_steps):
            (loss, renderred_gaussians), gradients = jax.value_and_grad(
                pixel_loss, argnums=[0, 1, 2, 3, 4], has_aux=True
            )(means, L, colors, rotmats, background_color, target_image)

            updates_means, opt_state_means = optimize_means.update(
                gradients[0], opt_state_means
            )
            means = optax.apply_updates(means, updates_means)

            updates_cov, opt_state_cov = optimize_cov.update(
                gradients[1], opt_state_cov
            )
            L = optax.apply_updates(L, updates_cov)

            updates_colors, opt_state_colors = optimize_colors.update(
                gradients[2], opt_state_colors
            )
            colors = optax.apply_updates(colors, updates_colors)

            updates_rotmats, opt_state_rotmats = optimize_rotmats.update(
                gradients[3], opt_state_rotmats
            )
            rotmats = optax.apply_updates(rotmats, updates_rotmats)

            updates_background, opt_state_background = optimize_background.update(
                gradients[4], opt_state_background
            )
            background_color = optax.apply_updates(background_color, updates_background)

            if jnp.isnan(loss):
                log.error(prev_stats)
                log.error(
                    f"{loss}, {[(jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients]}"
                )
                break
            if step % 50 == 0:
                log.info(
                    f"Loss: {loss:.5f}, step: {step}, at epoch {num_epoch} / {num_epochs}, num gaussians: {means.shape[0]}"
                )
                frames.append(np.array(renderred_gaussians))
            prev_stats = [
                (jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients
            ]

        smeans = []
        scovs = []
        scolors = []
        srotmats = []
        covariances = L.at[:, 0, 1].set(0) @ jnp.transpose(
            L.at[:, 0, 1].set(0), axes=[0, 2, 1]
        )
        for mean, cov, color, rotmat, grad_mean in zip(
            means, covariances, colors, rotmats, gradients[0]
        ):
            if jnp.linalg.norm(grad_mean) > 5e-5:
                mean, cov, color, rotmat = split_gaussian(
                    mean, cov, color, rotmat, grad_mean, key
                )
                smeans.append(mean.reshape(2, -1))
                scovs.append(cov.reshape(2, 2, 2))
                scolors.append(color.reshape(2, -1))
                srotmats.append(rotmat.reshape(2, 2, 2))
            elif jnp.linalg.norm(color) < 0.15:
                pass
            else:
                smeans.append(mean.reshape(1, 2))
                scovs.append(cov.reshape(1, 2, 2))
                scolors.append(color.reshape(1, -1))
                srotmats.append(rotmat.reshape(1, 2, 2))

        means, covariances, colors, rotmats = (
            jnp.concatenate(smeans),
            jnp.concatenate(scovs),
            jnp.concatenate(scolors) * 0.1,
            jnp.concatenate(srotmats),
        )
        background_color = background_color * 0.1
        L = jax.lax.linalg.cholesky(covariances)

        learning_rate_schedule = optax.cosine_onecycle_schedule(max_steps, lr)

        optimize_means = optax.adam(learning_rate_schedule)
        optimize_cov = optax.adam(lr)
        optimize_colors = optax.adam(lr)
        optimize_rotmats = optax.adam(lr)
        optimize_background = optax.adam(lr)

        opt_state_means = optimize_means.init(means)
        opt_state_cov = optimize_cov.init(L)
        opt_state_colors = optimize_colors.init(colors)
        opt_state_rotmats = optimize_rotmats.init(rotmats)
        opt_state_background = optimize_background.init(background_color)

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
