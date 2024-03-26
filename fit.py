import logging
from pathlib import Path

import cv2
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from drawingwithgaussians.gaussian import GaussianPC
from drawingwithgaussians.losses import pixel_loss


@hydra.main(version_base=None, config_path="./configs", config_name="fit_to_image.yaml")
def fit(cfg: DictConfig):
    log = logging.getLogger(__name__)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    out_dir = Path(hydra_cfg["runtime"]["output_dir"])

    key = jax.random.key(cfg.optim.seed)

    img = Image.open(cfg.image.path)

    height = cfg.image.height
    width = cfg.image.width
    lr = cfg.optim.lr
    num_epochs = cfg.optim.num_epochs
    max_steps = cfg.optim.num_steps

    target_image = jnp.array(img.resize((height, width)), dtype=jnp.float32)[:, :, :3] / 255

    gaussians = GaussianPC(num_gaussians=cfg.gaussians.num_gaussians, target_image=target_image, key=key)
    gaussians.set_up_optimizers(lr=cfg.optim.lr, max_steps=cfg.optim.num_steps)

    prev_stats = []
    frames = []
    for num_epoch in range(num_epochs):
        for step in range(max_steps):
            loss_grad = jax.value_and_grad(pixel_loss, argnums=[0, 1, 2, 3, 4], has_aux=True)
            (loss, renderred_gaussians), gradients = loss_grad(
                gaussians.means,
                gaussians.L,
                gaussians.colors,
                gaussians.rotmats,
                gaussians.background_color,
                gaussians.target_image,
            )

            gaussians.update(gradients)

            if jnp.isnan(loss):
                log.error(prev_stats)
                log.error(f"{loss}, {[(jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients]}")
                break
            if step % 50 == 0:
                log.info(
                    f"Loss: {loss:.5f}, step: {step}, at epoch {num_epoch} / {num_epochs}, num gaussians: {gaussians.num_gaussians}"
                )
                frames.append(np.array(renderred_gaussians))
            prev_stats = [(jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients]

        gaussians.split_n_prune(gradients, grad_thr=5e-5)

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
