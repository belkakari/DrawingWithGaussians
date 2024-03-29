import jax
import jax.numpy as jnp
import numpy as np
from dm_pix import ssim

from .rendering2d import rasterize
from .sds_pipeline import img2img


@jax.jit
def pixel_loss(means, L, colors, rotmats, background_color, target_image, ssim_weight=0.2):
    covariances = L.at[:, 0, 1].set(0) @ jnp.transpose(L.at[:, 0, 1].set(0), axes=[0, 2, 1])
    height, width, channels = target_image.shape
    background = jnp.repeat(jnp.repeat(background_color, height, axis=0), width, axis=1)
    renderred_gaussians, opacities, partitioning = rasterize(
        means, covariances, colors, rotmats, background, height, width
    )
    loss = (1 - ssim_weight) * jnp.abs(renderred_gaussians - target_image).mean()  # + ssim_weight * ssim(
    #     renderred_gaussians, target_image
    # )
    return loss, renderred_gaussians


def diffusion_guidance(means, L, colors, rotmats, background_color, prompt, key, shape, num_steps, strength):
    height, width = shape
    covariances = L.at[:, 0, 1].set(0) @ jnp.transpose(L.at[:, 0, 1].set(0), axes=[0, 2, 1])
    background = jnp.repeat(jnp.repeat(background_color, height, axis=0), width, axis=1)
    renderred_gaussians, opacities, partitioning = rasterize(
        means, covariances, colors, rotmats, background, height, width
    )
    print("image shape is ", renderred_gaussians.shape, renderred_gaussians.size)
    image = jax.lax.stop_gradient(img2img(renderred_gaussians, prompt, key, height, width, num_steps, strength))
    loss = jnp.abs(renderred_gaussians - image).mean()
    return loss, np.hstack([np.array(renderred_gaussians), np.array(image)])
