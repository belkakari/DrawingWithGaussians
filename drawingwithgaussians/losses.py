import jax.numpy as jnp

from .rendering2d import rasterize


def pixel_loss(means, L, colors, rotmats, background_color, target_image):
    covariances = L.at[:, 0, 1].set(0) @ jnp.transpose(
        L.at[:, 0, 1].set(0), axes=[0, 2, 1]
    )
    height, width, channels = target_image.shape
    background = jnp.repeat(jnp.repeat(background_color, height, axis=0), width, axis=1)
    renderred_gaussians, opacities, partitioning = rasterize(
        means, covariances, colors, rotmats, background, height, width
    )
    loss = jnp.abs(renderred_gaussians - target_image).mean()
    return loss, renderred_gaussians
