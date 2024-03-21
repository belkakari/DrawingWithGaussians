from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from einops import rearrange


@partial(jax.jit, static_argnames=["height", "width"])
def rasterize_single_gaussian(mean, covariance, color, rotmat, height=128, width=128):
    x, y = jnp.mgrid[0:height, 0:width]
    xy = jnp.column_stack([x.flatten(), y.flatten()])
    pdf = jax.scipy.stats.multivariate_normal.pdf(xy, mean, covariance @ rotmat)
    intensity = rearrange(pdf, "(h w) -> h w", h=height, w=width) / pdf.max()
    rasterized_color = (
        jnp.repeat(intensity[..., None], 3, axis=2) * color[None, :3] * color[None, 3]
    )
    rasterized_opacity = intensity[..., None] * color[None, 3]
    return rasterized_color, rasterized_opacity


@jax.jit
def alpha_compose(
    layers_rgb: jnp.array, layers_opacities: jnp.array, background: jnp.array
) -> List[jnp.array]:
    """https://github.com/leonidk/fuzzy-metaballs/blob/main/fm_render.py#L154

    Args:
        layers_rgb (jnp.array): _description_
        layers_opacities (jnp.array): _description_
        background (jnp.array): _description_

    Returns:
        List[jnp.array]: _description_
    """
    order_summed_density = jnp.cumsum(layers_opacities, axis=0)
    order_prior_density = order_summed_density - layers_opacities
    opacities = 1 - jnp.exp(-order_summed_density[-1])

    transmit = jnp.exp(-order_prior_density)
    layers_opacities = transmit * (1 - jnp.exp(-layers_opacities))

    wgt = layers_opacities.sum(0)
    div = jnp.where(wgt == 0, 1, wgt)
    partitioning = layers_opacities / div

    color = (
        background * (1 - opacities)
        + jnp.cumsum(layers_opacities * layers_rgb, axis=0)[-1]
    )
    return color, opacities, partitioning


def rasterize(
    means: jnp.array,
    covariances: jnp.array,
    colors: jnp.array,
    rotmats: jnp.array,
    background: jnp.array,
    height: int,
    width: int,
) -> jnp.array:
    assert means.shape[0] == covariances.shape[0] == colors.shape[0]

    vmaped_raster = jax.vmap(
        rasterize_single_gaussian, in_axes=[0, 0, 0, 0, None, None]
    )
    rasterized_colors, rasterized_opacities = vmaped_raster(
        means, covariances, colors, rotmats, height, width
    )
    return alpha_compose(
        rasterized_colors, rasterized_opacities, background, num_layers=means.shape[0]
    )


def pixel_loss(means, sigmas, colors, rotmats, background_color, target_image):
    covariances = jnp.stack([jnp.diag(sigma**2) for sigma in sigmas])
    height, width, channels = target_image.shape
    background = jnp.repeat(jnp.repeat(background_color, height, axis=0), width, axis=1)
    renderred_gaussians, opacities, partitioning = rasterize(
        means, covariances, colors, rotmats, background, height, width
    )
    loss = ((renderred_gaussians - target_image) ** 2).mean()
    return loss
