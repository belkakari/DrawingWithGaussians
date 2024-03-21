from functools import partial

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


@partial(jax.jit, static_argnames=["num_layers"])
def alpha_compose(layers_rgb, layers_opacities, background, num_layers):
    height, width, _ = layers_rgb[0].shape
    canvas = jnp.zeros((height, width, 3)) + background
    opacity_mask = jnp.zeros((height, width, 1))
    for layer_num in range(num_layers):
        layer_color, layer_opacity = layers_rgb[layer_num], layers_opacities[layer_num]
        opacity_mask = opacity_mask + layer_opacity
        canvas = canvas + jnp.where(
            opacity_mask < 1.0,
            layer_color,
            0,
        )
    return background + layers_rgb.sum(0), opacity_mask


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
    renderred_gaussians, opacity_mask = rasterize(
        means, covariances, colors, rotmats, background, height, width
    )
    loss = ((jax.nn.sigmoid(renderred_gaussians) - target_image) ** 2).mean()
    return loss
