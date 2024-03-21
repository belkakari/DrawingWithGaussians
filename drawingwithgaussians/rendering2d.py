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
    rasterized_color = jnp.repeat(intensity[..., None], 3, axis=2) * color[None, :3]
    layer_opacity = jax.nn.sigmoid(color[None, 3])
    return rasterized_color, layer_opacity


@partial(jax.jit, static_argnames=["num_layers"])
def alpha_compose(canvas, layers_rgb, layers_opacities, num_layers):
    opacity_mask = jnp.zeros((canvas.shape[0], canvas.shape[1], 1))
    for layer_num in range(num_layers):
        layer_color, layer_opacity = layers_rgb[layer_num], layers_opacities[layer_num]
        opacity_mask = opacity_mask + layer_color.mean(2)[..., None] * jax.nn.sigmoid(
            layer_opacity
        )
        mask = opacity_mask < 1.0
        canvas = canvas + layer_color * jax.nn.sigmoid(layer_opacity) * mask
    return canvas


def rasterize(
    means: jnp.array,
    covariances: jnp.array,
    colors: jnp.array,
    rotmats: jnp.array,
    height: int,
    width: int,
) -> jnp.array:
    assert means.shape[0] == covariances.shape[0] == colors.shape[0]
    canvas = jnp.zeros((height, width, 3))
    vmaped_raster = jax.vmap(
        rasterize_single_gaussian, in_axes=[0, 0, 0, 0, None, None]
    )
    rasterized_colors, layers_opacities = vmaped_raster(
        means, covariances, colors, rotmats, height, width
    )
    return alpha_compose(
        canvas, rasterized_colors, layers_opacities, num_layers=means.shape[0]
    )


def pixel_loss(means, covariances, colors, rotmats, target_image):
    height, width, channels = target_image.shape
    renderred_gaussians = rasterize(means, covariances, colors, rotmats, height, width)
    loss = ((renderred_gaussians - target_image) ** 2).mean() + (
        colors[None, 3] ** 2
    ).mean()
    return loss
