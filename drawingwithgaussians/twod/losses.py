from functools import partial

import jax
import jax.numpy as jnp

from .rendering import rasterize
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


def diffusion_guidance(
    means,
    L,
    colors,
    rotmats,
    background_color,
    prompt,
    key,
    shape,
    diffusion_shape,
    num_steps,
    strength,
    pipeline,
    params,
    dtype,
    cfg_scale,
    target_image=None,
):
    @partial(jax.jit, static_argnames=["shape", "diffusion_shape"])
    def preprocess(means, L, colors, rotmats, background_color, shape, diffusion_shape):
        height, width, c = shape
        covariances = L.at[:, 0, 1].set(0) @ jnp.transpose(L.at[:, 0, 1].set(0), axes=[0, 2, 1])
        background = jnp.repeat(jnp.repeat(background_color, height, axis=0), width, axis=1)
        renderred_gaussians, opacities, partitioning = rasterize(
            means, covariances, colors, rotmats, background, height, width
        )
        renderred_gaussians_ = jax.image.resize(renderred_gaussians, shape=diffusion_shape, method="bilinear")
        return renderred_gaussians, renderred_gaussians_

    height_d, width_d, c = diffusion_shape
    renderred_gaussians, renderred_gaussians_ = preprocess(
        means, L, colors, rotmats, background_color, shape, diffusion_shape
    )
    if target_image is None:
        image = jax.lax.stop_gradient(
            img2img(
                jax.lax.stop_gradient(renderred_gaussians_.astype(dtype)),
                prompt,
                key,
                height_d,
                width_d,
                num_steps,
                strength,
                cfg_scale,
                pipeline,
                params,
            )
        )
        image = jax.image.resize(image[0, 0], shape=shape, method="bilinear")
    else:
        image = jnp.copy(target_image)
    loss = jnp.abs(renderred_gaussians - jax.lax.stop_gradient(image)).mean()

    return loss, (renderred_gaussians, image)
