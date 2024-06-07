from functools import partial

import jax
import jax.numpy as jnp
import jaxsplat

from drawingwithgaussians.twod.sds_pipeline import img2img


@jax.jit
def pixel_loss(
    target,
    means3d,
    scales,
    quats,
    colors,
    opacities,
    viewmat,
    background,
    focal,
    center,
    shape,
    glob_scale,
    clip_thresh,
    block_size,
):
    renderred_gaussians = jaxsplat.render(
        means3d,  # jax.Array (N, 3)
        scales,  # jax.Array (N, 3)
        quats,  # jax.Array (N, 4) normalized
        colors,  # jax.Array (N, 3)
        opacities,  # jax.Array (N, 1)
        viewmat=viewmat,  # jax.Array (4, 4)
        background=background,  # jax.Array (3,)
        img_shape=shape,  # tuple[int, int] = (H, W)
        f=focal,  # tuple[float, float] = (fx, fy)
        c=center,  # tuple[int, int] = (cx, cy)
        glob_scale=glob_scale,  # float
        clip_thresh=clip_thresh,  # float
        block_size=block_size,  # int <= 16
    )
    loss = jnp.mean(jnp.square(renderred_gaussians - target))
    return loss, renderred_gaussians


def diffusion_guidance(
    gaussians_params,
    viewmat,
    background,
    focal,
    center,
    shape,
    glob_scale,
    clip_thresh,
    block_size,
    diffusion_shape,
    num_steps,
    strength,
    pipeline,
    params,
    dtype,
    cfg_scale,
    prompt,
    key,
    target_image=None,
):
    means3d, scales, quats, colors, opacities = gaussians_params

    renderred_gaussians = jaxsplat.render(
        means3d,  # jax.Array (N, 3)
        scales,  # jax.Array (N, 3)
        quats,  # jax.Array (N, 4) normalized
        colors,  # jax.Array (N, 3)
        opacities,  # jax.Array (N, 1)
        viewmat=viewmat,  # jax.Array (4, 4)
        background=background,  # jax.Array (3,)
        img_shape=shape,  # tuple[int, int] = (H, W)
        f=focal,  # tuple[float, float] = (fx, fy)
        c=center,  # tuple[int, int] = (cx, cy)
        glob_scale=glob_scale,  # float
        clip_thresh=clip_thresh,  # float
        block_size=block_size,  # int <= 16
    )

    height_d, width_d, c = diffusion_shape
    if target_image is None:
        image = jax.lax.stop_gradient(
            img2img(
                jax.lax.stop_gradient(renderred_gaussians.astype(dtype)),
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
        shape = (shape[0], shape[1], 3)
        image = jax.image.resize(image[0, 0], shape=shape, method="bilinear")
    else:
        image = jnp.copy(target_image)
    loss = jnp.mean(jnp.square(renderred_gaussians - jax.lax.stop_gradient(image)))
    return loss, (renderred_gaussians, image)
