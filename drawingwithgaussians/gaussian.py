from functools import partial

import jax
import jax.numpy as jnp
import optax
from einops import rearrange
from jax.scipy.spatial.transform import Rotation as R


def init_gaussians(num_gaussians, target_image, key, optimize_background=True):
    num_gaussians = num_gaussians
    key = key
    background_color = jax.random.uniform(key, (1, 1, 3))
    if not optimize_background:
        background_color *= 0.0
    height, weight, _ = target_image.shape
    target_image = target_image
    means = jax.random.uniform(key, (num_gaussians, 2), minval=0, maxval=height, dtype=jnp.float32)
    sigmas = jax.random.uniform(key, (num_gaussians, 2), minval=1, maxval=height / 8, dtype=jnp.float32)
    covariances = jnp.stack([jnp.diag(sigma**2) for sigma in sigmas])
    L = jax.lax.linalg.cholesky(covariances)
    colors = jax.random.uniform(key, (num_gaussians, 3), jnp.float32, 0, 1)
    opacities = jax.random.uniform(key, (num_gaussians, 1), jnp.float32, 0, 1)
    opacities = opacities / opacities.max()
    r = R.from_euler(
        "x",
        [
            0.0,
        ],
    )
    rotmats = jnp.repeat(r.as_matrix()[1:, 1:][None], num_gaussians, axis=0)
    return means, L, colors, rotmats, background_color


def set_up_optimizers(
    means, L, colors, rotmats, background_color, lr, max_steps, means_mode="cos", optimize_background=True
):
    if means_mode == "cos":
        learning_rate_schedule = optax.cosine_decay_schedule(lr, max_steps)
    elif means_mode == "const":
        learning_rate_schedule = optax.constant_schedule(lr)
    optimize_means = optax.adam(learning_rate_schedule)
    optimize_cov = optax.adam(lr)
    optimize_colors = optax.adam(lr)
    optimize_rotmats = optax.adam(lr)
    if optimize_background:
        optimize_background = optax.adam(lr)
    else:
        optimize_background = None

    opt_state_means = optimize_means.init(means)
    opt_state_cov = optimize_cov.init(L)
    opt_state_colors = optimize_colors.init(colors)
    opt_state_rotmats = optimize_rotmats.init(rotmats)
    if optimize_background:
        opt_state_background = optimize_background.init(background_color)
    else:
        opt_state_background = None

    return (
        (optimize_means, opt_state_means),
        (optimize_cov, opt_state_cov),
        (optimize_colors, opt_state_colors),
        (optimize_rotmats, opt_state_rotmats),
        (optimize_background, opt_state_background),
    )


def split_n_prune(
    means,
    L,
    colors,
    rotmats,
    background_color,
    gradients,
    key,
    grad_thr=5e-5,
    color_demp_coeff=0.1,
):
    covariances = L @ jnp.transpose(L, axes=[0, 2, 1])

    mask_to_split = jnp.where(jnp.linalg.norm(gradients[0], axis=1) > grad_thr, True, False)
    mask_to_erase = jnp.where(jnp.linalg.norm(colors) < 0.05, True, False)
    mask = jnp.where(~mask_to_split & ~mask_to_erase)

    split_vmap = jax.vmap(split_gaussian, in_axes=[0, 0, 0, 0, None, None])
    splitted_means, splitted_covariances, splited_colors, splitted_rotmats = split_vmap(
        means[mask_to_split],
        covariances[mask_to_split],
        colors[mask_to_split],
        rotmats[mask_to_split],
        key,
        1.6,
    )
    means, covariances, colors, rotmats = (
        jnp.concatenate([means[mask], rearrange(splitted_means, "n s d -> (n s) d")]),
        jnp.concatenate([covariances[mask], rearrange(splitted_covariances, "n s h w -> (n s) h w")]),
        jnp.concatenate([colors[mask], rearrange(splited_colors, "n s d -> (n s) d")]) * color_demp_coeff,
        jnp.concatenate([rotmats[mask], rearrange(splitted_rotmats, "n s h w -> (n s) h w")]),
    )
    background_color = background_color * color_demp_coeff
    L = jax.lax.linalg.cholesky(covariances)
    return means, L, colors, rotmats, background_color


def update(means, L, colors, rotmats, background_color, optimizers, gradients):
    (
        (optimize_means, opt_state_means),
        (optimize_cov, opt_state_cov),
        (optimize_colors, opt_state_colors),
        (optimize_rotmats, opt_state_rotmats),
        (optimize_background, opt_state_background),
    ) = optimizers
    updates_means, opt_state_means = optimize_means.update(gradients[0], opt_state_means)
    means = optax.apply_updates(means, updates_means)

    updates_cov, opt_state_cov = optimize_cov.update(gradients[1], opt_state_cov)
    L = optax.apply_updates(L, updates_cov)
    L = L.at[:, 0, 1].set(0)  # keeps upper triangular to 0

    updates_colors, opt_state_colors = optimize_colors.update(gradients[2], opt_state_colors)
    colors = optax.apply_updates(colors, updates_colors)

    updates_rotmats, opt_state_rotmats = optimize_rotmats.update(gradients[3], opt_state_rotmats)
    rotmats = optax.apply_updates(rotmats, updates_rotmats)

    if optimize_background is not None:
        updates_background, opt_state_background = optimize_background.update(gradients[4], opt_state_background)
        background_color = optax.apply_updates(background_color, updates_background)

    return (
        means,
        L,
        colors,
        rotmats,
        background_color,
        (
            (optimize_means, opt_state_means),
            (optimize_cov, opt_state_cov),
            (optimize_colors, opt_state_colors),
            (optimize_rotmats, opt_state_rotmats),
            (optimize_background, opt_state_background),
        ),
    )


@jax.jit
def split_gaussian(mean, covariance, color, rotmat, key, cov_scale=1.6):
    splitted_means = jax.random.multivariate_normal(key, mean, covariance, shape=(2,))
    splitted_covariances = jnp.concatenate([covariance, covariance]) / cov_scale
    splitted_colors = jnp.concatenate([color, color])
    splitted_rotmat = jnp.concatenate([rotmat, rotmat])
    return (
        splitted_means.reshape(2, -1),
        splitted_covariances.reshape(2, 2, 2),
        splitted_colors.reshape(2, -1),
        splitted_rotmat.reshape(2, 2, 2),
    )
