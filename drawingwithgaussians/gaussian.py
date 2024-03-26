from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax.scipy.spatial.transform import Rotation as R


def init_gaussians(num_gaussians, target_image, key):
    num_gaussians = num_gaussians
    key = key
    background_color = jax.random.uniform(key, (1, 1, 3))
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


def set_up_optimizers(means, L, colors, rotmats, background_color, lr, max_steps):
    learning_rate_schedule = optax.cosine_decay_schedule(lr, max_steps)
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

    return (
        (optimize_means, opt_state_means),
        (optimize_cov, opt_state_cov),
        (optimize_colors, opt_state_colors),
        (optimize_rotmats, opt_state_rotmats),
        (optimize_background, opt_state_background),
    )


def split_n_prune(means, L, colors, rotmats, background_color, gradients, key, grad_thr=5e-5, color_demp_coeff=0.1):
    covariances = L @ jnp.transpose(L, axes=[0, 2, 1])

    # TODO: refactor into something vmapable
    smeans = []
    scovs = []
    scolors = []
    srotmats = []
    for mean, cov, col, rotmap, mean_grad in zip(means, covariances, colors, rotmats, gradients[0]):
        mean, cov, col, rotmap = snp_single(mean, cov, col, rotmap, mean_grad, key, grad_thr)
        if mean is not None:
            smeans.append(mean)
            scovs.append(cov)
            scolors.append(col)
            srotmats.append(rotmap)
    means, covariances, colors, rotmats = (
        jnp.concatenate(smeans),
        jnp.concatenate(scovs),
        jnp.concatenate(scolors) * color_demp_coeff,
        jnp.concatenate(srotmats),
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
    return splitted_means, splitted_covariances, splitted_colors, splitted_rotmat


def snp_single(mean, cov, color, rotmat, grad_mean, key, grad_thr=2e-5):
    if jnp.linalg.norm(grad_mean) > grad_thr:
        mean, cov, color, rotmat = split_gaussian(mean, cov, color, rotmat, key)
        return mean.reshape(2, -1), cov.reshape(2, 2, 2), color.reshape(2, -1), rotmat.reshape(2, 2, 2)
    elif jnp.linalg.norm(color) < 0.15:
        return (None, None, None, None)
    else:
        return mean.reshape(1, 2), cov.reshape(1, 2, 2), color.reshape(1, -1), rotmat.reshape(1, 2, 2)
