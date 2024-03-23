import cv2
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.scipy.spatial.transform import Rotation as R
from PIL import Image

from drawingwithgaussians.rendering2d import pixel_loss, rasterize, split_gaussian

seed = 1
key = jax.random.key(seed)

path = "/Users/gleb/Downloads/avataro4ka.jpg"
img = Image.open(path)

height = 128
width = 128
angle = 0.0
num_gaussians = 2
lr = 1e-1
num_epochs = 10
max_steps = 2000
learning_rate_schedule = optax.cosine_decay_schedule(
    lr / 1000, max_steps
)  # cosine_onecycle_schedule(max_steps, lr)

target_image = jnp.array(img.resize((height, width)), dtype=jnp.float32)[:, :, :3] / 255
background_color = target_image.mean(0).mean(0)[None, None, :] * 0.0
means = jax.random.uniform(
    key, (num_gaussians, 2), minval=0, maxval=height, dtype=jnp.float32
)
sigmas = jax.random.uniform(
    key, (num_gaussians, 2), minval=5, maxval=20, dtype=jnp.float32
)
covariances = jnp.stack([jnp.diag(sigma**2) for sigma in sigmas])
L = jax.lax.linalg.cholesky(covariances)
colors = jax.random.uniform(key, (num_gaussians, 4), jnp.float32, 0, 1)
colors = colors.at[:, 3].set(colors[:, 3] / colors[:, 3].sum())
r = R.from_euler(
    "x",
    [
        angle,
    ],
)
rotmats = jnp.repeat(r.as_matrix()[1:, 1:][None], num_gaussians, axis=0)

optimize_means = optax.adam(learning_rate_schedule)
optimize_cov = optax.adam(learning_rate_schedule)
optimize_colors = optax.adam(learning_rate_schedule)
optimize_rotmats = optax.adam(learning_rate_schedule)
optimize_background = optax.adam(learning_rate_schedule)

opt_state_means = optimize_means.init(means)
opt_state_cov = optimize_cov.init(L)
opt_state_colors = optimize_colors.init(colors)
opt_state_rotmats = optimize_rotmats.init(rotmats)
opt_state_background = optimize_background.init(background_color)

prev_stats = []
frames = []
out = cv2.VideoWriter(
    "outpy.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 24, (width, height)
)
for num_epoch in range(num_epochs):
    for step in range(max_steps):
        (loss, renderred_gaussians), gradients = jax.value_and_grad(
            pixel_loss, argnums=[0, 1, 2, 3, 4], has_aux=True
        )(means, L, colors, rotmats, background_color, target_image)

        updates_means, opt_state_means = optimize_means.update(
            gradients[0], opt_state_means
        )
        means = optax.apply_updates(means, updates_means)

        updates_cov, opt_state_cov = optimize_cov.update(gradients[1], opt_state_cov)
        L = optax.apply_updates(L, updates_cov)

        updates_colors, opt_state_colors = optimize_colors.update(
            gradients[2], opt_state_colors
        )
        colors = optax.apply_updates(colors, updates_colors)

        updates_rotmats, opt_state_rotmats = optimize_rotmats.update(
            gradients[3], opt_state_rotmats
        )
        rotmats = optax.apply_updates(rotmats, updates_rotmats)

        updates_background, opt_state_background = optimize_background.update(
            gradients[4], opt_state_background
        )
        background_color = optax.apply_updates(background_color, updates_background)

        if jnp.isnan(loss):
            print(prev_stats)
            print(
                loss,
                [(jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients],
            )
            break
        if step % 50 == 0:
            print(loss, step, num_epoch, means.shape[0])
            out.write(
                np.array((renderred_gaussians[:, :, ::-1] * 255).astype(np.uint8))
            )
        prev_stats = [
            (jnp.linalg.norm(gradient), gradient.max()) for gradient in gradients
        ]

    smeans = []
    scovs = []
    scolors = []
    srotmats = []
    if num_epoch < num_epochs // 2 + 1:
        covariances = L.at[:, 0, 1].set(0) @ jnp.transpose(
            L.at[:, 0, 1].set(0), axes=[0, 2, 1]
        )
        for mean, cov, color, rotmat in zip(means, covariances, colors, rotmats):
            mean, cov, color, rotmat = split_gaussian(mean, cov, color, rotmat)
            smeans.append(mean.reshape(2, -1))
            scovs.append(cov.reshape(2, 2, 2))
            scolors.append(color.reshape(2, -1))
            srotmats.append(rotmat.reshape(2, 2, 2))

        means, covariances, colors, rotmats = (
            jnp.concatenate(smeans),
            jnp.concatenate(scovs),
            jnp.concatenate(scolors),
            jnp.concatenate(srotmats),
        )
        L = jax.lax.linalg.cholesky(covariances)

    learning_rate_schedule = optax.cosine_onecycle_schedule(max_steps, lr)

    optimize_means = optax.adam(learning_rate_schedule)
    optimize_cov = optax.adam(learning_rate_schedule)
    optimize_colors = optax.adam(learning_rate_schedule)
    optimize_rotmats = optax.adam(learning_rate_schedule)
    optimize_background = optax.adam(learning_rate_schedule)

    opt_state_means = optimize_means.init(means)
    opt_state_cov = optimize_cov.init(L)
    opt_state_colors = optimize_colors.init(colors)
    opt_state_rotmats = optimize_rotmats.init(rotmats)
    opt_state_background = optimize_background.init(background_color)
out.release()
