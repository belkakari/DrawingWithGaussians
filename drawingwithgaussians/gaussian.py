import jax
import jax.numpy as jnp
import optax
from jax.scipy.spatial.transform import Rotation as R


class GaussianPC:
    def __init__(self, num_gaussians, target_image, key):
        self.num_gaussians = num_gaussians
        self.key = key
        self.background_color = jax.random.uniform(key, (1, 1, 3))
        self.height, self.weight, _ = target_image.shape
        self.target_image = target_image
        self.means = jax.random.uniform(key, (num_gaussians, 2), minval=0, maxval=self.height, dtype=jnp.float32)
        sigmas = jax.random.uniform(key, (num_gaussians, 2), minval=1, maxval=self.height / 8, dtype=jnp.float32)
        covariances = jnp.stack([jnp.diag(sigma**2) for sigma in sigmas])
        self.L = jax.lax.linalg.cholesky(covariances)
        self.colors = jax.random.uniform(key, (num_gaussians, 3), jnp.float32, 0, 1)
        self.opacities = jax.random.uniform(key, (num_gaussians, 1), jnp.float32, 0, 1)
        self.opacities = self.opacities / self.opacities.max()
        r = R.from_euler(
            "x",
            [
                0.0,
            ],
        )
        self.rotmats = jnp.repeat(r.as_matrix()[1:, 1:][None], num_gaussians, axis=0)

    def set_up_optimizers(self, lr, max_steps):
        learning_rate_schedule = optax.cosine_decay_schedule(lr, max_steps)
        self.optimize_means = optax.adam(learning_rate_schedule)
        self.optimize_cov = optax.adam(lr)
        self.optimize_colors = optax.adam(lr)
        self.optimize_rotmats = optax.adam(lr)
        self.optimize_background = optax.adam(lr)

        self.opt_state_means = self.optimize_means.init(self.means)
        self.opt_state_cov = self.optimize_cov.init(self.L)
        self.opt_state_colors = self.optimize_colors.init(self.colors)
        self.opt_state_rotmats = self.optimize_rotmats.init(self.rotmats)
        self.opt_state_background = self.optimize_background.init(self.background_color)

    def split_n_prune(self, gradients, grad_thr=5e-5):
        covariances = self.L @ jnp.transpose(self.L, axes=[0, 2, 1])

        def snp_single(mean, cov, color, rotmat, grad_mean):
            if jnp.linalg.norm(grad_mean) > grad_thr:
                mean, cov, color, rotmat = self.split_gaussian(mean, cov, color, rotmat, grad_mean, self.key)
                return mean.reshape(2, -1), cov.reshape(2, 2, 2), color.reshape(2, -1), rotmat.reshape(2, 2, 2)
            elif jnp.linalg.norm(color) < 0.15:
                pass
            else:
                return mean.reshape(1, 2), cov.reshape(1, 2, 2), color.reshape(1, -1), rotmat.reshape(1, 2, 2)

        snp_vmaped = jax.vmap(snp_single, in_axes=[0, 0, 0, 0, 0])
        means, covariances, colors, rotmats = snp_vmaped(
            self.means, covariances, self.colors, self.rotmats, gradients[0]
        )
        print(means.shape, covariances.shape, colors.shape, rotmats.shape)
        background_color = background_color * 0.1
        L = jax.lax.linalg.cholesky(covariances)
        self.num_gaussians = means.shape[0]

    def update(self, gradients):
        updates_means, self.opt_state_means = self.optimize_means.update(gradients[0], self.opt_state_means)
        self.means = optax.apply_updates(self.means, updates_means)

        updates_cov, self.opt_state_cov = self.optimize_cov.update(gradients[1], self.opt_state_cov)
        self.L = optax.apply_updates(self.L, updates_cov)
        self.L = self.L.at[:, 0, 1].set(0)  # keeps upper triangular to 0

        updates_colors, self.opt_state_colors = self.optimize_colors.update(gradients[2], self.opt_state_colors)
        self.colors = optax.apply_updates(self.colors, updates_colors)

        updates_rotmats, self.opt_state_rotmats = self.optimize_rotmats.update(gradients[3], self.opt_state_rotmats)
        self.rotmats = optax.apply_updates(self.rotmats, updates_rotmats)

        updates_background, self.opt_state_background = self.optimize_background.update(
            gradients[4], self.opt_state_background
        )
        self.background_color = optax.apply_updates(self.background_color, updates_background)

    @staticmethod
    @jax.jit
    def split_gaussian(mean, covariance, color, rotmat, key, cov_scale=1.6):
        splitted_means = jax.random.multivariate_normal(key, mean, covariance, shape=(2,))
        splitted_covariances = jnp.concatenate([covariance, covariance]) / cov_scale
        splitted_colors = jnp.concatenate([color, color])
        splitted_rotmat = jnp.concatenate([rotmat, rotmat])
        return splitted_means, splitted_covariances, splitted_colors, splitted_rotmat
