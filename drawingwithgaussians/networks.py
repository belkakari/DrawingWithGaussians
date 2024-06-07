import functools
from typing import Any, Callable, Ellipsis

import jax
import jax.numpy as jnp
from flax import linen as nn


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """from https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/model_utils.py
    Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
        x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].
        min_deg: int, the minimum (inclusive) degree of the encoding.
        max_deg: int, the maximum (exclusive) degree of the encoding.
        legacy_posenc_order: bool, keep the same ordering as the original tf code.

    Returns:
        encoded: jnp.ndarray, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = jnp.array([2**i for i in range(min_deg, max_deg)])
    if legacy_posenc_order:
        xb = x[Ellipsis, None, :] * scales[:, None]
        four_feat = jnp.reshape(jnp.sin(jnp.stack([xb, xb + 0.5 * jnp.pi], -2)), list(x.shape[:-1]) + [-1])
    else:
        xb = jnp.reshape((x[Ellipsis, None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
        four_feat = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)


class MLP(nn.Module):
    """from https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/model_utils.py
    A simple MLP.
    """

    net_depth: int = 8  # The depth of the first part of MLP.
    net_width: int = 256  # The width of the first part of MLP.
    net_depth_condition: int = 1  # The depth of the second part of MLP.
    net_width_condition: int = 128  # The width of the second part of MLP.
    net_activation: Callable[Ellipsis, Any] = nn.relu  # The activation function.
    skip_layer: int = 4  # The layer to add skip layers to.
    num_rgb_channels: int = 3  # The number of RGB channels.
    num_sigma_channels: int = 1  # The number of sigma channels.

    @nn.compact
    def __call__(self, x, condition=None):
        """Evaluate the MLP.

        Args:
          x: jnp.ndarray(float32), [batch, num_samples, feature], points.
          condition: jnp.ndarray(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
          raw_rgb: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_rgb_channels].
          raw_sigma: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_sigma_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.reshape([-1, feature_dim])
        dense_layer = functools.partial(nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)
        raw_sigma = dense_layer(self.num_sigma_channels)(x).reshape([-1, num_samples, self.num_sigma_channels])

        if condition is not None:
            # Output of the first part of MLP.
            bottleneck = dense_layer(self.net_width)(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.reshape([-1, condition.shape[-1]])
            x = jnp.concatenate([bottleneck, condition], axis=-1)
            # Here use 1 extra layer to align with the original nerf model.
            for i in range(self.net_depth_condition):
                x = dense_layer(self.net_width_condition)(x)
                x = self.net_activation(x)
        raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape([-1, num_samples, self.num_rgb_channels])
        return raw_rgb, raw_sigma
