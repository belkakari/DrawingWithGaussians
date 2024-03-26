import jax.numpy as jnp


def jax_stable_exp(z, s=1, axis=0):
    z = s * z
    z = z - z.max(axis)
    z = jnp.exp(z)
    return z
