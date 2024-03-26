# from https://github.com/leonidk/fmb-plus/blob/main/util_render.py
import jax.numpy as jnp


def jax_stable_exp(z, s=1, axis=0):
    z = s * z
    z = z - z.max(axis)
    z = jnp.exp(z)
    return z


# numerically stable softmax
def local_softmax(z, s=1, axis=0):
    z = jax_stable_exp(z, s, axis)
    return z / z.sum(keepdims=True, axis=axis)
