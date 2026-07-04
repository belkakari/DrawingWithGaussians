# from https://github.com/leonidk/fmb-plus/blob/main/util_render.py
import mlx.core as mx


def mlx_stable_exp(z, s=1, axis=0):
    """Per-row numerically stable exp: subtracts the per-axis max before exponentiating.

    Matches the semantics of the original ``jax_stable_exp`` used in the JAX port.
    """
    z = s * z
    z = z - mx.max(z, axis=axis, keepdims=True)
    z = mx.exp(z)
    return z


# numerically stable softmax
def local_softmax(z, s=1, axis=0):
    z = mlx_stable_exp(z, s, axis)
    return z / mx.sum(z, axis=axis, keepdims=True)
