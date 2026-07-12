"""Fused row-selective Adam for Apple Metal.

Rows not present in ``tile_visible_mask`` are copied bit-for-bit: parameters,
moments, and their per-Gaussian bias-correction counter do not advance.
"""

from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    uint N = (uint)sizes[0];
    uint C = (uint)sizes[1];
    uint total = N * C;
    if (tid >= total) return;
    uint row = tid / C;
    bool active = visible[row] != 0u;
    float p = params[tid];
    float old_m = moment1[tid];
    float old_v = moment2[tid];
    uint old_step = counters[row];
    if (!active) {
        params_out[tid] = p;
        moment1_out[tid] = old_m;
        moment2_out[tid] = old_v;
        if ((tid % C) == 0u) counters_out[row] = old_step;
        return;
    }
    float g = grads[tid];
    float b1 = hyper[1], b2 = hyper[2];
    float m = metal::fma(b1, old_m, (1.0f - b1) * g);
    float v = metal::fma(b2, old_v, (1.0f - b2) * g * g);
    uint step = old_step + 1u;
    float bc1 = 1.0f - metal::pow(b1, (float)step);
    float bc2 = 1.0f - metal::pow(b2, (float)step);
    float update = (m / bc1) / (metal::sqrt(v / bc2) + hyper[3]);
    params_out[tid] = p - hyper[0] * update;
    moment1_out[tid] = m;
    moment2_out[tid] = v;
    if ((tid % C) == 0u) counters_out[row] = step;
"""

_KERNEL = mx.fast.metal_kernel(
    name="selective_adam_rows",
    input_names=["params", "grads", "moment1", "moment2", "counters", "visible", "hyper", "sizes"],
    output_names=["params_out", "moment1_out", "moment2_out", "counters_out"],
    source=_SOURCE,
)


def selective_adam_update(
    params: mx.array,
    grads: mx.array,
    moment1: mx.array,
    moment2: mx.array,
    counters: mx.array,
    tile_visible_mask: mx.array,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if params.shape != grads.shape or params.shape != moment1.shape or params.shape != moment2.shape:
        raise ValueError("params, grads, and Adam moments must have identical shapes")
    if params.ndim < 1 or counters.shape != (params.shape[0],) or tile_visible_mask.shape != counters.shape:
        raise ValueError("counters and tile_visible_mask must have one entry per parameter row")
    n = int(params.shape[0])
    row_width = int(params.size // max(n, 1))
    visible = tile_visible_mask.astype(mx.uint32)
    hyper = mx.array([learning_rate, beta1, beta2, eps], dtype=mx.float32)
    sizes = mx.array([n, row_width], dtype=mx.uint32)
    block = 256
    total = n * row_width
    return tuple(
        _KERNEL(  # type: ignore[operator]
            inputs=[params, grads, moment1, moment2, counters, visible, hyper, sizes],
            grid=((total + block - 1) // block * block, 1, 1),
            threadgroup=(block, 1, 1),
            output_shapes=[params.shape, params.shape, params.shape, counters.shape],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.uint32],
        )
    )


class SelectiveAdam:
    """Small optimizer façade for a dictionary of Gaussian row tensors."""

    def __init__(self, learning_rates: dict[str, float], beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rates = {k: float(v) for k, v in learning_rates.items()}
        self.beta1, self.beta2, self.eps = float(beta1), float(beta2), float(eps)
        self.state: dict = {}

    def init(self, params: dict[str, mx.array]) -> None:
        if set(params) != set(self.learning_rates):
            raise ValueError("SelectiveAdam learning rates must cover every parameter")
        n = int(next(iter(params.values())).shape[0])
        if any(int(p.shape[0]) != n for p in params.values()):
            raise ValueError("SelectiveAdam only accepts row-aligned Gaussian tensors")
        self.state = {
            "counters": mx.zeros((n,), dtype=mx.uint32),
            "params": {name: {"m": mx.zeros_like(p), "v": mx.zeros_like(p)} for name, p in params.items()},
        }

    def apply_gradients(
        self, grads: dict[str, mx.array], params: dict[str, mx.array], tile_visible_mask: mx.array
    ) -> dict[str, mx.array]:
        counters = self.state["counters"]
        updated = {}
        next_counters = None
        for name, param in params.items():
            st = self.state["params"][name]
            p, m, v, c = selective_adam_update(
                param,
                grads[name],
                st["m"],
                st["v"],
                counters,
                tile_visible_mask,
                self.learning_rates[name],
                self.beta1,
                self.beta2,
                self.eps,
            )
            updated[name] = p
            st["m"], st["v"] = m, v
            next_counters = c
        self.state["counters"] = next_counters
        return updated

    def remap(self, idx_keep, num_new: int) -> None:
        idx = mx.array(idx_keep, dtype=mx.int32)
        old_counters = self.state["counters"]
        self.state["counters"] = mx.concatenate([mx.take(old_counters, idx), mx.zeros((num_new,), dtype=mx.uint32)])
        for st in self.state["params"].values():
            for key in ("m", "v"):
                old = st[key]
                st[key] = mx.concatenate(
                    [mx.take(old, idx, axis=0), mx.zeros((num_new,) + old.shape[1:], dtype=old.dtype)], axis=0
                )

    def carry_from(self, old: "SelectiveAdam", idx_keep, num_new: int) -> None:
        self.state = {
            "counters": old.state["counters"],
            "params": {name: {"m": st["m"], "v": st["v"]} for name, st in old.state["params"].items()},
        }
        self.remap(idx_keep, num_new)
