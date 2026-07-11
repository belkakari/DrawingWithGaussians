"""Open-addressed signed-int3 hash set implemented as an MLX Metal kernel."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    uint N = (uint)shape[0];
    uint capacity = (uint)shape[1];
    if (tid >= N) return;
    uint x = as_type<uint>(coords[3 * tid]);
    uint y = as_type<uint>(coords[3 * tid + 1]);
    uint z = as_type<uint>(coords[3 * tid + 2]);
    uint hash = 2166136261u;
    hash = (hash ^ x) * 16777619u;
    hash = (hash ^ y) * 16777619u;
    hash = (hash ^ z) * 16777619u;
    hash ^= hash >> 16;
    hash *= 0x7feb352du;
    hash ^= hash >> 15;
    hash *= 0x846ca68bu;
    hash ^= hash >> 16;
    uint mask = capacity - 1u;
    for (uint probe = 0; probe < capacity; ++probe) {
        uint slot = (hash + probe) & mask;
        uint expected = 0u;
        bool claimed = false;
        // A weak CAS may fail spuriously while leaving expected == 0. Retry
        // that case; treating it as an occupied slot can insert a duplicate
        // key later in the probe chain.
        do {
            expected = 0u;
            claimed = atomic_compare_exchange_weak_explicit(
                &states[slot], &expected, 1u,
                metal::memory_order_relaxed, metal::memory_order_relaxed);
        } while (!claimed && expected == 0u);
        if (claimed) {
            atomic_store_explicit(&table_coords[3 * slot], x, metal::memory_order_relaxed);
            atomic_store_explicit(&table_coords[3 * slot + 1], y, metal::memory_order_relaxed);
            atomic_store_explicit(&table_coords[3 * slot + 2], z, metal::memory_order_relaxed);
            atomic_store_explicit(&states[slot], 2u, metal::memory_order_relaxed);
            return;
        }
        uint state = atomic_load_explicit(&states[slot], metal::memory_order_relaxed);
        while (state == 1u) {
            state = atomic_load_explicit(&states[slot], metal::memory_order_relaxed);
        }
        if (state == 2u) {
            uint tx = atomic_load_explicit(&table_coords[3 * slot], metal::memory_order_relaxed);
            uint ty = atomic_load_explicit(&table_coords[3 * slot + 1], metal::memory_order_relaxed);
            uint tz = atomic_load_explicit(&table_coords[3 * slot + 2], metal::memory_order_relaxed);
            if (tx == x && ty == y && tz == z) return;
        }
    }
    atomic_fetch_add_explicit(&overflow[0], 1u, metal::memory_order_relaxed);
"""

_KERNEL = mx.fast.metal_kernel(
    name="signed_int3_hash_set",
    input_names=["coords", "shape"],
    output_names=["states", "table_coords", "overflow"],
    source=_SOURCE,
    atomic_outputs=True,
)


class HashTableOverflow(RuntimeError):
    pass


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def unique_int3_metal(coords, capacity: int | None = None, max_load: float = 0.5) -> np.ndarray:
    """Return unique signed int3 rows using a Metal open-addressed hash set."""
    coords = mx.array(coords, dtype=mx.int32)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"expected (N,3) coordinates, got {coords.shape}")
    count = int(coords.shape[0])
    if count == 0:
        return np.empty((0, 3), dtype=np.int32)
    if capacity is None:
        capacity = _next_power_of_two(max(16, math.ceil(count / max_load)))
    if capacity <= 0 or capacity & (capacity - 1):
        raise ValueError("hash capacity must be a positive power of two")
    block = 256
    states, table, overflow = _KERNEL(  # type: ignore[operator]
        inputs=[coords, mx.array([count, capacity], dtype=mx.uint32)],
        grid=((count + block - 1) // block * block, 1, 1),
        threadgroup=(block, 1, 1),
        output_shapes=[(capacity,), (capacity, 3), (1,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
        init_value=0,
    )
    mx.eval(states, table, overflow)
    overflow_count = int(overflow[0])
    if overflow_count:
        raise HashTableOverflow(f"Metal int3 hash table overflowed for {overflow_count} insertions")
    occupied = np.asarray(states) == 2
    compact = np.asarray(table)[occupied].view(np.int32)
    compact = compact.reshape(-1, 3)
    order = np.lexsort((compact[:, 2], compact[:, 1], compact[:, 0]))
    compact = compact[order]
    # The weak-CAS retry above prevents duplicate insertion. Keep a cheap,
    # deterministic adjacent compaction as a final exactness guard: Metal only
    # exposes relaxed device atomics, and this is O(number of occupied slots)
    # after the table has already done the expensive reduction.
    if len(compact) > 1:
        keep = np.concatenate(([True], np.any(compact[1:] != compact[:-1], axis=1)))
        compact = compact[keep]
    return compact
