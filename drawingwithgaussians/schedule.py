"""DashGaussian (arXiv:2503.18402) resolution + primitive-count schedulers.

Pure numpy, deliberately kept out of the trainer so it stays unit-testable in
isolation. Two portable pieces are ported here (see EXPERIMENTS.md for the A/B):

* :func:`resolution_schedule` — frequency-guided coarse-to-fine render
  resolution (paper Eqs. 6-7). The FFT magnitude spectrum of the target image
  decides how far the render resolution can be dropped early (the low-frequency
  window must still keep ``1/a`` of the spectral energy) and when to step it
  back up. This repo densifies only at epoch boundaries, so instead of the
  paper's per-iteration curve we sample it once per epoch: a **1:1 epoch ->
  integer downscale factor** map, non-increasing, last epoch always full res.

* :class:`MomentumBudget` — automatic primitive-count budget (paper Eqs. 4-5),
  an alternative to the hand-tuned ``grad_thr`` capacity knob. ``update`` is fed
  the **realized, post-clamp** densification count (not the candidate count), so
  the momentum estimate tracks real scene demand instead of being pinned to a
  constant multiple of N (see the plan / EXPERIMENTS.md).

The N-images -> N=1 reduction is trivial (a single target image), and the FFT
runs once at setup in numpy, so nothing here touches MLX or the compiled step.
"""

import math

import numpy as np


def _magnitude_spectrum(target_np):
    """Centered FFT magnitude spectrum of an image (or a stack of views).

    ``target_np`` is ``(H, W)``, ``(H, W, C)``, or a view stack ``(N, H, W, C)``
    in any float range. Returns a ``(C, H, W)`` magnitude array (channel-first,
    ``C == 1`` for grayscale); a stack returns the **mean** spectrum over views,
    matching DashGaussian's multi-view ``scene_freq_image`` accumulation.
    Computed in float64 for a stable bisection.
    """
    img = np.asarray(target_np, dtype=np.float64)
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim == 4:  # (N, H, W, C) -> mean per-view spectrum
        acc = None
        for k in range(img.shape[0]):
            s = _magnitude_spectrum(img[k])
            acc = s if acc is None else acc + s
        return acc / img.shape[0]
    img = np.transpose(img, (2, 0, 1))  # (C, H, W)
    fft = np.fft.fftshift(np.fft.fft2(img, axes=(-2, -1)), axes=(-2, -1))
    return np.abs(fft)


def _win_significance(sig_map, scale):
    """Sum of spectral magnitude inside the centered ``(H/scale, W/scale)``
    window (DashGaussian ``compute_win_significance``)."""
    h, w = sig_map.shape[-2:]
    cy, cx = (h + 1) // 2, (w + 1) // 2
    wh, ww = int(h / scale), int(w / scale)
    win = sig_map[..., cy - wh // 2 : cy + wh // 2, cx - ww // 2 : cx + ww // 2]
    return float(win.sum())


def _scale_solver(sig_map, target_significance, iters=64):
    """Bisection for the downscale factor whose centered low-frequency window
    captures ``target_significance`` of the spectral energy (DashGaussian
    ``scale_solver``). Returns a factor ``>= 1``."""
    lo, hi, mid = 0.0, 1.0, 0.5
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _win_significance(sig_map, 1.0 / mid) < target_significance:
            lo = mid
        else:
            hi = mid
    return 1.0 / mid


def _res_scale_at(iteration, reso_scales, reso_level_begin, increase_reso_until):
    """Stateless port of DashGaussian ``get_res_scale``: the interpolated
    (not yet floored) downscale factor at ``iteration``, inverse-area
    interpolation between adjacent schedule levels."""
    if iteration >= increase_reso_until:
        return 1.0
    if iteration < reso_level_begin[1]:
        return float(reso_scales[0])
    i = 1
    while i + 1 < len(reso_level_begin) and iteration >= reso_level_begin[i + 1]:
        i += 1
    i_now, i_nxt = reso_level_begin[i], reso_level_begin[i + 1]
    s_lst, s_now = reso_scales[i - 1], reso_scales[i]
    if i_nxt == i_now:
        return float(s_now)
    frac = (iteration - i_now) / (i_nxt - i_now)
    inv_area = frac * (1.0 / s_now**2 - 1.0 / s_lst**2) + 1.0 / s_lst**2
    return (1.0 / inv_area) ** 0.5


def build_freq_schedule(
    target_np,
    increase_reso_until,
    start_significance_factor=4.0,
    max_reso_scale=8.0,
    reso_sample_num=32,
):
    """Build the DashGaussian frequency resolution schedule (Eqs. 6-7).

    Returns a schedule dict queryable at any global step by
    :func:`resolution_factor_at`, or ``None`` when the target has no headroom to
    downscale. Resolution ramps from the coarsest level at step 0 to full res at
    ``increase_reso_until``. ``target_np`` may be a single image or a view stack
    (mean spectrum). This is the step-queryable core shared by the per-epoch
    :func:`resolution_schedule` (fit3d) and the segment-based
    :func:`resolution_segments` (train_colmap3d).
    """
    if increase_reso_until <= 0:
        return None
    sig_map = _magnitude_spectrum(target_np)
    e_total = float(sig_map.sum())
    if e_total <= 0.0:
        return None

    # Cap the downscale so the lowest-res window still keeps 1/a of the energy.
    e_min_cap = e_total / float(start_significance_factor)
    max_reso_scale = min(float(max_reso_scale), _scale_solver(sig_map, e_min_cap))
    if max_reso_scale <= 1.0 + 1e-6:
        return None  # target has no headroom to downscale

    E_total = e_total
    E_min = _win_significance(sig_map, max_reso_scale)
    if not (E_min > 0.0 and E_total > E_min):
        return None
    denom = math.log(E_total / E_min)

    # Build the internal (reso_sample_num) level schedule, faithful to
    # DashGaussian.init_reso_scheduler (log-modulated cumulative-energy spacing).
    reso_scales = [max_reso_scale]
    reso_level_begin = [0]
    level_sig = [E_min]
    for i in range(1, reso_sample_num - 1):
        s_i = (E_total - E_min) * i / (reso_sample_num - 1) + E_min
        level_sig.append(s_i)
        reso_scales.append(_scale_solver(sig_map, s_i))
        level_sig[-2] = math.log(level_sig[-2] / E_min)
        reso_level_begin.append(int(increase_reso_until * level_sig[-2] / denom))
    reso_scales.append(1.0)
    level_sig[-1] = math.log(level_sig[-1] / E_min)
    reso_level_begin.append(int(increase_reso_until * level_sig[-1] / denom))
    reso_level_begin.append(int(increase_reso_until))
    return {
        "reso_scales": reso_scales,
        "reso_level_begin": reso_level_begin,
        "increase_reso_until": int(increase_reso_until),
    }


def resolution_factor_at(sched, iteration):
    """Integer (floored) downscale factor at global ``iteration`` for a schedule
    from :func:`build_freq_schedule`. ``None`` schedule -> full res (1)."""
    if sched is None:
        return 1
    scale = _res_scale_at(iteration, sched["reso_scales"], sched["reso_level_begin"], sched["increase_reso_until"])
    return max(1, int(scale))


def resolution_segments(
    target_np,
    increase_reso_until,
    start_significance_factor=4.0,
    max_reso_scale=8.0,
    reso_sample_num=32,
):
    """Coarse-to-fine resolution as ``(start_step, factor)`` segments.

    Scans steps ``[0, increase_reso_until]`` and emits one entry each time the
    integer downscale factor changes: a compact, non-increasing list starting at
    ``(0, coarsest)`` and reaching factor ``1`` by ``increase_reso_until``. Used
    by segment-driven trainers to add resolution boundaries independent of the
    densification boundaries. ``[(0, 1)]`` when the schedule is degenerate.
    """
    sched = build_freq_schedule(
        target_np, increase_reso_until, start_significance_factor, max_reso_scale, reso_sample_num
    )
    if sched is None:
        return [(0, 1)]
    segs = []
    prev = None
    for step in range(0, int(increase_reso_until) + 1):
        f = resolution_factor_at(sched, step)
        if f != prev:
            segs.append((step, f))
            prev = f
    if segs[-1][1] != 1:
        segs.append((int(increase_reso_until), 1))
    return segs


def resolution_schedule(
    target_np,
    num_epochs,
    num_steps,
    start_significance_factor=4.0,
    max_reso_scale=8.0,
    reso_sample_num=32,
):
    """Per-epoch integer downscale factors for coarse-to-fine training.

    Returns a list of length ``num_epochs``, non-increasing, with the last
    entry ``== 1`` (final epoch at full resolution). Resolution is increased
    across the first ``num_epochs - 1`` epochs and the last epoch renders full
    res; ``num_epochs <= 1`` disables the schedule (all ones).

    Args:
        target_np: the target image ``(H, W)`` or ``(H, W, C)``.
        num_epochs, num_steps: trainer schedule; total steps ``S`` implied.
        start_significance_factor: ``a`` in the paper; the lowest-res window
            keeps ``1/a`` of the spectral energy (larger -> lower start res).
        max_reso_scale: hard cap on the downscale factor.
        reso_sample_num: internal number of significance levels (>= 2).
    """
    if num_epochs <= 1:
        return [1] * max(1, num_epochs)

    increase_reso_until = (num_epochs - 1) * num_steps
    sched = build_freq_schedule(
        target_np, increase_reso_until, start_significance_factor, max_reso_scale, reso_sample_num
    )
    if sched is None:
        return [1] * num_epochs

    factors = [resolution_factor_at(sched, epoch * num_steps) for epoch in range(num_epochs)]
    factors[-1] = 1  # final epoch always full resolution
    # Enforce non-increasing (interpolation is monotone, but flooring is safe).
    for i in range(len(factors) - 1):
        factors[i] = max(factors[i], factors[i + 1])
    return factors


def clamp_densify_count(target_count, n_kept, max_rate=0.2):
    """Realized densification count ``k`` for one refine (DashGaussian's
    per-refine growth clamp): grow ``n_kept`` toward ``target_count`` but by at
    most ``max_rate`` of the surviving population. Guarantees
    ``0 <= k <= max_rate * n_kept``."""
    cap = int(max_rate * n_kept)
    k = int(target_count) - int(n_kept)
    k = max(0, min(k, cap))
    assert 0 <= k <= cap, (k, cap, target_count, n_kept)
    return k


class MomentumBudget:
    """Automatic primitive-count budget (DashGaussian Eqs. 4-5).

    The budget ``P_fin`` is an EMA-style momentum over the realized per-refine
    densification count, so it converges to ``P_init + eta/(1-gamma) * k`` for a
    sustained realized count ``k`` (paper Eq. 8). Set ``max_n_gaussian > 0`` to
    pin a fixed budget instead (momentum disabled). Internally tracks
    ``momentum = P_fin - P_init`` exactly like the reference implementation.
    """

    def __init__(self, p_init, gamma=0.98, eta=1.0, max_n_gaussian=-1, step_cap=1_000_000):
        self.p_init = int(p_init)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.step_cap = int(step_cap)
        if max_n_gaussian is not None and int(max_n_gaussian) > 0:
            self._momentum = None  # fixed budget
            self._p_fin = int(max_n_gaussian)
        else:
            self._momentum = 5 * self.p_init
            self._p_fin = self.p_init + self._momentum

    @property
    def p_fin(self):
        return self._p_fin

    @property
    def is_fixed(self):
        return self._momentum is None

    def update(self, k):
        """Fold the realized post-clamp densification count ``k`` into the
        momentum (Eq. 5). No-op for a fixed budget."""
        if self._momentum is None:
            return self._p_fin
        step = min(self.step_cap, self.eta * max(0.0, float(k)))
        self._momentum = max(self._momentum, int(self.gamma * self._momentum + step))
        self._p_fin = self.p_init + self._momentum
        return self._p_fin

    def target_count(self, r, step, total):
        """Target primitive count for downscale ``r`` at global ``step`` over
        ``total`` steps (Eq. 4). At full res (``r == 1``) this is ``P_fin``."""
        r = max(1.0, float(r))
        total = max(1, int(total))
        exponent = 2.0 - float(step) / float(total)
        return int((self._p_fin - self.p_init) / (r**exponent)) + self.p_init
