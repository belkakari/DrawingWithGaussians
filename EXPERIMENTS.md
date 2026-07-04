# Runtime optimization experiments

Goal: reduce per-step time of `fit.py` without changing output quality.
Reference for production patterns: [gsplat](https://github.com/nerfstudio-project/gsplat)
(local checkout at `~/repos/gsplat`) — fused CUDA forward/backward kernels that never
materialize the (N, P) gaussian-pixel matrix.

Further local references for future work:
- `~/repos/msplat` — full 3DGS *training* engine in pure Metal compute shaders
  (`core/metal/msplat_metal.metal`): tile-based rasterization with per-tile
  bitonic sort (the scaling fix if N grows past ~10k; our 3D path loops all
  gaussians per pixel), separable 11-tap SSIM fwd/bwd (recipe for the unused
  `ssim_weight` knob), prefix/suffix-transmittance backward, GPU-resident
  densification. Same shader language as our `mx.fast.metal_kernel` sources.
- `~/repos/MetalSplatter` — Swift 3DGS *viewer* (no training); its
  PLYIO/SplatIO modules document the .splat/PLY conventions for exporting
  `fit3d.py` results to an interactive viewer.
- `~/repos/gsplat-mlx` — third-party port of gsplat's ops as MLX *C++ custom
  extensions* (`mlx::core::Primitive` + Metal kernels + nanobind), validated
  against npz fixtures exported from gsplat CUDA. Pinned to mlx 0.30.0 and
  does **not** compile on mlx ≥ 0.31 (this repo runs 0.31.2), so it's a
  reference, not a dependency. Best bits: the `note/` source maps of
  gsplat's backward API / CUDA→Metal correspondences, the fixture-based
  validation methodology, tile-intersection kernels in MLX-flavored Metal,
  and SPZ export scripts (pairs with MetalSplatter).

Benchmark setup unless noted: 128x128 image (`configs/fit_to_image.yaml`), N=1500
gaussians, Apple Silicon / Metal, MLX 0.31.2. Times are steady-state per-step
(forward + backward + Adam updates + `mx.eval`), from `fit.py`'s own logging.
Microbenchmarks (value_and_grad only) noted separately.

## Baseline

- 40.8 ms/step at N=1500 (fit.py); 40.5 ms/iter for value_and_grad alone.
- ~1.0–1.3 ms/step at N=10–35 (early epochs are launch/graph-build dominated).
- The hot path is `rasterize`: builds several (N, P) = (1500, 16384) fp32
  intermediates (dx, dy, pdf, z, intensity ≈ 98 MB each) — bandwidth-bound,
  forward and especially backward.

## Exp 1: `mx.compile` on `value_and_grad` — KEPT

Wrap the pure `loss_and_grad` in `mx.compile` (fit.py); optimizer updates stay
outside. Fuses the (N, P) elementwise chains in forward/backward, cutting
intermediate memory traffic.

- 40.5 → 23.1 ms/iter microbench; fit.py step 40.8 → 23.4 ms (**1.74x**).
- Gradients **bit-identical** to uncompiled (loss value differs 2e-6 — fp32
  reduction order only). Full 3-epoch run at same seed: logged loss trajectory
  identical to baseline.
- The old worry that compile "fights with" the per-epoch `split_n_prune` shape
  change is unfounded: shape change just triggers one shape-keyed retrace per
  epoch (~8 retraces per run, negligible).

## Exp 2: cache the pixel grid — KEPT (neutral on speed, less graph churn)

`rasterize` rebuilt the (P,) pixel-coordinate arrays with `mx.repeat` on every
call (every step, forward + backward graph). Now cached per (H, W, dtype) and
evaluated once (`_pixel_grid` in rendering2d.py). No measurable speed change at
N=1500 (grid build is lazy and tiny vs the raster), but removes per-step graph
nodes and keeps constants stable across `mx.compile` traces.

## Exp 3: pdf as feature matmul — REJECTED (numerics)

Reformulate the quadratic form as `[x², y², xy, x, y, 1] @ coeffs` — one
(P, 6) x (6, N) GEMM instead of the (N, P) elementwise chain; dx/dy never
materialize.

- Fast: 12.1 ms uncompiled, **8.6 ms compiled** (4.7x vs baseline).
- But expanding `(x - mu)²` in fp32 causes catastrophic cancellation for
  small-variance gaussians (error ~ eps · p · max(x², mu²)). At realistic
  converged-like params (sigma down to 0.3): rendered max|diff| 0.89,
  mean|diff| 4.9e-2 (≫ 1/255), means-grad relative error ~1.9. Changes the
  optimization trajectory → violates the "no quality change" constraint.
  Centering coordinates at the image midpoint helps 4x but not enough.

## Exp 4: gsplat-style fused Metal kernels — KEPT (default path)

`drawingwithgaussians/rendering2d_fused.py`, used by `losses.pixel_loss`;
the dense `rendering2d.rasterize` stays as the reference implementation.
Port of gsplat's kernel structure via `mx.fast.metal_kernel` +
`mx.custom_function`: pixel-parallel forward accumulation, gaussian-parallel
analytic backward, exact per-pixel `x - mu` math (no expansion, so none of
Exp 3's cancellation), peak normalization included — that one needs an extra
per-gaussian min-pdf/argmin pass that gsplat doesn't have, and the min's
gradient (+S at the argmin pixel) is part of the backward. The (N, P) matrix
is never materialized. Small chains (L → cov → precision, loss) stay in MLX
autodiff.

- value_and_grad at N=1500: 40.5 → **3.96 ms** (first version).
- Numerics vs fp64 numpy ground truth (worst-case synthetic state, sigma
  down to 0.3, all gaussians overlapping): the fused path is *closer* to
  fp64 than the dense MLX path on every metric — loss err 1.3e-3 vs 2.7e-3,
  rendered max err 3.4e-3 vs 5.7e-3, dcolors 6.3e-5 vs 1.0e-4, dmeans equal.
  The fused-vs-dense delta is fp32 rounding noise, and the fused side of it
  is the more accurate one. Kahan compensated summation in all serial
  accumulation loops; `metal::precise::exp`.

## Exp 5: kernel tuning — KEPT

Profiling the 3.96 ms: pass A (min-pdf) cost 1.14 ms and ran twice (forward
+ recomputed in the VJP); the gaussian-parallel passes ran only N≈1500
threads each scanning all 16K pixels serially (poor occupancy).

- Return `minpdf`/`argmin` as extra outputs of the custom function so the
  VJP reuses them (custom_function's `outputs` arg) instead of recomputing.
- Chunk passes A and C over 64 pixel blocks — grid (chunks, N), ~96K
  threads — with the tiny cross-chunk reductions (argmin-of-partial-mins,
  sum-of-partial-grads, min's argmin-pixel gradient term) in regular MLX ops.
- value_and_grad at N=1500: 3.96 → **2.20 ms**. Numerics unchanged
  (same fp64-truth errors as Exp 4).

## Exp 6: compile the whole train step — KEPT

`fit.py` builds one `mx.compile` region per epoch containing loss/grad plus
all five Adam updates, with optimizer state threaded through
`inputs=`/`outputs=` (standard MLX pattern). Also: per-step NaN check reads
the already-realized loss (`math.isnan(loss.item())`) instead of building an
`mx.isnan` graph each step.

## 2D results (Exps 1–6)

fit.py per-step at N=1500 (128x128): **40.8 → 2.2 ms (18.5x)**.

Full default config run (8 epochs x 1000 steps, N grows 10 → ~600), same
seed, wall clock including ~2.5 s interpreter/Hydra startup:

| variant                     | wall time | final loss | final N |
| --------------------------- | --------- | ---------- | ------- |
| dense + compiled step       | 18.9 s    | 0.0181     | 407     |
| fused + compiled step       | 5.7 s     | 0.0165     | 588     |

Note the trajectories genuinely diverge (split/prune thresholds on gradient
norms amplify fp32-level differences into different gaussian counts) — but
the fused run converges as well or better, and per-step numerics are
verifiably closer to fp64 than the dense path, so this is trajectory noise,
not quality loss.

Phase-1-only numbers (kept for reference): `mx.compile` on value_and_grad
alone gave 40.8 → 23.4 ms/step with a bit-identical loss trajectory.

## Exp 7: 3D gaussian splatting in MLX (fit3d.py) — NEW FEATURE

MLX port of gsplat's `examples/image_fitting.py` (3dgs mode): a fixed number
of 3D gaussians (means, log-scales, wxyz quats, opacity/color logits) seen
through a fixed pinhole camera (fov_x 90°, camera at z=-8), fitted to an
image with a single Adam, no densification.

- `rendering3d.py`: EWA projection (quat/scale → cov3d → clamped-Jacobian
  perspective → conic, +0.3 px low-pass) in regular MLX autodiff ops, plus a
  dense reference rasterizer (alpha compositing via `mx.cumprod` on the
  materialized (N, P) matrix).
- `rendering3d_fused.py`: fused Metal kernels mirroring gsplat's
  `RasterizeToPixels3DGS*` CUDA kernels (untiled): pixel-parallel
  front-to-back forward with gsplat's exact thresholds (alpha clamp 0.99,
  1/255 skip, exclusive 1e-4 transmittance early termination),
  pixel-parallel back-to-front backward replay with `atomic_fetch_add`
  scatter (compositing order couples pixels to gaussians, so a
  gaussian-parallel backward would be O(N²P)). Depth sort + gather and
  background compositing stay in MLX autodiff; the cotangent on the final
  transmittance carries the background path into the kernel backward.

Validation (N=2000 synthetic, worst-case overlap): image max diff dense vs
fused 7e-4 (≪ 1/255), loss diff 9e-8. Gradient max diffs ~1e-6 absolute —
shown NOT to be early termination (disabling it changes nothing) but the
dense reference's own fp32 error: against fp64 compositing ground truth the
fused kernel is 53x more accurate on mean image error (1.7e-7 vs 9.0e-6) and
matches fp64 exactly at the worst-diff pixel. Atomic adds make fused
backward grads non-deterministic at the ulp level across runs (same as
gsplat).

Speed at N=2000, 128x128: dense value_and_grad 91 ms → fused 2.4 ms
(compiled), **37x**. Training run (`fit_to_image_3d.yaml`, N=5000, 2000
steps): 5.4 ms/step, ~12 s wall, loss 0.273 → 0.051.

![3D gaussian splatting reconstruction (left: render, right: target)](./static/eye_fitting_3d.png)

Deviations from gsplat (documented, deliberate): scales stored in log-space
(repo convention; gsplat passes linear scales), L1 pixel loss instead of MSE
(repo convention), fixed black background, no tile culling (single small
image; the 1/255 skip already ignores far-away gaussians).

## Densification: `split_n_prune` vs gsplat's DefaultStrategy — REVIEW (no code change)

Comparison of this repo's per-epoch `split_n_prune` (gaussian.py) against
gsplat's `strategy/default.py` (the 3DGS-paper strategy plus refinements).
The mechanics of the *split op itself* match gsplat: children means sampled
from the parent's covariance, child covariance = parent / 1.6, parent
removed. Everything around it differs:

| aspect | this repo | gsplat DefaultStrategy |
| --- | --- | --- |
| trigger signal | **instantaneous** means-grad from one extra `loss_and_grad` at epoch end | **running mean** of per-step screen-space means2d-grad norms, averaged over steps the gaussian was visible (`grad2d / count`); optional `absgrad` (better, per their EXPLORATION.md) |
| threshold | raw pixel-space norm > `grad_thr` (1e-5), resolution-dependent | grads normalized to [-1, 1] screen space, > 2e-4; scale thresholds relative to `scene_scale` |
| cadence | every epoch (1000 steps), never stops | every 100 steps, only inside `[500, 15000)` — densification *stops* well before training ends so the population can converge |
| grow small GSs | always split (children shrink by 1.6x even when the parent is tiny) | grad-high & small → **duplicate** (clone, parent kept); grad-high & large → split; optional revised-opacity correction for cloning |
| prune | color norm < 0.05 (color ≈ opacity in the additive renderer), or variance < 0.05 (degenerate slivers) | sigmoid(opacity) < 0.005; too-big-in-world (scale > 0.1 · scene_scale) after the first opacity reset |
| oversized GSs | no upper sigma/variance cap; large grad-high gaussians split via the grow-scale branch | no cap; too-big gaussians are pruned |
| newborn init | split children colors x0.1 (soft start); background x0.1 every refine | split children keep opacity (optional `revised_opacity` correction); no background damping |
| optimizer state | all five Adams **rebuilt from scratch** every epoch; the means' cosine schedule restarts too | Adam moments **preserved** for surviving gaussians, zeroed only for new rows (`_update_param_with_optimizer`) |

(A correction from the first version of this review: the old strategy does
*not* damp all colors — only the newborn children (x0.1 as init) and the
background. The per-epoch loss spike in the logs comes from the background
damp + cold optimizers, not a full-image wipe.)

## Exp 8: gsplat-style densification A/B — MIXED RESULT, partially KEPT

Implemented the gsplat DefaultStrategy elements on `split_n_prune` and ran
the ladder against the old strategy (reconstructed verbatim as the A/B
baseline), full config 10 epochs x 2000 steps, 128x128, same seed:

| variant | final loss | final N |
| --- | --- | --- |
| OLD strategy (baseline) | 0.00898 | 1138 |
| H0: full gsplat port (accum signal, dupli/split, carried moments, full-run cosine, no damping, children x0.5) | 0.02951 | 397 |
| H1: H0 + SGDR warm-restart means LR | 0.02475 | 464 |
| H2: H1 + grad_thr 1e-6 | 0.02230 | 490 |
| H5: H2 + scale moments not carried | 0.02556 | 404 |
| H6: H2 + full color reset every epoch | 0.02545 | 312 |
| H7: H2 + children x0.1 + bg damp x0.1 (old newborn init) | 0.01886 | 545 |
| H8: H7 + **no optimizer-state carry at all** | 0.00839 | 2706 |
| H9/H10 (=defaults): H8 + duplicate branch (+ thr 1e-6) | **0.00739 / 0.00756** | ~2650 |

Seed 2 confirms: old 0.01023 @ 1080 vs new defaults 0.00747 @ 2701.

Findings, in causal order:

1. **Carrying Adam moments across refines — gsplat's orthodoxy — is the
   single most harmful ingredient here** (H8 vs H7 is the largest jump).
   Sustained shrink momentum on `log_diag` collapses variances en masse
   (prune logs: ~80–130 variance-prunes per refine with carry, ~0
   without), the collapsed gaussians hit the variance prune, and the
   population stops doubling. Fresh state each refine — what the old code
   did by accident — is what lets N grow 10 → ~2700. `carry_optimizer_state`
   is kept in the code but **off by default**.
2. **Warm-restart LR is load-bearing** (H1 vs H0): each refine's newborns
   need high LR; a single full-run cosine strands late generations at
   near-zero LR. `means_mode: cos_restart` implements SGDR restarts via
   `step % period` so it composes with carried step counters.
3. **The old newborn init (children x0.1 + background damp) beats
   contribution-preserving x0.5 children** (H7 vs H2): near-invisible
   newborns + a forced global re-fit level the field each epoch.
4. **The duplicate branch for small gaussians helps** (H9 vs H8, ~12%)
   once the rest of the regime is right.
5. The accumulated-gradient signal is a wash at this scale (thr 1e-6 ≈
   thr 0): with 10 starting gaussians on one image, capacity growth — not
   split selectivity — is the bottleneck. It still replaces the extra
   end-of-epoch forward/backward probe (small speed win) and makes
   `grad_thr` a meaningful, stable knob for larger runs.

Net result vs old strategy: **~20% lower loss** (0.0076 vs 0.0090 at seed
1, 0.0075 vs 0.0102 at seed 2) at equal steps, with reproducible splits
(the old path sampled children via unseeded global `np.random`), a
vectorized split (children `L = L/sqrt(1.6)` exactly — the cov→cholesky
round-trip was unnecessary), refine-count logging, and no wasted refine
after the final epoch. Final N is ~2.3x larger (~2650 vs ~1140) — the
fused rasterizer makes that free (~3 ms/step at N=2700).

## Exp 9: densification for the 3D path (fit3d.py) — KEPT

Same strategy ported to 3D (`gaussian3d.py`), with the fit.py config API
(`num_epochs` x `num_steps`, refine at epoch boundaries, `num_epochs: 1` =
fixed-N training, i.e. the original gsplat image_fitting behavior).
3D-specific pieces: the original densification signal here was the true
*net* screen-space means2d gradient, obtained by adding a zero
`means2d_offset` parameter to the projected means (MLX's equivalent of
gsplat's `retain_grad`); split children take gsplat's `revised_opacity`
correction (`1 - sqrt(1 - a)`), since alpha compositing double-counts a
plain opacity copy; prune is `sigmoid(opacity) < 0.005`; fresh optimizer
state per refine (2D finding; `carry_optimizer_state` knob available).

A/B at equal total steps (5000), 128x128, before the later absgrad change:

| variant | final loss | final N | late step time |
| --- | --- | --- | --- |
| fixed N=5000 (`num_epochs=1`) | 0.031 | 5000 | 5.4 ms |
| densified 500 →, `grad_thr=1e-5` (default) | **0.0113** | 2452 | 3.3 ms |
| densified 500 →, `grad_thr=1e-6` | **0.0035** | 22407 | 24.6 ms |

Densification wins decisively: 2.8x lower loss than fixed-N with *half*
the gaussians (and faster steps), or 9x lower loss if capacity is allowed
to grow 4.5x. Unlike the 2D path, prunes are rare (opacities stay high),
so `grad_thr` is the capacity knob: split fractions grow as the fit
sharpens, so 1e-6 compounds into unbounded doubling — hence 1e-5 as the
default.

## Roadmap: prioritized borrowings from the reference repos

1. **Kernel regression fixtures** (gsplat-mlx methodology) — DONE, Exp 10.
2. **Real SSIM loss** (msplat's separable 11-tap fwd/bwd) — DONE, Exp 11.
3. **absgrad densification signal** (gsplat EXPLORATION.md) — DONE, Exp 12.
4. **Tile-based rasterization** (msplat kernels; gsplat-mlx intersect ops
   as MLX-flavored reference): removes the all-gaussians-per-pixel wall
   (24.6 ms/step at N=22k). Large effort — do when growth regimes or
   bigger images are actually wanted.
5. **PLY/SPZ export** (MetalSplatter SplatIO conventions, gsplat-mlx
   script shape): PLY export is DONE (`fit3d.py` writes `final.ply`, using
   gsplat's standard uncompressed PLY layout with SH degree 0). SPZ remains
   optional future work.

Skipped deliberately: fused-Adam kernels (mx.compile already fuses ours),
GPU-resident densification (per-epoch numpy refine is free at this
cadence), gsplat's MCMC strategy (research detour).

## Exp 10: kernel regression tests — KEPT

`tests/test_kernels.py` (methodology from gsplat-mlx's fixture testing, but
anchored to our own independent implementations instead of CUDA exports):
seeded synthetic scenes (2D incl. tiny sigmas + off-diagonals, 3D incl.
low/high opacities), three layers of checks —

1. fused vs dense reference, all gradients, tolerances set from the
   measured fp32 bounds (`dbg` gets a looser one: it flows through
   `sign(rendered - target)`, and pixels at the L1 sign boundary flip
   between implementations at 2/(3HW) each);
2. fused forward vs fp64 numpy compositing truth (must stay at least as
   accurate as the dense path);
3. golden npz fixtures of the fused outputs themselves (`--write` to
   regenerate) so silent drift is caught even if both live paths move
   together. 2D goldens are deterministic; 3D backward uses atomics, but
   reproduces to ~4e-12 at test size — tolerance 1e-5 leaves headroom.

Also covers SSIM against a direct fp64 numpy reference (matches to 1.2e-8).

## Exp 11: real SSIM loss — KEPT (default `ssim_weight: 0.2`)

`ssim_weight` had been dead since the JAX port (it only scaled L1 by
(1-w)). `losses.ssim` now implements the standard 11x11 sigma=1.5
gaussian-window SSIM as two separable depthwise `mx.conv2d` passes
(msplat's 121→22-taps formulation) in pure MLX ops — at 128x128 the convs
are negligible next to the rasterizer, so no custom kernel; autodiff
provides the backward. Loss: `(1-w)*L1 + w*(1-SSIM)` (3DGS convention).

A/B at default configs, final-frame metrics vs target (uint8-quantized
video frames):

| run | PSNR | SSIM | L1 | final N | late step |
| --- | --- | --- | --- | --- | --- |
| 2D, w=0 | 34.93 dB | 0.9555 | 0.0123 | 2559 | 4.9 ms |
| 2D, w=0.2 | **35.18 dB** | **0.9571** | 0.0124 | 2230 | 5.9 ms |
| 3D, w=0 | 32.75 dB | 0.9315 | 0.0145 | — | — |
| 3D, w=0.2 | **37.42 dB** | **0.9692** | **0.0098** | — | — |

2D: modest but real win, with 13% fewer gaussians, ~1 ms/step overhead.
3D: large win (+4.7 dB) — SSIM improves even the pure-L1 metric.
Default is now `ssim_weight: 0.2` in both configs.

## Exp 12: absgrad densification signal for 3D — KEPT (default)

The original 3D signal used the *net* screen-space means2d gradient, i.e.
the gradient that would flow through `means2d_offset`. At higher resolution
this was too conservative: the image loss is mean-reduced over pixels, so
512x512 diluted the per-pixel signal by ~16x relative to 128x128, and
opposing pixel gradients also cancel. A 512x512 run with
`grad_thr=1e-5`, `num_steps=2000` only split single-digit/low-double-digit
counts per refine and plateaued around N≈650.

Implemented gsplat-style `absgrad` in the fused 3D backward: each
per-pixel contribution to the means2d gradient is accumulated as
`abs(gmx), abs(gmy)` into an extra atomic output. The custom VJP exposes
that through an ignored zero `means2d_absgrad_sink` argument, so
`fit3d.py` can collect it inside the same compiled step without changing
the actual optimization gradients. Config knob: `gaussians.absgrad`
(default `true`); `false` restores the old net-gradient signal.

Smoke check at 512x512 with only 50 initial gaussians and 5 steps already
showed the signal is no longer suppressed: refine after epoch 0 split 34
of 50 gaussians (vs the previous low split counts after thousands of
steps). Kernel regression tests now also check that the absgrad VJP path is
finite and nonzero. `grad_thr` is again the capacity knob; with absgrad on,
old thresholds may be more aggressive and should be retuned for target N.

## Summary

| path                       | per step        | speedup   | quality check                              |
| -------------------------- | --------------- | --------- | ------------------------------------------ |
| 2D dense (baseline)        | 40.8 ms @ N=1500 | 1x       | —                                          |
| 2D fused + compiled step   | 2.2 ms @ N=1500  | **18.5x** | closer to fp64 truth than dense (Exp 4)   |
| 3D dense (reference)       | 91 ms @ N=2000¹  | —        | —                                          |
| 3D fused + compiled step   | 5.4 ms @ N=5000  | **~37x**¹ | 53x closer to fp64 truth than dense (Exp 7)|

¹ value_and_grad microbench at equal N=2000; the dense 3D path was never a
trainer, it exists as the validation reference.

Repro:

```bash
uv run python fit.py --config-name fit_to_image.yaml       # 2D with densification
uv run python fit3d.py --config-name fit_to_image_3d.yaml  # 3D with densification
```
