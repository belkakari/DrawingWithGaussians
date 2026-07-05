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

## Exp 13: fwd+bwd speedup sweep — IN PROGRESS

User-requested sweep of the top three forward+backward speedup ideas, tested
one after another against the same `scripts/bench_render.py` baseline. The
repo already had uncommitted 3D tiling/binning work at the start of this
sweep (`rendering3d.py`, `rendering3d_fused.py`, `bench_render.py`, plus new
`bench_binning.py`); all numbers below are from that working tree, MLX
0.31.2, 128x128, 100 iters unless noted. Raw logs are also saved under
`benchmark_results/`.

Baseline (`uv run python scripts/bench_render.py`):

| scene | path | N | fwd ms | fwd+bwd ms |
| --- | --- | ---: | ---: | ---: |
| clustered | 3D | 5k | 2.08 | 4.73 |
| clustered | 3D | 20k | 7.20 | 16.79 |
| clustered | 3D | 50k | 18.26 | 41.55 |
| spread | 3D | 5k | 1.76 | 4.00 |
| spread | 3D | 20k | 6.01 | 14.51 |
| spread | 3D | 50k | 15.10 | 35.71 |

Binning baseline (`uv run python scripts/bench_binning.py`): avg tiles/G is
~2.6–2.8, while exact `bin_pad=None` at 128x128 means pad=64. Tuned pad 8/16
is therefore a strong proxy for a compact intersect pipeline: it avoids
sorting many invalid padded keys while preserving exactness for these scenes
(except pad=8 has one fallback at 50k clustered/spread in the standalone
bench; pad=16 had zero fallbacks).

Attempt A: skip unused absgrad atomics in benchmark/training calls that do not
pass `means2d_absgrad_sink`. Added a no-absgrad backward kernel variant and
select it when the sink is `None`. This is a small specialization, not the
full Faster-GS bucketed backward. Result (30 iters): essentially noise-level
speedup, e.g. clustered 50k fwd+bwd 41.55 → 41.33 ms, spread 50k 35.71 →
35.69 ms. Conclusion: absgrad atomics are not the bottleneck; real bucketed
backward still needs a dedicated prototype.

Attempt B: expose `bin_pad` through `pixel_loss_3d` and `bench_render.py` and
benchmark tuned pads as a compact-binning proxy:

| scene | bin_pad | N | fwd ms | fwd+bwd ms | vs baseline fwd+bwd |
| --- | ---: | ---: | ---: | ---: | ---: |
| clustered | 16 | 50k | 15.49 | 38.54 | 1.08x |
| clustered | 8 | 50k | 15.22 | 38.21 | 1.09x |
| spread | 16 | 50k | 12.51 | 32.59 | 1.10x |
| spread | 8 | 50k | 12.10 | 32.30 | 1.11x |

At 20k, pad=8 similarly improved clustered 16.79 → 15.84 ms and spread
14.51 → 13.58 ms. This supports the compact-intersect idea, but the current
pad=8 numbers are not safe as a default because the standalone binning bench
reported rare fallbacks; pad=16 was exact in these synthetic scenes and still
wins. Next step is a true compact Metal/C++ intersect path (count → prefix →
encode real intersections → sort), likely borrowing `gsplat-mlx`'s
`gsplat_intersect.metal` structure and optionally Faster-GS exact tile tests.

Attempt C: tile geometry. The cheapest geometry variant was changing the
3D raster tile from 16x16 / 256 threads to 8x8 / 64 threads. This increases
average tile intersections (clustered ~2.6 → ~5.0 tiles/G; spread ~2.8 →
~5.6 tiles/G) but cuts per-threadgroup work and threadgroup memory. With a
safe tuned `bin_pad=16` (zero fallbacks in `bench_binning.py` for these
synthetic scenes), it was a clear fwd+bwd win over both baseline and the
16x16 tuned-pad proxy:

| scene | tile | bin_pad | N | fwd ms | fwd+bwd ms | vs baseline fwd+bwd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clustered | 8 | 16 | 50k | 13.41 | 34.43 | 1.21x |
| spread | 8 | 16 | 50k | 12.39 | 32.21 | 1.11x |

At 20k, tile8+pad16 improved clustered 16.79 → 14.26 ms and spread
14.51 → 13.39 ms. Correctness still passes `uv run python tests/test_kernels.py`.
However, the exact default path (`bin_pad=None`) gets much slower with tile8
because exact pad becomes 256 tiles: clustered 50k fwd+bwd 49.00 ms, spread
46.62 ms (50-iter check). So tile8 is only a keeper if paired with a tuned
or compact bin builder; it should not replace tile16 while exact padded bins
remain the default. A 32x32 / 1024-thread variant was rejected immediately:
its forward kernel requires ~40 KB threadgroup memory, exceeding Apple
Metal's 32 KB limit for this device.

Fit3d bin-pad tuning: plumbed `bin_pad` through `pixel_loss_3d` and added
`gaussians.bin_pad` config for `fit3d.py` (`auto | exact | integer`). `auto`
projects the current Gaussians at each epoch boundary, computes the max tile
bbox area, applies `bin_pad_margin`, caps at the number of tiles, and retraces
the compiled train step with that integer. Default config now uses
`bin_pad: auto`, `bin_pad_min: 16`, `bin_pad_margin: 2.0`.

Short 512x512 `fit3d.py` smoke benchmarks (default N=500, SSIM on,
`optim.num_steps=50`, `train.log_frequency=25`; logs in
`benchmark_results/`):

| run | epoch | bin_pad | late step time |
| --- | ---: | ---: | ---: |
| exact | 0 | exact (=4096 tiles for tile8) | 11.8 ms |
| auto margin 2 | 0 | 1566 | 10.6 ms |
| exact | 1 after refine to N=586 | exact | 12.4 ms |
| auto margin 2 | 1 after refine to N=586 | 1960 | 11.0 ms |

So the current `fit3d.py` default gets about **1.11–1.13x** late-step speedup
on this short 512x512 run. A margin-4 check chose 3132 then exact 4096 after
refine and matched the exact loss more closely, but gave little/no speedup;
margin 2 is the current speed/strictness trade-off. A direct initial-state
comparison showed exact and `bin_pad=1566` are bit-identical for the first
render (`max|image diff| = 0`). Full-run quality should still be checked
before treating this as final, because a smaller fixed pad can change the
optimization trajectory if Gaussians grow within an epoch.

Remaining planned attempt: full Faster-GS bucketed backward. The simpler
no-absgrad specialization above showed the extra absgrad atomics are not the
bottleneck, so the bucketed backward would need the real Faster-GS structure:
forward checkpoints every 32 Gaussians and a backward kernel over
(tile, bucket) with one lane/Gaussian accumulating across the tile's pixels.

Attempt D0 (context, from the parallel session): two flavors of reduced
gradient scatter in the tiled backward were tried before binning landed.
A **threadgroup-level** reduction (simd_sum → 8 partials in threadgroup
memory → one atomic per component per tile) was a clear REGRESSION —
clustered 50k fwd+bwd 39.3 → 48.0 ms: the two threadgroup barriers per
gaussian dominate in hot tiles (~10⁵ barriers/tile). The **simdgroup-only**
reduction (gsplat's warpSum: `simd_sum` + one atomic per 32 lanes, zero
barriers — `simd_reduce_add4` in rendering3d_fused.py) was neutral-to-
slightly-positive (39.3 → 38.2 clustered) and is kept. Conclusion matching
Attempt A: relaxed device atomics are cheap on Apple Silicon; atomic traffic
was never the 3D backward bottleneck at these sizes.

### Attempt D: the actual bottleneck was projection (batched 3x3 matmuls) — KEPT, ~7-13x

Per-stage decomposition at N=50k spread 128x128 (before this change):

| stage | ms |
| --- | ---: |
| projection alone (compiled) | 11.19 |
| full pre-kernel pipeline (projection+sort+gathers+radii+bins) | 11.87 |
| forward rasterization kernel | 0.67 |
| fwd+bwd rasterization kernels | 1.70 |

The rasterizer everyone was optimizing cost 1.7 ms of a ~33 ms step. The
batched `(N, 3, 3)` matmuls in `project_gaussians` (`R@S`, `M@M^T`,
`Rcw@Σ@Rcw^T`, `J@Σ@J^T`) dispatch GEMM kernels that are pathological for
50k tiny matrices, and their VJPs roughly double the cost in backward.

Fix: **fully scalarized projection** (gsplat's CUDA structure) — rotation
entries, cov3d, camera rotation and `J Σ J^T` written as elementwise
expressions over (N,) arrays so `mx.compile` fuses the chain. Same math,
fp reassociation only; autodiff untouched. Isolated projection: 11.19 →
**1.17 ms fwd, ~1.0 ms fwd+bwd**. (A transposed (D, N)-layout variant
halves forward again to 0.44 ms via contiguous row slices, but doesn't help
fwd+bwd — not worth the layout churn; noted for later.)

Full `bench_render.py` with the tuned `--bin-pad 16`, tile8 (128x128):

| scene | N | fwd ms | fwd+bwd ms | Exp 13 baseline f+b | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| clustered | 5k | 0.78 | 1.35 | 4.73 | 3.5x |
| clustered | 20k | 1.33 | 2.56 | 16.79 | 6.6x |
| clustered | 50k | 2.63 | **5.12** | 41.55 | **8.1x** |
| spread | 20k | 1.01 | 1.76 | 14.51 | 8.2x |
| spread | 50k | 1.47 | **2.70** | 35.71 | **13.2x** |

With the exact default pad (`bin_pad=None` → 256 at tile8), forward is
dominated by the 12.8M-key bin sort (26.8 ms fwd at 50k) — same tile8
caveat Exp 13 already recorded; the tuned/auto pad is where the wins are.

Validation: all fused-vs-dense gradient checks pass (max err 1.7e-5, well
inside tolerances); goldens regenerated (`--write`) for the documented
reassociation-only change. Training sanity at 512x512 (the config that
exposed the earlier bin-pad truncation bug): loss 0.431 → 0.278 over epoch
0 and 0.327 → 0.229 after refine — healthy, vs the stuck 0.436 → 0.424 of
the truncation era.

Observation from that run: `bin_pad: auto` escalated from 1566 to exact
4096 after the first refine (split children's bboxes grow). Next levers for
512x512+ and the new `train_colmap3d.py` path: Faster-GS exact tile-overlap
tests (`will_primitive_contribute`) to shrink per-gaussian tile areas below
the bbox count, a pad cap with the oversized few streamed exactly, and —
for the COLMAP trainer specifically — caching decoded images as mx arrays
(`_load_item` currently does PIL decode + resize + host->device transfer
inside the hot loop, per step).

## Exp 14: camera-batched rendering (no loop) — KEPT

`project_gaussians` and `rasterize3d_fused` now accept a camera batch in
gsplat's `[..., C, N]` convention (`_fully_fused_projection` as reference):

- **Projection**: free by construction after Exp 13's scalarization —
  camera entries are indexed with an ellipsis (`viewmat[..., i, j][..., None]`)
  so a (C, 4, 4) batch broadcasts as (C, 1) against (N,) gaussians. One
  fused elementwise chain, no loop, no vmap.
- **Rasterization**: one launch per pass for the whole batch — grid z = C,
  with tiles, pixels and gaussian rows laid out view-major. The bins for
  all views are built together (view-offset tile ids, one global sort; key
  dtype auto-switches uint32 -> int64 when `C * ntiles * C * N` overflows).
  Per-view depth `argsort(axis=-1)` + gathers of the shared (N, ...) params;
  their VJPs scatter-add gradients over views automatically, including the
  `means2d_offset` (net grad) and absgrad sinks.
- **Loss/SSIM**: `pixel_loss_3d` takes (B, H, W, 3) targets with batched
  viewmats/Ks; SSIM's depthwise convs are batched natively in NHWC. Since
  the loss is the *mean* over views, the per-step absgrad scale matches
  single-view runs — `grad_thr` semantics survive batching unchanged.

Correctness: `tests/test_kernels.py::test_batched_3d` — batched loss and
all seven gradients (params + offset + absgrad) match the per-view loop to
~1e-10.

`train_colmap3d.py` gains `--camera-batch B` plus two supporting fixes:

- **Preloaded views** (`_preload_views`): the old path ran PIL decode +
  resize + normalize *inside* the training loop, several ms of synchronous
  CPU work per step (x B with batching). All train views are now decoded
  once into a stacked uint8 MLX buffer (151 views @ 512x338 = 78 MB, 0.9 s)
  and normalized on the GPU stream; per-step "loading" is an `mx.take`.
  (Unified memory note: there is no host->device transfer in MLX — the
  numpy->MLX copy is a same-DRAM memcpy; the decode/resize was the cost.)
- **Multi-view-safe `bin_pad auto`**: the max tile-bbox area is now taken
  over ALL train cameras at epoch start (~1 ms/view with the scalarized
  projection), not one reference view — an undersized pad silently
  truncates bins.

Flowers @ 512x338, N=5000 init, 100 steps:

| camera_batch | time/step | time per rendered view |
| ---: | ---: | ---: |
| 1 | 86 ms | 86 ms |
| 4 | 176 ms | **44 ms (1.95x)** |

Plus the statistical benefit of 4-view-averaged gradients per optimizer
step. Both configs are currently dominated by the auto-pad escalation
(pad=2752 on this scene -> a 13.8M-key bin sort per step at B=1): the
FasterGS exact tile-overlap test / pad cap remains the top follow-up, and
its payoff now multiplies by B.

## Roadmap v3: next three speedups (post Exp 14), by expected value

1. **Compact intersection lists** (gsplat `isect_tiles` / gsplat-mlx
   `gsplat_intersect.metal` structure): count real tile overlaps in a
   gaussian-parallel kernel -> `mx.cumsum` offsets -> scatter kernel ->
   sort only the real ~1e5 intersections instead of N x pad padded keys
   (13.8M on flowers at pad=2752, x B with camera batching). Capacity
   buffer with INVALID tail keeps shapes static under mx.compile (retrace
   only on capacity bumps). Also permanently removes the pad-truncation
   failure mode. Est. COLMAP step 86 -> ~10-15 ms.
2. **Exact tile-overlap + opacity-scaled cutoff radius** (FasterGS
   `will_primitive_contribute`, `max_power_threshold = ln 255`): bbox ->
   exact ellipse-tile test, and radius `sqrt(2 ln(255 opac))` instead of
   flat 3.33 sigma (0.77x at opacity 0.1, far smaller for floaters).
   1.5-2.5x fewer intersections; folds into 1's counting kernel; dense
   reference gets the same radius for test parity.
3. **Profile-gated**: decompose after 1 lands, then either (a) FasterGS
   bucketed backward + per-32-gaussian forward checkpoints (if kernels
   dominate again at 512p/N>=50k), or (b) whisper-style `mx.async_eval`
   pipelining + eval-set trims (if launch/sync gaps dominate; ~10-20%,
   stacks with everything).

## Exp 15: compact exact intersections + opacity-scaled cutoff — KEPT (Roadmap v3 #1 + #2)

Implemented in `rendering3d_fused.py`: count -> `mx.cumsum` -> scatter of
*real* tile intersections (gsplat `isect_tiles` structure), sorted into a
static `capacity` buffer with an INVALID tail (compile-friendly; retrace
only when capacity is re-chosen). Two culling upgrades ride along, both
provably lossless because they reproduce the composite kernels' own skip
criterion `sigma <= ln(255 * opac)` (the dense reference applies the same
alpha >= 1/255 cut, so fused-vs-dense tests still pass):

- **exact convex ellipse-vs-tile test** (`min_sigma_rect`: unconstrained
  minimum inside the rect -> 0, else edge stationary points, clamped) —
  FasterGS's `will_primitive_contribute`;
- **opacity-scaled radius** `sqrt(2 ln(255 opac))` x marginal std instead
  of flat 3.33 sigma (0.77x at opac 0.1; transparent floaters shrink far
  more; opac <= 1/255 drops out entirely).

Review verdicts (audited): per-tile depth order preserved (keys =
tile x flatN + sorted rank); scatter is atomic-free/deterministic (prefix
offsets); uint32 key overflow falls back to the int64 padded path;
continuous-rect minimum only over-includes (false positives, never false
negatives). Capacity overflow (`pos < capacity`) silently drops tail
intersections — mitigated by per-epoch re-estimation with 2x margin over a
*stable* statistic (total intersections, vs the old fragile max-area pad)
plus utilization telemetry in the epoch log; a mid-epoch >2x explosion
remains theoretically silent (documented).

Trainer wiring: `fit3d.py` and `train_colmap3d.py` both size capacity per
epoch by counting real intersections (colmap: one batched projection+count
over ALL train views, worst = sum of top-B per-view totals). fit3d 512p
now runs capacity=178k keys where the padded path sorted 0.8-2.4M.

Measured (flowers, 512x338, N=5000, 100 steps; loss trajectories match the
padded runs exactly):

| config | before (padded, pad=2752) | after (compact) | per view |
| --- | ---: | ---: | ---: |
| B=1 | 86 ms | **~19-28 ms (3-4.5x)** | ~24 ms |
| B=4 | 176 ms | **~98-108 ms (1.7x)** | **~26 ms** |

`bench_render` 50k clustered `--bin-pad 16`: fwd+bwd 5.12 -> **4.18 ms**
(exact culling shrinks the backward) with fwd 2.63 -> 3.25 (the extra
count+scatter passes) — net win. fit3d 512p epoch-0 step 10.6 -> 13.1 ms
at N=500: the two extra kernel launches show at tiny N (launch-bound);
acceptable, revisit only if small-N matters.

Remaining follow-ups surfaced by telemetry were addressed in the next pass:
giant SfM-init splats are now attacked at init/refine time, and compact-bin
capacity now has exact preflight overflow detection before optimizer updates.

## Exp 16: giant-gaussian controls + safe tighter capacity — KEPT

Implemented the Roadmap v4 speed controls:

- **k-NN COLMAP init scales** in `train_colmap3d.py`: default
  `--init-scale-mode knn` computes per-point scale from 3-nearest-neighbor
  spacing (`--init-knn-k`, `--init-scale-mult`, `--init-scale-min`,
  `--init-scale-max`; default max 0.05). The trainer logs p50/p95/max
  initial scales so screen-filling outliers are visible immediately. The
  old flat scale remains available with `--init-scale-mode constant`.
- **`prune_scale3d`** in `split_n_prune_3d`: optional too-big pruning by
  `max(exp(log_scales)) > prune_scale3d * scene_scale`, reported separately
  as `n_prune_scale3d`. Defaults: `fit_to_image_3d.yaml` uses 0.2;
  COLMAP trainer CLI uses 0.1 and `<=0` disables.
- **mean/p95/worst capacity policy** for COLMAP camera batches:
  `--bin-capacity-stat mean|p95|worst` (default mean). Integer `--bin-pad`
  now means compact capacity `B * N * pad`, not a padded per-row expansion.
  The epoch log includes capacity, utilization estimate, mean/p99/max
  tiles/G telemetry.
- **exact overflow preflight**: before a compact-capacity training step,
  the sampled batch is projected and counted exactly. If
  `real_intersections > capacity`, the step is *not* run; capacity is bumped
  from the exact count with a modest overflow margin (`--bin-overflow-margin`,
  default 1.25; `bin_overflow_margin` in the Hydra config), the compiled step
  is rebuilt, and then the optimizer update proceeds. This closes the old silent-truncation hole. The
  earlier idea `bounds[-1] == capacity` is only a possible-overflow/full
  buffer signal; exact detection is the count sum.
- Added `scripts/profile_3d_step.py` to decompose projection, bin build,
  projected raster, full L1 fwd/fwd+bwd, and SSIM fwd/fwd+bwd before picking
  the next low-level kernel target.

Validation so far: `uv run pytest tests/test_kernels.py` passes. A small
synthetic smoke confirmed k-NN scales clamp as intended and `prune_scale3d`
removes oversized rows. Full flowers timing still needs a clean run after the
2DGS worktree settles.

## Exp 17: RGB-only 2DGS surfel mode — CAPABILITY KEPT, not a speedup

Implemented Phase 1 of 2DGS as a geometry/surface capability for the
COLMAP/scan path:

- `rendering2dgs.py`: differentiable MLX reference for gsplat's
  `_fully_fused_projection_2dgs` + `accumulate_2dgs` RGB path. Projection is
  scalarized like Exp 13 and supports camera batching. The AABB math was
  audited against gsplat: the AABB must be computed from `M = T_sl^T`, i.e.
  `M[..., 2]` corresponds to `T_sl`'s third **row** `(m20, m21, m22)`, not
  its third column. A fronto-parallel on-axis disk now projects to
  `(cx, cy)` exactly.
- `rendering2dgs_fused.py`: RGB fused Metal rasterizer with ray-splat
  intersection sigma `0.5 * min(u^2 + v^2, 2 ||pixel - mean2d||^2)`, a
  custom VJP through means2d/ray transform/opacities/colors, absgrad sink,
  camera-batch launch layout, and compact bbox bin builder.
- `pixel_loss_2dgs` and `train_colmap3d.py --mode 2dgs` wire the new path
  into training. 3DGS remains the default; 2DGS regularizers are not on by
  default.
- Pytest now includes a dense-vs-fused 2DGS RGB regression; current run:
  `uv run pytest tests/test_kernels.py` -> 5 passed.

Memory note: 2DGS is expected to use more memory than 3DGS in this Phase-1
implementation. The per-splat projected state is a full 3x3 ray transform
(9 floats) plus its gradient, versus a 3-float conic for 3DGS; the backward
also returns `dray_transforms` `(N, 3, 3)`. Compact bbox bins avoid the
worst padded-key blow-up, but if memory is tight use `--mode 3dgs`, lower
`--camera-batch`, or tighter capacity. Exp 18 packs the hot raster state into
`float4` records; remaining memory/speed reductions are no-absgrad/no-ray-grad
specializations for eval/preview and, later, avoiding materialized ray-transform
grads when 2DGS projection is fused deeper.

## Exp 18: 2DGS rasterizer bin/layout tuning — KEPT, still needs COLMAP timing

Three low-level changes from the Apple-Silicon optimization pass were tried and
kept in `rendering2dgs_fused.py`:

- **Reduced compact-bin work**: the 2DGS compact count/scatter kernels now use
  opacity-aware exact tile contribution tests instead of writing every projected
  AABB tile. The screen-space fallback (`||pixel - mean||^2 <= ln(255 opac)`) is
  checked first, then the ray-splat branch solves the exact quadratic condition
  `tu^2 + tv^2 <= 2 ln(255 opac) tw^2` over the tile rectangle. This should only
  remove tiles that the raster kernel would skip anyway.
- **Packed/vectorized raster state**: forward/backward now pack each splat into
  four `float4` records (`mean+opacity+R`, ray row 0 + G, ray row 1 + B, ray row
  2 + pad). The Metal kernels stage/read `float4`s instead of 15 scalar floats;
  the VJP scatters into a packed `dparams` buffer and unpacks to the public MLX
  gradients.
- **Fast exp in 2DGS raster only**: `metal::fast::exp` replaced
  `metal::precise::exp` in the 2DGS forward/backward alpha path. The dense-vs-
  fused tolerances still pass.

Validation:

```bash
uv run python tests/test_kernels.py
python -m py_compile drawingwithgaussians/rendering2dgs_fused.py train_colmap3d.py
```

Quick synthetic 2DGS timing (`scripts.bench_render.scene_3d`, spread scene,
128x128, SSIM off, 20 iters; both columns include the new packed/fast-exp
kernels, so this mainly compares exact padded bins vs compact `bin_pad=16`):

| N | bins | fwd ms | fwd+bwd ms |
| ---: | --- | ---: | ---: |
| 1k | exact | 1.07 | 1.18 |
| 1k | pad16 | 0.37 | 0.51 |
| 5k | exact | 1.38 | 1.63 |
| 5k | pad16 | 0.49 | 0.71 |
| 10k | exact | 3.21 | 3.59 |
| 10k | pad16 | 0.58 | 0.94 |

A direct N=5k exact-vs-pad16 render check was bit-identical (`max|image diff| =
0`, loss diff `0`). This is promising for 2DGS capacity tuning, but it is not a
substitute for a clean COLMAP run with real camera batches, overflow checks, and
memory telemetry.

Clean flowers timing at 512x338, B=1, N=20k, `bin_pad: auto`, SSIM 0.2,
`prune_scale3d: 0.1`, one 2k-step epoch (logs from 2026-07-05):

| mode | bins | steady step | wall incl. load/init/save | notes |
| --- | --- | ---: | ---: | --- |
| 2DGS | capacity=320k, mean≈40.8k, tiles/G mean=2.0 p99=19 max=2752 | ~11.0 ms | ~23 s | fewer bins, but more expensive ray-splat math + 3x3 transform grads |
| 3DGS | capacity=320k, mean≈56.8k, tiles/G mean=2.8 p99=29 max=2752 | ~8.2 ms | ~17 s | still faster despite more bin entries |

Conclusion: Exp 18 makes the 2DGS path usable and compact-bin efficient, but on
this flowers B=1 run 2DGS remains ~1.3-1.4x slower than 3DGS. The fact that 2DGS
has fewer intersections but slower steps strongly suggests the next 2DGS target
is raster/projection math or gradient payload, not more tile culling.

Added `scripts/profile_2dgs_step.py` to make that decision empirical. It mirrors
`profile_3d_step.py` and reports projection, compact bins, projected raster fwd,
projected raster fwd+bwd, full L1 fwd/fwd+bwd, and optional SSIM fwd/fwd+bwd.
Example:

```bash
uv run python scripts/profile_2dgs_step.py --n 20000 --width 512 --height 338 --iters 30 --ssim-weight 0.2
```

Small smoke (`N=5k`, `128x128`, spread, compact auto, SSIM off, 5 iters) ran
successfully and showed the profiler plumbing working. A same-dim synthetic run
matching the flowers resolution (`N=20k`, `512x338`, compact auto, SSIM 0.2,
30 iters, repeated 5x) produced stable high-level conclusions:

| stage | observed range |
| --- | ---: |
| projection only | 0.71-1.34 ms |
| compact bins only | 0.86-2.18 ms |
| projected raster fwd | 1.07-1.44 ms |
| projected raster fwd+bwd | 1.94-1.98 ms |
| full fwd+bwd L1 | 2.13-2.16 ms |
| full fwd+bwd SSIM 0.2 | 7.84-7.94 ms |

So on this synthetic profile, SSIM adds ~5.7 ms and dominates the training-step
cost far more than the 2DGS rasterizer. Caveat: the synthetic profile had
~229k real tile intersections (50% of 458k capacity), while the actual flowers
2DGS epoch log had mean≈40.8k intersections, so use a dataset-backed profiler
before making exact per-stage claims for COLMAP. The direction is clear enough:
optimize/skip/schedule SSIM before attempting more 2DGS raster micro-tuning.

## Exp 19: fused Metal SSIM — KEPT, big 512p win

Ported the 2D Metal structure from [`fused-ssim`](https://github.com/rahul-goel/fused-ssim) into
`drawingwithgaussians/ssim_fused.py` as an MLX `mx.fast.metal_kernel` +
`mx.custom_function` implementation for NHWC/BHWC images. Forward fuses the
five 11x11 Gaussian-window statistics (`mu1`, `mu2`, `E[x^2]`, `E[y^2]`,
`E[xy]`) inside one tiled kernel and stores the derivative maps needed by the
custom VJP. Backward performs the adjoint Gaussian filtering from fused-ssim and
returns gradients for the rendered image only; targets are constants in this
project. `losses.ssim` now routes through the fused path.

Validation:

```bash
uv run python tests/test_kernels.py
python -m py_compile drawingwithgaussians/ssim_fused.py drawingwithgaussians/losses.py
```

The existing SSIM numpy-reference test still passes (`ssim(x, x) == 1`, fp64
reference tolerance, finite gradients). On the same synthetic 2DGS profile that
made SSIM the bottleneck (`N=20k`, `512x338`, compact auto, SSIM 0.2, 30 iters),
full SSIM fwd+bwd dropped from ~7.9 ms to ~2.47-2.48 ms:

| stage | before fused SSIM | after fused SSIM |
| --- | ---: | ---: |
| full fwd+bwd L1 | 2.13-2.16 ms | 2.16-2.18 ms |
| full fwd SSIM 0.2 | 4.62-4.67 ms | 1.30-1.32 ms |
| full fwd+bwd SSIM 0.2 | 7.84-7.94 ms | 2.47-2.48 ms |

So the SSIM overhead at this resolution shrank from ~5.7 ms to ~0.3 ms. The
next real check is a clean `train_colmap3d.py` run; expected steady-state 2DGS
B=1 time should move much closer to the L1-only/raster bound.

## Exp 20: visibility-normalized COLMAP densification — PRELIMINARY KEEP

Stage 3 of the COLMAP trainer plan replaces the raw, resolution-dependent
absgrad norm with

`norm(absgrad * camera_batch * (width/2, height/2)) / visible_view_count`.

The fused 3DGS and 2DGS rasterizers can now return exact tile-intersection
counts in original parameter order. Compact mode reuses its builder counts and
undoes the depth sort; padded/exact mode and the uint32-key fallback invoke the
standalone exact count kernel. Counts stay device-side through each compiled
step. Unit coverage includes both rasterizers, compact and padded modes, forced
uint32 fallback, depth/parameter order, batch invariance, partial visibility,
and opacity/depth culling.

The fixed-N overhead gate passes on flowers:

| mode | legacy | normalized | overhead | budget |
| --- | ---: | ---: | ---: | ---: |
| compact, 20K, 512x338, B=4 | 21.98 ms | 22.18 ms | **0.9%** | 3% |
| padded exact, 2K, 256x169, B=4 | 9.27 ms | 9.40 ms | **1.4%** | 5% |

The first shadow sweep used 20K initial points and suggested `grad_thr=4e-4`.
That threshold did not transfer to the current 2K-point config: after the first
500-step epoch it selected 1 grow candidate while 568 points were pruned.
The expanded 2K sweep selected:

| threshold | duplicate | split | grow | pruned |
| ---: | ---: | ---: | ---: | ---: |
| 1e-5 | 87 | 500 | 587 | 568 |
| 2e-5 | 83 | 347 | 430 | 568 |
| 4e-5 | 66 | 157 | 223 | 568 |
| 1e-4 | 23 | 25 | 48 | 568 |
| 4e-4 | 0 | 1 | 1 | 568 |

At 2K initialization, seed 1, five 500-step epochs, `1e-5` fixes the population
collapse and improves the run:

| threshold | N by epoch boundary | best PSNR | final PSNR | final SSIM |
| --- | --- | ---: | ---: | ---: |
| 4e-4 | 2000→1433→1132→944→867 | 17.05 | 15.74 | 0.2758 |
| 1e-5 | 2000→2009→2276→3112→4856 | **17.39** | **16.28** | **0.3262** |

The full run therefore validates `1e-5` for the current 2K initialization.
This is not a universal threshold: the 20K initialization needs a higher
value. The default config now uses the 2K low-init regime with `1e-5`; the
20K sweep remains useful as the override point when `max_init_points` moves
back to the old setting. Multi-seed quality validation remains required before
treating the Stage 3b default as fully settled.

## Exp 21: remaining trainer-plan controls — IMPLEMENTED, A/B PENDING

Stages 4–6 are now available with conservative defaults:

- Periodic opacity reset caps opacity at `2 * prune_opa`, runs after
  split/prune and before optimizer reconstruction, and clears the opacity Adam
  moments after optional state carry. `reset_opacity_every: 6` is a no-op in
  the default five-epoch run.
- 2DGS normal and distortion regularizers use epoch-boundary warm-up at
  `0.233` and `0.1` of total steps. Effective weights are logged in every epoch
  header.
- Overflow policy accepts `preflight | lazy | off` plus legacy booleans.
  The default is now `lazy`. It uses the uncapped sum of exact builder counts,
  reports per-epoch event count and p50/p95/max real intersections, accepts
  the detected truncated step, then grows capacity and recompiles.
- Camera sampling accepts `random | shuffle`; the default is now `shuffle`.
  It produces deterministic fixed-size batches, avoids intra-batch duplicates
  when `n_views >= camera_batch`, and preserves balanced coverage across
  permutation wraps.

An integrated 2DGS smoke forced all non-default paths
(`reset_opacity_every=1`, `bin_pad=1`, lazy overflow, shuffle sampling). Lazy
overflow detected `19546 > 8000` intersections on the first step, rebuilt to
128K capacity, opacity reset fired at the first boundary, and both regularizers
activated at the next epoch. The run completed with exact overflow telemetry
and held-out evaluation. Unit tests cover reset/moment clearing, warm-up
boundaries, overflow parsing/detection/capacity growth, and shuffle edge cases.
The plan's multi-seed quality A/Bs for reset, warm-up, lazy overflow, and
shuffle remain pending, but the config now defaults to the low-init Stage 3–6
setup (`max_init_points=2000`, `densify_signal=normalized`, `grad_thr=1e-5`,
`bin_check_overflow=lazy`, `view_sampling=shuffle`).

## Exp 22: DashGaussian scheduling in fit3d — freq resolution KEPT (~23% faster), budget stabilizes N

Ported the two portable, framework-agnostic schedulers from DashGaussian
(arXiv:2503.18402, CVPR'25) into the `fit3d` path, plus its LR-delay. All in a
new pure-numpy `drawingwithgaussians/schedule.py` (kept out of the trainer so it
unit-tests in isolation); every knob defaults to the current behavior.

- **Frequency-guided coarse-to-fine resolution** (`resolution_mode: freq`, paper
  Eqs. 6-7). `resolution_schedule` runs one FFT of the target, picks the maximum
  downscale so the low-frequency window still keeps `1/a` of the spectral energy
  (`start_significance_factor`, default 4), and returns a **1:1 epoch -> integer
  downscale** map (non-increasing, last epoch full res). fit3d renders each epoch
  at `(H/r, W/r)` with a `cv2.INTER_AREA` target downsample and a `K` rebuilt
  from the resized dims (principal point stays centered). The per-epoch retrace
  already absorbs the shape change; `choose_bins`/overflow use the epoch dims.
- **Momentum primitive-count budget + top-k densification** (`densify_mode:
  budget`, Eqs. 4-5), an automatic alternative to the hand-tuned `grad_thr`
  capacity knob (Exp 8/9/12). `MomentumBudget` sets an EMA target `P_fin` and a
  resolution-coupled per-refine target `P_i`; `split_n_prune_3d` gains a budget
  branch that swaps the resolution-dilated absolute `grad_thr` gate for a
  `grad_percentile` relative gate and densifies only the **top-k by signal**
  (union of dupli+split, branch preserved), with `k = clamp(target - n_kept, 0,
  0.2*n_kept)`. Crucially the budget is fed the **realized post-clamp `k`**, not
  the ~0.5N candidate count, so `P_fin` tracks real scene demand instead of being
  pinned to a constant multiple of N.
- **LR delay** (`lr_decay_from_full_res`): holds the means LR constant across
  reduced-res epochs and starts `cos`/`cos_restart` decay at the first full-res
  epoch, via the existing global `step_offset` path. No-op with `means_mode:
  const` (the fit3d default).

Verification (short smoke runs): the `free`/`const` default is byte-identical to
the pre-change baseline (loss 0.43109/0.43409/0.43189, prune 474->26, split
19->40 — matched a stashed `git HEAD` run). `resolution_mode=freq` produced
schedule `[8,5,3,2,1,1]` (64->102->171->256->512 px), with the budget target
ramping `630->900->1383->3000` as resolution rose — coarse-to-fine count growth,
a smooth ramp rather than collapsed into the last epoch. `k` is correctly
rate-limited to `0.2*n_kept` per refine; bin capacity recomputes per epoch dims;
the video buffer upsamples reduced-res frames so the strip never shape-mismatches
the full-res target. Unit tests cover `resolution_schedule` (non-increasing,
last==1, degenerate cases), `MomentumBudget` (monotone, truncated fixed point,
fixed-budget/target-count), and the `split_n_prune_3d` budget top-k (exact top-k
survives; `k<=0` no growth). Kernel and trainer suites unchanged (8 + 48 pass).

**Prerequisite fix — the default collapsed.** On eye.jpeg the shipped defaults
(`grad_thr=1e-5`, `prune_scale3d=0.2`) collapse to ~20 gaussians: the init scales
`log(uniform(1e-3,1))` reach 1.0, and `prune_scale3d=0.2` (threshold `0.2*scene=0.4`)
scale-prunes ~92% at the first epoch boundary, after which the weak densification
never refills (0 duplicated, ~2-13 split/epoch vs 7-22 pruned). It is a
prune-dominated collapse, not a densification-rate problem. Single knob fix:
`prune_scale3d 0.2 -> 0.35` (threshold 0.7) stops nuking the oversized inits, N
stabilizes at ~1k, loss 0.35 -> ~0.13, `grad_thr` unchanged. A grad_thr x
prune_scale3d sweep confirmed a clean loss-vs-N Pareto (ps0.5 grows N to 5k-31k
with diminishing loss gains past ~5k); 0.35 is the fast, healthy ~1k knee. This
is now the fit3d default.

**A/B — 5 seeds x 3 variants at the tuned default (10 epochs x 500 steps, 512p).**
Seed-averaged (the fused 3D backward's atomic adds are ulp-nondeterministic, and
the discrete split-selection amplifies it, so N swings 362-1203 across seeds at
fixed strategy — averaged rather than made bitwise-deterministic):

| variant                    | loss (mean+/-std) | final N (mean+/-std) | wall  |
| -------------------------- | ----------------- | -------------------- | ----- |
| A free/const (baseline)    | 0.1493 +/- 0.0059 | 760 +/- 295          | 21.7s |
| B budget/const             | 0.1533 +/- 0.0113 | 439 +/- 80           | 20.9s |
| C budget/freq              | 0.1506 +/- 0.0082 | 450 +/- 54           | 16.6s |

Findings: (1) **Loss is equal within noise** across all three (gaps ~0.004 <
per-seed std 0.006-0.011; SEMs overlap) — the budget does not improve quality.
(2) **The count budget's payoff is N stability**, not loss: it cuts final-N
variance ~4-5x (std 295 -> 54-80) and uses fewer gaussians at equal loss — the
automatic-capacity benefit over the hand-tuned `grad_thr` made concrete.
(3) **`resolution_mode=freq` is the clear win: ~23% lower wall-clock at equal
quality** (16.6 vs 21.7s), from coarse-to-fine early epochs; budget/const is
baseline speed as expected (same resolution). The resolution schedule carries the
speedup even at this small scale and should widen on larger scenes / longer runs.

Verdict: KEEP `resolution_mode=freq` + `densify_mode=budget` as the recommended
non-default combo for fit3d (same fit, ~23% faster, predictable N). Not made
default pending confirmation on more scenes than eye.jpeg. Step D (LR-delay) is
still unexercised — it needs `means_mode != const`, so it is out of scope for the
`const`-default A/B above.

## Roadmap v5: next work, by expected value

1. **Run clean COLMAP timing after Exp 19**: flowers B=1/B=4 with k-NN init,
   `prune_scale3d`, mean capacity, overflow logging, and both `--mode 3dgs` and
   `--mode 2dgs`. Record p50/p95/max tiles/G, bin utilization, memory high-water,
   and whether 2DGS `bin_pad: auto` remains exact under per-epoch growth.
2. **Re-profile 2DGS/3DGS with fused SSIM**: use `scripts/profile_2dgs_step.py`
   and `scripts/profile_3d_step.py` to decide whether the next target is
   projection, compact count/scatter/sort, raster fwd/bwd, L1, or remaining
   SSIM overhead.
3. **Specialize eval/preview paths**: no-absgrad and possibly no-ray-grad 2DGS
   kernels should reduce backward/preview memory traffic. This is lower risk
   than changing compositing order and can be selected when the training signal
   does not need absgrad or ray-transform gradients.
4. **Only if profiling says raster bwd dominates**: prototype a Faster-GS-style
   bucketed 2DGS backward/checkpoint scheme. Otherwise focus on projection/bin
   build or SSIM convs.
5. **2DGS Phase 2**: normal/depth/distortion outputs and gsplat's two
   regularizers, off by default. Phase 3 TSDF meshing remains out of scope.
6. **2D image path borrowings from gsplat** remain open: port Exp 15's
   compact/tiled cutoff to the 2D renderer, add 2D absgrad, and A/B gsplat's
   every-100-step refine cadence. Not worth taking: packed rasterization modes,
   MCMC.

## Summary

| path                       | per step        | speedup   | quality check                              |
| -------------------------- | --------------- | --------- | ------------------------------------------ |
| 2D dense (baseline)        | 40.8 ms @ N=1500 | 1x       | —                                          |
| 2D fused + compiled step   | 2.2 ms @ N=1500  | **18.5x** | closer to fp64 truth than dense (Exp 4)   |
| 3D dense (reference)       | 91 ms @ N=2000¹  | —        | —                                          |
| 3D fused + compiled step   | 5.4 ms @ N=5000  | **~37x**¹ | 53x closer to fp64 truth than dense (Exp 7)|
| 3D after Exp 13 (tiles+bins+scalarized projection) | 5.1 ms fwd+bwd @ N=50000² | **8-13x** over Exp 13 baseline | fused-vs-dense ≤1.7e-5, goldens |

¹ value_and_grad microbench at equal N=2000; the dense 3D path was never a
trainer, it exists as the validation reference.
² `bench_render.py --bin-pad 16`, clustered scene; spread is 2.7 ms.

Repro:

```bash
uv run python fit.py --config-name fit_to_image.yaml       # 2D with densification
uv run python fit3d.py --config-name fit_to_image_3d.yaml  # 3D with densification
```
