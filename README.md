# Experiments with 2D gaussians

MLX-only port of the original JAX/Flax version. Fits 2D Gaussians to images on Apple Silicon (Metal GPU). Not production code — expect rough edges.

## Set up

```bash
git clone https://github.com/belkakari/DrawingWithGaussians.git
cd DrawingWithGaussians
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python fit.py --config-name fit_to_image.yaml
```

The MLX implementation runs on the Apple Silicon Metal backend by default. It uses the same per-epoch split/prune loop as the original JAX version.

## Overview

This is not a "production-ready project" by any means but rather my attempts at a low-level tweaking of different image representations based on gaussians and how to control them. I might move to 3D at some point but for now it's more about 2D and adapting some methods I like to 2D setup.

### MLX-specific notes

- L's diagonal is parameterized in log-space (`log_diag`); the actual diagonal is `exp(log_diag)`. This guarantees positivity by construction (gsplat convention) and avoids the late-epoch NaNs that the raw L parameterization suffered from.
- Multiple Adam optimizers are used (one per trainable parameter), each with its own state. The means optimizer uses a cosine-decayed LR schedule; the rest use a constant LR.
- The per-epoch `split_n_prune` is implemented in numpy because MLX 0.31 has no boolean indexing / `nonzero` / `compress`. It's a per-epoch op so the cost is negligible.
- `mx.compile` is intentionally avoided in the inner training step. The compile boundary fights with the shape change introduced by split/prune (the recompile boundary is brittle across epochs). MLX's lazy evaluation already gives good throughput.

## Fit 2D gaussians to an image

```bash
uv run python fit.py --config-name fit_to_image.yaml
```

![An example of fitting an image](./static/eye_fitting.gif)

Here I initialize 50 gaussians and split them every epoch based on the gradient values. After each epoch I multiply all of the gaussians colors by `cfg.gaussians.color_demp_coeff` which is 0.1 here.

The MLX implementation converges to similar losses as the JAX version (around 0.06 for the default config), and can run with up to ~1500 Gaussians before getting slow on Apple Silicon.

## Diffusion guidance

The original `diffusion_guidance` path (Stable Diffusion img2img) is **not** ported to MLX. Selecting that loss in `fit.py` raises `NotImplementedError`. The pixel-loss path (`fit_to_image.yaml`) is fully supported.

## ToDO
- [x] Move boilerplate to separate functions
- [x] Add SSIM
- [ ] Ability to copy optimizer state from before the pruning (copy for the splitted gaussians)
- [ ] Test "deferred rendering" like in [SpacetimeGaussians](https://oppo-us-research.github.io/SpacetimeGaussians-website/)
- [ ] Port diffusion guidance to MLX (was in original JAX version)
- [ ] Add basic 3D version
- [ ] Add alternative alpha-composing with occlusions (prune gaussians based on opacity, currently prunning based on color norm, probably won't do this untill I'll decide to move to 3D)

## References
Based on [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [fmb-plus](https://leonidk.com/fmb-plus/), [GaussianImage](https://arxiv.org/abs/2403.08551), [gsplat](https://github.com/nerfstudio-project/gsplat). Works on Apple Silicon (M1/M2) up to ~1500 Gaussians.
