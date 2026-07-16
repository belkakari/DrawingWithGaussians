# Repository Guidelines

## Project Scope

This is an experimental research project for fitting 2D and 3D Gaussians (Gaussian splatting) to images. It is an
MLX-only port of an earlier JAX/Flax implementation, not production software; expect research-project rough edges.

MLX targets Apple Silicon and Metal. CUDA and non-Apple development environments are unsupported. Dense reference
renderers are intended for small numerical checks (the historical local limit was roughly 1,500 Gaussians); use the
fused/tiled paths for realistic training sizes.

## Project Structure & Module Organization

Core code lives in `drawingwithgaussians/`; entry points are `fit.py`, `fit3d.py`, and `train_colmap3d.py`. These are
Hydra applications whose YAML configuration lives under `configs/`; override configuration values on the command
line. Utilities are in `scripts/`, and tests plus golden `.npz` fixtures are in `tests/`. Put local datasets in ignored
`inputs/`, documentation media in `static/`, and generated runs in `outputs/<date>/<time>/`.

## Environment, Build & Development Commands

Use `uv` and the Python 3.11 interpreter pinned by `.python-version` and `requires-python`. Some locked scientific
wheels are not consistently available on newer Python versions. Do not rely on a bare Conda environment.

```bash
uv python install 3.11                         # if the pinned interpreter is missing
uv sync                                        # install runtime and development dependencies
uv run python fit.py --config-name fit_to_image.yaml
uv run python fit.py --config-name fit_to_image.yaml optim.num_steps=2000
uv run python fit3d.py --config-name fit_to_image_3d.yaml
uv run python train_colmap3d.py --config-name train_colmap3d.yaml data.dir=/path/to/scene
uv run pytest -q
uv run pre-commit run --all-files
```

Hydra overrides can be appended to any entry point, for example
`uv run python fit.py --config-name fit_to_image.yaml optim.num_epochs=50`. Alternatively, activate uv's local
environment with `source .venv/bin/activate`; still let uv create and synchronize that environment.

## Package Internals

`gaussian.py` owns 2D parameters (means, log-Cholesky diagonal, raw off-diagonal, colors, and background), optimizer
setup, and split/prune refinement. The Cholesky diagonal is `exp(log_diag)` and is therefore positive by construction;
`L[1, 0]` remains unconstrained in raw space.

`gaussian3d.py` manages 3D means, log-scales, quaternions, opacity logits, degree-3 spherical harmonics, and
gsplat-style densification; `selective_adam.py` updates only visible rows. Split children use gsplat's
`revised_opacity` correction.

`rendering2d.py`, `rendering3d.py`, and `rendering2dgs.py` are dense numerical references. Their `_fused.py`
counterparts implement tiled `mx.fast.metal_kernel` rasterization and `mx.custom_function` VJPs following gsplat's
kernel structure. `losses.py` combines L1, fused SSIM, distortion, depth, and normal terms, while `lpips_mlx.py`
supplies perceptual evaluation. `schedule.py`, `utilization.py`, `sh.py`, and `photometric.py` handle resolution and
refinement scheduling, pruning telemetry, view-dependent color, and camera correction. `evaluation.py`,
`single_image_eval.py`, and `splat_export.py` write metrics, manifests, and 3DGS PLY files; `dtu.py`, `tsdf_mlx.py`,
and `spatial_hash_metal.py` cover DTU evaluation and surface fusion.

## MLX & Metal Gotchas

- Keep MLX computations functional and lazy, with realization boundaries (`mx.eval`) explicit. Training steps use
  `mx.value_and_grad` and compile loss/gradient plus Adam updates with optimizer state threaded through
  `mx.compile(inputs=state, outputs=state)`. Shape-changing refinement rebuilds the compiled step and optimizer as
  needed; one retrace per epoch/segment is expected and negligible.
- Performance work and A/B decisions are recorded in `EXPERIMENTS.md`. Benchmark realized outputs and gradients, not
  graph construction alone.
- Preserve dense renderers as references. After a kernel edit, compare images and gradients against the dense path and
  use an independent fp64 NumPy anchor for accuracy questions. Fused paths can be more accurate than reassociated dense
  fp32 expressions. The 3D backward uses atomic adds, so gradients can vary at the ulp level.
- Dynamic selected-row compaction in `split_n_prune` and `split_n_prune_3d` materializes through NumPy because pinned
  MLX 0.31 lacks dynamically sized boolean indexing/`mx.nonzero`/`mx.compress`. These operations run only at
  refinement boundaries, so the eager overhead is negligible.
- In MLX 0.31, `mx.linalg.cholesky`, `mx.linalg.inv`, and `mx.random.multivariate_normal` are CPU-only. Current hot paths
  avoid them; if one is introduced, dispatch it on an explicit CPU stream rather than accidentally placing it in a
  Metal training graph.

## Refinement & Densification

Densification follows gsplat's `DefaultStrategy` with deliberate, experiment-backed deviations. Do not assume one
entry point's defaults apply to another; read its YAML and `EXPERIMENTS.md` before changing behavior.

- Standalone 2D fitting defaults to fresh optimizer state per refinement (`carry_optimizer_state: false`), an SGDR
  warm-restart means schedule (`means_mode: cos_restart`), child colors multiplied by 0.1, and background damping.
- Fixed-camera 3D fitting also defaults to fresh optimizer state, uses `revised_opacity` children, and currently has a
  constant means LR by default; warm restart remains available. `optim.num_epochs: 1` disables densification.
- COLMAP training has its own selective-Adam, scheduling, budget, and state-carry policy. Treat its resolved config as
  authoritative.
- Refinement signals are accumulated inside the compiled step as mean per-step gradient norms (screen-space for 3D).
  `grad_thr` is the capacity knob in free mode; budget mode replaces it with momentum count targets and top-k
  selection.

## Coding Style & Naming Conventions

Use four-space indentation, helpful type hints, `snake_case` for functions and variables, and `PascalCase` for classes.
Black and isort enforce a 120-column line length with Black-compatible imports; match that line length in new code.

## Testing Guidelines

Pytest discovers `tests/test_*.py`; name cases `test_<behavior>`. Add focused trainer tests and fused-versus-dense
image/gradient comparisons for rendering changes. For kernel or loss edits, run at least:

```bash
uv run pytest -q tests/test_kernels.py
uv run pytest -q
```

Regenerate intentional numeric goldens only with `uv run pytest tests/test_kernels.py --write`, then review the fixture
diff. There is no CI, so also run a small end-to-end `fit.py`, `fit3d.py`, or COLMAP smoke appropriate to the changed
path.

Before submission, run `uv run pre-commit run --all-files`. Hooks include Black and isort with line length 120.

## Commit & Pull Request Guidelines

Changes go through focused pull requests into `main`, not direct pushes. History favors brief, imperative, lowercase
subjects such as `fix 2dgs densification bug`. Explain motivation, configuration or dataset, and validation; link
issues and include metrics or renders for visual, numeric, or performance changes. Never commit datasets, `.venv`, or
`outputs/` artifacts.
