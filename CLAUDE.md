# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Experimental research project: fitting 2D and 3D Gaussians (Gaussian splatting) to images. MLX-only port of the original JAX/Flax version. Not production code — expect rough edges.

## Commands

Dependencies are managed by Poetry; run everything through the Poetry env:

```bash
poetry install                                                          # setup
poetry run python fit.py --config-name fit_to_image.yaml                # fit Gaussians to an image
poetry run python fit.py --config-name fit_to_image.yaml optim.num_steps=2000  # longer run
poetry run python fit3d.py --config-name fit_to_image_3d.yaml               # 3D gaussian splatting (fixed camera)
```

`fit.py` (2D) and `fit3d.py` (3D, gsplat image_fitting analog) are the entry points — Hydra apps reading YAML from `configs/`. Override params on the CLI, e.g. `poetry run python fit.py --config-name fit_to_image.yaml optim.num_epochs=50`. **Only the `pixel` loss path is supported.** The `diffusion_guidance` config will raise `NotImplementedError`; the diffusion path was intentionally not ported to MLX.

### Environment gotcha

The locked deps don't build on Python 3.13+ (no wheels for `numpy 1.23.5` / `scipy 1.12.0` / `ml-dtypes 0.2.0`), so this repo's Poetry env is pinned to a **Python 3.11** interpreter (a conda env `dwg` created via `conda create -n dwg -c conda-forge --override-channels python=3.11`). Poetry borrows that interpreter but installs into its **own** virtualenv.

Do **not** `conda activate dwg` to run the project — that bare env has no `pip`/`poetry`/project deps and will fail with `ModuleNotFoundError`. Use one of:
- `poetry run python fit.py ...` (from a shell where `poetry` is on PATH, e.g. conda `base`), or
- activate the Poetry venv directly: `source $(poetry env info --path)/bin/activate`.

## Before committing

Run `pre-commit run --all-files` (or `poetry run pre-commit run`). Hooks: **black** and **isort** (black profile), both with **line length 120**. Match that line length in new code.

## Workflow

- Changes go through PRs into `main`, not direct pushes.
- Dev/test target is Apple Silicon (Metal GPU). The MLX implementation runs fine up to ~1500 Gaussians locally. CUDA is not supported by MLX.

## Gotchas

- Code is functional MLX: heavy use of `mx.value_and_grad` and small per-step functions with the lazy evaluation + `mx.eval` pattern. Both trainers compile the *whole* step (loss/grad + Adam updates) with optimizer state threaded through `mx.compile(inputs=state, outputs=state)`; `fit.py` rebuilds the step per epoch because `split_n_prune` changes N and the optimizers are recreated (a retrace per epoch is negligible). Performance experiments are logged in `EXPERIMENTS.md`.
- Rasterization runs as fused Metal kernels (`rendering2d_fused.py`, `rendering3d_fused.py`; `mx.fast.metal_kernel` + `mx.custom_function` VJPs, gsplat kernel structure). The dense implementations (`rendering2d.rasterize`, `rendering3d.rasterize3d_dense`) are kept as reference for validating kernel changes — compare images/grads against them (and, for accuracy questions, against fp64 numpy; the fused paths are *more* accurate than the dense ones, see EXPERIMENTS.md). The 3D backward uses atomic adds, so its grads are non-deterministic at the ulp level.
- L's diagonal is parameterized in **log-space** (`log_diag`) — the actual diagonal is `exp(log_diag)`. This guarantees positivity by construction (gsplat convention). The off-diagonal `L[1, 0]` is stored in raw space.
- MLX 0.31 has no boolean indexing / `mx.nonzero` / `mx.compress`, so `split_n_prune` (2D) and `split_n_prune_3d` are implemented eagerly in numpy (per-epoch ops, overhead negligible).
- Densification (both paths) follows gsplat's DefaultStrategy with deliberate deviations that won the A/B in EXPERIMENTS.md: **fresh optimizer state each refine** (`carry_optimizer_state: false` — carried Adam moments collapse variances), **SGDR warm-restart means LR** (`means_mode: cos_restart`), children colors x0.1 + background damp (2D), `revised_opacity` children (3D). The signal is the mean per-step (screen-space, for 3D) means-grad norm accumulated inside the compiled step. `grad_thr` is the capacity knob; `num_epochs: 1` disables densification in fit3d.
- MLX `mx.linalg.cholesky` and `mx.linalg.inv` are CPU-only — `init_gaussians` passes an explicit CPU stream. `mx.random.multivariate_normal` is also CPU-only for the same reason.
- Use `mlx_stable_exp` (in `drawingwithgaussians/utils.py`) instead of raw `mx.exp` where overflow is a risk.
- Kernel regression tests: `poetry run python tests/test_kernels.py` (fused-vs-dense tolerance checks, fp64 anchors, golden npz fixtures in `tests/fixtures/`; `--write` regenerates goldens after an *intentional* numeric change). Run them after touching any kernel or loss. There is no CI; also verify end-to-end by running `fit.py` on a small config.