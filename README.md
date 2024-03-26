# Experiments with 2D gaussians

## Set up

```bash
pip install poetry
poetry install
```

## Fit 2D gaussians to an image

```bash
python fit.py --config-name fit_to_image.yaml
```

## References
Based on [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [fmb-plus](https://leonidk.com/fmb-plus/), [GaussianImage](https://arxiv.org/abs/2403.08551), unoptimized and slow
