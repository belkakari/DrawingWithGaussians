# Experiments with 2D gaussians

## Set up

```bash
git clone https://github.com/belkakari/DrawingWithGaussians.git
cd DrawingWithGaussians
pip install poetry
poetry install
```

## Fit 2D gaussians to an image

```bash
python fit.py --config-name fit_to_image.yaml
```

![An example of fitting an image](./static/eye_fitting.gif)

## ToDO
- [ ] Add alternative alpha-composing with occlusions (prune gaussians based on opacity, currently prunning based on color norm)
- [ ] Move boilerplate to separate classes
- [ ] Add SDS with SD
- [ ] Add basic 3D version

## References
Based on [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [fmb-plus](https://leonidk.com/fmb-plus/), [GaussianImage](https://arxiv.org/abs/2403.08551), works ok on macbook m1 up to ~300 gaussians
