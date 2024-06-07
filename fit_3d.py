import jax
import jax.numpy as jnp
import jaxsplat
import numpy as np
import optax
from einops import rearrange
from jax.scipy.spatial.transform import Rotation as R
from PIL import Image

from drawingwithgaussians.threed.losses import diffusion_guidance, pixel_loss
from drawingwithgaussians.twod.sds_pipeline import FlaxStableDiffusionImg2ImgPipeline

key = jax.random.key(0)
num_gaussians = 3000
means3d = jax.random.uniform(key, (num_gaussians, 3))
scales = jax.random.uniform(key, (num_gaussians, 3))
quats = jax.random.uniform(key, (num_gaussians, 4))
quats = quats / jnp.linalg.norm(quats, axis=1, keepdims=True)
colors = jax.random.uniform(key, (num_gaussians, 3))
opacities = jax.random.uniform(key, (num_gaussians, 1))
# viewmat = jnp.identity(4)
background = jax.random.uniform(key, (3,))


viewmat = jnp.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ],
    dtype=jnp.float32,
)

W = 512
H = 512
img_shape = (W, H)
f = (W / 2, H / 2)
c = (W / 2, H / 2)
glob_scale = 1.0
clip_thresh = 0.1
block_size = 8

lr = 1e-2
optim = optax.adam(lr)
optim_state = optim.init((means3d, scales, quats, colors, opacities))


params = (means3d, scales, quats, colors, opacities)
target = Image.open("/home/ubuntu/DrawingWithGaussians/inputs/eye.jpeg")
target = jnp.array(target.resize((H, W)), dtype=jnp.float32)[:, :, :3] / 255

dtype = jnp.bfloat16
pipeline, pipeline_params = FlaxStableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)

loss = diffusion_guidance
loss_fn = jax.value_and_grad(
    loss,
    argnums=[
        0,
    ],
    has_aux=True,
)
for i in range(1000):
    viewmat = jnp.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=jnp.float32,
    )
    if i % 20 == 0:
        target_image = None
    else:
        target_image = jnp.copy(diffusion_image)

    viewmat.at[:3, :3].set(R.from_euler("xyz", jax.random.uniform(key, (3,)) / 10, degrees=True).as_matrix())
    (loss, (renderred_gaussians, diffusion_image)), grads = loss_fn(
        params,
        viewmat,
        background,
        f,
        c,
        img_shape,
        glob_scale,
        clip_thresh,
        block_size,
        diffusion_shape=[H, W, 3],
        num_steps=30,
        strength=0.8,
        pipeline=pipeline,
        params=pipeline_params,
        dtype=dtype,
        cfg_scale=6.0,
        prompt="a yellow vase",
        key=key,
        target_image=target_image,
    )
    updates, optim_state = optim.update(grads[0], optim_state)
    params = optax.apply_updates(params, updates)
    if i % 50 == 0:
        print(f"step {i} loss {loss:.5f}")
        image = (np.clip(np.array(jnp.array(renderred_gaussians.block_until_ready())), 0, 1) * 255).astype(np.uint8)
        diff_image = (np.clip(np.array(jnp.array(diffusion_image.block_until_ready())), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(image).save(f"/home/ubuntu/DrawingWithGaussians/outputs/test_{i}.jpg")
        Image.fromarray(diff_image).save(f"/home/ubuntu/DrawingWithGaussians/outputs/diff_test_{i}.jpg")


viewmat = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0.2],
        [0, 0, 1, 2],
        [0, 0, 0, 1],
    ],
    dtype=jnp.float32,
)

img = jaxsplat.render(
    means3d,  # jax.Array (N, 3)
    scales,  # jax.Array (N, 3)
    quats,  # jax.Array (N, 4) normalized
    colors,  # jax.Array (N, 3)
    opacities,  # jax.Array (N, 1)
    viewmat=viewmat,  # jax.Array (4, 4)
    background=background,  # jax.Array (3,)
    img_shape=img_shape,  # tuple[int, int] = (H, W)
    f=f,  # tuple[float, float] = (fx, fy)
    c=c,  # tuple[int, int] = (cx, cy)
    glob_scale=glob_scale,  # float
    clip_thresh=clip_thresh,  # float
    block_size=block_size,  # int <= 16
)
image = (np.clip(np.array(jnp.array(img.block_until_ready())), 0, 1) * 255).astype(np.uint8)
Image.fromarray(image).save(f"/home/ubuntu/DrawingWithGaussians/outputs/final.jpg")
