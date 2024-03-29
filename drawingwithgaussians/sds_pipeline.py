import jax
import jax.numpy as jnp
from jax import pmap
import numpy as np
from flax.jax_utils import replicate
from diffusers import FlaxStableDiffusionPipeline
from PIL import Image
from flax.training.common_utils import shard

seed = 0

key = jax.random.PRNGKey(seed)

dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)

prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)

# parameters
p_params = replicate(params)

# arrays
prompt_ids = shard(prompt_ids)
prompt_ids.shape

rng = jax.random.split(key, jax.device_count())
images = pipeline(prompt_ids, p_params, rng, jit=True)[0]

Image.fromarray(np.array(images[0, 0] * 255).astype(np.uint8)).save("./image.jpg")
