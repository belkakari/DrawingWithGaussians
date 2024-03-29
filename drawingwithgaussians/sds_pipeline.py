import jax
import jax.numpy as jnp
import numpy as np
from diffusers import FlaxStableDiffusionPipeline
from PIL import Image

seed = 0

key = jax.random.PRNGKey(seed)

dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)

prompt = "A cinematic film still of Morgan Freeman starring as Jimi Hendrix, portrait, 40mm lens, shallow depth of field, close up, split lighting, cinematic"
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape

images = pipeline(prompt_ids, params, key, jit=True)[0]

Image.fromarray(np.array(images[0] * 255).astype(np.uint8)).save("./image.jpg")
