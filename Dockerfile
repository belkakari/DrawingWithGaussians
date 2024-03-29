FROM ghcr.io/nvidia/jax:jax

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install numpy pillow matplotlib opencv-python einops optax hydra-core dm-pix flax diffusers transformers ftfy
