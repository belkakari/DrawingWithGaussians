FROM ghcr.io/nvidia/jax:jax

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 python3.10-venv -y
RUN apt remove cmake -y && pip install cmake --upgrade && pip install "pybind11[global]" poetry
