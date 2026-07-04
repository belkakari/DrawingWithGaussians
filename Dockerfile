FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY drawingwithgaussians ./drawingwithgaussians
COPY configs ./configs
COPY fit.py fit3d.py ./
RUN uv sync --locked
