# Image for the API and the agent (one image, two Fly process groups). CPU torch for the encoder; no CUDA.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
COPY packages packages
COPY apps/api apps/api
COPY apps/web apps/web
COPY agents agents
COPY data data
RUN uv sync --frozen --no-dev --package dreaming-api --package bookkeeper \
 && uv pip install --python .venv --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "dreaming[student]"  # CPU torch, no CUDA wheels
ENV PATH=/app/.venv/bin:$PATH
# DREAMING_IMAGE_DIGEST is set by the deploy recipe to the image digest Fly reports, so env_sha names this exact image
