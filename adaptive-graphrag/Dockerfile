# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app

FROM base AS builder
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install ".[ui]"

FROM python:3.11-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 ATLAS_DATA_DIR=/data
RUN useradd --create-home --uid 10001 atlas && mkdir -p /data && chown atlas /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=atlas data ./data
COPY --chown=atlas evals ./evals
COPY --chown=atlas ui ./ui
USER atlas
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["atlas", "serve", "--host", "0.0.0.0", "--port", "8000"]
