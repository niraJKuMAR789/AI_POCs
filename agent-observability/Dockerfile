# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install ".[ui]"

FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 ARGUS_DB_PATH=/data/argus.db
RUN useradd --create-home --uid 10001 argus && mkdir /data && chown argus /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY config ./config
COPY ui ./ui
USER argus
VOLUME ["/data"]
EXPOSE 8003
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')" || exit 1
CMD ["argus", "serve", "--host", "0.0.0.0", "--port", "8003"]
