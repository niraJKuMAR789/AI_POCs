# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install ".[ui]"

FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 QUILL_DB_PATH=/app/data/claims.db
RUN useradd --create-home --uid 10001 quill
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY config ./config
COPY ui ./ui
# The synthetic warehouse is generated at build time: deterministic, ~6 MB, no data shipped in git.
RUN quill init-db && chown -R quill /app/data
USER quill
EXPOSE 8001
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1
CMD ["quill", "serve", "--host", "0.0.0.0", "--port", "8001"]
