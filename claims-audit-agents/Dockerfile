# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv venv /opt/venv && VIRTUAL_ENV=/opt/venv uv pip install ".[ui]"

FROM python:3.11-slim
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 AEGIS_STATE_DIR=/state
RUN useradd --create-home --uid 10001 aegis && mkdir /state && chown aegis /state
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY config ./config
COPY data ./data
COPY ui ./ui
USER aegis
VOLUME ["/state"]
EXPOSE 8002
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1
CMD ["aegis", "serve", "--host", "0.0.0.0", "--port", "8002"]
