FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH="/app/src"

RUN uv sync --locked --no-dev

EXPOSE 9200

CMD ["python", "-m", "tldr", "serve", "--config", "/config/config.toml"]
