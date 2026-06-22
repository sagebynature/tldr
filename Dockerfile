FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY hooks ./hooks
COPY config.remote.example.toml config.apple-local.example.toml ./

RUN pip install --no-cache-dir .

EXPOSE 9200

CMD ["tts-summarizer", "serve", "--config", "/config/config.toml"]
