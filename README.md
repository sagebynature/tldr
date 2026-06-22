# tts-summarizer

Harness-neutral local TTS summarizer daemon for macOS/Apple Silicon. It accepts HTTP speech requests, keeps MLX summarizer/TTS models warm, and returns WAV bytes for client-side playback.

## Requirements

- Python 3.11+
- `uv`
- Apple Silicon Mac for MLX TTS runtime

## Install for local development

```bash
uv sync --dev
```

## Common commands

```bash
make build      # uv build
make test       # uv run python -m unittest discover -s tests -v
make typecheck  # uvx ty check src tests
make check      # typecheck, test, then build
make run        # uv run python -m tts_summarizer serve --config config.toml
```

Use another config:

```bash
make run CONFIG=/path/to/config.toml
```

## Run the daemon

```bash
uv run tts-summarizer serve --config config.toml
```

The daemon binds to `127.0.0.1`, writes its state under the configured `state_dir`, and loads MLX models lazily on first use.

FastAPI OpenAPI docs are available while the daemon is running:

- `http://127.0.0.1:9200/docs`
- `http://127.0.0.1:9200/redoc`
- `http://127.0.0.1:9200/openapi.json`

## Send a request

`/v1/speak` returns WAV bytes. Playback belongs to the client.

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -H 'X-TTS-Caller: manual' \
  -H 'X-TTS-Session-Id: demo' \
  -d '{"text":"Codex finished.","summarize":true}' \
  http://127.0.0.1:9200/v1/speak > speech.wav
```

Use `"summarize": false` to send text directly to TTS.

## Remote TTS backend

TTS profiles can call an OpenAI/MLX-Audio-compatible server instead loading `mlx_audio` in process:

```toml
[tts.profiles.remote-kokoro]
backend = "remote"
base_url = "http://127.0.0.1:9100/v1"
api_key = "omlx"
model = "mlx-community/Kokoro-82M-bf16"
stream = true
sample_rate = 24000

[tts.profiles.remote-kokoro.generate_kwargs]
voice = "af_heart"
lang_code = "a"
response_format = "wav"
```

The daemon posts to `{base_url}/audio/speech` and streams the returned WAV bytes unchanged. Set `[tts] default_profile = "remote-kokoro"` to use it by default.

## Check and stop

```bash
uv run tts-summarizer health --config config.toml
uv run tts-summarizer stop --config config.toml
```

## Config lookup order

1. `--config /path/to/config.toml`
2. `./config.toml`
3. `~/.config/tts-summarizer/config.toml`
4. built-in defaults

See `config.toml` for model, prompt, and server settings.

Model-specific `mlx-audio` generation arguments belong under `[tts.profiles.<name>.generate_kwargs]`:

```toml
[tts]
default_profile = "qwen"

[tts.profiles.qwen]
model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

[tts.profiles.qwen.generate_kwargs]
voice = "Chelsie"
lang_code = "English"
```

## Logging

The package ships `src/tts_summarizer/logging.conf`, using `colorlog.ColoredFormatter` like `korean-name-generator`.

Use a custom logging config from TOML:

```toml
[logging]
config_file = "/path/to/logging.conf"
```

## Versioning and releases

Version is stored in `pyproject.toml` at `project.version`.

Releases use Python Semantic Release with Conventional Commits:

- `fix:` bumps patch.
- `feat:` bumps minor.
- breaking changes bump major.

On pushes to `main`, `.github/workflows/release.yml` runs `make check`, creates a GitHub release/tag when semantic-release finds releasable commits, builds with `uv build`, and publishes to PyPI through Trusted Publishing.

Required GitHub setup:

- `SEMANTIC_RELEASE_TOKEN` secret with permission to push release commits/tags.
- PyPI Trusted Publisher configured for this repository and the `pypi` environment.
