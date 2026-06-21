# tts-summarizer

Harness-neutral local TTS summarizer daemon for macOS/Apple Silicon. It accepts normalized speech requests, keeps MLX summarizer/TTS models warm, and interrupts stale speech from the same caller session.

## Requirements

- Python 3.11+
- `uv`
- Apple Silicon Mac for the real MLX/Metal runtime

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
make run        # uv run python -m tts_summarizer serve --config config.example.toml
```

Use another config:

```bash
make run CONFIG=/path/to/config.toml
```

## Run the daemon

```bash
uv run tts-summarizer serve --config config.example.toml
```

The daemon binds to `127.0.0.1`, writes its state under the configured `state_dir`, and loads MLX models lazily on first use.

## Send a request

```bash
echo '{"caller":"manual","session_id":"demo","text":"Codex finished."}' \
  | uv run tts-summarizer speak --config config.example.toml
```

A later request with the same `caller` and `session_id` interrupts stale speech for that session.

## Check and stop

```bash
uv run tts-summarizer health --config config.example.toml
uv run tts-summarizer stop --config config.example.toml
```

## Config lookup order

1. `--config /path/to/config.toml`
2. `./config.toml`
3. `~/.config/tts-summarizer/config.toml`
4. built-in defaults

See `config.example.toml` for all model, prompt, session, server, and audio settings.

Model-specific `mlx-audio` generation arguments belong under `[tts.generate_kwargs]`:

```toml
[tts]
model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

[tts.generate_kwargs]
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
