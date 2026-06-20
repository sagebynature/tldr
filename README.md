# tts-summarizer

Harness-neutral local TTS summarizer daemon for macOS/Apple Silicon. It accepts normalized speech requests, keeps MLX summarizer/TTS models warm, and interrupts stale speech from the same caller session.

## Requirements

- Python 3.11+
- Apple Silicon Mac for MLX/Metal runtime
- `uv` for the default `make typecheck` target (`uvx ty ...`)

## Install for local development

```bash
python -m pip install -e .
```

## Common commands

```bash
make build      # byte-compile src/
make test       # run unittest suite
make typecheck  # run ty check src tests
make run        # start daemon with config.example.toml
```

Use another config:

```bash
make run CONFIG=/path/to/config.toml
```

## Run the daemon

```bash
tts-summarizer serve --config config.example.toml
```

The daemon binds to `127.0.0.1`, writes its state under the configured `state_dir`, and loads MLX models lazily on first use.

## Send a request

```bash
echo '{"caller":"manual","session_id":"demo","text":"Codex finished."}' \
  | tts-summarizer speak --config config.example.toml
```

A later request with the same `caller` and `session_id` interrupts stale speech for that session.

## Check and stop

```bash
tts-summarizer health --config config.example.toml
tts-summarizer stop --config config.example.toml
```

## Config lookup order

1. `--config /path/to/config.toml`
2. `./config.toml`
3. `~/.config/tts-summarizer/config.toml`
4. built-in defaults

See `config.example.toml` for all model, prompt, session, server, and audio settings.
