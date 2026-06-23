<div align="center">

# TL;DR

**Short agent replies, spoken cleanly**

</div>

---

## What is "TL;DR"?

Not every agent response is speech friendly. Responses are often too long, and
they frequently include details that are useful on screen but awkward out loud:
URLs, file links, markdown, code fences, long lists, and other visual-only
context.

TL;DR rewrites that text into a short spoken version, sends it to a
configured TTS backend, and streams the audio back as soon as bytes are
available.

## How It Works

1. Send agent text to TL;DR.
2. The summarizer creates a short version of what should be said.
3. The rewritten text is sent to TTS.
4. WAV audio streams back to the client and can start playing immediately.

The daemon exposes an HTTP API and an `tldr speak` helper. Hook scripts are
included for common agent harnesses so completions can be spoken automatically.

## Requirements

- Python 3.11+
- `uv`
- `curl` and `ffplay` for the playback examples
- A summarizer backend with an OpenAI-compatible chat completions API
- A TTS backend with an OpenAI/MLX-Audio-compatible `/audio/speech` API, or local
  MLX audio support on Apple Silicon
- Docker, only if you want to run the server in a container

## Install

Install the CLI from GitHub:

```bash
uv tool install git+https://github.com/sagebynature/tts-summarizer
```

From a local checkout:

```bash
uv tool install .
```

For local Apple Silicon MLX audio profiles, install the optional extras:

```bash
uv tool install 'tldr[mlx,kokoro] @ git+https://github.com/sagebynature/tts-summarizer'
```

The old `tts-summarizer` command is still installed as a compatibility alias.

## Configure

Generate a user config:

```bash
tldr init-config --profile remote
```

The generated config is written to:

```text
~/.config/tldr/config.toml
```

Use `--force` to replace an existing generated config:

```bash
tldr init-config --profile remote --force
```

For Apple Silicon local MLX defaults:

```bash
tldr init-config --profile apple-local
```

Config lookup order:

1. `--config /path/to/config.toml`
2. `./config.toml`
3. `~/.config/tldr/config.toml`
4. `~/.config/tts-summarizer/config.toml`
5. Built-in defaults

The checked-in `config.toml` points both summarization and remote TTS at
`http://127.0.0.1:9000/v1`. Update the `base_url`, `api_key`, `model`, voice, and
profile names for your backend.

## Run The Server

Start the HTTP daemon:

```bash
tldr serve --config config.toml
```

From a development checkout:

```bash
uv run python -m tldr serve --config config.toml
```

The default server listens on:

```text
http://127.0.0.1:9200
```

OpenAPI docs are available while the server is running:

- `http://127.0.0.1:9200/docs`
- `http://127.0.0.1:9200/redoc`
- `http://127.0.0.1:9200/openapi.json`

Check or stop the daemon:

```bash
tldr health --config config.toml
tldr stop --config config.toml
```

## Speak With curl

`POST /v1/speak` returns WAV bytes. This example streams the response directly
into `ffplay`:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"text":"The implementation is complete. I updated the README, verified the CLI options, and left the server configuration unchanged.","summarize":true}' \
  http://127.0.0.1:9200/v1/speak \
  | ffplay -nodisp -autoexit -loglevel error -i pipe:0
```

Save the audio instead:

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -d '{"text":"Speak this after summarizing it.","summarize":true}' \
  http://127.0.0.1:9200/v1/speak \
  --output reply.wav
```

Set `"summarize":false` to send text directly to TTS.

## Speak With The CLI

The CLI helper posts to the server and pipes the streamed WAV response to
`ffplay`:

```bash
tldr speak --session_id demo "Codex finished the task and the tests passed."
```

Use a specific server:

```bash
tldr speak --server 127.0.0.1 --port 9200 "Read this response out loud."
```

Skip summarization:

```bash
tldr speak --summarize false "Speak this exact text."
```

`--session_id` interrupts any previous playback for the same session before
starting the new one.

## Install A Harness Hook

Install a hook for your agent harness:

```bash
tldr install --harness codex
```

Supported harnesses:

- `codex`
- `claude`
- `omp`
- `pi`
- `hermes`

Examples:

```bash
tldr install --harness claude
tldr install --harness hermes
```

The installer copies the matching hook into the harness config directory and
updates the harness settings where needed. Hooks call the local daemon, so start
`tldr serve` before expecting spoken completions.

## Run With Docker

Build and run the daemon with Compose:

```bash
docker compose up --build
```

Compose publishes the service on host port `9200` and mounts:

```text
./config.docker.toml:/config/config.toml:ro
```

The Docker config binds the daemon to `0.0.0.0:9200` and points model backends at:

```text
http://host.docker.internal:9000/v1
```

Use a different config file:

```bash
TLDR_CONFIG=./config.toml docker compose up --build
```

The container runs the server only. Your summarizer and TTS model servers must be
reachable from inside the container.

## Development

```bash
make run        # uv run python -m tldr serve --config config.toml
make test       # typecheck, then unittest
make typecheck  # uv run ty check src tests
make check      # lint, format, typecheck, test
```

Use another config while developing:

```bash
make run CONFIG=/path/to/config.toml
```

## Logging

The default logging config lives at `src/tldr/logging.conf`. Use a
custom logging config from TOML:

```toml
[logging]
config_file = "/path/to/logging.conf"
```
