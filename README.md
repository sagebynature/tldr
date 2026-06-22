# tts-summarizer

`tts-summarizer` is a small HTTP daemon you bind to any AI application. It turns long AI responses into shorter, speech-friendly text, then returns TTS audio WAV bytes.

## What it is

- Local HTTP service for AI apps that want spoken responses.
- Summarizes long assistant output into text that fits spoken playback.
- Generates TTS audio for client-side playback.
- Uses configurable summarization and TTS models, local or remote.
- Works with local MLX models or OpenAI-compatible remote endpoints.

## Requirements

- Python 3.11+
- `uv`
- Remote OpenAI-compatible summarizer and TTS endpoints, or an Apple Silicon Mac for local MLX profiles.
- Docker, only for the Docker quick start.

## Quick start: local CLI install

```bash
uv tool install git+https://github.com/sagebynature/tts-summarizer
tts-summarizer init-config --profile remote
tts-summarizer serve
```

The generated config lives at `~/.config/tts-summarizer/config.toml`. Use `--force` to replace an existing generated config:

```bash
tts-summarizer init-config --profile remote --force
```

The remote profile expects your summarizer and TTS servers to expose OpenAI-compatible APIs before you start the daemon.

## Quick start: Docker

```bash
git clone https://github.com/sagebynature/tts-summarizer
cd tts-summarizer
docker compose up
```

Docker runs the HTTP daemon only. By default Compose mounts `config.docker.example.toml`, which binds the daemon to `0.0.0.0:9200` and points model backends at `http://host.docker.internal:9000/v1` so containers can reach model servers running on the Docker host.

Use another Docker config file when your model endpoints live elsewhere:

```bash
TTS_SUMMARIZER_CONFIG=./config.toml docker compose up
```

## Config lookup profiles

Config lookup order:

1. `--config /path/to/config.toml`
2. `./config.toml`
3. `~/.config/tts-summarizer/config.toml`
4. built-in defaults

Generate a remote-backend config:

```bash
tts-summarizer init-config --profile remote
```

Generate an Apple Silicon local MLX config:

```bash
tts-summarizer init-config --profile apple-local
```

Profiles are selected by `default_profile` under each section:

```toml
[summarizer]
default_profile = "remote-qwen25"

[tts]
default_profile = "remote-kokoro"
```

Switch individual profiles by changing only the relevant `default_profile` value; the daemon can keep remote and local profile definitions in one config.

## Apple local MLX notes

Apple local profiles use `mlx_audio` in process intended Apple Silicon Macs. Install optional local audio dependencies and the default Kokoro TTS profile dependencies using local MLX TTS:

```bash
uv tool install 'tts-summarizer[mlx,kokoro] @ git+https://github.com/sagebynature/tts-summarizer'
```

The Apple local example keeps remote profiles too, so you can switch individual profiles without changing the daemon.

## Run daemon

```bash
tts-summarizer serve --config config.toml
```

FastAPI OpenAPI docs are available while the daemon is running:

- `http://127.0.0.1:9200/docs`
- `http://127.0.0.1:9200/redoc`
- `http://127.0.0.1:9200/openapi.json`

## Send request

`/v1/speak` returns WAV bytes. Playback example:

```bash
curl -sS -X POST http://127.0.0.1:9200/v1/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"This is a long answer to summarize before speaking."}' \
  --output reply.wav
ffplay -nodisp -autoexit reply.wav
```

CLI request helper:

```bash
tts-summarizer speak --session_id demo "Codex finished."
```

Use `--summarize false` to send text directly to TTS. `--session_id` interrupts any previous playback for that session.

## Check and stop

```bash
tts-summarizer health --config config.toml
tts-summarizer stop --config config.toml
```

## Development commands

```bash
uv sync --dev
make build # uv build
make test # uv run python -m unittest discover -s tests -v
make typecheck # uvx ty check src tests
make check # typecheck, test, build
make run # uv run python -m tts_summarizer serve --config config.toml
```

Use another config during development:

```bash
make run CONFIG=/path/to/config.toml
```

## Configure summarization models

Summarizer profiles use OpenAI-compatible chat completion endpoints. Point them at a local server, an oMLX deployment, or another compatible remote endpoint.

```toml
[summarizer]
default_profile = "qwen25"

[summarizer.profiles.qwen25]
base_url = "http://127.0.0.1:9000/v1"
api_key = "omlx"
model = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
max_words = 40
```

## Configure TTS models

TTS profiles can run local `mlx_audio` models in process or call a remote OpenAI/MLX-Audio-compatible endpoint.

Local example:

```toml
[tts]
default_profile = "kokoro"

[tts.profiles.kokoro]
backend = "mlx"
model = "mlx-community/Kokoro-82M-bf16"

[tts.profiles.kokoro.generate_kwargs]
voice = "af_heart"
lang_code = "a"
```

Remote example:

```toml
[tts]
default_profile = "remote-kokoro"

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

Remote TTS posts to `{base_url}/audio/speech` and streams returned WAV bytes unchanged.

## Logging

The package ships `src/tts_summarizer/logging.conf`, using `colorlog.ColoredFormatter` like `korean-name-generator`. Use a custom logging config from TOML:

```toml
[logging]
config_file = "/path/to/logging.conf"
```
