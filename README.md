# tts-summarizer

`tts-summarizer` is a small HTTP daemon you can bind to any AI application. It turns long AI responses into shorter, speech-friendly text, then returns TTS audio as WAV bytes.

## What this is

- A local HTTP service for AI apps that want spoken responses.
- Summarizes long assistant output into text that fits spoken playback.
- Generates TTS audio for client-side playback.
- Lets you configure your own summarization and TTS models, local or remote.
- Works with local MLX models and OpenAI-compatible endpoints.

## Requirements

- Python 3.11+
- `uv`
- Apple Silicon Mac for local MLX TTS runtime

## Install for local development

```bash
uv sync --dev
```

## Common commands

```bash
make build # uv build
make test # uv run python -m unittest discover -s tests -v
make typecheck # uvx ty check src tests
make check # typecheck, test, build
make run # uv run python -m tts_summarizer serve --config config.toml
```

Use another config:

```bash
make run CONFIG=/path/to/config.toml
```

## Run the daemon

```bash
uv run tts-summarizer serve --config config.toml
```

The daemon binds `127.0.0.1`, writes state under configured `state_dir`, and loads local MLX models lazily on first use.

FastAPI OpenAPI docs are available while the daemon is running:

- `http://127.0.0.1:9200/docs`
- `http://127.0.0.1:9200/redoc`
- `http://127.0.0.1:9200/openapi.json`

## Send a request

`/v1/speak` returns WAV bytes. Playback belongs to the client.

```bash
tts-summarizer speak --session_id demo "Codex finished."
```

Use `--summarize false` to send text directly to TTS. `--session_id` interrupts any previous playback for that session.


```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -H 'X-TTS-Caller: manual' \
  -H 'X-TTS-Session-Id: demo' \
  -d '{"text":"Codex finished.","summarize":true}' \
  http://127.0.0.1:9200/v1/speak > speech.wav
```

Use `"summarize": false` to send text directly to TTS.

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

See `config.toml` for server settings, prompts, and model profiles.

## Logging

The package ships `src/tts_summarizer/logging.conf`, using `colorlog.ColoredFormatter` like `korean-name-generator`.

Use custom logging config TOML:

```toml
[logging]
config_file = "/path/to/logging.conf"
```
