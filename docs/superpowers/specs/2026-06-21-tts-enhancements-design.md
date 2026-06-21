# TTS Summarizer Enhancements Design

## Goal

Update `tts-summarizer` so it sanitizes URLs before summarization, uses an OpenAI-compatible summarizer endpoint, serves HTTP via FastAPI with OpenAPI docs, and can interrupt active same-session playback.

## Approved Scope

- Replace every URL in text sent to the summarizer with the exact words `supplied URL`.
- Replace the direct `mlx_lm` summarizer backend with an OpenAI-compatible chat completions backend configurable from `config.toml`.
- Implement the web server with FastAPI and expose FastAPI's OpenAPI docs.
- Allow a `speak` request to interrupt current playback when the new request has the same `session_id` and caller/session key.
- Use `ffplay`, not `afplay`, for local playback.
- Document `ffplay`/FFmpeg as a README runtime requirement.
- Note that installer work is intentionally deferred; the future installer must install or validate `ffplay`.

## Out of Scope

- Building the installer.
- Adding provider presets beyond OpenAI-compatible chat completions.
- Rewriting the service to asyncio end-to-end.
- Changing the public speech request JSON shape.
- Adding a new TTS provider.

## Current Architecture

Current request flow:

1. CLI creates `SpeechRequest`.
2. Client posts to the daemon.
3. `server.py` uses `http.server` handlers for `/health`, `/v1/speak`, and `/shutdown`.
4. `TtsService` enqueues work and tracks same-session generations with `SessionManager`.
5. `Summarizer` loads `mlx_lm` directly through `MlxLmBackend`.
6. `SpeechGenerator` currently calls `mlx_audio.generate_audio` with `play=True`, so playback ownership is inside MLX audio and cannot reliably be interrupted mid-playback by `WorkToken`.

## Proposed Architecture

Use the existing service boundary, but make the IO edges boring and explicit:

- `config.py`: extend `SummarizerConfig` for OpenAI-compatible endpoint settings.
- `summarizer.py`: sanitize URLs before prompt construction and send chat completion requests through stdlib HTTP.
- `server.py`: keep `TtsService`, replace stdlib request handler with a FastAPI app factory and uvicorn runner.
- `speech.py` / `audio.py`: make generated audio playable by `AudioPlayer`, and make `AudioPlayer` own the `ffplay` subprocess so cancellation can terminate it.
- `README.md` and config comments: document FastAPI docs and `ffplay` requirement.

This is the shortest path that preserves existing CLI/request/session behavior while fixing playback interrupt at the source.

## Configuration

`summarizer.model` remains the model identifier, but it is sent to the configured OpenAI-compatible endpoint instead of loaded locally.

New summarizer fields:

```toml
[summarizer]
enabled = true
base_url = "http://127.0.0.1:1234/v1"
api_key = ""
model = "local-model"
word_threshold = 0
max_words = 40
temperature = 0.2
max_tokens = 180
```

Rules:

- `base_url` defaults to a local OpenAI-compatible endpoint.
- `api_key` is optional. If non-empty, requests include `Authorization: Bearer <api_key>`.
- Chat completions endpoint is `{base_url}/chat/completions`, with exactly one slash between parts.
- Request body uses OpenAI-compatible fields: `model`, `messages`, `temperature`, `max_tokens`.
- Response parser reads `choices[0].message.content`.
- On summarizer failure, preserve current behavior: log and return the original text.

## URL Sanitization

Add `replace_urls(text: str) -> str` in `summarizer.py`.

Requirements:

- Replace HTTP and HTTPS URLs with `supplied URL`.
- Apply before building the user prompt sent to the summarizer.
- Do not mutate `SpeechRequest.text`; only sanitize the summarizer input.
- If summarization is disabled or skipped by word threshold, return original text unchanged.
- Keep implementation small; no new URL parsing dependency.

## FastAPI Server

Replace `BaseHTTPRequestHandler` with FastAPI:

- `create_app(config: Config, service: TtsService | None = None) -> FastAPI`
- `GET /health` returns `service.health()`.
- `POST /v1/speak` accepts the same JSON shape as `SpeechRequest.from_json()` and returns current accepted response.
- `POST /shutdown` stops service and requests server shutdown where practical.
- OpenAPI docs are available at FastAPI defaults:
  - `/docs`
  - `/redoc`
  - `/openapi.json`

Implementation constraints:

- Keep CLI command behavior: `tts-summarizer serve --config <path>` starts daemon on configured host/port.
- Add `fastapi` and `uvicorn` dependencies.
- Prefer synchronous route handlers because the service queue is thread-based today.
- Keep current health/speak/shutdown response payloads where tests and clients depend on them.

## Same-Session Playback Interrupt

`SessionManager` already increments generation for same-session work. The missing piece is playback ownership.

Design:

1. `TtsService.handle()` begins a new token and enqueues work as today.
2. Existing active work sees `token.cancelled()` become true when the same session submits newer work.
3. `TtsService._process()` checks cancellation:
   - before summarization
   - before TTS generation
   - before playback
 - during playback via `AudioPlayer.play(chunks, token=token)`
4. `AudioPlayer` writes WAV chunks/files and launches `ffplay`.
5. While `ffplay` is running, poll periodically. If `token.cancelled()` becomes true, terminate `ffplay` and return.

Playback command:

```bash
ffplay -nodisp -autoexit -loglevel error <wav>
```

Audio config:

```toml
[audio]
backend = "auto" # auto | ffplay | file
```

`afplay` should stop being the documented default. Existing code paths mentioning `afplay` should be renamed or removed unless tests require temporary compatibility; clean cutover is preferred.

## README / Installer Note

README requirements must include:

- Python 3.11+
- `uv`
- Apple Silicon Mac for MLX TTS runtime
- FFmpeg with `ffplay` available on `PATH`

Add a short note:

- Installer work is deferred.
- Future installer must install or validate FFmpeg/`ffplay` before starting the daemon.

## Testing Strategy

Add or update unittest coverage only where behavior can break:

- URL replacement:
 - Replaces concrete HTTP and HTTPS URLs, such as `http://example.test/a` and `https://example.test/b`, with `supplied URL`.
  - Leaves non-URL text unchanged.
  - Summarizer sends sanitized text to backend but returns backend summary normally.
- OpenAI-compatible backend:
  - Posts expected JSON to `/chat/completions`.
  - Omits `Authorization` when `api_key` is empty.
 - Sends `Authorization: Bearer test-token` when configured.
  - Parses `choices[0].message.content`.
- FastAPI server:
  - `/openapi.json` exists.
  - `/health` returns current health payload.
  - `/v1/speak` accepts current request JSON and enqueues through `TtsService`.
  - Invalid speak payload returns HTTP 400-compatible error.
- Playback interrupt:
  - Fake or short-lived player observes stale token and stops when same-session generation advances.
  - `ffplay` command construction is tested without requiring real audio playback.
- Config:
  - New summarizer keys load from TOML.
  - `audio.backend = "ffplay"` is accepted; README/config comments no longer document `afplay`.

Verification commands for implementation:

```bash
rtk uv run python -m unittest tests.test_summarizer -v
rtk uv run python -m unittest tests.test_server -v
rtk uv run python -m unittest tests.test_speech_audio -v
rtk make test
```

If dependencies change, also run:

```bash
rtk uv lock
rtk uv build
```

## Risks

- `mlx_audio.generate_audio` with `play=True` may not expose chunks/files. If so, switch to non-playing generation mode or the smallest available API that returns audio data. Playback must be owned by `AudioPlayer` for interrupt to work.
- `POST /shutdown` under uvicorn is less direct than `ThreadingHTTPServer.shutdown()`. Keep the route for CLI compatibility; implementation may signal service stop and let process-level shutdown remain best-effort.
- FastAPI test support may need `httpx` through FastAPI/Starlette dependencies. Avoid adding extra test dependencies unless already required by FastAPI's test client.
