# TTS Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sanitize URLs before summarization, switch summarization to an OpenAI-compatible endpoint, serve with FastAPI/OpenAPI, and interrupt same-session playback through `ffplay`.

**Architecture:** Keep the existing `TtsService` queue/session boundary. Replace the HTTP edge with FastAPI, replace the summarizer backend with a stdlib HTTP OpenAI-compatible backend, and move playback ownership to `AudioPlayer` so stale `WorkToken`s can terminate `ffplay`.

**Tech Stack:** Python 3.11, stdlib `unittest`, stdlib `urllib.request`, FastAPI, uvicorn, MLX audio, FFmpeg `ffplay`.

## Global Constraints

- Follow approved spec `docs/superpowers/specs/2026-06-21-tts-enhancements-design.md`.
- Do not build installer; only document deferred installer requirement.
- Do not add provider presets; use OpenAI-compatible `/chat/completions` only.
- Do not change public speech request JSON shape.
- Use `ffplay`, not `afplay`.
- Shell commands must be prefixed with `rtk`.
- Keep code boring: no new abstraction unless this plan names it.

---

## File Structure

- Modify `pyproject.toml`: add runtime dependencies `fastapi` and `uvicorn`; add dev dependency `httpx` for FastAPI `TestClient` if tests need it.
- Modify `config.toml`: add `[summarizer]` endpoint keys; change audio backend comment to `ffplay`.
- Modify `src/tts_summarizer/config.py`: add `base_url` and `api_key` to `SummarizerConfig`.
- Modify `src/tts_summarizer/summarizer.py`: add `replace_urls`; replace `MlxLmBackend` with `OpenAICompatibleBackend`.
- Modify `src/tts_summarizer/server.py`: add `create_app`; replace `http.server` runner with uvicorn/FastAPI while preserving `TtsService`.
- Modify `src/tts_summarizer/speech.py`: ensure generated audio is returned, not played inside MLX.
- Modify `src/tts_summarizer/audio.py`: use `ffplay` and terminate on token cancellation.
- Modify `README.md`: document FastAPI docs and FFmpeg/`ffplay` requirement plus deferred installer note.
- Modify tests: `tests/test_config.py`, `tests/test_summarizer.py`, `tests/test_server.py`, `tests/test_speech_audio.py`.

---

## Task 1: Config and Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.toml`
- Modify: `src/tts_summarizer/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `SummarizerConfig.base_url: str`, `SummarizerConfig.api_key: str`.
- Produces: accepted `AudioConfig.backend` values documented as `auto | ffplay | file`.
- Consumes: existing `_merge_dataclass` unknown-key validation.

- [ ] **Step 1: Write failing config tests**

Add to `tests/test_config.py`:

```python
    def test_summarizer_endpoint_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                '\n'.join(
                    [
                        '[summarizer]',
                        'base_url = "http://127.0.0.1:1234/v1"',
                        'api_key = "test-token"',
                        'model = "local-model"',
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

        self.assertEqual(cfg.summarizer.base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(cfg.summarizer.api_key, "test-token")
        self.assertEqual(cfg.summarizer.model, "local-model")

    def test_audio_ffplay_backend_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[audio]\nbackend = "ffplay"\n', encoding="utf-8")

            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

        self.assertEqual(cfg.audio.backend, "ffplay")
```

- [ ] **Step 2: Run test verify fails**

Run:

```bash
rtk uv run python -m unittest tests.test_config.ConfigTests.test_summarizer_endpoint_config_loads tests.test_config.ConfigTests.test_audio_ffplay_backend_config_loads -v
```

Expected: first test fails with unknown config key or missing `base_url`/`api_key`; second may pass before comments are updated.

- [ ] **Step 3: Add dependencies and config fields**

In `pyproject.toml`, change dependencies to include:

```toml
  "fastapi",
  "uvicorn",
```

In `[dependency-groups]`, ensure dev includes `httpx`:

```toml
dev = ["ty==0.0.23", "ruff>=0.5", "httpx"]
```

In `src/tts_summarizer/config.py`, change `SummarizerConfig` to:

```python
@dataclass(frozen=True)
class SummarizerConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = ""
    model: str = "local-model"
    word_threshold: int = 0
    max_words: int = 40
    temperature: float = 0.2
    max_tokens: int = 180
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE
```

In `config.toml`, update `[summarizer]`:

```toml
[summarizer]
enabled = true
base_url = "http://127.0.0.1:1234/v1"
api_key = ""
model = "local-model"
```

In `config.toml`, update `[audio]` comment:

```toml
backend = "auto" # auto | ffplay | file
```

- [ ] **Step 4: Run config tests verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_config -v
```

Expected: PASS.

- [ ] **Step 5: Refresh lockfile**

Run:

```bash
rtk uv lock
```

Expected: lockfile updates successfully with FastAPI, uvicorn, and httpx.

- [ ] **Step 6: Commit**

Run:

```bash
rtk git add pyproject.toml uv.lock config.toml src/tts_summarizer/config.py tests/test_config.py && rtk git commit -m "feat: configure openai summarizer endpoint"
```

---

## Task 2: URL Sanitization and OpenAI-Compatible Summarizer

**Files:**
- Modify: `src/tts_summarizer/summarizer.py`
- Modify: `tests/test_summarizer.py`

**Interfaces:**
- Produces: `replace_urls(text: str) -> str`.
- Produces: `OpenAICompatibleBackend.generate(messages: list[dict[str, str]], config: SummarizerConfig) -> str`.
- Consumes: `SummarizerConfig.base_url`, `api_key`, `model`, `temperature`, `max_tokens`.

- [ ] **Step 1: Update summarizer imports in tests**

Change the import in `tests/test_summarizer.py` from:

```python
from tts_summarizer.summarizer import MlxLmBackend, Summarizer, count_words
```

to:

```python
from tts_summarizer.summarizer import (
    OpenAICompatibleBackend,
    Summarizer,
    count_words,
    replace_urls,
)
```

- [ ] **Step 2: Write failing URL tests**

Add to `tests/test_summarizer.py`:

```python
    def test_replace_urls_replaces_http_and_https(self):
        self.assertEqual(
            replace_urls("Read http://example.test/a and https://example.test/b?x=1."),
            "Read supplied URL and supplied URL",
        )

    def test_summarizer_sends_sanitized_text_to_backend(self):
        backend = FakeBackend()
        config = SummarizerConfig(
            word_threshold=0,
            user_prompt_template="Say {max_words}: {text}",
        )
        summarizer = Summarizer(config, backend=backend)

        self.assertEqual(summarizer.summarize("open https://example.test/path"), "short result")
        self.assertEqual(backend.prompt, "Say 40: open supplied URL")
```

- [ ] **Step 3: Replace old MLX backend test with OpenAI backend tests**

Delete `test_mlx_backend_uses_sampler_not_temperature_kwarg` from `tests/test_summarizer.py`.

Add:

```python
    def test_openai_backend_posts_chat_completion_without_auth(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"short summary"}}]}'

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return Response()

        backend = OpenAICompatibleBackend(urlopen=fake_urlopen)
        result = backend.generate(
            [{"role": "user", "content": "hello"}],
            SummarizerConfig(
                base_url="http://localhost:1234/v1/",
                api_key="",
                model="local-model",
                temperature=0.3,
                max_tokens=50,
            ),
        )

        self.assertEqual(result, "short summary")
        request, timeout = calls[0]
        self.assertEqual(request.full_url, "http://localhost:1234/v1/chat/completions")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(timeout, 30)

    def test_openai_backend_posts_auth_when_configured(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"short summary"}}]}'

        def fake_urlopen(request, timeout):
            calls.append(request)
            return Response()

        backend = OpenAICompatibleBackend(urlopen=fake_urlopen)
        backend.generate(
            [{"role": "user", "content": "hello"}],
            SummarizerConfig(api_key="test-token"),
        )

        self.assertEqual(calls[0].headers["Authorization"], "Bearer test-token")
```

- [ ] **Step 4: Run tests verify fail**

Run:

```bash
rtk uv run python -m unittest tests.test_summarizer -v
```

Expected: FAIL because `replace_urls` and `OpenAICompatibleBackend` are missing and old MLX backend name no longer exists.

- [ ] **Step 5: Implement URL replacement and OpenAI backend**

In `src/tts_summarizer/summarizer.py`, replace MLX-specific imports/backend with:

```python
from typing import Callable, Protocol
from urllib.request import Request, urlopen
import json
import logging
import re

from .config import SummarizerConfig
```

Add:

```python
URL_PATTERN = re.compile(r"https?://[^\s<>)\]}]+")


def replace_urls(text: str) -> str:
    return URL_PATTERN.sub("supplied URL", text)
```

Add backend:

```python
class OpenAICompatibleBackend:
    def __init__(self, urlopen: Callable[..., object] = urlopen, timeout: float = 30):
        self.urlopen = urlopen
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], config: SummarizerConfig) -> str:
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(
            {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = Request(url, data=body, headers=headers, method="POST")
        with self.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])
```

Change `Summarizer.__init__` default backend:

```python
self.backend = backend or OpenAICompatibleBackend()
```

Change `Summarizer.summarize` prompt text:

```python
sanitized = replace_urls(text)
messages = [
    {"role": "system", "content": self.config.system_prompt},
    {
        "role": "user",
        "content": self.config.user_prompt_template.format(
            max_words=self.config.max_words,
            text=sanitized,
        ),
    },
]
```

Keep `clean_summary(summary, text, self.config)` so fallback returns original text.

- [ ] **Step 6: Run summarizer tests verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_summarizer -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add src/tts_summarizer/summarizer.py tests/test_summarizer.py && rtk git commit -m "feat: use openai compatible summarizer"
```

---

## Task 3: FastAPI Server

**Files:**
- Modify: `src/tts_summarizer/server.py`
- Modify: `tests/test_server.py`
- Modify: `src/tts_summarizer/cli.py` only if imports need adjustment after server change.

**Interfaces:**
- Produces: `create_app(config: Config, service: TtsService | None = None) -> FastAPI`.
- Preserves: `run_server(config: Config) -> int`.
- Preserves endpoints: `GET /health`, `POST /v1/speak`, `POST /shutdown`.

- [ ] **Step 1: Add FastAPI route tests**

Add imports in `tests/test_server.py`:

```python
from fastapi.testclient import TestClient
from tts_summarizer.server import TtsService, create_app
```

Add tests:

```python
    def test_fastapi_openapi_schema_exists(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/speak", response.json()["paths"])

    def test_fastapi_health_route(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_fastapi_speak_route_accepts_current_json(self):
        player = FakePlayer()
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=player)
        client = TestClient(create_app(Config(), service=service))

        response = client.post(
            "/v1/speak",
            json={"text": "hello", "caller": "c", "session_id": "s"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "accepted")
        self.assertTrue(service.process_pending())

    def test_fastapi_speak_route_rejects_invalid_json(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.post("/v1/speak", json={"text": ""})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
```

- [ ] **Step 2: Run tests verify fail**

Run:

```bash
rtk uv run python -m unittest tests.test_server -v
```

Expected: FAIL because `create_app` is missing.

- [ ] **Step 3: Implement FastAPI app factory**

In `src/tts_summarizer/server.py`, remove `BaseHTTPRequestHandler`, `ThreadingHTTPServer`, and manual JSON response handler.

Add imports:

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
```

Keep `TtsService` mostly unchanged.

Add:

```python
def create_app(config: Config, service: TtsService | None = None) -> FastAPI:
    service = service or TtsService(config)
    app = FastAPI(title="tts-summarizer")
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, object]:
        return service.health()

    @app.post("/v1/speak")
    def speak(payload: dict[str, object]):
        try:
            request = SpeechRequest.from_json(payload)
        except RequestError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return service.handle(request)

    @app.post("/shutdown")
    def shutdown() -> dict[str, object]:
        service.stop()
        return {"status": "shutting_down"}

    return app
```


Change `run_server`:

```python
def run_server(config: Config) -> int:
    service = TtsService(config)
    app = create_app(config, service)
    write_state(config.server.state_dir, config.server.host, config.server.port)
    worker = threading.Thread(target=service.run, daemon=True)
    worker.start()
    uvicorn.run(app, host=config.server.host, port=config.server.port, log_config=None)
    service.stop()
    return 0
```

Preserve existing `write_state` behavior from old `run_server`; if old code used bound port `0`, use uvicorn `Config`/`Server` only if tests require exact actual port state.

- [ ] **Step 4: Run server tests verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_server -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
rtk git add src/tts_summarizer/server.py src/tts_summarizer/cli.py tests/test_server.py && rtk git commit -m "feat: serve api with fastapi"
```

---

## Task 4: ffplay Playback and Same-Session Interrupt

**Files:**
- Modify: `src/tts_summarizer/speech.py`
- Modify: `src/tts_summarizer/audio.py`
- Modify: `src/tts_summarizer/server.py`
- Modify: `tests/test_speech_audio.py`
- Modify: `tests/test_server.py`

**Interfaces:**
- Preserves: `SpeechGenerator.generate(text: str) -> list[AudioChunk]`.
- Produces: `AudioPlayer.play(chunks: Iterable[AudioChunk], token: WorkToken | None = None) -> None` terminates playback when token is stale.
- Consumes: `WorkToken.cancelled()`.

- [ ] **Step 1: Write ffplay command test**

Add to `tests/test_speech_audio.py`:

```python
    def test_audio_player_uses_ffplay_for_auto_backend(self):
        calls = []

        class Proc:
            def __init__(self, command):
                calls.append(command)
                self.polls = 0

            def poll(self):
                self.polls += 1
                return 0

            def terminate(self):
                calls.append(["terminated"])

        with tempfile.TemporaryDirectory() as tmp:
            player = AudioPlayer(AudioConfig(backend="auto", output_dir=tmp, save=False))
            with unittest.mock.patch("tts_summarizer.audio.subprocess.Popen", Proc):
                player.play([AudioChunk(samples=[0.0], sample_rate=8000)])

        self.assertEqual(calls[0][0:4], ["ffplay", "-nodisp", "-autoexit", "-loglevel"])
        self.assertEqual(calls[0][4], "error")
```

Add `import unittest.mock` if missing.

- [ ] **Step 2: Write cancellation test**

Add to `tests/test_speech_audio.py`:

```python
    def test_audio_player_terminates_ffplay_when_token_cancelled(self):
        from tts_summarizer.session import SessionManager
        from tts_summarizer.request import SpeechRequest
        from tts_summarizer.config import SessionConfig

        events = []

        class Proc:
            def __init__(self, command):
                self.polls = 0

            def poll(self):
                self.polls += 1
                if self.polls == 1:
                    manager.begin(SpeechRequest(text="new", caller="c", session_id="s"))
                    return None
                return None

            def terminate(self):
                events.append("terminated")

        manager = SessionManager(SessionConfig(interrupt_same_session=True))
        token = manager.begin(SpeechRequest(text="old", caller="c", session_id="s"))

        with tempfile.TemporaryDirectory() as tmp:
            player = AudioPlayer(AudioConfig(backend="ffplay", output_dir=tmp, save=False))
            with unittest.mock.patch("tts_summarizer.audio.subprocess.Popen", Proc):
                player.play([AudioChunk(samples=[0.0], sample_rate=8000)], token=token)

        self.assertEqual(events, ["terminated"])
```

- [ ] **Step 3: Write service interrupt test**

Add to `tests/test_server.py`:

```python
    def test_same_session_request_cancels_active_playback_token(self):
        cancelled = []

        class BlockingPlayer:
            def play(self, chunks, token=None):
                service.handle(SpeechRequest(text="new", caller="c", session_id="s"))
                cancelled.append(token.cancelled())

        service = TtsService(
            Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=BlockingPlayer()
        )
        service.handle(SpeechRequest(text="old", caller="c", session_id="s"))

        self.assertTrue(service.process_pending())
        self.assertEqual(cancelled, [True])
```

- [ ] **Step 4: Run tests verify fail**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio tests.test_server -v
```

Expected: FAIL while `AudioPlayer` still uses `afplay` or service does not pass tokens through playback.

- [ ] **Step 5: Update speech generation to return audio**

In `src/tts_summarizer/speech.py`, make `MlxAudioBackend.generate` return chunks instead of playing:

```python
    def generate(self, text: str, config: TtsConfig) -> list[AudioChunk]:
        model = self._load(config.model)
        kwargs = dict(config.generate_kwargs)
        logger.info(
            "calling tts generate model=%s kwargs=%s text_chars=%s",
            config.model,
            sorted(kwargs),
            len(text),
        )
        results = model.generate(text=text, **kwargs)
        return [
            AudioChunk(
                samples=getattr(result, "audio", result),
                sample_rate=getattr(result, "sample_rate", config.sample_rate),
            )
            for result in results
        ]
```

Keep `_load` returning the loaded model. Remove unused `generate_audio` import if no longer used.

- [ ] **Step 6: Update AudioPlayer to use ffplay and token cancellation**

In `src/tts_summarizer/audio.py`, change the playback branch:

```python
            if self.config.backend in {"auto", "ffplay"} and not self.config.save:
                proc = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
                )
                while proc.poll() is None:
                    if token is not None and token.cancelled():
                        proc.terminate()
                        return
                    time.sleep(0.05)
```

Remove `afplay` references.

- [ ] **Step 7: Ensure service passes token to playback**

In `src/tts_summarizer/server.py`, ensure `_process` does:

```python
            if token.cancelled():
                logger.info("speech request cancelled before tts session=%s", request.session_key())
                return
            logger.info("generating speech session=%s", request.session_key())
            chunks = self.speech.generate(text)
            if token.cancelled():
                logger.info("speech request cancelled before playback session=%s", request.session_key())
                return
            self.player.play(chunks, token=token)
            logger.info("speech playback complete session=%s", request.session_key())
```

- [ ] **Step 8: Run audio/server tests verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio tests.test_server -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
rtk git add src/tts_summarizer/speech.py src/tts_summarizer/audio.py src/tts_summarizer/server.py tests/test_speech_audio.py tests/test_server.py && rtk git commit -m "feat: interrupt ffplay playback by session"
```

---

## Task 5: README and Final Verification

**Files:**
- Modify: `README.md`
- Optional modify: `CHANGELOG.md` only if project convention requires it for this branch.

**Interfaces:**
- Documents: FastAPI OpenAPI docs endpoints.
- Documents: FFmpeg/`ffplay` runtime requirement.
- Documents: installer deferred note.

- [ ] **Step 1: Update README requirements**

In `README.md`, update Requirements to include:

```markdown
- Python 3.11+
- `uv`
- Apple Silicon Mac for the MLX TTS runtime
- FFmpeg with `ffplay` available on `PATH`
```

- [ ] **Step 2: Update README daemon/docs section**

Add after daemon run instructions:

```markdown
FastAPI OpenAPI docs are available while the daemon is running:

- `http://127.0.0.1:9200/docs`
- `http://127.0.0.1:9200/redoc`
- `http://127.0.0.1:9200/openapi.json`
```

- [ ] **Step 3: Add installer note**

Add near Requirements:

```markdown
Installer work is deferred. The future installer must install or validate FFmpeg/`ffplay` before starting the daemon.
```

- [ ] **Step 4: Run README-adjacent checks**

Run:

```bash
rtk uv run python -m unittest tests.test_config tests.test_server tests.test_speech_audio tests.test_summarizer -v
```

Expected: PASS.

- [ ] **Step 5: Run full test/typecheck gate**

Run:

```bash
rtk make test
```

Expected: PASS, including `ty check` and unittest discovery.

- [ ] **Step 6: Run build because dependencies changed**

Run:

```bash
rtk uv build
```

Expected: PASS.

- [ ] **Step 7: Commit final docs**

Run:

```bash
rtk git add README.md CHANGELOG.md && rtk git commit -m "docs: note ffplay and api docs"
```

If `CHANGELOG.md` was not changed, run:

```bash
rtk git add README.md && rtk git commit -m "docs: note ffplay and api docs"
```

---

## Self-Review Checklist

- Spec coverage:
  - URL replacement: Task 2.
  - OpenAI-compatible summarizer endpoint: Tasks 1 and 2.
  - FastAPI/OpenAPI docs: Tasks 1, 3, and 5.
  - Same-session playback interrupt: Task 4.
  - `ffplay` requirement and installer note: Tasks 4 and 5.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps remain.
- Type consistency:
  - `replace_urls`, `OpenAICompatibleBackend`, `create_app`, and config fields are defined before later tasks use them.
  - Existing `SpeechGenerator.generate(text)` API is preserved for `TtsService`.
  - Existing CLI `run_server(config)` API is preserved for `cli.py`.
