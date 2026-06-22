# Serve Speak Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `serve` expose `/v1/speak` as a bytes-only TTS endpoint with request identity in headers, optional summarization, and no server playback/interruption machinery.

**Architecture:** Keep the existing FastAPI/uvicorn server. `/v1/speak` is a synchronous `def` route because summarization, TTS, and WAV encoding are blocking work; FastAPI runs sync routes in its threadpool. The route reads identity from headers, builds a `SpeechRequest`, generates chunks, encodes one WAV byte response, and returns `audio/wav`. Delete the worker queue, session cancellation, server playback, and `TtsService` lifecycle class; plain functions and FastAPI closure/state are enough.

**Tech Stack:** FastAPI, Pydantic, uvicorn, Python stdlib `io` and `wave`; existing `SpeechRequest`, `Summarizer`, `SpeechGenerator`, `AudioChunk`; `unittest`; `ruff`; `ty`.

## Global Constraints

- `/v1/speak` must never accept or branch on a `playback` request property.
- `/v1/speak` must always return generated audio bytes with `Content-Type: audio/wav` for valid requests.
- Request JSON payload contains `text`, optional `metadata`, optional `summarize` only.
- `caller` and `session_id` are request headers, not payload properties.
- Use headers `X-TTS-Caller` and `X-TTS-Session-Id`.
- Missing `X-TTS-Caller` defaults to `default`; missing `X-TTS-Session-Id` uses existing `fallback_session_id()`.
- `summarize` is optional and defaults to `True`.
- `summarize: false` skips the summarizer and sends original request text directly to TTS.
- Drop speech interruption: remove cancellation tokens, same-session generation tracking, server playback queueing, and interruption config/tests.
- Do not implement client playback in this plan.
- Keep the diff small. Delete dead architecture instead of preserving shims.

---

## File Structure

- Modify `src/tts_summarizer/request.py`
  - Remove `event` from `SpeechRequest`.
  - Add `summarize: bool = True`.
  - Stop parsing `caller` and `session_id` from JSON.
  - Extend `from_json(data, caller=None, session_id=None)` for header-fed identity.
  - Delete `from_cli()` after CLI `speak` is removed.
  - Keep `session_key()` for logging only.
- Modify `src/tts_summarizer/audio.py`
  - Add `chunks_to_wav_bytes(chunks: Iterable[AudioChunk]) -> bytes`.
  - Delete `AudioPlayer`, token handling, subprocess playback, and unused file-playback helpers if no references remain.
- Modify `src/tts_summarizer/server.py`
  - Delete `TtsService`, queue, worker loop, `SessionManager`, `WorkToken`, `AudioPlayer`, and `service.stop()` use.
  - Add `synthesize_speech(request, summarizer, speech) -> bytes`.
  - Change `create_app(config, summarizer=None, speech=None)` to close over injectable dependencies for tests.
  - Read `X-TTS-Caller` and `X-TTS-Session-Id` in the FastAPI `/v1/speak` route.
  - Return `fastapi.Response(content=body, media_type="audio/wav")`.
  - Keep `/shutdown` by setting `app.state.server.should_exit = True`; no service stop hook.
- Modify `src/tts_summarizer/config.py`
  - Remove `SessionConfig`, `Config.session`, and `[session]` from allowed sections.
- Modify `src/tts_summarizer/state.py`
  - Remove `config.session` from `config_fingerprint()`.
- Delete `src/tts_summarizer/session.py`.
- Modify `src/tts_summarizer/cli.py`
  - Remove the `speak` subcommand and request-posting path.
  - Remove unused `SpeechRequest`/`RequestError` imports.
- Modify `config.toml`
  - Remove the `[session]` block.
- Modify `README.md`
  - Remove stale CLI `speak`, same-session interruption, and session config references.
  - Document POST `/v1/speak` with JSON body and headers.
- Modify tests:
  - `tests/test_request.py`: request parsing contract.
  - `tests/test_speech_audio.py`: WAV byte encoder.
  - `tests/test_server.py`: FastAPI endpoint, header identity, summarization toggle, OpenAPI payload shape, no worker lifecycle.
  - `tests/test_config.py`: no session defaults and `[session]` rejection if explicit config still contains it.
  - `tests/test_state_client.py`: unchanged behavior after fingerprint edit.
  - `tests/test_cli_commands.py`: `speak` is not a command.
  - Delete `tests/test_session.py`.

---

## Task 1: Request Contract

**Files:**
- Modify: `src/tts_summarizer/request.py`
- Test: `tests/test_request.py`

**Interfaces:**
- Produces: `SpeechRequest.from_json(data, caller=None, session_id=None) -> SpeechRequest`.
- Produces: `SpeechRequest.summarize: bool` defaulting to `True`.
- Removes: `SpeechRequest.event`, payload-derived `caller`, payload-derived `session_id`, and `SpeechRequest.from_cli()`.

- [ ] **Step 1: Write failing request tests**

Replace stale identity/CLI tests in `tests/test_request.py` with:

```python
import unittest

from tts_summarizer.request import RequestError, SpeechRequest


class RequestTests(unittest.TestCase):
    def test_speech_request_uses_headers_for_identity(self):
        request = SpeechRequest.from_json(
            {"text": "hello", "caller": "body", "session_id": "body-session"},
            caller="header",
            session_id="header-session",
        )

        self.assertEqual(request.caller, "header")
        self.assertEqual(request.session_id, "header-session")
        self.assertEqual(request.session_key(), "header:header-session")
        self.assertNotIn("caller", request.to_json())
        self.assertNotIn("session_id", request.to_json())

    def test_speech_request_defaults_to_summarize_true(self):
        request = SpeechRequest.from_json({"text": "hello"}, caller="c", session_id="s")

        self.assertIs(request.summarize, True)
        self.assertNotIn("playback", request.to_json())
        self.assertNotIn("event", request.to_json())

    def test_speech_request_accepts_summarize_false(self):
        request = SpeechRequest.from_json(
            {"text": "hello", "summarize": False}, caller="c", session_id="s"
        )

        self.assertIs(request.summarize, False)
        self.assertEqual(request.to_json()["summarize"], False)

    def test_missing_text_fails(self):
        with self.assertRaises(RequestError):
            SpeechRequest.from_json({"metadata": {"x": 1}})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run request tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_request
```

Expected before implementation: failures because `from_json()` does not accept header identity and still serializes old fields.

- [ ] **Step 3: Implement minimal request contract**

Update `SpeechRequest` in `src/tts_summarizer/request.py` to:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


class RequestError(ValueError):
    pass


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    session_id: str
    caller: str = "default"
    metadata: dict[str, object] = field(default_factory=dict)
    summarize: bool = True

    @classmethod
    def from_json(
        cls,
        data: dict[str, object],
        caller: str | None = None,
        session_id: str | None = None,
    ) -> "SpeechRequest":
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RequestError("normalized request requires non-empty text")
        metadata = data.get("metadata")
        summarize = data.get("summarize")
        clean_metadata: dict[str, object] = {}
        if isinstance(metadata, dict):
            clean_metadata = {str(key): value for key, value in metadata.items()}
        return cls(
            text=text,
            caller=caller or "default",
            session_id=session_id or fallback_session_id(),
            metadata=clean_metadata,
            summarize=summarize if isinstance(summarize, bool) else True,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "summarize": self.summarize,
        }

    def session_key(self) -> str:
        return f"{self.caller}:{self.session_id}"


def fallback_session_id() -> str:
    cwd = Path.cwd()
    ppid = os.getppid()
    return f"{cwd}:{ppid}"
```

- [ ] **Step 4: Run request tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_request
```

Expected: all request tests pass.

---

## Task 2: WAV Byte Encoder

**Files:**
- Modify: `src/tts_summarizer/audio.py`
- Test: `tests/test_speech_audio.py`

**Interfaces:**
- Produces: `chunks_to_wav_bytes(chunks: Iterable[AudioChunk]) -> bytes`.
- Removes: `AudioPlayer`, `WorkToken` import, and token-aware playback plumbing.

- [ ] **Step 1: Write failing WAV byte tests**

In `tests/test_speech_audio.py`, keep sample conversion tests that still apply and replace playback tests with:

```python
import io
import unittest
import wave

from tts_summarizer.audio import chunks_to_wav_bytes
from tts_summarizer.speech import AudioChunk


class SpeechAudioTests(unittest.TestCase):
    def test_chunks_to_wav_bytes_returns_readable_wav(self):
        body = chunks_to_wav_bytes([AudioChunk(samples=[0.0, 0.5, -0.5], sample_rate=8000)])

        self.assertTrue(body.startswith(b"RIFF"))
        with wave.open(io.BytesIO(body), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 8000)
            self.assertEqual(wav.getnframes(), 3)

    def test_chunks_to_wav_bytes_appends_multiple_chunks(self):
        body = chunks_to_wav_bytes(
            [
                AudioChunk(samples=[0.0], sample_rate=8000),
                AudioChunk(samples=[1.0], sample_rate=8000),
            ]
        )

        with wave.open(io.BytesIO(body), "rb") as wav:
            self.assertEqual(wav.getnframes(), 2)

    def test_chunks_to_wav_bytes_rejects_mixed_sample_rates(self):
        with self.assertRaises(ValueError):
            chunks_to_wav_bytes(
                [
                    AudioChunk(samples=[0.0], sample_rate=8000),
                    AudioChunk(samples=[0.0], sample_rate=16000),
                ]
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run audio tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio
```

Expected before implementation: import failure for `chunks_to_wav_bytes` or failures from old playback tests.

- [ ] **Step 3: Implement WAV byte encoder and delete playback code**

Update `src/tts_summarizer/audio.py` to keep only audio conversion helpers and add:

```python
from __future__ import annotations

from collections.abc import Iterable as IterableABC
from typing import Any, Iterable, Protocol, cast
import io
import math
import wave

from .speech import AudioChunk


class SupportsToList(Protocol):
    def tolist(self) -> Any: ...


def chunks_to_wav_bytes(chunks: Iterable[AudioChunk]) -> bytes:
    buffer = io.BytesIO()
    sample_rate: int | None = None
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        for chunk in chunks:
            if sample_rate is None:
                sample_rate = chunk.sample_rate
                wav.setframerate(sample_rate)
            elif chunk.sample_rate != sample_rate:
                raise ValueError("all audio chunks must use the same sample_rate")
            samples = _to_float_list(chunk.samples)
            wav.writeframes(b"".join(_to_i16(sample) for sample in samples))
        if sample_rate is None:
            wav.setframerate(8000)
    return buffer.getvalue()


def _to_float_list(samples: object) -> list[float]:
    if hasattr(samples, "tolist"):
        raw = cast(SupportsToList, samples).tolist()
    else:
        raw = samples
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        rows = cast(list[list[Any]], raw)
        return [item for row in rows for item in row]
    if isinstance(raw, IterableABC) and not isinstance(raw, (str, bytes)):
        return list(cast(Iterable[Any], raw))
    return [raw]


def _to_i16(sample: float) -> bytes:
    value = max(-1.0, min(1.0, sample if math.isfinite(sample) else 0.0))
    return int(value * 32767).to_bytes(2, "little", signed=True)
```

- [ ] **Step 4: Run audio tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio
```

Expected: all audio tests pass.

---

## Task 3: FastAPI Bytes-Only Speak Endpoint

**Files:**
- Modify: `src/tts_summarizer/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `SpeechRequest.from_json(payload, caller=..., session_id=...)` from Task 1.
- Consumes: `chunks_to_wav_bytes(chunks)` from Task 2.
- Produces: `synthesize_speech(request, summarizer, speech) -> bytes`.
- Produces: `create_app(config, summarizer=None, speech=None) -> FastAPI`.
- Removes: `TtsService`.

- [ ] **Step 1: Write failing endpoint tests**

In `tests/test_server.py`, remove `TtsService`, `FakePlayer`, `process_pending()`, and playback/interruption tests. Use dependency injection through `create_app()`:

```python
from fastapi.testclient import TestClient

from tts_summarizer.config import Config
from tts_summarizer.server import create_app
from tts_summarizer.speech import AudioChunk


class FakeSummarizer:
    def summarize(self, text):
        return f"summary: {text}"


class CapturingSpeech:
    def __init__(self):
        self.text = ""

    def generate(self, text):
        self.text = text
        return [AudioChunk(samples=[0.0], sample_rate=8000)]
```

Add these tests:

```python
def test_fastapi_speak_route_returns_wav_and_uses_identity_headers(self):
    speech = CapturingSpeech()
    client = TestClient(create_app(Config(), summarizer=FakeSummarizer(), speech=speech))

    with self.assertLogs("tts_summarizer.server", level="INFO") as logs:
        response = client.post(
            "/v1/speak",
            json={"text": "hello"},
            headers={"X-TTS-Caller": "header", "X-TTS-Session-Id": "header-session"},
        )

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.headers["content-type"], "audio/wav")
    self.assertTrue(response.content.startswith(b"RIFF"))
    self.assertEqual(speech.text, "summary: hello")
    self.assertIn("session=header:header-session", "\n".join(logs.output))


def test_fastapi_speak_route_summarizes_by_default(self):
    speech = CapturingSpeech()
    client = TestClient(create_app(Config(), summarizer=FakeSummarizer(), speech=speech))

    response = client.post("/v1/speak", json={"text": "hello"})

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.headers["content-type"], "audio/wav")
    self.assertEqual(speech.text, "summary: hello")


def test_fastapi_speak_route_can_skip_summarizer_for_tts_testing(self):
    speech = CapturingSpeech()
    client = TestClient(create_app(Config(), summarizer=FakeSummarizer(), speech=speech))

    response = client.post("/v1/speak", json={"text": "hello", "summarize": False})

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.headers["content-type"], "audio/wav")
    self.assertEqual(speech.text, "hello")


def test_fastapi_speak_route_rejects_payload_identity_and_playback(self):
    client = TestClient(create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech()))

    for key in ("caller", "session_id", "event", "playback"):
        response = client.post("/v1/speak", json={"text": "hello", key: "bad"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
```

Update the OpenAPI test to assert:

```python
properties = components["SpeakRequestBody"]["properties"]
self.assertIn("text", properties)
self.assertIn("metadata", properties)
self.assertIn("summarize", properties)
self.assertNotIn("caller", properties)
self.assertNotIn("session_id", properties)
self.assertNotIn("event", properties)
self.assertNotIn("playback", properties)
```

- [ ] **Step 2: Run server tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_server
```

Expected before implementation: failures because `/v1/speak` returns JSON accepted status, payload identity is still accepted, or `TtsService` still owns playback.

- [ ] **Step 3: Implement FastAPI sync route and synthesis function**

In `src/tts_summarizer/server.py`:

```python
from __future__ import annotations

from dataclasses import replace
import logging
import os
import socket

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
import uvicorn

from .audio import chunks_to_wav_bytes
from .config import Config
from .request import RequestError, SpeechRequest
from .speech import SpeechGenerator
from .state import write_state
from .summarizer import Summarizer


logger = logging.getLogger(__name__)


class SpeakRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    metadata: dict[str, object] | None = None
    summarize: bool = True


def synthesize_speech(request: SpeechRequest, summarizer, speech) -> bytes:
    logger.info("incoming text session=%s text=%r", request.session_key(), request.text)
    if request.summarize:
        logger.info("summarizing speech request session=%s", request.session_key())
        text = summarizer.summarize(request.text)
        logger.info(
            "summary ready session=%s input_chars=%s output_chars=%s changed=%s",
            request.session_key(),
            len(request.text),
            len(text),
            text != request.text,
        )
        logger.info("summarized text session=%s text=%r", request.session_key(), text)
    else:
        text = request.text
        logger.info("summary skipped session=%s", request.session_key())
    logger.info("generating speech session=%s", request.session_key())
    return chunks_to_wav_bytes(speech.generate(text))
```

Update `create_app()` speak route to:

```python
def create_app(config: Config, summarizer=None, speech=None) -> FastAPI:
    summarizer = summarizer or Summarizer(config.summarizer)
    speech = speech or SpeechGenerator(config.tts)
    app = FastAPI(title="tts-summarizer")
    app.state.summarizer = summarizer
    app.state.speech = speech

    @app.exception_handler(RequestValidationError)
    def validation_error(_request: object, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "invalid request body"})

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "pid": os.getpid()}

    @app.post("/v1/speak")
    def speak(payload: SpeakRequestBody, http_request: Request) -> Response:
        try:
            speech_request = SpeechRequest.from_json(
                payload.model_dump(exclude_none=True),
                caller=http_request.headers.get("X-TTS-Caller"),
                session_id=http_request.headers.get("X-TTS-Session-Id"),
            )
        except RequestError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        body = synthesize_speech(speech_request, summarizer, speech)
        return Response(content=body, media_type="audio/wav")
```

Keep the existing `/v1/summarize` behavior, but replace `service.summarizer` with the captured `summarizer`:

```python
backend = getattr(summarizer, "backend", None)
summary = Summarizer(summarizer_config, backend=backend).summarize(text)
```

Update `/shutdown` to remove `service.stop()`:

```python
@app.post("/shutdown")
def shutdown() -> dict[str, object]:
    server = getattr(app.state, "server", None)
    if server is not None:
        server.should_exit = True
    return {"status": "shutting_down"}
```

Update `run_server()` to remove worker thread lifecycle:

```python
def run_server(config: Config) -> int:
    app = create_app(config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((config.server.host, config.server.port))
    sock.listen()
    host, port = sock.getsockname()[:2]
    write_state(config, str(host), int(port), os.getpid())
    server_config = uvicorn.Config(app, host=config.server.host, port=int(port), log_config=None)
    server = uvicorn.Server(server_config)
    app.state.server = server
    server.run(sockets=[sock])
    return 0
```

- [ ] **Step 4: Update run-server lifecycle test**

Replace the worker-thread test with one that asserts no stop/join lifecycle exists:

```python
def test_run_server_starts_uvicorn_without_worker_thread(self):
    events = []

    class FakeSocket:
        def setsockopt(self, *args):
            events.append(("setsockopt", args))

        def bind(self, address):
            events.append(("bind", address))

        def listen(self):
            events.append("listen")

        def getsockname(self):
            return ("127.0.0.1", 0)

    class FakeState:
        pass

    class FakeApp:
        def __init__(self):
            self.state = FakeState()

    class FakeServer:
        def __init__(self, config):
            self.config = config

        def run(self, sockets):
            events.append(("server", sockets))

    with (
        unittest.mock.patch("tts_summarizer.server.create_app", return_value=FakeApp()),
        unittest.mock.patch("tts_summarizer.server.socket.socket", return_value=FakeSocket()),
        unittest.mock.patch("tts_summarizer.server.write_state"),
        unittest.mock.patch("tts_summarizer.server.uvicorn.Server", FakeServer),
    ):
        self.assertEqual(run_server(Config()), 0)

    self.assertIn("listen", events)
    self.assertEqual(events[-1][0], "server")
```

- [ ] **Step 5: Run server tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_server
```

Expected: all server tests pass.

---

## Task 4: Remove Session Config and State References

**Files:**
- Modify: `src/tts_summarizer/config.py`
- Modify: `src/tts_summarizer/state.py`
- Modify: `config.toml`
- Delete: `src/tts_summarizer/session.py`
- Delete: `tests/test_session.py`
- Test: `tests/test_config.py`
- Test: `tests/test_state_client.py`

**Interfaces:**
- Removes: `SessionConfig`, `Config.session`, `interrupt_same_session`, `max_queue_per_session`, `cross_session_policy`, `SessionManager`, `WorkToken`.
- Keeps: `load_config()` rejecting unknown sections.

- [ ] **Step 1: Search references before deleting**

Use harness search for these patterns in `src tests README.md config.toml`:

```text
SessionManager
WorkToken
interrupt_same_session
max_queue_per_session
cross_session_policy
Config().session
config.session
[session]
```

Expected before deletion: references only in session architecture, state fingerprint, tests, README, and repo config.

- [ ] **Step 2: Write/update config tests**

In `tests/test_config.py`, ensure defaults no longer expose session and explicit old session config fails:

```python
def test_defaults_load_without_session_config(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = load_config(None, cwd=root / "cwd", home=root / "home")
    self.assertFalse(hasattr(cfg, "session"))


def test_session_config_section_is_rejected(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.toml"
        path.write_text('[session]\ninterrupt_same_session = true\n', encoding="utf-8")

        with self.assertRaises(ConfigError):
            load_config(str(path), cwd=Path(tmp), home=Path(tmp))
```

- [ ] **Step 3: Remove session config from code and repo config**

In `src/tts_summarizer/config.py`:

```python
# delete SessionConfig

@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    source: Path | None = None
```

Update `_apply()`:

```python
def _apply(raw: dict[str, Any], source: Path | None) -> Config:
    cfg = Config(source=source)
    allowed = {"server", "summarizer", "tts", "audio", "logging"}
    unknown_sections = sorted(set(raw) - allowed)
    if unknown_sections:
        raise ConfigError(f"unknown sections: {', '.join(unknown_sections)}")
    return Config(
        server=_merge_dataclass(cfg.server, raw.get("server", {})),
        summarizer=_merge_dataclass(cfg.summarizer, raw.get("summarizer", {})),
        tts=_merge_dataclass(cfg.tts, raw.get("tts", {})),
        audio=_merge_dataclass(cfg.audio, raw.get("audio", {})),
        logging=_merge_dataclass(cfg.logging, raw.get("logging", {})),
        source=source,
    )
```

In `src/tts_summarizer/state.py`, update the fingerprint basis:

```python
basis = repr((config.server, config.summarizer, config.tts, config.audio)).encode("utf-8")
```

Remove these lines from `config.toml`:

```toml
[session]
interrupt_same_session = true
max_queue_per_session = 1
cross_session_policy = "queue" # queue | mix | interrupt_all
```

Delete `src/tts_summarizer/session.py` and `tests/test_session.py`.

- [ ] **Step 4: Run config and state tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_config tests.test_state_client
```

Expected: all config and state tests pass.

---

## Task 5: Shelf CLI Speak

**Files:**
- Modify: `src/tts_summarizer/cli.py`
- Modify: `tests/test_cli_commands.py`

**Interfaces:**
- Removes: CLI `speak` command and stale request payload posting.
- Keeps: `serve`, `health`, `stop`, and `config-check` behavior.

- [ ] **Step 1: Write/update CLI tests**

Replace speak-posting tests in `tests/test_cli_commands.py` with:

```python
import unittest

from tts_summarizer import cli


class CliCommandTests(unittest.TestCase):
    def test_speak_command_is_shelved(self):
        code = cli.main(["speak", "--text", "hello"])

        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands
```

Expected before implementation: failure if `speak` still posts to `/v1/speak` and returns success.

- [ ] **Step 3: Remove the `speak` command path**

In `src/tts_summarizer/cli.py`:

```python
# delete speak parser setup
# delete args.command == "speak" branch
# delete SpeechRequest and RequestError imports
```

The remaining command handling should be only:

```python
if args.command == "serve":
    return run_server(config)

base_url = daemon_base_url(config, getattr(args, "config", None))
if base_url is None:
    print("tts-summarizer daemon unavailable", file=sys.stderr)
    return 0

timeout = config.server.request_timeout_ms / 1000
try:
    if args.command == "health":
        print(get_json(f"{base_url}/health", timeout))
        return 0
    if args.command == "stop":
        post_json(f"{base_url}/shutdown", {}, timeout)
        return 0
except Exception as exc:
    print(f"tts-summarizer request failed: {exc}", file=sys.stderr)
    return 0

return 0
```

- [ ] **Step 4: Run CLI tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands
```

Expected: all CLI command tests pass.

---

## Task 6: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Verify all touched source/tests.

**Interfaces:**
- Documents: client-owned playback, `/v1/speak` body, identity headers, and no same-session interruption guarantee.

- [ ] **Step 1: Update README after behavior works**

Replace stale request section with:

````markdown
## Send request

`/v1/speak` returns WAV bytes. Playback belongs client.

```bash
curl -sS \
  -H 'Content-Type: application/json' \
  -H 'X-TTS-Caller: manual' \
  -H 'X-TTS-Session-Id: demo' \
  -d '{"text":"Codex finished.","summarize":true}' \
  http://127.0.0.1:9200/v1/speak > speech.wav
```

Use `"summarize": false` send text directly TTS.
````

Remove README claims about CLI `speak`, same-session interruption, and session config.

- [ ] **Step 2: Run focused behavior tests**

Run:

```bash
rtk uv run python -m unittest tests.test_server tests.test_request tests.test_config tests.test_cli_commands tests.test_speech_audio tests.test_state_client
```

Expected: all focused tests pass.

- [ ] **Step 3: Run touched-file lint and format checks**

Run:

```bash
rtk uv run ruff format --check src/tts_summarizer/request.py src/tts_summarizer/server.py src/tts_summarizer/audio.py src/tts_summarizer/config.py src/tts_summarizer/state.py src/tts_summarizer/cli.py tests/test_server.py tests/test_request.py tests/test_config.py tests/test_cli_commands.py tests/test_speech_audio.py tests/test_state_client.py
rtk uv run ruff check src/tts_summarizer/request.py src/tts_summarizer/server.py src/tts_summarizer/audio.py src/tts_summarizer/config.py src/tts_summarizer/state.py src/tts_summarizer/cli.py tests/test_server.py tests/test_request.py tests/test_config.py tests/test_cli_commands.py tests/test_speech_audio.py tests/test_state_client.py
```

Expected: both commands pass.

- [ ] **Step 4: Run typecheck**

Run:

```bash
rtk uv run ty check src tests
```

Expected: `All checks passed!`

---

## Self-Review

- Spec coverage: API playback removed; client playback deferred; `summarize` added; identity moved to headers; interruption architecture removed; FastAPI shape preserved; sync route selected for blocking work.
- Placeholder scan: No `TBD`, `TODO`, or vague implementation steps remain.
- Type consistency: Plan uses `SpeechRequest.summarize`; no `SpeechRequest.playback`; `caller/session_id` are constructor/header inputs, not serialized payload fields; `TtsService` is deleted, not shrunk.
- YAGNI check: Removes queue/cancellation/session-manager/player/service machinery instead of adapting it for a single synchronous bytes endpoint.
