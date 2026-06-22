# Serve Speak Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `serve` expose `/v1/speak` as a bytes-only TTS endpoint with request metadata in headers, optional summarization, and no server playback/interruption machinery.

**Architecture:** `/v1/speak` accepts JSON body for synthesis options only, reads request identity from HTTP headers, runs optional summarization, generates TTS chunks, encodes one WAV response, and returns `audio/wav`. Server playback, queueing, cancellation tokens, and same-session interruption are removed because playback belongs to clients. The CLI `speak` feature is shelved; this plan does not preserve it.

**Tech Stack:** Python stdlib `http.server`, `urllib`, `wave`; existing `SpeechRequest`, `TtsService`, `SpeechGenerator`, `AudioChunk`, `chunks_to_wav_bytes`; `unittest`; `ruff`; `ty`.

## Global Constraints

- `/v1/speak` must never accept or branch on a `playback` request property.
- `/v1/speak` must always return generated audio bytes with `Content-Type: audio/wav` for valid requests.
- Request JSON payload contains `text`, optional `metadata`, optional `summarize` only.
- `caller` and `session_id` are request headers, not payload properties.
- Use headers `X-TTS-Caller` and `X-TTS-Session-Id` unless the user requests different names.
- Missing `X-TTS-Caller` defaults to `default`; missing `X-TTS-Session-Id` uses existing fallback session id.
- `summarize` is optional and defaults to `True`.
- `summarize: false` skips the summarizer and sends original request text directly to TTS.
- Drop speech interruption: remove cancellation tokens, same-session generation tracking, server playback queueing, and interruption config/tests.
- Do not implement `ffplay` client playback in this plan.
- Keep the diff small. Delete dead architecture instead of preserving shims.

---

## File Structure

- Modify `src/tts_summarizer/request.py`
  - Remove `playback` from `SpeechRequest`.
  - Add `summarize: bool = True`.
  - Remove `event` from `SpeechRequest`; use `metadata` for any future labels.
  - Stop parsing `caller` and `session_id` from JSON.
  - Add simple header-fed construction, either by extending `from_json(data, caller=None, session_id=None)` or adding a small `from_http(data, caller, session_id)` helper.
  - Keep `session_key()` for logging only.
- Modify `src/tts_summarizer/server.py`
  - Read `X-TTS-Caller` and `X-TTS-Session-Id` in `Handler._speak()`.
  - Always call `service.synthesize(request)` and `_send_audio()`.
  - Remove playback branch, job queue, worker loop, `SessionManager`, `WorkToken`, and `AudioPlayer` use.
  - Keep `/shutdown` by directly shutting down the HTTP server; no service stop hook.
- Modify `src/tts_summarizer/config.py`
  - Remove `SessionConfig` and `Config.session` if no remaining references after server cleanup.
- Delete `src/tts_summarizer/session.py` if all references disappear.
- Modify `src/tts_summarizer/audio.py`
  - Remove `WorkToken` import and `token` parameter from `AudioPlayer.play()` if `AudioPlayer` remains for future client/local use.
- Modify `tests/test_server.py`
  - Keep endpoint HTTP tests only.
  - Add header metadata tests.
  - Add default summarization and `summarize: false` direct-TTS tests.
  - Delete playback/queue/interruption tests.
- Modify `tests/test_request.py`
  - Add request parsing tests for default `summarize=True`, explicit `False`, and payload ignoring `caller`/`session_id`.
- Modify `tests/test_config.py`
  - Remove assertions for `session` config if present.
- Modify `tests/test_cli_commands.py` and `src/tts_summarizer/cli.py`
  - Since CLI `speak` is shelved, remove the `speak` subcommand and its tests, or mark it unsupported with the smallest code path. Recommended: remove command and tests now to avoid a known-broken client path.

---

## Task 1: Request Contract

**Files:**
- Modify: `src/tts_summarizer/request.py`
- Test: `tests/test_request.py`

**Interfaces:**
- Produces: `SpeechRequest.from_json(data, caller=None, session_id=None) -> SpeechRequest`.
- Produces: `SpeechRequest.summarize: bool` defaulting to `True`.
- Removes: payload-derived `caller`, payload-derived `session_id`, and `playback`.

- [ ] **Step 1: Write failing request tests**

Add tests to `tests/test_request.py`:

```python
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


def test_speech_request_accepts_summarize_false(self):
    request = SpeechRequest.from_json(
        {"text": "hello", "summarize": False}, caller="c", session_id="s"
    )

    self.assertIs(request.summarize, False)
    self.assertEqual(request.to_json()["summarize"], False)
```

- [ ] **Step 2: Run request tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_request
```

Expected before implementation: failures because `from_json()` does not accept header identity and still serializes old fields.

- [ ] **Step 3: Implement minimal request contract**

Update `SpeechRequest` to this shape:

```python
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
```

Update `from_cli()` only if it remains; it must no longer put `caller`, `session_id`, or `playback` into payload JSON.

- [ ] **Step 4: Run request tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_request
```

Expected: all request tests pass.

---

## Task 2: Header-Driven Bytes-Only Endpoint

**Files:**
- Modify: `src/tts_summarizer/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `SpeechRequest.from_json(payload, caller=..., session_id=...)` from Task 1.
- Produces: `/v1/speak` always responds with WAV bytes for valid requests.

- [ ] **Step 1: Write failing endpoint tests**

In `tests/test_server.py`, make `post_speak()` accept headers:

```python
def post_speak(service, payload, headers=None):
    handler = type("TestHandler", (Handler,), {"service": service})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    try:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/v1/speak",
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1) as response:
            return response.status, response.headers, response.read()
    finally:
        httpd.shutdown()
        thread.join()
        httpd.server_close()
```

Add or keep a capturing speech helper:

```python
class CapturingSpeech:
    def __init__(self):
        self.text = ""

    def generate(self, text):
        self.text = text
        return [AudioChunk(samples=[0.0], sample_rate=8000)]
```

Add tests:

```python
def test_speak_endpoint_uses_identity_headers_for_logs(self):
    speech = CapturingSpeech()
    service = TtsService(Config(), summarizer=FakeSummarizer(), speech=speech)

    with self.assertLogs("tts_summarizer.server", level="INFO") as logs:
        status, headers, body = post_speak(
            service,
            {"text": "hello", "caller": "body", "session_id": "body-session"},
            {"X-TTS-Caller": "header", "X-TTS-Session-Id": "header-session"},
        )

    self.assertEqual(status, 200)
    self.assertEqual(headers["Content-Type"], "audio/wav")
    self.assertTrue(body.startswith(b"RIFF"))
    self.assertIn("session=header:header-session", "\n".join(logs.output))
    self.assertNotIn("body:body-session", "\n".join(logs.output))


def test_speak_endpoint_summarizes_by_default(self):
    speech = CapturingSpeech()
    service = TtsService(Config(), summarizer=FakeSummarizer(), speech=speech)

    status, headers, body = post_speak(service, {"text": "hello"})

    self.assertEqual(status, 200)
    self.assertEqual(headers["Content-Type"], "audio/wav")
    self.assertTrue(body.startswith(b"RIFF"))
    self.assertEqual(speech.text, "summary: hello")


def test_speak_endpoint_can_skip_summarizer_for_tts_testing(self):
    speech = CapturingSpeech()
    service = TtsService(Config(), summarizer=FakeSummarizer(), speech=speech)

    status, headers, body = post_speak(service, {"text": "hello", "summarize": False})

    self.assertEqual(status, 200)
    self.assertEqual(headers["Content-Type"], "audio/wav")
    self.assertTrue(body.startswith(b"RIFF"))
    self.assertEqual(speech.text, "hello")
```

Delete playback endpoint tests.

- [ ] **Step 2: Run server tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_server
```

Expected before implementation: failures because headers are ignored, `summarize: false` still summarizes, or playback-specific tests still exist.

- [ ] **Step 3: Implement endpoint behavior**

In `Handler._speak()`:

```python
caller = self.headers.get("X-TTS-Caller")
session_id = self.headers.get("X-TTS-Session-Id")
request = SpeechRequest.from_json(payload, caller=caller, session_id=session_id)
body = self.service.synthesize(request)
self._send_audio(200, body)
```

In `TtsService._generate()`:

```python
def _generate(self, request: SpeechRequest):
    if request.summarize:
        logger.info("summarizing speech request session=%s", request.session_key())
        text = self.summarizer.summarize(request.text)
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
    return self.speech.generate(text)
```

- [ ] **Step 4: Run server tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_server
```

Expected: all server tests pass.

---

## Task 3: Remove Interruption and Server Playback Architecture

**Files:**
- Modify: `src/tts_summarizer/server.py`
- Modify: `src/tts_summarizer/audio.py`
- Modify: `src/tts_summarizer/config.py`
- Delete: `src/tts_summarizer/session.py` if no references remain
- Test: `tests/test_server.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Removes: server-side queue, worker, `SessionManager`, `WorkToken`, `interrupt_same_session`, `max_queue_per_session`, `cross_session_policy`.
- Keeps: synchronous `TtsService.synthesize(request) -> bytes`.

- [ ] **Step 1: Search references before deleting**

Use harness search for these patterns in `src tests`:

```text
SessionManager
WorkToken
interrupt_same_session
max_queue_per_session
cross_session_policy
process_pending
service.handle(
AudioPlayer(
```

Expected: only server playback/interruption code and tests reference them.

- [ ] **Step 2: Delete server queue/playback methods**

In `src/tts_summarizer/server.py`, reduce `TtsService` to config, summarizer, speech, and `synthesize()`/`_generate()`/`health()`:

```python
class TtsService:
    def __init__(self, config: Config, summarizer=None, speech=None):
        self.config = config
        self.summarizer = summarizer or Summarizer(config.summarizer)
        self.speech = speech or SpeechGenerator(config.tts)

    def synthesize(self, request: SpeechRequest) -> bytes:
        return chunks_to_wav_bytes(self._generate(request))
```

Remove imports no longer needed:

```python
from queue import Empty, Queue
import threading
from .audio import AudioPlayer
from .session import SessionManager, WorkToken
```

In `/shutdown`, remove `self.service.stop()` and directly start shutdown thread:

```python
self._send(200, {"status": "shutting_down"})
threading.Thread(target=self.server.shutdown).start()
```

Keep `import threading` only if `/shutdown` still uses it.

- [ ] **Step 3: Remove interruption config and token plumbing**

In `src/tts_summarizer/config.py`, delete:

```python
@dataclass(frozen=True)
class SessionConfig:
    interrupt_same_session: bool = True
    max_queue_per_session: int = 1
    cross_session_policy: str = "queue"
```

Delete `session: SessionConfig = field(default_factory=SessionConfig)` from `Config` if present.

In `src/tts_summarizer/audio.py`, remove `WorkToken` import and simplify `AudioPlayer.play()`:

```python
def play(self, chunks: Iterable[AudioChunk]) -> None:
    output_dir = Path(self.config.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        path = output_dir / f"speech-{time.time_ns()}.wav"
        write_wav(path, chunk)
        if self.config.backend in {"auto", "afplay"} and not self.config.save:
            proc = subprocess.Popen(["/usr/bin/afplay", str(path)])
            while proc.poll() is None:
                time.sleep(0.05)
```

Delete `src/tts_summarizer/session.py` after references are gone.

- [ ] **Step 4: Delete tests only covering removed architecture**

Remove server tests that assert async playback/interruption internals, including:

```python
test_service_speaks_request
test_service_returns_before_slow_tts_finishes
test_service_handle_does_not_start_worker_thread
test_speak_endpoint_queues_playback_when_requested
```

Update config tests to stop asserting session defaults.

- [ ] **Step 5: Run focused tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_server tests.test_config tests.test_speech_audio
```

Expected: all focused tests pass.

---

## Task 4: Shelf CLI Speak

**Files:**
- Modify: `src/tts_summarizer/cli.py`
- Modify: `tests/test_cli_commands.py`

**Interfaces:**
- Removes or disables CLI `speak` command so it does not advertise or test a stale payload contract.
- Keeps `serve` and `health` behavior.

- [ ] **Step 1: Write/update failing CLI tests**

Replace speak-posting tests with a serve-focused assertion or delete them if redundant. If keeping a guard, add:

```python
def test_speak_command_is_shelved(self):
    code = cli.main(["speak", "--text", "hello"])

    self.assertNotEqual(code, 0)
```

- [ ] **Step 2: Run CLI tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands
```

Expected before implementation: failure if `speak` still posts to `/v1/speak`.

- [ ] **Step 3: Remove the `speak` subcommand path**

In `src/tts_summarizer/cli.py`, either remove the `speak` parser and branch, or keep parser with a direct unsupported return. Recommended minimum:

```python
speak = subcommands.add_parser("speak")
speak.add_argument("--text")
```

and in command handling:

```python
if args.command == "speak":
    print("tts-summarizer speak is shelved; use serve /v1/speak and pipe audio client-side")
    return 2
```

Do not post JSON from CLI speak in this plan.

- [ ] **Step 4: Run CLI tests and verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands
```

Expected: all CLI command tests pass.

---

## Task 5: Final Verification

**Files:**
- Verify touched files only first.
- Then typecheck all `src tests` because public request/config/server shapes changed.

- [ ] **Step 1: Run focused behavior tests**

Run:

```bash
rtk uv run python -m unittest tests.test_server tests.test_request tests.test_config tests.test_cli_commands tests.test_speech_audio
```

Expected: all tests pass.

- [ ] **Step 2: Run touched-file lint and format checks**

Run:

```bash
rtk uv run ruff format --check src/tts_summarizer/request.py src/tts_summarizer/server.py src/tts_summarizer/audio.py src/tts_summarizer/speech.py src/tts_summarizer/config.py src/tts_summarizer/cli.py tests/test_server.py tests/test_request.py tests/test_config.py tests/test_cli_commands.py tests/test_speech_audio.py
rtk uv run ruff check src/tts_summarizer/request.py src/tts_summarizer/server.py src/tts_summarizer/audio.py src/tts_summarizer/speech.py src/tts_summarizer/config.py src/tts_summarizer/cli.py tests/test_server.py tests/test_request.py tests/test_config.py tests/test_cli_commands.py tests/test_speech_audio.py
```

Expected: both commands pass.

- [ ] **Step 3: Run typecheck**

Run:

```bash
rtk uv run ty check src tests
```

Expected: `All checks passed!`

- [ ] **Step 4: Report unrelated formatter drift**

If full repo format check is run, expect pre-existing drift in `src/tts_summarizer/summarizer.py`. Do not format unrelated files unless the user asks.

---

## Self-Review

- Spec coverage: API playback removed; client playback deferred; `summarize` added; identity moved from payload to headers; interruption architecture removed.
- Placeholder scan: No `TBD`, `TODO`, or vague implementation steps remain.
- Type consistency: Plan uses `SpeechRequest.summarize`; no `SpeechRequest.playback`; `caller/session_id` are constructor/header inputs, not serialized payload fields.
- YAGNI check: Removes queue/cancellation/session-manager machinery instead of adapting it for a single synchronous bytes endpoint.
