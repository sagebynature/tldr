# Remote TTS Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow TTS profiles to synthesize through a remote OpenAI/MLX-Audio-compatible `/v1/audio/speech` server using `base_url`, `api_key`, and `model`, while keeping local MLX-Audio profiles working.

**Architecture:** Keep `SpeechGenerator` as the public entry point. Add remote profile fields to config, add a tiny `AudioBytes` output type for already-encoded WAV bytes, and route default generation through a small backend dispatcher that chooses local MLX-Audio or remote HTTP by `TtsProfileConfig.backend`. Server streaming branches once: local `AudioChunk` streams go through `chunks_to_wav_stream`; remote `AudioBytes` chunks pass through unchanged.

**Tech Stack:** Python 3.11 dataclasses and typing, stdlib `urllib.request` and `json`, FastAPI `StreamingResponse`, existing `unittest` tests, existing `rtk uv run python -m unittest` verification.

## Global Constraints

- No OpenAI SDK dependency; use stdlib HTTP like `OpenAICompatibleBackend`.
- Local TTS profiles without `backend`, `base_url`, or `api_key` must keep working.
- Remote TTS uses `POST {base_url.rstrip("/")}/audio/speech`.
- Remote response format for `/v1/speak` is WAV only.
- Do not decode, resample, or re-encode remote audio.
- Do not change public `/v1/speak` request JSON shape.
- Prefix shell commands with `rtk`.

---

## File Structure

- Modify `src/tts_summarizer/config.py`
  - Add `backend`, `base_url`, `api_key` to `TtsProfileConfig`.
- Modify `src/tts_summarizer/speech.py`
  - Add `AudioBytes` and `SpeechOutput`.
  - Add `RemoteTtsBackend`.
  - Add `RoutingSpeechBackend` that chooses `MlxAudioBackend` or `RemoteTtsBackend`.
  - Keep `SpeechGenerator` constructor injectable for existing tests.
- Modify `src/tts_summarizer/server.py`
  - Add helper that converts `SpeechOutput` to response byte iterable.
- Modify `tests/test_config.py`
  - Assert remote TTS profile fields load from TOML.
- Modify `tests/test_speech_audio.py`
  - Assert remote backend URL, headers, body, and byte streaming.
  - Assert default `SpeechGenerator` routes remote profiles to the remote backend.
- Modify `tests/test_server.py`
  - Assert remote `AudioBytes` pass through without an added WAV header.
- Modify `config.toml`
  - Add a commented or named remote profile example only if it does not make local default unusable.
- Modify `README.md`
  - Document remote TTS config briefly.

---

## Task 1: Load Remote TTS Config

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/tts_summarizer/config.py`

**Interfaces:**
- Produces: `TtsProfileConfig.backend: str`, `TtsProfileConfig.base_url: str`, `TtsProfileConfig.api_key: str`.
- Consumes: existing `_merge_dataclass(TtsProfileConfig(), profile)` loader.

- [ ] **Step 1: Write failing config test**

Add this test to `ConfigTests` in `tests/test_config.py`:

```python
def test_tts_remote_profile_config_loads_endpoint_fields(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.toml"
        path.write_text(
            "\n".join(
                [
                    "[tts]",
                    'default_profile = "remote"',
                    "[tts.profiles.remote]",
                    'backend = "remote"',
                    'base_url = "http://127.0.0.1:9100/v1"',
                    'api_key = "omlx"',
                    'model = "mlx-community/Kokoro-82M-bf16"',
                    "sample_rate = 24000",
                    "[tts.profiles.remote.generate_kwargs]",
                    'voice = "af_heart"',
                    'response_format = "wav"',
                ]
            ),
            encoding="utf-8",
        )

        cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

    profile = cfg.tts.profiles["remote"]
    self.assertEqual(profile.backend, "remote")
    self.assertEqual(profile.base_url, "http://127.0.0.1:9100/v1")
    self.assertEqual(profile.api_key, "omlx")
    self.assertEqual(profile.model, "mlx-community/Kokoro-82M-bf16")
    self.assertEqual(profile.generate_kwargs["voice"], "af_heart")
    self.assertEqual(profile.generate_kwargs["response_format"], "wav")
```

- [ ] **Step 2: Run config test and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_config.ConfigTests.test_tts_remote_profile_config_loads_endpoint_fields
```

Expected before implementation: FAIL with `unknown config keys TtsProfileConfig: api_key, backend, base_url`.

- [ ] **Step 3: Implement config fields**

Update `TtsProfileConfig` in `src/tts_summarizer/config.py` to:

```python
@dataclass(frozen=True)
class TtsProfileConfig:
    model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    stream: bool = True
    sample_rate: int = 24000
    generate_kwargs: dict[str, object] = field(default_factory=dict)
    backend: str = "mlx"
    base_url: str = ""
    api_key: str = ""
```

Do not add custom merge code; `_merge_dataclass()` already rejects unknown keys and accepts new dataclass fields.

- [ ] **Step 4: Run config tests**

Run:

```bash
rtk uv run python -m unittest tests.test_config
```

Expected: PASS.

- [ ] **Step 5: Commit config fields**

```bash
rtk git add src/tts_summarizer/config.py tests/test_config.py
rtk git commit -m "feat: load remote tts profile config"
```

---

## Task 2: Add Remote TTS Backend

**Files:**
- Modify: `tests/test_speech_audio.py`
- Modify: `src/tts_summarizer/speech.py`

**Interfaces:**
- Consumes: `TtsProfileConfig.backend/base_url/api_key/model/stream/generate_kwargs` from Task 1.
- Produces: `AudioBytes(chunks: Iterable[bytes], content_type: str = "audio/wav")`.
- Produces: `RemoteTtsBackend.generate(text: str, config: TtsProfileConfig) -> AudioBytes`.

- [ ] **Step 1: Write failing remote backend tests**

Add imports at the top of `tests/test_speech_audio.py`:

```python
import json
```

Update the speech import to include `AudioBytes` and `RemoteTtsBackend`:

```python
from tts_summarizer.speech import AudioBytes, AudioChunk, RemoteTtsBackend, SpeechGenerator
```

Add these tests to `SpeechAudioTests`:

```python
def test_remote_tts_backend_posts_openai_audio_speech_request(self):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size=-1):
            return b"RIFFremote"

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return Response()

    config = TtsProfileConfig(
        backend="remote",
        base_url="http://127.0.0.1:9100/v1/",
        api_key="omlx",
        model="mlx-community/Kokoro-82M-bf16",
        stream=True,
        generate_kwargs={"voice": "af_heart", "response_format": "mp3"},
    )

    output = RemoteTtsBackend(urlopen=fake_urlopen, timeout=7).generate("hello", config)
    self.assertIsInstance(output, AudioBytes)
    self.assertEqual(b"".join(output.chunks), b"RIFFremote")

    request, timeout = calls[0]
    body = json.loads(request.data.decode("utf-8"))
    self.assertEqual(request.full_url, "http://127.0.0.1:9100/v1/audio/speech")
    self.assertEqual(timeout, 7)
    self.assertEqual(request.get_method(), "POST")
    self.assertEqual(request.headers["Content-type"], "application/json")
    self.assertEqual(request.headers["Authorization"], "Bearer omlx")
    self.assertEqual(body["model"], "mlx-community/Kokoro-82M-bf16")
    self.assertEqual(body["input"], "hello")
    self.assertIs(body["stream"], True)
    self.assertEqual(body["voice"], "af_heart")
    self.assertEqual(body["response_format"], "wav")


def test_remote_tts_backend_omits_empty_authorization(self):
    headers = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _size=-1):
            return b"RIFF"

    def fake_urlopen(request, timeout):
        headers.append(request.headers)
        return Response()

    config = TtsProfileConfig(
        backend="remote",
        base_url="http://127.0.0.1:9100/v1",
        api_key="",
        model="model",
    )

    output = RemoteTtsBackend(urlopen=fake_urlopen).generate("hello", config)
    self.assertEqual(b"".join(output.chunks), b"RIFF")
    self.assertNotIn("Authorization", headers[0])
```

- [ ] **Step 2: Run remote backend tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio.SpeechAudioTests.test_remote_tts_backend_posts_openai_audio_speech_request tests.test_speech_audio.SpeechAudioTests.test_remote_tts_backend_omits_empty_authorization
```

Expected before implementation: FAIL importing `AudioBytes` or `RemoteTtsBackend`.

- [ ] **Step 3: Implement remote backend types**

In `src/tts_summarizer/speech.py`, add imports:

```python
from urllib.request import Request, urlopen
import json
```

Add after `AudioChunk`:

```python
@dataclass(frozen=True)
class AudioBytes:
    chunks: Iterable[bytes]
    content_type: str = "audio/wav"


SpeechOutput = Iterable[AudioChunk] | AudioBytes
```

Change `SpeechBackend` to:

```python
class SpeechBackend(Protocol):
    def generate(self, text: str, config: TtsProfileConfig) -> SpeechOutput: ...
```

Add `RemoteTtsBackend` before `SpeechGenerator`:

```python
class RemoteTtsBackend:
    def __init__(self, urlopen: Any = urlopen, timeout: float = 30):
        self.urlopen = urlopen
        self.timeout = timeout

    def generate(self, text: str, config: TtsProfileConfig) -> AudioBytes:
        if not config.base_url:
            raise ValueError("remote TTS profile requires base_url")
        body = {
            "model": config.model,
            "input": text,
            "stream": config.stream,
            **config.generate_kwargs,
            "response_format": "wav",
        }
        body_bytes = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = Request(
            f"{config.base_url.rstrip('/')}/audio/speech",
            data=body_bytes,
            headers=headers,
            method="POST",
        )

        def chunks() -> Iterable[bytes]:
            with cast(Any, self.urlopen(request, timeout=self.timeout)) as response:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return AudioBytes(chunks())
```

The `response_format` key is intentionally after `generate_kwargs` so `/v1/speak` always receives WAV even if config contains a stale format.

- [ ] **Step 4: Run speech audio tests**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio
```

Expected: PASS.

- [ ] **Step 5: Commit remote backend**

```bash
rtk git add src/tts_summarizer/speech.py tests/test_speech_audio.py
rtk git commit -m "feat: add remote tts backend"
```

---

## Task 3: Route Profiles to Local or Remote Backend

**Files:**
- Modify: `tests/test_speech_audio.py`
- Modify: `src/tts_summarizer/speech.py`

**Interfaces:**
- Consumes: `MlxAudioBackend.generate(...) -> Iterable[AudioChunk]`.
- Consumes: `RemoteTtsBackend.generate(...) -> AudioBytes`.
- Produces: `RoutingSpeechBackend.generate(text, config) -> SpeechOutput`.
- Preserves: `SpeechGenerator(config, backend=FakeBackend())` injection for existing tests.

- [ ] **Step 1: Write failing routing tests**

Add this test to `SpeechAudioTests`:

```python
def test_speech_generator_routes_remote_profile_to_remote_backend(self):
    calls = []

    class LocalBackend:
        def generate(self, text, config):
            calls.append(("local", text, config.model))
            return [AudioChunk(samples=[0.0], sample_rate=8000)]

    class RemoteBackend:
        def generate(self, text, config):
            calls.append(("remote", text, config.model))
            return AudioBytes([b"RIFFremote"])

    generator = SpeechGenerator(
        TtsConfig(
            default_profile="remote",
            profiles={
                "local": TtsProfileConfig(model="local-model"),
                "remote": TtsProfileConfig(
                    backend="remote",
                    base_url="http://127.0.0.1:9100/v1",
                    model="remote-model",
                ),
            },
        )
    )
    generator.backend = tts_summarizer.speech.RoutingSpeechBackend(
        local=LocalBackend(), remote=RemoteBackend()
    )

    output = generator.generate("hello")

    self.assertEqual(output, AudioBytes([b"RIFFremote"]))
    self.assertEqual(calls, [("remote", "hello", "remote-model")])


def test_speech_generator_rejects_unknown_backend(self):
    generator = SpeechGenerator(
        TtsConfig(
            profiles={"bad": TtsProfileConfig(backend="wat", model="m")},
            default_profile="bad",
        )
    )

    with self.assertRaisesRegex(ValueError, "unknown TTS backend: wat"):
        generator.generate("hello")
```

- [ ] **Step 2: Run routing tests and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio.SpeechAudioTests.test_speech_generator_routes_remote_profile_to_remote_backend tests.test_speech_audio.SpeechAudioTests.test_speech_generator_rejects_unknown_backend
```

Expected before implementation: FAIL because `RoutingSpeechBackend` does not exist or unknown backend is not checked.

- [ ] **Step 3: Implement routing backend**

Add before `SpeechGenerator` in `src/tts_summarizer/speech.py`:

```python
class RoutingSpeechBackend:
    def __init__(
        self,
        local: SpeechBackend | None = None,
        remote: SpeechBackend | None = None,
    ):
        self.local = local or MlxAudioBackend()
        self.remote = remote or RemoteTtsBackend()

    def generate(self, text: str, config: TtsProfileConfig) -> SpeechOutput:
        if config.backend == "mlx":
            return self.local.generate(text, config)
        if config.backend == "remote":
            return self.remote.generate(text, config)
        raise ValueError(f"unknown TTS backend: {config.backend}")
```

Change `SpeechGenerator.__init__` default backend line to:

```python
self.backend = backend or RoutingSpeechBackend()
```

Change `SpeechGenerator.generate()` return annotation to:

```python
def generate(self, text: str, profile_name: str | None = None) -> SpeechOutput:
    return self.backend.generate(text, self.profile(profile_name))
```

- [ ] **Step 4: Run speech audio tests**

Run:

```bash
rtk uv run python -m unittest tests.test_speech_audio
```

Expected: PASS.

- [ ] **Step 5: Commit routing**

```bash
rtk git add src/tts_summarizer/speech.py tests/test_speech_audio.py
rtk git commit -m "feat: route tts profiles by backend"
```

---

## Task 4: Pass Remote WAV Bytes Through Server

**Files:**
- Modify: `tests/test_server.py`
- Modify: `src/tts_summarizer/server.py`

**Interfaces:**
- Consumes: `AudioBytes` and `SpeechOutput` from Task 2.
- Produces: `speech_output_to_wav_stream(output, sample_rate) -> Iterable[bytes]` or equivalent helper.

- [ ] **Step 1: Write failing server pass-through test**

Update import in `tests/test_server.py`:

```python
from tts_summarizer.speech import AudioBytes, AudioChunk
```

Add this fake class near `CapturingSpeech`:

```python
class RemoteSpeech:
    def sample_rate(self, profile_name=None):
        return 24000

    def generate(self, text, profile_name=None):
        return AudioBytes([b"RIFFremote-wav"])
```

Add this test to `ServerTests`:

```python
def test_fastapi_speak_route_passes_remote_wav_bytes_through(self):
    client = TestClient(
        create_app(Config(), summarizer=FakeSummarizer(), speech=RemoteSpeech())
    )

    response = client.post("/v1/speak", json={"text": "hello"})

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.headers["content-type"], "audio/wav")
    self.assertEqual(response.content, b"RIFFremote-wav")
```

- [ ] **Step 2: Run server pass-through test and verify failure**

Run:

```bash
rtk uv run python -m unittest tests.test_server.ServerTests.test_fastapi_speak_route_passes_remote_wav_bytes_through
```

Expected before implementation: FAIL because server treats `AudioBytes` as `AudioChunk` iterable and tries to wrap it.

- [ ] **Step 3: Implement server output branch**

In `src/tts_summarizer/server.py`, change import to include `AudioBytes`:

```python
from .speech import AudioBytes, SpeechGenerator
```

Add helper near `synthesize_speech()`:

```python
def _speech_output_to_wav_stream(output, sample_rate: int) -> Iterable[bytes]:
    if isinstance(output, AudioBytes):
        return output.chunks
    return chunks_to_wav_stream(output, sample_rate)
```

Change the return at the end of `synthesize_speech()` from direct `chunks_to_wav_stream(...)` to:

```python
output = speech.generate(text, profile_name=request.tts_profile)
return _speech_output_to_wav_stream(output, sample_rate)
```

Keep `sample_rate` lookup before generation unchanged so existing fake speech tests continue passing.

- [ ] **Step 4: Run server tests**

Run:

```bash
rtk uv run python -m unittest tests.test_server
```

Expected: PASS.

- [ ] **Step 5: Commit server pass-through**

```bash
rtk git add src/tts_summarizer/server.py tests/test_server.py
rtk git commit -m "feat: stream remote tts wav bytes"
```

---

## Task 5: Document Remote TTS and Run Final Checks

**Files:**
- Modify: `README.md`
- Modify: `config.toml`

**Interfaces:**
- Consumes: final config fields from Task 1.
- Produces: documented user-facing config example.

- [ ] **Step 1: Update sample config without changing default behavior**

In `config.toml`, keep `[tts] default_profile = "kokoro"`. Add this profile after local TTS profiles:

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

Do not make it the default in the committed sample config; local development should still work without a remote server.

- [ ] **Step 2: Update README config docs**

Add a short section after the run/send request instructions:

````markdown
## Remote TTS backend

TTS profiles can call an OpenAI/MLX-Audio-compatible server instead of loading `mlx_audio` in this process:

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
````


- [ ] **Step 3: Run focused test suite**

Run:

```bash
rtk uv run python -m unittest tests.test_config tests.test_speech_audio tests.test_server
```

Expected: PASS.

- [ ] **Step 4: Run project check**

Run:

```bash
rtk make check
```

Expected: PASS.

- [ ] **Step 5: Commit docs and sample config**

```bash
rtk git add README.md config.toml
rtk git commit -m "docs: document remote tts backend"
```

---

## Self-Review Checklist

- Spec coverage: config fields covered by Task 1; remote HTTP covered by Task 2; routing covered by Task 3; pass-through WAV streaming covered by Task 4; docs/sample config covered by Task 5.
- Type consistency: `AudioBytes`, `SpeechOutput`, `RemoteTtsBackend`, and `RoutingSpeechBackend` names are defined before use.
- YAGNI check: no SDK, no provider abstraction beyond two concrete backends, no decode/re-encode path, no retries or health checks.
