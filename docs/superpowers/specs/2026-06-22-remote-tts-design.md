# Remote TTS Backend Design

## Goal

Allow TTS profiles to use a remotely deployed OpenAI-compatible TTS server through `base_url`, `api_key`, and `model`, while preserving the existing in-process MLX-Audio backend.

## Approved Scope

- Add remote TTS profile configuration for `base_url`, `api_key`, and explicit `backend` selection.
- Support OpenAI/MLX-Audio-compatible `POST /v1/audio/speech` remote synthesis.
- Pass remote WAV bytes through `/v1/speak` unchanged.
- Keep current local MLX-Audio profiles working without config changes.
- Reuse stdlib HTTP code; do not add an OpenAI SDK dependency.

## Out of Scope

- Adding non-OpenAI TTS providers.
- Decoding, resampling, or re-encoding remote audio.
- Supporting remote response formats other than WAV for `/v1/speak`.
- Adding provider discovery, model listing, retries, or health checks for the remote TTS server.
- Changing the public `/v1/speak` request body.

## Current Architecture

Current TTS flow:

1. `server.py` receives `/v1/speak`.
2. `synthesize_speech()` optionally summarizes the text.
3. `SpeechGenerator.generate()` selects a TTS profile.
4. `MlxAudioBackend.generate()` loads an MLX-Audio model in-process and returns `AudioChunk` objects.
5. `chunks_to_wav_stream()` converts those chunks to a WAV stream returned by FastAPI.

The summarizer already uses the target HTTP pattern: profile-level `base_url`, `api_key`, and `model`; stdlib `urllib.request`; OpenAI-compatible endpoint construction.

## Configuration

Extend `TtsProfileConfig` with:

```python
backend: str = "mlx"
base_url: str = ""
api_key: str = ""
```

Local profiles remain valid because `backend` defaults to `"mlx"`:

```toml
[tts.profiles.kokoro]
model = "mlx-community/Kokoro-82M-bf16"

[tts.profiles.kokoro.generate_kwargs]
voice = "af_heart"
lang_code = "a"
```

Remote profile example:

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

Rules:

- `backend = "mlx"` uses existing in-process generation.
- `backend = "remote"` requires non-empty `base_url`.
- Unknown backend values raise `ValueError` when the profile is used.
- `api_key` is optional. If non-empty, requests include `Authorization: Bearer <api_key>`.
- Remote request URL is `{base_url.rstrip("/")}/audio/speech`.
- Remote request body includes `model`, `input`, `stream`, `response_format = "wav"`, then `generate_kwargs`.
- `generate_kwargs` may include MLX-Audio fields such as `voice`, `speed`, `lang_code`, `streaming_interval`, `temperature`, `max_tokens`, `ref_audio`, `ref_text`, or `instruct`.

## Backend Interface

The current `SpeechBackend.generate()` returns `Iterable[AudioChunk]`. Remote TTS returns encoded WAV bytes. Add a small output union instead of decoding remote audio:

```python
@dataclass(frozen=True)
class AudioBytes:
    chunks: Iterable[bytes]
    content_type: str = "audio/wav"
```

Then allow:

```python
SpeechOutput = Iterable[AudioChunk] | AudioBytes
```

`MlxAudioBackend.generate()` continues returning local `AudioChunk` iterable. `RemoteTtsBackend.generate()` returns `AudioBytes` that streams response bytes.

## Remote HTTP Behavior

`RemoteTtsBackend` uses stdlib `urllib.request` like `OpenAICompatibleBackend`.

Request:

```json
{
  "model": "mlx-community/Kokoro-82M-bf16",
  "input": "Text to speak",
  "stream": true,
  "response_format": "wav",
  "voice": "af_heart",
  "lang_code": "a"
}
```

Response handling:

- Treat successful response body as binary WAV bytes.
- Yield response bytes in fixed-size chunks.
- Do not parse or validate WAV headers.
- Let HTTP and JSON encoding errors propagate to the FastAPI error path; existing server behavior already returns failures as HTTP errors/logs.

## Server Streaming

Update `synthesize_speech()` or the adjacent server helper so it branches on `SpeechOutput`:

- `AudioBytes`: yield bytes directly.
- `Iterable[AudioChunk]`: call existing `chunks_to_wav_stream(chunks, sample_rate)`.

The response media type remains `audio/wav`.

## Testing Strategy

Add the smallest tests that catch real breakage:

- Config loads remote TTS profile fields: `backend`, `base_url`, `api_key`, and `model`.
- Remote backend posts to `/audio/speech` with expected JSON body.
- Remote backend includes `Authorization` only when `api_key` is non-empty.
- Remote backend yields binary response bytes unchanged.
- Existing local MLX backend tests still pass.
- Server helper returns remote bytes without wrapping them in a second WAV header.

## Deferred Work

- Provider presets.
- Automatic remote server health checks.
- Model listing.
- Retry/backoff.
- Non-WAV response formats.
- OpenAI SDK integration.

These are deliberately omitted until needed; the remote TTS server already speaks the one endpoint this daemon needs.
