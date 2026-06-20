# Standalone TTS CLI and Warm Daemon Design

## Goal

Replace `~/.codex/hooks/tts` with a harness-neutral Python CLI and optional long-running service that keeps summarizer and TTS models warm, uses native MLX SDKs, and lets small per-harness hooks normalize events before calling the core tool.

## Non-goals

- The core service will not know Codex, Claude, or any other harness payload schema.
- The core service will not require a harness to use a specific hook format.
- The core service will not expose network access beyond loopback.
- The first version will not implement multi-host audio routing or remote clients.
- Docker/local container hosting is out of scope; the production local service runs natively on macOS for MLX/Metal.

## Command shape

Install one Python entrypoint, tentatively named `tts-summarizer`.

Required commands:

```bash
tts-summarizer speak [--config path] [--session-id id] [--caller id] [--text text]
tts-summarizer serve [--config path]
tts-summarizer health [--config path]
tts-summarizer stop [--config path]
tts-summarizer config-check [--config path]
```

`speak` accepts either `--text` or JSON on stdin. JSON is already normalized by the harness mini-hook.

Normalized request schema:

```json
{
  "text": "assistant response or event text",
  "session_id": "stable harness session id",
  "caller": "optional caller/harness name",
  "event": "optional event name",
  "metadata": {}
}
```

Only `text` is semantically required. If `session_id` is missing, the client derives a best-effort session key from caller, current working directory, and process parent. Harness hooks should pass a real session id when available.

## Config discovery

Config files are read in this exact order:

1. Explicit `--config /path/to/config.toml`; if provided and invalid or missing, fail fast.
2. Current working directory: `./config.toml`, if present.
3. User config directory: `~/.config/tts-summarizer/config.toml`, if present.
4. Built-in defaults.

No Codex config is read by the core tool. Per-harness mini-hooks may decide where they live and how they pass `--config`.

## Config model

Example:

```toml
[server]
host = "127.0.0.1"
port = 0
state_dir = "~/.cache/tts-summarizer"
auto_start = true
startup_timeout_ms = 3000
request_timeout_ms = 5000

[session]
interrupt_same_session = true
max_queue_per_session = 1
cross_session_policy = "queue" # queue | mix | interrupt_all

[summarizer]
enabled = true
model = "mlx-community/Qwen3-0.6B-4bit"
word_threshold = 0
max_words = 40
temperature = 0.2
max_tokens = 180
system_prompt = """
You summarize assistant responses for text-to-speech.
Return only a spoken summary.
Do not mention that this is a summary.
If the content is a question, preserve the question instead of answering it.
Do not include markdown, code fences, file paths, URLs, bullets, or formatting.
"""
user_prompt_template = """
Summarize this response in {max_words} words or fewer.
Preserve the practical outcome and next action.

{text}
"""

[tts]
model = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
voice = "Chelsie"
lang_code = "English"
speed = 1.6
ref_audio = ""
ref_text = ""
stream = true

[audio]
backend = "auto" # auto | afplay | sounddevice | file
output_dir = "~/.cache/tts-summarizer/audio"
save = false
```

All model parameters are configurable through TOML. The default summarizer prompt is usable without configuration, but both system and user prompt templates are overridable.

## Service transport

Use loopback HTTP.

Endpoints:

- `POST /v1/speak`: submit normalized request.
- `GET /health`: report daemon status, loaded models, active sessions.
- `POST /shutdown`: stop daemon.

The daemon binds to `127.0.0.1`. With `port = 0`, it chooses a free port and writes a state file under `state_dir` containing host, port, pid, started-at timestamp, and config fingerprint. Clients discover the daemon through this state file.

HTTP is chosen over Unix sockets because the user requested endpoint-style transport and future tools can call it without Python-specific socket handling.

## Session-aware interruption

Speech is keyed by:

```text
session_key = caller + ":" + session_id
```

If `caller` is absent, use `default`. If `session_id` is absent, use the client-derived fallback key.

Default behavior:

- A new request from the same `session_key` cancels currently playing or generating speech for that session.
- The latest request replaces any queued request for that same session.
- Requests from different sessions do not cancel each other.
- Cross-session output defaults to a single global audio queue to avoid overlapping speech.

This gives the important behavior: if one harness session keeps producing new turns, stale speech from that same session is interrupted immediately, while unrelated sessions do not erase each other's state.

## Model lifecycle

The daemon keeps model objects warm.

- Summarizer loads lazily on the first request that needs summarization.
- TTS loads lazily on the first speech request.
- Models remain loaded until daemon shutdown.
- `health` reports whether each model is unloaded, loading, ready, or failed.

Summarizer and TTS models may be different. Model reload happens only when the daemon starts with a different config. Hot reload is deferred until needed.

## Request flow

1. Harness mini-hook receives harness event.
2. Mini-hook extracts a normalized request: `text`, `session_id`, `caller`, optional metadata.
3. Mini-hook calls `tts-summarizer speak` with JSON on stdin.
4. CLI reads config using the configured search order.
5. CLI starts daemon if enabled and not running.
6. CLI sends `POST /v1/speak` to loopback daemon.
7. Daemon cancels same-session stale work.
8. Daemon summarizes text if enabled and threshold rules require it.
9. Daemon generates speech through native `mlx-audio` APIs.
10. Daemon streams or plays audio through configured backend.

## Failure behavior

Hook clients should never fail the calling harness because speech failed.

- Missing disabled/enable gate: exit `0` without speaking.
- Daemon unavailable and auto-start disabled: log and exit `0`.
- Daemon startup timeout: log and exit `0` unless `--strict` is later added.
- Summarizer failure: log and use original text.
- TTS failure: log and exit `0`.
- Invalid explicit config path: fail command with nonzero status for manual invocations; mini-hooks may choose to suppress that.

## Harness mini-hooks

Harness-specific logic lives outside the core service.

Each mini-hook should:

- Read the harness payload.
- Extract text.
- Extract stable session identity when available.
- Pass normalized JSON to `tts-summarizer speak`.

Example normalized output from a Codex-specific hook:

```json
{
  "caller": "codex",
  "session_id": "019ee707-a155-72d2-8261-d6b543153602",
  "event": "Stop",
  "text": "Yes, exactly! Wonderful work...",
  "metadata": {
    "cwd": "/Users/sage/workspace/test-agent",
    "turn_id": "019ee710-d30b-7cc2-816e-c3787b77b5be"
  }
}
```

The core package should include a documented request schema, not built-in harness parsers as the primary integration point.

## Testing plan

Unit tests:

- Config discovery order: explicit path, cwd, user config, defaults.
- Prompt override behavior.
- Normalized request validation.
- Session key derivation.
- Same-session cancellation replaces stale request.
- Different sessions do not cancel each other.
- Summarizer failure falls back to original text.
- Daemon state-file discovery.

Integration tests:

- `speak --text` starts daemon and posts request.
- `health` reports daemon state.
- Two same-session requests interrupt the first.
- Two different-session requests preserve both requests under the configured cross-session policy.

Manual smoke test:

```bash
tts-summarizer serve --config ./config.toml
tts-summarizer health --config ./config.toml
echo '{"caller":"manual","session_id":"a","text":"First long message"}' | tts-summarizer speak --config ./config.toml
echo '{"caller":"manual","session_id":"a","text":"Replacement message"}' | tts-summarizer speak --config ./config.toml
```

Expected: replacement interrupts the first same-session utterance.

## Open decisions

None for the design. Specific package manager, module name, and audio playback backend can be selected during implementation based on the existing environment.
