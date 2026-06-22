# Speak Subcommand Design

## Goal

Add a minimal `tts-summarizer speak` CLI path that sends text to the running daemon, plays returned WAV bytes through `ffplay`, and interrupts any previous playback for the same explicit session id.

## Command shape

```bash
tts-summarizer speak \
  [--config ~/.config/tts-summarizer/config.toml] \
  [--server 127.0.0.1] \
  [--port 9000] \
  [--session_id demo] \
  [--summarize true] \
  "text to summarize"
```

Arguments:

- `--config`: optional config path. Default for this command is `~/.config/tts-summarizer/config.toml`; if missing, existing built-in defaults apply.
- `--server`: optional host override. Default is `config.server.host`, then `127.0.0.1`.
- `--port`: optional port override. Default is `config.server.port`, then `9000` when config has `0` or no usable value.
- `--session_id`: optional. When present, it becomes `X-TTS-Session-Id` and enables same-session interruption.
- `--summarize`: optional bool string, default `true`. Accept `true/false`, `1/0`, `yes/no`, `on/off`.
- `text_to_summarize`: required positional text. Multiple words join as one string.

## Runtime behavior

`speak` builds the same request the manual curl example uses:

```json
{"text":"...","summarize":true}
```

Headers:

- `Content-Type: application/json`
- `X-TTS-Caller: manual`
- `X-TTS-Session-Id: <session_id>` only when provided

The command pipes daemon WAV bytes to:

```bash
ffplay -nodisp -autoexit -loglevel error -i pipe:0
```

Implementation may use Python `subprocess` rather than shell parsing. No new dependency is needed.

## Session interruption

When `--session_id` is provided:

1. Resolve pid file under `<state_dir>/sessions/<session_id>.pid`.
2. If pid file contains a live pid, terminate that pid before starting new playback.
3. Start the new `curl | ffplay` pipeline.
4. Write the new process-group pid to the session pid file.
5. Remove the pid file when playback exits and it still points at the same pid.

If `--session_id` is omitted, skip pid tracking and interruption. That keeps anonymous one-shot playback simple and avoids inventing session identity.

## Config and defaults

Use existing `load_config()` to avoid a second TOML parser. For `speak`, pass the default config path explicitly when `--config` is not supplied. If that file is absent, fall back to `load_config(None)` so local `./config.toml` and built-in defaults still work.

Port fallback intentionally differs from `ServerConfig.port = 0`: the user-facing manual client defaults to `9000` when no configured server port exists.

## Dependency metadata

Move `mlx-audio` from required dependencies to an optional extra because remote TTS can run without local MLX audio. Keep Darwin/arm64 marker on the extra. Do not add new dependencies.

Suggested extras:

```toml
[project.optional-dependencies]
mlx = ["mlx-audio; platform_system == 'Darwin' and platform_machine == 'arm64'"]
kokoro = ["misaki", "num2words", "spacy", "espeakng-loader", "phonemizer-fork"]
```

This preserves a local-MLX install path while keeping remote-only installs lighter.

## Errors

Follow existing CLI convention: local malformed input/config returns `2`; daemon/curl/playback failures print to stderr and return `0` so hook callers are not broken by speech failures.

## Tests

Add the smallest behavior tests:

- parser/default resolution for `speak` including default summarize true and port fallback.
- same-session pid interruption kills an existing live pid and replaces the pid file.

No end-to-end audio test; that would require daemon, curl, and ffplay availability and would be brittle.
