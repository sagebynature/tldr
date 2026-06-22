# Hermes Shell Hook Design

## Goal

Add Hermes Agent support using its shell-hook system so final assistant responses can be spoken by `tts-summarizer` without changing visible chat output or blocking Hermes.

## Approved Scope

- Add one Hermes shell-hook script.
- Install it under `~/.hermes/agent-hooks/tts-summarizer/`.
- Register it in `~/.hermes/config.yaml` under `hooks.post_llm_call`.
- Keep hook execution non-blocking and fail-open.
- Add tests for hook behavior and installer idempotency.

## Out of Scope

- Gateway `HOOK.yaml` hooks under `~/.hermes/hooks/`.
- Plugin hooks.
- Transforming or replacing the assistant's visible response.
- Adding a YAML dependency just for config editing.
- Managing Hermes consent settings beyond documenting the registered command.

## Current Architecture

Existing harness support is file-based:

- `hooks/codex/codex_tts.py`
- `hooks/claude/claude_tts.py`
- `hooks/omp/tts.ts`
- `hooks/pi/tts.ts`

`src/tts_summarizer/installer.py` maps harness names to hook files, copies hooks into the user's home directory, and updates each harness config. Tests in `tests/test_hooks.py` exercise the hook scripts with stub `tts-summarizer` binaries.

## Hermes Hook Shape

Hermes shell hooks are configured in `~/.hermes/config.yaml`:

```yaml
hooks:
  post_llm_call:
    - command: "/Users/me/.hermes/agent-hooks/tts-summarizer/hermes_tts.py"
      timeout: 5
```

Hermes pipes JSON into stdin and reads JSON from stdout. The hook must print `{}` for no-op success. The selected event is `post_llm_call` because it fires once per turn after the tool-calling loop and its return value is ignored.

## Hook Behavior

Add `hooks/hermes/hermes_tts.py`:

1. Read stdin as JSON.
2. Ignore payloads where `hook_event_name != "post_llm_call"`.
3. Extract speech text from `extra.assistant_response` when it is a non-empty string.
4. Fallback to the last assistant entry in `extra.conversation_history` when available.
5. Build session id as `hermes:<session_id>` if present; otherwise `hermes:<cwd>`; otherwise `hermes`.
6. Spawn `tts-summarizer speak --session_id <session_id> <text>` using `subprocess.Popen` with ignored stdio and detached session where supported.
7. Print `{}` and exit zero.

Errors are swallowed after best-effort stderr logging. Missing `tts-summarizer`, malformed JSON, absent response text, and spawn failures must not break Hermes.

## Installer Behavior

Extend `installer.py`:

- Add `hermes` to hook filename mapping.
- Add `_install_hermes(home: Path) -> Path`.
- Copy `hooks/hermes/hermes_tts.py` to `~/.hermes/agent-hooks/tts-summarizer/hermes_tts.py`.
- `chmod +x` the installed script.
- Touch `~/.hermes/tts.enabled` for parity with other harnesses.
- Update `~/.hermes/config.yaml` idempotently.

No new YAML library. The installer should support the common generated shape by appending a small block when no matching command is present. If an existing config already contains the command, leave it unchanged.

## Packaging

Include `hooks/hermes/hermes_tts.py` in wheel resources so installed packages can copy the hook without the source tree.

## Testing

Add focused tests:

- Hermes hook speaks `extra.assistant_response`.
- Hermes hook falls back to last assistant message in `extra.conversation_history`.
- Hermes hook ignores other hook events.
- Hermes hook exits before delayed speech finishes.
- Installer writes the Hermes hook and a `post_llm_call` config entry.
- Installer is idempotent and does not duplicate the command.

Use the existing stub binary pattern from `tests/test_hooks.py`. No mocks unless subprocess boundary makes real execution impractical.

## Risks

- Hermes config may already have complex YAML. The lazy safe path is append-only text for the simple missing-entry case and exact command detection for idempotency. Add real YAML editing only if users need preserving arbitrary complex configs.
- `post_llm_call` payload field names may differ by Hermes version. The hook includes the conversation-history fallback to reduce that risk without adding complexity.
