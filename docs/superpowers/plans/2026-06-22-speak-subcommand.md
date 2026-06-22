# Speak Subcommand Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tts-summarizer speak` to send text to the daemon, play returned audio, interrupt prior playback for the same explicit session id, and make local MLX audio optional.

**Architecture:** Keep behavior in `src/tts_summarizer/cli.py` because the current CLI is small and adding a second module would be ceremony. Use existing `load_config()` for TOML/defaults. Use Python `subprocess` to run `curl` and `ffplay` without shell quoting.

**Tech Stack:** Python 3.11 stdlib, existing `unittest` tests, `uv` commands, no new runtime dependencies.

## Global Constraints

- No new dependencies.
- Shell commands must be prefixed with `rtk`.
- Existing CLI convention: malformed local input/config returns `2`; daemon/playback failures print stderr and return `0`.
- `--config` defaults to `~/.config/tts-summarizer/config.toml` for `speak`, with fallback to existing config search/defaults if that file is absent.
- `--server` defaults to config host then `127.0.0.1`.
- `--port` defaults to config port then `9000` when config port is `0`.
- Session pid tracking only happens when `--session_id` is provided.

---

### Task 1: Speak parser and request execution

**Files:**
- Modify: `src/tts_summarizer/cli.py`
- Modify: `tests/test_cli_commands.py`

**Interfaces:**
- Produces: `tts-summarizer speak [--config path] [--server host] [--port port] [--session_id id] [--summarize bool] text...`
- Produces helper functions in `cli.py`: `_parse_bool(value: str) -> bool`, `_load_speak_config(path: str | None) -> Config`, `_speak(args: argparse.Namespace) -> int`

- [ ] **Step 1: Write failing parser/default test**

Replace `tests/test_cli_commands.py` with tests that monkeypatch only the subprocess boundary:

```python
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tts_summarizer import cli


class CliCommandTests(unittest.TestCase):
    def test_speak_posts_text_to_configured_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                "[server]\nhost = '127.0.0.9'\nport = 7777\nstate_dir = '"
                + tmp
                + "'\n",
                encoding="utf-8",
            )
            proc = mock.Mock()
            proc.wait.return_value = 0
            with mock.patch("tts_summarizer.cli.subprocess.Popen", return_value=proc) as popen:
                code = cli.main(["speak", "--config", str(config), "hello world"])

        self.assertEqual(code, 0)
        curl_args = popen.call_args_list[0].args[0]
        ffplay_args = popen.call_args_list[1].args[0]
        self.assertEqual(curl_args[-1], "http://127.0.0.9:7777/v1/speak")
        self.assertIn('{"text":"hello world","summarize":true}', curl_args)
        self.assertEqual(ffplay_args[:4], ["ffplay", "-nodisp", "-autoexit", "-loglevel"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test verify fails**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands -v
```

Expected: FAIL because `speak` is not implemented or does not call `subprocess.Popen`.

- [ ] **Step 3: Implement minimal speak command**

Update `src/tts_summarizer/cli.py`:

- import `json`, `subprocess`, `Path`
- add `speak` subparser with args listed above
- add `_parse_bool`, `_load_speak_config`, `_speak`
- in `main`, dispatch `args.command == "speak"` before daemon health/stop flow
- `_speak` builds curl args and ffplay args, starts `curl`, pipes stdout to `ffplay`, waits for `ffplay`, then waits for `curl`, returns `0` for subprocess errors after printing stderr

- [ ] **Step 4: Run test verify passes**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands -v
```

Expected: PASS.

---

### Task 2: Same-session interruption

**Files:**
- Modify: `src/tts_summarizer/cli.py`
- Modify: `tests/test_cli_commands.py`

**Interfaces:**
- Produces helper functions in `cli.py`: `_session_pid_path(config: Config, session_id: str) -> Path`, `_pid_alive(pid: int) -> bool`, `_stop_session(config: Config, session_id: str) -> None`, `_write_session_pid(config: Config, session_id: str, pid: int) -> Path`, `_clear_session_pid(config: Config, session_id: str, pid: int) -> None`

- [ ] **Step 1: Write failing interruption test**

Add this test to `CliCommandTests`:

```python
    def test_speak_replaces_existing_session_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                "[server]\nport = 7777\nstate_dir = '" + tmp + "'\n",
                encoding="utf-8",
            )
            session_dir = Path(tmp) / "sessions"
            session_dir.mkdir()
            (session_dir / "demo.pid").write_text("123", encoding="utf-8")
            proc = mock.Mock()
            proc.pid = 456
            proc.wait.return_value = 0
            with (
                mock.patch("tts_summarizer.cli._pid_alive", return_value=True),
                mock.patch("tts_summarizer.cli.os.kill") as kill,
                mock.patch("tts_summarizer.cli.subprocess.Popen", return_value=proc),
            ):
                code = cli.main(["speak", "--config", str(config), "--session_id", "demo", "hello"])

        self.assertEqual(code, 0)
        kill.assert_called_once()
        self.assertEqual((session_dir / "demo.pid").read_text(encoding="utf-8"), "456")
```

- [ ] **Step 2: Run test verify fails**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands -v
```

Expected: FAIL because `_pid_alive` or session pid handling is missing.

- [ ] **Step 3: Implement pid tracking**

Update `src/tts_summarizer/cli.py`:

- import `os`, `signal`
- `_session_pid_path` stores pids under `Path(config.server.state_dir).expanduser() / "sessions" / f"{session_id}.pid"`
- `_stop_session` reads the old pid, calls `os.kill(pid, signal.SIGTERM)` when `_pid_alive(pid)` is true, ignores malformed/missing pid files
- `_speak` calls `_stop_session` before starting subprocesses when `session_id` is set, writes `ffplay.pid`, and clears only if the file still contains the same pid

- [ ] **Step 4: Run tests verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands -v
```

Expected: PASS.

---

### Task 3: Optional MLX audio dependency

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces optional extra: `mlx = ["mlx-audio; platform_system == 'Darwin' and platform_machine == 'arm64'"]`

- [ ] **Step 1: Edit dependency metadata**

Move `mlx-audio; platform_system == 'Darwin' and platform_machine == 'arm64'` from `[project].dependencies` to `[project.optional-dependencies].mlx`.

- [ ] **Step 2: Run metadata/build check**

Run:

```bash
rtk uv build
```

Expected: build succeeds.

---

### Task 4: Focused verification

**Files:**
- Verify only

- [ ] **Step 1: Run CLI and metadata checks**

Run:

```bash
rtk uv run python -m unittest tests.test_cli tests.test_cli_commands -v
rtk uv build
```

Expected: all tests pass and build succeeds.

- [ ] **Step 2: Review final status**

Run:

```bash
rtk git status --short
```

Expected: changes only in `src/tts_summarizer/cli.py`, `tests/test_cli_commands.py`, `pyproject.toml`, and this plan file.
