# Standalone TTS CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a harness-neutral Python CLI and loopback HTTP daemon that keeps MLX summarizer and TTS models warm and interrupts stale speech from the same caller session.

**Architecture:** A stdlib-first Python package exposes `tts-summarizer` commands. `speak` accepts normalized requests, discovers or starts a loopback daemon, and posts to `/v1/speak`; the daemon owns config, session cancellation, summarization, TTS generation, and audio playback. MLX wrappers are thin and injectable so tests run without loading real models.

**Tech Stack:** Python 3.11+, stdlib `argparse`, `dataclasses`, `http.server`, `json`, `subprocess`, `threading`, `tomllib`, `urllib`; runtime MLX dependencies `mlx-lm`, `mlx-audio`, `numpy`.

## Global Constraints

- Core service must be harness-neutral; no Codex/Claude payload parsing in package internals.
- Harness adapters must send normalized JSON: `text`, optional `session_id`, `caller`, `event`, `metadata`.
- Loopback HTTP only; bind to `127.0.0.1`.
- Docker/local container hosting is out of scope; production local service runs natively on macOS for MLX/Metal.
- Config lookup order is exact: explicit `--config`, then `./config.toml`, then `~/.config/tts-summarizer/config.toml`, then built-in defaults.
- Summarizer and TTS may use separate models.
- Summarizer system prompt must have a good default and be overridable in TOML.
- Same `caller:session_id` requests interrupt current speech for that session.
- Different sessions must not cancel each other by default.
- Hook/client failures should not fail the calling harness unless the user is running validation commands like `config-check`.
- Workspace is currently not a git repository; skip commit steps unless a repo is initialized before implementation.

## File Structure

- Create `pyproject.toml`: package metadata, console script, runtime dependencies.
- Create `src/tts_summarizer/__init__.py`: package version.
- Create `src/tts_summarizer/__main__.py`: `python -m tts_summarizer` entrypoint.
- Create `src/tts_summarizer/cli.py`: argparse commands and process exit behavior.
- Create `src/tts_summarizer/config.py`: TOML discovery, defaults, dataclasses, validation.
- Create `src/tts_summarizer/request.py`: normalized request parsing and session key derivation.
- Create `src/tts_summarizer/state.py`: daemon state file read/write and stale PID handling.
- Create `src/tts_summarizer/client.py`: daemon discovery, auto-start, HTTP calls.
- Create `src/tts_summarizer/server.py`: loopback HTTP daemon endpoints.
- Create `src/tts_summarizer/session.py`: per-session cancellation and queue policy.
- Create `src/tts_summarizer/summarizer.py`: MLX-LM summarizer wrapper with injectable fake.
- Create `src/tts_summarizer/speech.py`: MLX-Audio TTS wrapper with injectable fake.
- Create `src/tts_summarizer/audio.py`: WAV writing and macOS `afplay` playback process management.
- Create `config.example.toml`: complete example config matching defaults.
- Create tests under `tests/` using stdlib `unittest` only.

---

### Task 1: Package scaffold and CLI shell

**Files:**
- Create: `pyproject.toml`
- Create: `src/tts_summarizer/__init__.py`
- Create: `src/tts_summarizer/__main__.py`
- Create: `src/tts_summarizer/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `tts_summarizer.cli.main(argv: list[str] | None = None) -> int`
- Produces: console script `tts-summarizer = tts_summarizer.cli:main`
- Later tasks add real command handlers behind this shell.

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_cli.py`:

```python
import unittest
from tts_summarizer.cli import main


class CliTests(unittest.TestCase):
    def test_config_check_command_exists(self):
        self.assertEqual(main(["config-check"]), 0)

    def test_unknown_command_fails(self):
        self.assertNotEqual(main(["not-a-command"]), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_cli -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tts_summarizer'`.

- [ ] **Step 3: Add minimal package files**

Create `pyproject.toml`:

```toml
[project]
name = "tts-summarizer"
version = "0.1.0"
description = "Harness-neutral local TTS summarizer daemon using MLX"
requires-python = ">=3.11"
dependencies = [
  "mlx-lm",
  "mlx-audio",
  "numpy",
]

[project.scripts]
tts-summarizer = "tts_summarizer.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

Create `src/tts_summarizer/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `src/tts_summarizer/__main__.py`:

```python
from .cli import main

raise SystemExit(main())
```

Create `src/tts_summarizer/cli.py`:

```python
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tts-summarizer")
    subcommands = parser.add_subparsers(dest="command", required=True)

    config_check = subcommands.add_parser("config-check")
    config_check.add_argument("--config")

    speak = subcommands.add_parser("speak")
    speak.add_argument("--config")
    speak.add_argument("--session-id")
    speak.add_argument("--caller")
    speak.add_argument("--text")

    serve = subcommands.add_parser("serve")
    serve.add_argument("--config")

    health = subcommands.add_parser("health")
    health.add_argument("--config")

    stop = subcommands.add_parser("stop")
    stop.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command == "config-check":
        return 0
    return 0
```

- [ ] **Step 4: Run test and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_cli -v
```

Expected: PASS both tests.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add pyproject.toml src/tts_summarizer tests/test_cli.py
rtk git commit -m "feat: scaffold tts summarizer cli"
```

If not a git repository, skip commit.

---

### Task 2: Config discovery and validation

**Files:**
- Create: `src/tts_summarizer/config.py`
- Modify: `src/tts_summarizer/cli.py`
- Create: `config.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(explicit_path: str | None, cwd: Path | None = None, home: Path | None = None) -> Config`
- Produces: `Config` dataclass with `.server`, `.session`, `.summarizer`, `.tts`, `.audio`.
- Consumes: CLI `config-check` calls `load_config` and returns nonzero on invalid explicit config.

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config.py`:

```python
import tempfile
import unittest
from pathlib import Path

from tts_summarizer.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_load_without_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(None, cwd=root / "cwd", home=root / "home")
        self.assertEqual(cfg.server.host, "127.0.0.1")
        self.assertEqual(cfg.summarizer.max_words, 40)
        self.assertIn("text-to-speech", cfg.summarizer.system_prompt)

    def test_cwd_config_beats_user_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "cwd"
            home = root / "home"
            cwd.mkdir()
            (home / ".config" / "tts-summarizer").mkdir(parents=True)
            (home / ".config" / "tts-summarizer" / "config.toml").write_text(
                '[tts]\nvoice = "UserVoice"\n', encoding="utf-8"
            )
            (cwd / "config.toml").write_text('[tts]\nvoice = "CwdVoice"\n', encoding="utf-8")
            cfg = load_config(None, cwd=cwd, home=home)
        self.assertEqual(cfg.tts.voice, "CwdVoice")

    def test_explicit_missing_config_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.toml"
            with self.assertRaises(ConfigError):
                load_config(str(missing), cwd=Path(tmp), home=Path(tmp))

    def test_prompt_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[summarizer]\nsystem_prompt = "Speak plainly."\n', encoding="utf-8")
            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))
        self.assertEqual(cfg.summarizer.system_prompt, "Speak plainly.")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_config -v
```

Expected: FAIL with `ModuleNotFoundError` or missing `tts_summarizer.config`.

- [ ] **Step 3: Implement config loader**

Create `src/tts_summarizer/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import tomllib


class ConfigError(ValueError):
    pass


DEFAULT_SYSTEM_PROMPT = """You summarize assistant responses for text-to-speech.
Return only a spoken summary.
Do not mention that this is a summary.
If the content is a question, preserve the question instead of answering it.
Do not include markdown, code fences, file paths, URLs, bullets, or formatting."""

DEFAULT_USER_PROMPT_TEMPLATE = """Summarize this response in {max_words} words or fewer.
Preserve the practical outcome and next action.

{text}"""


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    state_dir: str = "~/.cache/tts-summarizer"
    auto_start: bool = True
    startup_timeout_ms: int = 3000
    request_timeout_ms: int = 5000


@dataclass(frozen=True)
class SessionConfig:
    interrupt_same_session: bool = True
    max_queue_per_session: int = 1
    cross_session_policy: str = "queue"


@dataclass(frozen=True)
class SummarizerConfig:
    enabled: bool = True
    model: str = "mlx-community/Qwen3-0.6B-4bit"
    word_threshold: int = 0
    max_words: int = 40
    temperature: float = 0.2
    max_tokens: int = 180
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE


@dataclass(frozen=True)
class TtsConfig:
    model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    voice: str = "Chelsie"
    lang_code: str = "English"
    speed: float = 1.6
    ref_audio: str = ""
    ref_text: str = ""
    stream: bool = True
    sample_rate: int = 24000


@dataclass(frozen=True)
class AudioConfig:
    backend: str = "auto"
    output_dir: str = "~/.cache/tts-summarizer/audio"
    save: bool = False


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    source: Path | None = None


def _expand(path: str, *, home: Path | None = None) -> Path:
    if path.startswith("~/") and home is not None:
        return home / path[2:]
    return Path(path).expanduser()


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid = instance.__dataclass_fields__.keys()
    unknown = sorted(set(values) - set(valid))
    if unknown:
        raise ConfigError(f"unknown config keys for {type(instance).__name__}: {', '.join(unknown)}")
    return replace(instance, **values)


def _apply(raw: dict[str, Any], source: Path | None) -> Config:
    cfg = Config(source=source)
    allowed = {"server", "session", "summarizer", "tts", "audio"}
    unknown_sections = sorted(set(raw) - allowed)
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {', '.join(unknown_sections)}")
    return Config(
        server=_merge_dataclass(cfg.server, raw.get("server", {})),
        session=_merge_dataclass(cfg.session, raw.get("session", {})),
        summarizer=_merge_dataclass(cfg.summarizer, raw.get("summarizer", {})),
        tts=_merge_dataclass(cfg.tts, raw.get("tts", {})),
        audio=_merge_dataclass(cfg.audio, raw.get("audio", {})),
        source=source,
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def load_config(explicit_path: str | None, cwd: Path | None = None, home: Path | None = None) -> Config:
    cwd = cwd or Path.cwd()
    home = home or Path.home()
    if explicit_path:
        path = _expand(explicit_path, home=home)
        if not path.exists():
            raise ConfigError(f"config not found: {path}")
        return _apply(_read(path), path)

    cwd_config = cwd / "config.toml"
    if cwd_config.exists():
        return _apply(_read(cwd_config), cwd_config)

    user_config = home / ".config" / "tts-summarizer" / "config.toml"
    if user_config.exists():
        return _apply(_read(user_config), user_config)

    return Config()
```

Modify `src/tts_summarizer/cli.py` so `config-check` validates config:

```python
from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tts-summarizer")
    subcommands = parser.add_subparsers(dest="command", required=True)

    config_check = subcommands.add_parser("config-check")
    config_check.add_argument("--config")

    speak = subcommands.add_parser("speak")
    speak.add_argument("--config")
    speak.add_argument("--session-id")
    speak.add_argument("--caller")
    speak.add_argument("--text")

    serve = subcommands.add_parser("serve")
    serve.add_argument("--config")

    health = subcommands.add_parser("health")
    health.add_argument("--config")

    stop = subcommands.add_parser("stop")
    stop.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command == "config-check":
        try:
            load_config(args.config)
        except ConfigError as exc:
            print(f"tts-summarizer config error: {exc}", file=sys.stderr)
            return 2
        return 0
    return 0
```

Create `config.example.toml` from the spec, including `tts.sample_rate = 24000`.

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_config tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/config.py src/tts_summarizer/cli.py tests/test_config.py config.example.toml
rtk git commit -m "feat: add config discovery"
```

If not a git repository, skip commit.

---

### Task 3: Normalized request parsing and session keys

**Files:**
- Create: `src/tts_summarizer/request.py`
- Modify: `src/tts_summarizer/cli.py`
- Test: `tests/test_request.py`

**Interfaces:**
- Produces: `SpeechRequest.from_json(data: dict[str, object]) -> SpeechRequest`
- Produces: `SpeechRequest.from_cli(text: str | None, stdin_text: str, caller: str | None, session_id: str | None) -> SpeechRequest`
- Produces: `SpeechRequest.session_key() -> str`
- Consumes: CLI `speak` builds `SpeechRequest` before client posting in later tasks.

- [ ] **Step 1: Write failing request tests**

Create `tests/test_request.py`:

```python
import json
import unittest

from tts_summarizer.request import RequestError, SpeechRequest


class RequestTests(unittest.TestCase):
    def test_json_request_keeps_session_identity(self):
        req = SpeechRequest.from_json({"text": "hello", "caller": "manual", "session_id": "abc"})
        self.assertEqual(req.text, "hello")
        self.assertEqual(req.session_key(), "manual:abc")

    def test_missing_text_fails(self):
        with self.assertRaises(RequestError):
            SpeechRequest.from_json({"caller": "manual"})

    def test_cli_text_wins_over_stdin(self):
        req = SpeechRequest.from_cli("from arg", '{"text":"from stdin"}', "cli", "s1")
        self.assertEqual(req.text, "from arg")
        self.assertEqual(req.session_key(), "cli:s1")

    def test_stdin_json_supported(self):
        payload = json.dumps({"text": "from stdin", "caller": "hook", "session_id": "s2"})
        req = SpeechRequest.from_cli(None, payload, None, None)
        self.assertEqual(req.text, "from stdin")
        self.assertEqual(req.session_key(), "hook:s2")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run request tests and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_request -v
```

Expected: FAIL because `tts_summarizer.request` does not exist.

- [ ] **Step 3: Implement request parsing**

Create `src/tts_summarizer/request.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os


class RequestError(ValueError):
    pass


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    session_id: str
    caller: str = "default"
    event: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "SpeechRequest":
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RequestError("normalized request requires non-empty text")
        caller = data.get("caller")
        session_id = data.get("session_id")
        event = data.get("event")
        metadata = data.get("metadata")
        return cls(
            text=text,
            caller=caller if isinstance(caller, str) and caller else "default",
            session_id=session_id if isinstance(session_id, str) and session_id else fallback_session_id(),
            event=event if isinstance(event, str) else "",
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    @classmethod
    def from_cli(
        cls,
        text: str | None,
        stdin_text: str,
        caller: str | None,
        session_id: str | None,
    ) -> "SpeechRequest":
        if text is not None:
            return cls(
                text=text,
                caller=caller or "default",
                session_id=session_id or fallback_session_id(),
            )
        stripped = stdin_text.strip()
        if not stripped:
            raise RequestError("provide --text or normalized JSON on stdin")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = {"text": stdin_text}
        if not isinstance(payload, dict):
            raise RequestError("stdin JSON must be an object")
        if caller is not None:
            payload["caller"] = caller
        if session_id is not None:
            payload["session_id"] = session_id
        return cls.from_json(payload)

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "caller": self.caller,
            "event": self.event,
            "metadata": self.metadata,
        }

    def session_key(self) -> str:
        return f"{self.caller}:{self.session_id}"


def fallback_session_id() -> str:
    cwd = Path.cwd()
    ppid = os.getppid()
    return f"{cwd}:{ppid}"
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_request tests.test_config tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/request.py tests/test_request.py
rtk git commit -m "feat: add normalized speech requests"
```

If not a git repository, skip commit.

---

### Task 4: Daemon state and HTTP client lifecycle

**Files:**
- Create: `src/tts_summarizer/state.py`
- Create: `src/tts_summarizer/client.py`
- Modify: `src/tts_summarizer/cli.py`
- Test: `tests/test_state_client.py`

**Interfaces:**
- Produces: `DaemonState(host: str, port: int, pid: int, config_fingerprint: str)`
- Produces: `write_state(config: Config, host: str, port: int, pid: int) -> Path`
- Produces: `read_state(config: Config) -> DaemonState | None`
- Produces: `post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]`
- Later server task will call `write_state` after binding a port.

- [ ] **Step 1: Write failing state tests**

Create `tests/test_state_client.py`:

```python
import tempfile
import unittest
from pathlib import Path

from tts_summarizer.config import load_config
from tts_summarizer.state import read_state, write_state


class StateClientTests(unittest.TestCase):
    def test_write_and_read_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.toml"
            cfg_path.write_text(f'[server]\nstate_dir = "{tmp}/state"\n', encoding="utf-8")
            cfg = load_config(str(cfg_path), cwd=Path(tmp), home=Path(tmp))
            write_state(cfg, "127.0.0.1", 4321, 12345)
            state = read_state(cfg)
        self.assertIsNotNone(state)
        self.assertEqual(state.host, "127.0.0.1")
        self.assertEqual(state.port, 4321)
        self.assertEqual(state.pid, 12345)

    def test_missing_state_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(None, cwd=Path(tmp), home=Path(tmp))
            self.assertIsNone(read_state(cfg))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run state tests and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_state_client -v
```

Expected: FAIL because `tts_summarizer.state` does not exist.

- [ ] **Step 3: Implement state and client helpers**

Create `src/tts_summarizer/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import time

from .config import Config


@dataclass(frozen=True)
class DaemonState:
    host: str
    port: int
    pid: int
    config_fingerprint: str
    started_at: float

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def state_path(config: Config) -> Path:
    return Path(config.server.state_dir).expanduser() / "daemon.json"


def config_fingerprint(config: Config) -> str:
    basis = repr((config.server, config.session, config.summarizer, config.tts, config.audio)).encode("utf-8")
    return hashlib.sha256(basis).hexdigest()[:16]


def write_state(config: Config, host: str, port: int, pid: int) -> Path:
    path = state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": host,
        "port": port,
        "pid": pid,
        "config_fingerprint": config_fingerprint(config),
        "started_at": time.time(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_state(config: Config) -> DaemonState | None:
    path = state_path(config)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = DaemonState(
            host=str(payload["host"]),
            port=int(payload["port"]),
            pid=int(payload["pid"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            started_at=float(payload["started_at"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    if not _pid_exists(state.pid):
        path.unlink(missing_ok=True)
        return None
    return state
```

Create `src/tts_summarizer/client.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .config import Config
from .state import read_state


class ClientError(RuntimeError):
    pass


def post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def get_json(url: str, timeout: float) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def start_daemon(config_path: str | None) -> None:
    args = [sys.executable, "-m", "tts_summarizer", "serve"]
    if config_path:
        args.extend(["--config", config_path])
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def wait_for_state(config: Config) -> str | None:
    deadline = time.monotonic() + config.server.startup_timeout_ms / 1000
    while time.monotonic() < deadline:
        state = read_state(config)
        if state is not None:
            return state.base_url
        time.sleep(0.05)
    return None


def daemon_base_url(config: Config, config_path: str | None) -> str | None:
    state = read_state(config)
    if state is not None:
        return state.base_url
    if not config.server.auto_start:
        return None
    start_daemon(config_path)
    return wait_for_state(config)
```

- [ ] **Step 4: Run state/client tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_state_client tests.test_config tests.test_request tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/state.py src/tts_summarizer/client.py tests/test_state_client.py
rtk git commit -m "feat: add daemon state discovery"
```

If not a git repository, skip commit.

---

### Task 5: Session cancellation manager

**Files:**
- Create: `src/tts_summarizer/session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Produces: `SessionManager.begin(request: SpeechRequest) -> WorkToken`
- Produces: `WorkToken.cancelled() -> bool`
- Produces: `SessionManager.finish(token: WorkToken) -> None`
- Later server task uses `begin` before summarization/TTS and checks token during playback.

- [ ] **Step 1: Write failing session tests**

Create `tests/test_session.py`:

```python
import unittest

from tts_summarizer.config import Config
from tts_summarizer.request import SpeechRequest
from tts_summarizer.session import SessionManager


class SessionTests(unittest.TestCase):
    def test_same_session_interrupts_previous_token(self):
        manager = SessionManager(Config().session)
        first = manager.begin(SpeechRequest(text="one", caller="c", session_id="s"))
        second = manager.begin(SpeechRequest(text="two", caller="c", session_id="s"))
        self.assertTrue(first.cancelled())
        self.assertFalse(second.cancelled())

    def test_different_session_does_not_interrupt(self):
        manager = SessionManager(Config().session)
        first = manager.begin(SpeechRequest(text="one", caller="c", session_id="s1"))
        second = manager.begin(SpeechRequest(text="two", caller="c", session_id="s2"))
        self.assertFalse(first.cancelled())
        self.assertFalse(second.cancelled())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run session tests and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_session -v
```

Expected: FAIL because `tts_summarizer.session` does not exist.

- [ ] **Step 3: Implement session manager**

Create `src/tts_summarizer/session.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import threading

from .config import SessionConfig
from .request import SpeechRequest


@dataclass(frozen=True)
class WorkToken:
    session_key: str
    generation: int
    manager: "SessionManager"

    def cancelled(self) -> bool:
        return self.manager.current_generation(self.session_key) != self.generation


class SessionManager:
    def __init__(self, config: SessionConfig):
        self.config = config
        self._lock = threading.Lock()
        self._generations: dict[str, int] = {}

    def begin(self, request: SpeechRequest) -> WorkToken:
        key = request.session_key()
        with self._lock:
            current = self._generations.get(key, 0)
            next_generation = current + 1 if self.config.interrupt_same_session else current or 1
            self._generations[key] = next_generation
            return WorkToken(key, next_generation, self)

    def finish(self, token: WorkToken) -> None:
        # Keep the generation number so stale workers can still observe cancellation.
        return None

    def current_generation(self, session_key: str) -> int:
        with self._lock:
            return self._generations.get(session_key, 0)
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_session tests.test_state_client tests.test_request tests.test_config tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/session.py tests/test_session.py
rtk git commit -m "feat: add session cancellation"
```

If not a git repository, skip commit.

---

### Task 6: Summarizer wrapper

**Files:**
- Create: `src/tts_summarizer/summarizer.py`
- Test: `tests/test_summarizer.py`

**Interfaces:**
- Produces: `Summarizer.summarize(text: str) -> str`
- Produces: `count_words(text: str) -> int`
- Consumes: `Config.summarizer`.
- Later server task uses `Summarizer` before speech generation.

- [ ] **Step 1: Write failing summarizer tests with fake generator**

Create `tests/test_summarizer.py`:

```python
import unittest

from tts_summarizer.config import SummarizerConfig
from tts_summarizer.summarizer import Summarizer, count_words


class FakeBackend:
    def __init__(self):
        self.prompt = ""

    def generate(self, messages, config):
        self.prompt = messages[-1]["content"]
        return "short result"


class SummarizerTests(unittest.TestCase):
    def test_count_words(self):
        self.assertEqual(count_words("one two\nthree"), 3)

    def test_threshold_skips_model(self):
        backend = FakeBackend()
        summarizer = Summarizer(SummarizerConfig(word_threshold=10), backend=backend)
        self.assertEqual(summarizer.summarize("short text"), "short text")
        self.assertEqual(backend.prompt, "")

    def test_prompt_template_used(self):
        backend = FakeBackend()
        config = SummarizerConfig(word_threshold=0, user_prompt_template="Limit {max_words}: {text}")
        summarizer = Summarizer(config, backend=backend)
        self.assertEqual(summarizer.summarize("long enough"), "short result")
        self.assertEqual(backend.prompt, "Limit 40: long enough")

    def test_backend_failure_returns_original_text(self):
        class BrokenBackend:
            def generate(self, messages, config):
                raise RuntimeError("boom")

        summarizer = Summarizer(SummarizerConfig(word_threshold=0), backend=BrokenBackend())
        self.assertEqual(summarizer.summarize("keep this"), "keep this")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run summarizer tests and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_summarizer -v
```

Expected: FAIL because `tts_summarizer.summarizer` does not exist.

- [ ] **Step 3: Implement summarizer wrapper**

Create `src/tts_summarizer/summarizer.py`:

```python
from __future__ import annotations

from typing import Protocol
import sys

from .config import SummarizerConfig


class SummaryBackend(Protocol):
    def generate(self, messages: list[dict[str, str]], config: SummarizerConfig) -> str: ...


class MlxLmBackend:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_name = ""

    def _load(self, model_name: str):
        if self._model is not None and self._model_name == model_name:
            return self._model, self._tokenizer
        from mlx_lm import generate, load  # type: ignore

        model, tokenizer = load(model_name)
        self._generate = generate
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        return model, tokenizer

    def generate(self, messages: list[dict[str, str]], config: SummarizerConfig) -> str:
        model, tokenizer = self._load(config.model)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self._generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            verbose=False,
        ).strip()


class Summarizer:
    def __init__(self, config: SummarizerConfig, backend: SummaryBackend | None = None):
        self.config = config
        self.backend = backend or MlxLmBackend()

    def summarize(self, text: str) -> str:
        if not self.config.enabled:
            return text
        if count_words(text) <= self.config.word_threshold:
            return text
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": self.config.user_prompt_template.format(
                    max_words=self.config.max_words,
                    text=text,
                ),
            },
        ]
        try:
            summary = self.backend.generate(messages, self.config).strip()
        except Exception as exc:
            print(f"tts-summarizer summary failed: {exc}", file=sys.stderr)
            return text
        return summary or text


def count_words(text: str) -> int:
    return len(text.split())
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_summarizer tests.test_session tests.test_request tests.test_config tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/summarizer.py tests/test_summarizer.py
rtk git commit -m "feat: add mlx summarizer wrapper"
```

If not a git repository, skip commit.

---

### Task 7: Speech generation and audio playback

**Files:**
- Create: `src/tts_summarizer/speech.py`
- Create: `src/tts_summarizer/audio.py`
- Test: `tests/test_speech_audio.py`

**Interfaces:**
- Produces: `SpeechGenerator.generate(text: str) -> list[AudioChunk]`
- Produces: `AudioChunk(samples: object, sample_rate: int)`
- Produces: `AudioPlayer.play(chunks: Iterable[AudioChunk], token: WorkToken | None = None) -> None`
- Later server task uses `SpeechGenerator` and `AudioPlayer`.

- [ ] **Step 1: Write failing speech/audio tests with fake model**

Create `tests/test_speech_audio.py`:

```python
import tempfile
import unittest
from pathlib import Path

from tts_summarizer.audio import AudioPlayer
from tts_summarizer.config import AudioConfig, TtsConfig
from tts_summarizer.speech import AudioChunk, SpeechGenerator


class FakeBackend:
    def generate(self, text, config):
        return [AudioChunk(samples=[0.0, 0.25, -0.25], sample_rate=8000)]


class SpeechAudioTests(unittest.TestCase):
    def test_speech_generator_passes_text(self):
        generator = SpeechGenerator(TtsConfig(sample_rate=8000), backend=FakeBackend())
        chunks = generator.generate("hello")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].sample_rate, 8000)

    def test_audio_player_file_backend_writes_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            player = AudioPlayer(AudioConfig(backend="file", output_dir=tmp, save=True))
            player.play([AudioChunk(samples=[0.0, 0.5, -0.5], sample_rate=8000)])
            files = list(Path(tmp).glob("*.wav"))
        self.assertEqual(len(files), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run speech/audio tests and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_speech_audio -v
```

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement speech and audio modules**

Create `src/tts_summarizer/speech.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import TtsConfig


@dataclass(frozen=True)
class AudioChunk:
    samples: object
    sample_rate: int


class SpeechBackend(Protocol):
    def generate(self, text: str, config: TtsConfig) -> list[AudioChunk]: ...


class MlxAudioBackend:
    def __init__(self):
        self._model = None
        self._model_name = ""

    def _load(self, model_name: str):
        if self._model is not None and self._model_name == model_name:
            return self._model
        from mlx_audio.tts.utils import load_model  # type: ignore

        self._model = load_model(model_name)
        self._model_name = model_name
        return self._model

    def generate(self, text: str, config: TtsConfig) -> list[AudioChunk]:
        model = self._load(config.model)
        kwargs = {
            "voice": config.voice or None,
            "lang_code": config.lang_code or None,
            "speed": config.speed,
            "ref_audio": config.ref_audio or None,
            "ref_text": config.ref_text or None,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        chunks: list[AudioChunk] = []
        for result in model.generate(text=text, **kwargs):
            sample_rate = int(getattr(result, "sample_rate", getattr(model, "sample_rate", config.sample_rate)))
            chunks.append(AudioChunk(samples=result.audio, sample_rate=sample_rate))
        return chunks


class SpeechGenerator:
    def __init__(self, config: TtsConfig, backend: SpeechBackend | None = None):
        self.config = config
        self.backend = backend or MlxAudioBackend()

    def generate(self, text: str) -> list[AudioChunk]:
        return self.backend.generate(text, self.config)
```

Create `src/tts_summarizer/audio.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math
import subprocess
import time
import wave

from .config import AudioConfig
from .session import WorkToken
from .speech import AudioChunk


class AudioPlayer:
    def __init__(self, config: AudioConfig):
        self.config = config

    def play(self, chunks: Iterable[AudioChunk], token: WorkToken | None = None) -> None:
        output_dir = Path(self.config.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            if token is not None and token.cancelled():
                return
            path = output_dir / f"speech-{time.time_ns()}.wav"
            write_wav(path, chunk)
            if self.config.backend in {"auto", "afplay"} and not self.config.save:
                proc = subprocess.Popen(["/usr/bin/afplay", str(path)])
                while proc.poll() is None:
                    if token is not None and token.cancelled():
                        proc.terminate()
                        return
                    time.sleep(0.05)
            if self.config.backend == "file" or self.config.save:
                continue


def write_wav(path: Path, chunk: AudioChunk) -> None:
    samples = _to_float_list(chunk.samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(chunk.sample_rate)
        wav.writeframes(b"".join(_to_i16(sample) for sample in samples))


def _to_float_list(samples: object) -> list[float]:
    if hasattr(samples, "tolist"):
        raw = samples.tolist()
    else:
        raw = samples
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = [item for row in raw for item in row]
    return [float(item) for item in raw]


def _to_i16(sample: float) -> bytes:
    value = max(-1.0, min(1.0, sample if math.isfinite(sample) else 0.0))
    return int(value * 32767).to_bytes(2, byteorder="little", signed=True)
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_speech_audio tests.test_summarizer tests.test_session tests.test_request tests.test_config tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/speech.py src/tts_summarizer/audio.py tests/test_speech_audio.py
rtk git commit -m "feat: add speech and audio playback"
```

If not a git repository, skip commit.

---

### Task 8: Loopback HTTP daemon

**Files:**
- Create: `src/tts_summarizer/server.py`
- Modify: `src/tts_summarizer/cli.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Produces: `run_server(config: Config) -> int`
- Produces: `TtsService.handle(request: SpeechRequest) -> dict[str, object]`
- Consumes: `SessionManager`, `Summarizer`, `SpeechGenerator`, `AudioPlayer`, `write_state`.

- [ ] **Step 1: Write failing service test using fakes**

Create `tests/test_server.py`:

```python
import unittest

from tts_summarizer.config import Config
from tts_summarizer.request import SpeechRequest
from tts_summarizer.server import TtsService
from tts_summarizer.speech import AudioChunk


class FakeSummarizer:
    def summarize(self, text):
        return f"summary: {text}"


class FakeSpeech:
    def generate(self, text):
        return [AudioChunk(samples=[0.0], sample_rate=8000)]


class FakePlayer:
    def __init__(self):
        self.played = []

    def play(self, chunks, token=None):
        self.played.extend(chunks)


class ServerTests(unittest.TestCase):
    def test_service_speaks_request(self):
        player = FakePlayer()
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=player)
        response = service.handle(SpeechRequest(text="hello", caller="c", session_id="s"))
        self.assertEqual(response["status"], "accepted")
        self.assertEqual(len(player.played), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run server test and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_server -v
```

Expected: FAIL because `tts_summarizer.server` does not exist.

- [ ] **Step 3: Implement service and HTTP endpoints**

Create `src/tts_summarizer/server.py`:

```python
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading

from .audio import AudioPlayer
from .config import Config
from .request import RequestError, SpeechRequest
from .session import SessionManager
from .speech import SpeechGenerator
from .state import write_state
from .summarizer import Summarizer


class TtsService:
    def __init__(self, config: Config, summarizer=None, speech=None, player=None):
        self.config = config
        self.sessions = SessionManager(config.session)
        self.summarizer = summarizer or Summarizer(config.summarizer)
        self.speech = speech or SpeechGenerator(config.tts)
        self.player = player or AudioPlayer(config.audio)
        self._audio_lock = threading.Lock()

    def handle(self, request: SpeechRequest) -> dict[str, object]:
        token = self.sessions.begin(request)
        try:
            text = self.summarizer.summarize(request.text)
            if token.cancelled():
                return {"status": "cancelled", "session_key": request.session_key()}
            chunks = self.speech.generate(text)
            if self.config.session.cross_session_policy == "queue":
                with self._audio_lock:
                    self.player.play(chunks, token=token)
            else:
                self.player.play(chunks, token=token)
            return {"status": "accepted", "session_key": request.session_key()}
        finally:
            self.sessions.finish(token)

    def health(self) -> dict[str, object]:
        return {"status": "ok", "pid": os.getpid()}


class Handler(BaseHTTPRequestHandler):
    service: TtsService
    httpd_ref: ThreadingHTTPServer

    def do_GET(self):
        if self.path == "/health":
            self._send(200, self.service.health())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/speak":
            self._speak()
            return
        if self.path == "/shutdown":
            self._send(200, {"status": "shutting_down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self._send(404, {"error": "not found"})

    def log_message(self, format, *args):
        return

    def _speak(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request = SpeechRequest.from_json(payload)
        except (json.JSONDecodeError, RequestError) as exc:
            self._send(400, {"error": str(exc)})
            return
        response = self.service.handle(request)
        self._send(200, response)

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(config: Config) -> int:
    service = TtsService(config)
    handler = type("ConfiguredHandler", (Handler,), {"service": service})
    httpd = ThreadingHTTPServer((config.server.host, config.server.port), handler)
    host, port = httpd.server_address
    write_state(config, host, port, os.getpid())
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return 0
```

Modify `src/tts_summarizer/cli.py` so `serve` calls `run_server(load_config(args.config))`.

- [ ] **Step 4: Run server tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_server tests.test_speech_audio tests.test_summarizer tests.test_session tests.test_state_client tests.test_request tests.test_config tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/server.py src/tts_summarizer/cli.py tests/test_server.py
rtk git commit -m "feat: add loopback tts daemon"
```

If not a git repository, skip commit.

---

### Task 9: Wire speak, health, and stop commands

**Files:**
- Modify: `src/tts_summarizer/cli.py`
- Modify: `src/tts_summarizer/client.py`
- Test: `tests/test_cli_commands.py`

**Interfaces:**
- Consumes: `SpeechRequest`, `daemon_base_url`, `post_json`, `get_json`.
- Produces: functional `speak`, `health`, and `stop` command paths.

- [ ] **Step 1: Write CLI command tests with monkeypatching**

Create `tests/test_cli_commands.py`:

```python
import io
import unittest
from unittest.mock import patch

from tts_summarizer import cli


class CliCommandTests(unittest.TestCase):
    def test_speak_posts_normalized_request(self):
        calls = []

        def fake_base_url(config, config_path):
            return "http://127.0.0.1:9999"

        def fake_post(url, payload, timeout):
            calls.append((url, payload, timeout))
            return {"status": "accepted"}

        with patch("tts_summarizer.cli.daemon_base_url", fake_base_url), patch(
            "tts_summarizer.cli.post_json", fake_post
        ):
            code = cli.main(["speak", "--caller", "manual", "--session-id", "s", "--text", "hello"])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][0], "http://127.0.0.1:9999/v1/speak")
        self.assertEqual(calls[0][1]["text"], "hello")
        self.assertEqual(calls[0][1]["session_id"], "s")

    def test_speak_no_daemon_is_best_effort_success(self):
        with patch("tts_summarizer.cli.daemon_base_url", lambda config, config_path: None):
            self.assertEqual(cli.main(["speak", "--text", "hello"]), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run CLI command tests and verify failure**

Run:

```bash
PYTHONPATH=src rtk python -m unittest tests.test_cli_commands -v
```

Expected: FAIL because `cli` does not import or call client helpers yet.

- [ ] **Step 3: Wire CLI commands**

Modify `src/tts_summarizer/cli.py` to include these imports:

```python
from .client import daemon_base_url, get_json, post_json
from .request import RequestError, SpeechRequest
from .server import run_server
```

Then replace the command dispatch in `main` with:

```python
    try:
        config = load_config(getattr(args, "config", None))
    except ConfigError as exc:
        print(f"tts-summarizer config error: {exc}", file=sys.stderr)
        return 2 if args.command == "config-check" else 0

    if args.command == "config-check":
        return 0

    if args.command == "serve":
        return run_server(config)

    base_url = daemon_base_url(config, getattr(args, "config", None))
    if base_url is None:
        print("tts-summarizer daemon unavailable", file=sys.stderr)
        return 0

    timeout = config.server.request_timeout_ms / 1000
    try:
        if args.command == "speak":
            request = SpeechRequest.from_cli(args.text, sys.stdin.read(), args.caller, args.session_id)
            post_json(f"{base_url}/v1/speak", request.to_json(), timeout)
            return 0
        if args.command == "health":
            print(get_json(f"{base_url}/health", timeout))
            return 0
        if args.command == "stop":
            post_json(f"{base_url}/shutdown", {}, timeout)
            return 0
    except RequestError as exc:
        print(f"tts-summarizer request error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"tts-summarizer request failed: {exc}", file=sys.stderr)
        return 0

    return 0
```

- [ ] **Step 4: Run full unit tests and verify pass**

Run:

```bash
PYTHONPATH=src rtk python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository, commit:

```bash
rtk git add src/tts_summarizer/cli.py src/tts_summarizer/client.py tests/test_cli_commands.py
rtk git commit -m "feat: wire tts cli commands"
```

If not a git repository, skip commit.

---

### Task 10: Final verification and smoke path

**Files:**
- Modify: `config.example.toml` if any config field is missing.
- No new docs unless the user asks.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified local package behavior without requiring real MLX model downloads in tests.

- [ ] **Step 1: Run all unit tests**

Run:

```bash
PYTHONPATH=src rtk python -m unittest discover -s tests -v
```

Expected: PASS all tests.

- [ ] **Step 2: Run config check against example config**

Run:

```bash
PYTHONPATH=src rtk python -m tts_summarizer config-check --config config.example.toml
```

Expected: exit code `0` and no config error.

- [ ] **Step 3: Run daemon health smoke test with file audio backend and fake text**

Create a temporary smoke config manually for the command:

```bash
rtk python - <<'PY'
from pathlib import Path
p = Path('/tmp/tts-summarizer-smoke.toml')
p.write_text('''
[server]
host = "127.0.0.1"
port = 0
state_dir = "/tmp/tts-summarizer-smoke-state"
auto_start = true
startup_timeout_ms = 3000
request_timeout_ms = 5000

[audio]
backend = "file"
output_dir = "/tmp/tts-summarizer-smoke-audio"
save = true
''')
print(p)
PY
```

Then run:

```bash
PYTHONPATH=src rtk python -m tts_summarizer health --config /tmp/tts-summarizer-smoke.toml
```

Expected: output contains `status` and `ok`. This starts the daemon without speaking.

- [ ] **Step 4: Stop smoke daemon**

Run:

```bash
PYTHONPATH=src rtk python -m tts_summarizer stop --config /tmp/tts-summarizer-smoke.toml
```

Expected: exit code `0`.

- [ ] **Step 5: Commit if repository exists**

Run:

```bash
rtk git status
```

If this is a git repository and final verification changed tracked files, commit:

```bash
rtk git add config.example.toml
rtk git commit -m "test: verify tts summarizer smoke path"
```

If not a git repository or no files changed, skip commit.

## Self-Review

- Spec coverage: config order, harness-neutral request schema, loopback HTTP, session-aware interruption, configurable prompts, separate summarizer/TTS models, warm daemon lifecycle, and no Docker production path are covered by tasks 2 through 10.
- Placeholder scan: plan contains concrete steps, exact paths, commands, and code blocks.
- Type consistency: `SpeechRequest`, `Config`, `SessionManager`, `Summarizer`, `SpeechGenerator`, `AudioPlayer`, and client/server helper names match across tasks.
- Testability: every behavioral unit has stdlib tests with fake MLX/audio backends; final smoke path avoids real model downloads.
