# Hermes Shell Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Hermes Agent shell-hook support that speaks final assistant responses through `tts-summarizer` without blocking or changing Hermes output.

**Architecture:** Add one executable Python shell hook under `hooks/hermes/hermes_tts.py`. Extend the existing installer to copy it into `~/.hermes/agent-hooks/tts-summarizer/` and append an idempotent `post_llm_call` entry to `~/.hermes/config.yaml`. Keep config editing dependency-free and exact-command based.

**Tech Stack:** Python 3.11, `unittest`, existing `tts-summarizer speak` CLI, Hermes shell hooks configured in YAML text.

## Global Constraints

- Use a Hermes shell hook, not gateway `HOOK.yaml` and not plugin hooks.
- Register under `hooks.post_llm_call` in `~/.hermes/config.yaml`.
- Install the hook script to `~/.hermes/agent-hooks/tts-summarizer/hermes_tts.py`.
- The hook must print `{}` to stdout for no-op success.
- The hook must fail open: malformed JSON, missing text, missing binary, and spawn errors must not break Hermes.
- Do not add a YAML parsing dependency.
- Use TDD: write each failing test first, run it red, implement minimal code, run it green.
- Shell commands in this repo must be prefixed with `rtk`.

---

## File Structure

- Create `hooks/hermes/hermes_tts.py`: standalone executable Hermes shell-hook script.
- Modify `tests/test_hooks.py`: add Hermes hook tests and installer tests, following existing stub binary helpers.
- Modify `src/tts_summarizer/installer.py`: add Hermes hook copying and config registration.
- Modify `src/tts_summarizer/cli.py`: include `hermes` in accepted install harness choices if choices are hard-coded there.
- Modify `pyproject.toml`: include the Hermes hook file in wheel resources.

---

### Task 1: Hermes Hook Script

**Files:**
- Create: `hooks/hermes/hermes_tts.py`
- Modify: `tests/test_hooks.py`

**Interfaces:**
- Consumes: Hermes stdin JSON payload with `hook_event_name`, `session_id`, `cwd`, and `extra`.
- Produces: Executable script that prints `{}` and spawns `tts-summarizer speak --session_id <id> <text>`.

- [ ] **Step 1: Write failing assistant-response test**

Add this near existing hook constants/classes in `tests/test_hooks.py`:

```python
HERMES_HOOK = ROOT / "hooks" / "hermes" / "hermes_tts.py"


class HermesHookTests(unittest.TestCase):
    def _hook_env(self, tmp: Path) -> dict[str, str]:
        return {
            **os.environ,
            "PATH": f"{tmp / 'bin'}{os.pathsep}{os.environ['PATH']}",
        }

    def _run_hermes_hook(
        self, tmp: Path, payload: dict[str, object], delay: float = 0
    ) -> tuple[dict[str, object], subprocess.CompletedProcess[str]]:
        capture = stub_tts(tmp, delay=delay)
        result = subprocess.run(
            [str(HERMES_HOOK)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
            env=self._hook_env(tmp),
            cwd=ROOT,
        )
        return read_json_when_ready(capture), result

    def test_hermes_hook_speaks_assistant_response_with_session_id(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            call, result = self._run_hermes_hook(
                Path(tmp_name),
                {
                    "hook_event_name": "post_llm_call",
                    "session_id": "hermes-session-123",
                    "extra": {"assistant_response": "Implemented Hermes hook."},
                },
            )

        self.assertEqual(result.stdout, "{}\n")
        self.assertEqual(
            call["argv"],
            [
                "speak",
                "--session_id",
                "hermes:hermes-session-123",
                "Implemented Hermes hook.",
            ],
        )
        self.assertEqual(call["stdin"], "")
```

- [ ] **Step 2: Run test verify fails**

```bash
rtk uv run python -m unittest tests.test_hooks.HermesHookTests.test_hermes_hook_speaks_assistant_response_with_session_id -v
```

Expected: FAIL or ERROR because `hooks/hermes/hermes_tts.py` does not exist.

- [ ] **Step 3: Write minimal hook implementation**

Create `hooks/hermes/hermes_tts.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any


def _read_payload() -> dict[str, Any]:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _last_assistant_text(history: Any) -> str:
    if not isinstance(history, list):
        return ""
    for entry in reversed(history):
        if not isinstance(entry, dict) or entry.get("role") != "assistant":
            continue
        content = entry.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            text = "".join(parts).strip()
            if text:
                return text
    return ""


def _assistant_text(payload: dict[str, Any]) -> str:
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        return ""
    response = extra.get("assistant_response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    return _last_assistant_text(extra.get("conversation_history"))


def _session_id(payload: dict[str, Any]) -> str:
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return f"hermes:{session_id.strip()}"
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return f"hermes:{cwd.strip()}"
    return "hermes"


def _spawn(text: str, session_id: str) -> None:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name != "nt":
        kwargs["start_new_session"] = True
    subprocess.Popen(
        ["tts-summarizer", "speak", "--session_id", session_id, text],
        **kwargs,
    )


def main() -> int:
    payload = _read_payload()
    try:
        if payload.get("hook_event_name") == "post_llm_call":
            text = _assistant_text(payload)
            if text:
                _spawn(text, _session_id(payload))
    except Exception as exc:
        print(f"tts-summarizer Hermes hook ignored error: {exc}", file=sys.stderr)
    print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Make executable:

```bash
rtk chmod +x hooks/hermes/hermes_tts.py
```

- [ ] **Step 4: Run test verify passes**

```bash
rtk uv run python -m unittest tests.test_hooks.HermesHookTests.test_hermes_hook_speaks_assistant_response_with_session_id -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add hooks/hermes/hermes_tts.py tests/test_hooks.py
rtk git commit -m "feat: add hermes shell hook script"
```

---

### Task 2: Hermes Hook Edge Cases

**Files:**
- Modify: `tests/test_hooks.py`
- Modify: `hooks/hermes/hermes_tts.py` only if tests expose a bug.

**Interfaces:**
- Consumes: `HermesHookTests` helpers from Task 1.
- Produces: coverage for fallback text, ignored events, and non-blocking spawn.

- [ ] **Step 1: Write fallback and ignore tests**

Append to `HermesHookTests`:

```python
    def test_hermes_hook_falls_back_to_conversation_history(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            call, _result = self._run_hermes_hook(
                Path(tmp_name),
                {
                    "hook_event_name": "post_llm_call",
                    "session_id": "fallback-session",
                    "extra": {
                        "conversation_history": [
                            {"role": "user", "content": "ignore"},
                            {
                                "role": "assistant",
                                "content": [
                                    {"type": "text", "text": "Fallback "},
                                    {"type": "text", "text": "Hermes text."},
                                ],
                            },
                        ]
                    },
                },
            )

        self.assertEqual(
            call["argv"],
            ["speak", "--session_id", "hermes:fallback-session", "Fallback Hermes text."],
        )

    def test_hermes_hook_ignores_other_events(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            capture = stub_tts(tmp)
            result = subprocess.run(
                [str(HERMES_HOOK)],
                input=json.dumps(
                    {
                        "hook_event_name": "pre_llm_call",
                        "session_id": "wrong-event",
                        "extra": {"assistant_response": "Do not speak."},
                    }
                ),
                text=True,
                capture_output=True,
                check=True,
                env=self._hook_env(tmp),
                cwd=ROOT,
            )

        self.assertEqual(result.stdout, "{}\n")
        self.assertFalse(capture.exists())
```

- [ ] **Step 2: Run tests verify behavior**

```bash
rtk uv run python -m unittest tests.test_hooks.HermesHookTests.test_hermes_hook_falls_back_to_conversation_history tests.test_hooks.HermesHookTests.test_hermes_hook_ignores_other_events -v
```

Expected before implementation: fallback fails if `_last_assistant_text()` is missing; ignore passes if Task 1 guarded event names.

- [ ] **Step 3: Fix hook only if needed**

If fallback failed, update `_last_assistant_text()` in `hooks/hermes/hermes_tts.py` to the implementation from Task 1 Step 3. Do not add formats the tests do not require.

- [ ] **Step 4: Write non-blocking test**

Append to `HermesHookTests`:

```python
    def test_hermes_hook_exits_before_speech_finishes(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            capture = stub_tts(tmp, delay=1.5)
            started = tmp / "tts-started"
            started_at = time.monotonic()
            result = subprocess.run(
                [str(HERMES_HOOK)],
                input=json.dumps(
                    {
                        "hook_event_name": "post_llm_call",
                        "session_id": "slow-session",
                        "extra": {"assistant_response": "Slow Hermes speech"},
                    }
                ),
                text=True,
                capture_output=True,
                check=True,
                env=self._hook_env(tmp),
                cwd=ROOT,
                timeout=0.5,
            )
            elapsed = time.monotonic() - started_at

            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and not started.exists():
                time.sleep(0.05)

            self.assertEqual(result.stdout, "{}\n")
            self.assertLess(elapsed, 1)
            self.assertTrue(started.exists())
            self.assertFalse(capture.exists())
```

- [ ] **Step 5: Run non-blocking test**

```bash
rtk uv run python -m unittest tests.test_hooks.HermesHookTests.test_hermes_hook_exits_before_speech_finishes -v
```

Expected: PASS if Task 1 used detached `Popen`; FAIL if the hook blocks.

- [ ] **Step 6: Run Hermes hook tests**

```bash
rtk uv run python -m unittest tests.test_hooks.HermesHookTests -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add hooks/hermes/hermes_tts.py tests/test_hooks.py
rtk git commit -m "test: cover hermes hook edge cases"
```

---

### Task 3: Hermes Installer

**Files:**
- Modify: `tests/test_hooks.py`
- Modify: `src/tts_summarizer/installer.py`
- Modify: `src/tts_summarizer/cli.py` if install choices are listed there.

**Interfaces:**
- Consumes: existing `cli.main(["install", "--harness", ...])` install command.
- Produces: `cli.main(["install", "--harness", "hermes"]) == 0` and installed hook/config files.

- [ ] **Step 1: Write failing installer test**

Append to `HookInstallerTests`:

```python
    def test_cli_install_hermes_hook_creates_idempotent_shell_hook_entry(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            hermes_dir = home / ".hermes"
            hermes_dir.mkdir()
            config_yaml = hermes_dir / "config.yaml"
            config_yaml.write_text("model: qwen\n", encoding="utf-8")

            old_home, old_path = with_home_and_path(home, os.environ["PATH"])
            try:
                self.assertEqual(cli.main(["install", "--harness", "hermes"]), 0)
                self.assertEqual(cli.main(["install", "--harness", "hermes"]), 0)
            finally:
                restore_home_and_path(old_home, old_path)

        installed = home / ".hermes" / "agent-hooks" / "tts-summarizer" / "hermes_tts.py"
        config = config_yaml.read_text(encoding="utf-8")
        command = str(installed)

        self.assertTrue(installed.exists())
        self.assertTrue(os.access(installed, os.X_OK))
        self.assertTrue((home / ".hermes" / "tts.enabled").exists())
        self.assertIn("model: qwen\n", config)
        self.assertIn("hooks:\n", config)
        self.assertIn("post_llm_call:\n", config)
        self.assertIn(f'command: "{command}"\n', config)
        self.assertIn("timeout: 5\n", config)
        self.assertEqual(config.count(command), 1)
```

- [ ] **Step 2: Run installer test verify fails**

```bash
rtk uv run python -m unittest tests.test_hooks.HookInstallerTests.test_cli_install_hermes_hook_creates_idempotent_shell_hook_entry -v
```

Expected: FAIL because `hermes` is not supported.

- [ ] **Step 3: Implement installer support**

In `src/tts_summarizer/installer.py`, extend `HOOK_FILENAMES`:

```python
HOOK_FILENAMES = {
    "codex": "codex_tts.py",
    "claude": "claude_tts.py",
    "omp": "tts.ts",
    "pi": "tts.ts",
    "hermes": "hermes_tts.py",
}
```

Add helpers and installer:

```python
def _quote_yaml_string(value: str) -> str:
    return json.dumps(value)


def _ensure_hermes_config_entry(config_yaml: Path, installed_hook: Path) -> None:
    command = str(installed_hook)
    existing = config_yaml.read_text(encoding="utf-8") if config_yaml.exists() else ""
    if command in existing:
        return
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    if "hooks:" in existing:
        block = (
            f"{prefix}  post_llm_call:\n"
            f"    - command: {_quote_yaml_string(command)}\n"
            "      timeout: 5\n"
        )
    else:
        block = (
            f"{prefix}hooks:\n"
            "  post_llm_call:\n"
            f"    - command: {_quote_yaml_string(command)}\n"
            "      timeout: 5\n"
        )
    config_yaml.parent.mkdir(parents=True, exist_ok=True)
    config_yaml.write_text(existing + block, encoding="utf-8")


def _install_hermes(home: Path) -> Path:
    hermes_dir = home / ".hermes"
    install_dir = hermes_dir / "agent-hooks" / "tts-summarizer"
    installed_hook = install_dir / "hermes_tts.py"
    config_yaml = hermes_dir / "config.yaml"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_hook("hermes", installed_hook)
    installed_hook.chmod(installed_hook.stat().st_mode | 0o700)
    (hermes_dir / "tts.enabled").touch()
    _ensure_hermes_config_entry(config_yaml, installed_hook)
    return installed_hook
```

Update `install_hook()` dispatch:

```python
def install_hook(harness: str, home: Path | None = None) -> Path:
    root = home or Path.home()
    if harness == "codex":
        return _install_codex(root)
    if harness == "claude":
        return _install_claude(root)
    if harness == "omp":
        return _install_omp(root)
    if harness == "pi":
        return _install_pi(root)
    if harness == "hermes":
        return _install_hermes(root)
    raise ValueError(f"unsupported harness: {harness}")
```

If `src/tts_summarizer/cli.py` hard-codes install harness choices, add `"hermes"` to that list.

- [ ] **Step 4: Run installer test verify passes**

```bash
rtk uv run python -m unittest tests.test_hooks.HookInstallerTests.test_cli_install_hermes_hook_creates_idempotent_shell_hook_entry -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/tts_summarizer/installer.py src/tts_summarizer/cli.py tests/test_hooks.py
rtk git commit -m "feat: install hermes shell hook"
```

---

### Task 4: Packaging

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_hooks.py`

**Interfaces:**
- Consumes: `_copy_hook("hermes", installed_hook)` resource fallback.
- Produces: packaged `tts_summarizer/hooks/hermes_tts.py` resource.

- [ ] **Step 1: Add copied-file assertion**

Append this assertion to the Hermes installer test from Task 3:

```python
        self.assertIn("post_llm_call", installed.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Update wheel force-include**

Modify `pyproject.toml` `[tool.hatch.build.targets.wheel]` `force-include` to include:

```toml
"hooks/hermes/hermes_tts.py" = "tts_summarizer/hooks/hermes_tts.py"
```

Keep existing entries unchanged.

- [ ] **Step 3: Run focused installer test**

```bash
rtk uv run python -m unittest tests.test_hooks.HookInstallerTests.test_cli_install_hermes_hook_creates_idempotent_shell_hook_entry -v
```

Expected: PASS.

- [ ] **Step 4: Build package**

```bash
rtk uv build
```

Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
rtk git add pyproject.toml tests/test_hooks.py
rtk git commit -m "build: package hermes shell hook"
```

---

### Task 5: Final Verification

**Files:**
- Verify all changed files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: verified Hermes shell-hook support.

- [ ] **Step 1: Run hook test module**

```bash
rtk uv run python -m unittest tests.test_hooks -v
```

Expected: PASS.

- [ ] **Step 2: Run project check**

```bash
rtk make check
```

Expected: PASS.

- [ ] **Step 3: Inspect working tree**

```bash
rtk git status --short
```

Expected: only intentional changes, or clean if all task commits were made.

- [ ] **Step 4: Commit final fixes if needed**

```bash
rtk git add hooks/hermes/hermes_tts.py tests/test_hooks.py src/tts_summarizer/installer.py src/tts_summarizer/cli.py pyproject.toml
rtk git commit -m "fix: verify hermes shell hook support"
```

Expected: commit created only if there were fixes after earlier commits.
