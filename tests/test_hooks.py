import json
import os
import shlex
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tts_summarizer import cli


ROOT = Path(__file__).resolve().parents[1]
CODEX_HOOK = ROOT / "hooks" / "codex" / "codex_tts.py"
CLAUDE_HOOK = ROOT / "hooks" / "claude" / "claude_tts.py"
HERMES_HOOK = ROOT / "hooks" / "hermes" / "hermes_tts.py"


def read_json_when_ready(path: Path, timeout: float = 3) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


def stub_tts(directory: Path, delay: float = 0) -> Path:
    bin_dir = directory / "bin"
    bin_dir.mkdir(exist_ok=True)
    capture = directory / "tts-call.json"
    started = directory / "tts-started"
    stub = bin_dir / "tts-summarizer"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys, time\n"
        f"pathlib.Path({str(started)!r}).write_text('1', encoding='utf-8')\n"
        f"time.sleep({delay!r})\n"
        f"pathlib.Path({str(capture)!r}).write_text(json.dumps({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return capture


def with_home_and_path(home: Path, path: str):
    old_home = os.environ.get("HOME")
    old_path = os.environ.get("PATH")
    os.environ["HOME"] = str(home)
    os.environ["PATH"] = path
    return old_home, old_path


def restore_home_and_path(old_home: str | None, old_path: str | None) -> None:
    if old_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = old_home
    if old_path is None:
        os.environ.pop("PATH", None)
    else:
        os.environ["PATH"] = old_path


class CodexHookTests(unittest.TestCase):
    def _hook_env(self, tmp: Path) -> dict[str, str]:
        state_file = tmp / "tts.enabled"
        state_file.write_text("1", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{tmp / 'bin'}{os.pathsep}{env['PATH']}",
                "CODEX_TTS_STATE_FILE": str(state_file),
                "CODEX_TTS_LOG": str(tmp / "hook.log"),
                "CODEX_TTS_PAYLOAD_LOG": str(tmp / "payload.json"),
            }
        )
        return env

    def _run_codex_hook(
        self, tmp: Path, payload: dict[str, object]
    ) -> dict[str, object]:
        capture = stub_tts(tmp)
        subprocess.run(
            [str(CODEX_HOOK)],
            input=json.dumps(payload),
            text=True,
            check=True,
            env=self._hook_env(tmp),
            cwd=ROOT,
        )
        return read_json_when_ready(capture)

    def test_codex_hook_speaks_payload_message_with_session_id(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            call = self._run_codex_hook(
                Path(tmp_name),
                {
                    "session_id": "codex-session-123",
                    "last_assistant_message": "Implemented the Codex hook.",
                },
            )

        self.assertEqual(
            call["argv"],
            [
                "speak",
                "--session_id",
                "codex-session-123",
                "Implemented the Codex hook.",
            ],
        )
        self.assertEqual(call["stdin"], "")

    def test_codex_hook_falls_back_to_transcript_last_assistant_message(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            transcript = tmp / "rollout.jsonl"
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {"type": "output_text", "text": "commentary"}
                                    ],
                                    "phase": "commentary",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response_item",
                                "payload": {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "Final answer text",
                                        }
                                    ],
                                    "phase": "final_answer",
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            call = self._run_codex_hook(
                tmp,
                {
                    "session_id": "codex-session-456",
                    "transcript_path": str(transcript),
                },
            )

        self.assertEqual(
            call["argv"],
            ["speak", "--session_id", "codex-session-456", "Final answer text"],
        )

    def test_codex_hook_exits_before_speech_finishes(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            capture = stub_tts(tmp, delay=1.5)
            payload = {
                "session_id": "codex-session-999",
                "last_assistant_message": "Long speech text.",
            }

            try:
                subprocess.run(
                    [str(CODEX_HOOK)],
                    input=json.dumps(payload),
                    text=True,
                    check=True,
                    env=self._hook_env(tmp),
                    cwd=ROOT,
                    timeout=0.5,
                )
            except subprocess.TimeoutExpired:
                self.fail("Codex hook waited for tts-summarizer instead of spawning it")

            self.assertFalse(capture.exists())
            call = read_json_when_ready(capture)
            self.assertEqual(
                call["argv"],
                ["speak", "--session_id", "codex-session-999", "Long speech text."],
            )


class ClaudeHookTests(unittest.TestCase):
    def _hook_env(self, tmp: Path) -> dict[str, str]:
        state_file = tmp / "tts.enabled"
        state_file.write_text("1", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{tmp / 'bin'}{os.pathsep}{env['PATH']}",
                "CLAUDE_TTS_STATE_FILE": str(state_file),
                "CLAUDE_TTS_LOG": str(tmp / "hook.log"),
                "CLAUDE_TTS_PAYLOAD_LOG": str(tmp / "payload.json"),
            }
        )
        return env

    def _run_claude_hook(
        self, tmp: Path, payload: dict[str, object]
    ) -> dict[str, object]:
        capture = stub_tts(tmp)
        subprocess.run(
            [str(CLAUDE_HOOK)],
            input=json.dumps(payload),
            text=True,
            check=True,
            env=self._hook_env(tmp),
            cwd=ROOT,
        )
        return read_json_when_ready(capture)

    def test_claude_hook_speaks_payload_message_with_session_id(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            call = self._run_claude_hook(
                Path(tmp_name),
                {
                    "session_id": "claude-session-123",
                    "last_assistant_message": "Implemented the Claude hook.",
                    "hook_event_name": "Stop",
                },
            )

        self.assertEqual(
            call["argv"],
            [
                "speak",
                "--session_id",
                "claude-session-123",
                "Implemented the Claude hook.",
            ],
        )
        self.assertEqual(call["stdin"], "")

    def test_claude_hook_falls_back_to_transcript_last_text_message(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            transcript = tmp / "claude.jsonl"
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "thinking", "thinking": "ignore"},
                                    ],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "assistant",
                                "message": {
                                    "role": "assistant",
                                    "content": [
                                        {"type": "text", "text": "Claude final text"}
                                    ],
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            call = self._run_claude_hook(
                tmp,
                {
                    "session_id": "claude-session-456",
                    "transcript_path": str(transcript),
                    "hook_event_name": "Stop",
                },
            )

        self.assertEqual(
            call["argv"],
            ["speak", "--session_id", "claude-session-456", "Claude final text"],
        )

    def test_claude_hook_exits_before_speech_finishes(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            capture = stub_tts(tmp, delay=1.5)
            payload = {
                "session_id": "claude-session-999",
                "last_assistant_message": "Long Claude speech.",
                "hook_event_name": "Stop",
            }

            try:
                subprocess.run(
                    [str(CLAUDE_HOOK)],
                    input=json.dumps(payload),
                    text=True,
                    check=True,
                    env=self._hook_env(tmp),
                    cwd=ROOT,
                    timeout=0.5,
                )
            except subprocess.TimeoutExpired:
                self.fail(
                    "Claude hook waited for tts-summarizer instead of spawning it"
                )

            self.assertFalse(capture.exists())
            call = read_json_when_ready(capture)
            self.assertEqual(
                call["argv"],
                ["speak", "--session_id", "claude-session-999", "Long Claude speech."],
            )



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
                        ],
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


class HookInstallerTests(unittest.TestCase):
    def test_cli_install_codex_hook_creates_idempotent_python_stop_entry(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            capture = stub_tts(home)
            codex_dir = home / ".codex"
            installed = codex_dir / "hooks" / "tts" / "codex_tts.py"
            codex_dir.mkdir()
            (codex_dir / "hooks.json").write_text(
                json.dumps(
                    {
                        "hooks": {
                            "Stop": [
                                {
                                    "hooks": [
                                        {
                                            "command": str(installed),
                                            "statusMessage": "Speaking completion",
                                            "timeout": 5,
                                            "type": "command",
                                        }
                                    ],
                                    "matcher": "*",
                                }
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            old_home, old_path = with_home_and_path(
                home, f"{home / 'bin'}{os.pathsep}{os.environ['PATH']}"
            )
            try:
                self.assertEqual(cli.main(["install", "--harness", "codex"]), 0)
                self.assertEqual(cli.main(["install", "--harness", "codex"]), 0)
            finally:
                restore_home_and_path(old_home, old_path)

            hooks = json.loads((codex_dir / "hooks.json").read_text(encoding="utf-8"))
            stop_entries = hooks["hooks"]["Stop"]
            expected_command = f"python3 {shlex.quote(str(installed))}"
            commands = [
                hook.get("command")
                for entry in stop_entries
                for hook in entry.get("hooks", [])
            ]
            matching = [
                entry
                for entry in stop_entries
                for hook in entry.get("hooks", [])
                if hook.get("command") == expected_command
            ]

            self.assertFalse(capture.exists())
            self.assertTrue(installed.exists())
            self.assertTrue(os.access(installed, os.X_OK))
            self.assertNotIn(str(installed), commands)
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["matcher"], "*")
            self.assertTrue((codex_dir / "tts.enabled").exists())

    def test_installed_codex_command_runs_when_tts_summarizer_is_installed(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            capture = stub_tts(home)
            (home / ".codex").mkdir()
            old_home, old_path = with_home_and_path(
                home, f"{home / 'bin'}{os.pathsep}/usr/bin:/bin"
            )
            try:
                self.assertEqual(cli.main(["install", "--harness", "codex"]), 0)
            finally:
                restore_home_and_path(old_home, old_path)

            hooks = json.loads(
                (home / ".codex" / "hooks.json").read_text(encoding="utf-8")
            )
            command = hooks["hooks"]["Stop"][0]["hooks"][0]["command"]
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{home / 'bin'}{os.pathsep}/usr/bin:/bin",
                    "CODEX_TTS_STATE_FILE": str(home / ".codex" / "tts.enabled"),
                    "CODEX_TTS_LOG": str(home / "hook.log"),
                    "CODEX_TTS_PAYLOAD_LOG": str(home / "payload.json"),
                }
            )
            subprocess.run(
                command,
                input=json.dumps(
                    {
                        "session_id": "codex-session-789",
                        "last_assistant_message": "Hook runner text.",
                    }
                ),
                text=True,
                shell=True,
                check=True,
                env=env,
                cwd=ROOT,
            )

            call = read_json_when_ready(capture)
            self.assertEqual(
                call["argv"],
                ["speak", "--session_id", "codex-session-789", "Hook runner text."],
            )

    def test_cli_install_claude_hook_creates_exec_form_stop_entry(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            claude_dir = home / ".claude"
            installed = claude_dir / "hooks" / "tts" / "claude_tts.py"
            claude_dir.mkdir()
            (claude_dir / "settings.json").write_text(
                json.dumps({"theme": "dark", "hooks": {"Stop": []}}),
                encoding="utf-8",
            )
            old_home, old_path = with_home_and_path(
                home, f"{home / 'bin'}{os.pathsep}{os.environ['PATH']}"
            )
            try:
                self.assertEqual(cli.main(["install", "--harness", "claude"]), 0)
                self.assertEqual(cli.main(["install", "--harness", "claude"]), 0)
            finally:
                restore_home_and_path(old_home, old_path)

            settings = json.loads(
                (claude_dir / "settings.json").read_text(encoding="utf-8")
            )
            stop_entries = settings["hooks"]["Stop"]
            matching = [
                entry
                for entry in stop_entries
                for hook in entry.get("hooks", [])
                if hook.get("command") == "python3"
                and hook.get("args") == [str(installed)]
            ]

            self.assertEqual(settings["theme"], "dark")
            self.assertTrue(installed.exists())
            self.assertTrue(os.access(installed, os.X_OK))
            self.assertEqual(len(matching), 1)
            self.assertNotIn("matcher", matching[0])
            self.assertTrue((claude_dir / "tts.enabled").exists())

    def test_installed_claude_command_runs_when_tts_summarizer_is_installed(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            capture = stub_tts(home)
            (home / ".claude").mkdir()
            old_home, old_path = with_home_and_path(
                home, f"{home / 'bin'}{os.pathsep}/usr/bin:/bin"
            )
            try:
                self.assertEqual(cli.main(["install", "--harness", "claude"]), 0)
            finally:
                restore_home_and_path(old_home, old_path)

            settings = json.loads(
                (home / ".claude" / "settings.json").read_text(encoding="utf-8")
            )
            hook = settings["hooks"]["Stop"][0]["hooks"][0]
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{home / 'bin'}{os.pathsep}/usr/bin:/bin",
                    "CLAUDE_TTS_STATE_FILE": str(home / ".claude" / "tts.enabled"),
                    "CLAUDE_TTS_LOG": str(home / "hook.log"),
                    "CLAUDE_TTS_PAYLOAD_LOG": str(home / "payload.json"),
                }
            )
            subprocess.run(
                [hook["command"], *hook["args"]],
                input=json.dumps(
                    {
                        "session_id": "claude-session-789",
                        "last_assistant_message": "Claude runner text.",
                        "hook_event_name": "Stop",
                    }
                ),
                text=True,
                check=True,
                env=env,
                cwd=ROOT,
            )

            call = read_json_when_ready(capture)
            self.assertEqual(
                call["argv"],
                ["speak", "--session_id", "claude-session-789", "Claude runner text."],
            )


class OmpHookTests(unittest.TestCase):
    def _write_node_runner(
        self,
        tmp: Path,
        event_name: str,
        event: dict[str, object],
        entries: list[dict[str, object]],
    ) -> Path:
        hook_mjs = tmp / "omp_tts.mjs"
        hook_mjs.write_text(
            (ROOT / "hooks" / "omp" / "tts.ts").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        runner = tmp / "run-omp-hook.mjs"
        runner.write_text(
            "\n".join(
                [
                    "import hook from './omp_tts.mjs';",
                    "const handlers = {};",
                    "const pi = { on(name, handler) { handlers[name] = handler; } };",
                    f"const event = {json.dumps(event)};",
                    f"const entries = {json.dumps(entries)};",
                    "hook(pi);",
                    f"if (handlers[{json.dumps(event_name)}]) {{",
                    f"  await handlers[{json.dumps(event_name)}](event, {{ cwd: '/repo/demo', sessionManager: {{ getEntries: () => entries }} }});",
                    "}",
                ]
            ),
            encoding="utf-8",
        )
        return runner

    def test_omp_hook_speaks_agent_end_current_assistant_text(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            capture = stub_tts(tmp)
            runner = self._write_node_runner(
                tmp,
                "agent_end",
                {
                    "type": "agent_end",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "thinking", "thinking": "ignore"},
                                {"type": "text", "text": "OMP final answer"},
                            ],
                        }
                    ],
                },
                [
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Previous persisted answer"}
                            ],
                        },
                    }
                ],
            )

            env = os.environ.copy()
            env.update(
                {
                    "OMP_TTS_BIN": str(tmp / "bin" / "tts-summarizer"),
                    "OMP_TTS_SESSION_ID": "omp-session-123",
                }
            )

            subprocess.run(
                ["node", str(runner)], check=True, cwd=tmp, env=env, timeout=1
            )
            call = read_json_when_ready(capture)

            self.assertEqual(
                call["argv"],
                ["speak", "--session_id", "omp-session-123", "OMP final answer"],
            )
            self.assertEqual(call["stdin"], "")

    def test_omp_hook_ignores_turn_end_intermediate_messages(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            capture = stub_tts(tmp)
            runner = self._write_node_runner(
                tmp,
                "turn_end",
                {
                    "type": "turn_end",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Intermediate assistant text"}
                        ],
                    },
                    "toolResults": [],
                },
                [
                    {
                        "type": "message",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Intermediate assistant text"}
                            ],
                        },
                    }
                ],
            )
            env = os.environ.copy()
            env["OMP_TTS_BIN"] = str(tmp / "bin" / "tts-summarizer")

            subprocess.run(
                ["node", str(runner)], check=True, cwd=tmp, env=env, timeout=1
            )
            time.sleep(0.2)

            self.assertFalse(capture.exists())

    def test_omp_hook_exits_before_speech_finishes(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            capture = stub_tts(tmp, delay=1.5)
            runner = self._write_node_runner(
                tmp,
                "agent_end",
                {
                    "type": "agent_end",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Slow OMP speech"}],
                        }
                    ],
                },
                [],
            )
            env = os.environ.copy()
            env["OMP_TTS_BIN"] = str(tmp / "bin" / "tts-summarizer")

            subprocess.run(
                ["node", str(runner)], check=True, cwd=tmp, env=env, timeout=0.5
            )
            call = read_json_when_ready(capture)

            self.assertEqual(
                call["argv"],
                ["speak", "--session_id", "omp:/repo/demo", "Slow OMP speech"],
            )

    def test_cli_install_omp_global_extension_file(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            old_home, old_path = with_home_and_path(home, os.environ["PATH"])
            try:
                self.assertEqual(cli.main(["install", "--harness", "omp"]), 0)
                self.assertEqual(cli.main(["install", "--harness", "omp"]), 0)
            finally:
                restore_home_and_path(old_home, old_path)

            installed = home / ".omp" / "agent" / "extensions" / "tts-summarizer.ts"
            self.assertTrue(installed.exists())
            self.assertIn("agent_end", installed.read_text(encoding="utf-8"))

    def test_cli_install_pi_agent_alias_copies_global_extension_file(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            old_home, old_path = with_home_and_path(home, os.environ["PATH"])
            try:
                self.assertEqual(cli.main(["install", "--harness", "pi"]), 0)
            finally:
                restore_home_and_path(old_home, old_path)

            installed = home / ".pi" / "agent" / "extensions" / "tts-summarizer.ts"
            self.assertTrue(installed.exists())
            self.assertIn("agent_end", installed.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
