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


def read_json_when_ready(path: Path, timeout: float = 3) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {path}")


class CodexHookTests(unittest.TestCase):
    def _stub_tts(self, directory: Path, delay: float = 0) -> Path:
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

    def _run_codex_hook(self, tmp: Path, payload: dict[str, object]) -> dict[str, object]:
        capture = self._stub_tts(tmp)
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
            ["speak", "--session_id", "codex-session-123", "Implemented the Codex hook."],
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
                                        {"type": "output_text", "text": "Final answer text"}
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
            capture = self._stub_tts(tmp, delay=1.5)
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


class HookInstallerTests(unittest.TestCase):
    def _with_home_and_path(self, home: Path, path: str):
        old_home = os.environ.get("HOME")
        old_path = os.environ.get("PATH")
        os.environ["HOME"] = str(home)
        os.environ["PATH"] = path
        return old_home, old_path

    def _restore_home_and_path(self, old_home: str | None, old_path: str | None) -> None:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path

    def test_cli_install_codex_hook_creates_idempotent_python_stop_entry(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            home = Path(tmp_name)
            capture = CodexHookTests()._stub_tts(home)
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
            old_home, old_path = self._with_home_and_path(
                home, f"{home / 'bin'}{os.pathsep}{os.environ['PATH']}"
            )
            try:
                self.assertEqual(cli.main(["install", "--harness", "codex"]), 0)
                self.assertEqual(cli.main(["install", "--harness", "codex"]), 0)
            finally:
                self._restore_home_and_path(old_home, old_path)

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
            capture = CodexHookTests()._stub_tts(home)
            (home / ".codex").mkdir()
            old_home, old_path = self._with_home_and_path(
                home, f"{home / 'bin'}{os.pathsep}/usr/bin:/bin"
            )
            try:
                self.assertEqual(cli.main(["install", "--harness", "codex"]), 0)
            finally:
                self._restore_home_and_path(old_home, old_path)

            hooks = json.loads((home / ".codex" / "hooks.json").read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    unittest.main()
