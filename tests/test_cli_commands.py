import io
import tempfile
import signal
import unittest
from pathlib import Path
from unittest import mock

from tts_summarizer import cli
from tts_summarizer.config import load_config


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
            with mock.patch("subprocess.Popen", return_value=proc) as popen:
                code = cli.main(["speak", "--config", str(config), "hello world"])

        self.assertEqual(code, 0)
        curl_args = popen.call_args_list[0].args[0]
        ffplay_args = popen.call_args_list[1].args[0]
        self.assertEqual(curl_args[-1], "http://127.0.0.9:7777/v1/speak")
        self.assertIn('{"text":"hello world","summarize":true}', curl_args)
        self.assertEqual(
            ffplay_args[:4], ["ffplay", "-nodisp", "-autoexit", "-loglevel"]
        )

    def test_speak_replaces_existing_session_pid_while_playing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                "[server]\nport = 7777\nstate_dir = '" + tmp + "'\n",
                encoding="utf-8",
            )
            session_dir = Path(tmp) / "sessions"
            session_dir.mkdir()
            pid_file = session_dir / "demo.pid"
            pid_file.write_text("123", encoding="utf-8")
            seen_during_playback = []
            curl_proc = mock.Mock()
            curl_proc.stdout = mock.Mock()
            curl_proc.wait.return_value = 0
            player_proc = mock.Mock()
            player_proc.pid = 456
            player_proc.wait.side_effect = lambda: seen_during_playback.append(
                pid_file.read_text(encoding="utf-8")
            )
            with (
                mock.patch("os.kill") as kill,
                mock.patch("subprocess.Popen", side_effect=[curl_proc, player_proc]),
            ):
                code = cli.main(
                    ["speak", "--config", str(config), "--session_id", "demo", "hello"]
                )

        self.assertEqual(code, 0)
        kill.assert_any_call(123, signal.SIGTERM)
        self.assertEqual(seen_during_playback, ["456"])
        self.assertFalse(pid_file.exists())

    def test_speak_cleans_up_curl_when_player_launch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text(
                "[server]\nport = 7777\nstate_dir = '" + tmp + "'\n",
                encoding="utf-8",
            )
            curl_proc = mock.Mock()
            curl_proc.stdout = mock.Mock()
            curl_proc.wait.return_value = 0
            with mock.patch(
                "subprocess.Popen", side_effect=[curl_proc, OSError("missing ffplay")]
            ):
                code = cli.main(["speak", "--config", str(config), "hello"])

        self.assertEqual(code, 0)
        curl_proc.terminate.assert_called_once_with()
        curl_proc.wait.assert_called_once_with()



    def _run_cli_with_home(self, argv, home):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch("tts_summarizer.cli.Path.home", return_value=home),
            mock.patch("sys.stdout", new=stdout),
            mock.patch("sys.stderr", new=stderr),
        ):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_init_config_writes_remote_config_to_user_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()

            code, stdout, stderr = self._run_cli_with_home(["init-config"], home)

            path = home / ".config" / "tts-summarizer" / "config.toml"
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(path), stdout)
            cfg = load_config(str(path), cwd=Path(tmp), home=home)
            self.assertEqual(cfg.summarizer.default_profile, "remote-qwen25")
            self.assertEqual(cfg.tts.default_profile, "remote-kokoro")

    def test_init_config_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            path = home / ".config" / "tts-summarizer" / "config.toml"
            path.parent.mkdir(parents=True)
            path.write_text("sentinel", encoding="utf-8")

            code, stdout, stderr = self._run_cli_with_home(["init-config"], home)

            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn(str(path), stderr)
            self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")

    def test_init_config_force_overwrites_existing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            path = home / ".config" / "tts-summarizer" / "config.toml"
            path.parent.mkdir(parents=True)
            path.write_text("sentinel", encoding="utf-8")

            code, stdout, stderr = self._run_cli_with_home(["init-config", "--force"], home)

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(path), stdout)
            self.assertNotEqual(path.read_text(encoding="utf-8"), "sentinel")

    def test_init_config_writes_apple_local_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()

            code, stdout, stderr = self._run_cli_with_home(
                ["init-config", "--profile", "apple-local"], home
            )

            path = home / ".config" / "tts-summarizer" / "config.toml"
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(path), stdout)
            cfg = load_config(str(path), cwd=Path(tmp), home=home)
            self.assertEqual(cfg.summarizer.default_profile, "qwen25")
            self.assertEqual(cfg.tts.default_profile, "kokoro")
if __name__ == "__main__":
    unittest.main()
