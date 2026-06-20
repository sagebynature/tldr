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
