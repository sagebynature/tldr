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
        default_summary = cfg.summarizer.profiles[cfg.summarizer.default_profile]
        self.assertEqual(default_summary.max_words, 40)
        self.assertIn("text-to-speech", default_summary.system_prompt)
        self.assertFalse(hasattr(cfg, "session"))

    def test_session_config_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "[session]\ninterrupt_same_session = true\n", encoding="utf-8"
            )

            with self.assertRaises(ConfigError):
                load_config(str(path), cwd=Path(tmp), home=Path(tmp))

    def test_summarizer_profiles_config_loads_named_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "\n".join(
                    [
                        "[summarizer]",
                        'default_profile = "fast"',
                        "[summarizer.profiles.qwen]",
                        'base_url = "http://127.0.0.1:1234/v1"',
                        'api_key = "test-token"',
                        'model = "local-model"',
                        "max_words = 40",
                        "[summarizer.profiles.qwen.extra_body.chat_template_kwargs]",
                        "enable_thinking = false",
                        "[summarizer.profiles.fast]",
                        'base_url = "http://127.0.0.1:9000/v1"',
                        'model = "fast-model"',
                        "max_words = 25",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

        self.assertEqual(cfg.summarizer.default_profile, "fast")
        self.assertEqual(cfg.summarizer.profiles["qwen"].api_key, "test-token")
        self.assertEqual(cfg.summarizer.profiles["qwen"].model, "local-model")
        self.assertEqual(
            cfg.summarizer.profiles["qwen"].extra_body,
            {"chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertEqual(cfg.summarizer.profiles["fast"].max_words, 25)

    def test_audio_ffplay_backend_config_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text('[audio]\nbackend = "ffplay"\n', encoding="utf-8")

            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

        self.assertEqual(cfg.audio.backend, "ffplay")

    def test_tts_profiles_config_loads_named_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "\n".join(
                    [
                        "[tts]",
                        'default_profile = "kokoro"',
                        "[tts.profiles.qwen]",
                        'model = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"',
                        "sample_rate = 24000",
                        "[tts.profiles.qwen.generate_kwargs]",
                        'voice = "Aiden"',
                        'lang_code = "english"',
                        "[tts.profiles.kokoro]",
                        'model = "mlx-community/Kokoro-82M-bf16"',
                        "sample_rate = 24000",
                        "[tts.profiles.kokoro.generate_kwargs]",
                        'voice = "af_heart"',
                        'lang_code = "a"',
                        "speed = 1.6",
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

        self.assertEqual(cfg.tts.default_profile, "kokoro")
        self.assertEqual(cfg.tts.profiles["qwen"].generate_kwargs["voice"], "Aiden")
        self.assertEqual(cfg.tts.profiles["kokoro"].generate_kwargs["speed"], 1.6)

    def test_cwd_config_beats_user_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "cwd"
            home = root / "home"
            cwd.mkdir()
            (home / ".config" / "tts-summarizer").mkdir(parents=True)
            (home / ".config" / "tts-summarizer" / "config.toml").write_text(
                '[tts.profiles.qwen.generate_kwargs]\nvoice = "UserVoice"\n',
                encoding="utf-8",
            )
            (cwd / "config.toml").write_text(
                '[tts.profiles.qwen.generate_kwargs]\nvoice = "CwdVoice"\n',
                encoding="utf-8",
            )
            cfg = load_config(None, cwd=cwd, home=home)
        self.assertEqual(cfg.tts.profiles["qwen"].generate_kwargs["voice"], "CwdVoice")

    def test_explicit_missing_config_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.toml"
            with self.assertRaises(ConfigError):
                load_config(str(missing), cwd=Path(tmp), home=Path(tmp))

    def test_prompt_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                '[summarizer.profiles.default]\nsystem_prompt = "custom system"\n',
                encoding="utf-8",
            )

            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

        self.assertEqual(
            cfg.summarizer.profiles[cfg.summarizer.default_profile].system_prompt,
            "custom system",
        )


if __name__ == "__main__":
    unittest.main()
