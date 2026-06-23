import tempfile
import unittest
from pathlib import Path

from tldr.config import ConfigError, load_config


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

    def test_tts_remote_profile_config_loads_endpoint_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            path.write_text(
                "\n".join(
                    [
                        "[tts]",
                        'default_profile = "remote"',
                        "[tts.profiles.remote]",
                        'backend = "remote"',
                        'base_url = "http://127.0.0.1:9100/v1"',
                        'api_key = "omlx"',
                        'model = "mlx-community/Kokoro-82M-bf16"',
                        "sample_rate = 24000",
                        "[tts.profiles.remote.generate_kwargs]",
                        'voice = "af_heart"',
                        'response_format = "wav"',
                    ]
                ),
                encoding="utf-8",
            )

            cfg = load_config(str(path), cwd=Path(tmp), home=Path(tmp))

        profile = cfg.tts.profiles["remote"]
        self.assertEqual(profile.backend, "remote")
        self.assertEqual(profile.base_url, "http://127.0.0.1:9100/v1")
        self.assertEqual(profile.api_key, "omlx")
        self.assertEqual(profile.model, "mlx-community/Kokoro-82M-bf16")
        self.assertEqual(profile.generate_kwargs["voice"], "af_heart")
        self.assertEqual(profile.generate_kwargs["response_format"], "wav")

    def test_remote_example_config_loads_remote_profiles(self):
        cfg = load_config(
            "config.remote.example.toml", cwd=Path.cwd(), home=Path.home()
        )

        self.assertEqual(cfg.server.host, "0.0.0.0")
        self.assertEqual(cfg.server.port, 9200)
        self.assertEqual(cfg.summarizer.default_profile, "remote-qwen25")
        self.assertEqual(
            cfg.summarizer.profiles["remote-qwen25"].base_url,
            "http://127.0.0.1:9000/v1",
        )
        self.assertEqual(cfg.tts.default_profile, "remote-kokoro")
        self.assertEqual(cfg.tts.profiles["remote-kokoro"].backend, "remote")
        self.assertEqual(
            cfg.tts.profiles["remote-kokoro"].base_url,
            "http://127.0.0.1:9000/v1",
        )

    def test_local_example_config_loads_local_defaults(self):
        cfg = load_config("config.local.example.toml", cwd=Path.cwd(), home=Path.home())

        self.assertEqual(cfg.server.host, "127.0.0.1")
        self.assertEqual(cfg.server.port, 9200)
        self.assertEqual(cfg.summarizer.default_profile, "qwen25")
        self.assertEqual(cfg.tts.default_profile, "kokoro")
        self.assertEqual(cfg.tts.profiles["kokoro"].backend, "mlx")
        self.assertEqual(
            cfg.tts.profiles["kokoro"].model, "mlx-community/Kokoro-82M-bf16"
        )
        self.assertEqual(cfg.tts.profiles["remote-kokoro"].backend, "remote")

    def test_cwd_config_beats_user_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cwd = root / "cwd"
            home = root / "home"
            cwd.mkdir()
            (home / ".config" / "tldr").mkdir(parents=True)
            (home / ".config" / "tldr" / "config.toml").write_text(
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
