import io
import logging
import tempfile
import unittest
from pathlib import Path

from tts_summarizer.config import load_config
from tts_summarizer.logging_setup import setup_logging


class LoggingSetupTests(unittest.TestCase):
    def test_default_logging_config_installs_handler(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(None, cwd=Path(tmp), home=Path(tmp))
            setup_logging(cfg)
        self.assertTrue(logging.getLogger().handlers)

    def test_custom_logging_config_is_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_stream = io.StringIO()
            logging_config = Path(tmp) / "logging.conf"
            logging_config.write_text(
                "\n".join(
                    [
                        "[loggers]",
                        "keys=root",
                        "[handlers]",
                        "keys=console",
                        "[formatters]",
                        "keys=plain",
                        "[formatter_plain]",
                        "format=CUSTOM %(levelname)s %(message)s",
                        "[handler_console]",
                        "class=StreamHandler",
                        "args=(sys.stderr,)",
                        "formatter=plain",
                        "level=DEBUG",
                        "[logger_root]",
                        "level=DEBUG",
                        "handlers=console",
                    ]
                ),
                encoding="utf-8",
            )
            cfg_path = Path(tmp) / "config.toml"
            cfg_path.write_text(f'[logging]\nconfig_file = "{logging_config}"\n', encoding="utf-8")
            cfg = load_config(str(cfg_path), cwd=Path(tmp), home=Path(tmp))
            setup_logging(cfg)
        self.assertTrue(logging.getLogger().handlers)


if __name__ == "__main__":
    unittest.main()
