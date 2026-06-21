from __future__ import annotations

from importlib import resources
from logging.config import fileConfig
from pathlib import Path

from .config import Config


def setup_logging(config: Config) -> None:
    config_file = config.logging.config_file
    if config_file:
        path = Path(config_file).expanduser()
        fileConfig(path, disable_existing_loggers=False)
        return

    with resources.as_file(resources.files("tts_summarizer") / "logging.conf") as path:
        fileConfig(path, disable_existing_loggers=False)
