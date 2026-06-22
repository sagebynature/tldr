from __future__ import annotations

import logging
from importlib import resources
from logging.config import fileConfig
from pathlib import Path

from .config import Config


def _log_config(config: Config) -> None:
    logger = logging.getLogger(__name__)
    for label, value in (
        ("source", config.source),
        ("host", config.server.host),
        ("port", config.server.port),
        ("sum.profile", config.summarizer.default_profile),
        (
            "sum.model",
            config.summarizer.profiles[config.summarizer.default_profile].model,
        ),
        ("tts.profile", config.tts.default_profile),
        ("tts.model", config.tts.profiles[config.tts.default_profile].model),
    ):
        logger.info("  %s=%s", label, value)


def setup_logging(config: Config) -> None:
    config_file = config.logging.config_file
    if config_file:
        path = Path(config_file).expanduser()
        fileConfig(path, disable_existing_loggers=False)
        _log_config(config)
        return

    with resources.as_file(resources.files("tts_summarizer") / "logging.conf") as path:
        fileConfig(path, disable_existing_loggers=False)

    _log_config(config)
