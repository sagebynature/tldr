from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol, cast

from .config import TtsConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    samples: object
    sample_rate: int


class SpeechBackend(Protocol):
    def generate(self, text: str, config: TtsConfig) -> list[AudioChunk]: ...


class MlxAudioBackend:
    def __init__(self):
        self._model = None
        self._model_name = ""

    def _load(self, model_name: str):
        if self._model is not None and self._model_name == model_name:
            return self._model
        logger.info("loading tts model=%s", model_name)
        from mlx_audio.tts.utils import load_model

        self._model = cast(Any, load_model)(model_name)
        self._model_name = model_name
        logger.info("tts model ready model=%s", model_name)
        return self._model

    def generate(self, text: str, config: TtsConfig) -> list[AudioChunk]:
        model = self._load(config.model)
        kwargs = dict(config.generate_kwargs)
        kwargs.setdefault("stream", config.stream)
        logger.info(
            "calling tts generate model=%s kwargs=%s text_chars=%s",
            config.model,
            sorted(kwargs),
            len(text),
        )
        results = model.generate(text=text, **kwargs)
        return [
            AudioChunk(
                samples=getattr(result, "audio", result),
                sample_rate=getattr(result, "sample_rate", config.sample_rate),
            )
            for result in results
        ]


class SpeechGenerator:
    def __init__(self, config: TtsConfig, backend: SpeechBackend | None = None):
        self.config = config
        self.backend = backend or MlxAudioBackend()

    def generate(self, text: str) -> list[AudioChunk]:
        return self.backend.generate(text, self.config)
