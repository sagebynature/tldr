from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from .config import TtsConfig


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
        from mlx_audio.tts.utils import load_model

        self._model = cast(Any, load_model)(model_name)
        self._model_name = model_name
        return self._model

    def generate(self, text: str, config: TtsConfig) -> list[AudioChunk]:
        model = self._load(config.model)
        kwargs = {
            "voice": config.voice or None,
            "language": config.language or None,
            "speed": config.speed,
            "ref_audio": config.ref_audio or None,
            "ref_text": config.ref_text or None,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        chunks: list[AudioChunk] = []
        for result in model.generate(text=text, **kwargs):
            sample_rate = int(getattr(result, "sample_rate", getattr(model, "sample_rate", config.sample_rate)))
            chunks.append(AudioChunk(samples=result.audio, sample_rate=sample_rate))
        return chunks


class SpeechGenerator:
    def __init__(self, config: TtsConfig, backend: SpeechBackend | None = None):
        self.config = config
        self.backend = backend or MlxAudioBackend()

    def generate(self, text: str) -> list[AudioChunk]:
        return self.backend.generate(text, self.config)
