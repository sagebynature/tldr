from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import json
import logging
from typing import Any, Protocol, cast
from urllib.request import Request, urlopen

from .config import TtsConfig, TtsProfileConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    samples: object
    sample_rate: int

@dataclass(frozen=True)
class AudioBytes:
    chunks: Iterable[bytes]
    content_type: str = "audio/wav"


SpeechOutput = Iterable[AudioChunk] | AudioBytes


class SpeechBackend(Protocol):
    def generate(self, text: str, config: TtsProfileConfig) -> SpeechOutput: ...


def _patch_kokoro_sinegen() -> None:
    try:
        from mlx_audio.tts.models.kokoro import istftnet
    except ImportError:
        return

    sine_gen = istftnet.SineGen
    if getattr(sine_gen, "_tts_summarizer_patched", False):
        return

    mx = istftnet.mx

    def patched_call(self, f0):
        fn = f0 * mx.arange(1, self.harmonic_num + 2)[None, None, :]
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        length = min(sine_waves.shape[1], uv.shape[1])
        if sine_waves.shape[1] != uv.shape[1]:
            logger.warning(
                "cropping Kokoro sine source length from %s/%s to %s",
                sine_waves.shape[1],
                uv.shape[1],
                length,
            )
            sine_waves = sine_waves[:, :length, :]
            uv = uv[:, :length, :]
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * mx.random.normal(sine_waves.shape)
        return sine_waves * uv + noise, uv, noise

    setattr(sine_gen, "_tts_summarizer_patched", True)
    setattr(sine_gen, "__call__", patched_call)


class MlxAudioBackend:
    def __init__(self):
        self._models: dict[str, object] = {}
        self._load_model = self._default_load_model

    def _default_load_model(self, model_name: str):
        if "kokoro" in model_name.lower():
            # ponytail: MLX-Audio 0.4.4 Kokoro can emit sine/noise lengths off by one frame.
            _patch_kokoro_sinegen()
        from mlx_audio.tts.utils import load_model

        return cast(Any, load_model)(model_name)

    def _load(self, model_name: str):
        if model_name in self._models:
            return self._models[model_name]
        logger.info("loading tts model=%s", model_name)
        model = self._load_model(model_name)
        self._models[model_name] = model
        logger.info("tts model ready model=%s", model_name)
        return model

    def generate(self, text: str, config: TtsProfileConfig) -> Iterable[AudioChunk]:
        model = cast(Any, self._load(config.model))
        kwargs = dict(config.generate_kwargs)
        kwargs.setdefault("stream", config.stream)
        logger.info(
            "calling tts generate model=%s kwargs=%s text_chars=%s",
            config.model,
            sorted(kwargs),
            len(text),
        )
        results = model.generate(text=text, **kwargs)
        return (
            AudioChunk(
                samples=getattr(result, "audio", result),
                sample_rate=getattr(result, "sample_rate", config.sample_rate),
            )
            for result in results
        )


class RemoteTtsBackend:
    def __init__(self, urlopen: Any = urlopen, timeout: float = 30):
        self.urlopen = urlopen
        self.timeout = timeout

    def generate(self, text: str, config: TtsProfileConfig) -> AudioBytes:
        if not config.base_url:
            raise ValueError("remote TTS profile requires base_url")
        body = {
            "model": config.model,
            "input": text,
            "stream": config.stream,
            **config.generate_kwargs,
            "response_format": "wav",
        }
        body_bytes = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = Request(
            f"{config.base_url.rstrip('/')}/audio/speech",
            data=body_bytes,
            headers=headers,
            method="POST",
        )

        def chunks() -> Iterable[bytes]:
            with cast(Any, self.urlopen(request, timeout=self.timeout)) as response:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return AudioBytes(chunks())


class RoutingSpeechBackend:
    def __init__(
        self,
        local: SpeechBackend | None = None,
        remote: SpeechBackend | None = None,
    ):
        self.local = local or MlxAudioBackend()
        self.remote = remote or RemoteTtsBackend()

    def generate(self, text: str, config: TtsProfileConfig) -> SpeechOutput:
        if config.backend == "mlx":
            return self.local.generate(text, config)
        if config.backend == "remote":
            return self.remote.generate(text, config)
        raise ValueError(f"unknown TTS backend: {config.backend}")


class SpeechGenerator:
    def __init__(self, config: TtsConfig, backend: SpeechBackend | None = None):
        self.config = config
        self.backend = backend or RoutingSpeechBackend()

    def profile(self, profile_name: str | None = None) -> TtsProfileConfig:
        name = profile_name or self.config.default_profile
        try:
            return self.config.profiles[name]
        except KeyError as exc:
            raise ValueError(f"unknown TTS profile: {name}") from exc

    def sample_rate(self, profile_name: str | None = None) -> int:
        return self.profile(profile_name).sample_rate

    def generate(
        self, text: str, profile_name: str | None = None
    ) -> SpeechOutput:
        return self.backend.generate(text, self.profile(profile_name))
