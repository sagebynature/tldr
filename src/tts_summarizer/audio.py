from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable as IterableABC
from typing import Any, Iterable, Protocol, cast
import math
import subprocess
import time
import wave

from .config import AudioConfig
from .session import WorkToken
from .speech import AudioChunk


class SupportsToList(Protocol):
    def tolist(self) -> Any: ...


class AudioPlayer:
    def __init__(self, config: AudioConfig):
        self.config = config

    def play(
        self, chunks: Iterable[AudioChunk], token: WorkToken | None = None
    ) -> None:
        output_dir = Path(self.config.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        for chunk in chunks:
            if token is not None and token.cancelled():
                return
            path = output_dir / f"speech-{time.time_ns()}.wav"
            write_wav(path, chunk)
            if self.config.backend in {"auto", "ffplay"} and not self.config.save:
                proc = subprocess.Popen(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
                )
                while proc.poll() is None:
                    if token is not None and token.cancelled():
                        proc.terminate()
                        return
                    time.sleep(0.05)
            if self.config.backend == "file" or self.config.save:
                continue


def write_wav(path: Path, chunk: AudioChunk) -> None:
    samples = _to_float_list(chunk.samples)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(chunk.sample_rate)
        wav.writeframes(b"".join(_to_i16(sample) for sample in samples))


def _to_float_list(samples: object) -> list[float]:
    raw: Any
    if hasattr(samples, "tolist"):
        raw = cast(SupportsToList, samples).tolist()
    else:
        raw = samples
    return [float(item) for item in _flatten(raw)]


def _flatten(raw: Any) -> list[Any]:
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        rows = cast(list[Any], raw)
        return [item for row in rows for item in cast(Iterable[Any], row)]
    if isinstance(raw, IterableABC) and not isinstance(raw, (str, bytes)):
        return list(cast(Iterable[Any], raw))
    return [raw]


def _to_i16(sample: float) -> bytes:
    value = max(-1.0, min(1.0, sample if math.isfinite(sample) else 0.0))
    return int(value * 32767).to_bytes(2, byteorder="little", signed=True)
