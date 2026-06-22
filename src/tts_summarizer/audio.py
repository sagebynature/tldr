from __future__ import annotations

from collections.abc import Iterable as IterableABC
from typing import Any, Iterable, Iterator, Protocol, cast
import io
import math
import struct
import wave

from .speech import AudioChunk


class SupportsToList(Protocol):
    def tolist(self) -> Any: ...


def chunks_to_wav_bytes(chunks: Iterable[AudioChunk]) -> bytes:
    buffer = io.BytesIO()
    sample_rate: int | None = None
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        for chunk in chunks:
            if sample_rate is None:
                sample_rate = chunk.sample_rate
                wav.setframerate(sample_rate)
            elif chunk.sample_rate != sample_rate:
                raise ValueError("all audio chunks must use the same sample_rate")
            samples = _to_float_list(chunk.samples)
            wav.writeframes(b"".join(_to_i16(sample) for sample in samples))
        if sample_rate is None:
            wav.setframerate(8000)
    return buffer.getvalue()


def chunks_to_wav_stream(
    chunks: Iterable[AudioChunk], sample_rate: int = 8000
) -> Iterator[bytes]:
    yield _wav_stream_header(sample_rate)
    for chunk in chunks:
        if chunk.sample_rate != sample_rate:
            raise ValueError("all audio chunks use sample_rate")
        samples = _to_float_list(chunk.samples)
        yield b"".join(_to_i16(sample) for sample in samples)


def _wav_stream_header(sample_rate: int) -> bytes:
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", 0xFFFFFFFF),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16),
            b"data",
            struct.pack("<I", 0xFFFFFFFF),
        )
    )


def _to_float_list(samples: object) -> list[float]:
    if hasattr(samples, "tolist"):
        raw = cast(SupportsToList, samples).tolist()
    else:
        raw = samples
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        rows = cast(list[list[Any]], raw)
        return [float(item) for row in rows for item in row]
    if isinstance(raw, IterableABC) and not isinstance(raw, (str, bytes)):
        return [float(item) for item in cast(Iterable[Any], raw)]
    return [float(cast(Any, raw))]


def _to_i16(sample: float) -> bytes:
    value = max(-1.0, min(1.0, sample if math.isfinite(sample) else 0.0))
    return int(value * 32767).to_bytes(2, "little", signed=True)
