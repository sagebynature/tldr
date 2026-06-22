from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


class RequestError(ValueError):
    pass


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    session_id: str
    caller: str = "default"
    metadata: dict[str, object] = field(default_factory=dict)
    summarize: bool = True
    tts_profile: str | None = None
    summarizer_profile: str | None = None

    @classmethod
    def from_json(
        cls,
        data: dict[str, object],
        caller: str | None = None,
        session_id: str | None = None,
    ) -> "SpeechRequest":
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RequestError("normalized request requires non-empty text")
        metadata = data.get("metadata")
        summarize = data.get("summarize")
        tts_profile = data.get("tts_profile")
        summarizer_profile = data.get("summarizer_profile")
        clean_metadata: dict[str, object] = {}
        if isinstance(metadata, dict):
            clean_metadata = {str(key): value for key, value in metadata.items()}
        return cls(
            text=text,
            caller=caller or "default",
            session_id=session_id or fallback_session_id(),
            metadata=clean_metadata,
            summarize=summarize if isinstance(summarize, bool) else True,
            tts_profile=tts_profile.strip() if isinstance(tts_profile, str) else None,
            summarizer_profile=(
                summarizer_profile.strip()
                if isinstance(summarizer_profile, str)
                else None
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "summarize": self.summarize,
            "tts_profile": self.tts_profile,
            "summarizer_profile": self.summarizer_profile,
        }

    def session_key(self) -> str:
        return f"{self.caller}:{self.session_id}"


def fallback_session_id() -> str:
    cwd = Path.cwd()
    ppid = os.getppid()
    return f"{cwd}:{ppid}"
