from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os


class RequestError(ValueError):
    pass


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    session_id: str
    caller: str = "default"
    event: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "SpeechRequest":
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RequestError("normalized request requires non-empty text")
        caller = data.get("caller")
        session_id = data.get("session_id")
        event = data.get("event")
        metadata = data.get("metadata")
        clean_metadata: dict[str, object] = {}
        if isinstance(metadata, dict):
            clean_metadata = {str(key): value for key, value in metadata.items()}
        return cls(
            text=text,
            caller=caller if isinstance(caller, str) and caller else "default",
            session_id=session_id if isinstance(session_id, str) and session_id else fallback_session_id(),
            event=event if isinstance(event, str) else "",
            metadata=clean_metadata,
        )

    @classmethod
    def from_cli(
        cls,
        text: str | None,
        stdin_text: str,
        caller: str | None,
        session_id: str | None,
    ) -> "SpeechRequest":
        if text is not None:
            return cls(
                text=text,
                caller=caller or "default",
                session_id=session_id or fallback_session_id(),
            )
        stripped = stdin_text.strip()
        if not stripped:
            raise RequestError("provide --text or normalized JSON on stdin")
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = {"text": stdin_text}
        if not isinstance(payload, dict):
            raise RequestError("stdin JSON must be an object")
        if caller is not None:
            payload["caller"] = caller
        if session_id is not None:
            payload["session_id"] = session_id
        return cls.from_json(payload)

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "session_id": self.session_id,
            "caller": self.caller,
            "event": self.event,
            "metadata": self.metadata,
        }

    def session_key(self) -> str:
        return f"{self.caller}:{self.session_id}"


def fallback_session_id() -> str:
    cwd = Path.cwd()
    ppid = os.getppid()
    return f"{cwd}:{ppid}"
