from __future__ import annotations

from dataclasses import dataclass
import threading

from .config import SessionConfig
from .request import SpeechRequest


@dataclass(frozen=True)
class WorkToken:
    session_key: str
    generation: int
    manager: "SessionManager"

    def cancelled(self) -> bool:
        return self.manager.current_generation(self.session_key) != self.generation


class SessionManager:
    def __init__(self, config: SessionConfig):
        self.config = config
        self._lock = threading.Lock()
        self._generations: dict[str, int] = {}

    def begin(self, request: SpeechRequest) -> WorkToken:
        key = request.session_key()
        with self._lock:
            current = self._generations.get(key, 0)
            next_generation = (
                current + 1 if self.config.interrupt_same_session else current or 1
            )
            self._generations[key] = next_generation
            return WorkToken(key, next_generation, self)

    def finish(self, token: WorkToken) -> None:
        # Keep the generation number so stale workers can still observe cancellation.
        return None

    def current_generation(self, session_key: str) -> int:
        with self._lock:
            return self._generations.get(session_key, 0)
