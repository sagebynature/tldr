from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import time

from .config import Config


@dataclass(frozen=True)
class DaemonState:
    host: str
    port: int
    pid: int
    config_fingerprint: str
    started_at: float

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def state_path(config: Config) -> Path:
    return Path(config.server.state_dir).expanduser() / "daemon.json"


def config_fingerprint(config: Config) -> str:
    basis = repr(
        (config.server, config.session, config.summarizer, config.tts, config.audio)
    ).encode("utf-8")
    return hashlib.sha256(basis).hexdigest()[:16]


def write_state(config: Config, host: str, port: int, pid: int) -> Path:
    path = state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": host,
        "port": port,
        "pid": pid,
        "config_fingerprint": config_fingerprint(config),
        "started_at": time.time(),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_state(config: Config) -> DaemonState | None:
    path = state_path(config)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = DaemonState(
            host=str(payload["host"]),
            port=int(payload["port"]),
            pid=int(payload["pid"]),
            config_fingerprint=str(payload["config_fingerprint"]),
            started_at=float(payload["started_at"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        path.unlink(missing_ok=True)
        return None
    if not _pid_exists(state.pid):
        path.unlink(missing_ok=True)
        return None
    return state
