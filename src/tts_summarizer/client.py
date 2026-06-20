from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

from .config import Config
from .state import read_state


class ClientError(RuntimeError):
    pass


def post_json(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def get_json(url: str, timeout: float) -> dict[str, object]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body else {}


def start_daemon(config_path: str | None) -> None:
    args = [sys.executable, "-m", "tts_summarizer", "serve"]
    if config_path:
        args.extend(["--config", config_path])
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def wait_for_state(config: Config) -> str | None:
    deadline = time.monotonic() + config.server.startup_timeout_ms / 1000
    while time.monotonic() < deadline:
        state = read_state(config)
        if state is not None:
            return state.base_url
        time.sleep(0.05)
    return None


def daemon_base_url(config: Config, config_path: str | None) -> str | None:
    state = read_state(config)
    if state is not None:
        return state.base_url
    if not config.server.auto_start:
        return None
    start_daemon(config_path)
    return wait_for_state(config)
