#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

TEXT_PATHS = (
    ("last_assistant_message",),
    ("payload", "last_assistant_message"),
    ("message",),
    ("payload", "message"),
    ("text",),
    ("payload", "text"),
)
SESSION_PATHS = (("session_id",), ("payload", "session_id"))
TRANSCRIPT_PATHS = (("transcript_path",), ("payload", "transcript_path"))


def nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def first_string(payload: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> str:
    for path in paths:
        value = nested_get(payload, path)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def assistant_text_from_transcript_item(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    message = item.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    ).strip()


def transcript_text(path: str) -> str:
    if not path:
        return ""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return ""
    last_text = ""
    try:
        with candidate.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = assistant_text_from_transcript_item(item)
                if text:
                    last_text = text
    except OSError:
        return ""
    return last_text


def load_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> int:
    log_file = Path(os.environ.get("CLAUDE_TTS_LOG", "/tmp/claude-tts.log"))
    payload_log = Path(os.environ.get("CLAUDE_TTS_PAYLOAD_LOG", "/tmp/claude-tts-payload.json"))
    state_file = Path(os.environ.get("CLAUDE_TTS_STATE_FILE", str(Path.home() / ".claude/tts.enabled")))
    tts_bin = os.environ.get("CLAUDE_TTS_BIN", "tts-summarizer")

    if not state_file.is_file():
        return 0

    raw = sys.stdin.read()
    if raw:
        payload_log.parent.mkdir(parents=True, exist_ok=True)
        payload_log.write_text(raw + "\n", encoding="utf-8")
    payload = load_payload(raw)

    session_id = os.environ.get("CLAUDE_TTS_SESSION_ID", "") or first_string(payload, SESSION_PATHS)
    text = os.environ.get("CLAUDE_TTS_TEXT", "") or first_string(payload, TEXT_PATHS)
    if not text:
        text = transcript_text(first_string(payload, TRANSCRIPT_PATHS))
    text = text or "Claude finished."

    if shutil.which(tts_bin) is None:
        append_log(log_file, f"claude_tts: {tts_bin} not found")
        return 0

    args = [tts_bin, "speak"]
    if session_id:
        args.extend(["--session_id", session_id])
    args.append(text)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = log_file.open("a", encoding="utf-8")
    try:
        subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
