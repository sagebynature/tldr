#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, cast


def _read_payload() -> dict[str, Any]:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _last_assistant_text(history: Any) -> str:
    if not isinstance(history, list):
        return ""
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        message = cast("dict[str, Any]", entry)
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                content_part = cast("dict[str, Any]", part)
                text_part = content_part.get("text")
                if isinstance(text_part, str):
                    parts.append(text_part)
            text = "".join(parts).strip()
            if text:
                return text
    return ""


def _assistant_text(payload: dict[str, Any]) -> str:
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        return ""
    response = extra.get("assistant_response")
    if isinstance(response, str) and response.strip():
        return response.strip()
    return _last_assistant_text(extra.get("conversation_history"))


def _session_id(payload: dict[str, Any]) -> str:
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return f"hermes:{session_id.strip()}"
    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        return f"hermes:{cwd.strip()}"
    return "hermes"


def _spawn(text: str, session_id: str) -> None:
    kwargs: dict[str, Any] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name != "nt":
        kwargs["start_new_session"] = True
    subprocess.Popen(
        ["tldr", "speak", "--session_id", session_id, text],
        **kwargs,
    )


def main() -> int:
    payload = _read_payload()
    try:
        if payload.get("hook_event_name") == "post_llm_call":
            text = _assistant_text(payload)
            if text:
                _spawn(text, _session_id(payload))
    except Exception as exc:
        print(f"TL;DR Hermes hook ignored error: {exc}", file=sys.stderr)
    print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
