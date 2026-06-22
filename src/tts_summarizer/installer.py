from __future__ import annotations

import json
import shlex
import shutil
from importlib import resources
from pathlib import Path
from typing import Any


def _copy_codex_hook(destination: Path) -> None:
    source_tree_hook = Path(__file__).resolve().parents[2] / "hooks" / "codex" / "codex_tts.py"
    if source_tree_hook.is_file():
        shutil.copyfile(source_tree_hook, destination)
        return
    resource = resources.files("tts_summarizer") / "hooks" / "codex_tts.py"
    with resources.as_file(resource) as source:
        shutil.copyfile(source, destination)


def _load_hooks(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _tts_bin() -> str:
    return shutil.which("tts-summarizer") or "tts-summarizer"


def _codex_command(installed_hook: Path) -> str:
    return (f"python3 {shlex.quote(str(installed_hook))}")


def _is_codex_tts_hook(command: Any, installed_hook: Path) -> bool:
    if not isinstance(command, str):
        return False
    if command == str(installed_hook):
        return True
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return bool(parts) and parts[-1] == str(installed_hook)


def _remove_existing_codex_tts_hooks(stop_entries: list[Any], installed_hook: Path) -> list[Any]:
    cleaned: list[Any] = []
    for entry in stop_entries:
        if not isinstance(entry, dict):
            cleaned.append(entry)
            continue
        hooks = entry.get("hooks")
        if not isinstance(hooks, list):
            cleaned.append(entry)
            continue
        entry["hooks"] = [
            hook
            for hook in hooks
            if not (
                isinstance(hook, dict)
                and _is_codex_tts_hook(hook.get("command"), installed_hook)
            )
        ]
        if entry["hooks"]:
            cleaned.append(entry)
    return cleaned


def _install_codex(home: Path) -> Path:
    codex_dir = home / ".codex"
    install_dir = codex_dir / "hooks" / "tts"
    installed_hook = install_dir / "codex_tts.py"
    hooks_json = codex_dir / "hooks.json"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_codex_hook(installed_hook)
    installed_hook.chmod(installed_hook.stat().st_mode | 0o700)
    (codex_dir / "tts.enabled").touch()

    data = _load_hooks(hooks_json)
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        data["hooks"] = hooks
    stop_entries = hooks.setdefault("Stop", [])
    if not isinstance(stop_entries, list):
        stop_entries = []

    stop_entries = _remove_existing_codex_tts_hooks(stop_entries, installed_hook)
    stop_entries.append(
        {
            "hooks": [
                {
                    "command": _codex_command(installed_hook),
                    "statusMessage": "Speaking completion",
                    "timeout": 5,
                    "type": "command",
                }
            ],
            "matcher": "*",
        }
    )
    hooks["Stop"] = stop_entries

    hooks_json.parent.mkdir(parents=True, exist_ok=True)
    hooks_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return installed_hook


def install_hook(harness: str, home: Path | None = None) -> Path:
    if harness != "codex":
        raise ValueError(f"unsupported harness: {harness}")
    return _install_codex(home or Path.home())
