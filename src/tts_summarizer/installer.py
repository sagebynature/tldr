from __future__ import annotations

import json
import shlex
import shutil
from importlib import resources
from pathlib import Path
from typing import Any

HOOK_FILENAMES = {
    "codex": "codex_tts.py",
    "claude": "claude_tts.py",
    "omp": "omp_tts.ts",
}


def _copy_hook(harness: str, destination: Path) -> None:
    filename = HOOK_FILENAMES[harness]
    source_tree_hook = Path(__file__).resolve().parents[2] / "hooks" / harness / filename
    if source_tree_hook.is_file():
        shutil.copyfile(source_tree_hook, destination)
        return

    resource = resources.files("tts_summarizer") / "hooks" / filename
    with resources.as_file(resource) as source:
        shutil.copyfile(source, destination)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _is_hook_for_script(hook: Any, installed_hook: Path) -> bool:
    if not isinstance(hook, dict):
        return False
    installed = str(installed_hook)
    args = hook.get("args")
    if isinstance(args, list) and installed in args:
        return True
    command = hook.get("command")
    if not isinstance(command, str):
        return False
    if command == installed:
        return True
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return bool(parts) and parts[-1] == installed


def _remove_existing_tts_hooks(stop_entries: list[Any], installed_hook: Path) -> list[Any]:
    cleaned: list[Any] = []
    for entry in stop_entries:
        if not isinstance(entry, dict):
            cleaned.append(entry)
            continue
        hooks = entry.get("hooks")
        if not isinstance(hooks, list):
            cleaned.append(entry)
            continue
        entry["hooks"] = [hook for hook in hooks if not _is_hook_for_script(hook, installed_hook)]
        if entry["hooks"]:
            cleaned.append(entry)
    return cleaned


def _hooks_root(data: dict[str, Any]) -> dict[str, Any]:
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        data["hooks"] = hooks
    return hooks


def _stop_entries(hooks: dict[str, Any]) -> list[Any]:
    entries = hooks.setdefault("Stop", [])
    return entries if isinstance(entries, list) else []


def _install_codex(home: Path) -> Path:
    codex_dir = home / ".codex"
    install_dir = codex_dir / "hooks" / "tts"
    installed_hook = install_dir / "codex_tts.py"
    hooks_json = codex_dir / "hooks.json"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_hook("codex", installed_hook)
    installed_hook.chmod(installed_hook.stat().st_mode | 0o700)
    (codex_dir / "tts.enabled").touch()

    data = _load_json(hooks_json)
    hooks = _hooks_root(data)
    stop_entries = _remove_existing_tts_hooks(_stop_entries(hooks), installed_hook)
    stop_entries.append(
        {
            "hooks": [
                {
                    "command": f"python3 {shlex.quote(str(installed_hook))}",
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


def _install_claude(home: Path) -> Path:
    claude_dir = home / ".claude"
    install_dir = claude_dir / "hooks" / "tts"
    installed_hook = install_dir / "claude_tts.py"
    settings_json = claude_dir / "settings.json"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_hook("claude", installed_hook)
    installed_hook.chmod(installed_hook.stat().st_mode | 0o700)
    (claude_dir / "tts.enabled").touch()

    data = _load_json(settings_json)
    hooks = _hooks_root(data)
    stop_entries = _remove_existing_tts_hooks(_stop_entries(hooks), installed_hook)
    stop_entries.append(
        {
            "hooks": [
                {
                    "command": "python3",
                    "args": [str(installed_hook)],
                    "statusMessage": "Speaking completion",
                    "timeout": 5,
                    "type": "command",
                }
            ]
        }
    )
    hooks["Stop"] = stop_entries

    settings_json.parent.mkdir(parents=True, exist_ok=True)
    settings_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return installed_hook


def _install_omp(home: Path) -> Path:
    install_dir = home / ".omp" / "agent" / "extensions"
    installed_hook = install_dir / "tts-summarizer.ts"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_hook("omp", installed_hook)
    return installed_hook


def install_hook(harness: str, home: Path | None = None) -> Path:
    root = home or Path.home()
    if harness == "codex":
        return _install_codex(root)
    if harness == "claude":
        return _install_claude(root)
    if harness == "omp":
        return _install_omp(root)
    raise ValueError(f"unsupported harness: {harness}")
