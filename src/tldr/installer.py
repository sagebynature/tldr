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
    "omp": "tts.ts",
    "pi": "tts.ts",
    "hermes": "hermes_tts.py",
}


def _copy_hook(harness: str, destination: Path) -> None:
    filename = HOOK_FILENAMES[harness]
    source_tree_hook = (
        Path(__file__).resolve().parents[2] / "hooks" / harness / filename
    )
    if source_tree_hook.is_file():
        shutil.copyfile(source_tree_hook, destination)
        return

    resource = resources.files("tldr") / "hooks" / filename
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


def _remove_existing_tts_hooks(
    stop_entries: list[Any], installed_hook: Path
) -> list[Any]:
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
            hook for hook in hooks if not _is_hook_for_script(hook, installed_hook)
        ]
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
    installed_hook = install_dir / "tldr.ts"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_hook("omp", installed_hook)
    return installed_hook


def _install_pi(home: Path) -> Path:
    install_dir = home / ".pi" / "agent" / "extensions"
    installed_hook = install_dir / "tldr.ts"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_hook("pi", installed_hook)
    return installed_hook


def _quote_yaml_string(value: str) -> str:
    return json.dumps(value)


def _hermes_hook_block(command: str) -> str:
    return (
        "  post_llm_call:\n"
        f"    - command: {_quote_yaml_string(command)}\n"
        "      timeout: 5\n"
    )


def _insert_into_hooks_block(existing: str, block: str) -> str | None:
    lines = existing.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    for index, line in enumerate(lines):
        if line.strip() != "hooks:" or line.startswith((" ", "\t")):
            continue

        insert_at = index + 1
        while insert_at < len(lines):
            stripped = lines[insert_at].strip()
            if stripped and not lines[insert_at].startswith((" ", "\t")):
                break
            insert_at += 1

        lines.insert(insert_at, block)
        return "".join(lines)

    return None


def _ensure_hermes_config_entry(config_yaml: Path, installed_hook: Path) -> None:
    command = str(installed_hook)
    existing = config_yaml.read_text(encoding="utf-8") if config_yaml.exists() else ""
    if command in existing:
        return

    block = _hermes_hook_block(command)
    updated = _insert_into_hooks_block(existing, block)
    if updated is None:
        prefix = "" if not existing or existing.endswith("\n") else "\n"
        updated = existing + f"{prefix}hooks:\n{block}"

    config_yaml.parent.mkdir(parents=True, exist_ok=True)
    config_yaml.write_text(updated, encoding="utf-8")


def _install_hermes(home: Path) -> Path:
    hermes_dir = home / ".hermes"
    install_dir = hermes_dir / "agent-hooks" / "tldr"
    installed_hook = install_dir / "hermes_tts.py"

    install_dir.mkdir(parents=True, exist_ok=True)
    _copy_hook("hermes", installed_hook)
    installed_hook.chmod(installed_hook.stat().st_mode | 0o700)
    (hermes_dir / "tts.enabled").touch()
    _ensure_hermes_config_entry(hermes_dir / "config.yaml", installed_hook)
    return installed_hook


def install_hook(harness: str, home: Path | None = None) -> Path:
    root = home or Path.home()
    if harness == "codex":
        return _install_codex(root)
    if harness == "claude":
        return _install_claude(root)
    if harness == "omp":
        return _install_omp(root)
    if harness == "pi":
        return _install_pi(root)
    if harness == "hermes":
        return _install_hermes(root)
    raise ValueError(f"unsupported harness: {harness}")
