from __future__ import annotations

import argparse
import importlib.resources as resources
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
from urllib.parse import quote

from .client import daemon_base_url, get_json, post_json
from .config import Config, ConfigError, load_config
from .logging_setup import setup_logging
from .server import run_server
from .installer import install_hook


DEFAULT_SPEAK_CONFIG = "~/.config/tts-summarizer/config.toml"
DEFAULT_USER_CONFIG = Path("~/.config/tts-summarizer/config.toml")
CONFIG_PROFILE_RESOURCES = {
    "remote": "config.remote.example.toml",
    "apple-local": "config.apple-local.example.toml",
}



def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("--summarize must be true or false")


def _load_speak_config(path: str | None) -> Config:
    if path:
        return load_config(path)
    default = Path(DEFAULT_SPEAK_CONFIG).expanduser()
    if default.exists():
        return load_config(str(default))
    return load_config(None)


def _session_pid_path(config: Config, session_id: str) -> Path:
    return (
        Path(config.server.state_dir).expanduser()
        / "sessions"
        / f"{quote(session_id, safe='')}.pid"
    )


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_session(config: Config, session_id: str) -> None:
    path = _session_pid_path(config, session_id)
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _write_session_pid(config: Config, session_id: str, pid: int) -> Path:
    path = _session_pid_path(config, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid), encoding="utf-8")
    return path


def _clear_session_pid(config: Config, session_id: str, pid: int) -> None:
    path = _session_pid_path(config, session_id)
    try:
        current = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return
    if current == pid:
        path.unlink()


def _speak(args: argparse.Namespace) -> int:
    try:
        config = _load_speak_config(args.config)
        setup_logging(config)
    except ConfigError as exc:
        print(f"tts-summarizer config error: {exc}", file=sys.stderr)
        return 2

    host = args.server or config.server.host or "127.0.0.1"
    port = args.port or config.server.port or 9000
    body = json.dumps(
        {"text": " ".join(args.text_to_summarize), "summarize": args.summarize},
        separators=(",", ":"),
    )
    curl_args = [
        "curl",
        "-sS",
        "-D",
        "/dev/stderr",
        "-H",
        "Content-Type: application/json",
        "-H",
        "X-TTS-Caller: manual",
    ]
    if args.session_id:
        curl_args.extend(["-H", f"X-TTS-Session-Id: {args.session_id}"])
    curl_args.extend(["-d", body, f"http://{host}:{port}/v1/speak"])
    ffplay_args = [
        "ffplay",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
    ]

    if args.session_id:
        _stop_session(config, args.session_id)

    curl = None
    try:
        curl = subprocess.Popen(curl_args, stdout=subprocess.PIPE)
        player = subprocess.Popen(ffplay_args, stdin=curl.stdout)
        if args.session_id:
            _write_session_pid(config, args.session_id, player.pid)
        if curl.stdout:
            curl.stdout.close()
        try:
            player.wait()
            curl.wait()
        finally:
            if args.session_id:
                _clear_session_pid(config, args.session_id, player.pid)
    except Exception as exc:
        if curl is not None:
            if curl.stdout:
                curl.stdout.close()
            try:
                curl.terminate()
            except ProcessLookupError:
                pass
            try:
                curl.wait()
            except Exception:
                pass
        print(f"tts-summarizer request failed: {exc}", file=sys.stderr)
        return 0
    return 0


def _read_config_resource(resource_name: str) -> str:
    resource = resources.files("tts_summarizer").joinpath(resource_name)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        source_checkout_resource = Path(__file__).resolve().parents[2] / resource_name
        return source_checkout_resource.read_text(encoding="utf-8")


def _init_config(args: argparse.Namespace) -> int:
    config_path = Path.home() / DEFAULT_USER_CONFIG.relative_to("~")
    if config_path.exists() and not args.force:
        print(
            f"tts-summarizer config exists: {config_path} (use --force overwrite)",
            file=sys.stderr,
        )
        return 2

    resource_name = CONFIG_PROFILE_RESOURCES[args.profile]
    text = _read_config_resource(resource_name)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    print(config_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tts-summarizer")
    subcommands = parser.add_subparsers(dest="command", required=True)

    config_check = subcommands.add_parser("config-check")
    config_check.add_argument("--config")

    init_config = subcommands.add_parser("init-config")
    init_config.add_argument("--profile", choices=sorted(CONFIG_PROFILE_RESOURCES), default="remote")
    init_config.add_argument("--force", action="store_true")

    serve = subcommands.add_parser("serve")
    serve.add_argument("--config")

    health = subcommands.add_parser("health")
    health.add_argument("--config")

    stop = subcommands.add_parser("stop")
    stop.add_argument("--config")

    install = subcommands.add_parser("install")
    install.add_argument(
        "--harness", choices=["codex", "claude", "omp", "pi", "hermes"], required=True
    )

    speak = subcommands.add_parser("speak")
    speak.add_argument("--config")
    speak.add_argument("--server")
    speak.add_argument("--port", type=int)
    speak.add_argument("--session_id")
    speak.add_argument("--summarize", type=_parse_bool, default=True)
    speak.add_argument("text_to_summarize", nargs="+")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    if args.command == "init-config":
        return _init_config(args)

    if args.command == "install":
        installed = install_hook(args.harness)
        print(f"Installed {args.harness} TTS hook: {installed}")
        return 0

    if args.command == "speak":
        return _speak(args)

    try:
        config = load_config(getattr(args, "config", None))
        setup_logging(config)
    except ConfigError as exc:
        print(f"tts-summarizer config error: {exc}", file=sys.stderr)
        return 2 if args.command == "config-check" else 0

    if args.command == "config-check":
        return 0

    if args.command == "serve":
        return run_server(config)

    base_url = daemon_base_url(config, getattr(args, "config", None))
    if base_url is None:
        print("tts-summarizer daemon unavailable", file=sys.stderr)
        return 0

    timeout = config.server.request_timeout_ms / 1000
    try:
        if args.command == "health":
            print(get_json(f"{base_url}/health", timeout))
            return 0
        if args.command == "stop":
            post_json(f"{base_url}/shutdown", {}, timeout)
            return 0
    except Exception as exc:
        print(f"tts-summarizer request failed: {exc}", file=sys.stderr)
        return 0
    return 0
