from __future__ import annotations

import argparse
import sys

from .client import daemon_base_url, get_json, post_json
from .config import ConfigError, load_config
from .request import RequestError, SpeechRequest
from .server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tts-summarizer")
    subcommands = parser.add_subparsers(dest="command", required=True)

    config_check = subcommands.add_parser("config-check")
    config_check.add_argument("--config")

    speak = subcommands.add_parser("speak")
    speak.add_argument("--config")
    speak.add_argument("--session-id")
    speak.add_argument("--caller")
    speak.add_argument("--text")

    serve = subcommands.add_parser("serve")
    serve.add_argument("--config")

    health = subcommands.add_parser("health")
    health.add_argument("--config")

    stop = subcommands.add_parser("stop")
    stop.add_argument("--config")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)

    try:
        config = load_config(getattr(args, "config", None))
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
        if args.command == "speak":
            request = SpeechRequest.from_cli(args.text, sys.stdin.read(), args.caller, args.session_id)
            post_json(f"{base_url}/v1/speak", request.to_json(), timeout)
            return 0
        if args.command == "health":
            print(get_json(f"{base_url}/health", timeout))
            return 0
        if args.command == "stop":
            post_json(f"{base_url}/shutdown", {}, timeout)
            return 0
    except RequestError as exc:
        print(f"tts-summarizer request error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"tts-summarizer request failed: {exc}", file=sys.stderr)
        return 0

    return 0
