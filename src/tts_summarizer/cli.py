from __future__ import annotations

import argparse
import sys

from .config import ConfigError, load_config


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

    if args.command == "config-check":
        try:
            load_config(args.config)
        except ConfigError as exc:
            print(f"tts-summarizer config error: {exc}", file=sys.stderr)
            return 2
        return 0
    return 0
