#!/usr/bin/env bash
set -euo pipefail

harness="${1:-codex}"
exec tts-summarizer install --harness "$harness"
