#!/usr/bin/env bash
set -euo pipefail

harness="${1:-codex}"
exec echobrief install --harness "$harness"
