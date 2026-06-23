#!/usr/bin/env bash
set -euo pipefail

harness="${1:-codex}"
exec tldr install --harness "$harness"
