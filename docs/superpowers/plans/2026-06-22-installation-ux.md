# Installation UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tts-summarizer` installable for non-developers with one clear install path, one config initializer, and a Docker daemon path for remote model backends.

**Architecture:** Keep the CLI small: `init-config` lives in `src/tts_summarizer/cli.py` beside existing subcommands and copies packaged TOML examples with `importlib.resources`. Root example configs double as Docker/user-facing artifacts and are force-included into the wheel so installed CLIs can write them. Docker is only a daemon wrapper around the installed package and a mounted config.

**Tech Stack:** Python 3.11 stdlib, `argparse`, `importlib.resources`, existing `unittest` tests, Hatchling wheel metadata, Docker Compose, no new Python dependencies.

## Global Constraints

- No PyPI publishing.
- No model server installation or management.
- No local MLX models bundled into Docker.
- No provider discovery, endpoint health probing, or interactive setup prompts.
- No new dependency for config generation.
- `init-config` default profile is `remote`.
- User config path is `~/.config/tts-summarizer/config.toml`.
- Existing config is not overwritten unless `--force` is present.
- Docker supports remote model backends only.
- Shell commands in this repo use `rtk` prefix.

---

## File Structure

- Create `config.remote.example.toml`: root user/Docker remote-only config example.
- Create `config.apple-local.example.toml`: root Apple Silicon local MLX config example with remote examples retained.
- Modify `pyproject.toml`: package both root config examples into wheel resources and include them in sdist.
- Modify `src/tts_summarizer/cli.py`: add `init-config` parser, resource copy helper, and dispatch before config loading.
- Modify `tests/test_cli_commands.py`: add focused `init-config` CLI tests.
- Modify `tests/test_config.py`: parse the two example configs so config keys stay valid.
- Create `Dockerfile`: install package image and run the daemon.
- Create `docker-compose.yml`: mount `./config.toml` read-only, expose `9200`, run remote-only daemon.
- Modify `README.md`: reorder installation UX before development commands.

---

### Task 1: Example configs packaged as resources

**Files:**
- Create: `config.remote.example.toml`
- Create: `config.apple-local.example.toml`
- Modify: `pyproject.toml`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces package resources read later by CLI:
  - `tts_summarizer/config.remote.example.toml`
  - `tts_summarizer/config.apple-local.example.toml`
- Produces valid TOML accepted by `tts_summarizer.config.load_config(path)`.

- [ ] **Step 1: Write failing config example tests**

Append these tests to `tests/test_config.py` inside `ConfigTests`:

```python
    def test_remote_example_config_loads_remote_profiles(self):
        cfg = load_config("config.remote.example.toml", cwd=Path.cwd(), home=Path.home())

        self.assertEqual(cfg.server.host, "0.0.0.0")
        self.assertEqual(cfg.server.port, 9200)
        self.assertEqual(cfg.summarizer.default_profile, "remote-qwen25")
        self.assertEqual(cfg.summarizer.profiles["remote-qwen25"].base_url, "http://127.0.0.1:9000/v1")
        self.assertEqual(cfg.tts.default_profile, "remote-kokoro")
        self.assertEqual(cfg.tts.profiles["remote-kokoro"].backend, "remote")
        self.assertEqual(cfg.tts.profiles["remote-kokoro"].base_url, "http://127.0.0.1:9000/v1")

    def test_apple_local_example_config_loads_local_defaults(self):
        cfg = load_config("config.apple-local.example.toml", cwd=Path.cwd(), home=Path.home())

        self.assertEqual(cfg.server.host, "127.0.0.1")
        self.assertEqual(cfg.server.port, 9200)
        self.assertEqual(cfg.summarizer.default_profile, "qwen25")
        self.assertEqual(cfg.tts.default_profile, "kokoro")
        self.assertEqual(cfg.tts.profiles["kokoro"].backend, "local")
        self.assertEqual(cfg.tts.profiles["remote-kokoro"].backend, "remote")
```

- [ ] **Step 2: Run tests verify fail**

Run:

```bash
rtk uv run python -m unittest tests.test_config -v
```

Expected: FAIL because `config.remote.example.toml` and `config.apple-local.example.toml` do not exist.

- [ ] **Step 3: Create remote example config**

Create `config.remote.example.toml`:

```toml
[server]
host = "0.0.0.0"
port = 9200
state_dir = "~/.cache/tts-summarizer"
auto_start = true
startup_timeout_ms = 3000
request_timeout_ms = 5000

[summarizer]
default_profile = "remote-qwen25"

[summarizer.profiles.remote-qwen25]
enabled = true
base_url = "http://127.0.0.1:9000/v1"
api_key = "omlx"
model = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
word_threshold = 0
max_words = 40
temperature = 0.1
max_tokens = 180
system_prompt = """
You summarize assistant responses text-to-speech.
Return only final spoken summary.
Do not include reasoning, analysis, planning, explanations, prefaces, markdown, code fences, file paths, URLs, bullets, or formatting.
If content question, preserve question instead answering it.
"""
user_prompt_template = """
Write one complete spoken summary in {max_words} words fewer.
Stop summary.

{text}
"""

[summarizer.profiles.remote-qwen25.extra_body.chat_template_kwargs]
enable_thinking = false

[tts]
default_profile = "remote-kokoro"

[tts.profiles.remote-qwen]
backend = "remote"
base_url = "http://127.0.0.1:9000/v1"
api_key = "omlx"
model = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
stream = true
sample_rate = 24000

[tts.profiles.remote-qwen.generate_kwargs]
voice = "Aiden"
lang_code = "english"
streaming_interval = 0.32

[tts.profiles.remote-kokoro]
backend = "remote"
base_url = "http://127.0.0.1:9000/v1"
api_key = "omlx"
model = "mlx-community/Kokoro-82M-bf16"
stream = true
sample_rate = 24000

[tts.profiles.remote-kokoro.generate_kwargs]
voice = "af_heart"
lang_code = "a"
response_format = "wav"
```

- [ ] **Step 4: Create Apple local example config**

Create `config.apple-local.example.toml` by copying existing `config.toml` content exactly. It already has Apple Silicon local MLX defaults plus useful remote examples:

- `[server] host = "127.0.0.1"`, `port = 9200`
- `[summarizer] default_profile = "qwen25"`
- local TTS profiles `qwen` and `kokoro`
- remote TTS profiles `remote-qwen` and `remote-kokoro`

Do not delete the root development `config.toml`; it remains the repo-local config.

- [ ] **Step 5: Package examples into wheel and sdist**

Update `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/tts_summarizer"]
force-include = {
  "src/tts_summarizer/logging.conf" = "tts_summarizer/logging.conf",
  "hooks/codex/codex_tts.py" = "tts_summarizer/hooks/codex_tts.py",
  "config.remote.example.toml" = "tts_summarizer/config.remote.example.toml",
  "config.apple-local.example.toml" = "tts_summarizer/config.apple-local.example.toml",
}
```

Update the existing sdist include list to include root example files:

```toml
[tool.hatch.build.targets.sdist]
include = [
  "src",
  "hooks",
  "tests",
  "README.md",
  "CHANGELOG.md",
  "Makefile",
  "config.remote.example.toml",
  "config.apple-local.example.toml",
]
```

- [ ] **Step 6: Run config tests verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_config -v
```

Expected: PASS.

- [ ] **Step 7: Build package verify resources included**

Run:

```bash
rtk uv build
```

Expected: build succeeds. The wheel contains `tts_summarizer/config.remote.example.toml` and `tts_summarizer/config.apple-local.example.toml`.

- [ ] **Step 8: Commit**

Run:

```bash
rtk git add config.remote.example.toml config.apple-local.example.toml pyproject.toml tests/test_config.py
rtk git commit -m "feat: add install config examples"
```

---

### Task 2: `init-config` CLI command

**Files:**
- Modify: `src/tts_summarizer/cli.py`
- Modify: `tests/test_cli_commands.py`

**Interfaces:**
- Produces CLI command: `tts-summarizer init-config [--profile remote|apple-local] [--force]`.
- Writes: `Path.home() / ".config" / "tts-summarizer" / "config.toml"`.
- Reads package resources from Task 1.
- Returns `0` on write, `2` when refusing overwrite.

- [ ] **Step 1: Write failing CLI tests**

Add imports to `tests/test_cli_commands.py`:

```python
import io
```

Append these tests inside `CliCommandTests`:

```python
    def _run_cli_with_home(self, argv, home):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch("tts_summarizer.cli.Path.home", return_value=home), \
             mock.patch("sys.stdout", new=stdout), \
             mock.patch("sys.stderr", new=stderr):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_init_config_writes_remote_config_to_user_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()

            code, stdout, stderr = self._run_cli_with_home(["init-config"], home)

            path = home / ".config" / "tts-summarizer" / "config.toml"
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(path), stdout)
            cfg = load_config(str(path), cwd=Path(tmp), home=home)
            self.assertEqual(cfg.summarizer.default_profile, "remote-qwen25")
            self.assertEqual(cfg.tts.default_profile, "remote-kokoro")

    def test_init_config_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            path = home / ".config" / "tts-summarizer" / "config.toml"
            path.parent.mkdir(parents=True)
            path.write_text("sentinel", encoding="utf-8")

            code, stdout, stderr = self._run_cli_with_home(["init-config"], home)

            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn(str(path), stderr)
            self.assertEqual(path.read_text(encoding="utf-8"), "sentinel")

    def test_init_config_force_overwrites_existing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            path = home / ".config" / "tts-summarizer" / "config.toml"
            path.parent.mkdir(parents=True)
            path.write_text("sentinel", encoding="utf-8")

            code, stdout, stderr = self._run_cli_with_home(["init-config", "--force"], home)

            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(path), stdout)
            self.assertNotEqual(path.read_text(encoding="utf-8"), "sentinel")

    def test_init_config_writes_apple_local_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            home.mkdir()

            code, stdout, stderr = self._run_cli_with_home(
                ["init-config", "--profile", "apple-local"], home
            )

            path = home / ".config" / "tts-summarizer" / "config.toml"
            self.assertEqual(code, 0)
            self.assertEqual(stderr, "")
            self.assertIn(str(path), stdout)
            cfg = load_config(str(path), cwd=Path(tmp), home=home)
            self.assertEqual(cfg.summarizer.default_profile, "qwen25")
            self.assertEqual(cfg.tts.default_profile, "kokoro")
```

Also update the import line:

```python
from tts_summarizer.config import load_config
```

- [ ] **Step 2: Run tests verify fail**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands -v
```

Expected: FAIL because `init-config` parser command does not exist.

- [ ] **Step 3: Add CLI resource copy implementation**

Update `src/tts_summarizer/cli.py` imports:

```python
import argparse
import importlib.resources as resources
import json
import os
from pathlib import Path
import subprocess
import sys
import signal
from urllib.parse import quote
```

Add constants and helper near `DEFAULT_SPEAK_CONFIG`:

```python
DEFAULT_USER_CONFIG = Path("~/.config/tts-summarizer/config.toml")
CONFIG_PROFILE_RESOURCES = {
    "remote": "config.remote.example.toml",
    "apple-local": "config.apple-local.example.toml",
}
```

Add this function before `build_parser()`:

```python
def _init_config(args: argparse.Namespace) -> int:
    config_path = Path.home() / ".config" / "tts-summarizer" / "config.toml"
    if config_path.exists() and not args.force:
        print(
            f"tts-summarizer config exists: {config_path} (use --force to overwrite)",
            file=sys.stderr,
        )
        return 2

    resource_name = CONFIG_PROFILE_RESOURCES[args.profile]
    text = resources.files("tts_summarizer").joinpath(resource_name).read_text(encoding="utf-8")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    print(config_path)
    return 0
```

- [ ] **Step 4: Add parser and dispatch**

In `build_parser()`, add before `serve`:

```python
    init_config = subcommands.add_parser("init-config")
    init_config.add_argument("--profile", choices=sorted(CONFIG_PROFILE_RESOURCES), default="remote")
    init_config.add_argument("--force", action="store_true")
```

In `main()`, dispatch before loading daemon config:

```python
    if args.command == "init-config":
        return _init_config(args)
```

Keep `argparse` choices as the only unknown-profile handling; do not add a custom validator.

- [ ] **Step 5: Run CLI tests verify pass**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands -v
```

Expected: PASS.

- [ ] **Step 6: Run config tests again**

Run:

```bash
rtk uv run python -m unittest tests.test_config -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
rtk git add src/tts_summarizer/cli.py tests/test_cli_commands.py tests/test_config.py
rtk git commit -m "feat: add init-config command"
```

---

### Task 3: Docker daemon artifacts

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

**Interfaces:**
- Docker image installs this package from the cloned repo.
- Container command is `tts-summarizer serve --config /config/config.toml`.
- Compose mounts `./config.toml` read-only to `/config/config.toml`.
- Compose publishes host port `9200` to container port `9200`.
- No model-specific environment variables.

- [ ] **Step 1: Create Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY hooks ./hooks
COPY config.remote.example.toml config.apple-local.example.toml ./

RUN pip install --no-cache-dir .

EXPOSE 9200

CMD ["tts-summarizer", "serve", "--config", "/config/config.toml"]
```

- [ ] **Step 2: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  tts-summarizer:
    build: .
    command: ["tts-summarizer", "serve", "--config", "/config/config.toml"]
    ports:
      - "9200:9200"
    volumes:
      - ./config.toml:/config/config.toml:ro
```

- [ ] **Step 3: Validate compose config**

Run:

```bash
rtk docker compose config
```

Expected: config renders successfully with one `tts-summarizer` service, one `9200:9200` port mapping, and one read-only config volume.

- [ ] **Step 4: Commit**

Run:

```bash
rtk git add Dockerfile docker-compose.yml
rtk git commit -m "feat: add docker daemon install path"
```

---

### Task 4: README installation UX

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents local CLI install first:
  - `uv tool install git+https://github.com/sagebynature/tts-summarizer`
  - `tts-summarizer init-config --profile remote`
  - `tts-summarizer serve`
- Documents Docker install:
  - `git clone https://github.com/sagebynature/tts-summarizer`
  - `cd tts-summarizer`
  - `cp config.remote.example.toml config.toml`
  - `docker compose up`
- Documents config lookup order and profile choices.

- [ ] **Step 1: Rewrite README order**

Replace README sections after the title with this order:

```markdown
# tts-summarizer

`tts-summarizer` is a small HTTP daemon you bind to any AI application. It turns long AI responses into shorter, speech-friendly text, then returns TTS audio WAV bytes.

## What it is

- Local HTTP service for AI apps that want spoken responses.
- Summarizes long assistant output into text that fits spoken playback.
- Generates TTS audio for client-side playback.
- Uses configurable summarization and TTS models, local or remote.
- Works with local MLX models or OpenAI-compatible remote endpoints.

## Requirements

- Python 3.11+
- `uv`
- Remote OpenAI-compatible summarizer and TTS endpoints, or Apple Silicon Mac for local MLX profiles.
- Docker, only for the Docker quick start.

## Quick start: local CLI install

```bash
uv tool install git+https://github.com/sagebynature/tts-summarizer
tts-summarizer init-config --profile remote
tts-summarizer serve
```

The generated config is `~/.config/tts-summarizer/config.toml`. Use `--force` to replace an existing generated config:

```bash
tts-summarizer init-config --profile remote --force
```

## Quick start: Docker

```bash
git clone https://github.com/sagebynature/tts-summarizer
cd tts-summarizer
cp config.remote.example.toml config.toml
docker compose up
```

Docker runs the HTTP daemon only. The summarizer and TTS profiles in `config.toml` must point at remote model backends reachable from the container.

## Config lookup and profiles

Config lookup order:

1. `--config /path/to/config.toml`
2. `./config.toml`
3. `~/.config/tts-summarizer/config.toml`
4. built-in defaults

Generate a remote-backend config:

```bash
tts-summarizer init-config --profile remote
```

Generate an Apple Silicon local MLX config:

```bash
tts-summarizer init-config --profile apple-local
```

## Apple local MLX notes

Apple local profiles use `mlx_audio` in process and are intended for Apple Silicon Macs. Install the optional local audio dependencies when using local MLX TTS:

```bash
uv tool install 'git+https://github.com/sagebynature/tts-summarizer[mlx]'
```

The Apple local example keeps remote profiles too, so you can switch individual profiles without changing the daemon.

## Run daemon

```bash
tts-summarizer serve --config config.toml
```

FastAPI OpenAPI docs are available while the daemon is running:

- `http://127.0.0.1:9200/docs`
- `http://127.0.0.1:9200/redoc`
- `http://127.0.0.1:9200/openapi.json`

## Send request

`/v1/speak` returns WAV bytes. Playback example:

```bash
curl -sS -X POST http://127.0.0.1:9200/v1/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"This is a long answer to summarize before speaking."}' \
  --output reply.wav
ffplay -nodisp -autoexit reply.wav
```

## Check and stop

```bash
tts-summarizer health --config config.toml
tts-summarizer stop --config config.toml
```

## Development commands

```bash
uv sync --dev
make build # uv build
make test # uv run python -m unittest discover -s tests -v
make typecheck # uvx ty check src tests
make check # typecheck, test, build
make run # uv run python -m tts_summarizer serve --config config.toml
```

Use another config during development:

```bash
make run CONFIG=/path/to/config.toml
```
```

Keep any existing detailed model snippets only if they do not move development commands above user install paths.

- [ ] **Step 2: Check README contains required commands**

Run:

```bash
rtk python - <<'PY'
from pathlib import Path
text = Path('README.md').read_text(encoding='utf-8')
required = [
    'uv tool install git+https://github.com/sagebynature/tts-summarizer',
    'tts-summarizer init-config --profile remote',
    'docker compose up',
    'Docker runs the HTTP daemon only',
    '~/.config/tts-summarizer/config.toml',
    '## Development commands',
]
missing = [item for item in required if item not in text]
assert not missing, missing
assert text.index('## Quick start: local CLI install') < text.index('## Development commands')
PY
```

Expected: command exits `0`.

- [ ] **Step 3: Commit**

Run:

```bash
rtk git add README.md
rtk git commit -m "docs: document user install paths first"
```

---

### Task 5: Final focused verification

**Files:**
- Verify only.

**Interfaces:**
- Confirms CLI, config examples, Docker metadata, package build, and docs checks work together.

- [ ] **Step 1: Run affected unit tests**

Run:

```bash
rtk uv run python -m unittest tests.test_cli_commands tests.test_config -v
```

Expected: PASS.

- [ ] **Step 2: Run typecheck**

Run:

```bash
rtk uv run ty check src tests
```

Expected: PASS.

- [ ] **Step 3: Run package build**

Run:

```bash
rtk uv build
```

Expected: PASS.

- [ ] **Step 4: Validate Docker compose**

Run:

```bash
rtk docker compose config
```

Expected: PASS. If Docker is not installed, record the exact command failure and still complete steps 1-3.

- [ ] **Step 5: Smoke-test installed resource read without installing globally**

Run:

```bash
rtk uv run python - <<'PY'
import importlib.resources as resources
for name in ('config.remote.example.toml', 'config.apple-local.example.toml'):
    text = resources.files('tts_summarizer').joinpath(name).read_text(encoding='utf-8')
    assert '[server]' in text, name
PY
```

Expected: command exits `0`.

- [ ] **Step 6: Commit verification-only fixes if any**

If verification required fixes, commit them:

```bash
rtk git add src tests pyproject.toml README.md Dockerfile docker-compose.yml config.remote.example.toml config.apple-local.example.toml
rtk git commit -m "fix: finish installation ux verification"
```

Skip this commit when no files changed during verification.
