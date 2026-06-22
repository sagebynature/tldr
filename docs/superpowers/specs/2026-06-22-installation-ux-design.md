# Installation UX Design

## Goal

Make `tts-summarizer` installable by users who are not developing the repo, with one clear mental model: install the daemon, then choose where model backends run.

## Approved Scope

- Add a persistent local install path using `uv tool install`.
- Add `tts-summarizer init-config [--profile remote|apple-local] [--force]`.
- Generate user config at `~/.config/tts-summarizer/config.toml`.
- Make `remote` the default generated profile because it works outside Apple Silicon and inside Docker.
- Add Docker support for running the daemon from a cloned repo with remote model backends only.
- Update README so user install paths come before development commands.

## Out Scope

- Publishing to PyPI.
- Installing or managing model servers.
- Bundling local MLX models into Docker.
- Adding provider discovery, endpoint health probing, or interactive setup prompts.
- Adding a new dependency for config generation.

## Current Architecture

Configuration lookup already supports the needed user path:

1. `--config /path/to/config.toml`
2. `./config.toml`
3. `~/.config/tts-summarizer/config.toml`
4. built-in defaults

The CLI currently supports `config-check`, `serve`, `health`, and `stop`. The repo has one development `config.toml` with both local and remote profiles. README currently documents local development first.

## User Flows

### Local CLI Install

```bash
uv tool install git+https://github.com/sagebynature/tts-summarizer
tts-summarizer init-config --profile remote
tts-summarizer serve
```

`uvx` remains useful for one-off execution, but README should not present it as persistent install. `uv tool install` is the install command.

### Docker Install

```bash
git clone https://github.com/sagebynature/tts-summarizer
cd tts-summarizer
cp config.remote.example.toml config.toml
docker compose up
```

Docker runs the HTTP daemon only. Remote summarizer and remote TTS endpoints must already exist and be reachable from the container.

### Apple Local Optional Path

```bash
tts-summarizer init-config --profile apple-local
tts-summarizer serve
```

This keeps the current MLX-friendly path available without making it the default for every installer.

## CLI Design

Add subcommand:

```bash
tts-summarizer init-config [--profile remote|apple-local] [--force]
```

Behavior:

- Default `--profile remote`.
- Create `~/.config/tts-summarizer` if missing.
- Write `config.toml` there.
- Refuse to overwrite an existing file unless `--force` is present.
- Print the written path on success.
- Return non-zero when refusing overwrite.

Keep this non-interactive. Flags are enough, and non-interactive setup works in scripts and docs.

## Config Artifacts

Add two example config files:

- `config.remote.example.toml`: remote summarizer and remote TTS profiles only.
- `config.apple-local.example.toml`: Apple Silicon local MLX defaults plus any useful remote examples already supported.

`init-config` can copy the matching package resource or write the matching literal text from the module. Prefer copying package data if examples are already shipped; otherwise keep the implementation small and obvious.

## Docker Artifacts

Add:

- `Dockerfile`
- `docker-compose.yml`

Docker behavior:

- Install the package into the image.
- Run `tts-summarizer serve --config /config/config.toml`.
- Mount local `./config.toml` read-only into `/config/config.toml`.
- Expose `9200`.
- Set no model-specific environment variables.

## README Changes

Order:

1. What it is.
2. Requirements.
3. Quick start: local CLI install.
4. Quick start: Docker.
5. Config lookup and profiles.
6. Apple local MLX notes.
7. Development commands.

The Docker section must explicitly say remote model backends are required.

## Error Handling

- Missing parent directory: create it.
- Existing config without `--force`: print a clear error and path.
- Unknown `--profile`: let `argparse` reject it.
- File write failure: surface the OS error through CLI stderr.

## Testing

Add focused CLI tests for `init-config`:

- writes remote config to a temporary home directory
- refuses overwrite without `--force`
- overwrites with `--force`
- writes apple-local config when selected

Update config tests only if example files introduce new parsing coverage.

## Non-Goals Kept Deliberately Lazy

- No interactive wizard. Flags are easier to test and script.
- No Docker model server orchestration. Users already have different remote backends.
- No config migration command. There is no installed config history yet.
