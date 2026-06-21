PYTHON ?= python
CONFIG ?= config.toml

.PHONY: build test lint format typecheck check run

build:
	uv build

test: typecheck
	uv run $(PYTHON) -m unittest discover -s tests -v

lint:
	uv run ruff check src tests --fix

format:
	uv run ruff format src tests

typecheck:
	uv run ty check src tests

check: lint format typecheck test build

run:
	uv run $(PYTHON) -m tts_summarizer serve --config $(CONFIG)
