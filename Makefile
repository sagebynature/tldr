PYTHON ?= python
CONFIG ?= config.toml

.PHONY: build test typecheck check run

build:
	uv build

test: typecheck
	uv run $(PYTHON) -m unittest discover -s tests -v

typecheck:
	uvx ty check src tests

check: typecheck test build

run:
	uv run $(PYTHON) -m tts_summarizer serve --config $(CONFIG)
