PYTHON ?= python
CONFIG ?= config.toml

.PHONY: test lint format typecheck check run

test: typecheck
	uv run $(PYTHON) -m unittest discover -s tests -v

lint:
	uv run ruff check src tests --fix

format:
	uv run ruff format src tests

typecheck:
	uv run ty check src tests

check: lint format typecheck test

run:
	uv run $(PYTHON) -m tldr serve --config $(CONFIG)
