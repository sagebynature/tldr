PYTHON ?= python
CONFIG ?= config.toml
PORT ?= 9200
IMAGE ?= tldr
CONTAINER ?= tldr
DOCKER_CONFIG ?= config.docker.toml


.PHONY: test lint format typecheck check run docker-build docker-run

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

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker rm -f $(CONTAINER) >/dev/null 2>&1 || true
	docker run --name $(CONTAINER) --rm -d -p $(PORT):9200 --add-host=host.docker.internal:host-gateway -v "$(CURDIR)/$(DOCKER_CONFIG):/config/config.toml:ro" $(IMAGE)
