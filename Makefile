PYTHON ?= python
TY ?= uvx ty
CONFIG ?= config.example.toml

.PHONY: build test typecheck run

build:
	PYTHONPATH=src $(PYTHON) -m compileall -q src

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

typecheck:
	$(TY) check src tests

run:
	PYTHONPATH=src $(PYTHON) -m tts_summarizer serve --config $(CONFIG)
