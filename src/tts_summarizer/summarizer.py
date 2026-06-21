from __future__ import annotations

from typing import Any, Callable, Protocol, cast
import logging
import re

from .config import SummarizerConfig


logger = logging.getLogger(__name__)


class SummaryBackend(Protocol):
    def generate(
        self, messages: list[dict[str, str]], config: SummarizerConfig
    ) -> str: ...


class MlxLmBackend:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_name = ""
        self._generate: Callable[..., str] | None = None
        self._make_sampler: Callable[..., object] | None = None

    def _load(self, model_name: str):
        if self._model is not None and self._model_name == model_name:
            return self._model, self._tokenizer
        logger.info("loading summarizer model=%s", model_name)
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler

        loaded = cast(Any, load)(model_name)
        model = loaded[0]
        tokenizer = loaded[1]
        self._generate = generate
        self._make_sampler = make_sampler
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        logger.info("summarizer model ready model=%s", model_name)
        return model, tokenizer

    def generate(self, messages: list[dict[str, str]], config: SummarizerConfig) -> str:
        model, tokenizer = self._load(config.model)
        logger.info("generating summary model=%s", config.model)
        try:
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        generate = self._generate
        make_sampler = self._make_sampler
        if generate is None:
            raise RuntimeError("mlx-lm generator was not loaded")
        if make_sampler is None:
            raise RuntimeError("mlx-lm sampler factory was not loaded")
        return generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            sampler=make_sampler(temp=config.temperature),
            verbose=False,
        ).strip()


class Summarizer:
    def __init__(self, config: SummarizerConfig, backend: SummaryBackend | None = None):
        self.config = config
        self.backend = backend or MlxLmBackend()

    def summarize(self, text: str) -> str:
        if not self.config.enabled:
            return text
        if count_words(text) <= self.config.word_threshold:
            return text
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": self.config.user_prompt_template.format(
                    max_words=self.config.max_words,
                    text=text,
                ),
            },
        ]
        try:
            summary = self.backend.generate(messages, self.config).strip()
        except Exception:
            logger.exception("summary failed; using original text")
            return text
        cleaned = clean_summary(summary, text, self.config)
        logger.info("summary output preview=%r", preview(cleaned))
        return cleaned


def clean_summary(summary: str, original: str, config: SummarizerConfig) -> str:
    cleaned = strip_thinking(summary).strip()
    for prompt_part in (
        config.system_prompt,
        config.user_prompt_template.format(max_words=config.max_words, text=original),
    ):
        prompt_part = prompt_part.strip()
        if cleaned == prompt_part:
            return original
        if cleaned.startswith(prompt_part):
            cleaned = cleaned[len(prompt_part) :].strip()
    return cleaned or original


def strip_thinking(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped.startswith("<think>"):
        return ""
    return stripped


def preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}..."


def count_words(text: str) -> int:
    return len(text.split())
