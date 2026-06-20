from __future__ import annotations

from typing import Protocol
import sys

from .config import SummarizerConfig


class SummaryBackend(Protocol):
    def generate(self, messages: list[dict[str, str]], config: SummarizerConfig) -> str: ...


class MlxLmBackend:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._model_name = ""
        self._generate = None

    def _load(self, model_name: str):
        if self._model is not None and self._model_name == model_name:
            return self._model, self._tokenizer
        from mlx_lm import generate, load  # type: ignore

        model, tokenizer = load(model_name)
        self._generate = generate
        self._model = model
        self._tokenizer = tokenizer
        self._model_name = model_name
        return model, tokenizer

    def generate(self, messages: list[dict[str, str]], config: SummarizerConfig) -> str:
        model, tokenizer = self._load(config.model)
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self._generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
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
        except Exception as exc:
            print(f"tts-summarizer summary failed: {exc}", file=sys.stderr)
            return text
        return summary or text


def count_words(text: str) -> int:
    return len(text.split())
