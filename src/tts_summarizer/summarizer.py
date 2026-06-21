from __future__ import annotations

from typing import Callable, Protocol
from urllib.request import Request, urlopen
import json
import logging
import re

from .config import SummarizerConfig


logger = logging.getLogger(__name__)
URL_PATTERN = re.compile(r"https?://[^\s<>)\]}]+")


class SummaryBackend(Protocol):
    def generate(
        self, messages: list[dict[str, str]], config: SummarizerConfig
    ) -> str: ...


def replace_urls(text: str) -> str:
    return URL_PATTERN.sub("supplied URL", text)


class OpenAICompatibleBackend:
    def __init__(self, urlopen: Callable[..., object] = urlopen, timeout: float = 30):
        self.urlopen = urlopen
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], config: SummarizerConfig) -> str:
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(
            {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = Request(url, data=body, headers=headers, method="POST")
        with self.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])


class Summarizer:
    def __init__(self, config: SummarizerConfig, backend: SummaryBackend | None = None):
        self.config = config
        self.backend = backend or OpenAICompatibleBackend()

    def summarize(self, text: str) -> str:
        if not self.config.enabled:
            return text
        if count_words(text) <= self.config.word_threshold:
            return text
        logger.info(
            "summarizing text chars=%s words=%s prompt=%s",
            len(text),
            count_words(text),
            self.config.system_prompt,
        )
        sanitized = replace_urls(text)
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {
                "role": "user",
                "content": self.config.user_prompt_template.format(
                    max_words=self.config.max_words,
                    text=sanitized,
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
    sanitized = replace_urls(original)
    for prompt_part in (
        config.system_prompt,
        config.user_prompt_template.format(max_words=config.max_words, text=original),
        config.user_prompt_template.format(max_words=config.max_words, text=sanitized),
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
