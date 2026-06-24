from __future__ import annotations
from dataclasses import replace

from typing import Any, Callable, Protocol, cast
from urllib.request import Request, urlopen
import json
import logging
import re
from mdclense.parser import MarkdownParser

from .config import SummarizerConfig, SummarizerProfileConfig


logger = logging.getLogger(__name__)
MARKDOWN_PARSER = MarkdownParser()
URL_PATTERN = re.compile(r"https?://[^\s<>)\]}]+", re.IGNORECASE)
CODE_BLOCK_PATTERN = re.compile(r"(?ms)^```[^\n]*\n.*?^```\s*")


def remove_code_blocks(text: str) -> str:
    return CODE_BLOCK_PATTERN.sub(" ", text)


class SummaryBackend(Protocol):
    def generate(
        self, messages: list[dict[str, str]], config: SummarizerProfileConfig
    ) -> str: ...


def markdown_to_plain_text(text: str) -> str:
    return " ".join(MARKDOWN_PARSER.parse(text).split())


def replace_urls(text: str) -> str:
    return URL_PATTERN.sub("supplied URL", text)


def sanitize_for_summary(text: str) -> str:
    return replace_urls(markdown_to_plain_text(remove_code_blocks(text)))


class OpenAICompatibleBackend:
    def __init__(self, urlopen: Callable[..., object] = urlopen, timeout: float = 30):
        self.urlopen = urlopen
        self.timeout = timeout

    def generate(
        self, messages: list[dict[str, str]], config: SummarizerProfileConfig
    ) -> str:
        token_limits = (config.max_tokens, config.max_tokens * 2)
        content = ""
        for max_tokens in token_limits:
            payload = self._post_completion(messages, config, max_tokens)
            choice = payload["choices"][0]
            content = str(choice["message"]["content"])
            if choice.get("finish_reason") != "length":
                return content
            logger.warning(
                "summary hit max_tokens=%s; retrying with larger limit", max_tokens
            )
        return content

    def _post_completion(
        self,
        messages: list[dict[str, str]],
        config: SummarizerProfileConfig,
        max_tokens: int,
    ):
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": max_tokens,
            **config.extra_body,
        }
        body_bytes = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        request = Request(url, data=body_bytes, headers=headers, method="POST")
        with cast(Any, self.urlopen(request, timeout=self.timeout)) as response:
            return json.loads(response.read().decode("utf-8"))


class Summarizer:
    def __init__(self, config: SummarizerConfig, backend: SummaryBackend | None = None):
        self.config = config
        self.backend = backend or OpenAICompatibleBackend()

    def profile(
        self,
        profile_name: str | None = None,
        overrides: dict[str, object] | None = None,
    ) -> SummarizerProfileConfig:
        name = profile_name or self.config.default_profile
        try:
            profile = self.config.profiles[name]
        except KeyError as exc:
            raise ValueError(f"unknown summarizer profile: {name}") from exc
        return replace(profile, **(overrides or {}))

    def summarize(
        self,
        text: str,
        profile_name: str | None = None,
        overrides: dict[str, object] | None = None,
    ) -> str:
        config = self.profile(profile_name, overrides)
        if not config.enabled:
            return text
        if count_words(text) <= config.word_threshold:
            return text
        logger.info(
            "summarizing text chars=%s words=%s profile=%s",
            len(text),
            count_words(text),
            profile_name or self.config.default_profile,
        )
        sanitized = sanitize_for_summary(text)
        messages = [
            {"role": "system", "content": config.system_prompt},
            {
                "role": "user",
                "content": config.user_prompt_template.format(
                    max_words=config.max_words,
                    text=sanitized,
                ),
            },
        ]
        try:
            summary = self.backend.generate(messages, config).strip()
        except Exception:
            logger.exception("summary failed; using original text")
            return text
        cleaned = clean_summary(summary, text, config)
        logger.info("summary output preview=%r", preview(cleaned))
        return cleaned


def clean_summary(summary: str, original: str, config: SummarizerProfileConfig) -> str:
    cleaned = strip_thinking(summary).strip()
    sanitized = sanitize_for_summary(original)
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
