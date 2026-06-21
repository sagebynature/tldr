import json
import unittest
from unittest.mock import patch

from tts_summarizer.config import SummarizerConfig
from tts_summarizer.summarizer import (
    OpenAICompatibleBackend,
    Summarizer,
    count_words,
    replace_urls,
)


class FakeBackend:
    def __init__(self):
        self.prompt = ""

    def generate(self, messages, config):
        self.prompt = messages[-1]["content"]
        return "short result"


class SummarizerTests(unittest.TestCase):
    def test_count_words(self):
        self.assertEqual(count_words("one two\nthree"), 3)

    def test_replace_urls_replaces_http_and_https(self):
        self.assertEqual(
            replace_urls("Read http://example.test/a https://example.test/b?x=1."),
            "Read supplied URL supplied URL",
        )

    def test_threshold_skips_model(self):
        backend = FakeBackend()
        summarizer = Summarizer(SummarizerConfig(word_threshold=10), backend=backend)
        self.assertEqual(summarizer.summarize("short text"), "short text")
        self.assertEqual(backend.prompt, "")

    def test_prompt_template_used(self):
        backend = FakeBackend()
        config = SummarizerConfig(
            word_threshold=0, user_prompt_template="Limit {max_words}: {text}"
        )
        summarizer = Summarizer(config, backend=backend)
        self.assertEqual(summarizer.summarize("long enough"), "short result")
        self.assertEqual(backend.prompt, "Limit 40: long enough")

    def test_summarizer_sends_sanitized_text_to_backend(self):
        backend = FakeBackend()
        config = SummarizerConfig(
            word_threshold=0,
            user_prompt_template="Say {max_words}: {text}",
        )
        summarizer = Summarizer(config, backend=backend)

        self.assertEqual(
            summarizer.summarize("open https://example.test/path"), "short result"
        )
        self.assertEqual(backend.prompt, "Say 40: open supplied URL")

    def test_system_prompt_echo_falls_back_to_original_text(self):
        class EchoBackend:
            def generate(self, messages, config):
                return config.system_prompt

        summarizer = Summarizer(
            SummarizerConfig(word_threshold=0), backend=EchoBackend()
        )
        self.assertEqual(
            summarizer.summarize("actual request text"), "actual request text"
        )

    def test_system_prompt_prefix_is_stripped_from_summary(self):
        class EchoBackend:
            def generate(self, messages, config):
                return f"{config.system_prompt}\n\nactual summary"

        summarizer = Summarizer(
            SummarizerConfig(word_threshold=0), backend=EchoBackend()
        )
        self.assertEqual(summarizer.summarize("actual request text"), "actual summary")

    def test_thinking_block_is_stripped_from_summary(self):
        class ThinkingBackend:
            def generate(self, messages, config):
                return "<think>private reasoning</think>spoken result"

        summarizer = Summarizer(
            SummarizerConfig(word_threshold=0), backend=ThinkingBackend()
        )
        self.assertEqual(summarizer.summarize("actual request text"), "spoken result")

    def test_thinking_only_output_falls_back_to_original_text(self):
        class ThinkingBackend:
            def generate(self, messages, config):
                return "<think>unfinished reasoning"

        summarizer = Summarizer(
            SummarizerConfig(word_threshold=0), backend=ThinkingBackend()
        )
        self.assertEqual(
            summarizer.summarize("actual request text"), "actual request text"
        )

    def test_openai_backend_posts_chat_completion_without_auth(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"short summary"}}]}'

        def fake_urlopen(request, timeout):
            calls.append((request, timeout))
            return Response()

        backend = OpenAICompatibleBackend(urlopen=fake_urlopen)
        result = backend.generate(
            [{"role": "user", "content": "hello"}],
            SummarizerConfig(
                base_url="http://localhost:1234/v1/",
                api_key="",
                model="local-model",
                temperature=0.3,
                max_tokens=50,
            ),
        )

        self.assertEqual(result, "short summary")
        request, timeout = calls[0]
        self.assertEqual(request.full_url, "http://localhost:1234/v1/chat/completions")
        self.assertEqual(request.get_method(), "POST")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(timeout, 30)
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "model": "local-model",
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.3,
                "max_tokens": 50,
            },
        )

    def test_openai_backend_posts_auth_when_configured(self):
        calls = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"short summary"}}]}'

        def fake_urlopen(request, timeout):
            calls.append(request)
            return Response()

        backend = OpenAICompatibleBackend(urlopen=fake_urlopen)
        backend.generate(
            [{"role": "user", "content": "hello"}],
            SummarizerConfig(api_key="test-token"),
        )

        self.assertEqual(calls[0].headers["Authorization"], "Bearer test-token")

    def test_backend_failure_returns_original_text(self):
        class BrokenBackend:
            def generate(self, messages, config):
                raise RuntimeError("boom")

        summarizer = Summarizer(
            SummarizerConfig(word_threshold=0), backend=BrokenBackend()
        )
        with patch("tts_summarizer.summarizer.logger.exception"):
            self.assertEqual(summarizer.summarize("keep this"), "keep this")


if __name__ == "__main__":
    unittest.main()
