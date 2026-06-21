import unittest
from unittest.mock import patch

from tts_summarizer.config import SummarizerConfig
from tts_summarizer.summarizer import MlxLmBackend, Summarizer, count_words


class FakeBackend:
    def __init__(self):
        self.prompt = ""

    def generate(self, messages, config):
        self.prompt = messages[-1]["content"]
        return "short result"


class SummarizerTests(unittest.TestCase):
    def test_count_words(self):
        self.assertEqual(count_words("one two\nthree"), 3)

    def test_threshold_skips_model(self):
        backend = FakeBackend()
        summarizer = Summarizer(SummarizerConfig(word_threshold=10), backend=backend)
        self.assertEqual(summarizer.summarize("short text"), "short text")
        self.assertEqual(backend.prompt, "")

    def test_prompt_template_used(self):
        backend = FakeBackend()
        config = SummarizerConfig(word_threshold=0, user_prompt_template="Limit {max_words}: {text}")
        summarizer = Summarizer(config, backend=backend)
        self.assertEqual(summarizer.summarize("long enough"), "short result")
        self.assertEqual(backend.prompt, "Limit 40: long enough")

    def test_system_prompt_echo_falls_back_to_original_text(self):
        class EchoBackend:
            def generate(self, messages, config):
                return config.system_prompt

        summarizer = Summarizer(SummarizerConfig(word_threshold=0), backend=EchoBackend())
        self.assertEqual(summarizer.summarize("actual request text"), "actual request text")

    def test_system_prompt_prefix_is_stripped_from_summary(self):
        class EchoBackend:
            def generate(self, messages, config):
                return f"{config.system_prompt}\n\nshort useful summary"

        summarizer = Summarizer(SummarizerConfig(word_threshold=0), backend=EchoBackend())
        self.assertEqual(summarizer.summarize("actual request text"), "short useful summary")

    def test_thinking_block_is_stripped_from_summary(self):
        class ThinkingBackend:
            def generate(self, messages, config):
                return "<think>\nI should summarize the request.\n</think>\nFinal summary only."

        summarizer = Summarizer(SummarizerConfig(word_threshold=0), backend=ThinkingBackend())
        self.assertEqual(summarizer.summarize("actual request text"), "Final summary only.")

    def test_thinking_only_output_falls_back_to_original_text(self):
        class ThinkingBackend:
            def generate(self, messages, config):
                return "<think>\nNo final answer yet.\n</think>"

        summarizer = Summarizer(SummarizerConfig(word_threshold=0), backend=ThinkingBackend())
        self.assertEqual(summarizer.summarize("actual request text"), "actual request text")


    def test_mlx_backend_uses_sampler_not_temperature_kwarg(self):
        calls = {}

        class Tokenizer:
            def __init__(self):
                self.kwargs = {}

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "prompt"

        def fake_generate(*args, **kwargs):
            calls.update(kwargs)
            if "temperature" in kwargs:
                raise TypeError("generate_step() got an unexpected keyword argument 'temperature'")
            return "summary"

        def fake_make_sampler(**kwargs):
            return ("sampler", kwargs)

        tokenizer = Tokenizer()
        backend = MlxLmBackend()
        backend._model = object()
        backend._tokenizer = tokenizer
        backend._model_name = "fake"
        backend._generate = fake_generate
        backend._make_sampler = fake_make_sampler

        result = backend.generate([{"role": "user", "content": "hello"}], SummarizerConfig(model="fake", temperature=0.2))

        self.assertEqual(result, "summary")
        self.assertNotIn("temperature", calls)
        self.assertEqual(calls["sampler"], ("sampler", {"temp": 0.2}))
        self.assertIs(tokenizer.kwargs["enable_thinking"], False)

    def test_backend_failure_returns_original_text(self):
        class BrokenBackend:
            def generate(self, messages, config):
                raise RuntimeError("boom")

        summarizer = Summarizer(SummarizerConfig(word_threshold=0), backend=BrokenBackend())
        with patch("tts_summarizer.summarizer.logger.exception"):
            self.assertEqual(summarizer.summarize("keep this"), "keep this")


if __name__ == "__main__":
    unittest.main()
