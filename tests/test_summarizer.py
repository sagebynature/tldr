import unittest

from tts_summarizer.config import SummarizerConfig
from tts_summarizer.summarizer import Summarizer, count_words


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

    def test_backend_failure_returns_original_text(self):
        class BrokenBackend:
            def generate(self, messages, config):
                raise RuntimeError("boom")

        summarizer = Summarizer(SummarizerConfig(word_threshold=0), backend=BrokenBackend())
        self.assertEqual(summarizer.summarize("keep this"), "keep this")


if __name__ == "__main__":
    unittest.main()
