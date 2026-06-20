import unittest

from tts_summarizer.config import Config
from tts_summarizer.request import SpeechRequest
from tts_summarizer.server import TtsService
from tts_summarizer.speech import AudioChunk


class FakeSummarizer:
    def summarize(self, text):
        return f"summary: {text}"


class FakeSpeech:
    def generate(self, text):
        return [AudioChunk(samples=[0.0], sample_rate=8000)]


class FakePlayer:
    def __init__(self):
        self.played = []

    def play(self, chunks, token=None):
        self.played.extend(chunks)


class ServerTests(unittest.TestCase):
    def test_service_speaks_request(self):
        player = FakePlayer()
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=player)
        response = service.handle(SpeechRequest(text="hello", caller="c", session_id="s"))
        self.assertEqual(response["status"], "accepted")
        self.assertEqual(len(player.played), 1)


if __name__ == "__main__":
    unittest.main()
