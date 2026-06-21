import threading
import time
import unittest
import unittest.mock
from fastapi.testclient import TestClient

from tts_summarizer.config import Config
from tts_summarizer.request import SpeechRequest
from tts_summarizer.server import TtsService, create_app
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
        self.done = threading.Event()

    def play(self, chunks, token=None):
        self.played.extend(chunks)
        self.done.set()


class ServerTests(unittest.TestCase):
    def test_service_speaks_request(self):
        player = FakePlayer()
        service = TtsService(
            Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=player
        )
        response = service.handle(
            SpeechRequest(text="hello", caller="c", session_id="s")
        )
        self.assertEqual(response["status"], "accepted")
        self.assertTrue(service.process_pending())
        self.assertTrue(player.done.wait(1))
        self.assertEqual(len(player.played), 1)

    def test_service_passes_summary_to_tts(self):
        class CapturingSpeech:
            def __init__(self):
                self.text = ""

            def generate(self, text):
                self.text = text
                return [AudioChunk(samples=[0.0], sample_rate=8000)]

        player = FakePlayer()
        speech = CapturingSpeech()
        service = TtsService(
            Config(), summarizer=FakeSummarizer(), speech=speech, player=player
        )
        response = service.handle(
            SpeechRequest(text="hello", caller="c", session_id="s")
        )
        self.assertEqual(response["status"], "accepted")
        self.assertTrue(service.process_pending())
        self.assertEqual(speech.text, "summary: hello")

    def test_service_logs_incoming_and_summarized_text(self):
        player = FakePlayer()
        service = TtsService(
            Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=player
        )
        with self.assertLogs("tts_summarizer.server", level="INFO") as logs:
            response = service.handle(
                SpeechRequest(text="hello", caller="c", session_id="s")
            )
            self.assertEqual(response["status"], "accepted")
            self.assertTrue(service.process_pending())
        output = "\n".join(logs.output)
        self.assertIn("incoming text session=c:s text='hello'", output)
        self.assertIn("summarized text session=c:s text='summary: hello'", output)

    def test_service_returns_before_slow_tts_finishes(self):
        class SlowSpeech:
            def generate(self, text):
                time.sleep(0.2)
                return [AudioChunk(samples=[0.0], sample_rate=8000)]

        player = FakePlayer()
        service = TtsService(
            Config(), summarizer=FakeSummarizer(), speech=SlowSpeech(), player=player
        )
        started = time.monotonic()
        response = service.handle(
            SpeechRequest(text="hello", caller="c", session_id="s")
        )
        elapsed = time.monotonic() - started
        self.assertEqual(response["status"], "accepted")
        self.assertLess(elapsed, 0.1)
        self.assertFalse(player.done.is_set())
        self.assertTrue(service.process_pending())
        self.assertTrue(player.done.wait(1))

    def test_service_handle_does_not_start_worker_thread(self):
        with unittest.mock.patch("tts_summarizer.server.threading.Thread") as thread:
            service = TtsService(
                Config(),
                summarizer=FakeSummarizer(),
                speech=FakeSpeech(),
                player=FakePlayer(),
            )
            response = service.handle(
                SpeechRequest(text="hello", caller="c", session_id="s")
            )
        self.assertEqual(response["status"], "accepted")
        thread.assert_not_called()

    def test_fastapi_openapi_schema_exists(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/speak", response.json()["paths"])

    def test_fastapi_health_route(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_fastapi_speak_route_accepts_current_json(self):
        player = FakePlayer()
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=player)
        client = TestClient(create_app(Config(), service=service))

        response = client.post(
            "/v1/speak",
            json={"text": "hello", "caller": "c", "session_id": "s"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "accepted")
        self.assertTrue(service.process_pending())
        self.assertTrue(player.done.wait(1))

    def test_fastapi_speak_route_rejects_invalid_json(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.post("/v1/speak", json={"text": ""})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


if __name__ == "__main__":
    unittest.main()
