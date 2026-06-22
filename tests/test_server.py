import threading
import socket
import time
import unittest
import unittest.mock
from fastapi.testclient import TestClient

from tts_summarizer.config import Config
from tts_summarizer.request import SpeechRequest
from tts_summarizer.server import TtsService, create_app, run_server
from tts_summarizer.speech import AudioChunk
from tts_summarizer.summarizer import Summarizer


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

    def test_same_session_request_cancels_active_playback_token(self):
        cancelled = []

        class BlockingPlayer:
            def play(self, chunks, token=None):
                service.handle(SpeechRequest(text="new", caller="c", session_id="s"))
                assert token is not None
                cancelled.append(token.cancelled())

        service = TtsService(
            Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=BlockingPlayer()
        )
        service.handle(SpeechRequest(text="old", caller="c", session_id="s"))

        self.assertTrue(service.process_pending())
        self.assertEqual(cancelled, [True])

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


    def test_run_server_joins_non_daemon_worker_after_uvicorn_stops(self):
        events = []

        class FakeSocket:
            def setsockopt(self, level, optname, value):
                events.append(("setsockopt", level, optname, value))

            def bind(self, address):
                events.append(("bind", address))

            def listen(self):
                events.append("listen")

            def getsockname(self):
                return ("127.0.0.1", 0)

        class FakeService:
            def __init__(self, config):
                self.config = config

            def run(self):
                events.append("run")

            def stop(self):
                events.append("stop")

        class FakeServer:
            def __init__(self, config):
                self.config = config

            def run(self, sockets):
                events.append(("server", sockets))

        class FakeThread:
            def __init__(self, target=None, daemon=None, **_kwargs):
                self.target = target
                self.daemon = daemon

            def start(self):
                events.append(("start", self.daemon))

            def join(self):
                events.append("join")

        with (
            unittest.mock.patch("tts_summarizer.server.TtsService", FakeService),
            unittest.mock.patch("tts_summarizer.server.socket.socket", return_value=FakeSocket()),
            unittest.mock.patch("tts_summarizer.server.write_state"),
            unittest.mock.patch("tts_summarizer.server.uvicorn.Server", FakeServer),
            unittest.mock.patch(
                "tts_summarizer.server.threading.Thread",
                side_effect=lambda *args, **kwargs: FakeThread(*args, **kwargs),
            ) as thread,
        ):
            self.assertEqual(run_server(Config()), 0)

        self.assertIs(thread.call_args.kwargs["daemon"], False)
        self.assertIn(("setsockopt", socket.SOL_SOCKET, socket.SO_REUSEADDR, 1), events)
        self.assertEqual(events[-2:], ["stop", "join"])
    def test_fastapi_openapi_schema_exists(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/speak", response.json()["paths"])


    def test_fastapi_openapi_schema_includes_summarize(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/summarize", response.json()["paths"])

    def test_fastapi_openapi_schema_has_concrete_request_bodies(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        schema = client.get("/openapi.json").json()
        components = schema["components"]["schemas"]
        self.assertIn("SpeakRequestBody", components)
        self.assertIn("SummarizeRequestBody", components)
        self.assertIn("SummarizeResponseBody", components)
        self.assertIn("text", components["SpeakRequestBody"]["properties"])
        self.assertIn("session_id", components["SpeakRequestBody"]["properties"])
        self.assertIn("max_words", components["SummarizeRequestBody"]["properties"])
        self.assertNotIn("model", components["SummarizeRequestBody"]["properties"])
        self.assertIn("summary", components["SummarizeResponseBody"]["properties"])

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

    def test_fastapi_speak_route_rejects_malformed_json_with_error_contract(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.post(
            "/v1/speak",
            content='{"text":',
            headers={"content-type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(set(response.json()), {"error"})

    def test_fastapi_speak_route_rejects_non_object_json_with_error_contract(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.post("/v1/speak", json=["hello"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(set(response.json()), {"error"})


    def test_fastapi_summarize_route_uses_allowed_overrides(self):
        class CapturingBackend:
            def __init__(self):
                self.config = None
                self.messages = []

            def generate(self, messages, config):
                self.messages = messages
                self.config = config
                return "short summary"

        backend = CapturingBackend()
        service = TtsService(
            Config(),
            summarizer=Summarizer(Config().summarizer, backend=backend),
            speech=FakeSpeech(),
            player=FakePlayer(),
        )
        client = TestClient(create_app(Config(), service=service))

        response = client.post(
            "/v1/summarize",
            json={
                "text": "Read HTTPS://example.test/private.",
                "word_threshold": 0,
                "max_words": 12,
                "temperature": 0.7,
                "max_tokens": 33,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"summary": "short summary", "changed": True})
        assert backend.config is not None
        self.assertEqual(backend.config.word_threshold, 0)
        self.assertEqual(backend.config.max_words, 12)
        self.assertEqual(backend.config.temperature, 0.7)
        self.assertEqual(backend.config.max_tokens, 33)
        self.assertEqual(backend.config.model, Config().summarizer.model)
        self.assertIn("supplied URL", backend.messages[-1]["content"])

    def test_fastapi_summarize_route_rejects_disallowed_override(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.post("/v1/summarize", json={"text": "hello", "model": "nope"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_fastapi_summarize_route_requires_text(self):
        service = TtsService(Config(), summarizer=FakeSummarizer(), speech=FakeSpeech(), player=FakePlayer())
        client = TestClient(create_app(Config(), service=service))

        response = client.post("/v1/summarize", json={"max_tokens": 10})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

if __name__ == "__main__":
    unittest.main()
