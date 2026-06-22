import unittest
import unittest.mock
from urllib.error import URLError

from fastapi.testclient import TestClient

from tts_summarizer.config import Config, SummarizerConfig, SummarizerProfileConfig
from tts_summarizer.server import create_app, run_server
from tts_summarizer.speech import AudioBytes, AudioChunk
from tts_summarizer.summarizer import Summarizer


class FakeSummarizer:
    def __init__(self):
        self.profile_name = None

    def summarize(self, text, profile_name=None, overrides=None):
        self.profile_name = profile_name
        return f"summary: {text}"


class CapturingSpeech:
    def __init__(self):
        self.text = ""
        self.profile_name = None

    def generate(self, text, profile_name=None):
        self.text = text
        self.profile_name = profile_name
        return [AudioChunk(samples=[0.0], sample_rate=8000)]


class RemoteSpeech:
    def sample_rate(self, profile_name=None):
        return 24000

    def generate(self, text, profile_name=None):
        return AudioBytes([b"RIFFremote-wav"])


class UnavailableRemoteSpeech:
    def sample_rate(self, profile_name=None):
        return 24000

    def generate(self, text, profile_name=None):
        raise URLError(ConnectionRefusedError(61, "Connection refused"))


class ServerTests(unittest.TestCase):
    def test_run_server_starts_uvicorn_without_worker_thread(self):
        events = []

        class FakeSocket:
            def setsockopt(self, *args):
                events.append(("setsockopt", args))

            def bind(self, address):
                events.append(("bind", address))

            def listen(self):
                events.append("listen")

            def getsockname(self):
                return ("127.0.0.1", 0)

        class FakeState:
            pass

        class FakeApp:
            def __init__(self):
                self.state = FakeState()

        class FakeServer:
            def __init__(self, config):
                self.config = config

            def run(self, sockets):
                events.append(("server", sockets))

        with (
            unittest.mock.patch(
                "tts_summarizer.server.create_app", return_value=FakeApp()
            ),
            unittest.mock.patch(
                "tts_summarizer.server.socket.socket", return_value=FakeSocket()
            ),
            unittest.mock.patch("tts_summarizer.server.write_state"),
            unittest.mock.patch("tts_summarizer.server.uvicorn.Server", FakeServer),
        ):
            self.assertEqual(run_server(Config()), 0)

        self.assertIn("listen", events)
        self.assertEqual(events[-1][0], "server")

    def test_fastapi_openapi_schema_exists(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/speak", response.json()["paths"])

    def test_fastapi_openapi_schema_includes_summarize(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/v1/summarize", response.json()["paths"])

    def test_fastapi_openapi_schema_has_concrete_request_bodies(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        schema = client.get("/openapi.json").json()
        components = schema["components"]["schemas"]
        self.assertIn("SpeakRequestBody", components)
        self.assertIn("SummarizeRequestBody", components)
        self.assertIn("SummarizeResponseBody", components)
        properties = components["SpeakRequestBody"]["properties"]
        self.assertIn("text", properties)
        self.assertIn("metadata", properties)
        self.assertIn("summarize", properties)
        self.assertNotIn("caller", properties)
        self.assertNotIn("session_id", properties)
        self.assertNotIn("event", properties)
        self.assertNotIn("playback", properties)
        self.assertIn("max_words", components["SummarizeRequestBody"]["properties"])
        self.assertNotIn("model", components["SummarizeRequestBody"]["properties"])
        self.assertIn("summary", components["SummarizeResponseBody"]["properties"])

    def test_fastapi_health_route(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_fastapi_speak_route_returns_wav_and_uses_identity_headers(self):
        speech = CapturingSpeech()
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=speech)
        )

        with self.assertLogs("tts_summarizer.server", level="INFO") as logs:
            response = client.post(
                "/v1/speak",
                json={"text": "hello"},
                headers={
                    "X-TTS-Caller": "header",
                    "X-TTS-Session-Id": "header-session",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertTrue(response.content.startswith(b"RIFF"))
        self.assertEqual(speech.text, "summary: hello")
        self.assertIn("session=header:header-session", "\n".join(logs.output))

    def test_fastapi_speak_route_streams_without_content_length(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.post("/v1/speak", json={"text": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("content-length", response.headers)

    def test_fastapi_speak_route_passes_remote_wav_bytes_through(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=RemoteSpeech())
        )

        response = client.post("/v1/speak", json={"text": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, b"RIFFremote-wav")

    def test_fastapi_speak_route_reports_remote_tts_unavailable(self):
        client = TestClient(
            create_app(
                Config(), summarizer=FakeSummarizer(), speech=UnavailableRemoteSpeech()
            ),
            raise_server_exceptions=False,
        )

        response = client.post("/v1/speak", json={"text": "hello"})

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"error": "remote TTS unavailable"})

    def test_fastapi_speak_route_summarizes_by_default(self):
        speech = CapturingSpeech()
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=speech)
        )

        response = client.post("/v1/speak", json={"text": "hello"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(speech.text, "summary: hello")

    def test_fastapi_speak_route_passes_summarizer_profile(self):
        summarizer = FakeSummarizer()
        client = TestClient(
            create_app(Config(), summarizer=summarizer, speech=CapturingSpeech())
        )

        response = client.post(
            "/v1/speak", json={"text": "hello", "summarizer_profile": "fast"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(summarizer.profile_name, "fast")

    def test_fastapi_speak_route_can_skip_summarizer_for_tts_testing(self):
        speech = CapturingSpeech()
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=speech)
        )

        response = client.post("/v1/speak", json={"text": "hello", "summarize": False})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(speech.text, "hello")

    def test_fastapi_speak_route_passes_tts_profile(self):
        speech = CapturingSpeech()
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=speech)
        )

        response = client.post(
            "/v1/speak",
            json={"text": "hello", "summarize": False, "tts_profile": "kokoro"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(speech.profile_name, "kokoro")

    def test_fastapi_speak_route_rejects_payload_identity_and_playback(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        for key in ("caller", "session_id", "event", "playback"):
            with self.subTest(key=key):
                response = client.post("/v1/speak", json={"text": "hello", key: "bad"})
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json())

    def test_fastapi_speak_route_rejects_invalid_json(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.post("/v1/speak", json={"text": ""})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_fastapi_speak_route_rejects_malformed_json_with_error_contract(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.post(
            "/v1/speak",
            content='{"text":',
            headers={"content-type": "application/json"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(set(response.json()), {"error"})

    def test_fastapi_speak_route_rejects_non_object_json_with_error_contract(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.post("/v1/speak", json=["hello"])

        self.assertEqual(response.status_code, 400)
        self.assertEqual(set(response.json()), {"error"})

    def test_fastapi_summarize_route_applies_safe_overrides(self):
        class CapturingBackend:
            def __init__(self):
                self.messages = []
                self.config = None

            def generate(self, messages, config):
                self.messages = messages
                self.config = config
                return "short summary"

        backend = CapturingBackend()
        client = TestClient(
            create_app(
                Config(),
                summarizer=Summarizer(Config().summarizer, backend=backend),
                speech=CapturingSpeech(),
            )
        )

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
        default_summary = Config().summarizer.profiles[
            Config().summarizer.default_profile
        ]
        self.assertEqual(backend.config.model, default_summary.model)
        self.assertIn("supplied URL", backend.messages[-1]["content"])

    def test_fastapi_summarize_route_selects_summarizer_profile(self):
        class CapturingBackend:
            def __init__(self):
                self.config = None

            def generate(self, messages, config):
                self.config = config
                return "profile summary"

        backend = CapturingBackend()
        config = Config(
            summarizer=SummarizerConfig(
                default_profile="default",
                profiles={
                    "default": SummarizerProfileConfig(model="default-model"),
                    "fast": SummarizerProfileConfig(model="fast-model", max_words=20),
                },
            )
        )
        client = TestClient(
            create_app(
                config,
                summarizer=Summarizer(config.summarizer, backend=backend),
                speech=CapturingSpeech(),
            )
        )

        response = client.post(
            "/v1/summarize",
            json={"text": "hello world", "summarizer_profile": "fast", "max_words": 12},
        )

        self.assertEqual(response.status_code, 200)
        assert backend.config is not None
        self.assertEqual(backend.config.model, "fast-model")
        self.assertEqual(backend.config.max_words, 12)

    def test_fastapi_summarize_route_rejects_disallowed_override(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.post("/v1/summarize", json={"text": "hello", "model": "nope"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_fastapi_summarize_route_requires_text(self):
        client = TestClient(
            create_app(Config(), summarizer=FakeSummarizer(), speech=CapturingSpeech())
        )

        response = client.post("/v1/summarize", json={"max_tokens": 10})

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())


if __name__ == "__main__":
    unittest.main()
