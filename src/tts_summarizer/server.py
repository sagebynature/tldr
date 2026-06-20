from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading

from .audio import AudioPlayer
from .config import Config
from .request import RequestError, SpeechRequest
from .session import SessionManager
from .speech import SpeechGenerator
from .state import write_state
from .summarizer import Summarizer


class TtsService:
    def __init__(self, config: Config, summarizer=None, speech=None, player=None):
        self.config = config
        self.sessions = SessionManager(config.session)
        self.summarizer = summarizer or Summarizer(config.summarizer)
        self.speech = speech or SpeechGenerator(config.tts)
        self.player = player or AudioPlayer(config.audio)
        self._audio_lock = threading.Lock()

    def handle(self, request: SpeechRequest) -> dict[str, object]:
        token = self.sessions.begin(request)
        try:
            text = self.summarizer.summarize(request.text)
            if token.cancelled():
                return {"status": "cancelled", "session_key": request.session_key()}
            chunks = self.speech.generate(text)
            if self.config.session.cross_session_policy == "queue":
                with self._audio_lock:
                    self.player.play(chunks, token=token)
            else:
                self.player.play(chunks, token=token)
            return {"status": "accepted", "session_key": request.session_key()}
        finally:
            self.sessions.finish(token)

    def health(self) -> dict[str, object]:
        return {"status": "ok", "pid": os.getpid()}


class Handler(BaseHTTPRequestHandler):
    service: TtsService

    def do_GET(self):
        if self.path == "/health":
            self._send(200, self.service.health())
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/v1/speak":
            self._speak()
            return
        if self.path == "/shutdown":
            self._send(200, {"status": "shutting_down"})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self._send(404, {"error": "not found"})

    def log_message(self, format, *args):
        return

    def _speak(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request = SpeechRequest.from_json(payload)
        except (json.JSONDecodeError, RequestError) as exc:
            self._send(400, {"error": str(exc)})
            return
        response = self.service.handle(request)
        self._send(200, response)

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(config: Config) -> int:
    service = TtsService(config)
    handler = type("ConfiguredHandler", (Handler,), {"service": service})
    httpd = ThreadingHTTPServer((config.server.host, config.server.port), handler)
    address = httpd.server_address
    host = str(address[0])
    port = int(address[1])
    write_state(config, host, port, os.getpid())
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return 0
