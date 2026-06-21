from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
import json
import logging
import os
import threading

from .audio import AudioPlayer
from .config import Config
from .request import RequestError, SpeechRequest
from .session import SessionManager, WorkToken
from .speech import SpeechGenerator
from .state import write_state
from .summarizer import Summarizer


logger = logging.getLogger(__name__)
Job = tuple[SpeechRequest, WorkToken] | None


class TtsService:
    def __init__(self, config: Config, summarizer=None, speech=None, player=None):
        self.config = config
        self.sessions = SessionManager(config.session)
        self.summarizer = summarizer or Summarizer(config.summarizer)
        self.speech = speech or SpeechGenerator(config.tts)
        self.player = player or AudioPlayer(config.audio)
        self._audio_lock = threading.Lock()
        self._jobs: Queue[Job] = Queue()

    def handle(self, request: SpeechRequest) -> dict[str, object]:
        token = self.sessions.begin(request)
        logger.info("accepted speech request session=%s chars=%s", request.session_key(), len(request.text))
        logger.info("incoming text session=%s text=%r", request.session_key(), request.text)
        self._jobs.put((request, token))
        return {"status": "accepted", "session_key": request.session_key()}

    def process_pending(self, timeout: float = 0.0) -> bool:
        try:
            job = self._jobs.get(timeout=timeout)
        except Empty:
            return False
        if job is None:
            return False
        request, token = job
        self._process(request, token)
        return True

    def run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            request, token = job
            self._process(request, token)

    def stop(self) -> None:
        self._jobs.put(None)

    def _process(self, request: SpeechRequest, token: WorkToken) -> None:
        try:
            logger.info("summarizing speech request session=%s", request.session_key())
            text = self.summarizer.summarize(request.text)
            logger.info(
                "summary ready session=%s input_chars=%s output_chars=%s changed=%s",
                request.session_key(),
                len(request.text),
                len(text),
                text != request.text,
            )
            logger.info("summarized text session=%s text=%r", request.session_key(), text)
            if token.cancelled():
                logger.info("speech request cancelled before tts session=%s", request.session_key())
                return
            logger.info("generating speech session=%s", request.session_key())
            chunks = self.speech.generate(text)
            logger.info("generated speech chunks session=%s chunks=%s", request.session_key(), len(chunks))
            if self.config.session.cross_session_policy == "queue":
                with self._audio_lock:
                    self.player.play(chunks, token=token)
            else:
                self.player.play(chunks, token=token)
            logger.info("speech playback complete session=%s", request.session_key())
        except Exception:
            logger.exception("speech request failed session=%s", request.session_key())
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
            self.service.stop()
            self._send(200, {"status": "shutting_down"})
            threading.Thread(target=self.server.shutdown).start()
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
    host, port = httpd.server_address[:2]
    write_state(config, str(host), int(port), os.getpid())
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()
    try:
        service.run()
    finally:
        httpd.shutdown()
        server_thread.join()
        httpd.server_close()
    return 0
