from __future__ import annotations

from queue import Empty, Queue
import logging
import os
import socket
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

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
        logger.info(
            "accepted speech request session=%s chars=%s",
            request.session_key(),
            len(request.text),
        )
        logger.info(
            "incoming text session=%s text=%r", request.session_key(), request.text
        )
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
            logger.info(
                "summarized text session=%s text=%r", request.session_key(), text
            )
            if token.cancelled():
                logger.info(
                    "speech request cancelled before tts session=%s",
                    request.session_key(),
                )
                return
            logger.info("generating speech session=%s", request.session_key())
            chunks = self.speech.generate(text)
            if token.cancelled():
                logger.info(
                    "speech request cancelled before playback session=%s",
                    request.session_key(),
                )
                return
            with self._audio_lock:
                self.player.play(chunks, token=token)
            logger.info("speech playback complete session=%s", request.session_key())
        except Exception:
            logger.exception("speech request failed session=%s", request.session_key())
        finally:
            self.sessions.finish(token)

    def health(self) -> dict[str, object]:
        return {"status": "ok", "pid": os.getpid()}


def create_app(config: Config, service: TtsService | None = None) -> FastAPI:
    service = service or TtsService(config)
    app = FastAPI(title="tts-summarizer")
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, object]:
        return service.health()

    @app.post("/v1/speak")
    def speak(payload: dict[str, object]):
        try:
            request = SpeechRequest.from_json(payload)
        except RequestError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return service.handle(request)

    @app.post("/shutdown")
    def shutdown() -> dict[str, object]:
        service.stop()
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True
        return {"status": "shutting_down"}

    return app


def run_server(config: Config) -> int:
    service = TtsService(config)
    app = create_app(config, service)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((config.server.host, config.server.port))
    sock.listen()
    host, port = sock.getsockname()[:2]
    write_state(config, str(host), int(port), os.getpid())
    server_config = uvicorn.Config(app, host=config.server.host, port=int(port), log_config=None)
    server = uvicorn.Server(server_config)
    app.state.server = server
    worker = threading.Thread(target=service.run, daemon=True)
    worker.start()
    try:
        server.run(sockets=[sock])
    finally:
        service.stop()
    return 0
