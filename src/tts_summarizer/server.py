from __future__ import annotations

from collections.abc import Iterable
import logging
import os
import socket

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
import uvicorn

from .audio import chunks_to_wav_stream
from .config import Config
from .request import RequestError, SpeechRequest
from .speech import SpeechGenerator
from .state import write_state
from .summarizer import Summarizer


logger = logging.getLogger(__name__)


class SpeakRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    metadata: dict[str, object] | None = None
    summarize: bool = True
    tts_profile: str | None = None
    summarizer_profile: str | None = None


class SummarizeRequestBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str
    word_threshold: int | None = None
    max_words: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    summarizer_profile: str | None = None


class SummarizeResponseBody(BaseModel):
    summary: str
    changed: bool


SUMMARY_OVERRIDE_KEYS = {
    "word_threshold",
    "max_words",
    "temperature",
    "max_tokens",
}


def synthesize_speech(request: SpeechRequest, summarizer, speech) -> Iterable[bytes]:
    logger.info("incoming text session=%s text=%r", request.session_key(), request.text)
    if request.summarize:
        logger.info("summarizing speech request session=%s", request.session_key())
        text = summarizer.summarize(
            request.text, profile_name=request.summarizer_profile
        )
        logger.info(
            "summary ready session=%s input_chars=%s output_chars=%s changed=%s",
            request.session_key(),
            len(request.text),
            len(text),
            text != request.text,
        )
        logger.info("summarized text session=%s text=%r", request.session_key(), text)
    else:
        text = request.text
        logger.info("summary skipped session=%s", request.session_key())
    logger.info(
        "generating speech session=%s profile=%s",
        request.session_key(),
        request.tts_profile,
    )
    sample_rate = (
        speech.sample_rate(request.tts_profile)
        if hasattr(speech, "sample_rate")
        else getattr(getattr(speech, "config", None), "sample_rate", 8000)
    )
    return chunks_to_wav_stream(
        speech.generate(text, profile_name=request.tts_profile), sample_rate
    )


def _summary_request(payload: dict[str, object], config: Config):
    unknown = sorted(set(payload) - (SUMMARY_OVERRIDE_KEYS | {"text", "summarizer_profile"}))
    if unknown:
        raise RequestError(f"unknown summarize request keys: {', '.join(unknown)}")
    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RequestError("summarize request requires non-empty text")
    overrides: dict[str, object] = {}
    summarizer_profile = payload.get("summarizer_profile")
    if summarizer_profile is not None and not isinstance(summarizer_profile, str):
        raise RequestError("summarizer_profile must be a string")
    for key in ("word_threshold", "max_words", "max_tokens"):
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise RequestError(f"{key} must be an integer")
        overrides[key] = value
    value = payload.get("temperature")
    if value is not None:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise RequestError("temperature must be a number")
        overrides["temperature"] = float(value)
    return text, summarizer_profile, overrides


def create_app(config: Config, summarizer=None, speech=None) -> FastAPI:
    summarizer = summarizer or Summarizer(config.summarizer)
    speech = speech or SpeechGenerator(config.tts)
    app = FastAPI(title="tts-summarizer")
    app.state.summarizer = summarizer
    app.state.speech = speech

    @app.exception_handler(RequestValidationError)
    def validation_error(
        _request: object, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": "invalid request body"})

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "pid": os.getpid()}

    @app.post("/v1/speak")
    def speak(payload: SpeakRequestBody, http_request: Request) -> Response:
        try:
            speech_request = SpeechRequest.from_json(
                payload.model_dump(exclude_none=True),
                caller=http_request.headers.get("X-TTS-Caller"),
                session_id=http_request.headers.get("X-TTS-Session-Id"),
            )
        except RequestError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            body = synthesize_speech(speech_request, summarizer, speech)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return StreamingResponse(body, media_type="audio/wav")

    @app.post("/v1/summarize", response_model=SummarizeResponseBody)
    def summarize(payload: SummarizeRequestBody):
        try:
            text, summarizer_profile, overrides = _summary_request(
                payload.model_dump(exclude_none=True), config
            )
        except RequestError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        try:
            summary = summarizer.summarize(
                text, profile_name=summarizer_profile, overrides=overrides
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
        return {"summary": summary, "changed": summary != text}

    @app.post("/shutdown")
    def shutdown() -> dict[str, object]:
        server = getattr(app.state, "server", None)
        if server is not None:
            server.should_exit = True
        return {"status": "shutting_down"}

    return app


def run_server(config: Config) -> int:
    app = create_app(config)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((config.server.host, config.server.port))
    sock.listen()
    host, port = sock.getsockname()[:2]
    write_state(config, str(host), int(port), os.getpid())
    server_config = uvicorn.Config(
        app, host=config.server.host, port=int(port), log_config=None
    )
    server = uvicorn.Server(server_config)
    app.state.server = server
    server.run(sockets=[sock])
    return 0
