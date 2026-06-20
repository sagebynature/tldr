from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import tomllib


class ConfigError(ValueError):
    pass


DEFAULT_SYSTEM_PROMPT = """You summarize assistant responses for text-to-speech.
Return only a spoken summary.
Do not mention that this is a summary.
If the content is a question, preserve the question instead of answering it.
Do not include markdown, code fences, file paths, URLs, bullets, or formatting."""

DEFAULT_USER_PROMPT_TEMPLATE = """Summarize this response in {max_words} words or fewer.
Preserve the practical outcome and next action.

{text}"""


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    state_dir: str = "~/.cache/tts-summarizer"
    auto_start: bool = True
    startup_timeout_ms: int = 3000
    request_timeout_ms: int = 5000


@dataclass(frozen=True)
class SessionConfig:
    interrupt_same_session: bool = True
    max_queue_per_session: int = 1
    cross_session_policy: str = "queue"


@dataclass(frozen=True)
class SummarizerConfig:
    enabled: bool = True
    model: str = "mlx-community/Qwen3-0.6B-4bit"
    word_threshold: int = 0
    max_words: int = 40
    temperature: float = 0.2
    max_tokens: int = 180
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE


@dataclass(frozen=True)
class TtsConfig:
    model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    voice: str = "Chelsie"
    lang_code: str = "English"
    speed: float = 1.6
    ref_audio: str = ""
    ref_text: str = ""
    stream: bool = True
    sample_rate: int = 24000


@dataclass(frozen=True)
class AudioConfig:
    backend: str = "auto"
    output_dir: str = "~/.cache/tts-summarizer/audio"
    save: bool = False


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    source: Path | None = None


def _expand(path: str, *, home: Path | None = None) -> Path:
    if path.startswith("~/") and home is not None:
        return home / path[2:]
    return Path(path).expanduser()


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid = instance.__dataclass_fields__.keys()
    unknown = sorted(set(values) - set(valid))
    if unknown:
        raise ConfigError(f"unknown config keys for {type(instance).__name__}: {', '.join(unknown)}")
    return replace(instance, **values)


def _apply(raw: dict[str, Any], source: Path | None) -> Config:
    cfg = Config(source=source)
    allowed = {"server", "session", "summarizer", "tts", "audio"}
    unknown_sections = sorted(set(raw) - allowed)
    if unknown_sections:
        raise ConfigError(f"unknown config sections: {', '.join(unknown_sections)}")
    return Config(
        server=_merge_dataclass(cfg.server, raw.get("server", {})),
        session=_merge_dataclass(cfg.session, raw.get("session", {})),
        summarizer=_merge_dataclass(cfg.summarizer, raw.get("summarizer", {})),
        tts=_merge_dataclass(cfg.tts, raw.get("tts", {})),
        audio=_merge_dataclass(cfg.audio, raw.get("audio", {})),
        source=source,
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def load_config(explicit_path: str | None, cwd: Path | None = None, home: Path | None = None) -> Config:
    cwd = cwd or Path.cwd()
    home = home or Path.home()
    if explicit_path:
        path = _expand(explicit_path, home=home)
        if not path.exists():
            raise ConfigError(f"config not found: {path}")
        return _apply(_read(path), path)

    cwd_config = cwd / "config.toml"
    if cwd_config.exists():
        return _apply(_read(cwd_config), cwd_config)

    user_config = home / ".config" / "tts-summarizer" / "config.toml"
    if user_config.exists():
        return _apply(_read(user_config), user_config)

    return Config()
