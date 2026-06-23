from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
import tomllib


class ConfigError(ValueError):
    pass


DEFAULT_SYSTEM_PROMPT = """You summarize assistant responses for text-to-speech. Return only the final spoken summary. Do not include reasoning, analysis, planning, explanations, prefaces, markdown, code fences, file paths, URLs, bullets, or formatting. If the content is a question, preserve the question instead of answering it."""

DEFAULT_USER_PROMPT_TEMPLATE = """Write one complete spoken summary in {max_words} words or fewer. Stop after the summary.

{text}"""


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 0
    state_dir: str = "~/.cache/echobrief"
    auto_start: bool = True
    startup_timeout_ms: int = 3000
    request_timeout_ms: int = 5000


@dataclass(frozen=True)
class SummarizerProfileConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = ""
    model: str = "local-model"
    word_threshold: int = 0
    max_words: int = 40
    temperature: float = 0.2
    max_tokens: int = 180
    extra_body: dict[str, object] = field(default_factory=dict)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE


@dataclass(frozen=True)
class SummarizerConfig:
    default_profile: str = "default"
    profiles: dict[str, SummarizerProfileConfig] = field(
        default_factory=lambda: {
            "default": SummarizerProfileConfig(),
        }
    )


@dataclass(frozen=True)
class TtsProfileConfig:
    model: str = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    stream: bool = True
    sample_rate: int = 24000
    generate_kwargs: dict[str, object] = field(default_factory=dict)
    backend: str = "mlx"
    base_url: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class TtsConfig:
    default_profile: str = "qwen"
    profiles: dict[str, TtsProfileConfig] = field(
        default_factory=lambda: {
            "qwen": TtsProfileConfig(),
        }
    )


@dataclass(frozen=True)
class LoggingConfig:
    config_file: str = ""


@dataclass(frozen=True)
class Config:
    server: ServerConfig = field(default_factory=ServerConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    source: Path | None = None


def _expand(path: str, *, home: Path | None = None) -> Path:
    if path.startswith("~/") and home is not None:
        return home / path[2:]
    return Path(path).expanduser()


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    valid = instance.__dataclass_fields__.keys()
    unknown = sorted(set(values) - set(valid))
    if unknown:
        raise ConfigError(
            f"unknown config keys for {type(instance).__name__}: {', '.join(unknown)}"
        )
    return replace(instance, **values)


def _merge_summarizer_config(
    instance: SummarizerConfig, values: dict[str, Any]
) -> SummarizerConfig:
    valid = {"default_profile", "profiles"}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ConfigError(
            f"unknown config keys for SummarizerConfig: {', '.join(unknown)}"
        )
    profiles = {
        name: _merge_dataclass(SummarizerProfileConfig(), profile)
        for name, profile in values.get("profiles", {}).items()
    }
    merged = replace(
        instance,
        default_profile=values.get("default_profile", instance.default_profile),
        profiles=profiles or instance.profiles,
    )
    if merged.default_profile not in merged.profiles:
        raise ConfigError(
            f"unknown default summarizer profile: {merged.default_profile}"
        )
    return merged


def _merge_tts_config(instance: TtsConfig, values: dict[str, Any]) -> TtsConfig:
    valid = {"default_profile", "profiles"}
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ConfigError(f"unknown config keys for TtsConfig: {', '.join(unknown)}")
    profiles = {
        name: _merge_dataclass(TtsProfileConfig(), profile)
        for name, profile in values.get("profiles", {}).items()
    }
    merged = replace(
        instance,
        default_profile=values.get("default_profile", instance.default_profile),
        profiles=profiles or instance.profiles,
    )
    if merged.default_profile not in merged.profiles:
        raise ConfigError(f"unknown default TTS profile: {merged.default_profile}")
    return merged


def _apply(raw: dict[str, Any], source: Path | None) -> Config:
    cfg = Config(source=source)
    allowed = {"server", "summarizer", "tts", "logging"}
    unknown_sections = sorted(set(raw) - allowed)
    if unknown_sections:
        raise ConfigError(f"unknown sections: {', '.join(unknown_sections)}")
    return Config(
        server=_merge_dataclass(cfg.server, raw.get("server", {})),
        summarizer=_merge_summarizer_config(cfg.summarizer, raw.get("summarizer", {})),
        tts=_merge_tts_config(cfg.tts, raw.get("tts", {})),
        logging=_merge_dataclass(cfg.logging, raw.get("logging", {})),
        source=source,
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


def load_config(
    explicit_path: str | None, cwd: Path | None = None, home: Path | None = None
) -> Config:
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
    user_config = home / ".config" / "echobrief" / "config.toml"
    if user_config.exists():
        return _apply(_read(user_config), user_config)
    legacy_user_config = home / ".config" / "tts-summarizer" / "config.toml"
    if legacy_user_config.exists():
        return _apply(_read(legacy_user_config), legacy_user_config)
    return Config()
