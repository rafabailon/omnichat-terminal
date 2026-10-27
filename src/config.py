from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_REQUEST_RETRIES = 1
DEFAULT_ANTHROPIC_MAX_TOKENS = 2048
MAX_REQUEST_RETRIES = 5
MAX_ANTHROPIC_MAX_TOKENS = 100_000


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    api_key: str
    model: str
    base_url: str
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer "
    max_tokens: int | None = None


@dataclass(frozen=True)
class Settings:
    openai: ProviderSettings
    gemini: ProviderSettings
    anthropic: ProviderSettings
    compatible: ProviderSettings
    database_path: Path
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    request_retries: int = DEFAULT_REQUEST_RETRIES

    @classmethod
    def from_environment(cls, env_file: Path | None = None) -> Settings:
        """Build settings from process environment values and an optional ``.env`` file.

        Process environment variables always take precedence over the file, matching
        the usual dotenv semantics. The file is read without mutating ``os.environ``,
        which keeps repeated lookups (and tests) free of cross-call side effects.
        """
        path = env_file or Path.cwd() / ".env"
        file_values = dotenv_values(path) if path.is_file() else {}

        def get(name: str, default: str = "") -> str:
            if name in os.environ:
                return os.environ[name]
            value = file_values.get(name)
            return default if value is None else value

        database_setting = get("OMNICHAT_DB_PATH").strip()
        database_path = (
            Path(database_setting).expanduser()
            if database_setting
            else _default_data_directory() / "history.sqlite3"
        )

        return cls(
            openai=ProviderSettings(
                "openai",
                get("OPENAI_API_KEY").strip(),
                _text(get("OPENAI_MODEL"), "gpt-4o-mini"),
                _text(get("OPENAI_BASE_URL"), "https://api.openai.com/v1").rstrip("/"),
            ),
            gemini=ProviderSettings(
                "gemini",
                get("GEMINI_API_KEY").strip(),
                _text(get("GEMINI_MODEL"), "gemini-2.5-flash"),
                _text(
                    get("GEMINI_BASE_URL"),
                    "https://generativelanguage.googleapis.com/v1beta",
                ).rstrip("/"),
            ),
            anthropic=ProviderSettings(
                "anthropic",
                get("ANTHROPIC_API_KEY").strip(),
                _text(get("ANTHROPIC_MODEL"), "claude-haiku-4-5-20251001"),
                _text(get("ANTHROPIC_BASE_URL"), "https://api.anthropic.com/v1").rstrip("/"),
                max_tokens=_positive_int(
                    get("ANTHROPIC_MAX_TOKENS"),
                    "ANTHROPIC_MAX_TOKENS",
                    DEFAULT_ANTHROPIC_MAX_TOKENS,
                    maximum=MAX_ANTHROPIC_MAX_TOKENS,
                ),
            ),
            compatible=ProviderSettings(
                "compatible",
                get("COMPATIBLE_API_KEY").strip(),
                get("COMPATIBLE_MODEL").strip(),
                get("COMPATIBLE_BASE_URL").strip().rstrip("/"),
                _text(get("COMPATIBLE_API_KEY_HEADER"), "Authorization"),
                get("COMPATIBLE_API_KEY_PREFIX", "Bearer "),
            ),
            database_path=database_path,
            request_timeout=_positive_float(
                get("OMNICHAT_REQUEST_TIMEOUT"),
                "OMNICHAT_REQUEST_TIMEOUT",
                DEFAULT_REQUEST_TIMEOUT,
            ),
            request_retries=_non_negative_int(
                get("OMNICHAT_REQUEST_RETRIES"),
                "OMNICHAT_REQUEST_RETRIES",
                DEFAULT_REQUEST_RETRIES,
                maximum=MAX_REQUEST_RETRIES,
            ),
        )


def _text(value: str, default: str) -> str:
    """Return a trimmed value, falling back to a default when it is blank."""
    trimmed = value.strip()
    return trimmed or default


def _positive_float(value: str, name: str, default: float) -> float:
    raw = value.strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number.") from None
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return parsed


def _positive_int(value: str, name: str, default: int, *, maximum: int) -> int:
    raw = value.strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.") from None
    if not 1 <= parsed <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
    return parsed


def _non_negative_int(value: str, name: str, default: int, *, maximum: int) -> int:
    raw = value.strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}.") from None
    if not 0 <= parsed <= maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}.")
    return parsed


def _default_data_directory() -> Path:
    """Return the conventional per-user data directory for the current platform."""
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base.expanduser() / "omnichat-terminal"
