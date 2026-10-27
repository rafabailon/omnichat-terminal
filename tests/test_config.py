from __future__ import annotations

import os
import sys

import pytest

from config import Settings

ENVIRONMENT_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MAX_TOKENS",
    "COMPATIBLE_API_KEY",
    "COMPATIBLE_API_KEY_HEADER",
    "COMPATIBLE_API_KEY_PREFIX",
    "COMPATIBLE_MODEL",
    "COMPATIBLE_BASE_URL",
    "OMNICHAT_DB_PATH",
    "OMNICHAT_REQUEST_TIMEOUT",
    "OMNICHAT_REQUEST_RETRIES",
    "XDG_DATA_HOME",
    "LOCALAPPDATA",
)


@pytest.fixture
def clean_environment(monkeypatch):
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_settings_load_dotenv_and_process_environment_takes_precedence(tmp_path, clean_environment):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=file-key\nOPENAI_MODEL=file-model\nOMNICHAT_DB_PATH=./from-file.sqlite3\n",
        encoding="utf-8",
    )
    clean_environment.setenv("OPENAI_MODEL", "environment-model")

    settings = Settings.from_environment(env_file)

    assert settings.openai.api_key == "file-key"
    assert settings.openai.model == "environment-model"
    assert settings.database_path.name == "from-file.sqlite3"
    assert settings.gemini.api_key == ""


def test_reading_dotenv_does_not_mutate_the_process_environment(tmp_path, clean_environment):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

    settings = Settings.from_environment(env_file)

    assert settings.openai.api_key == "file-key"
    assert "OPENAI_API_KEY" not in os.environ


def test_an_explicit_environment_value_wins_even_when_it_is_empty(tmp_path, clean_environment):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    clean_environment.setenv("OPENAI_API_KEY", "")

    assert Settings.from_environment(env_file).openai.api_key == ""


def test_blank_model_and_base_url_fall_back_to_safe_defaults(tmp_path, clean_environment):
    env_file = tmp_path / "missing.env"
    clean_environment.setenv("OPENAI_MODEL", "   ")
    clean_environment.setenv("OPENAI_BASE_URL", " https://custom.example/v1/ ")

    settings = Settings.from_environment(env_file)

    assert settings.openai.model == "gpt-4o-mini"
    assert settings.openai.base_url == "https://custom.example/v1"


def test_compatible_endpoint_keeps_blank_values_and_supports_empty_prefix(
    tmp_path, clean_environment
):
    env_file = tmp_path / "missing.env"
    clean_environment.setenv("COMPATIBLE_API_KEY", " local-key ")
    clean_environment.setenv("COMPATIBLE_BASE_URL", " https://local.example/v1/ ")
    clean_environment.setenv("COMPATIBLE_MODEL", " local-model ")
    clean_environment.setenv("COMPATIBLE_API_KEY_HEADER", " api-key ")
    clean_environment.setenv("COMPATIBLE_API_KEY_PREFIX", "")

    settings = Settings.from_environment(env_file)

    assert settings.compatible.api_key == "local-key"
    assert settings.compatible.base_url == "https://local.example/v1"
    assert settings.compatible.model == "local-model"
    assert settings.compatible.api_key_header == "api-key"
    assert settings.compatible.api_key_prefix == ""


def test_blank_compatible_endpoint_stays_unconfigured(tmp_path, clean_environment):
    settings = Settings.from_environment(tmp_path / "missing.env")

    assert settings.compatible.api_key == ""
    assert settings.compatible.base_url == ""
    assert settings.compatible.model == ""


def test_timeout_retries_and_anthropic_token_limit_are_configurable(tmp_path, clean_environment):
    env_file = tmp_path / "missing.env"
    clean_environment.setenv("OMNICHAT_REQUEST_TIMEOUT", " 12.5 ")
    clean_environment.setenv("OMNICHAT_REQUEST_RETRIES", "3")
    clean_environment.setenv("ANTHROPIC_MAX_TOKENS", "4096")

    settings = Settings.from_environment(env_file)

    assert settings.request_timeout == 12.5
    assert settings.request_retries == 3
    assert settings.anthropic.max_tokens == 4096


def test_default_request_policy_and_anthropic_token_limit(tmp_path, clean_environment):
    settings = Settings.from_environment(tmp_path / "missing.env")

    assert settings.request_timeout == 60.0
    assert settings.request_retries == 1
    assert settings.anthropic.max_tokens == 2048


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("OMNICHAT_REQUEST_TIMEOUT", "0", "positive number"),
        ("OMNICHAT_REQUEST_TIMEOUT", "-1", "positive number"),
        ("OMNICHAT_REQUEST_TIMEOUT", "soon", "positive number"),
        ("OMNICHAT_REQUEST_TIMEOUT", "nan", "positive number"),
        ("OMNICHAT_REQUEST_RETRIES", "-1", "between 0 and 5"),
        ("OMNICHAT_REQUEST_RETRIES", "6", "between 0 and 5"),
        ("OMNICHAT_REQUEST_RETRIES", "many", "between 0 and 5"),
        ("ANTHROPIC_MAX_TOKENS", "0", "between 1 and 100000"),
        ("ANTHROPIC_MAX_TOKENS", "100001", "between 1 and 100000"),
    ],
)
def test_invalid_numeric_configuration_is_rejected(
    tmp_path, clean_environment, variable, value, message
):
    clean_environment.setenv(variable, value)

    with pytest.raises(ValueError, match=message):
        Settings.from_environment(tmp_path / "missing.env")


def test_default_database_location_uses_the_platform_data_directory(tmp_path, clean_environment):
    if os.name == "nt" or sys.platform == "darwin":
        pytest.skip("XDG data directory check applies to Linux-like platforms.")
    clean_environment.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    settings = Settings.from_environment(tmp_path / "missing.env")

    assert settings.database_path == tmp_path / "xdg" / "omnichat-terminal" / "history.sqlite3"
