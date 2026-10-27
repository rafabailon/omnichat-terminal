from __future__ import annotations

import pytest

from _version import __version__
from cli import build_parser, main
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
)


@pytest.fixture
def isolated_environment(tmp_path, monkeypatch):
    for key in ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def test_parser_defaults_and_choices():
    parser = build_parser()

    assert parser.parse_args([]).provider is None
    assert parser.parse_args([]).db is None
    assert parser.parse_args(["--provider", "gemini"]).provider == "gemini"
    with pytest.raises(SystemExit):
        parser.parse_args(["--provider", "unknown"])


def test_version_option_uses_the_project_name(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"OmniChat Terminal {__version__}"


def test_list_providers_without_configuration(isolated_environment, capsys):
    assert main(["--list-providers"]) == 0
    assert "No providers configured" in capsys.readouterr().out


def test_list_providers_shows_configured_provider_and_model(isolated_environment, capsys):
    isolated_environment.setenv("OPENAI_API_KEY", "test-key")
    isolated_environment.setenv("OPENAI_MODEL", "test-model")

    assert main(["--list-providers"]) == 0

    output = capsys.readouterr().out
    assert "OpenAI / ChatGPT" in output
    assert "test-model" in output


def test_invalid_numeric_configuration_is_reported(isolated_environment, capsys):
    isolated_environment.setenv("OMNICHAT_REQUEST_TIMEOUT", "not-a-number")

    assert main(["--list-providers"]) == 2

    output = capsys.readouterr().out
    assert "Invalid configuration" in output
    assert "OMNICHAT_REQUEST_TIMEOUT" in output


def test_empty_database_path_is_rejected(isolated_environment, capsys):
    assert main(["--db", "   "]) == 2
    assert "non-empty path" in capsys.readouterr().out


def test_in_memory_database_is_rejected_gracefully(isolated_environment, capsys):
    assert main(["--db", ":memory:"]) == 2

    output = capsys.readouterr().out
    assert "Could not open the chat database" in output
    assert "persistent SQLite" in output


def test_no_providers_configured_returns_configuration_error(
    isolated_environment, tmp_path, capsys
):
    code = main(["--db", str(tmp_path / "history.sqlite3")])

    assert code == 2
    assert "No provider is configured" in capsys.readouterr().out


def test_settings_from_environment_remains_available_for_cli(isolated_environment):
    settings = Settings.from_environment()

    assert settings.openai.api_key == ""
    assert settings.request_retries == 1
