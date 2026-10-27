from __future__ import annotations

from pathlib import Path

from config import ProviderSettings, Settings
from providers import (
    AnthropicProvider,
    CompatibleProvider,
    GeminiProvider,
    OpenAIProvider,
)
from registry import PROVIDER_LABELS, PROVIDER_NAMES, configured_providers


def make_settings(**overrides) -> Settings:
    empty = ProviderSettings("", "", "", "")
    values = {
        "openai": empty,
        "gemini": empty,
        "anthropic": empty,
        "compatible": empty,
        "database_path": Path("unused.sqlite3"),
    }
    values.update(overrides)
    return Settings(**values)


def test_provider_names_match_labels():
    assert PROVIDER_NAMES == tuple(PROVIDER_LABELS)


def test_no_providers_are_returned_without_configuration():
    assert configured_providers(make_settings()) == {}


def test_each_configured_provider_gets_a_factory():
    settings = make_settings(
        openai=ProviderSettings("openai", "key", "gpt-test", "https://openai.example/v1"),
        gemini=ProviderSettings("gemini", "key", "gemini-test", "https://gemini.example"),
        anthropic=ProviderSettings(
            "anthropic", "key", "claude-test", "https://anthropic.example", max_tokens=1234
        ),
        compatible=ProviderSettings(
            "compatible", "", "local-test", "https://compatible.example/v1"
        ),
        request_timeout=12.0,
        request_retries=3,
    )

    factories = configured_providers(settings)

    assert list(factories) == ["openai", "gemini", "anthropic", "compatible"]
    expected_types = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "anthropic": AnthropicProvider,
        "compatible": CompatibleProvider,
    }
    for name, factory in factories.items():
        provider = factory()
        try:
            assert isinstance(provider, expected_types[name])
            assert provider.model == getattr(settings, name).model
            assert provider._retries == 3
            assert provider._client.timeout.read == 12.0
        finally:
            provider.close()


def test_anthropic_factory_uses_configured_token_limit():
    settings = make_settings(
        anthropic=ProviderSettings(
            "anthropic", "key", "claude-test", "https://anthropic.example", max_tokens=3210
        )
    )

    provider = configured_providers(settings)["anthropic"]()

    try:
        assert isinstance(provider, AnthropicProvider)
        assert provider._max_tokens == 3210
    finally:
        provider.close()


def test_compatible_provider_requires_both_url_and_model():
    only_url = make_settings(
        compatible=ProviderSettings("compatible", "", "", "https://compatible.example/v1")
    )
    only_model = make_settings(compatible=ProviderSettings("compatible", "", "local-test", ""))

    assert configured_providers(only_url) == {}
    assert configured_providers(only_model) == {}
