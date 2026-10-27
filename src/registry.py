from __future__ import annotations

from collections.abc import Callable

from config import Settings
from providers import (
    AnthropicProvider,
    CompatibleProvider,
    GeminiProvider,
    OpenAIProvider,
    Provider,
)

ProviderFactory = Callable[[], Provider]

PROVIDER_LABELS = {
    "openai": "OpenAI / ChatGPT",
    "gemini": "Google Gemini",
    "anthropic": "Anthropic Claude",
    "compatible": "OpenAI-compatible endpoint",
}
PROVIDER_NAMES = tuple(PROVIDER_LABELS)


def configured_providers(settings: Settings) -> dict[str, ProviderFactory]:
    """Return factories for every provider that has enough configuration to run."""
    factories: dict[str, ProviderFactory] = {}
    retries = settings.request_retries
    if settings.openai.api_key:
        factories["openai"] = lambda: OpenAIProvider(
            settings.openai,
            timeout=settings.request_timeout,
            retries=retries,
        )
    if settings.gemini.api_key:
        factories["gemini"] = lambda: GeminiProvider(
            settings.gemini,
            timeout=settings.request_timeout,
            retries=retries,
        )
    if settings.anthropic.api_key:
        factories["anthropic"] = lambda: AnthropicProvider(
            settings.anthropic,
            timeout=settings.request_timeout,
            retries=retries,
        )
    if settings.compatible.base_url and settings.compatible.model:
        factories["compatible"] = lambda: CompatibleProvider(
            settings.compatible,
            timeout=settings.request_timeout,
            retries=retries,
        )
    return factories
