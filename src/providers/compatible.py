from __future__ import annotations

from collections.abc import Sequence

from config import ProviderSettings
from models import ChatMessage

from .base import Provider


class CompatibleProvider(Provider):
    """Adapter for endpoints that implement the OpenAI Chat Completions API."""

    def __init__(self, settings: ProviderSettings, **kwargs) -> None:
        if not settings.base_url or not settings.model:
            raise ValueError("COMPATIBLE_BASE_URL and COMPATIBLE_MODEL are required.")
        if settings.api_key and not settings.api_key_header.strip():
            raise ValueError("COMPATIBLE_API_KEY_HEADER cannot be blank.")
        if "\r" in settings.api_key_header or "\n" in settings.api_key_header:
            raise ValueError("COMPATIBLE_API_KEY_HEADER contains invalid characters.")
        super().__init__(settings.name, settings.model, **kwargs)
        self._settings = settings
        self._api_key = settings.api_key
        self._base_url = settings.base_url

    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        messages = [{"role": message.role, "content": message.content} for message in history]
        messages.append({"role": "user", "content": prompt})
        headers = {}
        if self._api_key:
            headers[self._settings.api_key_header] = (
                f"{self._settings.api_key_prefix}{self._api_key}"
            )
        data = self._post_json(
            f"{self._base_url}/chat/completions",
            headers=headers,
            payload={"model": self.model, "messages": messages},
        )
        return self._openai_message_text(data, "Compatible endpoint")
