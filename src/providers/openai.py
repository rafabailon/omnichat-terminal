from __future__ import annotations

from collections.abc import Sequence

from config import ProviderSettings
from models import ChatMessage

from .base import Provider


class OpenAIProvider(Provider):
    def __init__(self, settings: ProviderSettings, **kwargs) -> None:
        if not settings.api_key:
            raise ValueError("OPENAI_API_KEY is required.")
        super().__init__(settings.name, settings.model, **kwargs)
        self._api_key = settings.api_key
        self._base_url = settings.base_url

    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        messages = [{"role": message.role, "content": message.content} for message in history]
        messages.append({"role": "user", "content": prompt})
        data = self._post_json(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            payload={"model": self.model, "messages": messages},
        )
        return self._openai_message_text(data, "OpenAI")
