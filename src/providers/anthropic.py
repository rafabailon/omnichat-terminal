from __future__ import annotations

from collections.abc import Sequence

from config import ProviderSettings
from models import ChatMessage

from .base import Provider, ProviderError

DEFAULT_MAX_TOKENS = 2048


class AnthropicProvider(Provider):
    def __init__(self, settings: ProviderSettings, **kwargs) -> None:
        if not settings.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required.")
        super().__init__(settings.name, settings.model, **kwargs)
        self._api_key = settings.api_key
        self._base_url = settings.base_url
        self._max_tokens = settings.max_tokens or DEFAULT_MAX_TOKENS

    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        messages = [
            {
                "role": "assistant" if message.role == "assistant" else "user",
                "content": message.content,
            }
            for message in history
        ]
        messages.append({"role": "user", "content": prompt})
        data = self._post_json(
            f"{self._base_url}/messages",
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            payload={
                "model": self.model,
                "max_tokens": self._max_tokens,
                "messages": messages,
            },
        )
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError("Anthropic returned an unexpected response format.")
        text = "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        )
        return self._required_text(text, "Anthropic")
