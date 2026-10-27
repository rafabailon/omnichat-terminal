from __future__ import annotations

from collections.abc import Sequence

from config import ProviderSettings
from models import ChatMessage

from .base import Provider, ProviderError


class GeminiProvider(Provider):
    def __init__(self, settings: ProviderSettings, **kwargs) -> None:
        if not settings.api_key:
            raise ValueError("GEMINI_API_KEY is required.")
        super().__init__(settings.name, settings.model, **kwargs)
        self._api_key = settings.api_key
        self._base_url = settings.base_url

    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        contents = [
            {
                "role": "model" if message.role == "assistant" else "user",
                "parts": [{"text": message.content}],
            }
            for message in history
        ]
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        data = self._post_json(
            f"{self._base_url}/models/{self.model}:generateContent",
            headers={},
            params={"key": self._api_key},
            payload={"contents": contents},
        )
        feedback = data.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            raise ProviderError(f"Gemini blocked the request ({feedback['blockReason']}).")
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError("Gemini returned an unexpected response format.") from error
        if not isinstance(parts, list):
            raise ProviderError("Gemini returned an unexpected response format.")
        text = "\n".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        return self._required_text(text, "Gemini")
