from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import httpx

from models import ChatMessage

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 10.0


class ProviderError(RuntimeError):
    """A safe, user-facing error returned by a model provider."""


class Provider(ABC):
    def __init__(
        self,
        name: str,
        model: str,
        *,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
        retries: int = 0,
        retry_delay: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 0:
            raise ValueError("Provider retries cannot be negative.")
        if retry_delay < 0:
            raise ValueError("Provider retry delay cannot be negative.")
        self.name = name
        self.model = model
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self._retries = retries
        self._retry_delay = retry_delay
        self._sleep = sleep

    def __enter__(self) -> Provider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @abstractmethod
    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        """Send a prompt with prior turns and return the provider's response text."""

    def _post_json(
        self,
        url: str,
        *,
        headers: dict[str, str],
        payload: dict,
        params: dict[str, str] | None = None,
    ) -> dict:
        """POST JSON and return the decoded object, retrying transient failures."""
        attempts = self._retries + 1
        for attempt in range(attempts):
            try:
                response = self._client.post(url, headers=headers, json=payload, params=params)
            except httpx.TimeoutException as error:
                if attempt < self._retries:
                    self._sleep(self._retry_delay * (2**attempt))
                    continue
                raise ProviderError("The provider request timed out. Please try again.") from error
            except httpx.RequestError as error:
                if attempt < self._retries:
                    self._sleep(self._retry_delay * (2**attempt))
                    continue
                raise ProviderError(
                    "Could not connect to the provider. Check your connection."
                ) from error

            if response.is_error:
                if response.status_code in RETRYABLE_STATUS_CODES and attempt < self._retries:
                    self._sleep(self._retry_delay_for(response, attempt))
                    continue
                raise ProviderError(self._error_detail(response.status_code))

            try:
                data = response.json()
            except ValueError as error:
                raise ProviderError("The provider returned an invalid JSON response.") from error
            if not isinstance(data, dict):
                raise ProviderError("The provider returned an unexpected response format.")
            return data

        # The loop always returns or raises; this guards against future refactors.
        raise ProviderError("The provider request failed after multiple attempts.")

    def _retry_delay_for(self, response: httpx.Response, attempt: int) -> float:
        delay = self._retry_delay * (2**attempt)
        retry_after = response.headers.get("Retry-After", "").strip()
        if retry_after.isdigit():
            delay = max(delay, min(float(retry_after), MAX_RETRY_AFTER_SECONDS))
        return delay

    @staticmethod
    def _error_detail(status_code: int) -> str:
        if status_code in {401, 403}:
            return "The provider rejected the API key or permissions."
        if status_code == 429:
            return "The provider rate limit or quota was reached. Try again later."
        if status_code >= 500:
            return f"The provider is temporarily unavailable (HTTP {status_code})."
        return f"The provider rejected the request (HTTP {status_code})."

    @staticmethod
    def _required_text(value: object, provider_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ProviderError(f"{provider_name} returned an empty response.")
        return value.strip()

    @classmethod
    def _openai_message_text(cls, data: dict, provider_name: str) -> str:
        """Extract text from an OpenAI Chat Completions style response."""
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                f"{provider_name} returned an unexpected response format."
            ) from error
        if not isinstance(message, dict):
            raise ProviderError(f"{provider_name} returned an unexpected response format.")

        text = message.get("content")
        if isinstance(text, list):
            text = "\n".join(
                part["text"]
                for part in text
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if not isinstance(text, str) or not text.strip():
            refusal = message.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                text = refusal
        return cls._required_text(text, provider_name)
