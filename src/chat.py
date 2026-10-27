from __future__ import annotations

from collections.abc import Sequence

from database import ChatStore
from models import ChatMessage, ChatSession
from providers import Provider


class ChatService:
    def __init__(self, store: ChatStore) -> None:
        self.store = store

    def ask(
        self,
        provider: Provider,
        prompt: str,
        session: ChatSession,
        history: Sequence[ChatMessage],
    ) -> str:
        """Return the model's actual response and persist the successful turn."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be blank.")
        response = provider.send_message(prompt, history)
        self.store.add_turn(session.id, prompt, response)
        return response
