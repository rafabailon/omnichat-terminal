from __future__ import annotations

from collections.abc import Sequence

import pytest
from rich.console import Console

from app import OmniChatApp
from chat import ChatService
from config import ProviderSettings, Settings
from database import ChatStore
from models import ChatMessage
from providers import Provider, ProviderError


class RecordingProvider(Provider):
    def __init__(self, answer: str = "The model's real answer is 42.") -> None:
        super().__init__("openai", "recording-model")
        self.answer = answer
        self.prompts: list[str] = []
        self.histories: list[list[ChatMessage]] = []

    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        self.prompts.append(prompt)
        self.histories.append(list(history))
        return self.answer


def empty_settings(database_path) -> Settings:
    empty = ProviderSettings("", "", "", "")
    return Settings(empty, empty, empty, empty, database_path)


def test_generic_prompt_is_sent_to_provider_and_exact_response_is_returned(store: ChatStore):
    session = store.create_session("openai", "recording-model")
    provider = RecordingProvider()

    answer = ChatService(store).ask(provider, "What is 6 multiplied by 7?", session, [])

    assert provider.prompts == ["What is 6 multiplied by 7?"]
    assert answer == "The model's real answer is 42."
    assert store.get_messages(session.id)[1].content == answer


def test_interactive_chat_displays_and_saves_provider_answer(tmp_path, store: ChatStore):
    providers = []

    def factory():
        provider = RecordingProvider("A direct provider result: 42.")
        providers.append(provider)
        return provider

    settings = empty_settings(tmp_path / "chat.sqlite3")
    inputs = iter(["What is 6 times 7?", "/quit"])
    console = Console(record=True, force_terminal=False, width=100)
    app = OmniChatApp(
        settings,
        store,
        console=console,
        input_fn=lambda _: next(inputs),
        providers={"openai": factory},
    )

    assert app.run() == 0
    output = console.export_text()
    saved = store.list_sessions()[0]

    assert "A direct provider result: 42." in output
    assert providers[0].prompts == ["What is 6 times 7?"]
    assert [message.content for message in store.get_messages(saved.id)] == [
        "What is 6 times 7?",
        "A direct provider result: 42.",
    ]


def test_chat_history_is_sent_on_follow_up_turn(store: ChatStore):
    session = store.create_session("openai", "recording-model")
    provider = RecordingProvider()
    service = ChatService(store)

    first_answer = service.ask(provider, "My name is Inez.", session, [])
    history = [
        ChatMessage(role=message.role, content=message.content)
        for message in store.get_messages(session.id)
    ]
    service.ask(provider, "What is my name?", session, history)

    assert first_answer == provider.answer
    assert provider.histories[1] == [
        ChatMessage(role="user", content="My name is Inez."),
        ChatMessage(role="assistant", content=provider.answer),
    ]


def test_loading_session_automatically_uses_its_saved_provider(tmp_path, store):
    session = store.create_session("openai", "recording-model")
    store.add_turn(session.id, "Remember this.", "Saved answer.")
    providers = []

    def factory():
        provider = RecordingProvider()
        providers.append(provider)
        return provider

    inputs = iter(["What did I say?", "/quit"])
    console = Console(record=True, force_terminal=False, width=100)
    app = OmniChatApp(
        empty_settings(tmp_path / "chat.sqlite3"),
        store,
        console=console,
        input_fn=lambda _: next(inputs),
        providers={"openai": factory},
    )

    assert app.run(session_id=session.id) == 0
    assert providers[0].histories[0] == [
        ChatMessage(role="user", content="Remember this."),
        ChatMessage(role="assistant", content="Saved answer."),
    ]
    assert "Loaded: Remember this." in console.export_text()


def test_blank_prompts_are_rejected_before_calling_the_provider(store: ChatStore):
    session = store.create_session("openai", "recording-model")
    provider = RecordingProvider()

    with pytest.raises(ValueError, match="Prompt cannot be blank"):
        ChatService(store).ask(provider, "   ", session, [])

    assert provider.prompts == []
    assert store.get_messages(session.id) == []


def test_provider_failures_do_not_persist_a_partial_turn(store: ChatStore):
    class FailingProvider(Provider):
        def __init__(self) -> None:
            super().__init__("openai", "recording-model")

        def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
            raise ProviderError("temporary upstream failure")

    session = store.create_session("openai", "recording-model")

    with pytest.raises(ProviderError, match="temporary upstream failure"):
        ChatService(store).ask(FailingProvider(), "Will this be saved?", session, [])

    assert store.get_messages(session.id) == []
