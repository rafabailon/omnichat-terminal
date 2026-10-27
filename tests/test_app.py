from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from app import OmniChatApp
from config import ProviderSettings, Settings
from database import ChatStore
from models import ChatMessage
from providers import Provider, ProviderError


class RecordingProvider(Provider):
    def __init__(
        self,
        answer: str = "Recorded answer.",
        *,
        name: str = "openai",
        model: str = "recording-model",
        error: BaseException | None = None,
    ) -> None:
        super().__init__(name, model)
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []
        self.histories: list[list[ChatMessage]] = []

    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        self.prompts.append(prompt)
        self.histories.append(list(history))
        if self.error is not None:
            raise self.error
        return self.answer


def empty_settings(database_path: Path) -> Settings:
    empty = ProviderSettings("", "", "", "")
    return Settings(empty, empty, empty, empty, database_path)


def build_app(
    tmp_path: Path,
    store: ChatStore,
    providers: dict,
    inputs: list[str],
    *,
    width: int = 140,
) -> tuple[OmniChatApp, Console]:
    values = iter(inputs)

    def input_fn(_: str) -> str:
        try:
            return next(values)
        except StopIteration as error:
            raise EOFError from error

    console = Console(record=True, force_terminal=False, width=width)
    app = OmniChatApp(
        empty_settings(tmp_path / "unused.sqlite3"),
        store,
        console=console,
        input_fn=input_fn,
        providers=providers,
    )
    return app, console


def test_help_lists_new_commands_and_commands_are_case_insensitive(tmp_path, store):
    app, console = build_app(tmp_path, store, {"openai": RecordingProvider}, ["/HELP", "/QUIT"])

    assert app.run() == 0

    output = console.export_text()
    assert "/search <text>" in output
    assert "/export [path]" in output
    assert "/clear" in output


def test_export_command_writes_the_current_conversation(tmp_path, store):
    destination = tmp_path / "my conversation.md"
    app, console = build_app(
        tmp_path,
        store,
        {"openai": RecordingProvider},
        ["Hello there", f"/export {destination}", "/quit"],
    )

    assert app.run() == 0

    assert destination.is_file()
    content = destination.read_text(encoding="utf-8")
    assert "# Hello there" in content
    assert "Recorded answer." in content
    assert "Conversation exported to" in console.export_text()


def test_export_failure_is_reported_without_crashing(tmp_path, store):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    app, console = build_app(
        tmp_path,
        store,
        {"openai": RecordingProvider},
        [f"/export {blocker / 'chat.md'}", "/quit"],
    )

    assert app.run() == 0
    assert "Could not export this conversation" in console.export_text()


def test_search_command_finds_older_sessions(tmp_path, store):
    old = store.create_session("openai", "old-model")
    store.add_turn(old.id, "Tell me about the tide.", "The tide rises at night.")
    app, console = build_app(
        tmp_path,
        store,
        {"openai": RecordingProvider},
        ["/search tide", "/quit"],
    )

    assert app.run() == 0

    output = console.export_text()
    assert old.id in output
    assert "The tide rises at night." in output
    assert "Use /load <session-id>" in output


def test_search_without_a_query_shows_usage(tmp_path, store):
    app, console = build_app(tmp_path, store, {"openai": RecordingProvider}, ["/search", "/quit"])

    assert app.run() == 0
    assert "Usage: /search <text>" in console.export_text()


def test_history_command_lists_saved_sessions(tmp_path, store):
    session = store.create_session("openai", "gpt-test")
    store.add_turn(session.id, "Remember this title.", "Okay.")
    app, console = build_app(
        tmp_path,
        store,
        {"openai": RecordingProvider},
        ["/history", "/quit"],
    )

    assert app.run() == 0
    output = console.export_text()
    assert session.id in output
    assert "Remember this title." in output


def test_delete_removes_an_explicit_session(tmp_path, store):
    obsolete = store.create_session("openai", "gpt-test")
    app, console = build_app(
        tmp_path,
        store,
        {"openai": RecordingProvider},
        [f"/delete {obsolete.id}", "/quit"],
    )

    assert app.run() == 0

    assert store.get_session(obsolete.id) is None
    assert len(store.list_sessions()) == 1
    assert "Chat history deleted." in console.export_text()


def test_clear_removes_the_current_session_and_starts_another(tmp_path, store):
    app, console = build_app(
        tmp_path,
        store,
        {"openai": RecordingProvider},
        ["/clear", "/quit"],
    )

    assert app.run() == 0

    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert store.get_messages(sessions[0].id) == []
    assert "New session" in console.export_text()


def test_invalid_provider_number_does_not_select_the_last_provider(tmp_path, store):
    app, console = build_app(
        tmp_path,
        store,
        {
            "openai": lambda: RecordingProvider(name="openai"),
            "gemini": lambda: RecordingProvider(name="gemini"),
        },
        ["0", "/quit"],
    )

    assert app.run() == 0

    output = console.export_text()
    assert "Invalid provider selection." in output
    assert store.list_sessions() == []


def test_switch_provider_starts_a_new_session_for_the_new_provider(tmp_path, store):
    app, console = build_app(
        tmp_path,
        store,
        {
            "openai": lambda: RecordingProvider(name="openai"),
            "gemini": lambda: RecordingProvider(name="gemini"),
        },
        ["1", "/switch", "2", "/quit"],
    )

    assert app.run() == 0

    sessions = store.list_sessions()
    assert len(sessions) == 2
    assert {session.provider for session in sessions} == {"openai", "gemini"}
    assert "Provider switched" in console.export_text()


def test_cancelling_switch_keeps_the_current_session(tmp_path, store):
    app, console = build_app(
        tmp_path,
        store,
        {
            "openai": lambda: RecordingProvider(name="openai"),
            "gemini": lambda: RecordingProvider(name="gemini"),
        },
        ["1", "/switch", "q", "/quit"],
    )

    assert app.run() == 0

    assert len(store.list_sessions()) == 1
    assert "Provider switched" not in console.export_text()


def test_load_command_replaces_the_active_history(tmp_path, store):
    saved = store.create_session("openai", "recording-model")
    store.add_turn(saved.id, "Remember this.", "Saved answer.")
    providers: list[RecordingProvider] = []

    def factory() -> RecordingProvider:
        provider = RecordingProvider()
        providers.append(provider)
        return provider

    app, console = build_app(
        tmp_path,
        store,
        {"openai": factory},
        [f"/load {saved.id}", "What did I say?", "/quit"],
    )

    assert app.run() == 0

    assert providers[1].histories[0] == [
        ChatMessage(role="user", content="Remember this."),
        ChatMessage(role="assistant", content="Saved answer."),
    ]
    assert "Loaded: Remember this." in console.export_text()


def test_provider_failure_is_reported_and_not_saved(tmp_path, store):
    app, console = build_app(
        tmp_path,
        store,
        {"openai": lambda: RecordingProvider(error=ProviderError("upstream down"))},
        ["hello", "/quit"],
    )

    assert app.run() == 0

    output = console.export_text()
    assert "Provider error:" in output
    assert "upstream down" in output
    session = store.list_sessions()[0]
    assert store.get_messages(session.id) == []


def test_keyboard_interrupt_during_a_request_cancels_only_that_turn(tmp_path, store):
    app, console = build_app(
        tmp_path,
        store,
        {"openai": lambda: RecordingProvider(error=KeyboardInterrupt())},
        ["hello", "/quit"],
    )

    assert app.run() == 0

    assert "Request cancelled." in console.export_text()
    session = store.list_sessions()[0]
    assert store.get_messages(session.id) == []


def test_no_configured_providers_returns_two(tmp_path, store):
    app, console = build_app(tmp_path, store, {}, ["/quit"])

    assert app.run() == 2
    assert "No provider is configured." in console.export_text()


def test_startup_with_missing_session_returns_two(tmp_path, store):
    app, console = build_app(tmp_path, store, {"openai": RecordingProvider}, [])

    assert app.run(session_id="missing") == 2
    assert "No chat session found" in console.export_text()


def test_startup_session_with_unconfigured_provider_returns_two(tmp_path, store):
    session = store.create_session("gemini", "gemini-test")
    app, console = build_app(tmp_path, store, {"openai": RecordingProvider}, [])

    assert app.run(session_id=session.id) == 2
    assert "not configured" in console.export_text()


def test_unknown_command_is_reported(tmp_path, store):
    app, console = build_app(
        tmp_path,
        store,
        {"openai": RecordingProvider},
        ["/unknown", "/quit"],
    )

    assert app.run() == 0
    assert "Unknown command /unknown" in console.export_text()


def test_empty_prompts_are_ignored(tmp_path, store):
    provider = RecordingProvider()
    app, _ = build_app(
        tmp_path,
        store,
        {"openai": lambda: provider},
        ["", "   ", "real prompt", "/quit"],
    )

    assert app.run() == 0
    assert provider.prompts == ["real prompt"]
