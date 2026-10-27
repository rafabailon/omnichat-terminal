from __future__ import annotations

import os
import sqlite3
import stat
from pathlib import Path

import pytest

from database import SCHEMA_VERSION, ChatStore


def test_sessions_store_ordered_turns_and_update_title(store: ChatStore):
    session = store.create_session("openai", "gpt-test")
    store.add_turn(session.id, "  Explain   the tide. ", "The tide is an ocean's rise and fall.")

    loaded = store.get_session(session.id)
    messages = store.get_messages(session.id)

    assert loaded is not None
    assert loaded.title == "Explain the tide."
    assert loaded.provider == "openai"
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "  Explain   the tide. "
    assert messages[1].content == "The tide is an ocean's rise and fall."
    assert store.list_sessions()[0].id == session.id


def test_deleting_a_session_cascades_to_messages(store: ChatStore):
    session = store.create_session("openai", "gpt-test")
    store.add_turn(session.id, "hello", "A real provider response")

    assert store.delete_session(session.id)
    assert store.get_session(session.id) is None
    assert store.get_messages(session.id) == []
    assert not store.delete_session(session.id)


def test_adding_turn_to_missing_session_fails_without_creating_history(store: ChatStore):
    with pytest.raises(KeyError, match="does not exist"):
        store.add_turn("missing", "question", "answer")

    assert store.list_sessions() == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permission bits are not portable.")
def test_database_file_is_private_to_the_current_user(store: ChatStore):
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600


def test_in_memory_database_is_rejected_as_unsupported():
    with pytest.raises(ValueError, match="persistent SQLite file"):
        ChatStore(Path(":memory:"))


def test_schema_version_is_recorded(tmp_path):
    store = ChatStore(tmp_path / "history.sqlite3")

    with sqlite3.connect(store.path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]

    assert version == SCHEMA_VERSION


def test_a_database_from_a_newer_version_is_rejected(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(RuntimeError, match="newer OmniChat Terminal version"):
        ChatStore(path)


def test_search_messages_matches_across_sessions_in_reverse_order(store: ChatStore):
    first = store.create_session("openai", "gpt-test")
    second = store.create_session("gemini", "gemini-test")
    store.add_turn(first.id, "Tell me about the tide.", "The tide is a rise and fall.")
    store.add_turn(second.id, "What is 100% of 5?", "It is 5. Be _sure_ of that.")

    tide_matches = store.search_messages("TIDE")
    percent_matches = store.search_messages("100%")
    underscore_matches = store.search_messages("_sure_")
    plain_sure_matches = store.search_messages("sure")

    assert [match.content for match in tide_matches] == [
        "The tide is a rise and fall.",
        "Tell me about the tide.",
    ]
    assert tide_matches[0].session_id == first.id
    assert tide_matches[0].session_title == "Tell me about the tide."
    assert tide_matches[0].provider == "openai"
    assert tide_matches[0].role == "assistant"
    assert tide_matches[1].role == "user"
    assert [match.message_id for match in percent_matches] == [store.get_messages(second.id)[0].id]
    assert len(underscore_matches) == 1
    assert len(plain_sure_matches) == 1


def test_search_messages_rejects_blank_queries_and_invalid_limits(store: ChatStore):
    with pytest.raises(ValueError, match="cannot be blank"):
        store.search_messages("   ")
    with pytest.raises(ValueError, match="between 1 and 500"):
        store.search_messages("tide", limit=0)
    with pytest.raises(ValueError, match="between 1 and 500"):
        store.search_messages("tide", limit=501)


def test_search_messages_honours_the_limit(store: ChatStore):
    session = store.create_session("openai", "gpt-test")
    for index in range(3):
        store.add_turn(session.id, f"question {index}", "shared answer")

    assert len(store.search_messages("shared", limit=2)) == 2


def test_session_list_rejects_non_positive_limits(store: ChatStore):
    with pytest.raises(ValueError, match="positive"):
        store.list_sessions(0)
