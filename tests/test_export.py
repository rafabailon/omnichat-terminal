from __future__ import annotations

import os
import stat

import pytest

from database import ChatStore
from export import (
    default_export_filename,
    export_session,
    render_transcript,
)


def test_render_transcript_contains_metadata_and_ordered_messages(store: ChatStore):
    session = store.create_session("openai", "gpt-test")
    store.add_turn(session.id, "What is the tide?", "A rise and fall of the sea.")
    loaded = store.get_session(session.id)
    assert loaded is not None

    transcript = render_transcript(loaded, store.get_messages(session.id))

    assert transcript.startswith("# What is the tide?\n")
    assert f"- **Session ID:** `{session.id}`" in transcript
    assert "- **Provider:** `openai`" in transcript
    assert "- **Model:** `gpt-test`" in transcript
    assert "## You\n\nWhat is the tide?" in transcript
    assert "## Assistant\n\nA rise and fall of the sea." in transcript
    assert transcript.endswith("\n")


def test_render_transcript_handles_an_empty_conversation(store: ChatStore):
    session = store.create_session("openai", "gpt-test")
    loaded = store.get_session(session.id)
    assert loaded is not None

    transcript = render_transcript(loaded, [])

    assert "# New conversation" in transcript
    assert "## You" not in transcript
    assert transcript.endswith("\n")


def test_export_session_writes_a_private_markdown_file(store: ChatStore, tmp_path):
    session = store.create_session("openai", "gpt-test")
    store.add_turn(session.id, "Hello", "Hi there.")
    destination = tmp_path / "exports" / "chat.md"

    result = export_session(store, session.id, destination)

    assert result == destination
    assert destination.read_text(encoding="utf-8").startswith("# Hello\n")
    assert "Hi there." in destination.read_text(encoding="utf-8")
    assert not list(destination.parent.glob("*.tmp"))
    if os.name == "posix":
        assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_export_session_uses_a_directory_and_adds_a_suffix(store: ChatStore, tmp_path):
    session = store.create_session("openai", "gpt-test")

    in_directory = export_session(store, session.id, tmp_path)
    without_suffix = export_session(store, session.id, tmp_path / "transcript")

    assert in_directory == tmp_path / default_export_filename(session)
    assert in_directory.is_file()
    assert without_suffix == tmp_path / "transcript.md"
    assert without_suffix.is_file()


def test_default_export_filename_uses_the_session_prefix(store: ChatStore):
    session = store.create_session("openai", "gpt-test")

    assert default_export_filename(session) == f"omnichat-{session.id[:8]}.md"


def test_export_missing_session_raises_key_error(store: ChatStore, tmp_path):
    with pytest.raises(KeyError, match="does not exist"):
        export_session(store, "missing", tmp_path / "chat.md")
