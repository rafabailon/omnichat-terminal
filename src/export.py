from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

from database import ChatStore
from models import ChatSession, StoredMessage


def default_export_filename(session: ChatSession) -> str:
    """Return the default Markdown filename for a conversation."""
    return f"omnichat-{session.id[:8]}.md"


def _format_timestamp(value: datetime) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def render_transcript(session: ChatSession, messages: list[StoredMessage]) -> str:
    """Render a conversation as portable Markdown."""
    lines = [
        f"# {session.title}",
        "",
        f"- **Session ID:** `{session.id}`",
        f"- **Provider:** `{session.provider}`",
        f"- **Model:** `{session.model}`",
        f"- **Created:** {_format_timestamp(session.created_at)}",
        f"- **Updated:** {_format_timestamp(session.updated_at)}",
        "",
        "---",
        "",
    ]
    for message in messages:
        speaker = "You" if message.role == "user" else "Assistant"
        lines.extend([f"## {speaker}", "", message.content, ""])
    return "\n".join(lines).rstrip() + "\n"


def export_session(store: ChatStore, session_id: str, destination: Path) -> Path:
    """Write one saved conversation to ``destination`` and return the final path."""
    session = store.get_session(session_id)
    if session is None:
        raise KeyError(f"Chat session {session_id!r} does not exist.")
    messages = store.get_messages(session_id)

    destination = destination.expanduser()
    if destination.is_dir():
        destination = destination / default_export_filename(session)
    elif destination.suffix == "":
        destination = destination.with_name(f"{destination.name}.md")
    destination.parent.mkdir(parents=True, exist_ok=True)

    content = render_transcript(session, messages)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        if os.name == "posix":
            temporary.chmod(0o600)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
