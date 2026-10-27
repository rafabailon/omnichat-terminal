from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class ChatSession:
    id: str
    title: str
    provider: str
    model: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class StoredMessage:
    id: int
    session_id: str
    role: Role
    content: str
    created_at: datetime


@dataclass(frozen=True)
class MessageMatch:
    """A message returned by a cross-session history search."""

    message_id: int
    session_id: str
    session_title: str
    provider: str
    role: Role
    content: str
    created_at: datetime
