from __future__ import annotations

from pathlib import Path

import pytest

from config import ProviderSettings, Settings
from database import ChatStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    empty = ProviderSettings("", "", "", "")
    return Settings(
        openai=ProviderSettings(
            "openai", "test-openai-key", "test-model", "https://example.test/v1"
        ),
        gemini=ProviderSettings("gemini", "", "gemini-test", "https://example.test/v1beta"),
        anthropic=ProviderSettings("anthropic", "", "claude-test", "https://example.test/v1"),
        compatible=empty,
        database_path=tmp_path / "history.sqlite3",
    )


@pytest.fixture
def store(settings: Settings) -> ChatStore:
    return ChatStore(settings.database_path)
