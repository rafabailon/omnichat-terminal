from __future__ import annotations

import os
import re

import pytest

from chat import ChatService
from config import Settings
from database import ChatStore
from registry import configured_providers

LIVE_ENABLED = os.getenv("OMNICHAT_LIVE_TEST", "").strip() == "1"
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not LIVE_ENABLED,
        reason="Set OMNICHAT_LIVE_TEST=1 to make a real, potentially billable model request.",
    ),
]


def test_generic_question_receives_a_real_provider_answer(tmp_path):
    """Ask a generic arithmetic question and assert the configured model actually answers it."""
    settings = Settings.from_environment()
    provider_name = os.getenv("OMNICHAT_TEST_PROVIDER", "openai").strip().lower()
    factories = configured_providers(settings)
    if provider_name not in factories:
        pytest.fail(
            f"Live test requested {provider_name!r}, but it is not configured. "
            "Set its API key and model in .env."
        )

    provider = factories[provider_name]()
    store = ChatStore(tmp_path / "live-test.sqlite3")
    session = store.create_session(provider.name, provider.model)
    try:
        answer = ChatService(store).ask(
            provider,
            "What is 6 multiplied by 7? Reply with the number only.",
            session,
            history=[],
        )
    finally:
        provider.close()

    assert answer.strip()
    assert re.search(r"\b42\b", answer), (
        f"{provider_name} returned non-empty text but did not answer correctly: {answer!r}"
    )
    saved_messages = store.get_messages(session.id)
    assert [message.role for message in saved_messages] == ["user", "assistant"]
    assert saved_messages[1].content == answer
    print(f"Generic prompt answered by real {provider_name} model: {answer}")
