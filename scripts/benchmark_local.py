#!/usr/bin/env python3
"""Measure local chat-service and SQLite overhead without contacting an AI provider."""

from __future__ import annotations

import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

from chat import ChatService
from database import ChatStore
from models import ChatMessage
from providers import Provider


class LocalBenchmarkProvider(Provider):
    def __init__(self) -> None:
        super().__init__("benchmark", "local-stub")

    def send_message(self, prompt: str, history: Sequence[ChatMessage]) -> str:
        return "Local benchmark response."


def main() -> None:
    samples: list[float] = []
    with tempfile.TemporaryDirectory(prefix="omnichat-benchmark-") as directory:
        store = ChatStore(Path(directory) / "benchmark.sqlite3")
        session = store.create_session("benchmark", "local-stub")
        provider = LocalBenchmarkProvider()
        service = ChatService(store)
        try:
            for index in range(100):
                started = time.perf_counter()
                service.ask(provider, f"Local benchmark prompt {index}", session, [])
                samples.append((time.perf_counter() - started) * 1000)
        finally:
            provider.close()
        ordered = sorted(samples)
        print(
            "100 local service + SQLite turns (stub provider; no network): "
            f"median={statistics.median(samples):.3f}ms "
            f"p95={ordered[94]:.3f}ms"
        )


if __name__ == "__main__":
    main()
