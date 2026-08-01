"""
Trivial "no memory system" floor baseline — ingests turns into an
in-memory list, retrieval just returns the most recent N regardless of
query relevance. Costs nothing, needs no external service.

Exists as a sanity check on the harness itself: if RECENT_ONLY ever beats
a real memory system on QA accuracy, something in the harness (not the
memory system) is broken.
"""
from __future__ import annotations

from benchmarks.adapters.base import MemoryAdapter


class RecentOnlyAdapter(MemoryAdapter):
    name = "recent_only"

    def __init__(self, window: int = 10):
        self.window = window
        self._store: dict = {}

    def setup(self) -> None:
        pass

    def reset(self, namespace: str) -> None:
        self._store[namespace] = []

    def ingest_session(self, namespace: str, session_id: str, turns: list) -> None:
        self._store.setdefault(namespace, [])
        for t in turns:
            content = t.get("content", "")
            if content:
                self._store[namespace].append(content)

    def retrieve(self, namespace: str, query: str, top_k: int = 10) -> list:
        turns = self._store.get(namespace, [])
        return turns[-min(top_k, self.window):]
