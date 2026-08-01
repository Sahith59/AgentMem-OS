"""
Letta baseline adapter — real, in-process (main venv; letta-client is a
thin httpx-based client, low collision risk with the main venv's pins).
Talks to a real Letta server (benchmarks/docker-compose.baselines.yml).

Deliberate scoping decision (LAUNCH_ROADMAP.md Phase 2 Task 20): one fresh
agent per namespace, and ingest_session writes each turn directly into
ARCHIVAL memory (agents.passages.create) — bypassing Letta's own
conversational agent loop entirely. Running the full conversational loop
is technically possible but multiplies cost/time for a benefit ("how good
is the underlying LLM at deciding what to remember mid-conversation") that
is hard to isolate from "how good is Letta's archival retrieval," which is
the thing actually comparable to the other three systems here. This
scoping MUST travel with every result artifact this adapter produces
(adapter_disclosures in real_baseline_eval.py, not built yet) — never
presented as "Letta" unqualified.

Cost note: agent create/delete/list are free (calls to our own local
Docker container) — default model/embedding is "letta/letta-free", the
only handle the stock server exposes with zero provider configuration
(confirmed: "openai/gpt-4o-mini" 404s with "must be one of []" until
OPENAI_API_KEY is set server-side). ingest_session's passages.create still
makes a real embedding-provider call per turn regardless of which handle
is configured (confirmed via a 404 from OpenAI's own embeddings endpoint
when no key is set even under letta/letta-free) — for a real, comparable-
to-the-other-three-systems run, pass model="openai/gpt-4o-mini",
embedding="openai/text-embedding-3-small" and set OPENAI_API_KEY on the
letta service in docker-compose.baselines.yml. This adapter is $0 to
build/import/create-agents-with at its defaults, but NOT $0 to actually
ingest real conversation turns through either way.
"""
from __future__ import annotations

import os

from benchmarks.adapters.base import MemoryAdapter

_AGENT_NAME_PREFIX = "agentmem-bench-"


class LettaAdapter(MemoryAdapter):
    name = "letta"

    def __init__(self, base_url: str | None = None,
                 model: str = "letta/letta-free",
                 embedding: str = "letta/letta-free"):
        self._base_url = base_url or os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
        self._model = model
        self._embedding = embedding
        self._client = None
        self._agent_ids: dict = {}  # namespace -> agent_id

    def setup(self) -> None:
        from letta_client import Letta
        self._client = Letta(base_url=self._base_url)
        self._client.agents.list(limit=1)  # verify the server is reachable

    def _agent_name(self, namespace: str) -> str:
        return f"{_AGENT_NAME_PREFIX}{namespace}"

    def reset(self, namespace: str) -> None:
        name = self._agent_name(namespace)
        for existing in self._client.agents.list(name=name).items:
            self._client.agents.delete(existing.id)
        agent = self._client.agents.create(
            name=name, model=self._model, embedding=self._embedding,
        )
        self._agent_ids[namespace] = agent.id

    def ingest_session(self, namespace: str, session_id: str, turns: list) -> None:
        agent_id = self._agent_ids[namespace]
        for turn in turns:
            content = turn.get("content", "")
            if content:
                self._client.agents.passages.create(agent_id, text=content)

    def retrieve(self, namespace: str, query: str, top_k: int = 10) -> list:
        agent_id = self._agent_ids[namespace]
        resp = self._client.agents.passages.search(agent_id, query=query, top_k=top_k)
        return [r.content for r in resp.results if r.content][:top_k]

    def teardown(self) -> None:
        for agent_id in self._agent_ids.values():
            try:
                self._client.agents.delete(agent_id)
            except Exception:
                pass
        self._agent_ids.clear()
