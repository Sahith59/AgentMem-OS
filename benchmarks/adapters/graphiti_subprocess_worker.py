"""
Graphiti (Zep's OSS temporal-KG core) worker — runs inside
benchmarks/adapters/.venv-graphiti, speaks newline-delimited JSON over
stdin/stdout to graphiti_adapter.py in the main venv. See base.py's
SubprocessAdapter for the protocol.

Requires Neo4j running locally (benchmarks/docker-compose.baselines.yml)
and gpt-4o-mini as the extraction LLM (same model used across every
adapter per LAUNCH_ROADMAP.md Phase 2 Task 12).

Uses `group_id` per namespace for multi-tenant isolation (Graphiti's own
mechanism, per Task 17) rather than one Neo4j database per namespace —
one running Graphiti/driver instance is reused across namespaces.

Cost note: every ingest_session call makes real OpenAI calls for
Graphiti's own LLM-based entity/edge extraction — this worker is $0 to
build/import/start, but NOT $0 to actually ingest real conversation turns
through.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "benchmarkpassword")

_graphiti: Graphiti | None = None
_indices_built = False
_reset_group_ids: set = set()


def _get_graphiti() -> Graphiti:
    global _graphiti
    if _graphiti is None:
        llm_config = LLMConfig(model="gpt-4o-mini", small_model="gpt-4o-mini")
        _graphiti = Graphiti(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD,
            llm_client=OpenAIClient(config=llm_config),
            embedder=OpenAIEmbedder(config=OpenAIEmbedderConfig(embedding_model="text-embedding-3-small")),
        )
    return _graphiti


async def _ensure_indices() -> None:
    global _indices_built
    if not _indices_built:
        await _get_graphiti().build_indices_and_constraints()
        _indices_built = True


async def _reset(namespace: str) -> None:
    """
    Graphiti has no per-group_id wipe primitive — delete every node/edge
    tagged with this group_id directly via the underlying driver, so a
    namespace can be re-run cleanly without wiping every other namespace's
    already-ingested graph data.
    """
    g = _get_graphiti()
    await _ensure_indices()
    await g.driver.execute_query(
        "MATCH (n {group_id: $group_id}) DETACH DELETE n",
        group_id=namespace,
    )
    _reset_group_ids.add(namespace)


async def _ingest_session(namespace: str, turns: list) -> None:
    g = _get_graphiti()
    await _ensure_indices()
    now = datetime.now(timezone.utc)
    for i, turn in enumerate(turns):
        content = turn.get("content", "")
        if not content:
            continue
        await g.add_episode(
            name=f"{namespace}-turn-{i}",
            episode_body=content,
            source_description=f"{turn.get('role', 'user')} turn",
            reference_time=now,
            source=EpisodeType.message,
            group_id=namespace,
        )


async def _retrieve(namespace: str, query: str, top_k: int) -> list:
    g = _get_graphiti()
    edges = await g.search(query, group_ids=[namespace], num_results=top_k)
    return [e.fact for e in edges if getattr(e, "fact", None)][:top_k]


def handle(req: dict) -> dict:
    op = req.get("op")
    loop = asyncio.get_event_loop()

    if op == "ping":
        return {"ok": True}

    if op == "reset":
        loop.run_until_complete(_reset(req["namespace"]))
        return {"ok": True}

    if op == "ingest_session":
        loop.run_until_complete(_ingest_session(req["namespace"], req["turns"]))
        return {"ok": True}

    if op == "retrieve":
        result = loop.run_until_complete(
            _retrieve(req["namespace"], req["query"], req.get("top_k", 10))
        )
        return {"ok": True, "result": result}

    if op == "shutdown":
        if _graphiti is not None:
            loop.run_until_complete(_graphiti.close())
        return {"ok": True}

    return {"ok": False, "error": f"unknown op '{op}'"}


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        try:
            resp = handle(req)
        except Exception as e:
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
        if req.get("op") == "shutdown":
            break


if __name__ == "__main__":
    main()
