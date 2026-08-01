"""
LangMem worker — runs inside benchmarks/adapters/.venv-langmem
(bootstrapped by setup_venvs.sh), speaks newline-delimited JSON over
stdin/stdout to langmem_adapter.py in the main venv. See base.py's
SubprocessAdapter for the protocol.

Uses create_memory_store_manager() for real LLM-based extraction per
LAUNCH_ROADMAP.md Phase 2 Task 22 — deliberately not a bare store.put()
call, which would be lower-friction but unfaithful to what the library
actually does (LangMem's whole pitch is the extraction step, not the
storage primitive underneath it). gpt-4o-mini as the extraction model,
matching every other adapter.

One fresh in-memory store per namespace (mirrors the per-namespace
isolation pattern in mem0_subprocess_worker.py) rather than one shared
store with LangMem's own namespace-tuple multi-tenancy — simpler to
reason about for a benchmark harness that resets namespaces independently.

Cost note: every ingest_session call makes real OpenAI calls for LangMem's
own memory extraction, and retrieve's semantic search needs a real
embedding call too — this worker is $0 to build/import/start, but NOT $0
to actually ingest/retrieve real conversation turns through.
"""
from __future__ import annotations

import json
import sys

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager

_NAMESPACE_PREFIX = ("memories",)
_managers: dict = {}  # namespace -> (manager, store)


def _get_manager(namespace: str):
    if namespace not in _managers:
        store = InMemoryStore(index={
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
            "fields": ["content"],
        })
        manager = create_memory_store_manager(
            "openai:gpt-4o-mini",
            namespace=(*_NAMESPACE_PREFIX, namespace),
            store=store,
        )
        _managers[namespace] = (manager, store)
    return _managers[namespace]


def _to_lc_message(turn: dict):
    content = turn.get("content", "")
    role = turn.get("role", "user")
    return AIMessage(content=content) if role == "assistant" else HumanMessage(content=content)


def handle(req: dict) -> dict:
    op = req.get("op")

    if op == "ping":
        return {"ok": True}

    if op == "reset":
        namespace = req["namespace"]
        _managers.pop(namespace, None)
        _get_manager(namespace)
        return {"ok": True}

    if op == "ingest_session":
        namespace = req["namespace"]
        turns = req["turns"]
        manager, _ = _get_manager(namespace)
        messages = [_to_lc_message(t) for t in turns if t.get("content")]
        if messages:
            manager.invoke({"messages": messages})
        return {"ok": True}

    if op == "retrieve":
        namespace = req["namespace"]
        query = req["query"]
        top_k = req.get("top_k", 10)
        manager, _ = _get_manager(namespace)
        items = manager.search(query=query, limit=top_k)
        texts = [item.value.get("content", str(item.value)) for item in items]
        return {"ok": True, "result": texts[:top_k]}

    if op == "shutdown":
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
