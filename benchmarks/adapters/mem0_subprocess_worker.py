"""
Mem0 worker — runs inside benchmarks/adapters/.venv-mem0 (bootstrapped by
setup_venvs.sh), speaks newline-delimited JSON over stdin/stdout to
mem0_adapter.py in the main venv. See base.py's SubprocessAdapter for the
protocol.

Configured fully self-hosted/local per LAUNCH_ROADMAP.md Phase 2 Task 12:
local Chroma persisted per-namespace under a temp directory, gpt-4o-mini as
the extraction LLM (the same model used across every adapter so the
comparison isn't confounded by different extraction-model quality).

Cost note: every ingest_session call makes real OpenAI calls for Mem0's own
fact extraction — this worker is $0 to build/import/start, but NOT $0 to
actually ingest real conversation turns through.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

os.environ.setdefault("MEM0_TELEMETRY", "False")  # no PostHog calls from a benchmark run

from mem0 import Memory

_BASE_DIR = tempfile.mkdtemp(prefix="agentmem_bench_mem0_")
_memories: dict = {}  # namespace -> Memory instance


def _config_for(namespace: str) -> dict:
    persist_dir = os.path.join(_BASE_DIR, namespace)
    os.makedirs(persist_dir, exist_ok=True)
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {"path": persist_dir, "collection_name": "mem0_bench"},
        },
        "llm": {
            "provider": "openai",
            "config": {"model": "gpt-4o-mini", "temperature": 0.0},
        },
        "embedder": {
            "provider": "openai",
            "config": {"model": "text-embedding-3-small"},
        },
    }


def _get_memory(namespace: str) -> Memory:
    if namespace not in _memories:
        _memories[namespace] = Memory.from_config(_config_for(namespace))
    return _memories[namespace]


def handle(req: dict) -> dict:
    op = req.get("op")

    if op == "ping":
        return {"ok": True}

    if op == "reset":
        namespace = req["namespace"]
        persist_dir = os.path.join(_BASE_DIR, namespace)
        shutil.rmtree(persist_dir, ignore_errors=True)
        _memories.pop(namespace, None)
        _get_memory(namespace)
        return {"ok": True}

    if op == "ingest_session":
        namespace = req["namespace"]
        turns = req["turns"]
        mem = _get_memory(namespace)
        messages = [{"role": t.get("role", "user"), "content": t.get("content", "")}
                    for t in turns if t.get("content")]
        if messages:
            mem.add(messages, user_id=namespace)
        return {"ok": True}

    if op == "retrieve":
        namespace = req["namespace"]
        query = req["query"]
        top_k = req.get("top_k", 10)
        mem = _get_memory(namespace)
        # mem0 2.x rejects user_id as a top-level search() kwarg — must go
        # through filters. top_k (not limit) caps result count.
        result = mem.search(query, filters={"user_id": namespace}, top_k=top_k)
        hits = result.get("results", result) if isinstance(result, dict) else result
        texts = [h.get("memory", "") for h in hits if h.get("memory")]
        return {"ok": True, "result": texts[:top_k]}

    if op == "shutdown":
        shutil.rmtree(_BASE_DIR, ignore_errors=True)
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
