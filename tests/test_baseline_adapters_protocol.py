"""
Protocol-only smoke tests for the subprocess-isolated competitor adapters
(mem0, graphiti, ... as they're added — LAUNCH_ROADMAP.md Phase 2).

Deliberately calls ONLY setup()/reset()/teardown() — never
ingest_session()/retrieve() — because those two make real LLM/embedding
API calls for each system's own extraction (mem0, graphiti both need a
real OPENAI_API_KEY and cost real money; that's the whole point of
integrating the real libraries rather than simulating them). This file
verifies the worker-subprocess wiring (venv exists, worker imports
cleanly, speaks the JSON protocol, can talk to its backing service) stays
$0 and needs no real API key — a dummy key is enough for client-object
construction in both libraries, confirmed by hand before writing this.

Each test skips (not fails) when its prerequisite isn't available, since
these need local infra a plain `pytest tests/` run won't always have:
  - mem0: benchmarks/adapters/.venv-mem0 (bootstrap: setup_venvs.sh)
  - graphiti: .venv-graphiti + Neo4j reachable at bolt://localhost:7687
    (bootstrap: docker compose -f benchmarks/docker-compose.baselines.yml up -d)
"""
import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-for-construction-test-only")

ADAPTERS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "adapters"


def _neo4j_reachable() -> bool:
    try:
        with socket.create_connection(("localhost", 7687), timeout=1):
            return True
    except OSError:
        return False


@pytest.mark.skipif(not (ADAPTERS_DIR / ".venv-mem0").exists(),
                     reason="benchmarks/adapters/.venv-mem0 not bootstrapped — run setup_venvs.sh")
def test_mem0_adapter_protocol_smoke():
    from benchmarks.adapters.registry import get_adapter
    adapter = get_adapter("mem0")
    adapter.setup()
    try:
        adapter.reset("pytest-smoke-ns")
    finally:
        adapter.teardown()


@pytest.mark.skipif(not (ADAPTERS_DIR / ".venv-graphiti").exists(),
                     reason="benchmarks/adapters/.venv-graphiti not bootstrapped — run setup_venvs.sh")
@pytest.mark.skipif(not _neo4j_reachable(),
                     reason="Neo4j not reachable at bolt://localhost:7687 — "
                            "run docker compose -f benchmarks/docker-compose.baselines.yml up -d")
def test_graphiti_adapter_protocol_smoke():
    from benchmarks.adapters.registry import get_adapter
    adapter = get_adapter("graphiti")
    adapter.setup()
    try:
        adapter.reset("pytest-smoke-ns")
    finally:
        adapter.teardown()
