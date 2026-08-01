"""
Toy-data unit tests for the Phase 2 adapter interface (LAUNCH_ROADMAP.md
Phase 2, Task 9): exercise AgentMemAdapter and RecentOnlyAdapter against a
tiny synthetic conversation before wiring either into the real
LoCoMo/LongMemEval harness (real_baseline_eval.py, not built yet).

Also regression-covers the two agent_id-scoping bugs found while building
this adapter (see benchmarks/adapters/agentmem_adapter.py's module
docstring and agentmem_os_known_issues.md #9/#10 in project memory):
  - KG/procedural memory is scoped by agent_id, not session_id — passing
    agent_id=None pools every namespace together.
  - kg_nodes.agent_id has a FK to agent_namespaces.agent_id — a real
    namespace string needs a matching row or every insert silently fails.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

from benchmarks.adapters.registry import get_adapter, ADAPTERS

TOY_TURNS_A = [
    {"role": "user", "content": "My name is Priya and I live in Austin."},
    {"role": "assistant", "content": "Nice to meet you, Priya."},
    {"role": "user", "content": "I work as a data engineer at a startup called Fernway."},
    {"role": "assistant", "content": "Got it — data engineer at Fernway."},
    {"role": "user", "content": "My favorite hobby is rock climbing on weekends."},
    {"role": "assistant", "content": "Rock climbing sounds fun."},
]

TOY_TURNS_B = [
    {"role": "user", "content": "My name is Derek and I live in Seattle."},
    {"role": "assistant", "content": "Nice to meet you, Derek."},
    {"role": "user", "content": "I work as a chef at a restaurant called Ember."},
    {"role": "assistant", "content": "Got it — chef at Ember."},
]


@pytest.mark.parametrize("adapter_name", sorted(ADAPTERS))
def test_adapter_ingest_and_retrieve_roundtrip(adapter_name):
    adapter = get_adapter(adapter_name)
    adapter.setup()
    try:
        adapter.reset("toy-ns-a")
        adapter.ingest_session("toy-ns-a", "sess-1", TOY_TURNS_A)
        results = adapter.retrieve("toy-ns-a", "Where does Priya live?", top_k=10)
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)
    finally:
        adapter.teardown()


def test_agentmem_adapter_isolates_namespaces_no_cross_contamination():
    """
    Regression test for the agent_id=None contamination bug: two distinct
    namespaces ingested into the same adapter instance must not leak
    entities into each other's retrieval.
    """
    adapter = get_adapter("agentmem_os")
    adapter.setup()
    try:
        adapter.reset("toy-ns-a")
        adapter.ingest_session("toy-ns-a", "sess-1", TOY_TURNS_A)

        adapter.reset("toy-ns-b")
        adapter.ingest_session("toy-ns-b", "sess-1", TOY_TURNS_B)

        ctx_a = " ".join(adapter.retrieve("toy-ns-a", "Tell me about this person.", top_k=10))
        ctx_b = " ".join(adapter.retrieve("toy-ns-b", "Tell me about this person.", top_k=10))

        assert "Derek" not in ctx_a and "Ember" not in ctx_a
        assert "Priya" not in ctx_b and "Fernway" not in ctx_b
    finally:
        adapter.teardown()


def test_recent_only_adapter_is_a_floor_not_a_ceiling():
    """
    Sanity check on the harness itself (per recent_only_adapter.py's own
    docstring): if RECENT_ONLY ever beats a real system on retrieval
    relevance, something in the harness is broken, not the memory system.
    Here we just confirm it returns bounded, literal recent turns with no
    query-relevance ranking — i.e. it can't accidentally look smart.
    """
    adapter = get_adapter("recent_only")
    adapter.setup()
    adapter.reset("toy-ns")
    adapter.ingest_session("toy-ns", "sess-1", TOY_TURNS_A)
    results = adapter.retrieve("toy-ns", "completely unrelated query about tax law", top_k=3)
    assert results == [t["content"] for t in TOY_TURNS_A][-3:]
