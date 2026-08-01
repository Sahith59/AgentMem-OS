"""
AgentMem OS adapter — the one with zero new dependencies, since it's the
real package this whole repo already is. Reuses the same synchronous
ingestion pattern already proven in ablation_study_real.py and
qa_accuracy_eval.py (bypasses save_turn()'s daemon-thread KG ingestion for
deterministic benchmark results) and the same TF-IDF-in-place-of-ChromaDB
swap (benchmarks/real_code_utils.py) so this runs with zero external
services.

Important scoping note, found while building this adapter: the entity
knowledge graph (EntityKnowledgeGraph.get_relevant_subgraph) and
procedural memory (ProceduralMemory.get_relevant_patterns) are both
scoped by `agent_id`, NOT `session_id` — by design, so a single agent's
world-model accumulates across all of its sessions in production. Passing
agent_id=None everywhere (as ablation_study_real.py and
qa_accuracy_eval.py both do) means every namespace would share one global
KG/procedural pool across an entire multi-namespace benchmark run,
contaminating retrieval with entities from unrelated namespaces. This
adapter passes `namespace` as `agent_id` consistently everywhere, so each
namespace gets its own isolated KG and procedural-pattern pool, matching
the isolation boundary the whole harness is built around.

Second bug found while fixing the first one: `kg_nodes.agent_id` has a
FOREIGN KEY to `agent_namespaces.agent_id` (nullable, which is exactly
why agent_id=None silently "worked" everywhere before — NULL bypasses FK
checks in SQLite). Passing a made-up agent_id string without a matching
`agent_namespaces` row makes every single KG insert fail with
IntegrityError, silently swallowed by store._ingest_kg()'s except clause
— confirmed 100% reproducible, zero nodes ever persisted, entirely
independent of the StaticPool/session issue documented elsewhere
(agentmem_os_known_issues.md) as a red herring I initially suspected.
Fix: ensure_agent_exists(namespace) before any ingestion.
"""
from __future__ import annotations

from benchmarks.adapters.base import MemoryAdapter
from benchmarks.real_code_utils import install_tfidf_chroma


class AgentMemAdapter(MemoryAdapter):
    name = "agentmem_os"

    def __init__(self):
        self._store = None
        self._assembler = None

    def setup(self) -> None:
        from agentmem_os.storage.store import ConversationStore
        from agentmem_os.llm.context_assembler import ContextAssembler
        from agentmem_os.agents.namespace_manager import AgentNamespaceManager
        from agentmem_os.db.engine import get_session

        install_tfidf_chroma(ContextAssembler)
        self._store = ConversationStore()
        self._assembler = ContextAssembler()
        self._namespace_mgr = AgentNamespaceManager(get_session)

    def reset(self, namespace: str) -> None:
        self._store.delete_session(namespace)
        self._store.get_or_create_session(namespace, name="real_baseline_eval")
        # Required before any KG ingestion — kg_nodes.agent_id has a FK to
        # agent_namespaces.agent_id (nullable, so agent_id=None bypasses
        # this, but a real per-namespace agent_id needs a matching row or
        # every insert fails with IntegrityError). Idempotent.
        self._namespace_mgr.ensure_agent_exists(namespace)

    def ingest_session(self, namespace: str, session_id: str, turns: list) -> None:
        from agentmem_os.db.models import Turn

        for turn in turns:
            content = turn.get("content", "")
            if not content:
                continue
            tokens = self._store.token_counter.count(content)
            t = Turn(session_id=namespace, role=turn.get("role", "user"),
                      content=content, token_count=tokens)
            self._store.db.add(t)
        self._store.db.commit()
        try:
            for turn in turns:
                if turn.get("content"):
                    # agent_id=namespace — see module docstring on KG/
                    # procedural scoping.
                    self._store._ingest_kg(namespace, namespace, turn["content"])
        except Exception:
            pass  # KG ingestion is best-effort — retrieval still works via
                   # the semantic/recent tiers if entity extraction fails
                   # on a particular turn's content.

    def retrieve(self, namespace: str, query: str, top_k: int = 10) -> list:
        ctx = self._assembler.assemble(namespace, query, agent_id=namespace)
        # ContextAssembler wraps each tier in <[LABEL]>...</[LABEL]> tags —
        # split into per-section chunks rather than returning one giant
        # string, so top_k has the same meaning as it does for every other
        # adapter (a ranked list of discrete memory units).
        import re
        sections = re.split(r"<\[[A-Z ]+\]>|\</\[[A-Z ]+\]>", ctx)
        chunks = [s.strip() for s in sections if s.strip()]
        return chunks[:top_k]
