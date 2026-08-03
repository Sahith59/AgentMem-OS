"""
AgentMem OS — Context Assembler (v2)
======================================
Upgraded from v1 placeholder to full 4-tier context assembly.

Budget allocation (60% of model window → leaves 40% for response):
  ┌─────────────────────────────────────────────────────────┐
  │  5%  System Prompt        (Tier 0 — instructions)       │
  │ 40%  Recent Turns         (Tier 2 — episodic)           │
  │ 25%  Branch Snapshot      (inherited context)            │
  │ 20%  Semantic Retrieval   (Tier 3 — ChromaDB MMR)       │
  │  7%  Global Map           (Tier 3 — Entity KG) ← NEW    │
  │  3%  Behavioral Patterns  (Tier 4 — Procedural) ← NEW   │
  └─────────────────────────────────────────────────────────┘

v1 had "global" as a 10% placeholder that was always skipped.
v2 replaces it with EntityKnowledgeGraph (7%) + ProceduralMemory (3%).
"""

from loguru import logger
from agentmem_os.llm.token_counter import TokenCounter


class ContextAssembler:
    """
    Assembles the full context string for an LLM call, strictly respecting
    per-section token budgets.

    All 4 memory tiers are queried. The result is a single structured string
    passed as the system message to the LLM.
    """

    def __init__(self, model_window: int = 128_000):
        self.model_window = model_window
        self.budget = int(model_window * 0.60)  # 60% for context, 40% for response
        self.allocations = {
            "system":    int(self.budget * 0.05),
            "recent":    int(self.budget * 0.40),
            "snapshot":  int(self.budget * 0.25),
            "semantic":  int(self.budget * 0.20),
            "global":    int(self.budget * 0.07),   # Entity KG (was 10% placeholder)
            "procedural":int(self.budget * 0.03),   # Procedural memory (was 0)
        }
        self.counter = TokenCounter()

        # Lazy-initialized to avoid circular imports at startup
        self._store = None
        self._chroma = None
        self._kg = None
        self._procedural = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def assemble(
        self,
        session_id: str,
        query: str,
        system_prompt: str = "You are an AI assistant with persistent memory, powered by AgentMem OS.",
        agent_id: str = None,
        disable: frozenset = frozenset(),
    ) -> str:
        """
        Build the full context string for the given session and query.

        Each section is capped at its token budget.
        Sections are labelled with XML-style tags for easy parsing in evaluations.

        disable: optional set of tier names to skip — {"semantic", "global",
        "procedural"}. Exists so ablation studies can exercise this real
        assembler directly instead of a hand-rolled simulation of it — see
        benchmarks/ablation_study_real.py. Defaults to empty (all tiers on),
        so this has no effect on any existing caller.
        """
        store = self._get_store()
        session = store.get_or_create_session(session_id)

        sections = []

        # ── Section 1: System Prompt ─────────────────────────────────────────
        sys_section = self._fit_to_budget(
            system_prompt, self.allocations["system"], "[SYSTEM]"
        )
        sections.append(sys_section)

        # ── Section 2: Branch Snapshot (inherited parent context) ────────────
        if session.inherited_context:
            snap_section = self._fit_to_budget(
                session.inherited_context,
                self.allocations["snapshot"],
                "[INHERITED CONTEXT]"
            )
            sections.append(snap_section)

        # ── Section 3: Semantic Memory (ChromaDB MMR retrieval) ──────────────
        if "semantic" not in disable:
            try:
                chroma = self._get_chroma()
                # Budget-aware retrieval depth. top_k=5 was a hardcoded
                # constant that left the 20% semantic budget ~99% unused on
                # long histories (measured live on LoCoMo: ~470 of 76,800
                # budgeted tokens used, gold evidence in 0/30 contexts).
                # Fetch enough to fill the allocation and let the budget cap
                # trim — head-keeping, so the best-ranked chunks survive.
                # ChromaManager.search clamps to collection size, so a large
                # top_k is safe for small live sessions.
                approx_chunk_tokens = 60  # short conversational turns dominate
                top_k = max(5, min(200, self.allocations["semantic"] // approx_chunk_tokens))
                chunks = chroma.search(session_id, query, top_k=top_k)
                if chunks:
                    sem_text = "\n---\n".join(chunks)
                    sem_section = self._fit_to_budget(
                        sem_text, self.allocations["semantic"], "[SEMANTIC MEMORY]",
                        keep="head",
                    )
                    sections.append(sem_section)
            except Exception as e:
                logger.debug(f"[ContextAssembler] Semantic retrieval skipped: {e}")

        # ── Section 4: Global Map (Entity Knowledge Graph) ───────────────────
        if "global" not in disable:
            try:
                kg = self._get_kg()
                world_model = kg.get_relevant_subgraph(
                    query=query,
                    agent_id=agent_id,
                    top_k=12,
                )
                if world_model:
                    kg_section = self._fit_to_budget(
                        world_model, self.allocations["global"], "[WORLD MODEL]"
                    )
                    sections.append(kg_section)
            except Exception as e:
                logger.debug(f"[ContextAssembler] Knowledge graph skipped: {e}")

        # ── Section 5: Procedural Memory (Behavioral Patterns) ───────────────
        if "procedural" not in disable:
            try:
                pm = self._get_procedural()
                patterns = pm.get_relevant_patterns(query, agent_id=agent_id, top_k=3)
                if patterns:
                    proc_section = self._fit_to_budget(
                        patterns, self.allocations["procedural"], "[BEHAVIORAL PATTERNS]"
                    )
                    sections.append(proc_section)
            except Exception as e:
                logger.debug(f"[ContextAssembler] Procedural memory skipped: {e}")

        # ── Section 6: Recent Turns (Episodic — always last) ─────────────────
        turns = store.get_history(session_id, last_n=20)

        # Branch inheritance: if this is a new branch with few turns,
        # borrow recent turns from parent session too
        if len(turns) < 15 and session.parent_session_id:
            parent_turns = store.get_history(
                session.parent_session_id, last_n=(15 - len(turns))
            )
            turns = parent_turns + turns

        recent_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in turns
        )
        recent_section = self._fit_to_budget(
            recent_text, self.allocations["recent"], "[RECENT TURNS]"
        )
        sections.append(recent_section)

        # ── Assemble final context ────────────────────────────────────────────
        full_context = "\n\n".join(s for s in sections if s)

        total_tokens = self.counter.count(full_context)
        logger.debug(
            f"[ContextAssembler] session={session_id} | "
            f"total_tokens={total_tokens}/{self.budget} | "
            f"sections={len(sections)}"
        )

        return full_context

    def get_budget_breakdown(self) -> dict:
        """Return token budget per section for debugging and paper evaluations."""
        return {
            "model_window": self.model_window,
            "total_context_budget": self.budget,
            "allocations": self.allocations,
            "utilization_pct": {
                k: f"{(v / self.budget * 100):.1f}%"
                for k, v in self.allocations.items()
            }
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_to_budget(self, text: str, token_budget: int, label: str, keep: str = "tail") -> str:
        """
        Truncate text to fit within token_budget.
        Uses character-level proxy (1 token ≈ 4 chars) for fast truncation,
        then verifies with tiktoken.

        keep: which end survives truncation. "tail" (default) suits
        chronological content — most recent last. "head" suits RANKED
        content (semantic retrieval, best chunk first): tail-keeping there
        would silently drop the best-scoring chunks and keep the worst.
        """
        if not text or not text.strip():
            return ""

        def _cut(t: str, n: int) -> str:
            return t[-n:] if keep == "tail" else t[:n]

        # Fast path: estimate via character count
        char_budget = token_budget * 4
        if len(text) > char_budget:
            text = _cut(text, char_budget)

        # Verify with token counter
        if self.counter.count(text) > token_budget:
            # Binary search for exact fit
            lo, hi = 0, len(text)
            while lo < hi - 10:
                mid = (lo + hi) // 2
                if self.counter.count(_cut(text, mid)) <= token_budget:
                    lo = mid
                else:
                    hi = mid
            text = _cut(text, lo)

        # Wrap with section label
        return f"<{label.strip('<>')}>\n{text.strip()}\n</{label.strip('<>')}>"

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy dependency getters (avoid circular imports)
    # ──────────────────────────────────────────────────────────────────────────

    def _get_store(self):
        if self._store is None:
            from agentmem_os.storage.store import ConversationStore
            self._store = ConversationStore()
        return self._store

    def _get_chroma(self):
        if self._chroma is None:
            from agentmem_os.db.chroma_client import ChromaManager
            self._chroma = ChromaManager()
        return self._chroma

    def _get_kg(self):
        if self._kg is None:
            from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph
            from agentmem_os.db.engine import get_session
            self._kg = EntityKnowledgeGraph(get_session)
        return self._kg

    def _get_procedural(self):
        if self._procedural is None:
            from agentmem_os.llm.procedural_memory import ProceduralMemory
            from agentmem_os.db.engine import get_session
            self._procedural = ProceduralMemory(get_session)
        return self._procedural
