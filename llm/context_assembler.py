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

Stage 5 (Consolidation v2): the 20% semantic allocation is now
facts-FIRST — dated atomic facts distilled by ConsolidationV2 claim the
budget first ([SEMANTIC FACTS]), raw-turn chunk retrieval fills the
remainder ([SEMANTIC MEMORY]) as provenance/fallback. With an empty
fact store the output is byte-identical to the pre-facts assembler.
"""

import os
import re

from loguru import logger
from agentmem_os.llm.token_counter import TokenCounter

# Share of the semantic allocation the facts tier may claim. The rest
# is RESERVED for raw-turn evidence (Gate C: facts at 100% evicted the
# fallback and the score stayed exactly at baseline).
#
# 0.65 -> 0.35, 2026-08-11, MEASURED (DECISION_AND_FAILURE_LOG §3.1z).
# 0.65 was chosen to stop facts starving raw turns ENTIRELY; it was never
# re-validated once the fact tier actually had a full corpus behind it.
#
# The metric it is now tuned against is GOLD-SESSION COVERAGE — how often
# EVERY session holding a question's evidence reaches the context — which
# is a $0 proxy with a measured relationship to accuracy:
#     ALL gold sessions present -> 84.5% correct  (ceiling 86.7%)
#     PARTIAL / NONE            -> ~44% correct
# Swept through THIS assembler on the real corpus, same 4,740-token
# budget, n=150:
#     facts 65% -> 103 ALL   (multi-session 20/39, temporal 17/40)
#     facts 50% -> 108
#     facts 35% -> 116 ALL   (multi-session 25/39, temporal 25/40)
#     facts 20% -> 117
#     facts 10% -> 120
#     facts  0% -> 122
# Coverage rises monotonically as facts yield budget, because multi-hop
# questions need 2-5 DIFFERENT gold sessions and raw turns are what carry
# them.
#
# 0.35 and NOT the coverage-maximising 0.0-0.10 deliberately: facts
# demonstrably earn their place (multi-session 18/39 -> 22/39 when the
# tier was added, §3.1x), so this takes the setting that restores
# coverage while keeping the tier meaningful, rather than the one that
# maximises the proxy. Optimising a proxy to its extreme is how a metric
# stops describing the product.
FACTS_BUDGET_SHARE = 0.35
# The profile gets its OWN slice of the semantic allocation, taken
# before facts and chunks divide the rest. It is small by design (who
# the user IS compresses to a few dozen lines) and it must never
# compete with the other tiers for space — the Gate C starvation
# lesson applies to the new tier first.
PROFILE_BUDGET_SHARE = 0.15

# Questions that ask what was SAID in an earlier conversation, rather than
# what the user is like. On these the user-model tiers are suppressed and
# the raw-turn tier — the only representation holding the assistant's words
# verbatim — receives the whole semantic allocation. See the long note at
# the routing site in assemble() for the measurements behind this.
#
# Deliberately conservative: it must fire on "what did you tell me", NEVER
# on "can you suggest a hotel for my trip" — the latter is an ADVICE
# question that needs the user model most. Requiring an explicit reference
# to a prior exchange is what separates them (measured: an earlier, looser
# 'you|recommend|suggest' pattern caught 70% of preference questions and
# would have suppressed the profile exactly where it helps).
# Aggregation-intent questions ("how many X across our chats") need the
# TALLY SHEET, not the diary: the fact tier stores ONE DATED ATOMIC FACT
# PER INSTANCE (the aggregation gate verified the instances exist — the
# three tanks, the five antiques, the workshop days are each their own
# fact), while raw turns scatter the same instances through ~30k chars of
# prose that language models demonstrably miscount (even the oracle
# does). On these questions the facts tier takes a larger share of the
# semantic budget. RECALL INTENT WINS on conflict ("our previous chat...
# how many times") because assistant-stated content is deliberately not
# extracted into facts — suppression must beat boosting there. Validated
# on the 350 held-out questions: 0 preference false-positives; the only
# assistant-question hits also match the recall rule and are taken by it.
_AGGREGATION_INTENT_RE = re.compile(
    r"\b(how many|how much|how often|how old|in total|altogether"
    r"|total (number|cost|amount|days))\b",
    re.IGNORECASE,
)
_AGGREGATION_FACTS_SHARE = 0.60

_CONVERSATION_RECALL_RE = re.compile(
    r"(previous|earlier|last|past|our)\s+"
    r"(conversation|chat|discussion|talk|session)"
    r"|\bwe (discussed|talked|spoke)\b"
    r"|\byou (told|mentioned|said|recommended|suggested|gave)\b"
    r"|\bremind me\b|\bgoing back to\b|\blooking back\b|\bfollow up on\b",
    re.IGNORECASE,
)


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
        # Populated by every assemble() so callers (gate runs) can
        # REPORT tier starvation instead of burying it in a log — the
        # Gate C lesson: an alarm that only reaches a log file is not
        # an alarm.
        self.last_tier_budget = {}

        # Lazy-initialized to avoid circular imports at startup
        self._store = None
        self._chroma = None
        self._kg = None
        self._procedural = None
        self._facts = None
        self._profile = None
        self.profile_session_ids = None
        self.profile_scoped_required = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def assemble(
        self,
        session_id: str,
        query: str,
        system_prompt: str = "You are an AI assistant with persistent memory, powered by AgentMem OS.",
        agent_id: str = None,
        user_id: str = None,
        disable: frozenset = frozenset(),
    ) -> str:
        """
        Build the full context string for the given session and query.

        Each section is capped at its token budget.
        Sections are labelled with XML-style tags for easy parsing in evaluations.

        disable: optional set of tier names to skip — {"profile", "facts", "semantic",
        "global", "procedural"}. Exists so ablation studies can exercise this
        real assembler directly instead of a hand-rolled simulation of it —
        see benchmarks/ablation_study_real.py. Defaults to empty (all tiers
        on), so this has no effect on any existing caller.

        user_id: second axis of the fact scope (make_scope_key) — the facts
        tier reads with the SAME scope derivation consolidation writes with.
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

        # ── Section 3a: Semantic FACTS (Stage 5 — facts first) ───────────────
        # The semantic tier's primary representation: dated atomic facts
        # distilled by ConsolidationV2, ranked lexical-primary with an
        # entity-linked recall floor (llm/fact_retrieval.py). Facts and
        # raw-turn chunks SHARE the semantic allocation — facts claim
        # first, chunks fill the remainder. With zero facts in store the
        # block is "" and everything below runs with the full allocation
        # untouched: the assembled output is byte-identical to the
        # pre-facts assembler (pinned in tests — the banked benchmark
        # numbers were measured through this code path).
        sem_budget = self.allocations["semantic"]

        # ── INTENT ROUTING (2026-08-11, measured) ────────────────────────
        # A question can ask two different things of memory:
        #   "what am I like / what did I do"  -> the USER MODEL (facts,
        #                                        profile)
        #   "what did YOU tell me last time"  -> the CONVERSATION itself,
        #                                        which only raw turns hold
        #                                        VERBATIM
        # Our extraction contract deliberately never stores assistant-
        # stated content (consolidation_v2.py:780), so on the second kind
        # the fact tier has nothing relevant — and injecting it anyway is
        # not merely useless, it is HARMFUL.
        #
        # MEASURED, run #1 (DECISION_AND_FAILURE_LOG §3.1q):
        # single-session-assistant 17/20 -> 3/20 with the fact tier on.
        # The cause is NOT eviction — a budget sweep showed the gold
        # evidence still survives into the context 18-19/20 at every
        # split. The model reads a wall of facts ABOUT THE USER, concludes
        # its memory lacks what the assistant said, and abstains without
        # using the raw turns below it (16 such abstentions measured).
        #
        # Why a lexical INTENT rule and not a relevance score: similarity
        # cannot detect this. On these questions the fact tier is
        # CONFIDENT AND WRONG — median top cosine 0.4132, HIGHER than
        # multi-session (0.3916) — because "what did you tell me about
        # slow cookers" matches the user's own cooking facts. Threshold
        # sweeps separate nothing (benchmarks/relevance_threshold_search.py).
        #
        # VALIDATED ON HELD-OUT DATA, not fitted: built from the 150-question
        # dev set, then checked against the 350 questions never used —
        # 35/36 = 97% recall on assistant-recall questions, and 0 false
        # positives in 314 others. The patterns are generic English recall
        # phrasings, not question-specific strings.
        # ABLATION SWITCH: a fix that cannot be turned OFF cannot be shown
        # to do anything. This exists so the routing's contribution can be
        # isolated from every other change in the same run — the confound
        # that made run #1 uninterpretable (§3.1s).
        recall_intent = bool(_CONVERSATION_RECALL_RE.search(query or "")) \
            and os.environ.get("AGENTMEM_OS_DISABLE_INTENT_ROUTING") != "1"
        if recall_intent:
            disable = frozenset(disable) | {"profile", "facts"}
        # OPT-IN, not default (2026-08-12): the paid probe FAILED its
        # pre-registered bar — of the 29 systematic failures it fixed 1
        # (bar >=4) and broke 2 stable answers. Instance DELIVERY
        # improved (rollercoasters 6->9 of 10 in context) and the model
        # still miscounted: the failure is distinguishing which delivered
        # instances are DISTINCT and IN-WINDOW, not seeing them. An
        # unproven change does not ship as a default; the mechanism stays
        # for corpora where it is measured to help.
        aggregation_intent = (
            not recall_intent
            and bool(_AGGREGATION_INTENT_RE.search(query or ""))
            and os.environ.get(
                "AGENTMEM_OS_ENABLE_AGGREGATION_ROUTING") == "1")
        facts_share = (_AGGREGATION_FACTS_SHARE if aggregation_intent
                       else FACTS_BUDGET_SHARE)

        # ── Section 2b: USER PROFILE (injected, never retrieved) ─────────
        # Who the user IS, always present when non-empty, from its own
        # reserved slice. Query-INDEPENDENT by design: this tier exists
        # precisely so that a preference's presence does not depend on
        # a query happening to match it (PROFILE_TIER_PLAN.md D4).
        profile_budget = int(sem_budget * PROFILE_BUDGET_SHARE)
        profile_used = 0
        if "profile" not in disable and profile_budget > 0:
            try:
                # Scoping parity with the facts tier (G3 R1 B7): an
                # eval that scopes facts per question MUST scope the
                # profile too, or the profile leaks every question's
                # attributes into every other question. Setting
                # profile_scoped_required makes an unset scope REFUSE
                # rather than silently read everything.
                sids = getattr(self, "profile_session_ids", None)
                if sids is None and getattr(
                        self, "profile_scoped_required", False):
                    raise RuntimeError(
                        "profile_session_ids unset while scoping is "
                        "required — refusing to inject an UNSCOPED "
                        "profile (that would leak across questions)")
                attrs = self._get_profile().current(
                    agent_id=agent_id, user_id=user_id, session_ids=sids)
                block = self._get_profile().render(
                    attrs, token_budget=profile_budget,
                    counter=self.counter)
                if block:
                    prof_section = self._fit_to_budget(
                        block, profile_budget, "[USER PROFILE]", keep="head")
                    if prof_section:
                        sections.append(prof_section)
                        profile_used = self.counter.count(prof_section)
                        sem_budget = max(0, sem_budget - profile_used)
            except Exception as e:
                # Same containment as the facts tier: a dead profile
                # must not take recall down with it, and must not be
                # silent either.
                logger.warning(
                    f"[ContextAssembler] Profile tier failed; continuing "
                    f"without it: {e}")

        # NO TIER MAY STARVE ANOTHER (Gate C 2026-08-09, measured):
        # facts consumed 99-100% of this allocation (4,737 of 4,740
        # tokens) and left raw-turn evidence 3 tokens. That did not
        # measure "facts + turns" — it measured "facts INSTEAD OF
        # turns", and the two cover DIFFERENT question shapes: 13
        # questions flipped to correct and 13 to wrong, netting zero.
        # The tiers are complements, not competitors, so the facts
        # tier may claim at most FACTS_BUDGET_SHARE and the remainder
        # is RESERVED for raw evidence. When there are few facts they
        # simply use less — the reservation only ever caps, never pads.
        facts_budget = int(sem_budget * facts_share)
        if "facts" not in disable:
            try:
                retriever = self._get_facts()
                # TOKEN budget, not chars (G3 R2 B1): rendered fact
                # blocks measure ~3.7-3.8 chars/token, so a chars=4×
                # proxy overfilled and _fit_to_budget's head-keeping
                # cut the chronologically-newest = rank-0 fact. The
                # retriever enforces BOTH of _fit_to_budget's cuts —
                # tokens AND the ×4 char fast path (G3 R4 B1: the char
                # half was still the working cut on ~5.9 chars/token
                # prose). COUPLING: changing _fit_to_budget's char
                # factor or keep direction must revisit
                # fact_retrieval._CALLER_CHAR_FACTOR — the high-ratio
                # sweep pin goes red if they drift apart.
                block = retriever.build_block(
                    query, agent_id=agent_id, user_id=user_id,
                    token_budget=facts_budget)
                if block:
                    facts_section = self._fit_to_budget(
                        block, sem_budget, "[SEMANTIC FACTS]", keep="head")
                    if facts_section:
                        sections.append(facts_section)
                        sem_budget = max(
                            0, sem_budget - self.counter.count(facts_section))
            except Exception as e:
                # Deliberately WARNING, not the debug-swallow the other
                # tiers use: a dead facts tier must not take recall down
                # with it (raw-turn fallback below still answers), but it
                # must never die silently either.
                logger.warning(
                    f"[ContextAssembler] Facts tier failed; falling back "
                    f"to raw retrieval: {e}")

        # G3 R1 B3: facts_used used to include the profile's tokens —
        # the alarm lied in the direction that hid the NEW tier's spend.
        # Every tier is now reported separately, with its own selection
        # note, so a gate run can print starvation instead of logging it.
        self.last_tier_budget = {
            # A routing decision that is not recorded cannot be audited
            # later — F-14's whole lesson is that a run must be able to
            # prove what it actually did, not what it was asked to do.
            "recall_intent": recall_intent,
            "aggregation_intent": aggregation_intent,
            "semantic_total": self.allocations["semantic"],
            "profile_cap": profile_budget,
            "profile_used": profile_used,
            "facts_cap": facts_budget,
            "facts_used": max(0, self.allocations["semantic"]
                              - profile_used - sem_budget),
            "chunks_left": sem_budget,
            "profile_selection": getattr(self._profile, "last_selection", None)
            if self._profile is not None else None}
        if "semantic" not in disable and sem_budget <= 0:
            # Loud, not silent (G3 R1 m2): a facts block that consumed
            # the whole semantic allocation starves raw-turn provenance
            # entirely — legitimate, but the operator must be able to
            # see it happened.
            logger.info(
                f"[ContextAssembler] session={session_id}: facts consumed "
                f"the full semantic allocation; raw-turn provenance skipped")

        # ── Section 3b: Semantic Memory (raw-turn provenance/fallback) ───────
        if "semantic" not in disable and sem_budget > 0:
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
                top_k = max(5, min(200, sem_budget // approx_chunk_tokens))
                chunks = chroma.search(session_id, query, top_k=top_k)
                if chunks:
                    # Rank decides WHAT survives; time decides HOW it reads.
                    # Retrieval returns chunks in similarity order — a shuffle
                    # of moments from many different days. Temporal questions
                    # ("how many days between X and Y") and aggregation
                    # questions ("how many N in total") are the two worst
                    # measured categories, and both need evidence laid out
                    # the way a person recounts it: chronologically. So the
                    # budget is filled by rank, then the SURVIVORS are
                    # reordered by their date stamps. No-ops gracefully when
                    # chunks carry no parseable dates.
                    chunks = self._order_evidence(
                        chunks, sem_budget
                    )
                    sem_text = "\n---\n".join(chunks)
                    sem_section = self._fit_to_budget(
                        sem_text, sem_budget, "[SEMANTIC MEMORY]",
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

    _EVIDENCE_DATE_RE = None  # compiled lazily

    def _order_evidence(self, chunks: list, token_budget: int) -> list:
        """
        Fill the token budget from the RANKED chunk list, then sort the
        survivors chronologically by their leading "[<date>]" stamp.

        Selection stays rank-based (the best evidence must survive the
        budget); only presentation changes. Dates in the "[YYYY/MM/DD ...]"
        form sort correctly as strings. If fewer than half the chunks carry
        a parseable date (e.g. corpora whose turns aren't date-stamped),
        the original ranked order is returned untouched — reordering by
        mostly-missing keys would be noise, not chronology.
        """
        import re as _re
        if ContextAssembler._EVIDENCE_DATE_RE is None:
            ContextAssembler._EVIDENCE_DATE_RE = _re.compile(r"^\[([^\]]+)\]")

        char_budget = token_budget * 4
        picked, used = [], 0
        for c in chunks:
            if used >= char_budget and picked:
                break
            picked.append(c)
            used += len(c) + 5

        dated = 0
        keys = []
        for i, c in enumerate(picked):
            m = ContextAssembler._EVIDENCE_DATE_RE.match(c)
            if m:
                dated += 1
                keys.append((m.group(1), i))
            else:
                keys.append(("~undated", i))  # "~" sorts after digits — undated last, stable
        if dated < max(2, len(picked) // 2):
            return picked
        keys.sort()
        return [picked[i] for _, i in keys]

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

    # Benchmark backend override (Stage 6 final pass, C1): install_*
    # in benchmarks/real_code_utils.py used to REPLACE _get_chroma on
    # the class — a permanent rewiring that silently ignored instance
    # _chroma assignments for the rest of the process, so importing
    # the benchmark adapter in one test file broke the byte-identity
    # pin in every later file. The override is now DATA the instance
    # attribute always beats, and tests reset it (conftest autouse).
    _chroma_override = None

    def _get_chroma(self):
        if self._chroma is None:
            if ContextAssembler._chroma_override is not None:
                self._chroma = ContextAssembler._chroma_override
            else:
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

    def _get_profile(self):
        if self._profile is None:
            from agentmem_os.db.profile import ProfileStore
            from agentmem_os.db.engine import get_session
            self._profile = ProfileStore(get_session)
        return self._profile

    def _get_facts(self):
        if self._facts is None:
            from agentmem_os.llm.fact_retrieval import FactRetriever
            from agentmem_os.db.engine import get_session
            self._facts = FactRetriever(get_session)
        return self._facts
