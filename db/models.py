"""
AgentMem OS — Database Models
==============================
Implements the full 4-tier memory hierarchy:

  Tier 1: Working Memory    → managed in Redis (not ORM)
  Tier 2: Episodic Memory   → Turn table
  Tier 3: Semantic Memory   → Summary + SemanticChunk tables
  Tier 4: Procedural Memory → ProceduralPattern table

Additional:
  - KnowledgeGraphNode / KnowledgeGraphEdge: Entity graph persistence
  - AgentNamespace: Multi-agent isolation
  - CostLog: API cost tracking
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime,
    ForeignKey, Text, Boolean, JSON,
    Index, UniqueConstraint, text as sql_text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ──────────────────────────────────────────────────────────────────────────────
# AGENT NAMESPACE  (multi-agent isolation — Phase 4)
# ──────────────────────────────────────────────────────────────────────────────

class AgentNamespace(Base):
    """
    Each agent gets its own namespace. Episodic memory is private per agent.
    Semantic memories can be promoted to a shared pool (shared_pool=True).
    """
    __tablename__ = "agent_namespaces"

    agent_id   = Column(String, primary_key=True)
    name       = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata_  = Column(JSON, default=dict)     # arbitrary agent config

    sessions   = relationship("Session", back_populates="agent")


# ──────────────────────────────────────────────────────────────────────────────
# SESSION  (branching conversation tree)
# ──────────────────────────────────────────────────────────────────────────────

class Session(Base):
    """
    Root entity for a conversation thread. Supports parent-child branching.
    branch_type: 'root' | 'hard' (new direction) | 'soft' (minor variant)
    """
    __tablename__ = "sessions"

    session_id         = Column(String, primary_key=True)
    agent_id           = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=True)
    parent_session_id  = Column(String, ForeignKey("sessions.session_id"), nullable=True)
    branch_point_turn  = Column(Integer, nullable=True)
    inherited_context  = Column(Text, nullable=True)   # snapshot of parent at branch time
    name               = Column(String, nullable=True)
    model              = Column(String, nullable=True)
    branch_type        = Column(String, default="root")
    created_at         = Column(DateTime, default=datetime.utcnow)
    total_tokens       = Column(Integer, default=0)
    total_cost_usd     = Column(Float, default=0.0)
    is_archived        = Column(Boolean, default=False)

    # Relationships
    agent    = relationship("AgentNamespace", back_populates="sessions")
    parent   = relationship("Session", remote_side=[session_id], back_populates="children")
    children = relationship("Session", back_populates="parent")
    turns    = relationship("Turn", back_populates="session", cascade="all, delete-orphan")
    summaries= relationship("Summary", back_populates="session", cascade="all, delete-orphan")


# ──────────────────────────────────────────────────────────────────────────────
# TIER 2: EPISODIC MEMORY — raw conversation turns
# ──────────────────────────────────────────────────────────────────────────────

class Turn(Base):
    """
    A single message in a conversation (user or assistant).
    importance_score is computed by the MemoryImportanceScorer before compression.
    """
    __tablename__ = "turns"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(String, ForeignKey("sessions.session_id"))
    role             = Column(String, nullable=False)      # 'user' | 'assistant' | 'system'
    content          = Column(Text, nullable=False)
    token_count      = Column(Integer, default=0)
    created_at       = Column(DateTime, default=datetime.utcnow)

    # AgentMem OS additions
    importance_score = Column(Float, default=0.0)          # 0.0–1.0, set by MemoryImportanceScorer
    entity_count     = Column(Integer, default=0)          # NER entity count, set at save time
    semantic_novelty = Column(Float, default=0.0)          # distance from existing summaries
    is_compressed    = Column(Boolean, default=False)       # True once included in a Summary

    # Phase 1: Conflict Detection
    is_active        = Column(Boolean, default=True)        # False = superseded by a later fact
    contradicted_by  = Column(Integer, ForeignKey("turns.id"), nullable=True)  # FK to the turn that supersedes this one

    session = relationship("Session", back_populates="turns")


# ──────────────────────────────────────────────────────────────────────────────
# TIER 3: SEMANTIC MEMORY — compressed, meaning-indexed
# ──────────────────────────────────────────────────────────────────────────────

class Summary(Base):
    """
    A compressed semantic summary of one or more turns.
    Generated by the SleepConsolidationEngine.
    Also indexed in ChromaDB for semantic retrieval.
    """
    __tablename__ = "summaries"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(String, ForeignKey("sessions.session_id"))
    agent_id        = Column(String, nullable=True)        # set for shared-pool promotions
    turn_range      = Column(String, nullable=False)       # e.g. "42-67"
    content         = Column(Text, nullable=False)
    entities        = Column(Text, nullable=True)          # comma-separated entity names
    cluster_id      = Column(Integer, nullable=True)       # DBSCAN cluster this belongs to
    abstraction_level = Column(Integer, default=1)         # 1=episode, 2=pattern, 3=principle
    is_shared       = Column(Boolean, default=False)       # promoted to multi-agent shared pool
    created_at      = Column(DateTime, default=datetime.utcnow)

    session = relationship("Session", back_populates="summaries")


class SemanticFact(Base):
    """
    An atomic, tri-temporally dated, individually-cited fact distilled from
    episodic turns by the consolidation engine (Consolidation v2 — see
    CONSOLIDATION_V2_DESIGN.md). The true semantic tier: Summary compresses
    text; this stores KNOWLEDGE.

    Atomic = one proposition, self-contained referents, numbers/dates kept
    verbatim (never merged with a fact whose numbers differ — the counts are
    the point). Transitions ("switched from X to Y") are NOT written into
    fact text; they are reconstructed at read time by walking the
    superseded_by chain (Mem0 writes them into the text because it has no
    chain to walk — we do).

    Three timestamps (tri-temporal — "bi-temporal" is Zep's two-time term,
    don't borrow it):
      t_occurred  — when the thing happened (event date if stated, else the
                    session date), sortable "YYYY/MM/DD" text, partial
                    "YYYY/MM" allowed, nullable
      t_mentioned — when the user said it (session date) — the ONLY valid
                    anchor for resolving "last week"-style references
      t_ingested  — when we recorded it (transaction time)

    superseded_by/superseded_at: invalidate-don't-delete. superseded_at is
    OUR decision time, kept separate from validity (Graphiti's
    expired_at-vs-invalid_at split, verified from its source).

    mention_count/last_confirmed_at: re-affirmation strengthens the one row
    instead of duplicating it (the bloat class behind Mem0 issue #4573's
    97.8%-junk audit).

    scope_key: non-null dedup scope (agent_id or user_id or "global") so the
    UNIQUE(scope_key, normalized_hash) constraint is the FINAL dedup
    authority even under concurrent writers (SQLite treats NULLs as distinct
    in unique constraints, so a nullable column can't serve — Mem0's TOCTOU
    duplicate race, issue #6531, is the failure mode this closes).
    """
    __tablename__ = "semantic_facts"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    agent_id          = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=True)
    user_id           = Column(String, nullable=True)
    scope_key         = Column(String, nullable=False, default="global")

    fact_text         = Column(Text, nullable=False)
    fact_type         = Column(String, nullable=False)     # event | state | preference | identity
    # Event dating as an explicit interval: a full date is a point
    # (t_occurred_end NULL); a month-only date stores its first and last day
    # so lexical range queries can never drop it ("2023/10" sorting before
    # "2023/10/01" was G3 finding 8).
    t_occurred        = Column(String, nullable=True)      # sortable "YYYY/MM/DD"
    t_occurred_end    = Column(String, nullable=True)      # NULL = point in time
    t_mentioned       = Column(String, nullable=False)     # sortable "YYYY/MM/DD"
    t_ingested        = Column(DateTime, default=datetime.utcnow)

    source_session_id  = Column(String, ForeignKey("sessions.session_id"), nullable=False)
    source_session_ids = Column(JSON, default=list)        # ALL sessions that affirmed this fact
    source_turn_ids    = Column(JSON, default=list)        # citations — every fact traceable
    entities          = Column(JSON, default=list)         # entity strings; KG linking in Stage 3
    lang_source       = Column(String, default="en")       # language of FIRST source
    langs             = Column(JSON, default=list)         # all source languages (cross-lingual)
    extraction_model  = Column(String, nullable=False)     # disclosure, per honesty rules

    normalized_hash   = Column(String, nullable=False)     # sha256 of normalized fact_text
    mention_count     = Column(Integer, nullable=False, default=1, server_default=sql_text("1"))
    last_confirmed_at = Column(DateTime, default=datetime.utcnow)

    superseded_by     = Column(Integer, ForeignKey("semantic_facts.id"), nullable=True)
    superseded_at     = Column(DateTime, nullable=True)    # when WE invalidated it

    __table_args__ = (
        UniqueConstraint("scope_key", "normalized_hash", name="uq_facts_scope_hash"),
        # Partial index for the hot "current facts" path. SQLite's planner
        # matches the predicate TEXTUALLY — every query on this path must
        # say `superseded_by IS NULL` literally or the index silently
        # goes unused (slow, not wrong; sqlite.org/partialindex.html).
        Index(
            "idx_facts_current", "scope_key", "fact_type", "t_occurred",
            sqlite_where=sql_text("superseded_by IS NULL"),
        ),
        # Serves the no-fact_type default read path (G3 finding 9: without
        # this, the default current_facts() full-scanned + temp-sorted).
        Index(
            "idx_facts_current_all", "scope_key", "t_occurred",
            sqlite_where=sql_text("superseded_by IS NULL"),
        ),
        Index("idx_facts_mentioned", "scope_key", "t_mentioned"),
        Index("idx_facts_valid_range", "scope_key", "t_occurred", "t_ingested"),
        Index("idx_facts_source_session", "source_session_id"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# TIER 4: PROCEDURAL MEMORY — learned patterns and behavioral templates
# ──────────────────────────────────────────────────────────────────────────────

class ProceduralPattern(Base):
    """
    A reusable behavioral pattern or skill extracted from episodic sequences.
    Example: "When user reports a bug → ask for minimal reproduction first"

    Extracted by: PatternExtractor (Phase 3D)
    Used by: ContextAssembler to inject relevant procedural context
    """
    __tablename__ = "procedural_patterns"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    agent_id        = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=True)
    trigger         = Column(Text, nullable=False)         # "When X..."
    action          = Column(Text, nullable=False)         # "...do Y"
    full_pattern    = Column(Text, nullable=False)         # human-readable if-then
    confidence      = Column(Float, default=0.5)          # 0.0–1.0
    support_count   = Column(Integer, default=1)           # how many episodes support this
    source_sessions = Column(Text, nullable=True)          # comma-sep session IDs
    embedding_id    = Column(String, nullable=True)        # ChromaDB doc ID
    created_at      = Column(DateTime, default=datetime.utcnow)
    last_used_at    = Column(DateTime, nullable=True)
    is_global       = Column(Boolean, default=False)       # shared across all agents


# ──────────────────────────────────────────────────────────────────────────────
# ENTITY KNOWLEDGE GRAPH — the Global Map (fills the 10% context slot)
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphNode(Base):
    """
    A named entity extracted from conversation turns via spaCy NER.
    Node in the persistent Entity Knowledge Graph.
    """
    __tablename__ = "kg_nodes"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    agent_id    = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=True)
    session_id  = Column(String, nullable=True)            # originating session
    entity_text = Column(String, nullable=False)           # "Python", "OpenAI", "Sahith"
    entity_type = Column(String, nullable=False)           # spaCy label: ORG, PERSON, TECH, etc.
    mention_count = Column(Integer, default=1)
    first_seen  = Column(DateTime, default=datetime.utcnow)
    last_seen   = Column(DateTime, default=datetime.utcnow)
    # Temporal KG (LAUNCH_ROADMAP.md Phase 6 Priority 2, ported from
    # X-MemoryArch's graph_builder.py): last time this entity was confirmed
    # by a NEW typed-relation edge, distinct from last_seen (which updates
    # on every co-occurrence mention, typed or not).
    last_confirmed_at = Column(DateTime, nullable=True)


class KnowledgeGraphEdge(Base):
    """
    A relationship between two entities.

    Two edge families, per X-MemoryArch's graph_builder.py (LAUNCH_ROADMAP.md
    Phase 6 Priority 2):
      CO_OCCURS (relation_type default) — accumulates forever, weight counts
        co-occurrence, never superseded. Historical co-occurrence is a
        permanent fact even after the entities' relationship changes.
      Typed relations (WORKS_AT / LIVES_AT / STUDIES_AT) — bi-temporally
        supersedable: when a NEW typed edge arrives for the same
        (source_id, relation_type), the previous active edge (valid_until
        IS NULL) gets valid_until set to the new edge's valid_from and
        superseded_by pointed at the new edge's id. Deterministic — same
        (subject, relation_type) match, later timestamp wins, zero LLM
        calls, matching conflict_detector.py's existing zero-LLM design.
    """
    __tablename__ = "kg_edges"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    source_id   = Column(Integer, ForeignKey("kg_nodes.id"), nullable=False)
    target_id   = Column(Integer, ForeignKey("kg_nodes.id"), nullable=False)
    weight      = Column(Float, default=1.0)               # co-occurrence count
    session_id  = Column(String, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)
    relation_type = Column(String, default="CO_OCCURS")    # CO_OCCURS | WORKS_AT | LIVES_AT | STUDIES_AT
    confidence    = Column(Float, default=0.5)             # 0.5 co-occurrence, 0.80-0.90 typed
    valid_from    = Column(DateTime, nullable=True)         # when this fact became true
    valid_until   = Column(DateTime, nullable=True)         # NULL = currently active
    superseded_by = Column(Integer, ForeignKey("kg_edges.id"), nullable=True)


# ──────────────────────────────────────────────────────────────────────────────
# COST LOG — API cost tracking per call
# ──────────────────────────────────────────────────────────────────────────────

class CostLog(Base):
    """Tracks API cost per LLM call with cached token savings."""
    __tablename__ = "cost_log"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(String, ForeignKey("sessions.session_id"))
    agent_id       = Column(String, nullable=True)
    model          = Column(String, nullable=False)
    input_tokens   = Column(Integer, default=0)
    output_tokens  = Column(Integer, default=0)
    cached_tokens  = Column(Integer, default=0)
    cost_usd       = Column(Float, default=0.0)
    timestamp      = Column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# CONSOLIDATION LOG — track Sleep Consolidation Engine runs
# ──────────────────────────────────────────────────────────────────────────────

class ConsolidationLog(Base):
    """
    Audit trail for every Sleep Consolidation Engine run.
    Useful for benchmarking and the research paper evaluation section.
    """
    __tablename__ = "consolidation_log"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    session_id          = Column(String, nullable=True)
    turns_processed     = Column(Integer, default=0)
    clusters_found      = Column(Integer, default=0)
    summaries_generated = Column(Integer, default=0)
    tokens_before       = Column(Integer, default=0)
    tokens_after        = Column(Integer, default=0)
    compression_ratio   = Column(Float, default=0.0)       # tokens_after / tokens_before
    duration_seconds    = Column(Float, default=0.0)
    triggered_by        = Column(String, default="scheduled")  # 'scheduled' | 'threshold' | 'manual'
    timestamp           = Column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 4: MEMORY FEDERATION PROTOCOL
# ──────────────────────────────────────────────────────────────────────────────

class AgentTrustScore(Base):
    """
    Pairwise trust scores between agents in the federation.

    Trust is directional: agent_from trusts agent_to with score ∈ [0, 1].
    When agent_from retrieves memories from agent_to and those memories
    prove useful (positive feedback), trust increases. Negative feedback
    or disuse causes slow decay.

    Trust update rule (exponential moving average):
        trust_new = α × trust_old + (1 − α) × feedback_signal
        α = 0.8  (trust earned slowly, lost slowly)

    Initialized to 0.5 (neutral) for all new pairs.
    Fork relationships start at 0.9 (child trusts parent by default).
    """
    __tablename__ = "agent_trust_scores"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    agent_from     = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=False)
    agent_to       = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=False)
    trust_score    = Column(Float, default=0.5)            # 0.0 = untrusted, 1.0 = fully trusted
    interaction_count = Column(Integer, default=0)         # total interactions that shaped this score
    last_updated   = Column(DateTime, default=datetime.utcnow)
    created_at     = Column(DateTime, default=datetime.utcnow)


class FederatedMemoryEntry(Base):
    """
    A memory promoted to the cross-agent shared pool by the MFP (Memory Federation Protocol).

    Only memories with abstraction_level >= 2 (pattern or principle) are eligible.
    Raw episodic turns NEVER enter the federated pool — this is a privacy guarantee.

    Fields:
      source_agent_id   — which agent produced this memory
      content           — the abstracted summary text
      abstraction_level — 2 (pattern) or 3 (principle)
      promotion_score   — the score that triggered promotion
      access_count      — how many agents have retrieved this entry
      last_accessed_at  — used for relevance aging / decay
      chroma_doc_id     — pointer to ChromaDB for vector retrieval
    """
    __tablename__ = "federated_memory"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    source_agent_id   = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=False)
    source_session_id = Column(String, nullable=True)
    content           = Column(Text, nullable=False)
    abstraction_level = Column(Integer, default=2)         # 2=pattern, 3=principle
    promotion_score   = Column(Float, default=0.0)         # score that qualified this entry
    access_count      = Column(Integer, default=0)         # times retrieved by other agents
    last_accessed_at  = Column(DateTime, nullable=True)
    chroma_doc_id     = Column(String, nullable=True)      # ChromaDB document ID
    is_active         = Column(Boolean, default=True)      # False = decayed/retired
    created_at        = Column(DateTime, default=datetime.utcnow)


class AgentForkRecord(Base):
    """
    Records the parent-child relationship when an agent is forked.

    Forking means: new_agent inherits all L2/L3 summaries from parent_agent
    as its initial semantic knowledge base, but builds its own episodic layer
    from scratch. Analogous to git branch from a commit.

    The fork_depth tracks inheritance chains:
      parent (depth=0) → child (depth=1) → grandchild (depth=2)
    """
    __tablename__ = "agent_fork_records"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    parent_agent_id = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=False)
    child_agent_id  = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=False)
    fork_depth      = Column(Integer, default=1)           # 1=direct child, 2=grandchild, etc.
    summaries_inherited = Column(Integer, default=0)       # how many summaries were copied
    patterns_inherited  = Column(Integer, default=0)       # how many patterns were copied
    forked_at       = Column(DateTime, default=datetime.utcnow)


class MemoryAccessLog(Base):
    """
    Log of every time an agent retrieves a FederatedMemoryEntry.
    Used for:
      - Decay: entries never accessed by other agents are eventually retired.
      - Trust updates: positive feedback from this agent about source_agent.
      - Analytics: which memories are most valuable across the federation.
    """
    __tablename__ = "memory_access_log"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    accessing_agent_id    = Column(String, ForeignKey("agent_namespaces.agent_id"), nullable=False)
    federated_entry_id    = Column(Integer, ForeignKey("federated_memory.id"), nullable=False)
    query_text            = Column(Text, nullable=True)    # what the agent was querying
    relevance_score       = Column(Float, default=0.0)     # cosine similarity at retrieval time
    feedback_signal       = Column(Float, nullable=True)   # 0.0–1.0, set after interaction
    accessed_at           = Column(DateTime, default=datetime.utcnow)
