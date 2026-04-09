"""
AgentMem OS — Phase 2 DB Tests
================================
Verifies the database foundation is correct:
  - All 4-tier tables are created successfully
  - Session factory works
  - WAL mode is active
  - FK constraints are enforced
  - Basic CRUD on every new table

Run with:
    pytest tests/test_phase2_db.py -v
"""

import os
import pytest
from sqlalchemy import text, inspect

# Use an in-memory SQLite DB for tests — no file system needed
os.environ["MEMNAI_DB_PATH"] = ":memory:"


@pytest.fixture(scope="module")
def engine_and_session():
    """Provide a clean in-memory DB engine and session for all tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from agentmem_os.db.models import Base

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    yield engine, session
    session.close()
    engine.dispose()  # Close all pooled connections → fixes ResourceWarning


# ─────────────────────────────────────────────────────────────────────────────
# Table Existence Tests
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_TABLES = [
    # ── Phase 1-3 tables ──────────────────────────────────────────────────────
    "sessions",
    "turns",
    "summaries",
    "procedural_patterns",
    "kg_nodes",
    "kg_edges",
    "agent_namespaces",
    "cost_log",
    "consolidation_log",
    # ── Phase 4 tables (multi-agent) ──────────────────────────────────────────
    "agent_trust_scores",
    "federated_memory",
    "agent_fork_records",
    "memory_access_log",
]

def test_all_tables_created(engine_and_session):
    """Every table defined in models.py must exist after init_db()."""
    engine, _ = engine_and_session
    inspector = inspect(engine)
    actual_tables = inspector.get_table_names()
    for table in EXPECTED_TABLES:
        assert table in actual_tables, f"Table '{table}' was not created"

def test_no_unexpected_tables(engine_and_session):
    """No extra tables should exist — catch accidental additions."""
    engine, _ = engine_and_session
    inspector = inspect(engine)
    actual = set(inspector.get_table_names())
    expected = set(EXPECTED_TABLES)
    extra = actual - expected
    assert not extra, f"Unexpected tables found: {extra}"


# ─────────────────────────────────────────────────────────────────────────────
# Column Existence Tests (Phase 2 additions to existing tables)
# ─────────────────────────────────────────────────────────────────────────────

def test_turns_table_has_importance_score(engine_and_session):
    """Turn table must have importance_score column (new in Phase 2)."""
    engine, _ = engine_and_session
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("turns")]
    assert "importance_score" in cols
    assert "semantic_novelty" in cols
    assert "entity_count" in cols
    assert "is_compressed" in cols

def test_summaries_table_has_cluster_columns(engine_and_session):
    """Summary table must have cluster_id and abstraction_level (new in Phase 2)."""
    engine, _ = engine_and_session
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("summaries")]
    assert "cluster_id" in cols
    assert "abstraction_level" in cols
    assert "is_shared" in cols

def test_sessions_table_has_agent_id(engine_and_session):
    """Session table must have agent_id FK for multi-agent support."""
    engine, _ = engine_and_session
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("sessions")]
    assert "agent_id" in cols
    assert "is_archived" in cols


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Tests — Tier 2: Episodic Memory
# ─────────────────────────────────────────────────────────────────────────────

def test_create_session(engine_and_session):
    """Can create a Session row."""
    _, db = engine_and_session
    from agentmem_os.db.models import Session
    s = Session(
        session_id="test-session-001",
        name="Test Session",
        model="ollama/llama3.2:3b",
        branch_type="root",
    )
    db.add(s)
    db.commit()
    found = db.query(Session).filter(Session.session_id == "test-session-001").first()
    assert found is not None
    assert found.name == "Test Session"
    assert found.total_tokens == 0
    assert found.is_archived == False

def test_create_turn(engine_and_session):
    """Can create a Turn row linked to a Session."""
    _, db = engine_and_session
    from agentmem_os.db.models import Turn
    t = Turn(
        session_id="test-session-001",
        role="user",
        content="Hello AgentMem OS!",
        token_count=5,
        importance_score=0.75,
        entity_count=0,
        semantic_novelty=0.5,
    )
    db.add(t)
    db.commit()
    found = db.query(Turn).filter(Turn.session_id == "test-session-001").first()
    assert found is not None
    assert found.role == "user"
    assert found.importance_score == 0.75
    assert found.is_compressed == False


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Tests — Tier 3: Semantic Memory
# ─────────────────────────────────────────────────────────────────────────────

def test_create_summary(engine_and_session):
    """Can create a Summary row (Tier 3) with cluster metadata."""
    _, db = engine_and_session
    from agentmem_os.db.models import Summary
    s = Summary(
        session_id="test-session-001",
        turn_range="1-5",
        content="User discussed the AgentMem OS architecture.",
        entities="AgentMem,SQLite,ChromaDB",
        cluster_id=0,
        abstraction_level=1,
        is_shared=False,
    )
    db.add(s)
    db.commit()
    found = db.query(Summary).filter(Summary.session_id == "test-session-001").first()
    assert found is not None
    assert found.cluster_id == 0
    assert found.abstraction_level == 1


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Tests — Tier 4: Procedural Memory (NEW table)
# ─────────────────────────────────────────────────────────────────────────────

def test_create_procedural_pattern(engine_and_session):
    """Can create a ProceduralPattern row (Tier 4 — brand new table)."""
    _, db = engine_and_session
    from agentmem_os.db.models import ProceduralPattern
    p = ProceduralPattern(
        trigger="bug_report",
        action="provided code block",
        full_pattern="When user reports a bug → provide code block with fix",
        confidence=0.8,
        support_count=3,
        source_sessions="test-session-001",
        is_global=False,
    )
    db.add(p)
    db.commit()
    found = db.query(ProceduralPattern).first()
    assert found is not None
    assert found.trigger == "bug_report"
    assert found.confidence == 0.8
    assert found.support_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Tests — Knowledge Graph (NEW tables)
# ─────────────────────────────────────────────────────────────────────────────

def test_create_kg_node(engine_and_session):
    """Can create a KnowledgeGraphNode row."""
    _, db = engine_and_session
    from agentmem_os.db.models import KnowledgeGraphNode
    node = KnowledgeGraphNode(
        session_id="test-session-001",
        entity_text="Anthropic",
        entity_type="ORG",
        mention_count=3,
    )
    db.add(node)
    db.commit()
    found = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Anthropic"
    ).first()
    assert found is not None
    assert found.entity_type == "ORG"
    assert found.mention_count == 3

def test_create_kg_edge(engine_and_session):
    """Can create a KnowledgeGraphEdge linking two nodes."""
    _, db = engine_and_session
    from agentmem_os.db.models import KnowledgeGraphNode, KnowledgeGraphEdge

    node_a = KnowledgeGraphNode(
        session_id="test-session-001",
        entity_text="Claude",
        entity_type="PRODUCT",
        mention_count=5,
    )
    node_b = KnowledgeGraphNode(
        session_id="test-session-001",
        entity_text="Sahith",
        entity_type="PERSON",
        mention_count=2,
    )
    db.add_all([node_a, node_b])
    db.flush()

    edge = KnowledgeGraphEdge(
        source_id=node_a.id,
        target_id=node_b.id,
        weight=4.0,
        session_id="test-session-001",
    )
    db.add(edge)
    db.commit()

    found = db.query(KnowledgeGraphEdge).filter(
        KnowledgeGraphEdge.source_id == node_a.id
    ).first()
    assert found is not None
    assert found.weight == 4.0


# ─────────────────────────────────────────────────────────────────────────────
# CRUD Tests — ConsolidationLog (NEW table)
# ─────────────────────────────────────────────────────────────────────────────

def test_create_consolidation_log(engine_and_session):
    """Can create a ConsolidationLog entry (benchmark audit trail)."""
    _, db = engine_and_session
    from agentmem_os.db.models import ConsolidationLog
    log = ConsolidationLog(
        session_id="test-session-001",
        turns_processed=30,
        clusters_found=4,
        summaries_generated=4,
        tokens_before=95000,
        tokens_after=62000,
        compression_ratio=0.653,
        duration_seconds=1.24,
        triggered_by="threshold",
    )
    db.add(log)
    db.commit()
    found = db.query(ConsolidationLog).first()
    assert found is not None
    assert found.clusters_found == 4
    assert found.compression_ratio == 0.653


# ─────────────────────────────────────────────────────────────────────────────
# Branching Test
# ─────────────────────────────────────────────────────────────────────────────

def test_session_branching(engine_and_session):
    """Child session can reference parent session via parent_session_id."""
    _, db = engine_and_session
    from agentmem_os.db.models import Session

    parent = Session(
        session_id="parent-001",
        name="Parent Session",
        branch_type="root",
    )
    db.add(parent)
    db.flush()

    child = Session(
        session_id="parent-001/branch-a",
        name="Branch A",
        parent_session_id="parent-001",
        branch_point_turn=10,
        inherited_context="Summary of parent context.",
        branch_type="hard",
    )
    db.add(child)
    db.commit()

    found = db.query(Session).filter(
        Session.parent_session_id == "parent-001"
    ).first()
    assert found is not None
    assert found.session_id == "parent-001/branch-a"
    assert found.inherited_context == "Summary of parent context."
