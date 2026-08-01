"""
Temporal Knowledge Graph tests — LAUNCH_ROADMAP.md Phase 6 Priority 2,
ported from X-MemoryArch's graph_builder.py/graph_retrieval.py.

Verifies the actual production code path (db/knowledge_graph.py's
EntityKnowledgeGraph.ingest_turn/get_relevant_subgraph, plus the typed
relation extraction reusing memory/conflict_detector.py's vocabulary) —
not a reimplementation of the supersession formula.

Uses an isolated in-memory SQLite engine (same pattern as
test_phase2_db.py), not the shared agentmem_os.db.engine module-level
engine — sidesteps that engine being bound to its DB path at import time
(see benchmarks/mfp_eval.py's docstring for why that matters) and keeps
this test process-safe alongside the rest of the suite.

Test sentences deliberately use well-known real-world company names
(Google, Microsoft) rather than fictional/novel ones — verified empirically
that AgentMem's spaCy-first NER reliably recognizes these as ORG entities
but does NOT reliably recognize an unfamiliar name like "Sarvam AI" (a
real, expected NER limitation, not a bug in this port) — using recognized
entities keeps these tests deterministic instead of coupled to spaCy's
training data for made-up examples.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest


@pytest.fixture()
def kg():
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from agentmem_os.db.models import Base
    from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

    from agentmem_os.db.models import AgentNamespace
    seed = SessionLocal()
    seed.add(AgentNamespace(agent_id="test-agent", name="test-agent"))
    seed.commit()
    seed.close()

    return EntityKnowledgeGraph(SessionLocal)


def test_typed_relation_created_on_ingest(kg):
    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala works at Google.")

    db = kg.get_db()
    from agentmem_os.db.models import KnowledgeGraphEdge
    edges = db.query(KnowledgeGraphEdge).filter_by(relation_type="WORKS_AT").all()
    db.close()

    assert len(edges) == 1
    assert edges[0].valid_until is None
    assert edges[0].confidence == 0.85


def test_second_statement_supersedes_the_first(kg):
    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala works at Google.")
    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala joined Microsoft.")

    db = kg.get_db()
    from agentmem_os.db.models import KnowledgeGraphEdge
    edges = (
        db.query(KnowledgeGraphEdge)
        .filter_by(relation_type="WORKS_AT")
        .order_by(KnowledgeGraphEdge.id)
        .all()
    )
    db.close()

    assert len(edges) == 2, "expected the old edge plus the new superseding edge"
    old, new = edges
    assert old.valid_until is not None, "old edge should be marked no-longer-active"
    assert old.superseded_by == new.id
    assert new.valid_until is None, "new edge should be the currently-active one"


def test_default_retrieval_shows_only_the_current_fact(kg):
    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala works at Google.")
    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala joined Microsoft.")

    kg._graph.clear()  # force a fresh load from DB, not the in-process cache
    result = kg.get_relevant_subgraph("Where does Sahith Thummala work?", "test-agent")

    assert "Microsoft" in result
    assert "Google" not in result, "superseded fact must not appear in the default view"


def test_as_of_shows_the_historical_fact(kg):
    from datetime import datetime

    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala works at Google.")
    # Real wall-clock midpoint between the two ingestions — NOT
    # first_edge.valid_from + a fixed offset, since a fast test runs both
    # ingestions within milliseconds of each other and a too-large offset
    # (e.g. +1 second) overshoots past the second edge's own valid_from,
    # landing as_of AFTER the supersession instead of before it (verified
    # empirically: that was this test's first, wrong version).
    midpoint = datetime.utcnow()

    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala joined Microsoft.")

    as_of_result = kg.get_relevant_subgraph(
        "Where does Sahith Thummala work?", "test-agent", as_of=midpoint,
    )
    current_result = kg.get_relevant_subgraph("Where does Sahith Thummala work?", "test-agent")

    assert "Google" in as_of_result, "point-in-time query should see the fact as it was then"
    assert "Microsoft" not in as_of_result
    assert "Microsoft" in current_result, "default (as_of=None) query should see the current fact"


def test_co_occurs_edge_is_never_superseded(kg):
    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala works at Google.")
    kg.ingest_turn("sess-1", "test-agent", "Sahith Thummala joined Microsoft.")

    db = kg.get_db()
    from agentmem_os.db.models import KnowledgeGraphEdge
    co_occurs = db.query(KnowledgeGraphEdge).filter_by(relation_type="CO_OCCURS").all()
    db.close()

    assert co_occurs, "expected at least one CO_OCCURS edge from entity co-mentions"
    assert all(e.valid_until is None for e in co_occurs), \
        "CO_OCCURS edges must never be superseded — historical co-occurrence is permanent"
