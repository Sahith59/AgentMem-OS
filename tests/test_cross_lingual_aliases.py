"""
Cross-lingual entity alias tests — the live wiring (db/entity_aliases.py +
db/knowledge_graph.py) of the measured benchmarks/cross_lingual_kg_eval.py
methodology: multilingual-e5-small, "query: " prefix, τ=0.90 default,
non-destructive ALIAS_OF edges, alias-gated Indic-script extraction.

Every similarity these tests rely on was measured against the real model
BEFORE the assertions were written (same prefix+normalize methodology as
the eval script):

    गूगल  vs Google           0.9506   (above τ — must alias)
    बेंगलुरु vs Bengaluru      0.9256   (above τ — must alias)
    पानी  vs Google           0.8526   (below τ — must be dropped)
    है/की/कल/मीटिंग vs any     ≤0.8577  (below τ — must never node)

Pairs that measured within ~0.01 of τ (Microsoft/माइक्रोसॉफ्ट at 0.9088,
Sahith/साहित at 0.9074) are deliberately not asserted on — too close to
the threshold to survive model-version drift.

Model-dependent tests skip when sentence-transformers isn't installed —
the core install is torch-free by design, and CI exercises the
disabled-path test only.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

try:
    import sentence_transformers  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False

needs_model = pytest.mark.skipif(
    not HAS_ST, reason="multilingual extra (sentence-transformers) not installed"
)


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
    SessionLocal = sessionmaker(
        bind=engine, autocommit=False, autoflush=False, expire_on_commit=False
    )

    from agentmem_os.db.models import AgentNamespace
    seed = SessionLocal()
    seed.add(AgentNamespace(agent_id="test-agent", name="test-agent"))
    seed.commit()
    seed.close()

    return EntityKnowledgeGraph(SessionLocal)


def _node_texts(kg):
    from agentmem_os.db.models import KnowledgeGraphNode
    db = kg.get_db()
    texts = [r.entity_text for r in db.query(KnowledgeGraphNode).all()]
    db.close()
    return texts


def _alias_edges(kg):
    from agentmem_os.db.models import KnowledgeGraphEdge
    db = kg.get_db()
    edges = db.query(KnowledgeGraphEdge).filter_by(relation_type="ALIAS_OF").all()
    db.close()
    return edges


def test_disabled_resolver_leaves_ingest_unchanged(kg, monkeypatch):
    """
    Off = zero NEW behavior: the NER-missed गूगल stays missed, and no
    ALIAS_OF edges appear. (en_core_web_sm may still tag arbitrary Indic
    spans as entities — that's pre-existing NER behavior with the resolver
    off, deliberately untouched; the resolver-ON path is what gates it.)
    """
    monkeypatch.setenv("AGENTMEM_OS_CROSS_LINGUAL", "0")
    kg.ingest_turn("s1", "test-agent", "Sahith Thummala works at Google.")
    kg.ingest_turn("s1", "test-agent", "गूगल की मीटिंग कल है")

    texts = _node_texts(kg)
    assert "Google" in texts
    assert "गूगल" not in texts
    assert _alias_edges(kg) == []


@needs_model
def test_script_token_alias_gates_node_creation(kg):
    kg.ingest_turn("s1", "test-agent", "Sahith Thummala works at Google.")
    kg.ingest_turn("s1", "test-agent", "गूगल की मीटिंग कल है")

    texts = _node_texts(kg)
    from agentmem_os.db.entity_aliases import contains_indic
    indic_nodes = [t for t in texts if contains_indic(t)]

    # गूगल alias-matched Google (0.9506 ≥ τ) so it gated in; the common
    # words की/मीटिंग/कल/है matched nothing ≥ τ and must NOT be nodes.
    assert indic_nodes == ["गूगल"]

    edges = _alias_edges(kg)
    assert len(edges) == 1
    assert edges[0].confidence >= 0.90
    assert edges[0].valid_until is None  # aliases never expire


@needs_model
def test_below_tau_token_is_dropped_entirely(kg):
    kg.ingest_turn("s1", "test-agent", "Sahith Thummala works at Google.")
    kg.ingest_turn("s1", "test-agent", "पानी")  # 0.8526 vs Google — below τ

    assert "पानी" not in _node_texts(kg)
    assert _alias_edges(kg) == []


@needs_model
def test_ner_caught_script_entity_still_gets_alias_edge(kg):
    """
    बेंगलुरु sometimes IS caught by en_core_web_sm as a GPE (verified
    empirically). Whether it arrives via NER (script_caught branch) or via
    the gated-token branch, the ALIAS_OF edge to Bengaluru (0.9256 ≥ τ)
    must exist either way — this assertion is deliberately branch-agnostic
    so spaCy version drift can't break it.
    """
    kg.ingest_turn("s1", "test-agent", "Sahith Thummala works at Google in Bengaluru.")
    kg.ingest_turn("s1", "test-agent", "The user asked about बेंगलुरु yesterday")

    texts = _node_texts(kg)
    assert "बेंगलुरु" in texts

    from agentmem_os.db.models import KnowledgeGraphNode
    db = kg.get_db()
    by_text = {
        n.entity_text: n.id for n in db.query(KnowledgeGraphNode).all()
    }
    db.close()

    pair = {by_text["बेंगलुरु"], by_text["Bengaluru"]}
    assert any({e.source_id, e.target_id} == pair for e in _alias_edges(kg))


@needs_model
def test_hindi_query_reaches_english_memory(kg):
    kg.ingest_turn("s1", "test-agent", "Sahith Thummala works at Google.")

    out = kg.get_relevant_subgraph("गूगल", agent_id="test-agent", top_k=5)
    assert "Google" in out


@needs_model
def test_world_model_serializes_alias_line(kg):
    kg.ingest_turn("s1", "test-agent", "Sahith Thummala works at Google.")
    kg.ingest_turn("s1", "test-agent", "गूगल की मीटिंग कल है")

    # A bare "Google" query gives NER no sentence context (returns no
    # entities → top-entities fallback, pre-existing behavior) — use a
    # query-shaped sentence, verified to NER-extract Google as ORG.
    out = kg.get_relevant_subgraph(
        "What is happening at Google?", agent_id="test-agent", top_k=10
    )
    assert "Aliases:" in out
    assert "गूगल" in out
