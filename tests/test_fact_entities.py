"""
Stage 3 G1: FactEntityLinker + the Stage-3 migration.

Attack surface pinned here (build log, Stage 3 design):
  - node upsert races (Graphiti #1331 class) — real 2-process attack
  - orphaned/unlinked facts recoverable via link_missing (#1083 class)
  - alias gate parity with turn ingestion (Indic surfaces never enter
    the graph unvetted; resolver-off degrades to skip, never to junk)
  - ALIAS_OF traversal in the read path, both directions
  - fact-scope vs node-scope isolation (one agent, many users)
  - migration: dedup-merge on a synthetic-duplicate DB (our dev DB has
    zero dup groups — this path must be exercised by tests, not luck),
    event_status backfill determinism, unrepairable-table refusal
"""
import multiprocessing
import sqlite3
import sys

import pytest

from tests.test_semantic_facts import (
    _bootstrap, _make_engine, _make_production_engine,
)


@pytest.fixture()
def linked_env(tmp_path):
    """File-backed store + linker (file, not memory: the migration and
    race tests need a real path; logic tests share it for one shape)."""
    from agentmem_os.db.fact_entities import FactEntityLinker
    from agentmem_os.db.semantic_facts import SemanticFactStore

    engine = _make_production_engine(tmp_path / "s3.db")
    SessionLocal = _bootstrap(engine)
    return (SemanticFactStore(SessionLocal), FactEntityLinker(SessionLocal),
            SessionLocal, tmp_path / "s3.db")


def _fact(store, text, **kw):
    defaults = dict(fact_type="state", t_mentioned="2023/05/20",
                    source_session_id="sess-1", extraction_model="test-model")
    defaults.update(kw)
    fact, _ = store.add_fact(text, **defaults)
    return fact


# ── link_fact basics ─────────────────────────────────────────────────────────

def test_link_creates_nodes_and_links(linked_env):
    from agentmem_os.db.models import KnowledgeGraphNode, SemanticFactEntity
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google in Paris.")
    rep = linker.link_fact(fact.id, fact_text=fact.fact_text)
    surfaces = {s for s, _, _ in rep["linked"]}
    assert "Google" in surfaces and "Paris" in surfaces
    db = SessionLocal()
    nodes = {n.entity_text for n in db.query(KnowledgeGraphNode).all()}
    links = db.query(SemanticFactEntity).filter(
        SemanticFactEntity.fact_id == fact.id).all()
    db.close()
    assert {"Google", "Paris"} <= nodes
    assert {l.linked_via for l in links} == {"ner"}
    assert all(l.confidence is None for l in links)


def test_existing_node_reused_and_mention_count_untouched(linked_env):
    from agentmem_os.db.models import KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    db.add(KnowledgeGraphNode(agent_id=None, entity_text="Google",
                              entity_type="ORG", mention_count=7))
    db.commit()
    db.close()

    fact = _fact(store, "The user works at Google.")
    linker.link_fact(fact.id, fact_text=fact.fact_text)
    db = SessionLocal()
    rows = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Google").all()
    db.close()
    assert len(rows) == 1                 # reused, not duplicated
    assert rows[0].mention_count == 7     # turn-mention counter untouched


def test_relink_is_idempotent(linked_env):
    from agentmem_os.db.models import SemanticFactEntity
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    r1 = linker.link_fact(fact.id, fact_text=fact.fact_text)
    r2 = linker.link_fact(fact.id, fact_text=fact.fact_text)
    assert len(r1["linked"]) == 1 and r2["linked"] == []
    db = SessionLocal()
    n = db.query(SemanticFactEntity).filter(
        SemanticFactEntity.fact_id == fact.id).count()
    db.close()
    assert n == 1


def test_link_unknown_fact_is_loud(linked_env):
    _, linker, _, _ = linked_env
    with pytest.raises(ValueError, match="not found"):
        linker.link_fact(99999, fact_text="The user works at Google.")


def test_link_requires_text_or_surfaces(linked_env):
    _, linker, _, _ = linked_env
    with pytest.raises(ValueError, match="fact_text or surfaces"):
        linker.link_fact(1)


def test_surface_dedup_casefold_and_length():
    from agentmem_os.db.fact_entities import FactEntityLinker
    out = FactEntityLinker._dedup_surfaces(
        [("Google", "ORG"), ("google", "ORG"), ("G", "ORG"),
         ("  ", "ORG"), ("Paris", "GPE")])
    assert out == [("Google", "ORG"), ("Paris", "GPE")]


def test_same_batch_two_facts_one_new_surface(linked_env):
    # Two facts in ONE caller-owned batch both mention a surface that has
    # no node yet: plans computed BEFORE the batch (B1 contract), first
    # apply creates the node (flushed), the second must SEE the flushed
    # row — one node, two link rows, no constraint trip.
    from agentmem_os.db.models import KnowledgeGraphNode, SemanticFactEntity
    store, linker, SessionLocal, _ = linked_env
    texts = ("The user works at Zomato.",
             "The user ordered lunch from Zomato.")
    plans = [linker.plan_surfaces(linker.extract_surfaces(t)) for t in texts]
    batch = SessionLocal()
    try:
        f1, _ = store.add_fact(texts[0],
                               fact_type="state", t_mentioned="2023/05/20",
                               source_session_id="sess-1",
                               extraction_model="t", db=batch)
        linker.link_fact(f1.id, plan=plans[0], db=batch)
        f2, _ = store.add_fact(texts[1],
                               fact_type="event", t_mentioned="2023/05/20",
                               t_occurred="2023/05/19",
                               source_session_id="sess-1",
                               extraction_model="t", db=batch)
        linker.link_fact(f2.id, plan=plans[1], db=batch)
        batch.commit()
    finally:
        batch.close()
    db = SessionLocal()
    nodes = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Zomato").count()
    links = db.query(SemanticFactEntity).count()
    db.close()
    assert nodes == 1
    assert links == 2


def test_caller_owned_session_refuses_to_plan(linked_env):
    # B1 contract is an API error, not a footnote: planning can load the
    # alias model (~87s cold, measured) under the caller's write lock.
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    batch = SessionLocal()
    try:
        with pytest.raises(ValueError, match="plan="):
            linker.link_fact(fact.id, fact_text=fact.fact_text, db=batch)
    finally:
        batch.close()


def test_apply_with_plan_never_touches_resolver(linked_env, monkeypatch):
    # B1 tripwire: applying a precomputed plan inside a batch must never
    # consult the resolver — if this fires, model compute is back under
    # the write lock.
    from agentmem_os.db import fact_entities as fe
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "उपयोगकर्ता गूगल में काम करता है।")
    plan = {"agent_id": None,
            "plan": [("गूगल", "FOREIGN", "ner", None, None)],
            "skipped": []}

    def _explode():
        raise AssertionError("resolver consulted during apply")

    monkeypatch.setattr(fe, "get_alias_resolver", _explode)
    batch = SessionLocal()
    try:
        # a write first, so the batch holds the lock like the engine does
        store.add_fact("The user lives in Pune.", fact_type="state",
                       t_mentioned="2023/05/20", source_session_id="sess-1",
                       extraction_model="t", db=batch)
        rep = linker.link_fact(fact.id, plan=plan, db=batch)
        batch.commit()
    finally:
        batch.close()
    assert [s for s, _, _ in rep["linked"]] == ["गूगल"]


# ── Alias gate ───────────────────────────────────────────────────────────────

def test_indic_surface_skipped_when_resolver_off(linked_env, monkeypatch):
    from agentmem_os.db import entity_aliases
    monkeypatch.setenv("AGENTMEM_OS_CROSS_LINGUAL", "0")
    entity_aliases.reset_alias_resolver()
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    rep = linker.link_fact(
        fact.id, surfaces=[("Google", "ORG"), ("गूगल", "FOREIGN")])
    entity_aliases.reset_alias_resolver()
    assert [s for s, _, _ in rep["linked"]] == ["Google"]
    assert rep["skipped"] == [("गूगल", "indic surface, alias resolver unavailable")]


def test_indic_surface_with_exact_node_links_without_resolver(
        linked_env, monkeypatch):
    # A stored Indic node already passed the gate when it was created —
    # linking to it must not require the resolver at all.
    from agentmem_os.db import entity_aliases
    from agentmem_os.db.models import KnowledgeGraphNode
    monkeypatch.setenv("AGENTMEM_OS_CROSS_LINGUAL", "0")
    entity_aliases.reset_alias_resolver()
    store, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    db.add(KnowledgeGraphNode(agent_id=None, entity_text="गूगल",
                              entity_type="FOREIGN", mention_count=2))
    db.commit()
    db.close()
    fact = _fact(store, "The user works at Google.")
    rep = linker.link_fact(fact.id, surfaces=[("गूगल", "FOREIGN")])
    entity_aliases.reset_alias_resolver()
    assert [(s, v) for s, _, v in rep["linked"]] == [("गूगल", "ner")]
    assert rep["skipped"] == []


class _FakeResolver:
    """Discriminating fake: only the true alias pair clears τ — a fake
    that matches EVERY Hindi token would hide over-admission bugs (and
    misrepresent the measured resolver, whose whole point is that common
    words do NOT clear τ)."""
    enabled = True

    @staticmethod
    def find_aliases(text, candidates):
        if text == "गूगल" and "Google" in candidates:
            return [("Google", 0.93)]
        return []


def test_indic_surface_admitted_through_alias_gate(linked_env, monkeypatch):
    # Resolver behavior is pinned deterministically here (the REAL model
    # runs in the G2 smoke): an Indic surface matching a known entity at
    # τ gets a FOREIGN node, an 'alias' link carrying the similarity,
    # and a non-destructive ALIAS_OF edge to the anchor.
    from agentmem_os.db import fact_entities as fe
    from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode
    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _FakeResolver())
    store, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    db.add(KnowledgeGraphNode(agent_id=None, entity_text="Google",
                              entity_type="ORG", mention_count=3))
    db.commit()
    db.close()
    fact = _fact(store, "उपयोगकर्ता गूगल में काम करता है।")
    rep = linker.link_fact(
        fact.id, surfaces=[("गूगल", "FOREIGN")])
    assert [(s, v) for s, _, v in rep["linked"]] == [("गूगल", "alias")]
    assert rep["alias_edges"] == 1
    db = SessionLocal()
    edge = db.query(KnowledgeGraphEdge).filter(
        KnowledgeGraphEdge.relation_type == "ALIAS_OF").one()
    node = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "गूगल").one()
    db.close()
    assert edge.confidence == pytest.approx(0.93)
    assert node.entity_type == "FOREIGN"


def test_unmatched_indic_surface_never_enters_graph(linked_env, monkeypatch):
    from agentmem_os.db import fact_entities as fe
    from agentmem_os.db.models import KnowledgeGraphNode

    class _NoMatch:
        enabled = True

        @staticmethod
        def find_aliases(text, candidates):
            return []

    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _NoMatch())
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    rep = linker.link_fact(fact.id, surfaces=[("मीटिंग", "FOREIGN")])
    assert rep["linked"] == []
    assert rep["skipped"] == [("मीटिंग", "indic surface, no alias anchor at tau")]
    db = SessionLocal()
    assert db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "मीटिंग").count() == 0
    db.close()


# ── Read path ────────────────────────────────────────────────────────────────

def test_facts_for_entity_timeline_and_liveness(linked_env):
    store, linker, SessionLocal, _ = linked_env
    e1 = _fact(store, "The user visited the Google office.",
               fact_type="event", t_occurred="2023/03/10")
    e2 = _fact(store, "The user interviewed at Google.",
               fact_type="event", t_occurred="2023/01/05")
    s1 = _fact(store, "The user works at Google.")
    s2 = _fact(store, "The user is on sabbatical from Google.")
    for f in (e1, e2, s1, s2):
        linker.link_fact(f.id, fact_text=f.fact_text)
    store.supersede(s1.id, s2.id)

    got = linker.facts_for_entity("Google")
    ids = [f.id for f in got]
    assert s1.id not in ids                      # superseded fact is dead
    assert ids.index(e2.id) < ids.index(e1.id)   # timeline order
    only_events = linker.facts_for_entity("Google", fact_type="event")
    assert {f.id for f in only_events} == {e1.id, e2.id}


def test_facts_for_entity_scope_isolation(linked_env):
    # One agent, two users: same entity node axis, but user B must never
    # see user A's facts through the entity index.
    store, linker, SessionLocal, _ = linked_env
    fa = _fact(store, "The user works at Google.", user_id="user-A")
    linker.link_fact(fa.id, fact_text=fa.fact_text)
    assert [f.id for f in linker.facts_for_entity(
        "Google", user_id="user-A")] == [fa.id]
    assert linker.facts_for_entity("Google", user_id="user-B") == []
    assert linker.facts_for_entity("Google") == []   # global scope ≠ user scope


def test_facts_for_entity_follows_alias_edges_both_directions(linked_env):
    from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    linker.link_fact(fact.id, fact_text=fact.fact_text)
    db = SessionLocal()
    google = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Google").one()
    hindi = KnowledgeGraphNode(agent_id=None, entity_text="गूगल",
                               entity_type="FOREIGN", mention_count=1)
    db.add(hindi)
    db.flush()
    a, b = sorted([google.id, hindi.id])
    db.add(KnowledgeGraphEdge(source_id=a, target_id=b, weight=1.0,
                              relation_type="ALIAS_OF", confidence=0.93))
    db.commit()
    db.close()

    via_alias = linker.facts_for_entity("गूगल")
    assert [f.id for f in via_alias] == [fact.id]
    assert linker.facts_for_entity("गूगल", follow_aliases=False) == []


def test_facts_for_entity_unknown_entity_empty(linked_env):
    _, linker, _, _ = linked_env
    assert linker.facts_for_entity("Nonexistent") == []


# ── Recovery sweep ───────────────────────────────────────────────────────────

def test_link_missing_sweeps_unlinked_facts(linked_env):
    from agentmem_os.db.models import SemanticFactEntity
    store, linker, SessionLocal, _ = linked_env
    f1 = _fact(store, "The user works at Google.")
    f2 = _fact(store, "The user lives in Paris.")
    rep = linker.link_missing()
    assert rep["swept"] == 2 and rep["failures"] == []
    assert rep["links_created"] == 2
    db = SessionLocal()
    linked_fact_ids = {r[0] for r in db.query(SemanticFactEntity.fact_id).all()}
    db.close()
    assert linked_fact_ids == {f1.id, f2.id}
    # Idempotent: already-linked facts are no longer candidates.
    assert linker.link_missing()["candidates"] == 0


def test_link_missing_survives_a_poison_fact(linked_env, monkeypatch):
    # One fact that blows up during linking must not take the sweep down
    # (the sweep IS the recovery path — it cannot itself be fragile).
    store, linker, SessionLocal, _ = linked_env
    bad = _fact(store, "The user works at Google.")
    good = _fact(store, "The user lives in Paris.")
    real = linker.link_fact

    def _explode(fid, **kw):
        if fid == bad.id:
            raise RuntimeError("boom")
        return real(fid, **kw)

    monkeypatch.setattr(linker, "link_fact", _explode)
    rep = linker.link_missing()
    assert rep["swept"] == 1
    assert rep["failures"] == [(bad.id, "boom")]
    monkeypatch.undo()
    rep2 = linker.link_missing()      # the failed fact is still sweepable
    assert rep2["swept"] == 1 and rep2["failures"] == []


# ── Race classifier ──────────────────────────────────────────────────────────

def test_race_classifier_only_matches_stage3_guards():
    from sqlalchemy.exc import IntegrityError
    from agentmem_os.db.fact_entities import _is_link_race

    def _err(msg):
        return IntegrityError("stmt", {}, sqlite3.IntegrityError(msg))

    assert _is_link_race(_err("UNIQUE constraint failed: index 'uq_kg_nodes_scope_text'"))
    assert _is_link_race(_err(
        "UNIQUE constraint failed: semantic_fact_entities.fact_id, "
        "semantic_fact_entities.node_id"))
    assert not _is_link_race(_err("FOREIGN KEY constraint failed"))
    assert not _is_link_race(_err(
        "NOT NULL constraint failed: semantic_fact_entities.surface_text"))
    assert not _is_link_race(_err(
        "UNIQUE constraint failed: semantic_facts.scope_key, "
        "semantic_facts.normalized_hash"))


# ── Cross-process node race (Graphiti #1331 class) ───────────────────────────

def _race_worker(db_path, fact_id, barrier_path):
    import os
    import time
    os.environ["AGENTMEM_OS_DB_PATH"] = db_path

    from sqlalchemy.orm import sessionmaker
    from tests.test_semantic_facts import _make_production_engine
    from agentmem_os.db.fact_entities import FactEntityLinker

    engine = _make_production_engine(db_path)
    SessionLocal = sessionmaker(bind=engine, autocommit=False,
                                autoflush=False, expire_on_commit=False)
    linker = FactEntityLinker(SessionLocal)
    while not os.path.exists(barrier_path):
        time.sleep(0.005)
    linker.link_fact(fact_id, surfaces=[("Zomato", "ORG")])
    sys.exit(0)


def test_cross_process_node_creation_race(linked_env, tmp_path):
    # Two OS processes link two different facts to the SAME brand-new
    # surface at once. The unique index is the authority: exactly ONE
    # node row survives and BOTH facts hold a link to it.
    from agentmem_os.db.models import KnowledgeGraphNode, SemanticFactEntity
    store, linker, SessionLocal, db_path = linked_env
    f1 = _fact(store, "The user works at Zomato.")
    f2 = _fact(store, "The user ordered from Zomato.",
               fact_type="event", t_occurred="2023/05/19")

    barrier = tmp_path / "go"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_race_worker,
                         args=(str(db_path), fid, str(barrier)))
             for fid in (f1.id, f2.id)]
    for p in procs:
        p.start()
    barrier.write_text("go")
    for p in procs:
        p.join(timeout=60)
    assert all(p.exitcode == 0 for p in procs), \
        [p.exitcode for p in procs]

    db = SessionLocal()
    nodes = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Zomato").all()
    links = db.query(SemanticFactEntity).count()
    db.close()
    assert len(nodes) == 1
    assert links == 2


# ── Migration ────────────────────────────────────────────────────────────────

_OLD_SCHEMA = """
CREATE TABLE kg_nodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT, agent_id VARCHAR,
  session_id VARCHAR, entity_text VARCHAR NOT NULL,
  entity_type VARCHAR NOT NULL, mention_count INTEGER,
  first_seen DATETIME, last_seen DATETIME, last_confirmed_at DATETIME);
CREATE TABLE kg_edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source_id INTEGER NOT NULL,
  target_id INTEGER NOT NULL, weight FLOAT, session_id VARCHAR,
  last_updated DATETIME, relation_type VARCHAR, confidence FLOAT,
  valid_from DATETIME, valid_until DATETIME, superseded_by INTEGER);
CREATE TABLE semantic_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, scope_key VARCHAR NOT NULL,
  fact_text TEXT NOT NULL, fact_type VARCHAR NOT NULL,
  t_occurred VARCHAR, t_occurred_end VARCHAR, t_mentioned VARCHAR NOT NULL,
  normalized_hash VARCHAR NOT NULL,
  UNIQUE (scope_key, normalized_hash));
CREATE TABLE consolidation_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id VARCHAR,
  turns_processed INTEGER, summaries_generated INTEGER);
"""


def _old_db(tmp_path, extra_sql=""):
    p = tmp_path / "old.db"
    conn = sqlite3.connect(p)
    conn.executescript(_OLD_SCHEMA + extra_sql)
    conn.commit()
    conn.close()
    return p


def _run_stage3_migration(monkeypatch, db_path):
    from agentmem_os.db import engine as engine_module
    monkeypatch.setattr(engine_module, "DB_PATH", str(db_path))
    return engine_module._migrate_stage3()


def test_migration_adds_columns_backfills_and_indexes(tmp_path, monkeypatch):
    p = _old_db(tmp_path, """
      INSERT INTO semantic_facts (scope_key, fact_text, fact_type,
        t_occurred, t_mentioned, normalized_hash) VALUES
        ('global','future event','event','2024/04/15','2023/05/20','h1'),
        ('global','past event','event','2023/01/01','2023/05/20','h2'),
        ('global','undated event','event',NULL,'2023/05/20','h3'),
        ('global','a preference','preference',NULL,'2023/05/20','h4');
    """)
    report = _run_stage3_migration(monkeypatch, p)
    assert "backfilled 3 event rows" in report["event_status"]
    assert report["kg_nodes_unique"] == "created"
    conn = sqlite3.connect(p)
    rows = dict(conn.execute(
        "SELECT fact_text, event_status FROM semantic_facts"))
    log_cols = [c[1] for c in conn.execute(
        "PRAGMA table_info(consolidation_log)")]
    idx = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' "
        "AND name='uq_kg_nodes_scope_text'").fetchone()
    conn.close()
    assert rows == {"future event": "planned", "past event": "occurred",
                    "undated event": "occurred", "a preference": None}
    assert "entities_linked" in log_cols
    assert idx is not None
    # Idempotent second run.
    report2 = _run_stage3_migration(monkeypatch, p)
    assert "backfilled 0 event rows" in report2["event_status"]
    assert report2["kg_nodes_unique"] == "verified"


def test_migration_merges_duplicate_nodes(tmp_path, monkeypatch):
    # PRODUCTION-shaped fixture (G3 R1 B2): real CO_OCCURS rows carry
    # the STRING 'CO_OCCURS' (ORM default — 34,905/34,905 in the dev
    # DB); a NULL-only fixture rubber-stamped a merge that was inert on
    # real data. One NULL row stays in the mix so BOTH storage shapes
    # merge as one family.
    p = _old_db(tmp_path, """
      INSERT INTO kg_nodes (id, agent_id, entity_text, entity_type,
        mention_count, last_seen) VALUES
        (1, NULL, 'Google', 'ORG', 3, '2023-01-01'),
        (2, NULL, 'Google', 'ORG', 5, '2023-06-01'),
        (3, NULL, 'Paris',  'GPE', 2, '2023-01-01'),
        (4, 'agent-A', 'Google', 'ORG', 9, '2023-01-01');
      INSERT INTO kg_edges (id, source_id, target_id, weight, relation_type) VALUES
        (10, 1, 3, 2.0, 'CO_OCCURS'),
        (11, 2, 3, 4.0, NULL),
        (12, 1, 2, 9.0, 'CO_OCCURS');   -- becomes a self-loop, must be dropped
      -- typed edge from the dup node: re-pointed, kept as-is
      INSERT INTO kg_edges (id, source_id, target_id, weight,
        relation_type, valid_from) VALUES
        (13, 2, 3, 1.0, 'WORKS_AT', '2023-06-01');
    """)
    report = _run_stage3_migration(monkeypatch, p)
    assert "after merge" in report["kg_nodes_unique"]
    conn = sqlite3.connect(p)
    nodes = conn.execute(
        "SELECT id, agent_id, entity_text, mention_count "
        "FROM kg_nodes ORDER BY id").fetchall()
    co = conn.execute(
        "SELECT source_id, target_id, weight FROM kg_edges "
        "WHERE relation_type IS NULL OR relation_type = 'CO_OCCURS'").fetchall()
    typed = conn.execute(
        "SELECT source_id, target_id, relation_type FROM kg_edges "
        "WHERE relation_type = 'WORKS_AT'").fetchall()
    conn.close()
    # Keeper = min id; scoped duplicate (agent-A) untouched; counts summed.
    assert nodes == [(1, None, "Google", 8), (3, None, "Paris", 2),
                     (4, "agent-A", "Google", 9)]
    # Google↔Paris CO_OCCURS rows merged ACROSS shapes with SUMMED
    # weight; the self-loop is gone.
    assert co == [(1, 3, 6.0)]
    assert typed == [(1, 3, "WORKS_AT")]


def test_migration_rebuilds_wrong_shape_index(tmp_path, monkeypatch):
    # G3 R1 M1: a scope-blind UNIQUE(entity_text) impostor with the
    # right NAME previously read as "verified" while permanently
    # rejecting a second agent's nodes. Verify-by-SQL rebuilds it.
    p = _old_db(tmp_path, """
      CREATE UNIQUE INDEX uq_kg_nodes_scope_text ON kg_nodes(entity_text);
    """)
    report = _run_stage3_migration(monkeypatch, p)
    assert "rebuilt (wrong shape)" in report["kg_nodes_unique"]
    conn = sqlite3.connect(p)
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='uq_kg_nodes_scope_text'"
    ).fetchone()[0]
    # And it now actually enforces scope-aware identity:
    conn.execute("INSERT INTO kg_nodes (agent_id, entity_text, entity_type)"
                 " VALUES ('a', 'Google', 'ORG')")
    conn.execute("INSERT INTO kg_nodes (agent_id, entity_text, entity_type)"
                 " VALUES ('b', 'Google', 'ORG')")   # different scope: OK
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO kg_nodes (agent_id, entity_text, "
                     "entity_type) VALUES ('a', 'Google', 'ORG')")
    conn.close()
    assert "coalesce(agent_id, '')" in sql
    # Second run: verified, not rebuilt again.
    assert _run_stage3_migration(monkeypatch, p)["kg_nodes_unique"] == "verified"


def test_migration_refuses_unrepairable_link_table(tmp_path, monkeypatch):
    p = _old_db(tmp_path, """
      CREATE TABLE semantic_fact_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact_id INTEGER NOT NULL, node_id INTEGER NOT NULL,
        surface_text VARCHAR NOT NULL);
    """)
    with pytest.raises(RuntimeError, match="WITHOUT a UNIQUE index"):
        _run_stage3_migration(monkeypatch, p)


def test_migration_honest_about_absent_tables(tmp_path, monkeypatch):
    p = tmp_path / "empty.db"
    sqlite3.connect(p).close()
    report = _run_stage3_migration(monkeypatch, p)
    assert "absent" in report["consolidation_log"]
    assert "absent" in report["event_status"]
    assert "absent" in report["semantic_fact_entities"]
    assert "absent" in report["kg_nodes_unique"]


# ── Coverage of the store-owned failure paths ────────────────────────────────

def _race_error():
    from sqlalchemy.exc import IntegrityError
    return IntegrityError(
        "stmt", {},
        sqlite3.IntegrityError(
            "UNIQUE constraint failed: index 'uq_kg_nodes_scope_text'"))


def test_store_owned_race_retries_once_then_succeeds(linked_env, monkeypatch):
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    real = linker._link_in_session
    calls = {"n": 0}

    def _lose_first(session, owns, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _race_error()
        return real(session, owns, *a, **k)

    monkeypatch.setattr(linker, "_link_in_session", _lose_first)
    rep = linker.link_fact(fact.id, fact_text=fact.fact_text)
    assert calls["n"] == 2
    assert [s for s, _, _ in rep["linked"]] == ["Google"]


def test_store_owned_second_race_loss_stays_loud(linked_env, monkeypatch):
    # One retry, never a loop (Stage-1 lesson): two losses in a row is a
    # bug signature, not a race pattern.
    from sqlalchemy.exc import IntegrityError
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    monkeypatch.setattr(linker, "_link_in_session",
                        lambda *a, **k: (_ for _ in ()).throw(_race_error()))
    with pytest.raises(IntegrityError):
        linker.link_fact(fact.id, fact_text=fact.fact_text)


def test_store_owned_non_race_integrity_error_not_retried(
        linked_env, monkeypatch):
    from sqlalchemy.exc import IntegrityError
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    calls = {"n": 0}

    def _fk_violation(*a, **k):
        calls["n"] += 1
        raise IntegrityError("stmt", {},
                             sqlite3.IntegrityError("FOREIGN KEY constraint failed"))

    monkeypatch.setattr(linker, "_link_in_session", _fk_violation)
    with pytest.raises(IntegrityError):
        linker.link_fact(fact.id, fact_text=fact.fact_text)
    assert calls["n"] == 1                     # a real failure never retries


def test_alias_edge_idempotent_and_junk_anchor_defended(
        linked_env, monkeypatch):
    # Relinking the same alias pair adds no second edge; a resolver that
    # returns an anchor with no node (junk) links the surface but writes
    # no edge — the resolver's output is never trusted blindly.
    from agentmem_os.db import fact_entities as fe
    from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    db.add(KnowledgeGraphNode(agent_id=None, entity_text="Google",
                              entity_type="ORG", mention_count=1))
    db.commit()
    db.close()
    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _FakeResolver())
    f1 = _fact(store, "उपयोगकर्ता गूगल में काम करता है।")
    f2 = _fact(store, "उपयोगकर्ता गूगल पर निर्भर है।")
    r1 = linker.link_fact(f1.id, surfaces=[("गूगल", "FOREIGN")])
    r2 = linker.link_fact(f2.id, surfaces=[("गूगल", "FOREIGN")])
    assert r1["alias_edges"] == 1 and r2["alias_edges"] == 0
    db = SessionLocal()
    assert db.query(KnowledgeGraphEdge).filter(
        KnowledgeGraphEdge.relation_type == "ALIAS_OF").count() == 1
    db.close()

    class _JunkAnchor:
        enabled = True

        @staticmethod
        def find_aliases(text, candidates):
            return [("Ghost", 0.99)]

    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _JunkAnchor())
    f3 = _fact(store, "उपयोगकर्ता ज़ोमैटो से खाना मंगाता है।")
    r3 = linker.link_fact(f3.id, surfaces=[("ज़ोमैटो", "FOREIGN")])
    # G3 R2 M1 hardening: an anchor that does not exist at apply time
    # (junk resolver output, or deleted between plan and apply) now
    # SKIPS the surface entirely — admitting it would create an ungated
    # Indic node, the exact class the gate exists to keep out.
    assert r3["linked"] == []
    assert r3["skipped"] == [("ज़ोमैटो", "alias anchor not found at apply")]
    assert r3["alias_edges"] == 0
    from agentmem_os.db.models import KnowledgeGraphNode as _KGN
    db = SessionLocal()
    assert db.query(_KGN).filter(
        _KGN.entity_text == "ज़ोमैटो").count() == 0
    db.close()


def test_read_side_alias_seed_for_indic_query(linked_env, monkeypatch):
    # facts_for_entity with an Indic query and NO exact node: the alias
    # resolver seeds the best-matched node — a Hindi query reaches facts
    # linked under the English surface (the read-side Gate-E payoff).
    from agentmem_os.db import fact_entities as fe
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    linker.link_fact(fact.id, fact_text=fact.fact_text)
    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _FakeResolver())
    assert [f.id for f in linker.facts_for_entity("गूगल")] == [fact.id]

    class _Off:
        enabled = False

    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _Off())
    assert linker.facts_for_entity("गूगल") == []


def test_facts_for_entity_caller_owned_session(linked_env):
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    linker.link_fact(fact.id, fact_text=fact.fact_text)
    db = SessionLocal()
    try:
        got = linker.facts_for_entity("Google", db=db)
        assert [f.id for f in got] == [fact.id]
    finally:
        db.close()


def test_no_entity_fact_links_nothing(linked_env):
    store, linker, _, _ = linked_env
    fact = _fact(store, "The user prefers quiet mornings.")
    rep = linker.link_fact(fact.id, fact_text=fact.fact_text)
    assert rep == {"linked": [], "skipped": [], "alias_edges": 0}


def test_ensure_alias_edge_double_call_writes_once(linked_env):
    # Direct double-ensure of the same pair (the cross-process-race
    # double-check): second call sees the edge and declines.
    from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    google = KnowledgeGraphNode(agent_id=None, entity_text="Google",
                                entity_type="ORG", mention_count=1)
    hindi = KnowledgeGraphNode(agent_id=None, entity_text="गूगल",
                               entity_type="FOREIGN", mention_count=1)
    db.add_all([google, hindi])
    db.flush()
    first = linker._ensure_alias_edge(db, hindi, google, 0.93, None)
    second = linker._ensure_alias_edge(db, hindi, google, 0.93, None)
    db.commit()
    n = db.query(KnowledgeGraphEdge).filter(
        KnowledgeGraphEdge.relation_type == "ALIAS_OF").count()
    db.close()
    assert (first, second, n) == (1, 0, 1)


def test_extract_surfaces_sees_devanagari_tokens():
    # en_core_web_sm is blind to Devanagari — without the script-token
    # augmentation a Hindi fact text gets ZERO surfaces and the alias
    # gate is never consulted (Gate-E write path dead on arrival).
    from agentmem_os.db.fact_entities import FactEntityLinker
    got = FactEntityLinker.extract_surfaces(
        "उपयोगकर्ता गूगल में काम करता है।")
    assert ("गूगल", "FOREIGN") in got


def test_hindi_fact_links_through_alias_gate_end_to_end(
        linked_env, monkeypatch):
    # Full write path for a Hindi fact: script token extracted, admitted
    # through the (pinned) alias gate, FOREIGN node + link + ALIAS_OF
    # edge — and readable from BOTH surfaces afterwards.
    from agentmem_os.db import fact_entities as fe
    from agentmem_os.db.models import KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    db.add(KnowledgeGraphNode(agent_id=None, entity_text="Google",
                              entity_type="ORG", mention_count=1))
    db.commit()
    db.close()
    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _FakeResolver())
    fact = _fact(store, "उपयोगकर्ता गूगल में काम करता है।")
    rep = linker.link_fact(fact.id, fact_text=fact.fact_text)
    assert [(s, v) for s, _, v in rep["linked"]] == [("गूगल", "alias")]
    assert rep["alias_edges"] == 1
    assert [f.id for f in linker.facts_for_entity("गूगल")] == [fact.id]
    assert [f.id for f in linker.facts_for_entity("Google")] == [fact.id]


# ── G3 R1 fixes: cursor sweep, alias closure, case reads, danda tokens ──────

def test_sweep_cursor_defeats_zero_surface_starvation(linked_env):
    # B3 repro pinned: 12 zero-surface facts ahead of 1 linkable one at
    # limit=10 — the cursor drain must reach the linkable fact and the
    # drain loop must terminate.
    from agentmem_os.db.models import SemanticFactEntity
    store, linker, SessionLocal, _ = linked_env
    for i in range(12):
        _fact(store, f"The user prefers quiet mornings variant {i:02d}.")
    target = _fact(store, "The user works at Google.")

    rounds, cursor, total_linked = 0, 0, 0
    while True:
        r = linker.link_missing(limit=10, after_id=cursor)
        if r["candidates"] == 0:
            break
        cursor = r["next_after_id"]
        total_linked += r["links_created"]
        rounds += 1
        assert rounds < 10   # termination guard — the old shape looped forever
    assert total_linked == 1
    db = SessionLocal()
    linked_ids = {r[0] for r in db.query(SemanticFactEntity.fact_id).all()}
    db.close()
    assert target.id in linked_ids


def test_alias_closure_reaches_chained_variants(linked_env):
    # M3 repro pinned: चेन्नई—चेन्नई।—Chennai chain. One-hop returned a
    # SUBSET of the entity's facts; the closure returns all of them
    # from every surface in the chain.
    from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    f_en = _fact(store, "The user visited Chennai.",
                 fact_type="event", t_occurred="2023/02/01")
    linker.link_fact(f_en.id, fact_text=f_en.fact_text)
    db = SessionLocal()
    chennai = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Chennai").one()
    v1 = KnowledgeGraphNode(agent_id=None, entity_text="चेन्नई",
                            entity_type="FOREIGN", mention_count=1)
    v2 = KnowledgeGraphNode(agent_id=None, entity_text="चेन्नई।",
                            entity_type="FOREIGN", mention_count=1)
    db.add_all([v1, v2])
    db.flush()
    # chain: v2 = v1, v1 = Chennai (no direct v2-Chennai edge)
    for a, b in ((v2.id, v1.id), (v1.id, chennai.id)):
        lo, hi = sorted((a, b))
        db.add(KnowledgeGraphEdge(source_id=lo, target_id=hi, weight=1.0,
                                  relation_type="ALIAS_OF", confidence=0.95))
    db.commit()
    hindi_fact = _fact(store, "उपयोगकर्ता चेन्नई। गए थे।")
    db = SessionLocal()
    from agentmem_os.db.models import SemanticFactEntity
    db.add(SemanticFactEntity(fact_id=hindi_fact.id, node_id=v2.id,
                              surface_text="चेन्नई।", linked_via="alias",
                              confidence=0.97))
    db.commit()
    db.close()
    for surface in ("Chennai", "चेन्नई", "चेन्नई।"):
        got = {f.id for f in linker.facts_for_entity(surface)}
        assert got == {f_en.id, hindi_fact.id}, surface
    # follow_aliases=False still means THIS surface only
    assert {f.id for f in linker.facts_for_entity("Chennai",
                                                  follow_aliases=False)} \
        == {f_en.id}


def test_case_insensitive_read_side(linked_env):
    # M4: node identity stays case-sensitive (turn-path parity) but the
    # READ side must not go blind over casing — 1.5% of real KG groups
    # are case variants.
    from agentmem_os.db.models import KnowledgeGraphNode, SemanticFactEntity
    store, linker, SessionLocal, _ = linked_env
    f1 = _fact(store, "The user works at Google.")
    linker.link_fact(f1.id, fact_text=f1.fact_text)
    f2 = _fact(store, "The user filed a google support ticket.",
               fact_type="event", t_occurred="2023/03/03")
    db = SessionLocal()
    variant = KnowledgeGraphNode(agent_id=None, entity_text="google",
                                 entity_type="ORG", mention_count=1)
    db.add(variant)
    db.flush()
    db.add(SemanticFactEntity(fact_id=f2.id, node_id=variant.id,
                              surface_text="google", linked_via="ner"))
    db.commit()
    db.close()
    for q in ("google", "Google", "GOOGLE"):
        assert {f.id for f in linker.facts_for_entity(q)} == {f1.id, f2.id}, q


def test_undated_facts_sort_last_in_timeline(linked_env):
    # Explicit nullslast tripwire (G3 R1 minor: deleting .nullslast()
    # left the suite green — now it can't).
    store, linker, SessionLocal, _ = linked_env
    undated = _fact(store, "The user works at Google.")
    dated = _fact(store, "The user visited the Google campus.",
                  fact_type="event", t_occurred="2023/04/01")
    for f in (undated, dated):
        linker.link_fact(f.id, fact_text=f.fact_text)
    got = [f.id for f in linker.facts_for_entity("Google")]
    assert got == [dated.id, undated.id]   # undated LAST, not first


def test_danda_never_part_of_script_tokens():
    # G3 R1 minor→class fix: danda/double-danda are punctuation inside
    # the Indic block; "है।" tokenizing WITH its danda cost 22-43% of τ
    # margin and manufactured sentence-final variant nodes (M3's cause).
    from agentmem_os.db.entity_aliases import EntityAliasResolver
    toks = EntityAliasResolver.extract_script_tokens(
        "उपयोगकर्ता गूगल में काम करता है। चेन्नई॥")
    assert "है" in toks and "चेन्नई" in toks
    assert all("।" not in t and "॥" not in t for t in toks)


def test_migration_repairs_duplicate_co_occurs_pairs(tmp_path, monkeypatch):
    # The turn-path lookup bug (relation_type IS NULL vs the ORM's
    # 'CO_OCCURS' default) created a NEW edge per co-occurrence; the
    # loader then kept only the last row's weight. The migration merges
    # the damage globally: weights summed, idempotent, typed edges and
    # non-duplicate pairs untouched.
    p = _old_db(tmp_path, """
      INSERT INTO kg_nodes (id, entity_text, entity_type) VALUES
        (1, 'Google', 'ORG'), (2, 'Paris', 'GPE'), (3, 'Alice', 'PERSON');
      INSERT INTO kg_edges (id, source_id, target_id, weight, relation_type) VALUES
        (10, 1, 2, 1.0, 'CO_OCCURS'),
        (11, 1, 2, 1.0, 'CO_OCCURS'),
        (12, 1, 2, 1.0, NULL),
        (13, 1, 3, 5.0, 'CO_OCCURS'),
        (14, 1, 2, 1.0, 'WORKS_AT');
    """)
    report = _run_stage3_migration(monkeypatch, p)
    assert report["co_occurs_dedup"] == "merged 1 duplicate pairs"
    conn = sqlite3.connect(p)
    co = conn.execute(
        "SELECT id, weight FROM kg_edges WHERE source_id=1 AND target_id=2 "
        "AND (relation_type IS NULL OR relation_type='CO_OCCURS')").fetchall()
    untouched = conn.execute(
        "SELECT weight FROM kg_edges WHERE id IN (13, 14) ORDER BY id").fetchall()
    conn.close()
    assert co == [(10, 3.0)]                  # both shapes summed into min id
    assert untouched == [(5.0,), (1.0,)]      # single pair + typed edge intact
    assert _run_stage3_migration(monkeypatch, p)["co_occurs_dedup"] == "clean"


def test_ingest_turn_increments_instead_of_duplicating(linked_env):
    # Tripwire for the turn-path fix itself: two ingests of the same
    # co-occurring pair must yield ONE edge with weight 2, not two rows
    # (the pre-fix behavior — 1,979 duplicate pairs in the real dev DB).
    from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph
    from agentmem_os.db.models import KnowledgeGraphEdge
    _, _, SessionLocal, _ = linked_env
    kg = EntityKnowledgeGraph(SessionLocal)
    kg.ingest_turn("sess-1", None, "Alice met Bob at Google in Paris.")
    kg.ingest_turn("sess-1", None, "Alice met Bob at Google in Paris.")
    db = SessionLocal()
    rows = db.query(KnowledgeGraphEdge).filter(
        (KnowledgeGraphEdge.relation_type.is_(None))
        | (KnowledgeGraphEdge.relation_type == "CO_OCCURS")).all()
    db.close()
    pairs = {(e.source_id, e.target_id) for e in rows}
    assert len(rows) == len(pairs)            # no duplicate pair rows
    assert all(e.weight == 2.0 for e in rows)  # increments, not clones


# ── G3 R2 fixes: unicode reads, plan validation, inverted-pair merge ─────────

def test_non_ascii_exact_read_not_lost_to_sqlite_lower(linked_env):
    # G3 R2 B1 pinned: SQLite lower() folds ASCII only — the
    # lower()==lower() seed alone turned a byte-identical 'Übermensch'
    # query into []. The exact arm must carry non-ASCII reads.
    from agentmem_os.db.models import KnowledgeGraphNode, SemanticFactEntity
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user admires the Übermensch concept.")
    db = SessionLocal()
    node = KnowledgeGraphNode(agent_id=None, entity_text="Übermensch",
                              entity_type="WORK_OF_ART", mention_count=1)
    db.add(node)
    db.flush()
    db.add(SemanticFactEntity(fact_id=fact.id, node_id=node.id,
                              surface_text="Übermensch", linked_via="ner"))
    db.commit()
    db.close()
    assert [f.id for f in linker.facts_for_entity("Übermensch")] == [fact.id]
    # Non-ASCII case folding is NOT claimed (disclosed ASCII-only):
    # SQLite lower() leaves Ü untouched, Python folds it — the lowercase
    # query matches neither arm. This assertion pins the DISCLOSURE.
    assert linker.facts_for_entity("übermensch") == []


def test_plan_scope_and_shape_validated_at_apply(linked_env):
    # G3 R2 M1: apply validated nothing — wrong-scope plans wrote into
    # the wrong agent's graph; malformed via landed in linked_via.
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    good = linker.plan_surfaces([("Google", "ORG")], agent_id="agent-A")
    with pytest.raises(ValueError, match="scopes must match"):
        linker.link_fact(fact.id, plan=good, agent_id="agent-Z")
    with pytest.raises(ValueError, match="plan_surfaces"):
        linker.link_fact(fact.id, plan=([("Google", "ORG", "ner", None,
                                          None)], []))
    with pytest.raises(ValueError, match="malformed plan entry"):
        linker.link_fact(fact.id, plan={"agent_id": None, "skipped": [],
                                        "plan": [("G", "ORG", "TYPO",
                                                  None, None)]})


def test_stale_alias_plan_skips_instead_of_ungated_node(linked_env,
                                                        monkeypatch):
    # G3 R2 M1 repro pinned: anchor deleted between plan and apply →
    # the surface must be SKIPPED, not admitted as an ungated node.
    from agentmem_os.db import fact_entities as fe
    from agentmem_os.db.models import KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    db.add(KnowledgeGraphNode(agent_id=None, entity_text="Google",
                              entity_type="ORG", mention_count=1))
    db.commit()
    db.close()
    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _FakeResolver())
    plan = linker.plan_surfaces([("गूगल", "FOREIGN")])
    assert plan["plan"][0][2] == "alias"
    db = SessionLocal()
    db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Google").delete()
    db.commit()
    db.close()
    fact = _fact(store, "उपयोगकर्ता गूगल में काम करता है।")
    rep = linker.link_fact(fact.id, plan=plan)
    assert rep["linked"] == []
    assert ("गूगल", "alias anchor not found at apply") in rep["skipped"]
    db = SessionLocal()
    assert db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "गूगल").count() == 0
    db.close()


def test_plan_surfaces_has_no_session_parameter():
    # G3 R2 M5: a db= parameter here would run the model load under a
    # caller's write lock — the exact hazard link_fact refuses loudly.
    import inspect
    from agentmem_os.db.fact_entities import FactEntityLinker
    assert "db" not in inspect.signature(
        FactEntityLinker.plan_surfaces).parameters


def test_migration_rejects_lookalike_indexes(tmp_path, monkeypatch):
    # G3 R2 M2: substring verification "verified" an index with an
    # extra trailing column (unique over nothing) and one with a WHERE
    # clause (NULL scope unguarded). Exact-DDL equality rebuilds both.
    for i, bad_ddl in enumerate((
        "CREATE UNIQUE INDEX uq_kg_nodes_scope_text ON kg_nodes"
        "(coalesce(agent_id,''), entity_text, id)",
        "CREATE UNIQUE INDEX uq_kg_nodes_scope_text ON kg_nodes"
        "(coalesce(agent_id,''), entity_text) WHERE agent_id IS NOT NULL",
    )):
        sub = tmp_path / f"lookalike{i}"
        sub.mkdir()
        p = _old_db(sub, "")
        conn = sqlite3.connect(p)
        conn.execute(bad_ddl)
        conn.commit()
        conn.close()
        report = _run_stage3_migration(monkeypatch, p)
        assert "rebuilt (wrong shape)" in report["kg_nodes_unique"], bad_ddl
        conn = sqlite3.connect(p)
        conn.execute("INSERT INTO kg_nodes (entity_text, entity_type) "
                     "VALUES ('Google','ORG')")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO kg_nodes (entity_text, entity_type) "
                         "VALUES ('Google','ORG')")   # NULL scope now guarded
        conn.close()


def test_node_merge_catches_inverted_pairs(tmp_path, monkeypatch):
    # G3 R2 M3 repro pinned: keeper=min(id) makes re-pointing INVERT
    # src<tgt ordering — the ordered per-group GROUP BY missed exactly
    # the pairs the merge itself created (measured: loader kept 5.0 of
    # a true 7.0). The global pass canonicalizes ordering first.
    p = _old_db(tmp_path, """
      INSERT INTO kg_nodes (id, agent_id, entity_text, entity_type,
        mention_count, last_seen) VALUES
        (1, NULL, 'Google', 'ORG', 1, '2023-01-01'),
        (3, NULL, 'Paris',  'GPE', 1, '2023-01-01'),
        (5, NULL, 'Google', 'ORG', 1, '2023-06-01');
      -- (3,5): src<tgt as written; re-pointing 5→1 INVERTS it to (3,1)
      INSERT INTO kg_edges (id, source_id, target_id, weight, relation_type) VALUES
        (10, 1, 3, 5.0, 'CO_OCCURS'),
        (11, 3, 5, 2.0, 'CO_OCCURS');
    """)
    report = _run_stage3_migration(monkeypatch, p)
    assert "after merge" in report["kg_nodes_unique"]
    conn = sqlite3.connect(p)
    co = conn.execute(
        "SELECT source_id, target_id, weight FROM kg_edges "
        "WHERE relation_type='CO_OCCURS' OR relation_type IS NULL").fetchall()
    conn.close()
    assert co == [(1, 3, 7.0)]     # 5.0 + 2.0, one canonical row


def test_global_repair_merges_reversed_rows(tmp_path, monkeypatch):
    # (a,b) and (b,a) are the SAME undirected pair — historical
    # reversed rows must merge, not survive as two.
    p = _old_db(tmp_path, """
      INSERT INTO kg_nodes (id, entity_text, entity_type) VALUES
        (1, 'Google', 'ORG'), (2, 'Paris', 'GPE');
      INSERT INTO kg_edges (id, source_id, target_id, weight, relation_type,
        last_updated) VALUES
        (10, 1, 2, 3.0, 'CO_OCCURS', '2023-01-01'),
        (11, 2, 1, 4.0, 'CO_OCCURS', '2023-06-01');
    """)
    report = _run_stage3_migration(monkeypatch, p)
    assert report["co_occurs_dedup"] == "merged 1 duplicate pairs"
    conn = sqlite3.connect(p)
    rows = conn.execute(
        "SELECT source_id, target_id, weight, last_updated "
        "FROM kg_edges").fetchall()
    # R4-M3 tripwire: the CO_OCCURS branch's backup INSERT was the one
    # with no test — the branch that deleted 2,711 rows on the real dev
    # DB. The deleted row must be IN the backup, pre-repair values intact.
    backup = conn.execute(
        "SELECT id, source_id, target_id, weight FROM "
        "kg_edges_dedup_backup").fetchall()
    conn.close()
    assert rows == [(1, 2, 7.0, "2023-06-01")]   # summed + newest recency
    # Canonicalization runs before dedup, so the backed-up loser row
    # carries canonical ordering with its ORIGINAL weight — the row as
    # it existed at deletion time.
    assert backup == [(11, 1, 2, 4.0)]


def test_node_merge_repoints_fact_links(tmp_path, monkeypatch):
    # G3 R2 minor: the raw migration connection runs with FKs OFF —
    # forgetting semantic_fact_entities silently orphaned fact links at
    # deleted node ids. Collisions with an existing keeper link drop.
    p = _old_db(tmp_path, """
      CREATE TABLE semantic_fact_entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fact_id INTEGER NOT NULL, node_id INTEGER NOT NULL,
        surface_text VARCHAR NOT NULL, linked_via VARCHAR NOT NULL DEFAULT 'ner',
        confidence FLOAT, created_at DATETIME,
        CONSTRAINT uq_fact_entity UNIQUE (fact_id, node_id));
      INSERT INTO kg_nodes (id, entity_text, entity_type) VALUES
        (1, 'Google', 'ORG'), (2, 'Google', 'ORG');
      INSERT INTO semantic_fact_entities (fact_id, node_id, surface_text) VALUES
        (100, 2, 'Google'),      -- re-points to keeper 1
        (200, 1, 'Google'),      -- already on keeper
        (200, 2, 'Google');      -- would collide → dropped
    """)
    _run_stage3_migration(monkeypatch, p)
    conn = sqlite3.connect(p)
    links = conn.execute(
        "SELECT fact_id, node_id FROM semantic_fact_entities "
        "ORDER BY fact_id").fetchall()
    conn.close()
    assert links == [(100, 1), (200, 1)]


def test_plan_surfaces_empty_input(linked_env):
    _, linker, _, _ = linked_env
    assert linker.plan_surfaces([]) == {"agent_id": None, "plan": [],
                                        "skipped": []}


def test_get_node_and_self_alias_defenses(linked_env):
    from agentmem_os.db.models import KnowledgeGraphNode
    _, linker, SessionLocal, _ = linked_env
    db = SessionLocal()
    assert linker._get_node(db, None, None) is None      # junk anchor text
    node = KnowledgeGraphNode(agent_id=None, entity_text="Google",
                              entity_type="ORG", mention_count=1)
    db.add(node)
    db.flush()
    assert linker._ensure_alias_edge(db, node, node, 0.99, None) == 0
    db.rollback()
    db.close()


def test_closure_truncation_is_loud(linked_env, caplog):
    # A chain deeper than the cap must WARN — a silent partial read is
    # the M3 class wearing a depth-cap hat.
    from agentmem_os.db.models import (
        KnowledgeGraphEdge, KnowledgeGraphNode, SemanticFactEntity,
    )
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user visited Chennai.")
    db = SessionLocal()
    nodes = []
    for i in range(13):
        n = KnowledgeGraphNode(agent_id=None, entity_text=f"variant-{i:02d}",
                               entity_type="FOREIGN", mention_count=1)
        db.add(n)
        nodes.append(n)
    db.flush()
    for a, b in zip(nodes, nodes[1:]):
        lo, hi = sorted((a.id, b.id))
        db.add(KnowledgeGraphEdge(source_id=lo, target_id=hi, weight=1.0,
                                  relation_type="ALIAS_OF", confidence=0.95))
    db.add(SemanticFactEntity(fact_id=fact.id, node_id=nodes[-1].id,
                              surface_text="variant-12", linked_via="alias",
                              confidence=0.95))
    db.commit()
    db.close()
    import logging
    caplog.set_level(logging.WARNING)
    got = linker.facts_for_entity("variant-00")
    # loguru does not propagate to caplog by default — assert via a sink
    from loguru import logger as _loguru
    msgs = []
    sink_id = _loguru.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        linker.facts_for_entity("variant-00")
    finally:
        _loguru.remove(sink_id)
    assert any("truncated at depth" in m for m in msgs)


# ── G3 R3 fixes: two-depth sweep, ALIAS_OF canonicalization, tripwires ──────

def test_rescan_recovers_skipped_surface_on_partially_linked_fact(
        linked_env, monkeypatch):
    # R3 B1 repro pinned: fact linked Microsoft but its गूगल surface was
    # skipped (no anchor yet). The DEFAULT zero-link sweep must NOT see
    # it (documented bound); rescan_all=True must recover it once the
    # anchor exists.
    from agentmem_os.db import fact_entities as fe
    from agentmem_os.db.models import KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at गूगल and Microsoft.")
    rep = linker.link_fact(fact.id, fact_text=fact.fact_text)
    assert [s for s, _, _ in rep["linked"]] == ["Microsoft"]
    assert any(s == "गूगल" for s, _ in rep["skipped"])

    db = SessionLocal()
    db.add(KnowledgeGraphNode(agent_id=None, entity_text="Google",
                              entity_type="ORG", mention_count=1))
    db.commit()
    db.close()
    monkeypatch.setattr(fe, "get_alias_resolver", lambda: _FakeResolver())

    shallow = linker.link_missing()
    assert shallow["candidates"] == 0        # the documented bound, pinned
    deep = linker.link_missing(rescan_all=True)
    assert deep["links_created"] >= 1
    got = {f.id for f in linker.facts_for_entity("Google")}
    assert fact.id in got                    # the fact naming गूगल is visible


def test_retry_does_not_double_append_skipped(linked_env, monkeypatch):
    # R3 minor pinned — REBUILT after R4-M2 proved the first version
    # could not detect reverting the fix (its mock raised BEFORE the
    # real body, so the shared list was never mutated). Now the first
    # attempt runs the REAL apply (which appends the anchor-miss skip)
    # and THEN loses the race — without the local copy, the retry
    # appends into the same validated-tuple list a second time.
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    plan = {"agent_id": None,
            "plan": [("गूगल", "FOREIGN", "alias", "Ghost", 0.99)],
            "skipped": [("में", "indic surface, no alias anchor at tau")]}
    real = linker._link_in_session
    calls = {"n": 0}

    def _lose_after_real_work(session, owns, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            real(session, owns, *a, **k)   # mutates shared list if uncopied
            raise _race_error()
        return real(session, owns, *a, **k)

    monkeypatch.setattr(linker, "_link_in_session", _lose_after_real_work)
    rep = linker.link_fact(fact.id, plan=plan)
    assert calls["n"] == 2
    # one planning skip + one apply skip — each exactly once
    assert sorted(rep["skipped"]) == sorted([
        ("में", "indic surface, no alias anchor at tau"),
        ("गूगल", "alias anchor not found at apply")])


def test_plan_validation_rejects_junk_shapes(linked_env):
    store, linker, _, _ = linked_env
    fact = _fact(store, "The user works at Google.")
    for bad in (
        {"agent_id": None, "skipped": [],
         "plan": [("", "ORG", "ner", None, None)]},          # empty surface
        {"agent_id": None, "skipped": [],
         "plan": [("गूगल", "FOREIGN", "alias", None, 0.9)]},  # alias sans anchor
        {"agent_id": None, "skipped": ["oops"],
         "plan": []},                                         # junk skipped
        {"agent_id": None, "skipped": [],
         "plan": [("a", "ORG", "ner", None, None)]},          # 1-char surface
        {"agent_id": None, "skipped": [],
         "plan": [(" x ", "ORG", "ner", None, None)]},        # unstripped
    ):
        with pytest.raises(ValueError, match="malformed"):
            linker.link_fact(fact.id, plan=bad)


def test_node_merge_inverted_alias_pairs_deduplicated(tmp_path, monkeypatch):
    # R3 M3 repro pinned: node-merge re-pointing inverted an ALIAS_OF
    # pair; the ordered exists-check then duplicated the edge. The
    # global pass now canonicalizes ALIAS_OF too and collapses exact
    # duplicates keeping the best confidence.
    p = _old_db(tmp_path, """
      INSERT INTO kg_nodes (id, agent_id, entity_text, entity_type) VALUES
        (1, NULL, 'Google', 'ORG'),
        (3, NULL, 'गूगल',  'FOREIGN'),
        (5, NULL, 'Google', 'ORG');
      INSERT INTO kg_edges (id, source_id, target_id, weight, relation_type,
        confidence) VALUES
        (10, 1, 3, 1.0, 'ALIAS_OF', 0.95),
        (11, 3, 5, 1.0, 'ALIAS_OF', 0.91);  -- re-points to (3,1): inverted dup
    """)
    _run_stage3_migration(monkeypatch, p)
    conn = sqlite3.connect(p)
    rows = conn.execute(
        "SELECT source_id, target_id, confidence FROM kg_edges "
        "WHERE relation_type='ALIAS_OF'").fetchall()
    backup = conn.execute(
        "SELECT count(*) FROM kg_edges_dedup_backup").fetchone()[0]
    conn.close()
    assert rows == [(1, 3, 0.95)]     # canonical, deduped, best confidence
    assert backup == 1                # the deleted row is preserved


def test_indic_digit_runs_never_tokenize():
    # R3 minor tripwire: reverting the isdigit() filter fails here.
    from agentmem_os.db.entity_aliases import EntityAliasResolver
    toks = EntityAliasResolver.extract_script_tokens("१२३४ गूगल ५६")
    assert toks == ["गूगल"]


def test_demo_reset_survives_linked_global_facts():
    # R3 M5: the R2-M4 fix (delete link rows before nodes) had no
    # regression test — reordering the deletes restored the 500 with a
    # green suite. This exercises the REAL endpoint: a global-scope
    # fact linked to a global-scope node, FKs ON, then POST /demo/reset.
    from fastapi.testclient import TestClient
    from agentmem_os.api.app import app
    from agentmem_os.db.engine import get_session
    from agentmem_os.db.fact_entities import FactEntityLinker
    from agentmem_os.db.semantic_facts import SemanticFactStore
    from agentmem_os.db.models import (
        KnowledgeGraphNode, SemanticFact, SemanticFactEntity,
        Session as SessionRow,
    )

    db = get_session()
    try:
        if db.get(SessionRow, "reset-t-1") is None:
            db.add(SessionRow(session_id="reset-t-1"))
            db.commit()
    finally:
        db.close()
    store = SemanticFactStore(get_session)
    linker = FactEntityLinker(get_session)
    fact, _ = store.add_fact("The user works at ResetCorp.",
                             fact_type="state", t_mentioned="2023/05/20",
                             source_session_id="reset-t-1",
                             extraction_model="t")
    rep = linker.link_fact(fact.id, fact_text=fact.fact_text)
    assert rep["linked"], "precondition: a global-scope link must exist"

    client = TestClient(app)
    resp = client.post("/demo/reset")
    assert resp.status_code == 200, resp.text

    db = get_session()
    try:
        # Global-scope nodes purged, OUR link purged, OUR fact survives.
        # Assertions scoped to THIS test's artifacts (R4-m5: a global
        # count on the shared scratch DB failed spuriously when another
        # test left an agent-scoped link behind — the endpoint only
        # promises the global namespace).
        assert db.query(KnowledgeGraphNode).filter(
            KnowledgeGraphNode.agent_id.is_(None)).count() == 0
        assert db.query(SemanticFactEntity).filter(
            SemanticFactEntity.fact_id == fact.id).count() == 0
        assert db.get(SemanticFact, fact.id) is not None
    finally:
        db.close()


def test_fact_id_bound_is_loud_and_enforced(linked_env, monkeypatch):
    # R4-m3 rebuilt: with only bound-many facts, dropping .limit() left
    # the test green (len==bound fired the warning anyway). THREE facts
    # against bound=2 pins both the truncation AND the warning.
    from agentmem_os.db import fact_entities as fe
    store, linker, SessionLocal, _ = linked_env
    facts = [
        _fact(store, "The user works at Google."),
        _fact(store, "The user visited the Google campus.",
              fact_type="event", t_occurred="2023/04/01"),
        _fact(store, "The user interviewed at Google.",
              fact_type="event", t_occurred="2023/03/01"),
    ]
    for f in facts:
        linker.link_fact(f.id, fact_text=f.fact_text)
    monkeypatch.setattr(fe, "_FACT_ID_BOUND", 2)
    from loguru import logger as _loguru
    msgs = []
    sink = _loguru.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        got = linker.facts_for_entity("Google")
    finally:
        _loguru.remove(sink)
    assert len(got) == 2                     # truncation actually enforced
    assert any("fact-id bound" in m for m in msgs)


def test_backup_table_survives_kg_edges_schema_drift(tmp_path, monkeypatch):
    # R4-M4 repro pinned: a stale backup table (frozen columns) + a
    # later ALTER TABLE kg_edges ADD COLUMN made INSERT..SELECT * raise
    # inside the import-time migration — the whole package became
    # unimportable on exactly the DBs that had duplicates. The stale
    # backup must be preserved under a versioned name, not lost.
    p = _old_db(tmp_path, """
      INSERT INTO kg_nodes (id, entity_text, entity_type) VALUES
        (1, 'Google', 'ORG'), (2, 'Paris', 'GPE');
      INSERT INTO kg_edges (id, source_id, target_id, weight, relation_type) VALUES
        (10, 1, 2, 3.0, 'CO_OCCURS'),
        (11, 1, 2, 4.0, 'CO_OCCURS');
      -- stale backup created before kg_edges grew a column:
      CREATE TABLE kg_edges_dedup_backup (
        id INTEGER, source_id INTEGER, target_id INTEGER, weight FLOAT);
      INSERT INTO kg_edges_dedup_backup VALUES (99, 5, 6, 1.0);
    """)
    report = _run_stage3_migration(monkeypatch, p)
    assert report["co_occurs_dedup"] == "merged 1 duplicate pairs"
    conn = sqlite3.connect(p)
    fresh = conn.execute(
        "SELECT id, weight FROM kg_edges_dedup_backup").fetchall()
    old = conn.execute(
        "SELECT id, weight FROM kg_edges_dedup_backup_old1").fetchall()
    conn.close()
    assert fresh == [(11, 4.0)]      # new-schema backup holds the loser
    assert old == [(99, 1.0)]        # stale backup preserved, not dropped


def test_facts_for_entity_rejects_empty_query(linked_env):
    _, linker, _, _ = linked_env
    for bad in (None, "", "   "):
        with pytest.raises(ValueError, match="non-empty"):
            linker.facts_for_entity(bad)


def test_closure_node_bound_is_loud(linked_env, monkeypatch):
    from agentmem_os.db import fact_entities as fe
    from agentmem_os.db.models import KnowledgeGraphEdge, KnowledgeGraphNode
    store, linker, SessionLocal, _ = linked_env
    fact = _fact(store, "The user visited Chennai.")
    linker.link_fact(fact.id, fact_text=fact.fact_text)
    db = SessionLocal()
    chennai = db.query(KnowledgeGraphNode).filter(
        KnowledgeGraphNode.entity_text == "Chennai").one()
    prev = chennai
    for i in range(4):
        n = KnowledgeGraphNode(agent_id=None, entity_text=f"cb-{i}",
                               entity_type="FOREIGN", mention_count=1)
        db.add(n)
        db.flush()
        lo, hi = sorted((prev.id, n.id))
        db.add(KnowledgeGraphEdge(source_id=lo, target_id=hi, weight=1.0,
                                  relation_type="ALIAS_OF", confidence=0.95))
        prev = n
    db.commit()
    db.close()
    monkeypatch.setattr(fe, "_CLOSURE_NODE_BOUND", 2)
    from loguru import logger as _loguru
    msgs = []
    sink = _loguru.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        got = linker.facts_for_entity("Chennai")
    finally:
        _loguru.remove(sink)
    assert [f.id for f in got] == [fact.id]   # seed's facts still returned
    assert any("exceeded 2 nodes" in m for m in msgs)
