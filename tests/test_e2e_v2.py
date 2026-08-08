"""
Stage 6 G1: the FULL Consolidation-v2 loop through the PRODUCT
surfaces — every drive call goes through mcp_server.handle_call_tool
(save_memory → consolidate_session → recall_memory /
get_knowledge_graph). The ONLY thing mocked is the single LLM
boundary each engine already isolates (ConsolidationV2._llm /
SupersessionJudge._llm — one urlopen each); validation, storage,
entity linking, judgment gates, retrieval and rendering all run REAL.

Assertions may PEEK at the store through the engine session (reads
only) — the drive path stays product-only, per the design record D1.

Note on system turns: the save_memory tool schema enums user/assistant
only; the handler accepts role="system", which is how session-date
headers ("Session dated YYYY/MM/DD") enter — the same convention the
benchmark corpus uses. Live MCP sessions without headers get
turn-timestamp dates, which is correct for live use.
"""
import json

import pytest


async def _call(name, args):
    from mcp_server.server import handle_call_tool

    result = await handle_call_tool(name, args)
    return json.loads(result[0].text)


def _mock_extraction(monkeypatch, facts):
    import agentmem_os.llm.consolidation_v2 as cv2

    monkeypatch.setattr(cv2.ConsolidationV2, "_llm",
                        lambda self, prompt: {"facts": facts})


def _mock_judge(monkeypatch, superseded_ids=None):
    import agentmem_os.llm.supersession as sup

    monkeypatch.setattr(
        sup.SupersessionJudge, "_llm",
        lambda self, prompt: {"reasoning": "e2e deterministic mock",
                              "superseded_ids": superseded_ids or [],
                              "cancelled_ids": []})


def _await_background(timeout=180.0):
    """Design D4: save_turn spawns background KG-ingestion threads;
    the E2E must wait for quiescence BOUNDED and LOUDLY — a recall
    racing a background writer is a flake, not a test. (Observed
    before this existed: the facts section vanished from one
    cross-session recall in one of three otherwise-identical runs.)

    The timeout covers the measured reality this wait EXPOSED: each
    process's first KG ingests serialize behind the ~87s cold
    alias-model load (the Stage 3 B1 measurement) — background KG
    visibility is eventually-consistent by contract, and 'eventual'
    is ~1-2 minutes on a cold process. Disclosed in the stage
    record."""
    import threading
    import time

    deadline = time.monotonic() + timeout
    main = threading.main_thread()

    def _ours(t):
        # tqdm's TMonitor is a global daemon watchdog that never
        # exits by design — waiting on it is an eternal wait.
        return (t is not main and t.is_alive()
                and not t.name.startswith(("pytest", "asyncio"))
                and not t.__class__.__module__.startswith("tqdm"))

    while time.monotonic() < deadline:
        busy = [t for t in threading.enumerate() if _ours(t)]
        if not busy:
            return
        for t in busy:
            t.join(timeout=max(0.1, deadline - time.monotonic()))
    raise AssertionError(
        f"background threads still alive after {timeout}s: "
        f"{[t.name for t in threading.enumerate() if _ours(t)]}")


async def _ingest(sid, lines, date=None):
    if date:
        await _call("save_memory", {"session_id": sid, "role": "system",
                                    "content": f"Session dated {date}"})
    for role, content in lines:
        await _call("save_memory", {"session_id": sid, "role": role,
                                    "content": content})
    _await_background()


def _facts_section(context):
    import re
    m = re.search(r"<\[SEMANTIC FACTS\]>(.*?)</\[SEMANTIC FACTS\]>",
                  context, re.S)
    return m.group(1) if m else None


# ── The full round trip ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_round_trip_facts_first(monkeypatch):
    await _ingest("e2e-rt-1", [
        ("user", "Big news: I just moved to Lisbon for my new job at "
                 "Beacon Labs."),
        ("assistant", "Congratulations on the Beacon Labs role!"),
    ], date="2023/04/10")
    _mock_extraction(monkeypatch, [
        {"text": "The user moved to Lisbon for a new job at Beacon "
                 "Labs.", "fact_type": "state"},
    ])
    _mock_judge(monkeypatch)

    report = await _call("consolidate_session", {"session_id": "e2e-rt-1"})
    assert report["created"] == 1
    assert report["judge_failure"] is None

    recall = await _call("recall_memory", {
        "session_id": "e2e-rt-1",
        "query": "Where does the user work now?"})
    section = _facts_section(recall["context"])
    assert section is not None
    assert "Beacon Labs" in section
    assert "[noted 2023/04/10]" in section  # t_mentioned from the header


@pytest.mark.asyncio
async def test_cross_session_recall_through_mcp(monkeypatch):
    await _ingest("e2e-far-1", [
        ("user", "I have been enjoying weekend hikes lately."),
    ], date="2023/04/12")
    recall = await _call("recall_memory", {
        "session_id": "e2e-far-1",
        "query": "Where does the user work now?"})
    section = _facts_section(recall["context"])
    assert section is not None and "Beacon Labs" in section
    # ...and the RAW-TURN tiers of THIS session cannot contain it.
    # (The WORLD MODEL tier is scope-wide like facts, so it may
    # legitimately name the entity — only the per-session raw tiers
    # are structurally blind here.)
    import re as _re
    for tier in ("SEMANTIC MEMORY", "RECENT TURNS"):
        m = _re.search(rf"<\[{tier}\]>(.*?)</\[{tier}\]>",
                       recall["context"], _re.S)
        if m:
            assert "Beacon Labs" not in m.group(1)


@pytest.mark.asyncio
async def test_scope_isolation_through_mcp():
    recall = await _call("recall_memory", {
        "session_id": "e2e-rt-1",
        "query": "Where does the user work now?",
        "agent_id": "no-such-agent-scope"})
    assert _facts_section(recall["context"]) is None


@pytest.mark.asyncio
async def test_double_consolidation_is_idempotent(monkeypatch):
    _mock_extraction(monkeypatch, [
        {"text": "The user moved to Lisbon for a new job at Beacon "
                 "Labs.", "fact_type": "state"},
    ])
    _mock_judge(monkeypatch)
    report = await _call("consolidate_session", {"session_id": "e2e-rt-1"})
    assert report["created"] == 0  # re-affirmed, never duplicated

    recall = await _call("recall_memory", {
        "session_id": "e2e-rt-1",
        "query": "Where does the user work now?"})
    section = _facts_section(recall["context"])
    assert section.count("Beacon Labs") == 1


# ── Supersession, end to end ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_supersession_fires_through_the_product_loop(monkeypatch):
    await _ingest("e2e-sup-1", [
        ("user", "I work at Google as an engineer."),
    ], date="2023/03/01")
    _mock_extraction(monkeypatch, [
        {"text": "The user works at Google.", "fact_type": "state"},
    ])
    _mock_judge(monkeypatch)
    r1 = await _call("consolidate_session", {"session_id": "e2e-sup-1"})
    assert r1["created"] == 1

    from agentmem_os.db.engine import get_session
    from agentmem_os.db.models import SemanticFact
    db = get_session()
    try:
        old = (db.query(SemanticFact)
               .filter(SemanticFact.fact_text ==
                       "The user works at Google.").one())
    finally:
        db.close()

    await _ingest("e2e-sup-2", [
        ("user", "Update: I left Google and now I work at Microsoft."),
    ], date="2023/06/15")
    _mock_extraction(monkeypatch, [
        {"text": "The user left Google and now works at Microsoft.",
         "fact_type": "state"},
    ])
    _mock_judge(monkeypatch, superseded_ids=[old.id])
    r2 = await _call("consolidate_session", {"session_id": "e2e-sup-2"})
    assert r2["created"] == 1
    sup = r2["supersession"] or {}
    assert any(pair[0] == old.id for pair in sup.get("superseded", []))

    recall = await _call("recall_memory", {
        "session_id": "e2e-sup-2",
        "query": "Where does the user work these days?"})
    section = _facts_section(recall["context"])
    assert "Microsoft" in section
    assert "[change history:" in section  # the old truth, visible as history
    lines = [l for l in section.split("\n")
             if l.strip().startswith("[")]
    assert not any(l.startswith("[noted 2023/03/01] (state) The user "
                                "works at Google.") for l in lines)


# ── Failure paths through the MCP contract ───────────────────────────────────

@pytest.mark.asyncio
async def test_dead_llm_fails_loudly_with_zero_writes(monkeypatch):
    import urllib.error

    import agentmem_os.llm.consolidation_v2 as cv2

    await _ingest("e2e-dead-1", [
        ("user", "I adopted a beagle named Otto last week."),
    ], date="2023/05/02")

    def _boom(self, prompt):
        raise urllib.error.URLError("ollama down (e2e)")

    monkeypatch.setattr(cv2.ConsolidationV2, "_llm", _boom)
    result = await _call("consolidate_session",
                         {"session_id": "e2e-dead-1"})
    assert "error" in result

    from agentmem_os.db.engine import get_session
    from agentmem_os.db.models import SemanticFact
    db = get_session()
    try:
        n = (db.query(SemanticFact)
             .filter(SemanticFact.source_session_id == "e2e-dead-1")
             .count())
    finally:
        db.close()
    assert n == 0  # loud failure, zero writes — the Stage 2 contract


@pytest.mark.asyncio
async def test_unknown_session_refused_before_any_work():
    result = await _call("consolidate_session",
                         {"session_id": "e2e-never-existed"})
    assert "error" in result and "not found" in result["error"]


def test_concurrent_kg_ingest_never_drops_turns(tmp_path):
    """Pin for the Stage 6 race fix: one background thread per saved
    turn, and the read-then-write node upsert raced its siblings — the
    loser dropped its turn's ENTIRE KG contribution (12 drops in one
    E2E run once the offline-first model load stopped serializing the
    threads). With retry-on-IntegrityError, every thread must land.
    (A barrier maximizes collision; the race is probabilistic per
    thread pair, but 16 barrier-released writers made the OLD code
    fail on every observed run.)"""
    import threading

    from tests.test_semantic_facts import _bootstrap, _make_production_engine
    from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph
    from agentmem_os.db.models import KnowledgeGraphNode

    engine = _make_production_engine(tmp_path / "kgrace.db")
    SessionLocal = _bootstrap(engine)
    kg = EntityKnowledgeGraph(SessionLocal)
    kg._extract_entities("warm up the shared NER before the barrier")

    n_threads = 16
    barrier = threading.Barrier(n_threads)
    results = [None] * n_threads

    def _worker(i):
        barrier.wait()
        results[i] = kg.ingest_turn(
            "sess-1", None, "Rachel Smith visited the office.")

    threads = [threading.Thread(target=_worker, args=(i,))
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert all(r == 1 for r in results), results  # nobody dropped
    db = SessionLocal()
    try:
        node = (db.query(KnowledgeGraphNode)
                .filter(KnowledgeGraphNode.entity_text == "Rachel Smith")
                .one())  # .one() also proves the unique index held
        assert node.mention_count == n_threads
    finally:
        db.close()


def _fresh_kg(tmp_path, name):
    from tests.test_semantic_facts import _bootstrap, _make_production_engine
    from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph

    engine = _make_production_engine(tmp_path / name)
    SessionLocal = _bootstrap(engine)
    return EntityKnowledgeGraph(SessionLocal), SessionLocal


def test_upsert_retry_is_deterministically_pinned(tmp_path, monkeypatch):
    """Final-pass W1: the 16-thread pin killed the retry mutant in
    isolation but NOT in-file (warm state closes the collision
    window). This pin constructs the collision through the _find_node
    seam — the lookup lies 'not found' exactly once, forcing the
    loser's INSERT path against an existing row — so a retry revert
    dies in ANY context."""
    from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph
    from agentmem_os.db.models import KnowledgeGraphNode

    kg, SessionLocal = _fresh_kg(tmp_path, "retrypin.db")
    assert kg.ingest_turn("sess-1", None,
                          "Rachel Smith visited the office.") == 1

    orig = EntityKnowledgeGraph._find_node
    lied = {"n": 0}

    def lying(db, text, agent_id):
        if lied["n"] == 0:
            lied["n"] += 1
            return None  # the race: reader saw nothing, row exists
        return orig(db, text, agent_id)

    monkeypatch.setattr(EntityKnowledgeGraph, "_find_node",
                        staticmethod(lying))
    assert kg.ingest_turn("sess-1", None,
                          "Rachel Smith visited the office.") == 1
    db = SessionLocal()
    try:
        node = (db.query(KnowledgeGraphNode)
                .filter(KnowledgeGraphNode.entity_text == "Rachel Smith")
                .one())
        assert node.mention_count == 2  # retried AND counted exactly once
    finally:
        db.close()


def test_lost_update_impossible_for_node_and_edge(tmp_path):
    """Final-pass W1/W2 (rebuilt after the confirmation round broke
    the first version): deterministic lost-update pins for BOTH atomic
    increments. Two sessions load the same row and EXPUNGE their
    objects — rollback() would EXPIRE them and the next attribute
    access silently refreshes from the DB, destroying the very
    staleness the pin depends on (the critic's X4/X5 mutants both
    survived the rollback version; its lesson verbatim: "rollback()
    destroys the very staleness a lost-update pin depends on"). The
    setup ASSERTS the objects are genuinely stale before asserting the
    outcome. Server-side arithmetic lands on start+2; an ORM
    read-modify-write revert writes the stale snapshot → start+1."""
    from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph
    from agentmem_os.db.models import (
        KnowledgeGraphEdge, KnowledgeGraphNode,
    )

    kg, SessionLocal = _fresh_kg(tmp_path, "lostupd.db")
    assert kg.ingest_turn(
        "sess-1", None,
        "Rachel Smith met David Chen at the office.") >= 2

    for model, bump, col in (
            (KnowledgeGraphNode, EntityKnowledgeGraph._bump_node,
             "mention_count"),
            (KnowledgeGraphEdge, EntityKnowledgeGraph._bump_edge,
             "weight")):
        dbA, dbB = SessionLocal(), SessionLocal()
        try:
            objA = dbA.query(model).order_by(model.id.asc()).first()
            objB = dbB.query(model).filter(model.id == objA.id).one()
            start = getattr(objB, col)
            # Detach WITH loaded values — the objects must stay stale.
            dbA.expunge(objA), dbB.expunge(objB)
            dbA.rollback(), dbB.rollback()
            bump(dbA, objA)
            dbA.commit()
            # Setup self-check: objB must still hold the OLD value —
            # if this fails, the pin has gone vacuous again.
            assert getattr(objB, col) == start
            bump(dbB, objB)
            dbB.commit()
        finally:
            dbA.close(), dbB.close()
        db = SessionLocal()
        try:
            final = getattr(
                db.query(model).filter(model.id == objA.id).one(), col)
            assert final == start + 2, (model.__name__, start, final)
        finally:
            db.close()


def test_failed_ingest_always_invalidates_graph_cache(tmp_path,
                                                      monkeypatch):
    """Final-pass m1 pin (the critic judged it pinnable, so it is
    pinned rather than disclosed): ingest_turn mutates the in-memory
    graph BEFORE commit, so ANY rolled-back attempt — not just the
    raced IntegrityError — must invalidate the cache or reads serve a
    graph that is permanently ahead of the DB."""
    from agentmem_os.db.knowledge_graph import EntityKnowledgeGraph

    kg, SessionLocal = _fresh_kg(tmp_path, "cacheinv.db")
    assert kg.ingest_turn("sess-1", None,
                          "Rachel Smith visited the office.") == 1
    assert len(kg._graph.nodes) > 0  # cache warm

    class _PoisonSession:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def commit(self):
            raise RuntimeError("disk full (constructed, non-Integrity)")

    monkeypatch.setattr(kg, "get_db",
                        lambda: _PoisonSession(SessionLocal()))
    assert kg.ingest_turn("sess-1", None,
                          "David Chen joined the meeting.") == 0
    assert len(kg._graph.nodes) == 0  # cache invalidated, not ahead


# ── KG surface ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_graph_surface_sees_the_entities():
    result = await _call("get_knowledge_graph",
                         {"session_id": "e2e-rt-1", "entity": "Lisbon"})
    assert "subgraph" in result
