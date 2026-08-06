"""
Semantic Fact Store tests — Consolidation v2 Stage 1 (G1 gate, expanded
after the G3 critic verdict).

Failure paths are seeded from REAL competitor production bugs
(CONSOLIDATION_V2_STAGE0_RESEARCH.md §5) AND from the G3 critic's measured
breaks of our own first cut (CONSOLIDATION_V2_BUILD_LOG.md Stage 1 G3):
threaded writers on the production engine config, racing supersedes, event
occurrences collapsing, scope collapse, false-success on FK violations,
impossible calendar dates, partial-date range dropping, chain hangs.
"""
import os
import sys
import threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest


def _make_engine(url="sqlite://"):
    """Isolated IN-MEMORY engine for the functional tests: StaticPool +
    foreign_keys only (WAL does not exist for in-memory DBs). This is NOT
    the production construction — concurrency tests must use
    _make_production_engine below (G3 round 2, M6: an earlier docstring
    here claimed production parity it did not have)."""
    from sqlalchemy import create_engine, event
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        url, connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    return engine


def _make_production_engine(db_file):
    """File-backed engine matching db/engine.py's construction (default
    QueuePool — exclusive connection per checked-out session — WAL,
    busy_timeout, foreign keys). NOTE: parity here is convenience for
    isolated fixtures; the binding regression test below
    (test_real_engine_pool_and_concurrency) exercises db/engine.py ITSELF
    so a pooling revert turns the suite red (R3 N4)."""
    from sqlalchemy import create_engine, event

    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    return engine


def _bootstrap(engine):
    from sqlalchemy.orm import sessionmaker
    from agentmem_os.db.models import (
        AgentNamespace, Base, Session as SessionRow, Turn,
    )

    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False,
                                expire_on_commit=False)
    seed = SessionLocal()
    seed.add(AgentNamespace(agent_id="agent-A", name="agent-A"))
    seed.add(SessionRow(session_id="sess-1"))
    seed.add(SessionRow(session_id="sess-2"))
    seed.add(Turn(id=101, session_id="sess-1", role="user", content="I work at Google now."))
    seed.add(Turn(id=102, session_id="sess-1", role="user", content="I ride rollercoasters."))
    seed.commit()
    seed.close()
    return SessionLocal


@pytest.fixture()
def store_and_factory():
    from agentmem_os.db.semantic_facts import SemanticFactStore
    engine = _make_engine()
    SessionLocal = _bootstrap(engine)
    return SemanticFactStore(SessionLocal), SessionLocal, engine


@pytest.fixture()
def store(store_and_factory):
    return store_and_factory[0]


def _add(store, text, **kw):
    defaults = dict(fact_type="state", t_mentioned="2023/05/20",
                    source_session_id="sess-1", extraction_model="test-model")
    defaults.update(kw)
    return store.add_fact(text, **defaults)


# ── Happy path ───────────────────────────────────────────────────────────────

def test_add_and_retrieve_current_fact(store):
    fact, created = _add(store, "The user works at Google.",
                         source_turn_ids=[101], entities=["Google"])
    assert created is True
    assert fact.t_mentioned == "2023/05/20"
    assert fact.scope_key == "global"
    got = store.current_facts()
    assert [f.id for f in got] == [fact.id]
    assert got[0].source_turn_ids == [101]
    assert got[0].source_session_ids == ["sess-1"]


def test_date_normalization_variants(store):
    fact, _ = _add(store, "The user attended a concert.",
                   fact_type="event",
                   t_mentioned="2023-05-20 (Sat) 02:21", t_occurred="2023/7")
    assert fact.t_mentioned == "2023/05/20"
    # month-only occurrence stored as an explicit interval, never a lexical trap
    assert fact.t_occurred == "2023/07/01"
    assert fact.t_occurred_end == "2023/07/31"


# ── Validation failure paths ─────────────────────────────────────────────────

def test_invalid_fact_type_rejected(store):
    with pytest.raises(ValueError):
        _add(store, "Some fact.", fact_type="opinion")


def test_empty_text_rejected(store):
    with pytest.raises(ValueError):
        _add(store, "   ")


def test_unparseable_and_impossible_dates_rejected_loudly(store):
    for bad in ("last Tuesday", "2023/13/45", "2023-02-31", "2023/00/00",
                "9999/99/99", "2023/10/1x"):
        with pytest.raises(ValueError):
            _add(store, "Some fact.", t_mentioned=bad)
    with pytest.raises(ValueError):
        _add(store, "Some fact.", t_occurred="2023/13")


def test_partial_t_mentioned_rejected(store):
    with pytest.raises(ValueError):
        _add(store, "Some fact.", t_mentioned="2023/05")


def test_bad_turn_ids_rejected_before_any_write(store):
    with pytest.raises(ValueError):
        _add(store, "Some fact.", source_turn_ids=["turn-101"])
    with pytest.raises(ValueError):   # bools are not citations (G3 F18)
        _add(store, "Some fact.", source_turn_ids=[True])
    assert store.current_facts() == []


def test_fk_violation_is_loud_never_false_success(store):
    # G3 finding 5: a ghost session must raise, not silently "re-affirm".
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        _add(store, "Fact citing nowhere.", source_session_id="GHOST-SESSION")
    assert store.current_facts() == []


# ── Re-affirmation, not duplication ──────────────────────────────────────────

def test_reaffirmation_strengthens_single_row(store):
    f1, created1 = _add(store, "The user has a dog named Poppy.",
                        source_turn_ids=[101])
    f2, created2 = _add(store, "  the user has a dog named  Poppy. ",
                        source_turn_ids=[102], entities=["Poppy"],
                        source_session_id="sess-2", lang_source="hi")
    assert created1 is True and created2 is False
    assert f1.id == f2.id
    assert f2.mention_count == 2
    assert f2.source_turn_ids == [101, 102]
    assert f2.source_session_ids == ["sess-1", "sess-2"]   # G3 F15
    assert f2.langs == ["en", "hi"]                        # cross-lingual record
    assert len(store.current_facts()) == 1


def test_reaffirmation_backfills_occurrence_date(store):
    f1, _ = _add(store, "The user visited the Louvre.")
    assert f1.t_occurred is None
    f2, created = _add(store, "The user visited the Louvre.",
                       t_occurred="2023/04/12")
    assert created is False and f2.t_occurred == "2023/04/12"


def test_numeric_difference_never_collapses(store):
    _add(store, "The user rode Space Mountain three times on September 24th.",
         fact_type="event")
    _add(store, "The user rode Space Mountain two times on September 24th.",
         fact_type="event")
    assert len(store.current_facts(fact_type="event")) == 2


def test_same_event_text_different_dates_never_collapses(store):
    # G3 finding 3 — the aggregation-thesis guard's other half.
    _add(store, "The user went for a 5-mile run.", fact_type="event",
         t_occurred="2023/10/01")
    _add(store, "The user went for a 5-mile run.", fact_type="event",
         t_occurred="2023/10/08")
    events = store.current_facts(fact_type="event")
    assert len(events) == 2
    assert {e.t_occurred for e in events} == {"2023/10/01", "2023/10/08"}
    # same text AND same date IS a re-affirmation
    f3, created = _add(store, "The user went for a 5-mile run.",
                       fact_type="event", t_occurred="2023/10/08")
    assert created is False and f3.mention_count == 2


# ── Supersession ─────────────────────────────────────────────────────────────

def test_supersession_leaves_exactly_one_live_fact(store):
    old, _ = _add(store, "The user works at Company A.", t_mentioned="2023/01/10")
    new, _ = _add(store, "The user works at Company B.", t_mentioned="2023/05/20")
    store.supersede(old.id, new.id)

    live = store.current_facts()
    assert [f.id for f in live] == [new.id]
    chain = store.chain(new.id)
    assert [f.id for f in chain] == [old.id, new.id]
    assert [f.id for f in store.facts_as_of("2023/02/01")] == [old.id]
    assert [f.id for f in store.facts_as_of("2023/06/01")] == [new.id]


def test_double_supersession_refused(store):
    a, _ = _add(store, "Fact A.")
    b, _ = _add(store, "Fact B.")
    c, _ = _add(store, "Fact C.")
    store.supersede(a.id, b.id)
    with pytest.raises(ValueError, match="already superseded"):
        store.supersede(a.id, c.id)


def test_self_and_cyclic_supersession_refused(store):
    a, _ = _add(store, "Fact A.")
    b, _ = _add(store, "Fact B.")
    c, _ = _add(store, "Fact C.")
    with pytest.raises(ValueError):
        store.supersede(a.id, a.id)
    store.supersede(a.id, b.id)
    store.supersede(b.id, c.id)
    with pytest.raises(ValueError, match="cycle"):
        store.supersede(c.id, a.id)   # multi-link cycle walk


def test_cross_scope_supersession_refused(store):
    a, _ = _add(store, "The user lives in Chennai.", agent_id=None)
    b, _ = _add(store, "The user lives in Bengaluru.", user_id="user-2")
    with pytest.raises(ValueError, match="scope"):
        store.supersede(a.id, b.id)


def test_many_to_one_merge_chain_complete(store):
    # G3 finding 16: Stage 2's dedup merge produces two-predecessors-one-
    # successor; the audit chain must show ALL of it, deterministically.
    a, _ = _add(store, "Old fact A.", t_mentioned="2023/01/10")
    b, _ = _add(store, "Old fact B.", t_mentioned="2023/02/10")
    c, _ = _add(store, "Merged fact C.", t_mentioned="2023/03/10")
    store.supersede(a.id, c.id)
    store.supersede(b.id, c.id)
    ids = [f.id for f in store.chain(c.id)]
    assert ids == [a.id, b.id, c.id]
    assert ids == [f.id for f in store.chain(a.id)]   # same view from any member


def test_transition_synthesized_at_read_time_rows_untouched(store):
    old, _ = _add(store, "The user prefers almond milk lattes.",
                  fact_type="preference", t_mentioned="2023/01/10")
    new, _ = _add(store, "The user prefers oat milk lattes.",
                  fact_type="preference", t_mentioned="2023/05/20")
    store.supersede(old.id, new.id)
    story = store.transition_text(new.id)
    assert "almond" in story and "oat" in story and "superseded" in story
    assert "almond" not in store.current_facts()[0].fact_text


def test_transition_text_empty_for_chainless_fact(store):
    f, _ = _add(store, "A lone fact.")
    assert store.transition_text(f.id) == ""


def test_not_found_paths_raise(store):
    with pytest.raises(ValueError, match="not found"):
        store.chain(9999)
    with pytest.raises(ValueError, match="not found"):
        store.provenance(9999)
    with pytest.raises(ValueError, match="not found"):
        store.supersede(9999, 9998)


# ── Scope isolation (G3 finding 4) ───────────────────────────────────────────

def test_agent_and_user_axes_never_collapse(store):
    from agentmem_os.db.semantic_facts import make_scope_key
    _add(store, "The user's favorite color is blue.",
         agent_id="agent-A", user_id="user-1")
    _add(store, "The user's favorite color is blue.",
         agent_id="agent-A", user_id="user-2")
    k1 = make_scope_key("agent-A", "user-1")
    k2 = make_scope_key("agent-A", "user-2")
    assert k1 != k2
    assert len(store.current_facts(scope_key=k1)) == 1
    assert len(store.current_facts(scope_key=k2)) == 1
    # read path derives identically without hand-building keys
    assert len(store.current_facts(agent_id="agent-A", user_id="user-2")) == 1


def test_agent_name_user_name_and_literal_global_all_distinct(store):
    from agentmem_os.db.semantic_facts import make_scope_key
    assert make_scope_key("alice", None) != make_scope_key(None, "alice")
    assert make_scope_key(None, "global") != make_scope_key(None, None)
    _add(store, "The user has a cat.")                       # true global
    _add(store, "The user has a cat.", user_id="global")     # user literally named global
    assert len(store.current_facts()) == 1
    assert len(store.current_facts(user_id="global")) == 1


# ── Concurrency (G3 R1 F1/F2, R2 B1 — threaded, PRODUCTION engine config) ───

def test_threaded_writers_and_readers_lose_nothing(tmp_path):
    # Round-2 B1: the killer was never writer-vs-writer — it was an
    # ordinary concurrent READ closing its session and rolling back the
    # shared StaticPool connection under a writer (202/300 writes lost).
    # Fixed at the engine level (per-session-checkout pooling + WAL);
    # this test runs writers AND a hot reader loop on that real config.
    from agentmem_os.db.semantic_facts import SemanticFactStore

    engine = _make_production_engine(tmp_path / "conc.db")
    SessionLocal = _bootstrap(engine)
    store = SemanticFactStore(SessionLocal)

    N_SHARED, N_THREADS = 100, 4
    errors = []
    stop_reading = threading.Event()

    def writer(tid):
        for i in range(N_SHARED):
            try:
                store.add_fact(f"Shared fact number {i}.", fact_type="state",
                               t_mentioned="2023/05/20",
                               source_session_id="sess-1",
                               extraction_model=f"writer-{tid}")
            except Exception as e:
                errors.append(e)

    def reader():
        while not stop_reading.is_set():
            try:
                store.current_facts(limit=10)
                store.facts_as_of("2023/06/01", limit=10)
            except Exception as e:
                errors.append(e)

    r = threading.Thread(target=reader)
    r.start()
    threads = [threading.Thread(target=writer, args=(t,)) for t in range(N_THREADS)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    stop_reading.set()
    r.join()

    assert errors == []
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    rows = db.query(SemanticFact).all()
    db.close()
    assert len(rows) == N_SHARED                                 # no losses, no dupes
    assert sum(r.mention_count for r in rows) == N_SHARED * N_THREADS
    engine.dispose()


def test_real_engine_pool_and_concurrency():
    # R3 N4: the earlier concurrency test ran on a local COPY of the
    # engine construction — reverting db/engine.py's pooling would not
    # have failed it. This one binds to the real module (safe: conftest
    # forced AGENTMEM_OS_DB_PATH to scratch before any import).
    import agentmem_os.db.engine as eng
    from agentmem_os.db.models import Session as SessionRow
    from agentmem_os.db.semantic_facts import SemanticFactStore

    assert type(eng.engine.pool).__name__ == "QueuePool"   # pooling tripwire

    db = eng.get_session()
    if db.query(SessionRow).filter(
            SessionRow.session_id == "real-eng-sess").first() is None:
        db.add(SessionRow(session_id="real-eng-sess"))
        db.commit()
    db.close()

    store = SemanticFactStore(eng.get_session)
    errors, stop = [], threading.Event()

    def writer(tid):
        for i in range(50):
            try:
                store.add_fact(f"Real-engine fact {i}.", fact_type="state",
                               t_mentioned="2023/05/20",
                               source_session_id="real-eng-sess",
                               extraction_model=f"w{tid}")
            except Exception as e:
                errors.append(e)

    def reader():
        while not stop.is_set():
            try:
                store.current_facts(limit=5)
            except Exception as e:
                errors.append(e)

    r = threading.Thread(target=reader)
    r.start()
    ws = [threading.Thread(target=writer, args=(t,)) for t in range(3)]
    [t.start() for t in ws]
    [t.join() for t in ws]
    stop.set()
    r.join()

    assert errors == []
    rows = [f for f in store.current_facts(limit=1000)
            if f.fact_text.startswith("Real-engine fact")]
    assert len(rows) == 50
    assert sum(f.mention_count for f in rows) == 150


def test_same_text_different_fact_types_never_absorbed(store):
    # R3 N3: an extractor typing "vegetarian" as state in one session and
    # preference in another must produce two rows, not a silent absorption
    # that typed retrieval then under-returns.
    f1, c1 = _add(store, "The user is vegetarian.", fact_type="state")
    f2, c2 = _add(store, "The user is vegetarian.", fact_type="preference")
    assert c1 is True and c2 is True and f1.id != f2.id
    assert len(store.current_facts(fact_type="state")) == 1
    assert len(store.current_facts(fact_type="preference")) == 1


def test_supersede_walk_depth_guard(store, monkeypatch):
    # R3 N6: supersede()'s own chain walk must refuse pathological chains.
    from agentmem_os.db import semantic_facts as sf
    monkeypatch.setattr(sf, "_CHAIN_DEPTH_CAP", 5)
    ids = []
    for i in range(8):
        f, _ = _add(store, f"Deep chain fact {i}.")
        ids.append(f.id)
    for old, new in zip(ids, ids[1:]):
        store.supersede(old, new)
    lone, _ = _add(store, "A lone unsuperseded fact.")
    with pytest.raises(ValueError, match="too deep"):
        store.supersede(lone.id, ids[0])   # new=chain tail → walk exceeds cap


def test_race_fallback_refetch_none_reraises(store, monkeypatch):
    # R3 N6 (line 241): if the constraint fired but the winning row cannot
    # be found on re-fetch, the original IntegrityError must surface.
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Query
    _add(store, "The user collects stamps.")
    real_first = Query.first
    calls = {"n": 0}

    def blind_twice(self):
        calls["n"] += 1
        return None if calls["n"] <= 2 else real_first(self)

    monkeypatch.setattr(Query, "first", blind_twice)
    with pytest.raises(IntegrityError):
        _add(store, "The user collects stamps.")


def test_caller_session_dedup_race_raises_loudly(store_and_factory, monkeypatch):
    # R3 N6 (line 250): in a caller-owned batch we cannot roll back to
    # recover from a lost insert race — it must raise, never silently merge.
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import Query
    store, SessionLocal, _ = store_and_factory
    _add(store, "The user brews coffee.")
    real_first = Query.first
    calls = {"n": 0}

    def blind_once(self):
        calls["n"] += 1
        return None if calls["n"] == 1 else real_first(self)

    monkeypatch.setattr(Query, "first", blind_once)
    db = SessionLocal()
    with pytest.raises(IntegrityError):
        store.add_fact("The user brews coffee.", fact_type="state",
                       t_mentioned="2023/05/20", source_session_id="sess-1",
                       extraction_model="m", db=db)
    db.rollback()
    db.close()


def test_rowcount_guard_refuses_stale_supersede(store, monkeypatch):
    # R2 M2: the atomic conditional UPDATE is the cross-process defense
    # behind the pre-check. Exercise IT, not the pre-check: blind the
    # pre-check with a stale view of old (superseded_by=None) while the DB
    # already has it superseded — the rowcount guard must refuse.
    from types import SimpleNamespace
    from sqlalchemy.orm import Query

    a, _ = _add(store, "Fact A.")
    b, _ = _add(store, "Fact B.")
    c, _ = _add(store, "Fact C.")
    store.supersede(a.id, b.id)

    real_first = Query.first
    calls = {"n": 0}
    stale_old = SimpleNamespace(id=a.id, scope_key="global", superseded_by=None)

    def fake_first(self):
        calls["n"] += 1
        return stale_old if calls["n"] == 1 else real_first(self)

    monkeypatch.setattr(Query, "first", fake_first)
    with pytest.raises(ValueError, match="concurrently"):
        store.supersede(a.id, c.id)


def test_chain_depth_cap_raises_never_truncates_silently(store, monkeypatch):
    # R2 M3: a lineage longer than the cap must be an ERROR, not two
    # different silently-incomplete views depending on the entry point.
    from agentmem_os.db import semantic_facts as sf
    monkeypatch.setattr(sf, "_CHAIN_DEPTH_CAP", 5)
    ids = []
    for i in range(8):
        f, _ = _add(store, f"Chained fact {i}.", t_mentioned=f"2023/01/{i+1:02d}")
        ids.append(f.id)
    for old, new in zip(ids, ids[1:]):
        store.supersede(old, new)
    with pytest.raises(ValueError, match="exceeds"):
        store.chain(ids[0])


def test_racing_supersedes_have_exactly_one_winner(tmp_path):
    # Runs on the production-parity engine: with no in-process lock, race
    # outcomes are decided by the DB's conditional UPDATE across separate
    # connections — the shared-connection in-memory fixture cannot
    # represent that (it is the config the engine fix retired).
    from agentmem_os.db.semantic_facts import SemanticFactStore
    engine = _make_production_engine(tmp_path / "race.db")
    SessionLocal = _bootstrap(engine)
    store = SemanticFactStore(SessionLocal)

    old, _ = _add(store, "The user works at Acme.")
    b, _ = _add(store, "The user works at BetaCorp.")
    c, _ = _add(store, "The user works at CygnusInc.")
    outcomes = []

    def racer(new_id):
        try:
            store.supersede(old.id, new_id)
            outcomes.append(("won", new_id))
        except ValueError:
            outcomes.append(("refused", new_id))

    t1 = threading.Thread(target=racer, args=(b.id,))
    t2 = threading.Thread(target=racer, args=(c.id,))
    [t.start() for t in (t1, t2)]
    [t.join() for t in (t1, t2)]

    assert sorted(o[0] for o in outcomes) == ["refused", "won"]
    live_texts = {f.fact_text for f in store.current_facts()}
    assert "The user works at Acme." not in live_texts
    assert len(live_texts) == 2                                  # exactly one contradiction resolved
    assert len(store.chain(old.id)) == 2                         # old + the single winner


def _reaffirm_worker(args):
    # Module-level for pickling. Each "process" affirms the SAME fact with
    # its own session id + language — nothing may be lost (R3 N1).
    db_file, wid, n = args
    import sys as _sys
    _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from agentmem_os.db.semantic_facts import SemanticFactStore

    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 30},
    )

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _):
        c = dbapi_connection.cursor()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=30000")
        c.close()

    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False,
                                expire_on_commit=False)   # production parity (R5-7)
    store = SemanticFactStore(SessionLocal)
    errs = 0
    for _ in range(n):
        try:
            store.add_fact("The user speaks multiple languages.",
                           fact_type="state", t_mentioned="2023/05/20",
                           source_session_id=f"proc-sess-{wid}",
                           lang_source=f"lang-{wid}",
                           extraction_model=f"proc-{wid}")
        except Exception:
            errs += 1
    engine.dispose()
    return errs


def test_cross_process_reaffirmation_loses_nothing(tmp_path):
    # R3 N1 proof-of-fix: 3 OS processes × 20 affirmations of ONE fact —
    # exact mention_count, every source session, every language present.
    import multiprocessing as mp
    from sqlalchemy.orm import sessionmaker
    from agentmem_os.db.models import SemanticFact, Session as SessionRow
    from sqlalchemy import create_engine

    db_file = tmp_path / "xproc.db"
    engine = _make_production_engine(db_file)
    SessionLocal = _bootstrap(engine)
    seed = SessionLocal()
    for w in range(3):
        seed.add(SessionRow(session_id=f"proc-sess-{w}"))
    seed.commit()
    seed.close()

    ctx = mp.get_context("spawn")
    with ctx.Pool(3) as pool:
        errs = pool.map(_reaffirm_worker, [(str(db_file), w, 20) for w in range(3)])
    assert sum(errs) == 0

    db = SessionLocal()
    rows = db.query(SemanticFact).filter(
        SemanticFact.fact_text == "The user speaks multiple languages.").all()
    db.close()
    engine.dispose()
    assert len(rows) == 1
    assert rows[0].mention_count == 60                       # exact, none lost
    assert set(rows[0].source_session_ids) == {f"proc-sess-{w}" for w in range(3)}
    assert set(rows[0].langs) == {f"lang-{w}" for w in range(3)}


def test_reaffirm_preserves_caller_staged_edits(store_and_factory):
    # R5-1: expire_all() silently destroyed unflushed caller edits in a
    # batch. Stage an edit on a loaded object, re-affirm via db=, and the
    # edit must survive and persist.
    from agentmem_os.db.models import Turn
    store, SessionLocal, _ = store_and_factory
    _add(store, "The user gardens on weekends.")
    db = SessionLocal()
    turn = db.query(Turn).filter(Turn.id == 101).first()
    turn.content = "MARKED-BY-CALLER"
    f, created = store.add_fact("The user gardens on weekends.",
                                fact_type="state", t_mentioned="2023/05/20",
                                source_session_id="sess-1",
                                extraction_model="m", db=db)
    assert created is False and f.mention_count == 2
    assert turn.content == "MARKED-BY-CALLER"      # staged edit intact
    db.commit()
    db.close()
    db = SessionLocal()
    assert db.query(Turn).filter(Turn.id == 101).first().content == "MARKED-BY-CALLER"
    db.close()


def test_reaffirm_vanished_fact_raises(store, monkeypatch):
    # R5-5: cover the vanished-row guard (concurrent hard delete).
    from sqlalchemy.orm import Query
    _add(store, "The user paints.")
    real_update = Query.update
    monkeypatch.setattr(Query, "update", lambda self, *a, **k: 0)
    with pytest.raises(ValueError, match="vanished"):
        _add(store, "The user paints.")
    monkeypatch.setattr(Query, "update", real_update)


def test_legacy_null_mention_count_recovers(store_and_factory):
    # R6 N2 tripwire: legacy rows (real dev DB shape) have nullable
    # mention_count; the coalesce increment must recover NULL -> 2, never
    # NULL forever. Mutation-tested: dropping coalesce turns this red.
    from sqlalchemy import text as sql_text
    store, SessionLocal, engine = store_and_factory
    f, _ = _add(store, "The user kayaks.")
    with engine.connect() as c:
        # Recreate the LEGACY column shape (nullable, no default) — the
        # real dev DB's — since model constraints only reach new tables.
        ddl = c.execute(sql_text(
            "SELECT sql FROM sqlite_master WHERE name='semantic_facts'"
        )).scalar_one()
        legacy_ddl = ddl.replace("mention_count INTEGER DEFAULT 1 NOT NULL",
                                 "mention_count INTEGER")
        assert legacy_ddl != ddl
        c.execute(sql_text("PRAGMA foreign_keys=OFF"))
        c.execute(sql_text("ALTER TABLE semantic_facts RENAME TO sf_old"))
        c.execute(sql_text(legacy_ddl))
        c.execute(sql_text("INSERT INTO semantic_facts SELECT * FROM sf_old"))
        c.execute(sql_text("DROP TABLE sf_old"))
        c.execute(sql_text("UPDATE semantic_facts SET mention_count = NULL"))
        c.execute(sql_text("PRAGMA foreign_keys=ON"))
        c.commit()
    f2, created = _add(store, "The user kayaks.")
    assert created is False and f2.mention_count == 2


# ── Windowing, ranges, provenance ────────────────────────────────────────────

def test_superseded_facts_never_leak_through_limit_windows(store):
    ids = []
    for i in range(5):
        f, _ = _add(store, f"The user's favorite number is {i}.",
                    t_mentioned=f"2023/01/{10 + i:02d}")
        ids.append(f.id)
    for old, new in zip(ids, ids[1:]):
        store.supersede(old, new)
    for limit in (1, 2, 3, 10):
        assert [f.id for f in store.current_facts(limit=limit)] == [ids[-1]]


def test_interval_vs_point_precision_never_merges_either_order(store):
    # R2 B4: "sometime in October" (interval) and "on October 1st" (point)
    # are different claims even though the interval starts that day — and
    # the stored precision must not depend on insertion order.
    for first, second in ((("2023/10",), ("2023/10/01",)),
                          (("2023/10/01",), ("2023/10",))):
        store_local_texts = f"The user hiked Mount Tam. [{first[0]}-case]"
        _add(store, store_local_texts, fact_type="event", t_occurred=first[0])
        f2, created = _add(store, store_local_texts, fact_type="event",
                           t_occurred=second[0])
        assert created is True, f"order {first[0]}→{second[0]} merged precision"
    events = store.current_facts(fact_type="event", limit=50)
    assert len(events) == 4


def test_facts_overlapping_fact_type_filter(store):
    _add(store, "The user attended a concert.", fact_type="event",
         t_occurred="2023/10/05")
    _add(store, "The user liked October weather.", fact_type="preference",
         t_occurred="2023/10/05")
    hits = store.facts_overlapping("2023/10/01", "2023/10/31",
                                   fact_type="event")
    assert len(hits) == 1 and hits[0].fact_type == "event"


def test_month_only_facts_survive_range_queries(store):
    # G3 finding 8: '2023/10' must not fall out of an October filter.
    _add(store, "The user rode three rollercoasters at SeaWorld.",
         fact_type="event", t_occurred="2023/10")
    _add(store, "The user rode the Xcelerator.", fact_type="event",
         t_occurred="2023/10/08")
    _add(store, "The user attended a July concert.", fact_type="event",
         t_occurred="2023/07/04")
    hits = store.facts_overlapping("2023/10/01", "2023/10/31")
    assert len(hits) == 2
    assert not any("July" in f.fact_text for f in hits)


def test_contains_filter_escapes_like_wildcards(store):
    _add(store, "The user saved 100% of the bonus.")
    _add(store, "The user saved some money.")
    hits = store.current_facts(contains="100%")
    assert len(hits) == 1 and "100%" in hits[0].fact_text


def test_facts_as_of_respects_fact_type_and_limit(store):
    _add(store, "The user likes tea.", fact_type="preference",
         t_mentioned="2023/01/10")
    _add(store, "The user works at Google.", t_mentioned="2023/01/11")
    prefs = store.facts_as_of("2023/02/01", fact_type="preference")
    assert len(prefs) == 1 and prefs[0].fact_type == "preference"
    assert len(store.facts_as_of("2023/02/01", limit=1)) == 1


def test_provenance_reports_intact_and_broken_citations(store):
    f, _ = _add(store, "The user works at Google.", source_turn_ids=[101, 102])
    p = store.provenance(f.id)
    assert p["citations_intact"] is True and p["turns_resolved"] == [101, 102]
    g, _ = _add(store, "The user visited Japan.", source_turn_ids=[999])
    p2 = store.provenance(g.id)
    assert p2["citations_intact"] is False and p2["turns_resolved"] == []


def test_cross_process_race_falls_back_to_reaffirmation(store, monkeypatch):
    # Simulate a SECOND PROCESS landing the row between our pre-check and
    # our commit: blind the pre-check exactly once, so the insert hits the
    # unique constraint and must recover via re-affirmation. (Also guards
    # the SQLite reality that unique errors name COLUMNS, not constraints.)
    from sqlalchemy.orm import Query
    _add(store, "The user plays chess.", source_turn_ids=[101])

    real_first = Query.first
    calls = {"n": 0}

    def blind_once(self):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return real_first(self)

    monkeypatch.setattr(Query, "first", blind_once)
    fact, created = _add(store, "The user plays chess.", source_turn_ids=[102])
    assert created is False
    assert fact.mention_count == 2
    assert fact.source_turn_ids == [101, 102]


def test_provenance_with_no_citations(store):
    f, _ = _add(store, "An uncited fact.")
    p = store.provenance(f.id)
    assert p["citations_intact"] is True and p["source_turn_ids"] == []


# ── Transactional batching (G3 finding 14) ───────────────────────────────────

def test_caller_session_batch_aborts_atomically(store_and_factory):
    store, SessionLocal, _ = store_and_factory
    db = SessionLocal()
    store.add_fact("Batch fact one.", fact_type="state",
                   t_mentioned="2023/05/20", source_session_id="sess-1",
                   extraction_model="m", db=db)
    store.add_fact("Batch fact two.", fact_type="state",
                   t_mentioned="2023/05/20", source_session_id="sess-1",
                   extraction_model="m", db=db)
    db.rollback()   # simulated mid-consolidation crash
    db.close()
    assert store.current_facts() == []          # nothing half-committed

    db = SessionLocal()
    f1, _ = store.add_fact("Batch fact one.", fact_type="state",
                           t_mentioned="2023/05/20", source_session_id="sess-1",
                           extraction_model="m", db=db)
    f1b, created = store.add_fact("Batch fact one.", fact_type="state",
                                  t_mentioned="2023/05/20",
                                  source_session_id="sess-2",
                                  extraction_model="m", db=db)
    assert created is False and f1b.mention_count == 2   # in-batch re-affirm (R2 M1)
    f2, _ = store.add_fact("Batch fact two.", fact_type="state",
                           t_mentioned="2023/05/21", source_session_id="sess-1",
                           extraction_model="m", db=db)
    store.supersede(f1.id, f2.id, db=db)        # supersession joins the batch
    db.commit()
    db.close()
    live = store.current_facts()
    assert [f.id for f in live] == [f2.id]      # committed atomically


# ── The DB as final dedup authority + query plans on REAL store SQL ─────────

def test_unique_constraint_is_final_authority(store_and_factory):
    from sqlalchemy.exc import IntegrityError
    from agentmem_os.db.models import SemanticFact
    from agentmem_os.db.semantic_facts import fact_hash
    store, SessionLocal, _ = store_and_factory
    _add(store, "The user runs marathons.")
    db = SessionLocal()
    db.add(SemanticFact(
        scope_key="global", fact_text="THE USER RUNS MARATHONS.",
        fact_type="state", t_mentioned="2023/05/20",
        source_session_id="sess-1", extraction_model="test-model",
        normalized_hash=fact_hash("THE USER RUNS MARATHONS.", "state"),
    ))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
    db.close()


def _captured_plans(engine, call):
    """Run `call` and EXPLAIN the exact SELECT statements the store EMITTED
    — never a re-declared lookalike (R1 F10, repeated as R2 B5: a
    re-declared query goes green even if the store's real query regresses
    tomorrow)."""
    import re as _re
    from sqlalchemy import event as sa_event

    captured = []

    def grab(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            captured.append((statement, parameters))

    sa_event.listen(engine, "before_cursor_execute", grab)
    try:
        call()
    finally:
        sa_event.remove(engine, "before_cursor_execute", grab)

    plans = []
    with engine.connect() as conn:
        for stmt, params in captured:
            rows = conn.exec_driver_sql(
                f"EXPLAIN QUERY PLAN {stmt}", params).fetchall()
            plans.append(" | ".join(str(r) for r in rows))
    return plans


def test_store_queries_use_partial_indexes_no_temp_btree(store_and_factory):
    import re as _re
    store, SessionLocal, engine = store_and_factory
    _add(store, "Seed fact.", fact_type="event", t_occurred="2023/10/08")

    typed_plans = _captured_plans(
        engine, lambda: store.current_facts(fact_type="event", limit=10))
    untyped_plans = _captured_plans(
        engine, lambda: store.current_facts(limit=10))

    for plans, idx_pattern in (
            (typed_plans, r"USING (?:COVERING )?INDEX idx_facts_current\b(?!_)"),
            (untyped_plans, r"USING (?:COVERING )?INDEX idx_facts_current_all\b")):
        joined = " || ".join(plans)
        assert _re.search(idx_pattern, joined), f"index miss in: {joined}"
        assert "TEMP B-TREE" not in joined.upper(), f"temp sort in: {joined}"


# ── Migration verification (G3 findings 12, 13) ─────────────────────────────

def test_migration_detects_missing_dedup_constraint(tmp_path):
    # A semantic_facts table WITHOUT the unique constraint can never be
    # repaired in place — init-time verification must refuse it loudly.
    import sqlite3
    bare = tmp_path / "bare.db"
    conn = sqlite3.connect(bare)
    conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, "
                 "scope_key TEXT, normalized_hash TEXT)")
    conn.execute("CREATE TABLE kg_edges (id INTEGER PRIMARY KEY, "
                 "source_id INTEGER, relation_type TEXT, valid_until DATETIME)")
    conn.commit()
    conn.close()

    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(bare)
        with pytest.raises(RuntimeError, match="UNIQUE index"):
            eng._migrate_semantic_tier()
    finally:
        eng.DB_PATH = original


def test_migration_rejects_wrong_unique_constraint(tmp_path):
    # R2 B2: "some unique autoindex exists" is NOT dedup — a table with
    # UNIQUE(fact_text) must be refused, because dedup on the REAL key
    # (scope_key, normalized_hash) is unenforced.
    import sqlite3
    wrong = tmp_path / "wrong.db"
    conn = sqlite3.connect(wrong)
    conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, "
                 "scope_key TEXT, normalized_hash TEXT, fact_text TEXT, "
                 "UNIQUE(fact_text))")
    conn.execute("CREATE TABLE kg_edges (id INTEGER PRIMARY KEY, "
                 "source_id INTEGER, relation_type TEXT, valid_until DATETIME)")
    conn.commit()
    conn.close()

    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(wrong)
        with pytest.raises(RuntimeError, match="UNIQUE index"):
            eng._migrate_semantic_tier()
    finally:
        eng.DB_PATH = original


def test_migration_rejects_partial_unique_index(tmp_path):
    # R3 N5: a PARTIAL unique index on the right columns enforces nothing
    # outside its predicate — the verifier must refuse it.
    import sqlite3
    partial = tmp_path / "partial.db"
    conn = sqlite3.connect(partial)
    conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY, "
                 "scope_key TEXT, normalized_hash TEXT, superseded_by INTEGER)")
    conn.execute("CREATE UNIQUE INDEX uq_partial ON semantic_facts"
                 "(scope_key, normalized_hash) WHERE superseded_by IS NULL")
    conn.execute("CREATE TABLE kg_edges (id INTEGER PRIMARY KEY, "
                 "source_id INTEGER, relation_type TEXT, valid_until DATETIME)")
    conn.commit()
    conn.close()

    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(partial)
        with pytest.raises(RuntimeError, match="UNIQUE index"):
            eng._migrate_semantic_tier()
    finally:
        eng.DB_PATH = original


def test_migration_repairs_missing_index_on_target_db(tmp_path):
    # R2 B3: the repair must land on the DB being migrated (DB_PATH), not
    # on whatever the module-level engine points at.
    import sqlite3
    from sqlalchemy import create_engine
    from agentmem_os.db.models import Base

    target = tmp_path / "target.db"
    e = create_engine(f"sqlite:///{target}")
    Base.metadata.create_all(bind=e)
    e.dispose()
    conn = sqlite3.connect(target)
    conn.execute("DROP INDEX idx_facts_mentioned")
    conn.commit()
    conn.close()

    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(target)
        report = eng._migrate_semantic_tier()
        assert report["semantic_facts"] == "verified"
        conn = sqlite3.connect(target)
        idx = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='semantic_facts'")}
        conn.close()
        assert "idx_facts_mentioned" in idx      # repaired IN PLACE, on target
    finally:
        eng.DB_PATH = original


def test_migration_reports_on_healthy_db(tmp_path):
    import sqlite3
    from sqlalchemy import create_engine
    from agentmem_os.db.models import Base

    healthy = tmp_path / "healthy.db"
    e = create_engine(f"sqlite:///{healthy}")
    Base.metadata.create_all(bind=e)

    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(healthy)
        report = eng._migrate_semantic_tier()
        assert report["kg_edges_index"] == "present"
        assert report["semantic_facts"] == "verified"
        conn = sqlite3.connect(healthy)
        idx = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='kg_edges'")}
        conn.close()
        assert "idx_kg_edges_active" in idx
    finally:
        eng.DB_PATH = original
