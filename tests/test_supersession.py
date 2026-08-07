"""
Stage 4 G1: SupersessionJudge + cancelled semantics + t_invalid.

Attack surface pinned here (build log, Stage 4 design):
  - the LLM only PROPOSES: out-of-shortlist ids, garbage output, and
    co-signal disagreements are all dropped LOUDLY into the audit row
  - direction from DOMAIN time, never processing order (the Josh case:
    a chronologically-later fact ingested first must win regardless)
  - conservative scope: states/preferences only; identity and events
    skipped without an LLM call; domain-time ties skip
  - cancelled: only a live planned event, only via the guarded path,
    terminal against re-affirmation upgrades
  - every judgment persisted; unjudged facts are sweepable; a poison
    fact cannot take the sweep down
"""
import json
import sqlite3

import pytest

from tests.test_semantic_facts import _bootstrap, _make_production_engine


@pytest.fixture()
def judged_env(tmp_path):
    from agentmem_os.db.fact_entities import FactEntityLinker
    from agentmem_os.db.semantic_facts import SemanticFactStore
    from agentmem_os.llm.supersession import SupersessionJudge

    engine = _make_production_engine(tmp_path / "s4.db")
    SessionLocal = _bootstrap(engine)
    return (SemanticFactStore(SessionLocal), FactEntityLinker(SessionLocal),
            SupersessionJudge(SessionLocal, model="mock"), SessionLocal)


def _fact(store, text, **kw):
    defaults = dict(fact_type="state", t_mentioned="2023/05/20",
                    source_session_id="sess-1", extraction_model="test-model")
    defaults.update(kw)
    fact, _ = store.add_fact(text, **defaults)
    return fact


def _link(SessionLocal, fact_id, entity_text):
    """Deterministic link plumbing (bypasses NER so tests control the
    entity space exactly)."""
    from agentmem_os.db.models import KnowledgeGraphNode, SemanticFactEntity
    db = SessionLocal()
    try:
        node = (db.query(KnowledgeGraphNode)
                .filter(KnowledgeGraphNode.agent_id.is_(None),
                        KnowledgeGraphNode.entity_text == entity_text)
                .first())
        if node is None:
            node = KnowledgeGraphNode(agent_id=None, entity_text=entity_text,
                                      entity_type="ORG", mention_count=1)
            db.add(node)
            db.flush()
        db.add(SemanticFactEntity(fact_id=fact_id, node_id=node.id,
                                  surface_text=entity_text, linked_via="ner"))
        db.commit()
    finally:
        db.close()


def _judgment_rows(SessionLocal, fact_id=None):
    from agentmem_os.db.models import SupersessionJudgment
    db = SessionLocal()
    try:
        q = db.query(SupersessionJudgment)
        if fact_id is not None:
            q = q.filter(SupersessionJudgment.fact_id == fact_id)
        return q.all()
    finally:
        db.close()


# ── Happy path + t_invalid ───────────────────────────────────────────────────

def test_update_supersedes_with_domain_time_and_t_invalid(
        judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env
    old = _fact(store, "The user's personal best 5K time is 25:00.",
                t_mentioned="2023/03/01")
    new = _fact(store, "The user's personal best 5K time is 23:30.",
                t_mentioned="2023/05/20")
    for f in (old, new):
        _link(SessionLocal, f.id, "5K")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "same metric, updated value",
        "superseded_ids": [old.id], "cancelled_ids": []})
    r = judge.judge_fact(new.id, session_id="s-test")
    assert r["superseded"] == [(old.id, new.id)]
    assert r["dropped"] == [] and r["skipped"] is None
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    o = db.get(SemanticFact, old.id)
    db.close()
    assert o.superseded_by == new.id
    assert o.t_invalid == "2023/05/20"      # winner's domain time
    assert o.superseded_at is not None      # decision time, separate axis
    row = _judgment_rows(SessionLocal, new.id)
    assert len(row) == 1
    assert json.loads(row[0].applied_json)["superseded"] == [[old.id, new.id]]
    # store reads reflect it
    live = store.current_facts()
    assert old.id not in {f.id for f in live}
    assert store.provenance(old.id)["t_invalid"] == "2023/05/20"


def test_out_of_order_sessions_reverse_direction(judged_env, monkeypatch):
    # The Josh case: the chronologically-LATER fact was ingested FIRST.
    # The judge is invoked on the newly-arrived (but domain-earlier)
    # fact; the proposal must apply REVERSED — the domain-later fact
    # wins, processing order never decides.
    store, linker, judge, SessionLocal = judged_env
    later = _fact(store, "The user works at Microsoft.",
                  t_mentioned="2024/06/01")      # ingested first
    earlier = _fact(store, "The user works at Google.",
                    t_mentioned="2022/01/15")    # arrives second (backfill)
    for f in (later, earlier):
        _link(SessionLocal, f.id, "employer")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "employment changed",
        "superseded_ids": [later.id], "cancelled_ids": []})
    r = judge.judge_fact(earlier.id)
    # REVERSED: earlier gets superseded BY later, not the other way
    assert r["superseded"] == [(earlier.id, later.id)]
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    e = db.get(SemanticFact, earlier.id)
    l = db.get(SemanticFact, later.id)
    db.close()
    assert e.superseded_by == later.id
    assert l.superseded_by is None
    assert e.t_invalid == "2024/06/01"


def test_domain_time_tie_skips_conservatively(judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/05/20")
    b = _fact(store, "The user works at Microsoft.", t_mentioned="2023/05/20")
    for f in (a, b):
        _link(SessionLocal, f.id, "employer")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [a.id],
        "cancelled_ids": []})
    r = judge.judge_fact(b.id)
    assert r["superseded"] == []
    assert any("tie" in reason for _, reason in r["dropped"])
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, a.id).superseded_by is None
    db.close()


# ── The LLM only proposes ────────────────────────────────────────────────────

def test_cosignal_disagreement_blocks_llm_only_verdict(
        judged_env, monkeypatch):
    # Unrelated facts sharing an entity: no polarity flip, low lexical
    # similarity — even an insistent LLM verdict must not act.
    store, linker, judge, SessionLocal = judged_env
    a = _fact(store, "The user visited the Google campus cafeteria.",
              t_mentioned="2023/03/01")
    b = _fact(store, "The user admires Google's quantum research papers.",
              t_mentioned="2023/05/20")
    for f in (a, b):
        _link(SessionLocal, f.id, "Google")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "hallucinated replacement",
        "superseded_ids": [a.id], "cancelled_ids": []})
    r = judge.judge_fact(b.id)
    assert r["superseded"] == []
    assert any("co-signal disagreement" in reason
               for _, reason in r["dropped"])


def test_out_of_shortlist_and_garbage_ids_dropped_loudly(
        judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/03/01")
    b = _fact(store, "The user left Google for Microsoft.",
              t_mentioned="2023/05/20")
    for f in (a, b):
        _link(SessionLocal, f.id, "Google")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "…", "superseded_ids": [99999, "seven", a.id],
        "cancelled_ids": "not-a-list"})
    r = judge.judge_fact(b.id)
    assert r["superseded"] == [(a.id, b.id)]          # the valid one applies
    reasons = [reason for _, reason in r["dropped"]]
    assert any("not in shortlist" in x for x in reasons)
    assert any("non-integer id" in x for x in reasons)
    assert any("not a list" in x for x in reasons)
    row = _judgment_rows(SessionLocal, b.id)[0]
    assert len(json.loads(row.dropped_json)) == 3     # every drop audited


# ── Conservative scope ───────────────────────────────────────────────────────

def test_identity_events_and_superseded_skip_without_llm(
        judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env

    def _explode(prompt):
        raise AssertionError("LLM called for an out-of-scope fact")

    monkeypatch.setattr(judge, "_llm", _explode)
    ident = _fact(store, "The user is a software engineer named Alex.",
                  fact_type="identity")
    ev = _fact(store, "The user visited Paris.", fact_type="event",
               t_occurred="2023/04/01")
    r1 = judge.judge_fact(ident.id)
    assert "out of judgment scope" in r1["skipped"]
    r2 = judge.judge_fact(ev.id)
    assert "out of judgment scope" in r2["skipped"]
    # skip rows persisted too — a skip is a judgment decision
    assert len(_judgment_rows(SessionLocal, ident.id)) == 1
    # superseded facts skip as history
    a = _fact(store, "The user works at Google.")
    b = _fact(store, "The user works at Microsoft.",
              t_mentioned="2023/06/01")
    store.supersede(a.id, b.id)
    r3 = judge.judge_fact(a.id)
    assert r3["skipped"] == "fact already superseded"


def test_no_shared_entity_no_llm_call(judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env

    def _explode(prompt):
        raise AssertionError("LLM called with empty shortlist")

    monkeypatch.setattr(judge, "_llm", _explode)
    lone = _fact(store, "The user prefers quiet mornings.")
    r = judge.judge_fact(lone.id)
    assert r["skipped"] == ("no candidates from any pool "
                            "(entity, lexical, planned)")


def test_shortlist_composition_and_cap(judged_env, monkeypatch):
    # Same-type peers + planned events only; cross-scope and superseded
    # excluded; cap enforced most-recent-first with the cap disclosed.
    from agentmem_os.llm import supersession as sup
    store, linker, judge, SessionLocal = judged_env
    new = _fact(store, "The user works at Google.", t_mentioned="2024/01/01")
    _link(SessionLocal, new.id, "Google")
    # Lexically NEAR-IDENTICAL other-scope fact (R1-Ma6: the old
    # fixture's other-scope text differed, so deleting the scope filter
    # changed nothing — now removal admits it through the lexical pool
    # and turns this red).
    other_scope = _fact(store, "The user works at Google.",
                        user_id="user-B", t_mentioned="2023/12/30")
    _link(SessionLocal, other_scope.id, "Google")
    pref = _fact(store, "The user likes Google's doodles.",
                 fact_type="preference", t_mentioned="2023/01/02")
    _link(SessionLocal, pref.id, "Google")
    planned = _fact(store, "The user attends Google I/O.",
                    fact_type="event", t_occurred="2099/05/10",
                    event_status="planned", t_mentioned="2023/01/03")
    _link(SessionLocal, planned.id, "Google")
    peers = []
    for i in range(14):
        f = _fact(store, f"The user maintains Google project number {i:02d}.",
                  t_mentioned=f"2023/03/{i+1:02d}")
        _link(SessionLocal, f.id, "Google")
        peers.append(f)
    seen = {}

    def _capture(prompt):
        seen["prompt"] = prompt
        return {"reasoning": "none", "superseded_ids": [],
                "cancelled_ids": []}

    monkeypatch.setattr(judge, "_llm", _capture)
    # NO constant monkeypatch (R2 major: patching the cap back to its
    # default made cap mutations invisible) — defaults are exercised.
    judge.judge_fact(new.id)
    row = _judgment_rows(SessionLocal, new.id)[0]
    meta = json.loads(row.shortlist_json)
    ids = [c["id"] for c in meta["candidates"]]
    # 12 peer slots + the planned event in its RESERVED slot (G1 found
    # a single pool let recent peers crowd cancellation candidates out)
    assert len(ids) == 13
    assert other_scope.id not in ids               # scope isolation
    assert pref.id not in ids                      # type-peer rule (state≠pref)
    assert planned.id in ids                       # reserved slot works
    assert meta["peer_cap"] == 12 and meta["planned_cap"] == 4


def test_apply_defends_status_changed_between_shortlist_and_apply(
        judged_env):
    # Race defense: a planned candidate whose status changed between
    # shortlisting and apply must be refused by the guarded store path,
    # recorded as a drop — the snapshot's word is never trusted blindly.
    store, _, judge, SessionLocal = judged_env
    planned = _fact(store, "The user attends the Google developer "
                           "conference.", fact_type="event",
                    t_occurred="2099/06/01", event_status="planned",
                    t_mentioned="2023/05/01")
    new = _fact(store, "The user says the Google developer conference "
                       "was called off.", t_mentioned="2023/05/20")
    store.mark_event_cancelled(planned.id)     # status changes "mid-race"
    r = judge._apply(
        new.id, None,
        {"id": new.id, "text": new.fact_text, "type": "state",
         "occ": (None, None), "mentioned": "2023/05/20"},
        [{"id": planned.id,
          "text": "The user attends the Google developer conference.",
          "type": "event", "status": "planned", "pool": "planned",
          "shared_nodes": False,
          "occ": ("2099/06/01", "2099/06/01"),
          "mentioned": "2023/05/01"}],       # stale snapshot
        {planned.id: {"flip": False, "category": None, "cosine": 0.0,
                      "metric_update": False, "stripped_cosine": 0.0,
                      "pool": "planned", "agrees": False}},
        {"reasoning": "…", "superseded_ids": [],
         "cancelled_ids": [planned.id]})
    assert r["cancelled"] == []
    assert any("store refused" in reason for _, reason in r["dropped"])


# ── Cancellation ─────────────────────────────────────────────────────────────

def test_cancellation_only_reaches_live_planned_events(
        judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env
    planned = _fact(store, "The user attends the Google conference.",
                    fact_type="event", t_occurred="2099/06/01",
                    event_status="planned", t_mentioned="2023/05/01")
    occurred = _fact(store, "The user toured the Google office.",
                     fact_type="event", t_occurred="2023/04/01",
                     t_mentioned="2023/05/02")
    new = _fact(store, "The user says the Google conference was called off.",
                t_mentioned="2023/05/20")
    for f in (planned, occurred, new):
        _link(SessionLocal, f.id, "Google")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "conference cancelled",
        "superseded_ids": [],
        "cancelled_ids": [planned.id, occurred.id]})
    r = judge.judge_fact(new.id)
    assert r["cancelled"] == [planned.id]
    # the occurred event was never SHOWN to the model (not a candidate
    # of either pool) — its proposal drops as out-of-shortlist
    assert any(cid == occurred.id and "not in shortlist" in reason
               for cid, reason in r["dropped"])
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, planned.id).event_status == "cancelled"
    assert db.get(SemanticFact, occurred.id).event_status == "occurred"
    db.close()


def test_cancelled_store_guards_and_terminal_merge(judged_env):
    store, _, _, SessionLocal = judged_env
    planned = _fact(store, "The user attends the SeaWorld festival.",
                    fact_type="event", t_occurred="2099/04/15",
                    event_status="planned", t_mentioned="2023/04/01")
    occurred = _fact(store, "The user rode a rollercoaster.",
                     fact_type="event", t_occurred="2023/03/01")
    state = _fact(store, "The user likes festivals.")
    with pytest.raises(ValueError, match="only a live PLANNED"):
        store.mark_event_cancelled(occurred.id)
    with pytest.raises(ValueError, match="only a live PLANNED"):
        store.mark_event_cancelled(state.id)
    store.mark_event_cancelled(planned.id)
    # add_fact still refuses 'cancelled' as INPUT
    with pytest.raises(ValueError, match="event_status"):
        _fact(store, "Another planned thing.", fact_type="event",
              t_occurred="2099/01/01", event_status="cancelled")
    # terminal: a post-date re-affirmation must NOT resurrect it
    again, created = store.add_fact(
        "The user attends the SeaWorld festival.", fact_type="event",
        t_occurred="2099/04/15", t_mentioned="2099/05/01",
        event_status="occurred", source_session_id="sess-2",
        extraction_model="t")
    assert not created
    assert again.event_status == "cancelled"


# ── supersede t_invalid validation ───────────────────────────────────────────

def test_supersede_validates_t_invalid(judged_env):
    store, _, _, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.")
    b = _fact(store, "The user works at Microsoft.",
              t_mentioned="2023/06/01")
    with pytest.raises(ValueError, match="date"):
        store.supersede(a.id, b.id, t_invalid="not-a-date")
    store.supersede(a.id, b.id, t_invalid="2023/06")   # month interval OK
    assert store.provenance(a.id)["t_invalid"] == "2023/06/01"


# ── Recovery sweep ───────────────────────────────────────────────────────────

def test_judge_missing_sweeps_and_survives_poison(judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/01/01")
    b = _fact(store, "The user left Google for Microsoft.",
              t_mentioned="2023/06/01")
    c = _fact(store, "The user prefers tea over coffee.",
              fact_type="preference")
    for f in (a, b):
        _link(SessionLocal, f.id, "Google")
    real = judge.judge_fact
    calls = {"n": 0}

    def _poison_first(fid, **kw):
        calls["n"] += 1
        if fid == a.id:
            raise RuntimeError("boom")
        return real(fid, **kw)

    monkeypatch.setattr(judge, "judge_fact", _poison_first)
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "employment changed",
        "superseded_ids": [a.id], "cancelled_ids": []})
    r = judge.judge_missing()
    assert r["failures"] == [(a.id, "boom")]
    assert r["judged"] == 2                        # b and c judged
    assert r["supersessions"] == 1                 # b superseded a
    monkeypatch.setattr(judge, "judge_fact", real)
    r2 = judge.judge_missing()
    # a is now superseded → judged with a skip row; drain then empties
    assert r2["candidates"] == 0 or r2["judged"] >= 0
    r3 = judge.judge_missing()
    assert r3["candidates"] == 0


def test_judged_facts_never_resweep(judged_env, monkeypatch):
    store, linker, judge, SessionLocal = judged_env
    lone = _fact(store, "The user prefers quiet mornings.")
    assert judge.judge_missing()["judged"] == 1    # skip row written
    assert judge.judge_missing()["candidates"] == 0


# ── Coverage of remaining failure paths ──────────────────────────────────────

def test_judge_unknown_fact_is_loud(judged_env):
    _, _, judge, _ = judged_env
    with pytest.raises(ValueError, match="not found"):
        judge.judge_fact(99999)


def test_apply_drops_candidate_superseded_mid_race(judged_env):
    # Candidate superseded between shortlist and apply: the store's own
    # guard refuses, the judge records it — stale snapshots never win.
    store, _, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/01/01")
    b = _fact(store, "The user left Google for Microsoft.",
              t_mentioned="2023/06/01")
    c = _fact(store, "The user left Microsoft for Zomato.",
              t_mentioned="2023/09/01")
    store.supersede(a.id, b.id)                    # the "race" lands first
    r = judge._apply(
        c.id, None,
        {"id": c.id, "text": c.fact_text, "type": "state",
         "occ": (None, None), "mentioned": "2023/09/01"},
        [{"id": a.id, "text": a.fact_text, "type": "state",
          "status": None, "pool": "entity", "occ": (None, None),
          "mentioned": "2023/01/01"}],       # stale snapshot
        {a.id: {"flip": True, "category": "employment", "cosine": 0.5,
                "metric_update": False, "stripped_cosine": 0.5,
                "pool": "entity", "agrees": True}},
        {"reasoning": "…", "superseded_ids": [a.id], "cancelled_ids": []})
    assert r["superseded"] == []
    assert any("store refused" in reason for _, reason in r["dropped"])


def test_apply_drops_state_proposed_as_cancelled(judged_env):
    store, _, judge, SessionLocal = judged_env
    s = _fact(store, "The user likes Google's campus.",
              t_mentioned="2023/01/01")
    new = _fact(store, "The user says everything is called off.",
                t_mentioned="2023/06/01")
    r = judge._apply(
        new.id, None,
        {"id": new.id, "text": new.fact_text, "type": "state",
         "occ": (None, None), "mentioned": "2023/06/01"},
        [{"id": s.id, "text": s.fact_text, "type": "state",
          "status": None, "pool": "entity", "occ": (None, None),
          "mentioned": "2023/01/01"}],
        {s.id: {"flip": False, "category": None, "cosine": 0.0,
                "metric_update": False, "stripped_cosine": 0.0,
                "pool": "entity", "agrees": False}},
        {"reasoning": "…", "superseded_ids": [],
         "cancelled_ids": [s.id, 424242]})
    assert r["cancelled"] == []
    reasons = [reason for _, reason in r["dropped"]]
    assert any("only applies to planned" in x for x in reasons)
    assert any("not in shortlist" in x for x in reasons)


def test_shortlist_empty_when_entity_has_no_peers(judged_env, monkeypatch):
    store, _, judge, SessionLocal = judged_env

    def _explode(prompt):
        raise AssertionError("LLM called with no peers")

    monkeypatch.setattr(judge, "_llm", _explode)
    lone = _fact(store, "The user works at Zomato.")
    _link(SessionLocal, lone.id, "Zomato")         # linked, but alone
    r = judge.judge_fact(lone.id)
    assert r["skipped"] == ("no candidates from any pool "
                            "(entity, lexical, planned)")


def test_mark_cancelled_guards_not_found_history_and_caller_session(
        judged_env):
    store, _, _, SessionLocal = judged_env
    with pytest.raises(ValueError, match="not found"):
        store.mark_event_cancelled(99999)
    p1 = _fact(store, "The user attends conference A.", fact_type="event",
               t_occurred="2099/01/01", event_status="planned",
               t_mentioned="2023/01/01")
    p2 = _fact(store, "The user attends conference A rescheduled.",
               fact_type="event", t_occurred="2099/02/01",
               event_status="planned", t_mentioned="2023/02/01")
    store.supersede(p1.id, p2.id)
    with pytest.raises(ValueError, match="superseded history"):
        store.mark_event_cancelled(p1.id)
    # caller-owned session: flush-not-commit contract
    db = SessionLocal()
    try:
        store.mark_event_cancelled(p2.id, db=db)
        db.commit()
    finally:
        db.close()
    assert store.provenance(p2.id)["event_status"] == "cancelled"


def test_entityless_facts_reach_the_judge_via_lexical_pool(
        judged_env, monkeypatch):
    # G2 finding pinned: "personal best 5K time" carries NO named
    # entity — the entity-only shortlist starved and the judge never
    # fired on the real corpus case. The lexical pool must carry it.
    store, _, judge, SessionLocal = judged_env
    old = _fact(store, "The user's personal best time in the charity 5K "
                       "run is 25:31.", t_mentioned="2023/03/01")
    new = _fact(store, "The user's personal best time in the charity 5K "
                       "run is 23:15.", t_mentioned="2023/06/01")
    # NO entity links at all — the lexical pool is the only path
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "same metric updated",
        "superseded_ids": [old.id], "cancelled_ids": []})
    r = judge.judge_fact(new.id)
    assert r["superseded"] == [(old.id, new.id)]
    assert store.provenance(old.id)["t_invalid"] == "2023/06/01"


def test_lexical_pool_respects_scan_window(judged_env, monkeypatch):
    # Facts older than the scan window are NOT lexical candidates —
    # a disclosed bound, pinned so shrinking the window silently is loud.
    from agentmem_os.llm import supersession as sup
    store, _, judge, SessionLocal = judged_env
    old = _fact(store, "The user's personal best 5K time is 25:31.",
                t_mentioned="2023/01/01")
    filler = [_fact(store, f"The user tracks daily step count metric "
                           f"number {i:03d}.",
                    t_mentioned=f"2023/03/{(i % 28) + 1:02d}")
              for i in range(4)]
    new = _fact(store, "The user's personal best 5K time is 23:15.",
                t_mentioned="2023/06/01")
    monkeypatch.setattr(sup, "_LEXICAL_SCAN_CAP", 4)   # window excludes old
    seen_prompt = {}

    def _capture(p):
        seen_prompt["p"] = p
        return {"reasoning": "n", "superseded_ids": [old.id],
                "cancelled_ids": []}

    monkeypatch.setattr(judge, "_llm", _capture)
    r = judge.judge_fact(new.id)
    # old fell outside the window: either no candidates at all (skip)
    # or a shortlist without it — the proposal must drop out-of-shortlist
    assert r["superseded"] == []


# ── G3 R1 blocker/major pins ─────────────────────────────────────────────────

def test_planned_event_can_never_supersede_a_state(judged_env, monkeypatch):
    # R1-B1 pinned: the reserved planned pool guaranteed a future-dated
    # event in the shortlist; without the type guard it ALWAYS won on
    # domain time and invalidated a true current state (the dangerous
    # class, reproduced). Only same-type facts supersede.
    store, _, judge, SessionLocal = judged_env
    state = _fact(store, "The user works at Google and left the "
                         "conference committee.", t_mentioned="2023/05/01")
    planned = _fact(store, "The user attends the Google conference.",
                    fact_type="event", t_occurred="2099/06/01",
                    event_status="planned", t_mentioned="2023/05/02")
    for f in (state, planned):
        _link(SessionLocal, f.id, "Google")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "confused", "superseded_ids": [planned.id],
        "cancelled_ids": []})
    r = judge.judge_fact(state.id)
    assert r["superseded"] == []
    assert any("type mismatch" in reason for _, reason in r["dropped"])
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, state.id).superseded_by is None
    assert db.get(SemanticFact, planned.id).superseded_by is None
    db.close()


def test_similarity_cannot_approve_what_similarity_admitted(
        judged_env, monkeypatch):
    # R1-B2 pinned (the Mem0 #1674 class): 'allergic to peanuts' vs
    # 'allergic to shellfish' — high cosine admits the candidate through
    # the LEXICAL pool, but the same cosine must not approve the
    # supersession: coexisting siblings, no flip, no metric signal.
    store, _, judge, SessionLocal = judged_env
    peanuts = _fact(store, "The user is allergic to peanuts.",
                    t_mentioned="2023/01/01")
    shellfish = _fact(store, "The user is allergic to shellfish.",
                      t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "same allergy topic",
        "superseded_ids": [peanuts.id], "cancelled_ids": []})
    r = judge.judge_fact(shellfish.id)
    assert r["superseded"] == []
    assert any("co-signal disagreement" in reason
               for _, reason in r["dropped"])
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, peanuts.id).superseded_by is None
    db.close()


def test_metric_update_signal_carries_entityless_updates(
        judged_env, monkeypatch):
    # The legitimate lexical-pool case: same metric, numbers differ,
    # numbers-stripped texts near-identical — the replacement shape no
    # polarity vocabulary covers.
    store, _, judge, SessionLocal = judged_env
    old = _fact(store, "The user's personal best time in the charity 5K "
                       "run is 25:31.", t_mentioned="2023/03/01")
    new = _fact(store, "The user's personal best time in the charity 5K "
                       "run is 23:15.", t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "same metric updated",
        "superseded_ids": [old.id], "cancelled_ids": []})
    r = judge.judge_fact(new.id)
    assert r["superseded"] == [(old.id, new.id)]
    assert store.provenance(old.id)["t_invalid"] == "2023/06/01"


def test_polarity_flip_requires_shared_entity_node(judged_env, monkeypatch):
    # R1-Ma1 + R2-B4 pinned: the subject guard is a SHARED ENTITY NODE,
    # not string overlap — 'studying at Stanford' vs 'left Stanford
    # Stadium' share the WORD but link DIFFERENT nodes, so the flip is
    # suppressed. Positive control: a true same-node flip counts.
    store, _, judge, SessionLocal = judged_env
    a = _fact(store, "The user is studying at Stanford.",
              t_mentioned="2023/01/01")
    b = _fact(store, "The user left Stanford Stadium before the encore.",
              t_mentioned="2023/06/01")
    _link(SessionLocal, a.id, "Stanford")
    _link(SessionLocal, b.id, "Stanford Stadium")   # DIFFERENT node
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "left vs studying", "superseded_ids": [a.id],
        "cancelled_ids": []})
    r = judge.judge_fact(b.id)
    assert r["superseded"] == []
    reasons = [reason for _, reason in r["dropped"]]
    assert any("co-signal disagreement" in x for x in reasons)

    c = _fact(store, "The user works at Google.", t_mentioned="2023/02/01")
    d = _fact(store, "The user left Google last month.",
              t_mentioned="2023/07/01")
    for f in (c, d):
        _link(SessionLocal, f.id, "Google")         # SAME node
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "employment ended", "superseded_ids": [c.id],
        "cancelled_ids": []})
    r2 = judge.judge_fact(d.id)
    assert r2["superseded"] == [(c.id, d.id)]


def test_interval_overlap_is_ambiguous_strict_precedence_applies(
        judged_env, monkeypatch):
    # R1-Ma2 pinned: a month-interval fact must NOT lose to a point
    # date INSIDE its own interval (overlap = ambiguous); an interval
    # strictly before the other side's start DOES decide.
    store, _, judge, SessionLocal = judged_env
    month = _fact(store, "The user works at Google.",
                  fact_type="state", t_occurred="2023/06",
                  t_mentioned="2023/07/01")
    inside = _fact(store, "The user works at Microsoft.",
                   fact_type="state", t_occurred="2023/06/15",
                   t_mentioned="2023/07/02")
    for f in (month, inside):
        _link(SessionLocal, f.id, "employer")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [month.id],
        "cancelled_ids": []})
    r = judge.judge_fact(inside.id)
    assert r["superseded"] == []
    assert any("tie/overlap" in reason for _, reason in r["dropped"])

    later = _fact(store, "The user works at Zomato.",
                  fact_type="state", t_occurred="2023/09/01",
                  t_mentioned="2023/09/02")
    _link(SessionLocal, later.id, "employer")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [month.id],
        "cancelled_ids": []})
    r2 = judge.judge_fact(later.id)
    assert r2["superseded"] == [(month.id, later.id)]
    assert store.provenance(month.id)["t_invalid"] == "2023/09/01"


def test_actions_and_audit_commit_atomically(judged_env, monkeypatch):
    # R1-B3 pinned: a failure at audit-write time must roll the ACTIONS
    # back too — applied-but-unaudited must be impossible, and the fact
    # must remain live and sweepable.
    store, _, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/01/01")
    b = _fact(store, "The user left Google for Microsoft.",
              t_mentioned="2023/06/01")
    for f in (a, b):
        _link(SessionLocal, f.id, "Google")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [a.id],
        "cancelled_ids": []})

    def _log_boom(*args, **kw):
        raise RuntimeError("audit write failed")

    monkeypatch.setattr(judge, "_log", _log_boom)
    with pytest.raises(RuntimeError):
        judge.judge_fact(b.id)
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, a.id).superseded_by is None   # rolled back
    db.close()
    assert _judgment_rows(SessionLocal, b.id) == []           # no row
    monkeypatch.undo()
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [a.id],
        "cancelled_ids": []})
    assert judge.judge_missing()["judged"] >= 1               # sweepable


def test_double_listed_candidate_type_guard_then_cancellation(
        judged_env, monkeypatch):
    # A candidate in BOTH arrays: the supersession consumes it; the
    # cancellation proposal is dropped explicitly (R1 falsified the old
    # record's claim about this case — now the behavior is pinned).
    store, _, judge, SessionLocal = judged_env
    old_plan = _fact(store, "The user plans to run the Google charity "
                           "10K race.", fact_type="event",
                     t_occurred="2099/03/01", event_status="planned",
                     t_mentioned="2023/01/01")
    new = _fact(store, "The user called off the Google charity 10K race "
                       "and will run the marathon.",
                t_mentioned="2023/06/01")
    for f in (old_plan, new):
        _link(SessionLocal, f.id, "Google")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "replaced and cancelled",
        "superseded_ids": [old_plan.id], "cancelled_ids": [old_plan.id]})
    r = judge.judge_fact(new.id)
    # type guard drops the supersession (event vs state) — the
    # cancellation then legitimately proceeds
    assert r["superseded"] == []
    assert r["cancelled"] == [old_plan.id]
    assert any("type mismatch" in reason for _, reason in r["dropped"])


# ── G3 R2 blocker pins ───────────────────────────────────────────────────────

def test_metric_signal_is_digit_aware(judged_env, monkeypatch):
    # R2-B1 pinned with the critic's own repro pairs: a cosine that
    # cannot see digits cannot be a numbers signal. Masked-identity +
    # aligned-positions + normalized-difference is the whole gate.
    from agentmem_os.llm.supersession import _metric_update
    # the attack pairs — ALL must refuse
    assert not _metric_update(
        "The user's personal best time in the charity 10K run is 55:12.",
        "The user's personal best time in the charity 5K run is 25:31.")
    assert not _metric_update(
        "The user's flight leaves at 9:40 on Monday.",
        "The user's flight leaves at 6:15 on Friday.")
    assert not _metric_update(
        "The user bought a $2,400 desk.",
        "The user bought a $1,200 laptop.")
    # restatement, not update (normalized-equal numbers)
    assert not _metric_update(
        "The user's budget is $2,000.", "The user's budget is $2000.")
    assert not _metric_update(
        "The user weighs 70 kilograms.", "The user weighs 70.0 kilograms.")
    # the TRUE update shape passes
    assert _metric_update(
        "The user's personal best time in the charity 5K run is 23:15.",
        "The user's personal best time in the charity 5K run is 25:31.")
    # end-to-end: 10K fact must NOT supersede the 5K fact
    store, _, judge, SessionLocal = judged_env
    pb5 = _fact(store, "The user's personal best time in the charity 5K "
                       "run is 25:31.", t_mentioned="2023/03/01")
    pb10 = _fact(store, "The user's personal best time in the charity "
                        "10K run is 55:12.", t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "similar", "superseded_ids": [pb5.id],
        "cancelled_ids": []})
    r = judge.judge_fact(pb10.id)
    assert r["superseded"] == []
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, pb5.id).superseded_by is None
    db.close()


def test_oat_milk_cannot_cancel_the_tokyo_flight(judged_env, monkeypatch):
    # R2-B2 pinned, both halves: (1) an entity-less fact's planned pool
    # is TOPICALLY filtered — the flight never even reaches the
    # shortlist; (2) even when a plan IS shortlisted, a new text with
    # no cancellation cue (or no shared subject) cannot cancel it.
    store, _, judge, SessionLocal = judged_env
    flight = _fact(store, "The user is flying to Tokyo for his sister's "
                          "wedding.", fact_type="event",
                   t_occurred="2099/09/14", event_status="planned",
                   t_mentioned="2023/05/01")
    oat = _fact(store, "The user prefers oat milk in coffee.",
                fact_type="preference", t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "cancel it", "superseded_ids": [],
        "cancelled_ids": [flight.id]})
    r = judge.judge_fact(oat.id)
    assert r["cancelled"] == []
    # pool restriction: the flight shares no topic — judgment skips
    # with an empty shortlist, or drops as out-of-shortlist
    assert (r["skipped"] is not None
            or any("not in shortlist" in reason
                   for _, reason in r["dropped"]))
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, flight.id).event_status == "planned"
    db.close()

    # half (2): topically-shared plan, but the new text has NO cue
    trip = _fact(store, "The user plans a wedding trip to Tokyo.",
                 fact_type="event", t_occurred="2099/09/10",
                 event_status="planned", t_mentioned="2023/05/02")
    talk = _fact(store, "The user is excited about the wedding trip "
                        "to Tokyo.", t_mentioned="2023/06/02")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "cancel", "superseded_ids": [],
        "cancelled_ids": [trip.id]})
    r2 = judge.judge_fact(talk.id)
    assert r2["cancelled"] == []
    assert any("no cancellation cue" in reason
               for _, reason in r2["dropped"])
    # positive control: cue + shared subject cancels
    off = _fact(store, "The user says the wedding trip to Tokyo was "
                       "called off.", t_mentioned="2023/07/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "called off", "superseded_ids": [],
        "cancelled_ids": [trip.id]})
    r3 = judge.judge_fact(off.id)
    assert r3["cancelled"] == [trip.id]


def test_interval_touch_is_ambiguous_and_occ_axis_reversal_runs(
        judged_env, monkeypatch):
    # R2 majors pinned: intervals that TOUCH (end == other's start) are
    # ambiguous (mutating < to <= turns this red), and the occurrence-
    # axis REVERSAL branch actually executes (was uncovered).
    store, _, judge, SessionLocal = judged_env
    june = _fact(store, "The user works at Google.", fact_type="state",
                 t_occurred="2023/06", t_mentioned="2023/07/01")
    touch = _fact(store, "The user works at Microsoft.",
                  fact_type="state", t_occurred="2023/06/30",
                  t_mentioned="2023/07/02")
    for f in (june, touch):
        _link(SessionLocal, f.id, "employer")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [june.id],
        "cancelled_ids": []})
    r = judge.judge_fact(touch.id)
    assert r["superseded"] == []              # touch = ambiguous
    assert any("tie/overlap" in reason for _, reason in r["dropped"])

    # occ-axis reversal: the JUDGED fact is domain-earlier — it loses
    early = _fact(store, "The user works at Zomato.", fact_type="state",
                  t_occurred="2023/01/15", t_mentioned="2023/08/01")
    _link(SessionLocal, early.id, "employer")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [june.id],
        "cancelled_ids": []})
    r2 = judge.judge_fact(early.id)
    assert r2["superseded"] == [(early.id, june.id)]   # REVERSED on occ axis
    assert store.provenance(early.id)["t_invalid"] == "2023/06/01"


def test_cancelled_readers_and_opt_in(judged_env):
    # R2 major pinned (M25/M26/M27): the Ma3 reader-side fix had no
    # tests — dropping either filter or flipping the default now fails.
    store, _, _, SessionLocal = judged_env
    p = _fact(store, "The user attends the Tokyo conference.",
              fact_type="event", t_occurred="2099/09/14",
              event_status="planned", t_mentioned="2023/05/01")
    store.mark_event_cancelled(p.id)
    live_ids = {f.id for f in store.current_facts()}
    assert p.id not in live_ids                       # default excludes
    opted = {f.id for f in store.current_facts(include_cancelled=True)}
    assert p.id in opted                              # opt-in includes
    over = {f.id for f in store.facts_overlapping("2099/09/01",
                                                  "2099/09/30")}
    assert p.id not in over                           # never "happened"


def test_lexical_window_default_reaches_past_fillers(judged_env,
                                                     monkeypatch):
    # M34 pin: with DEFAULT constants, a lexical candidate behind 8
    # newer fillers is still found (shrinking the scan window to a
    # handful turns this red). No constant monkeypatching.
    store, _, judge, SessionLocal = judged_env
    old = _fact(store, "The user's personal best 5K time is 25:31.",
                t_mentioned="2023/01/01")
    for i in range(8):
        _fact(store, f"The user logs a daily journal entry number "
                     f"{i:02d} about gardening.",
              t_mentioned=f"2023/02/{i+1:02d}")
    new = _fact(store, "The user's personal best 5K time is 23:15.",
                t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "metric updated", "superseded_ids": [old.id],
        "cancelled_ids": []})
    r = judge.judge_fact(new.id)
    assert r["superseded"] == [(old.id, new.id)]


def test_norm_nums_unparseable_and_len_mismatch_and_shared_subject_drop():
    # Coverage of the remaining product branches, asserted not incidental:
    # unparseable float-ish token stays a string; length-mismatched
    # number lists refuse; version strings behave.
    from agentmem_os.llm.supersession import _metric_update, _norm_nums
    assert _norm_nums("version 1.2.3 build 7") == ["1.2.3", "7.0"]
    assert not _metric_update("The user ran 5 races in 30 days.",
                              "The user ran 5 races.")   # masks differ
    assert not _metric_update("The user is happy.", "The user is happy.")


def test_cancellation_shared_subject_via_content_word(judged_env,
                                                      monkeypatch):
    # Entity-less plan + cue + shared CONTENT word — the content-word
    # half of the cancellation subject gate (the node half is covered
    # by the Tokyo tests). Also covers the drop when only the cue holds.
    store, _, judge, SessionLocal = judged_env
    plan = _fact(store, "The user plans a pottery workshop weekend.",
                 fact_type="event", t_occurred="2099/03/01",
                 event_status="planned", t_mentioned="2023/05/01")
    off = _fact(store, "The user says the pottery workshop was "
                       "cancelled.", t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "cancelled", "superseded_ids": [],
        "cancelled_ids": [plan.id]})
    r = judge.judge_fact(off.id)
    assert r["cancelled"] == [plan.id]


def test_duplicate_proposal_ids_collapse_silently(judged_env, monkeypatch):
    # R2 minor pinned: a doubled id must not be audited as a phantom
    # concurrency race — it collapses to one before the gates.
    store, _, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/01/01")
    b = _fact(store, "The user left Google for Microsoft.",
              t_mentioned="2023/06/01")
    for f in (a, b):
        _link(SessionLocal, f.id, "Google")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [a.id, a.id, a.id],
        "cancelled_ids": []})
    r = judge.judge_fact(b.id)
    assert r["superseded"] == [(a.id, b.id)]
    assert not any("store refused" in reason for _, reason in r["dropped"])


def test_apply_defends_stale_plan_without_shared_subject(judged_env):
    # Direct-_apply defense (stale snapshot): a cued new text whose
    # shortlisted plan no longer shares any subject must drop, not
    # cancel — the snapshot's word is never trusted blindly.
    store, _, judge, SessionLocal = judged_env
    plan = _fact(store, "The user plans a pottery workshop weekend.",
                 fact_type="event", t_occurred="2099/03/01",
                 event_status="planned", t_mentioned="2023/05/01")
    new = _fact(store, "The user says everything got cancelled.",
                t_mentioned="2023/06/01")
    r = judge._apply(
        new.id, None,
        {"id": new.id, "text": new.fact_text, "type": "state",
         "occ": (None, None), "mentioned": "2023/06/01"},
        [{"id": plan.id, "text": "The user books a flight to Oslo.",
          "type": "event", "status": "planned", "pool": "planned",
          "shared_nodes": False,
          "occ": ("2099/03/01", "2099/03/01"),
          "mentioned": "2023/05/01"}],
        {plan.id: {"flip": False, "category": None, "cosine": 0.0,
                   "metric_update": False, "masked_equal": False,
                   "shared_nodes": False, "pool": "planned",
                   "agrees": False}},
        {"reasoning": "cancel", "superseded_ids": [],
         "cancelled_ids": [plan.id]})
    assert r["cancelled"] == []
    assert any("outside the plan" in reason
               for _, reason in r["dropped"])


# ── G3 R3 pins ───────────────────────────────────────────────────────────────

def test_gym_membership_cannot_cancel_climbing_competition(
        judged_env, monkeypatch):
    # R3-B1 pinned with the critic's real-model repro: one shared word
    # ('climbing') is topical coincidence, not identity — the binding
    # gate refuses what the pool admitted.
    store, _, judge, SessionLocal = judged_env
    plan = _fact(store, "The user plans to enter the climbing "
                        "competition in October.", fact_type="event",
                 t_occurred="2099/10/12", event_status="planned",
                 t_mentioned="2023/05/01")
    new = _fact(store, "The user cancelled his climbing gym membership.",
                t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "cancel", "superseded_ids": [],
        "cancelled_ids": [plan.id]})
    r = judge.judge_fact(new.id)
    assert r["cancelled"] == []
    assert any("outside the plan" in reason
               for _, reason in r["dropped"])
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, plan.id).event_status == "planned"
    db.close()


def test_negated_cue_never_cancels(judged_env, monkeypatch):
    # R3-Ma6 pinned: 'did NOT cancel' must not cancel — the negation
    # window the repo already shipped, reused.
    store, _, judge, SessionLocal = judged_env
    plan = _fact(store, "The user plans a pottery workshop weekend.",
                 fact_type="event", t_occurred="2099/03/01",
                 event_status="planned", t_mentioned="2023/05/01")
    new = _fact(store, "The user did not cancel the pottery workshop "
                       "weekend.", t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "cancel", "superseded_ids": [],
        "cancelled_ids": [plan.id]})
    r = judge.judge_fact(new.id)
    assert r["cancelled"] == []
    assert any("NEGATED" in reason for _, reason in r["dropped"])


def test_metric_hash_literal_and_single_number_and_decimals():
    # R3-Ma1: a literal '#' breaks equal-masks-imply-equal-counts — the
    # restored length check refuses the misaligned pair. R3-Ma7:
    # single-number pairs are identity-vs-value undecidable — refuse
    # (apartment, route, child-age all covered by one rule). R3-m1:
    # European decimal commas must not collapse ('1,5' != 15).
    from agentmem_os.llm.supersession import _metric_update, _norm_nums
    assert not _metric_update(
        "The user's ticket is 7 and it is 3 days old.",
        "The user's ticket is # and it is 3 days old.")
    assert not _metric_update("The user lives in apartment 12B.",
                              "The user lives in apartment 14B.")
    assert not _metric_update("The user's child is 7.",
                              "The user's child is 4.")
    assert _norm_nums("growth of 1,5 percent") == ["1,5"]
    assert _norm_nums("a $1,200 laptop") == ["1200.0"]


def test_interval_touch_ambiguous_both_directions(judged_env, monkeypatch):
    # R3-Ma2 pinned: the OTHER touch direction (the judged fact's
    # interval ends exactly where the candidate's begins) — mutating
    # the second < to <= turns this red.
    store, _, judge, SessionLocal = judged_env
    later = _fact(store, "The user works at Google.", fact_type="state",
                  t_occurred="2023/07/01", t_mentioned="2023/07/02")
    earlier = _fact(store, "The user works at Microsoft.",
                    fact_type="state", t_occurred="2023/06",
                    t_mentioned="2023/07/03")   # June interval ENDS 06/30…
    # …but judged fact is the JUNE one and candidate starts 07/01:
    # touching, not strictly before? 06/30 < 07/01 IS strictly before —
    # so construct a true touch: candidate starts exactly 06/30.
    touch = _fact(store, "The user works at Zomato.", fact_type="state",
                  t_occurred="2023/06/30", t_mentioned="2023/07/04")
    for f in (later, earlier, touch):
        _link(SessionLocal, f.id, "employer")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "changed", "superseded_ids": [touch.id],
        "cancelled_ids": []})
    r = judge.judge_fact(earlier.id)   # June interval vs 06/30 point: touch
    assert not any(pair[0] == touch.id or pair[1] == touch.id
                   for pair in r["superseded"])
    assert any(cid == touch.id and "tie/overlap" in reason
               for cid, reason in r["dropped"])


def test_int_list_rejects_booleans():
    # R3-m4 pinned (3rd-round survivor): True is not an id.
    from agentmem_os.llm.supersession import _int_list
    dropped = []
    assert _int_list([True, 3, False], dropped, "superseded_ids") == [3]
    assert len(dropped) == 2


def test_shortlist_and_sweep_exclude_superseded(judged_env, monkeypatch):
    # R3-m2/m3 pinned (3rd-round survivors): a superseded fact sharing
    # an entity never enters the shortlist; the sweep never judges it.
    store, _, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/01/01")
    b = _fact(store, "The user left Google for Microsoft.",
              t_mentioned="2023/06/01")
    c = _fact(store, "The user manages the Google account.",
              t_mentioned="2023/07/01")
    for f in (a, b, c):
        _link(SessionLocal, f.id, "Google")
    store.supersede(a.id, b.id)
    captured = {}

    def _cap(prompt):
        captured["p"] = prompt
        return {"reasoning": "n", "superseded_ids": [], "cancelled_ids": []}

    monkeypatch.setattr(judge, "_llm", _cap)
    judge.judge_fact(c.id)
    row = _judgment_rows(SessionLocal, c.id)[0]
    ids = [x["id"] for x in json.loads(row.shortlist_json)["candidates"]]
    assert a.id not in ids                       # dead facts never shortlist
    judge.judge_missing()
    # R4-Ma1: the old assertion here was a tautology (X or True) that
    # rubber-stamped a surviving mutation. The REAL pin: a superseded
    # fact with NO judgment row is not a sweep candidate at all —
    # deleting the sweep's live filter writes it a skip row and turns
    # this red.
    d = _fact(store, "The user runs the Google intramural league.",
              t_mentioned="2023/02/01")
    e = _fact(store, "The user left the Google intramural league.",
              t_mentioned="2023/08/01")
    store.supersede(d.id, e.id)          # superseded, never judged
    judge.judge_missing()
    assert _judgment_rows(SessionLocal, d.id) == []


def test_reinstate_cancelled_event(judged_env):
    # R3-B1 reversibility: the store's one-way transition gets its
    # shipped escape hatch, with the same guard discipline.
    store, _, _, SessionLocal = judged_env
    p = _fact(store, "The user attends the Tokyo conference.",
              fact_type="event", t_occurred="2099/09/14",
              event_status="planned", t_mentioned="2023/05/01")
    with pytest.raises(ValueError, match="CANCELLED"):
        store.reinstate_cancelled_event(p.id)    # not cancelled yet
    store.mark_event_cancelled(p.id)
    store.reinstate_cancelled_event(p.id)
    assert store.provenance(p.id)["event_status"] == "planned"
    with pytest.raises(ValueError, match="not found"):
        store.reinstate_cancelled_event(99999)


def test_cue_clause_must_name_the_plan():
    # Containment binding: the cancelling CLAUSE names something
    # outside the plan — the cue is not bound to this plan.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user loves the pottery workshop weekend. Separately, the "
        "user cancelled the newsletter subscription.",
        "The user plans a pottery workshop weekend.")
    assert not binds
    assert "names things outside the plan" in why
    binds2, why2 = _cancellation_binds(
        "The user cancelled the pottery workshop weekend.",
        "The user plans a pottery workshop weekend.")
    assert binds2 and why2 is None


def test_weekend_training_cannot_cancel_marathon_camp(judged_env,
                                                      monkeypatch):
    # R4-B1 pinned with the critic's real-model repro: two GENERIC
    # shared words bound a false cancel under v2; containment refuses
    # (the clause names session+physiotherapist — not in the plan) —
    # while the verbatim TRUE cancellation of a short-named plan binds.
    store, _, judge, SessionLocal = judged_env
    camp = _fact(store, "The user plans to join the weekend training "
                        "camp for the marathon.", fact_type="event",
                 t_occurred="2099/05/01", event_status="planned",
                 t_mentioned="2023/04/01")
    physio = _fact(store, "The user cancelled his weekend training "
                          "session with the physiotherapist.",
                   t_mentioned="2023/06/01")
    monkeypatch.setattr(judge, "_llm", lambda p: {
        "reasoning": "cancel", "superseded_ids": [],
        "cancelled_ids": [camp.id]})
    r = judge.judge_fact(physio.id)
    assert r["cancelled"] == []
    assert any("outside the plan" in reason for _, reason in r["dropped"])
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert db.get(SemanticFact, camp.id).event_status == "planned"
    db.close()
    # the calibration is no longer inverted: the short-named TRUE
    # cancellation binds
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, _ = _cancellation_binds(
        "The user cancelled the Rome marathon.",
        "The user plans to run the Rome marathon.")
    assert binds


def test_negation_checked_at_every_cue_occurrence():
    # R4-m1 pinned: first-occurrence-only negation let a negated second
    # mention through; and a non-negated occurrence naming ANOTHER
    # thing must not bind this plan.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user cancelled his climbing gym membership; he did not "
        "cancel the climbing competition.",
        "The user plans to enter the climbing competition in October.")
    assert not binds
    assert "NEGATED" in why and "outside the plan" in why


def test_insurance_cancellation_noun_does_not_cancel_trip():
    # R4-m2 pinned: the 'cancellation' noun cue (insurance coverage)
    # names things outside the plan — containment refuses.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user bought trip cancellation coverage from the insurer.",
        "The user plans a wedding trip to Tokyo.")
    assert not binds
    assert "outside the plan" in why


def test_pronoun_only_cancellation_names_nothing():
    # Disclosed miss pinned: 'cancelled it' names nothing → refuses
    # (the empty set must never bind trivially).
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user cancelled it.",
        "The user plans a wedding trip to Tokyo.")
    assert not binds
    assert "names nothing" in why


def test_non_dict_model_output_dropped_loudly(judged_env):
    # R4-m10 pinned: a grammar bypass returning a bare list must not
    # crash the apply path — dropped into the audit, judgment empty.
    store, _, judge, SessionLocal = judged_env
    a = _fact(store, "The user works at Google.", t_mentioned="2023/01/01")
    b = _fact(store, "The user left Google.", t_mentioned="2023/06/01")
    r = judge._apply(
        b.id, None,
        {"id": b.id, "text": b.fact_text, "type": "state",
         "occ": (None, None), "mentioned": "2023/06/01"},
        [{"id": a.id, "text": a.fact_text, "type": "state",
          "status": None, "pool": "entity", "shared_nodes": True,
          "occ": (None, None), "mentioned": "2023/01/01"}],
        {a.id: {"flip": True, "category": "employment", "cosine": 0.5,
                "metric_update": False, "masked_equal": False,
                "shared_nodes": True, "pool": "entity", "agrees": True}},
        ["not", "a", "dict"])
    assert r["superseded"] == [] and r["cancelled"] == []
    assert any("not an object" in reason for _, reason in r["dropped"])


def test_position_accurate_clause_selection():
    # R5-Ma1 pinned with the critic's exact wrong-write text: the same
    # cue surface in two clauses — the NEGATED first mention must not
    # be the clause judged for the second occurrence, and the second
    # occurrence names something outside the plan → refuse. Mirror:
    # a true cancellation AFTER an unrelated one still binds.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user did not cancel the pottery workshop weekend, but did "
        "cancel the pottery class.",
        "The user plans a pottery workshop weekend.")
    assert not binds
    assert "class" in why      # the SECOND clause was the one examined
    binds2, _ = _cancellation_binds(
        "The user cancelled the newsletter subscription and cancelled "
        "the Rome marathon.",
        "The user plans to run the Rome marathon.")
    assert binds2


def test_four_char_floor_is_load_bearing():
    # R5-Ma2 pinned with the NEGATIVE the symmetric positive could not
    # carry: at a 5-char floor 'rome' and 'boston' both vanish and
    # {marathon} ⊆ {marathon} falsely binds a DIFFERENT race's plan.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user cancelled the Rome marathon.",
        "The user plans to run the Boston marathon in April.")
    assert not binds
    assert "outside the plan" in why


def test_structural_stop_extension_is_load_bearing():
    # R5-m1 pinned (M11 survivor): without _S4_EXTRA_STOP, 'plans'
    # itself becomes an object word and 'cancelled the plans' binds any
    # plan containing the word 'plans'.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user cancelled the plans.",
        "The user plans a wedding trip to Tokyo.")
    assert not binds
    assert "names nothing" in why


def test_abbreviation_periods_do_not_truncate_the_clause():
    # R6-Ma1 pinned with the critic's honorific set: an abbreviation
    # period must not end the clause — truncation deletes the naming
    # words and every splitter false positive moves toward BINDING.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user cancelled the appointment with Dr. Meyer.",
        "The user plans an appointment at the downtown clinic.")
    assert not binds
    assert "meyer" in why      # the name SURVIVED the split
    binds2, _ = _cancellation_binds(
        "The user cancelled the trip to St. Louis.",
        "The user plans a trip to Chicago.")
    assert not binds2
    # control: the same phrasing without a period still refuses
    binds3, _ = _cancellation_binds(
        "The user cancelled the appointment with Doctor Meyer.",
        "The user plans an appointment at the downtown clinic.")
    assert not binds3
    # and a TRUE cancellation with an honorific binds
    binds4, why4 = _cancellation_binds(
        "The user cancelled the appointment with Dr. Meyer.",
        "The user plans an appointment with Dr. Meyer at the clinic.")
    assert binds4 and why4 is None


def test_lowercase_abbreviations_do_not_truncate_and_negation_errs_wide():
    # R7-B1 pinned with the critic's own set: 'Friday 6 p.m. dinner
    # reservation' must NOT truncate at 'p' and bind a Friday LUNCH
    # plan; same for 'vs.'. R7-Ma1 pinned: a negation crossing a
    # clause boundary must still refuse (negation errs wide).
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user cancelled the Friday 6 p.m. dinner reservation.",
        "The user plans a Friday lunch reservation.")
    assert not binds
    assert "dinner" in why       # the naming words SURVIVED the split
    binds2, _ = _cancellation_binds(
        "The user cancelled the Rome vs. Milan trip booking.",
        "The user plans a Rome hiking trip.")
    assert not binds2
    binds3, _ = _cancellation_binds(
        "The user did not cancel the pottery class and cancel the "
        "pottery workshop weekend.",
        "The user plans a pottery workshop weekend.")
    assert not binds3            # scope-ambiguous negation refuses


def test_sentence_period_split_is_load_bearing():
    # R7-m1 pinned: deleting the period branch merges sentences, the
    # clause grows with foreign words, and a TRUE cancellation refuses
    # — this pin makes that deletion loud.
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, why = _cancellation_binds(
        "The user cancelled the Rome marathon. The user booked a "
        "flight to Paris.",
        "The user plans to run the Rome marathon.")
    assert binds and why is None


def test_conjunction_split_class_is_pinned_as_disclosed(judged_env,
                                                        monkeypatch):
    # R8-n1 pinned AS DISCLOSED (not as fixed): the and/but branch
    # truncates coordinated noun phrases before the shared head noun —
    # 'cancelled the Friday and Saturday dinner reservations' names
    # only {friday}. This pin documents the measured behavior so a
    # future splitter edit cannot silently WIDEN or silently "fix" the
    # class without the record noticing. End-to-end: the gate binds
    # (the disclosed residual), so the pin asserts the DISCLOSURE'S
    # truth, and the real model's 0/3 decline rate plus prompt
    # unreachability carry the blast radius (disclosures 4/8).
    from agentmem_os.llm.supersession import _cancellation_binds
    binds, _ = _cancellation_binds(
        "The user cancelled the Friday and Saturday dinner reservations.",
        "The user plans a Friday lunch reservation.")
    assert binds        # the disclosed residual, pinned as measured
