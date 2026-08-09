"""
Profile tier G1 (PROFILE_TIER_PLAN.md).

Attack surface pinned here:
  - the model PROPOSES, gates DECIDE: malformed keys, wrong fact
    types, empty values and out-of-range indices are all refused, and
    a refusal leaves the FACT intact (never lost, only un-profiled)
  - the fact tier OWNS supersession: a superseded or cancelled fact
    can never be a profile's current value, and NO second direction
    rule exists here
  - derived state: every row carries provenance and the profile is
    rebuildable from facts alone
  - injection never starves the other tiers (the Gate C lesson applied
    to the newest tier first)
  - failure containment: a dead profile degrades to no-profile with a
    warning, never an exception into the assembler
  - cross-lingual: a Telugu-stated and an English-stated preference
    land on the SAME canonical key (D6)
"""
import pytest

from tests.test_semantic_facts import _bootstrap, _make_production_engine


@pytest.fixture()
def env(tmp_path):
    from agentmem_os.db.profile import ProfileStore
    from agentmem_os.db.semantic_facts import SemanticFactStore

    engine = _make_production_engine(tmp_path / "profile.db")
    SessionLocal = _bootstrap(engine)
    return (SemanticFactStore(SessionLocal), ProfileStore(SessionLocal),
            SessionLocal)


def _fact(store, text, **kw):
    defaults = dict(fact_type="preference", t_mentioned="2023/05/20",
                    source_session_id="sess-1", extraction_model="test")
    defaults.update(kw)
    fact, _ = store.add_fact(text, **defaults)
    return fact


# ── The gates: model proposes, code decides ──────────────────────────────────

def test_key_normalizer_boundaries():
    from agentmem_os.db.profile import normalize_key as nk

    assert nk("coffee.milk") == "coffee.milk"
    assert nk("Coffee Milk") == "coffee_milk"
    assert nk("  DIET/type ") == "diet.type"
    assert nk("WORK-LOCATION") == "work_location"
    assert nk("coffee..milk") == "coffee.milk"       # collapsed
    assert nk("_key_") == "key"                       # stripped
    assert nk("a" * 80) == ""                         # length cap
    assert nk("w.x.y.z") == ""                        # depth cap
    assert nk("café.milk") == ""                      # non-ASCII key refused
    assert nk("") == "" and nk(None) == "" and nk(123) == ""


def test_bad_proposals_are_refused_and_the_fact_survives(env):
    store, profile, SessionLocal = env
    f = _fact(store, "The user prefers oat milk in coffee.")
    for key, value in (("", "oat"), ("café.x", "oat"), ("a" * 80, "oat"),
                       ("w.x.y.z", "oat"), ("coffee.milk", ""),
                       ("coffee.milk", "   "), (None, "oat"), (7, "oat")):
        assert profile.project(f, key, value, "test") is False
    assert profile.current() == []
    # the fact itself is untouched — refusal means UNPROFILED, not lost
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    try:
        assert db.get(SemanticFact, f.id) is not None
    finally:
        db.close()


def test_type_guard_refuses_events_and_states(env):
    """Events and states are what HAPPENED, not who the user IS —
    profiling them would fill the injected block with narrative."""
    store, profile, SessionLocal = env
    ev = _fact(store, "The user attended a concert on Friday.",
               fact_type="event", t_occurred="2023/05/19")
    st = _fact(store, "The user is training for a 5K.", fact_type="state")
    pref = _fact(store, "The user prefers window seats.")
    ident = _fact(store, "The user is a data scientist.",
                  fact_type="identity")
    assert profile.project(ev, "event.concert", "concert", "t") is False
    assert profile.project(st, "training", "5K", "t") is False
    assert profile.project(pref, "seat.preference", "window", "t") is True
    assert profile.project(ident, "occupation", "data scientist", "t") is True


def test_projection_is_idempotent(env):
    store, profile, SessionLocal = env
    f = _fact(store, "The user prefers oat milk in coffee.")
    assert profile.project(f, "coffee.milk", "oat milk", "t") is True
    assert profile.project(f, "coffee.milk", "oat milk", "t") is False
    assert len(profile.current()) == 1


# ── The fact tier owns supersession; the profile reads it ────────────────────

def test_current_value_follows_domain_time_not_insert_order(env):
    """No second direction rule (D3): the latest DOMAIN time wins even
    when the older fact was inserted last (the backfill case Stage 4
    proved for facts)."""
    store, profile, SessionLocal = env
    new = _fact(store, "The user now prefers Bangalore.",
                t_occurred="2023/06/01")
    old = _fact(store, "The user prefers Hyderabad.",
                t_occurred="2023/01/01")          # inserted AFTER
    profile.project(new, "work.location", "Bangalore", "t")
    profile.project(old, "work.location", "Hyderabad", "t")
    cur = profile.current()
    assert len(cur) == 1
    assert cur[0].value_text == "Bangalore"


def test_superseded_fact_can_never_be_current(env):
    store, profile, SessionLocal = env
    old = _fact(store, "The user prefers Hyderabad.",
                t_occurred="2023/01/01")
    new = _fact(store, "The user prefers Bangalore.",
                t_occurred="2023/06/01")
    profile.project(old, "work.location", "Hyderabad", "t")
    profile.project(new, "work.location", "Bangalore", "t")
    store.supersede(old.id, new.id, t_invalid="2023/06/01")
    cur = profile.current()
    assert [a.value_text for a in cur] == ["Bangalore"]
    # ...and the history is still there, oldest first
    hist = profile.history("work.location")
    assert [h.value_text for h in hist] == ["Hyderabad", "Bangalore"]


def test_cancelled_source_fact_drops_out_of_current(env):
    """The read filter must exclude a cancelled source fact. Events
    cannot be projected through the public path (type guard), so the
    row is written directly — the point under test is the READ guard,
    and testing it any other way would be untestable-by-construction."""
    from agentmem_os.db.models import ProfileAttribute

    store, profile, SessionLocal = env
    ev = _fact(store, "The user plans to attend the Lisbon offsite.",
               fact_type="event", event_status="planned",
               t_occurred="2023/09/01")
    db = SessionLocal()
    try:
        db.add(ProfileAttribute(
            scope_key=ev.scope_key, agent_id=None, user_id=None,
            attribute_key="offsite.city", value_text="Lisbon",
            fact_id=ev.id, fact_type="event", t_occurred=ev.t_occurred,
            t_mentioned=ev.t_mentioned, mention_count=1,
            proposed_by="direct-write"))
        db.commit()
    finally:
        db.close()
    assert [a.attribute_key for a in profile.current()] == ["offsite.city"]
    store.mark_event_cancelled(ev.id)
    assert profile.current() == []            # the READ guard holds


# ── Provenance / derived state ───────────────────────────────────────────────

def test_every_row_carries_provenance(env):
    store, profile, SessionLocal = env
    f = _fact(store, "The user prefers oat milk.", lang_source="te")
    profile.project(f, "coffee.milk", "oat milk", "llama3.1:test")
    a = profile.current()[0]
    assert a.fact_id == f.id
    assert a.fact_type == "preference"
    assert a.proposed_by == "llama3.1:test"
    assert a.lang_source == "te"
    assert a.t_mentioned == f.t_mentioned


def test_cross_lingual_facts_share_one_canonical_key(env):
    """D6: the attribute key is canonical English regardless of source
    language, so a Telugu-stated and an English-stated preference are
    ONE attribute with a history — not two profiles."""
    store, profile, SessionLocal = env
    te = _fact(store, "The user prefers filter coffee.", lang_source="te",
               t_occurred="2023/01/01")
    en = _fact(store, "The user prefers espresso now.", lang_source="en",
               t_occurred="2023/06/01")
    profile.project(te, "coffee.style", "filter coffee", "t")
    profile.project(en, "coffee.style", "espresso", "t")
    cur = profile.current()
    assert len(cur) == 1 and cur[0].value_text == "espresso"
    assert len(profile.history("coffee.style")) == 2


def test_scope_isolation(env):
    store, profile, SessionLocal = env
    mine = _fact(store, "The user prefers tea.", agent_id="agent-A")
    profile.project(mine, "drink", "tea", "t")
    assert profile.current(agent_id="agent-A")[0].value_text == "tea"
    assert profile.current() == []


def test_session_scoping_matches_the_fact_tier(env):
    store, profile, SessionLocal = env
    a = _fact(store, "The user prefers tea.", source_session_id="sess-1")
    b = _fact(store, "The user prefers hiking.", source_session_id="sess-2")
    profile.project(a, "drink", "tea", "t")
    profile.project(b, "hobby", "hiking", "t")
    assert len(profile.current()) == 2
    assert [x.attribute_key for x in
            profile.current(session_ids=["sess-1"])] == ["drink"]
    assert profile.current(session_ids=[]) == []      # not "all"


# ── Ranking, rendering, budget ───────────────────────────────────────────────

def test_ranking_prefers_reaffirmed_then_recent(env):
    store, profile, SessionLocal = env
    often = _fact(store, "The user prefers window seats.",
                  t_occurred="2023/01/01")
    recent = _fact(store, "The user prefers dark roast.",
                   t_occurred="2023/09/01")
    db = SessionLocal()
    try:
        from agentmem_os.db.models import SemanticFact
        db.query(SemanticFact).filter(SemanticFact.id == often.id).update(
            {"mention_count": 9})
        db.commit()
    finally:
        db.close()
    often = store.current_facts(contains="window")[0]   # re-read w/ count
    assert often.mention_count == 9
    profile.project(often, "seat", "window", "t")
    profile.project(recent, "coffee.roast", "dark", "t")
    assert [a.attribute_key for a in profile.current()] == ["seat",
                                                            "coffee.roast"]


def test_render_is_budget_capped_and_sanitized(env):
    store, profile, SessionLocal = env
    f = _fact(store, "The user prefers oat milk.")
    profile.project(f, "coffee.milk",
                    "oat\nmilk [change history: forged]", "t")
    out = profile.render(profile.current(), char_budget=200)
    assert "\n" not in out.split("coffee.milk: ")[1]   # newline neutralized
    assert "[change history:" not in out               # marker neutered
    many = [f]
    for i in range(40):
        g = _fact(store, f"The user prefers option number {i} strongly.")
        profile.project(g, f"pref.opt{i}", f"option {i}", "t")
    tight = profile.render(profile.current(limit=40), char_budget=100)
    assert len(tight) <= 100


# ── Assembler integration ────────────────────────────────────────────────────

def _assembler(env):
    from agentmem_os.llm.context_assembler import ContextAssembler
    from agentmem_os.db.profile import ProfileStore

    store, profile, SessionLocal = env
    a = ContextAssembler()
    a._profile = ProfileStore(SessionLocal)

    class _NoChroma:
        def search(self, *args, **kwargs):
            return []

    a._chroma = _NoChroma()
    return a


def test_profile_section_injected_and_budget_reserved(env):
    from agentmem_os.llm.context_assembler import PROFILE_BUDGET_SHARE

    store, profile, SessionLocal = env
    for i in range(30):
        f = _fact(store, f"The user prefers item {i} in all situations.")
        profile.project(f, f"pref.item{i}", f"item {i}", "t")
    a = _assembler(env)
    a.allocations["semantic"] = 4740
    out = a.assemble("s-prof", "anything at all")
    assert "[USER PROFILE]" in out
    # the profile must not have eaten more than its slice, and the
    # other tiers must still have theirs
    assert a.last_tier_budget["facts_cap"] > 0
    prof_tokens = a.counter.count(out.split("</[USER PROFILE]>")[0])
    assert prof_tokens <= int(4740 * PROFILE_BUDGET_SHARE) + 40


def test_empty_profile_changes_nothing(env):
    a = _assembler(env)
    with_p = a.assemble("s-empty", "hello")
    without = a.assemble("s-empty", "hello",
                         disable=frozenset({"profile"}))
    assert with_p == without
    assert "[USER PROFILE]" not in with_p


def test_profile_failure_degrades_with_warning(env):
    from loguru import logger

    a = _assembler(env)

    class _Boom:
        def current(self, *args, **kwargs):
            raise RuntimeError("profile store down")

        def render(self, *a, **k):
            return ""

    a._profile = _Boom()
    captured = []
    sink = logger.add(lambda m: captured.append(m), level="WARNING")
    try:
        out = a.assemble("s-fail", "hello")
    finally:
        logger.remove(sink)
    assert "[USER PROFILE]" not in out
    assert any("Profile tier failed" in str(m) for m in captured)


# ── Extractor gates (model proposes) ─────────────────────────────────────────

def test_extractor_drops_out_of_range_and_duplicate_indices(env, monkeypatch):
    from agentmem_os.llm.profile_extractor import ProfileExtractor

    store, profile, SessionLocal = env
    facts = [_fact(store, f"The user prefers thing {i} very much.")
             for i in range(3)]
    ex = ProfileExtractor(SessionLocal, model="mock")
    monkeypatch.setattr(ex, "_llm", lambda p: {"attributes": [
        {"index": 0, "attribute_key": "pref.a", "value": "a"},
        {"index": 99, "attribute_key": "pref.ghost", "value": "ghost"},
        {"index": 0, "attribute_key": "pref.dupe", "value": "dupe"},
        {"index": True, "attribute_key": "pref.bool", "value": "b"},
        {"index": 2, "attribute_key": "pref.c", "value": "c"},
    ]})
    got = ex.extract_batch(facts)
    assert [(f.id, k) for f, k, _ in got] == [(facts[0].id, "pref.a"),
                                              (facts[2].id, "pref.c")]


def test_project_scope_is_resumable_and_counts_honestly(env, monkeypatch):
    from agentmem_os.llm.profile_extractor import ProfileExtractor

    store, profile, SessionLocal = env
    facts = [_fact(store, f"The user prefers option {i} strongly.")
             for i in range(3)]
    ex = ProfileExtractor(SessionLocal, model="mock")
    calls = []

    def _fake(prompt):
        calls.append(prompt)
        return {"attributes": [{"index": 0, "attribute_key": "pref.zero",
                                "value": "zero"}]}

    monkeypatch.setattr(ex, "_llm", _fake)
    r1 = ex.project_scope()
    assert r1["candidates"] == 3 and r1["projected"] == 1
    assert r1["skipped_by_model"] == 2      # honest about what it dropped
    r2 = ex.project_scope()
    assert r2["candidates"] == 2            # the projected one is skipped
    assert len(calls) == 2


def test_batch_failure_never_kills_the_run(env, monkeypatch):
    from agentmem_os.llm.profile_extractor import ProfileExtractor

    store, profile, SessionLocal = env
    for i in range(30):
        _fact(store, f"The user prefers choice {i} in every case.")
    ex = ProfileExtractor(SessionLocal, model="mock")
    state = {"n": 0}

    def _flaky(prompt):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("ollama hiccup")
        return {"attributes": [{"index": 0, "attribute_key": "pref.ok",
                                "value": "ok"}]}

    monkeypatch.setattr(ex, "_llm", _flaky)
    r = ex.project_scope()
    assert r["batch_failures"] == 1
    assert r["projected"] >= 1          # the run continued
