"""
Consolidation Engine v2 tests — Stage 2 G1 gate (expanded after G3 R2).

R2's mutation sweep proved four fixes had no tripwire — every test here is
written so reverting the fix it guards turns it red. The LLM is mocked at
the _llm boundary; extraction QUALITY is G2's job.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest


@pytest.fixture()
def env():
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from agentmem_os.db.models import Base, Session as SessionRow, Turn
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2

    engine = create_engine("sqlite://",
                           connect_args={"check_same_thread": False},
                           poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autocommit=False,
                                autoflush=False, expire_on_commit=False)
    seed = SessionLocal()
    seed.add(SessionRow(session_id="s1"))
    seed.add(Turn(id=0, session_id="s1", role="system",
                  content="Session dated 2023/05/20"))
    seed.add(Turn(id=1, session_id="s1", role="user",
                  content="I rode rollercoasters three times at SeaWorld "
                          "yesterday."))
    seed.add(Turn(id=2, session_id="s1", role="assistant",
                  content="Thrilling! Rollercoasters at SeaWorld are "
                          "world-class attractions."))
    seed.add(Turn(id=3, session_id="s1", role="user",
                  content="Also I went to the gym 5 times last week."))
    seed.add(Turn(id=4, session_id="s1", role="user",
                  content="And I collect vinyl records from local shops."))
    seed.commit()
    seed.close()
    return ConsolidationV2(SessionLocal, model="mock"), SessionLocal


GOOD = {"facts": [
    {"text": "The user rode rollercoasters three times at SeaWorld.",
     "fact_type": "event", "t_occurred": "2023/05/19"},
    {"text": "The user collects vinyl records from local shops.",
     "fact_type": "preference", "t_occurred": None},
]}


def _counts(SessionLocal):
    from agentmem_os.db.models import ConsolidationLog, SemanticFact
    db = SessionLocal()
    f, l = db.query(SemanticFact).count(), db.query(ConsolidationLog).count()
    db.close()
    return f, l


def test_happy_path_creates_facts_and_log(env, monkeypatch):
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    r = engine.consolidate_session("s1")
    assert r["created"] == 2 and r["rejected"] == 0
    assert r["truncated_chars"] == 0 and r["ctx_clamped"] is False
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    ev = db.query(SemanticFact).filter(
        SemanticFact.fact_type == "event").one()
    db.close()
    assert ev.t_mentioned == "2023/05/20"
    assert ev.t_occurred == "2023/05/19"
    assert 1 in ev.source_turn_ids


def test_citations_include_all_roles_ranked(env, monkeypatch):
    # Tripwire (R2-B3): reverting citations to user-turns-only fails this —
    # the assistant turn shares 'rollercoasters/seaworld' evidence.
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    engine.consolidate_session("s1")
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    ev = db.query(SemanticFact).filter(
        SemanticFact.fact_type == "event").one()
    db.close()
    assert 2 in ev.source_turn_ids       # assistant turn cited as evidence


def test_citations_ranked_by_strength_not_turn_id(env, monkeypatch):
    # R2-B2: [:8] must keep the STRONGEST evidence, not the lowest ids.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="rank"))
    for i in range(30, 40):    # ten weak low-id user turns share one token
        db.add(Turn(id=i, session_id="rank", role="user",
                    content=f"dolphin seaworld note number {i}."))
    db.add(Turn(id=99, session_id="rank", role="user",
                content="I interacted with a dolphin up close at SeaWorld "
                        "San Diego during their Summer Nights event."))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user interacted with dolphins up close at SeaWorld "
                 "San Diego during the Summer Nights event.",
         "fact_type": "event", "t_occurred": None}]})
    r = engine.consolidate_session("rank")
    assert r["created"] == 1
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    f = db.query(SemanticFact).filter(
        SemanticFact.source_session_id == "rank").one()
    db.close()
    assert 99 in f.source_turn_ids       # strongest evidence survives cap
    assert any("supporting turns" in w for _, w in r["warnings"])  # cap disclosed


def test_tool_numbers_cannot_ride_incidental_words(env, monkeypatch):
    # R2-B1 verbatim: '4213 milliseconds' from tool output must be
    # rejected even though a user turn contains the word 'build'.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="tool"))
    db.add(Turn(id=59, session_id="tool", role="system",
                content="Session dated 2023/05/20"))
    db.add(Turn(id=60, session_id="tool", role="user",
                content="Please run the build command for me."))
    db.add(Turn(id=61, session_id="tool", role="system",
                content="webpack compiled successfully in 4213 ms with 0 "
                        "errors"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's build succeeded in 4213 milliseconds.",
         "fact_type": "state", "t_occurred": None},
        {"text": "The user ran the build command.",
         "fact_type": "event", "t_occurred": None},
    ]})
    r = engine.consolidate_session("tool")
    texts = [t for t, _ in r["rejections"]]
    assert any("4213" in t for t in texts)          # tool numbers rejected
    assert any("not found in user-stated" in " ".join(p)
               for _, p in r["rejections"])


def test_assistant_only_knowledge_rejected(env, monkeypatch):
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's favorite park offers nighttime fireworks "
                 "spectaculars every evening.", "fact_type": "state",
         "t_occurred": None}]})
    r = engine.consolidate_session("s1")
    assert r["created"] == 0 and r["rejected"] == 1
    assert "USER turn" in r["rejections"][0][1][0]
    assert _counts(SessionLocal) == (0, 1)


def test_short_atomic_fact_adaptive_threshold(env, monkeypatch):
    # Tripwire (R2-B3): under a fixed need=2 this fact (single distinctive
    # stemmed token 'rode') loses user support and gets rejected.
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user rode.", "fact_type": "event",
         "t_occurred": None}]})
    r = engine.consolidate_session("s1")
    assert r["created"] == 1


def test_morphology_does_not_false_reject(env, monkeypatch):
    # R2-M6: 'microchipping' must meet 'microchipped'; 'cat' survives the
    # token floor.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="cat"))
    db.add(Turn(id=69, session_id="cat", role="system",
                content="Session dated 2023/05/20"))
    db.add(Turn(id=70, session_id="cat", role="user",
                content="I'm thinking of getting my cat microchipped soon."))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user is considering microchipping their cat.",
         "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("cat")
    assert r["created"] == 1 and r["rejected"] == 0


def test_planned_event_stored_with_status_marker(env, monkeypatch):
    # F7 RESOLVED (2026-08-06): future-dated events stay dated events
    # (retype corrupted dedup identity, R3-B3) and carry
    # event_status='planned'. Two different plan dates = two rows.
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user rode rollercoasters at the upcoming SeaWorld "
                 "festival.", "fact_type": "event",
         "t_occurred": "2024/04/15"},
        {"text": "The user rode rollercoasters at the upcoming SeaWorld "
                 "festival.", "fact_type": "event",
         "t_occurred": "2025/04/21"}]})
    r = engine.consolidate_session("s1")
    assert r["created"] == 2                     # dates NOT merged
    assert sum("event_status='planned'" in w for _, w in r["warnings"]) == 2
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    facts = db.query(SemanticFact).all()
    dates = {f.t_occurred for f in facts}
    statuses = {f.event_status for f in facts}
    db.close()
    assert dates == {"2024/04/15", "2025/04/21"}
    assert statuses == {"planned"}               # the marker, not just a warning


def test_user_stated_dates_and_word_numerals_pass_numbers_gate(env, monkeypatch):
    # R3-B1: "three times" in user speech satisfies "3 times" in the fact;
    # a date the user typed verbatim is user-stated; the fact's own
    # t_occurred digits are never treated as tool output.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="nums"))
    db.add(Turn(id=75, session_id="nums", role="user",
                content="I have tickets for the Taylor Swift show on "
                        "2024/03/15 and I saw her three times before."))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user has tickets for the Taylor Swift show on "
                 "2024/03/15.", "fact_type": "state", "t_occurred": None},
        {"text": "The user saw Taylor Swift 3 times before.",
         "fact_type": "event", "t_occurred": None},
        {"text": "The user saw the show on 2023/09/09.",
         "fact_type": "event", "t_occurred": "2023/09/09"}]})
    r = engine.consolidate_session("nums")
    assert r["rejected"] == 0 and r["created"] == 3


def test_rejection_reasons_persisted(env, monkeypatch):
    # Tripwire (R3-M1): rejected_count>0 AND the reasons survive in
    # rejections_json — reverting persistence turns this red.
    from agentmem_os.db.models import ConsolidationLog
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's favorite park offers nightly fireworks "
                 "spectaculars downtown.", "fact_type": "state",
         "t_occurred": None}]})
    engine.consolidate_session("s1")
    db = SessionLocal()
    row = db.query(ConsolidationLog).one()
    db.close()
    assert row.rejected_count == 1
    assert "USER turn" in (row.rejections_json or "")


def test_word_numerals_whole_word_only(env, monkeypatch):
    # Tripwire (R4-B1): "someone"/"content" must NOT manufacture numbers.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="ww"))
    db.add(Turn(id=110, session_id="ww", role="user",
                content="Please run the build and show me the bundle "
                        "content for someone on my phone."))
    db.add(Turn(id=111, session_id="ww", role="system",
                content="compiled in 10 s, 1 warning"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's build compiled in 10 seconds.",
         "fact_type": "state", "t_occurred": None},
        {"text": "The user saw 1 warning in the build.",
         "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("ww")
    assert r["created"] == 0 and r["rejected"] == 2


def test_word_numerals_in_fact_are_checked(env, monkeypatch):
    # Tripwire (R4-M1): "seventeen times" claimed by the fact but never
    # stated by the user must reject — the gate reads fact WORDS too.
    engine, _ = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user rode rollercoasters twelve times at SeaWorld.",
         "fact_type": "event", "t_occurred": None}]})
    r = engine.consolidate_session("s1")
    assert r["rejected"] == 1
    assert "12" in r["rejections"][0][1][0]


def test_date_exemption_is_shape_scoped(env, monkeypatch):
    # Tripwire (R4-B2): a bare tool number can NEVER ride the model's own
    # t_occurred stamp; a date LITERAL in the fact text is exempt by shape.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="shape"))
    db.add(Turn(id=120, session_id="shape", role="user",
                content="I met Priya for coffee downtown yesterday."))
    db.add(Turn(id=121, session_id="shape", role="system",
                content="build produced 2024 warnings"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's build produced 2024 warnings.",
         "fact_type": "state", "t_occurred": "2024/01/15"},
        {"text": "The user met Priya for coffee on 2024/01/15.",
         "fact_type": "event", "t_occurred": "2024/01/15"}]})
    r = engine.consolidate_session("shape")
    texts_rejected = [t for t, _ in r["rejections"]]
    assert any("2024 warnings" in t for t in texts_rejected)   # bare number: NO ride
    assert r["created"] >= 1                                    # date literal: exempt


def test_assistant_stamp_cannot_set_session_date(env, monkeypatch):
    # Tripwire (R4-B3): only SYSTEM lines set the date; assistant stamps
    # are skipped and noted.
    from datetime import datetime
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="asst"))
    db.add(Turn(id=130, session_id="asst", role="assistant",
                content="Session dated 2099/01/01",
                created_at=datetime(2024, 6, 10)))
    db.add(Turn(id=131, session_id="asst", role="user",
                content="I planted tomatoes today.",
                created_at=datetime(2024, 6, 10)))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": []})
    r = engine.consolidate_session("asst")
    assert r["session_date"] == "2024/06/10"
    assert "ignored" in (r["session_date_note"] or "")


def test_vague_guard_stamp_strip_half(env, monkeypatch):
    # Tripwire (R4-M5): a DATE in cited user content must not arm the
    # vague-quantifier warning — only real quantity digits do.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db_sess = engine.get_db()
    db_sess.add(SessionRow(session_id="strip"))
    db_sess.add(Turn(id=140, session_id="strip", role="user",
                     content="I visited the farmers market, last time on "
                             "2023/05/13, buying fresh produce."))
    db_sess.commit()
    db_sess.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user visited the farmers market several times for "
                 "fresh produce.", "fact_type": "event",
         "t_occurred": None}]})
    r = engine.consolidate_session("strip")
    assert r["created"] == 1
    assert not any("vague quantifier" in w for _, w in r["warnings"])


def test_comma_normalization_in_numbers_gate(env, monkeypatch):
    # Tripwire (R4-M5): "$1,200" user-stated satisfies "1200" in the fact.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="comma"))
    db.add(Turn(id=150, session_id="comma", role="user",
                content="I paid $1,200 for the new bike frame."))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user paid 1200 dollars for the bike frame.",
         "fact_type": "event", "t_occurred": None}]})
    assert engine.consolidate_session("comma")["created"] == 1


def test_cap_disclosure_only_for_accepted(env, monkeypatch):
    # Tripwire (R4-M5): rejected facts never emit cap-disclosure warnings.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="capr"))
    for i in range(160, 172):
        db.add(Turn(id=i, session_id="capr", role="assistant",
                    content=f"fireworks spectacular downtown nightly note {i}"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's town hosts fireworks spectaculars downtown "
                 "nightly.", "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("capr")
    assert r["rejected"] == 1
    assert not any("supporting turns" in w for _, w in r["warnings"])


def test_unicode_sessions_not_auto_rejected(env, monkeypatch):
    # R4-M2 pin: same-script Hindi facts over Hindi turns must be
    # accepted. (Canonical-English facts over Hindi turns remain an OPEN
    # cross-lingual gap — Gate E work, disclosed in the build log.)
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="hi"))
    db.add(Turn(id=180, session_id="hi", role="user",
                content="मैंने कल जयपुर में संगीत समारोह देखा और बहुत आनंद आया।"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "उपयोगकर्ता ने जयपुर में संगीत समारोह देखा।",
         "fact_type": "event", "t_occurred": None}]})
    assert engine.consolidate_session("hi", lang_source="hi")["created"] == 1


def test_provenance_user_turns_resolved(env, monkeypatch):
    # Tripwire (R4-M6): provenance separates USER grounding from ranked
    # all-role evidence.
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    engine.consolidate_session("s1")
    from agentmem_os.db.models import SemanticFact
    from agentmem_os.db.semantic_facts import SemanticFactStore
    store = SemanticFactStore(SessionLocal)
    db = SessionLocal()
    ev = db.query(SemanticFact).filter(SemanticFact.fact_type == "event").one()
    db.close()
    p = store.provenance(ev.id)
    assert 1 in p["user_turns_resolved"]
    assert set(p["user_turns_resolved"]) <= set(p["turns_resolved"])


def test_month_year_dates_never_orphan_digits(env, monkeypatch):
    # Tripwire (R5-B1): "February 2023"/"mid-January 2023" must exempt
    # cleanly — the old regex orphaned "23" and killed true count facts.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="my"))
    db.add(Turn(id=200, session_id="my", role="user",
                content="I completed 30 miles in 2 hours and 15 minutes "
                        "during the charity ride, and I swim every "
                        "Wednesday since mid-January 2023."))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user completed 30 miles in 2 hours and 15 minutes "
                 "during the charity cycling event in February 2023.",
         "fact_type": "event", "t_occurred": "2023/02"},
        {"text": "The user swims every Wednesday since mid-January 2023.",
         "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("my")
    assert r["rejected"] == 0 and r["created"] == 2
    audit = dict(r["numbers_audit"])
    key = next(k for k in audit if "30 miles" in k)
    assert set(audit[key]["claimed"]) == {"30", "2", "15"}
    assert "2023" in audit[key]["date_exempt"]


def test_glued_units_are_seen(env, monkeypatch):
    # Tripwire (R5-B2): "4213ms"/"16GB" must yield their numbers — the
    # old \b requirement let formatting decide acceptance.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="glue"))
    db.add(Turn(id=210, session_id="glue", role="user",
                content="Please run the project build suite for me now."))
    db.add(Turn(id=211, session_id="glue", role="system",
                content="suite 4213ms; bundle 16GB"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's suite ran in 4213ms.", "fact_type": "state",
         "t_occurred": None},
        {"text": "The bundle size is 16GB for the user's project.",
         "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("glue")
    assert r["created"] == 0 and r["rejected"] == 2


def test_inline_stamps_never_license_numbers(env, monkeypatch):
    # Tripwire (R5-B3): "[2023/05/20 (Sat) 14:05]" digits (20, 14, 5...)
    # must not license tool numbers; a user-stated number still does.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="stamp"))
    db.add(Turn(id=220, session_id="stamp", role="user",
                content="[2023/05/20 (Sat) 14:05] Please run the build; "
                        "I expect roughly 7 warnings."))
    db.add(Turn(id=221, session_id="stamp", role="system",
                content="build finished: 20 warnings, 14 chunks"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's build produced 20 warnings.",
         "fact_type": "state", "t_occurred": None},
        {"text": "The user's build made 14 chunks.",
         "fact_type": "state", "t_occurred": None},
        {"text": "The user expects roughly 7 warnings from the build.",
         "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("stamp")
    assert r["created"] == 1                     # only the user-stated 7
    assert r["rejected"] == 2


def test_tokenizer_keeps_devanagari_words_whole():
    # Tripwire (R6-M1): assert on _tokens() OUTPUT — reverting the
    # tokenizer to \\w{3,} shreds the sentence into fragments and this
    # goes red (the behavioral test alone passed under the bug for the
    # wrong reason).
    from agentmem_os.llm.consolidation_v2 import _tokens
    toks = _tokens("मैंने कल जयपुर में संगीत समारोह देखा और बहुत आनंद आया।")
    assert "जयपुर" in toks and "समारोह" in toks
    assert len(toks) >= 6
    assert not any(len(tok) < 3 for tok in toks)


def test_devanagari_gate_functions_both_directions(env, monkeypatch):
    # R5-M2: the tokenizer must keep Hindi words whole so the gate can
    # both ACCEPT user-grounded facts and REJECT assistant knowledge.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="hi2"))
    db.add(Turn(id=230, session_id="hi2", role="user",
                content="मैंने कल जयपुर में संगीत समारोह देखा और बहुत आनंद आया।"))
    db.add(Turn(id=231, session_id="hi2", role="assistant",
                content="जयपुर भारत के राजस्थान राज्य की राजधानी है और गुलाबी नगरी "
                        "कहलाती है।"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "उपयोगकर्ता ने जयपुर में संगीत समारोह देखा।",
         "fact_type": "event", "t_occurred": None},
        {"text": "उपयोगकर्ता का शहर राजस्थान राज्य की राजधानी गुलाबी नगरी कहलाता है।",
         "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("hi2", lang_source="hi")
    assert r["created"] == 1                     # user-grounded accepted
    assert r["rejected"] == 1                    # assistant knowledge OUT


def test_numeric_fact_with_only_assistant_support_true_reason(env, monkeypatch):
    # Tripwire (R5-M4): assistant-only numeric facts report the TRUE cause.
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="ar"))
    db.add(Turn(id=240, session_id="ar", role="user",
                content="Tell me something interesting about space travel."))
    db.add(Turn(id=241, session_id="ar", role="assistant",
                content="The ISS orbits at 28000 km/h, quite amazing!"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user's ISS orbits at 28000 km/h.",
         "fact_type": "state", "t_occurred": None}]})
    r = engine.consolidate_session("ar")
    assert r["rejected"] == 1
    assert "USER turn" in r["rejections"][0][1][0]
    assert "numbers" not in r["rejections"][0][1][0]


def test_stamp_beyond_window_ignored(env, monkeypatch):
    # Tripwire (R5-M3d): widening STAMP_SCAN_TURNS to the whole session
    # turns this red — a system stamp at turn index 4 must NOT win.
    from datetime import datetime
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, _ = env
    db = engine.get_db()
    db.add(SessionRow(session_id="late"))
    for i in range(4):
        db.add(Turn(id=250 + i, session_id="late", role="user",
                    content=f"General chat number {i} here.",
                    created_at=datetime(2024, 3, 3)))
    db.add(Turn(id=254, session_id="late", role="system",
                content="Session dated 2099/01/01"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": []})
    r = engine.consolidate_session("late")
    assert r["session_date"] == "2024/03/03"
    assert "ignored" in (r["session_date_note"] or "")


def test_user_turns_resolved_excludes_assistant(env, monkeypatch):
    # Tripwire (R5-M3b): the assertion must FAIL if user_turns_resolved
    # were all cited turns — assistant turn 2 is cited but must be absent.
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    engine.consolidate_session("s1")
    from agentmem_os.db.models import SemanticFact
    from agentmem_os.db.semantic_facts import SemanticFactStore
    store = SemanticFactStore(SessionLocal)
    db = SessionLocal()
    ev = db.query(SemanticFact).filter(SemanticFact.fact_type == "event").one()
    db.close()
    p = store.provenance(ev.id)
    assert 2 in p["turns_resolved"]              # assistant IS cited...
    assert 2 not in p["user_turns_resolved"]     # ...but is not user grounding
    assert p["user_turns_resolved"] == [1]


def test_migration_raises_on_readonly_db(tmp_path):
    # Tripwire (R5-M3a): a non-duplicate-column failure must RAISE, never
    # report "verified" (reverting to bare except:pass turns this red).
    import os as _os
    import sqlite3
    ro = tmp_path / "ro.db"
    conn = sqlite3.connect(ro)
    conn.execute("CREATE TABLE consolidation_log (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE kg_edges (id INTEGER PRIMARY KEY, "
                 "source_id INTEGER, relation_type TEXT, valid_until DATETIME)")
    conn.commit()
    conn.close()
    _os.chmod(ro, 0o444)
    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(ro)
        with pytest.raises(RuntimeError):
            eng._migrate_semantic_tier()
    finally:
        eng.DB_PATH = original
        _os.chmod(ro, 0o644)


def test_migration_reports_absent_table_honestly(tmp_path):
    # Tripwire (R4-M4): a DB with NO consolidation_log must not say
    # "verified".
    import sqlite3
    bare = tmp_path / "notable.db"
    conn = sqlite3.connect(bare)
    conn.execute("CREATE TABLE kg_edges (id INTEGER PRIMARY KEY, "
                 "source_id INTEGER, relation_type TEXT, valid_until DATETIME)")
    conn.commit()
    conn.close()
    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(bare)
        report = eng._migrate_semantic_tier()
        assert "absent" in report["consolidation_log_columns"]
    finally:
        eng.DB_PATH = original


def test_migration_adds_consolidation_log_columns(tmp_path):
    import sqlite3
    legacy = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE consolidation_log (id INTEGER PRIMARY KEY, "
                 "session_id TEXT)")
    conn.execute("CREATE TABLE kg_edges (id INTEGER PRIMARY KEY, "
                 "source_id INTEGER, relation_type TEXT, valid_until DATETIME)")
    conn.commit()
    conn.close()
    import agentmem_os.db.engine as eng
    original = eng.DB_PATH
    try:
        eng.DB_PATH = str(legacy)
        report = eng._migrate_semantic_tier()
        assert "added" in report["consolidation_log_columns"]
        conn = sqlite3.connect(legacy)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(consolidation_log)")}
        conn.close()
        assert {"truncated_chars", "rejected_count", "rejections_json"} <= cols
    finally:
        eng.DB_PATH = original


def test_unparseable_date_drops_date_keeps_fact(env, monkeypatch):
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user rode rollercoasters three times at SeaWorld.",
         "fact_type": "event", "t_occurred": "2023/10/28-2023/10/29"},
        {"text": "The user collects vinyl records from local shops.",
         "fact_type": "state", "t_occurred": 2024}]})   # non-string too
    r = engine.consolidate_session("s1")
    assert r["created"] == 2
    assert sum("dropped" in w for _, w in r["warnings"]) == 2


def test_vague_quantifier_warns_never_rejects(env, monkeypatch):
    # Covers the guard's firing branch (uncovered in R2). Fires: the fact
    # cites the gym turn, whose user content has a real digit.
    engine, _ = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user went to the gym several times last week.",
         "fact_type": "event", "t_occurred": None}]})
    r = engine.consolidate_session("s1")
    assert r["created"] == 1                     # warned, NOT rejected
    assert any("vague quantifier" in w for _, w in r["warnings"])


def test_vague_guard_is_per_fact_not_session_global(env, monkeypatch):
    # Tripwire (R2-B3): the vinyl fact cites only digit-free user content;
    # the gym turn's '5' exists elsewhere in the session. Under the old
    # session-global guard this warns — red under revert.
    engine, _ = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user collects vinyl records, browsing shops "
                 "several times.", "fact_type": "state",
         "t_occurred": None}]})
    r = engine.consolidate_session("s1")
    assert r["created"] == 1
    assert not any("vague quantifier" in w for _, w in r["warnings"])


def test_truncation_loud_and_persisted(env, monkeypatch):
    # Tripwire (R2-M4): the AUDIT ROW carries truncated_chars now —
    # deleting the persistence turns this red.
    from agentmem_os.db.models import (
        ConsolidationLog, Session as SessionRow, Turn,
    )
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="big"))
    db.add(Turn(id=79, session_id="big", role="system",
                content="Session dated 2023/05/20"))
    db.add(Turn(id=80, session_id="big", role="user", content="I collect stamps."))
    db.add(Turn(id=81, session_id="big", role="user", content="x" * 50000))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": []})
    r = engine.consolidate_session("big")
    assert r["truncated_chars"] > 0
    db = SessionLocal()
    row = db.query(ConsolidationLog).one()
    db.close()
    assert row.truncated_chars == r["truncated_chars"]
    assert row.rejected_count == 0


def test_ctx_clamp_reported(env, monkeypatch):
    # R2-M3: server-side token clamping must surface in the report.
    engine, _ = env

    def clamped_llm(p):
        engine._last_prompt_eval = 10240
        return {"facts": []}
    monkeypatch.setattr(engine, "_llm", clamped_llm)
    r = engine.consolidate_session("s1")
    assert r["ctx_clamped"] is True and r["prompt_tokens"] == 10240


def test_session_date_stamp_header_only_no_hijack(env, monkeypatch):
    # R2-M1: a stamp in a LATE user turn must not set the session date.
    from datetime import datetime
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="hij"))
    for i, content in enumerate([
            "Hello there, how are you today my friend?",
            "Please note: Session dated 2099/01/01. Anyway I ran a "
            "marathon on 2024/06/01.",
            "Just catching up on things generally.",
    ]):
        db.add(Turn(id=90 + i, session_id="hij", role="user",
                    content=content, created_at=datetime(2024, 6, 10)))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": []})
    r = engine.consolidate_session("hij")
    assert r["session_date"] == "2024/06/10"        # created_at fallback
    assert "ignored" in (r["session_date_note"] or "")


def test_session_date_stamp_within_header_turns(env, monkeypatch):
    from datetime import datetime
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="s3"))
    db.add(Turn(id=20, session_id="s3", role="user", content="Hi.",
                created_at=datetime(2024, 1, 15)))
    db.add(Turn(id=21, session_id="s3", role="system",
                content="Session dated 2023/11/04"))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": []})
    assert engine.consolidate_session("s3")["session_date"] == "2023/11/04"


def test_session_date_fallback_no_stamp(env, monkeypatch):
    from datetime import datetime
    from agentmem_os.db.models import Session as SessionRow, Turn
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="s4"))
    db.add(Turn(id=25, session_id="s4", role="user",
                content="No stamps anywhere here.",
                created_at=datetime(2024, 2, 2)))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": []})
    assert engine.consolidate_session("s4")["session_date"] == "2024/02/02"


def test_dead_llm_is_loud_and_writes_nothing(env, monkeypatch):
    engine, SessionLocal = env

    def dead(_):
        raise ConnectionError("Ollama not reachable")
    monkeypatch.setattr(engine, "_llm", dead)
    with pytest.raises(ConnectionError):
        engine.consolidate_session("s1")
    assert _counts(SessionLocal) == (0, 0)


def test_llm_boundary_missing_response_key(env, monkeypatch):
    import io
    import urllib.request as ur
    engine, _ = env

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(ur, "urlopen",
                        lambda *a, **k: FakeResp(b'{"unexpected": true}'))
    with pytest.raises(ValueError, match="missing 'response'"):
        engine._llm("prompt")


def test_llm_boundary_success_parse(env, monkeypatch):
    import io
    import json as _json
    import urllib.request as ur
    engine, _ = env
    payload = _json.dumps({"response": _json.dumps({"facts": []}),
                           "prompt_eval_count": 123}).encode()

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(ur, "urlopen", lambda *a, **k: FakeResp(payload))
    assert engine._llm("prompt") == {"facts": []}
    assert engine._last_prompt_eval == 123


def test_batch_abort_is_atomic(env, monkeypatch):
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    calls = {"n": 0}
    real_add = engine.store.add_fact

    def failing_add(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("disk full")
        return real_add(*a, **k)
    monkeypatch.setattr(engine.store, "add_fact", failing_add)
    with pytest.raises(RuntimeError):
        engine.consolidate_session("s1")
    assert _counts(SessionLocal) == (0, 0)


def test_idempotent_reconsolidation(env, monkeypatch):
    engine, _ = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    engine.consolidate_session("s1")
    r2 = engine.consolidate_session("s1")
    assert r2["created"] == 0 and r2["reaffirmed"] == 2


def test_empty_session_skipped(env):
    engine, _ = env
    assert "skipped" in engine.consolidate_session("no-such-session")


def test_lang_source_passthrough(env, monkeypatch):
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    engine.consolidate_session("s1", lang_source="hi")
    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    assert all(f.lang_source == "hi" for f in db.query(SemanticFact).all())
    db.close()


# ── Stage 3: event_status single source + entity linking wiring ──────────────

def test_event_status_helper_single_source():
    from agentmem_os.llm.consolidation_v2 import _event_status
    # Non-events never carry the axis (undated plans extract as states —
    # disclosed boundary).
    assert _event_status("state", "2099/01/01", "2023/05/20") is None
    assert _event_status("preference", None, "2023/05/20") is None
    # Events: undated and past/present → occurred; strictly future → planned.
    assert _event_status("event", None, "2023/05/20") == "occurred"
    assert _event_status("event", "2023/05/19", "2023/05/20") == "occurred"
    assert _event_status("event", "2023/05/20", "2023/05/20") == "occurred"
    assert _event_status("event", "2023/05/21", "2023/05/20") == "planned"
    # Month interval overlapping the session date is not clearly future.
    assert _event_status("event", "2023/05", "2023/05/20") == "occurred"
    assert _event_status("event", "2023/06", "2023/05/20") == "planned"


def test_facts_link_to_kg_end_to_end(env, monkeypatch):
    # The full Stage-3 wire: mock LLM, REAL spaCy NER on the fact text,
    # real linker, one batch. Facts get entities (display cache), the
    # join table gets rows, the log row records the count.
    from agentmem_os.db.models import (
        ConsolidationLog, KnowledgeGraphNode, SemanticFact,
        SemanticFactEntity,
    )
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user rode rollercoasters three times at SeaWorld.",
         "fact_type": "event", "t_occurred": "2023/05/19"}]})
    r = engine.consolidate_session("s1")
    assert r["created"] == 1
    assert r["link_failure"] is None
    assert r["entities_linked"] >= 1          # SeaWorld at minimum
    db = SessionLocal()
    fact = db.query(SemanticFact).one()
    links = db.query(SemanticFactEntity).all()
    node_texts = {n.entity_text for n in db.query(KnowledgeGraphNode).all()}
    log = db.query(ConsolidationLog).one()
    db.close()
    assert "SeaWorld" in (fact.entities or [])     # display cache filled
    assert fact.event_status == "occurred"
    assert len(links) == r["entities_linked"]
    assert "SeaWorld" in node_texts
    assert log.entities_linked == r["entities_linked"]


def test_link_failure_never_takes_facts_down(env, monkeypatch):
    # Compute-level linking failure: facts and the log row still commit,
    # the failure is reported AND PERSISTED (G3 R1 M2 — the count alone
    # cannot distinguish "suspended" from "nothing to link"), linking is
    # suspended for the batch (the sweep recovers later). This is the
    # failure POLICY of record — a regression here silently couples fact
    # durability to linker health.
    from agentmem_os.db.models import (
        ConsolidationLog, SemanticFact, SemanticFactEntity,
    )
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    calls = {"batch": 0, "recovery": 0}

    def _boom(*a, **k):
        # Batch calls carry db=; the auto-recovery sweep (R4-M5) runs
        # store-owned calls — count them separately so the suspension
        # tripwire stays sharp.
        calls["batch" if k.get("db") is not None else "recovery"] += 1
        raise RuntimeError("resolver exploded")

    monkeypatch.setattr(engine.linker, "link_fact", _boom)
    r = engine.consolidate_session("s1")
    assert r["created"] == 2                   # facts survived
    assert "resolver exploded" in r["link_failure"]
    assert r["entities_linked"] == 0
    # Suspension tripwire (G3 R1 minor: flipping `if link_failure is
    # None` to `if True` left the suite green — now it can't): after
    # the FIRST failure, the second fact must never attempt linking
    # IN THE BATCH. The auto-recovery sweep then legitimately retries
    # both facts (still broken here → recorded as failures, loudly).
    assert calls["batch"] == 1
    assert calls["recovery"] == 2
    assert len(r["link_recovery"]["failures"]) == 2
    db = SessionLocal()
    facts = db.query(SemanticFact).count()
    links = db.query(SemanticFactEntity).count()
    log = db.query(ConsolidationLog).one()
    db.close()
    assert facts == 2 and links == 0
    assert "resolver exploded" in log.link_failure   # persisted, not just reported
    # And the sweep is the recovery path for exactly this state.
    monkeypatch.undo()
    swept = engine.linker.link_missing()
    assert swept["swept"] == 2 and swept["failures"] == []


def test_link_failure_triggers_automatic_recovery(env, monkeypatch):
    # R4-M5: the recovery sweep must have a PRODUCT caller. A batch
    # whose linking fails commits its facts, then consolidate_session
    # auto-drains the default sweep — the suspended facts (zero links)
    # come back linked without any human running a REPL loop.
    from agentmem_os.db.models import SemanticFactEntity
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    real = engine.linker.link_fact
    state = {"fail": True}

    def _fail_in_batch_only(*a, **k):
        if state["fail"] and k.get("db") is not None:
            raise RuntimeError("resolver exploded")
        return real(*a, **k)

    monkeypatch.setattr(engine.linker, "link_fact", _fail_in_batch_only)
    r = engine.consolidate_session("s1")
    assert r["created"] == 2
    assert "resolver exploded" in r["link_failure"]
    assert r["link_recovery"] is not None
    assert r["link_recovery"]["complete"] is True
    assert r["link_recovery"]["links_created"] >= 1   # SeaWorld recovered
    db = SessionLocal()
    links = db.query(SemanticFactEntity).count()
    db.close()
    assert links == r["link_recovery"]["links_created"]


def test_recover_links_deep_drain_terminates(env, monkeypatch):
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    engine.consolidate_session("s1")
    rec = engine.recover_links(deep=True, limit=1)
    assert rec["complete"] is True
    assert rec["deep"] is True
    assert rec["swept"] >= 2          # every fact revisited, drain ended


def test_recover_links_runaway_guard_is_loud(env, monkeypatch):
    # R5 minor: the max_rounds guard and complete=False had no tripwire
    # (coverage showed the warning line never executed).
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)
    engine.consolidate_session("s1")
    from loguru import logger as _loguru
    msgs = []
    sink = _loguru.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        rec = engine.recover_links(deep=True, limit=1, max_rounds=1)
    finally:
        _loguru.remove(sink)
    assert rec["complete"] is False
    assert any("max_rounds" in m for m in msgs)


# ── Stage 4: supersession wiring ─────────────────────────────────────────────

def test_created_state_prefs_judged_post_commit(env, monkeypatch):
    # Only created state/preference facts queue for judgment; events do
    # not. The judge runs AFTER the batch commit (its own sessions), and
    # every judged fact gets a persisted judgment row.
    from agentmem_os.db.models import SupersessionJudgment
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)   # 1 event + 1 pref
    monkeypatch.setattr(engine.judge, "_llm", lambda p: {
        "reasoning": "none", "superseded_ids": [], "cancelled_ids": []})
    r = engine.consolidate_session("s1")
    assert r["created"] == 2
    assert r["judge_failure"] is None
    assert r["supersession"]["judged"] == 1        # the preference only
    db = SessionLocal()
    rows = db.query(SupersessionJudgment).all()
    db.close()
    assert len(rows) == 1
    assert rows[0].session_id == "s1"


def test_judge_failure_is_best_effort_and_persisted(env, monkeypatch):
    from agentmem_os.db.models import ConsolidationLog, SemanticFact
    engine, SessionLocal = env
    monkeypatch.setattr(engine, "_llm", lambda p: GOOD)

    def _boom(fid, **kw):
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(engine.judge, "judge_fact", _boom)
    r = engine.consolidate_session("s1")
    assert r["created"] == 2                       # facts survived
    assert "judge exploded" in r["judge_failure"]
    assert r["supersession"]["failures"]
    db = SessionLocal()
    facts = db.query(SemanticFact).count()
    log = db.query(ConsolidationLog).one()
    db.close()
    assert facts == 2
    assert "judge exploded" in log.judge_failure   # persisted audit
    # recovery path: the unjudged fact is sweepable once the judge heals
    monkeypatch.undo()
    monkeypatch.setattr(engine.judge, "_llm", lambda p: {
        "reasoning": "none", "superseded_ids": [], "cancelled_ids": []})
    rec = engine.recover_judgments()
    assert rec["complete"] is True
    assert rec["judged"] == 1


def test_real_update_supersedes_across_sessions(env, monkeypatch):
    # Two sessions, an employment change: session 2's fact must
    # supersede session 1's via polarity-flip co-signal + mocked judge
    # verdict, with t_invalid = the new fact's domain time.
    from agentmem_os.db.models import Session as SessionRow, Turn, SemanticFact
    engine, SessionLocal = env
    db = SessionLocal()
    db.add(SessionRow(session_id="job1"))
    db.add(Turn(id=200, session_id="job1", role="system",
                content="Session dated 2023/01/10"))
    db.add(Turn(id=201, session_id="job1", role="user",
                content="I work at Google as an engineer."))
    db.add(SessionRow(session_id="job2"))
    db.add(Turn(id=210, session_id="job2", role="system",
                content="Session dated 2023/09/05"))
    db.add(Turn(id=211, session_id="job2", role="user",
                content="I left Google and joined Microsoft last month."))
    db.commit()
    db.close()
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user works at Google as an engineer.",
         "fact_type": "state", "t_occurred": None}]})
    r1 = engine.consolidate_session("job1")
    assert r1["created"] == 1
    monkeypatch.setattr(engine, "_llm", lambda p: {"facts": [
        {"text": "The user left Google and joined Microsoft.",
         "fact_type": "state", "t_occurred": None}]})

    def _judge_llm(prompt):
        import re as _re
        ids = [int(m) for m in _re.findall(r"\[(\d+)\]", prompt)]
        return {"reasoning": "employment changed",
                "superseded_ids": ids, "cancelled_ids": []}

    monkeypatch.setattr(engine.judge, "_llm", _judge_llm)
    r2 = engine.consolidate_session("job2")
    assert r2["created"] == 1
    assert len(r2["supersession"]["superseded"]) == 1
    db = SessionLocal()
    old = db.query(SemanticFact).filter(
        SemanticFact.source_session_id == "job1").one()
    new = db.query(SemanticFact).filter(
        SemanticFact.source_session_id == "job2").one()
    db.close()
    assert old.superseded_by == new.id
    assert old.t_invalid == "2023/09/05"           # new fact's domain time
    assert new.superseded_by is None


def test_prompt_types_dated_plans_as_planned_events():
    """Pin for the founder's plans-as-events decision (2026-08-08 —
    the parked F7 activation, decided at the BUILD READY checkpoint):
    dated future plans must extract as EVENTS (the store then stamps
    event_status='planned' deterministically via _event_status, which
    the tests above already pin); UNDATED plans stay states (F7's
    disclosed boundary, unchanged). A revert of the prompt flip goes
    red here."""
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2

    p = ConsolidationV2(lambda: None)._prompt("2023/05/20", "t")
    assert "plan/appointment for a SPECIFIC FUTURE DATE" in p
    assert "marks future-dated events as planned automatically" in p
    assert 'A plan with NO stated date is a "state"' in p
    assert "the date it is planned FOR" in p
    assert "never an event" not in p  # the pre-flip wording is gone


def test_output_truncation_retries_then_fails_loudly(env, monkeypatch):
    """Gate C finding pin (2026-08-09): Ollama silently returns JSON cut
    mid-string when the generation hits num_predict; the resulting
    JSONDecodeError killed the WHOLE session (3 of 3,631 lost, one of
    them GOLD EVIDENCE for a benchmark question — a silently depressed
    score). done_reason='length' makes the cap observable: retry once at
    double, then fail naming the real cause."""
    import json as _json
    from agentmem_os.llm import consolidation_v2 as cv2

    cv2engine, _SessionLocal = env
    calls = []

    class _Resp:
        def __init__(self, payload):
            self._p = _json.dumps(payload).encode()

        def read(self):
            return self._p

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        body = _json.loads(req.data.decode())
        calls.append((body["options"]["num_predict"],
                      "repeat_penalty" in body["options"]))
        # Truncate forever: proves escalation AND the loud terminal fail.
        return _Resp({"response": '{"facts": [{"text": "cut',
                      "done_reason": "length", "prompt_eval_count": 10})

    monkeypatch.setattr(cv2.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(ValueError, match="num_predict ceiling"):
        cv2engine._llm("prompt")
    # ROOT CAUSE FIRST: truncation is degeneration, so the first retry
    # applies anti-repetition at the SAME ceiling; only then does the
    # ceiling escalate. The default call must carry NO penalty (global
    # application measurably cuts good sessions from 9 facts to 2).
    assert calls[0] == (cv2.NUM_PREDICT, False)
    assert calls[1] == (cv2.NUM_PREDICT, True)
    assert calls[2] == (cv2.NUM_PREDICT * 2, True)
    assert calls[-1][0] == cv2.NUM_PREDICT_MAX
    assert all(c[1] for c in calls[1:])


def test_untruncated_output_never_retries(env, monkeypatch):
    """The escalation must fire ONLY on truncation — a normal reply
    makes exactly one call."""
    import json as _json
    from agentmem_os.llm import consolidation_v2 as cv2

    cv2engine, _SessionLocal = env
    calls = []

    class _Resp:
        def __init__(self, payload):
            self._p = _json.dumps(payload).encode()

        def read(self):
            return self._p

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        opts = _json.loads(req.data.decode())["options"]
        calls.append(opts)
        return _Resp({"response": '{"facts": []}', "done_reason": "stop",
                      "prompt_eval_count": 10})

    monkeypatch.setattr(cv2.urllib.request, "urlopen", _fake_urlopen)
    assert cv2engine._llm("prompt") == {"facts": []}
    assert len(calls) == 1
    assert "repeat_penalty" not in calls[0]  # untouched happy path
