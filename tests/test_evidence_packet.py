from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from agentmem_os.llm.context_assembler import ContextAssembler
from agentmem_os.llm.evidence_packet import (
    RetrievalHit,
    SourceSnapshot,
    SourceTurn,
    digest,
    pack,
)

NOW = datetime(2026, 1, 3)


def turn(i, text, *, session="s", position=0, role="user", when=NOW):
    return SourceTurn(i, session, position, role, when, text)


def hit(t, score=1.0):
    return RetrievalHit(t.id, digest(t.text), score)


def compile_(turns, hits, **kwargs):
    return pack(
        SourceSnapshot("owner", tuple(turns)),
        hits,
        scope="owner",
        as_of=NOW,
        char_budget=kwargs.pop("char_budget", 3000),
        **kwargs,
    )


def test_matched_later_turn_survives_not_replaced_by_session_opening():
    intro = turn("intro", "A different topic.", position=0)
    match = turn("match", "The workshop lasted two days.", position=1)
    text, report = compile_([intro, match], [hit(match)], neighbor_turns=0)
    assert match.text in text and intro.text not in text
    assert report["anchors"] == ["match"]


def test_small_later_hit_survives_oversized_first_hit_and_never_clips():
    big = turn("big", "x" * 5000)
    small = turn("small", "I changed the date.", session="t")
    text, report = compile_([big, small], [hit(big, 2), hit(small)], char_budget=300)
    assert small.text in text and big.text not in text
    assert dict(id="big", reason="anchor_budget_nonfit") in report["omissions"]
    assert len(text) <= 300


def test_neighbors_preserve_negation_and_do_not_cross_sessions():
    denial = turn("denial", "That was only a plan, not a completed event.", position=1)
    match = turn("match", "The library trip.", position=0)
    other = turn("other", "Foreign neighbor.", session="other")
    text, report = compile_([match, other, denial], [hit(match)])
    assert denial.text in text and other.text not in text
    assert {r["id"] for r in report["receipts"]} == {"match", "denial"}


def test_same_text_on_distinct_dates_is_not_deduplicated():
    a = turn("a", "I attended another class.", when=NOW - timedelta(days=1))
    b = turn("b", a.text, session="t")
    _, report = compile_([a, b], [hit(a), hit(b), hit(a)])
    assert len(report["receipts"]) == 2


def test_future_sources_excluded_before_ranking_and_from_neighbors():
    now = turn("now", "Current event.")
    future = turn("future", "Future event.", position=1, when=NOW + timedelta(days=1))
    text, report = compile_([now, future], [hit(future, 10), hit(now)])
    assert future.text not in text
    assert {r["id"] for r in report["receipts"]} == {"now"}


def test_tampered_hit_duplicate_source_and_scope_fail_closed():
    t = turn("a", "Original.")
    with pytest.raises(ValueError, match="Unbound"):
        compile_([t], [replace(hit(t), source_sha256="forged")])
    with pytest.raises(ValueError, match="source"):
        compile_([t, t], [hit(t)])
    with pytest.raises(ValueError, match="scope"):
        pack(
            SourceSnapshot("another_owner", (t,)),
            [hit(t)],
            scope="owner",
            as_of=NOW,
            char_budget=200,
        )


def test_all_receipts_roundtrip_and_missing_context_not_certified():
    a = turn("a", "I agreed.")
    b = turn("b", "Only conditionally. " * 300, position=1)
    text, report = compile_([a, b], [hit(a)], char_budget=250)
    for r in report["receipts"]:
        assert digest(text[r["start"] : r["end"]]) == r["sha256"]
    assert any(o["reason"] == "context_budget_nonfit" for o in report["omissions"])
    assert report["semantic_completeness"] == "NOT_CERTIFIED"


def test_real_assembler_entrypoint_has_no_database_or_model_dependency():
    a = turn("a", "My first topic was travel.")
    b = turn("b", "The workshop lasted two days.", position=1)
    text, report = ContextAssembler.assemble_source_packet(
        SourceSnapshot("owner", (a, b)),
        "How many workshop days?",
        scope="owner",
        as_of=NOW,
        char_budget=1000,
        neighbor_turns=0,
    )
    assert b.text in text and report["anchors"] == ["b"]


def test_anchor_skip_delivered_as_neighbor_is_not_a_final_omission():
    a = turn("a", "The library trip.")
    b = turn("b", "It was canceled.", position=1)
    _, report = compile_([a, b], [hit(a, 2), hit(b)], max_anchors=1)
    assert dict(id="b", reason="anchor_limit") in report["selection_skips"]
    assert report["omissions"] == []
    assert {r["id"]: r["kind"] for r in report["receipts"]} == {"a": "anchor", "b": "context"}


def test_mutable_snapshot_and_invalid_identifiers_are_rejected():
    t = turn("a", "Original.")
    with pytest.raises(ValueError, match="Immutable"):
        pack(SourceSnapshot("owner", [t]), [], scope="owner", as_of=NOW, char_budget=100)
    for bad in (replace(t, id=[]), replace(t, session={})):
        with pytest.raises(ValueError, match="source"):
            compile_([bad], [])


def test_future_text_does_not_influence_ranker_scores():
    from agentmem_os.llm.source_turn_retrieval import rank

    a = turn("a", "I visited the library.")
    future = turn("b", "library library archive", position=1, when=NOW + timedelta(days=1))
    current = rank(SourceSnapshot("owner", (a,)), "library", scope="owner", as_of=NOW)
    extended = rank(SourceSnapshot("owner", (a, future)), "library", scope="owner", as_of=NOW)
    assert current == extended
