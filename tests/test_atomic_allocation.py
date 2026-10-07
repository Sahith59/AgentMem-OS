"""Behavioral contracts for exact allocation of whole original neighborhoods."""

import math
from datetime import datetime, timedelta

import pytest
from agentmem_os.llm.context_assembler import ContextAssembler
from agentmem_os.llm.evidence_packet import RetrievalHit, SourceSnapshot, SourceTurn, digest
from agentmem_os.llm.source_aware_selection import select

NOW = datetime(2026, 1, 3, 12)


def turn(id, body, *, session=None, pos=0, role="user", when=NOW):
    return SourceTurn(id, session or id, pos, role, when,
                      when.strftime("[%Y/%m/%d (%a) %H:%M] ") + body)


def run(turns, scored, baseline="baseline", **policy):
    by_id = {t.id: t for t in turns}
    hits = [RetrievalHit(id, digest(by_id[id].text), score, i)
            for i, (id, score) in enumerate(scored)]
    return select(SourceSnapshot("q", tuple(turns)), hits, baseline,
                  scope="q", as_of=NOW, **policy)


def test_joint_allocation_fits_complementary_sources_greedy_crowds_out():
    a = turn("a", "Large relevant history. " + "A" * 900)
    b = turn("b", "March 4: I bought a book, only after the refund. " + "B" * 150)
    c = turn("c", "March 9: I considered a book but did not buy it. " + "C" * 150)
    args = dict(extra_budget=1200, neighbor_turns=0)
    greedy, g = run([a, b, c], [("a", 1), ("b", .7), ("c", .6)], **args)
    packet, r = run([a, b, c], [("a", 1), ("b", .7), ("c", .6)],
                    allocation="joint", **args)
    assert [x["id"] for x in g["receipts"]] == ["a"]
    assert {x["id"] for x in r["receipts"]} == {"b", "c"}
    assert b.text in packet and c.text in packet and a.text not in packet
    assert r["allocation"]["utility"] == math.fsum([.7, .6])
    assert r["allocation"]["rendered_block_chars"] == len(packet) - r["block_offset"]
    assert len(packet) - len("baseline") <= 1200 and greedy.startswith("baseline")
    assert r["bundles"][0]["status"] == "allocation_not_selected"


def test_shared_context_is_charged_once_and_incidental_complete_anchor_scores_once():
    a = turn("a", "Was the visit Friday?", session="s", pos=0)
    b = turn("b", "Only if the weather permits; otherwise Monday.",
             session="s", pos=1, role="assistant")
    c = turn("c", "It rained, so I went Monday.", session="s", pos=2)
    packet, r = run([a, b, c], [("a", 1), ("b", .9), ("c", .8)], allocation="joint")
    assert len(r["receipts"]) == 3
    assert r["allocation"]["utility"] == math.fsum([1, .9, .8])
    assert r["allocation"]["covered_anchor_ids"] == ["a", "b", "c"]
    assert r["allocation"]["seed_anchor_ids"] == ["a", "b"]
    assert packet.count(b.text) == 1
    for receipt in r["receipts"]:
        offset = r["block_offset"]
        assert digest(packet[offset + receipt["start"]:offset + receipt["end"]]) == (
            receipt["sha256"]
        )


def test_oversized_qualifier_never_becomes_a_naked_anchor():
    a = turn("a", "I replaced it.", session="s", pos=0)
    b = turn("b", "By it you meant the oven, not the toaster. " + "detail " * 500,
             session="s", pos=1, role="assistant")
    text, r = run([a, b], [("a", 1)], allocation="joint", extra_budget=400)
    assert text == "baseline" and not r["receipts"]
    assert r["bundles"][0]["status"] == "bundle_budget_nonfit"
    assert r["allocation"]["utility"] == 0


def test_complete_baseline_qualifier_is_reused_with_role_date_and_no_new_clipping():
    a = turn("a", "I replaced it Monday.", session="s", pos=0)
    b = turn("b", "You were referring to the old oven, not the toaster.",
             session="s", pos=1, role="assistant")
    baseline = f"<[RECENT TURNS]>\nASSISTANT: {b.text}\n</[RECENT TURNS]>"
    packet, r = run([a, b], [("a", 1)], baseline, allocation="joint", extra_budget=300)
    assert packet.startswith(baseline) and a.text in packet
    assert [x["id"] for x in r["receipts"]] == ["a"]
    assert r["bundles"][0]["baseline_reused"] == ["b"]
    assert r["bundles"][0]["status"] == "admitted"


@pytest.mark.parametrize("future", [False, True])
def test_gap_or_future_qualification_rejects_entire_bundle(future):
    a = turn("a", "My answer is yes.", session="s", pos=0)
    b = turn("b", "That answer was conditional.", session="s", pos=1 if future else 2,
             when=NOW + timedelta(days=1) if future else NOW, role="assistant")
    text, r = run([a, b], [("a", 1)], allocation="joint")
    assert text == "baseline" and not r["receipts"]
    assert r["bundles"][0]["status"] == ("future_neighbor" if future else "position_gap")


def test_no_lower_rank_backfill_after_the_fixed_eight():
    turns = [turn(str(i), "large " * 500) for i in range(8)] + [turn("tail", "Small fact.")]
    scores = [(t.id, 10 - i) for i, t in enumerate(turns)]
    text, r = run(turns, scores, allocation="joint", extra_budget=400, neighbor_turns=0)
    assert text == "baseline" and len(r["candidate_hits"]) == 8
    assert "tail" not in [h["source_id"] for h in r["candidate_hits"]]
    assert r["allocation"]["subset_count"] == 256


def test_equal_utility_prefers_lower_cost_then_original_rank_deterministically():
    a, b = turn("a", "long " * 80), turn("b", "short")
    _, r = run([a, b], [("a", 1), ("b", 1)], allocation="joint", extra_budget=580,
               neighbor_turns=0)
    assert r["admitted_anchors"] == ["b"]
    c, d = turn("c", "same"), turn("d", "same")
    one, r1 = run([c, d], [("c", 1), ("d", 1)], allocation="joint", extra_budget=200,
                  neighbor_turns=0)
    two, r2 = run([d, c], [("c", 1), ("d", 1)], allocation="joint", extra_budget=200,
                  neighbor_turns=0)
    assert one == two and r1["admitted_anchors"] == r2["admitted_anchors"] == ["c"]


@pytest.mark.parametrize("policy", [{"allocation": "unknown"}, {"allocation": None},
                                   {"allocation": "joint", "max_anchors": 9}])
def test_invalid_or_unbounded_allocation_fails_closed(policy):
    with pytest.raises(ValueError, match="policy"):
        run([turn("a", "A fact.")], [("a", 1)], **policy)


def test_zero_scores_or_zero_budget_do_not_force_additions():
    a = turn("a", "No positive relevance.")
    for score, budget in [(0, 1000), (1, 0)]:
        text, r = run([a], [("a", score)], allocation="joint", extra_budget=budget)
        assert text == "baseline" and r["allocation"]["utility"] == 0


def test_overflowing_aggregate_scores_fail_without_fallback():
    a, b = turn("a", "first"), turn("b", "second")
    with pytest.raises(ValueError, match="Nonfinite allocation utility"):
        run([a, b], [("a", 1e308), ("b", 1e308)], allocation="joint")


def test_core_entrypoint_forwards_joint_allocation_without_loading_a_model():
    import numpy as np

    class Encoder:
        def encode(self, texts, **kwargs):
            return np.array([[1., 0.] for _ in texts])

    a = turn("a", "I only planned the trip; I did not travel. 日本語")
    text, r = ContextAssembler.assemble_source_aware_packet(
        SourceSnapshot("q", (a,)), "trip", "baseline", scope="q", as_of=NOW,
        encoder=Encoder(), allocation="joint",
    )
    assert a.text in text and r["allocation"]["subset_count"] == 2
    assert r["allocation"]["rendered_block_chars"] == len(text) - r["block_offset"]
