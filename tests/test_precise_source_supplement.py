import hashlib

import pytest

from agentmem_os.benchmarks.precise_lexical_retrieval import MAX_TURNS, rank_turns
from agentmem_os.benchmarks.precise_source_supplement import supplement_packet
from benchmarks.english_screen.select_precision import select


def test_specific_bigram_beats_generic_recommendation_language():
    turns = [
        "I would recommend some new music for your next trip.",
        "My happy high school experience included debate team and AP economics.",
        "Could you recommend a language app for my next course?",
    ]
    assert rank_turns(turns, "Should I attend my high school reunion?")[0] == turns[1]


def test_weak_overlap_does_not_fill_available_space():
    turns = ["I am planning a completely unrelated project next week."]
    assert rank_turns(turns, "Which grocery store cost the most last month?") == []
    assert supplement_packet("baseline", [{"role": "user", "content": turns[0]}],
                             "Which grocery store cost the most last month?") == ("baseline", [])


def test_fixed_turn_limit_and_repeatability():
    turns = [f"Denver live music venue recommendation number {i}" for i in range(12)]
    first = rank_turns(turns, "Denver live music venue recommendations")
    assert len(first) == MAX_TURNS
    assert first == rank_turns(turns, "Denver live music venue recommendations")


def test_already_delivered_top_turns_do_not_trigger_noise_backfill():
    delivered = [f"Denver live music venue recommendation {i}" for i in range(MAX_TURNS)]
    weak_missing = "A general music recommendation for another city"
    packet = "\n".join(delivered)
    turns = [{"role": "user", "content": text} for text in delivered + [weak_missing]]
    result, receipts = supplement_packet(packet, turns, "Denver live music venues")
    assert result == packet
    assert receipts == []


def test_complete_source_receipts_and_prefix_preservation():
    packet = "[FACTS] older yoga schedule"
    source = "[2024/05/01] Yoga is now three times every week."
    result, receipts = supplement_packet(
        packet, [{"role": "user", "content": source}], "current yoga weekly frequency"
    )
    assert result.startswith(packet) and len(receipts) == 1
    receipt = receipts[0]
    assert result[receipt["packet_start"]:receipt["packet_end"]] == source
    assert receipt["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()


def test_duplicate_roles_caps_and_oversized_turns_fail_closed():
    ambiguous = [{"role": "user", "content": "Denver music venue"},
                 {"role": "assistant", "content": "Denver music venue"}]
    assert supplement_packet("base", ambiguous, "Denver music venue") == ("base", [])
    with pytest.raises(ValueError, match="exceeds"):
        supplement_packet("baseline", [], "Denver", char_cap=2)
    with pytest.raises(ValueError, match="non-negative"):
        supplement_packet("", [], "Denver", max_extra_chars=-1)
    huge = [{"role": "user", "content": "Denver music venue " * 100}]
    assert supplement_packet("base", huge, "Denver music venue", char_cap=100) == ("base", [])


def test_second_screen_replaces_all_prior_nonabstention_controls():
    cases = []
    jobs1, jobs2 = {}, {}
    kinds = ["knowledge-update", "multi-session", "single-session-assistant",
             "single-session-preference", "single-session-user", "temporal-reasoning"]
    for index in range(500):
        qid = f"q{index:03}"
        cases.append({"id": qid, "type": kinds[index % len(kinds)],
                      "abst": 92 <= index < 110,
                      "context_sha256": f"context-{index}",
                      "question_sha256": f"question-{index}"})
        if index < 72:
            jobs1[qid + "/judge"] = {"correct": False}
            jobs2[qid + "/judge"] = {"correct": False}
        elif index < 92:
            jobs1[qid + "/judge"] = {"correct": index % 2 == 0}
            jobs2[qid + "/judge"] = {"correct": index % 2 != 0}
        else:
            jobs1[qid + "/judge"] = {"correct": True}
            jobs2[qid + "/judge"] = {"correct": True}
    prior_nonabstention = {f"q{index:03}" for index in range(110, 150)}
    prior = {"cases": [
        {"question_id": qid, "cohort": "stable_pass_control",
         "abstention": next(case["abst"] for case in cases if case["id"] == qid)}
        for qid in ({f"q{index:03}" for index in range(92, 110)} | prior_nonabstention)
    ]}
    rows, counts = select({"cases": cases}, {"jobs": jobs1}, {"jobs": jobs2}, prior)
    controls = {row["question_id"] for row in rows
                if row["cohort"] == "stable_pass_control"}
    assert len(rows) == 150 and counts["fresh_nonabstention_controls"] == 40
    assert not controls & prior_nonabstention
    assert {f"q{index:03}" for index in range(92, 110)} <= controls
