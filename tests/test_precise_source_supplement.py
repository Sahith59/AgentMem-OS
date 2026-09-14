import hashlib

import pytest

from agentmem_os.benchmarks.precise_lexical_retrieval import MAX_TURNS, rank_turns
from agentmem_os.benchmarks.precise_source_supplement import supplement_packet


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
