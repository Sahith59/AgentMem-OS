import hashlib

import pytest

from agentmem_os.benchmarks.source_packet_supplement import supplement_packet


def test_adds_complete_source_with_valid_provenance_and_preserves_prefix():
    packet = "[FACTS] Older general schedule."
    turns = [{"role": "user", "content": "[2024/05/01] Yoga is three times weekly."}]
    result, receipts = supplement_packet(packet, turns, "yoga frequency")
    assert result.startswith(packet)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert result[receipt["packet_start"]:receipt["packet_end"]] == turns[0]["content"]
    assert receipt["source_sha256"] == hashlib.sha256(turns[0]["content"].encode()).hexdigest()
    assert receipt["role"] == "user"


def test_cap_includes_headers_and_never_clips_an_oversized_turn():
    packet = "baseline"
    turns = [{"role": "user", "content": "yoga " * 100},
             {"role": "assistant", "content": "yoga dates"}]
    result, receipts = supplement_packet(packet, turns, "yoga", char_cap=150)
    assert len(result) <= 150
    assert len(receipts) == 1
    assert turns[1]["content"] in result
    assert turns[0]["content"] not in result
    assert supplement_packet(packet, turns, "yoga", max_extra_chars=0) == (packet, [])


def test_already_present_and_ambiguous_speaker_turns_are_not_added():
    turns = [{"role": "user", "content": "yoga schedule"},
             {"role": "assistant", "content": "yoga schedule"},
             {"role": "user", "content": "yoga yesterday"}]
    assert supplement_packet("yoga yesterday", turns, "yoga") == ("yoga yesterday", [])


def test_refuses_to_truncate_input_and_does_not_fill_unrelated_text():
    with pytest.raises(ValueError, match="exceeds"):
        supplement_packet("old packet", [], "yoga", char_cap=3)
    with pytest.raises(ValueError, match="non-negative"):
        supplement_packet("", [], "yoga", max_extra_chars=-1)
    assert supplement_packet("old", [{"role": "user", "content": "other subject"}], "yoga") == ("old", [])
