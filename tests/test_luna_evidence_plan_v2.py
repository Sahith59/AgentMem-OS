"""Contract and original-text invariants; no claim of learned selection ability."""

import json
from dataclasses import dataclass

import pytest

from benchmarks.evidence_focus import FocusInput, SourceTurn
from benchmarks.luna_evidence_plan_v2 import (
    apply_plan,
    input_bound,
    parse_plan,
    plan_request,
)


def runtime():
    turns = (
        SourceTurn("a", "user", "2026-04-01", "I completed two runs."),
        SourceTurn("b", "user", "2026-04-02", "I plan another run next month."),
    )
    return FocusInput(
        "How many runs have I completed?", "", "\n".join(t.text for t in turns), turns
    )


def response():
    return {
        "requirements": {
            "operation": "count",
            "target": "completed runs",
            "time_window": "",
            "output_unit": "runs",
            "needed_facts": ["completed events"],
        },
        "turn_ids": ["a"],
        "rejected_turn_ids": ["b"],
        "sufficiency": "uncertain",
    }


def test_strict_schema_explicitly_specifies_every_status_and_field():
    request = plan_request(runtime())
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    root = schema["schema"]
    assert set(root["required"]) == set(response())
    assert root["additionalProperties"] is False
    assert root["properties"]["sufficiency"]["enum"] == ["complete", "uncertain"]
    req = root["properties"]["requirements"]
    assert set(req["required"]) == set(response()["requirements"])
    assert req["additionalProperties"] is False
    assert input_bound(request) > sum(len(m["content"].encode()) for m in request["messages"])
    assert input_bound(request) >= len(json.dumps(request["response_format"]).encode())


def test_extended_source_metadata_cannot_leak_into_provider_request():
    @dataclass(frozen=True)
    class LabelledTurn(SourceTurn):
        expected_answer: str = "DO_NOT_SEND_LABEL"

    turn = LabelledTurn("a", "user", "", "I own two chairs.")
    value = FocusInput("How many chairs?", "", turn.text, (turn,))
    request = plan_request(value)
    payload = json.loads(request["messages"][1]["content"])
    assert set(payload["sources"][0]) == {"id", "role", "observed_at", "text"}
    assert "DO_NOT_SEND_LABEL" not in json.dumps(request)


@pytest.mark.parametrize(
    "change",
    [
        {"sufficiency": "sufficient"},
        {"sufficiency": []},
        {"rejected_turn_ids": ["a"]},
        {"rejected_turn_ids": ["alien"]},
        {"rejected_turn_ids": ["b", "b"]},
        {"rejected_turn_ids": ["b", "a"]},
        {"rejected_turn_ids": [True]},
        {"turn_ids": ["alien"]},
        {"turn_ids": [], "sufficiency": "complete"},
        {"answer": 2},
    ],
)
def test_malformed_or_inconsistent_membership_fails_closed(change):
    data = response()
    data.update(change)
    with pytest.raises(ValueError):
        parse_plan(runtime(), json.dumps(data))


def test_only_supporting_originals_receive_extra_emphasis():
    value = runtime()
    raw = json.dumps(response())
    candidate, report = apply_plan(value, raw)
    assert candidate.startswith(value.packet)  # original distractor is still preserved
    added = candidate[len(value.packet) :]
    assert value.turns[0].text in added
    assert value.turns[1].text not in added
    assert "completed events" not in added  # model requirement prose is audit-only
    assert report["rejected_ids"] == ["b"]
    assert report["semantic_completeness"] == "NOT_CERTIFIED"
    receipt = report["receipts"][0]
    assert candidate[receipt["start"] : receipt["end"]] == value.turns[0].text


def test_semantic_mistake_can_still_be_schema_valid():
    data = response()
    data.update(turn_ids=["b"], rejected_turn_ids=["a"], sufficiency="complete")
    _, report = apply_plan(runtime(), json.dumps(data))
    assert report["semantic_completeness"] == "NOT_CERTIFIED"
    assert report["selected_ids"] == ["b"]  # shape validation cannot prove relevance


def test_nonfit_keeps_baseline_byte_identical():
    value = runtime()
    candidate, report = apply_plan(value, json.dumps(response()), char_cap=len(value.packet))
    assert candidate == value.packet
    assert report["status"] == "UNCHANGED_BUDGET_NONFIT"
    assert report["receipts"] == []
