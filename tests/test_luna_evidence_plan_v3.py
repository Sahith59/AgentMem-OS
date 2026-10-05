"""Offline consumer invariants, NOT tests of Luna's semantic selection ability."""

import json
from dataclasses import dataclass

import pytest

from benchmarks.evidence_focus import FocusInput, SourceTurn
from benchmarks.luna_evidence_plan import SETTINGS
from benchmarks.luna_evidence_plan_v2 import input_bound
from benchmarks.luna_evidence_plan_v3 import apply_plan, parse_plan, plan_request


def value(*texts):
    turns = tuple(SourceTurn(str(i), "user", "", text) for i, text in enumerate(texts))
    return FocusInput("What happened?", "", "\n\n".join(texts), turns)


def plan(support=(), qualifications=(), rejected=(), sufficiency="uncertain"):
    return {
        "requirements": {
            "operation": "direct_recall",
            "target": "event",
            "time_window": "",
            "output_unit": "",
            "needed_facts": ["DO_NOT_RENDER_GENERATED_FACT"],
        },
        "support_turn_ids": list(support),
        "qualification_turn_ids": list(qualifications),
        "rejected_turn_ids": list(rejected),
        "sufficiency": sufficiency,
    }


def test_request_retains_fixed_settings_and_explicit_source_projection():
    @dataclass(frozen=True)
    class Labelled(SourceTurn):
        expected_answer: str = "SECRET_EVALUATOR_LABEL"

    turn = Labelled("a", "user", "", "I have not decided.")
    request = plan_request(FocusInput("Where?", "", turn.text, (turn,)))
    assert all(request[k] == v for k, v in SETTINGS.items())
    assert "SECRET_EVALUATOR_LABEL" not in json.dumps(request)
    schema = request["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert set(schema["schema"]["required"]) == set(plan())
    assert schema["schema"]["additionalProperties"] is False
    assert input_bound(request) > len(json.dumps(request["response_format"]))


@pytest.mark.parametrize(
    "text,status",
    [
        ("I have not decided where to go.", "uncertain"),
        ("I did not visit any museum this weekend.", "complete"),
        ("I visited the museum, but cannot recall which day.", "uncertain"),
    ],
)
def test_qualification_only_is_retained_without_forcing_negative_to_uncertain(text, status):
    runtime = value(text)
    candidate, report = apply_plan(
        runtime, json.dumps(plan(qualifications=["0"], sufficiency=status))
    )
    assert report["status"] == "APPLIED"
    assert report["qualification_ids"] == ["0"]
    assert report["planner_sufficiency"] == status
    assert candidate.startswith(runtime.packet)
    added = candidate[len(runtime.packet) :]
    assert text in added
    assert "DO_NOT_RENDER_GENERATED_FACT" not in added
    assert "qualification" not in added
    assert report["semantic_completeness"] == "NOT_CERTIFIED"


def test_support_and_qualifier_union_keeps_source_order_and_whole_turns():
    runtime = value(
        "Earlier I was undecided.", "Later I visited a museum.", "Not sure which museum."
    )
    candidate, report = apply_plan(runtime, json.dumps(plan(["1"], ["0", "2"])))
    assert report["selected_ids"] == ["0", "1", "2"]
    for turn, receipt in zip(runtime.turns, report["receipts"], strict=True):
        assert candidate[receipt["start"] : receipt["end"]] == turn.text


def test_budget_nonfit_never_appends_support_without_its_qualifier():
    runtime = value("I finished it.", "Correction: I only started it. " * 100)
    candidate, report = apply_plan(runtime, json.dumps(plan(["0"], ["1"])), max_focus_chars=150)
    assert candidate == runtime.packet
    assert report["status"] == "UNCHANGED_BUDGET_NONFIT"
    assert report["receipts"] == []


def test_empty_uncertain_keeps_baseline_and_rejection_does_not_delete_sources():
    runtime = value("You could visit a museum.")
    candidate, report = apply_plan(runtime, json.dumps(plan(rejected=["0"])))
    assert candidate == runtime.packet
    assert report["status"] == "UNCHANGED_EMPTY_SELECTION"


@pytest.mark.parametrize(
    "change",
    [
        {"support_turn_ids": [False]},
        {"rejected_turn_ids": ["0"]},
        {"qualification_turn_ids": ["0"]},  # overlaps support
        {"qualification_turn_ids": ["2", "1"]},
        {"qualification_turn_ids": ["1", "1"]},
        {"qualification_turn_ids": [True]},
        {"qualification_turn_ids": ["unknown"]},
        {"qualification_turn_ids": None},
        {"rejected_turn_ids": ["1"]},  # overlaps qualification
        {"sufficiency": []},
        {"requirements": None},
        {"final_answer": "invented"},
    ],
)
def test_invalid_roles_fail_closed(change):
    raw = plan(["0"], ["1"])
    raw.update(change)
    with pytest.raises(ValueError):
        parse_plan(value("a", "b", "c"), json.dumps(raw))


def test_duplicate_json_keys_are_rejected():
    raw = json.dumps(plan())
    raw = raw[:-1] + ', "sufficiency":"complete"}'
    with pytest.raises(ValueError):
        parse_plan(value("a"), raw)


def test_total_evidence_limit_cannot_be_bypassed_with_two_roles():
    runtime = value(*(str(i) for i in range(9)))
    with pytest.raises(ValueError):
        parse_plan(
            runtime, json.dumps(plan([str(i) for i in range(5)], [str(i) for i in range(5, 9)]))
        )


def test_empty_cannot_claim_complete_and_schema_does_not_certify_semantics():
    runtime = value("You could visit a museum.")
    with pytest.raises(ValueError):
        parse_plan(runtime, json.dumps(plan(sufficiency="complete")))
    # Deliberately wrong semantic selection is shape-valid: a model test is required.
    _, report = apply_plan(runtime, json.dumps(plan(["0"], sufficiency="complete")))
    assert report["semantic_completeness"] == "NOT_CERTIFIED"
    assert report["evidence_roles"] == "UNVERIFIED_PLANNER_CLAIMS"


def test_omitted_source_is_unclassified_not_rejected():
    _, report = apply_plan(value("a", "b", "c"), json.dumps(plan(["0"], rejected=["1"])))
    assert report["unclassified_ids"] == ["2"]
    assert report["rejected_ids"] == ["1"]
