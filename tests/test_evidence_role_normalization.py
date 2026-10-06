import json

import pytest

from benchmarks.evidence_focus import FocusInput, SourceTurn
from benchmarks.evidence_role_normalization import apply_normalized, normalize
from benchmarks.guarded_focus import apply_guarded


def runtime():
    turns = (
        SourceTurn("a", "user", "", "I raised 100 dollars."),
        SourceTurn("b", "user", "", "I also raised 200 dollars."),
    )
    return FocusInput("How much did I raise?", "", "\n".join(t.text for t in turns), turns)


def plan():
    return dict(
        requirements=dict(
            operation="sum", target="", time_window="", output_unit="", needed_facts=[]
        ),
        support_turn_ids=["a", "b"],
        qualification_turn_ids=["a", "b"],
        rejected_turn_ids=[],
        sufficiency="uncertain",
    )


def test_overlap_resolves_without_changing_union_or_rendered_source_text():
    v = runtime()
    raw = json.dumps(plan())
    normalized, audit = normalize(v, raw)
    assert json.loads(normalized)["support_turn_ids"] == []
    assert audit["overlap_ids"] == ["a", "b"] and not audit["source_union_changed"]
    candidate, report = apply_normalized(v, raw)
    assert report["selected_ids"] == ["a", "b"]
    for receipt, turn in zip(report["receipts"], v.turns, strict=True):
        assert candidate[receipt["start"] : receipt["end"]] == turn.text


def test_valid_plans_remain_byte_identical_after_rendering():
    p = plan()
    p["qualification_turn_ids"] = []
    raw = json.dumps(p)
    v = runtime()
    a, _ = apply_guarded(v, raw)
    b, report = apply_normalized(v, raw)
    assert a == b and not report["role_normalization"]["changed"]


@pytest.mark.parametrize(
    "change",
    [
        {"support_turn_ids": ["a", "a"]},
        {"qualification_turn_ids": ["alien"]},
        {"rejected_turn_ids": ["a"]},
        {"support_turn_ids": ["b", "a"]},
        {"sufficiency": "sufficient"},
        {"answer": 300},
    ],
)
def test_other_defects_remain_errors(change):
    p = plan()
    p.update(change)
    with pytest.raises(ValueError):
        normalize(runtime(), json.dumps(p))
