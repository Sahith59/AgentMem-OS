import itertools
import json

import pytest

from benchmarks.evidence_focus import FocusInput, SourceTurn
from benchmarks.evidence_role_normalization_v2 import apply_normalized, normalize


def runtime():
    turns = tuple(SourceTurn(i, "user", "", f"Original source {i}.") for i in "abcd")
    return FocusInput("What happened?", "", "\n".join(t.text for t in turns), turns)


def plan():
    return dict(
        requirements=dict(
            operation="direct_recall", target="", time_window="", output_unit="", needed_facts=[]
        ),
        support_turn_ids=["a", "b"],
        qualification_turn_ids=[],
        rejected_turn_ids=["c", "d"],
        sufficiency="uncertain",
    )


def test_all_role_permutations_render_identically_and_preserve_original_text():
    value = runtime()
    p = plan()
    p["qualification_turn_ids"] = ["a", "b"]
    expected, _ = apply_normalized(value, json.dumps(p))
    for support, qualification, rejected in itertools.product(
        itertools.permutations("ab"),
        itertools.permutations("ab"),
        itertools.permutations("cd"),
    ):
        p.update(
            support_turn_ids=list(support),
            qualification_turn_ids=list(qualification),
            rejected_turn_ids=list(rejected),
        )
        actual, report = apply_normalized(value, json.dumps(p))
        assert actual == expected and report["selected_ids"] == ["a", "b"]
        assert not report["role_normalization"]["source_union_changed"]
        for receipt, turn in zip(report["receipts"], value.turns[:2], strict=True):
            assert actual[receipt["start"] : receipt["end"]] == turn.text


@pytest.mark.parametrize(
    "change",
    [
        {"rejected_turn_ids": ["unknown"]},
        {"support_turn_ids": ["a", "a"]},
        {"rejected_turn_ids": ["a"]},
        {"qualification_turn_ids": [None]},
        {"sufficiency": "certain"},
        {"answer": "invented"},
    ],
)
def test_semantic_and_contract_defects_are_not_repaired(change):
    p = plan()
    p.update(change)
    with pytest.raises(ValueError):
        normalize(runtime(), json.dumps(p))


def test_rejected_order_is_audited_but_changes_no_focus_context():
    p = plan()
    expected, _ = apply_normalized(runtime(), json.dumps(p))
    p["rejected_turn_ids"].reverse()
    candidate, report = apply_normalized(runtime(), json.dumps(p))
    assert candidate == expected
    assert report["role_normalization"]["reordered_fields"] == ["rejected_turn_ids"]
