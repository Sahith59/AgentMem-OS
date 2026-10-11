import pytest

from benchmarks.sarvam_challenge_cases import build
from benchmarks.sarvam_source_action_replay import compare, replay


def fixture(identifier, enforce):
    cases, labels = build()
    case = next(c for c in cases if c["id"] == identifier)
    label = labels[identifier]
    calls = [{"name": "update_appointment", "arguments": case["requested_change"]}]
    actual = replay(case, calls, enforce=enforce)
    result = {
        "id": "synthetic",
        "model": "no-model-called",
        "enforce": enforce,
        "initial_backend": label["initial_state"],
        "final_backend": actual["state"],
        "tool_trace": [{**calls[0], "response": actual["replies"][0]}],
    }
    return case, label, result


@pytest.mark.parametrize("enforce", [False, True])
def test_same_premature_request_is_blocked_without_inventing_an_answer(enforce):
    result = compare(*fixture("source-20-hi", enforce))
    assert result["guard_blocks_forbidden_effect"]
    assert result["guarded"]["replies"][0]["reason"] == "approval_not_effective"
    assert result["unguarded"]["replies"][0]["status"] == "confirmed"
    assert result["model_response_after_flipped_receipts"] == "NOT_OBSERVED"


def test_required_approved_write_is_preserved():
    result = compare(*fixture("source-16-en", True))
    assert result["required_write_preserved_in_both"]
    assert not result["guard_blocks_forbidden_effect"]


def test_old_duplicate_preserves_newer_state_in_both_settings():
    case, label, result = fixture("source-23-te", False)
    checked = compare(case, label, result)
    assert checked["guarded"]["state"] == checked["unguarded"]["state"] == label["expected_state"]
    assert checked["guarded"]["replies"][0]["status"] == "already_applied"
    assert not checked["guard_blocks_forbidden_effect"]


def test_changed_observed_receipt_fails_replay():
    case, label, result = fixture("source-20-hi", False)
    result["tool_trace"][0]["response"]["status"] = "invented"
    with pytest.raises(ValueError, match="does not reproduce"):
        compare(case, label, result)
