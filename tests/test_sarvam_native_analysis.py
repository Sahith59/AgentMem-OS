import json

import pytest

from benchmarks.sarvam_native_study.analyze import combine, pairs, usage_audit


def test_continuation_cannot_erase_failure_or_leave_missing_episode():
    failure = {"id": "a", "attempted": True, "task_success": False}
    unattempted = {"id": "b", "attempted": False}
    success = {"id": "b", "attempted": True, "task_success": True}
    assert combine(["a", "b"], [failure, unattempted], [success]) == [failure, success]
    with pytest.raises(ValueError, match="Repeated attempted"):
        combine(["a", "b"], [failure, unattempted], [failure, success])
    with pytest.raises(ValueError, match="Missing"):
        combine(["a", "b"], [failure, unattempted])


def test_reasoning_pairing_preserves_model_arm_and_losses():
    rows = []
    for model in ("m0", "m1"):
        for arm in ("full_history", "sqlite"):
            for case, before, after in (("one", False, True), ("two", True, False)):
                for reasoning, ok in ((None, before), ("low", after)):
                    rows.append(
                        {
                            "model": model,
                            "arm": arm,
                            "case_id": case,
                            "reasoning": reasoning,
                            "task_success": ok,
                        }
                    )
    result = pairs(rows, "reasoning", None, "low")
    assert len(result["groups"]) == 4
    assert all(g["right_gain"] == g["right_loss"] == 1 for g in result["groups"].values())
    with pytest.raises(ValueError, match="Incomplete"):
        pairs(rows[:-1], "reasoning", None, "low")
    with pytest.raises(ValueError, match="Duplicate"):
        pairs(rows + rows[:1], "reasoning", None, "low")


def test_missing_reasoning_telemetry_is_not_reported_zero(tmp_path):
    calls = tmp_path / "calls"
    calls.mkdir()
    for i, details in enumerate((None, {"reasoning_tokens": 0}, {"reasoning_tokens": 7})):
        job = "attribution-" + str(i)
        request = {"model": "m0", "reasoning_effort": "low"}
        body = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "completion_tokens_details": details,
            },
            "choices": [{"message": {"content": "test"}}],
        }
        (calls / (job + ".request.json")).write_text(json.dumps({"request": request}))
        (calls / (job + ".receipt.json")).write_text(
            json.dumps({"job": job, "status": 200, "body_text": json.dumps(body)})
        )
    result = usage_audit(tmp_path)["attribution|m0|low"]
    assert result["receipts"] == 3
    assert result["reasoning_token_count_unreported"] == 1
    assert result["calls_reporting_positive_reasoning_tokens"] == 1
    assert result["reported_reasoning_tokens"] == 7
