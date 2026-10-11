import pytest

from benchmarks.sarvam_native_study.analyze import combine, pairs


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
