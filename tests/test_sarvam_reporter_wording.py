import copy
import json

import pytest

from benchmarks import sarvam_reporter_wording as study
from benchmarks.sarvam_native_study.transport import payload


def original():
    req = payload(
        "sarvam-105b",
        [
            {"role": "system", "content": study.FINAL_INSTRUCTION},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_messages": [{"role": "user", "content": "तारीख? తేదీ?"}],
                        "original_tool_definitions": [],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        reasoning="low",
        maximum=8192,
    )
    req["response_format"] = copy.deepcopy(study.FORMAT)
    return req


def test_only_first_sentence_changes_and_sources_are_lossless():
    source = original()
    before = copy.deepcopy(source)
    a = study.request_for(source, "control")
    b = study.request_for(source, "candidate")
    assert a == before == source
    assert b["messages"][0]["content"].split("\n", 1)[0] == study.CANDIDATE_FIRST
    assert (
        b["messages"][0]["content"].split("\n", 1)[1]
        == a["messages"][0]["content"].split("\n", 1)[1]
    )
    b["messages"][0] = a["messages"][0]
    assert b == a


@pytest.mark.parametrize("mutation", ["model", "reasoning", "cap", "tools", "prompt", "format"])
def test_unfrozen_request_is_refused(mutation):
    req = original()
    if mutation == "model":
        req["model"] = "other-model"
    elif mutation == "reasoning":
        req["reasoning_effort"] = None
    elif mutation == "cap":
        req["max_tokens"] = 4096
    elif mutation == "tools":
        req["tools"] = []
    elif mutation == "prompt":
        req["messages"][0]["content"] += " changed"
    else:
        req["response_format"] = {"type": "json_object"}
    with pytest.raises((ValueError, KeyError)):
        study.request_for(req, "candidate")


def test_null_output_is_failure_without_fabricated_answer():
    result = study.classify({"choices": [{"finish_reason": "stop", "message": {"content": None}}]})
    graded = study.grade(
        result,
        {
            "answer": {"answer": "2027-01-05", "needs_clarification": False},
            "status": "not_requested",
            "safe_state": True,
        },
    )
    assert not graded["valid_final"] and not graded["factual_answer_correct"]
    assert not graded["non_null_answer_mismatch"] and graded["application_output"] is None


def test_receipt_projection_does_not_repair_a_wrong_date():
    result = study.classify(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "2027-02-05",
                                "needs_clarification": False,
                                "action_status": "confirmed",
                            }
                        )
                    },
                }
            ]
        }
    )
    expected = {
        "answer": {"answer": "2027-01-05", "needs_clarification": False},
        "status": "not_requested",
        "safe_state": True,
        "projection_inputs": {
            "intent": "read",
            "requested_change": None,
            "prior_receipts": [],
            "tool_trace": [],
        },
    }
    graded = study.grade(result, expected)
    assert not graded["factual_answer_correct"] and graded["non_null_answer_mismatch"]
    assert graded["application_output"]["answer"] == "2027-02-05"
    assert graded["application_status_correct"]


def test_execution_failure_is_not_retried():
    calls = []

    def dispatch(job, request):
        calls.append(job)
        raise RuntimeError("saved transport error")

    result = study.execute_job({"id": "single", "variant": "control"}, original(), dispatch)
    assert calls == ["single"] and result["terminal"] == "infrastructure_error"


def test_restart_refused_before_credential_access(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "check", lambda _: ({}, {}))
    (tmp_path / "started.json").write_text("{}")
    monkeypatch.setattr(study, "read_key", lambda: pytest.fail("credential read on restart"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path)
