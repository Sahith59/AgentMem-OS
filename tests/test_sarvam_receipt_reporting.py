import copy
import json

import pytest

from benchmarks.sarvam_receipt_reporting import project


def supplied():
    args = {
        "appointment_id": "booking",
        "date": "2026-12-05",
        "expected_version": 3,
        "idempotency_key": "change",
        "approval_id": "grant",
    }
    return {
        "intent": "change",
        "requested_change": args,
        "prior_receipts": [],
        "tool_trace": [],
        "raw_content": json.dumps(
            {"answer": "2026-11-05", "needs_clarification": False, "action_status": "confirmed"}
        ),
        "completed": True,
    }


def confirmed():
    return {"status": "confirmed", "date": "2026-12-05", "version": 4}


def test_unsupported_claim_removed_without_changing_answer_or_raw_output():
    args = supplied()
    before = copy.deepcopy(args)
    output = project(**args)
    assert output["application_output"] == {
        "answer": "2026-11-05",
        "needs_clarification": False,
        "action_status": "not_performed",
    }
    assert output["raw_content"] == args["raw_content"] and args == before
    assert output["status_changed"]


@pytest.mark.parametrize("status", ["confirmed", "rejected", "already_applied"])
def test_actual_receipt_authoritative(status):
    args = supplied()
    receipt = {**confirmed(), "status": status}
    if status == "already_applied":
        receipt["original_receipt"] = confirmed()
        receipt["date"], receipt["version"] = "2026-12-09", 7
    args["tool_trace"] = [
        {"name": "update_appointment", "arguments": args["requested_change"], "response": receipt}
    ]
    assert project(**args)["application_output"]["action_status"] == status


def test_prior_identical_request_is_completion_not_new_write():
    args = supplied()
    args["prior_receipts"] = [{"arguments": args["requested_change"], "receipt": confirmed()}]
    out = project(**args)
    assert out["application_output"]["action_status"] == "already_applied"
    assert out["application_output"]["answer"] == "2026-11-05"  # No answer correction here.


def test_unrelated_prior_receipt_not_authority():
    args = supplied()
    args["prior_receipts"] = [
        {
            "arguments": {**args["requested_change"], "idempotency_key": "other"},
            "receipt": confirmed(),
        }
    ]
    assert project(**args)["application_output"]["action_status"] == "not_performed"


def test_read_intent():
    args = supplied()
    args.update(intent="read", requested_change=None)
    assert project(**args)["application_output"]["action_status"] == "not_requested"


@pytest.mark.parametrize(
    "defect",
    [
        "incomplete",
        "backend_error",
        "multi_update",
        "wrong_args",
        "bad_receipt",
        "unknown_status",
        "read_update",
        "bad_request",
        "malformed_final",
    ],
)
def test_fail_closed_for_ambiguous_or_invalid_evidence(defect):
    args = supplied()
    event = {
        "name": "update_appointment",
        "arguments": args["requested_change"],
        "response": confirmed(),
    }
    if defect == "incomplete":
        args["completed"] = False
    elif defect == "backend_error":
        args["tool_trace"] = [{**event, "response": {"status": "error"}}]
    elif defect == "multi_update":
        args["tool_trace"] = [event, event]
    elif defect == "wrong_args":
        args["tool_trace"] = [{**event, "arguments": {**args["requested_change"], "date": "other"}}]
    elif defect == "bad_receipt":
        args["tool_trace"] = [{**event, "response": {**confirmed(), "version": 5}}]
    elif defect == "unknown_status":
        args["tool_trace"] = [{**event, "response": {"status": "maybe"}}]
    elif defect == "read_update":
        args.update(intent="read", requested_change=None, tool_trace=[event])
    elif defect == "bad_request":
        args["requested_change"] = {}
    else:
        args["raw_content"] = "not json"
    with pytest.raises(ValueError):
        project(**args)
