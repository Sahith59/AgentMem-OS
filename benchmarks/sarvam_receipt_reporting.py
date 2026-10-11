"""Offline candidate: application-owned action status, with raw model output preserved.

Only consume trusted application events, never model-authored receipt claims. This
bounded single-update projector neither executes actions nor corrects answer text.
It is ordinary interface engineering, not a memory algorithm or a new model score.
"""

import copy
from pathlib import Path

from benchmarks.sarvam_native_controls.approval_backend import ARGUMENTS
from benchmarks.sarvam_native_study.score import parse
from benchmarks.sarvam_native_study.transport import digest, save


def validate_request(args):
    if (
        not isinstance(args, dict)
        or set(args) != ARGUMENTS
        or type(args["expected_version"]) is not int
        or args["expected_version"] < 0
        or any(
            not isinstance(args[k], str) or not args[k] for k in ARGUMENTS - {"expected_version"}
        )
    ):
        raise ValueError("Incomplete trusted change request")


def validate_confirmation(receipt, args):
    if (
        receipt.get("status") != "confirmed"
        or receipt.get("date") != args["date"]
        or type(receipt.get("version")) is not int
        or receipt["version"] != args["expected_version"] + 1
    ):
        raise ValueError("Receipt does not establish this change")


def project(*, intent, requested_change, prior_receipts, tool_trace, raw_content, completed):
    """Return a separate application view; refuse incomplete/ambiguous action histories.

    No database/gold access. Dates, clarification and raw output are untouched.
    Caller authentication and event provenance must be guaranteed by the application.
    """
    if completed is not True or intent not in ("read", "change"):
        raise ValueError("Completed execution and trusted intent required")
    parsed = parse(raw_content)
    attempts = [t for t in tool_trace if t["name"] == "update_appointment"]
    if any(t["response"].get("status") == "error" for t in tool_trace):
        raise ValueError("Uncertain backend outcome")
    if len(attempts) > 1:
        raise ValueError("Multiple update outcomes require a richer action contract")
    if intent == "read":
        if requested_change is not None or attempts:
            raise ValueError("Read-only action boundary violated")
        status, basis = "not_requested", "trusted_read_intent"
    else:
        validate_request(requested_change)
        status, basis = "not_performed", "no_update_receipt"
        for prior in prior_receipts:
            if prior["arguments"] == requested_change:
                validate_confirmation(prior["receipt"], requested_change)
                status, basis = "already_applied", "matching_prior_receipt"
        if attempts:
            event = attempts[0]
            if event["arguments"] != requested_change:
                raise ValueError("Unexpected update arguments require review")
            receipt = event["response"]
            status = receipt.get("status")
            if status == "confirmed":
                validate_confirmation(receipt, requested_change)
            elif status == "already_applied":
                validate_confirmation(receipt.get("original_receipt", {}), requested_change)
            elif status != "rejected":
                raise ValueError("Unrecognized update outcome")
            basis = "actual_update_receipt"
    application = copy.deepcopy(parsed)
    application["action_status"] = status
    return {
        "raw_content": raw_content,
        "raw_parsed": parsed,
        "application_output": application,
        "status_changed": parsed["action_status"] != status,
        "status_basis": basis,
    }


def audit(root):
    """Post-hoc projection of saved episodes; never replace original scores or gate."""
    from benchmarks.sarvam_native_controls.integrated import read
    from benchmarks.sarvam_native_controls.integrated_audit import audit as replay
    from benchmarks.sarvam_native_controls.integrated_audit import grade

    original = replay(root)
    assert original == read(root / "verification.json")
    cases = {c["id"]: c for c in read(root / "runtime.json")}
    labels = read(root / "gold.json")
    rows = []
    for row in original["rows"]:
        result = read(root / "episodes" / row["id"] / "result.json")
        case = cases[result["case_id"]]
        projected = project(
            intent=case["intent"],
            requested_change=case["requested_change"],
            prior_receipts=result["seed_receipts"],
            tool_trace=result["tool_trace"],
            raw_content=result["final"]["content"],
            completed=result["terminal"] == "answer",
        )
        candidate = copy.deepcopy(result)
        candidate["final"]["parsed"] = projected["application_output"]
        measured = grade(case, labels[case["id"]], candidate)
        assert candidate["final_backend"] == result["final_backend"]
        assert projected["application_output"]["answer"] == projected["raw_parsed"]["answer"]
        assert (
            projected["application_output"]["needs_clarification"]
            == projected["raw_parsed"]["needs_clarification"]
        )
        rows.append(
            {
                "id": row["id"],
                **projected,
                "raw_correct_report": row["correct_report"],
                "raw_task_success": row["task_success"],
                "application_correct_report": measured["correct_report"],
                "application_task_success": measured["task_success"],
                "forbidden_update_attempt": measured["forbidden_update_attempt"],
            }
        )
    return {
        "verified": True,
        "kind": "POST_HOC_OFFLINE_APPLICATION_PROJECTION_NOT_NEW_MODEL_RUN",
        "original_verification_sha256": digest(original),
        "projector_source_sha256": digest(Path(__file__).read_text()),
        "original_gate_pass": original["gate_pass"],
        "new_api_calls": 0,
        "rows": rows,
        "changed_statuses": sum(r["status_changed"] for r in rows),
        "raw_correct_reports": sum(r["raw_correct_report"] for r in rows),
        "application_correct_reports": sum(r["application_correct_report"] for r in rows),
        "raw_task_successes": sum(r["raw_task_success"] for r in rows),
        "application_task_successes": sum(r["application_task_success"] for r in rows),
        "limitation": (
            "Addresses an interface contract on observed cases. Original gate stays failed; "
            "no new held-out result or model/memory improvement."
        ),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, audit(args.root))
