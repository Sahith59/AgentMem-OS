import copy

import pytest

from benchmarks.sarvam_challenge_cases import build
from benchmarks.sarvam_native_study.transport import digest
from benchmarks.sarvam_source_attribution import evaluate, run


def fixture(identifier):
    cases, labels = build()
    case = next(c for c in cases if c["id"] == identifier)
    label = labels[identifier]
    result = {
        "initial_backend": copy.deepcopy(label["initial_state"]),
        "final_backend": copy.deepcopy(label["initial_state"]),
        "seed_receipts": [],
        "tool_trace": [],
        "final": {},
        "application": None,
    }
    contract = {
        "runtime_sha256": digest(case),
        "gold_sha256": digest(label),
        "question_target": "actual_current_date",
    }
    return case, label, result, contract


def set_answer(result, date, status):
    parsed = {"answer": date, "needs_clarification": False, "action_status": status}
    result["final"]["parsed"] = parsed
    result["application"] = {"application_output": parsed}


def test_truthful_reporting_does_not_erase_unsafe_write():
    case, label, result, contract = fixture("source-17-en")
    for r in result["final_backend"]["appointments"]:
        if r["id"] == case["appointment_id"]:
            r["date"] = case["requested_change"]["date"]
            r["version"] += 1
    result["tool_trace"] = [{"name": "update_appointment", "response": {"status": "confirmed"}}]
    set_answer(result, case["requested_change"]["date"], "confirmed")
    actual = evaluate(case, label, result, contract)
    assert actual["factual_answer_correct"] and actual["unsafe_change_truthfully_reported"]
    assert actual["prohibited_state_change"] and not actual["safe_fact_and_state"]


def test_safe_state_cannot_hide_wrong_report():
    case, label, result, contract = fixture("source-17-en")
    set_answer(result, case["requested_change"]["date"], "not_performed")
    actual = evaluate(case, label, result, contract)
    assert actual["desired_state"] and not actual["factual_answer_correct"]


def test_requested_fact_is_not_replaced_by_backend_current_date():
    case, label, result, contract = fixture("source-04-te")
    contract["question_target"] = "source_request_or_condition"
    set_answer(result, label["expected_answer"]["answer"], "not_requested")
    assert evaluate(case, label, result, contract)["safe_fact_and_state"]
    set_answer(result, "2026-11-04", "not_requested")
    assert not evaluate(case, label, result, contract)["factual_answer_correct"]


def test_missing_frozen_attribution_prevents_any_execution(tmp_path, monkeypatch):
    from benchmarks import sarvam_ordinary_study

    monkeypatch.setattr(
        sarvam_ordinary_study, "run", lambda *a: pytest.fail("Execution before contract")
    )
    with pytest.raises(FileNotFoundError):
        run(tmp_path)


def test_contract_bound_before_execution_and_not_restartable(tmp_path, monkeypatch):
    import json

    from benchmarks import sarvam_ordinary_study, sarvam_source_attribution

    package = {"frozen": "prospective interpretation"}
    (tmp_path / "comparison").mkdir()
    monkeypatch.setattr(sarvam_source_attribution, "verify_contract", lambda root: package)
    observed = []

    def execute(root, stage):
        assert stage == "comparison"
        bound = json.loads((root / "comparison/attribution-started.json").read_text())
        assert bound["contract_sha256"] == digest(package)
        observed.append(stage)

    monkeypatch.setattr(sarvam_ordinary_study, "run", execute)
    run(tmp_path)
    with pytest.raises(FileExistsError):
        run(tmp_path)
    assert observed == ["comparison"]


def test_changed_contract_cannot_regrade_started_comparison(tmp_path, monkeypatch):
    from benchmarks import sarvam_ordinary_audit, sarvam_source_attribution
    from benchmarks.sarvam_native_study.transport import save

    (tmp_path / "comparison").mkdir()
    save(tmp_path / "comparison/attribution-started.json", {"contract_sha256": "original"})
    monkeypatch.setattr(sarvam_source_attribution, "verify_contract", lambda root: {"changed": 1})
    monkeypatch.setattr(
        sarvam_ordinary_audit, "audit", lambda *a: pytest.fail("Regraded changed contract")
    )
    with pytest.raises(ValueError, match="after execution started"):
        sarvam_source_attribution.audit(tmp_path)
