"""No paid calls: frozen projection, cost gate, and durable-run negative paths."""

import json

import pytest

from benchmarks.luna_evidence_plan import opaque_fixture_id
from benchmarks.luna_plan_diagnostic_v2 import (
    build_package,
    evaluate,
    fixture_cases,
    preflight,
    run,
)


def approval(package, directory, mode="offline-test"):
    plan = preflight(package)
    return {
        "approved": True,
        "authorization_text": "Test-only fake provider",
        "mode": mode,
        "package_sha256": plan["package_sha256"],
        "maximum_attempts": plan["calls_no_retries"],
        "budget_nusd": plan["maximum_reservation_nusd"],
        "output_directory": str(directory.resolve()),
    }


def fake_provider():
    fixture = {c["question"]: c for c in fixture_cases()}
    called = []

    def provider(request):
        data = json.loads(request["messages"][1]["content"])
        assert set(data) == {"question", "question_date", "sources"}
        assert "expected_answer" not in json.dumps(request)
        case = fixture[data["question"]]
        ids = [
            opaque_fixture_id(t["id"])
            for t in case["turns"]
            if t["id"] in case["required_evidence_ids"]
        ]
        called.append(case["id"])
        response = {
            "requirements": {
                "operation": "other",
                "target": "",
                "time_window": "",
                "output_unit": "",
                "needed_facts": [],
            },
            "turn_ids": ids,
            "rejected_turn_ids": [
                opaque_fixture_id(t["id"])
                for t in case["turns"]
                if t["id"] in {x["id"] for x in case["excluded_ids_and_reasons"]}
            ],
            "sufficiency": "uncertain",
        }
        return {
            "text": json.dumps(response),
            "finish_reason": "stop",
            "model": "gpt-5.6-luna",
            "id": "fake-" + case["id"],
            "request_id": "fake-req-" + case["id"],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
        }

    return provider, called


def test_package_reconstruction_and_no_label_leakage():
    package = build_package()
    assert preflight(package)["calls_no_retries"] == 13
    request = json.dumps(package["cases"][0]["request"])
    for key in (
        "required_evidence_ids",
        "excluded_ids_and_reasons",
        "expected_answer",
        "known_incorrect_answer",
    ):
        assert key not in request
    package["cases"][0]["request"]["model"] = "gpt-4o"
    with pytest.raises(ValueError, match="Changed diagnostic"):
        preflight(package)


def test_fake_run_and_resume_never_repeats_calls(tmp_path):
    package = build_package()
    provider, called = fake_provider()
    auth = approval(package, tmp_path)
    result = run(package, tmp_path, auth, mode="offline-test", provider=provider)
    assert result["completed"] == result["selection_passed"] == 13
    assert result["english_accuracy"] == "NOT_MEASURED"
    assert len(called) == 13
    again = run(package, tmp_path, auth, mode="offline-test", provider=provider)
    assert again == result and len(called) == 13
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    state["reserved_nusd"] -= 1
    with pytest.raises(ValueError, match="Reservation ledger"):
        evaluate(package, state)


def test_unresolved_attempt_never_retries(tmp_path):
    package = build_package()
    auth = approval(package, tmp_path)

    def fail(_request):
        raise RuntimeError("simulated uncertain provider state")

    with pytest.raises(RuntimeError, match="no automatic retry"):
        run(package, tmp_path, auth, mode="offline-test", provider=fail)
    provider, called = fake_provider()
    with pytest.raises(ValueError, match="Unresolved attempt"):
        run(package, tmp_path, auth, mode="offline-test", provider=provider)
    assert not called


def test_paid_mode_rejects_injected_provider_and_weak_approval(tmp_path):
    package = build_package()
    auth = approval(package, tmp_path, mode="paid")
    provider, called = fake_provider()
    with pytest.raises(ValueError, match="real provider"):
        run(package, tmp_path, auth, mode="paid", provider=provider)
    assert not called
    auth["budget_nusd"] -= 1
    with pytest.raises(ValueError, match="authorization"):
        run(package, tmp_path, auth, mode="paid")


def test_failed_schema_keeps_billed_usage_and_stops_before_next_call(tmp_path):
    package = build_package()
    auth = approval(package, tmp_path)
    valid_provider, calls = fake_provider()

    def malformed(request):
        result = valid_provider(request)
        data = json.loads(result["text"])
        data["sufficiency"] = "sufficient"
        result["text"] = json.dumps(data)
        return result

    with pytest.raises(RuntimeError, match="no automatic retry"):
        run(package, tmp_path, auth, mode="offline-test", provider=malformed)
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    audit = evaluate(package, state)
    assert len(calls) == audit["attempted"] == audit["failed_or_pending"] == 1
    assert audit["not_started"] == 12
    assert audit["completed"] == 0
    assert audit["usage_upper_nusd"] == 100 * 250 + 100 * 1200


def test_truncated_output_keeps_usage_without_retry_or_acceptance(tmp_path):
    package = build_package()
    auth = approval(package, tmp_path)
    valid_provider, calls = fake_provider()

    def truncated(request):
        result = valid_provider(request)
        result.update(finish_reason="length", text="{")
        return result

    with pytest.raises(RuntimeError, match="no automatic retry"):
        run(package, tmp_path, auth, mode="offline-test", provider=truncated)
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    audit = evaluate(package, state)
    assert audit["completed"] == 0
    assert audit["failed_or_pending"] == 1
    assert audit["usage_upper_nusd"] == 145000
    with pytest.raises(ValueError, match="Unresolved attempt"):
        run(package, tmp_path, auth, mode="offline-test", provider=truncated)
    assert len(calls) == 1
