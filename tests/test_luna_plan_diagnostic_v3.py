"""No paid calls: frozen projection, cost gate, and durable-run negative paths."""

import json

import pytest

from benchmarks.luna_evidence_plan import opaque_fixture_id
from benchmarks.luna_plan_diagnostic_v3 import (
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
        fresh = case["id"].startswith("fresh_")
        if fresh:
            selected = (
                [t["id"] for t in case["turns"][:8]]
                if case["capacity_exceeds_focus"]
                else case["acceptable_evidence_sets"][0]
            )
            support = [tid for tid in selected if "support" in case["allowed_roles"][tid]]
            qualification = [tid for tid in selected if tid not in support]
            rejected = [
                t["id"]
                for t in case["turns"]
                if t["id"] not in selected and "rejected" in case["allowed_roles"][t["id"]]
            ]
            sufficiency = case["allowed_sufficiency"][0]
        else:
            support = case["required_evidence_ids"]
            qualification = []
            rejected = [x["id"] for x in case["excluded_ids_and_reasons"]]
            sufficiency = "uncertain"
        called.append(case["id"])
        response = {
            "requirements": {
                "operation": "other",
                "target": "",
                "time_window": "",
                "output_unit": "",
                "needed_facts": [],
            },
            "support_turn_ids": [
                opaque_fixture_id(t["id"]) for t in case["turns"] if t["id"] in support
            ],
            "qualification_turn_ids": [
                opaque_fixture_id(t["id"]) for t in case["turns"] if t["id"] in qualification
            ],
            "rejected_turn_ids": [
                opaque_fixture_id(t["id"]) for t in case["turns"] if t["id"] in rejected
            ],
            "sufficiency": sufficiency,
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
    assert preflight(package)["calls_no_retries"] == 23
    request = json.dumps([row["request"] for row in package["cases"]])
    for key in (
        "required_evidence_ids",
        "excluded_ids_and_reasons",
        "expected_answer",
        "known_incorrect_answer",
        "acceptable_evidence_sets",
        "allowed_roles",
        "rationale",
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
    assert result["completed"] == result["selection_passed"] == 23
    assert result["english_accuracy"] == "NOT_MEASURED"
    assert len(called) == 23
    again = run(package, tmp_path, auth, mode="offline-test", provider=provider)
    assert again == result and len(called) == 23
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
    assert audit["not_started"] == 22
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


def test_population_totals_remain_separate(tmp_path):
    package = build_package()
    provider, _ = fake_provider()
    result = run(
        package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
    )
    assert result["population_results"] == {
        "exposed_development": {"completed": 13, "passed": 13},
        "internal_validation": {"completed": 9, "passed": 9},
        "capacity_awareness": {"completed": 1, "passed": 1},
    }


def test_fresh_semantics_wrong_status_or_role_and_missing_evidence_fail():
    from benchmarks.luna_role_validation import review

    fresh = next(c for c in fixture_cases() if c["id"] == "fresh_undecided")
    tid = fresh["turns"][0]["id"]
    assert not review(fresh, [], [], [], "uncertain")["pass"]
    assert not review(fresh, [tid], [], [], "uncertain")["pass"]
    assert not review(fresh, [], [tid], [], "complete")["pass"]
    assert review(fresh, [], [tid], [], "uncertain")["pass"]


def test_alternative_evidence_sets_and_capacity_do_not_claim_coverage():
    from benchmarks.luna_role_validation import review

    fresh = next(c for c in fixture_cases() if c["id"] == "fresh_resolved")
    a, b = [t["id"] for t in fresh["turns"]]
    assert review(fresh, [b], [], [a], "complete")["pass"]
    assert review(fresh, [b], [a], [], "complete")["pass"]
    cap = next(c for c in fixture_cases() if c["id"] == "fresh_capacity")
    ids = [t["id"] for t in cap["turns"][:8]]
    assert review(cap, ids, [], [], "uncertain")["pass"]
    assert not review(cap, ids, [], [], "complete")["pass"]


def test_semantic_failure_stays_in_denominator_without_retries(tmp_path):
    package = build_package()
    provider, calls = fake_provider()

    def omit_uncertainty(request):
        response = provider(request)
        if (
            json.loads(request["messages"][1]["content"])["question"]
            == "Which train did I book for my October 8 trip?"
        ):
            raw = json.loads(response["text"])
            raw["qualification_turn_ids"] = []
            response["text"] = json.dumps(raw)
        return response

    result = run(
        package,
        tmp_path,
        approval(package, tmp_path),
        mode="offline-test",
        provider=omit_uncertainty,
    )
    assert len(calls) == result["completed"] == 23
    assert result["selection_passed"] == 22
    assert result["selection_failed"] == 1
    assert result["all_case_gate"] == "FAIL_OR_INCOMPLETE"


def test_fresh_labels_are_well_formed_and_all_old_labels_stay_separate():
    cases = fixture_cases()
    assert len({c["id"] for c in cases}) == 23
    for case in cases[13:]:
        ids = {t["id"] for t in case["turns"]}
        assert len(ids) == len(case["turns"])
        assert set(case["allowed_roles"]) == ids
        assert set(case["allowed_sufficiency"]) <= {"complete", "uncertain"}
        for roles in case["allowed_roles"].values():
            assert roles and set(roles) <= {"support", "qualification", "rejected", "unclassified"}
        if case["capacity_exceeds_focus"]:
            assert len(ids) > 8 and not case["acceptable_evidence_sets"]
            assert case["allowed_sufficiency"] == ["uncertain"]
        else:
            assert case["acceptable_evidence_sets"]
            for option in case["acceptable_evidence_sets"]:
                assert len(option) == len(set(option)) <= 8
                assert set(option) <= ids
