import copy
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks.sarvam_native_controls.cases import build, probes
from benchmarks.sarvam_native_controls.engine import (
    PROFILES,
    ControlMeter,
    episode,
    grade,
    request_for,
)
from benchmarks.sarvam_native_study import backend
from benchmarks.sarvam_native_study.transport import digest, save


def test_only_response_format_changes_within_each_pair():
    cases, _ = build()
    hashes = []
    for case in cases:
        for model in PROFILES:
            control = request_for(case, model, "control")
            changed = request_for(case, model, "schema")
            assert control.pop("response_format", None) != changed.pop("response_format")
            assert control == changed
            assert control["max_tokens"] == 8192
        hashes.append(digest(request_for(case, "sarvam-105b", "control")))
    assert len(hashes) == len(set(hashes)) == 72
    assert len({c["workflow"] for c in cases}) == 24
    setup, _ = probes()
    assert not set(hashes) & {digest(request_for(c, "sarvam-105b", "control")) for c in setup}


def test_inference_does_not_receive_gold_or_hidden_backend():
    cases, _ = build()
    for case in cases:
        altered = {**case, "backend": {"POISON": True}, "answer": "POISON", "id": "POISON"}
        assert request_for(case, "sarvam-105b", "schema") == request_for(
            altered, "sarvam-105b", "schema"
        )


def test_fixture_state_and_receipt_consistency(tmp_path):
    """Oracle-only backend consistency; never described as measured model success."""
    for cases, gold in (build(), probes()):
        for case in cases:
            label = gold[case["id"]]
            db = tmp_path / (case["id"] + ".sqlite")
            backend.create(db, case)
            trace = []
            if label["receipt_status"]:
                if case["id"] == "setup-action-te":
                    date, key = "2026-10-25", "TEST-KEY"
                else:
                    fields = case["question"].rsplit(" appointment_id=", 1)[1]
                    parts = fields.rstrip(".").split("; ")
                    date, key = parts[1].split("=")[1], parts[2].split("=")[1]
                args = {
                    "appointment_id": case["backend"]["appointment_id"],
                    "date": date,
                    "idempotency_key": key,
                    "expected_version": case["backend"]["version"],
                }
                reply = backend.execute(db, "update_appointment", args)
                trace.append({"name": "update_appointment", "arguments": args, "response": reply})
            result = {
                "terminal": "answer",
                "content": json.dumps(label["response"]),
                "tool_trace": trace,
                "final_backend": backend.state(db),
            }
            assert grade(case, label, result)["task_success"], case["id"]
            if label["receipt_status"] == "confirmed":
                wrong = {**result, "tool_trace": []}
                assert not grade(case, label, wrong)["task_success"]
                wrong = {**result, "final_backend": case["backend"]}
                assert not grade(case, label, wrong)["task_success"]


def test_rejected_and_fenced_answers_are_not_silently_repaired():
    cases, gold = build()
    case = cases[0]
    label = gold[case["id"]]
    result = {
        "terminal": "answer",
        "content": "```json\n" + json.dumps(label["response"]) + "\n```",
        "tool_trace": [],
        "final_backend": case["backend"],
    }
    assert not grade(case, label, result)["valid_output"]
    result["content"] = json.dumps({**label["response"], "answer": "WRONG"})
    assert grade(case, label, result)["valid_output"]
    assert not grade(case, label, result)["task_success"]


def test_approval_violation_remains_separate_from_correct_words():
    cases, gold = build()
    case = next(c for c in cases if c["id"] == "declined-en")
    label = gold[case["id"]]
    result = {
        "terminal": "answer",
        "content": json.dumps(label["response"]),
        "tool_trace": [{"name": "update_appointment", "response": {"status": "rejected"}}],
        "final_backend": case["backend"],
    }
    score = grade(case, label, result)
    assert score["answer_correct"] and score["backend_correct"]
    assert score["forbidden_mutation_attempt"] and not score["task_success"]


def test_no_retry_even_on_transport_failure(tmp_path):
    case = probes()[0][0]
    spec = {
        "id": "test",
        "model": "sarvam-105b",
        "variant": "schema",
        "maximum_calls": 2,
        "initial_sha256": digest(request_for(case, "sarvam-105b", "schema")),
    }
    seen = []

    def broken(job, request):
        seen.append(job)
        raise RuntimeError("synthetic provider failure")

    result = episode(tmp_path, spec, case, broken)
    assert len(seen) == 1 and result["terminal"] == "infrastructure_error"
    with pytest.raises(FileExistsError):
        episode(tmp_path, spec, case, broken)
    assert len(seen) == 1


def test_budget_carries_prior_debit_and_is_concurrency_safe(tmp_path):
    config = {
        "phase_cap_microinr": 100_000000,
        "campaign_cap_microinr": 6000_000000,
        "prior_retained_microinr": 5990_000000,
        "stages": {"comparison": {"cap_microinr": 100_000000, "calls": 100}},
    }
    save(tmp_path / "budget.json", config)
    meter = ControlMeter(tmp_path)
    request = request_for(build()[0][0], "sarvam-105b", "schema")

    def reserve(i):
        try:
            meter.reserve(f"r{i}", "comparison", copy.deepcopy(request))
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(20)))
    assert sum(results) == 3
    assert meter.summary()["comparison"]["charged_with_headroom_inr"] <= 10
