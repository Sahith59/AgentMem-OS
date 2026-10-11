import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks.sarvam_native_study import backend
from benchmarks.sarvam_native_study.dataset import build
from benchmarks.sarvam_native_study.diagnostic import profile_request
from benchmarks.sarvam_native_study.paced import continuation_jobs
from benchmarks.sarvam_native_study.runner import initial
from benchmarks.sarvam_native_study.score import grade
from benchmarks.sarvam_native_study.transport import Meter, payload, save


def test_unknown_retains_reservation_and_no_duplicate_dispatch(tmp_path):
    meter = Meter(tmp_path)
    req = payload("sarvam-105b", [])
    reservation = meter.reserve("one", "setup", req)
    assert not meter.settle("one", {}, req["model"], req["max_tokens"])
    assert meter.summary()["setup"]["charged_with_headroom_inr"] == reservation / 1e6
    with pytest.raises(sqlite3.IntegrityError):
        meter.reserve("one", "setup", req)


def test_known_usage_settles_with_headroom(tmp_path):
    meter = Meter(tmp_path)
    req = payload("sarvam-105b", [])
    meter.reserve("one", "setup", req)
    assert meter.settle(
        "one",
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
        req["model"],
        4096,
    )
    s = meter.summary()["setup"]
    assert s["known_usage_estimate_inr"] == 0.0027
    assert s["charged_with_headroom_inr"] == 0.003375
    with pytest.raises(ValueError):
        meter.settle("one", {}, req["model"], 4096)


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 1},
        {"prompt_tokens": 1, "completion_tokens": 5000, "total_tokens": 5001},
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 8},
    ],
)
def test_invalid_usage_keeps_reservation(tmp_path, usage):
    meter = Meter(tmp_path)
    meter.reserve("a", "setup", payload("sarvam-105b", []))
    assert not meter.settle("a", {"usage": usage}, "sarvam-105b", 4096)
    assert meter.summary()["setup"]["unknown_or_pending"] == 1


def test_budget_ceiling_and_immutable_artifact(tmp_path):
    meter = Meter(tmp_path)
    with meter.connect() as db:
        db.execute("INSERT INTO calls VALUES ('used','setup',300000000,300000000,NULL,'unknown')")
    with pytest.raises(ValueError, match="Stage ceiling"):
        meter.reserve("a", "setup", payload("sarvam-105b", []))
    target = tmp_path / "receipt.json"
    save(target, {"old": 1})
    with pytest.raises(FileExistsError):
        save(target, {"new": 2})
    assert json.loads(target.read_text()) == {"old": 1}


def test_bank_group_splits_and_equivalence():
    cases, gold = build()
    assert len(cases) == len(gold) == 700
    assert len({c["base_id"] for c in cases}) == 200
    assert len({c["template_group"] for c in cases}) == 100
    groups = {}
    for c in cases:
        groups.setdefault(c["template_group"], set()).add(c["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    by_base = {}
    for c in cases:
        label = {k: v for k, v in gold[c["id"]].items() if k != "runtime_sha256"}
        if c["base_id"] in by_base:
            assert label == by_base[c["base_id"]]
        by_base[c["base_id"]] = label


def test_no_gold_or_backend_or_ids_in_inference_projection():
    cases, _ = build()
    case = cases[0]
    for arm in ("full_history", "sqlite"):
        request = initial(case, "sarvam-105b", arm)
        altered = {
            **case,
            "id": "POISONED",
            "review": "POISONED",
            "split": "POISONED",
            "backend": {"secret_label": "POISONED"},
            "answer": "POISONED",
        }
        assert initial(altered, "sarvam-105b", arm) == request


def test_real_sqlite_update_idempotency_rejection_restart(tmp_path):
    cases, _ = build()
    case = next(c for c in cases if c["id"] == "action_update-00-en")
    db = tmp_path / "ordinary.sqlite"
    backend.create(db, case)
    args = {
        "appointment_id": "O-1000",
        "date": "2026-12-01",
        "idempotency_key": "REQ-700",
        "expected_version": 4,
    }
    assert backend.execute(db, "update_appointment", args)["status"] == "confirmed"
    assert backend.execute(db, "update_appointment", args)["status"] == "already_applied"
    assert backend.state(db)["version"] == 5
    assert len(backend.state(db)["operations"]) == 1
    stale = {**args, "idempotency_key": "different"}
    assert backend.execute(db, "update_appointment", stale)["status"] == "rejected"
    assert backend.execute(db, "update_appointment", {**args, "expected_version": True}) == {
        "error": "invalid_arguments"
    }
    assert backend.execute(db, "read_records", {})["records"] == case["records"]


def test_correct_words_do_not_hide_missing_backend_action(tmp_path):
    cases, gold = build()
    case = next(c for c in cases if c["id"] == "action_update-00-en")
    db = tmp_path / "ordinary.sqlite"
    backend.create(db, case)
    result = {
        "terminal": "answer",
        "content": json.dumps(
            {k: gold[case["id"]][k] for k in ("answer", "needs_clarification", "action_status")}
        ),
        "final_backend": backend.state(db),
        "tool_trace": [],
    }
    g = grade(case, gold[case["id"]], result)
    assert g["answer_correct"] and not g["task_success"]
    assert not g["backend_correct"] and not g["required_tool_receipt"]


def test_oracle_backend_actions_and_read_answers_all_cases(tmp_path):
    """Checks fixture/scorer consistency, never claimed as model results."""
    cases, gold = build()
    for case in cases:
        label = gold[case["id"]]
        db = tmp_path / (case["id"] + ".sqlite")
        backend.create(db, case)
        trace = []
        if label["action_status"] in ("confirmed", "rejected"):
            i = int(case["base_id"].rsplit("-", 1)[1])
            args = {
                "appointment_id": case["backend"]["appointment_id"],
                "date": f"2026-12-{i+1:02}",
                "expected_version": case["backend"]["version"],
                "idempotency_key": f"REQ-{i+700}",
            }
            response = backend.execute(db, "update_appointment", args)
            trace = [{"name": "update_appointment", "arguments": args, "response": response}]
        obj = {k: label[k] for k in ("answer", "needs_clarification", "action_status")}
        result = {
            "terminal": "answer",
            "content": json.dumps(obj),
            "tool_trace": trace,
            "final_backend": backend.state(db),
        }
        assert grade(case, label, result)["task_success"], case["id"]


def test_reasoning_diagnostic_keeps_inputs_tools_and_equal_cap():
    cases, _ = build()
    for case in cases[:30]:
        for arm in ("full_history", "sqlite"):
            a = profile_request(case, "sarvam-105b", arm, None)
            b = profile_request(case, "sarvam-105b", arm, "low")
            assert a["max_tokens"] == b["max_tokens"] == 8192
            assert {k: v for k, v in a.items() if k != "reasoning_effort"} == {
                k: v for k, v in b.items() if k != "reasoning_effort"
            }


def test_concurrent_budget_reservations_cannot_overcommit(tmp_path):
    meter = Meter(tmp_path)
    req = payload("sarvam-105b", [])

    def reserve(i):
        try:
            meter.reserve(f"j{i}", "setup", req)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(reserve, range(140)))
    s = meter.summary()["setup"]
    assert 0 < sum(outcomes) < 140
    assert s["calls"] == sum(outcomes)
    assert s["charged_with_headroom_inr"] <= 300


def test_continuation_excludes_successes_and_errors(tmp_path):
    package = {"jobs": [{"id": "ok"}, {"id": "failed"}, {"id": "new"}]}
    execution = {
        "outcomes": [
            {"id": "ok", "terminal": "answer"},
            {"id": "failed", "terminal": "infrastructure_error"},
        ]
    }
    for job in ("ok", "failed"):
        folder = tmp_path / "episodes" / job
        folder.mkdir(parents=True)
        (folder / "result.json").write_text("{}")
    assert continuation_jobs(package, execution, tmp_path) == [{"id": "new"}]
    (tmp_path / "episodes/failed/result.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        continuation_jobs(package, execution, tmp_path)
