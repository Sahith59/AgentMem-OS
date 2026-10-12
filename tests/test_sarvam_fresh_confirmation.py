import copy
import json
from collections import Counter

import pytest

from benchmarks import sarvam_ordinary_v2 as ordinary
from benchmarks.sarvam_fresh_confirmation import audit, cases, engine, study
from benchmarks.sarvam_native_controls.engine import ControlMeter
from benchmarks.sarvam_native_controls.integrated import initialize, read
from benchmarks.sarvam_native_study.transport import ENDPOINT, digest, save


@pytest.fixture(scope="module")
def bank():
    return cases.build()


def response(answer=None, status="not_requested", clarify=False):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"answer": answer, "needs_clarification": clarify, "action_status": status}
                    ),
                },
            }
        ]
    }


def action(args):
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "one",
                            "function": {
                                "name": "update_appointment",
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
            }
        ]
    }


def spec(root, case):
    seeds = initialize(root / "initial.sqlite", case)
    return {
        "id": "sample",
        "case_id": case["id"],
        "model": "sarvam-105b",
        "enforce": True,
        "initial_sha256": digest(
            ordinary.initial_request(root / "initial.sqlite", case, "sarvam-105b", seeds)
        ),
        "report_order": ["candidate", "control"],
    }


def prepare(root, bank):
    runtime, gold = bank
    save(root / "runtime.json", runtime)
    save(root / "gold.json", gold)
    save(
        root / "review.json",
        {
            "runtime_sha256": digest(runtime),
            "gold_sha256": digest(gold),
            "review_type": "offline test fixture; not human review",
        },
    )
    save(
        root / "budget.json",
        {
            "phase_cap_microinr": 2600_000000,
            "prior_retained_microinr": 244451544,
            "prior_known_microinr": 169754685,
            "prior_calls": 6080,
            "campaign_cap_microinr": 6000_000000,
            "stages": {"confirmation": {"calls": 1200, "cap_microinr": 2600_000000}},
        },
    )
    save(root / "pacing-control.json", {"rpm": 90})
    (root / "PROTOCOL.md").write_text("Offline fixture simulation, not measured accuracy.\n")
    return study.prepare(root)


def test_inventory_binding_and_real_desired_backend(bank):
    runtime, labels = bank
    assert len(runtime) == 120 and len({c["workflow"] for c in runtime}) == 40
    assert Counter(c["language"] for c in runtime) == {"en": 40, "hi": 40, "te": 40}
    assert len({digest(c["source_messages"]) for c in runtime}) == 120
    assert sum(label["required_new_write"] for label in labels.values()) == 24
    assert sum(label["expected_answer"]["needs_clarification"] for label in labels.values()) == 24
    for case in runtime:
        label = labels[case["id"]]
        assert label["runtime_sha256"] == digest(case)
        assert "expected_answer" not in json.dumps(case)
        assert label["required_new_write"] == (label["initial_state"] != label["expected_state"])


def test_actions_once_reporters_same_evidence_and_wrong_date_not_repaired(tmp_path, bank):
    case = next(c for c in bank[0] if c["family"] == "approved_change")
    requests = []

    def dispatch(job, request):
        requests.append((job, copy.deepcopy(request)))
        return (
            action(case["requested_change"])
            if job.endswith("-a0")
            else response("WRONG", "not_performed")
        )

    result = engine.episode(tmp_path, spec(tmp_path, case), case, dispatch)
    assert len(requests) == 3 and len(result["original"]["tool_trace"]) == 1
    assert requests[1][1]["messages"][1] == requests[2][1]["messages"][1]
    assert "tools" not in requests[1][1] and "tools" not in requests[2][1]
    for variant in ("control", "candidate"):
        row = audit.grade(case, bank[1][case["id"]], result, variant)
        assert row["desired_state"] and not row["factual_answer_correct"]
        assert (
            row["application_status_matches_execution"] and not row["raw_status_matches_execution"]
        )
        assert result["reports"][variant]["application"]["application_output"]["answer"] == "WRONG"


@pytest.mark.parametrize("failed", ["control", "candidate"])
def test_one_reporter_failure_does_not_skip_other_or_retry(tmp_path, bank, failed):
    case = bank[0][0]
    called = []

    def dispatch(job, request):
        called.append(job)
        if job.endswith("-" + failed):
            raise RuntimeError("simulated provider error")
        return response()

    result = engine.episode(tmp_path, spec(tmp_path, case), case, dispatch)
    assert len(called) == 3 and len(set(called)) == 3
    assert result["reports"][failed]["terminal"] == "infrastructure_error"
    other = "candidate" if failed == "control" else "control"
    assert result["reports"][other]["terminal"] == "answer"


def test_truthful_old_date_does_not_hide_missed_approved_action(tmp_path, bank):
    case = next(c for c in bank[0] if c["family"] == "approval_boundary")
    old = case["fixture"]["appointments"][0]["date"]
    result = engine.episode(
        tmp_path, spec(tmp_path, case), case, lambda *_: response(old, "not_performed")
    )
    row = audit.grade(case, bank[1][case["id"]], result, "candidate")
    assert row["factual_answer_correct"] and row["missed_required_write"]
    assert not row["safe_fact_and_state"]


def test_three_native_call_limit_and_no_report_after_invalid_history(tmp_path, bank):
    case = bank[0][0]
    count = []

    def dispatch(job, request):
        count.append(job)
        body = action({"appointment_id": case["appointment_id"]})
        tc = body["choices"][0]["message"]["tool_calls"][0]
        tc["id"], tc["function"]["name"] = str(len(count)), "read_current"
        return body

    result = engine.episode(tmp_path, spec(tmp_path, case), case, dispatch)
    assert len(count) == 3
    assert result["original"]["terminal"] == "action_call_limit"
    assert all(v["terminal"] == "not_reached" for v in result["reports"].values())


def test_prepare_freezes_matched_profiles_order_and_affordable_maximum(tmp_path, bank):
    manifest = prepare(tmp_path, bank)
    assert manifest["maximum_calls"] == 1200
    assert manifest["worst_case_retained_microinr"] == 2580480000
    for model in study.PROFILES:
        assert Counter(j["report_order"][0] for j in manifest["jobs"] if j["model"] == model) == {
            "candidate": 60,
            "control": 60,
        }
    assert study.check(tmp_path)[0] == manifest


@pytest.mark.parametrize(
    "name",
    [
        "runtime.json",
        "gold.json",
        "PROTOCOL.md",
        "budget.json",
        "source/benchmarks/sarvam_fresh_confirmation/engine.py",
    ],
)
def test_changed_inputs_or_code_refused(tmp_path, bank, name):
    prepare(tmp_path, bank)
    with (tmp_path / name).open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError):
        study.check(tmp_path)


def test_restart_refused_before_key_access(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "check", lambda _: ({}, {}))
    save(tmp_path / "started.json", {})
    monkeypatch.setattr(study, "read_key", lambda: pytest.fail("unexpected key read"))
    with pytest.raises(FileExistsError):
        study.run(tmp_path)


def test_full_offline_simulation_replay_and_tamper_detection(tmp_path, bank, monkeypatch):
    manifest = prepare(tmp_path, bank)
    runtime = {c["id"]: c for c in bank[0]}
    jobs = {j["id"]: j for j in manifest["jobs"]}

    def fake_call(meter, job, stage, request, key):
        assert key == "offline"
        prefix = job.split("-a")[0] if "-a" in job else job.split("-final")[0]
        case = runtime[jobs[prefix]["case_id"]]
        label = bank[1][case["id"]]
        body = (
            action(case["requested_change"])
            if job.endswith("-a0") and label["required_new_write"]
            else response(
                label["expected_answer"]["answer"],
                clarify=label["expected_answer"]["needs_clarification"],
            )
        )
        body.update(
            id=job,
            model=request["model"],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )
        save(
            meter.root / (job + ".request.json"),
            {"endpoint": ENDPOINT, "request": request, "sha256": digest(request)},
        )
        meter.reserve(job, stage, request)
        save(
            meter.root / (job + ".receipt.json"),
            {
                "job": job,
                "request_sha256": digest(request),
                "status": 200,
                "body_text": json.dumps(body),
                "elapsed_seconds": 0,
            },
        )
        meter.settle(job, body, request["model"], 8192)
        return body

    class FakePacer:
        def __init__(self, root):
            pass

        def wait(self):
            return {"offline_simulation": True}

    monkeypatch.setattr(study, "call", fake_call)
    monkeypatch.setattr(study, "Pacer", FakePacer)
    monkeypatch.setattr(study, "read_key", lambda: "offline")
    study.run(tmp_path)
    result = audit.audit(tmp_path)
    assert result["verified"] and result["complete"] and result["calls"] == 720
    assert all(
        g["factual_answer_correct"] == 120 and g["safe_fact_and_state"] == 120
        for g in result["groups"]
    )
    assert not result["mechanical_screen_pass"]  # A tie does not pass the frozen gain criterion.
    p = next((tmp_path / "calls").glob("*.request.json"))
    changed = read(p)
    changed["request"]["temperature"] = 1
    p.write_text(json.dumps(changed))
    with pytest.raises(AssertionError):
        audit.audit(tmp_path)


def test_unknown_usage_retains_full_reservation_and_no_extra_budget(tmp_path):
    save(
        tmp_path / "budget.json",
        {
            "phase_cap_microinr": 3072000,
            "prior_retained_microinr": 0,
            "campaign_cap_microinr": 3072000,
            "stages": {"confirmation": {"calls": 1, "cap_microinr": 3072000}},
        },
    )
    meter = ControlMeter(tmp_path)
    req = {"model": "sarvam-105b", "max_tokens": 8192}
    meter.reserve("one", "confirmation", req)
    assert not meter.settle("one", {}, req["model"], 8192)
    assert meter.summary()["confirmation"]["charged_with_headroom_inr"] == 3.072
    with pytest.raises(ValueError):
        meter.reserve("two", "confirmation", req)
