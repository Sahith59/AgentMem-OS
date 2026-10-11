import copy
import json

import pytest

from benchmarks.sarvam_native_controls import approval_backend as backend
from benchmarks.sarvam_native_controls import integrated as engine
from benchmarks.sarvam_native_controls.integrated_audit import grade
from benchmarks.sarvam_native_controls.integrated_cases import build
from benchmarks.sarvam_native_study.transport import digest


def response(content="Done", calls=None, finish=None):
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content, "tool_calls": calls or []},
                "finish_reason": finish or ("tool_calls" if calls else "stop"),
            }
        ]
    }


def tc(args, name="update_appointment", identifier="call-1"):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def spec_for(tmp_path, case, enforce=True):
    db = tmp_path / "initial.sqlite"
    seeds = engine.initialize(db, case)
    return {
        "id": "test",
        "case_id": case["id"],
        "model": "sarvam-105b",
        "enforce": enforce,
        "initial_sha256": digest(engine.initial_request(db, case, "sarvam-105b", seeds)),
    }


def fake_dispatch(responses, requests):
    queue = iter(responses)

    def dispatch(job, request):
        requests.append(copy.deepcopy(request))
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return item

    return dispatch


def final(date, status):
    return response(
        json.dumps({"answer": date, "needs_clarification": False, "action_status": status})
    )


@pytest.mark.parametrize(
    "family,status", [("approved", "confirmed"), ("duplicate", "already_applied")]
)
def test_real_write_and_duplicate_receipts(tmp_path, family, status):
    cases, labels = build()
    case = next(c for c in cases if c["family"] == family)
    requests = []
    out = engine.episode(
        tmp_path,
        spec_for(tmp_path, case),
        case,
        fake_dispatch(
            [
                response("Claim is preserved", [tc(case["requested_change"])]),
                final(case["requested_change"]["date"], status),
            ],
            requests,
        ),
    )
    assert grade(case, labels[case["id"]], out)["task_success"]
    assert len(out["final_backend"]["operations"]) == 1
    assert "Claim is preserved" in requests[-1]["messages"][1]["content"]
    assert "tools" not in requests[-1] and "tool_choice" not in requests[-1]
    assert requests[-1]["reasoning_effort"] == "low"


@pytest.mark.parametrize("enforce", [False, True])
def test_pending_attempt_has_real_control_contrast(tmp_path, enforce):
    cases, labels = build()
    case = cases[1]
    expected_date = (
        case["fixture"]["appointments"][0]["date"] if enforce else case["requested_change"]["date"]
    )
    requests = []
    out = engine.episode(
        tmp_path,
        spec_for(tmp_path, case, enforce),
        case,
        fake_dispatch(
            [
                response(calls=[tc(case["requested_change"])]),
                final(expected_date, "rejected" if enforce else "confirmed"),
            ],
            requests,
        ),
    )
    result = grade(case, labels[case["id"]], out)
    assert result["correct_report"] and result["forbidden_update_attempt"]
    assert result["prohibited_state_change"] is not enforce
    assert not result[
        "false_confirmation"
    ]  # Unsafe actual writes can still be reported truthfully.
    serialized = json.dumps(requests[0])
    assert "enforce" not in serialized and "desired_date" not in serialized
    assert "forbid_new_write" not in serialized


@pytest.mark.parametrize("index", [1, 2, 3, 4, 5])
def test_no_tool_refusal_duplicate_and_read_finalization(tmp_path, index):
    cases, labels = build()
    case = cases[index]
    status = "already_applied" if index == 4 else "not_requested" if index == 5 else "not_performed"
    requests = []
    out = engine.episode(
        tmp_path,
        spec_for(tmp_path, case),
        case,
        fake_dispatch(
            [
                response("No new write performed."),
                final(labels[case["id"]]["desired_date"], status),
            ],
            requests,
        ),
    )
    assert grade(case, labels[case["id"]], out)["task_success"]
    assert not out["tool_trace"]


def test_read_tool_scoped_and_no_update_tool(tmp_path):
    case = build()[0][-1]
    requests = []
    out = engine.episode(
        tmp_path,
        spec_for(tmp_path, case),
        case,
        fake_dispatch(
            [
                response(calls=[tc({"appointment_id": case["appointment_id"]}, "read_current")]),
                response(),
                final(case["fixture"]["appointments"][0]["date"], "not_requested"),
            ],
            requests,
        ),
    )
    assert out["terminal"] == "answer" and len(requests) == 3
    assert [t["function"]["name"] for t in requests[0]["tools"]] == ["read_current"]
    assert out["initial_backend"] == out["final_backend"]
    assert engine.tool_reply(
        tmp_path / "initial.sqlite",
        "read_current",
        {"appointment_id": "other"},
        case,
        True,
        {"read_current"},
    ) == {"status": "rejected", "reason": "not_found_or_forbidden"}


@pytest.mark.parametrize("kind", ["truncated", "duplicate", "malformed"])
def test_invalid_batch_cannot_partially_write(tmp_path, kind):
    case = build()[0][0]
    calls = [tc(case["requested_change"])]
    finish = "length" if kind == "truncated" else "tool_calls"
    if kind == "duplicate":
        calls.append(copy.deepcopy(calls[0]))
    if kind == "malformed":
        calls.append({"id": "bad", "type": "function"})
    requests = []
    out = engine.episode(
        tmp_path,
        spec_for(tmp_path, case),
        case,
        fake_dispatch([response(calls=calls, finish=finish)], requests),
    )
    assert out["terminal"] == "invalid_action_response"
    assert len(requests) == 1 and out["initial_backend"] == out["final_backend"]


def test_unknown_tool_and_bad_json_are_recorded_without_writes(tmp_path):
    case = build()[0][0]
    invalid = tc(case["requested_change"])
    invalid["function"]["arguments"] = "{broken"
    requests = []
    out = engine.episode(
        tmp_path,
        spec_for(tmp_path, case),
        case,
        fake_dispatch(
            [
                response(calls=[tc({}, "invented", "unknown"), invalid]),
                final(case["fixture"]["appointments"][0]["date"], "rejected"),
            ],
            requests,
        ),
    )
    assert [t["response"]["reason"] for t in out["tool_trace"]] == [
        "tool_not_available",
        "invalid_arguments",
    ]
    assert out["initial_backend"] == out["final_backend"]


def test_provider_error_no_final_or_retry(tmp_path):
    case = build()[0][0]
    requests = []
    out = engine.episode(
        tmp_path, spec_for(tmp_path, case), case, fake_dispatch([RuntimeError("failure")], requests)
    )
    assert out["terminal"] == "infrastructure_error" and len(requests) == 1


def test_atomic_backend_failure_no_final_or_later_tool(tmp_path, monkeypatch):
    case = build()[0][0]
    spec = spec_for(tmp_path, case)
    original = engine.initialize

    def inject_failure(path, supplied):
        seeds = original(path, supplied)
        with backend.connect(path) as db:
            db.execute(
                (
                    "CREATE TRIGGER fail_receipt BEFORE INSERT ON operations BEGIN "
                    "SELECT RAISE(ABORT, 'receipt failure'); END"
                )
            )
        return seeds

    monkeypatch.setattr(engine, "initialize", inject_failure)
    requests = []
    out = engine.episode(
        tmp_path,
        spec,
        case,
        fake_dispatch(
            [response(calls=[tc(case["requested_change"]), tc({}, "read_current", "later")])],
            requests,
        ),
    )
    assert out["terminal"] == "backend_error" and len(requests) == 1
    assert len(out["tool_trace"]) == 1 and out["tool_trace"][0]["response"]["status"] == "error"
    assert out["initial_backend"] == out["final_backend"]


def test_final_tool_request_cannot_execute(tmp_path):
    case = build()[0][1]
    requests = []
    out = engine.episode(
        tmp_path,
        spec_for(tmp_path, case),
        case,
        fake_dispatch([response(), response(calls=[tc(case["requested_change"])])], requests),
    )
    assert out["terminal"] == "refused_final_tool_request"
    assert not out["tool_trace"] and out["initial_backend"] == out["final_backend"]


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "tool", "tool_call_id": "orphan", "content": "{}"}],
        [{"role": "assistant", "tool_calls": [tc({})]}],
        [
            {"role": "assistant", "tool_calls": [tc({})]},
            {"role": "assistant", "content": "premature"},
        ],
    ],
)
def test_finalizer_rejects_incomplete_history(messages):
    with pytest.raises(ValueError):
        engine.final_request({"model": "sarvam-105b", "messages": messages, "tools": []})


def test_frozen_check_precedes_key_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        engine, "read_key", lambda: pytest.fail("Read credential before checking frozen files")
    )
    with pytest.raises(FileNotFoundError):
        engine.run(tmp_path)


def test_complete_offline_replay_and_tamper_detection(tmp_path, monkeypatch):
    import socket

    from benchmarks.sarvam_native_controls.integrated_audit import audit
    from benchmarks.sarvam_native_controls.integrated_cases import write
    from benchmarks.sarvam_native_study.transport import ENDPOINT, save

    # These are explicit synthetic test responses, never research/API evidence.
    monkeypatch.setattr(socket, "socket", socket.socket)
    monkeypatch.setattr(socket, "create_connection", socket.create_connection)
    write(tmp_path)
    save(
        tmp_path / "budget.json",
        {
            "phase_cap_microinr": 220_000000,
            "prior_retained_microinr": 0,
            "prior_known_microinr": 0,
            "prior_calls": 0,
            "campaign_cap_microinr": 6000_000000,
            "stages": {"setup": {"cap_microinr": 220_000000, "calls": 96}},
        },
    )
    save(tmp_path / "pacing-control.json", {"rpm": 90})
    (tmp_path / "PROTOCOL.md").write_text("Synthetic offline test only")
    engine.prepare(tmp_path)
    package, cases = engine.check(tmp_path)
    save(tmp_path / "started.json", {"manifest_sha256": digest(package)})
    meter = engine.ControlMeter(tmp_path)
    outcomes = []
    for spec in package["jobs"]:
        case = cases[spec["case_id"]]
        new = case["family"] == "approved"
        already = case["family"] == "duplicate"
        date = (
            case["requested_change"]["date"]
            if new or already
            else case["fixture"]["appointments"][0]["date"]
        )
        status = (
            "confirmed"
            if new
            else "already_applied"
            if already
            else "not_requested"
            if case["intent"] == "read"
            else "not_performed"
        )
        queue = iter(
            [
                response(calls=[tc(case["requested_change"])]) if new else response(),
                final(date, status),
            ]
        )

        def dispatch(job, request):
            body = {
                **next(queue),
                "id": job,
                "model": request["model"],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            save(
                meter.root / (job + ".request.json"),
                {"endpoint": ENDPOINT, "request": request, "sha256": digest(request)},
            )
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
            save(meter.root / (job + ".dispatch.json"), {"synthetic_test": True})
            meter.reserve(job, "setup", request)
            assert meter.settle(job, body, request["model"], 8192)
            return body

        result = engine.episode(tmp_path, spec, case, dispatch)
        outcomes.append({"id": result["id"], "terminal": result["terminal"]})
    save(tmp_path / "execution.json", {"outcomes": outcomes, "meter": meter.summary()})
    verified = audit(tmp_path)
    assert verified["calls"] == 48 and verified["gate_pass"]
    assert all(r["task_success"] for r in verified["rows"])
    # An altered request must not receive a replay certificate.
    path = next((tmp_path / "calls").glob("*.request.json"))
    altered = json.loads(path.read_text())
    altered["request"]["temperature"] = 1
    path.write_text(json.dumps(altered))
    with pytest.raises(AssertionError):
        audit(tmp_path)
