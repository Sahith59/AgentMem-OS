import copy
import json
import socket
from collections import Counter

import pytest

from benchmarks import sarvam_ordinary_audit as audit_module
from benchmarks import sarvam_ordinary_study as study
from benchmarks import sarvam_ordinary_v2 as engine
from benchmarks.sarvam_challenge_cases import build, setup, write
from benchmarks.sarvam_native_controls.integrated import initialize, read
from benchmarks.sarvam_native_study.transport import ENDPOINT, digest, save


def answer(date, status="confirmed", clarification=False):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "answer": date,
                            "needs_clarification": clarification,
                            "action_status": status,
                        }
                    ),
                },
            }
        ]
    }


def tool(args, finish="tool_calls"):
    return {
        "choices": [
            {
                "finish_reason": finish,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "actual-call",
                            "type": "function",
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


def spec(tmp_path, case):
    path = tmp_path / "initial.sqlite"
    seeds = initialize(path, case)
    return {
        "id": "test",
        "case_id": case["id"],
        "model": "sarvam-105b",
        "enforce": True,
        "initial_sha256": digest(engine.initial_request(path, case, "sarvam-105b", seeds)),
    }


def simulate(tmp_path, case, replies):
    calls = []
    queue = iter(replies)

    def dispatch(job, request):
        calls.append(copy.deepcopy(request))
        result = next(queue)
        if isinstance(result, Exception):
            raise result
        return result

    return engine.episode(tmp_path, spec(tmp_path, case), case, dispatch), calls


def test_status_is_applied_in_execution_but_wrong_date_is_preserved(tmp_path):
    cases, labels = setup()
    case = cases[1]
    result, requests = simulate(tmp_path, case, [answer("WRONG"), answer("WRONG")])
    assert result["final"]["parsed"]["action_status"] == "confirmed"
    assert result["application"]["application_output"] == {
        "answer": "WRONG",
        "needs_clarification": False,
        "action_status": "not_performed",
    }
    assert not audit_module.grade(case, labels[case["id"]], result, "setup")["safe_task_success"]
    assert len(requests) == 2 and "tools" not in requests[-1]


@pytest.mark.parametrize("index", [0, 4, 5])
def test_approved_duplicate_and_read_paths(tmp_path, index):
    cases, labels = setup()
    case = cases[index]
    first = tool(case["requested_change"]) if index in (0, 4) else answer("no change")
    result, _ = simulate(
        tmp_path, case, [first, answer(labels[case["id"]]["desired_date"], "not_requested")]
    )
    measured = audit_module.grade(case, labels[case["id"]], result, "setup")
    assert measured["safe_task_success"]
    assert result["application"]["raw_content"] == result["final"]["content"]


@pytest.mark.parametrize("failure", ["truncation", "provider", "final_tool"])
def test_failures_cannot_become_projected_success(tmp_path, failure):
    case = setup()[0][0]
    replies = (
        [tool(case["requested_change"], "length")]
        if failure == "truncation"
        else [RuntimeError("recorded error")]
        if failure == "provider"
        else [answer("no change"), tool(case["requested_change"])]
    )
    result, _ = simulate(tmp_path, case, replies)
    assert result["application"] is None
    assert result["initial_backend"] == result["final_backend"]


def test_source_preserving_requested_date_question(tmp_path):
    cases, labels = build()
    case = next(c for c in cases if c["id"] == "source-04-te")
    expected = labels[case["id"]]["expected_answer"]["answer"]
    result, requests = simulate(tmp_path, case, [answer("no action"), answer(expected)])
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["source_messages"] == case["source_messages"]
    assert "expected_answer" not in json.dumps(requests)
    assert "expected_state" not in json.dumps(requests)
    evidence = json.loads(requests[-1]["messages"][1]["content"])
    assert evidence["original_messages"][1] == requests[0]["messages"][1]
    assert payload["application_evidence"]["current"]["date"] != expected
    assert audit_module.grade(case, labels[case["id"]], result, "comparison")["safe_task_success"]


def test_bank_translations_and_answer_contrasts():
    cases, labels = build()
    assert len(cases) == len(labels) == 72
    assert Counter(c["language"] for c in cases) == {"en": 24, "hi": 24, "te": 24}
    assert len({c["workflow"] for c in cases}) == 24
    assert len({c["family"] for c in cases}) == 8
    assert len({(c["question"], json.dumps(c["source_messages"])) for c in cases}) == 72
    for index in range(1, 25):
        group = [c for c in cases if c["workflow"] == f"source-{index:02}"]
        assert len({digest(c["fixture"]) for c in group}) == 1
        assert len({digest(labels[c["id"]]["expected_answer"]) for c in group}) == 1
        for c in group:
            assert labels[c["id"]]["runtime_sha256"] == digest(c)
    assert labels["source-07-en"]["expected_answer"]["answer"] == "2026-12-07"
    assert labels["source-08-en"]["expected_answer"]["answer"] == "2027-01-08"
    assert labels["source-09-en"]["expected_answer"] == {
        "answer": None,
        "needs_clarification": True,
    }
    assert labels["source-23-en"]["expected_answer"]["answer"] == "2027-01-23"
    assert all(not c["fixture"]["approvals"] for c in cases if int(c["workflow"][-2:]) <= 12)


def configure(root):
    write(root)
    save(
        root / "review.json",
        {
            **{
                f"{stage}_{kind}_sha256": digest(read(root / f"{stage}-{kind}.json"))
                for stage in ("setup", "comparison")
                for kind in ("runtime", "gold")
            },
            "assistant_source_review": "complete",
        },
    )
    save(
        root / "budget.json",
        {
            "phase_cap_microinr": 2700_000000,
            "prior_calls": 0,
            "prior_known_microinr": 0,
            "prior_retained_microinr": 0,
            "campaign_cap_microinr": 6000_000000,
            "stages": {
                "setup": {"calls": 48, "cap_microinr": 110_000000},
                "comparison": {"calls": 1152, "cap_microinr": 2590_000000},
            },
        },
    )
    save(root / "pacing-control.json", {"rpm": 90})
    (root / "PROTOCOL.md").write_text("Synthetic offline test only")
    (root / "CASE_REVIEW.md").write_text("Synthetic test fixture")


def test_full_setup_replay_gate_network_restore_and_manifest(tmp_path, monkeypatch):
    configure(tmp_path)
    with pytest.raises(FileNotFoundError):
        study.prepare(tmp_path, "comparison")
    assert not (tmp_path / "comparison").exists()
    study.prepare(tmp_path, "setup")
    package, cases = study.check(tmp_path, "setup")
    save(tmp_path / "setup/started.json", {"manifest_sha256": digest(package)})
    meter = study.ControlMeter(tmp_path)
    labels = read(tmp_path / "setup-gold.json")
    outcomes = []
    for job in package["jobs"]:
        case = cases[job["case_id"]]
        first = (
            tool(case["requested_change"]) if case["family"] == "approved" else answer("no action")
        )
        replies = iter([first, answer(labels[case["id"]]["desired_date"], "not_requested")])

        def dispatch(jid, request):
            body = {
                **next(replies),
                "id": jid,
                "model": request["model"],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
            save(
                meter.root / (jid + ".request.json"),
                {"endpoint": ENDPOINT, "request": request, "sha256": digest(request)},
            )
            save(
                meter.root / (jid + ".receipt.json"),
                {
                    "job": jid,
                    "request_sha256": digest(request),
                    "status": 200,
                    "body_text": json.dumps(body),
                    "elapsed_seconds": 0,
                },
            )
            save(meter.root / (jid + ".dispatch.json"), {"synthetic_test": True})
            meter.reserve(jid, "setup", request)
            assert meter.settle(jid, body, request["model"], 8192)
            return body

        result = engine.episode(tmp_path / "setup", job, case, dispatch)
        outcomes.append({"id": result["id"], "terminal": result["terminal"]})
    save(tmp_path / "setup/execution.json", {"outcomes": outcomes, "meter": meter.summary()})
    original_socket = socket.socket
    verification = audit_module.audit(tmp_path, "setup")
    assert verification["gate_pass"] and verification["episodes"] == 12
    assert socket.socket is original_socket
    save(tmp_path / "setup/verification.json", verification)
    study.prepare(tmp_path, "comparison")
    assert socket.socket is original_socket
    future, _ = study.check(tmp_path, "comparison")
    assert len(future["jobs"]) == 288 and future["maximum_calls"] == 1152
    assert not (tmp_path / "comparison/started.json").exists()
    monkeypatch.setattr(
        study, "read_key", lambda: pytest.fail("Credential read despite changed gate")
    )
    (tmp_path / "setup/verification.json").write_text("{}")
    with pytest.raises((AssertionError, ValueError, KeyError)):
        study.run(tmp_path, "comparison")
