import copy
import json
from unittest.mock import patch

import pytest

from benchmarks.sarvam_native_controls.cases import build
from benchmarks.sarvam_native_controls.engine import grade
from benchmarks.sarvam_native_controls.separated import (
    final_request,
    initial,
    paired_episode,
    prepare,
    run,
)
from benchmarks.sarvam_native_controls.separated_cases import probes
from benchmarks.sarvam_native_study import backend
from benchmarks.sarvam_native_study.transport import digest, save


def spec(case):
    return {
        "id": case["id"],
        "case_id": case["id"],
        "model": "sarvam-105b",
        "initial_sha256": digest(initial(case, "sarvam-105b")),
        "variant_order": ["control", "schema"],
    }


def response(content=None, tools=None, finish=None):
    return {
        "choices": [
            {
                "finish_reason": finish or ("tool_calls" if tools else "stop"),
                "message": {"content": content, "tool_calls": tools},
            }
        ]
    }


def tool(case):
    fields = case["question"].rsplit(" appointment_id=", 1)[1].rstrip(".").split("; ")
    args = {
        "appointment_id": fields[0],
        "date": fields[1].split("=")[1],
        "idempotency_key": fields[2].split("=")[1],
        "expected_version": case["backend"]["version"],
    }
    return {
        "id": "mock-call",
        "type": "function",
        "function": {"name": "update_appointment", "arguments": json.dumps(args)},
    }


def test_new_inputs_do_not_overlap_and_hidden_labels_cannot_change_prompt():
    cases, _ = probes()
    broad, _ = build()

    def project(c):
        return digest({"question": c["question"], "records": c["records"]})

    assert len({project(c) for c in cases}) == 7
    assert not {project(c) for c in cases} & {project(c) for c in broad}
    for case in cases:
        poisoned = {**case, "backend": {"secret": "POISON"}, "answer": "POISON"}
        assert initial(case, "sarvam-105b") == initial(poisoned, "sarvam-105b")


def test_both_final_requests_have_same_receipts_and_only_format_differs():
    case = probes()[0][0]
    original = initial(case, "sarvam-105b")
    assert "tools" in original and "response_format" not in original
    prefix = {
        "messages": copy.deepcopy(original["messages"])
        + [
            {"role": "assistant", "content": None, "tool_calls": [tool(case)]},
            {"role": "tool", "tool_call_id": "mock-call", "content": '{"status":"confirmed"}'},
        ]
    }
    before = copy.deepcopy(prefix)
    a = final_request(prefix, "sarvam-105b", "control")
    b = final_request(prefix, "sarvam-105b", "schema")
    assert a.pop("response_format") != b.pop("response_format")
    assert a == b
    assert "tools" not in a and "tool_choice" not in a
    assert a["messages"][:-1] == prefix["messages"]
    a["messages"][0]["content"] = "POISON"
    assert prefix == before


def test_real_database_and_shared_prefix_all_seven_probe_contracts(tmp_path):
    """Simulated oracle responses validate harness only, not native model quality."""
    cases, gold = probes()
    for case in cases:
        label, seen = gold[case["id"]], []

        def dispatch(job, request):
            seen.append(job)
            if "-act" in job:
                assert "response_format" not in request
                if label["receipt_status"]:
                    return response(tools=[tool(case)])
                return response(content="No new update is needed.")
            assert "tools" not in request
            return response(content=json.dumps(label["response"]))

        paired_episode(tmp_path, spec(case), case, dispatch)
        folder = tmp_path / "episodes" / case["id"]
        prefix = json.loads((folder / "prefix.json").read_text())
        assert prefix["final_backend"] == label["final_backend"]
        for variant in ("control", "schema"):
            result = json.loads((folder / (variant + ".json")).read_text())
            assert grade(case, label, result)["task_success"]
            assert result["shared_calls"] == prefix["calls"]
        assert len(seen) == (3 if case["family"].startswith("action_") else 2)


def test_final_tool_request_never_executes(tmp_path):
    case = probes()[0][0]
    seen = []

    def dispatch(job, request):
        seen.append(job)
        tc = tool(case)
        if "-final" in job:
            assert "tools" not in request
            args = json.loads(tc["function"]["arguments"])
            args.update(
                idempotency_key="UNAUTHORIZED-SECOND", expected_version=4, date="2026-12-31"
            )
            tc["function"]["arguments"] = json.dumps(args)
        return response(tools=[tc])

    paired_episode(tmp_path, spec(case), case, dispatch)
    folder = tmp_path / "episodes" / case["id"]
    state = backend.state(folder / "backend.sqlite")
    assert state["version"] == 4 and len(state["operations"]) == 1
    assert state["date"] == "2026-12-21"
    for variant in ("control", "schema"):
        assert (
            json.loads((folder / (variant + ".json")).read_text())["terminal"]
            == "final_tool_request"
        )
    assert len(seen) == 3


def test_incomplete_or_failed_action_never_gets_final_calls_or_retry(tmp_path):
    case = probes()[0][0]
    for kind in ("provider_error", "truncated_tool"):
        folder, seen = tmp_path / kind, []

        def dispatch(job, request):
            seen.append(job)
            if kind == "provider_error":
                raise RuntimeError("synthetic provider failure")
            return response(tools=[tool(case)], finish="length")

        paired_episode(folder, spec(case), case, dispatch)
        assert len(seen) == 1
        stored = folder / "episodes" / case["id"]
        assert backend.state(stored / "backend.sqlite") == case["backend"]
        for variant in ("control", "schema"):
            assert (
                json.loads((stored / (variant + ".json")).read_text())["terminal"]
                == "prefix_failure"
            )
        with pytest.raises(FileExistsError):
            paired_episode(folder, spec(case), case, dispatch)


def test_failed_gate_blocks_before_key_read(tmp_path):
    cases = probes()[0]
    save(tmp_path / "comparison-runtime.json", cases)
    save(tmp_path / "comparison-review.json", {"runtime_sha256": digest(cases)})
    save(tmp_path / "budget.json", {})
    (tmp_path / "PROTOCOL.md").write_text("Test only; no inference.")
    prepare(tmp_path, "comparison")
    (tmp_path / "probe").mkdir()
    save(tmp_path / "probe/gate.json", {"pass": False})
    with patch(
        "benchmarks.sarvam_native_controls.separated.read_key",
        side_effect=AssertionError("Must not read key"),
    ):
        with pytest.raises(ValueError, match="gate failed"):
            run(tmp_path, "comparison")
    assert not (tmp_path / "comparison/started.json").exists()
