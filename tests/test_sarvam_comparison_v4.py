"""Network-free validation of the narrow v4 provider compatibility profile."""

import copy
import socket

import pytest

from benchmarks.sarvam_comparison_v3 import contract as old
from benchmarks.sarvam_comparison_v3 import runner as old_runner
from benchmarks.sarvam_comparison_v4 import contract as c
from benchmarks.sarvam_comparison_v4 import runner as r
from benchmarks.sarvam_comparison_v4.score import evaluate


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(r, "SarvamProvider", forbidden)


def approval(package, directory):
    return {
        "approved": True,
        "mode": "offline-test",
        "package_sha256": c.sha(package),
        "maximum_attempts": 72,
        "budget_ninr": r.RESERVATION * 72,
        "output_directory": str(directory.resolve()),
        "accepted_review_level": package["review_level"],
        "authorization_text": "OFFLINE SIMULATION ONLY",
    }


def test_only_provider_unique_items_changes():
    parent = old_runner.build_package()
    package = r.build_package()
    assert package["parent_package_sha256"] == c.sha(parent)
    for spec, previous in zip(package["jobs"], parent["jobs"], strict=True):
        expected = copy.deepcopy(previous)
        if spec["stage"] == "extract":
            del expected["request"]["response_format"]["json_schema"]["schema"]["properties"][
                "facts"
            ]["items"]["properties"]["source_ids"]["uniqueItems"]
        assert spec == expected
    assert package["settings"] == parent["settings"]
    assert package["fixtures_sha256"] == parent["fixtures_sha256"]
    assert c.admit_facts is old.admit_facts
    assert c.source_packet is old.source_packet
    for name, digest in parent["code_sha256"].items():
        assert package["code_sha256"][name] == digest


@pytest.mark.parametrize("ids", [["s01", "s01"], ["foreign"], [], "s01"])
def test_local_validation_still_rejects_bad_citations(ids):
    case = c.runtime_cases()[0]
    text = c.canonical(
        {
            "facts": [
                {
                    "attribute": "pickup_counter",
                    "value": "K7",
                    "status": "current",
                    "condition": None,
                    "valid_from": None,
                    "valid_until": None,
                    "source_ids": ids,
                }
            ]
        }
    )
    admission = c.admit_facts(text, case)
    assert admission["status"] == "rejected" and admission["facts"] == []
    facts, originals = c.source_packet(case, admission)
    assert c.request("sqlite", case, facts, originals) == c.request("full_history", case)


class FakeProvider:
    def __init__(self, directory, fail=False):
        self.directory, self.fail, self.calls = directory, fail, 0

    def __call__(self, req):
        self.calls += 1
        state = c.loads((self.directory / "checkpoint.json").read_text())
        assert state["reserved_ninr"] == self.calls * r.RESERVATION
        assert list(state["jobs"].values())[-1]["dispatch_started"]
        if self.fail and self.calls == 2:
            return {"status_code": 500, "body_text": "{}", "elapsed_seconds": 0.001}
        data = c.loads(req["messages"][1]["content"])
        value = (
            {"facts": []}
            if "fields" not in data
            else {
                "needs_clarification": True,
                "fields": [{"name": n, "value": None, "source_ids": []} for n in data["fields"]],
            }
        )
        return {
            "status_code": 200,
            "elapsed_seconds": 0.001,
            "body_text": c.canonical(
                {
                    "model": c.MODEL,
                    "id": f"fake-{self.calls}",
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": c.canonical(value)}}
                    ],
                }
            ),
        }


def test_complete_persistent_control_and_no_duplicate_dispatch(tmp_path):
    package = r.build_package()
    a = approval(package, tmp_path)
    provider = FakeProvider(tmp_path)
    result = r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    assert result["resolved"] == result["dispatch_started"] == provider.calls == 72
    assert len(list(tmp_path.glob("*.sqlite"))) == 24
    score = evaluate(package, tmp_path)
    assert score["status"] == "SIMULATED_TEST_ONLY"
    assert score["paired"]["comparable"] == 24
    assert all(
        row["semantic_review"]["supported_meaning"] == "UNREVIEWED"
        for row in score["extraction_review"]
    )
    r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    assert provider.calls == 72


def test_unknown_usage_stops_and_forbids_relaunch(tmp_path):
    package = r.build_package()
    a = approval(package, tmp_path)
    provider = FakeProvider(tmp_path, fail=True)
    with pytest.raises(RuntimeError, match="Run stopped"):
        r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    score = evaluate(package, tmp_path)
    assert score["status"] == "SIMULATED_TEST_ONLY"
    assert score["execution"]["resolved"] == 1
    assert score["execution"]["attempts_with_unknown_usage"] == 1
    assert score["execution"]["reserved_ninr"] == 2 * r.RESERVATION
    assert all(s["accuracy_over_intended"] is None for s in score["arms"].values())
    with pytest.raises(ValueError, match="Unresolved"):
        r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    assert provider.calls == 2


def test_changed_package_and_increased_approval_fail_before_dispatch(tmp_path):
    package = r.build_package()
    bad = copy.deepcopy(package)
    bad["jobs"][0]["request"]["temperature"] = 1
    with pytest.raises(ValueError, match="Package differs"):
        r.preflight(bad)
    a = approval(package, tmp_path)
    a["maximum_attempts"] += 1
    with pytest.raises(ValueError, match="Approval"):
        r.run(package, tmp_path, a, mode="offline-test", provider=FakeProvider(tmp_path))
    assert not list(tmp_path.iterdir())
