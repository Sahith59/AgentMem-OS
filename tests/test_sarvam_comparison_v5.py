"""Offline contract transport and accounting checks; no live model score."""

import socket

import pytest

from benchmarks.sarvam_comparison_v4 import contract as old
from benchmarks.sarvam_comparison_v4 import runner as old_runner
from benchmarks.sarvam_comparison_v5 import contract as c
from benchmarks.sarvam_comparison_v5 import runner as r
from benchmarks.sarvam_comparison_v5.score import evaluate
from tests.test_sarvam_comparison_v4 import FakeProvider, approval


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(r, "SarvamProvider", forbidden)


def test_only_extraction_contract_transport_changes():
    parent = old_runner.build_package()
    package = r.build_package()
    assert package["parent_package_sha256"] == c.sha(parent)
    assert package["fixtures_sha256"] == parent["fixtures_sha256"]
    assert package["settings"] == parent["settings"]
    assert c.admit_facts is old.admit_facts
    for case in c.runtime_cases():
        before = old.request("extract", case)
        after = c.request("extract", case)
        schema = before["response_format"]["json_schema"]["schema"]
        before["response_format"] = {"type": "json_object"}
        before["messages"][0]["content"] += (
            "\nOutput must satisfy this JSON Schema, which the application validates locally:\n"
            + c.canonical(schema)
        )
        assert before == after
        for stage in ("full_history", "sqlite"):
            assert c.request(stage, case, [], c.eligible(case)) == old.request(
                stage, case, [], c.eligible(case)
            )


def test_complete_simulated_run_and_replay_without_new_calls(tmp_path):
    package = r.build_package()
    a = approval(package, tmp_path)
    provider = FakeProvider(tmp_path)
    result = r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    assert result["resolved"] == 72
    assert len(list(tmp_path.glob("*.sqlite"))) == 24
    score = evaluate(package, tmp_path)
    assert score["status"] == "SIMULATED_TEST_ONLY"
    assert score["paired"]["comparable"] == 24
    r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    assert provider.calls == 72


def test_truncation_still_stops_without_retry(tmp_path):
    package = r.build_package()
    a = approval(package, tmp_path)
    fake = FakeProvider(tmp_path)

    def provider(req):
        result = fake(req)
        if fake.calls == 2:
            body = c.loads(result["body_text"])
            body["choices"][0]["finish_reason"] = "length"
            result["body_text"] = c.canonical(body)
        return result

    with pytest.raises(RuntimeError, match="Run stopped"):
        r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    score = evaluate(package, tmp_path)
    assert score["execution"]["resolved"] == 1
    assert score["execution"]["attempts_with_unknown_usage"] == 0
    assert score["arms"]["sqlite"]["accuracy_over_intended"] is None
    with pytest.raises(ValueError, match="Unresolved"):
        r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    assert fake.calls == 2
