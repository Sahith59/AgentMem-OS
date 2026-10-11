"""Network-free contract, failure-policy and accounting tests; no model scores."""

import copy
import json
import socket
import sqlite3

import pytest

from benchmarks.sarvam_comparison_v2 import contract as previous
from benchmarks.sarvam_comparison_v3 import contract as c
from benchmarks.sarvam_comparison_v3 import runner as r
from benchmarks.sarvam_comparison_v3.score import evaluate
from benchmarks.sarvam_discovery.storage import read_packet


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(r, "SarvamProvider", forbidden)


def approval(package, directory):
    plan = r.preflight(package)
    return {
        "approved": True,
        "mode": "offline-test",
        "package_sha256": c.sha(package),
        "maximum_attempts": 72,
        "budget_ninr": plan["maximum_reservation_ninr"],
        "output_directory": str(directory.resolve()),
        "accepted_review_level": package["review_level"],
        "authorization_text": "OFFLINE SYNTHETIC TEST ONLY",
    }


def fact(**changes):
    return {
        "attribute": "pickup_instruction",
        "value": "Use pickup counter K7.",
        "status": "current",
        "condition": None,
        "valid_from": None,
        "valid_until": None,
        "source_ids": ["s01"],
        **changes,
    }


def receipt(text, index):
    return {
        "status_code": 200,
        "elapsed_seconds": 0.001,
        "request_id": f"fake-{index}",
        "body_text": c.canonical(
            {
                "model": c.MODEL,
                "id": f"fake-{index}",
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            }
        ),
    }


class FakeProvider:
    """Does not read gold; all responses are synthetic, mostly abstentions."""

    def __init__(self, directory, extract_text='{"facts":[]}', fault=None):
        self.directory, self.extract_text, self.fault = directory, extract_text, fault
        self.requests = []

    def __call__(self, req):
        state = c.loads((self.directory / "checkpoint.json").read_text())
        current = list(state["jobs"].values())[-1]
        assert current["status"] == "pending" and current["dispatch_started"]
        assert current["request"] == req and current["request_sha256"] == c.sha(req)
        assert state["reserved_ninr"] == len(state["jobs"]) * r.RESERVATION
        self.requests.append(copy.deepcopy(req))
        body = c.loads(req["messages"][1]["content"])
        if "fields" in body:
            text = c.canonical(
                {
                    "needs_clarification": True,
                    "fields": [
                        {"name": name, "value": None, "source_ids": []} for name in body["fields"]
                    ],
                }
            )
        else:
            text = self.extract_text
        result = receipt(text, len(self.requests))
        return self.fault(result, req, len(self.requests)) if self.fault else result


@pytest.fixture(scope="module")
def package():
    return r.build_package()


def test_requests_keep_answers_fixed_and_extraction_blind():
    for case in c.runtime_cases():
        for stage in ("full_history", "sqlite"):
            assert c.request(stage, case, [], c.eligible(case)) == previous.request(
                stage, case, [], c.eligible(case)
            )
        req = c.request("extract", case)
        body = c.loads(req["messages"][1]["content"])
        assert set(body) == {"as_of", "records"}
        assert all(set(v) == {"id", "role", "observed_at", "text"} for v in body["records"])
        changed = copy.deepcopy(case)
        changed.update(
            question="SECRET_GOLD", fields=["GOLD_FIELD"], family="GOLD_FAMILY", id="GOLD_ID"
        )
        assert req == c.request("extract", changed)
        assert req["model"] == "sarvam-105b" and req["reasoning_effort"] is None
        assert req["response_format"]["type"] == "json_schema"
        schema = req["response_format"]["json_schema"]
        assert schema["strict"] is True
        ids = schema["schema"]["properties"]["facts"]["items"]["properties"]["source_ids"]["items"][
            "enum"
        ]
        assert ids == [v["id"] for v in c.eligible(case)]


@pytest.mark.parametrize(
    "mutation",
    [
        {"valid_from": "2026-10-09T10:00:00+05:30"},
        {"valid_from": "2026-02-30"},
        {"valid_from": "20261009"},
        {"valid_from": "2026-11-01", "valid_until": "2026-10-01"},
        {"source_ids": []},
        {"source_ids": ["other-user"]},
        {"source_ids": ["s01", "s01"]},
        {"source_ids": "s01"},
        {"status": []},
        {"status": "invented"},
        {"condition": 123},
        {"attribute": " Observed At "},
        {"attribute": "tenant-id"},
        {"attribute": "user_id"},
        {"unexpected": "extra"},
        {"value": ""},
        {"value": "x" * 1001},
    ],
)
def test_invalid_fact_set_rejected_whole_without_mutation(mutation):
    case = c.runtime_cases()[0]
    obj = {"facts": [fact(), fact(**mutation)]}
    before = copy.deepcopy(obj)
    result = c.admit_facts(c.canonical(obj), case)
    assert result["status"] == "rejected" and result["facts"] == []
    assert obj == before
    facts, sources = c.source_packet(case, result)
    assert c.request("sqlite", case, facts, sources) == c.request("full_history", case)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"facts":[],"facts":[]}',
        '{"facts":NaN}',
        '{"facts":[null]}',
        '{"facts":{}}',
        "[]",
    ],
)
def test_malformed_extraction_rejected(text):
    assert c.admit_facts(text, c.runtime_cases()[0])["status"] == "rejected"


def test_saved_live_failure_rejected_and_sources_not_repaired():
    saved = (
        c.ROOT / "benchmarks/audits/sarvam-small-comparison-2026-10-11/live-run-001/diagnosis.json"
    )
    text = c.loads(saved.read_text())["final_content"]
    result = c.admit_facts(text, c.runtime_cases()[0])
    assert result["status"] == "rejected" and result["facts"] == []
    assert "2026-10-09T10:00:00+05:30" in text


def test_structural_acceptance_does_not_certify_semantics():
    case = c.runtime_cases()[0]
    # Deliberately false, but structurally valid: admission MUST NOT certify meaning.
    wrong = fact(value="Use pickup counter WRONG.", valid_from="2026-10-10")
    result = c.admit_facts(c.canonical({"facts": [wrong]}), case)
    assert result["status"] == "admitted_structure_only"
    assert result["semantic_review"] == "UNREVIEWED"
    assert result["facts"] == [wrong]


def test_conditions_future_dates_and_empty_facts_are_preserved():
    case = next(x for x in c.runtime_cases() if x["id"] == "D07-en")
    conditional = fact(
        attribute="callback",
        value="2026-10-18",
        status="conditional",
        condition="only if repair is approved",
        valid_from="2026-10-18",
    )
    future = fact(attribute="future_value", value="B4", valid_from="2026-10-20", source_ids=["s02"])
    obj = [conditional, future]
    assert c.admit_facts(c.canonical({"facts": obj}), case)["facts"] == obj
    empty = c.admit_facts('{"facts":[]}', case)
    assert empty["status"] == "admitted_structure_only" and empty["facts"] == []


def test_unauthorized_future_and_missing_identity_never_reach_extraction():
    case = copy.deepcopy(c.runtime_cases()[0])
    source = case["records"][0]
    case["records"] += [
        {**source, "id": "other", "customer": "OTHER", "text": "PRIVATE"},
        {**source, "id": "tenant", "tenant": "OTHER", "text": "PRIVATE"},
        {**source, "id": "future", "observed_at": "2099-01-01T00:00:00+00:00", "text": "PRIVATE"},
    ]
    assert "PRIVATE" not in c.canonical(c.request("extract", case))
    for forbidden in ("other", "tenant", "future"):
        assert (
            c.admit_facts(c.canonical({"facts": [fact(source_ids=[forbidden])]}), case)["status"]
            == "rejected"
        )
    case["authorized_customer"] = None
    assert c.loads(c.request("extract", case)["messages"][1]["content"])["records"] == []
    assert c.fact_schema(case)["properties"]["facts"]["maxItems"] == 0


@pytest.fixture(scope="module")
def rejected_run(package, tmp_path_factory):
    directory = tmp_path_factory.mktemp("rejected-v3")
    saved = (
        c.ROOT / "benchmarks/audits/sarvam-small-comparison-2026-10-11/live-run-001/diagnosis.json"
    )
    provider = FakeProvider(directory, c.loads(saved.read_text())["final_content"])
    summary = r.run(
        package, directory, approval(package, directory), mode="offline-test", provider=provider
    )
    return directory, summary, provider


def test_all_extractions_rejected_still_measures_both_controls(package, rejected_run):
    directory, summary, provider = rejected_run
    assert summary["resolved"] == len(provider.requests) == 72
    assert summary["rejected_extractions"] == 24 and summary["completed"] == 48
    assert summary["reserved_ninr"] == 72 * r.RESERVATION
    assert len(list(directory.glob("*.sqlite"))) == 24
    state = c.loads((directory / "checkpoint.json").read_text())
    for case in c.runtime_cases():
        for repeat in (1, 2):
            prefix = f"r{repeat}-{case['id']}-"
            history = state["jobs"][prefix + "full_history"]
            sql = state["jobs"][prefix + "sqlite"]
            assert sql["request"] == history["request"]
            assert read_packet(directory / sql["store_file"], case) == ([], c.eligible(case))
    score = evaluate(package, directory)
    assert score["status"] == "SIMULATED_TEST_ONLY" and score["paired"]["comparable"] == 24
    assert (
        score["arms"]["sqlite"]["by_evidence_path"]["originals_after_rejection"]["intended"] == 24
    )
    assert all(row["admission"]["status"] == "rejected" for row in score["extraction_review"])
    assert all(
        set(row["semantic_review"].values()) == {"UNREVIEWED"} for row in score["extraction_review"]
    )
    assert (
        score["metering"]["arms"]["sqlite"]["known_usage_ninr"]
        == 2 * score["metering"]["arms"]["full_history"]["known_usage_ninr"]
    )
    assert (
        sum(arm["known_usage_ninr"] for arm in score["metering"]["arms"].values())
        == summary["usage_ninr"]
    )
    assert (
        r.run(
            package, directory, approval(package, directory), mode="offline-test", provider=provider
        )
        == summary
    )
    assert len(provider.requests) == 72


def test_accepted_extractions_persist_and_do_not_certify_meaning(package, tmp_path):
    provider = FakeProvider(tmp_path, c.canonical({"facts": [fact()]}))
    result = r.run(
        package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
    )
    assert result["resolved"] == result["completed"] == 72
    score = evaluate(package, tmp_path)
    assert (
        score["arms"]["sqlite"]["by_evidence_path"]["originals_with_admitted_facts"]["intended"]
        == 24
    )
    assert all(
        row["admission"]["semantic_review"] == "UNREVIEWED" for row in score["extraction_review"]
    )


def test_invalid_answers_stay_in_intended_denominator(package, tmp_path):
    def invalid(out, req, index):
        if "fields" in c.loads(req["messages"][1]["content"]):
            return receipt('{"fields":[]}', index)
        return out

    provider = FakeProvider(tmp_path, fault=invalid)
    result = r.run(
        package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
    )
    assert result["resolved"] == 72 and result["invalid_answers"] == 48
    score = evaluate(package, tmp_path)
    for arm in score["arms"].values():
        assert arm["accuracy_among_valid"] is None
        assert arm["accuracy_over_intended"] == 0 and arm["intended"] == 24
        assert arm["status_counts"] == {"invalid_answer": 24}
    assert score["paired"]["comparable"] == 0


@pytest.mark.parametrize(
    "fault", ["http", "model", "usage", "truncated", "empty", "timeout", "interrupt"]
)
def test_infrastructure_failure_stops_and_never_resends(package, tmp_path, fault):
    def damage(out, req, index):
        if fault == "timeout":
            raise TimeoutError("SECRET_SENTINEL")
        if fault == "interrupt":
            raise KeyboardInterrupt()
        body = c.loads(out["body_text"])
        if fault == "http":
            out["status_code"] = 400
        if fault == "model":
            body["model"] = "UNAPPROVED"
        if fault == "usage":
            body["usage"]["prompt_tokens"] = True
        if fault == "truncated":
            body["choices"][0]["finish_reason"] = "length"
        if fault == "empty":
            body["choices"][0]["message"]["content"] = None
        out["body_text"] = c.canonical(body)
        return out

    provider = FakeProvider(tmp_path, fault=damage)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    state = c.loads((tmp_path / "checkpoint.json").read_text())
    summary = r.validate_state(package, state, tmp_path)
    assert summary["attempted"] == summary["dispatch_started"] == 1
    assert summary["reserved_ninr"] == r.RESERVATION
    assert summary["resolved"] == 0
    if fault in {"truncated", "empty"}:
        assert summary["usage_ninr"] > 0
    else:
        assert summary["attempts_with_unknown_usage"] == 1
    assert "SECRET_SENTINEL" not in c.canonical(state)
    before = (tmp_path / "checkpoint.json").read_bytes()
    with pytest.raises(ValueError, match="automatic retry"):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert len(provider.requests) == 1 and (tmp_path / "checkpoint.json").read_bytes() == before
    score = evaluate(package, tmp_path)
    assert all(arm["accuracy_over_intended"] is None for arm in score["arms"].values())


def test_duplicate_receipt_stops_and_is_scorable(package, tmp_path):
    def duplicate(out, req, index):
        body = c.loads(out["body_text"])
        body["id"] = "same-receipt"
        out["body_text"] = c.canonical(body)
        return out

    provider = FakeProvider(tmp_path, fault=duplicate)
    with pytest.raises(RuntimeError):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert len(provider.requests) == 2
    score = evaluate(package, tmp_path)
    assert score["execution"]["resolved"] == 1
    assert score["execution"]["attempted"] == 2


def test_persistence_failure_stops_before_dispatch(package, tmp_path, monkeypatch):
    def broken(*args):
        raise OSError("cannot persist")

    monkeypatch.setattr(r, "write_store", broken)
    provider = FakeProvider(tmp_path)
    with pytest.raises(RuntimeError):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    state = c.loads((tmp_path / "checkpoint.json").read_text())
    result = r.validate_state(package, state, tmp_path)
    assert result["attempted"] == 3 and result["dispatch_started"] == 2
    assert result["attempts_with_unknown_usage"] == 0
    meter = evaluate(package, tmp_path)["metering"]["stages"]["sqlite"]
    assert meter["reserved_attempt_slots"] == 1
    assert meter["dispatch_started_calls"] == meter["calls_with_unknown_usage"] == 0
    assert meter["not_dispatched_slots"] == 1


def test_persisted_scope_comes_only_from_original_records(tmp_path):
    case = copy.deepcopy(c.runtime_cases()[0])
    original = case["records"][0]
    case["records"] += [
        {**original, "id": "other", "customer": "OTHER", "text": "PRIVATE"},
        {**original, "id": "future", "observed_at": "2099-01-01T00:00:00+00:00"},
    ]
    admission = c.admit_facts(c.canonical({"facts": [fact()]}), case)
    path = tmp_path / "scope.sqlite"
    r.persist_packet(path, case, admission)
    with sqlite3.connect(path) as db:
        rows = db.execute("SELECT id,tenant,customer,payload FROM records").fetchall()
        assert len(rows) == 1 and rows[0][:3] == ("s01", "T1", "U17")
        assert c.loads(rows[0][3]) == original
        assert db.execute("SELECT tenant,customer FROM facts").fetchall() == [("T1", "U17")]
    with pytest.raises(FileExistsError):
        r.persist_packet(path, case, admission)


def test_completed_partial_prefix_can_continue_without_resending(package, tmp_path):
    provider = FakeProvider(tmp_path)
    original_atomic = r.atomic

    def stop_after_first_result(path, value):
        original_atomic(path, value)
        if path.name == "checkpoint.json" and len(value["jobs"]) == 1:
            if next(iter(value["jobs"].values()))["status"] == "complete":
                raise KeyboardInterrupt()

    from unittest.mock import patch

    with patch.object(r, "atomic", stop_after_first_result), pytest.raises(KeyboardInterrupt):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert len(provider.requests) == 1
    result = r.run(
        package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
    )
    assert result["resolved"] == len(provider.requests) == 72


@pytest.mark.parametrize("tamper", ["facts", "status", "request", "reservation", "store", "order"])
def test_tampering_rejected(package, rejected_run, tmp_path, tamper):
    import shutil

    source, _, _ = rejected_run
    shutil.copytree(source, tmp_path, dirs_exist_ok=True)
    app = approval(package, tmp_path)
    r.atomic(tmp_path / "approval.json", app)
    state = c.loads((tmp_path / "checkpoint.json").read_text())
    state["binding"]["approval_sha256"] = c.sha(app)
    if tamper == "facts":
        state["jobs"]["r1-D01-en-extract"]["admission"]["facts"] = [fact()]
    if tamper == "status":
        state["jobs"]["r1-D01-en-extract"]["status"] = "complete"
    if tamper == "request":
        state["jobs"]["r1-D01-en-full_history"]["request"]["model"] = "OTHER"
    if tamper == "reservation":
        state["reserved_ninr"] -= 1
    if tamper == "store":
        with sqlite3.connect(tmp_path / "r1-D01-en-sqlite.sqlite") as db:
            db.execute("DELETE FROM records")
    if tamper == "order":
        state["jobs"] = dict(reversed(list(state["jobs"].items())))
    with pytest.raises(ValueError):
        r.validate_state(package, state, tmp_path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("approved", False),
        ("budget_ninr", 1),
        ("maximum_attempts", 73),
        ("package_sha256", "wrong"),
    ],
)
def test_bad_approval_prevents_any_provider(package, tmp_path, key, value):
    app = approval(package, tmp_path)
    app[key] = value
    provider = FakeProvider(tmp_path)
    with pytest.raises(ValueError):
        r.run(package, tmp_path, app, mode="offline-test", provider=provider)
    assert not provider.requests and not (tmp_path / "checkpoint.json").exists()


def test_no_fake_provider_in_live_mode(package, tmp_path):
    app = approval(package, tmp_path)
    app["mode"] = "live"
    with pytest.raises(ValueError, match="fake provider"):
        r.run(package, tmp_path, app, provider=FakeProvider(tmp_path))
    assert not (tmp_path / "checkpoint.json").exists()


def test_preparation_contains_no_approved_scope_and_refuses_overwrite(tmp_path):
    import subprocess
    import sys

    directory = tmp_path / "package"
    command = [
        sys.executable,
        "-m",
        "benchmarks.sarvam_comparison_v3",
        "prepare",
        "--output-dir",
        str(directory),
    ]
    result = subprocess.run(command, cwd=c.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((directory / "approval-template.json").read_text())["approved"] is False
    assert subprocess.run(command, cwd=c.ROOT, capture_output=True).returncode != 0
    assert not (directory / "live-run").exists()
