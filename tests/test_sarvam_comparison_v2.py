"""Offline only. Simulated outputs never establish Sarvam answer quality."""

import copy
import json
import socket

import pytest

from benchmarks.sarvam_comparison_v2 import contract as c
from benchmarks.sarvam_comparison_v2 import runner as r
from benchmarks.sarvam_comparison_v2.score import evaluate, gold_labels
from benchmarks.sarvam_discovery.storage import read_packet, write_store


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in this test suite")

    monkeypatch.setattr(socket, "create_connection", forbidden)


def approval(package, directory):
    plan = r.preflight(package)
    return {
        "approved": True,
        "mode": "offline-test",
        "package_sha256": c.sha(package),
        "maximum_attempts": plan["calls_no_retries"],
        "budget_ninr": plan["maximum_reservation_ninr"],
        "output_directory": str(directory.resolve()),
        "accepted_review_level": package["review_level"],
        "authorization_text": "OFFLINE SYNTHETIC TEST ONLY; no paid calls authorized",
    }


def receipt(text, index=1):
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
    """No gold access. Deliberately abstains on all answer fields."""

    def __init__(self, directory, fault=None):
        self.directory, self.fault, self.calls = directory, fault, 0

    def __call__(self, req):
        self.calls += 1
        # Prove the durable reservation exists BEFORE any dispatch.
        state = c.loads((self.directory / "checkpoint.json").read_text())
        assert state["reserved_ninr"] == self.calls * r.RESERVATION
        assert list(state["jobs"].values())[-1]["status"] == "pending"
        assert list(state["jobs"].values())[-1]["request"] == req
        body = c.loads(req["messages"][1]["content"])
        if "fields" in body:
            result = {
                "needs_clarification": True,
                "fields": [
                    {"name": name, "value": None, "source_ids": []} for name in body["fields"]
                ],
            }
        else:
            result = {
                "facts": [
                    {
                        "attribute": "record",
                        "value": item["text"][:200],
                        "status": "current",
                        "condition": None,
                        "valid_from": None,
                        "valid_until": None,
                        "source_ids": [item["id"]],
                    }
                    for item in body["records"]
                ]
            }
        output = receipt(c.canonical(result), self.calls)
        if self.fault:
            return self.fault(output)
        return output


@pytest.fixture(scope="module")
def package():
    return r.build_package()


@pytest.fixture(scope="module")
def completed(package, tmp_path_factory):
    directory = tmp_path_factory.mktemp("sarvam-complete")
    provider = FakeProvider(directory)
    summary = r.run(
        package, directory, approval(package, directory), mode="offline-test", provider=provider
    )
    return directory, summary, provider.calls


def test_extractor_blind_to_questions_and_gold():
    for case in c.runtime_cases():
        body = c.loads(c.request("extract", case)["messages"][1]["content"])
        assert set(body) == {"as_of", "records"}
        changed = copy.deepcopy(case)
        changed.update(
            id="secret-case", question="GOLD_SENTINEL", fields=["GOLD_FIELD"], family="GOLD_FAMILY"
        )
        assert c.request("extract", changed) == c.request("extract", case)
        assert "GOLD_" not in c.canonical(c.request("extract", changed))


def test_plain_storage_retains_sources_even_when_extraction_empty(tmp_path):
    for case in c.runtime_cases():
        path = tmp_path / (case["id"] + ".sqlite")
        write_store(path, case, [])
        facts, sources = read_packet(path, case)
        assert facts == [] and sources == c.eligible(case)
        full = c.loads(c.request("full_history", case)["messages"][1]["content"])
        stored = c.loads(c.request("sqlite", case, facts, sources)["messages"][1]["content"])
        assert full == stored
        with pytest.raises(FileExistsError):
            write_store(path, case, [])


def test_complete_pipeline_and_scoring_watermark(package, completed):
    directory, summary, calls = completed
    assert calls == 72 and summary["completed"] == 72
    assert summary["reserved_ninr"] == 72 * r.RESERVATION
    assert summary["usage_ninr"] == 72 * (100 * c.RATES["input"] + 50 * c.RATES["output"])
    assert len(list(directory.glob("*.sqlite"))) == 24
    score = evaluate(package, directory)
    assert score["status"] == "SIMULATED_TEST_ONLY"
    assert score["paired"]["comparable"] == 24
    assert score["paired"]["sqlite_wins"] == score["paired"]["sqlite_losses"] == 0
    assert len(score["rows"]) == 48
    meter = score["metering"]
    assert meter["arms"]["sqlite"]["attempted_calls"] == 48
    assert meter["arms"]["full_history"]["attempted_calls"] == 24
    assert meter["arms"]["sqlite"]["known_usage_ninr"] == (
        2 * meter["arms"]["full_history"]["known_usage_ninr"]
    )
    assert sum(a["known_usage_ninr"] for a in meter["arms"].values()) == summary["usage_ninr"]


def test_completed_resume_dispatches_nothing(package, completed):
    directory, _, _ = completed
    provider = FakeProvider(directory)
    r.run(package, directory, approval(package, directory), mode="offline-test", provider=provider)
    assert provider.calls == 0


@pytest.mark.parametrize(
    "fault", ["model", "usage", "truncated", "reasoning_only", "bad_json", "rate_limit", "timeout"]
)
def test_failed_attempt_stays_reserved_and_cannot_retry(package, tmp_path, fault):
    def damage(result):
        body = c.loads(result["body_text"])
        if fault == "timeout":
            raise TimeoutError("Do not log SECRET_SENTINEL")
        if fault == "rate_limit":
            result["status_code"] = 429
        if fault == "model":
            body["model"] = "different-model"
        if fault == "usage":
            body["usage"]["prompt_tokens"] = True
        if fault == "truncated":
            body["choices"][0]["finish_reason"] = "length"
        if fault == "reasoning_only":
            body["choices"][0]["message"] = {"content": "", "reasoning_content": '{"facts":[]}'}
        if fault == "bad_json":
            body["choices"][0]["message"]["content"] = "NOT JSON"
        result["body_text"] = c.canonical(body)
        return result

    provider = FakeProvider(tmp_path, damage)
    with pytest.raises(RuntimeError):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert provider.calls == 1
    state = c.loads((tmp_path / "checkpoint.json").read_text())
    assert state["reserved_ninr"] == r.RESERVATION
    assert "SECRET_SENTINEL" not in c.canonical(state)
    if fault in {"truncated", "bad_json", "reasoning_only"}:
        assert next(iter(state["jobs"].values()))["usage_ninr"] > 0
    score = evaluate(package, tmp_path)
    assert score["execution"]["completed"] == 0
    assert score["arms"]["sqlite"]["valid_answers"] == 0
    with pytest.raises(ValueError, match="automatic retry"):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert provider.calls == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("approved", False),
        ("budget_ninr", 1),
        ("maximum_attempts", 73),
        ("accepted_review_level", "human"),
    ],
)
def test_bad_authorization_never_dispatches(package, tmp_path, key, value):
    auth = approval(package, tmp_path)
    auth[key] = value
    provider = FakeProvider(tmp_path)
    with pytest.raises(ValueError):
        r.run(package, tmp_path, auth, mode="offline-test", provider=provider)
    assert provider.calls == 0 and not (tmp_path / "checkpoint.json").exists()


@pytest.mark.parametrize("fault", ["gold", "code", "model", "prompt", "order"])
def test_package_tampering_rejected(package, fault):
    changed = copy.deepcopy(package)
    if fault in {"gold", "code"}:
        mapping = changed["fixtures_sha256" if fault == "gold" else "code_sha256"]
        mapping[next(iter(mapping))] = "000"
    elif fault == "model":
        changed["settings"]["model"] = "other"
    elif fault == "prompt":
        changed["jobs"][0]["request"]["messages"][0]["content"] = "answer from gold"
    else:
        changed["jobs"].reverse()
    with pytest.raises(ValueError):
        r.preflight(changed)


@pytest.mark.parametrize("fault", ["parsed", "receipt", "cost", "request", "reservation", "hole"])
def test_checkpoint_corruption_is_not_scored(package, completed, fault):
    directory, _, _ = completed
    state = c.loads((directory / "checkpoint.json").read_text())
    job = next(iter(state["jobs"].values()))
    if fault == "parsed":
        job["parsed"] = []
    elif fault == "receipt":
        job["receipt"]["body_text"] = "{}"
    elif fault == "cost":
        job["usage_ninr"] = 0
    elif fault == "request":
        job["request"]["model"] = "other"
    elif fault == "reservation":
        state["reserved_ninr"] = 0
    else:
        del state["jobs"][next(iter(state["jobs"]))]
    with pytest.raises(ValueError):
        r.validate_state(package, state, directory)


def test_pending_crash_cannot_redispatch(package, tmp_path):
    provider = FakeProvider(tmp_path, lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    state = json.loads((tmp_path / "checkpoint.json").read_text())
    assert next(iter(state["jobs"].values()))["status"] == "pending"
    with pytest.raises(ValueError, match="automatic retry"):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert provider.calls == 1


def test_duplicate_provider_id_stops_with_cost_retained(package, tmp_path):
    def duplicate(result):
        body = c.loads(result["body_text"])
        body["id"] = "same-provider-id"
        result["body_text"] = c.canonical(body)
        return result

    provider = FakeProvider(tmp_path, duplicate)
    with pytest.raises(RuntimeError):
        r.run(
            package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert provider.calls == 2
    state = c.loads((tmp_path / "checkpoint.json").read_text())
    summary = r.validate_state(package, state, tmp_path)
    assert summary["reserved_ninr"] == 2 * r.RESERVATION
    assert summary["attempts_with_unknown_usage"] == 0
    assert list(state["jobs"].values())[-1]["status"] == "error"


def test_persisted_database_tampering_rejected(package, tmp_path):
    import sqlite3

    provider = FakeProvider(tmp_path)
    r.run(package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider)
    path = next(tmp_path.glob("*.sqlite"))
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM records")
    with pytest.raises(ValueError, match="persisted store"):
        evaluate(package, tmp_path)


def test_preparation_cli_is_offline_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    from benchmarks.sarvam_comparison_v2.__main__ import main

    target = tmp_path / "prepared"
    monkeypatch.setattr("sys.argv", ["discovery", "prepare", "--output-dir", str(target)])
    main()
    plan = c.loads((target / "preflight.json").read_text())
    assert plan["calls_no_retries"] == 72
    auth = c.loads((target / "approval-template.json").read_text())
    assert auth["approved"] is False and auth["authorization_text"] == ""
    assert not (target / "live-run").exists()
    with pytest.raises(FileExistsError):
        main()
    assert "OFFLINE_PACKAGE_VALIDATED" in capsys.readouterr().out


def test_exact_subset_and_single_request_field_change(package):
    from benchmarks.sarvam_discovery import contract as legacy
    from benchmarks.sarvam_discovery import runner as parent

    cases = c.runtime_cases()
    assert [case["id"] for case in cases] == list(c.CASE_IDS)
    assert len(cases) == 12 and len({case["family"] for case in cases}) == 4
    assert len(gold_labels(cases)) == 12
    assert r.preflight(package)["maximum_reservation_ninr"] == 297908305920
    assert parent.preflight(parent.build_package())["calls_no_retries"] == 228
    assert legacy.SETTINGS["reasoning_effort"] == "low"
    for case in cases:
        for stage in ("extract", "full_history", "sqlite"):
            facts, sources = [], c.eligible(case)
            actual = c.request(stage, case, facts, sources)
            before = legacy.request(stage, case, facts, sources)
            assert actual == {**before, "reasoning_effort": None}
            assert '"reasoning_effort":null' in c.canonical(actual)
    for repeat in (1, 2):
        for i, case in enumerate(cases):
            jobs = [
                j for j in package["jobs"] if j["case_id"] == case["id"] and j["repeat"] == repeat
            ]
            expected = (
                ["full_history", "sqlite"]
                if (repeat - 1 + i) % 2 == 0
                else ["sqlite", "full_history"]
            )
            assert [j["stage"] for j in jobs] == ["extract", *expected]


def test_real_dependent_requests_keep_null_and_unchanged_originals(package, completed):
    directory, _, _ = completed
    state = c.loads((directory / "checkpoint.json").read_text())
    for case in c.runtime_cases():
        for repeat in (1, 2):
            full = state["jobs"][f"r{repeat}-{case['id']}-full_history"]["request"]
            stored = state["jobs"][f"r{repeat}-{case['id']}-sqlite"]["request"]
            assert stored["reasoning_effort"] is None and full["reasoning_effort"] is None
            assert (
                c.loads(full["messages"][1]["content"])["records"]
                == c.loads(stored["messages"][1]["content"])["records"]
                == c.eligible(case)
            )
    result = evaluate(package, directory)
    assert len(result["extraction_review"]) == 24
    assert all(
        set(row["semantic_review"].values()) == {"UNREVIEWED"}
        for row in result["extraction_review"]
    )


@pytest.mark.parametrize(
    "mutation", ["expand_subset", "reasoning_low", "reasoning_omitted", "old_package"]
)
def test_other_scope_never_starts(package, tmp_path, mutation):
    from benchmarks.sarvam_discovery import runner as parent

    changed = copy.deepcopy(package)
    if mutation == "expand_subset":
        changed["case_ids"].append("D08-en")
    elif mutation == "reasoning_low":
        changed["jobs"][0]["request"]["reasoning_effort"] = "low"
    elif mutation == "reasoning_omitted":
        del changed["jobs"][0]["request"]["reasoning_effort"]
    else:
        changed = parent.build_package()
    provider = FakeProvider(tmp_path)
    with pytest.raises(ValueError):
        r.run(
            changed, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider
        )
    assert provider.calls == 0 and not (tmp_path / "checkpoint.json").exists()


def test_no_gold_content_read_during_inference(package, tmp_path, monkeypatch):
    from pathlib import Path

    read = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name == "gold.json":
            raise AssertionError("Inference must not load answer labels")
        return read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    provider = FakeProvider(tmp_path)
    r.run(package, tmp_path, approval(package, tmp_path), mode="offline-test", provider=provider)
    assert provider.calls == 72


def test_live_mode_cannot_accept_fake_provider(package, tmp_path):
    auth = approval(package, tmp_path)
    auth["mode"] = "live"
    with pytest.raises(ValueError, match="fake provider"):
        r.run(package, tmp_path, auth, provider=FakeProvider(tmp_path))
    assert not (tmp_path / "checkpoint.json").exists()
