"""Offline tests for the saved-cap diagnostic; transport and credentials are injected."""

import hashlib
import json

import pytest

from benchmarks import sarvam_handoff_diagnostic as diagnostic
from benchmarks.sarvam_native_study.transport import ENDPOINT, digest, save
from benchmarks.sarvam_reporting_handoff import prepare_plan
from tests.test_sarvam_reporting_handoff import fixture, reply


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_fixture(tmp_path, monkeypatch, *, unsafe=False, opaque_gold=False):
    parent, root = tmp_path / "parent-v3", tmp_path / "diagnostic"
    parent.mkdir()
    root.mkdir()
    (root / "PROTOCOL.md").write_text("Fixed prospective one-cap protocol\n")
    inp = fixture()
    baseline_plan = prepare_plan(**inp)
    monkeypatch.setattr(diagnostic, "EXPECTED_PLAN_SHA256", baseline_plan["plan_sha256"])
    monkeypatch.setattr(
        diagnostic,
        "EXPECTED_REQUEST_SHA256",
        {arm: baseline_plan["reports"][arm]["request_sha256"] for arm in ("control", "candidate")},
    )
    case_id = "fresh-13-en"
    selected_id = diagnostic.EXPECTED_ELIGIBLE
    if unsafe:
        inp["tool_trace"] = [{"name": "update_appointment", "arguments": {}, "response": {}}]
    jobs, outcomes, rows = [], [], []
    initial_sha = digest(inp["initial_backend"])
    for index in range(diagnostic.PARENT_JOBS):
        eid = selected_id if index == 13 else f"ordinary-{index:03d}-m0"
        selected = eid == selected_id
        actor_calls = [eid + "-a0"]
        jobs.append(
            {
                "id": eid,
                "case_id": case_id,
                "model": "sarvam-105b",
                "initial_sha256": initial_sha,
                "report_order": ["control", "candidate"],
            }
        )
        outcomes.append({"id": eid, "actual_calls": actor_calls})
        reports = {
            arm: {"terminal": "not_reached" if selected else "answer"}
            for arm in ("control", "candidate")
        }
        for arm in reports:
            rows.append({"id": eid, "variant": arm, "terminal": reports[arm]["terminal"]})
        original = {
            "id": eid,
            "case_id": case_id,
            "model": "sarvam-105b",
            "initial_sha256": initial_sha,
            "terminal": "invalid_action_response" if selected else "answer",
            "calls": actor_calls,
            "tool_trace": inp["tool_trace"] if selected else [],
            "initial_backend": inp["initial_backend"],
            "final_backend": inp["final_backend"],
        }
        _write(
            parent / "episodes" / eid / "paired.json",
            {"original": original, "reports": reports, "actual_calls": actor_calls},
        )
    runtime = [{"id": case_id, "caller": inp["caller_scope"]}]
    _write(parent / "runtime.json", runtime)
    _write(parent / "execution.json", {"outcomes": outcomes})
    _write(
        parent / "verification.json",
        {
            "verified": True,
            "complete": False,
            "mechanical_screen_pass": False,
            "native_episodes": 240,
            "rows": rows,
            "research_calls": diagnostic.PRIOR_CALLS,
            "research_known_inr": diagnostic.PRIOR_KNOWN / 1e6,
            "research_retained_inr": diagnostic.PRIOR_RETAINED / 1e6,
            "unknown_calls": 0,
            "calls": 1,
        },
    )
    if opaque_gold:
        (parent / "gold.json").write_bytes(b"gold is deliberately not JSON")
    else:
        _write(
            parent / "gold.json",
            {
                case_id: {
                    "runtime_sha256": digest(runtime[0]),
                    "question_target": "source_request_or_condition",
                    "expected_answer": {"answer": None, "needs_clarification": True},
                    "forbid_new_write": True,
                }
            },
        )
    for name in ("budget.json", "review.json", "pacing-control.json"):
        _write(parent / name, {})
    (parent / "PROTOCOL.md").write_text("Frozen parent protocol\n")
    frozen = (
        "runtime.json",
        "budget.json",
        "review.json",
        "PROTOCOL.md",
        "pacing-control.json",
        "gold.json",
    )
    _write(
        parent / "manifest.json",
        {
            "jobs": jobs,
            "source_sha256": {},
            "files_sha256": {name: _sha(parent / name) for name in frozen},
        },
    )
    actor_job = selected_id + "-a0"
    _write(
        parent / "calls" / f"{actor_job}.request.json",
        {
            "endpoint": ENDPOINT,
            "request": inp["actor_request"],
            "sha256": digest(inp["actor_request"]),
        },
    )
    _write(parent / "calls" / f"{actor_job}.receipt.json", inp["native_receipt"])
    return parent, root


class FakePacer:
    def __init__(self, _root):
        pass

    def wait(self):
        return {"fake": True}


def fake_transport(*, unknown=False, malformed=False):
    dispatched = []

    def send(meter, job, stage, request, _key):
        dispatched.append(job)
        save(
            meter.root / f"{job}.request.json",
            {"endpoint": ENDPOINT, "request": request, "sha256": digest(request)},
        )
        meter.reserve(job, stage, request)
        body = reply(None, status="confirmed", clarification=True)
        body["model"] = request["model"]
        body["usage"] = {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}
        if unknown:
            del body["usage"]
        if malformed:
            del body["choices"]
        save(
            meter.root / f"{job}.receipt.json",
            {
                "job": job,
                "request_sha256": digest(request),
                "status": 200,
                "elapsed_seconds": 0.01,
                "body_text": json.dumps(body),
            },
        )
        known = meter.settle(job, body, request["model"], request["max_tokens"])
        if not known:
            raise RuntimeError("Unknown usage")
        return body

    return send, dispatched


def offline_run(root, transport, *, pacer_factory=FakePacer, key_loader=lambda: "fake"):
    return diagnostic.run(
        root,
        allow_paid=True,
        transport_call=transport,
        credential_loader=key_loader,
        pacer_factory=pacer_factory,
    )


def test_portable_prepare_and_two_metered_reports(tmp_path, monkeypatch):
    parent, root = parent_fixture(tmp_path, monkeypatch)
    package = diagnostic.prepare(parent, root)
    assert len(package["parent_census"]) == 240
    assert package["selected_id"] == diagnostic.EXPECTED_ELIGIBLE
    assert len(package["refusals"]) == 0
    assert package["request_sha256"]["control"] != package["request_sha256"]["candidate"]
    assert package["snapshot_sha256"]["sealed/gold.json"] == _sha(parent / "gold.json")
    # The package can be replayed without the live parent tree.
    parent.rename(tmp_path / "parent-detached")
    diagnostic._load(root)
    with pytest.raises(ValueError, match="allow-paid"):
        diagnostic.run(root)
    send, dispatched = fake_transport()
    result = offline_run(root, send)
    assert len(dispatched) == 2
    assert result["reports"]["candidate"]["parsed"]["action_status"] == "confirmed"
    report = diagnostic.audit(root)
    assert report["new_calls"] == 2 and report["known_usage_all"] is True
    assert report["candidate_primary_factual_match"] is True
    assert report["reports"]["candidate"]["raw_status_matches_read_only"] is False
    assert report["reports"]["candidate"]["receipt_owned_status_matches_read_only"] is True
    assert report["preserved_parent_snapshot_replayed"] is True
    assert not (root / "audit.json").exists()
    assert diagnostic.audit(root) == report
    diagnostic.audit(root, output=root / "audit.json")
    with pytest.raises(FileExistsError):
        diagnostic.audit(root, output=root / "audit.json")
    with pytest.raises(ValueError, match="already started"):
        offline_run(root, send, key_loader=lambda: pytest.fail("credential accessed on retry"))


def test_mutated_source_or_protocol_refuses_before_credential(tmp_path, monkeypatch):
    parent, root = parent_fixture(tmp_path, monkeypatch)
    diagnostic.prepare(parent, root)
    copied = root / "parent" / "runtime.json"
    copied.write_text(copied.read_text() + " ")
    with pytest.raises(ValueError, match="Frozen diagnostic input changed"):
        offline_run(
            root,
            lambda *_: pytest.fail("transport called"),
            key_loader=lambda: pytest.fail("credential accessed"),
        )
    copied.write_bytes((parent / "runtime.json").read_bytes())
    (root / "PROTOCOL.md").write_text("Mutated protocol")
    with pytest.raises(ValueError, match="Frozen diagnostic input changed"):
        diagnostic._load(root)


def test_unsafe_write_history_and_gold_isolation(tmp_path, monkeypatch):
    parent, root = parent_fixture(tmp_path, monkeypatch, unsafe=True)
    with pytest.raises(ValueError, match="Eligible capped-actor set differs"):
        diagnostic.prepare(parent, root)
    assert not (root / "manifest.json").exists()
    other = tmp_path / "other"
    other.mkdir()
    parent2, root2 = parent_fixture(other, monkeypatch, opaque_gold=True)
    diagnostic.prepare(parent2, root2)
    assert (root2 / "sealed" / "gold.json").read_bytes() == b"gold is deliberately not JSON"
    assert "gold is deliberately" not in json.dumps(diagnostic._load(root2)[1])
    send, _ = fake_transport()
    offline_run(root2, send)
    with pytest.raises(json.JSONDecodeError):
        diagnostic.audit(root2)


def test_unknown_usage_retains_full_reservation_and_stops_second(tmp_path, monkeypatch):
    parent, root = parent_fixture(tmp_path, monkeypatch)
    diagnostic.prepare(parent, root)
    send, dispatched = fake_transport(unknown=True)
    result = offline_run(root, send)
    assert len(dispatched) == 1
    assert result["reports"]["control"]["terminal"] == "infrastructure_error"
    assert result["reports"]["candidate"]["terminal"] == "not_attempted_after_prior_failure"
    report = diagnostic.audit(root)
    assert report["new_calls"] == 1 and report["known_usage_all"] is False
    assert report["new_known_microinr"] == 0
    assert (
        report["new_retained_microinr"]
        == report["attempts"][0]["usage"]["full_reservation_microinr"]
    )


def test_malformed_known_usage_and_predispatch_failure_replay(tmp_path, monkeypatch):
    parent, root = parent_fixture(tmp_path, monkeypatch)
    diagnostic.prepare(parent, root)
    send, dispatched = fake_transport(malformed=True)
    result = offline_run(root, send)
    assert len(dispatched) == 1
    assert result["reports"]["candidate"]["terminal"] == "not_attempted_after_prior_failure"
    report = diagnostic.audit(root)
    assert report["new_calls"] == 1 and report["attempts"][0]["usage"]["known"] is True
    other = tmp_path / "pre"
    other.mkdir()
    parent2, root2 = parent_fixture(other, monkeypatch)
    diagnostic.prepare(parent2, root2)

    class BrokenPacer(FakePacer):
        def wait(self):
            raise OSError("pre-dispatch failure")

    result2 = offline_run(
        root2, lambda *_: pytest.fail("transport called"), pacer_factory=BrokenPacer
    )
    assert result2["reports"]["control"]["terminal"] == "infrastructure_error"
    assert result2["reports"]["candidate"]["terminal"] == "not_attempted_after_prior_failure"
    report2 = diagnostic.audit(root2)
    assert report2["new_calls"] == 0


def test_audit_rejects_tampered_raw_report_and_unmetered_extra_request(tmp_path, monkeypatch):
    parent, root = parent_fixture(tmp_path, monkeypatch)
    diagnostic.prepare(parent, root)
    send, _ = fake_transport()
    offline_run(root, send)
    execution = json.loads((root / "execution.json").read_text())
    execution["reports"]["control"]["content"] = "invented"
    _write(root / "execution.json", execution)
    with pytest.raises(ValueError, match="does not replay"):
        diagnostic.audit(root)
    execution["reports"]["control"]["content"] = json.loads(
        (
            root / "calls" / (diagnostic.EXPECTED_ELIGIBLE + "-a0-handoff-control.receipt.json")
        ).read_text()
    )["body_text"]
    # Restore by deriving the saved assistant content, then check inventory.
    execution["reports"]["control"]["content"] = json.loads(
        execution["reports"]["control"]["content"]
    )["choices"][0]["message"]["content"]
    _write(root / "execution.json", execution)
    _write(root / "calls" / "extra-a0.request.json", {})
    with pytest.raises(ValueError, match="inventory mismatch"):
        diagnostic.audit(root)
