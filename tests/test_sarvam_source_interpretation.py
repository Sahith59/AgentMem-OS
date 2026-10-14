"""Offline checks for the 240-history, three-arm source study."""

import contextlib
import copy
import io
import json
import shutil
import time
from pathlib import Path

import pytest

from benchmarks import sarvam_source_interpretation as study
from benchmarks.sarvam_native_study.transport import ENDPOINT, digest, save
from benchmarks.sarvam_reporting_handoff import prepare_plan
from tests.test_sarvam_reporting_handoff import fixture, reply

EXPERIMENTS = Path(
    "/Volumes/Sahith_SSD/AgentMem-OS/codex-memory-2026-09-08/tracks/sarvam/experiments"
)


def synthetic_inputs():
    base = prepare_plan(**fixture())["reports"]["candidate"]["request"]
    requests = {"sarvam-105b": base}
    conversation = copy.deepcopy(base)
    conversation["model"] = "sarvam-105b-conversations"
    conversation["reasoning_effort"] = None
    requests[conversation["model"]] = conversation
    inputs = []
    for n in range(40):
        for language in ("en", "hi", "te"):
            for model in requests:
                ident = f"synthetic-{n:02d}-{language}-{model}"
                inputs.append(
                    {
                        "id": ident,
                        "case_id": ident,
                        "scenario": f"scenario-{n:02d}",
                        "language": language,
                        "family": "synthetic",
                        "model": model,
                        "request": requests[model],
                        "tool_trace": [],
                    }
                )
    return inputs


def test_six_orders_per_scenario_and_exact_budget():
    jobs = study._ordered_jobs(synthetic_inputs())
    assert len(jobs) == 720
    assert study._budget(jobs)["maximum_full_context_reservation_microinr"] == 1_548_288_000
    by_scenario = {}
    for job in jobs:
        by_scenario.setdefault(job["scenario"], {}).setdefault(job["history_id"], []).append(
            job["arm"]
        )
    for histories in by_scenario.values():
        assert len(histories) == 6
        assert len({tuple(order) for order in histories.values()}) == 6
    assert all(not job["treated_actor_prose"] for job in jobs)
    by_history = {(job["history_id"], job["arm"]): job for job in jobs}
    for item in synthetic_inputs():
        assert by_history[item["id"], study.ARMS[1]]["request_sha256"] == (
            by_history[item["id"], study.ARMS[2]]["request_sha256"]
        )


def test_bootstrap_resamples_all_40_scenarios_even_with_sparse_treatment():
    before = {"scenario": "scenario-00", "factual_answer_correct": False}
    after = {"scenario": "scenario-00", "factual_answer_correct": True}
    result = study._bootstrap(
        [(before, after)], {f"scenario-{n:02d}" for n in range(40)}, draws=1000
    )
    assert result["scenario_clusters"] == 40
    assert result["zero_denominator_draws"] > 0
    assert result["estimable_draws"] + result["zero_denominator_draws"] == 1000
    assert result["percentile_95_percentage_points"] == [100.0, 100.0]


def test_stop_after_two_cumulative_infrastructure_errors_and_refuse_resume(tmp_path, monkeypatch):
    inputs = synthetic_inputs()
    jobs = study._ordered_jobs(inputs)
    package = {
        "jobs": jobs,
        "maximum_calls": 720,
        "worst_case_retained_microinr": study.FULL_RESERVATION,
        "workers": 4,
        "batch_size": 4,
        "stop_after_batch_errors": 2,
    }
    monkeypatch.setattr(study, "check", lambda _root: (package, {x["id"]: x for x in inputs}))
    save(tmp_path / "budget.json", study._budget(jobs))
    calls = []

    class NoWait:
        def __init__(self, _root):
            pass

        def wait(self):
            return {"dispatch_time": time.time(), "pacing_control": {"rpm": 90}}

    def transport(_meter, job, _stage, _request, _key):
        calls.append(job)
        if len(calls) in (2, 4):
            raise RuntimeError("offline simulated infrastructure failure")
        return reply(None, status="not_requested", clarification=True)

    with contextlib.redirect_stdout(io.StringIO()):
        execution = study.run(
            tmp_path,
            allow_paid=True,
            transport_call=transport,
            credential_loader=lambda: "offline",
            pacer_factory=NoWait,
        )
    assert execution["attempted"] == 4 and execution["infrastructure_errors"] == 2
    assert len(execution["outcomes"]) == 720
    assert all(row["terminal"] == "not_attempted_after_stop" for row in execution["outcomes"][4:])
    with pytest.raises(ValueError, match="already started"):
        study.run(
            tmp_path,
            allow_paid=True,
            transport_call=transport,
            credential_loader=lambda: pytest.fail("credential accessed on resume"),
            pacer_factory=NoWait,
        )
    assert len(calls) == 4


def test_one_fatal_value_error_stops_after_started_batch(tmp_path, monkeypatch):
    inputs = synthetic_inputs()
    jobs = study._ordered_jobs(inputs)
    package = {
        "jobs": jobs,
        "maximum_calls": 720,
        "worst_case_retained_microinr": study.FULL_RESERVATION,
        "workers": 4,
        "batch_size": 4,
        "stop_after_batch_errors": 2,
    }
    monkeypatch.setattr(study, "check", lambda _root: (package, {x["id"]: x for x in inputs}))
    save(tmp_path / "budget.json", study._budget(jobs))

    class NoWait:
        def __init__(self, _root):
            pass

        def wait(self):
            return {"dispatch_time": time.time(), "pacing_control": {"rpm": 90}}

    def transport(_meter, job, _stage, _request, _key):
        if job == jobs[0]["id"]:
            raise ValueError("budget/source validation failed")
        return reply(None, status="not_requested", clarification=True)

    with contextlib.redirect_stdout(io.StringIO()):
        execution = study.run(
            tmp_path,
            allow_paid=True,
            transport_call=transport,
            credential_loader=lambda: "offline",
            pacer_factory=NoWait,
        )
    assert execution["attempted"] == 4 and execution["fatal_aborts"] == 1
    assert execution["infrastructure_errors"] == 1


def test_paid_path_refuses_by_default_before_source_or_key(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "check", lambda _root: pytest.fail("source checked"))
    with pytest.raises(ValueError, match="allow-paid"):
        study.run(tmp_path, credential_loader=lambda: pytest.fail("credential accessed"))


def test_actual_parent_offline_freeze_and_fake_720_replay_when_available(tmp_path, monkeypatch):
    parent = EXPERIMENTS / "2026-10-13-fresh-workflow-confirmation-v3"
    phase = EXPERIMENTS / "2026-10-13-source-interpretation-full"
    prior = EXPERIMENTS / "2026-10-13-capped-actor-recovery-diagnostic/completion.json"
    if not all(path.exists() for path in (parent, phase, prior)):
        pytest.skip("Frozen local Sarvam study artifacts unavailable")
    for name in ("PROTOCOL.md", "budget.json", "pacing-control.json", "source-preflight.json"):
        shutil.copyfile(phase / name, tmp_path / name)
    original_read = study._read

    def no_gold_read(path):
        if Path(path).name == "gold.json":
            pytest.fail("Gold parsed before offline audit")
        return original_read(path)

    with monkeypatch.context() as guarded:
        guarded.setattr(study, "_read", no_gold_read)
        package = study.prepare(tmp_path, parent, prior)
        checked, inputs = study.check(tmp_path)
    assert checked == package and len(inputs) == 240
    assert package["treated_actor_prose_histories"] == 184
    assert len(package["jobs"]) == 720
    assert not (tmp_path / "started.json").exists()
    frozen_inputs = (tmp_path / "inputs.json").read_bytes()
    (tmp_path / "inputs.json").write_bytes(frozen_inputs + b" ")
    with pytest.raises(ValueError, match="Frozen input changed"):
        study.check(tmp_path)
    (tmp_path / "inputs.json").write_bytes(frozen_inputs)

    class NoWait:
        def __init__(self, _root):
            pass

        def wait(self):
            return {"dispatch_time": time.time(), "pacing_control": {"rpm": 90}}

    def transport(meter, job, stage, request, _key):
        save(
            meter.root / f"{job}.request.json",
            {"endpoint": ENDPOINT, "request": request, "sha256": digest(request)},
        )
        meter.reserve(job, stage, request)
        body = reply(None, status="not_requested", clarification=True)
        body["id"] = "offline-" + job
        body["model"] = request["model"]
        body["usage"] = {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130}
        save(
            meter.root / f"{job}.receipt.json",
            {
                "job": job,
                "request_sha256": digest(request),
                "status": 200,
                "body_text": json.dumps(body),
                "elapsed_seconds": 0.01,
            },
        )
        assert meter.settle(job, body, request["model"], request["max_tokens"])
        return body

    with monkeypatch.context() as guarded:
        guarded.setattr(study, "_read", no_gold_read)
        with contextlib.redirect_stdout(io.StringIO()):
            execution = study.run(
                tmp_path,
                allow_paid=True,
                transport_call=transport,
                credential_loader=lambda: "offline",
                pacer_factory=NoWait,
            )
    assert execution["attempted"] == 720
    audit = study.audit(tmp_path)
    assert audit["verified"] and audit["complete"]
    assert len(audit["rows"]) == 720 and audit["reporter_receipts"] == 720
    assert audit["known_microinr"] == 2_376_000
    assert audit["screen_gate"]["checks"]["production_default_transport"] is False
    assert audit["contrasts"]["B_to_C"]["treated/all"]["n"] == 184
    assert audit["contrasts"]["B_to_C"]["no_treatment/all"]["n"] == 56
    assert len(audit["actual_dispatch_order"]) == 720
