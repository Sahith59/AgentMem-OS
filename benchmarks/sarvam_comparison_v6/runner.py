"""Versioned admission policy; no retries and no changes to historical runners."""

import fcntl
from pathlib import Path

from benchmarks.english_screen.runner import atomic
from benchmarks.sarvam_comparison_v5 import runner as previous
from benchmarks.sarvam_discovery.runner import (
    RESERVATION,
    SarvamProvider,
    extraction_id,
    parse_receipt,
    receipt_usage,
)
from benchmarks.sarvam_discovery.storage import read_packet, write_store

from .contract import (
    CASE_IDS,
    ROOT,
    SETTINGS,
    admit_facts,
    file_sha,
    loads,
    parse_answer,
    request,
    runtime_cases,
    sha,
    source_packet,
)

REPEATS = 2


def code_hashes():
    hashes = previous.code_hashes()
    paths = sorted((ROOT / "benchmarks/sarvam_comparison_v6").glob("*.py"))
    return {**hashes, **{str(p.relative_to(ROOT)): file_sha(p) for p in paths}}


def build_package():
    # Original builder validates all fixture/review bindings without loading gold contents.
    parent = previous.build_package()
    jobs = []
    for repeat in range(REPEATS):
        for i, case in enumerate(runtime_cases()):
            order = (
                ["full_history", "extract", "sqlite"]
                if (repeat + i) % 2 == 0
                else ["extract", "sqlite", "full_history"]
            )
            for stage in order:
                jobs.append(
                    {
                        "id": f"r{repeat+1}-{case['id']}-{stage}",
                        "case_id": case["id"],
                        "repeat": repeat + 1,
                        "stage": stage,
                        "request": None if stage == "sqlite" else request(stage, case),
                        "reservation_ninr": RESERVATION,
                    }
                )
    return {
        **parent,
        "schema": "sarvam-comparison-package-v6",
        "provider_schema_profile": (
            "JSON-object mode with same schema in prompt; unchanged local admission"
        ),
        "admission_policy": (
            "whole augmentation rejected if validity date lacks literal witness in cited text; "
            "retain originals; no normalization or repair"
        ),
        "answer_policy": "known-usage completed invalid answer counted invalid; no retry",
        "stop_policy": "transport/receipt/usage/budget/persistence failure stops; no downgrade",
        "semantic_admission": "STRUCTURE_AND_LITERAL_DATE_WITNESS_ONLY_SEMANTICS_UNREVIEWED",
        "parent_package_sha256": sha(parent),
        "settings": dict(SETTINGS),
        "code_sha256": code_hashes(),
        "case_ids": list(CASE_IDS),
        "jobs": jobs,
        "selection": "D01/D06/D07/D09; EN/HI/TE; prospective development subset",
        "rate_basis": "conservative original rates retained; recheck before live dispatch",
    }


def preflight(package):
    if package != build_package():
        raise ValueError("Package differs from fixed subset, review, code or configuration")
    return {
        "package_sha256": sha(package),
        "cases": 12,
        "families": 4,
        "calls_no_retries": 72,
        "answers": 48,
        "extractions": 24,
        "maximum_reservation_ninr": RESERVATION * 72,
        "maximum_reservation_inr": RESERVATION * 72 / 1e9,
        "review": package["review_level"],
        "provider_behavior": "JSON_OBJECT_WITH_LOCAL_SCHEMA; COMPARATIVE_QUALITY_UNKNOWN",
        "status": "OFFLINE_PACKAGE_VALIDATED",
    }


TERMINAL = {"complete", "rejected", "invalid_answer"}


def expected_request(job, case, jobs):
    if job["stage"] != "sqlite":
        return job["request"]
    facts, sources = source_packet(case, jobs[extraction_id(job)]["admission"])
    return request("sqlite", case, facts=facts, sources=sources)


def disposition(stage, text, case, req):
    if stage == "extract":
        admission = admit_facts(text, case)
        return {
            "status": "rejected" if admission["status"] == "rejected" else "complete",
            "parsed": admission["facts"],
            "admission": admission,
        }
    try:
        source_ids = [r["id"] for r in loads(req["messages"][1]["content"])["records"]]
        answer = parse_answer(text, case, source_ids)
        return {"status": "complete", "parsed": answer}
    except (ValueError, TypeError):
        return {"status": "invalid_answer", "parsed": None}


def validate_store(job, spec, case, jobs, directory):
    store = directory / job["store_file"]
    if store.name != spec["id"] + ".sqlite" or store.parent != directory:
        raise ValueError("Unexpected stored DB path")
    if file_sha(store) != job["store_sha256"]:
        raise ValueError("Changed persisted store")
    expected = source_packet(case, jobs[extraction_id(spec)]["admission"])
    if read_packet(store, case) != expected:
        raise ValueError("Store differs from admitted facts and original evidence")


def persist_packet(store, case, admission):
    facts, sources = source_packet(case, admission)
    # The application supplies scope. Never persist unauthorized/future originals.
    write_store(store, {**case, "records": sources}, facts)
    if read_packet(store, case) != (facts, sources):
        raise ValueError("Persistence changed original evidence")
    return {"store_file": store.name, "store_sha256": file_sha(store)}


def validate_state(package, state, directory):
    directory = Path(directory).resolve()
    plan = preflight(package)
    approval = loads((directory / "approval.json").read_text())
    if state["binding"] != {
        "package_sha256": plan["package_sha256"],
        "approval_sha256": sha(approval),
        "mode": approval["mode"],
        "budget_ninr": approval["budget_ninr"],
    }:
        raise ValueError("Changed recorded binding/authorization")
    validate_approval(package, directory, approval, state["binding"]["mode"])
    jobs = state["jobs"]
    if list(jobs) != [j["id"] for j in package["jobs"][: len(jobs)]]:
        raise ValueError("Checkpoint is not a prefix of the request schedule")
    if state["reserved_ninr"] != len(jobs) * RESERVATION:
        raise ValueError("Reservation ledger mismatch")
    if state["reserved_ninr"] > approval["budget_ninr"]:
        raise ValueError("Reservation exceeds approval")
    cases = {c["id"]: c for c in runtime_cases()}
    receipt_ids = set()
    for spec in package["jobs"][: len(jobs)]:
        job, case = jobs[spec["id"]], cases[spec["case_id"]]
        req = expected_request(spec, case, jobs)
        if job["request"] != req or job["request_sha256"] != sha(req):
            raise ValueError("Changed recorded request")
        if job["reservation_ninr"] != RESERVATION or type(job["dispatch_started"]) is not bool:
            raise ValueError("Changed reservation/dispatch status")
        if "usage_ninr" in job:
            _, cost, receipt_id = receipt_usage(job["receipt"])
            if cost != job["usage_ninr"] or receipt_id != job["provider_id"]:
                raise ValueError("Changed usage accounting")
            # A duplicate on the stopped last attempt is a recorded failure, never resolved.
            if receipt_id in receipt_ids and job["status"] in TERMINAL:
                raise ValueError("Duplicate resolved provider receipt ID")
            receipt_ids.add(receipt_id)
        if job["status"] not in TERMINAL:
            if spec["id"] != list(jobs)[-1] or job["status"] not in {"error", "pending"}:
                raise ValueError("Invalid stopped checkpoint")
            continue
        if not job["dispatch_started"] or "usage_ninr" not in job:
            raise ValueError("Resolved request lacks dispatch/known usage")
        text, _, _ = parse_receipt(job["receipt"])
        expected = disposition(spec["stage"], text, case, req)
        if any(job.get(key) != value for key, value in expected.items()):
            raise ValueError("Changed admission or parsed result")
        if spec["stage"] == "sqlite":
            validate_store(job, spec, case, jobs, directory)
    return {
        "intended_calls": len(package["jobs"]),
        "attempted": len(jobs),
        "dispatch_started": sum(j["dispatch_started"] for j in jobs.values()),
        "completed": sum(j["status"] == "complete" for j in jobs.values()),
        "resolved": sum(j["status"] in TERMINAL for j in jobs.values()),
        "rejected_extractions": sum(j["status"] == "rejected" for j in jobs.values()),
        "invalid_answers": sum(j["status"] == "invalid_answer" for j in jobs.values()),
        "reserved_ninr": state["reserved_ninr"],
        "usage_ninr": sum(j.get("usage_ninr", 0) for j in jobs.values()),
        "attempts_with_unknown_usage": sum(
            j["dispatch_started"] and "usage_ninr" not in j for j in jobs.values()
        ),
        "mode": state["binding"]["mode"],
        "quality": "NOT_SCORED_USE_SEPARATE_EVALUATOR",
    }


def validate_approval(package, directory, approval, mode):
    plan = preflight(package)
    if (
        mode not in {"live", "offline-test"}
        or approval.get("approved") is not True
        or approval.get("mode") != mode
        or approval.get("package_sha256") != plan["package_sha256"]
        or approval.get("maximum_attempts") != plan["calls_no_retries"]
        or type(approval.get("budget_ninr")) is not int
        or approval["budget_ninr"] != plan["maximum_reservation_ninr"]
        or approval.get("output_directory") != str(directory)
        or approval.get("accepted_review_level") != package["review_level"]
        or not isinstance(approval.get("authorization_text"), str)
        or not approval["authorization_text"].strip()
    ):
        raise ValueError("Approval must bind exact v6 scope, directory, review level and budget")


def run(package, directory, approval, *, mode="live", provider=None):
    directory = Path(directory).resolve()
    validate_approval(package, directory, approval, mode)
    plan = preflight(package)
    if mode == "live" and provider is not None:
        raise ValueError("Cannot inject a fake provider into live mode")
    if mode == "offline-test" and provider is None:
        raise ValueError("Offline mode requires a fake provider")
    directory.mkdir(parents=True, exist_ok=True)
    binding = {
        "package_sha256": plan["package_sha256"],
        "approval_sha256": sha(approval),
        "budget_ninr": approval["budget_ninr"],
        "mode": mode,
    }
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = directory / "checkpoint.json"
        if checkpoint.exists():
            state = loads(checkpoint.read_text())
            if state["binding"] != binding:
                raise ValueError("Changed resume authorization")
            validate_state(package, state, directory)
            if any(j["status"] not in TERMINAL for j in state["jobs"].values()):
                raise ValueError("Unresolved attempt; automatic retry is forbidden")
        else:
            if any(p.name != "run.lock" for p in directory.iterdir()):
                raise ValueError("New run directory must be empty")
            state = {"binding": binding, "reserved_ninr": 0, "jobs": {}}
            atomic(directory / "approval.json", approval)
            atomic(checkpoint, state)
        cases = {c["id"]: c for c in runtime_cases()}
        if provider is None and len(state["jobs"]) < len(package["jobs"]):
            provider = SarvamProvider()
        for spec in package["jobs"]:
            if spec["id"] in state["jobs"]:
                continue
            case = cases[spec["case_id"]]
            req = expected_request(spec, case, state["jobs"])
            if state["reserved_ninr"] + RESERVATION > binding["budget_ninr"]:
                raise ValueError("Budget would be exceeded before dispatch")
            attempt = {
                "status": "pending",
                "request": req,
                "request_sha256": sha(req),
                "reservation_ninr": RESERVATION,
                "dispatch_started": False,
            }
            state["jobs"][spec["id"]] = attempt
            state["reserved_ninr"] += RESERVATION
            # Durable reservation precedes the only outbound call. A killed pending
            # attempt remains unresolved on resume and is never silently resent.
            atomic(checkpoint, state)
            try:
                if spec["stage"] == "sqlite":
                    store = directory / (spec["id"] + ".sqlite")
                    attempt.update(
                        persist_packet(store, case, state["jobs"][extraction_id(spec)]["admission"])
                    )
                attempt["dispatch_started"] = True
                atomic(checkpoint, state)
                attempt["receipt"] = provider(req)
                atomic(checkpoint, state)  # Preserve raw receipt before any parsing.
                _, cost, receipt_id = receipt_usage(attempt["receipt"])
                attempt.update(usage_ninr=cost, provider_id=receipt_id)
                atomic(checkpoint, state)  # Invalid final JSON still consumed tokens.
                text, cost, receipt_id = parse_receipt(attempt["receipt"])
                for old in list(state["jobs"].values())[:-1]:
                    if old.get("provider_id") == receipt_id:
                        raise ValueError("Duplicate provider receipt ID")
                attempt.update(disposition(spec["stage"], text, case, req))
                atomic(checkpoint, state)
            except Exception as exc:
                # Error type only: exception text can contain authentication data.
                attempt.update(status="error", error_type=type(exc).__name__)
                atomic(checkpoint, state)
                raise RuntimeError(f"Run stopped at {spec['id']}; reservation retained") from None
        summary = validate_state(package, state, directory)
        atomic(directory / "execution.json", summary)
        return summary
