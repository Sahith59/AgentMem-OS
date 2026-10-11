"""Fixed 72-call comparison. Original pilot code and evidence remain immutable.

The execution/checkpoint lifecycle is versioned from sarvam_discovery.runner.
Provider, parsers, source storage and prompt builders reuse its frozen contracts.
No global settings are patched; the two packages can be loaded in one process.
"""

import fcntl
from pathlib import Path

from benchmarks.english_screen.runner import atomic
from benchmarks.sarvam_discovery import runner as original
from benchmarks.sarvam_discovery.runner import (
    RESERVATION,
    SarvamProvider,
    expected_packet,
    extraction_id,
    parse_receipt,
    receipt_usage,
)
from benchmarks.sarvam_discovery.storage import read_packet, write_store

from .contract import (
    CASE_IDS,
    ROOT,
    SETTINGS,
    file_sha,
    loads,
    parse_answer,
    parse_facts,
    request,
    runtime_cases,
    sha,
)

REPEATS = 2


def code_hashes():
    hashes = original.code_hashes()
    paths = sorted((ROOT / "benchmarks/sarvam_comparison_v2").glob("*.py"))
    return {**hashes, **{str(p.relative_to(ROOT)): file_sha(p) for p in paths}}


def build_package():
    # Original builder validates all fixture/review bindings without loading gold contents.
    parent = original.build_package()
    jobs = []
    for repeat in range(REPEATS):
        for i, case in enumerate(runtime_cases()):
            arms = (
                ["full_history", "sqlite"] if (repeat + i) % 2 == 0 else ["sqlite", "full_history"]
            )
            for stage in ("extract", *arms):
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
        "schema": "sarvam-comparison-package-v2",
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
        "provider_behavior": "ONE_EN_EXTRACTION_OBSERVED; COMPARATIVE_QUALITY_UNKNOWN",
        "status": "OFFLINE_PACKAGE_VALIDATED",
    }


def expected_request(job, case, jobs):
    if job["stage"] != "sqlite":
        return job["request"]
    facts = jobs[extraction_id(job)]["parsed"]
    facts, sources = expected_packet(case, facts)
    return request("sqlite", case, facts=facts, sources=sources)


def validate_state(package, state, directory):
    directory = Path(directory).resolve()
    plan = preflight(package)
    if state["binding"]["package_sha256"] != plan["package_sha256"]:
        raise ValueError("Changed package binding")
    approval = loads((directory / "approval.json").read_text())
    if (
        sha(approval) != state["binding"]["approval_sha256"]
        or approval["mode"] != state["binding"]["mode"]
        or approval["budget_ninr"] != state["binding"]["budget_ninr"]
        or approval["package_sha256"] != plan["package_sha256"]
    ):
        raise ValueError("Changed recorded authorization")
    completed_jobs = state["jobs"]
    # Dict insertion order records a prefix of the frozen schedule.
    if list(completed_jobs) != [j["id"] for j in package["jobs"][: len(completed_jobs)]]:
        raise ValueError("Checkpoint is not a prefix of the request schedule")
    if (
        state["reserved_ninr"] != len(completed_jobs) * RESERVATION
        or state["reserved_ninr"] > state["binding"]["budget_ninr"]
    ):
        raise ValueError("Reservation ledger mismatch")
    cases = {c["id"]: c for c in runtime_cases()}
    receipt_ids = set()
    for spec in package["jobs"][: len(completed_jobs)]:
        job = completed_jobs[spec["id"]]
        case = cases[spec["case_id"]]
        req = expected_request(spec, case, completed_jobs)
        if job["request"] != req or job["request_sha256"] != sha(req):
            raise ValueError("Changed recorded request")
        if job["reservation_ninr"] != RESERVATION:
            raise ValueError("Changed attempt reservation")
        if job["status"] != "complete":
            if spec["id"] != list(completed_jobs)[-1] or job["status"] not in {"pending", "error"}:
                raise ValueError("Invalid stopped checkpoint")
            if "usage_ninr" in job:
                _, cost, receipt_id = receipt_usage(job["receipt"])
                if cost != job["usage_ninr"] or receipt_id != job["provider_id"]:
                    raise ValueError("Changed failed-call usage accounting")
            continue
        text, cost, receipt_id = parse_receipt(job["receipt"])
        if receipt_id in receipt_ids:
            raise ValueError("Duplicate provider receipt ID")
        receipt_ids.add(receipt_id)
        if cost != job["usage_ninr"] or receipt_id != job["provider_id"]:
            raise ValueError("Changed usage accounting")
        if spec["stage"] == "extract":
            parsed = parse_facts(text, case)
        else:
            source_ids = [s["id"] for s in loads(req["messages"][1]["content"])["records"]]
            parsed = parse_answer(text, case, source_ids)
        if parsed != job["parsed"]:
            raise ValueError("Changed parsed final response")
        if spec["stage"] == "sqlite":
            store = directory / job["store_file"]
            if store.name != spec["id"] + ".sqlite" or store.parent != directory:
                raise ValueError("Unexpected stored DB path")
            if file_sha(store) != job["store_sha256"]:
                raise ValueError("Changed persisted store")
            facts = completed_jobs[extraction_id(spec)]["parsed"]
            if read_packet(store, case) != expected_packet(case, facts):
                raise ValueError("Store differs from model extraction/source input")
    return {
        "intended_calls": len(package["jobs"]),
        "attempted": len(completed_jobs),
        "completed": sum(j["status"] == "complete" for j in completed_jobs.values()),
        "reserved_ninr": state["reserved_ninr"],
        "usage_ninr": sum(j.get("usage_ninr", 0) for j in completed_jobs.values()),
        "attempts_with_unknown_usage": sum("usage_ninr" not in j for j in completed_jobs.values()),
        "mode": state["binding"]["mode"],
        "quality": "NOT_SCORED_USE_SEPARATE_EVALUATOR",
    }


def run(package, directory, approval, *, mode="live", provider=None):
    plan = preflight(package)
    directory = Path(directory).resolve()
    if (
        mode not in {"live", "offline-test"}
        or approval.get("approved") is not True
        or approval.get("mode") != mode
        or approval.get("package_sha256") != plan["package_sha256"]
        or approval.get("maximum_attempts") != plan["calls_no_retries"]
        or type(approval.get("budget_ninr")) is not int
        or approval["budget_ninr"] < plan["maximum_reservation_ninr"]
        or approval.get("output_directory") != str(directory)
        or approval.get("accepted_review_level") != package["review_level"]
        or not isinstance(approval.get("authorization_text"), str)
        or not approval["authorization_text"].strip()
    ):
        raise ValueError(
            "Explicit approval must bind package, directory, attempts, review level and budget"
        )
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
            if any(j["status"] != "complete" for j in state["jobs"].values()):
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
            store_meta = {}
            if spec["stage"] == "sqlite":
                store = directory / (spec["id"] + ".sqlite")
                facts = state["jobs"][extraction_id(spec)]["parsed"]
                write_store(store, case, facts)
                stored_facts, stored_sources = read_packet(store, case)
                if (stored_facts, stored_sources) != expected_packet(case, facts):
                    raise ValueError("Persistence changed the evidence packet")
                if request("sqlite", case, stored_facts, stored_sources) != req:
                    raise ValueError("Stored request differs from frozen derivation")
                store_meta = {"store_file": store.name, "store_sha256": file_sha(store)}
            if state["reserved_ninr"] + RESERVATION > binding["budget_ninr"]:
                raise ValueError("Budget would be exceeded before dispatch")
            attempt = {
                "status": "pending",
                "request": req,
                "request_sha256": sha(req),
                "reservation_ninr": RESERVATION,
                **store_meta,
            }
            state["jobs"][spec["id"]] = attempt
            state["reserved_ninr"] += RESERVATION
            # Durable reservation precedes the only outbound call. A killed pending
            # attempt remains unresolved on resume and is never silently resent.
            atomic(checkpoint, state)
            try:
                attempt["receipt"] = provider(req)
                atomic(checkpoint, state)  # Preserve raw receipt before any parsing.
                _, cost, receipt_id = receipt_usage(attempt["receipt"])
                attempt.update(usage_ninr=cost, provider_id=receipt_id)
                atomic(checkpoint, state)  # Invalid final JSON still consumed tokens.
                text, cost, receipt_id = parse_receipt(attempt["receipt"])
                for old in list(state["jobs"].values())[:-1]:
                    if old.get("provider_id") == receipt_id:
                        raise ValueError("Duplicate provider receipt ID")
                if spec["stage"] == "extract":
                    parsed = parse_facts(text, case)
                else:
                    source_ids = [r["id"] for r in loads(req["messages"][1]["content"])["records"]]
                    parsed = parse_answer(text, case, source_ids)
                attempt.update(
                    status="complete", parsed=parsed, usage_ninr=cost, provider_id=receipt_id
                )
                atomic(checkpoint, state)
            except Exception as exc:
                # Error type only: exception text can contain authentication data.
                attempt.update(status="error", error_type=type(exc).__name__)
                atomic(checkpoint, state)
                raise RuntimeError(f"Run stopped at {spec['id']}; reservation retained") from None
        summary = validate_state(package, state, directory)
        atomic(directory / "execution.json", summary)
        return summary
