"""Version three diagnostic: frozen development and internal semantic validation.

This is a development gate, not a LongMemEval accuracy experiment. The model
receives only the runtime projection; evaluator labels remain local.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import time
from pathlib import Path

from .english_screen.runner import OpenAIProvider, atomic, error_diagnostics
from .luna_evidence_plan import SETTINGS, canonical, opaque_fixture_id, project_fixture
from .luna_evidence_plan_v2 import input_bound
from .luna_evidence_plan_v3 import parse_plan, plan_request
from .luna_role_validation import review as role_review

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "benchmarks/fixtures/evidence_semantics_v1.json"
VALIDATION = ROOT / "benchmarks/fixtures/evidence_roles_validation_v3.json"
RATES = {"input": 200, "cache_write": 250, "output": 1200}  # nano-USD/token


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in (
            ROOT / "benchmarks/luna_evidence_plan.py",
            ROOT / "benchmarks/luna_plan_diagnostic_v3.py",
            ROOT / "benchmarks/luna_role_validation.py",
            ROOT / "benchmarks/luna_evidence_plan_v3.py",
            ROOT / "benchmarks/luna_evidence_plan_v2.py",
            ROOT / "benchmarks/evidence_focus.py",
            ROOT / "benchmarks/english_screen/runner.py",
            ROOT / "benchmarks/model_policy.py",
        )
    }


def fixture_cases():
    data = json.loads(FIXTURE.read_text())
    if data.get("status") != "PUBLISHED_DEVELOPMENT_FIXTURES_ONLY" or len(data["cases"]) != 13:
        raise ValueError("Unexpected development fixture population")
    fresh = json.loads(VALIDATION.read_text())
    if fresh.get("status") != "INTERNAL_VALIDATION_NO_MODEL_OUTPUTS" or len(fresh["cases"]) != 10:
        raise ValueError("Unexpected validation population")
    return data["cases"] + fresh["cases"]


def build_package():
    rows = []
    for case in fixture_cases():
        req = plan_request(project_fixture(case))
        rows.append({"id": case["id"], "request": req, "request_sha256": sha(canonical(req))})
    return {
        "schema": "luna-plan-diagnostic-v3",
        "purpose": "development-regression-and-internal-semantic-validation",
        "fixture_sha256": file_sha(FIXTURE),
        "validation_sha256": file_sha(VALIDATION),
        "code_sha256": code_hashes(),
        "settings": SETTINGS,
        "rates_nusd_per_token": RATES,
        "cases": rows,
        "accuracy": "NOT_MEASURED",
    }


def reservation(request):
    return (
        input_bound(request) * RATES["cache_write"]
        + request["max_completion_tokens"] * RATES["output"]
    )


def validate(package):
    if package != build_package():
        raise ValueError("Changed diagnostic package, fixture, request or code")
    ids = [row["id"] for row in package["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case")
    for row in package["cases"]:
        if input_bound(row["request"]) >= 272_000:
            raise ValueError("Outside frozen short-context price tier")
    return sha(canonical(package))


def preflight(package):
    identity = validate(package)
    amount = sum(reservation(row["request"]) for row in package["cases"])
    return {
        "package_sha256": identity,
        "model": SETTINGS["model"],
        "calls_no_retries": len(package["cases"]),
        "maximum_reservation_nusd": amount,
        "maximum_reservation_usd": amount / 1e9,
        "status": "OFFLINE_READY",
        "semantic_selection": "NOT_MEASURED",
        "english_accuracy": "NOT_MEASURED",
    }


def response_cost(request, response):
    # A truncated/refused output can still be billed. Acceptance is separate.
    model = response.get("model", "")
    if model != SETTINGS["model"] and not re.fullmatch(
        re.escape(SETTINGS["model"]) + r"-\d{4}-\d{2}-\d{2}", model
    ):
        raise ValueError("Unexpected returned model")
    if not response.get("id") or not response.get("request_id"):
        raise ValueError("Missing provider identity")
    usage = response.get("usage") or {}
    inp, out, total = (usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if not all(type(v) is int and v >= 0 for v in (inp, out, total)) or total != inp + out:
        raise ValueError("Invalid token accounting")
    if inp > input_bound(request) or out > SETTINGS["max_completion_tokens"]:
        raise ValueError("Usage exceeds reservation")
    details = usage.get("prompt_tokens_details") or {}
    cached, written = details.get("cached_tokens", 0), details.get("cache_write_tokens", 0)
    if not all(type(v) is int and v >= 0 for v in (cached, written)) or cached + written > inp:
        raise ValueError("Invalid cache usage")
    return inp * RATES["cache_write"] + out * RATES["output"]


def evaluate(package, state, *, complete=False):
    plan = preflight(package)
    binding = state["binding"]
    if binding["package_sha256"] != plan["package_sha256"]:
        raise ValueError("Checkpoint/package mismatch")
    jobs = state["jobs"]
    if set(jobs) - {c["id"] for c in package["cases"]}:
        raise ValueError("Unknown checkpoint case")
    if state["reserved_nusd"] != sum(j["reservation_nusd"] for j in jobs.values()):
        raise ValueError("Reservation ledger mismatch")
    if state["reserved_nusd"] > binding["budget_nusd"]:
        raise ValueError("Budget exceeded")
    fixture = {c["id"]: c for c in fixture_cases()}
    rows = []
    receipt_ids = set()
    for case in package["cases"]:
        job = jobs.get(case["id"])
        if not job:
            continue
        if job["request_sha256"] != case["request_sha256"] or job[
            "reservation_nusd"
        ] != reservation(case["request"]):
            raise ValueError("Changed job request or reservation")
        response = job.get("response")
        if response is None:
            if job["status"] == "complete":
                raise ValueError("Complete job has no response")
            rows.append({"id": case["id"], "status": job["status"], "usage_reconciled": False})
            continue
        if response["id"] in receipt_ids:
            raise ValueError("Duplicate provider receipt")
        receipt_ids.add(response["id"])
        charge = response_cost(case["request"], response)
        if job["usage_upper_nusd"] != charge or charge > job["reservation_nusd"]:
            raise ValueError("Invalid usage receipt")
        if job["status"] != "complete":
            rows.append(
                {
                    "id": case["id"],
                    "status": job["status"],
                    "usage_upper_nusd": charge,
                    "usage_reconciled": True,
                }
            )
            continue
        source = fixture[case["id"]]
        if response.get("finish_reason") != "stop":
            raise ValueError("Complete job has incomplete planner output")
        parsed = parse_plan(project_fixture(source), response["text"])
        mapping = {opaque_fixture_id(t["id"]): t["id"] for t in source["turns"]}
        support = [mapping[tid] for tid in parsed["support_turn_ids"]]
        qualification = [mapping[tid] for tid in parsed["qualification_turn_ids"]]
        rejected = [mapping[tid] for tid in parsed["rejected_turn_ids"]]
        review = role_review(source, support, qualification, rejected, parsed["sufficiency"])
        rows.append(
            {
                "id": case["id"],
                "status": "complete",
                "selection": review,
                "planner_sufficiency": parsed["sufficiency"],
                "rejected_original_ids": [mapping[tid] for tid in parsed["rejected_turn_ids"]],
                "support_original_ids": support,
                "qualification_original_ids": qualification,
                "selected_original_ids": [
                    t["id"] for t in source["turns"] if t["id"] in support + qualification
                ],
                "usage_upper_nusd": charge,
            }
        )
    if complete and (
        len(rows) != len(package["cases"]) or any(r["status"] != "complete" for r in rows)
    ):
        raise ValueError("Incomplete diagnostic")
    passed = sum(r.get("selection", {}).get("pass") is True for r in rows)
    return {
        "package_sha256": plan["package_sha256"],
        "mode": binding["mode"],
        "intended": len(package["cases"]),
        "attempted": len(jobs),
        "not_started": len(package["cases"]) - len(jobs),
        "failed_or_pending": sum(r["status"] != "complete" for r in rows),
        "completed": sum(r["status"] == "complete" for r in rows),
        "selection_passed": passed,
        "selection_failed": sum(r["status"] == "complete" for r in rows) - passed,
        "all_case_gate": "PASS"
        if len(rows) == len(package["cases"]) and passed == len(rows)
        else "FAIL_OR_INCOMPLETE",
        "population_results": {
            population: {
                "completed": sum(
                    r.get("selection", {}).get("population") == population for r in rows
                ),
                "passed": sum(
                    r.get("selection", {}).get("population") == population
                    and r["selection"]["pass"]
                    for r in rows
                ),
            }
            for population in ("exposed_development", "internal_validation", "capacity_awareness")
        },
        "english_accuracy": "NOT_MEASURED",
        "reserved_nusd": state["reserved_nusd"],
        "usage_upper_nusd": sum(r.get("usage_upper_nusd", 0) for r in rows),
        "unreconciled_usage_attempts": sum(r.get("usage_reconciled") is False for r in rows),
        "rows": rows,
    }


def run(package, directory, approval, *, mode="paid", provider=None):
    plan = preflight(package)
    directory = Path(directory).resolve()
    if (
        mode not in {"paid", "offline-test"}
        or approval.get("approved") is not True
        or approval.get("mode") != mode
        or approval.get("package_sha256") != plan["package_sha256"]
        or approval.get("maximum_attempts") != plan["calls_no_retries"]
        or type(approval.get("budget_nusd")) is not int
        or approval["budget_nusd"] < plan["maximum_reservation_nusd"]
        or not approval.get("authorization_text")
    ):
        raise ValueError("Exact package, attempts and budget authorization required")
    if mode == "paid" and (
        provider is not None or approval.get("output_directory") != str(directory)
    ):
        raise ValueError("Paid execution must bind one output directory and real provider")
    if mode == "offline-test" and provider is None:
        raise ValueError("Offline test requires fake provider")
    directory.mkdir(parents=True, exist_ok=True)
    binding = {
        "package_sha256": plan["package_sha256"],
        "approval_sha256": sha(canonical(approval)),
        "budget_nusd": approval["budget_nusd"],
        "mode": mode,
    }
    with (directory / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = directory / "checkpoint.json"
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state["binding"] != binding:
                raise ValueError("Changed resume binding")
            evaluate(package, state)
            if any(j["status"] != "complete" for j in state["jobs"].values()):
                raise ValueError("Unresolved attempt; no automatic retry")
        else:
            state = {"binding": binding, "jobs": {}, "reserved_nusd": 0}
            atomic(checkpoint, state)
        for case in package["cases"]:
            if case["id"] in state["jobs"]:
                continue
            request = case["request"]
            cost = reservation(request)
            if state["reserved_nusd"] + cost > approval["budget_nusd"]:
                raise ValueError("Budget stop before dispatch")
            job = {
                "status": "pending",
                "request_sha256": case["request_sha256"],
                "reservation_nusd": cost,
                "started_unix": time.time(),
            }
            state["jobs"][case["id"]] = job
            state["reserved_nusd"] += cost
            atomic(checkpoint, state)
            try:
                if provider is None:
                    provider = OpenAIProvider()
                response = provider(request)
                job["response"] = response
                job["usage_upper_nusd"] = response_cost(request, response)
                if response.get("finish_reason") != "stop":
                    raise ValueError("Incomplete planner output")
                # Validate schema and source membership before another dispatch.
                parse_plan(
                    project_fixture(next(c for c in fixture_cases() if c["id"] == case["id"])),
                    response["text"],
                )
                job["status"] = "complete"
                evaluate(package, state)
            except Exception as error:
                job["status"] = "error"
                job["error_class"] = type(error).__name__
                job["error_diagnostics"] = error_diagnostics(error)
                atomic(checkpoint, state)
                raise RuntimeError("Attempt failed; preserved; no automatic retry") from None
            finally:
                job["elapsed_seconds"] = time.time() - job["started_unix"]
                atomic(checkpoint, state)
        summary = evaluate(package, state, complete=True)
        atomic(directory / "summary.json", summary)
        return summary
