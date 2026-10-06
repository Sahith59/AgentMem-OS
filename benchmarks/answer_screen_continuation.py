"""Bind a new normalized experiment to immutable, already billed responses."""

import copy
import json
from pathlib import Path

from . import guarded_answer_screen as old
from . import guarded_answer_screen_normalized as new
from .english_screen.runner import atomic
from .evidence_role_normalization import normalize


def load_sources(records):
    data = {}
    for name, record in records.items():
        raw = Path(record["path"]).read_bytes()
        import hashlib

        if hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("Changed inherited artifact")
        data[name] = json.loads(raw)
    return data


def inherit(target, sources):
    if (
        target["continuation"]["source_checkpoint_sha256"]
        != target["continuation"]["sources"]["checkpoint"]["sha256"]
    ):
        raise ValueError("Checkpoint provenance hash mismatch")
    package, approval, state = (sources[k] for k in ("package", "approval", "checkpoint"))
    audit = old.verify(package, state)
    if (
        state["binding"]["approval_sha256"] != old.sha(old.canonical(approval))
        or state["binding"]["mode"] != approval["mode"]
        or not approval.get("approved")
        or audit["unreconciled_usage_attempts"]
    ):
        raise ValueError("Unverified source approval or usage")
    for field in (
        "cases",
        "answer_prompt",
        "answer_settings",
        "planner_settings",
        "judge_settings",
        "judge_rates",
        "gate",
        "request_bounds_nusd",
        "maximum_attempts",
        "maximum_reservation_nusd",
    ):
        if target[field] != package[field]:
            raise ValueError("Continuation changes population, inference or evaluation")
    jobs = copy.deepcopy(state["jobs"])
    cases = {c["id"]: c for c in target["cases"]}
    for key, job in jobs.items():
        cid, stage = key.rsplit(":", 1)
        if job["status"] != "complete":
            if job["status"] != "error" or stage != "plan" or not job.get("response"):
                raise ValueError("Cannot inherit unresolved dispatch")
            normalized, receipt = normalize(new.runtime(cases[cid]), job["response"]["text"])
            if not receipt["changed"]:
                raise ValueError("Only observed cross-role overlap is recoverable")
            job["normalization"] = receipt
        new.accept(cases[cid], stage, job["response"])
        job["inherited_original_status"] = job["status"]
        job["status"] = "complete"
        job["inherited_from"] = target["continuation"]["source_checkpoint_sha256"]
    if (
        sorted(jobs) != sorted(target["continuation"]["inherited_jobs"])
        or state["reserved_nusd"] != target["continuation"]["source_reserved_nusd"]
    ):
        raise ValueError("Changed inherited job inventory")
    return jobs


def bootstrap(package, directory, approval, *, mode="paid"):
    identity = new.validate(package)
    directory = Path(directory).resolve()
    c = package["continuation"]
    if (
        mode not in ("paid", "offline-test")
        or approval.get("approved") is not True
        or approval.get("mode") != mode
        or approval.get("package_sha256") != identity
        or approval.get("output_directory") != str(directory)
        or approval.get("maximum_attempts") != package["maximum_attempts"]
        or approval.get("maximum_new_attempts")
        != package["maximum_attempts"] - len(c["inherited_jobs"])
        or approval.get("additional_budget_nusd") != c["additional_cap_nusd"]
        or approval.get("budget_nusd") != c["source_reserved_nusd"] + c["additional_cap_nusd"]
        or not approval.get("authorization_text")
    ):
        raise ValueError("Exact continuation approval required")
    sources = load_sources(c["sources"])
    if mode == "paid" and sources["checkpoint"]["binding"]["mode"] != "paid":
        raise ValueError("Paid continuation cannot inherit fake responses")
    jobs = inherit(package, sources)
    state = dict(
        binding=dict(
            package_sha256=identity,
            approval_sha256=old.sha(old.canonical(approval)),
            budget_nusd=approval["budget_nusd"],
            mode=mode,
        ),
        jobs=jobs,
        reserved_nusd=c["source_reserved_nusd"],
    )
    new.verify(package, state)
    directory.mkdir(parents=True, exist_ok=False)  # never overwrite a prior checkpoint
    atomic(directory / "checkpoint.json", state)
    return state
