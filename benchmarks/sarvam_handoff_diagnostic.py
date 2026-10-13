"""Prospective two-call diagnostic for the one observed v3 capped native actor.

Preparation and audit are offline. Run requires explicit ``allow_paid``; this
module does not retry, change bookings, or replace any parent result. Caller
credentials are read only on the paid path. Gold bytes are parsed only in audit.
"""

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

from benchmarks.sarvam_native_controls.engine import ControlMeter
from benchmarks.sarvam_native_controls.transcript_finalizer import classify
from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.transport import (
    ENDPOINT,
    INPUT,
    MODELS,
    OUTPUT,
    call,
    digest,
    read_key,
    save,
)
from benchmarks.sarvam_receipt_reporting import project
from benchmarks.sarvam_reporting_handoff import dispatch_once, prepare_plan

EXPECTED_ELIGIBLE = "confirm-fresh-13-en-m0"  # predeclared scope check, never score-based selection
EXPECTED_PLAN_SHA256 = "0507096c37b93ede6b7392782a148a0ee8adbec36fb152b62ecc264da34cdfb5"
EXPECTED_REQUEST_SHA256 = {
    "control": "c69ec849f387fe37c0d8c81c61fc95a6b7fb948db1fec006e0f49c1ababadc22",
    "candidate": "218055cdd59e274a4078915c4e9e8c65d962a018599ef93a23d4c9de834a7ac6",
}
PARENT_JOBS = 240
PRIOR_CALLS = 7035
PRIOR_KNOWN = 270_657_780
PRIOR_RETAINED = 370_580_775
PHASE_CAP = 8_000_000
CAMPAIGN_CAP = 6_000_000_000
STAGE = "handoff"
CODE = (
    "benchmarks/sarvam_handoff_diagnostic.py",
    "benchmarks/sarvam_reporting_handoff.py",
    "benchmarks/sarvam_evidence_views.py",
    "benchmarks/sarvam_ordinary_v2.py",
    "benchmarks/sarvam_ordinary_study.py",
    "benchmarks/sarvam_reporter_wording.py",
    "benchmarks/sarvam_receipt_reporting.py",
    "benchmarks/sarvam_native_controls/engine.py",
    "benchmarks/sarvam_native_controls/integrated.py",
    "benchmarks/sarvam_native_controls/transcript_finalizer.py",
    "benchmarks/sarvam_native_controls/finalization_transport.py",
    "benchmarks/sarvam_native_controls/separated.py",
    "benchmarks/sarvam_native_controls/approval_backend.py",
    "benchmarks/sarvam_native_study/backend.py",
    "benchmarks/sarvam_native_study/runner.py",
    "benchmarks/sarvam_native_study/diagnostic.py",
    "benchmarks/sarvam_native_study/transport.py",
    "benchmarks/sarvam_native_study/paced.py",
    "benchmarks/sarvam_native_study/score.py",
)
PARENT_FILES = (
    "manifest.json",
    "execution.json",
    "verification.json",
    "runtime.json",
    "budget.json",
    "review.json",
    "PROTOCOL.md",
    "pacing-control.json",
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _read(path):
    return json.loads(Path(path).read_text())


def _strict_json(source):
    def pairs(items):
        obj = {}
        for key, value in items:
            _require(key not in obj, f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj

    return json.loads(source, object_pairs_hook=pairs)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_once(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as out:
        out.write(Path(source).read_bytes())
        out.flush()
        os.fsync(out.fileno())


def _code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: _sha(root / name) for name in CODE}


def _parent_files(parent, *, gold_path=None, full_call_inventory=False):
    parent = Path(parent)
    manifest = _read(parent / "manifest.json")
    _require(len(manifest.get("jobs", [])) == PARENT_JOBS, "Parent manifest is not 240 jobs")
    _require(len({j["id"] for j in manifest["jobs"]}) == PARENT_JOBS, "Duplicate parent job")
    for name, expected in manifest["files_sha256"].items():
        source = gold_path if name == "gold.json" and gold_path is not None else parent / name
        _require(_sha(source) == expected, f"Parent frozen input changed: {name}")
    for name, expected in manifest["source_sha256"].items():
        _require(
            digest((parent / "source" / name).read_text()) == expected,
            f"Parent source changed: {name}",
        )
    execution = _read(parent / "execution.json")
    verification = _read(parent / "verification.json")
    # The parent acceptance/completeness gate failed because these two reports
    # were not reached; every episode and raw execution record still exists.
    _require(
        verification.get("verified") is True
        and verification.get("complete") is False
        and verification.get("mechanical_screen_pass") is False,
        "Parent failed verification gate changed",
    )
    _require(
        verification.get("native_episodes") == PARENT_JOBS, "Parent native episode coverage changed"
    )
    _require(len(execution.get("outcomes", [])) == PARENT_JOBS, "Parent execution incomplete")
    _require(
        verification["research_calls"] == PRIOR_CALLS
        and round(verification["research_known_inr"] * 1_000_000) == PRIOR_KNOWN
        and round(verification["research_retained_inr"] * 1_000_000) == PRIOR_RETAINED
        and verification["unknown_calls"] == 0,
        "Parent cumulative ledger changed",
    )
    if full_call_inventory:
        _require(
            verification["calls"] == len(list((parent / "calls").glob("*.request.json"))),
            "Parent request inventory mismatch",
        )
    runtime = {c["id"]: c for c in _read(parent / "runtime.json")}
    return manifest, execution, verification, runtime


def _census(manifest, execution, verification, runtime, paired_dir, calls_dir):
    """Recheck all 240 routing decisions from raw paired files, never from scores."""
    outcomes = {row["id"]: row for row in execution["outcomes"]}
    _require(len(outcomes) == PARENT_JOBS, "Duplicate execution outcome")
    scores = {(row["id"], row["variant"]): row for row in verification["rows"]}
    _require(len(scores) == PARENT_JOBS * 2, "Parent verification row coverage mismatch")
    ids = {j["id"] for j in manifest["jobs"]}
    _require(
        set(outcomes) == ids and {p.stem for p in paired_dir.glob("*.json")} == ids,
        "Parent paired inventory mismatch",
    )
    both_not_reached, rows = [], []
    for job in manifest["jobs"]:
        eid = job["id"]
        pair_path = paired_dir / f"{eid}.json"
        pair = _read(pair_path)
        original = pair["original"]
        _require(
            original["id"] == eid
            and original["case_id"] == job["case_id"]
            and original["model"] == job["model"]
            and original["initial_sha256"] == job["initial_sha256"],
            f"Parent paired identity mismatch: {eid}",
        )
        _require(
            pair["actual_calls"] == outcomes[eid]["actual_calls"],
            f"Parent actual-call mismatch: {eid}",
        )
        for arm in ("control", "candidate"):
            _require(
                pair["reports"][arm]["terminal"] == scores[(eid, arm)]["terminal"],
                f"Parent report/verification mismatch: {eid}/{arm}",
            )
        _require(
            job["case_id"] in runtime and runtime[job["case_id"]]["id"] == job["case_id"],
            f"Parent runtime case absent: {eid}",
        )
        missed = all(
            pair["reports"][arm]["terminal"] == "not_reached" for arm in ("control", "candidate")
        )
        rows.append(
            {
                "id": eid,
                "paired_sha256": _sha(pair_path),
                "both_reporters_not_reached": missed,
                "actor_terminal": original["terminal"],
            }
        )
        if missed:
            both_not_reached.append((job, pair))
    _require(len(rows) == PARENT_JOBS, "Census incomplete")
    eligible, refusals = [], []
    for job, pair in both_not_reached:
        eid = job["id"]
        actor_calls = [x for x in pair["actual_calls"] if x.startswith(eid + "-a")]
        _require(actor_calls and len(actor_calls) <= 3, f"Missing bounded actor call: {eid}")
        actor_job = actor_calls[-1]
        request_file = calls_dir / f"{actor_job}.request.json"
        receipt_file = calls_dir / f"{actor_job}.receipt.json"
        try:
            recorded = _read(request_file)
            _require(
                recorded["endpoint"] == ENDPOINT
                and recorded["sha256"] == digest(recorded["request"]),
                "Recorded actor request mismatch",
            )
            receipt = _read(receipt_file)
            plan = prepare_plan(
                actor_job=actor_job,
                actor_request=recorded["request"],
                native_receipt=receipt,
                tool_trace=pair["original"]["tool_trace"],
                initial_backend=pair["original"]["initial_backend"],
                final_backend=pair["original"]["final_backend"],
                actor_calls_used=len(actor_calls),
                caller_scope=runtime[job["case_id"]]["caller"],
            )
            _require(
                pair["original"]["terminal"] == "invalid_action_response",
                "Parent actor was not an invalid action response",
            )
            _require(pair["actual_calls"] == actor_calls, "Parent reporters were already called")
            _require(
                pair["original"]["calls"] == actor_calls, "Original native call record differs"
            )
            _require(
                job["report_order"] == ["control", "candidate"],
                "Predeclared reporter order differs",
            )
            eligible.append(
                {
                    "id": eid,
                    "actor_job": actor_job,
                    "actor_request_file": request_file,
                    "actor_receipt_file": receipt_file,
                    "plan": plan,
                }
            )
        except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
            refusals.append({"id": eid, "reason": type(exc).__name__ + ": " + str(exc)})
    _require(
        [x["id"] for x in eligible] == [EXPECTED_ELIGIBLE],
        "Eligible capped-actor set differs from frozen protocol",
    )
    return rows, eligible[0], refusals


def _budget(model):
    amount = (MODELS[model] * INPUT + 8192 * OUTPUT) * 5 // 4
    _require(
        2 * amount <= PHASE_CAP and PRIOR_RETAINED + PHASE_CAP <= CAMPAIGN_CAP,
        "Diagnostic budget insufficient",
    )
    return {
        "phase_cap_microinr": PHASE_CAP,
        "stages": {STAGE: {"calls": 2, "cap_microinr": PHASE_CAP}},
        "prior_calls": PRIOR_CALLS,
        "prior_known_microinr": PRIOR_KNOWN,
        "prior_retained_microinr": PRIOR_RETAINED,
        "campaign_cap_microinr": CAMPAIGN_CAP,
        "rates_microinr": {"input": INPUT, "output": OUTPUT},
        "full_context_per_call_reservation_microinr": amount,
        "two_call_maximum_reservation_microinr": 2 * amount,
    }


def prepare(parent_root, diagnostic_root):
    """Offline census and portable freeze; no gold parsing, key read or HTTP."""
    parent, root = Path(parent_root), Path(diagnostic_root)
    _require(
        root.is_dir() and (root / "PROTOCOL.md").is_file(),
        "Existing diagnostic directory and protocol required",
    )
    _require(
        not any((root / name).exists() for name in ("manifest.json", "started.json", "calls")),
        "Diagnostic already prepared or attempted",
    )
    manifest, execution, verification, runtime = _parent_files(parent, full_call_inventory=True)
    # Parent paired files are not copied yet; this is a read-only census.
    paired_source = parent / "episodes"
    temporary_pairs = {}
    for job in manifest["jobs"]:
        q = paired_source / job["id"] / "paired.json"
        _require(q.is_file(), f"Parent paired file missing: {job['id']}")
        temporary_pairs[job["id"]] = q

    # Provide the census a flat, read-only view without creating files.
    class PairedView:
        def __truediv__(self, name):
            return temporary_pairs[name.removesuffix(".json")]

        def glob(self, _pattern):
            return [Path(eid + ".json") for eid in temporary_pairs]

    census, selected, refusals = _census(
        manifest, execution, verification, runtime, PairedView(), parent / "calls"
    )
    plan = selected["plan"]
    _require(
        plan["plan_sha256"] == EXPECTED_PLAN_SHA256
        and {arm: plan["reports"][arm]["request_sha256"] for arm in ("control", "candidate")}
        == EXPECTED_REQUEST_SHA256,
        "Observed reporter requests differ from predeclared protocol",
    )
    budget = _budget(plan["reports"]["control"]["request"]["model"])
    _require(
        all(
            plan["reports"][arm]["request"]["model"] == "sarvam-105b"
            for arm in ("control", "candidate")
        ),
        "Frozen model changed",
    )
    # Everything below is create-once. Gold is opaque bytes until audit.
    parent_copy = root / "parent"
    for name in PARENT_FILES:
        _copy_once(parent / name, parent_copy / name)
    _copy_once(parent / "gold.json", root / "sealed" / "gold.json")
    for name in manifest["source_sha256"]:
        _copy_once(parent / "source" / name, parent_copy / "source" / name)
    for row in census:
        _copy_once(temporary_pairs[row["id"]], parent_copy / "paired" / f"{row['id']}.json")
    actor_job = selected["actor_job"]
    for suffix in ("request", "receipt"):
        _copy_once(
            parent / "calls" / f"{actor_job}.{suffix}.json",
            parent_copy / "calls" / f"{actor_job}.{suffix}.json",
        )
    save(root / "plan.json", plan)
    save(root / "budget.json", budget)
    save(root / "pacing-control.json", {"rpm": 35, "basis": "below Starter40"})
    frozen = [
        root / "PROTOCOL.md",
        root / "plan.json",
        root / "budget.json",
        root / "pacing-control.json",
    ]
    frozen += [q for q in (root / "parent").rglob("*") if q.is_file()]
    frozen += [root / "sealed" / "gold.json"]
    package = {
        "kind": "OBSERVED_CAPPED_ACTOR_TWO_REPORTER_DIAGNOSTIC_NOT_V3_REPAIR",
        "selected_id": selected["id"],
        "actor_job": actor_job,
        "jobs": [plan["reports"][arm]["job"] for arm in ("control", "candidate")],
        "request_sha256": {
            arm: plan["reports"][arm]["request_sha256"] for arm in ("control", "candidate")
        },
        "plan_sha256": digest(plan),
        "parent_census": census,
        "parent_census_sha256": digest(census),
        "refusals": refusals,
        "snapshot_sha256": {str(q.relative_to(root)): _sha(q) for q in sorted(frozen)},
        "code_sha256": _code_hashes(),
        "maximum_new_calls": 2,
        "new_native_or_tool_calls": 0,
        "no_retries": True,
        "gold_parsed_in_prepare": False,
    }
    save(root / "manifest.json", package)
    return package


def _load(root):
    """Replay the complete portable parent census and plan before run or audit."""
    root = Path(root)
    package = _read(root / "manifest.json")
    _require(package["code_sha256"] == _code_hashes(), "Diagnostic code changed")
    for name, expected in package["snapshot_sha256"].items():
        _require(_sha(root / name) == expected, f"Frozen diagnostic input changed: {name}")
    parent = root / "parent"
    manifest, execution, verification, runtime = _parent_files(
        parent, gold_path=root / "sealed" / "gold.json"
    )
    census, selected, refusals = _census(
        manifest, execution, verification, runtime, parent / "paired", parent / "calls"
    )
    _require(
        census == package["parent_census"]
        and digest(census) == package["parent_census_sha256"]
        and refusals == package["refusals"],
        "Portable parent census changed",
    )
    plan = selected["plan"]
    _require(
        plan == _read(root / "plan.json")
        and digest(plan) == package["plan_sha256"]
        and selected["id"] == package["selected_id"]
        and [plan["reports"][arm]["job"] for arm in ("control", "candidate")] == package["jobs"],
        "Prepared reporting plan changed",
    )
    _require(
        _read(root / "budget.json") == _budget(plan["reports"]["control"]["request"]["model"]),
        "Budget changed",
    )
    return package, plan, selected, manifest, execution, verification


def run(
    root, *, allow_paid=False, transport_call=call, credential_loader=read_key, pacer_factory=Pacer
):
    """Prospective dispatch; the default is a refusal and tests inject a fake transport."""
    package, plan, _selected, _manifest, _execution, _verification = _load(root)
    if not allow_paid:
        raise ValueError("Paid diagnostic requires explicit --allow-paid")
    root = Path(root)
    _require(
        not any(
            (root / name).exists()
            for name in ("started.json", "execution.json", "handoff-claim.json")
        ),
        "Diagnostic already started; no resume or retry",
    )
    # Preflight both fixed requests and full context budget before credential access.
    for arm in ("control", "candidate"):
        req = plan["reports"][arm]["request"]
        _require(
            digest(req) == package["request_sha256"][arm]
            and "tools" not in req
            and "tool_choice" not in req,
            "Reporter request changed or executable",
        )
    _budget(plan["reports"]["control"]["request"]["model"])
    key = credential_loader()
    meter, pacer = ControlMeter(root), pacer_factory(root)
    execution_mode = (
        "production_default_transport"
        if transport_call is call and credential_loader is read_key and pacer_factory is Pacer
        else "injected_offline_transport"
    )
    save(
        root / "started.json",
        {
            "manifest_sha256": digest(package),
            "plan_sha256": digest(plan),
            "maximum_new_calls": 2,
            "execution_mode": execution_mode,
        },
    )
    prior_failure = False
    not_attempted = set()

    def dispatch(job, request):
        nonlocal prior_failure
        if prior_failure:
            not_attempted.add(job)
            raise RuntimeError("Prior infrastructure or usage failure; no second dispatch")
        try:
            timing = pacer.wait()
            save(root / "calls" / f"{job}.dispatch.json", timing)
            body = transport_call(meter, job, STAGE, request, key)
            classify(body)  # malformed infrastructure body stops the second provider call
            return body
        except Exception:
            prior_failure = True
            raise

    result = dispatch_once(plan, dispatch, set(), root / "handoff-claim.json")
    for arm in ("control", "candidate"):
        if (
            result["reports"][arm].get("terminal") == "infrastructure_error"
            and plan["reports"][arm]["job"] in not_attempted
        ):
            result["reports"][arm] = {
                "terminal": "not_attempted_after_prior_failure",
                "parsed": None,
                "application": None,
            }
    result["parent_selected_id"] = package["selected_id"]
    result["meter"] = meter.summary()
    save(root / "execution.json", result)
    return result


def _usage(receipt, request):
    """Independent per-call usage parsing; unknown retains full reservation."""
    try:
        body = _strict_json(receipt["body_text"])
        usage = body["usage"]
        inp, out, total = (usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        known = (
            all(type(v) is int and v >= 0 for v in (inp, out, total))
            and total == inp + out
            and inp <= MODELS[request["model"]]
            and out <= request["max_tokens"]
        )
    except (KeyError, ValueError, TypeError):
        body, inp, out, known = {}, None, None, False
    reservation = (MODELS[request["model"]] * INPUT + request["max_tokens"] * OUTPUT) * 5 // 4
    cost = inp * INPUT + out * OUTPUT if known else None
    retained = (cost * 5 + 3) // 4 if known else reservation
    return body, {
        "known": known,
        "prompt_tokens": inp,
        "completion_tokens": out,
        "known_cost_microinr": cost,
        "retained_microinr": retained,
        "full_reservation_microinr": reservation,
    }


def audit(root, *, output=None):
    """Offline replay and scoring; this is the sole function that parses copied gold."""
    root = Path(root)
    package, plan, selected, _manifest, _parent_execution, parent_verification = _load(root)
    result = _read(root / "execution.json")
    started = _read(root / "started.json")
    _require(
        started["manifest_sha256"] == digest(package)
        and started["plan_sha256"] == digest(plan)
        and started["maximum_new_calls"] == 2,
        "Started marker mismatch",
    )
    _require(
        _read(root / "handoff-claim.json")["plan_sha256"] == plan["plan_sha256"],
        "Handoff claim mismatch",
    )
    _require(result["parent_selected_id"] == selected["id"], "Execution selected wrong parent")
    meter_path = root / "calls" / "meter.sqlite"
    _require(meter_path.is_file(), "No metered execution")
    with sqlite3.connect(f"file:{meter_path}?mode=ro", uri=True) as db:
        meter_rows = {
            r[0]: r
            for r in db.execute("SELECT id,stage,reservation,charged,usage_cost,status FROM calls")
        }
    _require(len(meter_rows) <= 2, "More than two metered calls")
    reports, attempts = {}, []
    previous_failed = False
    for arm in ("control", "candidate"):
        job = plan["reports"][arm]["job"]
        request_path = root / "calls" / f"{job}.request.json"
        receipt_path = root / "calls" / f"{job}.receipt.json"
        observed = result["reports"][arm]
        if not request_path.exists():
            _require(not receipt_path.exists(), "Receipt exists without reporter request")
            if not previous_failed:
                _require(
                    arm == "control" and observed["terminal"] == "infrastructure_error",
                    "Missing reporter call without prior failure",
                )
                previous_failed = True
            else:
                _require(
                    observed["terminal"] == "not_attempted_after_prior_failure",
                    "Missing later reporter call has wrong terminal",
                )
            reports[arm] = {
                "terminal": observed["terminal"],
                "usage": None,
                "raw": None,
                "application": None,
            }
            continue
        _require(
            not previous_failed and receipt_path.exists() and job in meter_rows,
            "Unexpected or unmetered reporter call",
        )
        request_record = _read(request_path)
        _require(
            request_record["endpoint"] == ENDPOINT
            and request_record["request"] == plan["reports"][arm]["request"]
            and request_record["sha256"]
            == digest(request_record["request"])
            == plan["reports"][arm]["request_sha256"],
            "Recorded reporter request mismatch",
        )
        receipt = _read(receipt_path)
        _require(
            receipt["job"] == job and receipt["request_sha256"] == request_record["sha256"],
            "Reporter receipt/request mismatch",
        )
        body, usage = _usage(receipt, request_record["request"])
        row = meter_rows[job]
        _require(
            row[1] == STAGE
            and row[2] == usage["full_reservation_microinr"]
            and row[3] == usage["retained_microinr"]
            and row[4] == usage["known_cost_microinr"]
            and row[5] == ("known" if usage["known"] else "unknown"),
            "Meter/receipt usage mismatch",
        )
        good_transport = (
            receipt.get("status") == 200
            and usage["known"]
            and body.get("model") == request_record["request"]["model"]
        )
        if good_transport:
            try:
                replay = classify(body)
                replay["application"] = None
                if replay.get("parsed") is not None:
                    try:
                        replay["application"] = project(
                            **plan["projection_inputs"],
                            raw_content=replay["content"],
                            completed=True,
                        )
                    except (ValueError, KeyError, TypeError) as exc:
                        replay["application_error"] = type(exc).__name__
            except (ValueError, KeyError, TypeError, IndexError):
                good_transport = False
            else:
                _require(
                    all(
                        replay.get(key) == observed.get(key)
                        for key in (
                            "terminal",
                            "content",
                            "finish_reason",
                            "tool_calls",
                            "parsed",
                            "application",
                            "application_error",
                        )
                    ),
                    "Saved reporter output does not replay",
                )
        if not good_transport:
            _require(
                observed["terminal"] == "infrastructure_error",
                "Infrastructure/unknown usage not preserved",
            )
            previous_failed = True
        attempts.append(
            {
                "job": job,
                "arm": arm,
                "request_sha256": request_record["sha256"],
                "receipt_sha256": _sha(receipt_path),
                "http_status": receipt.get("status"),
                "usage": usage,
                "terminal": observed["terminal"],
                "elapsed_seconds": receipt.get("elapsed_seconds"),
            }
        )
        reports[arm] = {
            "terminal": observed["terminal"],
            "raw": observed.get("parsed"),
            "application": observed.get("application"),
            "usage": usage,
        }
    _require(set(meter_rows) == {a["job"] for a in attempts}, "Meter call coverage mismatch")
    _require(
        {p.name for p in (root / "calls").glob("*.request.json")}
        == {a["job"] + ".request.json" for a in attempts}
        and {p.name for p in (root / "calls").glob("*.receipt.json")}
        == {a["job"] + ".receipt.json" for a in attempts},
        "Reporter request/receipt inventory mismatch",
    )
    _require(len(attempts) <= package["maximum_new_calls"], "Call ceiling exceeded")
    # Gold remains opaque until all request/receipt/usage checks above pass.
    gold = _read(root / "sealed" / "gold.json")
    original = _read(root / "parent" / "paired" / f"{selected['id']}.json")["original"]
    runtime = {c["id"]: c for c in _read(root / "parent" / "runtime.json")}
    label = gold[original["case_id"]]
    _require(
        label["runtime_sha256"] == digest(runtime[original["case_id"]])
        and label["question_target"] == "source_request_or_condition"
        and label["expected_answer"] == {"answer": None, "needs_clarification": True}
        and label["forbid_new_write"] is True,
        "Selected source/gold contract changed",
    )
    expected = label["expected_answer"]
    for arm in ("control", "candidate"):
        raw = reports[arm]["raw"]
        projected = reports[arm]["application"]
        reports[arm]["raw_action_status"] = raw.get("action_status") if raw else None
        reports[arm]["receipt_owned_action_status"] = (
            projected["application_output"]["action_status"] if projected else None
        )
        reports[arm]["raw_status_matches_read_only"] = bool(
            raw and raw.get("action_status") == "not_requested"
        )
        reports[arm]["receipt_owned_status_matches_read_only"] = bool(
            projected and projected["application_output"]["action_status"] == "not_requested"
        )
        reports[arm]["date_only_match"] = bool(raw and raw["answer"] == expected["answer"])
        reports[arm]["clarification_match"] = bool(
            raw and raw["needs_clarification"] == expected["needs_clarification"]
        )
        reports[arm]["factual_match"] = (
            reports[arm]["date_only_match"] and reports[arm]["clarification_match"]
        )
    prior_rows = [r for r in parent_verification["rows"] if r["id"] == selected["id"]]
    _require(
        len(prior_rows) == 2 and all(r["terminal"] == "not_reached" for r in prior_rows),
        "Original failure was overwritten",
    )
    known = sum(a["usage"]["known_cost_microinr"] or 0 for a in attempts)
    retained = sum(a["usage"]["retained_microinr"] for a in attempts)
    report = {
        "kind": "PROSPECTIVE_CAPPED_ACTOR_DIAGNOSTIC_NOT_PARENT_REPAIR",
        "parent_selected_id": selected["id"],
        "parent_reporters_not_reached": True,
        "execution_mode": started["execution_mode"],
        "live_provider_observation": started["execution_mode"] == "production_default_transport",
        "selected_runtime_sha256": digest(runtime[original["case_id"]]),
        "sealed_gold_sha256": _sha(root / "sealed" / "gold.json"),
        "new_calls": len(attempts),
        "new_native_or_tool_calls": 0,
        "reports": reports,
        "attempts": attempts,
        "candidate_primary_factual_match": reports["candidate"]["factual_match"],
        "known_usage_all": all(a["usage"]["known"] for a in attempts) and len(attempts) == 2,
        "new_known_microinr": known,
        "new_retained_microinr": retained,
        "cumulative_known_microinr": PRIOR_KNOWN + known,
        "cumulative_retained_microinr": PRIOR_RETAINED + retained,
        "safe_backend_unchanged": original["initial_backend"] == original["final_backend"],
        "preserved_parent_snapshot_replayed": True,
        "not_matched_actual_compute": True,
        "one_observed_english_base_case_only": True,
    }
    if output is not None:
        save(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("parent")
    prep.add_argument("root")
    running = sub.add_parser("run")
    running.add_argument("root")
    running.add_argument("--allow-paid", action="store_true")
    checking = sub.add_parser("audit")
    checking.add_argument("root")
    checking.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.parent, args.root)
    elif args.command == "run":
        run(args.root, allow_paid=args.allow_paid)
    else:
        result = audit(args.root, output=args.output)
        if args.output is None:
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
