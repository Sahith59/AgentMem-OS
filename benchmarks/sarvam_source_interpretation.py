"""Prospective all-history source interpretation screen; no actor or tool calls.

Preparation and audit are offline. Paid execution is opt-in, one fixed reporter
request per job, with no continuation after any attempt. The saved histories are
development exposed; this module does not claim a memory or model improvement.
"""

import argparse
import json
import random
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from pathlib import Path

from benchmarks import sarvam_evidence_ablation as ablation
from benchmarks.sarvam_fresh_confirmation.engine import report_view
from benchmarks.sarvam_handoff_diagnostic import (
    _census,
    _copy_once,
    _parent_files,
    _sha,
    _usage,
)
from benchmarks.sarvam_native_controls.engine import PROFILES, ControlMeter
from benchmarks.sarvam_native_controls.transcript_finalizer import classify
from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.transport import (
    ENDPOINT,
    call,
    digest,
    read_key,
    save,
)
from benchmarks.sarvam_receipt_reporting import project
from benchmarks.sarvam_reporter_wording import reservation
from benchmarks.sarvam_reporting_handoff import prepare_plan
from benchmarks.sarvam_source_attribution import evaluate

ARMS = ablation.ARMS
STAGE = "source_interpretation"
PARENT_EPISODES = 240
PLANNED_CALLS = 720
FULL_RESERVATION = 1_548_288_000
PHASE_CAP = 1_550_000_000
PRIOR_CALLS = 7037
PRIOR_KNOWN = 271_224_630
PRIOR_RETAINED = 371_289_339
CAMPAIGN_CAP = 6_000_000_000
ORDER_SEED = 202610134
CODE = (
    "benchmarks/sarvam_source_interpretation.py",
    "benchmarks/sarvam_evidence_ablation.py",
    "benchmarks/sarvam_evidence_views.py",
    "benchmarks/sarvam_handoff_diagnostic.py",
    "benchmarks/sarvam_reporting_handoff.py",
    "benchmarks/sarvam_reporter_wording.py",
    "benchmarks/sarvam_ordinary_v2.py",
    "benchmarks/sarvam_ordinary_study.py",
    "benchmarks/sarvam_receipt_reporting.py",
    "benchmarks/sarvam_source_attribution.py",
    "benchmarks/sarvam_fresh_confirmation/engine.py",
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
    "started.json",
    "execution.json",
    "verification.json",
    "runtime.json",
    "budget.json",
    "review.json",
    "PROTOCOL.md",
    "pacing-control.json",
    "semantic-review.json",
)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read(path):
    return json.loads(Path(path).read_text())


def _code_hashes():
    repo = Path(__file__).resolve().parents[1]
    return {name: _sha(repo / name) for name in CODE}


def _source_inventory(parent, manifest, execution, verification, runtime):
    """Build all histories from the original v3 artifacts, never a diagnostic answer."""
    parent = Path(parent)
    ids = [job["id"] for job in manifest["jobs"]]
    _require(
        len(ids) == PARENT_EPISODES and len(set(ids)) == PARENT_EPISODES,
        "Parent manifest incomplete",
    )
    _require(
        _read(parent / "started.json")["manifest_sha256"] == digest(manifest),
        "Parent start/manifest mismatch",
    )
    _require([o["id"] for o in execution["outcomes"]] == ids, "Parent execution order mismatch")
    score = {(r["id"], r["variant"]): r for r in verification["rows"]}
    _require(len(score) == 480, "Parent reporter verification incomplete")
    inputs = []
    for job in manifest["jobs"]:
        episode_id = job["id"]
        pair_path = parent / "paired" / f"{episode_id}.json"
        pair = _read(pair_path)
        original = pair["original"]
        _require(
            original["id"] == episode_id
            and original["case_id"] == job["case_id"]
            and original["model"] == job["model"]
            and original["initial_sha256"] == job["initial_sha256"],
            f"Parent episode changed: {episode_id}",
        )
        _require(
            pair["actual_calls"]
            == next(x for x in execution["outcomes"] if x["id"] == episode_id)["actual_calls"],
            f"Parent actual calls changed: {episode_id}",
        )
        for arm in ("control", "candidate"):
            _require(
                pair["reports"][arm]["terminal"] == score[(episode_id, arm)]["terminal"],
                f"Parent report status changed: {episode_id}/{arm}",
            )
        case = runtime[job["case_id"]]
        _require(
            case["language"] == job["case_id"].rsplit("-", 1)[-1]
            and case["workflow"] == job["case_id"].rsplit("-", 1)[0]
            and case["caller"] == _read_actor_caller(original),
            f"Parent source/scope changed: {episode_id}",
        )
        candidate_path = parent / "calls" / f"{episode_id}-final-candidate.request.json"
        reconstructed = False
        actor_plan_hash = None
        if candidate_path.exists():
            record = _read(candidate_path)
            _require(
                record["endpoint"] == ENDPOINT and record["sha256"] == digest(record["request"]),
                f"Saved reporter request changed: {episode_id}",
            )
            request = record["request"]
            request_file_sha = _sha(candidate_path)
            _require(
                pair["reports"]["candidate"]["terminal"] != "not_reached",
                f"Saved candidate request but no parent report: {episode_id}",
            )
        else:
            _require(
                episode_id == "confirm-fresh-13-en-m0"
                and all(
                    pair["reports"][a]["terminal"] == "not_reached"
                    for a in ("control", "candidate")
                ),
                "Unexpected absent parent candidate request",
            )
            actor_calls = pair["actual_calls"]
            _require(
                len(actor_calls) == 1 and actor_calls[0] == episode_id + "-a0",
                "Capped actor history has other calls",
            )
            actor_request_path = parent / "calls" / f"{actor_calls[0]}.request.json"
            actor_receipt_path = parent / "calls" / f"{actor_calls[0]}.receipt.json"
            recorded = _read(actor_request_path)
            _require(
                recorded["endpoint"] == ENDPOINT
                and recorded["sha256"] == digest(recorded["request"]),
                "Capped actor request changed",
            )
            actor_plan = prepare_plan(
                actor_job=actor_calls[0],
                actor_request=recorded["request"],
                native_receipt=_read(actor_receipt_path),
                tool_trace=original["tool_trace"],
                initial_backend=original["initial_backend"],
                final_backend=original["final_backend"],
                actor_calls_used=1,
                caller_scope=case["caller"],
            )
            request = actor_plan["reports"]["candidate"]["request"]
            request_file_sha = None
            actor_plan_hash = actor_plan["plan_sha256"]
            reconstructed = True
        _require(request["model"] == job["model"], "Reporter model mismatch")
        _, sidecar, _ = ablation.render(request, original["tool_trace"], ARMS[0])
        _require(
            sidecar["original"]["original_messages"] == original["history"],
            f"Source history mismatch: {episode_id}",
        )
        _require(
            json.loads(original["history"][1]["content"])["source_messages"]
            == case["source_messages"],
            f"Original source messages changed: {episode_id}",
        )
        inputs.append(
            {
                "id": episode_id,
                "case_id": job["case_id"],
                "scenario": case["workflow"],
                "language": case["language"],
                "family": case["family"],
                "model": job["model"],
                "parent_paired_sha256": _sha(pair_path),
                "parent_request_file_sha256": request_file_sha,
                "parent_request_payload_sha256": digest(request),
                "capped_reconstruction": reconstructed,
                "actor_plan_sha256": actor_plan_hash,
                "request": request,
                "tool_trace": original["tool_trace"],
            }
        )
    _require(
        len(inputs) == PARENT_EPISODES
        and Counter(i["model"] for i in inputs) == {m: 120 for m in PROFILES}
        and Counter(i["language"] for i in inputs)
        == {language: 80 for language in ("en", "hi", "te")},
        "Model/language coverage changed",
    )
    scenarios = Counter(i["scenario"] for i in inputs)
    _require(
        len(scenarios) == 40 and set(scenarios.values()) == {6}, "40-by-6 scenario coverage changed"
    )
    _require(
        sum(i["capped_reconstruction"] for i in inputs) == 1,
        "Capped source reconstruction count changed",
    )
    return inputs


def _read_actor_caller(original):
    supplied = json.loads(original["history"][1]["content"])
    current = original["initial_backend"]["appointments"]
    rows = [row for row in current if row["id"] == supplied["appointment_id"]]
    _require(len(rows) == 1, "Parent caller scope ambiguous")
    return {key: rows[0][key] for key in ("tenant", "customer")}


def _ordered_jobs(inputs):
    by_scenario = defaultdict(list)
    for item in inputs:
        by_scenario[item["scenario"]].append(item)
    scenario_names = sorted(by_scenario)
    six = tuple(permutations(ARMS))
    assignment = {}
    for scenario_index, name in enumerate(scenario_names):
        members = sorted(by_scenario[name], key=lambda i: (i["language"], i["model"]))
        _require(len(members) == 6, "Scenario lacks six renderings")
        for slot, item in enumerate(members):
            assignment[item["id"]] = six[(scenario_index + slot) % 6]
    shuffled = list(inputs)
    random.Random(ORDER_SEED).shuffle(shuffled)
    jobs = []
    for item in shuffled:
        for position, arm in enumerate(assignment[item["id"]]):
            request, sidecar, omissions = ablation.render(item["request"], item["tool_trace"], arm)
            ablation.verify_omissions(request, sidecar, omissions, arm)
            jobs.append(
                {
                    "id": item["id"] + "-source-" + arm,
                    "history_id": item["id"],
                    "arm": arm,
                    "position": position,
                    "scenario": item["scenario"],
                    "request_sha256": digest(request),
                    "reservation_microinr": reservation(request),
                    "omission_count": len(omissions),
                }
            )
    _require(
        len(jobs) == PLANNED_CALLS
        and Counter(j["arm"] for j in jobs) == {arm: 240 for arm in ARMS}
        and Counter((j["arm"], j["position"]) for j in jobs)
        == {(arm, pos): 80 for arm in ARMS for pos in range(3)},
        "Three-arm order is unbalanced",
    )
    # When C omits nothing, B and C have identical bytes and form a no-treatment control.
    grouped = {(j["history_id"], j["arm"]): j for j in jobs}
    for item in inputs:
        b, c = (grouped[item["id"], arm] for arm in ARMS[1:])
        for arm in ARMS:
            grouped[item["id"], arm]["treated_actor_prose"] = bool(c["omission_count"])
        _require(
            bool(c["omission_count"]) or b["request_sha256"] == c["request_sha256"],
            "No-op actor-prose arm differs from grouped control",
        )
    return jobs


def _budget(jobs):
    worst = sum(job["reservation_microinr"] for job in jobs)
    _require(
        worst == FULL_RESERVATION
        and worst <= PHASE_CAP
        and PRIOR_RETAINED + PHASE_CAP <= CAMPAIGN_CAP,
        "Full-context budget changed",
    )
    return {
        "phase_cap_microinr": PHASE_CAP,
        "stages": {STAGE: {"calls": PLANNED_CALLS, "cap_microinr": PHASE_CAP}},
        "prior_calls": PRIOR_CALLS,
        "prior_known_microinr": PRIOR_KNOWN,
        "prior_retained_microinr": PRIOR_RETAINED,
        "campaign_cap_microinr": CAMPAIGN_CAP,
        "authority": (
            "Founder original total INR6000 research approval plus current instruction "
            "to proceed with proposed phase; not a renewed allowance"
        ),
        "maximum_full_context_reservation_microinr": worst,
    }


def _completion(path):
    result = _read(path)
    _require(
        result.get("all_required_publication_checks_complete") is True
        and result.get("new_calls") == 2
        and result.get("new_native_or_tool_calls") == 0
        and result.get("research_calls") == PRIOR_CALLS
        and round(result.get("research_known_inr", -1) * 1_000_000) == PRIOR_KNOWN
        and round(result.get("research_retained_inr", -1) * 1_000_000) == PRIOR_RETAINED,
        "Prior diagnostic completion or cumulative carry changed",
    )
    return result


def _preflight(path, inputs, jobs):
    reviewed = _read(path)
    _require(
        reviewed.get("status") == "INDEPENDENT_OFFLINE_SOURCE_PREFLIGHT_PASS"
        and reviewed.get("scope", {}).get("planned_reporter_requests") == PLANNED_CALLS
        and reviewed["scope"]["actor_prose_treated_histories"] == 184
        and reviewed["scope"]["actor_prose_no_treatment_histories"] == 56,
        "Independent source preflight changed",
    )
    cases = {row["id"]: row for row in reviewed["cases"]}
    _require(
        len(cases) == PARENT_EPISODES and set(cases) == {item["id"] for item in inputs},
        "Independent source preflight coverage changed",
    )
    by_job = {(j["history_id"], j["arm"]): j for j in jobs}
    treatment = Counter()
    for item in inputs:
        row = cases[item["id"]]
        treated = by_job[item["id"], ARMS[2]]["treated_actor_prose"]
        treatment[(item["model"], treated)] += 1
        _require(
            row["paired_file_sha256"] == item["parent_paired_sha256"]
            and row["saved_candidate_request_file_sha256"] == item["parent_request_file_sha256"]
            and row["actor_prose_treatment_present"] is treated
            and row["actor_prose_omission_ledger"]
            == ablation.render(item["request"], item["tool_trace"], ARMS[2])[2]
            and row["arm_request_sha256"]
            == {arm: by_job[item["id"], arm]["request_sha256"] for arm in ARMS},
            f"Independent source preflight disagrees: {item['id']}",
        )
    _require(
        treatment
        == {
            ("sarvam-105b", True): 98,
            ("sarvam-105b", False): 22,
            ("sarvam-105b-conversations", True): 86,
            ("sarvam-105b-conversations", False): 34,
        },
        "Treatment split changed",
    )
    return reviewed


def _flat_census(parent, manifest, execution, verification, runtime):
    rows, selected, refusals = _census(
        manifest, execution, verification, runtime, parent / "paired", parent / "calls"
    )
    _require(
        len(rows) == PARENT_EPISODES
        and not refusals
        and selected["id"] == "confirm-fresh-13-en-m0",
        "Parent routing census changed",
    )
    return rows


def prepare(root, parent, prior_completion):
    """Freeze 240 original histories and 720 reporter requests without reading gold."""
    root, parent, prior_completion = map(Path, (root, parent, prior_completion))
    _require(
        root.is_dir()
        and all(
            (root / name).is_file()
            for name in (
                "PROTOCOL.md",
                "budget.json",
                "pacing-control.json",
                "source-preflight.json",
            )
        ),
        "Reviewed protocol, budget and pacing files required",
    )
    _require(
        not any(
            (root / name).exists()
            for name in ("manifest.json", "inputs.json", "started.json", "calls", "provenance")
        ),
        "Study already prepared or attempted",
    )
    _completion(prior_completion)
    manifest, execution, verification, runtime = _parent_files(parent, full_call_inventory=True)
    _require(verification["calls"] == 844, "Original v3 call inventory changed")
    _require(
        _read(parent / "started.json")["manifest_sha256"] == digest(manifest),
        "Original v3 start binding changed",
    )
    for job in manifest["jobs"]:
        episode_id = job["id"]
        _require(
            (parent / "episodes" / episode_id / "paired.json").is_file(),
            f"Parent episode missing: {episode_id}",
        )
        candidate = parent / "calls" / f"{episode_id}-final-candidate.request.json"
        if episode_id != "confirm-fresh-13-en-m0":
            _require(candidate.is_file(), f"Saved candidate request missing: {episode_id}")
    # From here onward every write is create-once; a partial freeze cannot resume.
    archived = root / "provenance" / "parent"
    for name in PARENT_FILES:
        _copy_once(parent / name, archived / name)
    _copy_once(parent / "gold.json", root / "sealed" / "gold.json")
    _copy_once(prior_completion, root / "provenance" / "prior-completion.json")
    semantic = _read(parent / "semantic-review.json")
    _require(
        semantic.get("status") == "ASSISTANT_LANGUAGE_REVIEW_COMPLETE",
        "Parent semantic review incomplete",
    )
    for entry in semantic["review_files"].values():
        relative = Path(entry["file"])
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Parent review path escapes archive",
        )
        _copy_once(parent / relative, archived / relative)
    for name in manifest["source_sha256"]:
        _copy_once(parent / "source" / name, archived / "source" / name)
    for job in manifest["jobs"]:
        episode_id = job["id"]
        _copy_once(
            parent / "episodes" / episode_id / "paired.json",
            archived / "paired" / f"{episode_id}.json",
        )
        candidate = parent / "calls" / f"{episode_id}-final-candidate.request.json"
        if candidate.exists():
            _copy_once(candidate, archived / "calls" / candidate.name)
    selected_actor = "confirm-fresh-13-en-m0-a0"
    for suffix in ("request", "receipt"):
        name = f"{selected_actor}.{suffix}.json"
        _copy_once(parent / "calls" / name, archived / "calls" / name)
    copied_manifest, copied_execution, copied_verification, copied_runtime = _parent_files(
        archived, gold_path=root / "sealed" / "gold.json"
    )
    _flat_census(archived, copied_manifest, copied_execution, copied_verification, copied_runtime)
    inputs = _source_inventory(
        archived, copied_manifest, copied_execution, copied_verification, copied_runtime
    )
    jobs = _ordered_jobs(inputs)
    _preflight(root / "source-preflight.json", inputs, jobs)
    budget = _budget(jobs)
    _require(_read(root / "budget.json") == budget, "Reviewed budget differs from exact jobs")
    _require(
        _read(root / "pacing-control.json")
        == {
            "rpm": 90,
            "user_reported_account_rpm": 120,
            "workers": 4,
            "batch_size": 4,
            "stop_after_batch_errors": 2,
            "no_retries": True,
        },
        "Pacing must be 90 RPM",
    )
    sidecars = {}
    omissions = {}
    for item in inputs:
        _, sidecar, _ = ablation.render(item["request"], item["tool_trace"], ARMS[0])
        sidecars[item["id"]] = sidecar
    by_id = {item["id"]: item for item in inputs}
    for job in jobs:
        item = by_id[job["history_id"]]
        _, _, ledger = ablation.render(item["request"], item["tool_trace"], job["arm"])
        omissions[job["id"]] = ledger
    save(root / "inputs.json", inputs)
    save(root / "sidecars.json", sidecars)
    save(root / "omissions.json", omissions)
    frozen = [
        root / n
        for n in (
            "PROTOCOL.md",
            "budget.json",
            "pacing-control.json",
            "source-preflight.json",
            "inputs.json",
            "sidecars.json",
            "omissions.json",
        )
    ]
    frozen += [p for p in (root / "provenance").rglob("*") if p.is_file()]
    frozen += [root / "sealed" / "gold.json"]
    package = {
        "kind": "ALL_240_SAVED_HISTORY_SOURCE_INTERPRETATION_SCREEN",
        "jobs": jobs,
        "source_sha256": _code_hashes(),
        "files_sha256": {str(p.relative_to(root)): _sha(p) for p in sorted(frozen)},
        "parent_census_sha256": digest(
            _flat_census(
                archived, copied_manifest, copied_execution, copied_verification, copied_runtime
            )
        ),
        "maximum_calls": PLANNED_CALLS,
        "worst_case_retained_microinr": FULL_RESERVATION,
        "workers": 4,
        "batch_size": 4,
        "rpm": 90,
        "stop_after_batch_errors": 2,
        "no_retries": True,
        "new_native_action_trials": 0,
        "capped_reconstruction_count": 1,
        "treated_actor_prose_histories": sum(
            j["treated_actor_prose"] for j in jobs if j["arm"] == ARMS[2]
        ),
        "development_exposed": True,
    }
    save(root / "manifest.json", package)
    return package


def check(root):
    """Recreate every paid request from the portable source records, without gold."""
    root = Path(root)
    package = _read(root / "manifest.json")
    _require(
        package["kind"] == "ALL_240_SAVED_HISTORY_SOURCE_INTERPRETATION_SCREEN"
        and package["source_sha256"] == _code_hashes(),
        "Study source changed",
    )
    for name, expected in package["files_sha256"].items():
        _require(_sha(root / name) == expected, f"Frozen input changed: {name}")
    _completion(root / "provenance" / "prior-completion.json")
    archived = root / "provenance" / "parent"
    manifest, execution, verification, runtime = _parent_files(
        archived, gold_path=root / "sealed" / "gold.json"
    )
    _require(
        digest(_flat_census(archived, manifest, execution, verification, runtime))
        == package["parent_census_sha256"],
        "Portable parent census changed",
    )
    inputs = _source_inventory(archived, manifest, execution, verification, runtime)
    _require(inputs == _read(root / "inputs.json"), "Source inputs changed")
    jobs = _ordered_jobs(inputs)
    _preflight(root / "source-preflight.json", inputs, jobs)
    _require(
        jobs == package["jobs"]
        and len(jobs) == PLANNED_CALLS
        and _read(root / "budget.json") == _budget(jobs)
        and package["worst_case_retained_microinr"] == FULL_RESERVATION,
        "Frozen jobs or budget changed",
    )
    sidecars, omissions = _read(root / "sidecars.json"), _read(root / "omissions.json")
    by_id = {item["id"]: item for item in inputs}
    _require(
        set(sidecars) == set(by_id) and set(omissions) == {j["id"] for j in jobs},
        "Evidence sidecar/omission coverage changed",
    )
    for job in jobs:
        item = by_id[job["history_id"]]
        request, sidecar, ledger = ablation.render(item["request"], item["tool_trace"], job["arm"])
        ablation.verify_omissions(request, sidecar, ledger, job["arm"])
        _require(
            sidecars[item["id"]] == sidecar
            and omissions[job["id"]] == ledger
            and digest(request) == job["request_sha256"]
            and reservation(request) == job["reservation_microinr"]
            and len(ledger) == job["omission_count"],
            f"Reporter evidence changed: {job['id']}",
        )
    return package, by_id


def execute_job(job, item, dispatch):
    try:
        request, _, _ = ablation.render(item["request"], item["tool_trace"], job["arm"])
        _require(digest(request) == job["request_sha256"], "Dispatched request changed")
        return classify(dispatch(job["id"], request))
    except Exception as exc:
        return {
            "terminal": "infrastructure_error",
            "error_class": type(exc).__name__,
            "fatal_abort": isinstance(exc, ValueError),
        }


def run(
    root, *, allow_paid=False, transport_call=call, credential_loader=read_key, pacer_factory=Pacer
):
    """One no-resume launch; tests inject transport and never read a credential."""
    if not allow_paid:
        raise ValueError("Paid study requires explicit --allow-paid")
    root = Path(root)
    package, inputs = check(root)
    _require(
        not any((root / name).exists() for name in ("started.json", "execution.json", "calls")),
        "Study already started; no retry or continuation",
    )
    _require(
        package["maximum_calls"] == PLANNED_CALLS
        and package["worst_case_retained_microinr"] == FULL_RESERVATION
        and package["workers"] == package["batch_size"] == 4
        and package["stop_after_batch_errors"] == 2,
        "Run configuration changed",
    )
    mode = (
        "production_default_transport"
        if transport_call is call and credential_loader is read_key and pacer_factory is Pacer
        else "injected_offline_transport"
    )
    save(root / "started.json", {"manifest_sha256": digest(package), "execution_mode": mode})
    meter, pacer, key = ControlMeter(root), pacer_factory(root), credential_loader()

    def dispatch(job_id, request):
        timing = pacer.wait()
        save(meter.root / f"{job_id}.dispatch.json", timing)
        return transport_call(meter, job_id, STAGE, request, key)

    outcomes = []
    errors = fatal_aborts = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(package["jobs"]), 4):
            batch = package["jobs"][offset : offset + 4]
            results = [
                future.result()
                for future in [
                    pool.submit(execute_job, job, inputs[job["history_id"]], dispatch)
                    for job in batch
                ]
            ]
            for job, result in zip(batch, results):
                save(root / f"{job['id']}.result.json", result)
                outcomes.append({"id": job["id"], "terminal": result["terminal"]})
            errors += sum(result["terminal"] == "infrastructure_error" for result in results)
            fatal_aborts += sum(bool(result.get("fatal_abort")) for result in results)
            print(
                {
                    "attempted": len(outcomes),
                    "infrastructure_errors": errors,
                    "meter": meter.summary(),
                },
                flush=True,
            )
            if errors >= 2 or fatal_aborts:
                break
    for job in package["jobs"][len(outcomes) :]:
        outcomes.append({"id": job["id"], "terminal": "not_attempted_after_stop"})
    save(
        root / "execution.json",
        {
            "outcomes": outcomes,
            "meter": meter.summary(),
            "attempted": sum(o["terminal"] != "not_attempted_after_stop" for o in outcomes),
            "infrastructure_errors": errors,
            "fatal_aborts": fatal_aborts,
        },
    )
    return _read(root / "execution.json")


def _receipt_replay(root, job, result, ledger):
    """Replay one attempted output and charge from immutable local artifacts."""
    root = Path(root)
    request_path = root / "calls" / f"{job['id']}.request.json"
    receipt_path = root / "calls" / f"{job['id']}.receipt.json"
    if not request_path.exists():
        _require(
            not receipt_path.exists()
            and job["id"] not in ledger
            and result["terminal"] == "infrastructure_error",
            "Unrecorded or unmetered dispatch",
        )
        return {
            "usage": None,
            "receipt": None,
            "body": None,
            "cost_microinr": 0,
            "retained_microinr": 0,
        }
    request_record = _read(request_path)
    _require(
        request_record["endpoint"] == ENDPOINT
        and request_record["sha256"] == job["request_sha256"] == digest(request_record["request"]),
        "Recorded reporter request changed",
    )
    _require(job["id"] in ledger, "Unmetered reporter request")
    row = ledger[job["id"]]
    _require(
        row[0] == STAGE and row[1] == job["reservation_microinr"],
        "Reporter meter reservation changed",
    )
    if not receipt_path.exists():
        _require(
            result["terminal"] == "infrastructure_error"
            and row[2] == row[1]
            and row[3] is None
            and row[4] == "reserved",
            "Missing receipt has uncertain ledger state",
        )
        return {
            "usage": {
                "known": False,
                "prompt_tokens": None,
                "completion_tokens": None,
                "known_cost_microinr": None,
                "retained_microinr": row[2],
                "full_reservation_microinr": row[1],
            },
            "receipt": None,
            "body": None,
            "cost_microinr": 0,
            "retained_microinr": row[2],
        }
    receipt = _read(receipt_path)
    _require(
        receipt["job"] == job["id"] and receipt["request_sha256"] == job["request_sha256"],
        "Reporter receipt/request changed",
    )
    body, usage = _usage(receipt, request_record["request"])
    _require(
        row[2] == usage["retained_microinr"]
        and row[3] == usage["known_cost_microinr"]
        and row[4] == ("known" if usage["known"] else "unknown"),
        "Reporter usage/meter mismatch",
    )
    good = (
        receipt.get("status") == 200
        and usage["known"]
        and body.get("model") == request_record["request"]["model"]
    )
    if good:
        try:
            replay = classify(body)
        except (ValueError, KeyError, TypeError, IndexError):
            good = False
        else:
            _require(replay == result, "Saved reporter classification changed")
    if not good:
        _require(
            result["terminal"] == "infrastructure_error",
            "Reporter infrastructure failure not preserved",
        )
    return {
        "usage": usage,
        "receipt": receipt,
        "body": body,
        "cost_microinr": usage["known_cost_microinr"] or 0,
        "retained_microinr": usage["retained_microinr"],
    }


def _score(job, item, result, case, label, original, usage, receipt):
    projection = None
    if result.get("parsed") is not None:
        try:
            projection = project(
                intent=case["intent"],
                requested_change=case["requested_change"],
                prior_receipts=original["seed_receipts"],
                tool_trace=original["tool_trace"],
                raw_content=result["content"],
                completed=True,
            )
        except (ValueError, KeyError, TypeError):
            pass
    view = report_view(
        {"original": original, "reports": {job["arm"]: {**result, "application": projection}}},
        job["arm"],
    )
    attribution = evaluate(
        case,
        label,
        view,
        {
            "runtime_sha256": digest(case),
            "gold_sha256": digest(label),
            "question_target": label["question_target"],
        },
    )
    raw = result.get("parsed")
    target = attribution["actual_question_answer"]
    valid = raw is not None
    wrong_non_null = bool(valid and raw["answer"] is not None and raw["answer"] != target["answer"])
    false_abstention = bool(
        valid
        and target["answer"] is not None
        and (raw["answer"] is None or raw["needs_clarification"] is True)
    )
    unsupported_non_null = bool(valid and target["answer"] is None and raw["answer"] is not None)
    unknown_guess = bool(unsupported_non_null and item["family"] == "unresolved_condition")
    return {
        "id": job["id"],
        "history_id": item["id"],
        "case_id": item["case_id"],
        "scenario": item["scenario"],
        "language": item["language"],
        "family": item["family"],
        "model": item["model"],
        "arm": job["arm"],
        "position": job["position"],
        "treated_actor_prose": job["treated_actor_prose"],
        "terminal": result["terminal"],
        "valid_final": valid,
        "raw": raw,
        "application_output": projection["application_output"] if projection else None,
        "wrong_non_null_date": wrong_non_null,
        "false_abstention": false_abstention,
        "unsupported_non_null_on_null_target": unsupported_non_null,
        "unknown_condition_guess": unknown_guess,
        "raw_fully_correct": bool(
            attribution["factual_answer_correct"] and attribution["raw_status_matches_execution"]
        ),
        "projected_fully_correct": bool(
            attribution["factual_answer_correct"]
            and attribution["application_status_matches_execution"]
        ),
        "usage": usage,
        "elapsed_seconds": receipt.get("elapsed_seconds") if receipt else None,
        **attribution,
    }


def _aggregate(rows):
    return {
        "n": len(rows),
        "factual_correct": sum(bool(r["factual_answer_correct"]) for r in rows),
        "valid_final": sum(r["valid_final"] for r in rows),
        "invalid_or_missing_final": sum(not r["valid_final"] for r in rows),
        "wrong_non_null_dates": sum(r["wrong_non_null_date"] for r in rows),
        "false_abstentions": sum(r["false_abstention"] for r in rows),
        "unknown_condition_guesses": sum(r["unknown_condition_guess"] for r in rows),
        "raw_status_correct": sum(r["raw_status_matches_execution"] for r in rows),
        "projected_status_correct": sum(r["application_status_matches_execution"] for r in rows),
        "raw_fully_correct_reports": sum(r["raw_fully_correct"] for r in rows),
        "projected_fully_correct_reports": sum(r["projected_fully_correct"] for r in rows),
        "known_prompt_tokens": sum((r["usage"] or {}).get("prompt_tokens") or 0 for r in rows),
        "known_completion_tokens": sum(
            (r["usage"] or {}).get("completion_tokens") or 0 for r in rows
        ),
        "known_cost_microinr": sum(
            (r["usage"] or {}).get("known_cost_microinr") or 0 for r in rows
        ),
        "retained_microinr": sum((r["usage"] or {}).get("retained_microinr") or 0 for r in rows),
        "elapsed_seconds_sum": sum(r["elapsed_seconds"] or 0 for r in rows),
        "elapsed_seconds_count": sum(r["elapsed_seconds"] is not None for r in rows),
    }


def _bootstrap(pairs, scenario_universe, *, seed=ORDER_SEED, draws=10_000):
    """Percentile interval from paired scenario resampling, never row IID."""
    clusters = defaultdict(list)
    for before, after in pairs:
        clusters[before["scenario"]].append(
            int(after["factual_answer_correct"]) - int(before["factual_answer_correct"])
        )
    names = sorted(scenario_universe)
    _require(
        len(names) == 40 and set(clusters) <= set(names), "Bootstrap scenario universe changed"
    )
    randomizer = random.Random(seed)
    estimates = []
    zero_denominator_draws = 0
    for _ in range(draws):
        sample = [clusters[names[randomizer.randrange(len(names))]] for _ in names]
        denominator = sum(len(cluster) for cluster in sample)
        if denominator:
            estimates.append(sum(sum(cluster) for cluster in sample) / denominator)
        else:
            zero_denominator_draws += 1
    estimates.sort()
    return {
        "scenario_clusters": len(names),
        "draws": draws,
        "estimable_draws": len(estimates),
        "zero_denominator_draws": zero_denominator_draws,
        "seed": seed,
        "units": "fraction_of_selected_histories",
        "percentile_95": [
            estimates[int(0.025 * (len(estimates) - 1))],
            estimates[int(0.975 * (len(estimates) - 1))],
        ],
        "percentile_95_percentage_points": [
            100 * estimates[int(0.025 * (len(estimates) - 1))],
            100 * estimates[int(0.975 * (len(estimates) - 1))],
        ],
    }


def _contrasts(rows):
    by_history = defaultdict(dict)
    for row in rows:
        by_history[row["history_id"]][row["arm"]] = row
    _require(
        len(by_history) == PARENT_EPISODES
        and all(set(arms) == set(ARMS) for arms in by_history.values()),
        "Paired arm coverage changed",
    )
    scenario_universe = {arms[ARMS[0]]["scenario"] for arms in by_history.values()}
    _require(len(scenario_universe) == 40, "Paired scenario universe changed")
    contrasts = {}
    specs = (
        ("A_to_B", ARMS[0], ARMS[1]),
        ("B_to_C", ARMS[1], ARMS[2]),
        ("A_to_C", ARMS[0], ARMS[2]),
    )
    for name, left, right in specs:
        variants = {}
        for scope, selected in (
            ("all", list(by_history.values())),
            ("treated", [x for x in by_history.values() if x[ARMS[2]]["treated_actor_prose"]]),
            (
                "no_treatment",
                [x for x in by_history.values() if not x[ARMS[2]]["treated_actor_prose"]],
            ),
        ):
            for model in ("all", *PROFILES):
                pairs = [
                    (x[left], x[right])
                    for x in selected
                    if model == "all" or x[left]["model"] == model
                ]
                if not pairs:
                    continue
                gains = [
                    x
                    for x, y in pairs
                    if not x["factual_answer_correct"] and y["factual_answer_correct"]
                ]
                losses = [
                    x
                    for x, y in pairs
                    if x["factual_answer_correct"] and not y["factual_answer_correct"]
                ]
                key = f"{scope}/{model}"
                variants[key] = {
                    "n": len(pairs),
                    "left_correct": sum(x["factual_answer_correct"] for x, _ in pairs),
                    "right_correct": sum(y["factual_answer_correct"] for _, y in pairs),
                    "gain": len(gains),
                    "loss": len(losses),
                    "net": len(gains) - len(losses),
                    "restored_from_invalid": sum(not x["valid_final"] for x in gains),
                    "corrected_usable_answer": sum(x["valid_final"] for x in gains),
                    "workflow_net": {
                        scenario: sum(
                            int(y["factual_answer_correct"]) - int(x["factual_answer_correct"])
                            for x, y in pairs
                            if x["scenario"] == scenario
                        )
                        for scenario in sorted({x["scenario"] for x, _ in pairs})
                    },
                    "scenario_cluster_bootstrap": _bootstrap(pairs, scenario_universe),
                }
        contrasts[name] = variants
    return contrasts


def _gate(
    rows,
    contrasts,
    *,
    receipts_complete,
    unknown_calls,
    infrastructure_errors,
    production_transport,
):
    all_complete = receipts_complete and unknown_calls == 0 and infrastructure_errors == 0
    checks = {"all_720_clean": all_complete, "production_default_transport": production_transport}
    for model in PROFILES:
        treated = contrasts["B_to_C"][f"treated/{model}"]
        operational = contrasts["A_to_C"][f"all/{model}"]
        checks[f"treated_B_to_C_positive/{model}"] = treated["net"] > 0
        checks[f"two_positive_workflows/{model}"] = (
            sum(value > 0 for value in treated["workflow_net"].values()) >= 2
        )
        checks[f"all_C_to_A_positive/{model}"] = operational["net"] > 0
        by_arm = {
            arm: _aggregate([r for r in rows if r["model"] == model and r["arm"] == arm])
            for arm in ARMS
        }
        for metric in ("wrong_non_null_dates", "false_abstentions"):
            checks[f"C_no_higher_{metric}/{model}"] = all(
                by_arm[ARMS[2]][metric] <= by_arm[arm][metric] for arm in ARMS[:2]
            )
        for metric in ("raw_status_correct", "projected_status_correct"):
            checks[f"C_no_lower_{metric}/{model}"] = all(
                by_arm[ARMS[2]][metric] >= by_arm[arm][metric] for arm in ARMS[:2]
            )
    return {
        "checks": checks,
        "mechanical_pass_pending_semantic_review": all(checks.values()),
        "semantic_review_complete": False,
        "forward_gate_pass": None,
    }


def audit(root, *, output=None):
    """Independent offline receipt/charge replay; only here may gold be parsed."""
    root = Path(root)
    package, inputs = check(root)
    started = _read(root / "started.json")
    _require(
        started["manifest_sha256"] == digest(package)
        and started["execution_mode"]
        in ("production_default_transport", "injected_offline_transport"),
        "Started manifest binding changed",
    )
    execution = _read(root / "execution.json")
    jobs = package["jobs"]
    outcomes = execution["outcomes"]
    _require(
        len(outcomes) == PLANNED_CALLS
        and [row["id"] for row in outcomes] == [job["id"] for job in jobs],
        "Planned outcome inventory changed",
    )
    attempted = [row for row in outcomes if row["terminal"] != "not_attempted_after_stop"]
    _require(
        outcomes[: len(attempted)] == attempted
        and execution["attempted"] == len(attempted)
        and len(attempted) <= PLANNED_CALLS,
        "Stopped outcome prefix changed",
    )
    with sqlite3.connect(f"file:{root / 'calls' / 'meter.sqlite'}?mode=ro", uri=True) as db:
        ledger = {r[0]: r[1:] for r in db.execute("SELECT * FROM calls")}
    cases = {c["id"]: c for c in _read(root / "provenance" / "parent" / "runtime.json")}
    labels = _read(root / "sealed" / "gold.json")
    rows, response_ids, dispatch_order = [], set(), []
    known = retained = unknown = infrastructure = 0
    attempted_ids = {row["id"] for row in attempted}
    for job, outcome in zip(jobs, outcomes):
        item = inputs[job["history_id"]]
        pair = _read(root / "provenance" / "parent" / "paired" / f"{item['id']}.json")
        original = pair["original"]
        case = cases[item["case_id"]]
        label = labels[item["case_id"]]
        _require(label["runtime_sha256"] == digest(case), "Gold/runtime source binding changed")
        if job["id"] in attempted_ids:
            result = _read(root / f"{job['id']}.result.json")
            _require(result["terminal"] == outcome["terminal"], "Outcome/result changed")
            replay = _receipt_replay(root, job, result, ledger)
            usage, receipt, body = replay["usage"], replay["receipt"], replay["body"]
            known += replay["cost_microinr"]
            retained += replay["retained_microinr"]
            unknown += bool(job["id"] in ledger and ledger[job["id"]][4] != "known")
            infrastructure += result["terminal"] == "infrastructure_error"
            if receipt and receipt.get("status") == 200 and usage["known"] and body.get("id"):
                _require(body["id"] not in response_ids, "Duplicate provider response ID")
                response_ids.add(body["id"])
            marker = root / "calls" / f"{job['id']}.dispatch.json"
            if marker.exists():
                timing = _read(marker)
                stamp = timing.get("dispatch_time")
                if started["execution_mode"] == "production_default_transport":
                    _require(
                        type(stamp) in (int, float)
                        and timing.get("pacing_control", {}).get("rpm") == 90,
                        "Production dispatch timing changed",
                    )
                dispatch_order.append(
                    {"id": job["id"], "dispatch_time": stamp, "planned_position": job["position"]}
                )
            elif (
                started["execution_mode"] == "production_default_transport"
                and (root / "calls" / f"{job['id']}.request.json").exists()
            ):
                raise ValueError("Production request missing dispatch marker")
        else:
            _require(
                not (root / f"{job['id']}.result.json").exists(), "Stopped job has saved result"
            )
            result = {"terminal": "not_attempted_after_stop"}
            usage = receipt = None
        rows.append(_score(job, item, result, case, label, original, usage, receipt))
    _require(
        infrastructure == execution["infrastructure_errors"]
        and sum(
            bool(_read(root / f"{job['id']}.result.json").get("fatal_abort"))
            for job in jobs[: len(attempted)]
        )
        == execution["fatal_aborts"],
        "Infrastructure error count changed",
    )
    request_ids = {
        p.name.removesuffix(".request.json") for p in (root / "calls").glob("*.request.json")
    }
    receipt_ids = {
        p.name.removesuffix(".receipt.json") for p in (root / "calls").glob("*.receipt.json")
    }
    dispatch_ids = {
        p.name.removesuffix(".dispatch.json") for p in (root / "calls").glob("*.dispatch.json")
    }
    result_ids = {p.name.removesuffix(".result.json") for p in root.glob("*.result.json")}
    _require(
        result_ids == attempted_ids
        and request_ids <= attempted_ids
        and receipt_ids <= request_ids
        and dispatch_ids <= attempted_ids
        and set(ledger) == request_ids,
        "Reporter artifact or meter inventory changed",
    )
    meter_summary = (
        {
            STAGE: {
                "calls": len(ledger),
                "charged_with_headroom_inr": retained / 1e6,
                "known_usage_estimate_inr": known / 1e6,
                "unknown_or_pending": sum(row[4] != "known" for row in ledger.values()),
            }
        }
        if ledger
        else {}
    )
    _require(
        retained <= PHASE_CAP
        and PRIOR_RETAINED + retained <= CAMPAIGN_CAP
        and execution["meter"] == meter_summary,
        "Budget or saved meter summary changed",
    )
    groups = {}
    for dimensions in (
        ("model", "arm"),
        ("language", "arm"),
        ("family", "arm"),
        ("model", "language", "arm"),
        ("model", "family", "arm"),
        ("treated_actor_prose", "arm"),
    ):
        kind = "/".join(dimensions)
        keys = sorted({tuple(row[k] for k in dimensions) for row in rows})
        groups[kind] = {
            "/".join(map(str, key)): _aggregate(
                [row for row in rows if tuple(row[k] for k in dimensions) == key]
            )
            for key in keys
        }
    contrasts = _contrasts(rows)
    clean = (
        len(attempted) == PLANNED_CALLS
        and len(receipt_ids) == PLANNED_CALLS
        and len(ledger) == PLANNED_CALLS
    )
    gate = _gate(
        rows,
        contrasts,
        receipts_complete=clean,
        unknown_calls=unknown,
        infrastructure_errors=infrastructure,
        production_transport=started["execution_mode"] == "production_default_transport",
    )
    report = {
        "verified": True,
        "complete": clean and not infrastructure and not unknown,
        "kind": "ALL_240_SAVED_HISTORY_SOURCE_INTERPRETATION_NOT_PARENT_REPAIR",
        "execution_mode": started["execution_mode"],
        "planned_calls": PLANNED_CALLS,
        "attempted": len(attempted),
        "reporter_receipts": len(receipt_ids),
        "new_native_action_trials": 0,
        "known_microinr": known,
        "retained_microinr": retained,
        "unknown_calls": unknown,
        "infrastructure_errors": infrastructure,
        "cumulative_calls": PRIOR_CALLS + len(ledger),
        "cumulative_known_microinr": PRIOR_KNOWN + known,
        "cumulative_retained_microinr": PRIOR_RETAINED + retained,
        "rows": rows,
        "actual_dispatch_order": sorted(
            dispatch_order,
            key=lambda row: (row["dispatch_time"] is None, row["dispatch_time"] or 0, row["id"]),
        ),
        "groups": groups,
        "contrasts": contrasts,
        "screen_gate": gate,
        "scenario_clusters": 40,
        "treated_histories": 184,
        "no_treatment_histories": 56,
        "development_exposed": True,
        "independent_native_review": False,
    }
    if output is not None:
        save(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preparing = sub.add_parser("prepare")
    preparing.add_argument("root")
    preparing.add_argument("--parent", required=True)
    preparing.add_argument("--prior-completion", required=True)
    checking = sub.add_parser("check")
    checking.add_argument("root")
    running = sub.add_parser("run")
    running.add_argument("root")
    running.add_argument("--allow-paid", action="store_true")
    reviewing = sub.add_parser("audit")
    reviewing.add_argument("root")
    reviewing.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.root, args.parent, args.prior_completion)
    elif args.command == "check":
        check(args.root)
    elif args.command == "run":
        run(args.root, allow_paid=args.allow_paid)
    else:
        result = audit(args.root, output=args.output)
        if args.output is None:
            print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
