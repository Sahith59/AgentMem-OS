"""Bounded, saved-history-only Sarvam evidence-view screen.

Preparation and auditing are offline. ``run`` is a separate, explicitly gated
paid operation; this module never resumes or retries a dispatched call.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from benchmarks import sarvam_evidence_views as views
from benchmarks.sarvam_fresh_confirmation.engine import report_view
from benchmarks.sarvam_fresh_confirmation.study import sources as parent_source_hashes
from benchmarks.sarvam_native_controls.engine import FORMAT, PROFILES, ControlMeter
from benchmarks.sarvam_native_controls.integrated import read
from benchmarks.sarvam_native_controls.transcript_finalizer import classify
from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.transport import (
    ENDPOINT,
    INPUT,
    MODELS,
    OUTPUT,
    call,
    canonical,
    digest,
    read_key,
    save,
)
from benchmarks.sarvam_ordinary_v2 import FINAL_INSTRUCTION
from benchmarks.sarvam_receipt_reporting import project
from benchmarks.sarvam_reporter_wording import CANDIDATE_FIRST, reservation
from benchmarks.sarvam_source_attribution import evaluate

FIRST_SENTENCE = (
    "Answer the original service question using the evidence in the supplied "
    "source record; do not repeat the question."
)
SYSTEM = FIRST_SENTENCE + "\n" + FINAL_INSTRUCTION.split("\n", 1)[1]
ARMS = ("chronological", "grouped_lossless", "grouped_actor_prose_withheld")
ACTOR_FIELDS = {"role", "content", "reasoning_content", "refusal", "tool_calls"}
TEXT_FIELDS = ("content", "reasoning_content", "refusal")
SELECTED_WORKFLOWS = {
    "known_true": ("fresh-09", "fresh-11"),
    "known_false": ("fresh-12", "fresh-10"),
    "unknown": ("fresh-13", "fresh-16"),
}
MAXIMUM_RESERVATION = 232_243_200
STAGE = "evidence_views"
REPO = Path(__file__).parents[1]


def _original(request, trace):
    if request.get("model") not in PROFILES or {
        k: v for k, v in request.items() if k != "messages"
    } != {
        "model": request["model"],
        "temperature": 0,
        "reasoning_effort": PROFILES[request["model"]],
        "max_tokens": 8192,
        "stream": False,
        "n": 1,
        "response_format": FORMAT,
    }:
        raise ValueError("Reporter profile changed")
    if (
        len(request["messages"]) != 2
        or request["messages"][0]
        != {
            "role": "system",
            "content": CANDIDATE_FIRST + "\n" + FINAL_INSTRUCTION.split("\n", 1)[1],
        }
        or request["messages"][1].get("role") != "user"
    ):
        raise ValueError("Not the completed v3 candidate request")
    grouped = views.from_final_request(request, trace)
    return grouped["audit"], grouped


def render(request, trace, arm):
    """Return paid request plus local-only evidence sidecar and omission ledger."""
    if arm not in ARMS:
        raise ValueError("Undeclared arm")
    original, grouped = _original(request, trace)
    paid = copy.deepcopy(request)
    paid["messages"][0]["content"] = SYSTEM
    ledger = []
    if arm == "chronological":
        evidence = original
    else:
        evidence = copy.deepcopy(grouped)
        evidence.pop("audit")
        if arm == "grouped_actor_prose_withheld":
            for claim in evidence["actor_claims"]:
                message = claim["message"]
                if set(message) - ACTOR_FIELDS or "role" not in message:
                    raise ValueError("Unknown actor field: refuse lossy rendering")
                for field in TEXT_FIELDS:
                    if field in message:
                        value = message[field]
                        if value is not None and not isinstance(value, str):
                            raise ValueError("Unexpected nontext actor prose field")
                        if isinstance(value, str) and value:
                            ledger.append(
                                {
                                    "message_index": claim["message_index"],
                                    "field": field,
                                    "value_sha256": digest(message.pop(field)),
                                }
                            )
            if views.reconstruct(grouped, actual_tool_trace=trace) != original:
                raise ValueError("Original evidence failed reconstruction")
            if evidence["execution_view"] != grouped["execution_view"]:
                raise ValueError("Execution receipts changed")
            if evidence["source_view"] != grouped["source_view"]:
                raise ValueError("Source evidence changed")
    paid["messages"][1]["content"] = canonical(evidence)
    if any(k in paid for k in ("tools", "tool_choice")) or "audit" in evidence:
        raise ValueError("Sidecar or executable tools in paid request")
    forbidden = {"family", "gold", "expected_answer", "expected_state", "source_annotations"}

    def contains_forbidden_key(value):
        if isinstance(value, dict):
            return bool(forbidden & value.keys()) or any(
                contains_forbidden_key(child) for child in value.values()
            )
        return isinstance(value, list) and any(contains_forbidden_key(child) for child in value)

    if contains_forbidden_key(evidence):
        raise ValueError("Forbidden label or source annotation in paid evidence")
    return paid, {"original": original, "actual_tool_trace": copy.deepcopy(trace)}, ledger


def verify_omissions(paid, sidecar, ledger, arm):
    original = sidecar["original"]
    trace = sidecar["actual_tool_trace"]
    evidence = json.loads(paid["messages"][1]["content"])
    if arm == "chronological":
        if evidence != original or ledger:
            raise ValueError("Chronological evidence changed")
        return
    if arm == "grouped_lossless":
        if views.reconstruct(evidence, actual_tool_trace=trace) != original or ledger:
            raise ValueError("Lossless group changed")
        return
    if arm != "grouped_actor_prose_withheld":
        raise ValueError("Unknown arm")
    base = views.from_final_request(
        {
            **paid,
            "messages": [
                paid["messages"][0],
                {
                    "role": "user",
                    "content": canonical(original),
                },
            ],
        },
        trace,
    )
    base.pop("audit")
    expected = []
    for claim in base["actor_claims"]:
        for field in TEXT_FIELDS:
            if field in claim["message"]:
                value = claim["message"][field]
                if value is not None and not isinstance(value, str):
                    raise ValueError("Unexpected nontext actor prose field")
                if isinstance(value, str) and value:
                    expected.append(
                        {
                            "message_index": claim["message_index"],
                            "field": field,
                            "value_sha256": digest(claim["message"].pop(field)),
                        }
                    )
    if expected != ledger or evidence != base:
        raise ValueError("Omission ledger/view mismatch")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(Path(source).read_bytes())
        stream.flush()
        os.fsync(stream.fileno())


def _semantic_review(parent, parent_audit, parent_manifest):
    review = read(parent / "semantic-review.json")
    if review.get("status") != "ASSISTANT_LANGUAGE_REVIEW_COMPLETE" or review.get(
        "parent_verification_sha256"
    ) != _sha(parent_audit):
        raise ValueError("Complete assistant-language review is not bound to parent audit")
    episodes = review.get("episodes")
    expected_ids = [j["id"] for j in parent_manifest["jobs"]]
    if (
        not isinstance(episodes, list)
        or len(episodes) != 240
        or [item.get("id") for item in episodes] != expected_ids
        or any(
            set(item.get("reporters", [])) != {"control", "candidate"}
            or len(item["reporters"]) != 2
            for item in episodes
        )
    ):
        raise ValueError("Semantic review lacks exact 240/480 reporter coverage")
    selected_episode_ids = {
        f"confirm-{workflow}-{language}-m{i}"
        for workflows in SELECTED_WORKFLOWS.values()
        for workflow in workflows
        for language in ("en", "hi", "te")
        for i in range(2)
    }
    if any(
        item["id"] in selected_episode_ids and item.get("source_gold_ambiguous") is not False
        for item in episodes
    ):
        raise ValueError("Selected source/gold ambiguity remains unresolved")
    if Counter(j["case_id"].rsplit("-", 1)[-1] for j in parent_manifest["jobs"]) != {
        "en": 80,
        "hi": 80,
        "te": 80,
    }:
        raise ValueError("Parent language coverage changed")
    files = review.get("review_files")
    if not isinstance(files, dict) or set(files) != {"en", "hi", "te"}:
        raise ValueError("Missing final language review files")
    for language, entry in files.items():
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256"}:
            raise ValueError("Malformed language review reference")
        relative = PurePosixPath(entry["file"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or "\\" in entry["file"]
            or entry["file"].startswith(".")
        ):
            raise ValueError("Language review path escapes parent archive")
        if _sha(parent / str(relative)) != entry["sha256"]:
            raise ValueError(f"{language} review hash changed")
        document = read(parent / str(relative))
        language_ids = [
            j["id"] for j in parent_manifest["jobs"] if j["case_id"].endswith("-" + language)
        ]
        if document.get("reviewed_episode_ids") != language_ids:
            raise ValueError(f"{language} review lacks exact 80-episode coverage")
        reporter_reviews = document.get("reporter_reviews")
        if not isinstance(reporter_reviews, list) or len(reporter_reviews) != 160:
            raise ValueError(f"{language} review lacks 160 reporter judgments")
        observed = []
        for row in reporter_reviews:
            if not isinstance(row, dict) or any(
                row.get(key) not in {"correct", "incorrect", "ambiguous"}
                for key in ("factual_verdict", "status_verdict")
            ):
                raise ValueError(f"{language} reporter verdict missing")
            observed.append((row.get("episode_id"), row.get("variant")))
            if row.get("episode_id") in selected_episode_ids and (
                row["factual_verdict"] == "ambiguous" or row["status_verdict"] == "ambiguous"
            ):
                raise ValueError("Selected reporter judgment remains ambiguous")
        expected = {
            (episode_id, variant)
            for episode_id in language_ids
            for variant in ("control", "candidate")
        }
        if set(observed) != expected or len(set(observed)) != 160:
            raise ValueError(f"{language} review lacks exact reporter coverage")
    return review


def _parent_artifacts(parent, selection, parent_audit):
    """Read only source metadata until the audited, complete parent gate passes."""
    selected = read(selection)
    if selected.get("status") != "PROSPECTIVE_SOURCE_ONLY_SELECTION_NO_PAID_RUN":
        raise ValueError("Prospective source-only selection missing")
    if (
        selected.get("gold_or_saved_answers_read") is not False
        or selected.get("planned_reporter_calls_if_qualified") != 108
    ):
        raise ValueError("Selection provenance changed")
    if {k: tuple(v) for k, v in selected["selected_workflows"].items()} != SELECTED_WORKFLOWS:
        raise ValueError("Selected workflows changed")
    runtime = read(parent / "runtime.json")
    if _sha(parent / "runtime.json") != selected["runtime_file_sha256"]:
        raise ValueError("Selected source inventory changed")
    expected_ids = {
        f"{w}-{lang}"
        for ws in SELECTED_WORKFLOWS.values()
        for w in ws
        for lang in ("en", "hi", "te")
    }
    if set(selected["case_ids"]) != expected_ids or len(selected["case_ids"]) != 18:
        raise ValueError("Selection does not contain exactly 18 renderings")
    audit = read(parent_audit)
    if (
        audit.get("verified") is not True
        or audit.get("complete") is not True
        or audit.get("native_episodes") != 240
        or audit.get("unknown_calls") != 0
    ):
        raise ValueError("Parent 240-episode audit is not complete")
    execution = read(parent / "execution.json")
    if len(execution["outcomes"]) != 240:
        raise ValueError("Parent execution incomplete")
    parent_manifest = read(parent / "manifest.json")
    if (
        read(parent / "started.json").get("manifest_sha256") != digest(parent_manifest)
        or len(parent_manifest.get("jobs", [])) != 240
        or [j["id"] for j in parent_manifest["jobs"]] != [o["id"] for o in execution["outcomes"]]
    ):
        raise ValueError("Parent native manifest/start/execution binding changed")
    _semantic_review(parent, parent_audit, parent_manifest)
    for relative, expected in parent_manifest.get("files_sha256", {}).items():
        if _sha(parent / relative) != expected:
            raise ValueError("Parent frozen input changed")
    for relative, expected in parent_manifest.get("source_sha256", {}).items():
        if digest((parent / "source" / relative).read_text()) != expected:
            raise ValueError("Parent frozen source changed")
    cases = {c["id"]: c for c in runtime}
    if not expected_ids <= cases.keys():
        raise ValueError("Selected rendering absent")
    for ws in SELECTED_WORKFLOWS.values():
        for workflow in ws:
            for language in ("en", "hi", "te"):
                case = cases[f"{workflow}-{language}"]
                if case["workflow"] != workflow or case["language"] != language:
                    raise ValueError("Selection/rendering mismatch")
    return selected, cases


def source_hashes():
    result = parent_source_hashes()
    for relative in (
        "benchmarks/sarvam_evidence_ablation.py",
        "benchmarks/sarvam_evidence_views.py",
    ):
        result[relative] = digest((REPO / relative).read_text())
    return result


def prepare(root, parent, selection, parent_audit):
    """Freeze 36 native histories and 108 requests; hash but do not parse gold."""
    root, parent, selection, parent_audit = map(Path, (root, parent, selection, parent_audit))
    if (
        not root.is_dir()
        or any(
            (root / name).exists()
            for name in ("manifest.json", "inputs.json", "started.json", "calls")
        )
        or not all(
            (root / name).is_file()
            for name in ("budget.json", "pacing-control.json", "PROTOCOL.md")
        )
    ):
        raise FileExistsError(
            "Preparation requires reviewed config and no earlier frozen/run artifacts"
        )
    selected, cases = _parent_artifacts(parent, selection, parent_audit)
    histories = []
    for case_id in selected["case_ids"]:
        for i, model in enumerate(PROFILES):
            episode_id = f"confirm-{case_id}-m{i}"
            folder = parent / "episodes" / episode_id
            paired_path = folder / "paired.json"
            paired = read(paired_path)
            original = paired["original"]
            if original["model"] != model or original["case_id"] != case_id:
                raise ValueError("Parent episode identity mismatch")
            if original["history"][:2][1]["role"] != "user":
                raise ValueError("Native history missing user")
            candidate_path = parent / "calls" / (episode_id + "-final-candidate.request.json")
            saved = read(candidate_path)
            request = saved["request"]
            if saved["sha256"] != digest(request) or saved["endpoint"] != ENDPOINT:
                raise ValueError("Parent request wrapper changed")
            evidence, _ = _original(request, original["tool_trace"])
            if evidence["original_messages"] != original["history"]:
                raise ValueError("Parent native history changed")
            if request["model"] != model:
                raise ValueError("Parent model mismatch")
            if (
                json.loads(original["history"][1]["content"])["source_messages"]
                != cases[case_id]["source_messages"]
            ):
                raise ValueError("Source messages changed")
            histories.append(
                {
                    "id": episode_id,
                    "case_id": case_id,
                    "model": model,
                    "parent_paired_sha256": _sha(paired_path),
                    "parent_candidate_request_sha256": _sha(candidate_path),
                    "request": request,
                    "tool_trace": original["tool_trace"],
                }
            )
    if len(histories) != 36:
        raise ValueError("Expected 36 histories")
    shuffled = list(histories)
    random.Random(202610133).shuffle(shuffled)
    orderings = (ARMS, ARMS[1:] + ARMS[:1], ARMS[2:] + ARMS[:2])
    jobs = []
    order_counts = Counter()
    for history in shuffled:
        for arm in orderings[order_counts[history["model"]] % 3]:
            req, _, _ = render(history["request"], history["tool_trace"], arm)
            jobs.append(
                {
                    "id": history["id"] + "-" + arm,
                    "history_id": history["id"],
                    "arm": arm,
                    "request_sha256": digest(req),
                    "reservation_microinr": reservation(req),
                }
            )
        order_counts[history["model"]] += 1
    if len(jobs) != 108 or Counter(j["arm"] for j in jobs) != {a: 36 for a in ARMS}:
        raise ValueError("Unbalanced arms")
    worst = sum(j["reservation_microinr"] for j in jobs)
    if worst != MAXIMUM_RESERVATION:
        raise ValueError("Full-context reservation changed")
    budget = read(root / "budget.json") if (root / "budget.json").exists() else None
    if budget is None:
        raise ValueError("Create a reviewed budget.json before prepare")
    if (
        budget["phase_cap_microinr"] != 240_000_000
        or budget["campaign_cap_microinr"] != 6_000_000_000
        or budget["stages"][STAGE]["calls"] != 108
        or budget["stages"][STAGE]["cap_microinr"] != 240_000_000
        or worst + budget["prior_retained_microinr"] > budget["campaign_cap_microinr"]
    ):
        raise ValueError("Budget contract changed")
    parent_verified = read(parent_audit)
    if (
        budget["prior_calls"] != parent_verified["research_calls"]
        or budget["prior_known_microinr"]
        != round(parent_verified["research_known_inr"] * 1_000_000)
        or budget["prior_retained_microinr"]
        != round(parent_verified["research_retained_inr"] * 1_000_000)
    ):
        raise ValueError("Budget carry differs from the audited parent campaign")
    if read(root / "pacing-control.json")["rpm"] != 90:
        raise ValueError("Pacing must be 90 RPM")
    archived_parent = root / "provenance" / "parent"
    for name in (
        "manifest.json",
        "started.json",
        "execution.json",
        "runtime.json",
        "gold.json",
        "review.json",
        "budget.json",
        "PROTOCOL.md",
        "pacing-control.json",
    ):
        _copy_once(parent / name, archived_parent / name)
    _copy_once(parent_audit, archived_parent / "verification.json")
    _copy_once(selection, root / "provenance" / "prospective-selection.json")
    _copy_once(parent / "semantic-review.json", archived_parent / "semantic-review.json")
    semantic = read(parent / "semantic-review.json")
    for entry in semantic["review_files"].values():
        _copy_once(parent / entry["file"], archived_parent / entry["file"])
    for source in sorted((parent / "source").rglob("*")):
        if source.is_file():
            _copy_once(source, archived_parent / "source" / source.relative_to(parent / "source"))
    for history in histories:
        episode_id = history["id"]
        _copy_once(
            parent / "episodes" / episode_id / "paired.json",
            archived_parent / "episodes" / episode_id / "paired.json",
        )
        name = episode_id + "-final-candidate.request.json"
        _copy_once(parent / "calls" / name, archived_parent / "calls" / name)
    save(root / "inputs.json", histories)
    for history in histories:
        _, sidecar, _ = render(history["request"], history["tool_trace"], ARMS[0])
        destination = root / "sidecars" / (history["id"] + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        save(destination, sidecar)
    for job in jobs:
        history = next(h for h in histories if h["id"] == job["history_id"])
        _, _, omissions = render(history["request"], history["tool_trace"], job["arm"])
        destination = root / "omissions" / (job["id"] + ".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        save(destination, omissions)
    files = ("inputs.json", "budget.json", "pacing-control.json", "PROTOCOL.md")
    package = {
        "jobs": jobs,
        "source_sha256": source_hashes(),
        "files_sha256": {
            str(p.relative_to(root)): _sha(p)
            for p in [
                *(root / n for n in files),
                *sorted((root / "sidecars").glob("*.json")),
                *sorted((root / "omissions").glob("*.json")),
                *sorted(p for p in (root / "provenance").rglob("*") if p.is_file()),
            ]
        },
        "selection_file": "provenance/prospective-selection.json",
        "parent_root": "provenance/parent",
        "parent_manifest_sha256": _sha(archived_parent / "manifest.json"),
        "parent_verification_sha256": _sha(archived_parent / "verification.json"),
        "maximum_calls": 108,
        "worst_case_retained_microinr": worst,
        "workers": 4,
        "batch_size": 4,
        "stop_after_batch_errors": 2,
        "no_retries": True,
        "saved_native_history_only": True,
        "new_native_action_trials": 0,
        "development_exposed_selection": True,
    }
    for relative in package["source_sha256"]:
        destination = root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as stream:
            stream.write((REPO / relative).read_text())
    save(root / "manifest.json", package)
    return package


def check(root):
    root = Path(root)
    package = read(root / "manifest.json")
    if package["source_sha256"] != source_hashes():
        raise ValueError("Frozen source changed")
    for relative, expected in package["source_sha256"].items():
        if digest((root / "source" / relative).read_text()) != expected:
            raise ValueError("Frozen source copy changed")
    for relative, expected in package["files_sha256"].items():
        if _sha(root / relative) != expected:
            raise ValueError("Frozen input/configuration changed")
    if (
        package["parent_root"] != "provenance/parent"
        or package["selection_file"] != "provenance/prospective-selection.json"
    ):
        raise ValueError("Archive paths changed")
    parent = root / package["parent_root"]
    selection = root / package["selection_file"]
    if (
        _sha(parent / "manifest.json") != package["parent_manifest_sha256"]
        or _sha(parent / "verification.json") != package["parent_verification_sha256"]
    ):
        raise ValueError("Parent verification binding changed")
    _parent_artifacts(parent, selection, parent / "verification.json")
    verified = read(parent / "verification.json")
    budget = read(root / "budget.json")
    if (
        budget["prior_calls"] != verified["research_calls"]
        or budget["prior_known_microinr"] != round(verified["research_known_inr"] * 1_000_000)
        or budget["prior_retained_microinr"] != round(verified["research_retained_inr"] * 1_000_000)
    ):
        raise ValueError("Archived carry changed")
    inputs = {h["id"]: h for h in read(root / "inputs.json")}
    if len(inputs) != 36 or len(package["jobs"]) != 108:
        raise ValueError("Incomplete frozen design")
    for history in inputs.values():
        if (
            _sha(parent / "episodes" / history["id"] / "paired.json")
            != history["parent_paired_sha256"]
            or _sha(parent / "calls" / (history["id"] + "-final-candidate.request.json"))
            != history["parent_candidate_request_sha256"]
        ):
            raise ValueError("Parent history changed")
    for job in package["jobs"]:
        h = inputs[job["history_id"]]
        req, sidecar, ledger = render(h["request"], h["tool_trace"], job["arm"])
        verify_omissions(req, sidecar, ledger, job["arm"])
        if (
            read(root / "sidecars" / (h["id"] + ".json")) != sidecar
            or read(root / "omissions" / (job["id"] + ".json")) != ledger
        ):
            raise ValueError("Sidecar or omission ledger changed")
        if digest(req) != job["request_sha256"] or reservation(req) != job["reservation_microinr"]:
            raise ValueError("Frozen request changed")
    return package, inputs


def execute_job(job, history, dispatch):
    request, _, _ = render(history["request"], history["tool_trace"], job["arm"])
    try:
        return classify(dispatch(job["id"], request))
    except Exception as exc:
        return {"terminal": "infrastructure_error", "error_class": type(exc).__name__}


def run(root, *, allow_paid=False):
    if not allow_paid:
        raise ValueError("Paid run requires explicit --allow-paid after parent review")
    root = Path(root)
    package, inputs = check(root)
    if (root / "started.json").exists():
        raise FileExistsError("Already started: no retry or continuation")
    save(root / "started.json", {"manifest_sha256": digest(package)})
    meter, pacer, key = ControlMeter(root), Pacer(root), read_key()

    def dispatch(job, request):
        save(meter.root / (job + ".dispatch.json"), pacer.wait())
        return call(meter, job, STAGE, request, key)

    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(package["jobs"]), 4):
            jobs = package["jobs"][offset : offset + 4]
            results = [
                f.result()
                for f in [
                    pool.submit(execute_job, j, inputs[j["history_id"]], dispatch) for j in jobs
                ]
            ]
            for job, result in zip(jobs, results):
                save(root / (job["id"] + ".result.json"), result)
                outcomes.append({"id": job["id"], "terminal": result["terminal"]})
            errors = sum(r["terminal"] == "infrastructure_error" for r in results)
            print(
                {"completed": len(outcomes), "errors": errors, "meter": meter.summary()}, flush=True
            )
            if errors >= 2:
                break
    check(root)
    save(root / "execution.json", {"outcomes": outcomes, "meter": meter.summary()})


def audit(root):
    with (
        patch("socket.socket", side_effect=RuntimeError("Offline audit")),
        patch("socket.create_connection", side_effect=RuntimeError("Offline audit")),
    ):
        return _audit(Path(root))


def _audit(root):
    package, inputs = check(root)
    if read(root / "started.json") != {"manifest_sha256": digest(package)}:
        raise ValueError("Start binding changed")
    execution = read(root / "execution.json")
    jobs = package["jobs"][: len(execution["outcomes"])]
    if [j["id"] for j in jobs] != [o["id"] for o in execution["outcomes"]]:
        raise ValueError("Execution order changed")
    ids = {j["id"] for j in jobs}
    with sqlite3.connect(f"file:{root}/calls/meter.sqlite?mode=ro", uri=True) as db:
        ledger = {r[0]: r[1:] for r in db.execute("SELECT * FROM calls")}
    if set(ledger) != ids or len(ids) != len(jobs):
        raise ValueError("Ledger inventory changed")
    for suffix in ("request", "receipt", "dispatch"):
        if {
            p.name.removesuffix(f".{suffix}.json")
            for p in (root / "calls").glob(f"*.{suffix}.json")
        } != ids:
            raise ValueError("Call artifact inventory changed")
    if {p.name.removesuffix(".result.json") for p in root.glob("*.result.json")} != ids:
        raise ValueError("Result inventory changed")
    parent = root / package["parent_root"]
    cases = {c["id"]: c for c in read(parent / "runtime.json")}
    labels = read(parent / "gold.json")  # Offline audit only; never in preparation/run requests.
    rows, response_ids = [], set()
    known = retained = unknown = 0
    for job, outcome in zip(jobs, execution["outcomes"]):
        history = inputs[job["history_id"]]
        request, sidecar, omissions = render(history["request"], history["tool_trace"], job["arm"])
        verify_omissions(request, sidecar, omissions, job["arm"])
        if read(root / "calls" / (job["id"] + ".request.json")) != {
            "endpoint": ENDPOINT,
            "request": request,
            "sha256": digest(request),
        }:
            raise ValueError("Sent request changed")
        dispatch = read(root / "calls" / (job["id"] + ".dispatch.json"))
        if dispatch.get("pacing_control", {}).get("rpm") != 90 or not isinstance(
            dispatch.get("dispatch_time"), (int, float)
        ):
            raise ValueError("Paced dispatch record changed")
        receipt = read(root / "calls" / (job["id"] + ".receipt.json"))
        if receipt["job"] != job["id"] or receipt["request_sha256"] != digest(request):
            raise ValueError("Receipt/request binding changed")
        stage, reserved, charged, cost, status = ledger[job["id"]]
        if stage != STAGE or reserved != reservation(request):
            raise ValueError("Reservation mismatch")
        try:
            body = json.loads(receipt.get("body_text", ""))
        except ValueError:
            body = {}
        if status == "known":
            usage = body["usage"]
            if (
                any(
                    type(usage[k]) is not int or usage[k] < 0
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                or usage["prompt_tokens"] > MODELS[request["model"]]
                or usage["completion_tokens"] > 8192
                or cost != usage["prompt_tokens"] * INPUT + usage["completion_tokens"] * OUTPUT
                or charged != (cost * 5 + 3) // 4
            ):
                raise ValueError("Usage/charge mismatch")
            known += cost
        elif status == "unknown" and cost is None and charged == reserved:
            unknown += 1
        else:
            raise ValueError("Unknown ledger state")
        retained += charged
        if status == "known" and receipt.get("status") == 200:
            if body.get("model") != request["model"] or not isinstance(body.get("id"), str):
                raise ValueError("Saved provider identity changed")
            if body["id"] in response_ids:
                raise ValueError("Duplicate provider response ID")
            response_ids.add(body["id"])

        def replay(_job, _request):
            if _job != job["id"] or _request != request:
                raise ValueError("Replay request changed")
            if (
                receipt.get("status") != 200
                or status != "known"
                or body.get("model") != request["model"]
            ):
                raise RuntimeError("Saved provider failure")
            return body

        result = execute_job(job, history, replay)
        if (
            result != read(root / (job["id"] + ".result.json"))
            or result["terminal"] != outcome["terminal"]
        ):
            raise ValueError("Saved classification changed")
        paired = read(parent / "episodes" / history["id"] / "paired.json")
        original = paired["original"]
        case = cases[history["case_id"]]
        label = labels[history["case_id"]]
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
        contract = {
            "runtime_sha256": digest(case),
            "gold_sha256": digest(label),
            "question_target": label["question_target"],
        }
        attribution = evaluate(case, label, view, contract)
        rows.append(
            {
                "id": job["id"],
                "history_id": history["id"],
                "case_id": case["id"],
                "model": history["model"],
                "arm": job["arm"],
                "terminal": result["terminal"],
                "valid_final": result.get("parsed") is not None,
                "raw": result.get("parsed"),
                "application_output": projection["application_output"] if projection else None,
                "omission_ledger": omissions,
                "sidecar_sha256": digest(sidecar),
                **attribution,
            }
        )
    budget = read(root / "budget.json")
    if (
        retained > budget["phase_cap_microinr"]
        or retained + budget["prior_retained_microinr"] > budget["campaign_cap_microinr"]
    ):
        raise ValueError("Budget exceeded")
    if execution["meter"] != {
        STAGE: {
            "calls": len(ids),
            "charged_with_headroom_inr": retained / 1e6,
            "known_usage_estimate_inr": known / 1e6,
            "unknown_or_pending": unknown,
        }
    }:
        raise ValueError("Execution meter summary changed")
    groups = [
        {
            "model": model,
            "arm": arm,
            "n": sum(r["model"] == model and r["arm"] == arm for r in rows),
            "factual_answer_correct": sum(
                bool(r["factual_answer_correct"])
                for r in rows
                if r["model"] == model and r["arm"] == arm
            ),
            "raw_status_matches_execution": sum(
                bool(r["raw_status_matches_execution"])
                for r in rows
                if r["model"] == model and r["arm"] == arm
            ),
        }
        for model in PROFILES
        for arm in ARMS
    ]
    return {
        "verified": True,
        "complete": (
            len(rows) == 108
            and unknown == 0
            and all(r["terminal"] != "infrastructure_error" for r in rows)
        ),
        "calls": len(rows),
        "rows": rows,
        "groups": groups,
        "known_inr": known / 1e6,
        "retained_inr": retained / 1e6,
        "unknown_calls": unknown,
        "new_native_action_trials": 0,
        "limitation": (
            "Development-exposed saved histories; reporter-only causal screen, "
            "no memory or fresh-task claim."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "check", "run", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--parent-audit", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if not all((args.parent, args.selection, args.parent_audit)):
            parser.error("prepare requires --parent, --selection and --parent-audit")
        prepare(args.root, args.parent, args.selection, args.parent_audit)
    elif args.command == "check":
        check(args.root)
    elif args.command == "run":
        run(args.root, allow_paid=args.allow_paid)
    else:
        if not args.output:
            parser.error("audit requires --output")
        save(args.output, audit(args.root))
