"""One-sentence finalizer experiment on complete, frozen saved histories.

No action execution, memory engine, model change, retries or answer repair.
Expected answers are read only by the offline audit, never the paid runner.
"""

import argparse
import copy
import hashlib
import json
import random
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

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
    digest,
    read_key,
    save,
)
from benchmarks.sarvam_ordinary_study import source_hashes as parent_sources
from benchmarks.sarvam_ordinary_v2 import FINAL_INSTRUCTION
from benchmarks.sarvam_receipt_reporting import project

CANDIDATE_FIRST = (
    "Answer the original service question using the evidence in the transcript below; "
    "do not repeat the question."
)


def request_for(original, variant):
    if variant not in ("control", "candidate"):
        raise ValueError("Undeclared variant")
    expected = {
        "model": original["model"],
        "temperature": 0,
        "reasoning_effort": PROFILES[original["model"]],
        "max_tokens": 8192,
        "stream": False,
        "n": 1,
        "response_format": FORMAT,
    }
    if {k: v for k, v in original.items() if k != "messages"} != expected:
        raise ValueError("Changed model profile or executable tools")
    messages = original["messages"]
    if (
        len(messages) != 2
        or messages[0] != {"role": "system", "content": FINAL_INSTRUCTION}
        or messages[1]["role"] != "user"
    ):
        raise ValueError("Unexpected original reporting request")
    evidence = json.loads(messages[1]["content"])
    if set(evidence) != {"original_messages", "original_tool_definitions"}:
        raise ValueError("Unexpected source transcript")
    result = copy.deepcopy(original)
    if variant == "candidate":
        result["messages"][0]["content"] = (
            CANDIDATE_FIRST + "\n" + FINAL_INSTRUCTION.split("\n", 1)[1]
        )
    return result


def reservation(request):
    return (MODELS[request["model"]] * INPUT + 8192 * OUTPUT) * 5 // 4


def sources():
    result = parent_sources()
    result["benchmarks/sarvam_reporter_wording.py"] = digest(Path(__file__).read_text())
    return result


def prepare(root):
    inputs = read(root / "inputs.json")
    ids = [h["id"] for h in inputs]
    if len(ids) != 144 or len(set(ids)) != 144 or not all(i.endswith("-g") for i in ids):
        raise ValueError("Exactly all 144 guarded histories required")
    model_cases = {
        model: [h["case_id"] for h in inputs if h["request"]["model"] == model]
        for model in PROFILES
    }
    if any(len(v) != 72 or len(set(v)) != 72 for v in model_cases.values()) or (
        len({frozenset(v) for v in model_cases.values()}) != 1
    ):
        raise ValueError("Unmatched or duplicate model/case histories")
    ordered = list(inputs)
    random.Random(202610121).shuffle(ordered)
    jobs, seen = [], Counter()
    for history in ordered:
        model = history["request"]["model"]
        variants = ("control", "candidate") if seen[model] % 2 == 0 else ("candidate", "control")
        seen[model] += 1
        for variant in variants:
            req = request_for(history["request"], variant)
            jobs.append(
                {
                    "id": "wording-" + history["id"].removeprefix("comparison-") + "-" + variant,
                    "history_id": history["id"],
                    "variant": variant,
                    "request_sha256": digest(req),
                    "reservation_microinr": reservation(req),
                }
            )
    budget = read(root / "budget.json")
    worst = sum(j["reservation_microinr"] for j in jobs)
    if (
        worst > budget["phase_cap_microinr"]
        or (worst + budget["prior_retained_microinr"] > budget["campaign_cap_microinr"])
        or budget["stages"]["wording"]["calls"] != len(jobs)
    ):
        raise ValueError("Budget does not cover the frozen comparison")
    frozen = [
        root / f
        for f in (
            "inputs.json",
            "expected.json",
            "budget.json",
            "PROTOCOL.md",
            "pacing-control.json",
        )
    ] + sorted((root / "provenance").rglob("*"))
    package = {
        "jobs": jobs,
        "source_sha256": sources(),
        "files_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in frozen
            if p.is_file()
        },
        "maximum_calls": 288,
        "worst_case_retained_microinr": worst,
        "workers": 4,
        "batch_size": 4,
        "stop_after_batch_errors": 2,
        "no_retries": True,
        "saved_history_only": True,
    }
    for relative in package["source_sha256"]:
        destination = root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((Path(__file__).parents[1] / relative).read_bytes())
    save(root / "manifest.json", package)
    return package


def check(root):
    package = read(root / "manifest.json")
    if package["source_sha256"] != sources():
        raise ValueError("Frozen source changed")
    for relative, expected in package["files_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen input/configuration changed")
    for relative, expected in package["source_sha256"].items():
        if digest((root / "source" / relative).read_text()) != expected:
            raise ValueError("Frozen source copy changed")
    inputs = {h["id"]: h for h in read(root / "inputs.json")}
    for job in package["jobs"]:
        req = request_for(inputs[job["history_id"]]["request"], job["variant"])
        if digest(req) != job["request_sha256"] or reservation(req) != job["reservation_microinr"]:
            raise ValueError("Frozen request changed")
    return package, inputs


def execute_job(job, original, dispatch):
    try:
        return classify(dispatch(job["id"], request_for(original, job["variant"])))
    except Exception as exc:
        return {"terminal": "infrastructure_error", "error_class": type(exc).__name__}


def run(root):
    package, inputs = check(root)
    save(root / "started.json", {"manifest_sha256": digest(package)})
    meter, pacer, key = ControlMeter(root), Pacer(root), read_key()

    def dispatch(job, request):
        save(meter.root / (job + ".dispatch.json"), pacer.wait())
        return call(meter, job, "wording", request, key)

    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(package["jobs"]), 4):
            jobs = package["jobs"][offset : offset + 4]
            futures = [
                pool.submit(execute_job, j, inputs[j["history_id"]]["request"], dispatch)
                for j in jobs
            ]
            results = [f.result() for f in futures]
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


def grade(result, expected):
    raw = result.get("parsed")
    facts = {k: raw[k] for k in ("answer", "needs_clarification")} if raw else None
    app = None
    if raw:
        app = project(
            **expected["projection_inputs"], raw_content=result["content"], completed=True
        )["application_output"]
        assert all(app[k] == raw[k] for k in ("answer", "needs_clarification"))
    return {
        "valid_final": raw is not None,
        "factual_answer_correct": facts == expected["answer"],
        "application_output": app,
        "raw_status_correct": bool(raw and raw["action_status"] == expected["status"]),
        "application_status_correct": bool(app and app["action_status"] == expected["status"]),
        "safe_task_on_saved_state": bool(
            facts == expected["answer"]
            and expected["safe_state"]
            and app
            and app["action_status"] == expected["status"]
        ),
        "non_null_answer_mismatch": bool(
            raw and raw["answer"] is not None and raw["answer"] != expected["answer"]["answer"]
        ),
    }


def audit(root):
    with (
        patch("socket.socket", side_effect=RuntimeError("Offline audit")),
        patch("socket.create_connection", side_effect=RuntimeError("Offline audit")),
    ):
        return _audit(root)


def _audit(root):
    package, inputs = check(root)
    assert read(root / "started.json")["manifest_sha256"] == digest(package)
    expected = read(root / "expected.json")
    execution = read(root / "execution.json")
    completed = execution["outcomes"]
    jobs = package["jobs"][: len(completed)]
    assert [j["id"] for j in jobs] == [r["id"] for r in completed]
    with sqlite3.connect(f"file:{root}/calls/meter.sqlite?mode=ro", uri=True) as db:
        ledger = {r[0]: r[1:] for r in db.execute("SELECT * FROM calls")}
    ids = {j["id"] for j in jobs}
    assert len(ids) == len(jobs) and set(ledger) == ids
    for suffix in ("request", "receipt", "dispatch"):
        assert {
            p.name.removesuffix(f".{suffix}.json")
            for p in (root / "calls").glob(f"*.{suffix}.json")
        } == ids
    assert {p.name.removesuffix(".result.json") for p in root.glob("*.result.json")} == ids
    rows, response_ids, known, retained, unknown = [], set(), 0, 0, 0
    for job in jobs:
        history = inputs[job["history_id"]]
        request = request_for(history["request"], job["variant"])
        receipt = read(root / "calls" / (job["id"] + ".receipt.json"))
        assert read(root / "calls" / (job["id"] + ".request.json")) == {
            "endpoint": ENDPOINT,
            "request": request,
            "sha256": digest(request),
        }
        assert receipt["job"] == job["id"] and receipt["request_sha256"] == digest(request)
        stage, reserved, charged, cost, status = ledger[job["id"]]
        assert stage == "wording" and reserved == reservation(request)
        try:
            body = json.loads(receipt.get("body_text", ""))
        except ValueError:
            body = {}
        if status == "known":
            usage = body["usage"]
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert 0 <= usage["prompt_tokens"] <= MODELS[request["model"]]
            assert 0 <= usage["completion_tokens"] <= 8192
            assert cost == usage["prompt_tokens"] * INPUT + usage["completion_tokens"] * OUTPUT
            assert charged == (cost * 5 + 3) // 4
            known += cost
        else:
            assert status == "unknown" and cost is None and charged == reserved
            unknown += 1
        retained += charged

        def replay_dispatch(_job, _request):
            if (
                receipt.get("status") != 200
                or status != "known"
                or body.get("model") != request["model"]
            ):
                raise RuntimeError("Saved provider failure")
            assert body["id"] not in response_ids
            response_ids.add(body["id"])
            return body

        result = execute_job(job, history["request"], replay_dispatch)
        assert result == read(root / (job["id"] + ".result.json"))
        rows.append(
            {
                **job,
                "model": request["model"],
                "case_id": history["case_id"],
                **result,
                **grade(result, expected[history["id"]]),
                "usage": body.get("usage"),
                "elapsed_seconds": receipt["elapsed_seconds"],
            }
        )
    assert completed == [{"id": r["id"], "terminal": r["terminal"]} for r in rows]
    groups = []
    for model in PROFILES:
        for variant in ("control", "candidate"):
            selected = [r for r in rows if r["model"] == model and r["variant"] == variant]
            groups.append(
                {
                    "model": model,
                    "variant": variant,
                    "calls": len(selected),
                    **{
                        k: sum(bool(r[k]) for r in selected)
                        for k in (
                            "valid_final",
                            "factual_answer_correct",
                            "raw_status_correct",
                            "application_status_correct",
                            "safe_task_on_saved_state",
                            "non_null_answer_mismatch",
                        )
                    },
                }
            )
    pairs = []
    for hid in inputs:
        selected = {r["variant"]: r for r in rows if r["history_id"] == hid}
        if set(selected) == {"control", "candidate"}:
            a, b = selected["control"], selected["candidate"]
            pairs.append(
                {
                    "history_id": hid,
                    "model": a["model"],
                    "control_correct": a["factual_answer_correct"],
                    "candidate_correct": b["factual_answer_correct"],
                    "control_id": a["id"],
                    "candidate_id": b["id"],
                }
            )
    budget = read(root / "budget.json")
    assert retained <= budget["phase_cap_microinr"]
    assert retained + budget["prior_retained_microinr"] <= budget["campaign_cap_microinr"]
    deltas = {
        m: sum(p["candidate_correct"] - p["control_correct"] for p in pairs if p["model"] == m)
        for m in PROFILES
    }
    complete = len(rows) == 288 and len(pairs) == 144 and not unknown
    return {
        "verified": True,
        "calls": len(rows),
        "rows": rows,
        "groups": groups,
        "pairs": pairs,
        "all_planned_complete_and_usage_known": bool(complete),
        "deltas": deltas,
        "mechanical_screen_pass": bool(
            complete
            and any(v > 0 for v in deltas.values())
            and all(v >= 0 for v in deltas.values())
        ),
        "semantic_regression_review": "REQUIRED_SEPARATELY_BEFORE_ANY_SCREEN_PASS",
        "known_inr": known / 1e6,
        "retained_inr": retained / 1e6,
        "unknown_calls": unknown,
        "research_known_inr": (known + budget["prior_known_microinr"]) / 1e6,
        "research_retained_inr": (retained + budget["prior_retained_microinr"]) / 1e6,
        "research_calls": budget["prior_calls"] + len(rows),
        "new_action_trials": 0,
        "limitation": "Development histories; no fresh end-to-end or holdout claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        if not args.output:
            parser.error("audit requires a new --output path")
        save(args.output, audit(args.root))
    else:
        (prepare if args.command == "prepare" else run)(args.root)
