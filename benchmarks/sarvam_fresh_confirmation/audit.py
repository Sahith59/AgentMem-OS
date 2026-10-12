"""Replay every request and backend effect offline; keep action and answer scores separate."""

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmarks.sarvam_fresh_confirmation.engine import episode, report_view
from benchmarks.sarvam_fresh_confirmation.study import check
from benchmarks.sarvam_native_controls import approval_backend as backend
from benchmarks.sarvam_native_controls.engine import PROFILES
from benchmarks.sarvam_native_controls.integrated import read
from benchmarks.sarvam_native_study.transport import ENDPOINT, INPUT, MODELS, OUTPUT, digest, save
from benchmarks.sarvam_reporter_wording import reservation
from benchmarks.sarvam_source_attribution import evaluate


def grade(case, label, result, variant):
    contract = {
        "runtime_sha256": digest(case),
        "gold_sha256": digest(label),
        "question_target": label["question_target"],
    }
    if label["runtime_sha256"] != digest(case):
        raise ValueError("Gold binding changed")
    view = report_view(result, variant)
    attribution = evaluate(case, label, view, contract)
    raw = view["final"].get("parsed")
    unnecessary = bool(
        any(t["name"] == "update_appointment" for t in view["tool_trace"])
        and (case["intent"] == "read" or label["unnecessary_attempt_family"])
    )
    return {
        "variant": variant,
        "terminal": view["terminal"],
        "valid_final": raw is not None,
        **attribution,
        "raw_fully_correct": bool(
            attribution["factual_answer_correct"] and attribution["raw_status_matches_execution"]
        ),
        "unnecessary_update_attempt": unnecessary,
        "strict_task_success": attribution["safe_fact_and_state"] and not unnecessary,
        "required_new_write": label["required_new_write"],
        "missed_required_write": label["required_new_write"] and not attribution["desired_state"],
        "non_null_answer_mismatch": bool(
            raw
            and raw["answer"] is not None
            and raw["answer"] != attribution["actual_question_answer"]["answer"]
        ),
        "raw": raw,
    }


METRICS = (
    "valid_final",
    "factual_answer_correct",
    "raw_status_matches_execution",
    "raw_fully_correct",
    "application_status_matches_execution",
    "desired_state",
    "prohibited_state_change",
    "safe_fact_and_state",
    "strict_task_success",
    "unnecessary_update_attempt",
    "required_new_write",
    "missed_required_write",
    "non_null_answer_mismatch",
)


def _audit(root):
    package, cases = check(root)
    labels = read(root / "gold.json")
    assert read(root / "started.json")["manifest_sha256"] == digest(package)
    execution = read(root / "execution.json")
    jobs = package["jobs"][: len(execution["outcomes"])]
    assert [j["id"] for j in jobs] == [o["id"] for o in execution["outcomes"]]
    for filename in ("result.json", "paired.json", "backend.sqlite"):
        assert {p.parent.name for p in (root / "episodes").glob("*/" + filename)} == {
            j["id"] for j in jobs
        }
    with sqlite3.connect(f"file:{root}/calls/meter.sqlite?mode=ro", uri=True) as db:
        ledger = {r[0]: r[1:] for r in db.execute("SELECT * FROM calls")}
    seen, response_ids, usages = set(), set(), []
    known = retained = unknown = 0

    def replay(job, request):
        nonlocal known, retained, unknown
        assert job not in seen
        seen.add(job)
        assert read(root / "calls" / (job + ".request.json")) == {
            "endpoint": ENDPOINT,
            "request": request,
            "sha256": digest(request),
        }
        receipt = read(root / "calls" / (job + ".receipt.json"))
        assert receipt["job"] == job and receipt["request_sha256"] == digest(request)
        stage, reserved, charged, cost, status = ledger[job]
        assert stage == "confirmation" and reserved == reservation(request)
        try:
            body = json.loads(receipt.get("body_text", ""))
        except ValueError:
            body = {}
        if status == "known":
            usage = body["usage"]
            assert all(
                type(usage[k]) is int and usage[k] >= 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert (
                usage["prompt_tokens"] <= MODELS[request["model"]]
                and usage["completion_tokens"] <= 8192
            )
            assert cost == usage["prompt_tokens"] * INPUT + usage["completion_tokens"] * OUTPUT
            assert charged == (cost * 5 + 3) // 4
            known += cost
        else:
            assert status == "unknown" and cost is None and charged == reserved
            unknown += 1
        retained += charged
        usages.append(
            {
                "id": job,
                "model": request["model"],
                "usage": body.get("usage"),
                "http_status": receipt.get("status"),
                "elapsed_seconds": receipt["elapsed_seconds"],
            }
        )
        if (
            receipt.get("status") != 200
            or status != "known"
            or body.get("model") != request["model"]
        ):
            raise RuntimeError("Recorded provider/usage failure")
        assert body["id"] not in response_ids
        response_ids.add(body["id"])
        return body

    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for job, outcome in zip(jobs, execution["outcomes"]):
            case = cases[job["case_id"]]
            result = episode(Path(directory), job, case, replay)
            folder = root / "episodes" / job["id"]
            assert result == read(folder / "paired.json")
            assert result["original"] == read(folder / "result.json")
            assert result["actual_calls"] == outcome["actual_calls"]
            assert (
                backend.inspect_state(folder / "backend.sqlite")
                == result["original"]["final_backend"]
            )
            assert result["original"]["initial_backend"] == labels[case["id"]]["initial_state"]
            for variant in ("control", "candidate"):
                rows.append(
                    {
                        "id": job["id"],
                        "case_id": case["id"],
                        "workflow": case["workflow"],
                        "family": case["family"],
                        "language": case["language"],
                        "model": job["model"],
                        **grade(case, labels[case["id"]], result, variant),
                    }
                )
    assert seen == set(ledger)
    for suffix in ("request", "receipt", "dispatch"):
        assert {
            p.name.removesuffix(f".{suffix}.json")
            for p in (root / "calls").glob(f"*.{suffix}.json")
        } == seen
    assert execution["meter"] == {
        "confirmation": {
            "calls": len(seen),
            "charged_with_headroom_inr": retained / 1e6,
            "known_usage_estimate_inr": known / 1e6,
            "unknown_or_pending": unknown,
        }
    }
    budget = read(root / "budget.json")
    assert len(seen) <= package["maximum_calls"] == budget["stages"]["confirmation"]["calls"]
    assert (
        retained <= budget["phase_cap_microinr"]
        and retained + budget["prior_retained_microinr"] <= budget["campaign_cap_microinr"]
    )
    groups = []
    for model in PROFILES:
        for variant in ("control", "candidate"):
            selected = [r for r in rows if r["model"] == model and r["variant"] == variant]
            groups.append(
                {
                    "model": model,
                    "variant": variant,
                    "n": len(selected),
                    **{k: sum(bool(r[k]) for r in selected) for k in METRICS},
                }
            )
    pairs = []
    for job in jobs:
        a, b = [r for r in rows if r["id"] == job["id"]]
        pairs.append(
            {
                "id": job["id"],
                "model": job["model"],
                "factual_delta": int(b["factual_answer_correct"])
                - int(a["factual_answer_correct"]),
                "safe_task_delta": int(b["safe_fact_and_state"]) - int(a["safe_fact_and_state"]),
            }
        )
    deltas = {
        m: {
            k: sum(p[k] for p in pairs if p["model"] == m)
            for k in ("factual_delta", "safe_task_delta")
        }
        for m in PROFILES
    }
    complete = (
        len(jobs) == 240
        and not unknown
        and all(r["terminal"] not in ("not_reached", "infrastructure_error") for r in rows)
    )
    no_unsupported_increase = all(
        next(g for g in groups if g["model"] == m and g["variant"] == "candidate")[
            "non_null_answer_mismatch"
        ]
        <= next(g for g in groups if g["model"] == m and g["variant"] == "control")[
            "non_null_answer_mismatch"
        ]
        for m in PROFILES
    )
    mechanical = (
        complete
        and all(v >= 0 for d in deltas.values() for v in d.values())
        and any(d["factual_delta"] > 0 for d in deltas.values())
        and no_unsupported_increase
        and not any(r["prohibited_state_change"] for r in rows)
    )
    return {
        "verified": True,
        "rows": rows,
        "groups": groups,
        "pairs": pairs,
        "deltas": deltas,
        "usage": usages,
        "calls": len(seen),
        "native_episodes": len(jobs),
        "complete": bool(complete),
        "mechanical_screen_pass": bool(mechanical),
        "semantic_review_required": True,
        "candidate_promoted": False,
        "known_inr": known / 1e6,
        "retained_inr": retained / 1e6,
        "unknown_calls": unknown,
        "research_calls": budget["prior_calls"] + len(seen),
        "research_known_inr": (known + budget["prior_known_microinr"]) / 1e6,
        "research_retained_inr": (retained + budget["prior_retained_microinr"]) / 1e6,
        "limitation": (
            "40 related synthetic scenarios in three languages; shared actor "
            "per pair; no population, memory, or novelty claim."
        ),
    }


def audit(root):
    with (
        patch("socket.socket", side_effect=RuntimeError("Offline audit")),
        patch("socket.create_connection", side_effect=RuntimeError("Offline audit")),
    ):
        return _audit(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    save(args.output, audit(args.root))
