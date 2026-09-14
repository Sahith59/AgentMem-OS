"""Freeze a second 150-row screen with fresh non-abstention controls."""

import argparse
import hashlib
import json
from pathlib import Path

from .verification import digest, file_sha


SEED = "source-precision-controls-v2:"


def select(package, first, second, previous):
    jobs = [first["jobs"], second["jobs"]]
    stable_miss = {
        case["id"] for case in package["cases"]
        if all(not run[case["id"] + "/judge"]["correct"] for run in jobs)
    }
    disagreement = {
        case["id"] for case in package["cases"]
        if jobs[0][case["id"] + "/judge"]["correct"]
        != jobs[1][case["id"] + "/judge"]["correct"]
    }
    stable_pass = {
        case["id"] for case in package["cases"]
        if all(run[case["id"] + "/judge"]["correct"] for run in jobs)
    }
    if (len(stable_miss), len(disagreement), len(stable_pass)) != (72, 20, 408):
        raise ValueError("Historical cohorts changed")
    by_id = {case["id"]: case for case in package["cases"]}
    previous_nonabstention = {
        row["question_id"] for row in previous["cases"]
        if row["cohort"] == "stable_pass_control" and not row["abstention"]
    }
    mandatory = {
        qid for qid in stable_pass if by_id[qid]["abst"]
    }
    if len(mandatory) != 18:
        raise ValueError("Stable-pass abstention population changed")
    pools = {}
    for qid in stable_pass - mandatory - previous_nonabstention:
        pools.setdefault(by_id[qid]["type"], []).append(qid)
    for qids in pools.values():
        qids.sort(key=lambda qid: hashlib.sha256((SEED + qid).encode()).hexdigest())
    fresh = []
    types = sorted(pools)
    while len(fresh) < 40:
        progressed = False
        for kind in types:
            if pools[kind] and len(fresh) < 40:
                fresh.append(pools[kind].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("Insufficient fresh controls")
    controls = mandatory | set(fresh)
    if controls & previous_nonabstention or len(controls) != 58:
        raise ValueError("Fresh control policy violated")
    cohorts = {qid: "stable_miss" for qid in stable_miss}
    cohorts.update({qid: "disagreement" for qid in disagreement})
    cohorts.update({qid: "stable_pass_control" for qid in controls})
    rows = []
    for case in package["cases"]:
        if case["id"] not in cohorts:
            continue
        rows.append({
            "question_id": case["id"],
            "cohort": cohorts[case["id"]],
            "baseline_sha256": case["context_sha256"],
            "candidate_sha256": None,
            "question_sha256": case["question_sha256"],
            "type": case["type"],
            "abstention": case["abst"],
        })
    return rows, {
        "stable_miss": len(stable_miss),
        "disagreement": len(disagreement),
        "stable_pass_control": len(controls),
        "reused_stable_pass_abstention_controls": len(mandatory),
        "fresh_nonabstention_controls": len(fresh),
        "retired_previous_nonabstention_controls": len(previous_nonabstention),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--repeat1", type=Path, required=True)
    parser.add_argument("--repeat2", type=Path, required=True)
    parser.add_argument("--previous-selection", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "baseline_package": args.package,
        "repeat1": args.repeat1,
        "repeat2": args.repeat2,
        "previous_selection": args.previous_selection,
        "candidate_results": args.candidate_results,
        "selector": Path(__file__),
    }
    package = json.loads(args.package.read_text())
    rows, counts = select(
        package,
        json.loads(args.repeat1.read_text()),
        json.loads(args.repeat2.read_text()),
        json.loads(args.previous_selection.read_text()),
    )
    candidates = {
        row["question_id"]: row
        for row in json.loads(args.candidate_results.read_text())
    }
    for row in rows:
        row["candidate_sha256"] = candidates[row["question_id"]]["candidate_sha256"]
    result = {
        "status": "FROZEN_BEFORE_PRECISION_CANDIDATE_QA_OUTPUTS",
        "paid_authorization": False,
        "purpose": (
            "Second development screen for the precision-limited supplement; "
            "all hard/unstable rows, every abstention, and fresh non-abstention controls."
        ),
        "selection": (
            "All72 stable misses and20 disagreements; all18 stable-pass abstentions "
            "reused to retain the complete30-abstention cohort;40 deterministic "
            "type-round-robin controls selected after excluding every previously "
            "tested non-abstention control. No outcome-specific ID exclusion."
        ),
        "control_seed": SEED,
        "development_exposure": (
            "The precision algorithm was developed after the broad screen. Controls "
            "are untested with this candidate but the full dataset is development-exposed."
        ),
        "unchanged_models_prompts_and_judges": True,
        "gates": {
            "complete_pairs": 150,
            "candidate_minus_baseline_correct_min": 10,
            "stable_pass_control_losses_max": 2,
            "stable_pass_candidate_correct_min": 56,
            "abstention_net_gain_min": 0,
            "unresolved_jobs_max": 0,
            "selective_regrades_or_retries": False,
        },
        "counts": counts,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": file_sha(path)}
            for name, path in paths.items()
        },
        "cases": rows,
    }
    if len(rows) != 150 or sum(row["abstention"] for row in rows) != 30:
        raise ValueError("Screen size or abstention coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selection_sha256": file_sha(args.output), **counts}, indent=2))


if __name__ == "__main__":
    main()
