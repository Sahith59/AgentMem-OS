"""Independently verify precision-screen membership and source bindings."""

import argparse
import hashlib
import json
from pathlib import Path


SEED = "source-precision-controls-v2:"


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify(path):
    selection = json.loads(Path(path).read_text())
    if selection["status"] != "FROZEN_BEFORE_PRECISION_CANDIDATE_QA_OUTPUTS":
        raise ValueError("Selection was not frozen prospectively")
    if selection["paid_authorization"] is not False:
        raise ValueError("Selection cannot authorize spending")
    if selection["control_seed"] != SEED:
        raise ValueError("Control seed changed")
    inputs = selection["inputs"]
    for source in inputs.values():
        if file_sha(source["path"]) != source["sha256"]:
            raise ValueError("Selection input changed")
    package = json.loads(Path(inputs["baseline_package"]["path"]).read_text())
    repeats = [
        json.loads(Path(inputs[name]["path"]).read_text())["jobs"]
        for name in ("repeat1", "repeat2")
    ]
    previous = json.loads(Path(inputs["previous_selection"]["path"]).read_text())
    candidates = {
        row["question_id"]: row
        for row in json.loads(Path(inputs["candidate_results"]["path"]).read_text())
    }
    by_id = {case["id"]: case for case in package["cases"]}
    stable_miss = {
        qid for qid in by_id
        if all(not run[qid + "/judge"]["correct"] for run in repeats)
    }
    disagreement = {
        qid for qid in by_id
        if repeats[0][qid + "/judge"]["correct"]
        != repeats[1][qid + "/judge"]["correct"]
    }
    stable_pass = set(by_id) - stable_miss - disagreement
    mandatory = {qid for qid in stable_pass if by_id[qid]["abst"]}
    prior_nonabstention = {
        row["question_id"] for row in previous["cases"]
        if row["cohort"] == "stable_pass_control" and not row["abstention"]
    }
    pools = {}
    for qid in stable_pass - mandatory - prior_nonabstention:
        pools.setdefault(by_id[qid]["type"], []).append(qid)
    for qids in pools.values():
        qids.sort(key=lambda qid: hashlib.sha256((SEED + qid).encode()).hexdigest())
    fresh = []
    kinds = sorted(pools)
    while len(fresh) < 40:
        for kind in kinds:
            if pools[kind] and len(fresh) < 40:
                fresh.append(pools[kind].pop(0))
    controls = mandatory | set(fresh)
    cohorts = {qid: "stable_miss" for qid in stable_miss}
    cohorts.update({qid: "disagreement" for qid in disagreement})
    cohorts.update({qid: "stable_pass_control" for qid in controls})
    expected = []
    for case in package["cases"]:
        if case["id"] not in cohorts:
            continue
        expected.append({
            "question_id": case["id"],
            "cohort": cohorts[case["id"]],
            "baseline_sha256": case["context_sha256"],
            "candidate_sha256": candidates[case["id"]]["candidate_sha256"],
            "question_sha256": case["question_sha256"],
            "type": case["type"],
            "abstention": case["abst"],
        })
    if selection["cases"] != expected:
        raise ValueError("Selection membership or metadata mismatch")
    if selection["counts"] != {
        "stable_miss": 72,
        "disagreement": 20,
        "stable_pass_control": 58,
        "reused_stable_pass_abstention_controls": 18,
        "fresh_nonabstention_controls": 40,
        "retired_previous_nonabstention_controls": 40,
    }:
        raise ValueError("Selection counts changed")
    if len(expected) != 150 or sum(row["abstention"] for row in expected) != 30:
        raise ValueError("Population or abstention count changed")
    return {
        "status": "PASS",
        "selection_sha256": file_sha(path),
        "cases": 150,
        "stable_misses": 72,
        "disagreements": 20,
        "stable_controls": 58,
        "fresh_nonabstention_controls": 40,
        "reused_abstention_controls": 18,
        "all_abstentions": 30,
        "prior_nonabstention_control_overlap": 0,
        "qa_accuracy": "NOT_MEASURED",
        "paid_authorization": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.selection)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
