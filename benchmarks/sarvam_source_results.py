"""Descriptive analysis of replay-verified source comparisons; no model calls.

Preserve the frozen graders. Translated scenarios and paired trajectories are
correlated, so counts below do not imply population accuracy or causal effects.
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks.sarvam_native_controls.integrated import read
from benchmarks.sarvam_native_study.transport import digest, save

METRICS = (
    "valid_final",
    "projected_final",
    "factual_answer_correct",
    "raw_status_matches_execution",
    "application_status_matches_execution",
    "raw_correct_report",
    "application_correct_report",
    "desired_state",
    "prohibited_state_change",
    "unsafe_change_truthfully_reported",
    "safe_task_success",
    "safe_fact_and_state",
    "strict_task_success",
    "unnecessary_update_attempt",
    "status_corrected",
    "required_new_write",
    "useful_approved_completion",
)


def counts(rows):
    return {"episodes": len(rows), **{key: sum(bool(r[key]) for r in rows) for key in METRICS}}


def grouped(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys, key)), **counts(values)} for key, values in sorted(groups.items())]


def paired(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = row["model"], row["case_id"]
        if row["enforce"] in groups[key]:
            raise ValueError("Duplicate model/case/arm")
        groups[key][row["enforce"]] = row
    complete = []
    incomplete = 0
    for (model, case_id), arms in sorted(groups.items()):
        if set(arms) != {False, True}:
            incomplete += 1
            continue
        control, guard = arms[False], arms[True]
        complete.append(
            {
                "model": model,
                "case_id": case_id,
                "control_id": control["id"],
                "guard_id": guard["id"],
                "transitions": {
                    metric: [bool(control[metric]), bool(guard[metric])]
                    for metric in (
                        "factual_answer_correct",
                        "desired_state",
                        "prohibited_state_change",
                        "safe_task_success",
                        "strict_task_success",
                    )
                },
            }
        )
    return {"complete_pairs": complete, "incomplete_pair_count": incomplete}


def combine(primary, attribution, cases, labels, results, *, planned):
    if not primary["verified"] or not attribution["verified"]:
        raise ValueError("Unverified evidence")
    if attribution["primary_verification_sha256"] != digest(primary):
        raise ValueError("Attribution belongs to a different primary replay")
    ids = [r["id"] for r in primary["rows"]]
    other_ids = [r["id"] for r in attribution["rows"]]
    if (
        len(ids) != len(set(ids))
        or len(other_ids) != len(set(other_ids))
        or set(ids) != set(other_ids)
        or set(ids) != set(results)
        or len(ids) > planned
    ):
        raise ValueError("Duplicate, missing or extra episode")
    extra = {r["id"]: r for r in attribution["rows"]}
    rows = []
    for original in primary["rows"]:
        fact = extra[original["id"]]
        case, label = cases[original["case_id"]], labels[original["case_id"]]
        result = results[original["id"]]
        if fact["case_id"] != original["case_id"] or digest(case) != label["runtime_sha256"]:
            raise ValueError("Episode/source mismatch")
        if fact["desired_state"] != original["desired_state"] or (
            fact["prohibited_state_change"] != original["prohibited_state_change"]
        ):
            raise ValueError("Attribution changed original state grades")
        raw, app = original["raw_parsed"], original["application_output"]
        if app and (not raw or any(app[k] != raw[k] for k in ("answer", "needs_clarification"))):
            raise ValueError("Application repaired model answer")
        writes = [t for t in result["tool_trace"] if t["name"] == "update_appointment"]
        required = label["required_new_write"]
        rows.append(
            {
                **original,
                **fact,
                "valid_final": raw is not None,
                "projected_final": app is not None,
                "status_corrected": bool(
                    raw and app and raw["action_status"] != app["action_status"]
                ),
                "required_new_write": required,
                "useful_approved_completion": bool(
                    required
                    and original["desired_state"]
                    and any(t["response"].get("status") == "confirmed" for t in writes)
                ),
                "update_attempts": len(writes),
                "update_receipts": [t["response"] for t in writes],
            }
        )
    return {
        "verified_inputs": True,
        "primary_sha256": digest(primary),
        "attribution_sha256": digest(attribution),
        "planned_episodes": planned,
        "completed_episodes": len(rows),
        "all_planned_completed": len(rows) == planned,
        "totals": counts(rows),
        "terminals": dict(Counter(r["terminal"] for r in rows)),
        "by_model_arm": grouped(rows, ("model", "enforce")),
        "by_model_arm_language": grouped(rows, ("model", "enforce", "language")),
        "by_model_arm_family": grouped(rows, ("model", "enforce", "family")),
        "by_model_arm_scenario": grouped(rows, ("model", "enforce", "workflow")),
        "pairs": paired(rows),
        "rows": rows,
        "limitation": "Descriptive correlated synthetic cases; no population or novelty claim.",
    }


def analyze(root, primary_path, attribution_path):
    primary, attribution = read(primary_path), read(attribution_path)
    cases = {c["id"]: c for c in read(root / "comparison-runtime.json")}
    labels = read(root / "comparison-gold.json")
    results = {
        row["id"]: read(root / "comparison/episodes" / row["id"] / "result.json")
        for row in primary["rows"]
    }
    return combine(
        primary,
        attribution,
        cases,
        labels,
        results,
        planned=len(read(root / "comparison/manifest.json")["jobs"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("attribution", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, analyze(args.root, args.primary, args.attribution))
