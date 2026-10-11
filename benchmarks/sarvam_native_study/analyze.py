"""Offline aggregation; preserve failed attempts and keep diagnostic settings separate."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .score import score
from .transport import Meter, digest, save

METRICS = (
    "valid_output",
    "answer_correct",
    "backend_correct",
    "forbidden_mutation_attempt",
    "required_tool_receipt",
    "task_success",
)


def combine(expected_ids, *parts):
    """A continuation fills unattempted IDs, never replaces an earlier failure."""
    rows = {}
    for part in parts:
        for row in part:
            if not row["attempted"]:
                continue
            if row["id"] in rows:
                raise ValueError("Repeated attempted episode: " + row["id"])
            rows[row["id"]] = row
    if set(rows) != set(expected_ids):
        raise ValueError("Missing or unexpected attempted episodes")
    return [rows[j] for j in expected_ids]


def groups(rows, dimensions):
    output = defaultdict(Counter)
    for row in rows:
        key = "|".join(str(row.get(d)) for d in dimensions)
        output[key]["attempted"] += 1
        for field in METRICS:
            output[key][field] += bool(row[field])
        output[key]["terminal=" + row["terminal"]] += 1
    return dict(output)


def pairs(rows, field, left, right):
    """Within-model paired differences; absent counterparts are never successes."""
    matched = defaultdict(dict)
    for row in rows:
        other = row["arm"] if field == "reasoning" else row.get("reasoning")
        key = (row["model"], other, row["case_id"])
        value = row[field]
        if value in matched[key]:
            raise ValueError("Duplicate pair cell")
        matched[key][value] = row
    output = defaultdict(Counter)
    for (model, other, _), cells in matched.items():
        if set(cells) != {left, right}:
            raise ValueError("Incomplete paired comparison")
        a, b = cells[left]["task_success"], cells[right]["task_success"]
        out = output[model + "|" + str(other)]
        out["pairs"] += 1
        out["left_success"] += bool(a)
        out["right_success"] += bool(b)
        out["right_gain"] += not a and b
        out["right_loss"] += a and not b
        out["both_success"] += a and b
        out["both_fail"] += not a and not b
    return {"left": left, "right": right, "groups": dict(output)}


def decomposition(rows):
    """Mechanical, exclusive categories, not a semantic failure attribution."""
    output = defaultdict(Counter)
    for row in rows:
        group = row["model"] + "|" + row["arm"] + "|" + str(row.get("reasoning"))
        if row["terminal"] != "answer":
            category = "terminal:" + row["terminal"]
        elif not row["valid_output"]:
            category = "invalid_json_contract"
        elif row["forbidden_mutation_attempt"]:
            category = "forbidden_mutation_attempt"
        elif not row["backend_correct"]:
            category = "incorrect_backend_state"
        elif not row["required_tool_receipt"]:
            category = "missing_required_tool_receipt"
        elif not row["answer_correct"]:
            fields = sorted(k for k, v in row["expected"].items() if row["parsed"][k] != v)
            category = "exact_field_mismatch:" + ",".join(fields)
        else:
            category = "success"
        output[group][category] += 1
    return dict(output)


def summarize(rows, diagnostic=False):
    dims = ["model", "arm"] + (["reasoning"] if diagnostic else [])
    return {
        "episodes": len(rows),
        "base_scenarios": len({r["base_id"] for r in rows}),
        "renderings": len({r["case_id"] for r in rows}),
        "overall": groups(rows, dims),
        "by_family": groups(rows, dims + ["family"]),
        "by_language": groups(rows, dims + ["language"]),
        "mechanical_failure_categories": decomposition(rows),
        "paired": pairs(rows, "reasoning", None, "low")
        if diagnostic
        else pairs(rows, "arm", "full_history", "sqlite"),
        "rows": rows,
    }


def analyze(root):
    root = Path(root)
    phases = ("discovery", "discovery-continuation-001", "attribution", "confirmation")
    raw = {}
    for phase in phases:
        execution = json.loads((root / phase / "execution.json").read_text())
        scored = score(root, root / "bank", phase)
        done = {r["id"] for r in scored["rows"] if r["attempted"]}
        if done != {r["id"] for r in execution["outcomes"]}:
            raise ValueError("Execution/result inventory mismatch")
        raw[phase] = scored["rows"]
    original = json.loads((root / "discovery/manifest.json").read_text())
    discovery = combine(
        [j["id"] for j in original["jobs"]],
        raw["discovery"],
        raw["discovery-continuation-001"],
    )
    for phase in ("confirmation", "attribution"):
        if not all(r["attempted"] for r in raw[phase]):
            raise ValueError("Unattempted jobs in " + phase)
    return {
        "bank_sha256": digest(json.loads((root / "bank/runtime.json").read_text())),
        "discovery": summarize(discovery),
        "confirmation": summarize(raw["confirmation"]),
        "attribution": summarize(raw["attribution"], diagnostic=True),
        "meter": Meter(root / "calls").summary(),
        "limits": [
            "Strict synthetic task success, not general model accuracy",
            "Translations and parameter variants are correlated",
            "No independent native language validation",
            "Diagnostic is failure-enriched and changes reasoning, not architecture",
            "SQLite costs extra turns; arms do not have equal actual compute",
            "JSON mode applies only when no tools are available",
            "No speech, Romanized, all-language or full context-window test",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign")
    args = parser.parse_args()
    result = analyze(args.campaign)
    save(Path(args.campaign) / "analysis.json", result)
    stages = ("discovery", "confirmation", "attribution")
    print(json.dumps({s: result[s]["overall"] for s in stages}))
