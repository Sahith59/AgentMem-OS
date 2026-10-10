"""Deterministic evaluator, isolated from inference. No model judge or network."""

from collections import Counter, defaultdict
from pathlib import Path

from .contract import FIXTURES, eligible, exact_keys, loads, runtime_cases
from .runner import receipt_usage, validate_state


def gold_labels(cases):
    data = loads((FIXTURES / "gold.json").read_text())
    exact_keys(data, {"schema", "evaluator_only", "cases"})
    if data["schema"] != "sarvam-discovery-gold-v1" or data["evaluator_only"] is not True:
        raise ValueError("Unexpected gold contract")
    if set(data["cases"]) != {c["id"] for c in cases}:
        raise ValueError("Gold coverage differs from runtime cases")
    for case in cases:
        label = data["cases"][case["id"]]
        exact_keys(label, {"needs_clarification", "values", "required_evidence_alternatives"})
        if (
            type(label["needs_clarification"]) is not bool
            or set(label["values"]) != set(case["fields"])
            or set(label["required_evidence_alternatives"]) != set(case["fields"])
        ):
            raise ValueError("Gold fields differ from answer contract")
        allowed = {r["id"] for r in eligible(case)}
        for name, value in label["values"].items():
            if value is not None and not isinstance(value, str):
                raise ValueError("Invalid gold value")
            alternatives = label["required_evidence_alternatives"][name]
            if not alternatives or any(
                not isinstance(a, list) or not a or not set(a) <= allowed for a in alternatives
            ):
                raise ValueError("Gold evidence outside authorized scope")
    return data["cases"]


def grade(answer, gold):
    fields = {f["name"]: f for f in answer["fields"]}
    values = {name: fields[name]["value"] == value for name, value in gold["values"].items()}
    evidence = {
        name: any(set(required) <= set(fields[name]["source_ids"]) for required in alternatives)
        for name, alternatives in gold["required_evidence_alternatives"].items()
    }
    clarification = answer["needs_clarification"] == gold["needs_clarification"]
    return {
        "value_correct": values,
        "required_evidence_covered": evidence,
        "clarification_correct": clarification,
        "answer_correct": all(values.values()) and clarification,
        "answer_and_required_evidence": all(values.values())
        and clarification
        and all(evidence.values()),
        "citation_count": sum(len(f["source_ids"]) for f in fields.values()),
    }


def summarize(rows):
    valid = [r for r in rows if r["status"] == "complete"]
    correct = sum(r["grade"]["answer_correct"] for r in valid)
    return {
        "intended": len(rows),
        "valid_answers": len(valid),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "correct_answers": correct,
        "accuracy_among_valid": correct / len(valid) if valid else None,
        "correct_with_required_evidence": sum(
            r["grade"]["answer_and_required_evidence"] for r in valid
        ),
        "field_count": sum(len(r["grade"]["value_correct"]) for r in valid),
        "correct_fields": sum(sum(r["grade"]["value_correct"].values()) for r in valid),
        "clarification_correct": sum(r["grade"]["clarification_correct"] for r in valid),
        "citation_count": sum(r["grade"]["citation_count"] for r in valid),
    }


def metering(package, state):
    """Charge every extraction to SQLite, including repeated/failed extraction."""
    stages = {}
    for stage in ("extract", "full_history", "sqlite"):
        selected = [
            state["jobs"][s["id"]]
            for s in package["jobs"]
            if s["stage"] == stage and s["id"] in state["jobs"]
        ]
        known = [j for j in selected if "usage_ninr" in j]
        usage = [receipt_usage(j["receipt"])[0]["usage"] for j in known]
        stages[stage] = {
            "attempted_calls": len(selected),
            "calls_with_known_usage": len(known),
            "calls_with_unknown_usage": len(selected) - len(known),
            "prompt_tokens": sum(u["prompt_tokens"] for u in usage),
            "completion_tokens": sum(u["completion_tokens"] for u in usage),
            "known_usage_ninr": sum(j["usage_ninr"] for j in known),
            "reserved_ninr": sum(j["reservation_ninr"] for j in selected),
        }
    arms = {
        "full_history": dict(stages["full_history"]),
        "sqlite": {key: stages["sqlite"][key] + stages["extract"][key] for key in stages["sqlite"]},
    }
    return {
        "basis": "provider usage at frozen uncached rates; not an invoice",
        "allocation": "all extraction (including failures/repeats) charged to SQLite",
        "stages": stages,
        "arms": arms,
    }


def evaluate(package, directory):
    directory = Path(directory).resolve()
    state = loads((directory / "checkpoint.json").read_text())
    execution = validate_state(package, state, directory)
    cases = runtime_cases()
    labels = gold_labels(cases)
    indexed = {c["id"]: c for c in cases}
    rows = []
    for spec in package["jobs"]:
        if spec["stage"] == "extract":
            continue
        case = indexed[spec["case_id"]]
        job = state["jobs"].get(spec["id"], {})
        row = {
            "job_id": spec["id"],
            "case_id": case["id"],
            "family": case["family"],
            "repeat": spec["repeat"],
            "arm": spec["stage"],
            "rendering": case["rendering"],
            "status": job.get("status", "not_attempted"),
        }
        if row["status"] == "complete":
            row["grade"] = grade(job["parsed"], labels[case["id"]])
        rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["repeat"])].append(row)
    pairs = []
    for (case_id, repeat), pair in groups.items():
        arms = {r["arm"]: r for r in pair}
        comparable = all(r["status"] == "complete" for r in pair)
        delta = (
            (
                int(arms["sqlite"]["grade"]["answer_correct"])
                - int(arms["full_history"]["grade"]["answer_correct"])
            )
            if comparable
            else None
        )
        pairs.append(
            {
                "case_id": case_id,
                "repeat": repeat,
                "comparable": comparable,
                "sqlite_minus_history": delta,
            }
        )
    subgroup = {}
    for name, key in (
        ("family", lambda r: r["family"]),
        ("source_language", lambda r: r["rendering"]["source_language"]),
        ("query_language", lambda r: r["rendering"]["query_language"]),
        ("repeat", lambda r: str(r["repeat"])),
    ):
        subgroup[name] = {
            value: {
                arm: summarize([r for r in rows if key(r) == value and r["arm"] == arm])
                for arm in ("full_history", "sqlite")
            }
            for value in sorted({key(r) for r in rows})
        }
    return {
        "schema": "sarvam-discovery-score-v1",
        "status": "SIMULATED_TEST_ONLY"
        if execution["mode"] == "offline-test"
        else "EXPLORATORY_COMPLETE"
        if execution["completed"] == execution["intended_calls"]
        else "INCOMPLETE_NO_HEADLINE_SCORE",
        "package_sha256": state["binding"]["package_sha256"],
        "execution": execution,
        "metering": metering(package, state),
        "independent_human_validation": False,
        "limits": [
            "Assistant-authored and reviewed labels; exploratory only.",
            "Translations, contrasts and repeats are correlated, not independent samples.",
            "Exact canonical fields; no free-text or full explanation quality judgment.",
            "Required citation coverage does not prove every cited source entails an answer.",
            "Invalid/unattempted answers stay visible; valid-only accuracy is conditional.",
            "Same answer settings and original sources; SQLite adds extraction compute.",
            "No causal architecture, novelty, production or all-language quality claim.",
        ],
        "arms": {
            arm: summarize([r for r in rows if r["arm"] == arm])
            for arm in ("full_history", "sqlite")
        },
        "paired": {
            "intended": len(pairs),
            "comparable": sum(p["comparable"] for p in pairs),
            "sqlite_wins": sum(p["sqlite_minus_history"] == 1 for p in pairs),
            "sqlite_losses": sum(p["sqlite_minus_history"] == -1 for p in pairs),
            "ties": sum(p["sqlite_minus_history"] == 0 for p in pairs),
            "rows": pairs,
        },
        "subgroups": subgroup,
        "rows": rows,
    }
