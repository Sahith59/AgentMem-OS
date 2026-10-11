"""Same exact-field evaluator; source-grounded fact review stays separate."""

from collections import defaultdict
from pathlib import Path

from benchmarks.sarvam_discovery.contract import runtime_cases as all_cases
from benchmarks.sarvam_discovery.score import (
    gold_labels as original_gold_labels,
)
from benchmarks.sarvam_discovery.score import (
    grade,
    metering,
    summarize,
)

from .contract import eligible, loads, runtime_cases
from .runner import validate_state


def gold_labels(cases):
    labels = original_gold_labels(all_cases())
    if [c["id"] for c in cases] != [c["id"] for c in runtime_cases()]:
        raise ValueError("Gold selection differs from frozen subset")
    return {c["id"]: labels[c["id"]] for c in cases}


def extraction_review(package, state, indexed):
    rows = []
    for spec in package["jobs"]:
        if spec["stage"] != "extract":
            continue
        job = state["jobs"].get(spec["id"], {})
        rows.append(
            {
                "job_id": spec["id"],
                "case_id": spec["case_id"],
                "repeat": spec["repeat"],
                "status": job.get("status", "not_attempted"),
                "original_records": eligible(indexed[spec["case_id"]]),
                "facts": job.get("parsed"),
                "semantic_review": {
                    name: "UNREVIEWED"
                    for name in (
                        "supported_meaning",
                        "metadata_duplication",
                        "dates_preserved",
                        "conditions_preserved",
                        "requested_vs_confirmed",
                        "identity_scope",
                    )
                },
            }
        )
    return rows


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
        "schema": "sarvam-comparison-score-v2",
        "extraction_review": extraction_review(package, state, indexed),
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
