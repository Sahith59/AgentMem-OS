#!/usr/bin/env python3
"""Audit a frozen English run by evidence, answering, abstention, and judge stage.

The diagnostic may read benchmark annotations and gold answers because it runs
after measurement. Those fields are never inputs to retrieval or generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def date_line(case: dict) -> str:
    return f"\nToday's date is {case['date']}." if case.get("date") else ""


def generation_request(package: dict, case: dict) -> dict:
    content = package["prompt"].format(
        context=case["context"],
        question=case["question"],
        today_line=date_line(case),
    )
    return dict(
        package["settings"]["generate"],
        messages=[{"role": "user", "content": content}],
    )


def judge_request(package: dict, case: dict, answer: str) -> dict:
    content = case["judge_template"].replace("{response}", answer)
    return dict(
        package["settings"]["judge"],
        messages=[{"role": "user", "content": content}],
    )


def operation(question: str, abstention: bool = False) -> str:
    if abstention:
        return "evidence_sufficiency"
    text = question.lower()
    if re.search(r"\b(how many|how much|how often|total|combined|altogether)\b", text):
        return "aggregation_or_amount"
    if re.search(
        r"\b(first|last|before|after|ago|current|currently|recent|latest|"
        r"order|when|how long|days?|weeks?|months?|years?)\b",
        text,
    ):
        return "temporal_or_update"
    if re.search(r"\b(should|suggest|recommend|advice|ideas?|what do you think)\b", text):
        return "preference_or_advice"
    return "direct_recall_or_synthesis"


def failure_stage(case: dict, annotated_turns: list[dict]) -> tuple[str, int]:
    present = sum(turn["text"] in case["context"] for turn in annotated_turns)
    if case["abst"]:
        return "evidence_sufficiency_or_judge", present
    if annotated_turns and present == 0:
        return "retrieval_zero_exact_annotated_turns", present
    if present < len(annotated_turns):
        return "retrieval_partial_exact_annotated_turns", present
    return "answer_selection_reasoning_or_judge", present


def fact_lineage(
    context: str,
    turn: dict,
    connection: sqlite3.Connection,
    cache: dict,
) -> dict:
    """Trace one missing annotated source turn through extraction and delivery."""
    # The frozen derived store is keyed by the content-addressed source_key;
    # annotation session_id is the upstream pre-migration identifier.
    session_id = turn["source_key"]
    if session_id not in cache:
        stored_turns = connection.execute(
            "SELECT id, content FROM turns WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
        stored_facts = []
        for fact_id, text, source_ids in connection.execute(
            "SELECT id, fact_text, source_turn_ids FROM semantic_facts "
            "WHERE source_session_id=? ORDER BY id",
            (session_id,),
        ):
            stored_facts.append(
                (fact_id, text, set(json.loads(source_ids or "[]")))
            )
        cache[session_id] = (stored_turns, stored_facts)

    stored_turns, stored_facts = cache[session_id]
    source_text = turn["text"]
    turn_ids = {
        turn_id
        for turn_id, content in stored_turns
        if content == source_text or content.endswith(source_text)
    }
    linked = [
        (fact_id, text)
        for fact_id, text, source_ids in stored_facts
        if turn_ids & source_ids
    ]
    if not turn_ids:
        status = "source_turn_not_in_corpus"
    elif not linked:
        status = "no_fact_linked_to_source_turn"
    elif any(text in context for _, text in linked):
        status = "linked_fact_delivered"
    else:
        status = "linked_fact_not_delivered"
    return {
        "source_key": turn["source_key"],
        "upstream_session_id": turn["session_id"],
        "corpus_session_id": session_id,
        "turn_index": turn["turn_index"],
        "status": status,
        "corpus_turn_ids": sorted(turn_ids),
        "linked_fact_ids": [fact_id for fact_id, _ in linked],
        "linked_fact_texts": [text for _, text in linked],
    }


def diagnostic_stage(raw_stage: str, lineage: list[dict]) -> str:
    """Assign the earliest evidenced pipeline stage without hiding co-causes."""
    if raw_stage == "evidence_sufficiency_or_judge":
        return raw_stage
    if raw_stage == "answer_selection_reasoning_or_judge":
        return raw_stage
    statuses = {item["status"] for item in lineage}
    if "source_turn_not_in_corpus" in statuses:
        return "source_corpus_gap"
    if "no_fact_linked_to_source_turn" in statuses:
        return "extraction_lineage_gap"
    if "linked_fact_not_delivered" in statuses:
        return "fact_retrieval_or_ranking_gap"
    return "fact_lineage_delivered_raw_turn_missing"


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("repeat1", type=Path)
    parser.add_argument("repeat2", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retrieval-source", type=Path, action="append", default=[])
    parser.add_argument(
        "--facts-db",
        type=Path,
        required=True,
        help="Exact immutable fact/turn store used to build the frozen packets",
    )
    args = parser.parse_args()

    package = load(args.package)
    checkpoint = load(args.checkpoint)
    annotations = {row["question_id"]: row for row in load(args.annotations)}
    repeat1 = load(args.repeat1)
    repeat2 = load(args.repeat2)
    cases = {case["id"]: case for case in package["cases"]}
    if len(cases) != 500 or set(cases) != set(annotations):
        raise ValueError("Expected the same 500 IDs in package and annotations")

    request_checks = []
    rows = []
    raw_stage_counts = Counter()
    stage_counts = Counter()
    lineage_counts = Counter()
    operation_counts = Counter()
    type_stage_counts: dict[str, Counter] = defaultdict(Counter)
    facts = sqlite3.connect(f"file:{args.facts_db}?mode=ro", uri=True)
    lineage_cache = {}
    for qid, case in cases.items():
        generated = checkpoint["jobs"][f"{qid}/generate"]
        judged = checkpoint["jobs"][f"{qid}/judge"]
        gen_request = generation_request(package, case)
        grade_request = judge_request(package, case, generated["answer"])
        request_checks.append(
            generated["request_sha256"] == digest(canonical(gen_request))
            and judged["request_sha256"] == digest(canonical(grade_request))
        )
        if judged["correct"]:
            continue
        annotated = annotations[qid]["annotated_turns"]
        stage, present = failure_stage(case, annotated)
        op = operation(case["question"], case["abst"])
        raw_stage_counts[stage] += 1
        operation_counts[op] += 1
        missing_turns = [
            turn for turn in annotated if turn["text"] not in case["context"]
        ]
        lineage = [
            fact_lineage(case["context"], turn, facts, lineage_cache)
            for turn in missing_turns
        ]
        diagnosed = diagnostic_stage(stage, lineage)
        stage_counts[diagnosed] += 1
        type_stage_counts[case["type"]][diagnosed] += 1
        lineage_counts.update(item["status"] for item in lineage)
        rows.append(
            {
                "question_id": qid,
                "question_type": case["type"],
                "abstention": case["abst"],
                "operation": op,
                "raw_failure_stage": stage,
                "failure_stage": diagnosed,
                "question": case["question"],
                "reference": case["gold"],
                "answer": generated["answer"],
                "annotated_turns": len(annotated),
                "exact_annotated_turns_present": present,
                "missing_annotated_turns": lineage,
                "precision_added_turns": case["precision_added_turns"],
                "precision_gained_annotated_turns": case[
                    "precision_gained_annotated_turns"
                ],
                "context_changed": case["context_sha256"]
                != case["baseline_context_sha256"],
                "prior_repeat1_correct": repeat1["jobs"][f"{qid}/judge"]["correct"],
                "prior_repeat2_correct": repeat2["jobs"][f"{qid}/judge"]["correct"],
            }
        )

    facts.close()
    source_text = "\n".join(path.read_text() for path in args.retrieval_source)
    hardcoded_ids = sorted(qid for qid in cases if qid in source_text)
    correct = sum(
        checkpoint["jobs"][f"{qid}/judge"]["correct"] for qid in cases
    )
    result = {
        "status": "PASS_COMPLETE_FAILURE_STAGE_AUDIT",
        "score": {"correct": correct, "total": 500, "accuracy": correct / 500},
        "target_gap": {"target_correct": 450, "additional_correct_needed": 450 - correct},
        "failures": len(rows),
        "stage_counts": dict(sorted(stage_counts.items())),
        "raw_exact_turn_stage_counts": dict(sorted(raw_stage_counts.items())),
        "missing_turn_lineage_counts": dict(sorted(lineage_counts.items())),
        "facts_db": {
            "path": str(args.facts_db.resolve()),
            "sha256": hashlib.sha256(args.facts_db.read_bytes()).hexdigest(),
        },
        "operation_counts": dict(sorted(operation_counts.items())),
        "type_stage_counts": {
            kind: dict(sorted(counts.items()))
            for kind, counts in sorted(type_stage_counts.items())
        },
        "integrity": {
            "all_1000_saved_requests_reconstruct_exactly": all(request_checks),
            "generation_request_fields": [
                "frozen answer prompt",
                "frozen context",
                "question",
                "question date",
                "frozen model settings",
            ],
            "gold_used_only_after_generation_for_judging": True,
            "retrieval_source_question_ids_found": hardcoded_ids,
            "retrieval_source_has_no_question_id_hardcoding": not hardcoded_ids,
        },
        "interpretation_limits": [
            "Exact annotated-turn absence proves an evidence-delivery gap, not whether extraction or retrieval alone caused it.",
            "Exact annotated-turn presence does not prove that every required operand is salient or unambiguous.",
            "The answer-selection stage includes possible official-judge/reference defects until separately adjudicated.",
            "A fact linked to an annotated turn may omit the answer-bearing detail; lineage is necessary but not semantic-equivalence proof.",
            "All 500 outcomes are development-exposed; fixes require fresh or explicitly development-labeled validation.",
        ],
        "rows": rows,
    }
    if not all(request_checks):
        raise ValueError("Saved request reconstruction failed")
    if hardcoded_ids:
        raise ValueError(f"Retrieval source contains benchmark IDs: {hardcoded_ids}")
    if len(rows) != 500 - correct:
        raise ValueError("Failure count does not reconcile")

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# English full500 failure-stage audit",
        "",
        f"Score audited: **{correct}/500**. Failures: **{len(rows)}**. "
        f"Additional correct answers needed for 90%: **{450 - correct}**.",
        "",
        "## Diagnosed primary stage",
        "",
    ]
    for stage, count in sorted(stage_counts.items()):
        lines.append(f"- `{stage}`: {count}")
    lines.extend(
        [
            "",
            "",
            "## Raw exact-turn delivery split",
            "",
        ]
    )
    for stage, count in sorted(raw_stage_counts.items()):
        lines.append(f"- `{stage}`: {count}")
    lines.extend(
        [
            "",
            "All 1,000 generation/judge request hashes reconstruct exactly. Generation uses only the frozen prompt, context, question, question date and model settings; the reference enters only the later judge request. The audited retrieval source contains no benchmark question IDs.",
            "",
            "The complete77-row record, including every question, answer, reference, evidence count and missing source pointer, is in `audit.json`.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "score", "target_gap", "failures", "stage_counts", "operation_counts", "integrity")}, indent=2))


if __name__ == "__main__":
    main()
