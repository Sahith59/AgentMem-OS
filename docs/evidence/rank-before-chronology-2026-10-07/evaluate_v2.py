"""Evaluator-only reader correction: physical JSONL lines preserve Unicode separators.

Original frozen generation and failed evaluator are retained in audit.py and
evaluation.log. Only the two JSONL reader expressions differ in evaluate().
"""
import json
import sys
from audit import HERE, TOOLS, check_freeze, digest, dump_new, inputs, sha

def evaluate():
    phase1 = json.loads((HERE / "phase1.json").read_text())
    freeze = json.loads((HERE / "freeze.json").read_text())
    check_freeze(freeze)
    if phase1["status"] != "PASS_500_CONTEXTS_NO_LABELS_LOADED":
        raise ValueError("Incomplete phase1")
    if sha(HERE / "legacy-traces.jsonl") != phase1["legacy_trace_sha256"] or sha(HERE / "candidate-traces.jsonl") != phase1["candidate_trace_sha256"]:
        raise ValueError("Trace changed after generation")
    sys.path.insert(0, str(TOOLS))
    from build_question_local_english_cache import identity, render
    cache = json.loads(inputs()["cache"].read_text())
    upstream = {q["question_id"]: q for q in json.loads(inputs()["upstream_evaluator_only"].read_text())}
    corrected = {c["id"]: c for c in json.loads(inputs()["corrected_package"].read_text())["cases"]}
    precision = {c["id"]: c for c in json.loads(inputs()["precision_package"].read_text())["cases"]}
    checkpoint = json.loads(inputs()["paid_checkpoint_evaluator_only"].read_text())
    memories = {m["mid"]: m for m in cache["memories"]}
    queries = {q["question_id"]: q for q in cache["queries"]}
    candidate_traces = [json.loads(s) for s in (HERE / "candidate-traces.jsonl").open()]
    legacy_traces = [json.loads(s) for s in (HERE / "legacy-traces.jsonl").open()]
    if len(candidate_traces) != 500 or len(legacy_traces) != 500:
        raise ValueError("Trace count mismatch")
    rows = []
    for control, candidate in zip(legacy_traces, candidate_traces, strict=True):
        qid = control["id"]
        if qid != candidate["id"]:
            raise ValueError("Candidate/control order differs")
        original = upstream[qid]
        query = queries[qid]
        baseline = precision[qid]["context"]
        control_text = (HERE / "contexts" / "legacy" / (qid + ".txt")).read_text()
        candidate_text = (HERE / "contexts" / "candidate" / (qid + ".txt")).read_text()
        if control_text != baseline or digest(candidate_text) != candidate["final_context_sha256"]:
            raise ValueError("Final context binding differs: " + qid)
        grade = checkpoint["jobs"][qid + "/judge"]
        if grade["status"] != "complete" or type(grade["correct"]) is not bool:
            raise ValueError("Missing historical grade: " + qid)
        annotations = []
        for sid, date, turns in zip(original["haystack_session_ids"], original["haystack_dates"], original["haystack_sessions"], strict=True):
            key = identity(sid, date, turns)
            _, stamped = render(turns, date)
            if key not in query["scope_keys"] or memories[key]["turns"] != stamped:
                raise ValueError("Source lineage mismatch: " + qid)
            for position, (turn, source) in enumerate(zip(turns, stamped, strict=True)):
                if turn.get("has_answer") is not True:
                    continue
                text = source["content"]
                annotations.append({
                    "session_id": sid, "source_key": key, "position": position,
                    "role": turn["role"], "text_sha256": digest(text),
                    "legacy_full_body": text in control_text,
                    "candidate_full_body": text in candidate_text,
                })
        gains = [a for a in annotations if not a["legacy_full_body"] and a["candidate_full_body"]]
        losses = [a for a in annotations if a["legacy_full_body"] and not a["candidate_full_body"]]
        rows.append({
            "id": qid, "historically_correct": grade["correct"],
            "abstention": precision[qid]["abst"], "question_type": precision[qid]["type"],
            "legacy_context_sha256": digest(control_text),
            "candidate_context_sha256": digest(candidate_text),
            "pre_supplement_changed": candidate["pre_supplement_changed"],
            "supplement_changed": candidate["supplement_changed"],
            "annotations": annotations, "gained_count": len(gains), "lost_count": len(losses),
        })
    def cohort(group):
        return {"cases": len(group),
            "gain_cases": sum(r["gained_count"] > 0 for r in group),
            "loss_cases": sum(r["lost_count"] > 0 for r in group),
            "gained_turns": sum(r["gained_count"] for r in group),
            "lost_turns": sum(r["lost_count"] for r in group)}
    misses = [r for r in rows if not r["historically_correct"]]
    correct = [r for r in rows if r["historically_correct"]]
    allstats = cohort(rows); missstats = cohort(misses)
    gate = {
        "historical_miss_gain_cases_at_least_10": missstats["gain_cases"] >= 10,
        "historical_miss_net_source_cases_at_least_5": missstats["gain_cases"] - missstats["loss_cases"] >= 5,
        "no_annotated_losses_all500": allstats["loss_cases"] == 0,
    }
    dump_new(HERE / "audit.json", {
        "schema": "rank-before-chronology-offline-source-audit-v1",
        "status": "PASS_SOURCE_TRIAGE" if all(gate.values()) else "FAIL_SOURCE_TRIAGE",
        "freeze_sha256": sha(HERE / "freeze.json"), "phase1_sha256": sha(HERE / "phase1.json"),
        "cohorts": {"all500": allstats, "historical_miss": missstats,
            "historical_correct": cohort(correct), "abstention": cohort([r for r in rows if r["abstention"]])},
        "gate": gate, "semantic_completeness": "NOT_CERTIFIED",
        "answer_accuracy": "NOT_MEASURED", "paid_calls": 0, "new_answers": 0,
        "limits": ["Annotated complete-body presence is not semantic sufficiency or source identity.",
            "Precision supplement may change because the assembled baseline and remaining room changed; any apparent gain needs stage-specific tracing and source review.",
            "All 500 questions are development-exposed; no hidden accuracy claim."],
        "rows": rows,
    })


if __name__ == "__main__":
    evaluate()
