#!/usr/bin/env python3
"""Build and score the precision-limited supplement without network/model calls."""

import argparse
import hashlib
import json
from pathlib import Path
import socket

from agentmem_os.benchmarks.precise_lexical_retrieval import MAX_TURNS, MIN_SCORE
from agentmem_os.benchmarks.precise_source_supplement import supplement_packet


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "package": root / "full500-corrected-measurement-package-001/package.json",
        "cache": root / "integrity-audit-001/corrected-cache-v2/longmemeval_s.json",
        "annotations": root / "raw-vocabulary-audit-002/results.json",
        "candidate": Path(__file__).with_name("precise_source_supplement.py"),
        "ranker": Path(__file__).with_name("precise_lexical_retrieval.py"),
        "runner": Path(__file__),
    }
    inputs = {
        name: {"path": str(path.resolve()), "sha256": file_sha(path)}
        for name, path in paths.items()
    }
    policy = {
        "status": "FROZEN_BEFORE_EXACT_V2_PACKET_GENERATION",
        "development_exposure": (
            "Parameters were selected after reviewing the first paid screen and "
            "offline threshold sweeps. This is not held-out model selection."
        ),
        "inputs": inputs,
        "character_cap": 40_000,
        "extra_character_cap": 4_000,
        "max_added_turns": MAX_TURNS,
        "minimum_cosine_score_exclusive": MIN_SCORE,
        "vectorizer": {
            "sublinear_tf": True,
            "min_df": 1,
            "stop_words": "english",
            "ngram_range": [1, 2],
        },
        "mechanism": (
            "Preserve the complete baseline packet; append at most five complete "
            "source turns ranked by stopword-aware unigram/bigram TF-IDF, each "
            "strictly above 0.07 cosine similarity."
        ),
        "prohibited_selection_inputs": [
            "gold answer", "benchmark annotation", "case ID blacklist",
            "paid-screen correctness", "cohort label",
        ],
        "gate": {
            "exact_prefix_count": 500,
            "stable_miss_gain_questions_min": 20,
            "all500_annotated_turn_losses_max": 0,
            "every_added_turn_complete_and_scoped": True,
        },
        "precision_gate": {"added_turns_reduction_min_fraction": 0.70},
        "paid_execution": "NOT_AUTHORIZED",
    }
    (output / "policy.json").write_text(json.dumps(policy, indent=2) + "\n")

    def deny(*args, **kwargs):
        raise RuntimeError("Offline audit: network prohibited")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    package = json.loads(paths["package"].read_text())
    cache = json.loads(paths["cache"].read_text())
    annotations = {
        row["question_id"]: row
        for row in json.loads(paths["annotations"].read_text())
    }
    queries = {row["question_id"]: row for row in cache["queries"]}
    memories = {row["mid"]: row for row in cache["memories"]}
    (output / "contexts").mkdir()
    (output / "receipts").mkdir()
    rows = []
    for case in package["cases"]:
        qid = case["id"]
        turns = [
            turn
            for key in queries[qid]["scope_keys"]
            for turn in memories[key]["turns"]
        ]
        candidate, receipts = supplement_packet(
            case["context"], turns, case["question"]
        )
        if not candidate.startswith(case["context"]):
            raise ValueError("Baseline prefix changed")
        if len(candidate) > 40_000 or len(candidate) - len(case["context"]) > 4_000:
            raise ValueError("Candidate exceeds character cap")
        if len(receipts) > MAX_TURNS:
            raise ValueError("Candidate exceeds fixed turn count")
        for receipt in receipts:
            body = candidate[receipt["packet_start"]:receipt["packet_end"]]
            if hashlib.sha256(body.encode()).hexdigest() != receipt["source_sha256"]:
                raise ValueError("Source receipt hash mismatch")
            if not any(
                turn["content"] == body and turn["role"] == receipt["role"]
                for turn in turns
            ):
                raise ValueError("Unscoped source receipt")
        observed = annotations[qid]
        gained = [
            row for row in observed["annotated_turns"]
            if not row["baseline_exact_turn"] and row["text"] in candidate
        ]
        lost = [
            row for row in observed["annotated_turns"]
            if row["baseline_exact_turn"] and row["text"] not in candidate
        ]
        row = {
            "question_id": qid,
            "stable_miss": observed["stable_miss"],
            "stable_pass": observed["stable_pass"],
            "abstention": case["abst"],
            "baseline_sha256": case["context_sha256"],
            "candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            "prefix_preserved": True,
            "extra_chars": len(candidate) - len(case["context"]),
            "added_turns": len(receipts),
            "gained_annotated_turns": len(gained),
            "lost_annotated_turns": len(lost),
            "gained_source_turns": [
                {"source_key": item["source_key"], "turn_index": item["turn_index"]}
                for item in gained
            ],
        }
        (output / "contexts" / f"{qid}.txt").write_text(candidate)
        (output / "receipts" / f"{qid}.json").write_text(
            json.dumps(receipts, indent=2) + "\n"
        )
        rows.append(row)

    def summarize(selected):
        return {
            "questions": len(selected),
            "changed": sum(row["extra_chars"] > 0 for row in selected),
            "questions_gaining_annotated_turns": sum(
                row["gained_annotated_turns"] > 0 for row in selected
            ),
            "gained_annotated_turns": sum(
                row["gained_annotated_turns"] for row in selected
            ),
            "lost_annotated_turns": sum(
                row["lost_annotated_turns"] for row in selected
            ),
            "extra_chars_mean": sum(row["extra_chars"] for row in selected)
            / len(selected),
            "added_turns": sum(row["added_turns"] for row in selected),
        }

    summary = {
        "all500": summarize(rows),
        "stable_miss": summarize([row for row in rows if row["stable_miss"]]),
        "stable_pass": summarize([row for row in rows if row["stable_pass"]]),
    }
    original = json.loads(
        (root / "source-supplement-audit-001/summary.json").read_text()
    )["all500"]
    reduction = 1 - summary["all500"]["added_turns"] / original["added_turns"]
    summary["comparison_to_broad_supplement"] = {
        "broad_added_turns": original["added_turns"],
        "precise_added_turns": summary["all500"]["added_turns"],
        "added_turns_reduction_fraction": reduction,
        "broad_added_chars_mean": original["extra_chars_mean"],
        "precise_added_chars_mean": summary["all500"]["extra_chars_mean"],
    }
    gate = policy["gate"]
    precision_gate = policy["precision_gate"]
    summary["checks"] = {
        "all500_prefix_preserved": len(rows) == 500
        and all(row["prefix_preserved"] for row in rows),
        "stable_miss_gain_questions": summary["stable_miss"][
            "questions_gaining_annotated_turns"
        ] >= gate["stable_miss_gain_questions_min"],
        "no_annotated_turn_loss": summary["all500"]["lost_annotated_turns"] == 0,
        "added_turns_reduction": reduction
        >= precision_gate["added_turns_reduction_min_fraction"],
        "all_source_receipts_and_caps": True,
        "inputs_preserved": all(
            file_sha(path) == inputs[name]["sha256"]
            for name, path in paths.items()
        ),
    }
    summary["offline_gate_pass"] = all(summary["checks"].values())
    summary["qa_accuracy"] = "NOT_MEASURED"
    summary["paid_calls"] = 0
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
