"""Independent, standard-library verifier for source-supplement artifacts.

Does not import the candidate, the project database, or any model client.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_packet(baseline, candidate, turns, receipts, char_cap=40000, extra_cap=4000):
    if not candidate.startswith(baseline):
        raise ValueError("Baseline prefix changed")
    if len(candidate) > char_cap or len(candidate) - len(baseline) > extra_cap:
        raise ValueError("Character cap exceeded")
    allowed = {(turn["role"], turn["content"]) for turn in turns}
    rendered = []
    seen = set()
    for receipt in receipts:
        start, end = receipt["packet_start"], receipt["packet_end"]
        if not len(baseline) <= start < end <= len(candidate):
            raise ValueError("Invalid source offsets")
        body = candidate[start:end]
        role = receipt["role"]
        digest = hashlib.sha256(body.encode()).hexdigest()
        if digest != receipt["source_sha256"] or (role, body) not in allowed:
            raise ValueError("Unverified source or role")
        if body in seen or body in baseline:
            raise ValueError("Duplicate source supplement")
        if len({r for r, text in allowed if text == body}) != 1:
            raise ValueError("Ambiguous source role")
        seen.add(body)
        rendered.append(f"[source {digest[:16]} | {role}]\n{body}\n")
    expected = baseline
    if rendered:
        expected += "\n\n[ADDITIONAL SOURCE EVIDENCE]\n" + "".join(rendered)
    if candidate != expected:
        raise ValueError("Unreceipted text or invalid rendering")
    return len(rendered)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_dir", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    policy = json.loads((args.audit_dir / "policy.json").read_text())
    for value in policy["inputs"].values():
        if file_sha(value["path"]) != value["sha256"]:
            raise ValueError("Changed source: " + value["path"])
    inputs = policy["inputs"]
    package = json.loads(Path(inputs["package"]["path"]).read_text())
    cache = json.loads(Path(inputs["cache"]["path"]).read_text())
    annotations = {r["question_id"]: r for r in json.loads(Path(inputs["annotations"]["path"]).read_text())}
    queries = {q["question_id"]: q for q in cache["queries"]}
    memories = {m["mid"]: m for m in cache["memories"]}
    rows = json.loads((args.audit_dir / "results.json").read_text())
    by_id = {row["question_id"]: row for row in rows}
    if len(by_id) != 500 or len(rows) != 500 or len(package["cases"]) != 500:
        raise ValueError("Expected500 unique cases")
    added = gained = lost = stable_gain_questions = 0
    receipts_manifest = []
    for case in package["cases"]:
        qid = case["id"]
        row = by_id[qid]
        candidate_path = args.audit_dir / "contexts" / f"{qid}.txt"
        receipts_path = args.audit_dir / "receipts" / f"{qid}.json"
        candidate = candidate_path.read_text()
        turns = [t for key in queries[qid]["scope_keys"] for t in memories[key]["turns"]]
        n_added = verify_packet(case["context"], candidate, turns,
                                json.loads(receipts_path.read_text()),
                                policy["character_cap"], policy["extra_character_cap"])
        before = hashlib.sha256(case["context"].encode()).hexdigest()
        after = file_sha(candidate_path)
        ann = annotations[qid]
        n_gained = sum(a["text"] not in case["context"] and a["text"] in candidate for a in ann["annotated_turns"])
        n_lost = sum(a["text"] in case["context"] and a["text"] not in candidate for a in ann["annotated_turns"])
        checks = (before == case["context_sha256"] == row["baseline_sha256"],
                  after == row["candidate_sha256"], n_added == row["added_turns"],
                  n_gained == row["gained_annotated_turns"], n_lost == row["lost_annotated_turns"],
                  len(candidate) - len(case["context"]) == row["extra_chars"],
                  row["stable_miss"] == ann["stable_miss"], row["stable_pass"] == ann["stable_pass"])
        if not all(checks):
            raise ValueError("Result row mismatch: " + qid)
        added += n_added
        gained += n_gained
        lost += n_lost
        stable_gain_questions += ann["stable_miss"] and n_gained > 0
        receipts_manifest.append({"question_id": qid, "context_sha256": after,
                                  "receipt_sha256": file_sha(receipts_path)})
    summary = json.loads((args.audit_dir / "summary.json").read_text())
    if (summary["all500"]["added_turns"] != added
            or summary["all500"]["gained_annotated_turns"] != gained
            or summary["all500"]["lost_annotated_turns"] != lost
            or summary["stable_miss"]["questions_gaining_annotated_turns"] != stable_gain_questions):
        raise ValueError("Summary mismatch")
    gate = policy["gate"]
    if stable_gain_questions < gate["stable_miss_gain_questions_min"] or lost > gate["all500_annotated_turn_losses_max"]:
        raise ValueError("Offline evidence gate failed")
    result = {"status": "PASS", "packets_verified": 500, "source_turns_verified": added,
              "gained_annotated_turns": gained, "lost_annotated_turns": lost,
              "stable_miss_questions_gaining": stable_gain_questions,
              "qa_accuracy": "NOT_MEASURED", "manifest": receipts_manifest}
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
