"""Replay observed tool calls with ordinary approval enforcement flipped.

This is a post-hoc backend mechanism check, not a new model trajectory or answer
score. No model is called and no hypothetical downstream response is invented.
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchmarks.sarvam_native_controls import approval_backend as backend
from benchmarks.sarvam_native_controls.integrated import initialize, read, tool_reply
from benchmarks.sarvam_native_study.transport import digest, save


def replay(case, events, *, enforce):
    allowed = {"read_current"}
    if case["intent"] == "change":
        allowed.add("update_appointment")
    with tempfile.TemporaryDirectory() as temporary:
        db = Path(temporary) / "backend.sqlite"
        initialize(db, case)
        replies = [
            tool_reply(db, e["name"], e["arguments"], case, enforce, allowed) for e in events
        ]
        return {"replies": replies, "state": backend.inspect_state(db)}


def compare(case, label, result):
    if digest(case) != label["runtime_sha256"]:
        raise ValueError("Unbound case")
    actual = replay(case, result["tool_trace"], enforce=result["enforce"])
    if actual["state"] != result["final_backend"] or actual["replies"] != [
        event["response"] for event in result["tool_trace"]
    ]:
        raise ValueError("Observed backend execution does not reproduce")
    flipped = replay(case, result["tool_trace"], enforce=not result["enforce"])
    guarded = actual if result["enforce"] else flipped
    unguarded = flipped if result["enforce"] else actual
    return {
        "id": result["id"],
        "case_id": case["id"],
        "observed_enforce": result["enforce"],
        "observed_model": result["model"],
        "observed_replayed_exactly": True,
        "same_observed_calls": [
            {"name": e["name"], "arguments": e["arguments"]} for e in result["tool_trace"]
        ],
        "guarded": guarded,
        "unguarded": unguarded,
        "guard_blocks_forbidden_effect": bool(
            label["forbid_new_write"]
            and unguarded["state"] != result["initial_backend"]
            and guarded["state"] == result["initial_backend"]
        ),
        "required_write_preserved_in_both": bool(
            label["required_new_write"]
            and guarded["state"] == unguarded["state"] == label["expected_state"]
        ),
        "model_response_after_flipped_receipts": "NOT_OBSERVED",
    }


def audit(root, primary_path):
    primary = read(primary_path)
    if not primary["verified"]:
        raise ValueError("Verified primary replay required")
    cases = {c["id"]: c for c in read(root / "comparison-runtime.json")}
    labels = read(root / "comparison-gold.json")
    rows = []
    with (
        patch("socket.socket", side_effect=RuntimeError("Offline backend replay")),
        patch("socket.create_connection", side_effect=RuntimeError("Offline backend replay")),
    ):
        for row in primary["rows"]:
            case = cases[row["case_id"]]
            result = read(root / "comparison/episodes" / row["id"] / "result.json")
            rows.append(compare(case, labels[case["id"]], result))
    return {
        "verified": True,
        "kind": "POST_HOC_SAME_CALLS_BACKEND_CHECK_NOT_NEW_MODEL_RUN",
        "primary_verification_sha256": digest(primary),
        "new_calls": 0,
        "rows": rows,
        "guard_blocks_forbidden_effect": sum(r["guard_blocks_forbidden_effect"] for r in rows),
        "required_write_preserved_in_both": sum(
            r["required_write_preserved_in_both"] for r in rows
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("primary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save(args.output, audit(args.root, args.primary))
