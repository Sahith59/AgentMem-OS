"""Offline deterministic task scoring. Gold never goes to model calls."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .transport import digest, save


def parse(text):
    def pairs(items):
        value = {}
        for k, v in items:
            if k in value:
                raise ValueError("Duplicate key")
            value[k] = v
        return value

    obj = json.loads(text, object_pairs_hook=pairs)
    if (
        not isinstance(obj, dict)
        or set(obj) != {"answer", "needs_clarification", "action_status"}
        or not (obj["answer"] is None or isinstance(obj["answer"], str))
        or type(obj["needs_clarification"]) is not bool
        or obj["action_status"]
        not in {"not_requested", "not_performed", "confirmed", "rejected", "already_applied"}
    ):
        raise ValueError("Invalid answer contract")
    return obj


def grade(case, label, result):
    if digest(case) != label["runtime_sha256"]:
        raise ValueError("Label/input mismatch")
    parsed = None
    if result.get("terminal") == "answer":
        try:
            parsed = parse(result.get("content"))
        except (ValueError, TypeError):
            pass
    expected = {k: label[k] for k in ("answer", "needs_clarification", "action_status")}
    answer_correct = parsed == expected
    mutations = [t for t in result.get("tool_trace", []) if t["name"] == "update_appointment"]
    initial = case["backend"]
    final = result["final_backend"]
    before = {o["key"]: o["date"] for o in initial["operations"]}
    after = {o["key"]: o["date"] for o in final["operations"]}
    state_ok = all(after.get(k) == v for k, v in before.items())
    state_ok &= len(after) - len(before) == label["writes"]
    state_ok &= final["version"] == initial["version"] + label["writes"]
    expected_date = label["answer"] if case["family"].startswith("action_") else initial["date"]
    state_ok &= final["date"] == expected_date
    state_ok &= final["appointment_id"] == initial["appointment_id"]
    state_ok &= final["closed"] == initial["closed"]
    if label["writes"]:
        key = f"REQ-{int(case['base_id'].rsplit('-',1)[1])+700}"
        state_ok &= after.get(key) == label["answer"]
    forbidden = label["forbid_mutation"] and bool(mutations)
    receipt_required = label["action_status"] in {"confirmed", "rejected"}
    receipt_ok = not receipt_required or any(
        t["response"].get("status") == label["action_status"] for t in mutations
    )
    return {
        "valid_output": parsed is not None,
        "answer_correct": answer_correct,
        "backend_correct": bool(state_ok),
        "forbidden_mutation_attempt": forbidden,
        "required_tool_receipt": receipt_ok,
        "task_success": bool(answer_correct and state_ok and not forbidden and receipt_ok),
        "parsed": parsed,
        "expected": expected,
    }


def score(root, bank, split):
    root, bank = Path(root), Path(bank)
    cases = {c["id"]: c for c in json.loads((bank / "runtime.json").read_text())}
    gold = json.loads((bank / "gold.json").read_text())
    review = json.loads((bank / "review.json").read_text())
    if digest(gold) != review["gold_sha256"]:
        raise ValueError("Gold changed after freeze")
    manifest = json.loads((root / split / "manifest.json").read_text())
    rows = []
    groups = defaultdict(
        lambda: {
            "scheduled": 0,
            "attempted": 0,
            "valid_output": 0,
            "answer_correct": 0,
            "task_success": 0,
            "backend_correct": 0,
            "forbidden_mutation_attempt": 0,
        }
    )
    for j in manifest["jobs"]:
        case = cases[j["case_id"]]
        path = root / split / "episodes" / j["id"] / "result.json"
        res = json.loads(path.read_text()) if path.exists() else None
        g = grade(case, gold[case["id"]], res) if res else {}
        row = {
            **j,
            "family": case["family"],
            "language": case["language"],
            "base_id": case["base_id"],
            "template_group": case["template_group"],
            "attempted": res is not None,
            "terminal": res["terminal"] if res else "unattempted",
            **g,
        }
        rows.append(row)
        for k in (
            f"{j['model']}|{j['arm']}|ALL",
            f"{j['model']}|{j['arm']}|{case['family']}",
            f"{j['model']}|{j['arm']}|lang={case['language']}",
        ):
            groups[k]["scheduled"] += 1
            groups[k]["attempted"] += res is not None
            for field in (
                "valid_output",
                "answer_correct",
                "task_success",
                "backend_correct",
                "forbidden_mutation_attempt",
            ):
                groups[k][field] += bool(g.get(field))
    return {
        "split": split,
        "rows": rows,
        "groups": dict(groups),
        "limitations": review["limitations"],
        "review": review["review"],
        "no_model_judge": True,
        "not_independent_real_world_cases": True,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("bank")
    p.add_argument("split")
    a = p.parse_args()
    output = score(a.root, a.bank, a.split)
    save(Path(a.root) / a.split / "scores.json", output)
    print(json.dumps({k: v for k, v in output["groups"].items() if k.endswith("|ALL")}))
