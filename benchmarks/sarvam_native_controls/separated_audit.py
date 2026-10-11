"""Offline replay, strict scoring and honest cost attribution for shared trajectories."""

import argparse
import hashlib
import json
import random
import socket
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from benchmarks.sarvam_native_study import backend
from benchmarks.sarvam_native_study.transport import ENDPOINT, INPUT, OUTPUT, digest, save

from .engine import grade
from .separated import paired_episode


def read(path):
    return json.loads(path.read_text())


def cluster_interval(rows, seed=202610116):
    """Descriptive paired workflow bootstrap; translated views stay together."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["workflow"]].append(int(row["schema_success"]) - int(row["control_success"]))
    if len(groups) < 2:
        return None
    blocks, values, rng = list(groups.values()), [], random.Random(seed)
    for _ in range(2000):
        sampled = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        values.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
    values.sort()
    return {
        "low": values[49],
        "high": values[1949],
        "resamples": 2000,
        "unit": "workflow",
        "workflow_count": len(blocks),
        "limitation": "Synthetic assistant-authored workflows, not population uncertainty.",
    }


def audit(root, stage):
    def denied(*args, **kwargs):
        raise RuntimeError("Offline audit forbids networking")

    socket.socket = denied
    socket.create_connection = denied
    directory = root / stage
    package = read(directory / "manifest.json")
    cases = {c["id"]: c for c in read(root / (stage + "-runtime.json"))}
    labels = read(root / (stage + "-gold.json"))
    review = read(root / (stage + "-review.json"))
    assert digest(list(cases.values())) == package["runtime_sha256"]
    assert digest(labels) == review["gold_sha256"]
    assert digest(review) == package["review_sha256"]
    assert digest(read(root / "budget.json")) == package["budget_sha256"]
    assert digest((root / "PROTOCOL.md").read_text()) == package["protocol_sha256"]
    assert read(directory / "started.json")["manifest_sha256"] == digest(package)
    for relative, expected in package["source_sha256"].items():
        assert digest((directory / "source" / relative).read_text()) == expected
        assert digest(Path(relative).read_text()) == expected
    rows, seen, provider_ids = [], set(), set()
    execution = read(directory / "execution.json")
    attempted = {o["id"] for o in execution["outcomes"]}
    assert len(attempted) == len(execution["outcomes"])
    assert attempted <= {j["id"] for j in package["jobs"]}
    for spec in package["jobs"]:
        folder = directory / "episodes" / spec["id"]
        if spec["id"] not in attempted:
            assert not folder.exists()
            continue
        case = cases[spec["case_id"]]
        prefix = read(folder / "prefix.json")
        assert backend.state(folder / "backend.sqlite") == prefix["final_backend"]
        with tempfile.TemporaryDirectory() as tmp:

            def dispatch(job, request):
                assert job not in seen
                saved = read(root / "calls" / (job + ".request.json"))
                receipt = read(root / "calls" / (job + ".receipt.json"))
                assert saved["request"] == request and saved["endpoint"] == ENDPOINT
                assert saved["sha256"] == receipt["request_sha256"] == digest(request)
                assert receipt["job"] == job
                seen.add(job)
                if receipt.get("status") != 200:
                    raise RuntimeError("Preserved provider failure")
                body = json.loads(receipt["body_text"])
                assert body["id"] not in provider_ids
                provider_ids.add(body["id"])
                if body["model"] != request["model"]:
                    raise RuntimeError("Preserved returned-model mismatch")
                return body

            replay = paired_episode(Path(tmp), spec, case, dispatch)
            assert replay == next(o for o in execution["outcomes"] if o["id"] == spec["id"])
            regenerated = Path(tmp) / "episodes" / spec["id"]
            assert read(regenerated / "prefix.json") == prefix
            requests = []
            for variant in ("control", "schema"):
                result = read(folder / (variant + ".json"))
                assert read(regenerated / (variant + ".json")) == result
                rows.append(
                    {
                        "pair_id": spec["id"],
                        "case_id": case["id"],
                        "workflow": case["workflow"],
                        "family": case["family"],
                        "language": case["language"],
                        "model": spec["model"],
                        "variant": variant,
                        "terminal": result["terminal"],
                        "shared_calls": result["shared_calls"],
                        "final_calls": result["final_calls"],
                        **grade(case, labels[case["id"]], result),
                    }
                )
                if result["final_calls"]:
                    req = read(root / "calls" / (result["final_calls"][0] + ".request.json"))[
                        "request"
                    ]
                    assert "tools" not in req and "tool_choice" not in req
                    req.pop("response_format")
                    requests.append(req)
            if len(requests) == 2:
                assert requests[0] == requests[1]
    receipts = {
        p.name.removesuffix(".receipt.json")
        for p in (root / "calls").glob(stage + "-*.receipt.json")
    }
    requests = {
        p.name.removesuffix(".request.json")
        for p in (root / "calls").glob(stage + "-*.request.json")
    }
    assert receipts == requests == seen
    with sqlite3.connect(f"file:{root}/calls/meter.sqlite?mode=ro", uri=True) as db:
        ledger = db.execute(
            "SELECT id,charged,usage_cost,status FROM calls WHERE stage=?", (stage,)
        ).fetchall()
        cumulative = db.execute(
            "SELECT SUM(charged),SUM(usage_cost),COUNT(*) FROM calls"
        ).fetchone()
    assert {r[0] for r in ledger} == seen
    by_call, known, retained, unknown = {}, 0, 0, 0
    times = []
    for job, charged, cost, status in ledger:
        receipt = read(root / "calls" / (job + ".receipt.json"))
        request = read(root / "calls" / (job + ".request.json"))["request"]
        if status == "known":
            usage = json.loads(receipt["body_text"])["usage"]
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert usage["completion_tokens"] <= request["max_tokens"]
            assert cost == usage["prompt_tokens"] * INPUT + usage["completion_tokens"] * OUTPUT
            assert charged == (cost * 5 + 3) // 4
            known += cost
        else:
            usage = {}
            unknown += 1
        retained += charged
        by_call[job] = {
            "known_microinr": cost or 0,
            "retained_microinr": charged,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "request_seconds": receipt["elapsed_seconds"],
        }
        timing = read(root / "calls" / (job + ".dispatch.json"))
        times.append(timing["dispatch_time"])
    budget = read(root / "budget.json")
    assert cumulative[0] <= budget["phase_cap_microinr"]
    assert cumulative[0] + budget["prior_retained_microinr"] <= budget["campaign_cap_microinr"]
    assert retained <= budget["stages"][stage]["cap_microinr"]
    assert len(seen) <= min(budget["stages"][stage]["calls"], package["maximum_calls"])
    groups = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for suffix in ("ALL", "lang=" + row["language"], "family=" + row["family"]):
            g = groups[row["model"] + "|" + row["variant"] + "|" + suffix]
            g["rows"] += 1
            for field in (
                "valid_output",
                "answer_correct",
                "backend_correct",
                "task_success",
                "forbidden_mutation_attempt",
                "required_receipt",
            ):
                g[field] += row[field]
            # Full common prefix is charged to each arm for a fair path estimate.
            ids = row["shared_calls"] + row["final_calls"]
            g["attributed_path_calls"] += len(ids)
            for call_id in ids:
                for key, value in by_call[call_id].items():
                    g["attributed_path_" + key] += value
    indexed = {(r["pair_id"], r["variant"]): r for r in rows}
    pairs = []
    for a in rows:
        if a["variant"] != "control":
            continue
        b = indexed[a["pair_id"], "schema"]
        pairs.append(
            {
                "case_id": a["case_id"],
                "workflow": a["workflow"],
                "model": a["model"],
                "control_success": a["task_success"],
                "schema_success": b["task_success"],
                "gain": not a["task_success"] and b["task_success"],
                "loss": a["task_success"] and not b["task_success"],
                "control_valid": a["valid_output"],
                "schema_valid": b["valid_output"],
                "both_valid": a["valid_output"] and b["valid_output"],
            }
        )
    counts = {}
    for model in sorted({p["model"] for p in pairs}):
        selected = [p for p in pairs if p["model"] == model]
        counts[model] = {
            "pairs": len(selected),
            "gains": sum(p["gain"] for p in selected),
            "losses": sum(p["loss"] for p in selected),
            "gains_from_invalid_control": sum(
                p["gain"] and not p["control_valid"] for p in selected
            ),
            "both_valid_gains": sum(p["gain"] and p["both_valid"] for p in selected),
            "both_valid_losses": sum(p["loss"] and p["both_valid"] for p in selected),
            "workflow_bootstrap_delta_interval": cluster_interval(selected),
        }
    return {
        "verified": True,
        "stage": stage,
        "scheduled_pairs": len(package["jobs"]),
        "attempted_pairs": len(attempted),
        "rows": rows,
        "pairs": pairs,
        "groups": {k: dict(v) for k, v in groups.items()},
        "paired_counts": counts,
        "actual_unique_calls": len(seen),
        "successful_unique_provider_ids": len(provider_ids),
        "known_inr": known / 1e6,
        "retained_inr": retained / 1e6,
        "unknown_calls": unknown,
        "phase_cumulative_known_inr": (cumulative[1] or 0) / 1e6,
        "research_retained_inr": (cumulative[0] + budget["prior_retained_microinr"]) / 1e6,
        "maximum_dispatches_in_rolling_60s": max(
            sum(t <= s < t + 60 for s in times) for t in times
        ),
        "limitations": [
            "Shared trajectories do not compare independent action policies",
            "Same-code replay is not independent semantic review",
            "Cost attribution counts the shared prefix in each arm, not twice in actual spending",
            "Synthetic language renderings share workflow ancestry",
        ],
    }


def gate(root, result):
    schema = [r for r in result["rows"] if r["variant"] == "schema"]
    files = [
        root / "probe/manifest.json",
        root / "probe/started.json",
        root / "probe/execution.json",
        root / "probe-runtime.json",
        root / "probe-gold.json",
        root / "probe-review.json",
    ] + list((root / "probe/episodes").glob("*/*.json"))
    files += list((root / "calls").glob("probe-*.request.json"))
    files += list((root / "calls").glob("probe-*.receipt.json"))
    return {
        "pass": len(schema) == result["scheduled_pairs"]
        and all(r["task_success"] for r in schema)
        and result["unknown_calls"] == 0,
        "candidate_passed": sum(r["task_success"] for r in schema),
        "candidate_expected": result["scheduled_pairs"],
        "evidence_sha256": {str(p.relative_to(root)): digest(p.read_text()) for p in files},
        "database_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "probe/episodes").glob("*/backend.sqlite")
        },
        "meaning": "Excluded mechanics only; not a general quality or deployment gate",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("probe", "comparison"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--write-gate", action="store_true")
    a = parser.parse_args()
    result = audit(a.root, a.stage)
    save(a.output, result)
    if a.write_gate:
        assert a.stage == "probe"
        save(a.root / "probe/gate.json", gate(a.root, result))
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "pairs", "groups")}))
