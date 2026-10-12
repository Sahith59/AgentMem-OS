"""Immutable, metered fresh-workflow comparison; no paid runner reads gold."""

import argparse
import hashlib
import random
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks import sarvam_ordinary_v2 as ordinary
from benchmarks.sarvam_fresh_confirmation.engine import episode
from benchmarks.sarvam_native_controls.engine import PROFILES, ControlMeter
from benchmarks.sarvam_native_controls.integrated import initialize, read
from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.transport import call, digest, read_key, save
from benchmarks.sarvam_reporter_wording import reservation
from benchmarks.sarvam_reporter_wording import sources as parent_sources

FROZEN = (
    "runtime.json",
    "gold.json",
    "review.json",
    "budget.json",
    "PROTOCOL.md",
    "pacing-control.json",
)


def sources():
    result = parent_sources()
    repo = Path(__file__).parents[2]
    paths = list(Path(__file__).parent.glob("*.py")) + [
        repo / "benchmarks/sarvam_source_attribution.py"
    ]
    result.update({str(p.relative_to(repo)): digest(p.read_text()) for p in paths})
    return result


def prepare(root):
    cases, labels, review = (read(root / n) for n in ("runtime.json", "gold.json", "review.json"))
    if review["runtime_sha256"] != digest(cases) or review["gold_sha256"] != digest(labels):
        raise ValueError("Unreviewed cases/labels")
    if (
        len(cases) != 120
        or len({c["id"] for c in cases}) != 120
        or len({c["workflow"] for c in cases}) != 40
    ):
        raise ValueError("Expected 120 renderings / 40 scenario instances")
    if Counter(c["language"] for c in cases) != {"en": 40, "hi": 40, "te": 40}:
        raise ValueError("Language inventory changed")
    if set(labels) != {c["id"] for c in cases} or any(
        labels[c["id"]]["runtime_sha256"] != digest(c) for c in cases
    ):
        raise ValueError("Gold/runtime binding failed")
    jobs = []
    with tempfile.TemporaryDirectory() as temporary:
        for case in cases:
            path = Path(temporary) / (case["id"] + ".sqlite")
            seeds = initialize(path, case)
            for i, model in enumerate(PROFILES):
                request = ordinary.initial_request(path, case, model, seeds)
                jobs.append(
                    {
                        "id": f"confirm-{case['id']}-m{i}",
                        "case_id": case["id"],
                        "model": model,
                        "enforce": True,
                        "initial_sha256": digest(request),
                        "maximum_reservation_microinr": reservation(request) * 5,
                    }
                )
    random.Random(202610122).shuffle(jobs)
    counts = Counter()
    for job in jobs:
        job["report_order"] = (
            ["control", "candidate"] if counts[job["model"]] % 2 == 0 else ["candidate", "control"]
        )
        counts[job["model"]] += 1
    budget = read(root / "budget.json")
    worst = sum(j["maximum_reservation_microinr"] for j in jobs)
    if (
        worst > budget["phase_cap_microinr"]
        or worst + budget["prior_retained_microinr"] > budget["campaign_cap_microinr"]
        or budget["stages"]["confirmation"]["calls"] != 1200
    ):
        raise ValueError("Insufficient frozen budget")
    package = {
        "jobs": jobs,
        "source_sha256": sources(),
        "files_sha256": {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in FROZEN},
        "maximum_calls": 1200,
        "worst_case_retained_microinr": worst,
        "workers": 4,
        "batch_size": 4,
        "stop_after_batch_errors": 2,
        "no_retries": True,
        "native_action_runs_per_case_model": 1,
        "native_call_limit": 3,
    }
    for relative in package["source_sha256"]:
        dest = root / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x") as f:
            f.write((Path(__file__).parents[2] / relative).read_text())
    save(root / "manifest.json", package)
    return package


def check(root):
    package = read(root / "manifest.json")
    if package["source_sha256"] != sources():
        raise ValueError("Frozen source changed")
    for relative, expected in package["files_sha256"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen input/configuration changed")
    for relative, expected in package["source_sha256"].items():
        if digest((root / "source" / relative).read_text()) != expected:
            raise ValueError("Frozen source copy changed")
    return package, {c["id"]: c for c in read(root / "runtime.json")}


def run(root):
    package, cases = check(root)
    save(root / "started.json", {"manifest_sha256": digest(package)})
    meter, pacer, key = ControlMeter(root), Pacer(root), read_key()

    def dispatch(job, request):
        save(meter.root / (job + ".dispatch.json"), pacer.wait())
        return call(meter, job, "confirmation", request, key)

    outcomes = []
    with ThreadPoolExecutor(max_workers=package["workers"]) as pool:
        for offset in range(0, len(package["jobs"]), package["batch_size"]):
            futures = [
                pool.submit(episode, root, j, cases[j["case_id"]], dispatch)
                for j in package["jobs"][offset : offset + package["batch_size"]]
            ]
            results = [f.result() for f in futures]
            errors = sum(
                r["original"]["terminal"] in ("infrastructure_error", "backend_error")
                or any(v["terminal"] == "infrastructure_error" for v in r["reports"].values())
                for r in results
            )
            outcomes.extend(
                {"id": r["original"]["id"], "actual_calls": r["actual_calls"]} for r in results
            )
            print(
                {"episodes": len(outcomes), "errors": errors, "meter": meter.summary()}, flush=True
            )
            if errors >= package["stop_after_batch_errors"]:
                break
    check(root)
    save(root / "execution.json", {"outcomes": outcomes, "meter": meter.summary()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "check", "run"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    {"prepare": prepare, "check": check, "run": run}[args.command](args.root)
