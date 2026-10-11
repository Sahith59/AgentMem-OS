"""Frozen, metered study driver for the versioned ordinary integration."""

import argparse
import hashlib
import random
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks import sarvam_ordinary_v2 as engine
from benchmarks.sarvam_native_controls.engine import PROFILES, ControlMeter
from benchmarks.sarvam_native_controls.integrated import (
    initialize,
    read,
)
from benchmarks.sarvam_native_controls.integrated import (
    source_hashes as old_sources,
)
from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.transport import call, digest, read_key, save


def source_hashes():
    result = old_sources()
    for name in (
        "sarvam_ordinary_v2.py",
        "sarvam_ordinary_study.py",
        "sarvam_ordinary_audit.py",
        "sarvam_challenge_cases.py",
        "sarvam_receipt_reporting.py",
    ):
        path = Path(__file__).parent / name
        result["benchmarks/" + name] = digest(path.read_text())
    return result


def prepare(root, stage):
    if stage not in ("setup", "comparison"):
        raise ValueError("Undeclared stage")
    if stage == "comparison":
        from benchmarks.sarvam_ordinary_audit import audit

        audited = audit(root, "setup")
        if audited != read(root / "setup/verification.json") or not audited["gate_pass"]:
            raise ValueError("New version's integration gate has not passed")
    cases, review = read(root / f"{stage}-runtime.json"), read(root / "review.json")
    if digest(cases) != review[stage + "_runtime_sha256"]:
        raise ValueError("Unreviewed runtime")
    if review["assistant_source_review"] != "complete" or len(cases) != (
        6 if stage == "setup" else 72
    ):
        raise ValueError("Bank scope/review not ready")
    directory = root / stage
    directory.mkdir()
    jobs = []
    with tempfile.TemporaryDirectory() as temporary:
        for case in cases:
            db = Path(temporary) / (case["id"] + ".sqlite")
            seeds = initialize(db, case)
            for i, model in enumerate(PROFILES):
                for enforce in (True,) if stage == "setup" else (False, True):
                    jobs.append(
                        {
                            "id": f"{stage}-{case['id']}-m{i}-{'g' if enforce else 'c'}",
                            "case_id": case["id"],
                            "model": model,
                            "enforce": enforce,
                            "initial_sha256": digest(
                                engine.initial_request(db, case, model, seeds)
                            ),
                        }
                    )
    random.Random(202610120).shuffle(jobs)
    files = (
        f"{stage}-runtime.json",
        f"{stage}-gold.json",
        "review.json",
        "budget.json",
        "PROTOCOL.md",
        "CASE_REVIEW.md",
        "pacing-control.json",
    )
    package = {
        "stage": stage,
        "jobs": jobs,
        "maximum_calls": len(jobs) * 4,
        "workers": 4,
        "batch_size": 4,
        "no_retries": True,
        "stop_after_batch_errors": 2,
        "source_sha256": source_hashes(),
        "files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
    }
    if stage == "comparison":
        package["setup_gate_sha256"] = digest(read(root / "setup/verification.json"))
    for name in package["source_sha256"]:
        destination = directory / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as f:
            f.write((Path(__file__).parents[1] / name).read_text())
    save(directory / "manifest.json", package)


def check(root, stage):
    package = read(root / stage / "manifest.json")
    if package["stage"] != stage or package["source_sha256"] != source_hashes():
        raise ValueError("Frozen stage/source changed")
    for name, value in package["files_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != value:
            raise ValueError("Frozen configuration changed")
    for name, value in package["source_sha256"].items():
        if digest((root / stage / "source" / name).read_text()) != value:
            raise ValueError("Frozen source copy changed")
    return package, {c["id"]: c for c in read(root / f"{stage}-runtime.json")}


def run(root, stage):
    package, cases = check(root, stage)
    if stage == "comparison":
        from benchmarks.sarvam_ordinary_audit import audit

        gate = audit(root, "setup")
        if (
            not gate["gate_pass"]
            or digest(gate) != package["setup_gate_sha256"]
            or gate != read(root / "setup/verification.json")
        ):
            raise ValueError("Integration gate failed before credential access")
    directory = root / stage
    save(directory / "started.json", {"manifest_sha256": digest(package)})
    meter, pacer, key = ControlMeter(root), Pacer(root), read_key()

    def dispatch(job, request):
        save(meter.root / (job + ".dispatch.json"), pacer.wait())
        return call(meter, job, stage, request, key)

    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(package["jobs"]), 4):
            futures = [
                pool.submit(engine.episode, directory, job, cases[job["case_id"]], dispatch)
                for job in package["jobs"][offset : offset + 4]
            ]
            results = [f.result() for f in futures]
            outcomes.extend({"id": r["id"], "terminal": r["terminal"]} for r in results)
            errors = sum(
                r["terminal"] in ("infrastructure_error", "backend_error") for r in results
            )
            print(
                {
                    "stage": stage,
                    "completed": len(outcomes),
                    "batch_errors": errors,
                    "meter": meter.summary(),
                },
                flush=True,
            )
            if errors >= package["stop_after_batch_errors"]:
                break
    save(directory / "execution.json", {"outcomes": outcomes, "meter": meter.summary()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("setup", "comparison"))
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args.root, args.stage)
