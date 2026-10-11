"""Versioned pacing/continuation. Failed or completed episodes are never resent."""

import argparse
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import diagnostic, runner
from .transport import Meter, call, digest, read_key, save


class Pacer:
    def __init__(self, root):
        self.root = Path(root)
        self.control = self.root / "pacing-control.json"
        if not self.control.exists():
            save(
                self.control,
                {"rpm": 35, "basis": "below documented Starter40; account tier unknown"},
            )
        self.db = self.root / "pacing.sqlite"
        with sqlite3.connect(self.db) as db:
            db.execute("CREATE TABLE IF NOT EXISTS pacing (id INTEGER PRIMARY KEY, next_time REAL)")
            db.execute("INSERT OR IGNORE INTO pacing VALUES (1,0)")

    def wait(self):
        control = json.loads(self.control.read_text())
        rpm = control["rpm"]
        if type(rpm) not in (int, float) or not 1 <= rpm <= 100:
            raise ValueError("Invalid research request pacing")
        with sqlite3.connect(self.db, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT next_time FROM pacing WHERE id=1").fetchone()[0]
            slot = max(time.time(), previous)
            db.execute("UPDATE pacing SET next_time=? WHERE id=1", (slot + 60 / rpm,))
        while (delay := slot - time.time()) > 0:
            time.sleep(min(delay, 1))
        return {"dispatch_time": time.time(), "pacing_control": control}


def continuation_jobs(package, execution, original_dir):
    attempted = {r["id"] for r in execution["outcomes"]}
    scheduled = {j["id"] for j in package["jobs"]}
    if not attempted <= scheduled or len(attempted) != len(execution["outcomes"]):
        raise ValueError("Invalid original execution inventory")
    for j in package["jobs"]:
        exists = (Path(original_dir) / "episodes" / j["id"] / "result.json").exists()
        if exists != (j["id"] in attempted):
            raise ValueError("Incomplete or unexpected original episode")
    return [j for j in package["jobs"] if j["id"] not in attempted]


def continue_discovery(root, bank):
    directory = root / "discovery-continuation-001"
    directory.mkdir()
    old = json.loads((root / "discovery/manifest.json").read_text())
    executed = json.loads((root / "discovery/execution.json").read_text())
    jobs = continuation_jobs(old, executed, root / "discovery")
    # Original IDs must be entirely unattempted, even a single API turn.
    for j in jobs:
        if list((root / "calls").glob(j["id"] + "-t*.request.json")):
            raise ValueError("Refusing any previously attempted job")
    package = {
        **old,
        "jobs": jobs,
        "maximum_calls": len(jobs) * 4,
        "original_manifest_sha256": digest(old),
        "original_execution_sha256": digest(executed),
        "model_requests_unchanged": True,
        "execution_change": "shared paced dispatch only",
        "code_sha256": {p.name: digest(p.read_text()) for p in Path(__file__).parent.glob("*.py")},
    }
    save(directory / "manifest.json", package)
    for p in Path(__file__).parent.glob("*.py"):
        (directory / (p.name + ".frozen")).write_bytes(p.read_bytes())
    cases = {c["id"]: c for c in json.loads((bank / "runtime.json").read_text())}
    assert digest(list(json.loads((bank / "runtime.json").read_text()))) == package["bank_sha256"]
    meter, key = Meter(root / "calls"), read_key()
    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(jobs), 8):
            batch = jobs[offset : offset + 8]
            futures = [
                pool.submit(runner.episode, directory, meter, key, j, cases[j["case_id"]])
                for j in batch
            ]
            results = [f.result() for f in futures]
            outcomes.extend(results)
            errors = sum(r["terminal"] == "infrastructure_error" for r in results)
            print(
                json.dumps(
                    {
                        "episodes_done": len(outcomes),
                        "total": len(jobs),
                        "batch_errors": errors,
                        "meter": meter.summary(),
                    }
                ),
                flush=True,
            )
            if errors >= 3:
                break
    save(
        directory / "execution.json",
        {"scheduled": len(jobs), "outcomes": outcomes, "meter": meter.summary()},
    )


def execute(mode, root, bank):
    root, bank = Path(root), Path(bank)
    pacer = Pacer(root)

    def dispatch(meter, job, stage, request, key):
        timing = pacer.wait()
        save(root / "calls" / (job + ".dispatch.json"), timing)
        return call(meter, job, stage, request, key)

    # Explicit dependency injection in this isolated process; request construction is unchanged.
    runner.call = dispatch
    diagnostic.call = dispatch
    if mode == "continue":
        continue_discovery(root, bank)
    elif mode == "attribution":
        save(
            root / "attribution/pacing-addendum.json",
            {
                "change": "shared pacing added before first diagnostic call; requests unchanged",
                "pacing_source_sha256": digest(Path(__file__).read_text()),
            },
        )
        diagnostic.run(root, bank)
    elif mode == "confirmation":
        runner.run(root, bank, "confirmation")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("continue", "attribution", "confirmation"))
    p.add_argument("root")
    p.add_argument("bank")
    a = p.parse_args()
    execute(a.mode, a.root, a.bank)
