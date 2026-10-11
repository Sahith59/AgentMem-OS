"""Offline replay of every published campaign run; no credentials or new model calls."""

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def deny_network(event, args):
    if event.startswith("socket."):
        raise RuntimeError("Evidence replay must not access the network")


sys.addaudithook(deny_network)

from benchmarks.english_screen.runner import atomic  # noqa: E402
from benchmarks.sarvam_discovery.contract import file_sha, loads, sha  # noqa: E402


def verify():
    manifest = loads((HERE / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        assert file_sha(HERE / name) == digest, f"Changed artifact: {name}"
    results = []
    for saved in sorted(HERE.glob("run-*")):
        package = loads((saved / "package.json").read_text())
        version = package["schema"].rsplit("-", 1)[1]
        assert version in {"v3", "v4", "v5", "v6"}
        runner = importlib.import_module(f"benchmarks.sarvam_comparison_{version}.runner")
        scorer = importlib.import_module(f"benchmarks.sarvam_comparison_{version}.score")
        runner.preflight(package)
        state = loads((saved / "checkpoint.json").read_text())
        approval = loads((saved / "approval.json").read_text())
        assert sha(approval) == state["binding"]["approval_sha256"]
        assert approval["package_sha256"] == sha(package) == state["binding"]["package_sha256"]
        assert state["binding"]["mode"] == approval["mode"] == "live"
        with tempfile.TemporaryDirectory(prefix="sarvam-saved-replay-") as temp:
            directory = Path(temp).resolve()
            # Originals remain byte-for-byte unchanged. A temporary copy changes
            # only the bound local output path and its derived approval hash.
            approval["output_directory"] = str(directory)
            state["binding"]["approval_sha256"] = sha(approval)
            atomic(directory / "approval.json", approval)
            atomic(directory / "checkpoint.json", state)
            for store in saved.glob("*.sqlite"):
                shutil.copyfile(store, directory / store.name)
            score = scorer.evaluate(package, directory)
            assert score == loads((saved / "score.json").read_text())
        known_input = known_output = 0
        for job in state["jobs"].values():
            if "usage_ninr" in job:
                body = loads(job["receipt"]["body_text"])
                known_input += body["usage"]["prompt_tokens"]
                known_output += body["usage"]["completion_tokens"]
        results.append(
            {
                "run": saved.name,
                "status": score["status"],
                "dispatch_started": score["execution"]["dispatch_started"],
                "unknown_usage_calls": score["execution"]["attempts_with_unknown_usage"],
                "known_input_tokens": known_input,
                "known_output_tokens": known_output,
                "known_current_rate_estimate_micro_inr": known_input * 15 + known_output * 60,
                "reserved_ninr": state["reserved_ninr"],
                "answers": {
                    arm: {
                        k: v[k] for k in ("correct_answers", "intended", "accuracy_over_intended")
                    }
                    for arm, v in score["arms"].items()
                },
            }
        )
    assert len(results) <= 5
    assert sum(v["dispatch_started"] for v in results) <= 360
    assert sum(v["reserved_ninr"] for v in results) <= 1489541529600
    return {
        "status": "PASS_SAVED_EVIDENCE_REPLAY",
        "new_network_calls": 0,
        "temporary_changes": "output directory and derived approval hash only",
        "semantic_review": "Assistant assessment; not automatically established by replay",
        "provider_attestation": "Saved receipts; not independently provider-attested",
        "runs": results,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
