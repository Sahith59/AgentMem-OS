"""Independent offline mixed-outcome simulation; never a live accuracy result."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


def deny_network(event, args):
    if event.startswith("socket."):
        raise RuntimeError("Network is forbidden in this audit")


sys.addaudithook(deny_network)

from benchmarks.sarvam_comparison_v3 import contract as c  # noqa: E402
from benchmarks.sarvam_comparison_v3 import runner as r  # noqa: E402
from benchmarks.sarvam_comparison_v3.score import evaluate  # noqa: E402
from benchmarks.sarvam_discovery.storage import read_packet  # noqa: E402


def verify(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    package = c.loads((HERE / "package.json").read_text())
    plan = r.preflight(package)
    prior = ROOT / "benchmarks/audits/sarvam-small-comparison-2026-10-11/live-run-001"
    saved = c.loads((prior / "diagnosis.json").read_text())["final_content"]
    counters = {"calls": 0, "extract": 0}

    def synthetic_provider(req):
        counters["calls"] += 1
        checkpoint = c.loads((directory / "checkpoint.json").read_text())
        attempt = list(checkpoint["jobs"].values())[-1]
        assert attempt["status"] == "pending" and attempt["dispatch_started"]
        assert attempt["request"] == req
        assert checkpoint["reserved_ninr"] == counters["calls"] * r.RESERVATION
        body = c.loads(req["messages"][1]["content"])
        if "fields" in body:
            result = c.canonical(
                {
                    "needs_clarification": True,
                    "fields": [
                        {"name": name, "value": None, "source_ids": []} for name in body["fields"]
                    ],
                }
            )
            if counters["calls"] == 1:
                result = "SYNTHETIC_MALFORMED_ANSWER"
        else:
            counters["extract"] += 1
            selector = counters["extract"] % 3
            if selector == 1:
                result = saved  # Saved output is injected, never described as a fresh model result.
            elif selector == 2:
                source = body["records"][0]
                result = c.canonical(
                    {
                        "facts": [
                            {
                                "attribute": "retained_statement",
                                "value": source["text"],
                                "status": "current",
                                "condition": None,
                                "valid_from": None,
                                "valid_until": None,
                                "source_ids": [source["id"]],
                            }
                        ]
                    }
                )
            else:
                result = '{"facts":[]}'
        return {
            "status_code": 200,
            "elapsed_seconds": 0.001,
            "body_text": c.canonical(
                {
                    "id": f"synthetic-audit-{counters['calls']}",
                    "model": c.MODEL,
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    "choices": [{"finish_reason": "stop", "message": {"content": result}}],
                }
            ),
        }

    approval = {
        "approved": True,
        "mode": "offline-test",
        "package_sha256": c.sha(package),
        "maximum_attempts": 72,
        "budget_ninr": plan["maximum_reservation_ninr"],
        "output_directory": str(directory),
        "accepted_review_level": package["review_level"],
        "authorization_text": "SYNTHETIC OFFLINE AUDIT ONLY; no live authorization",
    }
    execution = r.run(
        package, directory, approval, mode="offline-test", provider=synthetic_provider
    )
    assert execution["resolved"] == counters["calls"] == 72
    assert execution["completed"] == 63
    assert execution["rejected_extractions"] == 8 and execution["invalid_answers"] == 1
    state = c.loads((directory / "checkpoint.json").read_text())
    score = evaluate(package, directory)
    assert score["status"] == "SIMULATED_TEST_ONLY"
    assert score["paired"]["comparable"] == 23
    paths = score["arms"]["sqlite"]["by_evidence_path"]
    assert {k: v["intended"] for k, v in paths.items()} == {
        "originals_after_rejection": 8,
        "originals_with_admitted_facts": 8,
        "originals_with_empty_extraction": 8,
    }
    originals, fallback_pairs = 0, 0
    for case in c.runtime_cases():
        for repeat in (1, 2):
            prefix = f"r{repeat}-{case['id']}-"
            sql = state["jobs"][prefix + "sqlite"]
            history = state["jobs"][prefix + "full_history"]
            facts, records = read_packet(directory / sql["store_file"], case)
            originals += len(records)
            sql_body = c.loads(sql["request"]["messages"][1]["content"])
            control_body = c.loads(history["request"]["messages"][1]["content"])
            assert records == c.eligible(case) == sql_body["records"] == control_body["records"]
            assert facts == sql_body["facts"]
            if not facts:
                assert sql["request"] == history["request"]
                fallback_pairs += 1
    assert originals == 60 and fallback_pairs == 16
    assert len(list(directory.glob("*.sqlite"))) == 24
    assert all(
        set(row["semantic_review"].values()) == {"UNREVIEWED"} for row in score["extraction_review"]
    )
    assert (
        sum(a["known_usage_ninr"] for a in score["metering"]["arms"].values())
        == execution["usage_ninr"]
    )
    assert score["metering"]["arms"]["sqlite"]["dispatch_started_calls"] == 48
    assert score["metering"]["arms"]["full_history"]["dispatch_started_calls"] == 24
    assert (
        r.run(package, directory, approval, mode="offline-test", provider=synthetic_provider)
        == execution
    )
    assert counters["calls"] == 72
    r.atomic(directory / "score.json", score)
    result = {
        "status": "PASS_OFFLINE_AUDIT_NOT_MODEL_QUALITY",
        "package_sha256": c.sha(package),
        "simulated_calls": 72,
        "rejected_extractions": 8,
        "invalid_answers": 1,
        "sqlite_stores_reopened": 24,
        "original_record_occurrences_preserved": originals,
        "empty_or_rejected_packet_pairs_equal_control": fallback_pairs,
        "all_semantic_reviews_unreviewed": True,
        "all_extraction_compute_charged_to_sqlite": True,
        "completed_resume_new_dispatches": 0,
        "network_calls": 0,
        "paid_calls": 0,
        "sarvam_accuracy": "NOT_MEASURED",
        "live_schema_support": "UNTESTED",
    }
    r.atomic(directory / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(verify(parser.parse_args().output_dir), indent=2))
