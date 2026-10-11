"""Replay saved public evidence without credentials, networking or model calls."""

import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


def no_network(event, args):
    if event.startswith("socket."):
        raise RuntimeError("This evidence replay must never access the network")


sys.addaudithook(no_network)

from benchmarks.sarvam_comparison_v2.contract import (  # noqa: E402
    eligible,
    loads,
    parse_facts,
    request,
    runtime_cases,
    sha,
)
from benchmarks.sarvam_comparison_v2.runner import preflight  # noqa: E402


def read(name):
    return loads((HERE / name).read_text())


def verify():
    diagnosis = read("diagnosis.json")
    req = read("request.json")
    case = next(c for c in runtime_cases() if c["id"] == "D01-en")
    package = loads((HERE.parent / "package.json").read_text())
    plan = preflight(package)
    assert plan["package_sha256"] == diagnosis["package_sha256"]
    assert req == request("extract", case) == package["jobs"][0]["request"]
    assert sha(req) == diagnosis["request_sha256"]
    prior = ROOT / "benchmarks/audits/sarvam-discovery-2026-10-10/thinking-off-probe"
    assert req == loads((prior / "request.json").read_text())
    assert req["reasoning_effort"] is None
    assert req["response_format"] == {"type": "json_object"}
    source_input = loads(req["messages"][1]["content"])
    assert set(source_input) == {"as_of", "records"}
    facts = loads(diagnosis["final_content"])["facts"]
    assert facts == diagnosis["unaccepted_facts_for_review_only"]
    try:
        parse_facts(diagnosis["final_content"], case)
    except ValueError as exc:
        assert "Invalid isoformat string" in str(exc)
    else:
        raise AssertionError("The saved extraction must fail the unchanged parser")
    errors = []
    for i, fact in enumerate(facts):
        for key in ("valid_from", "valid_until"):
            if fact[key] is not None:
                try:
                    assert date.fromisoformat(fact[key]).isoformat() == fact[key]
                except (ValueError, AssertionError):
                    errors.append((i, key, fact[key]))
    assert errors == [(4, "valid_from", source_input["records"][0]["observed_at"])]
    mapping = {
        "user_id": "customer",
        "tenant_id": "tenant",
        "session_id": "session",
        "user_role": "role",
        "observed_at": "observed_at",
        "instruction": "text",
    }
    assert len(facts) == len(mapping)
    assert {f["attribute"] for f in facts} == set(mapping)
    source = eligible(case)[0]
    for fact in facts:
        assert fact["value"] == source[mapping[fact["attribute"]]]
        assert fact["source_ids"] == [source["id"]]
    usage = diagnosis["usage"]
    assert usage["prompt_tokens"] == 366 and usage["completion_tokens"] == 293
    assert usage["total_tokens"] == 659
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 0
    current = (Decimal(366) * 15 + Decimal(293) * 60) / 1_000_000
    frozen = (Decimal(366) * 29280 + Decimal(293) * 73200) / 1_000_000_000
    assert current == Decimal(str(diagnosis["current_rate_usage_estimate_inr"]))
    assert frozen == Decimal(str(diagnosis["frozen_rate_usage_estimate_inr"]))
    score = read("score.json")
    assert score["execution"] == diagnosis["execution"]
    assert score["execution"]["attempted"] == 1
    assert score["execution"]["completed"] == 0
    assert diagnosis["unattempted_calls"] == 71 and diagnosis["retries"] == 0
    assert score["status"] == diagnosis["status"] == "INCOMPLETE_NO_HEADLINE_SCORE"
    for arm in score["arms"].values():
        assert arm["status_counts"] == {"not_attempted": 24}
        assert arm["accuracy_among_valid"] is None
    counterfactual = read("source-only-counterfactual.json")
    control = request("full_history", case)
    source_only = request("sqlite", case, [], eligible(case))
    assert control == source_only
    assert sha(control) == counterfactual["candidate_answer_request_sha256"]
    assert counterfactual["kept_originals"] == eligible(case)
    assert counterfactual["new_calls"] == 0
    return {
        "status": "PASS_SAVED_EVIDENCE_REPLAY",
        "network_calls": 0,
        "schema_rejection_reproduced": True,
        "exact_request_and_package_verified": True,
        "usage_costs_recomputed": True,
        "source_only_counterfactual_matches_control": True,
        "accuracy": "NOT_MEASURED",
        "provider_attestation": "NOT_INDEPENDENTLY_VERIFIED",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
