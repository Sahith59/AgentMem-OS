"""Evidence-bound date admission: actual failure class and adversarial limits."""

import copy
import socket

import pytest

from benchmarks.sarvam_comparison_v5 import contract as old
from benchmarks.sarvam_comparison_v5 import runner as old_runner
from benchmarks.sarvam_comparison_v6 import contract as c
from benchmarks.sarvam_comparison_v6 import runner as r
from benchmarks.sarvam_comparison_v6.score import evaluate
from tests.test_sarvam_comparison_v4 import FakeProvider, approval


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(r, "SarvamProvider", forbidden)


def candidate(**changes):
    return c.canonical(
        {
            "facts": [
                {
                    "attribute": "service_desk",
                    "value": "A2",
                    "status": "current",
                    "condition": None,
                    "valid_from": "2026-10-09",
                    "valid_until": None,
                    "source_ids": ["s01"],
                    **changes,
                }
            ]
        }
    )


def test_observation_timestamp_is_not_a_fact_effective_date():
    case = c.runtime_cases()[0]
    text = candidate(attribute="parcel_pickup_counter", value="K7")
    assert old.admit_facts(text, case)["status"] == "admitted_structure_only"
    result = c.admit_facts(text, case)
    assert result["reason"] == "VALIDITY_DATE_WITHOUT_CITED_LITERAL_WITNESS"
    facts, originals = c.source_packet(case, result)
    assert facts == [] and originals == c.eligible(case)
    assert c.request("sqlite", case, facts, originals) == c.request("full_history", case)


def test_uncited_date_and_unverified_derived_endpoint_are_rejected():
    case = next(v for v in c.runtime_cases() if v["id"] == "D06-en")
    text = candidate(valid_from=None, valid_until="2026-10-20")
    assert c.admit_facts(text, case)["status"] == "rejected"  # s01 lacks date
    cited = candidate(valid_from=None, valid_until="2026-10-20", source_ids=["s01", "s02"])
    assert c.admit_facts(cited, case)["status"] == "admitted_structure_only"
    derived = candidate(valid_from=None, valid_until="2026-10-19", source_ids=["s01", "s02"])
    assert c.admit_facts(derived, case)["status"] == "rejected"


@pytest.mark.parametrize(
    "prefix,suffix", [("x", ""), ("", "0"), ("", "T10:00:00"), ("-", ""), ("", "-01")]
)
def test_longer_identifier_or_timestamp_cannot_supply_date(prefix, suffix):
    case = copy.deepcopy(c.runtime_cases()[0])
    case["records"][0]["text"] = prefix + "2026-10-09" + suffix
    assert c.admit_facts(candidate(), case)["status"] == "rejected"


@pytest.mark.parametrize(
    "language_text",
    [
        "From 2026-10-09 use A2.",
        "2026-10-09 से A2 का उपयोग करें।",
        "2026-10-09 నుంచి A2 వాడండి.",
    ],
)
def test_explicit_iso_witness_allowed_across_source_scripts(language_text):
    case = copy.deepcopy(c.runtime_cases()[0])
    case["records"][0]["text"] = language_text
    assert c.admit_facts(candidate(), case)["date_witness_check"] == "LITERAL_ONLY_NOT_ENTAILMENT"


def test_present_date_does_not_prove_meaning_and_is_not_marked_reviewed():
    case = copy.deepcopy(c.runtime_cases()[0])
    case["records"][0]["text"] = "A different person's appointment is 2026-10-09."
    result = c.admit_facts(candidate(value="FABRICATED"), case)
    assert result["status"] == "admitted_structure_only"
    assert result["semantic_review"] == "UNREVIEWED"


def test_observed_false_conditional_approval_remains_an_explicit_limit():
    case = next(v for v in c.runtime_cases() if v["id"] == "D07-hi")
    # Exact failure form in campaign004: the date occurs, but the original text
    # makes approval a precondition. It does not establish an approval event.
    text = candidate(
        attribute="repair_approval_status",
        value="approved",
        status="requested",
        condition="repair_approval",
        valid_from="2026-10-18",
    )
    result = c.admit_facts(text, case)
    assert result["status"] == "admitted_structure_only"
    assert result["semantic_review"] == "UNREVIEWED"


def test_scope_and_whole_set_rejection():
    case = copy.deepcopy(c.runtime_cases()[0])
    extra = {**case["records"][0], "id": "s02", "text": "Starts 2026-10-09", "customer": "OTHER"}
    case["records"].append(extra)
    assert c.admit_facts(candidate(source_ids=["s02"]), case)["status"] == "rejected"
    rows = c.loads(candidate(valid_from=None))["facts"] + c.loads(candidate())["facts"]
    assert c.admit_facts(c.canonical({"facts": rows}), case)["facts"] == []


def test_requests_models_labels_and_schedules_unchanged():
    before, after = old_runner.build_package(), r.build_package()
    assert after["parent_package_sha256"] == c.sha(before)
    assert after["jobs"] == before["jobs"]
    assert after["settings"] == before["settings"]
    assert after["fixtures_sha256"] == before["fixtures_sha256"]
    assert c.request is old.request


def test_fallback_survives_persistence_and_charges_extraction(tmp_path):
    package = r.build_package()
    a = approval(package, tmp_path)
    fake = FakeProvider(tmp_path)

    def provider(req):
        result = fake(req)
        if "fields" not in c.loads(req["messages"][1]["content"]):
            body = c.loads(result["body_text"])
            body["choices"][0]["message"]["content"] = candidate()
            result["body_text"] = c.canonical(body)
        return result

    r.run(package, tmp_path, a, mode="offline-test", provider=provider)
    score = evaluate(package, tmp_path)
    assert score["status"] == "SIMULATED_TEST_ONLY"
    assert score["execution"]["rejected_extractions"] == 24
    assert len(list(tmp_path.glob("*.sqlite"))) == 24
    assert (
        score["arms"]["sqlite"]["by_evidence_path"]["originals_after_rejection"]["intended"] == 24
    )
    assert score["metering"]["arms"]["sqlite"]["dispatch_started_calls"] == 48
    assert fake.calls == 72
