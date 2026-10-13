"""Offline contract checks for the opt-in evidence layout. No model calls."""

import copy
import json

import pytest

from benchmarks import sarvam_evidence_views as views
from benchmarks.sarvam_native_study.transport import canonical, digest


def sample(*, with_receipt=False):
    booking = "BOOK-X"
    user = {
        "question": "From the stated condition, which date should I choose? Do not change it.",
        "intent": "read",
        "appointment_id": booking,
        "requested_change": None,
        "source_messages": [
            {
                "id": "utterance-α",
                "role": "customer",
                "content": "ఉదయం అందుబాటులో ఉంటే 2028-04-01, లేకపోతే 2028-05-01.",
            },
            {
                "id": "utterance-β",
                "role": "customer",
                "content": "Nobody has checked availability yet. Do not guess or change it.",
            },
        ],
        "application_evidence": {
            "current": {"date": "2028-03-01", "version": 3, "closed": False},
            "approvals": [],
            "trusted_issuers": ["service"],
            "prior_execution_receipts": [],
            "server_time": 123456,
        },
    }
    history = [
        {"role": "system", "content": "Actor instructions, preserved as data."},
        {"role": "user", "content": canonical(user)},
        {
            "role": "assistant",
            "content": "I confirmed 2028-04-01 and the morning slot is definitely available.",
            "tool_calls": None,
        },
    ]
    trace = []
    tools = []
    if with_receipt:
        call = {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "read_current",
                "arguments": canonical({"appointment_id": booking}),
            },
        }
        history[2]["tool_calls"] = [call]
        history.append(
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": canonical(user["application_evidence"]["current"]),
            }
        )
        history.append(
            {"role": "assistant", "content": "The April branch is conditional.", "tool_calls": None}
        )
        tools = [{"type": "function", "function": {"name": "read_current"}}]
        trace = [
            {
                "call_id": "call-1",
                "name": "read_current",
                "arguments": {"appointment_id": booking},
                "response": user["application_evidence"]["current"],
            }
        ]
    evidence = {"original_messages": history, "original_tool_definitions": tools}
    final = {
        "model": "fixed-model",
        "temperature": 0,
        "response_format": {"type": "json_schema"},
        "messages": [
            {"role": "system", "content": "Report from the complete source transcript."},
            {"role": "user", "content": canonical(evidence)},
        ],
    }
    return final, trace, evidence


@pytest.mark.parametrize("with_receipt", [False, True])
def test_exact_coverage_order_and_request_profile(with_receipt):
    final, trace, evidence = sample(with_receipt=with_receipt)
    before = copy.deepcopy(final)
    transformed = views.request_for(final, trace)
    bundle = json.loads(transformed["messages"][1]["content"])
    sidecar = views.from_final_request(final, trace)
    assert final == before
    assert {k: v for k, v in transformed.items() if k != "messages"} == {
        k: v for k, v in final.items() if k != "messages"
    }
    assert transformed["messages"][0] == final["messages"][0]
    assert "audit" not in bundle
    assert sidecar["audit"] == evidence
    assert views.reconstruct(bundle) == evidence
    assert bundle["protocol_context"]["original_system_message"] == evidence["original_messages"][0]
    assert (
        bundle["protocol_context"]["original_tool_definitions"]
        == evidence["original_tool_definitions"]
    )
    assert bundle["provenance"]["original_evidence_sha256"] == digest(evidence)
    assert bundle["provenance"]["message_sha256_in_order"] == [
        digest(m) for m in evidence["original_messages"]
    ]
    assert bundle["provenance"]["message_roles_in_order"] == [
        m["role"] for m in evidence["original_messages"]
    ]
    assert (
        bundle["source_view"]["source_messages"]
        == json.loads(evidence["original_messages"][1]["content"])["source_messages"]
    )
    assert bundle["source_view"]["source_message_sha256"] == [
        digest(m) for m in bundle["source_view"]["source_messages"]
    ]
    assert [c["message_index"] for c in bundle["actor_claims"]] == ([2, 4] if with_receipt else [2])
    assert all(c["authority"] == "actor_claim_only" for c in bundle["actor_claims"])
    assert [r["message_index"] for r in bundle["execution_view"]["actual_tool_receipts"]] == (
        [3] if with_receipt else []
    )


def test_wrong_actor_assertion_is_not_upgraded_to_a_fact_or_receipt():
    final, trace, _ = sample()
    bundle = views.from_final_request(final, trace)
    assert "definitely available" in bundle["actor_claims"][0]["message"]["content"]
    assert "definitely available" not in canonical(bundle["source_view"])
    assert bundle["execution_view"]["actual_tool_receipts"] == []
    assert bundle["execution_view"]["application_evidence"]["current"]["date"] == "2028-03-01"
    # The transformer preserves the unresolved condition; it does not invent a branch answer.
    assert "answer" not in bundle and "availability" not in bundle["execution_view"]


def test_reasoning_fields_remain_in_separate_actor_claim():
    final, trace, evidence = sample()
    evidence["original_messages"][2]["reasoning_content"] = "Unsupported private reasoning."
    final["messages"][1]["content"] = canonical(evidence)
    grouped = json.loads(views.request_for(final, trace)["messages"][1]["content"])
    assert grouped["actor_claims"][0]["message"]["reasoning_content"] == (
        "Unsupported private reasoning."
    )
    assert views.reconstruct(grouped) == evidence


def test_change_request_and_confirmed_receipt_stay_in_execution_view():
    final, _, evidence = sample()
    user = json.loads(evidence["original_messages"][1]["content"])
    user["intent"] = "change"
    user["requested_change"] = {
        "appointment_id": "BOOK-X",
        "date": "2028-04-01",
        "expected_version": 3,
        "approval_id": "server-grant",
        "idempotency_key": "request-key",
    }
    user["application_evidence"]["approvals"] = [
        {"appointment_id": "BOOK-X", "id": "server-grant", "granted": True}
    ]
    evidence["original_messages"][1]["content"] = canonical(user)
    args = user["requested_change"]
    receipt = {"status": "confirmed", "date": "2028-04-01", "version": 4}
    evidence["original_messages"][2]["tool_calls"] = [
        {
            "id": "write-1",
            "type": "function",
            "function": {"name": "update_appointment", "arguments": canonical(args)},
        }
    ]
    evidence["original_messages"].append(
        {"role": "tool", "tool_call_id": "write-1", "content": canonical(receipt)}
    )
    evidence["original_tool_definitions"] = [
        {"type": "function", "function": {"name": "update_appointment"}}
    ]
    final["messages"][1]["content"] = canonical(evidence)
    trace = [
        {
            "call_id": "write-1",
            "name": "update_appointment",
            "arguments": args,
            "response": receipt,
        }
    ]
    grouped = json.loads(views.request_for(final, trace)["messages"][1]["content"])
    assert grouped["execution_view"]["requested_change"] == args
    assert grouped["execution_view"]["actual_tool_receipts"][0]["message"]["content"] == (
        canonical(receipt)
    )
    assert "confirmed" in grouped["actor_claims"][0]["message"]["content"]
    assert views.reconstruct(grouped) == evidence


@pytest.mark.parametrize("claim", ["available", "unavailable", "unknown"])
def test_condition_statement_changes_only_source_view(claim):
    final, trace, _ = sample(with_receipt=True)
    evidence = json.loads(final["messages"][1]["content"])
    source = json.loads(evidence["original_messages"][1]["content"])
    source["source_messages"][1]["content"] = f"The customer says availability is {claim}."
    evidence["original_messages"][1]["content"] = canonical(source)
    final["messages"][1]["content"] = canonical(evidence)
    bundle = views.from_final_request(final, trace)
    assert bundle["source_view"]["source_messages"][1]["content"].endswith(f"{claim}.")
    assert bundle["execution_view"]["application_evidence"]["current"]["date"] == "2028-03-01"
    assert (
        bundle["execution_view"]["actual_tool_receipts"][0]["message"]
        == evidence["original_messages"][3]
    )


@pytest.mark.parametrize(
    "failure", ["receipt_body", "orphan", "missing_trace", "extra_trace", "call_args"]
)
def test_actual_receipt_linkage_refuses_tampering(failure):
    final, trace, evidence = sample(with_receipt=True)
    if failure == "receipt_body":
        evidence["original_messages"][3]["content"] = canonical({"date": "forged"})
    elif failure == "orphan":
        evidence["original_messages"][3]["tool_call_id"] = "other-call"
    elif failure == "missing_trace":
        trace = []
    elif failure == "extra_trace":
        trace.append(copy.deepcopy(trace[0]))
    else:
        evidence["original_messages"][2]["tool_calls"][0]["function"]["arguments"] = canonical(
            {"appointment_id": "OTHER-BOOKING"}
        )
    final["messages"][1]["content"] = canonical(evidence)
    with pytest.raises(ValueError):
        views.from_final_request(final, trace)


@pytest.mark.parametrize(
    "failure",
    ["gold", "approval_scope", "prior_scope", "request_scope", "duplicate_source", "tools"],
)
def test_input_contract_refuses_label_and_scope_leaks(failure):
    final, trace, evidence = sample()
    user = json.loads(evidence["original_messages"][1]["content"])
    if failure == "gold":
        user["expected_answer"] = "2028-04-01"
    elif failure == "approval_scope":
        user["application_evidence"]["approvals"] = [{"appointment_id": "OTHER"}]
    elif failure == "prior_scope":
        user["application_evidence"]["prior_execution_receipts"] = [
            {"arguments": {"appointment_id": "OTHER"}, "receipt": {"status": "confirmed"}}
        ]
    elif failure == "request_scope":
        user["intent"] = "change"
        user["requested_change"] = {"appointment_id": "OTHER"}
    elif failure == "duplicate_source":
        user["source_messages"][1]["id"] = user["source_messages"][0]["id"]
    else:
        final["tools"] = []
    evidence["original_messages"][1]["content"] = canonical(user)
    final["messages"][1]["content"] = canonical(evidence)
    with pytest.raises(ValueError):
        views.from_final_request(final, trace)


@pytest.mark.parametrize("location", ["final", "user", "tool", "message_item"])
def test_duplicate_json_keys_and_invalid_message_items_refused(location):
    final, trace, evidence = sample(with_receipt=True)
    if location == "final":
        final["messages"][1][
            "content"
        ] = '{"original_messages":[],"original_messages":[],"original_tool_definitions":[]}'
    elif location == "user":
        evidence["original_messages"][1]["content"] = (
            evidence["original_messages"][1]["content"][:-1] + ',"question":"duplicate"}'
        )
    elif location == "tool":
        evidence["original_messages"][3]["content"] = '{"date":"a","date":"b"}'
    else:
        evidence["original_messages"][2] = "invalid message"
    if location != "final":
        final["messages"][1]["content"] = canonical(evidence)
    with pytest.raises(ValueError):
        views.from_final_request(final, trace)


def test_grouped_view_reconstruction_refuses_missing_actor_or_changed_hash():
    final, trace, _ = sample(with_receipt=True)
    grouped = json.loads(views.request_for(final, trace)["messages"][1]["content"])
    grouped["actor_claims"].pop()
    with pytest.raises(ValueError):
        views.reconstruct(grouped)
    grouped = json.loads(views.request_for(final, trace)["messages"][1]["content"])
    grouped["source_view"]["source_messages"][0]["content"] = "changed"
    with pytest.raises(ValueError):
        views.reconstruct(grouped)


def test_grouped_view_refuses_actor_receipt_role_migration_and_wrong_trace():
    final, trace, _ = sample(with_receipt=True)
    grouped = json.loads(views.request_for(final, trace)["messages"][1]["content"])
    actor = grouped["actor_claims"].pop(0)
    grouped["execution_view"]["actual_tool_receipts"].append(
        {"message_index": actor["message_index"], "message": actor["message"]}
    )
    with pytest.raises(ValueError, match="Non-tool message"):
        views.reconstruct(grouped)

    grouped = json.loads(views.request_for(final, trace)["messages"][1]["content"])
    receipt = grouped["execution_view"]["actual_tool_receipts"].pop()
    grouped["actor_claims"].append({**receipt, "authority": "actor_claim_only"})
    with pytest.raises(ValueError, match="Non-assistant message"):
        views.reconstruct(grouped)

    grouped = json.loads(views.request_for(final, trace)["messages"][1]["content"])
    altered_trace = copy.deepcopy(trace)
    altered_trace[0]["response"]["date"] = "forged"
    with pytest.raises(ValueError, match="Actual tool trace hash mismatch"):
        views.reconstruct(grouped, actual_tool_trace=altered_trace)
