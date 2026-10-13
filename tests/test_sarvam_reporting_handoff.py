"""Offline tests for the opt-in capped actor handoff; no provider or key access."""

import copy
import json

import pytest

from benchmarks.sarvam_native_study.transport import canonical, digest, payload
from benchmarks.sarvam_reporting_handoff import dispatch_once, prepare_plan

# Body from the 2026-10-13 13-en-m0 HTTP-200, 8192-token native action cap.
# The surrounding actor request is a portable, local stand-in with the same profile.
LIVE_CAP_BODY = (
    '{"id":"20261013_79496396-3427-463e-94fc-cb7f4b2a4437",'
    '"choices":[{"finish_reason":"length","index":0,"logprobs":null,'
    '"message":{"content":null,"refusal":null,"reasoning_content":null,'
    '"role":"assistant","tool_calls":null}}],"created":1791908684,'
    '"model":"sarvam-105b","object":"chat.completion","service_tier":null,'
    '"system_fingerprint":"vllm-0.29.1rc1.dev145+gfe61f3317-tp2-1c00f7c7",'
    '"usage":{"completion_tokens":8192,"prompt_tokens":531,"total_tokens":8723,'
    '"completion_tokens_details":{"accepted_prediction_tokens":null,'
    '"audio_tokens":null,"reasoning_tokens":8191,"rejected_prediction_tokens":null},'
    '"prompt_tokens_details":null}}'
)


def fixture(intent="read", prior=None, requested_change=None):
    current = {"date": "2027-03-14", "version": 5, "closed": False}
    supplied = {
        "appointment_id": "VISIT-2113",
        "intent": intent,
        "question": (
            "Which conditional date applies? Read only."
            if intent == "read"
            else "What is actually confirmed?"
        ),
        "source_messages": [
            {
                "id": "note-13-0",
                "role": "customer",
                "content": "Morning availability has not been checked.",
            }
        ],
        "requested_change": requested_change,
        "application_evidence": {
            "approvals": [],
            "current": current,
            "prior_execution_receipts": prior or [],
            "server_time": 1791748800,
            "trusted_issuers": ["reviewer"],
        },
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_current",
                "description": "Read current booking",
                "parameters": {
                    "type": "object",
                    "properties": {"appointment_id": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "update_appointment",
                "description": "Change booking",
                "parameters": {"type": "object"},
            },
        },
    ]
    request = payload(
        "sarvam-105b",
        [
            {"role": "system", "content": "Synthetic service protocol"},
            {"role": "user", "content": canonical(supplied)},
        ],
        tools,
        reasoning="low",
        maximum=8192,
    )
    backend = {
        "appointments": [
            {
                "tenant": "fresh-confirmation",
                "customer": "person-13",
                "id": "VISIT-2113",
                "date": current["date"],
                "version": 5,
                "closed": 0,
            }
        ],
        "operations": [],
    }
    job = "confirm-fresh-13-en-m0-a0"
    receipt = {
        "job": job,
        "request_sha256": digest(request),
        "status": 200,
        "body_text": LIVE_CAP_BODY,
    }
    return {
        "actor_job": job,
        "actor_request": request,
        "native_receipt": receipt,
        "tool_trace": [],
        "initial_backend": backend,
        "final_backend": copy.deepcopy(backend),
        "actor_calls_used": 1,
        "caller_scope": {"tenant": "fresh-confirmation", "customer": "person-13"},
    }


def reply(answer, status="confirmed", clarification=False):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "tool_calls": None,
                    "content": json.dumps(
                        {
                            "answer": answer,
                            "needs_clarification": clarification,
                            "action_status": status,
                        }
                    ),
                },
            }
        ]
    }


def bind_second_turn(inputs):
    inputs["actor_calls_used"] = 2
    inputs["actor_job"] = "confirm-fresh-13-en-m0-a1"
    inputs["native_receipt"]["job"] = inputs["actor_job"]
    inputs["native_receipt"]["request_sha256"] = digest(inputs["actor_request"])


def test_live_cap_response_hands_off_losslessly_and_never_repairs_answer(tmp_path):
    inputs = fixture()
    plan = prepare_plan(**inputs)
    assert plan["actor_terminal"] == "truncated_no_tool_calls"
    assert plan["actor_failure"]["body_text"] == LIVE_CAP_BODY
    assert plan["actor_calls_used"] == 1 and plan["maximum_total_calls"] == 5
    requests = [plan["reports"][v]["request"] for v in ("control", "candidate")]
    assert all("tools" not in q and "tool_choice" not in q for q in requests)
    evidence = [json.loads(q["messages"][1]["content"]) for q in requests]
    assert evidence[0] == evidence[1]
    assert evidence[0]["original_messages"] == inputs["actor_request"]["messages"]
    assert LIVE_CAP_BODY not in requests[0]["messages"][1]["content"]
    seen, dispatched = set(), []
    claim = tmp_path / "claim.json"

    def dispatch(job, request):
        assert claim.exists()  # durable claim precedes both possible external calls
        dispatched.append((job, request))
        return reply("2027-01-01")  # deliberately wrong source answer and raw action claim

    result = dispatch_once(plan, dispatch, seen, claim)
    assert len(dispatched) == 2 and len(seen) == 2
    assert json.loads(claim.read_text())["plan_sha256"] == plan["plan_sha256"]
    assert result["no_action_retry"] is True
    for report in result["reports"].values():
        assert report["parsed"]["answer"] == "2027-01-01"
        assert report["parsed"]["action_status"] == "confirmed"
        assert report["application"]["application_output"]["answer"] == "2027-01-01"
        assert report["application"]["application_output"]["action_status"] == "not_requested"
    with pytest.raises(ValueError, match="Durable handoff claim already exists"):
        dispatch_once(plan, dispatch, set(), claim)
    assert len(dispatched) == 2


def test_previous_valid_scoped_read_receipt_is_kept_and_cross_linked():
    inputs = fixture()
    request = inputs["actor_request"]
    tc = {
        "id": "read-1",
        "type": "function",
        "function": {"name": "read_current", "arguments": '{"appointment_id":"VISIT-2113"}'},
    }
    response = {"date": "2027-03-14", "version": 5, "closed": False}
    request["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": "read-1", "content": canonical(response)},
        ]
    )
    inputs["tool_trace"] = [
        {
            "call_id": "read-1",
            "name": "read_current",
            "arguments": {"appointment_id": "VISIT-2113"},
            "response": response,
        }
    ]
    inputs["native_receipt"]["request_sha256"] = digest(request)
    inputs["actor_calls_used"] = 2
    inputs["actor_job"] = "confirm-fresh-13-en-m0-a1"
    inputs["native_receipt"]["job"] = inputs["actor_job"]
    plan = prepare_plan(**inputs)
    source = json.loads(plan["reports"]["control"]["request"]["messages"][1]["content"])
    assert source["original_messages"][-1]["content"] == canonical(response)
    assert plan["actual_tool_trace"] == inputs["tool_trace"]
    broken = copy.deepcopy(inputs)
    broken["tool_trace"][0]["response"]["date"] = "2027-09-09"
    with pytest.raises(ValueError, match="Tool receipt content mismatch"):
        prepare_plan(**broken)
    duplicate_args = copy.deepcopy(inputs)
    duplicate_args["actor_request"]["messages"][2]["tool_calls"][0]["function"][
        "arguments"
    ] = '{"appointment_id":"VISIT-2113","appointment_id":"VISIT-2113"}'
    duplicate_args["native_receipt"]["request_sha256"] = digest(duplicate_args["actor_request"])
    with pytest.raises(ValueError, match="Duplicate JSON key in tool arguments"):
        prepare_plan(**duplicate_args)


def test_matching_prior_receipt_projects_already_applied_without_new_action(tmp_path):
    change = {
        "appointment_id": "VISIT-2113",
        "date": "2027-03-14",
        "expected_version": 4,
        "approval_id": "AUTH-13",
        "idempotency_key": "REQUEST-13",
    }
    prior = [
        {
            "arguments": change,
            "receipt": {"status": "confirmed", "date": "2027-03-14", "version": 5},
        }
    ]
    inputs = fixture(intent="change", requested_change=change, prior=prior)
    plan = prepare_plan(**inputs)
    assert plan["actual_tool_trace"] == []
    result = dispatch_once(
        plan, lambda *_: reply("2027-03-14", "not_performed"), set(), tmp_path / "claim.json"
    )
    for report in result["reports"].values():
        assert report["application"]["application_output"]["action_status"] == "already_applied"
        assert report["parsed"]["action_status"] == "not_performed"


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda x: x["native_receipt"].update(status=502), "Unknown or failed HTTP"),
        (lambda x: x["native_receipt"].update(request_sha256="bad"), "request/receipt mismatch"),
        (lambda x: x["native_receipt"].update(body_text="{invalid"), "Invalid JSON in native body"),
        (
            lambda x: x["native_receipt"].update(
                body_text=json.dumps({**json.loads(LIVE_CAP_BODY), "usage": None})
            ),
            "Native usage unknown",
        ),
        (
            lambda x: x["native_receipt"].update(
                body_text=json.dumps({**json.loads(LIVE_CAP_BODY), "model": "other"})
            ),
            "Native model mismatch",
        ),
        (
            lambda x: x["native_receipt"].update(
                body_text=json.dumps(
                    {
                        **json.loads(LIVE_CAP_BODY),
                        "choices": [
                            {**json.loads(LIVE_CAP_BODY)["choices"][0], "finish_reason": "stop"}
                        ],
                    }
                )
            ),
            "Not a known length cap",
        ),
        (
            lambda x: x["native_receipt"].update(
                body_text=json.dumps(
                    {
                        **json.loads(LIVE_CAP_BODY),
                        "choices": [
                            {
                                **json.loads(LIVE_CAP_BODY)["choices"][0],
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [{"id": "maybe", "type": "function"}],
                                },
                            }
                        ],
                    }
                )
            ),
            "contains tool calls",
        ),
        (lambda x: x["actor_request"].update(reasoning_effort="high"), "Actor profile changed"),
        (lambda x: x.update(actor_calls_used=4), "Actor call cap exceeded"),
        (lambda x: x.update(actor_calls_used=2), "Actor call count/job mismatch"),
        (
            lambda x: x["final_backend"]["appointments"][0].update(date="2027-04-14"),
            "Backend state uncertain",
        ),
    ],
)
def test_unsafe_or_unbound_caps_are_rejected(mutation, reason):
    inputs = fixture()
    mutation(inputs)
    with pytest.raises(ValueError, match=reason):
        prepare_plan(**inputs)


def test_pending_or_action_receipts_and_backend_errors_fail_closed():
    inputs = fixture()
    tc = {
        "id": "read-1",
        "type": "function",
        "function": {"name": "read_current", "arguments": '{"appointment_id":"VISIT-2113"}'},
    }
    inputs["actor_request"]["messages"].append(
        {"role": "assistant", "content": None, "tool_calls": [tc]}
    )
    bind_second_turn(inputs)
    with pytest.raises(ValueError, match="Missing tool receipts"):
        prepare_plan(**inputs)
    inputs = fixture()
    tc["function"]["name"] = "update_appointment"
    response = {"status": "rejected", "reason": "approval_not_effective"}
    inputs["actor_request"]["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": "read-1", "content": canonical(response)},
        ]
    )
    inputs["tool_trace"] = [
        {
            "call_id": "read-1",
            "name": "update_appointment",
            "arguments": {"appointment_id": "VISIT-2113"},
            "response": response,
        }
    ]
    bind_second_turn(inputs)
    with pytest.raises(ValueError, match="Prior action or unknown tool"):
        prepare_plan(**inputs)
    inputs = fixture()
    tc["function"]["name"] = "read_current"
    inputs["actor_request"]["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [tc]},
            {"role": "tool", "tool_call_id": "read-1", "content": canonical({"status": "error"})},
        ]
    )
    inputs["tool_trace"] = [
        {
            "call_id": "read-1",
            "name": "read_current",
            "arguments": {"appointment_id": "VISIT-2113"},
            "response": {"status": "error"},
        }
    ]
    bind_second_turn(inputs)
    with pytest.raises(ValueError, match="Uncertain backend read"):
        prepare_plan(**inputs)


def test_dispatch_exception_counts_once_and_does_not_retry(tmp_path):
    plan = prepare_plan(**fixture())
    seen, count = set(), []

    def dispatch(job, _request):
        count.append(job)
        if job.endswith("control"):
            raise RuntimeError("offline transport failure")
        return reply(None, "not_requested", True)

    claim = tmp_path / "claim.json"
    result = dispatch_once(plan, dispatch, seen, claim)
    assert claim.exists()
    assert len(count) == 2 and len(seen) == 2
    assert result["reports"]["control"]["terminal"] == "infrastructure_error"
    assert result["reports"]["candidate"]["terminal"] == "answer"
    with pytest.raises(ValueError, match="Durable handoff claim already exists"):
        dispatch_once(plan, dispatch, set(), claim)
    assert len(count) == 2


def test_alternate_native_action_field_is_not_a_call_free_cap():
    inputs = fixture()
    body = json.loads(LIVE_CAP_BODY)
    body["choices"][0]["message"]["function_call"] = {"name": "update_appointment"}
    inputs["native_receipt"]["body_text"] = json.dumps(body)
    with pytest.raises(ValueError, match="alternate action call"):
        prepare_plan(**inputs)


def test_duplicate_json_and_unknown_gold_fields_are_rejected():
    inputs = fixture()
    body = LIVE_CAP_BODY.replace('"model":"sarvam-105b"', '"model":"x","model":"sarvam-105b"')
    inputs["native_receipt"]["body_text"] = body
    with pytest.raises(ValueError, match="Duplicate JSON key in native body"):
        prepare_plan(**inputs)

    inputs = fixture()
    user = inputs["actor_request"]["messages"][1]
    user["content"] = '{"question":"first","question":"second"}'
    inputs["native_receipt"]["request_sha256"] = digest(inputs["actor_request"])
    with pytest.raises(ValueError, match="Duplicate JSON key in app evidence"):
        prepare_plan(**inputs)

    inputs = fixture()
    user = inputs["actor_request"]["messages"][1]
    supplied = json.loads(user["content"])
    supplied["gold"] = {"answer": "2027-04-14"}
    user["content"] = canonical(supplied)
    inputs["native_receipt"]["request_sha256"] = digest(inputs["actor_request"])
    with pytest.raises(ValueError, match="Unexpected original input fields"):
        prepare_plan(**inputs)


def test_scope_and_unrelated_bad_prior_receipt_are_rejected():
    inputs = fixture()
    inputs["caller_scope"]["customer"] = "person-other"
    with pytest.raises(ValueError, match="Backend/caller scope mismatch"):
        prepare_plan(**inputs)

    change = {
        "appointment_id": "VISIT-2113",
        "date": "2027-04-14",
        "expected_version": 5,
        "approval_id": "AUTH-13",
        "idempotency_key": "REQUEST-13",
    }
    different = {**change, "date": "2027-02-14", "idempotency_key": "REQUEST-OLDER"}
    bad_prior = [
        {
            "arguments": different,
            "receipt": {"status": "confirmed", "date": "2027-02-14", "version": 99},
        }
    ]
    inputs = fixture(intent="change", requested_change=change, prior=bad_prior)
    with pytest.raises(ValueError, match="Receipt does not establish this change"):
        prepare_plan(**inputs)


def test_mutated_plan_cannot_dispatch_reporter_tools_or_change_projection(tmp_path):
    plan = prepare_plan(**fixture())
    plan["reports"]["candidate"]["request"]["tools"] = [{"type": "function"}]
    calls = []
    with pytest.raises(ValueError, match="Handoff plan changed"):
        dispatch_once(plan, lambda *args: calls.append(args), set(), tmp_path / "claim1.json")
    assert calls == []
    plan = prepare_plan(**fixture())
    plan["projection_inputs"]["intent"] = "change"
    with pytest.raises(ValueError, match="Handoff plan changed"):
        dispatch_once(plan, lambda *args: calls.append(args), set(), tmp_path / "claim2.json")
    assert calls == []
