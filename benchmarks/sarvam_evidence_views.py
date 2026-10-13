"""Opt-in, lossless evidence layout for a later Sarvam reporting comparison.

This module only reorganizes an already completed ordinary episode. It does not
infer dates, decide permissions, execute tools, read labels, or repair answers.
The caller must supply the actual application-owned tool trace; model-authored
tool calls or prose alone cannot establish an execution receipt.
"""

import copy
import json

from benchmarks.sarvam_native_study.transport import canonical, digest

INPUT_KEYS = {
    "question",
    "intent",
    "appointment_id",
    "requested_change",
    "application_evidence",
    "source_messages",
}
APPLICATION_KEYS = {
    "current",
    "approvals",
    "trusted_issuers",
    "prior_execution_receipts",
    "server_time",
}


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _json_object(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be JSON text")

    def unique_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}: {key}")
            result[key] = item
        return result

    try:
        return _object(json.loads(value, object_pairs_hook=unique_keys), name)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON") from exc


def _source_envelope(messages):
    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or any(not isinstance(m, dict) for m in messages)
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
        or any(m.get("role") not in ("assistant", "tool") for m in messages[2:])
    ):
        raise ValueError("Unexpected original message roles")
    supplied = _json_object(messages[1].get("content"), "original user envelope")
    if messages[1]["content"] != canonical(supplied):
        raise ValueError("Original user envelope must be canonical for exact reconstruction")
    if set(supplied) != INPUT_KEYS:
        raise ValueError("Unexpected original input fields, including possible labels")
    if not isinstance(supplied["question"], str) or not supplied["question"]:
        raise ValueError("Question is missing")
    if supplied["intent"] not in ("read", "change"):
        raise ValueError("Unknown intent")
    booking = supplied["appointment_id"]
    if not isinstance(booking, str) or not booking:
        raise ValueError("Booking scope is missing")
    change = supplied["requested_change"]
    if (supplied["intent"] == "read" and change is not None) or (
        supplied["intent"] == "change"
        and (not isinstance(change, dict) or change.get("appointment_id") != booking)
    ):
        raise ValueError("Request/booking scope mismatch")
    source = supplied["source_messages"]
    if not isinstance(source, list):
        raise ValueError("Source messages must be a list")
    ids = set()
    for item in source:
        _object(item, "source message")
        if (
            not isinstance(item.get("id"), str)
            or not item["id"]
            or item["id"] in ids
            or not isinstance(item.get("role"), str)
            or not item["role"]
            or not isinstance(item.get("content"), str)
        ):
            raise ValueError("Invalid or duplicate source message")
        ids.add(item["id"])
    app = _object(supplied["application_evidence"], "application evidence")
    if set(app) != APPLICATION_KEYS or not isinstance(app["approvals"], list):
        raise ValueError("Unexpected application evidence fields")
    for approval in app["approvals"]:
        if _object(approval, "approval").get("appointment_id") != booking:
            raise ValueError("Approval belongs to another booking")
    if not isinstance(app["prior_execution_receipts"], list):
        raise ValueError("Prior receipts must be a list")
    for prior in app["prior_execution_receipts"]:
        args = _object(_object(prior, "prior receipt").get("arguments"), "prior arguments")
        if args.get("appointment_id") != booking:
            raise ValueError("Prior receipt belongs to another booking")
        _object(prior.get("receipt"), "prior execution result")
    return supplied


def _linked_trace(messages, tool_trace, tool_definitions):
    if not isinstance(tool_trace, list) or not isinstance(tool_definitions, list):
        raise ValueError("Actual trace and tool definitions must be lists")
    names = set()
    for definition in tool_definitions:
        function = _object(_object(definition, "tool definition").get("function"), "function")
        name = function.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Invalid or duplicate tool definition")
        names.add(name)
    pending = {}
    seen = set()
    receipts = []
    trace_at = 0
    for index, message in enumerate(messages[2:], 2):
        if message["role"] == "assistant":
            if pending:
                raise ValueError("Unanswered tool call")
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise ValueError("Malformed tool calls")
            for call in calls:
                call = _object(call, "tool call")
                function = _object(call.get("function"), "tool function")
                call_id = call.get("id")
                name = function.get("name")
                if (
                    not isinstance(call_id, str)
                    or not call_id
                    or call_id in seen
                    or call.get("type") != "function"
                    or name not in names
                ):
                    raise ValueError("Duplicate, missing or undeclared tool call")
                pending[call_id] = (name, _json_object(function.get("arguments"), "tool arguments"))
                seen.add(call_id)
        else:
            call_id = message.get("tool_call_id")
            if call_id not in pending or trace_at >= len(tool_trace):
                raise ValueError("Orphan or untrusted tool receipt")
            name, args = pending.pop(call_id)
            actual = _object(tool_trace[trace_at], "actual tool trace")
            response = _json_object(message.get("content"), "tool response")
            if (
                actual.get("call_id") != call_id
                or actual.get("name") != name
                or actual.get("arguments") != args
                or actual.get("response") != response
            ):
                raise ValueError("Transcript tool message differs from actual receipt")
            receipts.append({"message_index": index, "message": copy.deepcopy(message)})
            trace_at += 1
    if pending or trace_at != len(tool_trace):
        raise ValueError("Incomplete or extra actual tool trace")
    return receipts


def from_final_request(final_request, tool_trace):
    """Create two evidence views, while retaining the exact input transcript.

    ``audit`` is a local sidecar, excluded from the model request. The grouped
    view preserves each original message and tool definition once. Hashes and
    indexes verify reconstruction without promoting actor claims to receipts.
    """
    request = _object(final_request, "final request")
    messages = request.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or any(not isinstance(m, dict) for m in messages)
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
        or "tools" in request
    ):
        raise ValueError("Expected a no-tools final reporting request")
    original = _json_object(messages[1].get("content"), "final evidence")
    if set(original) != {"original_messages", "original_tool_definitions"}:
        raise ValueError("Unexpected final evidence fields")
    history = original["original_messages"]
    supplied = _source_envelope(history)
    receipts = _linked_trace(history, tool_trace, original["original_tool_definitions"])
    claims = [
        {"message_index": i, "message": copy.deepcopy(message), "authority": "actor_claim_only"}
        for i, message in enumerate(history[2:], 2)
        if message["role"] == "assistant"
    ]
    grouped = {
        "view_version": 1,
        "protocol_context": {
            "original_system_message": copy.deepcopy(history[0]),
            "original_tool_definitions": copy.deepcopy(original["original_tool_definitions"]),
        },
        "source_view": {
            "question": supplied["question"],
            "source_messages": copy.deepcopy(supplied["source_messages"]),
            "source_message_sha256": [digest(item) for item in supplied["source_messages"]],
        },
        "execution_view": {
            "intent": supplied["intent"],
            "appointment_id": supplied["appointment_id"],
            "requested_change": copy.deepcopy(supplied["requested_change"]),
            "application_evidence": copy.deepcopy(supplied["application_evidence"]),
            "actual_tool_receipts": receipts,
        },
        "actor_claims": claims,
        "provenance": {
            "message_roles_in_order": [m["role"] for m in history],
            "message_sha256_in_order": [digest(m) for m in history],
            "original_evidence_sha256": digest(original),
            "actual_tool_trace_sha256": digest(tool_trace),
        },
    }
    if reconstruct(grouped, actual_tool_trace=tool_trace) != original:
        raise ValueError("Grouped evidence did not reconstruct original transcript")
    return {**grouped, "audit": copy.deepcopy(original)}


def reconstruct(grouped, *, actual_tool_trace=None):
    """Reconstruct original evidence; optionally verify against the caller's trace."""
    if grouped.get("view_version") != 1:
        raise ValueError("Unknown view version")
    protocol = _object(grouped.get("protocol_context"), "protocol context")
    source = _object(grouped.get("source_view"), "source view")
    execution = _object(grouped.get("execution_view"), "execution view")
    provenance = _object(grouped.get("provenance"), "provenance")
    if source.get("source_message_sha256") != [
        digest(item) for item in source.get("source_messages", [])
    ]:
        raise ValueError("Source message hash mismatch")
    supplied = {
        "question": source["question"],
        "source_messages": source["source_messages"],
        "intent": execution["intent"],
        "appointment_id": execution["appointment_id"],
        "requested_change": execution["requested_change"],
        "application_evidence": execution["application_evidence"],
    }
    indexed = {
        0: copy.deepcopy(protocol["original_system_message"]),
        1: {"role": "user", "content": canonical(supplied)},
    }
    for claim in grouped.get("actor_claims", []):
        claim = _object(claim, "actor claim entry")
        if claim.get("authority") != "actor_claim_only":
            raise ValueError("Actor claim authority changed")
        if _object(claim.get("message"), "actor claim").get("role") != "assistant":
            raise ValueError("Non-assistant message in actor claims")
        index = claim.get("message_index")
        if type(index) is not int or index in indexed or index < 2:
            raise ValueError("Duplicate or invalid actor index")
        indexed[index] = copy.deepcopy(claim["message"])
    for receipt in execution["actual_tool_receipts"]:
        receipt = _object(receipt, "tool receipt entry")
        if _object(receipt.get("message"), "tool receipt").get("role") != "tool":
            raise ValueError("Non-tool message in actual receipts")
        index = receipt.get("message_index")
        if type(index) is not int or index in indexed or index < 2:
            raise ValueError("Duplicate or invalid receipt index")
        indexed[index] = copy.deepcopy(receipt["message"])
    if set(indexed) != set(range(len(indexed))):
        raise ValueError("Original message coverage gap")
    history = [indexed[i] for i in range(len(indexed))]
    _source_envelope(history)
    original = {
        "original_messages": history,
        "original_tool_definitions": copy.deepcopy(protocol["original_tool_definitions"]),
    }
    if (
        provenance.get("message_roles_in_order") != [m["role"] for m in history]
        or provenance.get("message_sha256_in_order") != [digest(m) for m in history]
        or provenance.get("original_evidence_sha256") != digest(original)
    ):
        raise ValueError("Original evidence provenance mismatch")
    if actual_tool_trace is not None:
        if provenance.get("actual_tool_trace_sha256") != digest(actual_tool_trace):
            raise ValueError("Actual tool trace hash mismatch")
        _linked_trace(history, actual_tool_trace, original["original_tool_definitions"])
    return original


def request_for(original_final_request, tool_trace):
    """Change only the final request's evidence layout; preserve its other fields."""
    result = copy.deepcopy(original_final_request)
    grouped = from_final_request(result, tool_trace)
    grouped.pop("audit")
    result["messages"][1]["content"] = canonical(grouped)
    return result
