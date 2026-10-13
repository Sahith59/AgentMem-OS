"""Opt-in, offline capped-actor reporting handoff. No transport or action executor.

A caller owns/authenticates the saved HTTP receipt, tool trace, scope and backend snapshots.
Hashes below link caller-provided objects; they do not authenticate an external server.
This helper changes no frozen run and makes no claim that reporter accuracy improves.
The caller must enforce its real spend cap. The required claim file prevents reuse
of one plan across process restarts; the caller must supply stable writable storage.
"""

import copy
import hashlib
import json
import os
from pathlib import Path

from benchmarks import sarvam_evidence_views as views
from benchmarks import sarvam_ordinary_v2 as ordinary
from benchmarks.sarvam_native_controls.engine import PROFILES
from benchmarks.sarvam_native_controls.transcript_finalizer import classify
from benchmarks.sarvam_native_study.transport import canonical, digest
from benchmarks.sarvam_receipt_reporting import project, validate_confirmation, validate_request
from benchmarks.sarvam_reporter_wording import request_for


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _strict_object(text, name):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(text, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {name}") from exc
    _require(isinstance(result, dict), f"{name} must be object")
    return result


def _context(request, tool_trace, initial_backend, final_backend, caller_scope):
    _require(isinstance(request, dict), "Actor request required")
    model = request.get("model")
    _require(model in PROFILES, "Unknown model/profile")
    expected = {
        "model": model,
        "temperature": 0,
        "reasoning_effort": PROFILES[model],
        "max_tokens": 8192,
        "stream": False,
        "n": 1,
        "tool_choice": "auto",
    }
    _require(all(request.get(k) == v for k, v in expected.items()), "Actor profile changed")
    _require(set(request) == set(expected) | {"messages", "tools"}, "Actor request shape changed")
    messages, tools = request["messages"], request["tools"]
    _require(isinstance(messages, list) and len(messages) >= 2, "Missing original history")
    _require(isinstance(tools, list) and bool(tools), "Missing native tool definitions")
    _require(
        messages[0].get("role") == "system" and messages[1].get("role") == "user",
        "Missing protocol context",
    )
    _require(isinstance(messages[1].get("content"), str), "Missing app evidence")
    supplied = _strict_object(messages[1]["content"], "app evidence")
    _require(isinstance(supplied, dict), "Invalid app evidence")
    _require(supplied.get("intent") in ("read", "change"), "Trusted intent required")
    _require(isinstance(supplied.get("appointment_id"), str), "Scoped appointment required")
    app = supplied.get("application_evidence")
    _require(
        isinstance(app, dict) and isinstance(app.get("prior_execution_receipts"), list),
        "Trusted prior receipts required",
    )
    _require(isinstance(tool_trace, list), "Executed tool trace required")
    _require(
        isinstance(caller_scope, dict)
        and set(caller_scope) == {"tenant", "customer"}
        and all(isinstance(v, str) and v for v in caller_scope.values()),
        "Trusted caller scope required",
    )
    _require(
        isinstance(initial_backend, dict) and initial_backend == final_backend,
        "Backend state uncertain or changed",
    )
    scoped = [
        a
        for a in final_backend.get("appointments", [])
        if a.get("id") == supplied["appointment_id"]
    ]
    _require(len(scoped) == 1, "Scoped backend state missing")
    current = scoped[0]
    _require(
        all(current.get(k) == caller_scope[k] for k in ("tenant", "customer")),
        "Backend/caller scope mismatch",
    )
    _require(
        app.get("current")
        == {
            "date": current["date"],
            "version": current["version"],
            "closed": bool(current["closed"]),
        },
        "App/backend current mismatch",
    )
    if supplied["intent"] == "read":
        _require(supplied.get("requested_change") is None, "Read intent has change")
    else:
        validate_request(supplied.get("requested_change"))
    for prior in app["prior_execution_receipts"]:
        _require(
            isinstance(prior, dict) and "arguments" in prior and "receipt" in prior,
            "Malformed prior receipt",
        )
        validate_request(prior["arguments"])
        validate_confirmation(prior["receipt"], prior["arguments"])
    # Reuse the existing lossless history validator; it rejects pending/orphan receipts.
    ordinary.final_request(copy.deepcopy(request))
    calls = []
    for message in messages[2:]:
        role = message.get("role")
        if role == "assistant":
            tc = message.get("tool_calls")
            _require(isinstance(tc, list) and bool(tc), "Unexpected prior actor answer")
            calls.extend(tc)
        elif role != "tool":
            raise ValueError("Unexpected history role")
    _require(len(calls) == len(tool_trace), "Tool trace/history coverage mismatch")
    receipt_messages = [m for m in messages[2:] if m.get("role") == "tool"]
    _require(len(receipt_messages) == len(calls), "Missing tool receipt")
    for tc, message, event in zip(calls, receipt_messages, tool_trace):
        _require(
            tc["id"] == message.get("tool_call_id") == event.get("call_id"),
            "Tool receipt ID mismatch",
        )
        name = tc["function"]["name"]
        _require(
            name == event.get("name") == "read_current", "Prior action or unknown tool is unsafe"
        )
        args = _strict_object(tc["function"]["arguments"], "tool arguments")
        _require(
            args == event.get("arguments") == {"appointment_id": supplied["appointment_id"]},
            "Tool scope mismatch",
        )
        response = event.get("response")
        _require(isinstance(response, dict) and "status" not in response, "Uncertain backend read")
        _require(message.get("content") == canonical(response), "Tool receipt content mismatch")
        _require(
            response
            == {
                "date": current["date"],
                "version": current["version"],
                "closed": bool(current["closed"]),
            },
            "Read receipt/backend mismatch",
        )
    return supplied


def prepare_plan(
    *,
    actor_job,
    actor_request,
    native_receipt,
    tool_trace,
    initial_backend,
    final_backend,
    actor_calls_used,
    caller_scope,
):
    """Return two no-tools requests only for a known, call-free length completion.

    The failed native message is retained in ``actor_failure`` and is never inserted
    into either reporting transcript. ``actor_calls_used`` counts this failed call.
    """
    _require(
        isinstance(actor_job, str)
        and actor_job
        and not actor_job.endswith(("-control", "-candidate")),
        "Invalid actor job",
    )
    _require(
        type(actor_calls_used) is int and 1 <= actor_calls_used <= 3, "Actor call cap exceeded"
    )
    prior_actor_calls = sum(
        message.get("role") == "assistant" for message in actor_request.get("messages", [])[2:]
    )
    _require(
        actor_calls_used == prior_actor_calls + 1 and actor_job.endswith(f"-a{prior_actor_calls}"),
        "Actor call count/job mismatch",
    )
    supplied = _context(actor_request, tool_trace, initial_backend, final_backend, caller_scope)
    _require(isinstance(native_receipt, dict), "Native HTTP receipt required")
    _require(native_receipt.get("job") == actor_job, "Actor job/receipt mismatch")
    _require(
        native_receipt.get("request_sha256") == digest(actor_request),
        "Actor request/receipt mismatch",
    )
    _require(native_receipt.get("status") == 200, "Unknown or failed HTTP outcome")
    body_text = native_receipt.get("body_text")
    _require(isinstance(body_text, str), "Missing native body")
    body = _strict_object(body_text, "native body")
    _require(
        isinstance(body, dict) and body.get("model") == actor_request["model"],
        "Native model mismatch",
    )
    choices = body.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1, "Native choice count unknown")
    choice = choices[0]
    _require(
        isinstance(choice, dict)
        and choice.get("index") == 0
        and choice.get("finish_reason") == "length",
        "Not a known length cap",
    )
    message = choice.get("message")
    _require(
        isinstance(message, dict) and message.get("role") == "assistant", "Invalid capped message"
    )
    _require(
        message.get("tool_calls") is None or message.get("tool_calls") == [],
        "Capped response contains tool calls",
    )
    _require(
        not any(message.get(k) for k in ("function_call", "tool_call", "parallel_tool_calls")),
        "Capped response contains alternate action call",
    )
    usage = body.get("usage")
    _require(isinstance(usage, dict), "Native usage unknown")
    completion, prompt, total = (
        usage.get("completion_tokens"),
        usage.get("prompt_tokens"),
        usage.get("total_tokens"),
    )
    _require(
        all(type(v) is int and v >= 0 for v in (completion, prompt, total)), "Native usage unknown"
    )
    _require(
        completion == actor_request["max_tokens"] and prompt + completion == total,
        "Length cap/usage mismatch",
    )
    final = ordinary.final_request(copy.deepcopy(actor_request))
    # This validator rejects unknown/gold user fields, duplicate JSON keys, scope
    # leaks, missing receipts and non-authoritative actor claims promoted as tools.
    views.from_final_request(final, tool_trace)
    requests = {variant: request_for(final, variant) for variant in ("control", "candidate")}
    _require(
        all("tools" not in req and "tool_choice" not in req for req in requests.values()),
        "Executable reporter tools",
    )
    evidence = [json.loads(req["messages"][1]["content"]) for req in requests.values()]
    _require(
        evidence[0]
        == evidence[1]
        == {
            "original_messages": actor_request["messages"],
            "original_tool_definitions": actor_request["tools"],
        },
        "Unequal or lossy reporter evidence",
    )
    plan = {
        "kind": "OPT_IN_CAPPED_ACTOR_REPORTING_PLAN_NOT_ORIGINAL_RUN",
        "actor_terminal": "truncated_no_tool_calls",
        "actor_job": actor_job,
        "actor_calls_used": actor_calls_used,
        "actor_request_sha256": digest(actor_request),
        "actor_failure": {
            "http_status": 200,
            "body_text": body_text,
            "body_text_sha256": hashlib.sha256(body_text.encode()).hexdigest(),
        },
        "actual_tool_trace": copy.deepcopy(tool_trace),
        "caller_scope": copy.deepcopy(caller_scope),
        "projection_inputs": {
            "intent": supplied["intent"],
            "requested_change": copy.deepcopy(supplied.get("requested_change")),
            "prior_receipts": copy.deepcopy(
                supplied["application_evidence"]["prior_execution_receipts"]
            ),
            "tool_trace": copy.deepcopy(tool_trace),
        },
        "reports": {
            variant: {
                "job": actor_job + "-handoff-" + variant,
                "request": req,
                "request_sha256": digest(req),
            }
            for variant, req in requests.items()
        },
        "maximum_total_calls": 5,
        "planned_total_calls": actor_calls_used + 2,
    }
    plan["plan_sha256"] = digest(plan)
    return plan


def _check_plan(plan):
    """Detect mutation between preparation and dispatch; caller authenticates inputs."""
    _require(
        isinstance(plan, dict)
        and plan.get("kind") == "OPT_IN_CAPPED_ACTOR_REPORTING_PLAN_NOT_ORIGINAL_RUN",
        "Invalid handoff plan",
    )
    _require(
        plan.get("plan_sha256") == digest({k: v for k, v in plan.items() if k != "plan_sha256"}),
        "Handoff plan changed",
    )
    requests = [plan["reports"][variant] for variant in ("control", "candidate")]
    for report in requests:
        request = report["request"]
        _require(digest(request) == report["request_sha256"], "Reporter request changed")
        _require(
            "tools" not in request and "tool_choice" not in request, "Executable reporter tools"
        )
    evidence = [json.loads(report["request"]["messages"][1]["content"]) for report in requests]
    _require(evidence[0] == evidence[1], "Unequal reporter evidence")
    original = evidence[0]["original_messages"]
    supplied = json.loads(original[1]["content"])
    projection = plan["projection_inputs"]
    _require(
        projection
        == {
            "intent": supplied["intent"],
            "requested_change": supplied["requested_change"],
            "prior_receipts": supplied["application_evidence"]["prior_execution_receipts"],
            "tool_trace": plan["actual_tool_trace"],
        },
        "Projection/source binding changed",
    )


def dispatch_once(plan, dispatch, used_job_ids, claim_path):
    """Atomically claim this plan before dispatch; caller enforces spend budget."""
    _check_plan(plan)
    _require(
        callable(dispatch) and isinstance(used_job_ids, set), "Dispatcher and call ledger required"
    )
    jobs = [plan["reports"][v]["job"] for v in ("control", "candidate")]
    _require(
        len(set(jobs)) == 2 and not any(j in used_job_ids for j in jobs), "Duplicate reporter job"
    )
    claim = Path(claim_path)
    _require(claim.is_absolute() and claim.parent.is_dir(), "Absolute claim path required")
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("Durable handoff claim already exists") from exc
    # Never remove a claim after an error: a partial or uncertain dispatch must
    # remain unavailable to automatic retry, including after a process restart.
    with os.fdopen(fd, "w", encoding="utf-8") as marker:
        marker.write(canonical({"plan_sha256": plan["plan_sha256"], "jobs": jobs}) + "\n")
        marker.flush()
        os.fsync(marker.fileno())
    directory_fd = os.open(claim.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    used_job_ids.update(jobs)
    reports = {}
    for variant, job in zip(("control", "candidate"), jobs):
        _check_plan(plan)
        try:
            body = dispatch(job, copy.deepcopy(plan["reports"][variant]["request"]))
            result = classify(body)
        except Exception as exc:
            result = {
                "terminal": "infrastructure_error",
                "error_class": type(exc).__name__,
                "parsed": None,
            }
        result["application"] = None
        if result.get("parsed") is not None:
            try:
                result["application"] = project(
                    **plan["projection_inputs"], raw_content=result["content"], completed=True
                )
            except (ValueError, KeyError, TypeError) as exc:
                result["application_error"] = type(exc).__name__
        reports[variant] = result
    return {
        "kind": "OPT_IN_CAPPED_ACTOR_REPORTING_RESULT_NOT_ORIGINAL_RUN",
        "actor_terminal": plan["actor_terminal"],
        "actor_failure": copy.deepcopy(plan["actor_failure"]),
        "reports": reports,
        "report_jobs": jobs,
        "claim_path": str(claim),
        "no_action_retry": True,
    }
