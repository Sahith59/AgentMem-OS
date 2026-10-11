"""Native tool execution followed by a matched, non-executable final-answer pair."""

import argparse
import copy
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.sarvam_native_study import backend
from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.transport import call, digest, payload, read_key, save

from .engine import FORMAT, PROFILES, ControlMeter, request_for

UNKNOWN_POLICY = """
For this task, if the records do not establish the requested value, answer must be JSON null
and needs_clarification must be true. Do not use the string "null" or an absence description.
Otherwise needs_clarification must be false. This explicit policy also applies to read-only tasks.
Keep the JSON keys and action_status enum values exactly as specified, regardless of language.
"""
FINAL_INSTRUCTION = """The execution phase is over. No more tools or actions are available.
Answer the original question using the original records and actual preceding tool results.
An earlier assistant claim is not an execution receipt. Do not claim a new action was confirmed
unless the actual tool result confirms it. An already-applied request can be established by an
original server receipt or the actual tool result.
Preserve current dates and original request scope.
Return only the required JSON object, with the answer as the bare requested value (not a sentence).
Use exactly the specified English field names and action_status values in every language.
When no value is established, use JSON null and needs_clarification=true.
"""


def initial(case, model):
    request = request_for(case, model, "control")
    request["messages"][0]["content"] += UNKNOWN_POLICY
    return request


def action_prefix(folder, spec, case, dispatch):
    """Gold-blind native action phase. Stop on a real outcome or a direct model reply."""
    folder.mkdir(parents=True)
    db = folder / "backend.sqlite"
    backend.create(db, case)
    request = initial(case, spec["model"])
    if digest(request) != spec["initial_sha256"]:
        raise ValueError("Frozen action request changed")
    result = {"calls": [], "tool_trace": [], "terminal": "read_only"}
    if request.get("tools"):
        result["terminal"] = "action_call_limit"
        try:
            for turn in range(3):
                job = spec["id"] + f"-act{turn}"
                result["calls"].append(job)
                body = dispatch(job, request)
                choice = body["choices"][0]
                message = choice["message"]
                if choice.get("finish_reason") not in ("stop", "tool_calls"):
                    result["terminal"] = "action_incomplete"
                    break
                toolcalls = message.get("tool_calls") or []
                if not toolcalls:
                    request["messages"].append(
                        {"role": "assistant", "content": message.get("content")}
                    )
                    result["terminal"] = "model_finished"
                    break
                if len(toolcalls) > 4:
                    result["terminal"] = "tool_limit"
                    break
                request["messages"].append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": toolcalls,
                    }
                )
                allowed = {t["function"]["name"] for t in request["tools"]}
                got_outcome = False
                for tc in toolcalls:
                    name = tc["function"]["name"]
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                    except (TypeError, ValueError):
                        arguments = None
                    reply = (
                        backend.execute(db, name, arguments)
                        if name in allowed
                        else {"error": "tool_not_available"}
                    )
                    result["tool_trace"].append(
                        {"name": name, "arguments": arguments, "response": reply}
                    )
                    request["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(reply, ensure_ascii=False),
                        }
                    )
                    got_outcome |= reply.get("status") in (
                        "confirmed",
                        "already_applied",
                        "rejected",
                    )
                if got_outcome:
                    result["terminal"] = "tool_outcome"
                    break
        except Exception as exc:
            result.update(terminal="infrastructure_error", error_class=type(exc).__name__)
    result["messages"] = request["messages"]
    result["final_backend"] = backend.state(db)
    save(folder / "prefix.json", result)
    return result


def final_request(prefix, model, variant):
    if variant not in ("control", "schema"):
        raise ValueError("Unknown final format")
    request = payload(
        model,
        copy.deepcopy(prefix["messages"]) + [{"role": "user", "content": FINAL_INSTRUCTION}],
        reasoning=PROFILES[model],
        maximum=8192,
    )
    if variant == "schema":
        request["response_format"] = copy.deepcopy(FORMAT)
    # No tool definitions or executable dispatch path exists in final generation.
    assert "tools" not in request and "tool_choice" not in request
    return request


def paired_episode(root, spec, case, dispatch):
    folder = root / "episodes" / spec["id"]
    prefix = action_prefix(folder, spec, case, dispatch)
    results = []
    for variant in spec["variant_order"]:
        result = {
            "id": spec["id"] + "-" + variant,
            "case_id": spec["case_id"],
            "model": spec["model"],
            "variant": variant,
            "shared_calls": prefix["calls"],
            "final_calls": [],
            "tool_trace": prefix["tool_trace"],
            "final_backend": prefix["final_backend"],
            "prefix_sha256": digest(prefix),
            "terminal": "prefix_failure",
        }
        if prefix["terminal"] in ("read_only", "tool_outcome", "model_finished"):
            req = final_request(prefix, spec["model"], variant)
            result["final_request_sha256"] = digest(req)
            job = result["id"] + "-final"
            result["final_calls"].append(job)
            try:
                body = dispatch(job, req)
                choice = body["choices"][0]
                message = choice["message"]
                result.update(
                    content=message.get("content"), finish_reason=choice.get("finish_reason")
                )
                if message.get("tool_calls"):
                    # Preserve the violation. Never execute a final-response tool request.
                    result.update(terminal="final_tool_request", tool_calls=message["tool_calls"])
                else:
                    result["terminal"] = (
                        "answer" if choice.get("finish_reason") == "stop" else "incomplete"
                    )
            except Exception as exc:
                result.update(terminal="infrastructure_error", error_class=type(exc).__name__)
        assert backend.state(folder / "backend.sqlite") == prefix["final_backend"]
        save(folder / (variant + ".json"), result)
        results.append(result)
    return {
        "id": spec["id"],
        "prefix_terminal": prefix["terminal"],
        "final_terminals": [r["terminal"] for r in results],
    }


def source_hashes():
    root = Path(__file__).parents[2]
    relatives = [
        "benchmarks/sarvam_native_controls/" + name
        for name in (
            "engine.py",
            "cases.py",
            "separated.py",
            "separated_cases.py",
            "separated_audit.py",
        )
    ] + [
        "benchmarks/sarvam_native_study/" + name
        for name in ("backend.py", "transport.py", "score.py", "runner.py", "paced.py")
    ]
    return {p: digest((root / p).read_text()) for p in relatives}


def prepare(root, stage):
    directory = root / stage
    directory.mkdir()
    bank = json.loads((root / (stage + "-runtime.json")).read_text())
    review = json.loads((root / (stage + "-review.json")).read_text())
    if digest(bank) != review["runtime_sha256"]:
        raise ValueError("Unreviewed bank change")
    rng = random.Random(202610115)
    jobs = []
    for case in bank:
        for i, model in enumerate(PROFILES):
            order = ["control", "schema"]
            rng.shuffle(order)
            jobs.append(
                {
                    "id": f"{stage}-m{i}-{case['id']}",
                    "case_id": case["id"],
                    "model": model,
                    "variant_order": order,
                    "initial_sha256": digest(initial(case, model)),
                    "maximum_calls": 5 if case["family"].startswith("action_") else 2,
                }
            )
    rng.shuffle(jobs)
    package = {
        "jobs": jobs,
        "runtime_sha256": digest(bank),
        "review_sha256": digest(review),
        "source_sha256": source_hashes(),
        "maximum_calls": sum(j["maximum_calls"] for j in jobs),
        "budget_sha256": digest(json.loads((root / "budget.json").read_text())),
        "protocol_sha256": digest((root / "PROTOCOL.md").read_text()),
        "no_retries": True,
        "workers": 4,
        "purpose": "SHARED_EXECUTION_FORMAT_MECHANISM_NOT_INDEPENDENT_ACTION_COMPARISON",
    }
    save(directory / "manifest.json", package)
    for relative in package["source_sha256"]:
        dest = directory / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x") as f:
            f.write((Path(__file__).parents[2] / relative).read_text())
    return package


def run(root, stage):
    directory = root / stage
    package = json.loads((directory / "manifest.json").read_text())
    bank = json.loads((root / (stage + "-runtime.json")).read_text())
    if digest(bank) != package["runtime_sha256"] or source_hashes() != package["source_sha256"]:
        raise ValueError("Frozen source or bank changed")
    for file, field in (
        ("budget.json", "budget_sha256"),
        (stage + "-review.json", "review_sha256"),
    ):
        if digest(json.loads((root / file).read_text())) != package[field]:
            raise ValueError("Frozen configuration changed")
    if digest((root / "PROTOCOL.md").read_text()) != package["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    if stage == "comparison":
        gate = json.loads((root / "probe/gate.json").read_text())
        if not gate["pass"]:
            raise ValueError("Separated execution gate failed")
        for relative, expected in gate["evidence_sha256"].items():
            if digest((root / relative).read_text()) != expected:
                raise ValueError("Gate evidence changed")
        for relative, expected in gate["database_sha256"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                raise ValueError("Gate database changed")
    save(directory / "started.json", {"manifest_sha256": digest(package)})
    cases = {c["id"]: c for c in bank}
    meter, pacer, key = ControlMeter(root), Pacer(root), read_key()

    def dispatch(job, request):
        save(meter.root / (job + ".dispatch.json"), pacer.wait())
        return call(meter, job, stage, request, key)

    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(package["jobs"]), 8):
            batch = package["jobs"][offset : offset + 8]
            results = [
                pool.submit(paired_episode, directory, j, cases[j["case_id"]], dispatch)
                for j in batch
            ]
            results = [future.result() for future in results]
            outcomes.extend(results)
            errors = sum(
                r["prefix_terminal"] == "infrastructure_error"
                or "infrastructure_error" in r["final_terminals"]
                for r in results
            )
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "pairs": len(outcomes),
                        "scheduled": len(package["jobs"]),
                        "batch_errors": errors,
                        "meter": meter.summary(),
                    }
                ),
                flush=True,
            )
            if errors >= 2:
                break
    save(directory / "execution.json", {"outcomes": outcomes, "meter": meter.summary()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("root", type=Path)
    parser.add_argument("stage", choices=("probe", "comparison"))
    a = parser.parse_args()
    (prepare if a.command == "prepare" else run)(a.root, a.stage)
