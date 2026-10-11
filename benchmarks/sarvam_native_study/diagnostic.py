"""Prospective matched-cap reasoning diagnostic; immutable native study remains unchanged."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import backend
from .runner import initial
from .transport import MODELS, Meter, call, digest, read_key, save


def profile_request(case, model, arm, reasoning):
    request = initial(case, model, arm)
    request["reasoning_effort"] = reasoning
    request["max_tokens"] = 8192
    return request


def diagnostic_episode(root, meter, key, spec, case):
    folder = root / "episodes" / spec["id"]
    folder.mkdir(parents=True)
    db = folder / "backend.sqlite"
    backend.create(db, case)
    request = profile_request(case, spec["model"], spec["arm"], spec["reasoning"])
    if digest(request) != spec["initial_request_sha256"]:
        raise ValueError("Frozen initial request changed")
    result = {**spec, "calls": [], "tool_trace": [], "terminal": "call_limit"}
    try:
        for turn in range(4):
            jid = spec["id"] + f"-t{turn}"
            result["calls"].append(jid)
            body = call(meter, jid, "attribution", request, key)
            choice = body["choices"][0]
            message = choice["message"]
            toolcalls = message.get("tool_calls", [])
            if toolcalls:
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
                allowed = {t["function"]["name"] for t in request.get("tools", [])}
                for tc in toolcalls:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (ValueError, TypeError):
                        args = None
                    response = (
                        backend.execute(db, name, args)
                        if name in allowed
                        else {"error": "tool_not_available"}
                    )
                    result["tool_trace"].append(
                        {"name": name, "arguments": args, "response": response}
                    )
                    request["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(response, ensure_ascii=False),
                        }
                    )
                continue
            result["finish_reason"] = choice.get("finish_reason")
            result["content"] = message.get("content")
            result["terminal"] = (
                "answer" if choice.get("finish_reason") == "stop" else "incomplete_output"
            )
            break
    except Exception as exc:
        result["terminal"] = "infrastructure_error"
        result["error_class"] = type(exc).__name__
    result["final_backend"] = backend.state(db)
    save(folder / "result.json", result)
    return {"id": spec["id"], "terminal": result["terminal"]}


LEADS = [
    "arithmetic-04",
    "arithmetic-05",
    "effective_date-06",
    "effective_date-05",
    "action_update-00",
    "action_update-01",
]
CONTROL_FAMILIES = ["conditional", "backend_outcome", "identity", "distractors"]


def prepare(root, bank):
    root, bank = Path(root), Path(bank)
    directory = root / "attribution"
    directory.mkdir()
    cases = json.loads((bank / "runtime.json").read_text())
    base_ids = set(LEADS)
    controls = []
    for family in CONTROL_FAMILIES:
        choices = {
            c["base_id"] for c in cases if c["family"] == family and c["split"] == "discovery"
        }
        selected = min(choices, key=lambda x: digest("native-diagnostic-control-v1:" + x))
        controls.append(selected)
        base_ids.add(selected)
    selected = [c for c in cases if c["base_id"] in base_ids]
    jobs = []
    for case in selected:
        for m, model in enumerate(MODELS):
            for arm in ("full_history", "sqlite"):
                for reasoning in (None, "low"):
                    profile = "low" if reasoning else "null"
                    jobs.append(
                        {
                            "id": f"attribution-{profile}-m{m}-{case['id']}-{arm}",
                            "case_id": case["id"],
                            "model": model,
                            "arm": arm,
                            "reasoning": reasoning,
                            "initial_request_sha256": digest(
                                profile_request(case, model, arm, reasoning)
                            ),
                        }
                    )
    random.Random(20261012).shuffle(jobs)
    package = {
        "purpose": "TARGETED_DIAGNOSTIC_NOT_GENERAL_ACCURACY",
        "lead_and_contrast_bases": LEADS,
        "output_independent_control_bases": controls,
        "case_ids": [c["id"] for c in selected],
        "jobs": jobs,
        "maximum_calls": len(jobs) * 4,
        "workers": 4,
        "stage_ceiling_inr": 1000,
        "max_tokens": 8192,
        "reasoning_comparison": [None, "low"],
        "no_provider_retries": True,
        "bank_sha256": digest(cases),
        "code_sha256": {p.name: digest(p.read_text()) for p in Path(__file__).parent.glob("*.py")},
    }
    save(directory / "manifest.json", package)
    for p in Path(__file__).parent.glob("*.py"):
        (directory / (p.name + ".frozen")).write_bytes(p.read_bytes())
    print(
        json.dumps(
            {
                "diagnostic_cases": len(selected),
                "base_scenarios": len(base_ids),
                "episodes": len(jobs),
                "maximum_calls": len(jobs) * 4,
            }
        )
    )


def run(root, bank):
    root, bank = Path(root), Path(bank)
    directory = root / "attribution"
    package = json.loads((directory / "manifest.json").read_text())
    cases = json.loads((bank / "runtime.json").read_text())
    if digest(cases) != package["bank_sha256"]:
        raise ValueError("Frozen bank changed")
    for name, expected in package["code_sha256"].items():
        if digest((Path(__file__).parent / name).read_text()) != expected:
            raise ValueError("Frozen code changed")
    save(directory / "started.json", {"manifest_sha256": digest(package)})
    byid = {c["id"]: c for c in cases}
    meter, key = Meter(root / "calls"), read_key()
    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(package["jobs"]), 8):
            batch = package["jobs"][offset : offset + 8]
            futures = [
                pool.submit(diagnostic_episode, directory, meter, key, j, byid[j["case_id"]])
                for j in batch
            ]
            result = [f.result() for f in futures]
            outcomes.extend(result)
            errors = sum(r["terminal"] == "infrastructure_error" for r in result)
            print(
                json.dumps(
                    {
                        "episodes_done": len(outcomes),
                        "total": len(package["jobs"]),
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
        {"scheduled": len(package["jobs"]), "outcomes": outcomes, "meter": meter.summary()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("root")
    parser.add_argument("bank")
    args = parser.parse_args()
    (prepare if args.command == "prepare" else run)(args.root, args.bank)
