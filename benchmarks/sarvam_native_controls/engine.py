"""One-lever response-format control with immutable requests and real SQLite tools."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmarks.sarvam_native_study import backend
from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.runner import SYSTEM
from benchmarks.sarvam_native_study.score import parse
from benchmarks.sarvam_native_study.transport import (
    INPUT,
    MODELS,
    OUTPUT,
    Meter,
    call,
    digest,
    payload,
    read_key,
    save,
)

FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "service_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": ["string", "null"]},
                "needs_clarification": {"type": "boolean"},
                "action_status": {
                    "type": "string",
                    "enum": [
                        "not_requested",
                        "not_performed",
                        "confirmed",
                        "rejected",
                        "already_applied",
                    ],
                },
            },
            "required": ["answer", "needs_clarification", "action_status"],
            "additionalProperties": False,
        },
    },
}
PROFILES = {"sarvam-105b": "low", "sarvam-105b-conversations": None}


class ControlMeter(Meter):
    """New immutable ledger, explicitly debiting prior research reservations."""

    def __init__(self, root):
        self.config = json.loads((Path(root) / "budget.json").read_text())
        super().__init__(Path(root) / "calls")

    def reserve(self, job, stage, request):
        cfg = self.config
        if stage not in cfg["stages"] or request["model"] not in MODELS:
            raise ValueError("Undeclared stage/model")
        if request["max_tokens"] != 8192:
            raise ValueError("Unfrozen output cap")
        amount = (MODELS[request["model"]] * INPUT + 8192 * OUTPUT) * 5 // 4
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            total = db.execute("SELECT COALESCE(SUM(charged),0) FROM calls").fetchone()[0]
            subtotal, count = db.execute(
                "SELECT COALESCE(SUM(charged),0),COUNT(*) FROM calls WHERE stage=?", (stage,)
            ).fetchone()
            limits = cfg["stages"][stage]
            if total + amount > cfg["phase_cap_microinr"]:
                raise ValueError("Phase budget exhausted")
            if total + amount + cfg["prior_retained_microinr"] > cfg["campaign_cap_microinr"]:
                raise ValueError("Research budget exhausted")
            if subtotal + amount > limits["cap_microinr"] or count >= limits["calls"]:
                raise ValueError("Stage budget/count exhausted")
            db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?)",
                (job, stage, amount, amount, None, "reserved"),
            )
        return amount


def request_for(case, model, variant):
    if variant not in ("control", "schema") or model not in PROFILES:
        raise ValueError("Unfrozen model/variant")
    tools = []
    if case.get("probe_read"):
        tools = [backend.READ_TOOL]
    elif case["family"].startswith("action_"):
        tools = [backend.UPDATE_TOOL]
    supplied = {"question": case["question"]}
    if case.get("probe_read"):
        supplied["records_available_via"] = "read_records"
    else:
        supplied["records"] = case["records"]
    req = payload(
        model,
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(supplied, ensure_ascii=False)},
        ],
        tools,
        reasoning=PROFILES[model],
        maximum=8192,
    )
    if variant == "schema":
        req["response_format"] = json.loads(json.dumps(FORMAT))
    return req


def episode(root, spec, case, dispatch):
    folder = root / "episodes" / spec["id"]
    folder.mkdir(parents=True)
    db = folder / "backend.sqlite"
    backend.create(db, case)
    req = request_for(case, spec["model"], spec["variant"])
    if digest(req) != spec["initial_sha256"]:
        raise ValueError("Frozen request changed")
    result = {**spec, "calls": [], "tool_trace": [], "terminal": "call_limit"}
    try:
        for turn in range(spec["maximum_calls"]):
            jid = spec["id"] + f"-t{turn}"
            result["calls"].append(jid)
            body = dispatch(jid, req)
            choice = body["choices"][0]
            message = choice["message"]
            toolcalls = message.get("tool_calls") or []
            if toolcalls:
                if len(toolcalls) > 4:
                    result["terminal"] = "tool_limit"
                    break
                req["messages"].append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": toolcalls,
                    }
                )
                allowed = {t["function"]["name"] for t in req.get("tools", [])}
                for tc in toolcalls:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (TypeError, ValueError):
                        args = None
                    reply = (
                        backend.execute(db, name, args)
                        if name in allowed
                        else {"error": "tool_not_available"}
                    )
                    result["tool_trace"].append(
                        {"name": name, "arguments": args, "response": reply}
                    )
                    req["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps(reply, ensure_ascii=False),
                        }
                    )
                continue
            result["content"] = message.get("content")
            result["finish_reason"] = choice.get("finish_reason")
            result["terminal"] = "answer" if choice.get("finish_reason") == "stop" else "incomplete"
            break
    except Exception as exc:
        result["terminal"] = "infrastructure_error"
        result["error_class"] = type(exc).__name__
    result["final_backend"] = backend.state(db)
    save(folder / "result.json", result)
    return result


def grade(case, label, result):
    if label["runtime_sha256"] != digest(case):
        raise ValueError("Label/input mismatch")
    parsed = None
    if result["terminal"] == "answer":
        try:
            parsed = parse(result.get("content"))
        except (ValueError, TypeError):
            pass
    mutations = [t for t in result["tool_trace"] if t["name"] == "update_appointment"]
    forbidden = label["forbid_mutation"] and bool(mutations)
    receipt_ok = not label["receipt_status"] or any(
        t["response"].get("status") == label["receipt_status"] for t in mutations
    )
    state_ok = result["final_backend"] == label["final_backend"]
    answer_ok = parsed == label["response"]
    return {
        "valid_output": parsed is not None,
        "answer_correct": answer_ok,
        "backend_correct": state_ok,
        "forbidden_mutation_attempt": forbidden,
        "required_receipt": receipt_ok,
        "task_success": bool(answer_ok and state_ok and not forbidden and receipt_ok),
        "parsed": parsed,
    }


def source_hashes():
    paths = [*Path(__file__).parent.glob("*.py")]
    paths += [
        Path(backend.__file__).parent / name
        for name in ("backend.py", "transport.py", "score.py", "runner.py", "paced.py")
    ]
    return {str(p.relative_to(Path(__file__).parents[2])): digest(p.read_text()) for p in paths}


def prepare(root, stage):
    directory = root / stage
    directory.mkdir()
    bank = json.loads((root / (stage + "-runtime.json")).read_text())
    review = json.loads((root / (stage + "-review.json")).read_text())
    if digest(bank) != review["runtime_sha256"]:
        raise ValueError("Unreviewed bank change")
    jobs = []
    for case in bank:
        for i, model in enumerate(PROFILES):
            for variant in ["schema"] if stage == "probe" else ["control", "schema"]:
                jobs.append(
                    {
                        "id": f"{stage}-m{i}-{case['id']}-{variant}",
                        "case_id": case["id"],
                        "model": model,
                        "variant": variant,
                        "maximum_calls": 2 if stage == "probe" else 4,
                        "initial_sha256": digest(request_for(case, model, variant)),
                    }
                )
    random.Random(202610113).shuffle(jobs)
    package = {
        "stage": stage,
        "jobs": jobs,
        "runtime_sha256": digest(bank),
        "review_sha256": digest(review),
        "source_sha256": source_hashes(),
        "budget_sha256": digest(json.loads((root / "budget.json").read_text())),
        "protocol_sha256": digest((root / "PROTOCOL.md").read_text()),
        "workers": 4,
        "no_retries": True,
        "stop_on_batch_infrastructure_errors": 2,
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
        check = json.loads((root / "probe/gate.json").read_text())
        if not check["pass"]:
            raise ValueError("Compatibility gate failed")
    save(directory / "started.json", {"manifest_sha256": digest(package)})
    byid = {c["id"]: c for c in bank}
    meter, pacer, key = ControlMeter(root), Pacer(root), read_key()

    def dispatch(job, req):
        save(meter.root / (job + ".dispatch.json"), pacer.wait())
        return call(meter, job, stage, req, key)

    outcomes = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for offset in range(0, len(package["jobs"]), 8):
            batch = package["jobs"][offset : offset + 8]
            futures = [
                pool.submit(episode, directory, j, byid[j["case_id"]], dispatch) for j in batch
            ]
            results = [f.result() for f in futures]
            outcomes.extend({"id": r["id"], "terminal": r["terminal"]} for r in results)
            errors = sum(r["terminal"] == "infrastructure_error" for r in results)
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "episodes": len(outcomes),
                        "total": len(package["jobs"]),
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
