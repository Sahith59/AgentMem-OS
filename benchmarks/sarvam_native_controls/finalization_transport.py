"""Bounded transport diagnostic on saved native histories; no executable backend."""

import argparse
import copy
import hashlib
import json
import socket
import sqlite3
from pathlib import Path

from benchmarks.sarvam_native_study.paced import Pacer
from benchmarks.sarvam_native_study.score import parse
from benchmarks.sarvam_native_study.transport import (
    ENDPOINT,
    INPUT,
    MODELS,
    OUTPUT,
    call,
    digest,
    read_key,
    save,
)

from .engine import ControlMeter
from .separated import final_request, source_hashes


def read(path):
    return json.loads(path.read_text())


def request_for(history, variant):
    """Retain definitions required by native history; disable new tool selection."""
    definitions = history["tools"]
    names = {t["function"]["name"] for t in definitions}
    pending, observed = set(), set()
    for message in history["messages"]:
        for tool in message.get("tool_calls") or []:
            if tool["function"]["name"] not in names or tool["id"] in observed:
                raise ValueError("Missing definition or duplicate tool identity")
            pending.add(tool["id"])
            observed.add(tool["id"])
        if message["role"] == "tool":
            identity = message["tool_call_id"]
            if identity not in pending:
                raise ValueError("Unmatched actual tool receipt")
            pending.remove(identity)
    if pending or not observed:
        raise ValueError("Diagnostic requires complete native tool history")
    request = final_request(history, history["model"], variant)
    request.update(tools=copy.deepcopy(definitions), tool_choice="none")
    return request


def classify(body):
    choice = body["choices"][0]
    message = choice["message"]
    result = {
        "content": message.get("content"),
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": message.get("tool_calls") or [],
        "parsed": None,
    }
    if result["tool_calls"]:
        result["terminal"] = "refused_final_tool_request"
    elif choice.get("finish_reason") != "stop":
        result["terminal"] = "incomplete"
    else:
        result["terminal"] = "answer"
        try:
            result["parsed"] = parse(result["content"])
        except (TypeError, ValueError):
            pass
    return result


def prepare(root):
    histories = read(root / "histories.json")
    jobs = []
    for index, history in enumerate(histories):
        for variant in ("control", "schema") if index == 0 else ("schema", "control"):
            jobs.append(
                {
                    "id": f"repair-m{index}-{variant}",
                    "history": index,
                    "variant": variant,
                    "request_sha256": digest(request_for(history, variant)),
                }
            )
    if len(jobs) != 4 or {h["model"] for h in histories} != set(MODELS):
        raise ValueError("This diagnostic is limited to two histories/four calls")
    sources = source_hashes()
    relative = "benchmarks/sarvam_native_controls/finalization_transport.py"
    sources[relative] = digest(Path(__file__).read_text())
    frozen = [
        root / name
        for name in (
            "histories.json",
            "expected.json",
            "budget.json",
            "PROTOCOL.md",
            "pacing-control.json",
        )
    ] + sorted((root / "provenance").rglob("*"))
    package = {
        "jobs": jobs,
        "source_sha256": sources,
        "files_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in frozen
            if p.is_file()
        },
        "no_retries": True,
        "scope": "TRANSPORT_ONLY_SAVED_SUCCESSFUL_HISTORIES_NOT_NEW_ACTION_TRIALS",
    }
    for relative in sources:
        destination = root / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as f:
            f.write((Path(__file__).parents[2] / relative).read_text())
    save(root / "manifest.json", package)


def check(root):
    package = read(root / "manifest.json")
    for relative, expected in package["files_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
    for relative, expected in package["source_sha256"].items():
        assert digest((root / "source" / relative).read_text()) == expected
        assert digest((Path(__file__).parents[2] / relative).read_text()) == expected
    histories = read(root / "histories.json")
    for job in package["jobs"]:
        assert (
            digest(request_for(histories[job["history"]], job["variant"])) == job["request_sha256"]
        )
    return package, histories


def run(root):
    package, histories = check(root)
    save(root / "started.json", {"manifest_sha256": digest(package)})
    meter, pacer, key = ControlMeter(root), Pacer(root), read_key()
    results = []
    for job in package["jobs"]:
        request = request_for(histories[job["history"]], job["variant"])
        save(meter.root / (job["id"] + ".dispatch.json"), pacer.wait())
        try:
            result = classify(call(meter, job["id"], "repair", request, key))
        except Exception as exc:
            result = {"terminal": "infrastructure_error", "error_class": type(exc).__name__}
        save(root / (job["id"] + ".result.json"), result)
        results.append({"id": job["id"], "terminal": result["terminal"]})
        print(json.dumps(results[-1]), flush=True)
    check(root)  # Includes byte-identical saved databases; no action executor exists here.
    save(root / "execution.json", {"results": results, "meter": meter.summary()})


def audit(root):
    def denied(*args, **kwargs):
        raise RuntimeError("Offline transport audit forbids network")

    socket.socket = denied
    socket.create_connection = denied
    package, histories = check(root)
    assert read(root / "started.json")["manifest_sha256"] == digest(package)
    labels, rows, ids = read(root / "expected.json"), [], set()
    known, retained, unknown = 0, 0, 0
    with sqlite3.connect(f"file:{root}/calls/meter.sqlite?mode=ro", uri=True) as db:
        ledger = {r[0]: r[1:] for r in db.execute("SELECT * FROM calls")}
    assert set(ledger) == {j["id"] for j in package["jobs"]}
    assert {
        p.name.removesuffix(".request.json") for p in (root / "calls").glob("*.request.json")
    } == set(ledger)
    assert {
        p.name.removesuffix(".receipt.json") for p in (root / "calls").glob("*.receipt.json")
    } == set(ledger)
    for job in package["jobs"]:
        request = request_for(histories[job["history"]], job["variant"])
        saved = read(root / "calls" / (job["id"] + ".request.json"))
        receipt = read(root / "calls" / (job["id"] + ".receipt.json"))
        assert saved == {"endpoint": ENDPOINT, "request": request, "sha256": digest(request)}
        assert receipt["request_sha256"] == digest(request) and receipt["job"] == job["id"]
        stage, reservation, charged, cost, status = ledger[job["id"]]
        assert stage == "repair"
        assert reservation == (MODELS[request["model"]] * INPUT + 8192 * OUTPUT) * 5 // 4
        try:
            body = json.loads(receipt.get("body_text", ""))
        except ValueError:
            body = {}
        if status == "known":
            usage = body["usage"]
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
            assert 0 <= usage["completion_tokens"] <= 8192
            assert 0 <= usage["prompt_tokens"] <= MODELS[request["model"]]
            assert cost == usage["prompt_tokens"] * INPUT + usage["completion_tokens"] * OUTPUT
            assert charged == (cost * 5 + 3) // 4
            known += cost
        else:
            assert status == "unknown" and cost is None and charged == reservation
            unknown += 1
        retained += charged
        if receipt.get("status") == 200 and status == "known" and body["model"] == request["model"]:
            assert body["id"] not in ids
            ids.add(body["id"])
            result = classify(body)
        else:
            result = {"terminal": "infrastructure_error", "error_class": "RuntimeError"}
        assert read(root / (job["id"] + ".result.json")) == result
        rows.append(
            {
                "id": job["id"],
                "model": request["model"],
                "variant": job["variant"],
                "http_status": receipt.get("status"),
                **result,
                "success": result.get("parsed") == labels[job["history"]],
            }
        )
    execution = read(root / "execution.json")
    assert execution["results"] == [{"id": r["id"], "terminal": r["terminal"]} for r in rows]
    for history in histories:
        a, b = [request_for(history, v) for v in ("control", "schema")]
        a.pop("response_format")
        b.pop("response_format")
        assert a == b
    budget = read(root / "budget.json")
    assert retained <= budget["phase_cap_microinr"]
    assert retained + budget["prior_retained_microinr"] <= budget["campaign_cap_microinr"]
    return {
        "verified": True,
        "rows": rows,
        "calls": len(rows),
        "known_inr": known / 1e6,
        "retained_inr": retained / 1e6,
        "unknown_calls": unknown,
        "research_known_inr": (known + budget["prior_known_microinr"]) / 1e6,
        "research_retained_inr": (retained + budget["prior_retained_microinr"]) / 1e6,
        "research_calls": budget["prior_calls"] + len(rows),
        "all_four_compatible_and_correct": all(r["success"] for r in rows) and not unknown,
        "limitation": "Two saved successful histories; no new action trials or accuracy claim",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "audit":
        if not args.output:
            parser.error("audit requires a new --output path")
        save(args.output, audit(args.root))
    else:
        (prepare if args.command == "prepare" else run)(args.root)
