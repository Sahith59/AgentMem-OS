"""Replay saved native responses against fresh SQLite without any provider call."""

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

from . import backend
from .diagnostic import profile_request
from .runner import initial
from .score import grade
from .transport import INPUT, MODELS, OUTPUT, digest


def verify(root, bank, splits):
    root, bank = Path(root), Path(bank)
    cases_list = json.loads((bank / "runtime.json").read_text())
    cases = {c["id"]: c for c in cases_list}
    gold = json.loads((bank / "gold.json").read_text())
    receipt_ids = set()
    used_calls = set()
    episodes = 0
    successes = {}
    for split in splits:
        folder = root / split
        package = json.loads((folder / "manifest.json").read_text())
        assert package["bank_sha256"] == digest(cases_list)
        for name, expected in package["code_sha256"].items():
            assert digest((folder / (name + ".frozen")).read_text()) == expected, name
        rows = []
        for spec in package["jobs"]:
            path = folder / "episodes" / spec["id"] / "result.json"
            if not path.exists():
                continue
            res = json.loads(path.read_text())
            case = cases[spec["case_id"]]
            request = (
                profile_request(case, spec["model"], spec["arm"], spec["reasoning"])
                if split == "attribution"
                else initial(case, spec["model"], spec["arm"])
            )
            assert digest(request) == spec["initial_request_sha256"]
            assert len(res["calls"]) <= 4
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp) / "replay.sqlite"
                backend.create(db, case)
                trace = []
                for job in res["calls"]:
                    assert job not in used_calls, job
                    used_calls.add(job)
                    req = json.loads((root / "calls" / (job + ".request.json")).read_text())
                    rec = json.loads((root / "calls" / (job + ".receipt.json")).read_text())
                    assert req["request"] == request, job
                    assert req["sha256"] == rec["request_sha256"] == digest(request)
                    if rec.get("status") != 200:
                        assert res["terminal"] == "infrastructure_error"
                        break
                    body = json.loads(rec["body_text"])
                    assert body["model"] == spec["model"]
                    assert body["id"] not in receipt_ids, body["id"]
                    receipt_ids.add(body["id"])
                    usage = body["usage"]
                    assert (
                        usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
                    )
                    assert 0 <= usage["prompt_tokens"] <= MODELS[spec["model"]]
                    assert 0 <= usage["completion_tokens"] <= request["max_tokens"]
                    cost = usage["prompt_tokens"] * INPUT + usage["completion_tokens"] * OUTPUT
                    with sqlite3.connect(root / "calls/meter.sqlite") as ledger:
                        entry = ledger.execute(
                            "SELECT usage_cost,charged,status FROM calls WHERE id=?", (job,)
                        ).fetchone()
                    assert entry == (cost, (cost * 5 + 3) // 4, "known")
                    choice = body["choices"][0]
                    message = choice["message"]
                    tcalls = message.get("tool_calls", [])
                    if tcalls:
                        if len(tcalls) > 4:
                            assert res["terminal"] == "tool_limit"
                            break
                        request["messages"].append(
                            {
                                "role": "assistant",
                                "content": message.get("content"),
                                "tool_calls": tcalls,
                            }
                        )
                        allowed = {t["function"]["name"] for t in request.get("tools", [])}
                        for tc in tcalls:
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
                            trace.append({"name": name, "arguments": args, "response": response})
                            request["messages"].append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": json.dumps(response, ensure_ascii=False),
                                }
                            )
                    else:
                        assert res["content"] == message.get("content")
                        assert res["finish_reason"] == choice.get("finish_reason")
                assert trace == res["tool_trace"], spec["id"]
                assert backend.state(db) == res["final_backend"], spec["id"]
                stored = folder / "episodes" / spec["id"] / "backend.sqlite"
                assert backend.state(stored) == backend.state(db)
                assert backend.execute(stored, "read_records", {})["records"] == case["records"]
                rows.append(grade(case, gold[case["id"]], res))
                episodes += 1
        successes[split] = sum(r["task_success"] for r in rows)
    return {
        "verified_episodes": episodes,
        "verified_calls": len(used_calls),
        "unique_provider_ids": len(receipt_ids),
        "task_successes": successes,
        "new_network_calls": 0,
        "scope": "mechanical replay, not independent native semantic validation",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("bank")
    p.add_argument("splits", nargs="+")
    a = p.parse_args()

    def no_network(event, args):
        if event.startswith("socket."):
            raise RuntimeError("Network disabled during verification")

    sys.addaudithook(no_network)
    print(json.dumps(verify(a.root, a.bank, a.splits), indent=2))
