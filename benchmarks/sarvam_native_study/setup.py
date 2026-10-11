"""Excluded live setup checks, separately metered from scored research cases."""

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .transport import MODELS, Meter, call, digest, payload, read_key, save

TOOL = {
    "type": "function",
    "function": {
        "name": "get_counter",
        "description": "Read the authorized parcel's counter from SQLite.",
        "parameters": {
            "type": "object",
            "properties": {"parcel_id": {"type": "string"}},
            "required": ["parcel_id"],
        },
    },
}
QUERIES = {
    "en": 'The parcel counter is X7, not X2. Return JSON {"counter":"..."}.',
    "hi": 'पार्सल का काउंटर X7 है, X2 नहीं। JSON में जवाब दें: {"counter":"..."}।',
    "te": 'పార్సిల్ కౌంటర్ X7, X2 కాదు. JSON రూపంలో జవాబు ఇవ్వండి: {"counter":"..."}.',
}


def specs():
    jobs = []
    for i, model in enumerate(MODELS):
        for lang, text in QUERIES.items():
            jobs.append(
                {
                    "id": f"setup-m{i}-{lang}",
                    "kind": "json",
                    "request": payload(model, [{"role": "user", "content": text}]),
                }
            )
        jobs.append(
            {
                "id": f"setup-m{i}-reasoning",
                "kind": "reasoning",
                "request": payload(
                    model,
                    [
                        {
                            "role": "user",
                            "content": 'Return JSON {"sum":37+28} with the computed sum.',
                        }
                    ],
                    reasoning="low",
                    maximum=8192,
                ),
            }
        )
        jobs.append(
            {
                "id": f"setup-m{i}-tools",
                "kind": "tools",
                "request": payload(
                    model,
                    [
                        {
                            "role": "system",
                            "content": "Use the database tool to look up the parcel. "
                            'Then return JSON {"counter":"..."}. Do not guess.',
                        },
                        {"role": "user", "content": "What is the counter for parcel P-setup?"},
                    ],
                    [TOOL],
                ),
            }
        )
    return jobs


def execute(root):
    root = Path(root)
    directory = root / "setup"
    directory.mkdir()
    files = [Path(__file__), Path(__file__).with_name("transport.py")]
    manifest = {
        "purpose": "EXCLUDED_COMPATIBILITY_CHECKS",
        "jobs": specs(),
        "maximum_calls": 12,
        "stage_ceiling_inr": 300,
        "source_sha256": {p.name: digest(p.read_text()) for p in files},
        "authorization": "Founder approved evidence-first study within INR6000",
    }
    save(directory / "manifest.json", manifest)
    for p in files:
        (directory / (p.name + ".frozen")).write_bytes(p.read_bytes())
    dbpath = directory / "ordinary.sqlite"
    with sqlite3.connect(dbpath) as db:
        db.execute("CREATE TABLE parcels (id TEXT PRIMARY KEY, counter TEXT)")
        db.execute("INSERT INTO parcels VALUES ('P-setup','X7')")
    meter, key = Meter(root / "calls"), read_key()

    def run(spec):
        result = {"id": spec["id"], "kind": spec["kind"], "model": spec["request"]["model"]}
        try:
            body = call(meter, spec["id"], "setup", spec["request"], key)
            choice = body["choices"][0]
            if spec["kind"] == "tools":
                message = choice["message"]
                toolcalls = message.get("tool_calls", [])
                if len(toolcalls) != 1 or toolcalls[0]["function"]["name"] != "get_counter":
                    raise ValueError("Expected one get_counter call")
                tc = toolcalls[0]
                args = json.loads(tc["function"]["arguments"])
                if args != {"parcel_id": "P-setup"}:
                    raise ValueError("Unexpected tool arguments")
                with sqlite3.connect(dbpath) as db:
                    row = db.execute(
                        "SELECT counter FROM parcels WHERE id=?", (args["parcel_id"],)
                    ).fetchone()
                answer = {"counter": row[0]}
                follow = dict(spec["request"])
                follow["messages"] = spec["request"]["messages"] + [
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "tool_calls": toolcalls,
                    },
                    {"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(answer)},
                ]
                body = call(meter, spec["id"] + "-answer", "setup", follow, key)
                choice = body["choices"][0]
                result["sqlite_lookup"] = answer
            result["finish_reason"] = choice.get("finish_reason")
            result["content"] = choice["message"].get("content")
            expected = {"sum": 65} if spec["kind"] == "reasoning" else {"counter": "X7"}
            result["pass"] = (
                choice.get("finish_reason") == "stop" and json.loads(result["content"]) == expected
            )
            result["usage"] = body.get("usage")
        except Exception as exc:
            result["pass"] = False
            result["error_class"] = type(exc).__name__
        save(directory / (spec["id"] + ".result.json"), result)
        print(json.dumps({k: result[k] for k in ("id", "pass")}), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, manifest["jobs"]))
    summary = {"results": results, "meter": meter.summary(), "not_accuracy_benchmark": True}
    save(directory / "summary.json", summary)
    print(
        json.dumps(
            {
                "setup_passed": sum(r["pass"] for r in results),
                "setup_checks": len(results),
                "meter": meter.summary(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    execute(parser.parse_args().directory)
