"""Offline application-control evidence; never a model accuracy benchmark."""

import argparse
import hashlib
import json
import socket
import tempfile
from pathlib import Path

from benchmarks.sarvam_native_study.transport import save

from . import approval_backend as backend


def read(path):
    return json.loads(path.read_text())


def exercise(root, output):
    def deny(*args, **kwargs):
        raise RuntimeError("Offline approval replay forbids network")

    socket.socket = deny
    socket.create_connection = deny
    output.mkdir()
    manifest = read(root / "manifest.json")
    for relative, sha in manifest["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == sha
    for relative, sha in manifest["sources"].items():
        assert (
            hashlib.sha256((Path(__file__).parents[2] / relative).read_bytes()).hexdigest() == sha
        )
        assert hashlib.sha256((root / "source" / relative).read_bytes()).hexdigest() == sha
    cases, labels = read(root / "runtime.json"), read(root / "expected.json")
    rows = []
    for case in cases:
        for arm, enforce in (("unenforced", False), ("enforced", True)):
            db = output / (case["id"] + "-" + arm + ".sqlite")
            backend.create(db, case["fixture"])
            receipts = []
            for step in case["steps"]:
                if step["operation"] == "read":
                    reply = backend.current(db, step["appointment_id"], caller=case["caller"])
                else:
                    assert step["operation"] == "update"
                    reply = backend.execute(
                        db,
                        step["args"],
                        caller=case["caller"],
                        now=step["now"],
                        enforce=enforce,
                    )
                receipts.append(reply)
            state = backend.inspect_state(db)
            expected = labels[case["id"]][arm]
            assert receipts == expected["receipts"], (case["id"], arm, receipts)
            projection = {
                "appointments": state["appointments"],
                "operation_count": len(state["operations"]),
            }
            assert projection == expected["final_state"], (case["id"], arm, projection)
            rows.append({"id": case["id"], "arm": arm, "receipts": receipts, "state": state})
    result = {
        "verified": True,
        "cases": len(cases),
        "variants": len(rows),
        "rows": rows,
        "api_calls": 0,
        "claim": "Deterministic application controls, not model accuracy",
    }
    save(output / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        exercise(args.root, args.output)
    else:
        with tempfile.TemporaryDirectory() as temp:
            result = exercise(args.root, Path(temp) / "replay")
            assert result == read(args.root / "executed/results.json")
            for path in (args.root / "executed").glob("*.sqlite"):
                assert backend.inspect_state(path) == backend.inspect_state(
                    Path(temp) / "replay" / path.name
                )
            save(args.output, {k: v for k, v in result.items() if k != "rows"})
    print("Offline approval states and actual receipts verified; zero API calls.")
