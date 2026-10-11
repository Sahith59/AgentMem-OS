"""Ordinary SQLite test backend. No semantic extraction or model-generated state."""

import json
import sqlite3
from pathlib import Path

from .transport import canonical

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read_records",
        "description": "Read the original records for this test session from SQLite. No arguments.",
        "parameters": {"type": "object", "properties": {}},
    },
}
UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_appointment",
        "description": (
            "Attempt a date update. Closed appointments reject it. "
            "Use the user's exact request key. A repeated identical key is idempotent. "
            "Supply the current version; a stale version rejects. "
            "Returns actual outcome and current date."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "date": {"type": "string"},
                "idempotency_key": {"type": "string"},
                "expected_version": {"type": "integer"},
            },
            "required": ["appointment_id", "date", "idempotency_key", "expected_version"],
        },
    },
}


def create(path, case):
    path = Path(path)
    with path.open("xb"):
        pass
    state = case["backend"]
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE records (position INTEGER PRIMARY KEY, payload TEXT NOT NULL)")
        db.executemany(
            "INSERT INTO records VALUES (?,?)",
            [(i, canonical(r)) for i, r in enumerate(case["records"])],
        )
        db.execute(
            "CREATE TABLE appointment (id TEXT PRIMARY KEY, date TEXT, "
            "version INTEGER, closed INTEGER)"
        )
        db.execute(
            "INSERT INTO appointment VALUES (?,?,?,?)",
            (state["appointment_id"], state["date"], state["version"], int(state["closed"])),
        )
        db.execute("CREATE TABLE operations (key TEXT PRIMARY KEY, date TEXT)")
        db.executemany(
            "INSERT INTO operations VALUES (?,?)",
            [(r["key"], r["date"]) for r in state["operations"]],
        )


def state(path):
    with sqlite3.connect(path) as db:
        row = db.execute("SELECT id,date,version,closed FROM appointment").fetchone()
        operations = db.execute("SELECT key,date FROM operations ORDER BY key").fetchall()
    return {
        "appointment_id": row[0],
        "date": row[1],
        "version": row[2],
        "closed": bool(row[3]),
        "operations": [{"key": r[0], "date": r[1]} for r in operations],
    }


def execute(path, name, args):
    if not isinstance(args, dict):
        return {"error": "arguments_must_be_object"}
    with sqlite3.connect(path) as db:
        if name == "read_records":
            if args:
                return {"error": "unexpected_arguments"}
            return {
                "records": [
                    json.loads(r[0])
                    for r in db.execute("SELECT payload FROM records ORDER BY position")
                ]
            }
        if name != "update_appointment":
            return {"error": "unknown_tool"}
        required = {"appointment_id", "date", "idempotency_key", "expected_version"}
        if (
            set(args) != required
            or type(args["expected_version"]) is not int
            or any(not isinstance(args[k], str) for k in required - {"expected_version"})
        ):
            return {"error": "invalid_arguments"}
        # A real date is required; date text is never repaired or guessed.
        from datetime import date

        try:
            if date.fromisoformat(args["date"]).isoformat() != args["date"]:
                return {"error": "invalid_date"}
        except ValueError:
            return {"error": "invalid_date"}
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT date,version,closed FROM appointment WHERE id=?", (args["appointment_id"],)
        ).fetchone()
        if row is None:
            return {"status": "rejected", "reason": "unknown_appointment"}
        existing = db.execute(
            "SELECT date FROM operations WHERE key=?", (args["idempotency_key"],)
        ).fetchone()
        if existing:
            return {
                "status": "already_applied" if existing[0] == args["date"] else "rejected",
                "date": row[0],
                "version": row[1],
            }
        if row[2] or row[1] != args["expected_version"]:
            return {
                "status": "rejected",
                "date": row[0],
                "version": row[1],
                "reason": "closed" if row[2] else "version_mismatch",
            }
        db.execute(
            "UPDATE appointment SET date=?,version=version+1 WHERE id=?",
            (args["date"], args["appointment_id"]),
        )
        db.execute("INSERT INTO operations VALUES (?,?)", (args["idempotency_key"], args["date"]))
        return {"status": "confirmed", "date": args["date"], "version": row[1] + 1}
