"""Research-only ordinary SQLite authorization; no model text or gold-file authority."""

import json
import sqlite3
from datetime import date
from pathlib import Path

from benchmarks.sarvam_native_study.transport import canonical

ARGUMENTS = {"appointment_id", "date", "expected_version", "idempotency_key", "approval_id"}
UPDATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_appointment",
        "description": (
            "Request a date change using an existing server-issued approval ID. "
            "An approval must cover this customer, booking, date, operation and version, "
            "and be currently granted, unrevoked and unexpired. Never invent an approval. "
            "Use the exact request key. Returns actual outcome and authorized current state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                **{key: {"type": "string"} for key in sorted(ARGUMENTS - {"expected_version"})},
                "expected_version": {"type": "integer"},
            },
            "required": sorted(ARGUMENTS),
            "additionalProperties": False,
        },
    },
}


def connect(path, *, readonly=False):
    return sqlite3.connect(
        Path(path).resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
        uri=True,
        timeout=30,
    )


def create(path, fixture):
    """Only the trusted test application supplies fixtures; never tool/model arguments."""
    with Path(path).open("xb"):
        pass
    with connect(path) as db:
        db.execute("CREATE TABLE trusted_issuers (issuer TEXT PRIMARY KEY)")
        db.executemany("INSERT INTO trusted_issuers VALUES (?)", [(v,) for v in fixture["issuers"]])
        db.execute(
            "CREATE TABLE appointments (tenant TEXT, customer TEXT, id TEXT, date TEXT, "
            "version INTEGER NOT NULL CHECK(version>=0), closed INTEGER NOT NULL, "
            "PRIMARY KEY(tenant,customer,id))"
        )
        for row in fixture["appointments"]:
            db.execute(
                "INSERT INTO appointments VALUES (?,?,?,?,?,?)",
                (
                    row["tenant"],
                    row["customer"],
                    row["id"],
                    row["date"],
                    row["version"],
                    int(row["closed"]),
                ),
            )
        db.execute(
            "CREATE TABLE approvals (tenant TEXT, customer TEXT, id TEXT, appointment_id TEXT, "
            "operation TEXT, date TEXT, expected_version INTEGER, issuer TEXT, issued_at INTEGER, "
            "not_before INTEGER, expires_at INTEGER, granted INTEGER, revoked INTEGER, "
            "PRIMARY KEY(tenant,customer,id))"
        )
        for row in fixture["approvals"]:
            db.execute(
                "INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["tenant"],
                    row["customer"],
                    row["id"],
                    row["appointment_id"],
                    row["operation"],
                    row["date"],
                    row["expected_version"],
                    row["issuer"],
                    row["issued_at"],
                    row["not_before"],
                    row["expires_at"],
                    int(row["granted"]),
                    int(row["revoked"]),
                ),
            )
        db.execute(
            "CREATE TABLE operations (tenant TEXT, customer TEXT, key TEXT, "
            "request_json TEXT NOT NULL, receipt_json TEXT NOT NULL, "
            "PRIMARY KEY(tenant,customer,key))"
        )


def validate_caller(caller):
    if (
        not isinstance(caller, dict)
        or set(caller) != {"tenant", "customer"}
        or any(not isinstance(v, str) or not v for v in caller.values())
    ):
        raise ValueError("Trusted application caller scope required")


def current(path, appointment_id, *, caller):
    validate_caller(caller)
    with connect(path, readonly=True) as db:
        row = db.execute(
            "SELECT date,version,closed FROM appointments WHERE tenant=? AND customer=? AND id=?",
            (caller["tenant"], caller["customer"], appointment_id),
        ).fetchone()
    if row is None:
        return {"status": "rejected", "reason": "not_found_or_forbidden"}
    return {"date": row[0], "version": row[1], "closed": bool(row[2])}


def approval_problem(db, args, caller, now):
    row = db.execute(
        "SELECT appointment_id,operation,date,expected_version,issuer,issued_at,not_before,"
        "expires_at,granted,revoked FROM approvals WHERE tenant=? AND customer=? AND id=?",
        (caller["tenant"], caller["customer"], args["approval_id"]),
    ).fetchone()
    if row is None:
        return "approval_missing"
    appointment, operation, target, version, issuer, issued, start, expiry, granted, revoked = row
    if not db.execute("SELECT 1 FROM trusted_issuers WHERE issuer=?", (issuer,)).fetchone():
        return "untrusted_issuer"
    if (appointment, operation, target, version) != (
        args["appointment_id"],
        "update_appointment",
        args["date"],
        args["expected_version"],
    ):
        return "approval_scope_mismatch"
    if not granted:
        return "approval_pending"
    if revoked:
        return "approval_revoked"
    if not (issued <= start < expiry) or now < start:
        return "approval_not_effective"
    if now >= expiry:
        return "approval_expired"
    return None


def execute(path, args, *, caller, now, enforce=True):
    """Trusted caller/time/policy cannot be selected or overridden in tool arguments."""
    validate_caller(caller)
    if type(now) is not int or now < 0 or type(enforce) is not bool:
        raise ValueError("Trusted integer clock and boolean policy required")
    if (
        not isinstance(args, dict)
        or set(args) != ARGUMENTS
        or type(args["expected_version"]) is not int
        or args["expected_version"] < 0
        or any(
            not isinstance(args[k], str) or not args[k] for k in ARGUMENTS - {"expected_version"}
        )
    ):
        return {"status": "rejected", "reason": "invalid_arguments"}
    try:
        if date.fromisoformat(args["date"]).isoformat() != args["date"]:
            raise ValueError("Not canonical")
    except ValueError:
        return {"status": "rejected", "reason": "invalid_date"}
    tenant, customer = caller["tenant"], caller["customer"]
    with connect(path) as db:
        # Serializes approval check, version check, write and operation receipt together.
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT date,version,closed FROM appointments WHERE tenant=? AND customer=? AND id=?",
            (tenant, customer, args["appointment_id"]),
        ).fetchone()
        if row is None:
            return {"status": "rejected", "reason": "not_found_or_forbidden"}
        state = {"date": row[0], "version": row[1]}
        previous = db.execute(
            "SELECT request_json,receipt_json FROM operations "
            "WHERE tenant=? AND customer=? AND key=?",
            (tenant, customer, args["idempotency_key"]),
        ).fetchone()
        if previous:
            if previous[0] != canonical(args):
                return {"status": "rejected", "reason": "idempotency_conflict", **state}
            return {
                "status": "already_applied",
                **state,
                "original_receipt": json.loads(previous[1]),
            }
        if row[2] or row[1] != args["expected_version"]:
            return {
                "status": "rejected",
                **state,
                "reason": "closed" if row[2] else "version_mismatch",
            }
        reason = approval_problem(db, args, caller, now) if enforce else None
        if reason:
            return {"status": "rejected", "reason": reason, **state}
        db.execute(
            "UPDATE appointments SET date=?,version=version+1 "
            "WHERE tenant=? AND customer=? AND id=?",
            (args["date"], tenant, customer, args["appointment_id"]),
        )
        receipt = {"status": "confirmed", "date": args["date"], "version": row[1] + 1}
        db.execute(
            "INSERT INTO operations VALUES (?,?,?,?,?)",
            (
                tenant,
                customer,
                args["idempotency_key"],
                canonical(args),
                canonical(receipt),
            ),
        )
        return receipt


def inspect_state(path):
    """Offline audit only; this unscoped view is never exposed as a model tool."""
    with connect(path, readonly=True) as db:
        db.row_factory = sqlite3.Row
        return {
            "appointments": [
                dict(r)
                for r in db.execute("SELECT * FROM appointments ORDER BY tenant,customer,id")
            ],
            "operations": [
                dict(r) for r in db.execute("SELECT * FROM operations ORDER BY tenant,customer,key")
            ],
        }
