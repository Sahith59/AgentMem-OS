"""Ordinary SQLite evidence store, with no application DB or model dependencies."""

import sqlite3
from contextlib import closing

from .contract import canonical, eligible, instant, loads


def write_store(path, case, facts):
    if path.exists():
        raise FileExistsError("Refusing to replace an experiment store")
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "CREATE TABLE records (id TEXT PRIMARY KEY, tenant TEXT, customer TEXT, "
            "observed_at TEXT, payload TEXT)"
        )
        db.execute(
            "CREATE TABLE facts (position INTEGER PRIMARY KEY, tenant TEXT, "
            "customer TEXT, payload TEXT)"
        )
        db.executemany(
            "INSERT INTO records VALUES (?,?,?,?,?)",
            [
                (r["id"], r["tenant"], r["customer"], r["observed_at"], canonical(r))
                for r in case["records"]
            ],
        )
        db.executemany(
            "INSERT INTO facts VALUES (?,?,?,?)",
            [
                (i, case["tenant"], case["authorized_customer"], canonical(fact))
                for i, fact in enumerate(facts)
            ],
        )


def read_packet(path, case):
    # Reopen a persisted DB. Fetch all facts AND all original authorized records
    # for these short histories. Extraction omission cannot hide raw evidence.
    if case["authorized_customer"] is None:
        return [], []
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        facts = [
            loads(row[0])
            for row in db.execute(
                "SELECT payload FROM facts WHERE tenant=? AND customer=? ORDER BY position",
                (case["tenant"], case["authorized_customer"]),
            )
        ]
        wanted = {s for fact in facts for s in fact["source_ids"]}
        sources = [
            loads(row[0])
            for row in db.execute(
                "SELECT payload FROM records WHERE tenant=? AND customer=? ORDER BY rowid",
                (case["tenant"], case["authorized_customer"]),
            )
            if instant(loads(row[0])["observed_at"]) <= instant(case["as_of"])
        ]
    if not wanted <= {r["id"] for r in eligible(case)}:
        raise ValueError("Stored source outside authorized/time scope")
    if sources != eligible(case) or not wanted <= {r["id"] for r in sources}:
        raise ValueError("Missing stored source")
    return facts, sources
