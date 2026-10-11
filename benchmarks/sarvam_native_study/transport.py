"""Durable spend reservations, immutable receipts, explicit no-retry HTTP."""

import hashlib
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"
MODELS = {"sarvam-105b": 131072, "sarvam-105b-conversations": 32768}
# micro-INR/token, no cache discount; 25% headroom in the enforced ledger.
INPUT, OUTPUT = 15, 60
ENVELOPES = {
    "setup": 300,
    "discovery": 2700,
    "attribution": 1000,
    "confirmation": 1500,
    "contingency": 500,
}


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def save(path, obj):
    """Create once; never replace a request, receipt or frozen manifest."""
    with Path(path).open("x") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_key():
    """Parse a private assignment as data; never source shell or print its value."""
    path = Path.home() / ".config/agentmem-os/sarvam.env"
    values = []
    for line in path.read_text().splitlines():
        if line.strip().startswith("SARVAM_API_KEY="):
            value = line.strip().split("=", 1)[1].strip().strip("\"'")
            if value:
                values.append(value)
    if len(values) != 1:
        raise ValueError("Expected one nonempty local Sarvam credential")
    return values[0]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Meter:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "meter.sqlite"
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS calls "
                "(id TEXT PRIMARY KEY, stage TEXT, reservation INTEGER, "
                "charged INTEGER, usage_cost INTEGER, status TEXT)"
            )

    def connect(self):
        return sqlite3.connect(self.db, timeout=30)

    def reserve(self, job, stage, payload):
        model = payload["model"]
        if stage not in ENVELOPES or model not in MODELS:
            raise ValueError("Undeclared stage/model")
        limit = payload["max_tokens"]
        if type(limit) is not int or not 0 < limit <= 16384:
            raise ValueError("Invalid output cap")
        amount = (MODELS[model] * INPUT + limit * OUTPUT) * 5 // 4
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            total, count = db.execute(
                "SELECT COALESCE(SUM(charged),0),COUNT(*) FROM calls"
            ).fetchone()
            subtotal = db.execute(
                "SELECT COALESCE(SUM(charged),0) FROM calls WHERE stage=?", (stage,)
            ).fetchone()[0]
            if count >= 20000 or total + amount > 6000_000000:
                raise ValueError("Campaign ceiling reached")
            if subtotal + amount > ENVELOPES[stage] * 1_000000:
                raise ValueError("Stage ceiling reached")
            db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?)",
                (job, stage, amount, amount, None, "reserved"),
            )
        return amount

    def settle(self, job, body, expected_model, maximum):
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        inp, out, total = [
            usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        ]
        valid = (
            all(type(x) is int and x >= 0 for x in (inp, out, total))
            and total == inp + out
            and inp <= MODELS[expected_model]
            and out <= maximum
        )
        with self.connect() as db:
            row = db.execute("SELECT reservation,status FROM calls WHERE id=?", (job,)).fetchone()
            if not row or row[1] != "reserved":
                raise ValueError("Cannot settle a missing/already settled call")
            if valid:
                cost = inp * INPUT + out * OUTPUT
                charged = (cost * 5 + 3) // 4
                db.execute(
                    "UPDATE calls SET charged=?,usage_cost=?,status='known' WHERE id=?",
                    (charged, cost, job),
                )
            else:
                db.execute("UPDATE calls SET status='unknown' WHERE id=?", (job,))
        return valid

    def summary(self):
        with self.connect() as db:
            rows = db.execute(
                "SELECT stage,COUNT(*),SUM(charged),SUM(usage_cost),"
                "SUM(status!='known') FROM calls GROUP BY stage"
            ).fetchall()
        return {
            r[0]: {
                "calls": r[1],
                "charged_with_headroom_inr": r[2] / 1e6,
                "known_usage_estimate_inr": (r[3] or 0) / 1e6,
                "unknown_or_pending": r[4],
            }
            for r in rows
        }


def call(meter, job, stage, payload, key):
    """One dispatch. Unknown errors retain the full reservation; never resend."""
    if not job.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Unsafe job identifier")
    request_path = meter.root / (job + ".request.json")
    save(request_path, {"endpoint": ENDPOINT, "request": payload, "sha256": digest(payload)})
    meter.reserve(job, stage, payload)
    req = urllib.request.Request(
        ENDPOINT,
        data=canonical(payload).encode(),
        headers={"Content-Type": "application/json", "api-subscription-key": key},
    )
    started = time.monotonic()
    receipt = {"job": job, "request_sha256": digest(payload)}
    try:
        try:
            response = urllib.request.build_opener(NoRedirect).open(req, timeout=120)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read(2_000_001)
            receipt.update(
                status=response.code,
                request_id=response.headers.get("x-request-id"),
                body_text=raw.decode("utf-8", errors="replace"),
            )
        if len(raw) > 2_000_000:
            raise ValueError("Receipt limit")
    except Exception as exc:
        # Exception strings may include request details; persist only exception class.
        receipt["transport_error"] = type(exc).__name__
    receipt["elapsed_seconds"] = time.monotonic() - started
    save(meter.root / (job + ".receipt.json"), receipt)
    try:
        body = json.loads(receipt.get("body_text", ""))
    except (ValueError, TypeError):
        body = {}
    known = meter.settle(job, body, payload["model"], payload["max_tokens"])
    if receipt.get("status") != 200 or not known:
        raise RuntimeError("Provider/usage failure; receipt saved, no retry")
    if body.get("model") != payload["model"]:
        raise RuntimeError("Returned model differs; receipt saved")
    return body


def payload(model, messages, tools=None, reasoning=None, maximum=4096):
    obj = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "reasoning_effort": reasoning,
        "max_tokens": maximum,
        "stream": False,
        "n": 1,
    }
    if tools:
        obj.update(tools=tools, tool_choice="auto")
    else:
        obj["response_format"] = {"type": "json_object"}
    return obj
