"""
Gate C extraction, LIVE pipeline: consolidate EVERY session in the 79
slice questions' haystacks (noise included — no oracle shortcut)
through the REAL ConsolidationV2 — extraction + validation + entity
linking + supersession judgment — into a PERSISTENT dedicated DB that
the Gate C eval will read. $0, local llama3.1.

This replaces the RETIRED proxy runner (consolidation_v2_extract.py,
draft prompt, JSONL output, stopped at 48/3,631): Gate C must measure
what the SHIPPED pipeline stores, not a proxy of it.

Discipline carried over (the Graphiti lesson): parallel workers +
restart-safe checkpointing. Checkpoint source of truth is the DB
ITSELF — a session is done iff it has a consolidation_v2 row in
consolidation_log — so the resume state can never drift from the
facts. Failures are LOUD: per-session retry once, then recorded to
failures JSONL and counted in the summary; the run never dies to one
bad session.

Target DB: benchmarks/extracted_memories/gate_c_facts.db (gitignored,
persistent — THE Gate C artifact). Redis disabled. No assembler runs
here (write path only), so the storage tree is untouched.

Usage:
    python3 benchmarks/consolidation_v2_extract_live.py [workers] [limit]
    (limit is for smokes; omit for the full run)
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
DB_PATH = os.environ.get(
    "GATE_C_DB", str(HERE / "extracted_memories" / "gate_c_facts.db"))
os.environ["AGENTMEM_OS_DB_PATH"] = DB_PATH
os.environ["AGENTMEM_OS_DISABLE_REDIS"] = "1"

sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
# Sharding for the SLURM array (shard index/count, 0-based). Each shard
# owns a DISJOINT stride of the sorted worklist and its OWN DB (set via
# GATE_C_DB) — no cross-shard writers on one SQLite file, ever.
SHARD = int(os.environ.get("GATE_C_SHARD", "0"))
NSHARDS = int(os.environ.get("GATE_C_NSHARDS", "1"))
FAILLOG = Path(os.environ.get(
    "GATE_C_FAILLOG",
    str(HERE / "extracted_memories" / "gate_c_failures.jsonl")))


def build_worklist():
    """Union of scope_keys of the 79 slice questions actually run —
    exactly the haystacks the Gate C eval will query. Deterministic
    order (sorted) so resumed runs walk the same list."""
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    slice_qs = {r["question"] for r in json.load(open(
        HERE / "qa_accuracy_longmemeval_answerer54mini.json"))["results"]}
    qmap = {q["question"]: q for q in ds["queries"]}
    mems = {m["mid"]: m for m in ds["memories"]}
    sids = set()
    for qtext in slice_qs:
        for k in qmap[qtext]["scope_keys"]:
            if k in mems:
                sids.add(k)
    return sorted(sids), mems


def already_done(get_session):
    from agentmem_os.db.models import ConsolidationLog
    db = get_session()
    try:
        rows = (db.query(ConsolidationLog.session_id)
                .filter(ConsolidationLog.triggered_by
                        == "consolidation_v2").all())
        return {r[0] for r in rows}
    finally:
        db.close()


def ingest(get_session, mems, sid):
    from agentmem_os.db.models import Session as SessionRow, Turn
    db = get_session()
    try:
        if db.query(SessionRow).filter(
                SessionRow.session_id == sid).first():
            return
        db.add(SessionRow(session_id=sid))
        for line in mems[sid]["content"].split("\n"):
            line = line.strip()
            if not line:
                continue
            role = "user" if line.startswith("User:") else \
                "assistant" if line.startswith("Assistant:") else "system"
            db.add(Turn(session_id=sid, role=role, content=line))
        db.commit()
    finally:
        db.close()


def main():
    sids, mems = build_worklist()
    full_n = len(sids)
    if LIMIT:
        sids = sids[:LIMIT]
        print(f"SMOKE MODE: limited to first {LIMIT} sessions")
    if NSHARDS > 1:
        # Stride slicing on the DETERMINISTIC sorted worklist: shard i
        # takes every NSHARDS-th session. Strided (not blocked) so an
        # uneven size/latency distribution spreads across shards
        # instead of loading one. Union of shards == worklist exactly,
        # verified by the merge step.
        sids = sids[SHARD::NSHARDS]
        print(f"SHARD {SHARD}/{NSHARDS}: {len(sids)} of {full_n} sessions")

    from agentmem_os.db.engine import get_session
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2

    done = already_done(get_session)
    todo = [s for s in sids if s not in done]
    print(f"Gate C LIVE extraction: {len(sids)} sessions total, "
          f"{len(done)} done, {len(todo)} to go | {WORKERS} workers | "
          f"DB={DB_PATH}", flush=True)

    cv2 = ConsolidationV2(get_session)
    lock = threading.Lock()
    state = {"ok": 0, "fail": 0, "facts": 0, "planned": 0, "t0": time.time()}

    def work(sid):
        for attempt in (0, 1):
            try:
                ingest(get_session, mems, sid)
                r = cv2.consolidate_session(sid)
                with lock:
                    state["ok"] += 1
                    state["facts"] += r.get("created", 0)
                    n = state["ok"] + state["fail"]
                    rate = n / max(1e-9, time.time() - state["t0"])
                    eta_h = (len(todo) - n) / max(1e-9, rate) / 3600
                    if n % 10 == 0 or n == len(todo):
                        print(f"  {n}/{len(todo)} | {state['facts']} facts "
                              f"| {rate * 3600:.0f}/h | ETA {eta_h:.1f}h",
                              flush=True)
                return
            except Exception as e:
                if attempt == 1:
                    with lock:
                        state["fail"] += 1
                    with open(FAILLOG, "a") as f:
                        f.write(json.dumps(
                            {"sid": sid, "error": f"{type(e).__name__}: "
                                                  f"{str(e)[:300]}"}) + "\n")
                    print(f"  FAILED after retry: {sid}: "
                          f"{type(e).__name__}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo))

    hours = (time.time() - state["t0"]) / 3600
    print(f"\nDONE: ok={state['ok']} failed={state['fail']} "
          f"facts_created={state['facts']} in {hours:.2f}h")
    from agentmem_os.db.models import SemanticFact
    db = get_session()
    try:
        total = db.query(SemanticFact).count()
        planned = (db.query(SemanticFact)
                   .filter(SemanticFact.event_status == "planned").count())
    finally:
        db.close()
    print(f"DB totals: {total} facts, {planned} planned events")
    if state["fail"]:
        print(f"FAILURES logged to {FAILLOG} — rerun to retry (resume "
              f"skips completed sessions)")


if __name__ == "__main__":
    main()
