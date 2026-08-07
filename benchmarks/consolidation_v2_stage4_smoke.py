"""
Stage 4 G2 smoke: REAL knowledge-update sessions (LongMemEval `_s`)
through the FULL pipeline — real llama3.1 extraction, real Stage-3
linking, and the REAL llama3.1 supersession judge (its first live run).
Evidence goes into CONSOLIDATION_V2_BUILD_LOG.md.

Two passes over the same real question's gold sessions:
  A. In order (session 1 then session 2): the updated fact must
     supersede the original; transition_text and facts_as_of must show
     the history; t_invalid must carry the winner's domain time.
  B. OUT OF ORDER (session 2 first, then session 1 — the backfill
     case): the domain-time direction rule must produce the SAME final
     state; processing order must not decide truth order.
Whatever the real model does is reported honestly — including
under-supersession (a missed judgment degrades to a stale duplicate,
the disclosed mild class).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

# FORCED (the R5 lesson): a smoke must never touch a live DB, inherited
# env included.
os.environ["AGENTMEM_OS_DB_PATH"] = str(
    Path(tempfile.mkdtemp(prefix="agentmem-s4smoke-env-")) / "env.db")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))


def _fresh():
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from agentmem_os.db.models import Base

    db_file = Path(tempfile.mkdtemp(prefix="agentmem-s4smoke-")) / "s4.db"
    engine = create_engine(f"sqlite:///{db_file}",
                           connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def _ingest(SessionLocal, mems, sid):
    from agentmem_os.db.models import Session as SessionRow, Turn
    db = SessionLocal()
    db.add(SessionRow(session_id=sid))
    for line in mems[sid]["content"].split("\n"):
        line = line.strip()
        if not line:
            continue
        role = "user" if line.startswith("User:") else \
               "assistant" if line.startswith("Assistant:") else "system"
        db.add(Turn(session_id=sid, role=role, content=line))
    db.commit()
    db.close()


def _run_pass(label, order, mems, question):
    from agentmem_os.db.models import SemanticFact, SupersessionJudgment
    from agentmem_os.db.semantic_facts import SemanticFactStore
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2

    print(f"\n== {label} (question: {question[:70]!r}) ==")
    SessionLocal = _fresh()
    cv2 = ConsolidationV2(SessionLocal)
    store = SemanticFactStore(SessionLocal)
    for sid in order:
        _ingest(SessionLocal, mems, sid)
        r = cv2.consolidate_session(sid)
        sup = r["supersession"] or {}
        print(f"{sid}: {r['created']} created, "
              f"{r['entities_linked']} links, "
              f"judged={sup.get('judged', 0)} "
              f"superseded={sup.get('superseded', [])} "
              f"dropped={sup.get('dropped', 0)} "
              f"judge_failure={r['judge_failure']}", flush=True)

    db = SessionLocal()
    superseded = db.query(SemanticFact).filter(
        SemanticFact.superseded_by.isnot(None)).all()
    judgments = db.query(SupersessionJudgment).all()
    print(f"judgment rows: {len(judgments)}; superseded facts: "
          f"{len(superseded)}")
    for f in superseded:
        winner = db.get(SemanticFact, f.superseded_by)
        print(f"  SUPERSEDED [{f.t_mentioned}] {f.fact_text[:70]}")
        print(f"     -> BY   [{winner.t_mentioned}] {winner.fact_text[:70]}")
        print(f"     t_invalid={f.t_invalid} superseded_at set="
              f"{f.superseded_at is not None}")
        print(f"     transition: {store.transition_text(f.id)[:140]}")
        print(f"     as-of {f.t_mentioned}: "
              f"{[x.fact_text[:45] for x in store.facts_as_of(f.t_mentioned) if x.id == f.id]}")
    for j in judgments:
        if j.raw_output_json:
            raw = json.loads(j.raw_output_json)
            print(f"  judgment(fact {j.fact_id}): "
                  f"reasoning={raw.get('reasoning', '')[:100]!r} "
                  f"proposed={raw.get('superseded_ids')} "
                  f"applied={j.applied_json} dropped={j.dropped_json}")
        elif j.skipped:
            print(f"  judgment(fact {j.fact_id}): SKIP {j.skipped}")
    db.close()
    return len(superseded)


def main():
    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    mems = {m["mid"]: m for m in ds["memories"]}
    qs = [q for q in ds["queries"]
          if q.get("question_type") == "knowledge-update"
          and len(q["gold_keys"]) == 2]

    # Demonstration case: a STATE-shaped update (employment change with
    # a named entity — exercises the entity pool AND the polarity
    # vocabulary), both session orders.
    q = qs[22]
    sids = q["gold_keys"]
    a = _run_pass("A. IN ORDER", sids, mems, q["question"])
    b = _run_pass("B. OUT OF ORDER (backfill)", list(reversed(sids)),
                  mems, q["question"])

    # Boundary case, DISCLOSED: the 5K personal best extracts as an
    # EVENT — events are excluded from judgment by design (occurrences
    # never supersede), so this knowledge-update class stays untouched.
    q5k = qs[0]
    c = _run_pass("C. BOUNDARY (event-typed metric — judge excludes "
                  "events by design)", q5k["gold_keys"], mems,
                  q5k["question"])

    # D. Store-level REAL-model metric-update pass (the digit-aware
    # signal's first live exercise — labeled honestly: store-level
    # facts, real judge LLM).
    from agentmem_os.db.semantic_facts import SemanticFactStore
    from agentmem_os.llm.supersession import SupersessionJudge
    print("\n== D. Store-level metric update (REAL judge LLM) ==")
    from agentmem_os.db.models import Session as SessionRow
    SessionLocal = _fresh()
    db = SessionLocal()
    db.add(SessionRow(session_id="metric-demo"))
    db.commit()
    db.close()
    store = SemanticFactStore(SessionLocal)
    judge = SupersessionJudge(SessionLocal)
    old, _ = store.add_fact(
        "The user's personal best time in the charity 5K run is 25:31.",
        fact_type="state", t_mentioned="2023/05/23",
        source_session_id="metric-demo", extraction_model="store-direct")
    new, _ = store.add_fact(
        "The user's personal best time in the charity 5K run is 23:15.",
        fact_type="state", t_mentioned="2023/05/30",
        source_session_id="metric-demo", extraction_model="store-direct")
    rd = judge.judge_fact(new.id)
    print(f"judge verdict: superseded={rd['superseded']} "
          f"dropped={rd['dropped']} skipped={rd['skipped']}")
    if rd["superseded"]:
        print(f"  t_invalid={store.provenance(old.id)['t_invalid']} "
              f"transition={store.transition_text(old.id)[:120]}")

    print(f"\nRESULT: demonstration in-order={a}, out-of-order={b} "
          f"(each supersession that fired must show the domain-earlier "
          f"fact losing); boundary-case={c} (0 expected — event-typed "
          f"metrics are the disclosed exclusion); Part D "
          f"superseded={len(rd['superseded'])} — STORE-INJECTED, "
          f"HAND-AUTHORED facts, real judge LLM: the metric signal has "
          f"never fired on an EXTRACTED fact (disclosed)")


if __name__ == "__main__":
    main()
