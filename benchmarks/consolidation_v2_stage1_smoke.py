"""
Consolidation v2 — Stage 1 G2 smoke: the SemanticFact store exercised with
REAL extraction output (Gate B's llama3.1 facts from real LongMemEval
sessions), on a fresh scratch DB. Evidence goes into
CONSOLIDATION_V2_BUILD_LOG.md.

Checks:
  1. Bulk insert of real facts with real session dates.
  2. Idempotency at scale: re-inserting the whole batch creates ZERO new
     rows, only re-affirmations.
  3. The partial index actually serves the current-facts query
     (EXPLAIN QUERY PLAN must show idx_facts_current).
  4. Point-in-time and provenance queries run against the real rows.

Fact typing here is deliberately crude (event-verb regex) — proper typing is
the Stage 2 extractor's job; this smoke tests STORAGE, not extraction.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))

import tempfile

SCRATCH_DB = Path(tempfile.mkdtemp(prefix="agentmem-smoke-")) / "stage1_smoke.db"

MENTIONED_RE = re.compile(r"\s*\[mentioned:\s*([0-9/-]+)\]\s*$")
EVENT_RE = re.compile(r"\b(rode|attended|went|visited|bought|ran|finished|watched|baked|earned)\b", re.I)
MONTH_DAY_RE = re.compile(
    r"\bon (January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})", re.I)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"])}


def main():
    if SCRATCH_DB.exists():
        SCRATCH_DB.unlink()

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from agentmem_os.db.models import Base, Session as SessionRow
    from agentmem_os.db.semantic_facts import SemanticFactStore

    engine = create_engine(f"sqlite:///{SCRATCH_DB}")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    store = SemanticFactStore(SessionLocal)

    gate_b = json.load(open(HERE / "gate_b_llama31.json"))
    sessions = gate_b["sessions"]

    seed = SessionLocal()
    for sid in sessions:
        seed.add(SessionRow(session_id=sid))
    seed.commit()
    seed.close()

    def insert_all():
        created = reaffirmed = dated = 0
        for sid, data in sessions.items():
            for raw in data["facts"]:
                m = MENTIONED_RE.search(raw)
                text = MENTIONED_RE.sub("", raw).strip()
                t_mentioned = m.group(1) if m else data["session_date"]
                if not text:
                    continue
                t_occurred = None
                md = MONTH_DAY_RE.search(text)
                if md:
                    # Crude year inference (proper resolution is the Stage 2
                    # extractor's job). Guard: a derived date AFTER the
                    # mention date means the inference is wrong or the event
                    # is planned-future — skip rather than store a lie
                    # (G3 finding 7: 2/8 first-cut dates were impossible).
                    year = t_mentioned.split("/")[0]
                    candidate = f"{year}/{_MONTHS[md.group(1).lower()]:02d}/{int(md.group(2)):02d}"
                    if candidate <= t_mentioned:
                        t_occurred = candidate
                        dated += 1
                fact_type = "event" if EVENT_RE.search(text) else "state"
                try:
                    _, was_created = store.add_fact(
                        text, fact_type=fact_type, t_mentioned=t_mentioned,
                        source_session_id=sid, t_occurred=t_occurred,
                        extraction_model=gate_b["model"],
                    )
                except ValueError as e:
                    print(f"  [rejected] {text[:60]!r}: {e}")
                    continue
                if was_created:
                    created += 1
                else:
                    reaffirmed += 1
        return created, reaffirmed, dated

    c1, r1, d1 = insert_all()
    print(f"PASS 1: {c1} created, {r1} re-affirmed, {d1} with in-text event dates")

    c2, r2, _ = insert_all()
    print(f"PASS 2 (idempotency): {c2} created, {r2} re-affirmed")
    assert c2 == 0, "re-insertion must create zero rows"

    from agentmem_os.db.models import SemanticFact
    db = SessionLocal()
    total_rows = db.query(SemanticFact).count()
    db.close()
    live = store.current_facts(limit=total_rows + 1)
    print(f"current facts: {len(live)} of {total_rows} rows "
          f"(== {c1} created: {len(live) == c1})")
    assert len(live) == c1 == total_rows

    events = store.current_facts(fact_type="event", limit=1000)
    print(f"event facts: {len(events)}; sample dated rows:")
    for f in [f for f in events if f.t_occurred][:3]:
        print(f"  [{f.t_occurred} | said {f.t_mentioned}] {f.fact_text[:90]}")

    asof = store.facts_as_of("2023/06/01")
    print(f"facts_as_of 2023/06/01: {len(asof)}")

    p = store.provenance(live[0].id)
    print(f"provenance sample: session={p['source_session_id']} "
          f"model={p['extraction_model']} intact={p['citations_intact']}")

    # EXPLAIN the statements the store ACTUALLY EMITTED — captured off the
    # engine during real store calls, never re-declared (G3 R1 F10 and R2
    # B5: a lookalike query goes green even if the store's real query
    # regresses).
    from sqlalchemy import event as sa_event
    captured = []

    def grab(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith("SELECT"):
            captured.append((statement, parameters))

    sa_event.listen(engine, "before_cursor_execute", grab)
    store.current_facts(fact_type="event", limit=10)
    store.current_facts(limit=10)
    sa_event.remove(engine, "before_cursor_execute", grab)

    plans = []
    with engine.connect() as conn:
        for stmt, params in captured:
            plans.append(" | ".join(str(r) for r in conn.exec_driver_sql(
                f"EXPLAIN QUERY PLAN {stmt}", params).fetchall()))
    typed_plan, untyped_plan = plans[0], plans[1]
    print(f"typed-path plan:   {typed_plan}")
    print(f"untyped-path plan: {untyped_plan}")
    assert re.search(r"USING (?:COVERING )?INDEX idx_facts_current\b(?!_)", typed_plan)
    assert re.search(r"USING (?:COVERING )?INDEX idx_facts_current_all\b", untyped_plan)
    for p in plans:
        assert "TEMP B-TREE" not in p.upper(), "temp sort — LIMIT can't short-circuit"
    print("BOTH HOT PATHS CONFIRMED ON CAPTURED STORE SQL, NO TEMP SORT")


if __name__ == "__main__":
    main()
