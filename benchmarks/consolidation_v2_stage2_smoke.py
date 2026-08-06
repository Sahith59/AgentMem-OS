"""
Stage 2 G2 smoke: the REAL pipeline end-to-end — real LongMemEval sessions
ingested as Turn rows, llama3.1 extraction via Ollama, validation, atomic
write into SemanticFacts. Evidence goes into CONSOLIDATION_V2_BUILD_LOG.md.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))


def main():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from agentmem_os.db.models import Base, Session as SessionRow, Turn
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2

    db_file = Path(tempfile.mkdtemp(prefix="agentmem-s2smoke-")) / "s2.db"
    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    mems = {m["mid"]: m for m in ds["memories"]}
    q = next(x for x in ds["queries"]
             if x["question"].startswith("How many times did I ride rollercoasters"))
    sids = q["gold_keys"]

    db = SessionLocal()
    for sid in sids:
        db.add(SessionRow(session_id=sid))
        for i, line in enumerate(mems[sid]["content"].split("\n")):
            line = line.strip()
            if not line:
                continue
            role = "user" if line.startswith("User:") else \
                   "assistant" if line.startswith("Assistant:") else "system"
            db.add(Turn(session_id=sid, role=role, content=line))
    db.commit()
    db.close()

    cv2 = ConsolidationV2(SessionLocal)
    total_created = 0
    for sid in sids:
        r = cv2.consolidate_session(sid)
        total_created += r["created"]
        print(f"{sid}: {r['candidates']} candidates -> {r['created']} created, "
              f"{r['rejected']} rejected", flush=True)
        for t, p in r["rejections"]:
            print(f"    rejected: {t!r} — {p}")

    from agentmem_os.db.semantic_facts import SemanticFactStore
    store = SemanticFactStore(SessionLocal)
    events = [f for f in store.current_facts(fact_type="event", limit=200)
              if re.search(r"coaster|rode|ride", f.fact_text, re.I)]
    print(f"\nTOTAL created: {total_created}; rollercoaster events: {len(events)}")
    for f in events:
        p = store.provenance(f.id)
        occ = f.t_occurred + (f"..{f.t_occurred_end}" if f.t_occurred_end else "") if f.t_occurred else "???"
        print(f"  [{occ} | said {f.t_mentioned}] {f.fact_text[:95]}")
        print(f"      cites turns {p['source_turn_ids'][:4]} intact={p['citations_intact']}")


if __name__ == "__main__":
    main()
