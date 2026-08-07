"""
Stage 3 G2 smoke: the REAL pipeline end-to-end with linking — real
LongMemEval sessions, real llama3.1 extraction, real spaCy NER, real
multilingual-e5-small alias resolution. Evidence goes into
CONSOLIDATION_V2_BUILD_LOG.md.

Parts (each labeled by exactly what is real):
  A. Real-corpus English: the rollercoaster gold sessions → facts +
     entity links; entity-timeline readout via facts_for_entity.
  B. Hindi session through the FULL pipeline (llama3.1 extracts whatever
     it extracts — reported honestly, including the fact language).
  C. Store-level Hindi fact + REAL resolver (the Gate-E write shape,
     bypassing the LLM so the alias gate itself is measured): script
     token → τ-match → FOREIGN node + ALIAS_OF + link; read back from
     BOTH surfaces.
"""
import json
import os
import re
import sys
import tempfile
from pathlib import Path

# R4-m1: pin the engine's DB path BEFORE any agentmem_os import —
# importing the package runs init_db() against whatever path resolves,
# and a smoke run must never migrate the live dev DB as a side effect.
# FORCED, not setdefault (R5 minor + conftest's own R3-N8 rule): an
# inherited env var pointing at the real DB must lose to the scratch
# path — a smoke must NEVER touch a live DB, inherited env included.
os.environ["AGENTMEM_OS_DB_PATH"] = str(
    Path(tempfile.mkdtemp(prefix="agentmem-s3smoke-env-")) / "env.db")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))


def main():
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker
    from agentmem_os.db.models import (
        Base, ConsolidationLog, KnowledgeGraphEdge, KnowledgeGraphNode,
        Session as SessionRow, SemanticFactEntity, Turn,
    )
    from agentmem_os.db.fact_entities import FactEntityLinker
    from agentmem_os.db.semantic_facts import SemanticFactStore
    from agentmem_os.llm.consolidation_v2 import ConsolidationV2

    db_file = Path(tempfile.mkdtemp(prefix="agentmem-s3smoke-")) / "s3.db"
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
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    ds = json.load(open(HERE / "benchmark_cache/longmemeval_s.json"))
    mems = {m["mid"]: m for m in ds["memories"]}
    q = next(x for x in ds["queries"]
             if x["question"].startswith("How many times did I ride rollercoasters"))
    sids = q["gold_keys"]

    db = SessionLocal()
    for sid in sids:
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

    cv2 = ConsolidationV2(SessionLocal)
    linker = FactEntityLinker(SessionLocal)
    store = SemanticFactStore(SessionLocal)

    print("== A. Real-corpus English sessions (full pipeline) ==")
    total_created = total_linked = 0
    for sid in sids:
        r = cv2.consolidate_session(sid)
        total_created += r["created"]
        total_linked += r["entities_linked"]
        print(f"{sid}: {r['candidates']} candidates -> {r['created']} created, "
              f"{r['rejected']} rejected, {r['entities_linked']} entity links, "
              f"link_failure={r['link_failure']}", flush=True)
        for s, reason in r["entity_links_skipped"]:
            print(f"    link skipped: {s!r} — {reason}")

    db = SessionLocal()
    n_nodes = db.query(KnowledgeGraphNode).count()
    n_links = db.query(SemanticFactEntity).count()
    log_linked = [l.entities_linked for l in db.query(ConsolidationLog).all()]
    top_nodes = [r[0] for r in db.query(KnowledgeGraphNode.entity_text)
                 .join(SemanticFactEntity,
                       SemanticFactEntity.node_id == KnowledgeGraphNode.id)
                 .group_by(KnowledgeGraphNode.entity_text).all()]
    db.close()
    # Independent artifacts: report counters vs PERSISTED log rows vs
    # ACTUAL join-table rows — assert, don't just print (a printout that
    # nobody compares is a lookalike check).
    assert total_linked == sum(log_linked) == n_links, \
        (total_linked, sum(log_linked), n_links)
    print(f"\nTOTAL: {total_created} facts, {total_linked} links, "
          f"{n_nodes} nodes, {n_links} join rows == log sum "
          f"{sum(log_linked)} (asserted)")
    print(f"linked entities: {sorted(top_nodes)}")

    probe = next((t for t in top_nodes
                  if "seaworld" in t.lower() or "disney" in t.lower()),
                 top_nodes[0] if top_nodes else None)
    if probe:
        print(f"\nfacts_for_entity({probe!r}) timeline:")
        for f in linker.facts_for_entity(probe):
            occ = (f.t_occurred or "undated") + \
                  (f"..{f.t_occurred_end}" if f.t_occurred_end else "")
            p = store.provenance(f.id)
            print(f"  [{occ} | {f.fact_type}/{f.event_status}] "
                  f"{f.fact_text[:80]}")
            print(f"      cites {p['source_turn_ids'][:4]} "
                  f"intact={p['citations_intact']}")

    print("\n== B. Hindi session (full pipeline, llama3.1) ==")
    db = SessionLocal()
    db.add(SessionRow(session_id="hi-1"))
    db.add(Turn(session_id="hi-1", role="system",
                content="Session dated 2023/06/10"))
    db.add(Turn(session_id="hi-1", role="user",
                content="मैं गूगल में सॉफ्टवेयर इंजीनियर के रूप में काम करता हूँ "
                        "और मुझे यह बहुत पसंद है।"))
    db.add(Turn(session_id="hi-1", role="assistant",
                content="बहुत बढ़िया! गूगल एक शानदार कंपनी है।"))
    db.commit()
    db.close()
    rb = cv2.consolidate_session("hi-1", lang_source="hi")
    print(f"hi-1: {rb['candidates']} candidates -> {rb['created']} created, "
          f"{rb['rejected']} rejected, {rb['entities_linked']} links")
    for t, p in rb["rejections"]:
        print(f"    rejected: {t!r} — {p}")
    db = SessionLocal()
    from agentmem_os.db.models import SemanticFact
    hi_facts = db.query(SemanticFact).filter(
        SemanticFact.source_session_id == "hi-1").all()
    for f in hi_facts:
        print(f"    fact: {f.fact_text[:90]!r} entities={f.entities}")
    db.close()

    print("\n== C. Store-level Hindi fact + REAL resolver (Gate-E write shape) ==")
    # Anchor first, through the FULL pipeline: an English session that
    # mentions Google — its fact links create the English node the alias
    # gate needs (the first run proved the NEGATIVE path: with no anchor,
    # गूगल and every common Hindi word were correctly kept OUT at τ).
    db = SessionLocal()
    db.add(SessionRow(session_id="en-1"))
    db.add(Turn(session_id="en-1", role="system",
                content="Session dated 2023/06/09"))
    db.add(Turn(session_id="en-1", role="user",
                content="I started working at Google as a software "
                        "engineer this spring and I really enjoy it."))
    db.commit()
    db.close()
    ra = cv2.consolidate_session("en-1")
    print(f"en-1 (anchor): {ra['created']} created, "
          f"{ra['entities_linked']} links")

    fact, _ = store.add_fact(
        "उपयोगकर्ता गूगल में काम करता है।", fact_type="state",
        t_mentioned="2023/06/10", source_session_id="hi-1",
        extraction_model="store-direct(gate-e-shape)", lang_source="hi")
    rc = linker.link_fact(fact.id, fact_text=fact.fact_text)
    print(f"linked: {rc['linked']}  skipped: {rc['skipped']}  "
          f"alias_edges: {rc['alias_edges']}")
    db = SessionLocal()
    for e in db.query(KnowledgeGraphEdge).filter(
            KnowledgeGraphEdge.relation_type == "ALIAS_OF").all():
        src = db.get(KnowledgeGraphNode, e.source_id)
        tgt = db.get(KnowledgeGraphNode, e.target_id)
        print(f"ALIAS_OF: {src.entity_text} = {tgt.entity_text} "
              f"(cosine {e.confidence:.4f})")
    db.close()
    for query_text in ("गूगल", "Google"):
        got = linker.facts_for_entity(query_text)
        print(f"facts_for_entity({query_text!r}): "
              f"{[f.fact_text[:60] for f in got]}")


if __name__ == "__main__":
    main()
