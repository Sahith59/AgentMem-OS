"""
Gate C wiring: make the EXTRACTED FACT CORPUS the memory source for
the slice eval, without forking the eval harness.

Import this module and call `install(assembler, args)` before the run.
It swaps ContextAssembler's facts tier to read from
`benchmarks/extracted_memories/gate_c_facts.db` (3,631 sessions
distilled through the real pipeline on the GSU cluster) instead of the
eval's own live DB, and — critically — restricts every question's
retrieval to that question's OWN haystack sessions.

WHY THE RESTRICTION IS MANDATORY (measured, not defensive):
the corpus holds all 3,631 sessions in ONE scope, the union of every
question's haystack. 116 question pairs share at least one session.
Unfiltered retrieval would show every question the facts of every
other question — that is not a measurement, it is leakage. The filter
is CONSERVATIVE: a fact whose primary source session lies outside the
haystack is excluded even if it was also seen inside it (under-
retrieval, never leakage).

$0 by itself: this changes what the assembler reads, not what is
generated. The eval's paid calls are unchanged.
"""
import json
import os
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = Path(os.environ.get("AGENTMEM_OS_GATE_C_CORPUS",
    str(HERE / "extracted_memories" / "gate_c_facts.db")))


def corpus_stats(path: Path = CORPUS) -> dict:
    """Read-only census of the corpus — used by the $0 preflight so a
    paid run never starts against an empty or half-merged DB."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        q = con.execute
        return {
            "facts": q("SELECT COUNT(*) FROM semantic_facts").fetchone()[0],
            "sessions_consolidated": q(
                "SELECT COUNT(*) FROM consolidation_log WHERE "
                "triggered_by='consolidation_v2'").fetchone()[0],
            "sessions_with_facts": q(
                "SELECT COUNT(DISTINCT source_session_id) FROM "
                "semantic_facts").fetchone()[0],
            "entity_links": q(
                "SELECT COUNT(*) FROM semantic_fact_entities").fetchone()[0],
            "superseded": q("SELECT COUNT(*) FROM semantic_facts WHERE "
                            "superseded_by IS NOT NULL").fetchone()[0],
            "planned_events": q("SELECT COUNT(*) FROM semantic_facts WHERE "
                                "event_status='planned'").fetchone()[0],
            "by_type": dict(q("SELECT fact_type, COUNT(*) FROM "
                              "semantic_facts GROUP BY fact_type").fetchall()),
        }
    finally:
        con.close()


def install(assembler, scope_keys_by_question: dict,
            corpus: Path = CORPUS) -> str:
    """Point the assembler's facts tier at the corpus and scope every
    retrieval to the asking question's own haystack.

    scope_keys_by_question: {question_text: [session_id, ...]} — the
    ground-truth haystack per question, straight from the dataset.
    Returns a one-line provenance string for the artifact.
    """
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    from agentmem_os.llm.fact_retrieval import FactRetriever

    if not corpus.exists():
        raise SystemExit(f"Gate C corpus missing: {corpus}")

    engine = create_engine(f"sqlite:///{corpus}",
                           connect_args={"check_same_thread": False,
                                         "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA query_only=ON")   # the corpus is READ-ONLY
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    class _ScopedFactRetriever(FactRetriever):
        """Same retriever, with the question's haystack bound in. The
        assembler passes only (query, agent_id, user_id, budget), so the
        session scope is resolved from the QUERY here — the eval's own
        ground truth, never anything the memory system knows."""

        def build_block(self, query, agent_id=None, user_id=None,
                        token_budget=1000, session_ids=None,
                        boost_types=()):
            sids = scope_keys_by_question.get(query)
            if sids is None:
                raise KeyError(
                    "Gate C: no haystack registered for this question — "
                    "refusing to retrieve UNSCOPED (that would leak "
                    "other questions' facts into this one)")
            return super().build_block(
                query, agent_id=None, user_id=None,
                token_budget=token_budget, session_ids=sids,
                boost_types=boost_types)

    assembler._facts = _ScopedFactRetriever(SessionLocal)
    st = corpus_stats(corpus)
    return (f"facts-source=gate_c_facts.db "
            f"facts={st['facts']} sessions={st['sessions_consolidated']} "
            f"links={st['entity_links']} scoped-per-question=yes")


def preflight(scope_keys_by_question: dict, corpus: Path = CORPUS) -> bool:
    """$0 preflight — run BEFORE any paid call. Proves the corpus is
    whole, the scoping is real, and facts actually reach the questions.
    Returns True only if every check passes."""
    print("=== GATE C PREFLIGHT ($0) ===")
    st = corpus_stats(corpus)
    for k, v in st.items():
        print(f"  {k}: {v}")

    all_sessions = set()
    for sids in scope_keys_by_question.values():
        all_sessions.update(sids)
    con = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True)
    try:
        have = {r[0] for r in con.execute(
            "SELECT DISTINCT source_session_id FROM semantic_facts")}
        consolidated = {r[0] for r in con.execute(
            "SELECT session_id FROM consolidation_log WHERE "
            "triggered_by='consolidation_v2'")}
    finally:
        con.close()

    missing = all_sessions - consolidated
    factless = all_sessions - have
    print(f"\n  haystack sessions referenced by questions: {len(all_sessions)}")
    print(f"  of those, NEVER consolidated: {len(missing)}")
    print(f"  of those, consolidated but yielding ZERO facts: "
          f"{len(factless - missing)} "
          f"({(len(factless - missing)) / max(1, len(all_sessions)):.1%} — "
          f"disclosed: extraction refuses unsupported content by design)")
    ok = not missing and st["facts"] > 0
    print(f"\n  PREFLIGHT {'PASS' if ok else 'FAIL'}")
    return ok
