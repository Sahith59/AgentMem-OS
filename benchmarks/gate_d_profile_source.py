"""
Gate D wiring for the PROFILE tier — the profile equivalent of
`gate_c_facts_source`, built because R2-blocker-2 named its absence:
`profile_scoped_required` closes the leak half, but nothing set it,
nothing projected profile rows into the eval's corpus, and there was
no preflight that STOPS before a paid call.

Three jobs, in this order:
  1. PROJECT — run the extractor over the corpus's preference/identity
     facts (local model, $0) so a profile exists at all.
  2. PREFLIGHT — return False, loudly, if the profile is empty, if
     scoping is not registered for every question, or if the corpus is
     missing. `qa_accuracy_eval` refuses to spend when this is False.
  3. INSTALL — bind a per-question scope and turn scoping REFUSAL on,
     so an unregistered question can never read an unscoped profile
     (the Gate C leakage class, in the tier that is INJECTED and would
     therefore leak into every single answer).

The corpus is opened read-only for reads; projection uses its own
writable handle and is idempotent, so a re-run costs nothing.
"""
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = HERE / "extracted_memories" / "gate_c_facts.db"


def _sessionmaker(corpus: Path, read_only: bool):
    from sqlalchemy import create_engine, event
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(f"sqlite:///{corpus}",
                           connect_args={"check_same_thread": False,
                                         "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection, _):
        cur = dbapi_connection.cursor()
        if read_only:
            cur.execute("PRAGMA query_only=ON")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    return sessionmaker(bind=engine, expire_on_commit=False)


def project(corpus: Path = CORPUS, limit: int = 100000) -> dict:
    """Populate the profile from the corpus's facts. Idempotent.

    Creates `profile_attributes` if the corpus predates the tier
    (G3 R3 blocker 1: the Gate C corpus has 17 tables and this is not
    one of them, so project() raised OperationalError on the only
    database it exists to serve — the wiring was written and the entry
    point was dead)."""
    from agentmem_os.db.models import Base, ProfileAttribute
    from agentmem_os.llm.profile_extractor import ProfileExtractor

    Session = _sessionmaker(corpus, read_only=False)
    engine = Session.kw["bind"]
    ProfileAttribute.__table__.create(bind=engine, checkfirst=True)
    return ProfileExtractor(Session).project_scope(limit=limit)


def stats(corpus: Path = CORPUS) -> dict:
    con = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True)
    try:
        q = con.execute
        if not q("SELECT 1 FROM sqlite_master WHERE type='table' AND "
                 "name='profile_attributes'").fetchone():
            return {"profile_rows": 0, "keys": 0, "sessions": 0}
        return {
            "profile_rows": q("SELECT COUNT(*) FROM profile_attributes"
                              ).fetchone()[0],
            "keys": q("SELECT COUNT(DISTINCT attribute_key) FROM "
                      "profile_attributes").fetchone()[0],
            "sessions": q("SELECT COUNT(DISTINCT f.source_session_id) FROM "
                          "profile_attributes p JOIN semantic_facts f ON "
                          "f.id = p.fact_id").fetchone()[0],
        }
    finally:
        con.close()


def preflight(scope_keys_by_question: dict, corpus: Path = CORPUS) -> bool:
    """$0 gate. Returns False — loudly — rather than letting a paid run
    measure a silently absent or silently unscoped profile."""
    print("=== GATE D PROFILE PREFLIGHT ($0) ===")
    if not corpus.exists():
        print(f"  FAIL: corpus missing at {corpus}")
        return False
    st = stats(corpus)
    for k, v in st.items():
        print(f"  {k}: {v}")
    if st["profile_rows"] == 0:
        print("  FAIL: profile is EMPTY — run project() first, or the run "
              "would measure a system without the tier under test")
        return False
    unregistered = [q for q, s in scope_keys_by_question.items() if not s]
    print(f"  questions with a registered haystack: "
          f"{len(scope_keys_by_question) - len(unregistered)}"
          f"/{len(scope_keys_by_question)}")
    if unregistered:
        print(f"  FAIL: {len(unregistered)} questions have no scope — an "
              "unscoped profile injects every question's attributes into "
              "every answer")
        return False
    # COVERAGE, not presence (G3 R3 blocker 2): 3 rows covering 2 of
    # 2,965 sessions passed the old check with all 150 questions
    # "scoped" — the paid run would have measured the tier's ABSENCE
    # and reported it as the tier's effect. The facts tier checks the
    # union of question haystacks against what was actually built; so
    # must this one.
    con = sqlite3.connect(f"file:{corpus}?mode=ro", uri=True)
    try:
        profiled_sessions = {r[0] for r in con.execute(
            "SELECT DISTINCT f.source_session_id FROM profile_attributes p "
            "JOIN semantic_facts f ON f.id = p.fact_id")}
    finally:
        con.close()
    per_q = [len(set(s) & profiled_sessions) for s in
             scope_keys_by_question.values()]
    empty_q = sum(1 for n in per_q if n == 0)
    print(f"  questions whose haystack contains PROFILED sessions: "
          f"{len(per_q) - empty_q}/{len(per_q)} "
          f"(median profiled sessions/question: "
          f"{sorted(per_q)[len(per_q) // 2] if per_q else 0})")
    if empty_q:
        print(f"  FAIL: {empty_q} questions would see an EMPTY profile — "
              "the run would measure the tier's absence and report it as "
              "the tier's effect")
        return False
    print(f"  PREFLIGHT PASS ({len(per_q)} questions scoped and covered, "
          f"{st['profile_rows']} attribute rows)")
    return True


def install(assembler, scope_keys_by_question: dict,
            corpus: Path = CORPUS) -> str:
    """Bind the corpus + per-question scope, and turn REFUSAL on."""
    from agentmem_os.db.profile import ProfileStore

    Session = _sessionmaker(corpus, read_only=True)

    class _ScopedProfileStore(ProfileStore):
        """Scope is resolved from the REGISTERED MAP, never from
        caller-set external state (G3 R3 major 4: install() ignored
        its own scope_keys_by_question, so per-question binding lived
        in a mutable attribute and a STALE scope passed the is-None
        check). Same shape as _ScopedFactRetriever."""

        def __init__(self, get_db, scope_map):
            super().__init__(get_db)
            self._scope_map = scope_map
            self.current_question = None

        def current(self, scope_key=None, agent_id=None, user_id=None,
                    limit=40, session_ids=None, db=None):
            sids = session_ids
            if self.current_question is not None:
                sids = self._scope_map.get(self.current_question)
                if sids is None:
                    raise KeyError(
                        "Gate D: no haystack registered for this question "
                        "— refusing to read an UNSCOPED profile (it is "
                        "INJECTED, so it would leak into every answer)")
            return super().current(scope_key=None, agent_id=None,
                                   user_id=None, limit=limit,
                                   session_ids=sids, db=db)

    assembler._profile = _ScopedProfileStore(Session,
                                             scope_keys_by_question)
    assembler.profile_scoped_required = True     # unscoped => REFUSE
    st = stats(corpus)
    return (f"profile-source=gate_c_facts.db rows={st['profile_rows']} "
            f"keys={st['keys']} scoped-per-question=yes refuse-if-unscoped=yes")


if __name__ == "__main__":
    # G3 R3 blocker 1: there was no supported way to RUN the projection.
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    print(f"projecting up to {n} facts into {CORPUS} ...")
    print(project(limit=n))
    print(stats())
