"""
AgentMem OS — Semantic Fact Store (Consolidation v2, Stage 1)
==============================================================
CRUD + lifecycle for SemanticFact rows: the true semantic tier.

Design rules enforced here (CONSOLIDATION_V2_DESIGN.md + build log; the G3
critic findings and their resolutions are recorded in the build log's
Stage 1 record):

  - NO locks. Every mutation is guarded at the DATABASE so the guarantees
    hold identically in-process and cross-process: inserts by the
    UNIQUE(scope_key, normalized_hash) constraint (losing a race falls
    back to re-affirmation), supersession by a rowcount-checked
    conditional UPDATE, re-affirmation by a relative increment + merge performed
    UNDER the SQLite write lock in one transaction (R4-1). (History: R1 measured the shared-connection StaticPool
    engine destroying concurrent writes — fixed in db/engine.py itself;
    an interim module lock proved both insufficient across processes and
    deadlock-prone against the SQLite write lock, R3 N1/N2, and was
    removed.)
  - Re-affirmation, not duplication: a fact stated again strengthens the
    ONE row (mention_count, last_confirmed_at, citation/session/lang
    union, t_occurred backfill for undated states) instead of adding
    another.
  - EVENTS dedup on (text + occurrence date): "went for a run" on Oct 1
    and on Oct 8 are two facts. Collapsing them under-counts exactly the
    aggregation questions this tier exists to answer (G3 finding 3).
  - Invalidate, don't delete: supersession is an ATOMIC conditional
    update (WHERE superseded_by IS NULL, rowcount-checked) — two racers
    cannot both win (G3 finding 2).
  - Scopes are composite: agent axis and user axis never collapse
    (G3 finding 4 — one agent serving many users must not mix their
    facts). Derive scope keys ONLY via make_scope_key().
  - Every "current facts" query says `superseded_by IS NULL` literally —
    SQLite matches partial-index predicates textually.
  - Callers may pass their own session (db=) to make a whole
    consolidation batch one transaction; the store then flushes and
    NEVER commits/closes — the caller owns the transaction boundary.
  - Transitions are synthesized at READ time from the supersession
    chain, never written into fact_text (design-freeze decision).
"""

import calendar
import hashlib
import re
from datetime import datetime

from loguru import logger
from sqlalchemy.exc import IntegrityError

FACT_TYPES = frozenset({"event", "state", "preference", "identity"})

# Event-status axis (F7, founder-resolved 2026-08-06). 'cancelled' is
# deliberately NOT accepted yet — it is Stage 4 judgment territory and a
# value the store accepts must have defined merge semantics, not reserved
# ones. Events default to 'occurred' (the extractor contract says events
# already happened); 'planned' = future-dated at mention time. NOT part
# of the dedup hash; re-affirmation upgrades planned→occurred only.
EVENT_STATUSES = frozenset({"occurred", "planned"})

# Concurrency model (G3 round 3): NO in-process lock. Every mutation is
# either constraint-guarded (insert → unique constraint, race falls back to
# re-affirmation), rowcount-guarded (supersede → conditional UPDATE), or
# write-lock-guarded (re-affirmation: relative increment acquires the
# SQLite write lock, then merges happen in the same transaction — R4-1).
# These guards work identically in-process and cross-process — the earlier
# module lock only protected one process, silently lost cross-process
# re-affirmations (R3 N1), and deadlocked against the SQLite write lock
# when a caller-batch session and a store-owned session interleaved (R3 N2).
_CHAIN_DEPTH_CAP = 1000

_DATE_FULL = re.compile(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?=$|\s)")
_DATE_PARTIAL = re.compile(r"^(\d{4})[/-](\d{1,2})$")


def make_scope_key(agent_id: str = None, user_id: str = None) -> str:
    """The ONLY way scope keys are derived — write path and read path must
    agree byte-for-byte. Composite so the agent and user axes never collapse
    (an agent named 'alice' and a user named 'alice' are different scopes,
    and a literal user_id 'global' is not the global scope)."""
    if agent_id is None and user_id is None:
        return "global"
    return f"a={agent_id or ''}\x1fu={user_id or ''}"


def normalize_date(raw: str, allow_partial: bool = True):
    """
    Normalize to sortable 'YYYY/MM/DD', calendar-validated (month 13,
    Feb 31, day 00 all raise — an 8B extractor WILL hallucinate dates and
    they must be loud, not stored). Partial 'YYYY/MM' returns the month as
    an explicit (first_day, last_day) interval so range queries can never
    lexically drop it. Returns (start, end_or_None).
    """
    raw = (raw or "").strip()
    m = _DATE_FULL.match(raw)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            datetime(y, mo, d)
        except ValueError:
            raise ValueError(f"impossible calendar date: {raw!r}")
        return f"{y}/{mo:02d}/{d:02d}", None
    if allow_partial:
        m = _DATE_PARTIAL.match(raw)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if not 1 <= mo <= 12:
                raise ValueError(f"impossible month: {raw!r}")
            last = calendar.monthrange(y, mo)[1]
            return f"{y}/{mo:02d}/01", f"{y}/{mo:02d}/{last:02d}"
    raise ValueError(f"unparseable date: {raw!r}")


def normalize_fact_text(text: str) -> str:
    """Cosmetic-only normalization behind the dedup hash. 'rode 3 times'
    and 'rode 2 times' must NEVER normalize together."""
    t = re.sub(r"\s+", " ", text.strip().lower())
    return t.rstrip(".!,;: ")


def fact_hash(text: str, fact_type: str = "state", t_occurred: str = None,
              t_occurred_end: str = None) -> str:
    """Dedup identity = (type, text[, occurrence dates for events]).

    fact_type is ALWAYS in the key (R3 N3: without it, "The user is
    vegetarian." typed state in one session and preference in another was
    silently absorbed into whichever arrived first, and typed retrieval
    under-returned). Cross-type merging is a semantic judgment for Stage
    2's LLM adjudication, never a hash accident.

    Events additionally carry their occurrence date WITH its precision:
    the same sentence on two dates is two occurrences, and "sometime in
    October" (interval) is a different claim from "on October 1st" (point)
    (R1 F3, R2 B4)."""
    key = fact_type + "\x1f" + normalize_fact_text(text)
    if fact_type == "event":
        key += "\x1f" + (t_occurred or "") + "\x1f" + (t_occurred_end or "")
    return hashlib.sha256(key.encode()).hexdigest()


def _escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class SemanticFactStore:
    """
    Constructed with a session factory (same pattern as
    EntityKnowledgeGraph) so tests can bind an isolated engine.
    """

    def __init__(self, get_db_session):
        self.get_db = get_db_session

    # ── Session plumbing ────────────────────────────────────────────────────

    def _acquire(self, db):
        """(session, owns_it). Caller-supplied sessions are never
        committed or closed here — the caller owns the transaction."""
        if db is not None:
            return db, False
        return self.get_db(), True

    # ── Write path ──────────────────────────────────────────────────────────

    def add_fact(
        self,
        fact_text: str,
        fact_type: str,
        t_mentioned: str,
        source_session_id: str,
        source_turn_ids: list = None,
        t_occurred: str = None,
        entities: list = None,
        lang_source: str = "en",
        extraction_model: str = "unknown",
        agent_id: str = None,
        user_id: str = None,
        event_status: str = None,
        db=None,
    ):
        """
        Insert a fact, or re-affirm the existing one if this scope already
        holds it. Returns (fact, created: bool). All validation happens
        BEFORE any write.

        event_status (F7): events only — 'occurred' | 'planned'; None on
        an event defaults to 'occurred' so the invariant "every event
        carries a status" holds at the STORE, not by caller courtesy.
        Non-events must pass None (a status on a preference is a caller
        bug, loud).
        """
        from agentmem_os.db.models import SemanticFact

        if not fact_text or not fact_text.strip():
            raise ValueError("fact_text must be non-empty")
        if fact_type not in FACT_TYPES:
            raise ValueError(f"fact_type must be one of {sorted(FACT_TYPES)}, got {fact_type!r}")
        if source_turn_ids is not None and not all(
                type(i) is int for i in source_turn_ids):
            raise ValueError("source_turn_ids must be a list of turn row ids (ints, not bools)")
        if fact_type == "event":
            if event_status is None:
                event_status = "occurred"
            elif event_status not in EVENT_STATUSES:
                raise ValueError(
                    f"event_status must be one of {sorted(EVENT_STATUSES)}, "
                    f"got {event_status!r} ('cancelled' is Stage 4 — the "
                    "store only accepts statuses whose merge semantics exist)")
        elif event_status is not None:
            raise ValueError(
                f"event_status is an event axis; {fact_type!r} facts must not carry one")
        t_mentioned, _ = normalize_date(t_mentioned, allow_partial=False)
        t_occ_start = t_occ_end = None
        if t_occurred is not None:
            t_occ_start, t_occ_end = normalize_date(t_occurred)

        scope_key = make_scope_key(agent_id, user_id)
        h = fact_hash(fact_text, fact_type, t_occ_start, t_occ_end)

        session, owns = self._acquire(db)
        try:
                existing = (
                    session.query(SemanticFact)
                    .filter(SemanticFact.scope_key == scope_key,
                            SemanticFact.normalized_hash == h)
                    .first()
                )
                if existing is not None:
                    return self._reaffirm(
                        session, owns, existing, source_session_id,
                        source_turn_ids, entities, t_occ_start, t_occ_end,
                        lang_source, event_status,
                    ), False

                fact = SemanticFact(
                    agent_id=agent_id,
                    user_id=user_id,
                    scope_key=scope_key,
                    fact_text=fact_text.strip(),
                    fact_type=fact_type,
                    t_occurred=t_occ_start,
                    t_occurred_end=t_occ_end,
                    t_mentioned=t_mentioned,
                    event_status=event_status,
                    source_session_id=source_session_id,
                    source_session_ids=[source_session_id],
                    source_turn_ids=list(source_turn_ids or []),
                    entities=list(entities or []),
                    lang_source=lang_source,
                    langs=[lang_source],
                    extraction_model=extraction_model,
                    normalized_hash=h,
                )
                session.add(fact)
                try:
                    if owns:
                        session.commit()
                    else:
                        session.flush()
                except IntegrityError as e:
                    # Only the dedup constraint means "lost a cross-process
                    # race" — anything else (FK violation, locked DB) is a
                    # REAL failure and must stay loud (G3 finding 5).
                    # SQLite names the COLUMNS, not the constraint, in its
                    # error text — match both spellings.
                    msg = str(e.orig)
                    is_dedup_hit = (
                        "uq_facts_scope_hash" in msg
                        or ("UNIQUE constraint failed" in msg
                            and "normalized_hash" in msg)
                    )
                    if not is_dedup_hit:
                        if owns:
                            session.rollback()
                        raise
                    if not owns:
                        # Caller owns the transaction; we cannot roll back
                        # just our part. Surface it.
                        raise
                    session.rollback()
                    existing = (
                        session.query(SemanticFact)
                        .filter(SemanticFact.scope_key == scope_key,
                                SemanticFact.normalized_hash == h)
                        .first()
                    )
                    if existing is None:
                        raise
                    logger.debug("[SemanticFactStore] insert race → re-affirmation")
                    return self._reaffirm(
                        session, owns, existing, source_session_id,
                        source_turn_ids, entities, t_occ_start, t_occ_end,
                        lang_source, event_status,
                    ), False
                if owns:
                    session.refresh(fact)
                return fact, True
        finally:
            if owns:
                session.close()

    @staticmethod
    def _reaffirm(session, owns, fact, source_session_id, source_turn_ids,
                  entities, t_occ_start, t_occ_end, lang_source,
                  event_status=None):
        """
        Deterministic re-affirmation (R4-1/R5-1): a RELATIVE increment of
        mention_count acquires the SQLite write lock; the re-read and the
        citation/session/language merges then happen INSIDE that same
        transaction, where no other writer can interleave. No retry loop,
        no version check. Store-owned sessions first end their read
        snapshot (rollback) so the write transaction starts fresh. Only
        the affected fact is expired — never the caller's identity map.
        (Precisely: a caller's unflushed edit to OTHER objects survives;
        an unflushed edit to THIS fact is overwritten by the merge — this
        row is the one the store is contractually merging.)
        """
        from agentmem_os.db.models import SemanticFact

        fact_id = fact.id
        # R4-1: the earlier optimistic CAS read and wrote in SEPARATE
        # transactions — under sustained contention the retry budget
        # exhausted (~2%/call, loudly). Deterministic shape instead:
        #   1. relative increment (always exact, no version needed) — this
        #      UPDATE acquires the SQLite write lock;
        #   2. re-read + JSON merges UNDER that write lock, same txn —
        #      no other writer can interleave until commit.
        # Store-owned sessions first end their read snapshot so the write
        # txn starts fresh (no stale-snapshot BUSY). Caller-owned batches
        # already hold their txn; a cross-process commit landing mid-batch
        # surfaces as a loud OperationalError to the batch owner.
        if owns:
            session.rollback()
        from sqlalchemy import func
        inc = {"mention_count": func.coalesce(SemanticFact.mention_count, 1) + 1,
               "last_confirmed_at": datetime.utcnow()}
        updated = (session.query(SemanticFact)
                   .filter(SemanticFact.id == fact_id)
                   .update(inc, synchronize_session=False))
        if updated != 1:
            raise ValueError(f"fact {fact_id} vanished during re-affirmation")
        # R5-1: expire ONLY the row being re-affirmed. expire_all() wiped
        # the caller's whole identity map — unflushed caller edits in a
        # batch vanished silently while the store reported success.
        session.expire(fact)
        fresh = session.query(SemanticFact).filter(
            SemanticFact.id == fact_id).first()
        sessions_l = list(fresh.source_session_ids or [fresh.source_session_id])
        if source_session_id and source_session_id not in sessions_l:
            sessions_l.append(source_session_id)
        turns_l = list(fresh.source_turn_ids or [])
        turns_l += [t for t in (source_turn_ids or []) if t not in turns_l]
        ents_l = list(fresh.entities or [])
        ents_l += [e for e in (entities or []) if e not in ents_l]
        langs_l = list(fresh.langs or [fresh.lang_source])
        if lang_source and lang_source not in langs_l:
            langs_l.append(lang_source)
        values = {
            "source_session_ids": sessions_l,
            "source_turn_ids": turns_l,
            "entities": ents_l,
            "langs": langs_l,
        }
        # Backfill an occurrence date a later mention supplies. Events
        # never reach here with a different date (it is in their hash).
        if fresh.t_occurred is None and t_occ_start is not None:
            values["t_occurred"] = t_occ_start
            values["t_occurred_end"] = t_occ_end
        # F7 merge rule: planned→occurred only. A re-affirmation whose
        # mention time is past the event date arrives as 'occurred' —
        # the claim is no longer prospective from ANY source's view. The
        # reverse (a pre-date restatement of an already-occurred claim)
        # never downgrades. NULL on the row (pre-Stage-3 writer against
        # a migrated DB) backfills from the incoming value.
        if event_status is not None and fresh.fact_type == "event":
            # cancelled is TERMINAL against merges (Stage 4): a judged
            # cancellation is a deliberate decision — a later
            # restatement never silently un-cancels it. The shipped
            # reversal is reinstate_cancelled_event() below (R3-B1: a
            # one-way destructive transition with no API back is worse
            # than supersession, which chain() can always audit).
            if fresh.event_status != "cancelled" and (
                    fresh.event_status is None
                    or (fresh.event_status == "planned"
                        and event_status == "occurred")):
                values["event_status"] = event_status
        (session.query(SemanticFact)
         .filter(SemanticFact.id == fact_id)
         .update(values, synchronize_session=False))
        if owns:
            session.commit()
        session.refresh(fresh)
        return fresh

    def supersede(self, old_fact_id: int, new_fact_id: int,
                  t_invalid: str = None, db=None):
        """
        Mark old_fact superseded by new_fact. The write is an ATOMIC
        conditional update — `WHERE id = old AND superseded_by IS NULL`,
        rowcount-checked — so two racing supersedes cannot both win and a
        cycle cannot form through a race (G3 finding 2). Guards stay loud:
        distinct ids, both exist, same scope, not already superseded, no
        cycle. Many-to-one supersession (two facts merged into one) is
        ALLOWED — it is the shape Stage 2's dedup merge produces; chain()
        returns the full set.
        """
        from agentmem_os.db.models import SemanticFact

        if old_fact_id == new_fact_id:
            raise ValueError("a fact cannot supersede itself")
        # Stage 4: t_invalid = DOMAIN time the old fact stopped being
        # true (usually the superseding fact's own domain time) —
        # validated like every other date; distinct from superseded_at,
        # which is OUR decision time and is always stamped here.
        t_inv = None
        if t_invalid is not None:
            t_inv, _ = normalize_date(t_invalid)

        session, owns = self._acquire(db)
        try:
                old = session.query(SemanticFact).filter(
                    SemanticFact.id == old_fact_id).first()
                new = session.query(SemanticFact).filter(
                    SemanticFact.id == new_fact_id).first()
                if old is None or new is None:
                    raise ValueError(
                        f"fact not found (old={old_fact_id}: {old is not None}, "
                        f"new={new_fact_id}: {new is not None})")
                if old.scope_key != new.scope_key:
                    raise ValueError("cannot supersede across scopes "
                                     f"({old.scope_key!r} vs {new.scope_key!r})")
                if old.superseded_by is not None:
                    raise ValueError(
                        f"fact {old_fact_id} is already superseded by "
                        f"{old.superseded_by}; refusing to rewrite history")
                # Cycle guard: old must not be an ancestor-of/reachable from
                # new by following superseded_by forward.
                cursor, depth, seen = new, 0, set()
                while cursor is not None and cursor.superseded_by is not None:
                    depth += 1
                    if depth > _CHAIN_DEPTH_CAP or cursor.superseded_by in seen:
                        raise ValueError("supersession chain too deep or cyclic")
                    if cursor.superseded_by == old_fact_id:
                        raise ValueError("supersession would create a cycle")
                    seen.add(cursor.superseded_by)
                    cursor = session.query(SemanticFact).filter(
                        SemanticFact.id == cursor.superseded_by).first()

                updated = (
                    session.query(SemanticFact)
                    .filter(SemanticFact.id == old_fact_id,
                            SemanticFact.superseded_by.is_(None))
                    .update({"superseded_by": new_fact_id,
                             "superseded_at": datetime.utcnow(),
                             "t_invalid": t_inv},
                            synchronize_session=False)
                )
                if updated != 1:
                    if owns:
                        session.rollback()
                    raise ValueError(
                        f"fact {old_fact_id} was superseded concurrently; "
                        "refusing to overwrite")
                if owns:
                    session.commit()
                else:
                    session.flush()
        finally:
            if owns:
                session.close()

    def mark_event_cancelled(self, fact_id: int, db=None):
        """
        The ONLY path that writes event_status='cancelled' (Stage 4 —
        add_fact still refuses it as input; a status the store accepts
        must have defined transition semantics, and this is the whole
        definition): a LIVE event currently 'planned' becomes
        'cancelled'. occurred never cancels (a past claim), superseded
        facts are history (refused), and cancelled is TERMINAL — the
        re-affirmation merge never upgrades it (disclosed). Atomic
        conditional UPDATE, rowcount-checked, same discipline as
        supersede().
        """
        from agentmem_os.db.models import SemanticFact

        session, owns = self._acquire(db)
        try:
            fact = session.query(SemanticFact).filter(
                SemanticFact.id == fact_id).first()
            if fact is None:
                raise ValueError(f"fact {fact_id} not found")
            if fact.fact_type != "event" or fact.event_status != "planned":
                raise ValueError(
                    f"only a live PLANNED event can be cancelled; fact "
                    f"{fact_id} is {fact.fact_type}/{fact.event_status}")
            if fact.superseded_by is not None:
                raise ValueError(
                    f"fact {fact_id} is superseded history; refusing")
            updated = (
                session.query(SemanticFact)
                .filter(SemanticFact.id == fact_id,
                        SemanticFact.event_status == "planned",
                        SemanticFact.superseded_by.is_(None))
                .update({"event_status": "cancelled"},
                        synchronize_session=False)
            )
            if updated != 1:
                if owns:
                    session.rollback()
                raise ValueError(
                    f"fact {fact_id} changed concurrently; refusing")
            if owns:
                session.commit()
            else:
                session.flush()
        finally:
            if owns:
                session.close()

    def reinstate_cancelled_event(self, fact_id: int, db=None):
        """Operator escape hatch (R3-B1: mark_event_cancelled was the
        store's only one-way destructive transition — no API back).
        A LIVE cancelled event returns to 'planned'. Atomic conditional
        UPDATE, rowcount-checked, same discipline as the forward path.
        Deliberately NOT reachable from the judge — reinstatement is a
        human decision, informed by the audit row that cancelled it."""
        from agentmem_os.db.models import SemanticFact

        session, owns = self._acquire(db)
        try:
            fact = session.query(SemanticFact).filter(
                SemanticFact.id == fact_id).first()
            if fact is None:
                raise ValueError(f"fact {fact_id} not found")
            if fact.fact_type != "event" or fact.event_status != "cancelled":
                raise ValueError(
                    f"only a live CANCELLED event can be reinstated; "
                    f"fact {fact_id} is {fact.fact_type}/{fact.event_status}")
            if fact.superseded_by is not None:
                raise ValueError(
                    f"fact {fact_id} is superseded history; refusing")
            updated = (
                session.query(SemanticFact)
                .filter(SemanticFact.id == fact_id,
                        SemanticFact.event_status == "cancelled",
                        SemanticFact.superseded_by.is_(None))
                .update({"event_status": "planned"},
                        synchronize_session=False)
            )
            if updated != 1:
                if owns:
                    session.rollback()
                raise ValueError(
                    f"fact {fact_id} changed concurrently; refusing")
            if owns:
                session.commit()
            else:
                session.flush()
        finally:
            if owns:
                session.close()

    # ── Read path ───────────────────────────────────────────────────────────

    def current_facts(
        self,
        scope_key: str = None,
        fact_type: str = None,
        contains: str = None,
        limit: int = 100,
        agent_id: str = None,
        user_id: str = None,
        include_cancelled: bool = False,
        session_ids: list = None,
        db=None,
    ) -> list:
        """Live facts for a scope, newest event first (SQLite puts NULL
        t_occurred last in DESC natively — no leading expression, so the
        partial index serves the sort and LIMIT short-circuits). Pass either
        scope_key or agent_id/user_id; the latter derive via
        make_scope_key so read and write can never disagree."""
        from agentmem_os.db.models import SemanticFact

        if scope_key is None:
            scope_key = make_scope_key(agent_id, user_id)
        session, owns = self._acquire(db)
        try:
            q = (
                session.query(SemanticFact)
                .filter(SemanticFact.superseded_by.is_(None),
                        SemanticFact.scope_key == scope_key)
            )
            if not include_cancelled:
                # Stage 4 (G3 S4-R1 Ma3): a judged-cancelled planned
                # event is a VOIDED claim — surfacing it as "current"
                # misleads. It is not superseded (no successor fact),
                # so the status filter is its only reader-side guard.
                q = q.filter((SemanticFact.event_status.is_(None))
                             | (SemanticFact.event_status != "cancelled"))
            if session_ids is not None:
                # Session-scoped reads ("what does this project//haystack
                # know?"). CONSERVATIVE BY DESIGN: matches the PRIMARY
                # source_session_id only — a fact first learned outside
                # the given set but re-affirmed inside it is EXCLUDED.
                # Under-retrieval is the safe direction; the alternative
                # (JSON array membership on source_session_ids) is
                # unindexable in SQLite and would leak on a partial
                # match. Empty list means "no sessions" -> no facts,
                # never "unfiltered".
                q = q.filter(SemanticFact.source_session_id.in_(
                    list(session_ids)))
            if fact_type is not None:
                q = q.filter(SemanticFact.fact_type == fact_type)
            if contains:
                q = q.filter(SemanticFact.fact_text.ilike(
                    f"%{_escape_like(contains)}%", escape="\\"))
            q = q.order_by(SemanticFact.t_occurred.desc(),
                           SemanticFact.id.desc())
            return q.limit(limit).all()
        finally:
            if owns:
                session.close()

    def facts_as_of(self, as_of_date: str, scope_key: str = None,
                    fact_type: str = None, limit: int = 100,
                    agent_id: str = None, user_id: str = None,
                    db=None) -> list:
        """
        Point-in-time reconstruction: facts mentioned by the given date and
        not yet superseded in conversation time (their superseding fact, if
        any, was mentioned after it).

        DISCLOSED LIMIT (Stage 4): cancellation carries no decision-time
        axis of its own — a cancelled planned event appears cancelled at
        EVERY as-of date, including dates before the cancellation was
        judged. Modeling cancellation time is deferred until a reader
        needs it (Stage 5 territory).
        """
        from sqlalchemy.orm import aliased
        from agentmem_os.db.models import SemanticFact

        if scope_key is None:
            scope_key = make_scope_key(agent_id, user_id)
        d, _ = normalize_date(as_of_date, allow_partial=False)
        session, owns = self._acquire(db)
        try:
            successor = aliased(SemanticFact)
            q = (
                session.query(SemanticFact)
                .outerjoin(successor, SemanticFact.superseded_by == successor.id)
                .filter(SemanticFact.scope_key == scope_key,
                        SemanticFact.t_mentioned <= d)
                .filter((SemanticFact.superseded_by.is_(None))
                        | (successor.t_mentioned > d))
            )
            if fact_type is not None:
                q = q.filter(SemanticFact.fact_type == fact_type)
            return (q.order_by(SemanticFact.t_mentioned.desc(),
                               SemanticFact.id.desc())
                    .limit(limit).all())
        finally:
            if owns:
                session.close()

    def facts_overlapping(self, range_start: str, range_end: str,
                          scope_key: str = None, fact_type: str = None,
                          limit: int = 100, agent_id: str = None,
                          user_id: str = None, db=None) -> list:
        """
        Live facts whose occurrence interval overlaps [range_start,
        range_end] — the primitive behind time-aware query filtering
        (Stage 0 §1.5). Month-only facts participate via their explicit
        interval; undated facts are excluded by definition.
        """
        from sqlalchemy import func
        from agentmem_os.db.models import SemanticFact

        if scope_key is None:
            scope_key = make_scope_key(agent_id, user_id)
        a, _ = normalize_date(range_start, allow_partial=False)
        b, _ = normalize_date(range_end, allow_partial=False)
        session, owns = self._acquire(db)
        try:
            q = (
                session.query(SemanticFact)
                .filter(SemanticFact.superseded_by.is_(None),
                        SemanticFact.scope_key == scope_key,
                        SemanticFact.t_occurred.isnot(None),
                        SemanticFact.t_occurred <= b,
                        func.coalesce(SemanticFact.t_occurred_end,
                                      SemanticFact.t_occurred) >= a)
                # cancelled events never happened in the range (Ma3)
                .filter((SemanticFact.event_status.is_(None))
                        | (SemanticFact.event_status != "cancelled"))
            )
            if fact_type is not None:
                q = q.filter(SemanticFact.fact_type == fact_type)
            return (q.order_by(SemanticFact.t_occurred.asc(),
                               SemanticFact.id.asc())
                    .limit(limit).all())
        finally:
            if owns:
                session.close()

    def chain(self, fact_id: int, db=None) -> list:
        """
        The COMPLETE supersession lineage containing this fact — all
        ancestors (many-to-one merges included: G3 finding 16) and all
        successors — deterministically ordered by (t_mentioned, id).
        Seen-guards + depth caps on every walk: a corrupt chain terminates
        with what was reachable instead of hanging the audit path.
        """
        from agentmem_os.db.models import SemanticFact

        session, owns = self._acquire(db)
        try:
            root = session.query(SemanticFact).filter(
                SemanticFact.id == fact_id).first()
            if root is None:
                raise ValueError(f"fact {fact_id} not found")

            # Closure over BOTH directions: from any member, pull in every
            # predecessor (facts a member supersedes... i.e. facts whose
            # superseded_by points at a member) AND every successor (facts a
            # member points at). Iterate to fixpoint so a merge's sibling
            # branches are reached from ANY entry point — one-directional
            # walks miss them (G3 finding 16's re-test caught exactly that).
            members = {root.id: root}
            exhausted = True
            for _ in range(_CHAIN_DEPTH_CAP):
                ids = list(members.keys())
                grew = False
                preds = (session.query(SemanticFact)
                         .filter(SemanticFact.superseded_by.in_(ids))
                         .order_by(SemanticFact.id.asc()).all())
                succ_ids = [f.superseded_by for f in members.values()
                            if f.superseded_by is not None
                            and f.superseded_by not in members]
                succs = []
                if succ_ids:
                    succs = (session.query(SemanticFact)
                             .filter(SemanticFact.id.in_(succ_ids)).all())
                for row in list(preds) + list(succs):
                    if row.id not in members:
                        members[row.id] = row
                        grew = True
                if not grew:
                    exhausted = False
                    break
            if exhausted:
                # A truncated audit trail presented as complete is worse
                # than an error (G3 round 2, M3) — chains this long are
                # corruption or abuse, and the caller must know.
                raise ValueError(
                    f"supersession chain around fact {fact_id} exceeds "
                    f"{_CHAIN_DEPTH_CAP} links; refusing to return a "
                    "silently incomplete lineage")
            return sorted(members.values(),
                          key=lambda f: (f.t_mentioned, f.id))
        finally:
            if owns:
                session.close()

    def transition_text(self, fact_id: int, db=None) -> str:
        """Read-time transition synthesis (design-freeze decision): the
        readable change story, reconstructed from the chain without
        denormalizing the rows. Returns "" for chain-less facts."""
        links = self.chain(fact_id, db=db)
        if len(links) < 2:
            return ""
        parts = [links[0].fact_text]
        for nxt in links[1:]:
            when = nxt.t_occurred or nxt.t_mentioned
            parts.append(f"→ superseded {when}: {nxt.fact_text}")
        return " ".join(parts)

    def provenance(self, fact_id: int, db=None) -> dict:
        """Fact → every source session + cited turns, with an integrity
        flag: do the cited turn rows actually exist?"""
        from agentmem_os.db.models import SemanticFact, Turn

        session, owns = self._acquire(db)
        try:
            fact = session.query(SemanticFact).filter(
                SemanticFact.id == fact_id).first()
            if fact is None:
                raise ValueError(f"fact {fact_id} not found")
            cited = list(fact.source_turn_ids or [])
            found, user_found = [], []
            if cited:
                rows = session.query(Turn).filter(Turn.id.in_(cited)).all()
                found = [t.id for t in rows]
                user_found = [t.id for t in rows if t.role == "user"]
            return {
                "fact_id": fact.id,
                "fact_text": fact.fact_text,
                "source_session_id": fact.source_session_id,
                "source_session_ids": list(fact.source_session_ids
                                           or [fact.source_session_id]),
                "source_turn_ids": cited,
                "turns_resolved": found,
                "user_turns_resolved": user_found,   # R3-M3: ranked evidence
                                                     # spans roles; USER
                                                     # grounding shown apart
                # Three-state (Stage-2 G3-R1 B2): an empty citation set is
                # "uncited", never a clean bill of health.
                "citations_intact": ("uncited" if not cited
                                     else set(found) == set(cited)),
                "extraction_model": fact.extraction_model,
                "langs": list(fact.langs or [fact.lang_source]),
                "mention_count": fact.mention_count,
                "event_status": fact.event_status,   # F7 axis (events only)
                "t_invalid": fact.t_invalid,         # domain validity end (S4)
            }
        finally:
            if owns:
                session.close()
