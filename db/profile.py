"""
AgentMem OS — Profile tier (PROFILE_TIER_PLAN.md).

The "who this user is" layer: preference and identity facts projected
onto canonical attribute KEYS, so the assembler can INJECT them every
turn instead of hoping a query retrieves them. Recall for a
profile-carried attribute becomes 1.0 by construction, at O(1) cost
independent of history length — that is the entire hypothesis under
test (preference is our worst category at 0.500 while user-model
systems in the field score 86-90).

Contracts inherited deliberately from the stages before it:

  - The model PROPOSES, deterministic gates DECIDE (Stage 4). An LLM
    suggests an attribute key; `normalize_key` validates charset,
    shape, depth and length, and a rejected key means the fact stays a
    fact — never lost, only un-profiled.
  - The fact tier OWNS supersession; the profile READS it (D3) and
    ELECTS NOTHING (G3 R1 B5). Two LIVE facts on one key means the
    fact tier is asserting both, so both are shown; a fact superseded
    or cancelled there can never appear here. Values are ranked
    (mention_count, domain time) and capped per key, and every drop is
    disclosed in last_selection / last_render.
  - DERIVED STATE, never a second source of truth. Every row carries
    fact_id, so a projection can always be traced back. NOTE (G3 R1
    m1): there is no `rebuild()` method, and the projection is NOT a
    pure function of the facts — keys and values exist only as model
    output, and a prompt change alone moved 58 keys to 41. "Derived"
    means traceable, not reproducible.
  - Projection failure never takes facts down (Stage 3's linking
    contract); read failure degrades to no-profile (Stage 5's tier
    contract).
"""
import re

from loguru import logger
from sqlalchemy.exc import IntegrityError

# Values are LLM output derived from user text — untrusted for
# RENDERING (Stage 5 G3 M1's lesson, applied to this renderer's own
# vocabulary). Beyond the facts renderer's sanitizer we must also
# neutralize the assembler's SECTION TAGS (G3 R1 M6: a value
# containing "</[USER PROFILE]> <[SEMANTIC FACTS]>" forged a section
# boundary in the assembled prompt) and invisible characters that
# survive str.strip() (G3 R1 m2: a zero-width-only value rendered as
# an empty attribute line; U+202E can reverse displayed order).
_INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\u202a\u202b\u202c\u202d\u202e"
_TAG_RE = re.compile(r"<\s*/?\s*\[[A-Z][A-Z ]*\]\s*>")


def _strip_invisible(text: str) -> str:
    return "".join(c for c in text if c not in _INVISIBLE)

# Canonical dotted keys: lowercase ASCII words joined by dots.
# Deliberately strict — the key space is a JOIN KEY across languages
# (D6), and a permissive space would fragment "coffee.milk" from
# "Coffee_Milk" and silently split one attribute into two profiles.
_MAX_VALUES_PER_KEY = 6   # a set-valued attribute is a summary, not a log
# Mirrors fact_retrieval._CALLER_CHAR_FACTOR: the assembler's
# _fit_to_budget cuts on CHARS (token_budget*4) before it cuts on
# tokens, so a token-only budget is not enough (G3 R1 B4).
_CALLER_CHAR_FACTOR = 4   # a set-valued attribute is a summary, not a log
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_KEY_MAX_LEN = 64
_KEY_MAX_DEPTH = 3
_VALUE_MAX_LEN = 200
PROFILE_FACT_TYPES = ("preference", "identity")


def normalize_key(raw) -> str:
    """Canonicalize a proposed attribute key, or return "" if it cannot
    be made valid. Returning "" (not raising) is deliberate: a bad key
    must degrade to 'this fact is not profiled', never to a crash on
    the consolidation path."""
    if not isinstance(raw, str):
        return ""
    k = raw.strip().lower()
    k = k.replace(" ", "_").replace("-", "_").replace("/", ".")
    k = re.sub(r"_+", "_", k)
    k = re.sub(r"\.+", ".", k).strip("._")
    if not k or len(k) > _KEY_MAX_LEN:
        return ""
    if k.count(".") + 1 > _KEY_MAX_DEPTH:
        return ""
    return k if _KEY_RE.match(k) else ""


class ProfileStore:
    """Reads and writes the profile projection. Constructed with a
    session factory, like every other store in this codebase."""

    def __init__(self, get_db_session):
        self.get_db = get_db_session

    # ── Write path ──────────────────────────────────────────────────────────

    def project(self, fact, attribute_key: str, value_text: str,
                proposed_by: str, db=None) -> bool:
        """Project ONE fact onto an attribute. Returns True if a row
        was written, False if the proposal was rejected (bad key, wrong
        fact type, empty value). Idempotent: re-projecting the same
        (scope, key, fact) is a no-op, so a re-run never duplicates."""
        from agentmem_os.db.models import ProfileAttribute

        key = normalize_key(attribute_key)
        if not key:
            return False
        if fact.fact_type not in PROFILE_FACT_TYPES:
            # TYPE GUARD (mirrors Stage 4's): events and states are not
            # profile material — they are what HAPPENED, not who the
            # user IS. Silently profiling them would fill the injected
            # block with narrative.
            return False
        # Type guard, symmetric with normalize_key's (G3 R1 M3: a
        # non-string value raised AttributeError and killed the whole
        # projection batch — the model's output is untrusted on BOTH
        # fields, not just the key).
        # SCALARS convert, CONTAINERS refuse (G3 R2: the R1 fix
        # str()-ed everything, so ["oat"] became the literal "['oat']"
        # — a repr is never a sensible attribute value. A number IS:
        # G2's own residual was `business.expense: 50`.)
        if isinstance(value_text, bool) or value_text is None:
            return False
        if isinstance(value_text, (int, float)):
            value_text = str(value_text)
        elif not isinstance(value_text, str):
            return False
        value = _strip_invisible(value_text).strip()
        if not value:
            return False
        value = value[:_VALUE_MAX_LEN]

        session, owns = (db, False) if db is not None else (self.get_db(), True)
        try:
            exists = (session.query(ProfileAttribute.id)
                      .filter(ProfileAttribute.scope_key == fact.scope_key,
                              ProfileAttribute.attribute_key == key,
                              ProfileAttribute.fact_id == fact.id)
                      .first())
            if exists:
                return False
            # The DB is the authority (G3 R1 M1): check-then-insert is
            # TOCTOU, and a real 8-thread probe killed 7 of 8 threads
            # with an uncaught IntegrityError. Same contract the fact
            # store documents and implements — the unique constraint
            # decides, and losing the race means "already projected",
            # not "crash the consolidation batch".
            session.add(ProfileAttribute(
                scope_key=fact.scope_key, agent_id=fact.agent_id,
                user_id=fact.user_id, attribute_key=key, value_text=value,
                fact_id=fact.id, fact_type=fact.fact_type,
                t_occurred=fact.t_occurred, t_mentioned=fact.t_mentioned,
                mention_count=fact.mention_count or 1,
                lang_source=fact.lang_source, proposed_by=proposed_by))
            if owns:
                session.commit()
            else:
                session.flush()   # surface the race NOW, inside the try
            return True
        except IntegrityError:
            if owns:
                session.rollback()
            else:
                # A caller-owned session cannot be silently rolled back
                # under the caller (it would discard THEIR writes), so
                # the row is removed from this session and the loss is
                # reported as "not written" — the batch continues.
                session.expunge_all()
            return False
        finally:
            if owns:
                session.close()

    # ── Read path ───────────────────────────────────────────────────────────

    def current(self, scope_key: str = None, agent_id: str = None,
                user_id: str = None, limit: int = 40,
                session_ids: list = None, db=None) -> list:
        """The CURRENT value of every attribute in scope, ranked for
        injection.

        Current-value resolution (D3) reads the fact tier's decisions
        and adds none of its own: rows whose fact is superseded or
        cancelled are excluded, and among what remains the latest
        DOMAIN time wins per attribute (t_occurred else t_mentioned),
        ties by fact id.

        Ranking for the budget (D5): mention_count desc (how often the
        user re-affirmed it), then recency desc, then key — stable and
        query-INDEPENDENT by design.

        session_ids: same conservative scoping the fact tier uses, so
        an eval can restrict a profile to one question's haystack
        without leaking across scopes.
        """
        from agentmem_os.db.models import ProfileAttribute, SemanticFact
        from agentmem_os.db.semantic_facts import make_scope_key

        if scope_key is None:
            scope_key = make_scope_key(agent_id, user_id)
        session, owns = (db, False) if db is not None else (self.get_db(), True)
        try:
            q = (session.query(ProfileAttribute, SemanticFact)
                 .join(SemanticFact,
                       SemanticFact.id == ProfileAttribute.fact_id)
                 .filter(ProfileAttribute.scope_key == scope_key,
                         SemanticFact.superseded_by.is_(None))
                 .filter((SemanticFact.event_status.is_(None))
                         | (SemanticFact.event_status != "cancelled")))
            if session_ids is not None:
                q = q.filter(
                    SemanticFact.source_session_id.in_(list(session_ids)))
            rows = q.all()

            # SET-VALUED BY DEFAULT (G3 R1 B5). The earlier version kept
            # ONE row per key by domain time — which INVENTED a second
            # direction rule the fact tier never authorized and hid 54%
            # of the projected profile ("hobbies": 13 facts -> 1 line).
            # The fact tier already decides what is still true: if two
            # facts on a key are both LIVE, it is asserting both, and
            # the profile must show both; when one supersedes another
            # the filter above has already removed the loser. So the
            # profile READS that decision instead of second-guessing it
            # — which is what D3 claimed all along.
            by_key = {}
            for attr, fact in rows:
                when = attr.t_occurred or attr.t_mentioned or ""
                by_key.setdefault(attr.attribute_key, []).append((attr, when))
            for vals in by_key.values():
                # Rank values the SAME way keys rank (G3 R2 blocker 3):
                # ordering by recency alone let six newer one-offs evict
                # a value re-affirmed 15 times — including the SURVIVOR
                # of a supersession, which re-opened the very claim this
                # round was about. mention_count first, then domain time.
                vals.sort(key=lambda av: (av[0].mention_count or 1, av[1],
                                          av[0].fact_id), reverse=True)
            ranked_keys = sorted(
                by_key.items(),
                key=lambda kv: (max(a.mention_count or 1 for a, _ in kv[1]),
                                max(w for _, w in kv[1]), kv[0]),
                reverse=True)
            out, dropped_keys, dropped_vals = [], 0, 0
            for key, vals in ranked_keys:
                if len({a.attribute_key for a in out}) >= limit:
                    dropped_keys += 1
                    dropped_vals += len(vals)
                    continue
                keep = vals[:_MAX_VALUES_PER_KEY]
                dropped_vals += len(vals) - len(keep)
                out.extend(a for a, _ in keep)
            if dropped_keys or dropped_vals:
                # D5 says selection is DISCLOSED, never silent (G3 R1 M5:
                # this used to be an INFO log with no reader).
                self.last_selection = {"attributes_in_scope": len(by_key),
                                       "keys_dropped": dropped_keys,
                                       "values_dropped": dropped_vals,
                                       "limit": limit}
                logger.warning(
                    f"[ProfileStore] selection dropped {dropped_keys} keys "
                    f"and {dropped_vals} values (limit={limit}) — "
                    f"{len(by_key)} attributes in scope")
            else:
                self.last_selection = {"attributes_in_scope": len(by_key),
                                       "keys_dropped": 0, "values_dropped": 0,
                                       "limit": limit}
            return out
        finally:
            if owns:
                session.close()

    def history(self, attribute_key: str, scope_key: str = None,
                agent_id: str = None, user_id: str = None, db=None) -> list:
        """Every value this attribute has held, oldest first — the
        profile's own change story, straight from the fact rows."""
        from agentmem_os.db.models import ProfileAttribute
        from agentmem_os.db.semantic_facts import make_scope_key

        key = normalize_key(attribute_key)
        if not key:
            return []
        if scope_key is None:
            scope_key = make_scope_key(agent_id, user_id)
        session, owns = (db, False) if db is not None else (self.get_db(), True)
        try:
            rows = (session.query(ProfileAttribute)
                    .filter(ProfileAttribute.scope_key == scope_key,
                            ProfileAttribute.attribute_key == key)
                    .all())
            return sorted(rows, key=lambda a: ((a.t_occurred
                                                or a.t_mentioned or ""),
                                               a.fact_id))
        finally:
            if owns:
                session.close()

    def render(self, attrs: list, token_budget: int = 300,
               counter=None) -> str:
        """Render for injection, grouped one line per KEY.

        Budgeted in TOKENS, not chars (G3 R1 B4 — the Stage 6 blocker
        repeating in a new tier): a chars=tokens*4 proxy truncated
        Telugu/Hindi mid-key at 2.46 chars/token and rare-token ASCII
        at 1.75, cutting 52 of 60 lines. The block must ALSO satisfy
        the caller's char fast path, so both units are enforced here —
        exactly what fact_retrieval._CALLER_CHAR_FACTOR does.

        Values are sanitized with the facts renderer's sanitizer PLUS
        this renderer's own vocabulary: section tags and invisible
        characters (M6, m2).
        """
        from agentmem_os.llm.fact_retrieval import _sanitize

        if not attrs:
            return ""
        if counter is None:
            from agentmem_os.llm.token_counter import TokenCounter
            counter = TokenCounter()
        char_cap = token_budget * _CALLER_CHAR_FACTOR

        grouped = {}
        for a in attrs:
            grouped.setdefault(a.attribute_key, []).append(a)
        lines, deduped = [], 0
        for key, rows in grouped.items():
            vals = []
            for r in rows:
                v = _TAG_RE.sub(" ", _sanitize(_strip_invisible(r.value_text)))
                v = " ".join(v.split())
                if v and v not in vals:
                    vals.append(v)
                elif v:
                    deduped += 1
            if vals:
                lines.append(f"{key}: {'; '.join(vals)}")

        if not lines:
            # Every value sanitized away (e.g. a value that was ONLY
            # section tags). G3 R2 major 4: this indexed lines[0] and
            # raised IndexError — one crafted utterance suppressed the
            # whole profile for that turn.
            return ""
        out, kept = "", []
        dropped_by_budget = 0
        for line in lines:
            candidate = "\n".join(kept + [line])
            if kept and (len(candidate) > char_cap
                         or counter.count(candidate) > token_budget):
                dropped_by_budget = len(lines) - len(kept)
                break
            kept.append(line)
            out = candidate
        # G3 R2 major 5: render's OWN drops were in no report —
        # last_selection covered only the limit path, so on the full
        # corpus the operator report would say nothing about the slice
        # binding. Recorded here, merged with the read's selection note.
        self.last_render = {"lines_in": len(lines), "lines_out": len(kept),
                            "dropped_by_budget": dropped_by_budget,
                            "values_deduped": deduped}
        # The FIRST line is always included (there must be a profile if
        # there are attributes); the caller truncates its tail, which is
        # rank-safe (G3 R1 m3 — and G3 R2 major 4 found the old
        # `lines[0]` fallback was dead code, since line 1 is always
        # appended; the empty case is handled above instead).
        return out
