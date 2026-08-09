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
  - The fact tier OWNS supersession; the profile READS it (D3). No
    second direction rule is invented here: current value = latest by
    DOMAIN time (t_occurred else t_mentioned), ties by fact id, and a
    fact already superseded in the fact tier can never be current.
  - DERIVED STATE, never a second source of truth. Every row carries
    fact_id; `rebuild` can reconstruct the whole profile from facts.
  - Projection failure never takes facts down (Stage 3's linking
    contract); read failure degrades to no-profile (Stage 5's tier
    contract).
"""
import re

from loguru import logger

# Canonical dotted keys: lowercase ASCII words joined by dots.
# Deliberately strict — the key space is a JOIN KEY across languages
# (D6), and a permissive space would fragment "coffee.milk" from
# "Coffee_Milk" and silently split one attribute into two profiles.
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
        value = (value_text or "").strip()
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
            session.add(ProfileAttribute(
                scope_key=fact.scope_key, agent_id=fact.agent_id,
                user_id=fact.user_id, attribute_key=key, value_text=value,
                fact_id=fact.id, fact_type=fact.fact_type,
                t_occurred=fact.t_occurred, t_mentioned=fact.t_mentioned,
                mention_count=fact.mention_count or 1,
                lang_source=fact.lang_source, proposed_by=proposed_by))
            if owns:
                session.commit()
            return True
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

            best = {}
            for attr, fact in rows:
                when = attr.t_occurred or attr.t_mentioned or ""
                cur = best.get(attr.attribute_key)
                if cur is None or (when, attr.fact_id) > (cur[1], cur[0].fact_id):
                    best[attr.attribute_key] = (attr, when)
            picked = sorted(
                best.values(),
                key=lambda av: (av[0].mention_count or 1, av[1],
                                av[0].attribute_key),
                reverse=True)
            if len(picked) > limit:
                logger.info(
                    f"[ProfileStore] {len(picked)} attributes in scope; "
                    f"injecting the top {limit} by (mentions, recency)")
            return [a for a, _ in picked[:limit]]
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

    def render(self, attrs: list, char_budget: int = 1200) -> str:
        """Render for injection: one line per attribute, most-confirmed
        first, hard-capped. Values are sanitized with the SAME renderer
        the facts block uses — profile values are LLM-derived from user
        text and are equally untrusted (Stage 5 G3 M1)."""
        from agentmem_os.llm.fact_retrieval import _sanitize

        if not attrs:
            return ""
        lines, used = [], 0
        for a in attrs:
            line = f"{a.attribute_key}: {_sanitize(a.value_text)}"
            if lines and used + len(line) + 1 > char_budget:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
