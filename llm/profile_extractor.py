"""
Profile attribute extraction: the model PROPOSES canonical attribute
keys for preference/identity facts; `db.profile` DECIDES what is
stored (PROFILE_TIER_PLAN.md D2).

Same contract as Stage 4's supersession judge, for the same reason:
a local 8B model is not reliable enough to be trusted with a write.
Here the gates are `normalize_key` (charset/shape/depth/length), the
fact-type guard, and the value guard — a rejected proposal leaves the
fact untouched and unprofiled rather than storing something wrong.

Batched by design: one call per BATCH of facts, not per fact — the
Gate C corpus holds 7,164 preference/identity facts and a per-fact
call would be 7,164 LLM round-trips.

Grammar-forced JSON (Ollama `format`) with reasoning-FIRST field
ordering — the graphiti#1666 measurement (46.7% -> 93.3%) that Stage 4
already relies on.
"""
import json
import os
import urllib.request

from loguru import logger

OLLAMA_URL = os.environ.get(
    "AGENTMEM_OS_OLLAMA_URL", "http://localhost:11434") \
    .rstrip("/") + "/api/generate"
DEFAULT_MODEL = "llama3.1:latest"
BATCH_SIZE = 12          # facts per call; keeps output well under the cap
NUM_PREDICT = 3000
NUM_CTX = 8192

ATTRS_SCHEMA = {
    "type": "object",
    "properties": {
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "attribute_key": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["index", "attribute_key", "value"],
            },
        }
    },
    "required": ["attributes"],
}

_PROMPT = """You convert facts about a user into PROFILE ATTRIBUTES.

For each numbered fact, output:
  index         — the fact's number, exactly as given
  attribute_key — a canonical dotted key in ENGLISH, lowercase, at most
                  3 levels (e.g. coffee.milk, work.location, diet.type,
                  music.genre, sport.played, name, occupation). Use the
                  SAME key for the same kind of attribute every time,
                  whatever language the fact came from.
  value         — the attribute's value ONLY, short and self-contained
                  (e.g. "oat milk", "Bangalore", "vegetarian"). Not a
                  sentence, not "the user prefers X" — just X.

Rules:
- Skip a fact entirely (do not emit it) if it is not a stable property
  of the user — one-off events, momentary moods, or anything that will
  not still be true next month.
- Never invent an attribute the fact does not state.
- Keys must be English even when the fact mentions other languages.
- NEVER put a specific value, name, brand or title inside the key.
  WRONG: possession.omega_speedmaster / search.great_gatsby
  RIGHT: possession -> "Omega Speedmaster watch"
- REUSE an existing key below whenever the fact is the same KIND of
  attribute, even if the value differs. Two facts about what music the
  user likes share ONE key. Only invent a new key when nothing listed
  fits.

Keys already in use for this user (REUSE THESE FIRST):
{vocabulary}

Facts:
{facts}"""


class ProfileExtractor:
    """Proposes attribute keys; the store's gates decide."""

    def __init__(self, get_db_session, model: str = DEFAULT_MODEL,
                 timeout: int = 600):
        self.get_db = get_db_session
        self.model = model
        self.timeout = timeout
        self._store = None

    def _get_store(self):
        if self._store is None:
            from agentmem_os.db.profile import ProfileStore
            self._store = ProfileStore(self.get_db)
        return self._store

    def _llm(self, prompt: str) -> dict:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": self.model, "prompt": prompt, "stream": False,
                "format": ATTRS_SCHEMA,
                "options": {"temperature": 0, "num_ctx": NUM_CTX,
                            "num_predict": NUM_PREDICT},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            body = json.loads(r.read())
        if "response" not in body:
            raise ValueError(f"Ollama reply missing 'response': {body}")
        if body.get("done_reason") == "length":
            # The Gate C lesson: truncation must be observable, never a
            # parser error. A smaller batch is the remedy here.
            raise ValueError(
                "profile extraction output truncated — reduce BATCH_SIZE")
        return json.loads(body["response"])

    def _vocabulary(self, scope_key: str, cap: int = 60) -> str:
        """Keys already used in this scope, most-used first. Feeding
        these back is what makes a profile COLLAPSE (G2 finding: keyed
        blind, batch 2 reinvents batch 1's vocabulary — 'hobbies',
        'hobby.activity', 'hobby.topic' and 'interests' all appeared
        for one concept, and only 8 of 58 keys carried more than one
        fact)."""
        from sqlalchemy import func

        from agentmem_os.db.models import ProfileAttribute

        db = self.get_db()
        try:
            rows = (db.query(ProfileAttribute.attribute_key,
                             func.count(ProfileAttribute.id).label("n"))
                    .filter(ProfileAttribute.scope_key == scope_key)
                    .group_by(ProfileAttribute.attribute_key)
                    .order_by(func.count(ProfileAttribute.id).desc())
                    .limit(cap).all())
        finally:
            db.close()
        return ", ".join(r[0] for r in rows) if rows else "(none yet)"

    def extract_batch(self, facts: list, vocabulary: str = "(none yet)") -> list:
        """(fact, key, value) proposals for a batch. Malformed or
        out-of-range indices are DROPPED loudly — the model may not
        invent facts that were not in the batch."""
        if not facts:
            return []
        listing = "\n".join(f"{i}. {f.fact_text}" for i, f in enumerate(facts))
        raw = self._llm(_PROMPT.format(facts=listing,
                                       vocabulary=vocabulary))
        out, seen = [], set()
        for item in raw.get("attributes", []):
            idx = item.get("index")
            if not isinstance(idx, int) or isinstance(idx, bool):
                continue
            if not (0 <= idx < len(facts)) or idx in seen:
                # Out-of-range = the model referring to a fact it was
                # never shown; duplicate = it emitting one fact twice.
                # Both are dropped rather than guessed at.
                continue
            seen.add(idx)
            out.append((facts[idx], item.get("attribute_key"),
                        item.get("value")))
        return out

    def project_scope(self, agent_id: str = None, user_id: str = None,
                      limit: int = 5000, session_ids: list = None) -> dict:
        """Project every un-profiled preference/identity fact in scope.

        Resumable and idempotent: a fact that already has a profile row
        is skipped, so a re-run costs nothing and a crash loses nothing
        (the link_missing pattern). Per-batch failures are counted and
        reported, never fatal — projection must not be able to take the
        fact tier down (D7).
        """
        from agentmem_os.db.models import ProfileAttribute, SemanticFact
        from agentmem_os.db.profile import PROFILE_FACT_TYPES
        from agentmem_os.db.semantic_facts import make_scope_key

        scope_key = make_scope_key(agent_id, user_id)
        db = self.get_db()
        try:
            done = {r[0] for r in db.query(ProfileAttribute.fact_id)
                    .filter(ProfileAttribute.scope_key == scope_key).all()}
            q = (db.query(SemanticFact)
                 .filter(SemanticFact.scope_key == scope_key,
                         SemanticFact.fact_type.in_(PROFILE_FACT_TYPES),
                         SemanticFact.superseded_by.is_(None)))
            if session_ids is not None:
                q = q.filter(
                    SemanticFact.source_session_id.in_(list(session_ids)))
            todo = [f for f in q.order_by(SemanticFact.id.asc())
                    .limit(limit).all() if f.id not in done]
        finally:
            db.close()

        report = {"candidates": len(todo), "projected": 0, "rejected": 0,
                  "batch_failures": 0, "skipped_by_model": 0}
        store = self._get_store()
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            try:
                proposals = self.extract_batch(
                    batch, vocabulary=self._vocabulary(scope_key))
            except Exception as e:
                report["batch_failures"] += 1
                logger.warning(
                    f"[ProfileExtractor] batch {i // BATCH_SIZE} failed: "
                    f"{type(e).__name__}: {e}")
                continue
            report["skipped_by_model"] += len(batch) - len(proposals)
            db = self.get_db()
            try:
                wrote = rejected = 0
                for fact, key, value in proposals:
                    fresh = db.get(SemanticFact, fact.id)
                    if fresh is None:
                        continue
                    if store.project(fresh, key, value, self.model, db=db):
                        wrote += 1
                    else:
                        rejected += 1
                db.commit()
                # Counted only AFTER the commit that makes them real
                # (G3 R1 M2: the report claimed 2 projected with 0 rows
                # in the DB — a rolled-back batch still incremented).
                report["projected"] += wrote
                report["rejected"] += rejected
            except Exception as e:
                db.rollback()
                report["batch_failures"] += 1
                logger.warning(
                    f"[ProfileExtractor] batch {i // BATCH_SIZE} write "
                    f"failed: {type(e).__name__}: {e}")
            finally:
                db.close()
        logger.info(f"[ProfileExtractor] scope={scope_key!r} {report}")
        return report
