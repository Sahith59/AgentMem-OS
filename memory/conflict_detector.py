"""
AgentMem OS — ConflictDetector  (Phase 1)
==========================================
Detects when a new user turn contradicts an earlier stored fact and
soft-deletes the stale turn by setting is_active=False.

Design principles:
  - Zero extra LLM calls: pattern-based detection only (fast, free, no hallucination risk)
  - Only user turns are checked — assistants don't make factual claims about the user
  - Three detection layers (cheapest first):
      1. Entity overlap  — only compare turns sharing a named entity
      2. Topic match     — the two turns are about the same subject
      3. Polarity flip   — one asserts positive, the other negative about that subject
  - Conservative threshold: prefer false negatives over false positives
    (missing a contradiction is better than deleting a valid memory)
"""

import re
import math
from datetime import datetime
from typing import Optional
from loguru import logger

from agentmem_os.db.engine import get_session
from agentmem_os.db.models import Turn


# ─────────────────────────────────────────────────────────────────────────────
# Contradiction pattern library
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (positive_pattern, negative_pattern, category)
# Both are applied to the same entity context to detect polarity flip.
_CONTRADICTION_PAIRS = [
    # Preferences — extended positive vocabulary
    (r"\b(love|like|enjoy|prefer|use|adore|fan of|favourite|favorite|great at|good at)\b",
     r"\b(hate|dislike|avoid|don't use|do not use|stopped using|can't stand|no longer use|switched away)\b",
     "preference"),

    # Employment. Third-person "works at" included (Stage 4 root-cause
    # fix: the missing s-form was noticed during the KG typed-relation
    # port and patched LOCALLY there instead of here — db/
    # knowledge_graph.py's _build_relation_patterns still carries that
    # patch, now redundant but harmless. Canonical fact text is
    # third-person by design, so the supersession judge's polarity
    # co-signal was blind to every "The user works at X" fact.)
    (r"\b(work(?:s|ing)? at|work(?:s|ing)? for|employed at|joined|started at)\b",
     r"\b(left|quit|fired from|no longer at|resigned from|moved on from)\b",
     "employment"),

    # Location
    (r"\b(live(?:s)? in|based in|located in|moved to|living in)\b",
     r"\b(left|moved from|no longer in|moved away from)\b",
     "location"),

    # Study / education
    (r"\b(study(?:ing)? at|attend(?:ing)?|enrolled at|go(?:ing)? to)\b",
     r"\b(left|dropped out|graduated from|transferred from)\b",
     "education"),

    # Positive/negative sentiment — only used when entity overlap is confirmed
    (r"\b(good|great|excellent|fantastic|amazing|best|love|wonderful)\b",
     r"\b(bad|terrible|awful|horrible|hate|worst|disappointed|dreadful)\b",
     "sentiment"),
]

# Negation words that can flip polarity within a single sentence
_NEGATIONS = re.compile(
    r"\b(not|no|never|don't|doesn't|didn't|won't|can't|cannot|isn't|aren't|wasn't)\b",
    re.I,
)


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _tfidf_cosine(a: str, b: str) -> float:
    """Lightweight TF-IDF cosine similarity — no external deps."""
    def tf(tokens):
        freq: dict[str, int] = {}
        for w in tokens:
            freq[w] = freq.get(w, 0) + 1
        n = len(tokens) or 1
        return {w: c / n for w, c in freq.items()}

    ta, tb = _tokenise(a), _tokenise(b)
    vocab = set(ta) | set(tb)
    # IDF proxy: words appearing in both docs are more specific
    df = {w: (1 if w in ta else 0) + (1 if w in tb else 0) for w in vocab}
    tfa, tfb = tf(ta), tf(tb)
    vec_a = {w: tfa.get(w, 0) * math.log(1 + 1 / df[w]) for w in vocab}
    vec_b = {w: tfb.get(w, 0) * math.log(1 + 1 / df[w]) for w in vocab}
    dot = sum(vec_a.get(w, 0) * vec_b.get(w, 0) for w in vocab)
    na = math.sqrt(sum(v ** 2 for v in vec_a.values())) or 1e-9
    nb = math.sqrt(sum(v ** 2 for v in vec_b.values())) or 1e-9
    return dot / (na * nb)


def _extract_entities(text: str) -> set[str]:
    """Capitalised multi-word phrases as a lightweight entity proxy."""
    matches = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b", text)
    return {m.lower() for m in matches if len(m) > 2}


# Words excluded from content-word overlap: temporal/function words that often
# appear in both sentences without indicating a shared TOPIC.
_CONTENT_STOPWORDS = frozenset({
    "every", "always", "never", "often", "after", "early", "later", "other",
    "there", "their", "these", "those", "where", "which", "would", "could",
    "should", "might", "morning", "evening", "tonight", "today", "yesterday",
    "really", "quite", "about", "still", "great", "little", "being",
    "getting", "doing", "having", "going", "coming", "think", "things",
    "people", "place", "right", "first", "again", "still", "using",
    "before", "since", "while", "every", "truly", "until",
})


def _content_word_overlap(a: str, b: str) -> bool:
    """
    True if a and b share at least one significant content word (≥5 chars,
    not a temporal/function word). This catches lowercase topics like
    'rock climbing' or 'chess' that bypass the capitalised entity extractor.
    """
    def _keywords(text: str) -> set[str]:
        return {
            w for w in re.findall(r'[a-z]{5,}', text.lower())
            if w not in _CONTENT_STOPWORDS
        }
    return bool(_keywords(a) & _keywords(b))


def _has_polarity_flip(old: str, new: str) -> tuple[bool, str]:
    """
    Returns (True, category) if old and new have opposing polarity
    about a shared subject, False otherwise.

    Strategy:
      1. For each contradiction pair, check polarity flip (with negation awareness)
      2. For generic "sentiment" pairs, also require entity overlap — this prevents
         false positives where two unrelated sentences happen to share a polarity word
         (e.g. "I like coffee" vs "I hate Mondays" both contain polarity but different subjects)
    """
    def _negated(text: str, pattern: str) -> bool:
        for m in re.finditer(pattern, text, re.I):
            window = text[max(0, m.start() - 40): m.start()]
            if _NEGATIONS.search(window):
                return True
        return False

    def _matches(text: str, pattern: str) -> bool:
        hit = bool(re.search(pattern, text, re.I))
        if hit and _negated(text, pattern):
            return False
        return hit

    has_shared_entity = _entity_overlap(old, new)
    # Fallback for lowercase topics (e.g. "rock climbing", "chess"):
    # content-word overlap catches shared nouns/activities that aren't proper
    # nouns and therefore miss the capitalised-entity extractor. Temporal and
    # function words ("morning", "every", "getting") are excluded from this
    # check to avoid false positives where sentences share context words but
    # discuss unrelated subjects.
    if not has_shared_entity:
        has_shared_entity = _content_word_overlap(old, new)

    for pos_pat, neg_pat, category in _CONTRADICTION_PAIRS:
        old_pos = _matches(old, pos_pat)
        old_neg = _matches(old, neg_pat)
        new_pos = _matches(new, pos_pat)
        new_neg = _matches(new, neg_pat)

        if not ((old_pos and new_neg) or (old_neg and new_pos)):
            continue

        # Preference and sentiment are generic polarity words — "I like coffee"
        # and "I hate Mondays" would both match without entity overlap, causing a
        # false positive. Require entity overlap for any non-structural category.
        # Structural patterns (employment, location, education) are specific
        # enough ("work at X", "live in X") that they self-identify their subject.
        if category in ("preference", "sentiment") and not has_shared_entity:
            continue

        return True, category

    return False, ""


def _entity_overlap(old: str, new: str) -> bool:
    """True if old and new share at least one entity (cheap gate)."""
    return bool(_extract_entities(old) & _extract_entities(new))


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class ConflictDetector:
    """
    Scans recent user turns for contradictions with a newly saved turn.

    Call check_and_resolve() from a background thread after every
    user turn is saved. It is fully self-contained: opens its own DB
    session and closes it, so it is safe to call from any thread.
    """

    # How many recent user turns to compare against (keep small for speed)
    LOOKBACK = 40
    # Minimum topic similarity before checking polarity.
    # Kept low (0.10) because entity overlap is the primary topic gate —
    # two short sentences sharing one entity will have low TF-IDF cosine
    # but the entity overlap already confirms they're on the same subject.
    TOPIC_THRESHOLD = 0.10

    def check_and_resolve(
        self,
        session_id: str,
        new_turn_id: int,
        new_content: str,
        db_path: Optional[str] = None,
    ) -> int:
        """
        Compare new_content against the last LOOKBACK active user turns
        in session_id. For each pair that shares an entity AND has a polarity
        flip, soft-delete the older turn.

        db_path: when set, uses a raw sqlite3 connection instead of SQLAlchemy.
                 Required when called from compare_chat (raw sqlite3 writes) to
                 avoid the StaticPool WAL snapshot issue where the SQLAlchemy
                 connection doesn't see commits from a separate raw connection.

        Returns the number of turns deactivated.
        """
        if db_path:
            return self._check_sqlite(session_id, new_turn_id, new_content, db_path)
        return self._check_sqlalchemy(session_id, new_turn_id, new_content)

    def _check_sqlite(
        self, session_id: str, new_turn_id: int, new_content: str, db_path: str
    ) -> int:
        """Raw sqlite3 path — same connection sees its own committed writes."""
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(db_path)
        deactivated = 0
        try:
            rows = conn.execute(
                "SELECT id, content FROM turns "
                "WHERE session_id=? AND role='user' AND is_active=1 AND id!=? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, new_turn_id, self.LOOKBACK),
            ).fetchall()

            for turn_id, old_content in rows:
                if not _entity_overlap(old_content, new_content):
                    continue
                similarity = _tfidf_cosine(old_content, new_content)
                if similarity < self.TOPIC_THRESHOLD:
                    continue
                flipped, category = _has_polarity_flip(old_content, new_content)
                if not flipped:
                    continue

                conn.execute(
                    "UPDATE turns SET is_active=0, contradicted_by=? WHERE id=?",
                    (new_turn_id, turn_id),
                )
                deactivated += 1
                logger.info(
                    f"[ConflictDetector] Contradiction ({category}): "
                    f"turn {turn_id} superseded by turn {new_turn_id} "
                    f"in session {session_id!r} (similarity={similarity:.3f})"
                )

            if deactivated:
                conn.commit()
        except Exception as e:
            logger.warning(f"[ConflictDetector] sqlite3 error: {e}")
        finally:
            conn.close()
        return deactivated

    def _check_sqlalchemy(
        self, session_id: str, new_turn_id: int, new_content: str
    ) -> int:
        """SQLAlchemy path — used when turns were saved via store.save_turn()."""
        db = get_session()
        deactivated = 0
        try:
            prior_turns = (
                db.query(Turn)
                .filter(
                    Turn.session_id == session_id,
                    Turn.role == "user",
                    Turn.is_active == True,
                    Turn.id != new_turn_id,
                )
                .order_by(Turn.id.desc())
                .limit(self.LOOKBACK)
                .all()
            )

            for prior in prior_turns:
                old_content = prior.content
                if not _entity_overlap(old_content, new_content):
                    continue
                similarity = _tfidf_cosine(old_content, new_content)
                if similarity < self.TOPIC_THRESHOLD:
                    continue
                flipped, category = _has_polarity_flip(old_content, new_content)
                if not flipped:
                    continue

                prior.is_active = False
                prior.contradicted_by = new_turn_id
                deactivated += 1
                logger.info(
                    f"[ConflictDetector] Contradiction ({category}): "
                    f"turn {prior.id} superseded by turn {new_turn_id} "
                    f"in session {session_id!r} (similarity={similarity:.3f})"
                )

            if deactivated:
                db.commit()

        except Exception as e:
            logger.warning(f"[ConflictDetector] Error: {e}")
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()

        return deactivated


def forget_about(session_id: str, topic: str) -> int:
    """
    Explicitly soft-delete all active user turns in session_id whose content
    is topically similar to `topic` (TF-IDF cosine >= 0.25).

    Returns the number of turns deactivated.
    """
    db = get_session()
    deactivated = 0
    try:
        turns = (
            db.query(Turn)
            .filter(
                Turn.session_id == session_id,
                Turn.role == "user",
                Turn.is_active == True,
            )
            .all()
        )
        # Keyword overlap: any significant word (≥4 chars) from the topic that
        # appears in the turn content counts as a match. This is more reliable
        # than TF-IDF cosine for short sentences sharing a single key term
        # (e.g. "Python programming language" vs "My primary language is Python").
        topic_keywords = {w for w in re.findall(r'[a-z]{4,}', topic.lower())}
        for t in turns:
            turn_words = set(re.findall(r'[a-z]{4,}', t.content.lower()))
            if topic_keywords & turn_words:
                t.is_active = False
                deactivated += 1
                logger.info(
                    f"[ConflictDetector] Forgot turn {t.id} "
                    f"(matched topic={topic!r})"
                )
        if deactivated:
            db.commit()
    except Exception as e:
        logger.warning(f"[ConflictDetector] forget_about error: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()
    return deactivated
