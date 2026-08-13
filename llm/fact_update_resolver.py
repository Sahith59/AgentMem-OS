"""Read-time, scope-aware update resolution for the facts tier.

WHY (DECISION_AND_FAILURE_LOG 2026-08-13, the n=500 knowledge-update
autopsy): write-time supersession is correct for production (one user,
one timeline) but structurally cannot cover a multi-user store read
through per-question scopes — marking a fact superseded globally would
corrupt scopes that contain only the older session. And the write-time
judge's pools never pair facts that land in different extraction
shards, different sessions months apart, or different fact_types
("wakes at 8:30" arrived as an event; "likes to wake at 7:30" as a
state — same attribute, never judged). Measured result: 503 superseded
of 98,372 facts, and answer models presented with both values picked
the stale one 5 times out of 5 chances.

WHAT: at read time, over the SCOPED candidate set the retriever
already holds, deterministically pair facts that describe the same
attribute with different values, and ANNOTATE the older line with the
newer value and its date. Non-destructive by design (the ALIAS_OF
DNA): nothing is dropped, nothing is rewritten in the store, a wrong
pairing adds a visible annotation but can never delete truth.

Pairing is deliberately conservative; every gate exists because of a
concrete adversarial case (see tests):
  - both facts carry value tokens (numbers/times) and they DIFFER;
  - non-value content words overlap strongly (same attribute...);
  - ...and no conflicting time-qualifier ("weekdays" vs "Saturdays"
    describe different attributes even with high word overlap);
  - both are dated, on different days (same-day pairs are ambiguity,
    not updates — measured on the followers case);
  - different source sessions (within-session contradictions are the
    write-time judge's jurisdiction, and it already handles them).
"""
from __future__ import annotations

import re

_VALUE_RE = re.compile(
    r"\b\d[\d,.:]*\s?(?:am|pm|AM|PM)?\b")
_STOP = frozenset(
    "the a an of in on at to for with and or is are was were has have had "
    "user their they them my i we he she it that this be been being do "
    "does did not no like likes want wants around usually about".split())


def _stem(w: str) -> str:
    """Crude plural/3rd-person folding ('wakes'->'wake'): enough for
    the attribute vocabulary this compares, no NLP dependency."""
    if len(w) > 3 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w
# Qualifier classes: two facts whose qualifiers land in DIFFERENT
# classes describe different attributes, never an update pair.
_QUALIFIERS = {
    "weekday": {"weekday", "weekdays", "monday", "tuesday", "wednesday",
                "thursday", "friday", "mondays", "tuesdays", "wednesdays",
                "thursdays", "fridays"},
    "weekend": {"weekend", "weekends", "saturday", "sunday", "saturdays",
                "sundays"},
    "morning": {"morning", "mornings"},
    "evening": {"evening", "evenings", "night", "nights"},
}


def value_tokens(text: str) -> frozenset:
    return frozenset(m.group(0).strip().lower()
                     for m in _VALUE_RE.finditer(text))


def content_words(text: str) -> frozenset:
    words = frozenset(_stem(w) for w in re.findall(r"[a-z]+", text.lower())
                      if w not in _STOP)
    vals = {_stem(v) for v in
            re.findall(r"[a-z]+", " ".join(value_tokens(text)))}
    return words - vals


def qualifier_classes(text: str) -> frozenset:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return frozenset(name for name, vocab in _QUALIFIERS.items()
                     if words & vocab)


def is_update_pair(old, new) -> bool:
    """old/new: objects with .fact_text, .t_mentioned,
    .source_session_id. Direction (old vs new) is the caller's job."""
    if not old.t_mentioned or not new.t_mentioned:
        return False
    if str(old.t_mentioned)[:10] == str(new.t_mentioned)[:10]:
        return False
    if old.source_session_id == new.source_session_id:
        return False
    ov, nv = value_tokens(old.fact_text), value_tokens(new.fact_text)
    if not ov or not nv or ov == nv:
        return False
    oq, nq = qualifier_classes(old.fact_text), qualifier_classes(new.fact_text)
    if oq != nq:
        return False
    oc, nc = content_words(old.fact_text), content_words(new.fact_text)
    if not oc or not nc:
        return False
    jaccard = len(oc & nc) / len(oc | nc)
    return jaccard >= 0.5


def find_update_pairs(facts: list) -> dict:
    """Map old_fact_id -> newer fact, over a scoped candidate list.
    Each old fact gets at most its NEWEST matching update. O(n²) over
    the retriever's scan cap; measured fine at that scale."""
    dated = [f for f in facts if getattr(f, "t_mentioned", None)]
    newest = {}
    for i, a in enumerate(dated):
        for b in dated[i + 1:]:
            old, new = ((a, b) if str(a.t_mentioned) < str(b.t_mentioned)
                        else (b, a))
            if not is_update_pair(old, new):
                continue
            cur = newest.get(old.id)
            if cur is None or str(new.t_mentioned) > str(cur.t_mentioned):
                newest[old.id] = new
    return newest


def annotation_for(newer) -> str:
    """The suffix appended to the OLDER fact's rendered line."""
    date = str(newer.t_mentioned)[:10]
    return f"  [UPDATED {date}: {newer.fact_text}]"
