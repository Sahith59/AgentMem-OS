"""Source-bound evidence packing. No inference, database access or benchmark labels.

Scores select anchors; the packer always returns that exact turn, never a
session surrogate. Nearby turns are optional context, not certified evidence.
"""

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SourceTurn:
    id: str
    session: str
    position: int
    role: str
    observed_at: datetime
    text: str


@dataclass(frozen=True)
class SourceSnapshot:
    scope: str
    turns: tuple[SourceTurn, ...]


@dataclass(frozen=True)
class RetrievalHit:
    source_id: str
    source_sha256: str
    score: float
    tie_order: int = 0


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def eligible_sources(snapshot, *, scope, as_of):
    if (
        type(snapshot) is not SourceSnapshot
        or type(scope) is not str
        or not scope
        or snapshot.scope != scope
    ):
        raise ValueError("Explicit matching scope required")
    if type(snapshot.turns) is not tuple:
        raise ValueError("Immutable source tuple required")
    if type(as_of) is not datetime:
        raise ValueError("Explicit question time required")
    seen, positions = set(), set()
    eligible = []
    for t in snapshot.turns:
        if (
            type(t) is not SourceTurn
            or type(t.id) is not str
            or not t.id
            or type(t.session) is not str
            or not t.session
            or t.id in seen
            or type(t.position) is not int
            or t.position < 0
            or (t.session, t.position) in positions
            or t.role not in {"user", "assistant"}
            or not isinstance(t.text, str)
            or not t.text
            or type(t.observed_at) is not datetime
        ):
            raise ValueError("Invalid or ambiguous source")
        seen.add(t.id)
        positions.add((t.session, t.position))
        try:
            if t.observed_at <= as_of:
                eligible.append(t)
        except TypeError as error:
            raise ValueError("Incompatible source time zone") from error
    return tuple(eligible)


def pack(snapshot, hits, *, scope, as_of, char_budget, max_anchors=8, neighbor_turns=1):
    """Pack ranked anchors first; spend remaining budget on same-session neighbors.

    Whole turns only. No semantic completeness, event membership, arithmetic or
    final answer is inferred. Omissions remain visible in the returned report.
    Neighbor/context insufficiency must not be mistaken for an absence of facts.
    """
    if any(type(n) is not int or n < 0 for n in (char_budget, max_anchors, neighbor_turns)):
        raise ValueError("Invalid evidence budget")
    eligible = eligible_sources(snapshot, scope=scope, as_of=as_of)
    by_id = {t.id: t for t in snapshot.turns}
    allowed = {t.id for t in eligible}
    by_session = {}
    for t in eligible:
        by_session.setdefault(t.session, []).append(t)
    for turns in by_session.values():
        turns.sort(key=lambda t: t.position)
    ranked, seen, omitted = [], set(), []
    for hit in hits:
        if (
            type(hit) is not RetrievalHit
            or hit.source_id not in by_id
            or not math.isfinite(hit.score)
            or hit.score < 0
            or type(hit.tie_order) is not int
            or hit.tie_order < 0
            or hit.source_sha256 != digest(by_id[hit.source_id].text)
        ):
            raise ValueError("Unbound retrieval hit")
        if hit.source_id in seen:
            continue
        seen.add(hit.source_id)
        if hit.source_id not in allowed:
            omitted.append(dict(id=hit.source_id, reason="future_source"))
        else:
            ranked.append(hit)
    ranked.sort(key=lambda h: (-h.score, h.tie_order, h.source_id))
    chosen, anchor_ids = set(), []

    def render(ids):
        text = "[ORIGINAL SOURCE EVIDENCE: relevance and completeness are unverified]\n"
        receipts = []
        for t in sorted(
            (by_id[i] for i in ids), key=lambda t: (t.observed_at, t.session, t.position)
        ):
            header = f"[{t.id} | {t.role} | observed {t.observed_at.isoformat()}]\n"
            start = len(text) + len(header)
            text += header + t.text + "\n"
            receipts.append(
                dict(
                    id=t.id,
                    session=t.session,
                    position=t.position,
                    role=t.role,
                    observed_at=t.observed_at.isoformat(),
                    sha256=digest(t.text),
                    start=start,
                    end=start + len(t.text),
                    kind="anchor" if t.id in anchor_ids else "context",
                )
            )
        return (text, receipts) if ids else ("", [])

    for h in ranked:
        if len(anchor_ids) >= max_anchors:
            omitted.append(dict(id=h.source_id, reason="anchor_limit"))
            continue
        if len(render(chosen | {h.source_id})[0]) > char_budget:
            omitted.append(dict(id=h.source_id, reason="anchor_budget_nonfit"))
            continue
        chosen.add(h.source_id)
        anchor_ids.append(h.source_id)
    for anchor in anchor_ids:
        t = by_id[anchor]
        turns = by_session[t.session]
        center = next(i for i, v in enumerate(turns) if v.id == anchor)
        for v in turns[max(0, center - neighbor_turns) : center + neighbor_turns + 1]:
            if v.id in chosen:
                continue
            if len(render(chosen | {v.id})[0]) > char_budget:
                omitted.append(dict(id=v.id, reason="context_budget_nonfit", anchor=anchor))
            else:
                chosen.add(v.id)
    text, receipts = render(chosen)
    # Anchor selection can skip a source that later fits as local context.
    # Preserve that decision trace without falsely reporting it as absent.
    selection_skips = omitted
    omitted = [item for item in selection_skips if item["id"] not in chosen]
    return text, dict(
        schema="source-evidence-packet-v1",
        scope=scope,
        as_of=as_of.isoformat(),
        packet_sha256=digest(text),
        anchors=anchor_ids,
        receipts=receipts,
        selection_skips=selection_skips,
        omissions=omitted,
        char_budget=char_budget,
        used_chars=len(text),
        semantic_completeness="NOT_CERTIFIED",
        answer_accuracy="NOT_MEASURED",
    )
