"""Bind the existing dense/lexical ranking to original scoped source records."""

from .evidence_packet import RetrievalHit, digest, eligible_sources, pack
from .multi_vector_retrieval import MultiVectorRetriever


def rank(snapshot, question, *, scope, as_of, encoder=None):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Explicit question required")
    # Mirror the legacy indexer's blank filtering BEFORE mapping positional IDs.
    turns = tuple(t for t in eligible_sources(snapshot, scope=scope, as_of=as_of) if t.text.strip())
    if not turns:
        return ()
    retriever = MultiVectorRetriever(encoder=encoder)
    retriever.index([t.text for t in turns])
    return tuple(
        RetrievalHit(turns[i].id, digest(turns[i].text), score, order)
        for order, (i, score) in enumerate(retriever.ranked_indices(question))
    )


def supplement(
    snapshot,
    question,
    baseline,
    *,
    scope,
    as_of,
    encoder=None,
    char_budget=40000,
    extra_budget=4000,
    max_anchors=8,
    neighbor_turns=1,
):
    """Preserve baseline facts/context and append a bounded original-source block.

    Rank the entire eligible scope, then freeze top-N anchors BEFORE packing.
    No low-ranked backfill, answer labels, generated evidence or model fallback.
    Source text already in baseline may be repeated: identity cannot safely be
    inferred from a text match alone when events repeat on different dates.
    """
    if (
        type(baseline) is not str
        or any(
            type(v) is not int or v < 0
            for v in (char_budget, extra_budget, max_anchors, neighbor_turns)
        )
        or len(baseline) > char_budget
    ):
        raise ValueError("Invalid baseline or packet budget")
    hits = rank(snapshot, question, scope=scope, as_of=as_of, encoder=encoder)
    candidates = hits[:max_anchors]
    allowance = max(0, min(extra_budget, char_budget - len(baseline)) - 2)
    block, report = pack(
        snapshot,
        candidates,
        scope=scope,
        as_of=as_of,
        char_budget=allowance,
        max_anchors=max_anchors,
        neighbor_turns=neighbor_turns,
    )
    offset = len(baseline) + 2 if block else len(baseline)
    text = baseline + ("\n\n" + block if block else "")
    report.update(
        baseline_sha256=digest(baseline),
        baseline_chars=len(baseline),
        block_offset=offset,
        candidate_sha256=digest(text),
        total_chars=len(text),
        total_char_budget=char_budget,
        extra_budget=extra_budget,
        ranked_sources=len(hits),
        candidate_hits=[h.__dict__ for h in candidates],
        baseline_preserved=text.startswith(baseline),
        ranking="existing-dense-lexical-rrf-k60",
    )
    return text, report
