"""Query-intent router for adaptive retrieval packing (F-19).

Two packing needs proved incompatible under one static policy
(F-18/F-18b, measured): single-conversation lookback questions need
DEPTH (the answer is a long passage inside one session), while
cross-session aggregate questions need BREADTH (one fact per session,
many sessions). This router reads the QUERY's meaning and picks the
packing mode per question.

Mechanism: cosine similarity of the query embedding against two small
prototype banks, using the same multilingual-e5 encoder the product
ships (so routing works across languages by construction — a Hindi
lookback question routes like an English one). No keyword rules; the
prototypes are embedded meaning, not string matches. Margin below
threshold -> NEUTRAL (default packing). Env kill switch:
AGENTMEM_OS_DISABLE_ADAPTIVE_PACKING=1.
"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger

LOOKBACK_PROTOTYPES = [
    "I was looking back at our previous conversation about this topic",
    "in our previous chat you told me something, can you remind me",
    "what did you say in our earlier conversation about it",
    "I remember you gave me a detailed answer before, what was it",
    "going back to what we discussed, what were the exact details you "
    "mentioned",
]

AGGREGATE_PROTOTYPES = [
    "how many of these have I done in total across all our chats",
    "what is the total number of times this happened",
    "count how many items I mentioned over the past months",
    "what is the combined total amount from all the occasions",
    "how many different ones have I bought or tried altogether",
]

_MARGIN = float(os.environ.get("AGENTMEM_OS_INTENT_MARGIN", "0.02"))
_FLOOR = float(os.environ.get("AGENTMEM_OS_INTENT_FLOOR", "0.80"))

_state: dict = {}


def _banks():
    if "look" not in _state:
        from agentmem_os.db.entity_aliases import get_shared_encoder
        model = get_shared_encoder()
        if model is None:
            return None
        _state["look"] = model.encode(
            [f"query: {p}" for p in LOOKBACK_PROTOTYPES],
            normalize_embeddings=True, show_progress_bar=False)
        _state["agg"] = model.encode(
            [f"query: {p}" for p in AGGREGATE_PROTOTYPES],
            normalize_embeddings=True, show_progress_bar=False)
        _state["model"] = model
    return _state


def route(query: str) -> Optional[str]:
    """Return 'deep' | 'breadth' | None (neutral / disabled)."""
    if os.environ.get("AGENTMEM_OS_DISABLE_ADAPTIVE_PACKING"):
        return None
    s = _banks()
    if s is None:
        return None
    q = s["model"].encode([f"query: {query}"], normalize_embeddings=True,
                          show_progress_bar=False)[0]
    look = float((s["look"] @ q).max())
    agg = float((s["agg"] @ q).max())
    top, mode = (look, "deep") if look >= agg else (agg, "breadth")
    if top < _FLOOR or abs(look - agg) < _MARGIN:
        return None
    logger.debug(f"[IntentRouter] {mode} (look={look:.3f} agg={agg:.3f}) "
                 f"for: {query[:60]}")
    return mode
