"""Benchmark adapter that adds a bounded dated-event raw-turn reserve."""

from __future__ import annotations

from agentmem_os.benchmarks.dated_event_reserve import (
    prepend_reserve,
    select_dated_event_turns,
)
from agentmem_os.benchmarks.real_code_utils import TfIdfChromaAdapter
from agentmem_os.llm.context_assembler import ContextAssembler


class DatedEventTfIdfAdapter:
    """Wrap the measured TF-IDF adapter without changing its base ranking.

    ``reference_dates_by_query`` is required benchmark metadata.  Production
    callers do not have that mapping, which is one reason this remains an
    explicit benchmark adapter instead of a ContextAssembler default.
    """

    def __init__(self, reference_dates_by_query, reserve_limit=3, base=None,
                 turn_loader=None):
        if reserve_limit < 0:
            raise ValueError("reserve_limit must be non-negative")
        self.reference_dates_by_query = dict(reference_dates_by_query)
        self.reserve_limit = reserve_limit
        self.base = base or TfIdfChromaAdapter()
        self.turn_loader = turn_loader or self._load_turns
        self.last_receipt = None

    @staticmethod
    def _load_turns(session_id):
        from agentmem_os.db.engine import get_session as get_db
        from agentmem_os.db.models import Turn

        db = get_db()
        try:
            return list(
                db.query(Turn)
                .filter(Turn.session_id == session_id)
                .order_by(Turn.id.asc())
                .all()
            )
        finally:
            db.close()

    def search(self, session_id: str, query: str, top_k: int = 5) -> list:
        base_chunks = self.base.search(session_id, query, top_k=top_k)
        reference_date = self.reference_dates_by_query.get(query)
        if reference_date is None or self.reserve_limit == 0:
            self.last_receipt = {
                "session_id": session_id,
                "query": query,
                "reserve_count": 0,
                "reason": "unregistered_query" if reference_date is None
                else "reserve_disabled",
            }
            return base_chunks

        reserve = select_dated_event_turns(
            query, reference_date, self.turn_loader(session_id),
            limit=self.reserve_limit)
        merged = prepend_reserve(base_chunks, reserve)[:top_k]
        self.last_receipt = {
            "session_id": session_id,
            "query": query,
            "reserve_count": len(reserve),
            "reserve": list(reserve),
            "base_count": len(base_chunks),
            "returned_count": len(merged),
        }
        return merged


class DatedEventContextAssembler(ContextAssembler):
    """Keep admitted reserve turns ahead of chronologically ordered raw turns.

    The ordinary assembler first selects by rank, sorts the survivors by date,
    then trims from the head. A selected recent reserve turn can therefore be
    moved behind older evidence and trimmed away. This benchmark subclass
    protects at most ``reserve_budget_share`` of the raw character budget and
    leaves remaining chunks on the product's existing chronology path.
    """

    reserve_budget_share = 0.15

    def _order_evidence(self, chunks, token_budget):
        receipt = getattr(self._chroma, "last_receipt", None) or {}
        requested = receipt.get("reserve", [])
        if not requested:
            return super()._order_evidence(chunks, token_budget)

        chunk_set = set(chunks)
        reserve_cap = int(token_budget * 4 * self.reserve_budget_share)
        reserved = []
        used = 0
        for chunk in requested:
            if chunk not in chunk_set or chunk in reserved:
                continue
            cost = len(chunk) + (5 if reserved else 0)
            if reserved and used + cost > reserve_cap:
                break
            # Keep one admitted event even if it alone exceeds the fractional
            # cap; the caller's exact budget cut still bounds the whole block.
            reserved.append(chunk)
            used += cost

        if not reserved:
            return super()._order_evidence(chunks, token_budget)
        reserved_set = set(reserved)
        remaining = [chunk for chunk in chunks if chunk not in reserved_set]
        remaining_tokens = max(0, (token_budget * 4 - used) // 4)
        ordered_remaining = (super()._order_evidence(
            remaining, remaining_tokens) if remaining_tokens else [])
        return reserved + ordered_remaining
