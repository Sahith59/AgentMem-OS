"""Benchmark adapter that adds a bounded dated-event raw-turn reserve."""

from __future__ import annotations

from agentmem_os.benchmarks.dated_event_reserve import (
    prepend_reserve,
    select_dated_event_turns,
)
from agentmem_os.benchmarks.real_code_utils import TfIdfChromaAdapter


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
