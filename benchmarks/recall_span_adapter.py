"""Bounded source-span reserve for questions about prior assistant replies.

This is a benchmark adapter.  The caller supplies the original turn groups for
the already-scoped history.  Selection sees only the question and source turns;
it never receives the reference answer or annotated source-session labels.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from agentmem_os.benchmarks.dated_event_adapter import DatedEventContextAssembler
from agentmem_os.benchmarks.dated_event_reserve import prepend_reserve
from agentmem_os.benchmarks.real_code_utils import TfIdfChromaAdapter


def _field(turn, name: str):
    if isinstance(turn, dict):
        return turn.get(name)
    return getattr(turn, name, None)


def _best_window(text: str, query: str, max_chars: int) -> str:
    """Return one overlapping source-only window from an oversized turn."""
    if len(text) <= max_chars:
        return text
    step = max(1, max_chars * 3 // 4)
    starts = list(range(0, max(1, len(text) - max_chars + 1), step))
    final_start = len(text) - max_chars
    if starts[-1] != final_start:
        starts.append(final_start)
    windows = [text[start:start + max_chars] for start in starts]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=20_000,
        sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform(windows)
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    return windows[max(range(len(windows)), key=lambda index: (scores[index], -index))]


def select_recall_spans(
    query: str,
    turn_groups: Iterable[Sequence],
    *,
    session_limit: int = 1,
    min_similarity: float = 0.18,
    max_chars_per_turn: int = 3_200,
) -> list[str]:
    """Select matched user prompts and their immediately paired replies.

    A character-within-word score is used because the measured recall failures
    include inflection, punctuation and compound-word differences.  Ranking is
    over user turns only.  Once a user request is selected, its next assistant
    turn is reserved as provenance-preserving source text.
    """
    if session_limit <= 0 or max_chars_per_turn <= 0:
        return []
    groups = [list(group) for group in turn_groups]
    candidates = []
    for group_index, turns in enumerate(groups):
        for turn_index, turn in enumerate(turns):
            content = str(_field(turn, "content") or "")
            if str(_field(turn, "role") or "").lower() == "user" and content:
                candidates.append((group_index, turn_index, content))
    if not candidates:
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=20_000,
        sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform([row[2] for row in candidates])
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    ranked = sorted(
        range(len(candidates)), key=lambda index: (-scores[index], index))

    selected_groups = set()
    reserve = []
    for index in ranked:
        if scores[index] < min_similarity:
            break
        group_index, turn_index, user_content = candidates[index]
        if group_index in selected_groups:
            continue
        selected_groups.add(group_index)
        reserve.append(_best_window(user_content, query, max_chars_per_turn))
        turns = groups[group_index]
        if turn_index + 1 < len(turns):
            reply = turns[turn_index + 1]
            reply_content = str(_field(reply, "content") or "")
            if (str(_field(reply, "role") or "").lower() == "assistant"
                    and reply_content):
                reserve.append(_best_window(
                    reply_content, query, max_chars_per_turn))
        if len(selected_groups) >= session_limit:
            break
    return reserve


class RecallSpanTfIdfAdapter:
    """Add a bounded recall span ahead of the unchanged TF-IDF ranking."""

    def __init__(
        self,
        turn_groups_by_session: Mapping[str, Iterable[Sequence]],
        *,
        session_limit: int = 1,
        min_similarity: float = 0.18,
        max_chars_per_turn: int = 3_200,
        base=None,
    ):
        self.turn_groups_by_session = {
            key: [list(group) for group in groups]
            for key, groups in turn_groups_by_session.items()
        }
        self.session_limit = session_limit
        self.min_similarity = min_similarity
        self.max_chars_per_turn = max_chars_per_turn
        self.base = base or TfIdfChromaAdapter()
        self.last_receipt = None

    def search(self, session_id: str, query: str, top_k: int = 5) -> list:
        base_chunks = self.base.search(session_id, query, top_k=top_k)
        base_receipt = getattr(self.base, "last_receipt", None) or {}
        base_reserve = list(base_receipt.get("reserve", []))

        def no_recall(reason: str) -> list:
            self.last_receipt = {
                "session_id": session_id,
                "query": query,
                "reserve_count": len(base_reserve),
                "reserve": base_reserve,
                "recall_reserve_count": 0,
                "nested_reserve_count": len(base_reserve),
                "reason": reason,
            }
            return base_chunks

        groups = self.turn_groups_by_session.get(session_id)
        if not groups:
            return no_recall("no_turn_groups")

        from agentmem_os.llm.context_assembler import _CONVERSATION_RECALL_RE

        if not _CONVERSATION_RECALL_RE.search(query or ""):
            return no_recall("not_recall_intent")
        recall_reserve = select_recall_spans(
            query,
            groups,
            session_limit=self.session_limit,
            min_similarity=self.min_similarity,
            max_chars_per_turn=self.max_chars_per_turn,
        )
        if not recall_reserve:
            return no_recall("no_admission")
        reserve = prepend_reserve(recall_reserve, base_reserve)
        merged = prepend_reserve(recall_reserve, base_chunks)[:top_k]
        self.last_receipt = {
            "session_id": session_id,
            "query": query,
            "reserve_count": len(reserve),
            "reserve": list(reserve),
            "recall_reserve_count": len(recall_reserve),
            "nested_reserve_count": len(base_reserve),
            "base_count": len(base_chunks),
            "returned_count": len(merged),
        }
        return merged


class RecallSpanContextAssembler(DatedEventContextAssembler):
    """Keep admitted recall spans ahead of ordinary ranked raw turns."""

    reserve_budget_share = 0.25
