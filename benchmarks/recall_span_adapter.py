"""Bounded source-span reserve for questions about prior assistant replies.

This is a benchmark adapter.  The caller supplies the original turn groups for
the already-scoped history.  Selection sees only the question and source turns;
it never receives the reference answer or annotated source-session labels.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

from agentmem_os.benchmarks.dated_event_adapter import DatedEventContextAssembler
from agentmem_os.benchmarks.dated_event_reserve import prepend_reserve
from agentmem_os.benchmarks.real_code_utils import TfIdfChromaAdapter


_DECISION_RECALL_RE = re.compile(
    r"\b(?:decid(?:e|ed)|chos(?:e|en)|choose|sett(?:le|led)|finally|"
    r"end(?:ed)?\s+up|went\s+with)\b",
    re.IGNORECASE,
)
_POSITIVE_ACCEPTANCE_RE = re.compile(
    r"\b(?:love|cool|good|great|perfect|exactly|works|favorite|favourite|"
    r"go\s+with|settle\s+on|choose|chose)\b",
    re.IGNORECASE,
)
_NAMING_RECALL_RE = re.compile(r"\b(?:name|named|call|called)\b", re.IGNORECASE)
_NAMING_EVIDENCE_RE = re.compile(
    r"\b(?:name|names|named|call|called)\b", re.IGNORECASE)
_MORE_OPTIONS_RE = re.compile(
    r"\b(?:any\s+other|more|another|few)\b[^.!?\n]{0,48}\b"
    r"(?:name|option|idea|suggestion)s?\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{4,}")
_DECISION_STOPWORDS = {
    "about", "after", "again", "could", "finally", "great", "really",
    "their", "there", "these", "thing", "think", "those", "would",
}


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


def _local_window(turns: Sequence, turn_index: int) -> list:
    """Keep a source-ordered proposal, user response, and continuation."""
    start = turn_index
    if (turn_index > 0
            and str(_field(turns[turn_index - 1], "role") or "").lower()
            == "assistant"):
        start -= 1
    end = turn_index + 1
    if (end < len(turns)
            and str(_field(turns[end], "role") or "").lower()
            == "assistant"):
        end += 1
    return list(turns[start:end])


def _decision_bonus(
    turns: Sequence,
    turn_index: int,
    *,
    naming_recall: bool,
) -> float:
    """Favor accepted proposals over requests for additional options."""
    user = str(_field(turns[turn_index], "content") or "")
    if _MORE_OPTIONS_RE.search(user):
        return -0.30
    if not _POSITIVE_ACCEPTANCE_RE.search(user):
        return 0.0
    window = _local_window(turns, turn_index)
    window_text = "\n".join(
        str(_field(turn, "content") or "") for turn in window)
    if naming_recall and not _NAMING_EVIDENCE_RE.search(window_text):
        return 0.0
    if len(window) != 3:
        return 0.25
    token_sets = []
    for turn in window:
        token_sets.append({
            token.lower() for token in _WORD_RE.findall(
                str(_field(turn, "content") or ""))
            if token.lower() not in _DECISION_STOPWORDS
        })
    # Continued use of the same distinctive term after positive user language
    # is source-only evidence that a proposal was adopted.
    repeated = set.intersection(*token_sets)
    return 0.65 if repeated else 0.40


def select_recall_spans(
    query: str,
    turn_groups: Iterable[Sequence],
    *,
    session_limit: int = 1,
    min_similarity: float = 0.18,
    group_similarity_weight: float = 1.0,
    max_chars_per_turn: int = 3_200,
) -> list[str]:
    """Select matched user prompts and their immediately paired replies.

    A character-within-word score is used because the measured recall failures
    include inflection, punctuation and compound-word differences.  Each user
    turn receives a bounded score from its complete source group so distinctive
    entities elsewhere in the same conversation can disambiguate generic recall
    wording.  Once a user request is selected, its next assistant turn is
    reserved as provenance-preserving source text.
    """
    if (session_limit <= 0 or max_chars_per_turn <= 0
            or group_similarity_weight < 0):
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
    group_texts = [
        "\n".join(str(_field(turn, "content") or "") for turn in turns)
        for turns in groups
    ]
    group_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=20_000,
        sublinear_tf=True, min_df=1)
    group_matrix = group_vectorizer.fit_transform(group_texts)
    group_scores = cosine_similarity(
        group_vectorizer.transform([query]), group_matrix)[0]
    # Keep the established turn-level admission boundary. Group context may
    # rerank an already-qualified recall query, but it must not broaden which
    # queries receive a reserve.
    if max(scores) < min_similarity:
        return []
    decision_recall = bool(_DECISION_RECALL_RE.search(query or ""))
    naming_recall = bool(_NAMING_RECALL_RE.search(query or ""))
    if decision_recall:
        local_texts = [
            "\n".join(str(_field(turn, "content") or "")
                      for turn in _local_window(groups[row[0]], row[1]))
            for row in candidates
        ]
        local_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), max_features=20_000,
            sublinear_tf=True, min_df=1)
        local_matrix = local_vectorizer.fit_transform(local_texts)
        local_scores = cosine_similarity(
            local_vectorizer.transform([query]), local_matrix)[0]
    else:
        local_scores = [0.0] * len(candidates)
    base_ranking_scores = [
        scores[index] + group_similarity_weight * group_scores[row[0]]
        for index, row in enumerate(candidates)
    ]
    within_group_scores = []
    for index, row in enumerate(candidates):
        group_index, turn_index, _ = row
        decision_bonus = (
            _decision_bonus(
                groups[group_index], turn_index,
                naming_recall=naming_recall,
            )
            if decision_recall else 0.0
        )
        within_group_scores.append(
            base_ranking_scores[index] + local_scores[index] + decision_bonus)
    # Select source sessions with the already-verified group-aware score. Apply
    # decision evidence only after that boundary so generic acceptance language
    # in an unrelated session cannot change which session is admitted.
    candidates_by_group = {}
    for index, row in enumerate(candidates):
        candidates_by_group.setdefault(row[0], []).append(index)
    ranked_groups = sorted(
        candidates_by_group,
        key=lambda group_index: (
            -max(base_ranking_scores[index]
                 for index in candidates_by_group[group_index]),
            group_index,
        ),
    )

    selected_groups = set()
    reserve = []
    for group_index in ranked_groups:
        index = max(
            candidates_by_group[group_index],
            key=lambda candidate_index: (
                within_group_scores[candidate_index], -candidate_index),
        )
        _, turn_index, _ = candidates[index]
        selected_groups.add(group_index)
        turns = groups[group_index]
        selected_turns = (
            _local_window(turns, turn_index)
            if decision_recall else [turns[turn_index]]
        )
        if not decision_recall and turn_index + 1 < len(turns):
            reply = turns[turn_index + 1]
            if str(_field(reply, "role") or "").lower() == "assistant":
                selected_turns.append(reply)
        for selected_turn in selected_turns:
            selected_content = str(_field(selected_turn, "content") or "")
            if selected_content:
                reserve.append(_best_window(
                    selected_content, query, max_chars_per_turn))
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
        group_similarity_weight: float = 1.0,
        max_chars_per_turn: int = 3_200,
        base=None,
    ):
        self.turn_groups_by_session = {
            key: [list(group) for group in groups]
            for key, groups in turn_groups_by_session.items()
        }
        self.session_limit = session_limit
        self.min_similarity = min_similarity
        self.group_similarity_weight = group_similarity_weight
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
            group_similarity_weight=self.group_similarity_weight,
            max_chars_per_turn=self.max_chars_per_turn,
        )
        if not recall_reserve:
            return no_recall("no_admission")
        # prepend_reserve accepts (ordinary, reserve). Keep the newly matched
        # recall span first, followed by any nested dated-event reserve.
        reserve = prepend_reserve(base_reserve, recall_reserve)
        merged = prepend_reserve(base_chunks, recall_reserve)[:top_k]
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
