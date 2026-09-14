"""Precision-limited source-turn ranking for the second offline candidate.

This remains benchmark-only and opt-in. Parameters were selected after the
first broad supplement's paid diagnostic exposed distraction failures; this
is development-set iteration, not held-out model selection.
"""

from __future__ import annotations

from collections.abc import Sequence


MIN_SCORE = 0.07
MAX_TURNS = 5


def rank_turns(contents: Sequence[str], query: str) -> list[str]:
    """Return at most five complete turns above the fixed relevance floor."""
    if not contents or not query.strip():
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        min_df=1,
        stop_words="english",
        ngram_range=(1, 2),
    )
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(content) for content in contents):
        return []
    matrix = vectorizer.fit_transform(contents)
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    order = scores.argsort()[::-1]
    return [contents[index] for index in order if scores[index] > MIN_SCORE][
        :MAX_TURNS
    ]
