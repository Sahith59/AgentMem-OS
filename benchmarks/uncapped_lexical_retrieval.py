"""Experimental source-only lexical ranker; not a default retrieval path.

The existing TF-IDF benchmark adapter caps its vocabulary at 512 terms. This
candidate retains the vocabulary so rare query terms remain representable.
It changes no prompts, scopes, facts, labels, or context budgets. Evaluation
must measure evidence losses as well as gains before opting into this ranker.
"""

from __future__ import annotations

from collections.abc import Sequence


def rank_turns(contents: Sequence[str], query: str, top_k: int = 5) -> list[str]:
    """Return original source strings, ranked with uncapped word TF-IDF.

    Uses the existing adapter's NumPy sorting convention. Empty or
    out-of-vocabulary queries return no matches, never arbitrary source text.
    This pure function has no database, network, or model dependency.
    """
    if top_k <= 0 or not contents or not query.strip():
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1)
    # Empty-vocabulary input is valid, e.g. histories containing only emoji.
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(content) for content in contents):
        return []
    matrix = vectorizer.fit_transform(contents)
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    order = scores.argsort()[-top_k:][::-1]
    return [contents[index] for index in order if scores[index] > 0.01]
