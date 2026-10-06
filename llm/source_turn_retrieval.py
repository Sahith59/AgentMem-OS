"""Optional lexical adapter returning source-bound hits instead of flattened text."""

from .evidence_packet import RetrievalHit, digest, eligible_sources


def rank(snapshot, question, *, scope, as_of):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Explicit question required")
    turns = eligible_sources(snapshot, scope=scope, as_of=as_of)
    if not turns:
        return ()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(t.text) for t in turns):
        return ()
    matrix = vectorizer.fit_transform([t.text for t in turns])
    scores = cosine_similarity(vectorizer.transform([question]), matrix)[0]
    return tuple(
        RetrievalHit(t.id, digest(t.text), float(score))
        for t, score in zip(turns, scores, strict=True)
        if score > 0
    )
