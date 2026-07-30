"""
Shared utilities for benchmark scripts that exercise real agentmem_os
internals (ablation_study_real.py, qa_accuracy_eval.py) rather than a
hand-rolled simulation of them.

Kept separate from _tier_lib.py, which the pure-simulation scripts import:
_tier_lib.py stays free of the SQLAlchemy/scikit-learn dependency footprint
this module needs.
"""


class TfIdfChromaAdapter:
    """
    Drop-in replacement for ChromaManager — TF-IDF semantic search over a
    session's SQLite turns. Same swap tests/test_e2e_claude.py makes for
    its own E2E run, so any script using this runs with zero external
    dependencies: no Ollama, no Chroma server, no embedding API key.
    """

    def search(self, session_id: str, query: str, top_k: int = 5) -> list:
        from agentmem_os.db.engine import get_session as get_db
        from agentmem_os.db.models import Turn

        db = get_db()
        try:
            rows = (
                db.query(Turn)
                .filter(Turn.session_id == session_id)
                .order_by(Turn.id.asc())
                .all()
            )
            contents = [r.content for r in rows if r.content]
        finally:
            db.close()

        if len(contents) < 3:
            return contents

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        matrix = vec.fit_transform(contents)
        q_vec = vec.transform([query])
        sims = cosine_similarity(q_vec, matrix)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        return [contents[i] for i in top_idx if sims[i] > 0.01]


def install_tfidf_chroma(context_assembler_cls) -> None:
    """Monkey-patch ContextAssembler._get_chroma to use TfIdfChromaAdapter,
    for the lifetime of the process. Only affects scripts that call this
    explicitly — production code is unaffected."""
    adapter = TfIdfChromaAdapter()
    context_assembler_cls._get_chroma = lambda self: adapter
