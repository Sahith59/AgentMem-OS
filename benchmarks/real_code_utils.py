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


class DenseChromaAdapter:
    """
    Drop-in replacement for ChromaManager backed by the product's dense
    retriever (llm/multi_vector_retrieval.py, multilingual-e5-small).

    NOT the default benchmark backend — measured worse than TF-IDF on the
    LoCoMo evidence diagnostic in every variant tried (see
    install_best_chroma's table). Its role is what lexical matching
    fundamentally cannot do: cross-lingual retrieval (a Hindi query
    finding English turns — proven in tests/test_multi_vector_retrieval).
    Kept per-session with staleness detection by turn count.
    """

    def __init__(self):
        self._retrievers: dict = {}

    def search(self, session_id: str, query: str, top_k: int = 5) -> list:
        from agentmem_os.db.engine import get_session as get_db
        from agentmem_os.db.models import Turn
        from agentmem_os.llm.multi_vector_retrieval import MultiVectorRetriever

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

        retriever = self._retrievers.get(session_id)
        if retriever is None or retriever.n_docs != len(contents):
            retriever = MultiVectorRetriever()
            retriever.index(contents)
            self._retrievers[session_id] = retriever

        return retriever.search(query, top_k=top_k)


def install_tfidf_chroma(context_assembler_cls) -> None:
    """Monkey-patch ContextAssembler._get_chroma to use TfIdfChromaAdapter,
    for the lifetime of the process. Only affects scripts that call this
    explicitly — production code is unaffected."""
    adapter = TfIdfChromaAdapter()
    context_assembler_cls._get_chroma = lambda self: adapter


def install_best_chroma(context_assembler_cls) -> str:
    """
    Install the MEASURED-best semantic-search stand-in for benchmark runs,
    and return its name so every result records its provenance.

    Six variants, same 30-question LoCoMo evidence diagnostic (does the
    gold answer reach the assembled context / the 24k-char generation cap):

        tf-idf, bare turns          11/30 full   11/30 in-cap   <- champion
        tf-idf, 15-turn windows      9/30         6/30
        dense, turns+windows        10/30         7/30
        dense, +/-2-turn spans       9/30         6/30
        hybrid RRF, +/-2 spans      10/30         7/30
        hybrid RRF, bare turns       7/30         7/30

    Plain lexical TF-IDF over bare turns wins outright: LoCoMo-style
    questions share rare entity terms with their evidence (lexical
    matching's home turf), a small multilingual encoder ranks topical
    chatter above needles, and rank-fusing the weak dense signal DILUTES
    the strong lexical one. Dense retrieval stays available explicitly
    (install_dense_chroma) for the one thing lexical cannot do at all:
    cross-lingual queries. The next measured lever for beating 11/30 is
    fact decomposition at ingestion (X-MemoryArch's A5), not a better
    same-granularity ranker.
    """
    install_tfidf_chroma(context_assembler_cls)
    return "tfidf"


def install_dense_chroma(context_assembler_cls) -> str:
    """
    Explicit opt-in: the dense cross-lingual retriever (falls back to
    TF-IDF when the multilingual extra isn't installed).
    """
    from agentmem_os.llm.multi_vector_retrieval import is_available

    if is_available():
        adapter = DenseChromaAdapter()
        context_assembler_cls._get_chroma = lambda self: adapter
        return "dense"
    install_tfidf_chroma(context_assembler_cls)
    return "tfidf"
