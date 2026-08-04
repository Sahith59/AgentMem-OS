"""
Multi-vector semantic retrieval over conversation turns — the port of
X-MemoryArch's measured retrieval lessons into AgentMem OS product code.

Design (adapting X-MemoryArch's no-LLM champion, "A4mvr" — 0.822
aggregate R@5 with NO extraction step). Its transferable insight is
score-at-fine-granularity; what gets RETURNED had to be re-measured here:
  • Score individual TURNS only. Two window variants were both measured
    WORSE on the same 30-question evidence diagnostic (TF-IDF windows:
    6/30 vs 11/30; dense turn+window chunks: 7/30 within the generation
    cap vs 11/30) — long chunks win on topical similarity, flood the
    context, and crowd sharp evidence out of the cap. Fine-grained
    scoring is the part of A4mvr that survives contact with this data.
  • HYBRID ranking: TF-IDF and dense rankings fused with reciprocal-rank
    fusion (RRF, k=60). Measured one signal at a time first: pure dense
    (multilingual-e5-small) scored 6/30 within the generation cap vs
    lexical TF-IDF's 11/30 on the same questions — entity-heavy questions
    share rare terms with their evidence, which is lexical matching's
    home turf, while dense catches paraphrase and cross-lingual queries
    that share no tokens at all. Fusion keeps both strengths.
  • Dense side embeds with multilingual-e5-small (the model
    db/entity_aliases.py standardized on — one shared instance) using the
    model card's asymmetric prefixes: "query: "/"passage: ".
  • Return each selected turn expanded with ±CONTEXT_TURNS neighbors —
    evidence plus its local referents ("she", "it", "that trip") —
    never re-selecting covered turns.

FINAL MEASURED VERDICT (LoCoMo evidence diagnostic, 6 variants — full
table in benchmarks/real_code_utils.py): on entity-heavy English
benchmark questions, every dense/hybrid variant LOST to plain TF-IDF over
bare turns, so this retriever is NOT the default benchmark backend. Its
value is the thing lexical matching cannot do at any granularity:
cross-lingual retrieval — a Hindi query retrieves English turns and vice
versa (proven in tests) — the same property the entity-alias layer has,
now at the retrieval level. The next lever for benchmark evidence
coverage is fact decomposition at ingestion, not a better ranker.

Optional dependency: sentence-transformers (multilingual extra). When it
isn't installed, is_available() is False and callers fall back (the
benchmark harness falls back to TF-IDF and says so).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

CONTEXT_TURNS = 2  # neighbors returned on each side of a scored hit


def is_available() -> bool:
    try:
        from agentmem_os.db.entity_aliases import get_shared_encoder
        return get_shared_encoder() is not None
    except Exception:
        return False


class MultiVectorRetriever:
    """
    Immutable-corpus retriever: index() a list of turn strings once,
    search() many times. Rebuild (re-index) when the corpus changes —
    callers detect that by comparing n_docs.
    """

    def __init__(self, context_turns: int = CONTEXT_TURNS):
        self.context_turns = context_turns
        self.n_docs = 0
        self._turns: List[str] = []
        self._matrix = None  # normalized per-turn embeddings

    def index(self, turns: List[str]) -> None:
        from agentmem_os.db.entity_aliases import get_shared_encoder

        model = get_shared_encoder()
        if model is None:
            raise RuntimeError(
                "sentence-transformers not installed — check is_available() first"
            )

        self._turns = [t for t in turns if t and t.strip()]
        self.n_docs = len(self._turns)
        self._matrix = model.encode(
            [f"passage: {t}" for t in self._turns],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf_vec = TfidfVectorizer(max_features=512, sublinear_tf=True, min_df=1)
        self._tfidf_matrix = self._tfidf_vec.fit_transform(self._turns)
        logger.debug(f"[MultiVectorRetriever] indexed {self.n_docs} turns (dense + tfidf)")

    def search(self, query: str, top_k: int = 5) -> List[str]:
        if self._matrix is None or not self._turns:
            return []

        from agentmem_os.db.entity_aliases import get_shared_encoder

        model = get_shared_encoder()
        q = model.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )[0]
        dense_sims = self._matrix @ q

        from sklearn.metrics.pairwise import cosine_similarity

        tfidf_sims = cosine_similarity(
            self._tfidf_vec.transform([query]), self._tfidf_matrix
        )[0]

        # Reciprocal-rank fusion (k=60, the standard constant): robust to
        # the two signals' incomparable score scales, no tuned weights.
        n = len(self._turns)
        rrf = [0.0] * n
        for sims in (dense_sims, tfidf_sims):
            for rank, idx in enumerate(sims.argsort()[::-1]):
                rrf[int(idx)] += 1.0 / (60 + rank + 1)

        import numpy as np

        order = np.argsort(rrf)[::-1]
        covered = set()
        selected: List[str] = []
        n = len(self._turns)
        for idx in order:
            if len(selected) >= top_k:
                break
            i = int(idx)
            if i in covered:
                continue  # this evidence is already inside a returned span
            start = max(0, i - self.context_turns)
            end = min(n, i + self.context_turns + 1)
            covered |= set(range(start, end))
            selected.append("\n".join(self._turns[start:end]))
        return selected
