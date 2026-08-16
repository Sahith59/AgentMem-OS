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

# Snippet packing (2026-08-14, F-18). The truncated-harness era hid a
# breadth collapse: with full turns, one hit's span can cost 3-20k chars,
# so a fixed context budget holds HALF the distinct sessions it used to
# (measured on the n=500 dual-stack audit: KU 10.0 -> 5.4 sessions in
# packet, pref gold-session coverage 0.93 -> 0.67 — and KU/pref scores
# regressed with it). A turn longer than SNIPPET_CHARS therefore
# contributes a SNIPPET, not its whole body: the scored hit turn keeps
# the region that best matches the query (sentence-window scored with
# the already-fitted TF-IDF vectorizer — cheap, deterministic); neighbor
# turns keep their head (they are referent context, not evidence).
# Elisions are marked so the reader model knows text was cut. 0 disables
# (byte-identical to the old behaviour).
SNIPPET_CHARS = 800


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

    def __init__(self, context_turns: int = CONTEXT_TURNS,
                 snippet_chars: Optional[int] = None):
        import os
        self.context_turns = context_turns
        self.snippet_chars = (
            snippet_chars if snippet_chars is not None
            else int(os.environ.get("AGENTMEM_OS_SNIPPET_CHARS",
                                    str(SNIPPET_CHARS))))
        self.snippet_deep_hits = int(os.environ.get(
            "AGENTMEM_OS_SNIPPET_DEEP_HITS", "0"))
        self.snippet_deep_chars = int(os.environ.get(
            "AGENTMEM_OS_SNIPPET_DEEP_CHARS", "3000"))
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

    def _snippet(self, text: str, query: str, is_hit: bool,
                 cap_override: Optional[int] = None) -> str:
        """Trim a turn longer than snippet_chars to its most relevant
        region (hit turns) or its head (neighbor turns). Sentence-window
        scoring reuses the fitted TF-IDF vectorizer; ties and failures
        fall back to the head — deterministic either way. Elisions are
        marked with [...] so the reader knows text was cut."""
        cap = cap_override if cap_override is not None else self.snippet_chars
        if cap <= 0 or len(text) <= cap:
            return text
        if not is_hit:
            return text[:cap] + " [...]"
        import re as _re
        # sentence-ish segments; keeps code/lists intact well enough
        segs = [s for s in _re.split(r"(?<=[.!?\n])\s+", text) if s]
        if len(segs) < 2:
            return text[:cap] + " [...]"
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(
                self._tfidf_vec.transform([query]),
                self._tfidf_vec.transform(segs))[0]
        except Exception:
            return text[:cap] + " [...]"
        # best window of consecutive segments within the cap, greedy
        # around the single best segment
        best = int(sims.argmax())
        lo = hi = best
        size = len(segs[best])
        while size < cap:
            left = sims[lo - 1] if lo > 0 else -1.0
            right = sims[hi + 1] if hi + 1 < len(segs) else -1.0
            if left < 0 and right < 0:
                break
            if right >= left:
                if size + len(segs[hi + 1]) > cap:
                    if left >= 0 and size + len(segs[lo - 1]) <= cap:
                        lo -= 1
                        size += len(segs[lo])
                        continue
                    break
                hi += 1
                size += len(segs[hi])
            else:
                if size + len(segs[lo - 1]) > cap:
                    if right >= 0 and size + len(segs[hi + 1]) <= cap:
                        hi += 1
                        size += len(segs[hi])
                        continue
                    break
                lo -= 1
                size += len(segs[lo])
        out = " ".join(segs[lo:hi + 1])
        pre = "[...] " if lo > 0 else ""
        post = " [...]" if hi + 1 < len(segs) else ""
        return pre + out + post

    def search(self, query: str, top_k: int = 5,
               deep_hits: Optional[int] = None) -> List[str]:
        """deep_hits — BREADTH-THEN-DEPTH span policy (2026-08-12,
        measured). Fixed-width spans force a bad trade: at ±2 every hit
        returns 5 turns, so ~5x fewer DISTINCT evidence points fit a
        budget (fatal for multi-hop counting: full gold-session coverage
        108/150); at 0 every hit is naked and referent-dependent answers
        break ('how many bikes in MARCH' lost the neighbor carrying the
        date — 4 previously-stable questions failed). Measured on both
        axes: ctx=2 -> 108 coverage / context intact; ctx=0 -> 139 / 4
        known context-breaks; deep_hits=4 with ±1 -> 134 coverage AND
        context kept on all 4 known breaks.
        None (default) = every hit expanded ±context_turns, byte-identical
        to the old behaviour for every existing caller. An int N = the
        top-N ranked hits carry ±1 neighbor, the rest arrive hit-only."""
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
            if deep_hits is None:
                ctx = self.context_turns
            else:
                ctx = 1 if len(selected) < deep_hits else 0
            start = max(0, i - ctx)
            end = min(n, i + ctx + 1)
            covered |= set(range(start, end))
            # F-18b (2026-08-16): asymmetric caps. The n=500 forensics
            # showed uniform 800-char snippets elide the answer sentence
            # inside the TOP hits (5/8 lost ssa answers, 14/14 broken
            # multi-session counting questions had elided items). The
            # ranker's most confident hits keep near-full depth (bounded
            # so one 17k-char turn cannot eat the budget); only tail
            # hits pay the 800-char breadth price.
            deep_cap = self.snippet_deep_chars if (
                self.snippet_chars > 0
                and len(selected) < self.snippet_deep_hits) else None
            # Deep treatment covers the WHOLE span: in assistant-recall
            # questions the query matches the USER turn (the hit) while
            # the answer lives in its NEIGHBOR — protecting only the hit
            # was measured useless (the 8 ssa losses kept ans=False).
            # DEFAULT OFF (deep_hits=0): span-wide depth restored only
            # 3/8 ssa answers while collapsing ku breadth 9.4 -> 6.0
            # sessions (measured 2026-08-16) — the b5e68ff "packing
            # trades, it does not add" conclusion at yet another
            # operating point. Kept as an env-gated experiment lever.
            selected.append("\n".join(
                self._snippet(self._turns[j], query, is_hit=(j == i),
                              cap_override=deep_cap)
                for j in range(start, end)))
        return selected
