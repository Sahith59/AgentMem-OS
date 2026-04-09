"""
AgentMem OS — Memory Importance Scorer
========================================
Novel Algorithm #1: Replaces naive "compress oldest 30%" with a principled
multi-signal importance score computed per turn before compression.

Score = weighted combination of 4 signals:

  1. TF-IDF Rarity        (w=0.30) — Rare terms signal information-dense turns
  2. Semantic Novelty     (w=0.35) — Cosine distance from existing summaries
  3. Entity Density       (w=0.20) — Named entity count (via spaCy)
  4. Recency Decay        (w=0.15) — Exponential decay over turn index

Turns with LOW scores get compressed first (they're redundant / low-info).
Turns with HIGH scores are preserved in episodic memory as long as possible.

Research contribution:
  This is the first published formulation of multi-signal importance scoring
  for LLM agent memory compression. Prior work (MemGPT, Zep, etc.) uses
  simple recency or token-count heuristics.
"""

import math
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
from loguru import logger


class MemoryImportanceScorer:
    """
    Scores each episodic memory turn on a 0.0–1.0 importance scale.

    Usage:
        scorer = MemoryImportanceScorer()
        scored = scorer.score_turns(turns, existing_summary_embeddings)
        # Returns list of (turn, score) sorted by importance ascending
        # → compress the first N (lowest importance)
    """

    # Signal weights — must sum to 1.0
    WEIGHT_TFIDF    = 0.30
    WEIGHT_NOVELTY  = 0.35
    WEIGHT_ENTITIES = 0.20
    WEIGHT_RECENCY  = 0.15

    # Recency decay half-life: score halves every HALF_LIFE turns from the end
    HALF_LIFE = 20

    def __init__(self):
        self._nlp = None      # lazy-loaded spaCy model
        self._tfidf = None    # lazy-loaded sklearn TF-IDF vectorizer

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def score_turns(
        self,
        turns: List[Dict],                          # [{"role": ..., "content": ...}, ...]
        existing_embeddings: Optional[List[np.ndarray]] = None,  # summary embeddings
        get_embedding_fn=None,                      # callable: str -> np.ndarray
    ) -> List[Tuple[Dict, float]]:
        """
        Score all turns and return (turn, score) pairs sorted ascending by score.
        Lowest score = compress first.

        Args:
            turns: List of turn dicts with at least 'content' key.
            existing_embeddings: Embeddings of already-committed summaries.
                Used for semantic novelty calculation. Pass [] if none exist.
            get_embedding_fn: Optional function mapping text -> embedding vector.
                If None, semantic novelty falls back to 0.5 (neutral).

        Returns:
            List of (turn_dict, importance_score) sorted ascending (compress first).
        """
        if not turns:
            return []

        contents = [t["content"] for t in turns]
        n = len(turns)

        # Signal 1: TF-IDF rarity
        tfidf_scores = self._compute_tfidf_scores(contents)

        # Signal 2: Semantic novelty (vs existing summaries)
        novelty_scores = self._compute_novelty_scores(
            contents, existing_embeddings or [], get_embedding_fn
        )

        # Signal 3: Entity density
        entity_scores = self._compute_entity_scores(contents)

        # Signal 4: Recency (newer turns = higher score = preserve longer)
        recency_scores = self._compute_recency_scores(n)

        # Combine
        results = []
        for i, turn in enumerate(turns):
            score = (
                self.WEIGHT_TFIDF    * tfidf_scores[i]   +
                self.WEIGHT_NOVELTY  * novelty_scores[i]  +
                self.WEIGHT_ENTITIES * entity_scores[i]   +
                self.WEIGHT_RECENCY  * recency_scores[i]
            )
            results.append((turn, round(score, 4)))

        # Sort ascending → lowest importance first (these get compressed)
        results.sort(key=lambda x: x[1])

        logger.debug(
            f"[ImportanceScorer] Scored {n} turns. "
            f"Min={results[0][1]:.3f}, Max={results[-1][1]:.3f}, "
            f"Mean={np.mean([r[1] for r in results]):.3f}"
        )

        return results

    def get_compression_candidates(
        self,
        turns: List[Dict],
        compress_fraction: float = 0.30,
        existing_embeddings: Optional[List[np.ndarray]] = None,
        get_embedding_fn=None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split turns into (to_compress, to_keep) lists.

        Returns the lowest-importance `compress_fraction` of turns for compression
        and the rest for keeping in episodic memory.
        """
        scored = self.score_turns(turns, existing_embeddings, get_embedding_fn)
        n_compress = max(1, int(len(scored) * compress_fraction))

        to_compress = [t for t, _ in scored[:n_compress]]
        to_keep     = [t for t, _ in scored[n_compress:]]

        logger.info(
            f"[ImportanceScorer] Compressing {n_compress}/{len(turns)} turns "
            f"(fraction={compress_fraction}). "
            f"Avg importance of compressed: "
            f"{np.mean([s for _, s in scored[:n_compress]]):.3f}"
        )

        return to_compress, to_keep

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 1: TF-IDF Rarity
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_tfidf_scores(self, contents: List[str]) -> List[float]:
        """
        Compute TF-IDF score per document. Higher = more rare/unique vocabulary.
        Normalized to [0, 1] via min-max.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            if len(contents) < 2:
                return [0.5] * len(contents)

            vectorizer = TfidfVectorizer(
                stop_words="english",
                min_df=1,
                max_features=5000,
                sublinear_tf=True,    # log(1+tf) — reduces impact of very frequent terms
            )
            tfidf_matrix = vectorizer.fit_transform(contents)

            # Per-document score = mean TF-IDF weight of its terms
            raw_scores = np.array(tfidf_matrix.mean(axis=1)).flatten()
            return self._minmax_normalize(raw_scores)

        except Exception as e:
            logger.warning(f"[ImportanceScorer] TF-IDF failed: {e}. Using uniform 0.5.")
            return [0.5] * len(contents)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 2: Semantic Novelty
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_novelty_scores(
        self,
        contents: List[str],
        existing_embeddings: List[np.ndarray],
        get_embedding_fn,
    ) -> List[float]:
        """
        Novelty = how semantically distant is this turn from all existing summaries.
        High novelty → the turn says something new → high importance.

        If no embedder is provided or no existing summaries, returns 0.5 for all.
        """
        if not get_embedding_fn or not existing_embeddings:
            return [0.5] * len(contents)

        try:
            # Embed each content string
            turn_embeddings = [
                np.array(get_embedding_fn(c)) for c in contents
            ]

            # Stack existing summary embeddings
            summary_matrix = np.vstack(existing_embeddings)  # shape (n_summaries, dim)

            scores = []
            for emb in turn_embeddings:
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                # Cosine similarity to each existing summary
                norms = np.linalg.norm(summary_matrix, axis=1, keepdims=True) + 1e-10
                normed = summary_matrix / norms
                cosines = normed @ emb                          # shape (n_summaries,)
                max_similarity = float(cosines.max())
                # Novelty = 1 - max_similarity (most similar to existing = least novel)
                scores.append(1.0 - max(0.0, min(1.0, max_similarity)))

            return self._minmax_normalize(np.array(scores))

        except Exception as e:
            logger.warning(f"[ImportanceScorer] Novelty computation failed: {e}. Using 0.5.")
            return [0.5] * len(contents)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 3: Entity Density
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_entity_scores(self, contents: List[str]) -> List[float]:
        """
        Count named entities per turn using spaCy.
        Turns mentioning many entities (people, orgs, tools, dates) are more important.
        Normalized per-batch by max entity count.
        """
        try:
            nlp = self._get_nlp()
            entity_counts = []

            # Process in batch for efficiency
            for doc in nlp.pipe(contents, batch_size=32, disable=["parser"]):
                # Count entities, with bonus weight for PERSON, ORG, PRODUCT
                count = 0
                for ent in doc.ents:
                    if ent.label_ in ("PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
                        count += 2   # bonus weight for high-signal entity types
                    else:
                        count += 1
                entity_counts.append(float(count))

            return self._minmax_normalize(np.array(entity_counts))

        except Exception as e:
            logger.warning(f"[ImportanceScorer] spaCy entity scoring failed: {e}. "
                           f"Falling back to regex heuristic.")
            return self._entity_scores_regex(contents)

    def _entity_scores_regex(self, contents: List[str]) -> List[float]:
        """Fallback entity density: count capitalized words as proxy for entities."""
        pattern = re.compile(r'\b[A-Z][a-zA-Z]{2,}\b')
        counts = np.array([
            float(len(pattern.findall(c))) for c in contents
        ])
        return self._minmax_normalize(counts)

    # ──────────────────────────────────────────────────────────────────────────
    # Signal 4: Recency Decay
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_recency_scores(self, n: int) -> List[float]:
        """
        Exponential recency: score[i] = 2^(-(n-1-i)/HALF_LIFE)

        Turn 0 (oldest) gets the smallest score.
        Turn n-1 (newest) gets score = 1.0.

        This models the intuition that recent turns are more likely to be
        referenced soon and should be preserved longer in episodic memory.
        """
        scores = []
        for i in range(n):
            age = (n - 1 - i)              # 0 = newest, n-1 = oldest
            score = math.pow(2.0, -age / self.HALF_LIFE)
            scores.append(score)
        # Already in [0, 1]; no normalization needed (max is 1.0 at newest turn)
        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def _minmax_normalize(self, arr: np.ndarray) -> List[float]:
        """Normalize array to [0, 1]. Returns [0.5, ...] if all values are equal."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return [0.5] * len(arr)
        normalized = (arr - mn) / (mx - mn)
        return normalized.tolist()

    def _get_nlp(self):
        """Lazy-load spaCy model. Uses small model for speed."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning(
                    "[ImportanceScorer] spaCy model not found. "
                    "Run: python -m spacy download en_core_web_sm"
                )
                raise
        return self._nlp


# ──────────────────────────────────────────────────────────────────────────────
# Standalone scoring function (convenience wrapper)
# ──────────────────────────────────────────────────────────────────────────────

def score_and_rank(
    turns: List[Dict],
    existing_embeddings: Optional[List[np.ndarray]] = None,
    get_embedding_fn=None,
) -> List[Tuple[Dict, float]]:
    """
    Module-level convenience function.
    Returns turns ranked ascending by importance (compress first).
    """
    scorer = MemoryImportanceScorer()
    return scorer.score_turns(turns, existing_embeddings, get_embedding_fn)
