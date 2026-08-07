"""
Cross-lingual entity alias resolution — the live wiring of the measured
benchmarks/cross_lingual_kg_eval.py methodology into the knowledge graph.

Method must stay byte-identical to the eval script's: multilingual-e5-small,
"query: " prefix (the model card's own convention for short texts),
normalize_embeddings=True, cosine via dot product. The eval's measured
threshold table (benchmarks/cross_lingual_kg_eval_results.json) is only
valid for THIS exact setup — changing the model, prefix, or normalization
invalidates the measured τ.

Default τ=0.90 is the eval's own recommended_threshold (precision 0.762,
recall 0.533 on the hand-labeled EN/HI/TA set). Known measured limitation
at this τ: one hard negative (Chennai/China, similarity 0.9010) still
clears it — which is why the KG stores alias links as non-destructive
ALIAS_OF edges instead of merging nodes: a false alias adds retrieval
noise but never corrupts or deletes a fact, and it stays inspectable in
the DB (confidence column = the cosine similarity that created it).

Scope: only entity mentions containing Indic-script characters
(U+0900–U+0D7F: Devanagari through Malayalam) participate as the NEW side
of an alias check. General same-script entity dedup (e.g. Bengaluru vs
Bangalore) is deliberately out of scope — the eval's hard-negative table
shows near-name Latin pairs (Chennai/China 0.9010) are exactly where this
model false-merges, and that feature was never measured.

Optional dependency: sentence-transformers (pip install
"agentmem-os[multilingual]"). Absent → resolver reports enabled=False and
every caller no-ops; core installs stay torch-free.

Env:
  AGENTMEM_OS_CROSS_LINGUAL=0        disable even when installed (default: on)
  AGENTMEM_OS_CROSS_LINGUAL_TAU=0.95 override τ (default: 0.90)
"""
import os
import re
import threading
from typing import Dict, Iterable, List, Optional, Tuple

from loguru import logger

DEFAULT_TAU = 0.90
_MODEL_NAME = "intfloat/multilingual-e5-small"

# Devanagari (0900) through Malayalam (0D7F) — one contiguous block covering
# Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada,
# Malayalam. Two+ chars so stray combining marks never match alone.
# Danda/double-danda (U+0964/0965) are PUNCTUATION inside that block and
# are excluded (Stage-3 G3 R1): "है।" tokenizing WITH its danda cost
# 22-43% of the τ margin in real similarity checks and manufactured
# sentence-final duplicate nodes ("चेन्नई" vs "चेन्नई।") that then
# alias-chained to each other instead of to the Latin anchor.
_INDIC_TOKEN_RE = re.compile("[\\u0900-\\u0963\\u0966-\\u0D7F]{2,}")


def contains_indic(text: str) -> bool:
    return bool(_INDIC_TOKEN_RE.search(text))


class EntityAliasResolver:
    """
    Embedding-based alias matcher with a per-instance embedding cache.
    Thread-safe; the model loads lazily on first real use and a missing
    sentence-transformers install degrades to enabled=False, never raises.
    """

    def __init__(self, tau: Optional[float] = None):
        if tau is None:
            tau = float(os.environ.get("AGENTMEM_OS_CROSS_LINGUAL_TAU", DEFAULT_TAU))
        self.tau = tau
        self._model = None
        self._model_failed = False
        self._cache: Dict[str, "object"] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        if os.environ.get("AGENTMEM_OS_CROSS_LINGUAL", "1") == "0":
            return False
        return self._get_model() is not None

    def _get_model(self):
        if self._model is None and not self._model_failed:
            with self._lock:
                if self._model is None and not self._model_failed:
                    try:
                        from sentence_transformers import SentenceTransformer
                        self._model = SentenceTransformer(_MODEL_NAME)
                    except Exception as e:
                        self._model_failed = True
                        logger.debug(
                            f"[EntityAliases] cross-lingual resolution off — "
                            f"sentence-transformers unavailable: {e}"
                        )
        return self._model

    def _embed(self, texts: List[str]):
        model = self._get_model()
        missing = [t for t in texts if t not in self._cache]
        if missing:
            with self._lock:
                missing = [t for t in missing if t not in self._cache]
                if missing:
                    vecs = model.encode(
                        [f"query: {t}" for t in missing],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    )
                    for t, v in zip(missing, vecs):
                        self._cache[t] = v
        return {t: self._cache[t] for t in texts}

    def find_aliases(
        self, text: str, candidates: Iterable[str]
    ) -> List[Tuple[str, float]]:
        """
        Candidates whose cosine similarity to `text` clears τ, best first.
        Casefold-identical candidates are excluded (same surface form is
        the same node already, not an alias).
        """
        if not self.enabled:
            return []
        cands = [c for c in candidates if c and c.casefold() != text.casefold()]
        if not cands:
            return []
        embs = self._embed([text] + cands)
        anchor = embs[text]
        scored = [
            (c, float(anchor @ embs[c]))
            for c in cands
        ]
        matches = [(c, s) for c, s in scored if s >= self.tau]
        matches.sort(key=lambda x: -x[1])
        return matches

    @staticmethod
    def extract_script_tokens(text: str) -> List[str]:
        """Deduplicated Indic-script tokens in order of first
        appearance. Pure digit runs are dropped (G3 R2 minor: the Indic
        block contains each script's digits — ०१२ etc. — and a numeral
        is never an entity mention; str.isdigit() covers all of them)."""
        seen = set()
        out = []
        for tok in _INDIC_TOKEN_RE.findall(text):
            if tok not in seen and not tok.isdigit():
                seen.add(tok)
                out.append(tok)
        return out


def get_shared_encoder():
    """
    The raw shared SentenceTransformer, or None when the multilingual
    extra isn't installed. Exists so other consumers of the same model
    (llm/multi_vector_retrieval.py) reuse one loaded instance instead of
    duplicating ~120MB of weights in RAM.
    """
    return get_alias_resolver()._get_model()


_resolver: Optional[EntityAliasResolver] = None
_resolver_lock = threading.Lock()


def get_alias_resolver() -> EntityAliasResolver:
    """Process-wide singleton so the model and embedding cache are shared."""
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                _resolver = EntityAliasResolver()
    return _resolver


def reset_alias_resolver() -> None:
    """Testing hook: drop the singleton so env overrides take effect."""
    global _resolver
    with _resolver_lock:
        _resolver = None
