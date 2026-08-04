"""
Tests for llm/multi_vector_retrieval.py — the dense multi-vector semantic
retriever (X-MemoryArch A4mvr port: turn + window chunks, per-chunk
embeddings, coverage-deduped greedy selection).

Assertions are CONTENT-based (does the returned context contain the
evidence?), not rank-based — a window chunk containing the answer is as
good as the bare turn for downstream generation, and rank order between
them is model-version-sensitive.

All tests skip when sentence-transformers (multilingual extra) isn't
installed; CI stays torch-free by design.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest

try:
    import sentence_transformers  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False

needs_model = pytest.mark.skipif(
    not HAS_ST, reason="multilingual extra (sentence-transformers) not installed"
)

TURNS = [
    "Hey, how was your weekend?",
    "Pretty good, went hiking near the lake.",
    "Nice! By the way, Sahith Thummala started working at Google in Bengaluru last month.",
    "Oh that's great news for him.",
    "Yeah, he is on the maps infrastructure team.",
    "Did you watch the cricket match yesterday?",
    "Yes, what a finish that was!",
    "I also adopted a puppy named Biscuit last week.",
    "Aww, what breed is Biscuit?",
    "A golden retriever, very playful.",
]


@pytest.fixture(scope="module")
def retriever():
    from agentmem_os.llm.multi_vector_retrieval import MultiVectorRetriever
    r = MultiVectorRetriever(context_turns=2)
    r.index(TURNS)
    return r


@needs_model
def test_english_needle_is_retrieved(retriever):
    ctx = " ".join(retriever.search("Where does Sahith work?", top_k=3))
    assert "Google" in ctx


@needs_model
def test_hindi_query_retrieves_english_evidence(retriever):
    # Cross-lingual retrieval falls out of the multilingual encoder — the
    # semantic tier itself, not just entity aliasing, crosses languages.
    ctx = " ".join(retriever.search("साहित कहाँ काम करता है?", top_k=3))
    assert "Google" in ctx


@needs_model
def test_second_needle_is_retrieved(retriever):
    ctx = " ".join(retriever.search("What is the name of the puppy?", top_k=3))
    assert "Biscuit" in ctx


@needs_model
def test_every_selected_chunk_adds_new_content(retriever):
    # The dedupe invariant the design actually promises: a chunk is never
    # selected if every turn in it is already covered by earlier picks.
    # (A window MAY contain an already-picked turn — as long as it also
    # brings new ones.)
    hits = retriever.search("Where does Sahith work?", top_k=5)
    covered = set()
    for h in hits:
        turns_in_hit = {i for i, t in enumerate(TURNS) if t in h}
        assert turns_in_hit - covered, "chunk selected without adding any new turn"
        covered |= turns_in_hit


def test_install_best_chroma_is_the_measured_champion():
    # TF-IDF over bare turns won the 6-variant evidence measurement (see
    # install_best_chroma's docstring) — "best" must mean measured-best,
    # not newest.
    from benchmarks.real_code_utils import install_best_chroma

    class _Dummy:
        pass

    assert install_best_chroma(_Dummy) == "tfidf"
    assert _Dummy()._get_chroma() is not None


@needs_model
def test_install_dense_chroma_is_available_as_opt_in():
    from benchmarks.real_code_utils import install_dense_chroma

    class _Dummy:
        pass

    assert install_dense_chroma(_Dummy) == "dense"
    assert _Dummy()._get_chroma() is not None
