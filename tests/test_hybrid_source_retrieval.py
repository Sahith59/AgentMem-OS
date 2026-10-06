import builtins
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest
from agentmem_os.llm.context_assembler import ContextAssembler
from agentmem_os.llm.evidence_packet import RetrievalHit, SourceSnapshot, SourceTurn, digest, pack
from agentmem_os.llm.hybrid_source_retrieval import rank, supplement
from agentmem_os.llm.multi_vector_retrieval import MultiVectorRetriever

NOW = datetime(2026, 1, 3)


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((tuple(texts), kwargs))
        return np.array([[1.0, 0.0] for _ in texts])


def turn(i, text, pos=0, session="s", when=NOW):
    return SourceTurn(i, session, pos, "user", when, text)


def test_injection_preserves_encoder_contract_and_avoids_database_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("agentmem_os.db", "sentence_transformers")):
            raise AssertionError("Unexpected database or model loader import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    encoder = Encoder()
    retriever = MultiVectorRetriever(encoder=encoder, context_turns=0, snippet_chars=0)
    retriever.index(["alpha story", "beta story"])
    assert retriever.search("beta", top_k=1) == ["beta story"]
    assert encoder.calls == [
        (
            ("passage: alpha story", "passage: beta story"),
            dict(normalize_embeddings=True, show_progress_bar=False, batch_size=64),
        ),
        (("query: beta",), dict(normalize_embeddings=True, show_progress_bar=False)),
    ]


def test_existing_rrf_order_and_legacy_neighbor_output_are_preserved():
    retriever = MultiVectorRetriever(encoder=Encoder(), context_turns=1, snippet_chars=0)
    retriever.index(["alpha story", "beta story", "gamma story"])
    ranked = retriever.ranked_indices("absent")
    # Both similarity vectors tie. Existing NumPy reverse argsort favors later indices.
    assert [i for i, _ in ranked] == [2, 1, 0]
    assert [score for _, score in ranked] == pytest.approx([2 / 61, 2 / 62, 2 / 63])
    assert retriever.search("absent", top_k=2) == [
        "beta story\ngamma story",
        "alpha story\nbeta story",
    ]


def test_blank_sources_and_future_sources_cannot_shift_bound_ids():
    encoder = Encoder()
    a = turn("nonsequential-a", "alpha story", 0)
    blank = turn("blank", " \n", 1)
    b = turn("nonsequential-z", "beta story", 2)
    future = turn("future", "beta story", 3, when=NOW + timedelta(days=1))
    hits = rank(
        SourceSnapshot("scope", (a, blank, b, future)),
        "beta",
        scope="scope",
        as_of=NOW,
        encoder=encoder,
    )
    assert hits[0].source_id == b.id
    assert {h.source_id for h in hits} == {a.id, b.id}
    assert hits[0].source_sha256 == digest(b.text)
    assert len(encoder.calls[0][0]) == 2


def test_future_text_cannot_change_hybrid_ranks_or_scores():
    a = turn("a", "alpha story")
    f = turn("f", "alpha future", 1, when=NOW + timedelta(days=1))

    def get(turns):
        return rank(
            SourceSnapshot("scope", turns), "alpha", scope="scope", as_of=NOW, encoder=Encoder()
        )

    assert get((a,)) == get((a, f))


def test_explicit_tie_order_survives_packet_sorting():
    a, z = turn("a", "alpha story"), turn("z", "beta story", session="t")
    hits = [RetrievalHit(z.id, digest(z.text), 1.0, 0), RetrievalHit(a.id, digest(a.text), 1.0, 1)]
    _, report = pack(
        SourceSnapshot("scope", (a, z)),
        hits,
        scope="scope",
        as_of=NOW,
        char_budget=1000,
        max_anchors=1,
        neighbor_turns=0,
    )
    assert report["anchors"] == ["z"]
    for invalid in [-1, True]:
        with pytest.raises(ValueError, match="Unbound"):
            pack(
                SourceSnapshot("scope", (a, z)),
                [replace(hits[0], tie_order=invalid)],
                scope="scope",
                as_of=NOW,
                char_budget=1000,
            )


def test_real_core_entrypoint_retains_baseline_and_exact_scoped_receipts():
    a = turn("a", "The workshop lasted two days.")
    b = turn("b", "It was only planned.", 1)
    foreign = turn("foreign", "Unrelated session.", session="other")
    baseline = "[SEMANTIC FACTS] Baseline fact stays."

    class TopicEncoder(Encoder):
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] if "workshop" in text else [0.0, 1.0] for text in texts])

    text, report = ContextAssembler.assemble_hybrid_source_packet(
        SourceSnapshot("scope", (a, b, foreign)),
        "workshop",
        baseline,
        scope="scope",
        as_of=NOW,
        encoder=TopicEncoder(),
        max_anchors=1,
        extra_budget=1500,
    )
    assert text.startswith(baseline) and len(text) <= 40000
    assert foreign.text not in text
    assert {r["id"] for r in report["receipts"]} == {a.id, b.id}
    for r in report["receipts"]:
        offset = report["block_offset"]
        assert digest(text[offset + r["start"] : offset + r["end"]]) == r["sha256"]


def test_nonfitting_top_anchor_does_not_backfill_low_ranked_source(monkeypatch):
    import agentmem_os.llm.hybrid_source_retrieval as hybrid

    a, b = turn("a", "long " * 1000), turn("b", "small story", session="t")
    monkeypatch.setattr(
        hybrid,
        "rank",
        lambda *args, **kwargs: (
            RetrievalHit(a.id, digest(a.text), 2),
            RetrievalHit(b.id, digest(b.text), 1),
        ),
    )
    text, report = supplement(
        SourceSnapshot("scope", (a, b)),
        "query",
        "baseline",
        scope="scope",
        as_of=NOW,
        max_anchors=1,
        extra_budget=300,
    )
    assert text == "baseline" and report["ranked_sources"] == 2
    assert report["omissions"][0]["reason"] == "anchor_budget_nonfit"


def test_invalid_embeddings_fail_without_fallback():
    class BadEncoder(Encoder):
        def encode(self, texts, **kwargs):
            return np.array([[np.nan, 0.0] for _ in texts])

    with pytest.raises(ValueError, match="passage"):
        MultiVectorRetriever(encoder=BadEncoder()).index(["alpha story"])
    r = MultiVectorRetriever(encoder=Encoder())
    r.index(["alpha story"])
    r._encoder = BadEncoder()
    with pytest.raises(ValueError, match="query"):
        r.ranked_indices("alpha")


def test_invalid_budget_fails_and_no_headroom_keeps_baseline():
    a = turn("a", "alpha story")
    snapshot = SourceSnapshot("scope", (a,))
    with pytest.raises(ValueError, match="budget"):
        supplement(snapshot, "query", "too big", scope="scope", as_of=NOW, char_budget=1)
    text, _ = supplement(
        snapshot, "query", "full", scope="scope", as_of=NOW, char_budget=4, encoder=Encoder()
    )
    assert text == "full"


@pytest.mark.parametrize("value", [[], [[1, 0], [1, 0]], [[1]], [[1j, 0j]], [["x", "y"]]])
def test_malformed_query_batch_fails_closed(value):
    retriever = MultiVectorRetriever(encoder=Encoder())
    retriever.index(["alpha story"])

    class Invalid:
        def encode(self, *args, **kwargs):
            return np.array(value)

    retriever._encoder = Invalid()
    with pytest.raises(ValueError, match="query"):
        retriever.ranked_indices("alpha")


@pytest.mark.parametrize("value", [[[1j, 0j]], [["x", "y"]], [[np.nan, 0]]])
def test_rejected_reindex_cannot_mix_stale_lexical_and_new_dense_state(value):
    retriever = MultiVectorRetriever(encoder=Encoder())
    retriever.index(["alpha story"])

    class Invalid:
        def encode(self, *args, **kwargs):
            return np.array(value)

    retriever._encoder = Invalid()
    with pytest.raises(ValueError, match="passage"):
        retriever.index(["new story"])
    assert retriever.search("alpha") == []
    assert retriever.n_docs == 0


def test_empty_vocabulary_reindex_is_explicit_and_clears_state():
    retriever = MultiVectorRetriever(encoder=Encoder())
    retriever.index(["alpha story"])
    with pytest.raises(ValueError, match="empty vocabulary"):
        retriever.index(["!!!"])
    assert retriever.ranked_indices("alpha") == []
