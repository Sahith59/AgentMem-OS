"""Pins for F-18 snippet packing in MultiVectorRetriever."""
import pytest

from llm.multi_vector_retrieval import MultiVectorRetriever, is_available

pytestmark = pytest.mark.skipif(
    not is_available(), reason="multilingual extra not installed")


def _mk(turns, snippet_chars):
    r = MultiVectorRetriever(context_turns=1, snippet_chars=snippet_chars)
    r.index(turns)
    return r


LONG_FILLER = ("The weather stayed calm and nothing notable happened "
               "during the morning commute. " * 30)
EVIDENCE = "I finally bought the 70-200mm zoom lens at the camera shop."
LONG_HIT = LONG_FILLER + EVIDENCE + " " + LONG_FILLER


def test_short_turns_untouched():
    turns = ["User: hello there", "Assistant: hi, how can I help?",
             "User: what lens did I buy?"]
    r = _mk(turns, 1200)
    out = r.search("what lens did I buy?", top_k=2)
    joined = "\n".join(out)
    assert "[...]" not in joined


def test_long_hit_turn_keeps_query_relevant_region():
    turns = ["User: hello", LONG_HIT, "User: unrelated goodbye chatter"]
    r = _mk(turns, 400)
    out = r.search("which zoom lens did I purchase?", top_k=1)
    joined = "\n".join(out)
    assert "70-200mm" in joined, "snippet must keep the evidence region"
    assert len(max(joined.split("\n"), key=len)) < len(LONG_HIT)
    assert "[...]" in joined, "elision must be marked"


def test_neighbor_turns_head_capped_in_tail_spans():
    # F-18b: neighbors inside DEEP spans keep depth (the ssa answer
    # lives in a neighbor); this pin covers TAIL spans, where the
    # 800-style head cap still applies. deep_hits=0 makes every span a
    # tail span.
    turns = [LONG_FILLER + " neighbor tail marker ZZZ",
             "User: I bought the 70-200mm zoom lens yesterday.",
             "User: ok"]
    r = _mk(turns, 300)
    r.snippet_deep_hits = 0
    out = r.search("70-200mm zoom lens", top_k=1)
    joined = "\n".join(out)
    assert "ZZZ" not in joined, "tail-span neighbor beyond cap must be cut"
    assert "70-200mm" in joined


def test_deep_span_neighbor_keeps_depth():
    long_answer = ("Assistant: Sure, here are the details. "
                   + "Filler about many things. " * 60
                   + "The code you asked about was ULTRAVIOLET-77. "
                   + "More trailing filler. " * 20)
    turns = ["User: what was that code from our chat about the project?",
             long_answer, "User: thanks"]
    r = _mk(turns, 300)
    r.snippet_deep_hits = 4
    r.snippet_deep_chars = 8000
    out = r.search("what was the code from our previous conversation?",
                   top_k=1)
    assert "ULTRAVIOLET-77" in "\n".join(out), \
        "deep-span neighbor must keep the deep answer"


def test_zero_disables_byte_identical():
    turns = ["User: hello", LONG_HIT, "User: goodbye"]
    r_off = _mk(turns, 0)
    out = r_off.search("which zoom lens did I purchase?", top_k=1)
    assert LONG_HIT in "\n".join(out), "cap=0 must return whole turns"
    assert "[...]" not in "\n".join(out)


def test_env_default_used(monkeypatch):
    monkeypatch.setenv("AGENTMEM_OS_SNIPPET_CHARS", "777")
    r = MultiVectorRetriever()
    assert r.snippet_chars == 777


def test_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("AGENTMEM_OS_SNIPPET_CHARS", "777")
    r = MultiVectorRetriever(snippet_chars=555)
    assert r.snippet_chars == 555


def test_deep_hits_keep_more_depth():
    long_evidence = ("Filler sentence about ordinary things. " * 40
                     + "The rare serial number is XK-4471-OMEGA. "
                     + "More filler about the weather afterwards. " * 40)
    turns = ["User: hello", long_evidence, "User: bye",
             "User: unrelated topic entirely about gardening"]
    r = MultiVectorRetriever(context_turns=0, snippet_chars=200)
    r.snippet_deep_hits = 1
    r.snippet_deep_chars = 5000
    r.index(turns)
    out = r.search("what is the rare serial number?", top_k=2)
    assert "XK-4471-OMEGA" in "\n".join(out)
    # the deep hit kept (nearly) everything: far more than 200 chars
    assert max(len(s) for s in out) > 2000


def test_deep_cap_disabled_with_kill_switch():
    long_turn = "word " * 2000
    r = MultiVectorRetriever(context_turns=0, snippet_chars=0)
    r.index(["User: hi", long_turn, "User: bye"])
    out = r.search("word", top_k=1)
    assert long_turn.strip() in "\n".join(out)
