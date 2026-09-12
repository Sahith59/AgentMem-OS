from agentmem_os.benchmarks.recall_span_adapter import (
    RecallSpanTfIdfAdapter,
    _best_window,
    select_recall_spans,
)


class _Base:
    def __init__(self, rows, reserve=None):
        self.rows = rows
        self.last_receipt = {"reserve": list(reserve or [])}

    def search(self, session_id, query, top_k=5):
        return self.rows[:top_k]


def test_selects_user_request_and_immediate_assistant_reply():
    groups = [[
        {"role": "user", "content": "[2023/05/26] List work from home jobs for seniors"},
        {"role": "assistant", "content": "[2023/05/26] 1. Tutor 2. Editor 7. Transcriptionist"},
    ], [
        {"role": "user", "content": "[2023/05/20] List outdoor games"},
        {"role": "assistant", "content": "[2023/05/20] Chess and tennis"},
    ]]
    selected = select_recall_spans(
        "What was the seventh work from home job for seniors?", groups)
    assert selected == [groups[0][0]["content"], groups[0][1]["content"]]


def test_group_context_disambiguates_generic_recall_wording():
    groups = [[
        {"role": "assistant", "content": "Ideas for the Radiation Amplified zombie."},
        {"role": "user", "content": "Can you suggest some one-word names?"},
        {"role": "assistant", "content": "Radik, Irradon, and Fissionator."},
        {"role": "user", "content": "Fissionator is a really cool one."},
        {"role": "assistant", "content": "The Fissionator design can use a protective suit."},
    ], [
        {"role": "user", "content": "What was the word you were not supposed to remember?"},
        {"role": "assistant", "content": "I cannot provide that word."},
    ]]
    selected = select_recall_spans(
        "What did we finally decide to name the Radiation Amplified zombie?",
        groups,
    )
    assert selected[0] in {groups[0][1]["content"], groups[0][3]["content"]}
    assert selected[1] in {groups[0][2]["content"], groups[0][4]["content"]}


def test_group_context_cannot_bypass_similarity_floor_when_unrelated():
    groups = [[
        {"role": "user", "content": "Tell me about garden soil."},
        {"role": "assistant", "content": "Use compost."},
    ]]
    assert select_recall_spans(
        "What did we decide to name the Radiation Amplified zombie?",
        groups,
        min_similarity=0.95,
    ) == []


def test_oversized_turn_returns_source_only_relevant_window():
    text = "A" * 3500 + " Construction of the house began in 2014. " + "B" * 3500
    window = _best_window(text, "When did construction of the house begin?", 3200)
    assert len(window) == 3200
    assert "began in 2014" in window
    assert window in text


def test_no_admission_is_byte_preserving_even_with_duplicate_chunks():
    base_rows = ["duplicate", "duplicate", "other"]
    adapter = RecallSpanTfIdfAdapter(
        {"s": [[{"role": "user", "content": "unrelated"}]]},
        min_similarity=0.99,
        base=_Base(base_rows),
    )
    assert adapter.search("s", "our previous conversation about astronomy") == base_rows
    assert adapter.last_receipt["reason"] == "no_admission"


def test_non_recall_query_is_an_exact_noop():
    base_rows = ["one", "two"]
    adapter = RecallSpanTfIdfAdapter(
        {"s": [[
            {"role": "user", "content": "Suggest a hotel in Amsterdam"},
            {"role": "assistant", "content": "International Budget Hostel"},
        ]]},
        base=_Base(base_rows),
    )
    assert adapter.search("s", "Where should I stay in Amsterdam?") == base_rows
    assert adapter.last_receipt["reason"] == "not_recall_intent"


def test_preserves_nested_reserve_metadata_for_context_assembler():
    base_rows = ["dated event", "ordinary"]
    adapter = RecallSpanTfIdfAdapter(
        {}, base=_Base(base_rows, reserve=["dated event"]))
    assert adapter.search("s", "What happened two weeks ago?") == base_rows
    assert adapter.last_receipt["reserve"] == ["dated event"]
    assert adapter.last_receipt["nested_reserve_count"] == 1


def test_admitted_recall_span_is_actually_promoted_ahead_of_base_chunks():
    groups = [[
        {"role": "user", "content": "List work from home jobs for seniors"},
        {"role": "assistant", "content": "7. Transcriptionist"},
    ]]
    adapter = RecallSpanTfIdfAdapter(
        {"s": groups}, base=_Base(["ordinary one", "ordinary two"]))
    result = adapter.search(
        "s", "Remind me about work from home jobs for seniors", top_k=3)
    assert result == [groups[0][0]["content"], groups[0][1]["content"],
                      "ordinary one"]
    assert adapter.last_receipt["reserve"][:2] == result[:2]
