from agentmem_os.benchmarks.dated_event_adapter import (
    DatedEventContextAssembler,
    DatedEventTfIdfAdapter,
)


class FakeBase:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, session_id, query, top_k=5):
        self.calls.append((session_id, query, top_k))
        return self.chunks[:top_k]


def _turn(content, role="user"):
    return {"role": role, "content": content}


def test_adapter_prepends_bounded_event_and_preserves_base_order():
    query = "Where was the art event two weeks ago?"
    target = "[2023/01/15] I attended an exhibit at the Metropolitan Museum."
    base = FakeBase(["ordinary-a", "ordinary-b", target])
    adapter = DatedEventTfIdfAdapter(
        {query: "2023/02/01"}, reserve_limit=1, base=base,
        turn_loader=lambda _: [
            _turn(target),
            _turn("[2023/01/15] I finished assembling a model car."),
        ])

    assert adapter.search("session", query, top_k=3) == [
        target, "ordinary-a", "ordinary-b"]
    assert adapter.last_receipt["reserve_count"] == 1
    assert base.calls == [("session", query, 3)]


def test_adapter_is_byte_equivalent_for_unregistered_query():
    base = FakeBase(["a", "b"])
    adapter = DatedEventTfIdfAdapter(
        {}, base=base, turn_loader=lambda _: (_ for _ in ()).throw(
            AssertionError("turn loader must not run")))
    assert adapter.search("session", "ordinary question", top_k=2) == ["a", "b"]
    assert adapter.last_receipt["reason"] == "unregistered_query"


def test_adapter_deduplicates_reserved_chunk_already_in_base():
    query = "Which trip was in the past one month?"
    target = "[2023/05/15] I completed a camping trip to Yosemite."
    base = FakeBase([target, "other"])
    adapter = DatedEventTfIdfAdapter(
        {query: "2023/06/01"}, reserve_limit=1, base=base,
        turn_loader=lambda _: [_turn(target)])
    assert adapter.search("session", query, top_k=2) == [target, "other"]


def test_zero_limit_preserves_base_and_skips_loader():
    query = "Which trip was in the past one month?"
    base = FakeBase(["a"])
    adapter = DatedEventTfIdfAdapter(
        {query: "2023/06/01"}, reserve_limit=0, base=base,
        turn_loader=lambda _: (_ for _ in ()).throw(
            AssertionError("turn loader must not run")))
    assert adapter.search("session", query) == ["a"]
    assert adapter.last_receipt["reason"] == "reserve_disabled"


def test_context_assembler_keeps_reserve_before_chronological_raw_turns():
    assembler = DatedEventContextAssembler()
    reserved = "[2023/01/15] I attended the target event."
    assembler._chroma = type("Adapter", (), {
        "last_receipt": {"reserve": [reserved]}})()
    chunks = [
        reserved,
        "[2023/01/01] Older evidence.",
        "[2023/01/10] Middle evidence.",
    ]
    assert assembler._order_evidence(chunks, token_budget=100) == [
        reserved,
        "[2023/01/01] Older evidence.",
        "[2023/01/10] Middle evidence.",
    ]


def test_context_assembler_is_identical_without_a_reserve():
    chunks = ["[2023/01/10] Newer.", "[2023/01/01] Older."]
    assembler = DatedEventContextAssembler()
    assembler._chroma = type("Adapter", (), {"last_receipt": None})()
    assert assembler._order_evidence(chunks, 100) == [
        "[2023/01/01] Older.", "[2023/01/10] Newer."]
