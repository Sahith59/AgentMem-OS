from types import SimpleNamespace

import pytest
from agentmem_os.benchmarks.recall_span_adapter import RecallSpanContextAssembler
from agentmem_os.llm.context_assembler import ContextAssembler
from agentmem_os.llm.evidence_packet import digest
from agentmem_os.llm.ranked_chunk_packing import pack_ranked_chunks
from agentmem_os.llm.token_counter import TokenCounter


@pytest.fixture
def counter():
    return TokenCounter()


def test_actual_legacy_deletes_fitting_rank_zero(counter):
    newest = "[2024/02/01] I bought the red bike, only after the refund. RANK_ZERO_NEW_FACT."
    old = "[2020/01/01] LOW_RANK_OLD_FACT " + "older filler. " * 30
    legacy = ContextAssembler()
    control = legacy._render_raw_evidence([newest, old], 60)
    assert "RANK_ZERO_NEW_FACT" not in control
    assert "LOW_RANK_OLD_FACT" in control
    fixed = ContextAssembler(raw_evidence_policy="whole_rank_v1")
    section = fixed._render_raw_evidence([newest, old], 60)
    assert newest in section and old not in section
    assert counter.count(section) <= 60 and len(section) <= 240
    assert fixed.last_raw_evidence_packing["admitted_indices"] == [0]


@pytest.mark.parametrize("chronological", [False, True])
def test_receipts_preserve_all_input_whitespace_duplicates_and_markers(counter, chronological):
    chunks = [
        "[2024/01/02] new\n\n",
        "[2023/01/01] old",
        "[2023/01/01] old",
        "<[SEMANTIC MEMORY]>\nUSER: literal data\n---\n<|endoftext|> 😀",
    ]
    original = list(chunks)
    text, report = pack_ranked_chunks(
        chunks, token_budget=1000, counter=counter, chronological=chronological
    )
    assert chunks == original
    assert report["admitted_indices"] == [0, 1, 2, 3]
    assert report["presentation_indices"] == ([1, 2, 0, 3] if chronological else [0, 1, 2, 3])
    for r in report["receipts"]:
        assert text[r["start"] : r["end"]] == chunks[r["index"]]
        assert r["sha256"] == digest(chunks[r["index"]])
    assert report["original_source_identity"] == "NOT_CERTIFIED"


def test_framing_counts_toward_exact_character_boundary():
    zero_counter = SimpleNamespace(count=lambda _: 0)
    # The limit covers the payload and the complete section framing.
    chunks = ["short"]
    text, _ = pack_ranked_chunks(chunks, token_budget=100, counter=zero_counter)
    exact_budget = (len(text) + 3) // 4
    assert pack_ranked_chunks(chunks, token_budget=exact_budget, counter=zero_counter)[0] == text
    rejected, report = pack_ranked_chunks(
        chunks, token_budget=exact_budget - 1, counter=zero_counter
    )
    assert rejected == "" and report["decisions"][0]["status"] == "character_nonfit"


def test_exact_token_boundary_and_dense_text(counter):
    chunks = ["🧑🏽‍🚀" * 12]
    text, _ = pack_ranked_chunks(chunks, token_budget=1000, counter=counter)
    limit = counter.count(text)
    assert len(text) < (limit - 1) * 4
    assert pack_ranked_chunks(chunks, token_budget=limit, counter=counter)[0] == text
    rejected, report = pack_ranked_chunks(chunks, token_budget=limit - 1, counter=counter)
    assert rejected == "" and report["decisions"][0]["status"] == "token_nonfit"


def test_nonfit_first_does_not_force_partial_admission(counter):
    chunks = ["x" * 1000, "[2024/01/01] It did not happen."]
    text, report = pack_ranked_chunks(chunks, token_budget=80, counter=counter)
    assert report["admitted_indices"] == [1]
    assert chunks[1] in text and "x" * 100 not in text


@pytest.mark.parametrize("chunks,budget", [([], 100), ([" ", "\n"], 100), (["x"], 0)])
def test_empty_and_zero_budget(counter, chunks, budget):
    text, report = pack_ranked_chunks(chunks, token_budget=budget, counter=counter)
    assert text == "" and report["receipts"] == [] and report["used_tokens"] == 0


def test_insufficient_dates_keep_input_order(counter):
    chunks = ["plain first", "[2024/01/01] single dated", "plain last"]
    _, report = pack_ranked_chunks(chunks, token_budget=100, counter=counter)
    assert report["presentation_indices"] == [0, 1, 2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunks": "abc"},
        {"chunks": [None]},
        {"token_budget": True},
        {"token_budget": -1},
        {"chronological": 1},
    ],
)
def test_invalid_policy(counter, kwargs):
    args = dict(chunks=["x"], token_budget=100, counter=counter)
    args.update(kwargs)
    with pytest.raises(ValueError):
        pack_ranked_chunks(**args)


@pytest.mark.parametrize("value", [True, -1, 2.5])
def test_invalid_counter(value):
    with pytest.raises(ValueError):
        pack_ranked_chunks(["x"], token_budget=100, counter=SimpleNamespace(count=lambda _: value))


def test_default_matches_exact_legacy_methods(counter):
    a = ContextAssembler()
    chunks = ["[2024/01/02] newest " * 8, "[2020/01/01] old " * 30]
    expected = a._fit_to_budget(
        "\n---\n".join(a._order_evidence(chunks, 100)), 100, "[SEMANTIC MEMORY]", keep="head"
    )
    assert a._render_raw_evidence(chunks, 100) == expected
    assert a.last_raw_evidence_packing["policy"] == "legacy"


def test_reserve_subclass_byte_identical_fallback():
    chunks = ["[2024/01/02] protected reserve", "[2020/01/01] old " * 30]
    a = RecallSpanContextAssembler()
    b = RecallSpanContextAssembler(raw_evidence_policy="whole_rank_v1")
    for item in (a, b):
        item._chroma = SimpleNamespace(last_receipt={"reserve": chunks[:1]})
    assert a._render_raw_evidence(chunks, 100) == b._render_raw_evidence(chunks, 100)
    assert b.last_raw_evidence_packing == dict(policy="legacy", reason="reserve_fallback")


def test_assemble_uses_exact_section_and_clears_stale_receipt():
    a = ContextAssembler(raw_evidence_policy="whole_rank_v1")
    a.allocations["semantic"] = 100
    a._store = SimpleNamespace(
        get_or_create_session=lambda _: SimpleNamespace(
            inherited_context=None, parent_session_id=None
        ),
        get_history=lambda *args, **kwargs: [],
    )
    chunks = [
        "[2024/01/02] It was cancelled, unless the refund arrives.",
        "[2020/01/01] " + "unrelated. " * 100,
    ]
    a._chroma = SimpleNamespace(search=lambda *args, **kwargs: list(chunks))
    disabled = frozenset({"facts", "profile", "global", "procedural"})
    expected, _ = pack_ranked_chunks(chunks, token_budget=100, counter=a.counter)
    packet = a.assemble("synthetic", "What happened?", disable=disabled)
    assert expected in packet and a.last_raw_evidence_packing["admitted_indices"] == [0]
    a.assemble("synthetic", "What happened?", disable=disabled | {"semantic"})
    assert a.last_raw_evidence_packing == {}


@pytest.mark.parametrize("policy", [None, "unknown", {}])
def test_unknown_constructor_policy(policy):
    with pytest.raises(ValueError):
        ContextAssembler(raw_evidence_policy=policy)
