"""Capacity bounds are annotation-conditioned diagnostics, not runtime output."""

from datetime import datetime, timedelta
from itertools import combinations

import pytest
from agentmem_os.benchmarks.source_capacity_audit import measure_capacity
from agentmem_os.llm.evidence_packet import SourceSnapshot, SourceTurn

NOW = datetime(2026, 1, 2)


def t(id, pos, body="fact", session="s", when=NOW):
    return SourceTurn(id, session, pos, "user" if pos % 2 == 0 else "assistant",
                      when, body)


def run(turns, targets, **kwargs):
    return measure_capacity(SourceSnapshot("q", tuple(turns)), scope="q", as_of=NOW,
                            targets=targets, present_ids=kwargs.pop("present_ids", []),
                            **kwargs)


def rendered_cost(turns):
    if not turns:
        return 0
    return len("\n\n[ORIGINAL SOURCE EVIDENCE: relevance and completeness are unverified]\n") + sum(
        len(f"[{x.id} | {x.role} | observed {x.observed_at.isoformat()}]\n{x.text}\n")
        for x in turns
    )


def test_adjacent_anchor_can_cover_target_more_cheaply_than_target_anchor():
    turns = [t("a", 0), t("b", 1), t("c", 2, "X" * 3000)]
    result = run(turns, ["b"])
    assert result["all_cover"]["anchor_ids"] == ["a"]
    assert result["all_cover"]["source_ids"] == ["a", "b"]
    assert result["all_cover"]["chars"] == rendered_cost(turns[:2])
    restricted = run(turns, ["b"], anchor_ids=["b"])
    assert restricted["all_cover"]["chars"] == rendered_cost(turns)


def test_shared_qualifier_costs_once_and_complete_baseline_source_is_reused():
    turns = [t("a", 0), t("b", 1, "Only if the return is refunded."), t("c", 2)]
    result = run(turns, ["a", "c"], anchor_ids=["a", "c"], present_ids=["b"])
    assert result["all_cover"]["source_ids"] == ["a", "c"]
    assert result["all_cover"]["chars"] == rendered_cost([turns[0], turns[2]])


@pytest.mark.parametrize("future", [False, True])
def test_gap_or_future_neighbor_blocks_whole_bundle(future):
    turns = [t("a", 0), t("b", 1 if future else 2,
                              when=NOW + timedelta(days=1) if future else NOW)]
    result = run(turns, ["a"], anchor_ids=["a"])
    assert result["all_cover"] is None and result["any_cover"] is None
    assert result["unreachable_targets"] == ["a"]


def test_future_target_does_not_become_available_through_past_anchor():
    result = run([t("a", 0), t("b", 1, when=NOW + timedelta(days=1))], ["b"])
    assert result["eligible_targets"] == [] and result["all_cover"] is None


def test_no_targets_is_distinct_from_unreachable_targets():
    result = run([t("a", 0)], [])
    assert result["all_cover"]["chars"] == 0 and result["any_cover"] is None
    assert result["targets"] == []


@pytest.mark.parametrize("kwargs", [dict(targets=["unknown"]),
                                   dict(targets=["a"], present_ids=["a"]),
                                   dict(targets=["a"], anchor_ids=["unknown"]),
                                   dict(targets=["a"], neighbor_turns=-1)])
def test_invalid_evaluator_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        run([t("a", 0)], **kwargs)


def test_reject_ineligible_presence_and_unbounded_target_set():
    with pytest.raises(ValueError):
        run([t("a", 0, when=NOW + timedelta(days=1))], [], present_ids=["a"])
    turns = [t(str(i), i) for i in range(13)]
    with pytest.raises(ValueError):
        run(turns, [x.id for x in turns])


def test_exact_search_agrees_with_exhaustive_all_subsets_with_unicode_and_ties():
    turns = [t(str(i), i, "条件だけ。" * (1 + (i * 3) % 7)) for i in range(7)]
    for targets in [("0", "3", "6"), ("1", "2"), ("0", "6")]:
        result = run(turns, targets)
        choices = []
        for n in range(8):
            for seeds in combinations(range(7), n):
                ids = {i for seed in seeds for i in range(max(0, seed - 1), min(7, seed + 2))}
                if set(targets) <= {str(i) for i in ids}:
                    choices.append(rendered_cost([turns[i] for i in ids]))
        assert result["all_cover"]["chars"] == min(choices)
        assert result["status"] == "EVALUATOR_ONLY_NOT_AN_ACCURACY_CEILING"
