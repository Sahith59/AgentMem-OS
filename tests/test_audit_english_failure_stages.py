import sqlite3

from benchmarks.audit_english_failure_stages import (
    diagnostic_stage,
    fact_lineage,
    failure_stage,
    operation,
)


def case(context="", abstention=False):
    return {"context": context, "abst": abstention}


def test_failure_stage_separates_missing_partial_present_and_abstention():
    turns = [{"text": "alpha"}, {"text": "beta"}]
    assert failure_stage(case(), turns)[0] == "retrieval_zero_exact_annotated_turns"
    assert failure_stage(case("alpha"), turns) == (
        "retrieval_partial_exact_annotated_turns",
        1,
    )
    assert failure_stage(case("alpha beta"), turns) == (
        "answer_selection_reasoning_or_judge",
        2,
    )
    assert failure_stage(case("alpha", True), turns) == (
        "evidence_sufficiency_or_judge",
        1,
    )


def test_operation_uses_question_shape_without_ids_or_references():
    assert operation("How many conferences did I attend?") == "aggregation_or_amount"
    assert operation("Which museum did I visit first?") == "temporal_or_update"
    assert operation("Can you suggest a guitar?") == "preference_or_advice"
    assert operation("What company did I mention?") == "direct_recall_or_synthesis"
    assert operation("What company did I mention?", True) == "evidence_sufficiency"


def test_fact_lineage_separates_extraction_retrieval_and_delivery():
    connection = sqlite3.connect(":memory:")
    connection.execute("create table turns (id integer, session_id text, content text)")
    connection.execute(
        "create table semantic_facts "
        "(id integer, source_session_id text, fact_text text, source_turn_ids text)"
    )
    connection.executemany(
        "insert into turns values (?, ?, ?)",
        [(1, "s1", "User: evidence one"), (2, "s2", "User: evidence two")],
    )
    connection.executemany(
        "insert into semantic_facts values (?, ?, ?, ?)",
        [(10, "s1", "preserved fact", "[1]"), (20, "s2", "other fact", "[]")],
    )
    cache = {}
    delivered = fact_lineage(
        "context has preserved fact",
        {"session_id": "up1", "source_key": "s1", "turn_index": 0, "text": "evidence one"},
        connection,
        cache,
    )
    omitted = fact_lineage(
        "empty",
        {"session_id": "up1", "source_key": "s1", "turn_index": 0, "text": "evidence one"},
        connection,
        {},
    )
    unextracted = fact_lineage(
        "empty",
        {"session_id": "up2", "source_key": "s2", "turn_index": 0, "text": "evidence two"},
        connection,
        cache,
    )
    assert delivered["status"] == "linked_fact_delivered"
    assert omitted["status"] == "linked_fact_not_delivered"
    assert unextracted["status"] == "no_fact_linked_to_source_turn"
    assert diagnostic_stage("retrieval_zero_exact_annotated_turns", [delivered]) == (
        "fact_lineage_delivered_raw_turn_missing"
    )
    assert diagnostic_stage("retrieval_zero_exact_annotated_turns", [omitted]) == (
        "fact_retrieval_or_ranking_gap"
    )
    assert diagnostic_stage("retrieval_zero_exact_annotated_turns", [unextracted]) == (
        "extraction_lineage_gap"
    )
