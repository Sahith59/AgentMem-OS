"""Pure planning and source-boundary checks; no model-quality claim."""
import json
from pathlib import Path

import pytest

from benchmarks.evidence_focus import FocusInput, SourceTurn, semantic_review
from benchmarks.luna_evidence_plan import (apply_plan, plan_request,
    opaque_fixture_id, project_fixture)


FIXTURES = json.loads((Path(__file__).resolve().parents[1] /
    'benchmarks/fixtures/evidence_semantics_v1.json').read_text())['cases']


def runtime(case):
    return project_fixture(case)


def opaque(ids):
    return [opaque_fixture_id(tid) for tid in ids]


def response(ids, *, sufficiency='uncertain'):
    return json.dumps(dict(requirements=dict(operation='count', target='requested events',
        time_window='', output_unit='', needed_facts=['eligible event mentions']),
        turn_ids=ids, sufficiency=sufficiency))


@pytest.mark.parametrize('case', FIXTURES, ids=[c['id'] for c in FIXTURES])
def test_oracle_selection_preserves_original_text_and_receipts(case):
    value = runtime(case)
    candidate, report = apply_plan(value, response(opaque(case['required_evidence_ids'])))
    assert candidate.startswith(value.packet)
    assert report['planner'] == report['answerer'] == 'gpt-5.6-luna'
    assert report['semantic_completeness'] == 'NOT_CERTIFIED'
    original = {opaque_fixture_id(t['id']): t['id'] for t in case['turns']}
    assert semantic_review(case, [original[i] for i in report['selected_ids']])['pass']
    for receipt in report['receipts']:
        original = next(t for t in value.turns if t.id == receipt['id'])
        assert candidate[receipt['start']:receipt['end']] == original.text
    # Oracle IDs test transport, not a model's selection quality.


def test_request_contains_only_runtime_projection():
    case = FIXTURES[0]
    request = plan_request(runtime(case))
    assert request['model'] == 'gpt-5.6-luna'
    data = json.loads(request['messages'][1]['content'])
    assert set(data) == {'question', 'question_date', 'sources'}
    assert 'expected_answer' not in json.dumps(request)
    assert 'known_incorrect_answer' not in json.dumps(request)
    assert 'required_evidence_ids' not in json.dumps(request)
    assert 'excluded_ids_and_reasons' not in json.dumps(request)
    assert 'category-1' not in json.dumps(data)  # descriptive source ID stays local


@pytest.mark.parametrize('raw', [
    '{"requirements":{},"turn_ids":[],"sufficiency":"uncertain"}',
    '{"requirements":{},"turn_ids":[],"sufficiency":"uncertain","answer":"2"}',
    '{"requirements":{},"requirements":{},"turn_ids":[],"sufficiency":"uncertain"}',
    response(['unknown']), response(opaque(['category-2', 'category-1'])),
    response(opaque(['category-1', 'category-1'])), response([], sufficiency='complete'),
    response(opaque(['category-1'])).replace('"count"', '"execute_code"'),
])
def test_invalid_or_unknown_plans_fail_closed(raw):
    with pytest.raises(ValueError):
        apply_plan(runtime(FIXTURES[0]), raw)


def test_wrong_semantic_selection_remains_visible():
    case = FIXTURES[0]
    _, report = apply_plan(runtime(case), response(opaque(['category-2']), sufficiency='complete'))
    assert report['status'] == 'APPLIED'
    assert report['planner_sufficiency'] == 'complete'
    assert report['semantic_completeness'] == 'NOT_CERTIFIED'
    original = {opaque_fixture_id(t['id']): t['id'] for t in case['turns']}
    assert not semantic_review(case, [original[i] for i in report['selected_ids']])['pass']


def test_nonfit_preserves_exact_baseline_and_records_no_receipts():
    value = runtime(FIXTURES[0])
    candidate, report = apply_plan(value, response(opaque(['category-1'])), char_cap=len(value.packet))
    assert candidate == value.packet
    assert report['status'] == 'UNCHANGED_BUDGET_NONFIT'
    assert report['receipts'] == []


def test_cross_scope_turn_cannot_enter_plan_request():
    value = FocusInput('How many?', '', 'I have two books.',
        (SourceTurn('alien', 'user', '', 'Another user has seven books.'),))
    with pytest.raises(ValueError, match='already delivered'):
        plan_request(value)
