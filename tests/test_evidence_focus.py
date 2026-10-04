"""Provenance and regression tests, not paid semantic selection results."""
import json
from pathlib import Path

import pytest
from benchmarks.evidence_focus import (FocusInput, SourceTurn, selection_request,
    apply_selection, semantic_review)

FIXTURES = json.loads((Path(__file__).resolve().parents[1] /
                      'benchmarks/fixtures/evidence_semantics_v1.json').read_text())['cases']


def project(fixture):
    turns = tuple(SourceTurn(**{k: t[k] for k in ('id', 'role', 'observed_at', 'text')})
                  for t in fixture['turns'])
    return FocusInput(fixture['question'], '', '\n\n'.join(t.text for t in turns), turns)


@pytest.mark.parametrize('fixture', FIXTURES, ids=[f['id'] for f in FIXTURES])
def test_every_semantic_fixture_preserves_prose_and_receipts(fixture):
    runtime = project(fixture)
    # Oracle-selected IDs here test only application/receipts, NOT selector ability.
    raw = json.dumps({'turn_ids': fixture['required_evidence_ids'], 'sufficiency': 'uncertain'})
    candidate, report = apply_selection(runtime, raw)
    assert candidate.startswith(runtime.packet) and len(candidate) <= 40000
    for receipt in report['receipts']:
        source = next(t for t in runtime.turns if t.id == receipt['id'])
        assert candidate[receipt['start']:receipt['end']] == source.text
    assert semantic_review(fixture, report['selected_ids'])['pass']
    req = selection_request(runtime)
    assert set(json.loads(req['messages'][1]['content'])) == {'question', 'question_date', 'sources'}
    assert 'expected_answer' not in json.dumps(req)


@pytest.mark.parametrize('raw', [
    '{"turn_ids":["unknown"],"sufficiency":"complete"}',
    '{"turn_ids":[],"sufficiency":"complete"}',
    '{"turn_ids":[],"sufficiency":"uncertain","answer":"2"}',
    '{"turn_ids":[],"turn_ids":[],"sufficiency":"uncertain"}',
    '{"turn_ids":"category-1","sufficiency":"complete"}',
    '{"turn_ids":["category-1","category-1"],"sufficiency":"complete"}',
    '{"turn_ids":["category-2","category-1"],"sufficiency":"complete"}',
    '{"turn_ids":[],"sufficiency":"maybe"}',
])
def test_bad_selector_outputs_fail(raw):
    with pytest.raises(ValueError):
        apply_selection(project(FIXTURES[0]), raw)


def test_budget_nonfit_is_explicit_unchanged():
    value = project(FIXTURES[0])
    raw = json.dumps(dict(turn_ids=['category-1'], sufficiency='complete'))
    candidate, report = apply_selection(value, raw, char_cap=len(value.packet))
    assert candidate == value.packet and report['status'] == 'UNCHANGED_BUDGET_NONFIT'
    assert report['receipts'] == []


def test_wrong_semantic_selection_is_not_hidden_by_valid_provenance():
    value = project(FIXTURES[0])
    _, report = apply_selection(value, json.dumps(dict(turn_ids=['category-2'], sufficiency='complete')))
    # Real source, wrong category: a provenance pass must not become quality PASS.
    assert report['status'] == 'APPLIED'
    result = semantic_review(FIXTURES[0], report['selected_ids'])
    assert not result['pass'] and result['missing_required'] and result['selected_excluded']


def test_unknown_or_cross_scope_source_cannot_enter():
    value = FocusInput('Count items', '', 'Owned two pens.',
                       (SourceTurn('x', 'user', '', 'Another tenant owns six pens.'),))
    with pytest.raises(ValueError, match='already delivered'):
        selection_request(value)


def test_conflicting_attribution_rejected():
    value = FocusInput('What did I do?', '', 'Visit the park.',
                       (SourceTurn('x', 'user', '', 'Visit the park.'),
                        SourceTurn('y', 'assistant', '', 'Visit the park.')))
    with pytest.raises(ValueError, match='Ambiguous'):
        selection_request(value)


def test_focus_adapter_independently_rejects_changed_dates(tmp_path):
    from benchmarks.evaluator_v1.prepare_focus import prepare
    from benchmarks.evaluator_v1.verify_focus import verify, hash_file
    from benchmarks.evidence_focus import digest
    text = '[2026/01/01] I bought three pens.'
    package = {'cases':[{'id':str(i),'question':'How many pens?', 'context':text,
                         'context_sha256':digest(text)} for i in range(500)]}
    cache = {'memories':[{'mid':'m','turns':[{'role':'user','content':text}]}],
             'queries':[{'question_id':str(i),'question':'How many pens?',
                         'question_date':'2026/01/02','scope_keys':['m']} for i in range(500)]}
    pp, cp = tmp_path/'p.json', tmp_path/'c.json'
    pp.write_text(json.dumps(package)); cp.write_text(json.dumps(cache))
    out = tmp_path/'focus'; prepare(pp, cp, out)
    assert verify(out)['original_turns_checked'] == 500
    report = json.loads((out/'report.json').read_text())
    inp = Path(report['rows'][0]['input_file']['path'])
    value = json.loads(inp.read_text()); value['turns'][0]['observed_at'] = '2099/01/01'
    inp.write_text(json.dumps(value))
    # Even an updated file digest cannot conceal wrong source attribution.
    report['rows'][0]['input_file']['sha256'] = hash_file(inp)
    (out/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Changed source'):
        verify(out)
