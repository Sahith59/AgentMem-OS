"""Check the offline structural bound against the actual focus renderer."""
import hashlib
import json

import pytest

from benchmarks.audit_focus_capacity import audit, capacity
from benchmarks.evidence_focus import FocusInput, SourceTurn, apply_selection


def value():
    turns = (SourceTurn('a', 'user', '2026-01-01', 'A short source.'),
             SourceTurn('b', 'user', '2026-01-02', 'A much longer source. ' * 5),
             SourceTurn('c', 'assistant', '2026-01-03', 'తెలుగు source.'))
    return FocusInput('What happened?', '2026-01-04', '\n'.join(t.text for t in turns), turns)


def test_capacity_matches_renderer_for_every_subset_across_budgets():
    v = value()
    for extra in (0, 1, 80, 160, 240, 400, 4000):
        measured = capacity(v, char_cap=len(v.packet) + extra)
        feasible = []
        for mask in range(1, 8):
            ids = [t.id for i, t in enumerate(v.turns) if mask & (1 << i)]
            _, report = apply_selection(v, json.dumps({'turn_ids': ids, 'sufficiency': 'uncertain'}),
                                        char_cap=len(v.packet) + extra)
            if report['status'] == 'APPLIED':
                feasible.append(len(ids))
        assert measured['maximum_shortest_turns_fit'] == max(feasible, default=0)


def test_no_fit_does_not_silently_clip_or_change_baseline():
    v = value()
    candidate, report = apply_selection(v, '{"turn_ids":["a"],"sufficiency":"uncertain"}',
                                        char_cap=len(v.packet))
    assert candidate == v.packet and report['status'] == 'UNCHANGED_BUDGET_NONFIT'
    assert capacity(v, char_cap=len(v.packet))['maximum_shortest_turns_fit'] == 0


def test_invalid_baseline_and_boolean_budget_rejected():
    with pytest.raises(ValueError):
        capacity(value(), char_cap=1)
    with pytest.raises(ValueError):
        capacity(value(), char_cap=True)


def artifacts(tmp_path):
    from dataclasses import asdict
    v = value()
    inp = tmp_path / 'input.json'
    inp.write_text(json.dumps(asdict(v)))
    sha = hashlib.sha256(v.packet.encode()).hexdigest()
    original = tmp_path / 'original.json'
    original.write_text(json.dumps({'cases': [{'id': 'case', 'question': v.question,
        'date': v.question_date, 'context': v.packet}]}))
    pool = tmp_path / 'pool.json'
    pool.write_text(json.dumps({'cases': 1, 'sources': {'package': {'path': str(original),
        'sha256': hashlib.sha256(original.read_bytes()).hexdigest()}}, 'rows': [{'id': 'case', 'original_context_sha256': sha,
        'input_file': {'path': str(inp), 'sha256': hashlib.sha256(inp.read_bytes()).hexdigest()}}]}))
    bridge = tmp_path / 'bridge.json'
    bridge.write_text(json.dumps({'total': 1, 'status': 'PASS_INTEGRITY', 'rows': [{'id': 'case', 'question': v.question, 'context_sha256': sha, 'new_grade': False}]}))
    return inp, pool, bridge


def test_tampered_input_rejected(tmp_path):
    inp, pool, bridge = artifacts(tmp_path)
    inp.write_text(inp.read_text() + ' ')
    with pytest.raises(ValueError, match='Input file hash'):
        audit(pool, bridge, expected_cases=1)


def test_population_duplicate_rejected(tmp_path):
    _, pool, bridge = artifacts(tmp_path)
    b = json.loads(bridge.read_text())
    b['rows'] *= 2
    bridge.write_text(json.dumps(b))
    with pytest.raises(ValueError, match='population'):
        audit(pool, bridge, expected_cases=1)


def test_audit_counts_all_cases_and_no_accuracy_claim(tmp_path):
    _, pool, bridge = artifacts(tmp_path)
    result = audit(pool, bridge, expected_cases=1)
    assert result['all_cases']['cases'] == result['judge_misses']['cases'] == 1
    assert result['judge_correct']['cases'] == 0
    assert result['accuracy_lift'] == 'NOT_MEASURED' and result['model_calls'] == 0


def test_jointly_truncated_population_rejected(tmp_path):
    _, pool, bridge = artifacts(tmp_path)
    # Both files agree on one case, but the frozen CLI requires all 500.
    with pytest.raises(ValueError, match='population'):
        audit(pool, bridge)


@pytest.mark.parametrize('field', ['question', 'question_date'])
def test_rehashed_runtime_projection_cannot_change_request(tmp_path, field):
    inp, pool, bridge = artifacts(tmp_path)
    value = json.loads(inp.read_text())
    value[field] = 'different runtime value'
    inp.write_text(json.dumps(value))
    p = json.loads(pool.read_text())
    p['rows'][0]['input_file']['sha256'] = hashlib.sha256(inp.read_bytes()).hexdigest()
    pool.write_text(json.dumps(p))
    with pytest.raises(ValueError, match='Runtime projection'):
        audit(pool, bridge, expected_cases=1)
