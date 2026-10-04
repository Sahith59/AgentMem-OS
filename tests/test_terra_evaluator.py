"""Offline tests only. Fake receipts exercise integrity, never judge quality."""
import copy
import json
from pathlib import Path

import pytest
from benchmarks.evaluator_v1 import contract as c
from benchmarks.evaluator_v1.build import make_package
from benchmarks.evaluator_v1.calibration import dataset
from benchmarks.evaluator_v1.runner import run
from benchmarks.evaluator_v1.verify import verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError('No network in evaluator tests')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


@pytest.fixture
def package(tmp_path):
    data = dataset(ROOT / 'benchmarks/fixtures/evidence_semantics_v1.json')
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(data))
    cases = [v for v in data['cases'] if v['split'] == 'development'][:3]
    data['cases'] = cases
    path.write_text(json.dumps(data))
    return make_package('development', cases, {'calibration': {'path': str(path), 'sha256': c.file_sha(path)}})


def approval(package):
    p = c.preflight(package)
    return dict(package_sha256=p['package_sha256'], budget_nusd=p['maximum_reservation_nusd'],
                maximum_attempts=p['calls_no_retries'], approved=True, mode='offline-test',
                authorization_text='Synthetic test only; no spend')


class Fake:
    def __init__(self, texts=None):
        self.calls = 0
        self.texts = texts
    def __call__(self, req):
        self.calls += 1
        return dict(text=self.texts[self.calls - 1] if self.texts else 'yes', finish_reason='stop',
                    model=req['model'], id=f'mock-{self.calls}', request_id=f'mock-req-{self.calls}',
                    usage=dict(prompt_tokens=20, completion_tokens=10, total_tokens=30,
                               completion_tokens_details={'reasoning_tokens': 9}))


def test_dataset_counts_strata_and_family_split():
    data = dataset(ROOT / 'benchmarks/fixtures/evidence_semantics_v1.json')
    val = [v for v in data['cases'] if v['split'] == 'validation']
    dev = [v for v in data['cases'] if v['split'] == 'development']
    assert len(val) == 120 and sum(v['expected'] for v in val) == 60
    assert len(dev) == 32 and len({v['family'] for v in val}) == 60
    assert {v['type'] for v in val} == c.TYPES
    assert sum(v['abstention'] for v in val) == 10
    assert not ({v['family'] for v in val} & {v['family'] for v in dev})
    assert sum(v['critical'] for v in val) == 2


def test_request_does_not_expose_evaluator_metadata(package):
    case = copy.deepcopy(package['cases'][0])
    before = c.request(case)
    for field in ('id', 'expected', 'source', 'rationale', 'critical', 'family', 'split'):
        case[field] = 'LEAK_SENTINEL'
    assert c.request(case) == before
    assert 'LEAK_SENTINEL' not in json.dumps(c.request(case))
    assert before['reasoning_effort'] == 'low'


@pytest.mark.parametrize('value', ['yes because', 'no, but yes', '', 'yesterday', 'yes.', None])
def test_strict_parser(value):
    with pytest.raises(ValueError):
        c.verdict(value)


def test_resume_completed_run_never_dispatches_twice(package, tmp_path):
    fake = Fake()
    report = run(package, tmp_path / 'run', approval(package), mode='offline-test', provider=fake)
    assert fake.calls == 3 and report['complete']
    assert not report['paid_accuracy_claim_allowed']
    assert report['gate'] == 'MOCK_ONLY_NO_MODEL_QUALITY_EVIDENCE'
    run(package, tmp_path / 'run', approval(package), mode='offline-test', provider=fake)
    assert fake.calls == 3


def test_failure_reserves_and_blocks_retry(package, tmp_path):
    fake = Fake(['yes', 'invalid'])
    out = tmp_path / 'run'
    with pytest.raises(RuntimeError):
        run(package, out, approval(package), mode='offline-test', provider=fake)
    state = json.loads((out / 'checkpoint.json').read_text())
    assert fake.calls == 2 and len(state['jobs']) == 2
    assert state['reserved_nusd'] == sum(j['reservation_nusd'] for j in state['jobs'].values())
    with pytest.raises(ValueError, match='Unresolved'):
        run(package, out, approval(package), mode='offline-test', provider=fake)
    assert fake.calls == 2


@pytest.mark.parametrize('mutation', ['budget', 'hash', 'attempts', 'mode', 'approved'])
def test_bad_approval_stops_before_directory(package, tmp_path, mutation):
    a = approval(package)
    a[{'budget': 'budget_nusd', 'hash': 'package_sha256', 'attempts': 'maximum_attempts',
       'mode': 'mode', 'approved': 'approved'}[mutation]] = 0
    out = tmp_path / 'absent'
    with pytest.raises(ValueError):
        run(package, out, a, mode='offline-test', provider=Fake())
    assert not out.exists()


@pytest.mark.parametrize('field,value', [('finish_reason','length'), ('model','gpt-4o'),
    ('model','gpt-5.6-terra-unreviewed'), ('request_id',None), ('text','no maybe')])
def test_receipt_rejections(field, value):
    req = dict(c.SETTINGS, messages=[{'content': 'short'}])
    response = Fake()(req)
    response[field] = value
    with pytest.raises(ValueError):
        c.response_cost(req, response)


@pytest.mark.parametrize('field,value', [('prompt_tokens',-1), ('completion_tokens',True),
                                       ('total_tokens',31), ('completion_tokens',2049)])
def test_usage_rejections(field, value):
    req = dict(c.SETTINGS, messages=[{'content': 'short'}])
    response = Fake()(req)
    response['usage'][field] = value
    with pytest.raises(ValueError):
        c.response_cost(req, response)


def test_forged_grade_and_extra_jobs_fail_replay(package, tmp_path):
    out = tmp_path / 'run'
    run(package, out, approval(package), mode='offline-test', provider=Fake())
    state = json.loads((out / 'checkpoint.json').read_text())
    first = next(iter(state['jobs'].values()))
    first['correct'] = False
    with pytest.raises(ValueError, match='grade mismatch'):
        verify(package, state)
    first['correct'] = True
    state['jobs']['unknown'] = first
    with pytest.raises(ValueError, match='Unknown job'):
        verify(package, state)


def test_changed_code_input_and_labels_rejected(package):
    bad = copy.deepcopy(package)
    bad['code_sources']['benchmarks/lme_judge.py'] = 'bad'
    with pytest.raises(ValueError, match='code'):
        c.validate(bad)
    bad = copy.deepcopy(package)
    bad['cases'][0]['expected'] = not bad['cases'][0]['expected']
    with pytest.raises(ValueError, match='pinned'):
        c.validate(bad)
    Path(package['inputs']['calibration']['path']).write_text('{}')
    with pytest.raises(ValueError, match='input artifact'):
        c.validate(package)


def test_validation_gate_is_not_mock_accuracy(tmp_path):
    data = dataset(ROOT / 'benchmarks/fixtures/evidence_semantics_v1.json')
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps(data))
    cases = [v for v in data['cases'] if v['split'] == 'validation']
    pkg = make_package('validation', cases, {'calibration': {'path':str(path),'sha256':c.file_sha(path)}})
    report = run(pkg, tmp_path / 'run', approval(pkg), mode='offline-test',
                 provider=Fake(['yes' if v['expected'] else 'no' for v in cases]))
    assert report['correct'] == 120 and report['false_accepts'] == 0
    assert report['gate'] == 'MOCK_ONLY_NO_MODEL_QUALITY_EVIDENCE'


def test_paid_mode_rejects_injected_provider(package, tmp_path):
    a = approval(package)
    a['mode'] = 'paid'
    with pytest.raises(ValueError, match='Injected provider'):
        run(package, tmp_path / 'run', a, provider=Fake())
    assert not (tmp_path / 'run').exists()


def test_validation_cannot_start_without_paid_development_receipt(tmp_path):
    data = dataset(ROOT / 'benchmarks/fixtures/evidence_semantics_v1.json')
    path = tmp_path / 'cal.json'; path.write_text(json.dumps(data))
    cases = [v for v in data['cases'] if v['split'] == 'validation']
    pkg = make_package('validation', cases, {'calibration': {'path':str(path),'sha256':c.file_sha(path)}})
    a = approval(pkg); a['mode'] = 'paid'
    with pytest.raises(ValueError, match='preceding-stage'):
        run(pkg, tmp_path / 'run', a)
    assert not (tmp_path / 'run').exists()


@pytest.mark.parametrize('errors,expected_gate', [(3,True),(4,False)])
def test_false_accept_gate_boundary(tmp_path, errors, expected_gate):
    data = dataset(ROOT / 'benchmarks/fixtures/evidence_semantics_v1.json')
    path = tmp_path / 'cal.json'; path.write_text(json.dumps(data))
    cases = [v for v in data['cases'] if v['split'] == 'validation']
    pkg = make_package('validation', cases, {'calibration': {'path':str(path),'sha256':c.file_sha(path)}})
    texts = ['yes' if case['expected'] else 'no' for case in cases]
    negatives = [i for i, case in enumerate(cases) if not case['expected'] and not case['critical']]
    for i in negatives[:errors]: texts[i] = 'yes'
    report = run(pkg, tmp_path/'run', approval(pkg), mode='offline-test', provider=Fake(texts))
    assert report['false_accepts'] == errors
    assert report['thresholds_met'] is expected_gate
    assert report['gate'] == 'MOCK_ONLY_NO_MODEL_QUALITY_EVIDENCE'


def test_critical_error_fails_even_with_119_correct(tmp_path):
    data = dataset(ROOT / 'benchmarks/fixtures/evidence_semantics_v1.json')
    path = tmp_path / 'cal.json'; path.write_text(json.dumps(data))
    cases = [v for v in data['cases'] if v['split'] == 'validation']
    pkg = make_package('validation', cases, {'calibration': {'path':str(path),'sha256':c.file_sha(path)}})
    texts = ['yes' if case['expected'] else 'no' for case in cases]
    texts[next(i for i,case in enumerate(cases) if case['critical'])] = 'yes'
    report = run(pkg, tmp_path/'run', approval(pkg), mode='offline-test', provider=Fake(texts))
    assert report['correct'] == 119 and report['critical_errors'] == 1
    assert report['thresholds_met'] is False


def test_paid_approval_cannot_be_reused_in_a_new_directory(package, tmp_path):
    a = approval(package); a['mode'] = 'paid'
    a['output_directory'] = str(tmp_path / 'approved-only')
    with pytest.raises(ValueError, match='one output directory'):
        run(package, tmp_path / 'different', a)
    assert not (tmp_path / 'different').exists()
