"""Network-free screen integrity, failure, resumption and decision tests."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks.english_screen import build, packet, runner, verification as v


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket
    def denied(*args, **kwargs):
        raise AssertionError('Network forbidden in offline tests')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    root = tmp_path_factory.mktemp('screen-fixture')
    contexts = root / 'contexts'; contexts.mkdir()
    old = {'settings': {'generate': {'model': 'gpt-5.6-luna', 'max_completion_tokens': 4200},
                       'judge': {'model': 'gpt-4o', 'max_tokens': 5, 'temperature': 0}},
           'rates_nusd_per_token': {'generate': {'input': 200, 'output': 1200},
                                    'judge': {'input': 2500, 'output': 10000}},
           'answer_max_bytes': 4096, 'prompt': '{context}\n{question}{today_line}', 'cases': []}
    selected = {'cases': []}
    for i in range(500):
        c = {'id': f'q{i:03}', 'question': f'Question {i}?', 'context': f'Source {i}',
             'gold': str(i), 'date': '2026-09-12', 'type': 'test',
             'abst': i < 8 or 72 <= i < 74 or 92 <= i < 112,
             'judge_template': f'Q{i} gold {i}; response: {{response}}'}
        for field in ('context', 'question', 'gold', 'judge_template'):
            c[field + '_sha256'] = v.digest(c[field])
        old['cases'].append(c)
        if i < 150:
            candidate = c['context'] + '\nADDED evidence'
            (contexts / (c['id'] + '.txt')).write_text(candidate)
            selected['cases'].append({'question_id': c['id'],
                'cohort': 'stable_miss' if i < 72 else 'disagreement' if i < 92 else 'stable_pass_control',
                'baseline_sha256': v.digest(c['context']), 'candidate_sha256': v.digest(candidate),
                'question_sha256': c['question_sha256'], 'type': c['type'], 'abstention': c['abst']})
    (root / 'baseline.json').write_text(json.dumps(old))
    (root / 'selection.json').write_text(json.dumps(selected))
    result, _ = build.build(root / 'baseline.json', root / 'selection.json', contexts, {})
    return result


class Fake:
    def __init__(self, mutation=None):
        self.calls = 0
        self.mutation = mutation

    def __call__(self, req):
        self.calls += 1
        stage = 'generate' if req['model'] == 'gpt-5.6-luna' else 'judge'
        content = req['messages'][0]['content']
        # Distinct answers on half the rows, identical answers on the remainder.
        if stage == 'generate':
            number = int(content.splitlines()[0].split()[1])
            text = 'ANSWER: ' + ('candidate' if number % 2 and 'ADDED' in content else 'baseline')
        else:
            text = 'yes'
        result = {'text': text, 'finish_reason': 'stop',
                  'model': 'gpt-5.6-luna' if stage == 'generate' else 'gpt-4o-2024-08-06',
                  'id': f'resp-{self.calls}', 'request_id': f'req-{self.calls}',
                  'usage': {'prompt_tokens': 100, 'completion_tokens': 1}}
        if self.mutation:
            self.mutation(result)
        return result


@pytest.fixture(scope='module')
def completed(package, tmp_path_factory):
    directory = tmp_path_factory.mktemp('screen-complete')
    provider = Fake()
    report = packet.run(package, directory, provider, package['proposed_budget_nusd'], 'synthetic-only', 'offline-test')
    assert report['complete'] and provider.calls == 525
    return json.loads((directory / 'checkpoint.json').read_text())


def write_state(directory, state):
    (directory / 'checkpoint.json').write_text(json.dumps(state))


def test_contract_and_independent_accounting(package, completed):
    assert packet.validate(package) == v.verify_contract(package)
    result = v.verify_state(package, completed)
    assert result['provider_calls'] == 525 and result['shared_judgments'] == 75
    assert result['quality']['screen'] == 'FAIL'  # all-correct synthetic arms have no net gain
    assert len(completed['jobs']) == 600
    assert result['usage_cost_without_cache_discount_nusd'] == sum(j.get('usage_cost_nusd', 0) for j in completed['jobs'].values())


def test_completed_resume_makes_zero_calls(package, completed, tmp_path):
    write_state(tmp_path, completed)
    provider = Fake()
    packet.run(package, tmp_path, provider, package['proposed_budget_nusd'], 'synthetic-only', 'offline-test')
    assert provider.calls == 0


def test_valid_prefix_resumes_without_duplicate_dispatch(package, completed, tmp_path):
    state = copy.deepcopy(completed)
    state['jobs'] = dict(list(state['jobs'].items())[:7])
    state['reserved_nusd'] = sum(j['reservation_nusd'] for j in state['jobs'].values())
    n = sum('response' in j for j in state['jobs'].values())
    assert v.verify_state(package, state, require_complete=False)['quality']['screen'] == 'INCOMPLETE'
    write_state(tmp_path, state)
    provider = Fake(); provider.calls = n
    packet.run(package, tmp_path, provider, package['proposed_budget_nusd'], 'synthetic-only', 'offline-test')
    assert provider.calls == 525
    resumed = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert v.verify_state(package, resumed)['complete']
    assert list(resumed['jobs'].items())[:7] == list(state['jobs'].items())


@pytest.mark.parametrize('fault', ['answer', 'grade', 'request', 'usage', 'reservation', 'hole', 'sharing', 'pending', 'model', 'receipt', 'ledger'])
def test_corrupt_resume_rejected_before_provider(package, completed, tmp_path, fault):
    state = copy.deepcopy(completed)
    jobs = state['jobs']; gen = next(j for k, j in jobs.items() if k.endswith('/generate'))
    grade = next(j for k, j in jobs.items() if k.endswith('/judge') and 'response' in j)
    if fault == 'answer': gen['answer'] = 'tampered'
    elif fault == 'grade': grade['correct'] = not grade['correct']
    elif fault == 'request': gen['request_sha256'] = '0' * 64
    elif fault == 'usage': gen['usage_cost_nusd'] += 1
    elif fault == 'reservation':
        gen['reservation_nusd'] += 1; state['reserved_nusd'] += 1
    elif fault == 'hole':
        del jobs[next(iter(jobs))]; state['reserved_nusd'] = sum(j['reservation_nusd'] for j in jobs.values())
    elif fault == 'sharing': next(j for j in jobs.values() if j.get('shared_from'))['shared_from'] = 'q149/baseline/judge'
    elif fault == 'pending': gen['status'] = 'pending'
    elif fault == 'model': gen['response']['model'] = 'gpt-5.6-luna-other'
    elif fault == 'receipt': grade['response']['id'] = gen['response']['id']
    elif fault == 'ledger': state['reserved_nusd'] += 1
    write_state(tmp_path, state)
    provider = Fake()
    with pytest.raises(ValueError):
        packet.run(package, tmp_path, provider, package['proposed_budget_nusd'], 'synthetic-only', 'offline-test')
    assert provider.calls == 0


@pytest.mark.parametrize('fault', ['arm', 'gold', 'source', 'runtime', 'gates', 'model', 'prompt', 'order'])
def test_contract_tampering_fails(package, fault):
    p = copy.deepcopy(package)
    if fault == 'arm':
        c = p['cases'][0]; c['arm_contexts']['candidate'] += 'tamper'
        c['arm_context_sha256']['candidate'] = v.digest(c['arm_contexts']['candidate'])
    elif fault == 'gold':
        p['cases'][0]['gold'] += 'changed'; p['cases'][0]['gold_sha256'] = v.digest(p['cases'][0]['gold'])
    elif fault == 'source': p['source_artifacts']['baseline_package']['sha256'] = '0' * 64
    elif fault == 'runtime': p['runtime_sha256']['runner.py'] = '0' * 64
    elif fault == 'gates': p['gates']['net_gain_min'] = 0
    elif fault == 'model': p['settings']['judge']['temperature'] = 1
    elif fault == 'prompt': p['prompts']['candidate'] += 'Guess'
    elif fault == 'order': p['cases'].reverse()
    with pytest.raises(ValueError): packet.validate(p)


@pytest.mark.parametrize('fault', ['timeout', 'truncated', 'model', 'missing_id', 'usage_bool', 'cache_bool', 'oversize'])
def test_provider_failure_persists_reservation_and_never_retries(package, tmp_path, fault):
    def mutate(response):
        if fault == 'timeout': raise TimeoutError('PRIVATE SHOULD NOT BE SAVED')
        if fault == 'truncated': response['finish_reason'] = 'length'
        if fault == 'model': response['model'] = 'gpt-5.6-luna-new'
        if fault == 'missing_id': response['request_id'] = None
        if fault == 'usage_bool': response['usage']['prompt_tokens'] = True
        if fault == 'cache_bool': response['usage']['prompt_tokens_details'] = {'cached_tokens': False}
        if fault == 'oversize': response['text'] = 'x' * 4097
    provider = Fake(mutate)
    with pytest.raises(RuntimeError): packet.run(package, tmp_path, provider, package['proposed_budget_nusd'], 'test', 'offline-test')
    saved = (tmp_path / 'checkpoint.json').read_text(); state = json.loads(saved)
    assert state['reserved_nusd'] > 0 and len(state['jobs']) == 1 and 'PRIVATE' not in saved
    with pytest.raises(ValueError): packet.run(package, tmp_path, provider, package['proposed_budget_nusd'], 'test', 'offline-test')
    assert provider.calls == 1


def test_budget_stop_before_dispatch(package, tmp_path):
    provider = Fake()
    with pytest.raises(ValueError, match='Budget stop'): packet.run(package, tmp_path, provider, 1, 'test', 'offline-test')
    assert provider.calls == 0
    state = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert state['jobs'] == {} and state['reserved_nusd'] == 0


def test_cache_write_cost_is_bounded(package):
    req = packet.request(package, package['cases'][0], 'baseline', 'generate')
    response = Fake()(req)
    response['usage']['prompt_tokens_details'] = {'cache_write_tokens': 100, 'cached_tokens': 0}
    assert v.response_cost(package, 'generate', req, response) == 26200
    assert v.response_cost(package, 'generate', req, response) <= v.reservation(package, 'generate', req)


def test_gate_rejects_both_arm_control_collapse(package):
    rows = [{'id': c['id'], 'cohort': c['cohort'], 'abstention': c['abstention'],
             'baseline': c['cohort'] != 'stable_miss', 'candidate': True} for c in package['cases']]
    assert v.quality_gates(package, rows, True)['screen'] == 'PASS'
    controls = [r for r in rows if r['cohort'] == 'stable_pass_control']
    for row in controls[:3]: row.update(baseline=False, candidate=False)
    result = v.quality_gates(package, rows, True)
    assert result['stable_control_losses'] == 0 and result['screen'] == 'FAIL'
    assert not result['checks']['stable_control_absolute_floor']


def test_gate_exact_thresholds_and_abstention(package):
    rows = [{'id': c['id'], 'cohort': c['cohort'], 'abstention': c['abstention'],
             'baseline': True, 'candidate': True} for c in package['cases']]
    for row in rows[8:18]: row['baseline'] = False
    assert v.quality_gates(package, rows, True)['screen'] == 'PASS'
    rows[18]['baseline'] = False; rows[0]['candidate'] = False
    result = v.quality_gates(package, rows, True)
    assert result['net_gain'] == 10 and result['screen'] == 'FAIL'
    assert not result['checks']['abstention_preserved']
    assert v.quality_gates(package, rows[:-1], False)['screen'] == 'INCOMPLETE'


def test_approval_binding(package, completed, tmp_path):
    state = copy.deepcopy(completed); state['binding']['mode'] = 'paid'
    approval = {'status': 'FOUNDER_APPROVED', 'package_sha256': v.digest(v.canonical(package)),
                'budget_nusd': package['proposed_budget_nusd'], 'run_directory': str(tmp_path),
                'founder_message': 'Synthetic unit test; never use for paid execution'}
    state['binding']['authorization'] = v.digest(v.canonical(approval))
    assert v.verify_paid(package, state, approval, tmp_path)['complete']
    for key in ('status', 'package_sha256', 'budget_nusd', 'run_directory', 'founder_message'):
        bad = dict(approval); bad[key] = None
        with pytest.raises((ValueError, TypeError)): v.verify_paid(package, state, bad, tmp_path)


def test_cli_default_and_missing_approval_never_construct_provider(package, tmp_path, monkeypatch, capsys):
    path = tmp_path / 'package.json'; path.write_text(json.dumps(package))
    def denied(): raise AssertionError('Provider must not be constructed')
    monkeypatch.setattr(runner, 'OpenAIProvider', denied)
    monkeypatch.setattr(sys, 'argv', ['packet', str(path)])
    packet.main(); assert 'maximum_requests_no_retries' in capsys.readouterr().out
    monkeypatch.setattr(sys, 'argv', ['packet', str(path), '--execute-paid'])
    with pytest.raises(SystemExit): packet.main()


def test_provider_has_no_sdk_retries(monkeypatch):
    captured = {}
    def client(**kwargs): captured.update(kwargs); return SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=client))
    monkeypatch.setenv('OPENAI_API_KEY', 'offline-fake-value')
    runner.OpenAIProvider()
    assert captured['max_retries'] == 0 and captured['timeout'] == 90
    assert captured['base_url'] == 'https://api.openai.com/v1'


def test_ambiguous_judge_halts_without_regrade(package, tmp_path):
    provider = Fake()
    def ambiguous(req):
        response = provider(req)
        if req['model'] == 'gpt-4o': response['text'] = 'yes, but it is incomplete'
        return response
    with pytest.raises(RuntimeError): packet.run(package, tmp_path, ambiguous, package['proposed_budget_nusd'], 'test', 'offline-test')
    assert provider.calls == 2
    with pytest.raises(ValueError): packet.run(package, tmp_path, ambiguous, package['proposed_budget_nusd'], 'test', 'offline-test')
    assert provider.calls == 2


def test_unapproved_cli_blocks_provider_even_with_all_flags(package, tmp_path, monkeypatch):
    path = tmp_path / 'package.json'; path.write_text(json.dumps(package))
    approval = tmp_path / 'approval.json'; approval.write_text(json.dumps({'status': 'PROPOSED'}))
    def denied(): raise AssertionError('Provider must not be constructed')
    monkeypatch.setattr(runner, 'OpenAIProvider', denied)
    monkeypatch.setattr(sys, 'argv', ['packet', str(path), '--execute-paid', '--output', str(tmp_path / 'paid'),
                                     '--budget-usd', str(package['proposed_budget_nusd'] / 1e9), '--approval-record', str(approval)])
    with pytest.raises(SystemExit): packet.main()
    assert not (tmp_path / 'paid').exists()


def test_unchanged_answer_and_judge_requests(package):
    old = json.loads(Path(package['source_artifacts']['baseline_package']['path']).read_text())
    for case in package['cases']:
        expected = old['prompt'].format(context=case['context'], question=case['question'], today_line=runner.date_line(case))
        assert packet.request(package, case, 'baseline', 'generate')['messages'][0]['content'] == expected
        for arm in ('baseline', 'candidate'):
            assert packet.request(package, case, arm, 'judge', 'test {braces}') == v.request(package, case, arm, 'judge', 'test {braces}')


def test_active_run_lock_prevents_dispatch(package, tmp_path):
    import fcntl
    provider = Fake()
    with (tmp_path / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            packet.run(package, tmp_path, provider, package['proposed_budget_nusd'], 'test', 'offline-test')
    assert provider.calls == 0 and not (tmp_path / 'checkpoint.json').exists()


@pytest.mark.parametrize('change', ['budget', 'authorization', 'mode'])
def test_resume_binding_mismatch_blocks_calls(package, completed, tmp_path, change):
    write_state(tmp_path, completed)
    provider = Fake()
    budget = package['proposed_budget_nusd'] + (1 if change == 'budget' else 0)
    auth = 'changed' if change == 'authorization' else 'synthetic-only'
    mode = 'paid' if change == 'mode' else 'offline-test'
    with pytest.raises(ValueError, match='Resume manifest'):
        packet.run(package, tmp_path, provider, budget, auth, mode)
    assert provider.calls == 0
