#!/usr/bin/env python3
"""V2 paired English experiments with shared identical-input judgments. Default is offline validation; no project imports.

Money is integer nano-USD. Reservations are never released automatically, including
on errors. This bounds this runner's dispatch at the frozen rates, not account spend.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def atomic(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate(package):
    if package.get('judge_policy') != 'shared-exact-request-v2':
        raise ValueError('V2 requires explicit prospective judge policy; cannot replay old protocols')
    if package['schema'] != 1 or not package['cases']:
        raise ValueError('Unsupported or empty package')
    if not isinstance(package.get('answer_max_bytes', 8192), int) or not 1 <= package.get('answer_max_bytes', 8192) <= 65536:
        raise ValueError('Invalid answer byte limit')
    ids = [c['id'] for c in package['cases']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate question IDs')
    if set(package['prompts']) != {'baseline', 'candidate'}:
        raise ValueError('Expected exactly two paired arms')
    for arm, prompt in package['prompts'].items():
        if digest(prompt) != package['prompt_sha256'][arm]:
            raise ValueError('Changed prompt')
    for c in package['cases']:
        for field in ('context', 'question', 'gold'):
            if digest(c[field]) != c[field + '_sha256']:
                raise ValueError('Changed case content')
        # Only these three fields enter generation. Labels cannot select a prompt.
        for prompt in package['prompts'].values():
            prompt.format(context=c['context'], question=c['question'],
                          today_line=date_line(c))
        if c['judge_template'].count('{response}') != 1:
            raise ValueError('Invalid frozen judge template')
    expected = {'generate': ('gpt-5.6-luna', {'input': 200, 'output': 1200}),
                'judge': ('gpt-4o', {'input': 2500, 'output': 10000})}
    for stage in ('generate', 'judge'):
        if (package['settings'][stage]['model'], package['rates_nusd_per_token'][stage]) != expected[stage]:
            raise ValueError('Unreviewed model/rate combination')
        settings = package['settings'][stage]
        if set(settings) - {'model', 'max_completion_tokens', 'max_tokens', 'temperature'}:
            raise ValueError('Unexpected model setting')
        if not isinstance(output_cap(settings), int) or output_cap(settings) <= 0:
            raise ValueError('Invalid output cap')
        for rate in package['rates_nusd_per_token'][stage].values():
            if not isinstance(rate, int) or rate <= 0:
                raise ValueError('Invalid frozen rate')
    return digest(canonical(package))


def date_line(case):
    return f"\nToday's date is {case['date']}." if case['date'] else ''


def output_cap(settings):
    return settings.get('max_completion_tokens', settings.get('max_tokens'))


def request(package, case, arm, stage, answer=None):
    if stage == 'generate':
        content = package['prompts'][arm].format(context=case['context'],
                     question=case['question'], today_line=date_line(case))
    else:
        # replace rather than format: user evidence may contain literal braces.
        content = case['judge_template'].replace('{response}', answer)
    return dict(package['settings'][stage], messages=[{'role': 'user', 'content': content}])


def reserve(package, stage, req):
    # UTF-8 bytes bound byte-BPE content tokens; 1024 additional tokens cover the
    # fixed single-message wrapper. No tools, images, histories or hidden prompts.
    input_bound = len(req['messages'][0]['content'].encode()) + 1024
    rates = package['rates_nusd_per_token'][stage]
    # September11 official Luna page lists 1.25x cache-write pricing.
    # Reserve that input premium even when no cache write is reported.
    input_cost = input_bound * rates['input']
    if stage == 'generate':
        input_cost = (input_cost * 125 + 99) // 100
    return input_cost + output_cap(req) * rates['output']


def preflight(package):
    identity = validate(package)
    bound = 0
    for case in package['cases']:
        for arm in package['prompts']:
            bound += reserve(package, 'generate', request(package, case, arm, 'generate'))
            # Frozen maximum answer bytes, enforced before judge dispatch.
            # Oversized answers are errors, never silently truncated.
            bound += reserve(package, 'judge', request(package, case, arm, 'judge',
                                'x' * package.get('answer_max_bytes', 8192)))
    return {'package_sha256': identity, 'cases': len(package['cases']),
            'intended_answers': 2 * len(package['cases']),
            'maximum_requests_no_retries': 4 * len(package['cases']),
            'conservative_full_reservation_nusd': bound,
            'conservative_full_reservation_usd': bound / 1e9,
            'scope': 'Offline structural check, not answer-quality evidence',
            'cost_limitations': 'Frozen rates; no cache discount assumed. Per-run only. Large bound is not expected invoice.'}


def parse_answer(text):
    # Preserve qa_accuracy_eval.py historical first-ANSWER/first-line behavior.
    text = text.strip()
    match = re.search(r'ANSWER:\s*(.+)', text, re.IGNORECASE | re.DOTALL)
    return (match.group(1).strip() if match else text).split('\n')[0].strip()


def verdict(text):
    # Same interpretation for normal yes/no outputs; ambiguous outputs are errors.
    normalized = text.strip().lower().rstrip('.!')
    if normalized not in ('yes', 'no'):
        raise ValueError('Invalid judge verdict; requires yes/no')
    return normalized == 'yes'


def summary(package, state):
    arms = {}
    for arm in package['prompts']:
        rows = [state['jobs'].get(f'{c["id"]}/{arm}/judge', {}) for c in package['cases']]
        arms[arm] = {'intended': len(rows),
                     'completed': sum(r.get('status') == 'complete' for r in rows),
                     'correct': sum(r.get('correct') is True for r in rows)}
    paired = []
    for c in package['cases']:
        a = state['jobs'].get(f'{c["id"]}/baseline/judge', {})
        b = state['jobs'].get(f'{c["id"]}/candidate/judge', {})
        if a.get('status') == b.get('status') == 'complete':
            paired.append({'id': c['id'], 'abstention': c['abstention'],
                           'baseline': a['correct'], 'candidate': b['correct']})
    return {'mode': state['binding']['mode'], 'arms': arms, 'paired': paired,
            'complete': all(a['completed'] == a['intended'] for a in arms.values()),
            'reserved_usd': state['reserved_nusd'] / 1e9,
            'observed_usage_cost_without_cache_discount_usd': sum(j.get('usage_cost_nusd', 0) for j in state['jobs'].values()) / 1e9,
            'unresolved': [k for k, v in state['jobs'].items() if v['status'] != 'complete'],
            'shared_judgments': sum(bool(j.get('shared_from')) for j in state['jobs'].values()),
            'provider_calls': sum(bool(j.get('response')) for j in state['jobs'].values()),
            'reconciled_attempts_with_unknown_billing': state.get('reconciled_attempts', []),
            'quality_claim': 'INCOMPLETE unless complete=true; diagnostic cohort is not population accuracy'}


def validate_shared_judgments(state):
    for key, job in state['jobs'].items():
        if not job.get('shared_from'):
            continue
        source_key = job['shared_from']
        source = state['jobs'].get(source_key, {})
        parts, parent = key.split('/'), source_key.split('/')
        if (len(parts) != 3 or len(parent) != 3 or parts[0] != parent[0]
            or parts[1] == parent[1] or parts[2] != 'judge' or parent[2] != 'judge'
            or job.get('status') != 'complete' or source.get('status') != 'complete'
            or source.get('shared_from') or not source.get('response')
            or job.get('request_sha256') != source.get('request_sha256')
            or job.get('correct') != source.get('correct')
            or job.get('reservation_nusd') != 0 or job.get('response') is not None
            or job.get('usage_cost_nusd', 0) != 0):
            raise ValueError('Invalid shared-judgment provenance')


def run(package, directory, provider, budget_nusd, authorization, mode):
    identity = validate(package)
    if mode not in ('offline-test', 'paid') or not authorization or budget_nusd <= 0:
        raise ValueError('Explicit mode, authorization and positive budget required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'checkpoint.json'
        binding = {'package_sha256': identity, 'budget_nusd': budget_nusd,
                   'authorization': authorization, 'mode': mode,
                   'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if path.exists():
            state = json.loads(path.read_text())
            if state['binding'] != binding:
                raise ValueError('Resume manifest/budget/authorization/code mismatch')
            if state['reserved_nusd'] != sum(j['reservation_nusd'] for j in state['jobs'].values()) + sum(j['reservation_nusd'] for j in state.get('reconciled_attempts', [])):
                raise ValueError('Corrupt reservation ledger')
            validate_shared_judgments(state)
            from .verification import verify_state
            verify_state(package, state, require_complete=False)
            if any(j['status'] != 'complete' for j in state['jobs'].values()):
                raise ValueError('Unresolved prior attempt: no automatic retry; a new reviewed package is required')
        else:
            state = {'binding': binding, 'reserved_nusd': 0, 'jobs': {}}
            atomic(path, state)
        for case in package['cases']:
            # Counterbalanced by question hash; no outcome-dependent execution order.
            arms = ['baseline', 'candidate']
            if int(digest(case['id'])[-1], 16) % 2:
                arms.reverse()
            for arm in arms:
                for stage in ('generate', 'judge'):
                    key = f'{case["id"]}/{arm}/{stage}'
                    if key in state['jobs']:
                        continue
                    answer = state['jobs'].get(f'{case["id"]}/{arm}/generate', {}).get('answer')
                    req = request(package, case, arm, stage, answer)
                    if stage == 'judge':
                        other = 'candidate' if arm == 'baseline' else 'baseline'
                        source_key = f'{case["id"]}/{other}/judge'
                        source = state['jobs'].get(source_key, {})
                        req_hash = digest(canonical(req))
                        if (source.get('status') == 'complete'
                            and not source.get('shared_from')
                            and source.get('request_sha256') == req_hash):
                            state['jobs'][key] = {'status': 'complete',
                                'correct': source['correct'], 'request_sha256': req_hash,
                                'reservation_nusd': 0, 'shared_from': source_key,
                                'started_unix': time.time(), 'elapsed_seconds': 0}
                            atomic(path, state)
                            continue
                    cost = reserve(package, stage, req)
                    if state['reserved_nusd'] + cost > budget_nusd:
                        atomic(directory / 'summary.json', summary(package, state))
                        raise ValueError('Budget stop before dispatch')
                    job = {'status': 'pending', 'reservation_nusd': cost,
                           'request_sha256': digest(canonical(req)), 'started_unix': time.time()}
                    state['jobs'][key] = job
                    state['reserved_nusd'] += cost
                    atomic(path, state)  # durable before any external call
                    try:
                        result = provider(req)
                        job['response'] = result
                        usage = result['usage']
                        inp, out = usage['prompt_tokens'], usage['completion_tokens']
                        if not all(type(n) is int and n >= 0 for n in (inp, out)):
                            raise ValueError('Invalid usage')
                        from .verification import response_cost
                        job['usage_cost_nusd'] = response_cost(package, stage, req, result)
                        if out > output_cap(req) or inp > len(req['messages'][0]['content'].encode()) + 1024:
                            raise ValueError('Usage token bounds exceeded')
                        if not (result['model'] == req['model'] or result['model'].startswith(req['model'] + '-')):
                            raise ValueError('Unexpected returned model')
                        if job['usage_cost_nusd'] > cost:
                            raise ValueError('Observed usage exceeds reservation; halt for reconciliation')
                        if result['finish_reason'] != 'stop':
                            raise ValueError('Incomplete model output')
                        if stage == 'generate':
                            job['answer'] = parse_answer(result['text'])
                            if not job['answer'] or len(job['answer'].encode()) > package.get('answer_max_bytes', 8192):
                                raise ValueError('Empty or oversized answer')
                        else:
                            job['correct'] = verdict(result['text'])
                        job['status'] = 'complete'
                    except Exception as error:
                        job['status'] = 'error'
                        # Exception text may include request data or secrets; retain class only.
                        job['error_class'] = type(error).__name__
                        atomic(path, state)
                        atomic(directory / 'summary.json', summary(package, state))
                        raise RuntimeError('Attempt failed; checkpoint preserved; automatic retry disabled') from None
                    finally:
                        job['elapsed_seconds'] = time.time() - job['started_unix']
                        atomic(path, state)
        report = summary(package, state)
        atomic(directory / 'summary.json', report)
        return report


class OpenAIProvider:
    def __init__(self):
        from openai import OpenAI
        # No .env load, project imports, alternate endpoint, or SDK retries.
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],
                             base_url='https://api.openai.com/v1', max_retries=0, timeout=90)

    def __call__(self, req):
        result = self.client.chat.completions.create(**req)
        choice = result.choices[0]
        return {'text': choice.message.content or '', 'finish_reason': choice.finish_reason,
                'model': result.model, 'id': result.id,
                'request_id': getattr(result, '_request_id', None),
                'usage': result.usage.model_dump() if result.usage else None,
                'created': result.created}
