"""Durable one-attempt Terra grading; never auto-retries an uncertain request."""
import fcntl
import json
from pathlib import Path
import time

from ..english_screen.runner import atomic, error_diagnostics, OpenAIProvider
from .contract import canonical, sha, preflight, request, reservation, response_cost, verdict
from .verify import verify


def check_approval(package, approval, mode):
    plan = preflight(package)
    if mode not in {'offline-test', 'paid'}:
        raise ValueError('Explicit execution mode required')
    if (approval.get('package_sha256') != plan['package_sha256']
        or type(approval.get('budget_nusd')) is not int
        or approval['budget_nusd'] < plan['maximum_reservation_nusd']
        or approval.get('maximum_attempts') != plan['calls_no_retries']
        or approval.get('mode') != mode or approval.get('approved') is not True
        or not approval.get('authorization_text')):
        raise ValueError('Exact package/budget/attempt authorization required')
    if package['purpose'] in {'validation', 'bridge'} and mode == 'paid':
        # Each paid stage binds the actual successful preceding stage.
        evidence = approval.get('prerequisite', {})
        if not isinstance(evidence, dict) or set(evidence) != {'package', 'checkpoint'}:
            raise ValueError('Missing preceding-stage evidence')
        from .contract import file_sha
        for key in ('package', 'checkpoint'):
            if file_sha(evidence[key]['path']) != evidence[key]['sha256']:
                raise ValueError('Changed calibration evidence')
        cp = json.loads(Path(evidence['package']['path']).read_text())
        cs = json.loads(Path(evidence['checkpoint']['path']).read_text())
        result = verify(cp, cs, complete=True)
        if package['purpose'] == 'validation' and cp['inputs'].get('calibration') != package['inputs']['calibration']:
            raise ValueError('Development and validation must bind the same calibration set')
        required = 'development' if package['purpose'] == 'validation' else 'validation'
        if required == 'development':
            passed = result.get('correct') == 32 and result['completed'] == 32
        else:
            passed = result.get('gate') == 'PASS_INTERNAL_CALIBRATION'
        if cp['purpose'] != required or not passed or result['mode'] != 'paid':
            raise ValueError('Requires actual passing ' + required)

    return plan


def run(package, directory, approval, *, mode='paid', provider=None):
    plan = check_approval(package, approval, mode)
    # Production path constructs the no-retry provider only after approval checks.
    if mode == 'paid' and provider is not None:
        raise ValueError('Injected provider allowed only in offline-test mode')
    if mode == 'offline-test' and provider is None:
        raise ValueError('Offline mode requires a fake provider')
    directory = Path(directory).resolve()
    if mode == 'paid' and approval.get('output_directory') != str(directory):
        raise ValueError('Paid approval must bind one output directory; no duplicate run directory')
    directory.mkdir(parents=True, exist_ok=True)
    binding = {'package_sha256': plan['package_sha256'], 'approval_sha256': sha(canonical(approval)),
               'budget_nusd': approval['budget_nusd'], 'mode': mode}
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = directory / 'checkpoint.json'
        if path.exists():
            state = json.loads(path.read_text())
            if state['binding'] != binding:
                raise ValueError('Changed resume binding')
            verify(package, state)
            if any(j['status'] != 'complete' for j in state['jobs'].values()):
                raise ValueError('Unresolved attempt; no automatic retry or reservation release')
        else:
            state = {'binding': binding, 'jobs': {}, 'reserved_nusd': 0}
            atomic(path, state)
        for case in package['cases']:
            if case['id'] in state['jobs']:
                continue
            req = request(case)
            cost = reservation(req)
            if state['reserved_nusd'] + cost > approval['budget_nusd']:
                raise ValueError('Budget stop before dispatch')
            job = {'status': 'pending', 'request_sha256': sha(canonical(req)),
                   'reservation_nusd': cost, 'started_unix': time.time()}
            state['jobs'][case['id']] = job
            state['reserved_nusd'] += cost
            atomic(path, state)
            try:
                if provider is None:
                    provider = OpenAIProvider()
                response = provider(req)
                job['response'] = response
                job['usage_upper_nusd'] = response_cost(req, response)
                job['correct'] = verdict(response['text'])
                job['status'] = 'complete'
                # Detect duplicate receipts before another dispatch.
                if sum(j.get('response', {}).get('id') == response['id'] for j in state['jobs'].values()) != 1:
                    raise ValueError('Duplicate provider receipt')
            except Exception as error:
                job['status'] = 'error'
                job['error_class'] = type(error).__name__
                job['error_diagnostics'] = error_diagnostics(error)
                atomic(path, state)
                raise RuntimeError('Attempt failed; preserved; no automatic retry') from None
            finally:
                job['elapsed_seconds'] = time.time() - job['started_unix']
                atomic(path, state)
        report = verify(package, state, complete=True)
        atomic(directory / 'summary.json', report)
        return report
