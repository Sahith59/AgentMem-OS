"""Read-only receipt replay. Does not trust runner's stored grades or totals."""
from collections import Counter
from .contract import canonical, sha, validate, request, reservation, response_cost, verdict


def verify(package, state, *, complete=False):
    identity = validate(package)
    if state['binding']['package_sha256'] != identity:
        raise ValueError('Package binding mismatch')
    if state['binding']['mode'] not in {'paid', 'offline-test'}:
        raise ValueError('Invalid mode')
    jobs = state['jobs']
    cases = {c['id']: c for c in package['cases']}
    if set(jobs) - set(cases):
        raise ValueError('Unknown job')
    reserved = 0
    receipt_ids = set()
    returned_models = set()
    for cid, job in jobs.items():
        req = request(cases[cid])
        if job['request_sha256'] != sha(canonical(req)) or job['reservation_nusd'] != reservation(req):
            raise ValueError('Request/reservation mismatch')
        reserved += reservation(req)
        if job['status'] not in {'pending', 'error', 'complete'}:
            raise ValueError('Invalid job state')
        if job['status'] == 'complete':
            response = job['response']
            if response['id'] in receipt_ids:
                raise ValueError('Reused receipt')
            receipt_ids.add(response['id'])
            returned_models.add(response['model'])
            if job['usage_upper_nusd'] != response_cost(req, response) or job['correct'] is not verdict(response['text']):
                raise ValueError('Receipt/grade mismatch')
    if reserved != state['reserved_nusd'] or reserved > state['binding']['budget_nusd']:
        raise ValueError('Reservation ledger mismatch')
    finished = len(jobs) == len(cases) and all(j['status'] == 'complete' for j in jobs.values())
    if complete and not finished:
        raise ValueError('Incomplete run')
    rows = [(cases[cid], j['correct']) for cid, j in jobs.items() if j['status'] == 'complete']
    report = {'mode': state['binding']['mode'], 'complete': finished,
        'intended': len(cases), 'completed': len(rows), 'reserved_nusd': reserved,
        'usage_upper_nusd': sum(j.get('usage_upper_nusd', 0) for j in jobs.values()),
        'returned_models': sorted(returned_models), 'paid_accuracy_claim_allowed': state['binding']['mode'] == 'paid' and finished}
    if package['purpose'] == 'bridge':
        report.update(new_judge_correct=sum(v for _, v in rows),
                      historical_correct=423, interpretation='Judge-scale shift only; answers unchanged.')
    else:
        false_accepts = sum(v and not c['expected'] for c, v in rows)
        false_rejects = sum(not v and c['expected'] for c, v in rows)
        critical = sum(c['critical'] and v != c['expected'] for c, v in rows)
        correct = len(rows) - false_accepts - false_rejects
        strata = {}
        for c, value in rows:
            key = 'abstention' if c['abstention'] else c['type']
            entry = strata.setdefault(key, {'completed': 0, 'correct': 0})
            entry['completed'] += 1
            entry['correct'] += value == c['expected']
        families = Counter(c['family'] for c, v in rows if v != c['expected'])
        report.update(correct=correct, false_accepts=false_accepts, false_rejects=false_rejects,
            critical_errors=critical, rubric_strata=strata, families_with_error=len(families),
            gate='NOT_A_VALIDATION_RUN')
        if package['purpose'] == 'validation':
            passed = finished and correct >= 114 and false_accepts <= 3 and false_rejects <= 3 and critical == 0
            report['thresholds_met'] = passed
            report['gate'] = ('PASS_INTERNAL_CALIBRATION' if passed else 'FAIL_OR_INCOMPLETE')
        if state['binding']['mode'] != 'paid':
            report['gate'] = 'MOCK_ONLY_NO_MODEL_QUALITY_EVIDENCE'
    return report
