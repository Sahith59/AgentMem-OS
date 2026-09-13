"""Independent contract, checkpoint and quality-gate verification.

No runner/parser/provider/project-DB imports. Artifact hashes bind this verifier
alongside the runtime. Shares only response envelope checks with the dispatcher.
"""
import hashlib
import json
from pathlib import Path
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def request(package, case, arm, stage, answer=None):
    if stage == 'generate':
        content = package['prompts'][arm].format(
            context=case['arm_contexts'][arm], question=case['question'],
            today_line=f"\nToday's date is {case['date']}." if case['date'] else '')
    else:
        content = case['judge_template'].replace('{response}', answer)
    return dict(package['settings'][stage], messages=[{'role': 'user', 'content': content}])


def reservation(package, stage, req):
    rates = package['rates_nusd_per_token'][stage]
    n = (len(req['messages'][0]['content'].encode()) + 1024) * rates['input']
    if stage == 'generate':
        n = (n * 125 + 99) // 100
    return n + req.get('max_completion_tokens', req.get('max_tokens')) * rates['output']


def response_cost(package, stage, req, response):
    usage = response['usage']
    inp, out = usage['prompt_tokens'], usage['completion_tokens']
    require(all(integer(n) for n in (inp, out)), 'Invalid usage')
    require(inp <= len(req['messages'][0]['content'].encode()) + 1024, 'Input bound exceeded')
    require(out <= req.get('max_completion_tokens', req.get('max_tokens')), 'Output cap exceeded')
    require(response['finish_reason'] == 'stop', 'Incomplete provider response')
    require(response['model'] in package['allowed_returned_models'][stage], 'Returned model drift')
    require(bool(response.get('id')) and bool(response.get('request_id')), 'Missing provider receipt IDs')
    require(isinstance(response.get('text'), str), 'Nontext response')
    details = usage.get('prompt_tokens_details') or {}
    cached = details.get('cached_tokens', 0)
    written = details.get('cache_write_tokens', 0)
    require(integer(cached) and integer(written) and cached + written <= inp, 'Invalid cache usage')
    rates = package['rates_nusd_per_token'][stage]
    cost = inp * rates['input'] + out * rates['output']
    if stage == 'generate':
        cost += (written * rates['input'] + 3) // 4
    require(cost <= reservation(package, stage, req), 'Usage exceeds reservation')
    return cost


def answer(text):
    text = text.strip()
    match = re.search(r'ANSWER:\s*(.+)', text, flags=re.I | re.S)
    return (match.group(1).strip() if match else text).split('\n')[0].strip()


def verdict(text):
    value = text.strip().lower().rstrip('.!')
    require(value in ('yes', 'no'), 'Malformed judge verdict')
    return value == 'yes'


def verify_contract(package):
    require(package.get('screen_kind') == 'source-supplement-150-v1', 'Wrong screen kind')
    require(package['schema'] == 1 and package['judge_policy'] == 'shared-exact-request-v2', 'Wrong protocol')
    require(set(package['runtime_sha256']) == {'runner.py', 'packet.py', 'verification.py'}, 'Missing runtime binding')
    for name, expected in package['runtime_sha256'].items():
        require(file_sha(Path(__file__).with_name(name)) == expected, 'Runtime hash mismatch: ' + name)
    for source in package['source_artifacts'].values():
        require(file_sha(source['path']) == source['sha256'], 'Source artifact changed')
    source = json.loads(Path(package['source_artifacts']['baseline_package']['path']).read_text())
    original = {c['id']: c for c in source['cases']}
    selection = json.loads(Path(package['source_artifacts']['selection']['path']).read_text())
    require(package['selection'] == selection, 'Frozen selection changed')
    require(len(package['cases']) == 150 and len(original) == 500, 'Wrong population')
    ids = [c['id'] for c in package['cases']]
    require(len(set(ids)) == 150 and ids == [r['question_id'] for r in selection['cases']], 'Membership mismatch')
    require(package['settings'] == source['settings'], 'Model settings changed')
    require(package['rates_nusd_per_token'] == source['rates_nusd_per_token'], 'Unreviewed rates')
    require(package['answer_max_bytes'] == source['answer_max_bytes'], 'Answer limit changed')
    require(package['prompts'] == {'baseline': source['prompt'], 'candidate': source['prompt']}, 'Prompt changed')
    require(package['allowed_returned_models'] == {'generate': ['gpt-5.6-luna'], 'judge': ['gpt-4o-2024-08-06']}, 'Unreviewed returned models')
    require(package['gates'] == {'net_gain_min': 10, 'stable_control_losses_max': 2,
            'stable_control_correct_min': 56, 'abstention_net_gain_min': 0}, 'Gate changed')
    require(integer(package['proposed_budget_nusd']) and package['proposed_budget_nusd'] > 0, 'Invalid proposed budget')
    require(package['packet_runner_sha256'] == package['runtime_sha256']['packet.py'], 'Packet runtime binding')
    for arm in ('baseline', 'candidate'):
        require(digest(package['prompts'][arm]) == package['prompt_sha256'][arm], 'Prompt hash mismatch')
    cohorts = {}; abstentions = 0
    for c, selected in zip(package['cases'], selection['cases']):
        old = original[c['id']]
        require('/' not in c['id'], 'Invalid job identifier')
        for field in ('context', 'question', 'gold', 'judge_template'):
            require(digest(c[field]) == c[field + '_sha256'], 'Case hash mismatch: ' + field)
        require(c['judge_template'].count('{response}') == 1, 'Judge response slot')
        require(c['question_sha256'] == selected['question_sha256'] and c['type'] == selected['type'], 'Selection metadata changed')
        for field in ('question', 'question_sha256', 'gold', 'gold_sha256', 'date', 'type', 'judge_template'):
            require(c[field] == old[field], 'Original case metadata changed: ' + field)
        require(c['abstention'] == old['abst'] == selected['abstention'], 'Abstention metadata changed')
        require(c['cohort'] == selected['cohort'], 'Cohort metadata changed')
        cohorts[c['cohort']] = cohorts.get(c['cohort'], 0) + 1
        abstentions += c['abstention']
        require(c['context'] == old['context'] == c['arm_contexts']['baseline'], 'Baseline changed')
        new = c['arm_contexts']['candidate']
        require(new.startswith(old['context']) and len(new) <= 40000 and len(new) - len(old['context']) <= 4000, 'Packet preservation/cap failed')
        for arm in ('baseline', 'candidate'):
            require(digest(c['arm_contexts'][arm]) == c['arm_context_sha256'][arm] == selected[arm + '_sha256'], 'Arm hash mismatch')
            req = request(package, c, arm, 'generate')
            require(len(req['messages'][0]['content'].encode()) + 1024 < 272000, 'Long-context price tier')
    require(cohorts == {'stable_miss': 72, 'disagreement': 20, 'stable_pass_control': 58} and abstentions == 30, 'Cohort counts changed')
    return digest(canonical(package))


def quality_gates(package, rows, complete):
    controls = [r for r in rows if r['cohort'] == 'stable_pass_control']
    abstentions = [r for r in rows if r['abstention']]
    gains = sum(not r['baseline'] and r['candidate'] for r in rows)
    losses = sum(r['baseline'] and not r['candidate'] for r in rows)
    control_losses = sum(r['baseline'] and not r['candidate'] for r in controls)
    control_correct = sum(r['candidate'] for r in controls)
    abst_net = sum(int(r['candidate']) - int(r['baseline']) for r in abstentions)
    g = package['gates']
    checks = {'complete_150_pairs': complete and len(rows) == 150,
              'net_gain': gains - losses >= g['net_gain_min'],
              'stable_control_losses': control_losses <= g['stable_control_losses_max'],
              'stable_control_absolute_floor': len(controls) == 58 and control_correct >= g['stable_control_correct_min'],
              'abstention_preserved': len(abstentions) == 30 and abst_net >= g['abstention_net_gain_min']}
    return {'screen': 'PASS' if all(checks.values()) else 'FAIL' if complete else 'INCOMPLETE',
            'checks': checks, 'gains': gains, 'losses': losses, 'net_gain': gains-losses,
            'stable_control_losses': control_losses, 'stable_control_correct': control_correct,
            'abstention_net_gain': abst_net,
            'baseline_correct': sum(r['baseline'] for r in rows),
            'candidate_correct': sum(r['candidate'] for r in rows), 'denominator': len(rows)}


def verify_state(package, state, require_complete=True):
    require(state['binding']['package_sha256'] == digest(canonical(package)), 'Checkpoint package mismatch')
    require(state['binding']['runner_sha256'] == file_sha(Path(__file__).with_name('runner.py')), 'Checkpoint runtime mismatch')
    require(state['binding']['mode'] in ('paid', 'offline-test'), 'Invalid execution mode')
    require(bool(state['binding']['authorization']), 'Missing authorization binding')
    require(not state.get('reconciled_attempts'), 'Retries/reconciliation outside frozen scope')
    schedule = []
    by_id = {c['id']: c for c in package['cases']}
    for c in package['cases']:
        arms = ['baseline', 'candidate']
        if int(digest(c['id'])[-1], 16) % 2:
            arms.reverse()
        schedule.extend(f'{c["id"]}/{arm}/{stage}' for arm in arms for stage in ('generate', 'judge'))
    jobs = state['jobs']
    require(list(jobs) == schedule[:len(jobs)], 'Checkpoint is not a valid schedule prefix')
    require(all(j['status'] == 'complete' for j in jobs.values()), 'Unresolved attempt; no automatic retry')
    require(not require_complete or len(jobs) == len(schedule), 'Incomplete run')
    response_ids = set(); request_ids = set(); reserved = 0; usage_cost = 0; shared = 0
    for key, job in jobs.items():
        qid, arm, stage = key.split('/')
        c = by_id[qid]
        saved_answer = jobs.get(f'{qid}/{arm}/generate', {}).get('answer')
        req = request(package, c, arm, stage, saved_answer)
        require(job['request_sha256'] == digest(canonical(req)), 'Checkpoint request mismatch')
        if job.get('shared_from'):
            parent_key = job['shared_from']; parent = jobs.get(parent_key, {})
            other = 'candidate' if arm == 'baseline' else 'baseline'
            require(stage == 'judge' and parent_key == f'{qid}/{other}/judge', 'Cross-case sharing')
            require(list(jobs).index(parent_key) < list(jobs).index(key), 'Future shared source')
            require(not parent.get('shared_from') and bool(parent.get('response')), 'Invalid sharing chain')
            require(parent['request_sha256'] == job['request_sha256'] and parent['correct'] == job['correct'], 'Shared verdict mismatch')
            require(job['reservation_nusd'] == 0 and not job.get('response') and job.get('usage_cost_nusd', 0) == 0, 'Shared billing duplicated')
            shared += 1
        else:
            require(job['reservation_nusd'] == reservation(package, stage, req), 'Reservation mismatch')
            response = job['response']
            require(response['id'] not in response_ids and response['request_id'] not in request_ids, 'Duplicate provider receipt')
            response_ids.add(response['id']); request_ids.add(response['request_id'])
            cost = response_cost(package, stage, req, response)
            require(job['usage_cost_nusd'] == cost, 'Usage cost mismatch')
            usage_cost += cost
            if stage == 'generate':
                parsed = answer(response['text'])
                require(parsed and len(parsed.encode()) <= package['answer_max_bytes'] and parsed == job['answer'], 'Saved answer mismatch')
            else:
                require(type(job['correct']) is bool and verdict(response['text']) == job['correct'], 'Saved grade mismatch')
        reserved += job['reservation_nusd']
    require(integer(state['reserved_nusd']) and state['reserved_nusd'] == reserved, 'Reservation ledger mismatch')
    require(integer(state['binding']['budget_nusd']) and 0 < state['binding']['budget_nusd'] and reserved <= state['binding']['budget_nusd'], 'Budget mismatch')
    rows = []
    for c in package['cases']:
        left, right = [jobs.get(f'{c["id"]}/{arm}/judge') for arm in ('baseline', 'candidate')]
        if left and right:
            rows.append({'id': c['id'], 'cohort': c['cohort'], 'abstention': c['abstention'],
                         'baseline': left['correct'], 'candidate': right['correct']})
    complete = len(jobs) == len(schedule)
    return {'status': 'PASS_INTEGRITY', 'mode': state['binding']['mode'], 'complete': complete,
            'provider_calls': len(response_ids), 'shared_judgments': shared,
            'reserved_nusd': reserved, 'usage_cost_without_cache_discount_nusd': usage_cost,
            'quality': quality_gates(package, rows, complete), 'rows': rows,
            'scope': 'Development-exposed paired screen, not full500 accuracy or invoice reconciliation.'}


def verify_paid(package, state, approval, directory, require_complete=True):
    identity = verify_contract(package)
    require(approval.get('status') == 'FOUNDER_APPROVED' and bool(approval.get('founder_message')), 'No founder approval')
    require(approval.get('package_sha256') == identity, 'Approval package mismatch')
    require(approval.get('budget_nusd') == package['proposed_budget_nusd'] == state['binding']['budget_nusd'], 'Approval budget mismatch')
    require(Path(approval.get('run_directory', '')).resolve() == Path(directory).resolve(), 'Approval output mismatch')
    require(state['binding']['mode'] == 'paid' and state['binding']['authorization'] == digest(canonical(approval)), 'Approval checkpoint mismatch')
    return verify_state(package, state, require_complete=require_complete)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--approval-record', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    package = json.loads(args.package.read_text())
    identity = verify_contract(package)
    result = {'status': 'PASS_CONTRACT_ONLY', 'package_sha256': identity, 'paid_execution': False}
    if args.checkpoint:
        state = json.loads(args.checkpoint.read_text())
        if state['binding']['mode'] == 'paid':
            if not args.approval_record:
                parser.error('Paid verification requires the original founder approval record')
            approval = json.loads(args.approval_record.read_text())
            result = verify_paid(package, state, approval, args.checkpoint.parent, not args.allow_incomplete)
        else:
            result = verify_state(package, state, not args.allow_incomplete)
            result['warning'] = 'SYNTHETIC OFFLINE OUTPUT: no model-quality evidence'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
