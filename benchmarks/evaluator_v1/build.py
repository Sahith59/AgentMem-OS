"""Offline package builder. Exclusive output creation; no model or DB imports."""
import argparse
import json
from pathlib import Path

from .calibration import dataset
from .contract import ROOT, SETTINGS, RATES, SYSTEM, code_sources, file_sha, sha, canonical, preflight


def input_record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha(path)}


def bridge_cases(old_package, checkpoint):
    source = json.loads(Path(old_package).read_text())
    state = json.loads(Path(checkpoint).read_text())
    if len(source['cases']) != 500 or len(state['jobs']) != 1000:
        raise ValueError('Expected exact historical full500')
    if state['binding']['package_sha256'] != sha(canonical(source)):
        raise ValueError('Historical package/checkpoint binding mismatch')
    if sum(j.get('correct') is True for k, j in state['jobs'].items() if k.endswith('/judge')) != 423:
        raise ValueError('Wrong historical baseline')
    from ..english_screen.runner import parse_answer
    result = []
    for case in source['cases']:
        job = state['jobs'][case['id'] + '/generate']
        grade = state['jobs'][case['id'] + '/judge']
        req = dict(source['settings']['generate'], messages=[{'role': 'user', 'content':
            source['prompt'].format(context=case['context'], question=case['question'],
                today_line=f"\nToday's date is {case['date']}." if case['date'] else '')}])
        if (job['status'] != 'complete' or grade['status'] != 'complete'
            or job['request_sha256'] != sha(canonical(req))
            or job['answer'] != parse_answer(job['response']['text'])
            or job['response']['model'] != 'gpt-5.6-luna'):
            raise ValueError('Saved answer provenance mismatch')
        for key in ('question', 'context', 'gold'):
            if sha(case[key]) != case[key + '_sha256']:
                raise ValueError('Historical field hash mismatch')
        result.append(dict(id=case['id'], type=case['type'], abstention=case['abst'],
            question=case['question'], gold=case['gold'], response=job['answer']))
    return result


def make_package(purpose, cases, inputs):
    return {'schema': 'terra-reference-v1', 'purpose': purpose,
        'settings': dict(SETTINGS), 'rates_nusd_per_token': dict(RATES), 'system': SYSTEM,
        'code_sources': code_sources(), 'inputs': inputs,
        'price_source': 'https://developers.openai.com/api/docs/models/gpt-5.6-terra',
        'prices_checked': '2026-10-04',
        'scope': 'New internal judge series, not official GPT-4o benchmark parity. No paid authorization.',
        'cases': cases}


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def build(out, old_package=None, checkpoint=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    fixtures = ROOT / 'benchmarks/fixtures/evidence_semantics_v1.json'
    data = dataset(fixtures)
    write_new(out / 'calibration.json', data)
    inputs = {'calibration': input_record(out / 'calibration.json'), 'development_fixtures': input_record(fixtures),
              'reference_regressions': input_record(fixtures.with_name('judge_reference_regressions_v1.json'))}
    reports = {}
    for purpose in ('development', 'validation'):
        package = make_package(purpose, [c for c in data['cases'] if c['split'] == purpose], inputs)
        write_new(out / f'{purpose}-package.json', package)
        reports[purpose] = preflight(package)
    if old_package is not None and checkpoint is not None:
        package = make_package('bridge', bridge_cases(old_package, checkpoint),
            {'historical_package': input_record(old_package), 'historical_checkpoint': input_record(checkpoint)})
        write_new(out / 'bridge-package.json', package)
        reports['bridge'] = preflight(package)
    write_new(out / 'preflight.json', reports)
    return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--historical-package', type=Path)
    ap.add_argument('--historical-checkpoint', type=Path)
    args = ap.parse_args()
    print(json.dumps(build(args.output, args.historical_package, args.historical_checkpoint), indent=2))


if __name__ == '__main__':
    main()
