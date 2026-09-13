"""Build the previously selected source screen. Offline; refuses existing output."""
import argparse
import copy
import json
from pathlib import Path

from . import packet
from .verification import digest, file_sha


GATES = {'net_gain_min': 10, 'stable_control_losses_max': 2,
         'stable_control_correct_min': 56, 'abstention_net_gain_min': 0}


def build(baseline_path, selection_path, candidate_dir, source_paths):
    baseline_path, selection_path, candidate_dir = map(Path, (baseline_path, selection_path, candidate_dir))
    old = json.loads(baseline_path.read_text())
    selection = json.loads(selection_path.read_text())
    originals = {c['id']: c for c in old['cases']}
    sources = dict(source_paths, baseline_package=baseline_path, selection=selection_path)
    package = {
        'schema': 1, 'screen_kind': 'source-supplement-150-v1',
        'experiment_kind': 'paired-context-retrieval-v1', 'judge_policy': 'shared-exact-request-v2',
        'scope': 'Development-exposed 150-pair screen; not population accuracy or a held-out evaluation.',
        'settings': old['settings'], 'rates_nusd_per_token': old['rates_nusd_per_token'],
        'answer_max_bytes': old['answer_max_bytes'],
        'prompts': {a: old['prompt'] for a in ('baseline', 'candidate')},
        'prompt_sha256': {a: digest(old['prompt']) for a in ('baseline', 'candidate')},
        'allowed_returned_models': {'generate': ['gpt-5.6-luna'], 'judge': ['gpt-4o-2024-08-06']},
        'gates': dict(GATES), 'selection': selection, 'proposed_budget_nusd': 1,
        'runtime_sha256': {name: file_sha(Path(__file__).with_name(name))
                           for name in ('runner.py', 'packet.py', 'verification.py')},
        'source_artifacts': {name: {'path': str(Path(path).resolve()), 'sha256': file_sha(path)}
                             for name, path in sources.items()}, 'cases': []}
    package['packet_runner_sha256'] = package['runtime_sha256']['packet.py']
    for selected in selection['cases']:
        c = copy.deepcopy(originals[selected['question_id']])
        candidate = (candidate_dir / (c['id'] + '.txt')).read_text()
        c.update(abstention=c['abst'], cohort=selected['cohort'],
                 arm_contexts={'baseline': c['context'], 'candidate': candidate},
                 arm_context_sha256={'baseline': digest(c['context']), 'candidate': digest(candidate)})
        package['cases'].append(c)
    # Ceiling to cents. Reservations stay spent even after provider errors.
    receipt = packet.preflight(package)
    bound = receipt['conservative_full_reservation_nusd']
    package['proposed_budget_nusd'] = ((bound + 9_999_999) // 10_000_000) * 10_000_000
    return package, packet.preflight(package)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--source', action='append', default=[], help='name=path for an additional bound evidence artifact')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    for value in args.source:
        name, path = value.split('=', 1)
        if name in sources or name in ('baseline_package', 'selection'):
            parser.error('Duplicate or reserved source name')
        sources[name] = Path(path)
    package, receipt = build(args.baseline, args.selection, args.candidate_dir, sources)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'package.json').write_text(json.dumps(package, indent=2, ensure_ascii=False) + '\n')
    (args.output / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(receipt, proposed_budget_usd=package['proposed_budget_nusd'] / 1e9), indent=2))


if __name__ == '__main__':
    main()
