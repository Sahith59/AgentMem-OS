"""Exercise the real frozen package with synthetic responses and no network."""
import argparse
import json
from pathlib import Path
import socket

from . import packet, runner
from .verification import digest, verify_state


class SyntheticProvider:
    def __init__(self, identical=False):
        self.calls = 0
        self.identical = identical

    def __call__(self, req):
        self.calls += 1
        generation = req['model'] == 'gpt-5.6-luna'
        value = 'same synthetic answer' if self.identical else digest(req['messages'][0]['content'])
        return {'text': 'ANSWER: ' + value if generation else 'yes', 'finish_reason': 'stop',
                'model': 'gpt-5.6-luna' if generation else 'gpt-4o-2024-08-06',
                'id': f'SYNTHETIC-response-{self.calls}', 'request_id': f'SYNTHETIC-request-{self.calls}',
                'usage': {'prompt_tokens': 100, 'completion_tokens': 1}}


def smoke(package, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    results = {}
    unchanged = sum(
        case['arm_contexts']['baseline'] == case['arm_contexts']['candidate']
        for case in package['cases']
    )
    expected_distinct = 4 * len(package['cases']) - unchanged
    expected_identical = 3 * len(package['cases'])
    for name, identical, expected in [
        ('distinct', False, expected_distinct),
        ('identical', True, expected_identical),
    ]:
        provider = SyntheticProvider(identical)
        folder = output / name
        packet.run(package, folder, provider, package['proposed_budget_nusd'], 'SYNTHETIC-NOT-APPROVAL', 'offline-test')
        state = json.loads((folder / 'checkpoint.json').read_text())
        report = verify_state(package, state)
        if provider.calls != expected or not report['complete']:
            raise ValueError('Synthetic dispatch count mismatch')
        resumed = SyntheticProvider(identical)
        packet.run(package, folder, resumed, package['proposed_budget_nusd'], 'SYNTHETIC-NOT-APPROVAL', 'offline-test')
        if resumed.calls != 0:
            raise ValueError('Completed resume duplicated calls')
        report['warning'] = 'SYNTHETIC: tests execution only; not QA accuracy or actual spend'
        runner.atomic(folder / 'independent-verification.json', report)
        results[name] = {k: report[k] for k in ('complete', 'provider_calls', 'shared_judgments')}
        results[name]['resume_provider_calls'] = resumed.calls
    receipt = {'status': 'PASS_OFFLINE_EXECUTION', 'package_sha256': packet.validate(package),
               'real_provider_calls': 0, 'real_spend_usd': 0, 'scenarios': results,
               'unchanged_context_pairs': unchanged,
               'accuracy_claim': 'NONE. All model responses and grades are synthetic.'}
    runner.atomic(output / 'receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    def denied(*args, **kwargs):
        raise RuntimeError('Network forbidden in offline execution')
    socket.socket.connect = denied
    socket.create_connection = denied
    print(json.dumps(smoke(json.loads(args.package.read_text()), args.output), indent=2))


if __name__ == '__main__':
    main()
