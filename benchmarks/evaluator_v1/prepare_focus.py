"""Prepare original-turn focus inputs from a frozen packet and authorized scopes.

Offline only. No labels, old outcomes or gold evidence choose the source pool.
The adapter checks EVERY full500 scope, then reports eligibility without ranking
or claiming that a selector has run. Conflicting attribution rejects the case.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re

from ..evidence_focus import FocusInput, SourceTurn, selection_request, digest, validate_input
from .build import input_record, write_new


def prepare(package_path, cache_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    package = json.loads(Path(package_path).read_text())
    cache = json.loads(Path(cache_path).read_text())
    queries = {q['question_id']: q for q in cache['queries']}
    memories = {m['mid']: m for m in cache['memories']}
    if len(queries) != 500 or len(package['cases']) != 500:
        raise ValueError('Expected complete full500')
    rows = []
    for case in package['cases']:
        q = queries[case['id']]
        if case['question'] != q['question'] or digest(case['context']) != case['context_sha256']:
            raise ValueError('Question or frozen context mismatch')
        turns = []
        for mid in q['scope_keys']:
            for index, source in enumerate(memories[mid]['turns']):
                text = source['content']
                if source['role'] not in {'user', 'assistant'} or not text or text not in case['context']:
                    continue
                date = re.match(r'^\[([^\]]+)\]', text)
                turns.append(SourceTurn(digest(mid + ':' + str(index))[:24], source['role'],
                                        date.group(1) if date else '', text))
        # Original packet order; IDs resolve ties without knowledge of answers.
        turns.sort(key=lambda t: (case['context'].index(t.text), t.id))
        runtime = FocusInput(q['question'], q.get('question_date', ''), case['context'], tuple(turns))
        row = {'id': case['id'], 'original_context_sha256': digest(case['context']),
               'whole_original_turns_delivered': len(turns), 'baseline_chars': len(case['context']),
               'free_chars_within_40000': 40000 - len(case['context'])}
        try:
            validate_input(runtime)
            request = selection_request(runtime)
            row['status'] = 'READY_FOR_SELECTOR_QUALITY_TEST' if turns else 'NO_COMPLETE_SOURCE_TURNS'
            row['request_sha256'] = digest(json.dumps(request, sort_keys=True))
            row['request_input_utf8_bytes'] = sum(len(m['content'].encode()) for m in request['messages'])
            path = out / 'inputs' / (case['id'] + '.json')
            write_new(path, asdict(runtime))
            row['input_file'] = input_record(path)
        except ValueError:
            row['status'] = 'REJECTED_AMBIGUOUS_SOURCE'
        rows.append(row)
    report = {'status': 'OFFLINE_POOL_PREPARATION_ONLY', 'cases': len(rows),
        'counts': dict(Counter(r['status'] for r in rows)),
        'model_calls': 0, 'new_answers': 0, 'accuracy': 'NOT_MEASURED',
        'algorithm': 'Only complete original turns already in baseline, authorized scope, packet order.',
        'limitation': 'Focus cannot retrieve missing turns; raw-turn availability is not semantic completeness.',
        'sources': {'package': input_record(package_path), 'cache': input_record(cache_path),
                    'adapter': input_record(__file__),
                    'selector': input_record(Path(__file__).parents[1] / 'evidence_focus.py')},
        'rows': rows}
    write_new(out / 'report.json', report)
    return {k: v for k, v in report.items() if k != 'rows'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('package', type=Path)
    ap.add_argument('cache', type=Path)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    print(json.dumps(prepare(a.package, a.cache, a.output), indent=2))
