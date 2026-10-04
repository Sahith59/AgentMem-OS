"""Independent read-only check of prepared source pools against frozen scopes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    root = Path(directory)
    report = json.loads((root / 'report.json').read_text())
    for item in report['sources'].values():
        if hash_file(item['path']) != item['sha256']:
            raise ValueError('Changed source')
    package = json.loads(Path(report['sources']['package']['path']).read_text())
    cache = json.loads(Path(report['sources']['cache']['path']).read_text())
    cases = {c['id']: c for c in package['cases']}
    queries = {q['question_id']: q for q in cache['queries']}
    memories = {m['mid']: m for m in cache['memories']}
    rows = report['rows']
    if len(rows) != 500 or {r['id'] for r in rows} != set(cases):
        raise ValueError('Incomplete case membership')
    checked = 0
    free = []
    counts = []
    for row in rows:
        if row['status'] != 'READY_FOR_SELECTOR_QUALITY_TEST':
            raise ValueError('Pool not ready')
        record = row['input_file']
        if hash_file(record['path']) != record['sha256']:
            raise ValueError('Changed runtime input')
        value = json.loads(Path(record['path']).read_text())
        case, query = cases[row['id']], queries[row['id']]
        if (set(value) != {'question','question_date','packet','turns'} or
            value['packet'] != case['context'] or value['question'] != query['question'] or
            value['question_date'] != query['question_date']):
            raise ValueError('Runtime projection or frozen packet mismatch')
        allowed = {}
        for mid in query['scope_keys']:
            for index, turn in enumerate(memories[mid]['turns']):
                if turn['role'] in {'user','assistant'} and turn['content'] and turn['content'] in case['context']:
                    key = hashlib.sha256((mid + ':' + str(index)).encode()).hexdigest()[:24]
                    allowed[key] = turn
        if {t['id'] for t in value['turns']} != set(allowed) or len(value['turns']) != len(allowed):
            raise ValueError('Wrong scoped source population')
        if value['turns'] != sorted(value['turns'], key=lambda t: (case['context'].index(t['text']), t['id'])):
            raise ValueError('Changed source order')
        for turn in value['turns']:
            source = allowed[turn['id']]
            text = source['content']
            date = text[1:].split(']', 1)[0] if text.startswith('[') and ']' in text else ''
            if (set(turn) != {'id','role','observed_at','text'} or turn['observed_at'] != date
                or turn['role'] != source['role'] or turn['text'] != text):
                raise ValueError('Changed source text/role')
            checked += 1
        free.append(40000 - len(value['packet']))
        counts.append(len(value['turns']))
    if report['counts'] != dict(Counter(r['status'] for r in rows)):
        raise ValueError('Counts mismatch')
    return {'status':'PASS_OFFLINE_SOURCE_BINDINGS', 'verifier_sha256':hash_file(__file__), 'cases':500, 'original_turns_checked':checked,
        'median_source_turns':statistics.median(counts), 'minimum_free_chars':min(free),
        'median_free_chars':statistics.median(free), 'no_original_context_changes':True,
        'model_calls':0, 'selector_quality':'NOT_MEASURED', 'english_accuracy':'NOT_MEASURED'}


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    a=ap.parse_args()
    result=verify(a.directory)
    with (a.directory/'independent-verification.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
