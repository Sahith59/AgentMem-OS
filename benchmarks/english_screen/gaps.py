"""Recompute annotated-turn delivery for every persistent miss, without models.

Annotation coverage is a diagnostic proxy, not a sufficient-evidence judgment.
"""
import argparse
import json
from pathlib import Path


def inventory(run_root):
    root = Path(run_root)
    def read(path): return json.loads((root / path).read_text())
    original = {c['id']: c for c in read('full500-corrected-measurement-package-001/package.json')['cases']}
    annotations = {r['question_id']: r for r in read('raw-vocabulary-audit-002/results.json')}
    triage = {r['question_id']: r for r in read('stable-source-audit-001/triage.json')}
    repeats = [read(f'paid-full500-corrected-repeat{n}-001/checkpoint.json')['jobs'] for n in (1, 2)]
    stable = sorted(qid for qid in original if all(not s[qid + '/judge']['correct'] for s in repeats))
    if len(stable) != 72 or set(stable) != set(triage):
        raise ValueError('Persistent-miss population mismatch')
    rows = []
    for qid in stable:
        before = original[qid]['context']
        after = (root / 'source-supplement-audit-001/contexts' / (qid + '.txt')).read_text()
        turns = annotations[qid]['annotated_turns']
        missing_before = [a for a in turns if a['text'] not in before]
        missing_after = [a for a in turns if a['text'] not in after]
        gained = [a for a in turns if a['text'] not in before and a['text'] in after]
        rows.append({'question_id': qid, 'abstention': original[qid]['abst'],
                     'annotated_turns': len(turns), 'missing_before': len(missing_before),
                     'missing_after': len(missing_after), 'gained': len(gained),
                     'remaining_source_turns': [{'source_key': a['source_key'], 'turn_index': a['turn_index']} for a in missing_after],
                     'review_priority': triage[qid]['priority'], 'review_note': triage[qid]['note'],
                     'status': 'QA_UNMEASURED; cause remains provisional'} )
    answerable = [r for r in rows if not r['abstention']]
    summary = {'stable_misses': len(rows), 'answerable': len(answerable), 'abstention': len(rows)-len(answerable),
               'gaining_annotated_turns': sum(r['gained'] > 0 for r in rows),
               'answerable_gaining_annotated_turns': sum(r['gained'] > 0 for r in answerable),
               'answerable_all_annotated_turns_before': sum(r['missing_before'] == 0 for r in answerable),
               'answerable_all_annotated_turns_after': sum(r['missing_after'] == 0 for r in answerable),
               'answerable_still_missing_annotated_turns': sum(r['missing_after'] > 0 for r in answerable)}
    return {'summary': summary, 'rows': rows,
            'limitations': ['Raw annotation coverage is not fact sufficiency. Retained facts may convey a missing raw turn.',
                           'Abstention annotations do not prove the requested answer exists.',
                           'No new answer was generated or graded; this does not prove any answer fixed.',
                           'The source supplement does not repair reasoning, abstention prompts, disputed labels, or extraction lineage.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); result = inventory(args.run_root)
    with args.output.open('x') as stream: stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__': main()
