"""Offline feasibility diagnostic for the frozen, append-only focus experiment.

This reads saved artifacts only. It makes no model calls and uses outcomes only
for post-run grouping, never for selecting evidence or changing runtime inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from .evidence_focus import HEADER, FocusInput, SourceTurn, validate_input


def capacity(value: FocusInput, *, char_cap: int = 40_000,
             focus_cap: int = 4_000, turn_cap: int = 8) -> dict:
    validate_input(value)
    if any(type(n) is not int or n < 0 for n in (char_cap, focus_cap, turn_cap)):
        raise ValueError('Budgets must be nonnegative integers')
    headroom = char_cap - len(value.packet)
    if headroom < 0:
        raise ValueError('Baseline exceeds context cap')
    sizes = sorted(len(f'[{t.id} | {t.role} | observed {t.observed_at}]\n'
                       + t.text + '\n') for t in value.turns)
    budget = min(focus_cap, headroom)
    used, count = len(HEADER), 0
    for size in sizes[:turn_cap]:
        if used + size > budget:
            break
        used += size
        count += 1
    return dict(turns_available=len(sizes), headroom=headroom,
                effective_focus_cap=budget,
                shortest_focus_chars=len(HEADER) + sizes[0] if sizes else None,
                maximum_shortest_turns_fit=count,
                any_turns_that_cannot_fit_alone=sum(len(HEADER) + s > budget for s in sizes))


def summarize(rows: list[dict]) -> dict:
    return dict(cases=len(rows),
        median_headroom=statistics.median(r['headroom'] for r in rows) if rows else None,
        cannot_fit_even_shortest=sum(r['maximum_shortest_turns_fit'] == 0 for r in rows),
        cannot_fit_two_shortest=sum(r['maximum_shortest_turns_fit'] < 2 for r in rows),
        cannot_fit_eight_shortest=sum(r['maximum_shortest_turns_fit'] < 8 for r in rows),
        has_some_unfit_turns=sum(r['any_turns_that_cannot_fit_alone'] > 0 for r in rows))


def audit(pool_path: Path, bridge_path: Path, *, expected_cases: int = 500) -> dict:
    pool = json.loads(pool_path.read_text())
    bridge = json.loads(bridge_path.read_text())
    if (pool.get('cases') != expected_cases or bridge.get('total') != expected_cases
            or len(pool['rows']) != expected_cases or len(bridge['rows']) != expected_cases
            or bridge.get('status') != 'PASS_INTEGRITY'):
        raise ValueError('Incomplete or unverified population')
    record = pool['sources']['package']
    original_path = Path(record['path'])
    original_bytes = original_path.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest() != record['sha256']:
        raise ValueError('Original package hash mismatch')
    original_cases = json.loads(original_bytes)['cases']
    original = {c['id']: c for c in original_cases}
    if len(original) != expected_cases or len(original_cases) != expected_cases:
        raise ValueError('Original population mismatch')
    grades = {r['id']: r for r in bridge['rows']}
    if len(grades) != len(bridge['rows']) or len({r['id'] for r in pool['rows']}) != len(pool['rows']):
        raise ValueError('Duplicate case IDs')
    if {r['id'] for r in pool['rows']} != set(grades) or set(grades) != set(original):
        raise ValueError('Population mismatch')
    rows = []
    for item in pool['rows']:
        raw = Path(item['input_file']['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item['input_file']['sha256']:
            raise ValueError('Input file hash mismatch')
        inp = json.loads(raw)
        value = FocusInput(inp['question'], inp['question_date'], inp['packet'],
                          tuple(SourceTurn(**t) for t in inp['turns']))
        source = original[item['id']]
        if (value.question != source['question'] or value.question != grades[item['id']]['question']
                or value.question_date != source['date'] or value.packet != source['context']):
            raise ValueError('Runtime projection mismatch')
        digest = hashlib.sha256(value.packet.encode()).hexdigest()
        if digest != item['original_context_sha256'] or digest != grades[item['id']]['context_sha256']:
            raise ValueError('Context hash mismatch')
        grade = grades[item['id']]['new_grade']
        if type(grade) is not bool:
            raise ValueError('Grade must be boolean')
        rows.append(dict(id=item['id'], terra_correct=grade, **capacity(value)))
    return dict(status='OFFLINE_STRUCTURAL_CAPACITY_ONLY',
                all_cases=summarize(rows),
                judge_misses=summarize([r for r in rows if not r['terra_correct']]),
                judge_correct=summarize([r for r in rows if r['terra_correct']]),
                rows=rows, model_calls=0, new_answers=0, accuracy_lift='NOT_MEASURED',
                limits='Shortest-turn capacity is an optimistic length bound, not semantic recall. '
                       'Eight turns are a cap, not a requirement. Non-fit does not prove an answer failure.',
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (pool_path, bridge_path, original_path, Path(__file__),
                                   Path(__file__).with_name('evidence_focus.py'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pool', type=Path)
    parser.add_argument('bridge', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.pool, args.bridge)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'rows', 'sources'}}, indent=2))


if __name__ == '__main__':
    main()
