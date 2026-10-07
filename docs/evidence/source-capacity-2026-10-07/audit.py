"""Frozen evaluator-only capacity audit; never creates candidate answer packets."""
import hashlib
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEM = ROOT.parents[1]
REPO = MEM.parent / 'AgentMem-OS'
OLD = ROOT.parent / '2026-10-06-hybrid-source-audit'
ATOMIC = ROOT.parent / '2026-10-07-atomic-allocation'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parse = lambda s: datetime.strptime(s, '%Y/%m/%d (%a) %H:%M')
norm = lambda s: ' '.join(s.split())
spec = importlib.util.spec_from_file_location('agentmem_os', REPO / '__init__.py', submodule_search_locations=[str(REPO)])
m = importlib.util.module_from_spec(spec)
sys.modules['agentmem_os'] = m
spec.loader.exec_module(m)
from agentmem_os.benchmarks.source_capacity_audit import measure_capacity
from agentmem_os.llm.evidence_packet import SourceSnapshot, SourceTurn

checkpoint = MEM / 'runs/english-improvement-2026-09-11/paid-full500-precision-measurement1-001/checkpoint.json'
inputs = [OLD / 'runtime.json', OLD / 'labels.json', ATOMIC / 'packets.jsonl', ATOMIC / 'audit.json', checkpoint]
freeze = dict(policy_sha256=sha(ROOT / 'POLICY.md'), script_sha256=sha(Path(__file__)),
              code_sha256={f:sha(REPO / f) for f in ['benchmarks/source_capacity_audit.py', 'llm/evidence_packet.py']},
              inputs_sha256={str(p):sha(p) for p in inputs}, paid_calls=0, candidate_packets_created=0)
with (ROOT / 'freeze.json').open('x') as f:
    json.dump(freeze, f, indent=2)
# Labels deliberately enter only this evaluator-only tool; there is no selector.
runtime = json.loads((OLD / 'runtime.json').read_text())
labels = {r['id']:r for r in json.loads((OLD / 'labels.json').read_text())}
packets = {r['id']:r for r in map(json.loads, (ATOMIC / 'packets.jsonl').read_text().splitlines())}
prior = {r['id']:r for r in json.loads((ATOMIC / 'audit.json').read_text())['rows']}
jobs = json.loads(checkpoint.read_text())['jobs']
rows = []
for c in runtime['cases']:
    turns = tuple(SourceTurn(t['id'], sid, t['position'], t['role'], parse(runtime['sessions'][sid]['date']), t['text'])
                  for sid in c['scope'] for t in runtime['sessions'][sid]['turns'])
    snap = SourceSnapshot(c['id'], turns)
    byid = {t.id:t for t in turns}
    report = packets[c['id']]['control']['report']
    present = {r['id'] for r in report['presence']['receipts']}
    missing = {r['id'] for r in labels[c['id']]['flagged'] if norm(r['body']) not in norm(c['baseline'])}
    assert missing == {r['id'] for r in labels[c['id']]['flagged']} - set(prior[c['id']]['flagged_before'])
    assert not (missing & present)
    # Validate existing presence certificates before granting free reuse.
    for r in report['presence']['receipts']:
        t = byid[r['id']]
        assert c['baseline'][r['start']:r['end']] == t.text
        assert t.observed_at <= parse(c['date']) and r['sha256'] == hashlib.sha256(t.text.encode()).hexdigest()
    anchors = [h['source_id'] for h in report['candidate_hits']]
    available = min(4000, 40000 - len(c['baseline']))
    assert available >= 0
    item = dict(id=c['id'], historical_correct=jobs[c['id'] + '/judge']['correct'],
                annotated_turns=len(labels[c['id']]['flagged']), missing=sorted(missing),
                available_chars=available, baseline_chars=len(c['baseline']), fixed_anchor_ids=anchors)
    for name, pool in [('fixed_top8', anchors), ('unrestricted', None)]:
        r = measure_capacity(snap, scope=c['id'], as_of=parse(c['date']), present_ids=present,
                             targets=missing, anchor_ids=pool)
        for field in ['any_cover', 'all_cover']:
            r[field + '_fits'] = r[field] is not None and r[field]['chars'] <= available
        item[name] = r
    duplicates = []
    for sid in sorted(present):
        count = c['baseline'].count(byid[sid].text)
        if count > 1:
            duplicates.append(dict(id=sid, occurrences=count, chars=len(byid[sid].text)))
    item['duplicate_body_screening'] = dict(
        status='OPTIMISTIC_SUBSTRING_SCREEN_NOT_CERTIFIED_REMOVAL', sources=duplicates,
        extra_occurrences=sum(d['occurrences'] - 1 for d in duplicates),
        raw_chars_upper_bound=sum((d['occurrences'] - 1) * d['chars'] for d in duplicates))
    item['sections'] = {match[1]:len(match[2]) for match in re.finditer(r'<\[([A-Z][A-Z ]*)\]>\n(.*?)\n</\[\1\]>', c['baseline'], re.S)}
    rows.append(item)

def summarize(items):
    affected = [x for x in items if x['missing']]
    result = dict(cases=len(items), no_annotations=sum(x['annotated_turns'] == 0 for x in items),
                  all_annotated_bodies_already_present=sum(x['annotated_turns'] > 0 and not x['missing'] for x in items),
                  cases_with_missing=len(affected), missing_turns=sum(len(x['missing']) for x in affected))
    for name in ['fixed_top8', 'unrestricted']:
        result[name] = dict(any_missing_delivery_fits=sum(x[name]['any_cover_fits'] for x in affected),
                            all_missing_delivery_fits=sum(x[name]['all_cover_fits'] for x in affected),
                            any_unreachable=sum(x[name]['any_cover'] is None for x in affected),
                            any_over_budget=sum(x[name]['any_cover'] is not None and not x[name]['any_cover_fits'] for x in affected),
                            all_unreachable=sum(x[name]['all_cover'] is None for x in affected),
                            all_over_budget=sum(x[name]['all_cover'] is not None and not x[name]['all_cover_fits'] for x in affected))
    result['duplicate_body_screening'] = dict(cases=sum(bool(x['duplicate_body_screening']['sources']) for x in items),
        extra_occurrences=sum(x['duplicate_body_screening']['extra_occurrences'] for x in items),
        raw_chars_upper_bound=sum(x['duplicate_body_screening']['raw_chars_upper_bound'] for x in items))
    result['baseline_chars_median'] = statistics.median(x['baseline_chars'] for x in items)
    result['baseline_at_least_36k'] = sum(x['baseline_chars'] >= 36000 for x in items)
    result['section_chars_median_across_all_cases'] = {
        name:statistics.median(x['sections'].get(name, 0) for x in items)
        for name in sorted({n for x in items for n in x['sections']})}
    return result

assert len(rows) == 500
assert sum(not r['historical_correct'] for r in rows) == 77
assert all(sha(Path(p)) == h for p,h in freeze['inputs_sha256'].items())
result = dict(schema='source-capacity-diagnosis-v1', status='EVALUATOR_ONLY_NO_RUNTIME_CANDIDATE',
    all500=summarize(rows), historical77misses=summarize([r for r in rows if not r['historical_correct']]),
    historical423correct=summarize([r for r in rows if r['historical_correct']]),
    paid_calls=0, new_answers=0, answer_accuracy='NOT_MEASURED', candidate_readiness='NOT_APPLICABLE',
    freeze_sha256=sha(ROOT / 'freeze.json'), rows=rows)
with (ROOT / 'audit.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))
