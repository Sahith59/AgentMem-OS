"""Post-generation, label-free mechanism checks; no answer or grade imports."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2] / 'AgentMem-OS'
spec = importlib.util.spec_from_file_location('agentmem_os', REPO / '__init__.py',
                                            submodule_search_locations=[str(REPO)])
module = importlib.util.module_from_spec(spec)
sys.modules['agentmem_os'] = module
spec.loader.exec_module(module)
from agentmem_os.llm.token_counter import TokenCounter

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
counter = TokenCounter()
phase = json.loads((ROOT / 'phase1.json').read_text())
assert phase['candidate_complete'] == phase['legacy_exact'] == 500
old_path, new_path = ROOT / 'legacy-traces.jsonl', ROOT / 'candidate-traces.jsonl'
assert sha(old_path) == phase['legacy_trace_sha256']
assert sha(new_path) == phase['candidate_trace_sha256']
old = [json.loads(line) for line in old_path.open()]
new = [json.loads(line) for line in new_path.open()]
rows = []
for a, b in zip(old, new, strict=True):
    assert a['id'] == b['id'] and a['raw_ranked_chunks'] == b['raw_ranked_chunks']
    chunks = a['raw_ranked_chunks'] or []
    report = b['raw_packing_receipt']
    active = report.get('policy') == 'whole_rank_v1'
    root = chunks[0] if chunks else ''
    single = '<[SEMANTIC MEMORY]>\n' + root + '\n</[SEMANTIC MEMORY]>'
    budget = b['raw_budget_tokens']
    fits = bool(root.strip()) and budget is not None and (
        len(single) <= budget * 4 and counter.count(single) <= budget)
    legacy_final = (ROOT / 'contexts/legacy' / (a['id'] + '.txt')).read_text()
    candidate_final = (ROOT / 'contexts/candidate' / (a['id'] + '.txt')).read_text()
    row = dict(id=a['id'], policy=report.get('policy'), ranked_chunks=len(chunks),
               rank0_fits_as_whole_section=fits,
               legacy_raw_rank0_complete=bool(root) and root in a['raw_section'],
               candidate_raw_rank0_complete=bool(root) and root in b['raw_section'],
               legacy_final_rank0_complete=bool(root) and root in legacy_final,
               candidate_final_rank0_complete=bool(root) and root in candidate_final)
    if active:
        assert len(b['raw_section']) <= budget * 4
        assert counter.count(b['raw_section']) <= budget
        for r in report['receipts']:
            assert b['raw_section'][r['start']:r['end']] == chunks[r['index']]
        if fits:
            assert 0 in report['admitted_indices'] and row['candidate_raw_rank0_complete']
            assert row['candidate_final_rank0_complete']
    rows.append(row)
active = [r for r in rows if r['policy'] == 'whole_rank_v1']
affected = [r for r in active if r['rank0_fits_as_whole_section']
            and not r['legacy_raw_rank0_complete']]
result = dict(status='PASS_MECHANICAL_INVARIANT_ONLY', cases=len(rows),
              active_cases=len(active), legacy_fallback_cases=len(rows) - len(active),
              fitting_rank0_cases=sum(r['rank0_fits_as_whole_section'] for r in active),
              legacy_raw_missing_fitting_rank0_cases=len(affected),
              candidate_recovers_fitting_rank0_cases=sum(r['candidate_raw_rank0_complete']
                                                       for r in affected),
              newly_delivered_rank0_in_final_cases=sum(not r['legacy_final_rank0_complete']
                                                       for r in affected),
              affected_rows=affected, rows=rows,
              input_sha256={p.name: sha(p) for p in [old_path, new_path, ROOT / 'phase1.json']},
              script_sha256=sha(Path(__file__)), paid_calls=0, new_answers=0,
              limitation='Rank is retrieval preference, not semantic truth. Whole input text '
              'presence is not certified original-source identity or an answer gain.')
with (ROOT / 'mechanism-audit.json').open('x') as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: v for k, v in result.items() if k not in {'rows', 'affected_rows'}}, indent=2))
