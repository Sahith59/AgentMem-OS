"""Structural stress audit only: no source selector, labels, answers or paid calls."""
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]/'AgentMem-OS'
INPUT=ROOT.parent/'2026-10-06-hybrid-source-audit/runtime.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
spec=importlib.util.spec_from_file_location('agentmem_os',REPO/'__init__.py',submodule_search_locations=[str(REPO)])
m=importlib.util.module_from_spec(spec);sys.modules['agentmem_os']=m;spec.loader.exec_module(m)
from agentmem_os.llm.evidence_packet import SourceTurn,SourceSnapshot,digest,eligible_sources
from agentmem_os.llm.source_unit_contract import QuoteSpan,SourceUnit,MAX_SPANS
from agentmem_os.llm.source_unit_compiler import compile_unit,snapshot_digest

FILES=['llm/source_unit_contract.py','llm/source_unit_compiler.py','llm/evidence_packet.py','tests/test_source_unit_compiler.py']
freeze=dict(policy_sha256=sha(ROOT/'POLICY.md'),script_sha256=sha(Path(__file__)),
            code_sha256={p:sha(REPO/p) for p in FILES},runtime_sha256=sha(INPUT),
            selection='ALL eligible original turns in batches of128; FULL original bodies',
            purpose='STRUCTURE_ONLY; NOT candidate retrieval or compaction; labels never loaded',
            stress_budget_chars=100_000_000,answer_budget_change=False,paid_calls=0)
assert freeze['runtime_sha256']=='d913dd05f2d28791264f1c7d5df431b8c6e05ea2554096db0f889a8a658fcedb'
with (ROOT/'freeze.json').open('x') as f:json.dump(freeze,f,indent=2)
runtime=json.loads(INPUT.read_text())
rows=[];total_turns=total_batches=future_probes=budget_probes=empty_cases=0
start=time.monotonic()
with (ROOT/'batches.jsonl').open('x') as log:
 for case in runtime['cases']:
  turns=tuple(SourceTurn(t['id'],sid,t['position'],t['role'],parse(runtime['sessions'][sid]['date']),t['text']) for sid in case['scope'] for t in runtime['sessions'][sid]['turns'])
  snap=SourceSnapshot(case['id'],turns);asof=parse(case['date']);eligible=eligible_sources(snap,scope=case['id'],as_of=asof)
  binding=snapshot_digest(snap);byid={t.id:t for t in turns};batches=0
  def make(ts,index):
   spans=tuple(QuoteSpan(t.id,t.id,digest(t.text),0,len(t.text),digest(t.text)) for t in ts)
   return SourceUnit('structural-audit-'+str(index),binding,digest(case['question']),asof,tuple(t.id for t in ts),spans)
  for offset in range(0,len(eligible),MAX_SPANS):
   chosen=eligible[offset:offset+MAX_SPANS];unit=make(chosen,offset)
   preview,report=compile_unit(snap,unit,question=case['question'],scope=case['id'],as_of=asof,char_budget=100_000_000)
   assert report['status']=='PREVIEW_ONLY' and report['answer_path_eligible'] is False and report['semantic_completeness']=='NOT_CERTIFIED'
   assert report['used_chars']==len(preview)==report['required_chars'] and not report['omissions']
   lines=[json.loads(x) for x in preview.splitlines()]
   meta={x['source_id']:x for x in lines if x.get('kind')=='source'}
   quotes={x['source_id']:x for x in lines if x.get('kind')=='quote'}
   assert len(lines)==1+2*len(chosen) and len(report['receipts'])==len(chosen)
   for t in chosen:
    assert quotes[t.id]['text']==t.text and quotes[t.id]['start']==0 and quotes[t.id]['end']==len(t.text)
    assert meta[t.id]==dict(kind='source',source_id=t.id,session=t.session,position=t.position,role=t.role,observed_at=t.observed_at.isoformat(),source_sha256=digest(t.text))
   for r in report['receipts']:
    t=byid[r['source_id']];decoded=json.loads(preview[r['payload_start']:r['payload_end']])
    assert decoded[r['decoded_start']:r['decoded_end']]==t.text and r['quote_sha256']==digest(t.text)
    assert r['start']==r['decoded_start']==0 and r['end']==r['decoded_end']==len(t.text)
   if offset==0:
    rejected,rr=compile_unit(snap,unit,question=case['question'],scope=case['id'],as_of=asof,char_budget=len(preview)-1)
    assert rejected=='' and rr['reasons']==['budget_nonfit'] and not rr['receipts'] and not rr['emitted_ranges'] and not rr['omissions']
    budget_probes+=1
   log.write(json.dumps(dict(case_id=case['id'],offset=offset,snapshot_sha256=binding,source_ids=[t.id for t in chosen],preview_sha256=digest(preview),report_sha256=digest(json.dumps(report,sort_keys=True)),used_chars=len(preview)),sort_keys=True)+'\n')
   total_turns+=len(chosen);total_batches+=1;batches+=1
  future=next((t for t in turns if t.observed_at>asof),None)
  if future:
   rejected,rr=compile_unit(snap,make((future,),'future'),question=case['question'],scope=case['id'],as_of=asof,char_budget=100_000_000)
   assert rejected=='' and rr['reasons']==['future_source'] and not rr['receipts']
   future_probes+=1
  empty_cases+=not eligible
  rows.append(dict(id=case['id'],snapshot_sha256=binding,eligible_turns=len(eligible),scoped_turns=len(turns),batches=batches,future_probe=bool(future)))
  if len(rows)%50==0:print(json.dumps(dict(cases=len(rows),turns=total_turns,seconds=round(time.monotonic()-start,1))),flush=True)
assert len(rows)==500 and all(sha(REPO/p)==h for p,h in freeze['code_sha256'].items()) and sha(INPUT)==freeze['runtime_sha256']
result=dict(status='PASS_STRUCTURAL_ONLY',cases=500,eligible_turns_verified=total_turns,batches=total_batches,
            empty_eligible_cases=empty_cases,budget_refusals_verified=budget_probes,future_refusals_verified=future_probes,
            freeze_sha256=sha(ROOT/'freeze.json'),batches_sha256=sha(ROOT/'batches.jsonl'),
            semantic_completeness='NOT_CERTIFIED',source_selection_readiness='NOT_ESTABLISHED',answer_accuracy='NOT_MEASURED',
            paid_calls=0,new_answers=0,new_embeddings=0,large_stress_budget_not_a_benchmark_budget=True,
            elapsed_seconds=round(time.monotonic()-start,3),rows=rows)
with (ROOT/'audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
