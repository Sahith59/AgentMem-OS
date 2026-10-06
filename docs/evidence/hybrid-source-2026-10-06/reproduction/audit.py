"""Offline generation then separately evaluated coverage. No SDK or model loader."""
import hashlib,importlib.util,importlib.metadata,json,sys,time
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]/'AgentMem-OS'
spec=importlib.util.spec_from_file_location('agentmem_os',REPO/'__init__.py',submodule_search_locations=[str(REPO)])
module=importlib.util.module_from_spec(spec);sys.modules['agentmem_os']=module;spec.loader.exec_module(module)
from loguru import logger
logger.disable('agentmem_os')
from agentmem_os.llm.context_assembler import ContextAssembler
from agentmem_os.llm.evidence_packet import SourceSnapshot,SourceTurn,digest
parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
file_sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
preparation=json.loads((ROOT/'preparation.json').read_text())
for path,expected in preparation['input_sha256'].items():
 assert file_sha(Path(path))==expected, path
checkpoint=ROOT.parents[1]/'runs/english-improvement-2026-09-11/paid-full500-precision-measurement1-001/checkpoint.json'
checkpoint_sha=file_sha(checkpoint)

meta=json.loads((ROOT/'embeddings-complete.json').read_text())
assert importlib.metadata.version('scikit-learn')==meta['versions']['scikit-learn']
assert importlib.metadata.version('numpy')==meta['versions']['numpy']
assert meta['runtime_sha256']==file_sha(ROOT/'runtime.json')
assert meta['keys_sha256']==file_sha(ROOT/'embedding-keys.json')
assert meta['vectors_sha256']==file_sha(ROOT/'embeddings.npy')
keys=json.loads((ROOT/'embedding-keys.json').read_text());lookup={k:i for i,k in enumerate(keys)}
assert len(keys)==len(lookup)
vectors=np.load(ROOT/'embeddings.npy',mmap_mode='r');assert vectors.shape==(len(keys),384)
class CachedEncoder:
 def encode(self,texts,**kwargs):
  assert kwargs.get('normalize_embeddings') is True
  return vectors[[lookup[digest(text)] for text in texts]]
encoder=CachedEncoder()
runtime=json.loads((ROOT/'runtime.json').read_text());start=time.monotonic();rows=[]
code_files=['llm/multi_vector_retrieval.py','llm/evidence_packet.py','llm/hybrid_source_retrieval.py','llm/context_assembler.py']
code={name:file_sha(REPO/name) for name in code_files}
with (ROOT/'audit-freeze.json').open('x') as f:json.dump(dict(code_sha256=code,plan_sha256=file_sha(ROOT/'PLAN.md'),script_sha256=file_sha(Path(__file__)),runtime_sha256=meta['runtime_sha256'],labels_sha256=file_sha(ROOT/'labels.json'),checkpoint_sha256=checkpoint_sha,preparation_sha256=file_sha(ROOT/'preparation.json'),model=meta['model'],model_files_sha256=file_sha(ROOT/'model-files.json')),f,indent=2)
# Runtime phase: no evaluator labels, reference answers or historical grades loaded.
with (ROOT/'packets.jsonl').open('x') as output:
 for c in runtime['cases']:
  turns=tuple(SourceTurn(t['id'],mid,t['position'],t['role'],parse(runtime['sessions'][mid]['date']),t['text']) for mid in c['scope'] for t in runtime['sessions'][mid]['turns'])
  snapshot=SourceSnapshot(c['id'],turns)
  text,report=ContextAssembler.assemble_hybrid_source_packet(snapshot,c['question'],c['baseline'],scope=c['id'],as_of=parse(c['date']),encoder=encoder,char_budget=40000,extra_budget=4000,max_anchors=8,neighbor_turns=1)
  assert text.startswith(c['baseline']) and digest(c['baseline'])==c['baseline_sha256']
  assert len(text)<=40000 and len(text)-len(c['baseline'])<=4000
  by_id={t.id:t for t in turns};seen=set()
  for receipt in report['receipts']:
   t=by_id[receipt['id']];a=report['block_offset']+receipt['start'];b=report['block_offset']+receipt['end']
   assert t.id not in seen;seen.add(t.id)
   assert text[a:b]==t.text and digest(text[a:b])==receipt['sha256']
   assert t.observed_at<=parse(c['date']) and t.session in c['scope']
   assert (t.session,t.position,t.role,t.observed_at.isoformat())==(receipt['session'],receipt['position'],receipt['role'],receipt['observed_at'])
  assert not (seen & {v['id'] for v in report['omissions']})
  row=dict(id=c['id'],packet=text,report=report)
  output.write(json.dumps(row)+'\n');output.flush();rows.append(row)
  if len(rows)%50==0:print(json.dumps(dict(packed=len(rows),elapsed=round(time.monotonic()-start,1))),flush=True)
assert code=={name:file_sha(REPO/name) for name in code_files}
# Evaluator phase starts only after all500 packets exist.
assert file_sha(ROOT/'labels.json')==preparation['input_sha256'][str(ROOT/'labels.json')]
assert file_sha(checkpoint)==checkpoint_sha
labels={r['id']:r for r in json.loads((ROOT/'labels.json').read_text())}
checkpoint=ROOT.parents[1]/'runs/english-improvement-2026-09-11/paid-full500-precision-measurement1-001/checkpoint.json'
grades=json.loads(checkpoint.read_text())['jobs']
normalize=lambda s:' '.join(s.split())
evaluated=[]
for c,out in zip(runtime['cases'],rows,strict=True):
 assert c['id']==out['id'];label=labels[c['id']]
 baseline=normalize(c['baseline']);candidate=normalize(out['packet']);report=out['report']
 before={t['id'] for t in label['flagged'] if normalize(t['body']) in baseline}
 after={t['id'] for t in label['flagged'] if normalize(t['body']) in candidate}
 sessions_before={mid for mid in label['gold_sessions'] if any(normalize(t['text']) in baseline for t in runtime['sessions'][mid]['turns'])}
 sessions_after={mid for mid in label['gold_sessions'] if any(normalize(t['text']) in candidate for t in runtime['sessions'][mid]['turns'])}
 by_id={t['id']:dict(t,session=mid,date=runtime['sessions'][mid]['date']) for mid in c['scope'] for t in runtime['sessions'][mid]['turns']}
 received={t['id'] for t in report['receipts']};ranked={t['source_id'] for t in report['candidate_hits']}
 missing=[]
 for t in label['flagged']:
  if t['id'] not in after:
   reasons=[o['reason'] for o in report['omissions'] if o['id']==t['id']]
   reason='future_ineligible' if parse(by_id[t['id']]['date'])>parse(c['date']) else ('+'.join(sorted(set(reasons))) if reasons else 'not_top8_or_neighbor')
   missing.append(dict(id=t['id'],reason=reason))
 duplicated=[t['id'] for t in report['receipts'] if normalize(by_id[t['id']]['text']) in baseline]
 grade=grades[c['id']+'/judge'];assert grade['status']=='complete' and type(grade['correct']) is bool
 evaluated.append(dict(id=c['id'],type=label['type'],historical_correct=grade['correct'],abstention=label['abstention'],flagged_total=len(label['flagged']),flagged_before=sorted(before),flagged_after=sorted(after),gained=sorted(after-before),lost=sorted(before-after),gold_sessions_total=len(label['gold_sessions']),gold_sessions_before=sorted(sessions_before),gold_sessions_after=sorted(sessions_after),appended_sources=len(received),appended_anchors=sum(t['kind']=='anchor' for t in report['receipts']),appended_neighbors=sum(t['kind']=='context' for t in report['receipts']),duplicate_stamped_text_sources=duplicated,extra_chars=len(out['packet'])-len(c['baseline']),omissions=report['omissions'],missing=missing))
assert sum(x['historical_correct'] for x in evaluated)==423

def summarize(rs):
 flagged=[r for r in rs if r['flagged_total']]
 return dict(cases=len(rs),cases_with_additions=sum(r['extra_chars']>0 for r in rs),added_sources=sum(r['appended_sources'] for r in rs),duplicate_stamped_text_sources=sum(len(r['duplicate_stamped_text_sources']) for r in rs),gained_flagged_turns=sum(len(r['gained']) for r in rs),lost_flagged_turns=sum(len(r['lost']) for r in rs),cases_gaining_flagged=sum(bool(r['gained']) for r in rs),cases_losing_flagged=sum(bool(r['lost']) for r in rs),flagged_denominator=len(flagged),flagged_any_before=sum(bool(r['flagged_before']) for r in flagged),flagged_any_after=sum(bool(r['flagged_after']) for r in flagged),flagged_all_before=sum(len(r['flagged_before'])==r['flagged_total'] for r in flagged),flagged_all_after=sum(len(r['flagged_after'])==r['flagged_total'] for r in flagged),gold_any_before=sum(bool(r['gold_sessions_before']) for r in rs),gold_any_after=sum(bool(r['gold_sessions_after']) for r in rs),gold_all_before=sum(len(r['gold_sessions_before'])==r['gold_sessions_total'] for r in rs),gold_all_after=sum(len(r['gold_sessions_after'])==r['gold_sessions_total'] for r in rs))
misses=[r for r in evaluated if not r['historical_correct']]
passed=sum(bool(r['gained']) for r in misses)>=10 and not any(r['lost'] for r in evaluated)
result=dict(status='OFFLINE_COVERAGE_ONLY',readiness_signal='PASS_REQUIRES_REVIEW' if passed else 'FAIL',all500=summarize(evaluated),historical77misses=summarize(misses),historical423correct=summarize([r for r in evaluated if r['historical_correct']]),abstentions=summarize([r for r in evaluated if r['abstention']]),paid_calls=0,new_answers=0,answer_accuracy='NOT_MEASURED',source_receipts_verified=sum(r['appended_sources'] for r in evaluated),input_sha256=json.loads((ROOT/'preparation.json').read_text())['input_sha256'],packets_sha256=file_sha(ROOT/'packets.jsonl'),checkpoint_sha256=file_sha(checkpoint),code_sha256=code,rows=evaluated)
with (ROOT/'audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','input_sha256','code_sha256']},indent=2))
