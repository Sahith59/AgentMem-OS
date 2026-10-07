"""One frozen offline candidate/control; evaluate only after all packets exist."""
import hashlib,importlib.util,json,sys,warnings,time,math
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'2026-10-06-hybrid-source-audit';PREV=ROOT.parent/'2026-10-06-source-aware-selection';REPO=ROOT.parents[2]/'AgentMem-OS'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
spec=importlib.util.spec_from_file_location('agentmem_os',REPO/'__init__.py',submodule_search_locations=[str(REPO)]);module=importlib.util.module_from_spec(spec);sys.modules['agentmem_os']=module;spec.loader.exec_module(module)
from loguru import logger
logger.disable('agentmem_os')
from agentmem_os.llm.evidence_packet import SourceSnapshot,SourceTurn,digest
from agentmem_os.llm.hybrid_source_retrieval import rank
from agentmem_os.llm.source_aware_selection import select
prep=json.loads((OLD/'preparation.json').read_text());assert all(sha(Path(f))==h for f,h in prep['input_sha256'].items())
meta=json.loads((OLD/'embeddings-complete.json').read_text());assert sha(OLD/'runtime.json')==meta['runtime_sha256'];assert sha(OLD/'embedding-keys.json')==meta['keys_sha256'];assert sha(OLD/'embeddings.npy')==meta['vectors_sha256']
keys=json.loads((OLD/'embedding-keys.json').read_text());lookup={k:i for i,k in enumerate(keys)};vectors=np.load(OLD/'embeddings.npy',mmap_mode='r')
class CachedEncoder:
 def encode(self,texts,**kwargs):
  assert kwargs.get('normalize_embeddings') is True
  return vectors[[lookup[digest(t)] for t in texts]]
encoder=CachedEncoder();runtime=json.loads((OLD/'runtime.json').read_text());previous={x['id']:x for x in map(json.loads,(PREV/'packets-v2.jsonl').read_text().splitlines())}
code_files=['llm/source_presence.py','llm/source_aware_selection.py','llm/hybrid_source_retrieval.py','llm/multi_vector_retrieval.py','llm/evidence_packet.py','llm/context_assembler.py']
checkpoint=ROOT.parents[1]/'runs/english-improvement-2026-09-11/paid-full500-precision-measurement1-001/checkpoint.json'
freeze=dict(policy_sha256=sha(ROOT/'POLICY.md'),script_sha256=sha(Path(__file__)),code_sha256={f:sha(REPO/f) for f in code_files},input_sha256={str(f):sha(f) for f in [OLD/'runtime.json',OLD/'labels.json',PREV/'packets-v2.jsonl',PREV/'audit-v2.json',OLD/'embeddings-complete.json',OLD/'embedding-keys.json',OLD/'embeddings.npy',checkpoint]},paid_calls=0)
with (ROOT/'freeze.json').open('x') as f:json.dump(freeze,f,indent=2)
start=time.monotonic();outputs=[];warning_cases=0;warning_types=Counter();receipt_total=0;certificate_total=0
with (ROOT/'packets.jsonl').open('x') as output:
 for c in runtime['cases']:
  turns=tuple(SourceTurn(t['id'],mid,t['position'],t['role'],parse(runtime['sessions'][mid]['date']),t['text']) for mid in c['scope'] for t in runtime['sessions'][mid]['turns']);snap=SourceSnapshot(c['id'],turns);byid={t.id:t for t in turns};asof=parse(c['date'])
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');hits=rank(snap,c['question'],scope=c['id'],as_of=asof,encoder=encoder)
  warning_cases+=bool(caught);warning_types.update(str(w.message) for w in caught)
  row={'id':c['id']}
  for name,allocation in [('control','greedy'),('candidate','joint')]:
   use=True
   text,report=select(snap,hits,c['baseline'],scope=c['id'],as_of=asof,use_presence=use,allocation=allocation,char_budget=40000,extra_budget=4000,max_anchors=8,neighbor_turns=1)
   assert text.startswith(c['baseline']) and len(text)<=40000 and len(text)-len(c['baseline'])<=4000
   present=set()
   for r in report['presence']['receipts']:
    t=byid[r['id']];assert c['baseline'][r['start']:r['end']]==t.text and digest(t.text)==r['sha256'];assert t.observed_at<=asof;assert (t.session,t.position,t.role,t.observed_at.isoformat())==(r['session'],r['position'],r['role'],r['observed_at']);present.add(t.id)
   delivered=set()
   for r in report['receipts']:
    t=byid[r['id']];a=report['block_offset']+r['start'];b=report['block_offset']+r['end'];assert text[a:b]==t.text and digest(t.text)==r['sha256'];assert t.observed_at<=asof;assert (t.session,t.position,t.role,t.observed_at.isoformat())==(r['session'],r['position'],r['role'],r['observed_at']);assert t.id not in delivered;delivered.add(t.id)
   for b in report['bundles']:
    anchor=byid[b['anchor']];session=[t for t in turns if t.session==anchor.session];expected=[t.id for t in sorted(session,key=lambda t:t.position) if abs(t.position-anchor.position)<=1];assert b['required']==expected
    if b['status']=='admitted':assert set(b['required'])<=delivered | (present if use else set()) and not b['missing_positions']
   if use:assert not (delivered&present)
   receipt_total+=len(delivered);certificate_total+=len(present) if use else 0
   row[name]={'packet':text,'report':report}
  assert row['control']==previous[c['id']]['candidate']
  assert row['control']['report']['presence']==row['candidate']['report']['presence']
  cr=row['control']['report'];ar=row['candidate']['report'];assert cr['candidate_hits']==ar['candidate_hits']
  greedy_utility=math.fsum(h['score'] for h in cr['candidate_hits'] if h['source_id'] in cr['admitted_anchors'])
  assert ar['allocation']['utility']>=greedy_utility and ar['allocation']['rendered_block_chars']==len(row['candidate']['packet'])-ar['block_offset']
  outputs.append(row);output.write(json.dumps(row)+'\n');output.flush()
  if len(outputs)%50==0:print(json.dumps({'packed':len(outputs),'seconds':round(time.monotonic()-start,1)}),flush=True)
assert len(outputs)==500
assert freeze['code_sha256']=={f:sha(REPO/f) for f in code_files}
assert all(sha(Path(f))==h for f,h in freeze['input_sha256'].items())
# Evaluation starts after all packets. These labels never enter the selector.
labels={x['id']:x for x in json.loads((OLD/'labels.json').read_text())};jobs=json.loads(checkpoint.read_text())['jobs'];norm=lambda s:' '.join(s.split());rows=[]
for c,o in zip(runtime['cases'],outputs,strict=True):
 assert c['id']==o['id'];lab=labels[c['id']];byid={t['id']:t for mid in c['scope'] for t in runtime['sessions'][mid]['turns']};texts={name:norm(o[name]['packet']) for name in ['control','candidate']};texts['baseline']=norm(c['baseline']);sets={name:{t['id'] for t in lab['flagged'] if norm(t['body']) in text} for name,text in texts.items()};grade=jobs[c['id']+'/judge'];assert grade['status']=='complete'
 r=dict(id=c['id'],historical_correct=grade['correct'],abstention=lab['abstention'],type=lab['type'],flagged_total=len(lab['flagged']),flagged_before=sorted(sets['baseline']),flagged_control=sorted(sets['control']),flagged_candidate=sorted(sets['candidate']),gained=sorted(sets['candidate']-sets['baseline']),lost=sorted(sets['baseline']-sets['candidate']),gain_vs_control=sorted(sets['candidate']-sets['control']),loss_vs_control=sorted(sets['control']-sets['candidate']),presence_certificates=len(o['candidate']['report']['presence']['receipts']),presence_ambiguous=len(o['candidate']['report']['presence']['ambiguous']),presence_hazards=o['candidate']['report']['presence']['hazards'])
 for name in ['control','candidate']:
  report=o[name]['report'];r[name]=dict(extra_chars=len(o[name]['packet'])-len(c['baseline']),added_sources=len(report['receipts']),duplicate_stamped_text_sources=sum(norm(byid[v['id']]['text']) in texts['baseline'] for v in report['receipts']),bundle_statuses=dict(Counter(b['status'] for b in report['bundles'])),reused_sources=len({i for b in report['bundles'] if b['status']=='admitted' for i in b['baseline_reused']}))
 r['control_utility']=math.fsum(h['score'] for h in o['control']['report']['candidate_hits'] if h['source_id'] in o['control']['report']['admitted_anchors']);r['candidate_utility']=o['candidate']['report']['allocation']['utility'];r['packet_changed_vs_control']=o['control']['packet']!=o['candidate']['packet'];r['allocation']=o['candidate']['report']['allocation']
 rows.append(r)
def summary(rs):
 with_flags=[r for r in rs if r['flagged_total']]
 d=dict(cases=len(rs),packet_cases_changed=sum(r['packet_changed_vs_control'] for r in rs),utility_improved_cases=sum(r['candidate_utility']>r['control_utility'] for r in rs),cases_gaining=sum(bool(r['gained']) for r in rs),turns_gained=sum(len(r['gained']) for r in rs),turns_lost=sum(len(r['lost']) for r in rs),cases_gaining_vs_control=sum(bool(r['gain_vs_control']) for r in rs),cases_losing_vs_control=sum(bool(r['loss_vs_control']) for r in rs),turns_gained_vs_control=sum(len(r['gain_vs_control']) for r in rs),turns_lost_vs_control=sum(len(r['loss_vs_control']) for r in rs),flagged_denominator=len(with_flags),presence_certificates=sum(r['presence_certificates'] for r in rs),cases_with_presence_hazards=sum(bool(r['presence_hazards']) for r in rs))
 for name in ['control','candidate']:
  d[name]=dict(added_sources=sum(r[name]['added_sources'] for r in rs),duplicate_stamped_text_sources=sum(r[name]['duplicate_stamped_text_sources'] for r in rs),cases_with_additions=sum(bool(r[name]['extra_chars']) for r in rs),bundle_statuses=dict(sum((Counter(r[name]['bundle_statuses']) for r in rs),Counter())))
 for name in ['before','control','candidate']:
  d['flagged_any_'+name]=sum(bool(r['flagged_'+name]) for r in with_flags);d['flagged_all_'+name]=sum(len(r['flagged_'+name])==r['flagged_total'] for r in with_flags)
 return d
misses=[r for r in rows if not r['historical_correct']];ms=summary(misses);passed=ms['cases_gaining']>=10 and ms['cases_gaining_vs_control']-ms['cases_losing_vs_control']>=5 and not any(r['lost'] for r in rows)
result=dict(schema='atomic-allocation-audit-v1',status='OFFLINE_COVERAGE_ONLY',readiness='PASS_REQUIRES_SOURCE_REVIEW' if passed else 'FAIL',all500=summary(rows),historical77misses=ms,historical423correct=summary([r for r in rows if r['historical_correct']]),abstentions=summary([r for r in rows if r['abstention']]),receipt_total_verified=receipt_total,presence_certificates_verified=certificate_total,matched_greedy_control_reproduced_all500=True,same_top8_anchors_all500=True,numeric_warning_cases=warning_cases,numeric_warning_types=dict(warning_types),paid_calls=0,new_answers=0,answer_accuracy='NOT_MEASURED',packets_sha256=sha(ROOT/'packets.jsonl'),freeze_sha256=sha(ROOT/'freeze.json'),rows=rows)
with (ROOT/'audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
