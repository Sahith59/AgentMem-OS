"""Structural reviewer-input replay; NOT a semantic selector or answer experiment."""
import hashlib,importlib.util,json,socket,sys,time
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
MEM=HERE.parents[1];REPO=MEM.parent/'AgentMem-OS';PRIOR=HERE.parent/'2026-10-06-hybrid-source-audit'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(path,obj):
 with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')
spec=importlib.util.spec_from_file_location('agentmem_os',REPO/'__init__.py',submodule_search_locations=[str(REPO)])
m=importlib.util.module_from_spec(spec);sys.modules['agentmem_os']=m;spec.loader.exec_module(m)
from agentmem_os.llm.evidence_packet import SourceSnapshot,SourceTurn,RetrievalHit,digest,eligible_sources
from agentmem_os.llm.source_unit_contract import QuoteSpan,SourceUnit
from agentmem_os.llm.source_unit_compiler import snapshot_digest
from agentmem_os.benchmarks.source_unit_review import build_review_dossier,serialize_dossier,REVIEW_QUESTIONS
parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
def deny(*a,**k):raise RuntimeError('Network forbidden in review-dossier audit')
socket.create_connection=deny;socket.socket.connect=deny
runtime=json.loads((PRIOR/'runtime.json').read_text())
assert sha(PRIOR/'runtime.json')=='d913dd05f2d28791264f1c7d5df431b8c6e05ea2554096db0f889a8a658fcedb'
prior={r['id']:r for r in map(json.loads,(PRIOR/'packets.jsonl').open())}
assert len(prior)==len(runtime['cases'])==500
selected=sorted(runtime['cases'],key=lambda c:digest('source-unit-review-v1:'+c['question']))[:3]
review_ids=[c['id'] for c in selected]
code=['benchmarks/source_unit_review.py','llm/source_unit_compiler.py','llm/source_unit_contract.py','llm/evidence_packet.py','tests/test_source_unit_review.py']
freeze={'policy_sha256':sha(HERE/'POLICY.md'),'script_sha256':sha(Path(__file__)),'code_sha256':{x:sha(REPO/x) for x in code},'runtime_sha256':sha(PRIOR/'runtime.json'),'prior_hits_sha256':sha(PRIOR/'packets.jsonl'),'proposal_policy':'Existing first8 hybrid hits, whole-turn mechanical examples; no automatic semantic selection','unit_preview_char_budget':4000,'review_dossier_budget':'UNBUDGETED_OFFLINE_ONLY','review_sample_ids':review_ids,'paid_calls':0,'evaluator_labels_loaded':False}
dump(HERE/'freeze.json',freeze);out=HERE/'dossiers';out.mkdir()
start=time.monotonic();rows=[];total_sources=total_proposals=refused=previews=future=0
for c in runtime['cases']:
 turns=tuple(SourceTurn(t['id'],sid,t['position'],t['role'],parse(runtime['sessions'][sid]['date']),t['text']) for sid in c['scope'] for t in runtime['sessions'][sid]['turns'])
 snapshot=SourceSnapshot(c['id'],turns);asof=parse(c['date']);binding=snapshot_digest(snapshot)
 eligible=eligible_sources(snapshot,scope=c['id'],as_of=asof);byid={t.id:t for t in turns}
 hits=tuple(RetrievalHit(**h) for h in prior[c['id']]['report']['candidate_hits'][:8]);units=[]
 for rank,h in enumerate(hits):
  t=byid[h.source_id];span=QuoteSpan('anchor',t.id,digest(t.text),0,len(t.text),digest(t.text))
  units.append(SourceUnit('review-'+str(rank),binding,digest(c['question']),asof,('anchor',),(span,)))
 d=build_review_dossier(snapshot,hits,tuple(units),question=c['question'],baseline=c['baseline'],scope=c['id'],as_of=asof,unit_char_budget=4000)
 encoded=serialize_dossier(d);decoded=json.loads(encoded);assert decoded==d
 assert len(encoded.splitlines())==1
 assert d['baseline']['text']==c['baseline'] and d['baseline']['sha256']==c['baseline_sha256']
 assert d['snapshot_sha256']==binding and d['answer_path_eligible'] is False and d['paid_run_ready'] is False
 assert [s['id'] for s in d['sources']]==[t.id for t in eligible]
 for t,s in zip(eligible,d['sources'],strict=True):
  assert s==dict(id=t.id,session=t.session,position=t.position,role=t.role,observed_at=t.observed_at.isoformat(),text=t.text,sha256=digest(t.text))
 assert d['future_source_ids']==[t.id for t in turns if t.observed_at>asof]
 assert len(d['proposals'])==len(hits)
 for h,r in zip(hits,d['ranked_hits'],strict=True):
  assert (r['source_id'],r['source_sha256'],r['score'],r['tie_order'])==(h.source_id,h.source_sha256,float(h.score),h.tie_order)
 for u in d['proposals']:
  assert u['review']==dict.fromkeys(REVIEW_QUESTIONS,'UNREVIEWED') and len(u['review'])==13
  rr=u['report'];assert rr['answer_path_eligible'] is False and rr['semantic_completeness']=='NOT_CERTIFIED'
  assert digest(u['preview'])==rr['preview_sha256']
  refused+=rr['status']=='REFUSED';previews+=rr['status']=='PREVIEW_ONLY'
 body={k:v for k,v in d.items() if k!='dossier_sha256'};assert digest(serialize_dossier(body))==d['dossier_sha256']
 path=out/(c['id']+'.json')
 with path.open('x') as f:f.write(encoded+'\n')
 total_sources+=len(eligible);total_proposals+=len(hits);future+=len(d['future_source_ids'])
 rows.append(dict(id=c['id'],sources=len(eligible),future_excluded=len(d['future_source_ids']),proposals=len(hits),dossier_sha256=d['dossier_sha256'],file_sha256=sha(path),file_chars=len(encoded)+1,review_sample=c['id'] in review_ids))
 if len(rows)%100==0:print(json.dumps(dict(done=len(rows),elapsed=round(time.monotonic()-start,1))),flush=True)
assert freeze['code_sha256']=={x:sha(REPO/x) for x in code}
dump(HERE/'audit.json',dict(status='PASS_REVIEW_INPUT_INTEGRITY_ONLY',cases=len(rows),eligible_source_occurrences=total_sources,future_source_occurrences_excluded=future,whole_hit_example_units=total_proposals,compiler_preview_only=previews,compiler_refused=refused,review_sample_ids=review_ids,all_semantic_statuses='UNREVIEWED',new_answer_packets=0,paid_calls=0,new_answers=0,semantic_proposer='NOT_IMPLEMENTED',answer_accuracy='NOT_MEASURED',rows=rows,elapsed_seconds=time.monotonic()-start))
print('DONE',json.dumps(dict(cases=500,sources=total_sources,proposals=total_proposals,refused=refused,sample=review_ids)),flush=True)
