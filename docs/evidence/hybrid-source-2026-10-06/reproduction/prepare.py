"""Freeze runtime projection and separate evaluator labels; no product imports."""
import hashlib,json
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parent
RUN=ROOT.parents[1]/'runs/english-improvement-2026-09-11'
BASE=RUN/'full500-precision-measurement-package-001/package.json'
RAW=RUN/'integrity-audit-001/longmemeval_s_cleaned.json'
CACHE=RUN/'integrity-audit-001/corrected-cache-v2/longmemeval_s.json'
PROV=CACHE.parent/'variant-provenance.json'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
baseline=json.loads(BASE.read_text());raw=json.loads(RAW.read_text());cache=json.loads(CACHE.read_text());prov=json.loads(PROV.read_text())
mem={m['mid']:m for m in cache['memories']};queries={q['question_id']:q for q in cache['queries']};original={q['question_id']:q for q in raw}
lookup={(v['original_session_id'],v['session_date'],v['content_sha256']):mid for mid,v in prov.items()}
assert len(lookup)==len(prov)
sessions={};cases=[];labels=[];occurrences=0;future=0
for b in baseline['cases']:
 q=queries[b['id']];r=original[b['id']]
 assert b['question']==q['question']==r['question'] and b['date']==q['question_date']==r['question_date']
 assert sha(b['context'])==b['context_sha256']
 scope=[];flagged=[];gold=[]
 for sid,date,turns in zip(r['haystack_session_ids'],r['haystack_dates'],r['haystack_sessions'],strict=True):
  stamped=f'Session dated {date}\n'+'\n'.join(f"{t.get('role','?').capitalize()}: [{date}] "+(t.get('content','') or '') for t in turns)
  mid=lookup[(sid,date,sha(stamped))];scope.append(mid);m=mem[mid]
  assert m['content']==stamped and len(m['turns'])==len(turns)
  source_turns=[]
  for ti,(t,mt) in enumerate(zip(turns,m['turns'],strict=True)):
   assert mt['role']==t['role'] and mt['content']==f"[{date}] "+t['content']
   source_id=sha(f'{mid}:{ti}')[:24]
   source_turns.append(dict(id=source_id,position=ti,role=mt['role'],text=mt['content']))
   if t.get('has_answer'):
    flagged.append(dict(id=source_id,session=mid,position=ti,body=t['content']))
  sessions[mid]=dict(date=date,turns=source_turns)
  if sid in r['answer_session_ids']:gold.append(mid)
  occurrences+=1;future+=parse(date)>parse(b['date'])
 assert scope==q['scope_keys'] and set(gold)==set(q['gold_keys'])
 assert b['gold']==q['gold_answer']==str(r['answer'])
 cases.append(dict(id=b['id'],question=b['question'],date=b['date'],scope=scope,baseline=b['context'],baseline_sha256=b['context_sha256']))
 labels.append(dict(id=b['id'],gold_sessions=gold,flagged=flagged,type=b['type'],abstention=b['abst']))
assert len(cases)==500 and len({c['id'] for c in cases})==500
runtime=dict(schema='hybrid-source-audit-runtime-v1',sessions=sessions,cases=cases)
for filename,data in [('runtime.json',runtime),('labels.json',labels)]:
 with (ROOT/filename).open('x') as f:json.dump(data,f,ensure_ascii=False)
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [BASE,RAW,CACHE,PROV,ROOT/'runtime.json',ROOT/'labels.json']}
meta=dict(cases=500,scope_occurrences=occurrences,variant_sessions=len(sessions),future_occurrences= future,flagged_turns=sum(len(l['flagged']) for l in labels),gold_session_references=sum(len(l['gold_sessions']) for l in labels),unflagged=sum(not l['flagged'] for l in labels),input_sha256=files,paid_calls=0)
with (ROOT/'preparation.json').open('x') as f:json.dump(meta,f,indent=2)
print(json.dumps(meta,indent=2))
