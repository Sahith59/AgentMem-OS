import hashlib,json,warnings
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'runtime.json').read_text());keys=json.loads((ROOT/'embedding-keys.json').read_text());indices={k:i for i,k in enumerate(keys)};v=np.load(ROOT/'embeddings.npy',mmap_mode='r')
sha=lambda t:hashlib.sha256(t.encode()).hexdigest();parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
rows=[]
for c in r['cases']:
 texts=['passage: '+t['text'] for mid in c['scope'] if parse(r['sessions'][mid]['date'])<=parse(c['date']) for t in r['sessions'][mid]['turns'] if t['text'].strip()]
 matrix=v[[indices[sha(t)] for t in texts]];q=v[indices[sha('query: '+c['question'])]]
 assert np.isfinite(matrix).all() and np.isfinite(q).all()
 assert np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4) and abs(np.linalg.norm(q)-1)<1e-4
 with warnings.catch_warnings(record=True) as captured:
  warnings.simplefilter('always');scores=matrix@q
 reference=np.einsum('ij,j->i',matrix.astype(np.float64),q.astype(np.float64))
 error=float(np.max(np.abs(scores-reference))) if len(scores) else 0.0
 rows.append(dict(id=c['id'],finite=bool(np.isfinite(scores).all()),max_absolute_error=error,warnings=[str(w.message) for w in captured],sources=len(texts),min_score=float(scores.min()) if len(scores) else None,max_score=float(scores.max()) if len(scores) else None))
 assert rows[-1]['finite'] and error<1e-6
result=dict(status='PASS',cases=len(rows),empty_scopes=sum(r['sources']==0 for r in rows),cases_with_warnings=sum(bool(r['warnings']) for r in rows),max_absolute_error=max(r['max_absolute_error'] for r in rows),all_finite=all(r['finite'] for r in rows),rows=rows,interpretation='Every actual query/source dot product finite and agrees within1e-6 with direct float64 contraction. Numerical warning cause not proven; no nonfinite/overflow result observed. Original warnings preserved.')
with (ROOT/'numeric-audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
