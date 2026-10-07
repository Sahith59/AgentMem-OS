"""Measure zero lexical contributions and vocabulary loss; no new candidate."""
import json,hashlib,sys,importlib.util,warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'2026-10-06-hybrid-source-audit';REPO=ROOT.parents[2]/'AgentMem-OS';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
r=json.loads((OLD/'runtime.json').read_text());a={x['id']:x for x in json.loads((ROOT/'audit-v2.json').read_text())['rows']};packets={x['id']:x for x in map(json.loads,(ROOT/'packets-v2.jsonl').read_text().splitlines())};rows=[]
for c in r['cases']:
 ts=[t for mid in c['scope'] if parse(r['sessions'][mid]['date'])<=parse(c['date']) for t in r['sessions'][mid]['turns'] if t['text'].strip()];ids={t['id']:i for i,t in enumerate(ts)}
 if not ts:rows.append(dict(id=c['id'],historical_correct=a[c['id']]['historical_correct'],empty=True));continue
 vec=TfidfVectorizer(max_features=512,sublinear_tf=True,min_df=1);mat=vec.fit_transform([t['text'] for t in ts]);q=vec.transform([c['question']]);s=cosine_similarity(q,mat)[0];tokens=set(vec.build_analyzer()(c['question']));source_vocab=set(w for t in ts for w in vec.build_analyzer()(t['text']));dropped=sorted((tokens&source_vocab)-set(vec.vocabulary_));true_oov=sorted(tokens-source_vocab);selected=[h['source_id'] for h in packets[c['id']]['candidate']['report']['candidate_hits']];zeros=[tid for tid in selected if s[ids[tid]]==0]
 rows.append(dict(id=c['id'],historical_correct=a[c['id']]['historical_correct'],empty=False,passages=len(ts),positive_keyword_passages=int(np.sum(s>0)),zero_keyword_passages=int(np.sum(s==0)),query_vector_zero=q.nnz==0,query_terms_present_in_corpus_but_dropped_by_512_cap=dropped,query_terms_absent_from_corpus=true_oov,selected_novel_anchors=len(selected),selected_zero_keyword_anchors=zeros))
nonempty=[x for x in rows if not x['empty']]
def summary(rs):
 return dict(cases=len(rs),zero_query_cases=sum(x['query_vector_zero'] for x in rs),passages=sum(x['passages'] for x in rs),zero_keyword_passages=sum(x['zero_keyword_passages'] for x in rs),selected_anchors=sum(x['selected_novel_anchors'] for x in rs),selected_zero_keyword_anchors=sum(len(x['selected_zero_keyword_anchors']) for x in rs),cases_with_selected_zero_keyword_anchors=sum(bool(x['selected_zero_keyword_anchors']) for x in rs),cases_with_cap_dropped_query_terms=sum(bool(x['query_terms_present_in_corpus_but_dropped_by_512_cap']) for x in rs))
result=dict(schema='lexical-channel-diagnostic-v1',all500=summary(nonempty),historical77misses=summary([x for x in nonempty if not x['historical_correct']]),rows=rows,input_sha256={str(p):sha(p) for p in [OLD/'runtime.json',ROOT/'audit-v2.json',ROOT/'packets-v2.jsonl']},script_sha256=sha(Path(__file__)),new_candidates=0,paid_calls=0,limits=['Current legacy RRF assigns positive rank contribution even to zero lexical matches.','Zero keyword match does not imply a source is irrelevant: dense evidence can be relevant.','Query term drops can be generic words or meaningful terms; not all are answer failures.','This is diagnosis only, not a new ranker or accuracy result.'])
with (ROOT/'lexical-diagnosis.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
