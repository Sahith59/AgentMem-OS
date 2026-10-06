"""Compare actual pre-change search code against new code with the same local vectors."""
import hashlib,importlib.util,json,subprocess,sys,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]/'AgentMem-OS'
spec=importlib.util.spec_from_file_location('agentmem_os',REPO/'__init__.py',submodule_search_locations=[str(REPO)])
m=importlib.util.module_from_spec(spec);sys.modules['agentmem_os']=m;spec.loader.exec_module(m)
from agentmem_os.llm.multi_vector_retrieval import MultiVectorRetriever
from loguru import logger
logger.remove()
old=subprocess.check_output(['git','show','a7e44e3:llm/multi_vector_retrieval.py'],cwd=REPO,text=True)
legacy=types.ModuleType('legacy');exec(compile(old,'legacy_snapshot','exec'),legacy.__dict__)
texts=['alpha topic. A longer sentence mentioning the small library. Another detail.', 'beta topic. A longer sentence about a journey. Additional detail.', 'gamma event. A completely different event.', 'delta followup. alpha detail and a contradiction.', 'epsilon preference. beta discussion and a followup.']
queries=['alpha','beta','absent']
class Encoder:
 def __init__(self,tied):
  rng=np.random.default_rng(2301)
  values=np.ones((len(texts)+len(queries),4)) if tied else rng.normal(size=(len(texts)+len(queries),4))
  values/=np.linalg.norm(values,axis=1,keepdims=True)
  self.values=dict(zip(['passage: '+t for t in texts]+['query: '+q for q in queries],values,strict=True))
 def encode(self,items,**kwargs):return np.array([self.values[t] for t in items])
comparisons=0
for tied in [False,True]:
 encoder=Encoder(tied)
 sys.modules['agentmem_os.db.entity_aliases']=types.SimpleNamespace(get_shared_encoder=lambda:encoder)
 for neighbors in [0,1,2]:
  for snippet in [0,35,100]:
   previous=legacy.MultiVectorRetriever(context_turns=neighbors,snippet_chars=snippet)
   current=MultiVectorRetriever(context_turns=neighbors,snippet_chars=snippet,encoder=encoder)
   previous.index(texts);current.index(texts)
   for deep in [None,0,1,4]:
    for query in queries:
     assert previous.search(query,top_k=3,deep_hits=deep)==current.search(query,top_k=3,deep_hits=deep)
     comparisons+=1
result=dict(status='PASS',comparisons=comparisons,reference_commit='a7e44e3',reference_source_sha256=hashlib.sha256(old.encode()).hexdigest(),new_source_sha256=hashlib.sha256((REPO/'llm/multi_vector_retrieval.py').read_bytes()).hexdigest(),paid_calls=0,scope='Deterministic local vector parity across tied/non-tied scores, 3 neighbor widths, 3 snippet caps, 4 deep policies, 3 queries. Not accuracy or all-input equivalence proof.')
with (ROOT/'legacy-parity.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
