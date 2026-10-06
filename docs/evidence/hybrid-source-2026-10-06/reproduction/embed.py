"""Local E5 embeddings only. Reads runtime projection, never evaluation labels."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import gc,hashlib,json,time,importlib.metadata
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'runtime.json').read_text());parse=lambda s:datetime.strptime(s,'%Y/%m/%d (%a) %H:%M')
texts=set()
for c in r['cases']:
 texts.add('query: '+c['question'])
 for mid in c['scope']:
  s=r['sessions'][mid]
  if parse(s['date'])<=parse(c['date']):texts.update('passage: '+t['text'] for t in s['turns'] if t['text'].strip())
items=sorted(texts,key=lambda x:(len(x),x));del r,texts;gc.collect()
keys=[hashlib.sha256(x.encode()).hexdigest() for x in items]
with (ROOT/'embedding-keys.json').open('x') as f:json.dump(keys,f)
torch.set_num_threads(4)
device='mps' if torch.backends.mps.is_available() else 'cpu'
model=SentenceTransformer(str(ROOT/'model'),device=device,local_files_only=True)
shape=(len(items),model.get_sentence_embedding_dimension())
assert not (ROOT/'embeddings.npy').exists()
vectors=np.lib.format.open_memmap(ROOT/'embeddings.npy',mode='w+',dtype='float32',shape=shape)
start=time.monotonic()
print(json.dumps(dict(device=device,items=len(items),dimensions=shape[1],max_seq_length=model.max_seq_length)),flush=True)
for offset in range(0,len(items),512):
 batch=items[offset:offset+512]
 values=model.encode(batch,normalize_embeddings=True,show_progress_bar=False,batch_size=64)
 assert values.shape==(len(batch),shape[1]) and np.isfinite(values).all()
 assert np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-4)
 vectors[offset:offset+len(batch)]=values;vectors.flush()
 progress=dict(done=offset+len(batch),total=len(items),elapsed_seconds=round(time.monotonic()-start,2))
 (ROOT/'embedding-progress.json').write_text(json.dumps(progress))
 if offset%4096==0:print(json.dumps(progress),flush=True)
meta=dict(progress,device=device,max_seq_length=model.max_seq_length,model=json.loads((ROOT/'model.json').read_text()),versions={k:importlib.metadata.version(k) for k in ['torch','sentence-transformers','numpy','scikit-learn']},paid_calls=0,runtime_sha256=hashlib.sha256((ROOT/'runtime.json').read_bytes()).hexdigest(),keys_sha256=hashlib.sha256((ROOT/'embedding-keys.json').read_bytes()).hexdigest(),vectors_sha256=hashlib.sha256((ROOT/'embeddings.npy').read_bytes()).hexdigest())
with (ROOT/'embeddings-complete.json').open('x') as f:json.dump(meta,f,indent=2)
print(json.dumps(meta),flush=True)
