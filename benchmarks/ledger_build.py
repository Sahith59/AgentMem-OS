"""Build question-BLIND ledgers (full text) for the scopes behind the
43 autopsied counting failures. Local llama, $0. Output:
benchmarks/ledgers_f22.json {question: ledger_text} and
benchmarks/f22_questions.json [questions]. The model never sees any
question; ledgers are per-scope, derived from facts only."""
import json, re, sqlite3, sys, urllib.request
from pathlib import Path
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
CORPUS = HERE / "extracted_memories" / "gate_c_facts_luna.db"
PROMPT = open(HERE / "ledger_prototype.py").read().split('PROMPT = """')[1].split('"""')[0]

def llm(prompt):
    req = urllib.request.Request("http://localhost:11434/api/generate",
        data=json.dumps({"model": "llama3.1:latest", "prompt": prompt, "stream": False,
                         "format": "json", "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 2000}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.loads(r.read())["response"])

autopsy = json.load(open(HERE / "autopsy_luna_p1_87.json"))
counting = [r for r in autopsy if r["is_counting"]]
from corpus_loaders import load_longmemeval
ds = load_longmemeval(n_queries=500, seed=42, split="s")
by_q = {q.question: q for q in ds.queries}
con = sqlite3.connect(f"file:{CORPUS}?mode=ro", uri=True)
out, qs = {}, []
for i, r in enumerate(counting):
    q = by_q.get(r["question"])
    if not q: continue
    rows = []
    for sid in q.scope_keys:
        rows += [f"[{d or 'undated'}] {t}" for t, d in con.execute(
            "SELECT fact_text, t_occurred FROM semantic_facts WHERE source_session_id=? AND fact_type IN ('event','state')", (sid,)).fetchall()]
    ledgers = []
    for s in range(0, len(rows), 120):
        try: ledgers += llm(PROMPT.format(facts="\n".join(rows[s:s+120]))).get("ledgers", [])
        except Exception as e: print("chunk err", e, flush=True)
    lines = []
    for l in ledgers:
        if isinstance(l.get("count"), (int, float)) and l.get("count", 0) >= 2:
            dates = ", ".join(str(d) for d in (l.get("dates") or [])[:12])
            lines.append(f"- {l.get('topic')}: {int(l['count'])} distinct instances" + (f" ({dates})" if dates else ""))
    out[q.question] = "\n".join(lines) if lines else "- (no recurring topics found)"
    qs.append(q.question)
    print(f"  {i+1}/{len(counting)} {len(lines)} ledger lines", flush=True)
json.dump(out, open(HERE / "ledgers_f22.json", "w"), indent=1, ensure_ascii=False)
json.dump(qs, open(HERE / "f22_questions.json", "w"), indent=1, ensure_ascii=False)
print("wrote ledgers_f22.json + f22_questions.json")
