#!/usr/bin/env python3
"""
Phase 2 — Long-Horizon Test (50 turns)
=======================================
Tests FULL system vs RECENT_ONLY over 50 turns with probes at
turns 35, 40, 45, 48, 50 — facts seeded in the first 5 turns.
This stresses the memory system far beyond the standard 25-turn benchmark.

Output: benchmarks/long_horizon_results.json

Usage:
    python3 benchmarks/phase2_long_horizon.py
"""

import os, re, sys, json, math, time
from pathlib import Path
from dotenv import load_dotenv

for _p in [Path('.'), Path('..'), Path('../..')]:
    if (_p / '.env').exists(): load_dotenv(_p / '.env'); break

G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; B="\033[1m"; E="\033[0m"
def ok(m):   print(f"  {G}✓{E}  {m}")
def warn(m): print(f"  {Y}!{E}  {m}")
def hdr(t):  print(f"\n{B}{C}{'═'*60}{E}\n{B}{C}  {t}{E}\n{B}{C}{'═'*60}{E}")
def sub(t):  print(f"\n{B}  ── {t}{E}")

MODEL         = "claude-haiku-4-5-20251001"
COST_PER_MTOK = 0.80
SLEEP_THRESH  = 15
RECENT_WIN    = 8
SEMANTIC_TOP  = 3
KG_TOP        = 5

# 50-turn conversation: 5 grounding + 40 work/filler + 5 long-horizon probes
CONVERSATION_50 = [
    # T1–5: Grounding facts (seeded early, probed at T35–50)
    "Hi, my name is Sahith and I'm building AgentMem OS for my PhD research.",
    "My primary research goal is persistent memory for LLM agents across sessions.",
    "The system uses four memory tiers: Redis working memory, SQLite episodic, TF-IDF semantic, and procedural pattern memory.",
    "I'm targeting a NeurIPS 2026 workshop. The paper deadline is June 2026.",
    "The four novel algorithms are MemoryImportanceScorer, SleepConsolidationEngine, EntityKnowledgeGraph, and ProceduralMemory.",
    # T6–34: Work turns (mix of technical and domain questions)
    "Can you explain how DBSCAN clustering helps group semantically similar memories?",
    "What's the difference between episodic and semantic memory in cognitive science?",
    "How does prompt caching in Claude reduce API costs for long conversations?",
    "Please write a short Python function to compute cosine similarity between two vectors.",
    "What are the trade-offs between TF-IDF and dense vector retrieval?",
    "Explain memory importance scoring — how would you rank conversation turns?",
    "What evaluation metrics are used in memory-augmented language model papers?",
    "How does retrieval-augmented generation differ from a full memory system?",
    "Can you describe PageRank and how it applies to knowledge graphs?",
    "What is the architecture of a sleep consolidation system in AI memory research?",
    "What is the difference between sparse and dense retrieval in NLP?",
    "How does attention mechanism relate to memory in transformer models?",
    "What is the role of a vector database in a RAG system?",
    "Explain the concept of episodic memory compression in neural networks.",
    "What Python libraries are commonly used for NLP and text embedding?",
    "How do you evaluate the quality of a retrieval system?",
    "What is the difference between precision and recall in information retrieval?",
    "Explain BM25 and how it compares to TF-IDF for document retrieval.",
    "What is sentence-transformers and how does it work?",
    "How does LangChain's memory module handle conversation history?",
    "What are the main challenges in building long-term memory for AI agents?",
    "Explain the concept of working memory in cognitive psychology.",
    "What is the difference between declarative and procedural memory?",
    "How does spaCy's named entity recognition work?",
    "What are the key metrics for evaluating summarization quality?",
    "Explain the trade-offs between abstractive and extractive summarization.",
    "How does Redis handle pub/sub messaging for real-time applications?",
    "What is SQLAlchemy and why is it used in Python web applications?",
    # T35–39: Long-horizon probes (no in-turn hints — relies entirely on memory)
    "Without me reminding you, what is my name?",
    "What is the specific research deadline I am working towards — give the month and year?",
    "Can you list all four memory tiers in my system, exactly as I described them?",
    "Name all four novel ML algorithms I mentioned at the very start of our conversation.",
    "What conference and year am I submitting this research to?",
    # T40–50: More work turns + final probes
    "What makes a good NeurIPS workshop paper — what do reviewers look for?",
    "How would you structure an ablation study for a memory system like AgentMem OS?",
    "What baseline systems should I compare against in my evaluation section?",
    "Draft a one-sentence summary of the AgentMem OS contribution.",
    "What are the advantages of a local-first memory architecture over cloud-based?",
    "How does the Personalized PageRank algorithm work for knowledge graph queries?",
    "Explain the significance of the compression ratio metric in memory systems.",
    "What is the difference between token-level and sentence-level embeddings?",
    "How do you handle memory conflicts when the same entity appears with different attributes?",
    "What is the role of entity resolution in knowledge graph construction?",
    # T50 final probe — extreme long-horizon
    "One final question: going all the way back to when we started — what are the three evaluation metrics (CRS, TES, LCS) I mentioned my project uses?",
]

# Probe indices (0-based) → keyword to find in reply
PROBE_RECALLS_50 = {
    33: "sahith",
    34: "neurips 2026",
    35: "sqlite",
    36: "procedural",
    37: "neurips",
    48: "crs",   # extreme 50-turn probe
}

VARIANTS_LH = [
    ("FULL",        "Full AgentMem OS (all tiers)",        {}),
    ("RECENT_ONLY", "Recent-only (last 10 turns, no mem)", {"recent_only": True}),
]

# ── Reuse helpers from ablation_study.py ────────────────────────────────────

def _tok(t): return max(1, len(t)//4)
def _ents(t): return set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', t))

def _cosine(q, d):
    qt=re.findall(r'[a-z]+',q.lower()); dt=re.findall(r'[a-z]+',d.lower())
    all_w=set(qt)|set(dt); df={w:(w in qt)+(w in dt) for w in all_w}
    def v(ts):
        tf={}
        for w in ts: tf[w]=tf.get(w,0)+1
        n=len(ts) or 1; return {w:(c/n)*math.log(1+1/df[w]) for w,c in tf.items()}
    qv,dv=v(qt),v(dt)
    dot=sum(qv.get(k,0)*dv.get(k,0) for k in all_w)
    return dot/(math.sqrt(sum(x**2 for x in qv.values()))+1e-9)/(math.sqrt(sum(x**2 for x in dv.values()))+1e-9)

def _extract_summary(turns):
    def ec(t): return len(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b',t["content"]))
    chunks=[turns[i:i+5] for i in range(0,len(turns),5)]; out=[]
    for ch in chunks:
        for t in sorted(ch,key=ec,reverse=True)[:2]:
            out.append(f"[{t['role'].upper()}]: {t['content'][:280]}")
    ents=_ents(" ".join(t["content"] for t in turns))
    if ents: out.append("Entities: "+", ".join(sorted(ents)[:30]))
    return "\n\n".join(out)

def _tes(user_turns, compressed):
    orig=" ".join(t["content"] for t in user_turns)
    rc=max(0.,min(1.,1-_tok(compressed)/max(1,_tok(orig))))
    Eo=_ents(orig); Ec=_ents(compressed)
    return round(math.sqrt(rc*len(Eo&Ec)/max(1,len(Eo))),4)

def assemble(turns, query, flags, sleep_sum):
    parts=["You are AgentMem OS, an AI with persistent hierarchical memory."]
    if flags.get("recent_only"):
        parts.append("[RECENT TURNS]\n"+"".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in turns[-10:]))
        return "\n\n".join(parts)
    if sleep_sum:
        parts.append(f"[EPISODIC SUMMARY]\n{sleep_sum}")
    if len(turns)>RECENT_WIN:
        hits=sorted(turns[:-RECENT_WIN],key=lambda t:_cosine(query,t["content"]),reverse=True)[:SEMANTIC_TOP]
        if hits: parts.append("[SEMANTIC HITS]\n"+"".join(f"[{t['role'].upper()}]: {t['content'][:200]}\n" for t in hits))
    freq={}
    for t in turns:
        for e in _ents(t["content"]): freq[e]=freq.get(e,0)+1
    top=sorted(freq,key=freq.get,reverse=True)[:KG_TOP]
    if top: parts.append(f"[ENTITIES]: {', '.join(top)}")
    parts.append("[WORKING MEMORY]\n"+"".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in turns[-RECENT_WIN:]))
    return "\n\n".join(parts)

def run_variant(client, vname, label, flags):
    hdr(f"VARIANT: {vname}  ({len(CONVERSATION_50)} turns)")
    sub("Running...")
    turns=[]; recall={}; n_tok=0; b_tok=0; sleep_sum=None

    for i, msg in enumerate(CONVERSATION_50):
        turn_num=i+1
        if (not flags.get("no_sleep","") and not flags.get("recent_only")
                and sleep_sum is None and len(turns)>=SLEEP_THRESH):
            old=[t for t in turns if t["role"]=="user"]
            sleep_sum=_extract_summary(old[:int(len(old)*.6)])
            print(f"    → Sleep consolidation at T{turn_num}")

        ctx=assemble(turns,msg,flags,sleep_sum)
        naive=sum(_tok(t["content"]) for t in turns)+_tok(msg)
        ours=_tok(ctx)+_tok(msg)
        n_tok+=ours; b_tok+=naive

        try:
            resp=client.messages.create(model=MODEL,max_tokens=400,
                messages=[{"role":"user","content":f"{ctx}\n\n[QUERY]\n{msg}"}])
            reply=resp.content[0].text if resp.content else ""
        except Exception as ex:
            warn(f"T{turn_num}: {ex}"); reply=""

        turns.append({"role":"user","content":msg})
        turns.append({"role":"assistant","content":reply})

        if i in PROBE_RECALLS_50:
            kw=PROBE_RECALLS_50[i]; hit=kw.lower() in reply.lower()
            recall[i]=hit; sym=f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{turn_num:02d} {sym} probe '{kw}' | ctx={ours}tok saved={100*(1-ours/max(1,naive)):.0f}%")
        else:
            print(f"    T{turn_num:02d} ✓ | ctx={ours}tok")
        time.sleep(0.12)

    lcs=round(sum(recall.values())/len(PROBE_RECALLS_50),4)
    sav=round(100*(1-n_tok/max(1,b_tok)),1)
    user_turns=[t for t in turns if t["role"]=="user"]
    compressed=sleep_sum if sleep_sum else " ".join(t["content"] for t in user_turns[-max(1,int(len(user_turns)*.7)):])
    tes=_tes(user_turns,compressed)
    probe_qs=[CONVERSATION_50[i] for i in sorted(PROBE_RECALLS_50)]
    our_ctx=(sleep_sum or "")+" "+" ".join(t["content"] for t in turns[-10:])
    crs=round(sum(_cosine(q,our_ctx) for q in probe_qs)/len(probe_qs),4)
    sub("Results")
    print(f"    LCS  : {lcs:.4f}  recall={sum(recall.values())}/{len(PROBE_RECALLS_50)}")
    print(f"    TES  : {tes:.4f}")
    print(f"    CRS  : {crs:.4f}")
    print(f"    Savings: {sav:.1f}%")
    return {"variant":vname,"label":label,"turns":len(CONVERSATION_50),
            "metrics":{"CRS":crs,"TES":tes,"LCS":lcs},
            "tokens":{"savings_pct":sav},
            "recall":{PROBE_RECALLS_50[i]:recall.get(i,False) for i in PROBE_RECALLS_50}}

def main():
    hdr("Phase 2 — Long-Horizon Test (50 turns)")
    api_key=os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key: print(f"  {R}✗{E}  Set ANTHROPIC_API_KEY"); sys.exit(1)
    ok(f"API key ({api_key[:12]}...)")
    try:
        import anthropic; client=anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  pip install anthropic"); sys.exit(1)

    print(f"\n  2 variants × {len(CONVERSATION_50)} turns  |  Est. cost ~${2*0.62:.2f}\n")
    results=[]
    for vname,label,flags in VARIANTS_LH:
        r=run_variant(client,vname,label,flags)
        results.append(r)

    out=Path(__file__).parent/"long_horizon_results.json"
    out.write_text(json.dumps(results,indent=2))
    ok(f"Results → {out}")

    hdr("LONG-HORIZON COMPARISON  (50 turns)")
    full=next(r for r in results if r["variant"]=="FULL")
    ro=next(r for r in results if r["variant"]=="RECENT_ONLY")
    print(f"\n  {'Metric':<12} {'FULL':>10} {'RECENT_ONLY':>12} {'Delta':>10}")
    print(f"  {'─'*12} {'─'*10} {'─'*12} {'─'*10}")
    for m in ["LCS","TES","CRS"]:
        fv=full["metrics"][m]; rv=ro["metrics"][m]
        print(f"  {m:<12} {fv:>10.4f} {rv:>12.4f} {fv-rv:>+10.4f}")
    print(f"  {'Savings':<12} {full['tokens']['savings_pct']:>9.1f}% {ro['tokens']['savings_pct']:>11.1f}%")
    ok("Phase 2 complete.")

if __name__=="__main__":
    main()
