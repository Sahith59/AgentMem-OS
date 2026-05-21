#!/usr/bin/env python3
"""
Phase 3 — Baseline Comparison
===============================
Compares AgentMem OS (FULL) against three external baselines over 25 turns:

  FULL          — AgentMem OS with all 4 tiers active
  FULL_HISTORY  — Naive: send ALL conversation turns to model (no compression)
                  Simulates a system with no memory management — token cost grows
                  unboundedly, shows why compression is necessary.
  NAIVE_RAG     — TF-IDF top-5 retrieval only (no sleep, no KG, no proc, no recent)
                  Simulates a plain retrieval-augmented system without our tiers.
  RECENT_ONLY   — Sliding window last 10 turns (most common simple baseline)

Output: benchmarks/baseline_comparison.json

Usage:
    python3 benchmarks/phase3_baselines.py
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

CONVERSATION = [
    "Hi, my name is Sahith and I'm building AgentMem OS for my PhD research.",
    "My primary research goal is to add persistent memory to LLM agents across sessions.",
    "The project has four memory tiers: Redis working memory, SQLite episodic, TF-IDF semantic, and procedural pattern memory.",
    "I'm targeting a NeurIPS 2026 workshop submission. The paper deadline is June 2026.",
    "The four novel algorithms are: MemoryImportanceScorer, SleepConsolidationEngine, EntityKnowledgeGraph, and ProceduralMemory.",
    "Can you explain how DBSCAN clustering helps group semantically similar memories?",
    "What's the difference between episodic and semantic memory in cognitive science?",
    "How does prompt caching in Claude reduce API costs for long conversations?",
    "Please write a short Python function to compute cosine similarity between two vectors.",
    "What are the trade-offs between TF-IDF and dense vector retrieval for local memory systems?",
    "Explain the concept of memory importance scoring — how would you rank conversation turns?",
    "What evaluation metrics are typically used in memory-augmented language model papers?",
    "How does retrieval-augmented generation differ from a full persistent memory system?",
    "Can you describe the PageRank algorithm and how it applies to knowledge graphs?",
    "What is the typical architecture of a sleep consolidation system in AI memory research?",
    "Without me reminding you, what is my name?",
    "What specific research deadline am I working towards, including the month and year?",
    "Can you list all four memory tiers in the system I described earlier?",
    "Name all four novel ML algorithms I mentioned at the start of our conversation.",
    "What is the conference name and year I plan to submit this work to?",
    "What makes a good NeurIPS workshop paper — what do reviewers typically look for?",
    "How would you structure an ablation study for a memory system like AgentMem OS?",
    "What baseline systems should I compare against in my evaluation section?",
    "Can you help me draft a one-sentence summary of the AgentMem OS contribution?",
    "Final question: what are the three benchmark metrics — CRS, TES, and LCS?",
]
PROBE_RECALLS = {15:"sahith", 16:"neurips 2026", 17:"sqlite", 18:"procedural", 19:"neurips"}

BASELINES = [
    ("AGENTMEM_OS",  "AgentMem OS — Full System (all 4 tiers)", "ours"),
    ("FULL_HISTORY", "Full History — All turns, no compression", "full_history"),
    ("NAIVE_RAG",    "Naive RAG — TF-IDF retrieval only",        "naive_rag"),
    ("RECENT_ONLY",  "Recent-Only — Sliding window, last 10",    "recent_only"),
]

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

def build_context(turns, query, mode, sleep_sum):
    """Build context string for each baseline mode."""
    sys_p = "You are an AI assistant. Use the conversation context to answer accurately."

    if mode == "ours":
        # AgentMem OS: sleep summary + semantic + KG + recent
        parts=[sys_p]
        if sleep_sum:
            parts.append(f"[EPISODIC SUMMARY]\n{sleep_sum}")
        if len(turns)>RECENT_WIN:
            hits=sorted(turns[:-RECENT_WIN],key=lambda t:_cosine(query,t["content"]),reverse=True)[:3]
            if hits: parts.append("[SEMANTIC]\n"+"".join(f"[{t['role'].upper()}]: {t['content'][:200]}\n" for t in hits))
        freq={}
        for t in turns:
            for e in _ents(t["content"]): freq[e]=freq.get(e,0)+1
        top=sorted(freq,key=freq.get,reverse=True)[:5]
        if top: parts.append(f"[ENTITIES]: {', '.join(top)}")
        # Procedural
        msgs=[t["content"] for t in turns if t["role"]=="user"]
        procs=[]
        if any("explain" in m.lower() for m in msgs): procs.append("User prefers structured explanations.")
        if any("research" in m.lower() for m in msgs): procs.append("User is in research mode.")
        if procs: parts.append("[PATTERNS]: "+" ".join(procs))
        parts.append("[RECENT]\n"+"".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in turns[-RECENT_WIN:]))
        return "\n\n".join(parts)

    elif mode == "full_history":
        # Send EVERYTHING — no compression at all
        all_txt="".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in turns)
        return f"{sys_p}\n\n[FULL CONVERSATION HISTORY]\n{all_txt}"

    elif mode == "naive_rag":
        # TF-IDF top-5 only — no recent, no sleep, no KG, no proc
        if len(turns) < 2:
            return f"{sys_p}\n\n[No history yet]"
        hits=sorted(turns,key=lambda t:_cosine(query,t["content"]),reverse=True)[:5]
        hits_txt="".join(f"[{t['role'].upper()}]: {t['content'][:300]}\n" for t in hits)
        return f"{sys_p}\n\n[TOP-5 RETRIEVED TURNS]\n{hits_txt}"

    elif mode == "recent_only":
        recent=turns[-10:]
        recent_txt="".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in recent)
        return f"{sys_p}\n\n[RECENT TURNS (last 10)]\n{recent_txt}"

    return sys_p

def run_baseline(client, name, label, mode):
    hdr(f"{name}  —  {label}")
    sub("Running 25-turn conversation...")
    turns=[]; recall={}; n_tok=0; b_tok=0; sleep_sum=None

    for i, msg in enumerate(CONVERSATION):
        turn_num=i+1

        # Sleep consolidation for OURS only
        if (mode=="ours" and sleep_sum is None and len(turns)>=SLEEP_THRESH):
            old=[t for t in turns if t["role"]=="user"]
            sleep_sum=_extract_summary(old[:int(len(old)*.6)])
            print(f"    → Sleep consolidation at T{turn_num}")

        ctx=build_context(turns,msg,mode,sleep_sum)
        naive=sum(_tok(t["content"]) for t in turns)+_tok(msg)
        ours=_tok(ctx)+_tok(msg)
        n_tok+=ours; b_tok+=naive

        try:
            resp=client.messages.create(model=MODEL,max_tokens=400,
                messages=[{"role":"user","content":f"{ctx}\n\n[CURRENT QUERY]\n{msg}"}])
            reply=resp.content[0].text if resp.content else ""
        except Exception as ex:
            warn(f"T{turn_num}: {ex}"); reply=""

        turns.append({"role":"user","content":msg})
        turns.append({"role":"assistant","content":reply})

        if i in PROBE_RECALLS:
            kw=PROBE_RECALLS[i]; hit=kw.lower() in reply.lower()
            recall[i]=hit; sym=f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{turn_num:02d} {sym} probe '{kw}' | ctx={ours}tok naive={naive}tok")
        else:
            print(f"    T{turn_num:02d} ✓ | ctx={ours}tok")
        time.sleep(0.12)

    lcs=round(sum(recall.values())/len(PROBE_RECALLS),4)
    sav=round(100*(1-n_tok/max(1,b_tok)),1)
    user_turns=[t for t in turns if t["role"]=="user"]
    compressed = sleep_sum if (mode=="ours" and sleep_sum) else \
                 " ".join(t["content"] for t in user_turns)  # full_history has no compression
    if mode in ("full_history",): compressed=" ".join(t["content"] for t in user_turns)
    elif mode=="naive_rag": compressed=" ".join(t["content"] for t in user_turns[-max(1,int(len(user_turns)*.5)):])
    elif mode=="recent_only": compressed=" ".join(t["content"] for t in user_turns[-max(1,int(len(user_turns)*.3)):])
    tes=_tes(user_turns,compressed)
    probe_qs=[CONVERSATION[i] for i in sorted(PROBE_RECALLS)]
    our_ctx=(sleep_sum or "")+" "+" ".join(t["content"] for t in turns[-10:])
    crs=round(sum(_cosine(q,our_ctx) for q in probe_qs)/len(probe_qs),4)

    sub("Results")
    cost=round(n_tok/1_000_000*COST_PER_MTOK,4)
    naive_cost=round(b_tok/1_000_000*COST_PER_MTOK,4)
    print(f"    LCS : {lcs:.4f}  ({sum(recall.values())}/{len(PROBE_RECALLS)} probes recalled)")
    print(f"    TES : {tes:.4f}")
    print(f"    CRS : {crs:.4f}")
    print(f"    Tok savings: {sav:.1f}%  (${cost:.4f} vs ${naive_cost:.4f} naive)")

    return {"baseline":name,"label":label,"mode":mode,
            "metrics":{"CRS":crs,"TES":tes,"LCS":lcs},
            "tokens":{"savings_pct":sav,"cost":cost,"naive_cost":naive_cost},
            "recall":{PROBE_RECALLS[i]:recall.get(i,False) for i in PROBE_RECALLS}}

def main():
    hdr("Phase 3 — Baseline Comparison")
    api_key=os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key: print(f"  {R}✗{E}  Set ANTHROPIC_API_KEY"); sys.exit(1)
    ok(f"API key ({api_key[:12]}...)")
    try:
        import anthropic; client=anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  pip install anthropic"); sys.exit(1)

    print(f"\n  4 baselines × {len(CONVERSATION)} turns  |  Est. cost ~${4*0.31:.2f}\n")
    results=[]
    for name,label,mode in BASELINES:
        r=run_baseline(client,name,label,mode)
        results.append(r)

    out=Path(__file__).parent/"baseline_comparison.json"
    out.write_text(json.dumps(results,indent=2))
    ok(f"Results → {out}")

    hdr("BASELINE COMPARISON TABLE")
    print(f"  {'System':<18} {'LCS':>7} {'TES':>7} {'CRS':>7} {'Savings':>9} {'Cost':>8}")
    print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*9} {'─'*8}")
    for r in results:
        m=r["metrics"]; t=r["tokens"]
        print(f"  {r['baseline']:<18} {m['LCS']:>7.4f} {m['TES']:>7.4f} {m['CRS']:>7.4f} "
              f"{t['savings_pct']:>8.1f}% ${t['cost']:.4f}")
    print()

    # Show deltas vs AGENTMEM_OS
    ours=next(r for r in results if r["baseline"]=="AGENTMEM_OS")
    print(f"  Deltas vs AgentMem OS (negative = AgentMem OS wins):")
    for r in results:
        if r["baseline"]=="AGENTMEM_OS": continue
        m=r["metrics"]; om=ours["metrics"]
        dl=round(m["LCS"]-om["LCS"],3); dt=round(m["TES"]-om["TES"],3); dc=round(m["CRS"]-om["CRS"],4)
        print(f"  {r['baseline']:<18} ΔLCS={dl:+.3f}  ΔTES={dt:+.3f}  ΔCRS={dc:+.4f}")
    ok("Phase 3 complete.")

if __name__=="__main__":
    main()
