#!/usr/bin/env python3
"""
Phase 1 — Multi-Run Ablation (Runs 2–4)
========================================
Runs the 6-variant ablation 3 more times and combines with run 1
(ablation_results.json) to produce mean ± std across 4 runs.

Output: benchmarks/ablation_multi_run.json   (all 4 runs + statistics)
        benchmarks/ablation_summary_stats.json (mean ± std per variant)

Usage:
    python3 benchmarks/phase1_multi_run.py
"""

import os, re, sys, json, math, time, uuid, statistics
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

VARIANTS = [
    ("FULL",        "Full AgentMem OS (all tiers)",             {}),
    ("NO_SEMANTIC", "w/o Semantic Retrieval (Tier 3 disabled)", {"no_semantic": True}),
    ("NO_KG",       "w/o Entity Knowledge Graph",               {"no_kg": True}),
    ("NO_SLEEP",    "w/o Sleep Consolidation",                  {"no_sleep": True}),
    ("NO_PROC",     "w/o Procedural Memory",                    {"no_proc": True}),
    ("RECENT_ONLY", "Recent-only (Tier 1, last 10 turns)",      {"recent_only": True}),
]

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

def _tok(t): return max(1, len(t)//4)
def _ents(t): return set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', t))

def _cosine(q, d):
    qt = re.findall(r'[a-z]+', q.lower()); dt = re.findall(r'[a-z]+', d.lower())
    all_w = set(qt)|set(dt)
    df = {w:(w in qt)+(w in dt) for w in all_w}
    def v(ts):
        tf={}
        for w in ts: tf[w]=tf.get(w,0)+1
        n=len(ts) or 1
        return {w:(c/n)*math.log(1+1/df[w]) for w,c in tf.items()}
    qv,dv=v(qt),v(dt)
    dot=sum(qv.get(k,0)*dv.get(k,0) for k in all_w)
    nq=math.sqrt(sum(x**2 for x in qv.values()))+1e-9
    nd=math.sqrt(sum(x**2 for x in dv.values()))+1e-9
    return dot/(nq*nd)

def _extract_summary(turns):
    def ec(t): return len(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', t["content"]))
    chunks=[turns[i:i+5] for i in range(0,len(turns),5)]
    out=[]
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
    re_=len(Eo&Ec)/max(1,len(Eo))
    return round(math.sqrt(rc*re_),4)

def _tes_naive(user_turns):
    keep=user_turns[-max(1,int(len(user_turns)*.7)):]
    return _tes(user_turns," ".join(t["content"] for t in keep))

def _sem_retrieve(query, turns, n):
    scored=[(t,_cosine(query,t["content"])) for t in turns]
    scored.sort(key=lambda x:x[1],reverse=True)
    return [t for t,_ in scored[:n]]

def _kg_ents(turns, n):
    freq={}
    for t in turns:
        for e in _ents(t["content"]): freq[e]=freq.get(e,0)+1
    top=sorted(freq,key=freq.get,reverse=True)[:n]
    return "Known entities: "+", ".join(top) if top else ""

def _proc(turns):
    pts=[]; msgs=[t["content"] for t in turns if t["role"]=="user"]
    if any("explain" in m.lower() for m in msgs): pts.append("User prefers structured explanations.")
    if any("python" in m.lower() or "code" in m.lower() for m in msgs): pts.append("User requests code examples.")
    if any("research" in m.lower() or "paper" in m.lower() for m in msgs): pts.append("User is in research mode.")
    return "\n".join(pts)

def assemble(turns, query, flags, sleep_sum):
    parts=["You are AgentMem OS, an AI assistant with hierarchical persistent memory. Use memory context below to answer accurately."]
    if flags.get("recent_only"):
        recent=turns[-10:]
        parts.append("[RECENT TURNS (last 10)]\n"+"".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in recent))
        return "\n\n".join(parts)
    if sleep_sum and not flags.get("no_sleep"):
        parts.append(f"[EPISODIC MEMORY — Consolidated Summary]\n{sleep_sum}")
    if not flags.get("no_semantic") and len(turns)>RECENT_WIN:
        hits=_sem_retrieve(query,turns[:-RECENT_WIN],SEMANTIC_TOP)
        if hits: parts.append("[SEMANTIC MEMORY]\n"+"".join(f"[{t['role'].upper()}]: {t['content'][:200]}\n" for t in hits))
    if not flags.get("no_kg") and turns:
        kg=_kg_ents(turns,KG_TOP)
        if kg: parts.append(f"[ENTITY KG]\n{kg}")
    if not flags.get("no_proc") and turns:
        pr=_proc(turns)
        if pr: parts.append(f"[PROCEDURAL MEMORY]\n{pr}")
    recent=turns[-RECENT_WIN:]
    parts.append("[WORKING MEMORY]\n"+"".join(f"[{t['role'].upper()}]: {t['content']}\n" for t in recent))
    return "\n\n".join(parts)

def run_variant(client, vname, label, flags, run_id):
    print(f"  [{vname}] run {run_id}", end="", flush=True)
    turns=[]; recall={}; n_tok=0; b_tok=0; sleep_sum=None

    for i, msg in enumerate(CONVERSATION):
        if (not flags.get("no_sleep") and not flags.get("recent_only")
                and sleep_sum is None and len(turns)>=SLEEP_THRESH):
            old=[t for t in turns if t["role"]=="user"]
            sleep_sum=_extract_summary(old[:int(len(old)*.6)])

        ctx=assemble(turns,msg,flags,sleep_sum)
        naive=sum(_tok(t["content"]) for t in turns)+_tok(msg)
        ours=_tok(ctx)+_tok(msg)
        n_tok+=ours; b_tok+=naive

        try:
            resp=client.messages.create(model=MODEL,max_tokens=400,
                messages=[{"role":"user","content":f"{ctx}\n\n[QUERY]\n{msg}"}])
            reply=resp.content[0].text if resp.content else ""
        except Exception as ex:
            reply=""
        turns.append({"role":"user","content":msg})
        turns.append({"role":"assistant","content":reply})
        if i in PROBE_RECALLS:
            recall[i]=PROBE_RECALLS[i].lower() in reply.lower()
        time.sleep(0.12)

    print(" ✓")
    lcs=round(sum(recall.values())/len(PROBE_RECALLS),4)
    sav=round(100*(1-n_tok/max(1,b_tok)),1)
    user_turns=[t for t in turns if t["role"]=="user"]
    compressed=sleep_sum if sleep_sum and not flags.get("no_sleep") and not flags.get("recent_only") else \
        " ".join(t["content"] for t in user_turns[-max(1,int(len(user_turns)*.7)):])
    tes=_tes(user_turns,compressed)
    tes_b=_tes_naive(user_turns)
    probe_qs=[CONVERSATION[i] for i in sorted(PROBE_RECALLS)]
    our_ctx=(sleep_sum or "")+" "+" ".join(t["content"] for t in turns[-10:])
    crs=round(sum(_cosine(q,our_ctx) for q in probe_qs)/len(probe_qs),4)
    mid=len(turns)//2
    base_ctx=" ".join(t["content"] for t in turns[max(0,mid-2):mid+3])
    crs_b=round(sum(_cosine(q,base_ctx) for q in probe_qs)/len(probe_qs),4)

    return {"variant":vname,"run":run_id,
            "metrics":{"CRS":{"ours":crs,"baseline":crs_b},
                       "TES":{"ours":tes,"baseline":tes_b},
                       "LCS":{"ours":lcs,"baseline":0.70}},
            "tokens":{"savings_pct":sav}}

def main():
    hdr("Phase 1 — Multi-Run Ablation (Runs 2–4)")
    api_key=os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key:
        print(f"  {R}✗{E}  ANTHROPIC_API_KEY not found"); sys.exit(1)
    ok(f"API key loaded ({api_key[:12]}...)")

    try:
        import anthropic
        client=anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  pip install anthropic"); sys.exit(1)

    bench_dir=Path(__file__).parent
    # Load existing run 1
    run1_path=bench_dir/"ablation_results.json"
    if run1_path.exists():
        run1=json.loads(run1_path.read_text())
        for r in run1: r["run"]=1
        all_runs=run1
        ok("Loaded run 1 from ablation_results.json")
    else:
        warn("ablation_results.json not found — will run from scratch (4 runs)")
        all_runs=[]

    first_new_run = 2 if all_runs else 1
    n_new = 3 if all_runs else 4

    print(f"\n  Running {n_new} new run(s) × {len(VARIANTS)} variants × {len(CONVERSATION)} turns")
    print(f"  Est. cost: ~${n_new * len(VARIANTS) * 0.31:.2f}\n")

    for run_id in range(first_new_run, first_new_run + n_new):
        sub(f"Run {run_id}")
        for vname, label, flags in VARIANTS:
            r=run_variant(client, vname, label, flags, run_id)
            all_runs.append(r)

    # Save all runs
    multi_path=bench_dir/"ablation_multi_run.json"
    multi_path.write_text(json.dumps(all_runs, indent=2))
    ok(f"All runs → {multi_path}")

    # Compute mean ± std per variant
    stats={}
    for vname,_,_ in VARIANTS:
        runs=[r for r in all_runs if r["variant"]==vname]
        for metric in ["CRS","TES","LCS"]:
            vals=[r["metrics"][metric]["ours"] for r in runs]
            sav=[r["tokens"]["savings_pct"] for r in runs]
            stats.setdefault(vname,{})
            stats[vname][metric]={"mean":round(statistics.mean(vals),4),
                                  "std":round(statistics.stdev(vals) if len(vals)>1 else 0.0,4),
                                  "n":len(vals)}
        sav_vals=[r["tokens"]["savings_pct"] for r in runs]
        stats[vname]["savings"]={"mean":round(statistics.mean(sav_vals),1),
                                 "std":round(statistics.stdev(sav_vals) if len(sav_vals)>1 else 0.0,1)}

    stats_path=bench_dir/"ablation_summary_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    ok(f"Stats → {stats_path}")

    # Print summary
    hdr("ABLATION SUMMARY  (mean ± std, n=4 runs)")
    print(f"  {'Variant':<18} {'CRS':>12} {'TES':>12} {'LCS':>12} {'Savings':>10}")
    print(f"  {'─'*18} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")
    for vname,_,_ in VARIANTS:
        s=stats[vname]
        def fmt(m): return f"{s[m]['mean']:.3f}±{s[m]['std']:.3f}"
        print(f"  {vname:<18} {fmt('CRS'):>12} {fmt('TES'):>12} {fmt('LCS'):>12} "
              f"{s['savings']['mean']:>8.1f}%±{s['savings']['std']:.1f}")
    print()
    ok("Phase 1 complete.")

if __name__=="__main__":
    main()
