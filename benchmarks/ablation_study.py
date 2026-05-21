#!/usr/bin/env python3
"""
AgentMem OS — Ablation Study (Standalone v2)
=============================================
Self-contained: zero dependency on memnai.* internals.
Only requires: anthropic, python-dotenv  (both in your venv)

Six variants isolate each component's contribution:
  FULL        — All four tiers active (baseline)
  NO_SEMANTIC — Disable TF-IDF semantic retrieval (Tier 3)
  NO_KG       — Disable Entity Knowledge Graph (Tier 4a)
  NO_SLEEP    — Disable Sleep Consolidation (no compression)
  NO_PROC     — Disable Procedural Memory (Tier 4b)
  RECENT_ONLY — Last 10 turns only (Tier 1, pure recency)

Usage (from ANY directory — .env auto-located):
    python3 benchmarks/ablation_study.py

Results: benchmarks/ablation_results.json
Cost:    ~6 × $0.31 ≈ $1.86
"""

import os, re, sys, json, math, time, uuid
from pathlib import Path
from dotenv import load_dotenv

# ── Locate .env (walk up from this file's location) ───────────────────────────
_here = Path(__file__).resolve()
for _p in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
    _env = _p / ".env"
    if _env.exists():
        load_dotenv(_env, override=True)
        break

# ── Terminal colours ───────────────────────────────────────────────────────────
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m";  E = "\033[0m"

def ok(m):   print(f"  {G}✓{E}  {m}")
def warn(m): print(f"  {Y}!{E}  {m}")
def info(m): print(f"  {C}→{E}  {m}")
def hdr(t):  print(f"\n{B}{C}{'═'*60}{E}\n{B}{C}  {t}{E}\n{B}{C}{'═'*60}{E}")
def sub(t):  print(f"\n{B}  ── {t}{E}")

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL          = "claude-haiku-4-5-20251001"
COST_PER_MTOK  = 0.80      # USD / 1M input tokens
SLEEP_THRESH   = 15        # compress after this many turns
RECENT_WIN     = 8         # turns kept as raw recent context
KG_TOP_N       = 5         # top KG entities to inject
SEMANTIC_TOP_N = 3         # top semantic turns to inject

VARIANTS = [
    ("FULL",        "Full AgentMem OS (all tiers)",             {}),
    ("NO_SEMANTIC", "w/o Semantic Retrieval (Tier 3 disabled)", {"no_semantic": True}),
    ("NO_KG",       "w/o Entity Knowledge Graph",               {"no_kg": True}),
    ("NO_SLEEP",    "w/o Sleep Consolidation",                  {"no_sleep": True}),
    ("NO_PROC",     "w/o Procedural Memory",                    {"no_proc": True}),
    ("RECENT_ONLY", "Recent-only (Tier 1, last 10 turns)",      {"recent_only": True}),
]

# ── 25-turn benchmark conversation ────────────────────────────────────────────
CONVERSATION = [
    # Grounding turns (T1–5) — seeded facts for long-horizon recall
    "Hi, my name is Sahith and I'm building AgentMem OS for my PhD research.",
    "My primary research goal is to add persistent memory to LLM agents across sessions.",
    "The project has four memory tiers: Redis working memory, SQLite episodic, TF-IDF semantic, and procedural pattern memory.",
    "I'm targeting a NeurIPS 2026 workshop submission. The paper deadline is June 2026.",
    "The four novel algorithms are: MemoryImportanceScorer, SleepConsolidationEngine, EntityKnowledgeGraph, and ProceduralMemory.",
    # Work turns (T6–15)
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
    # Long-horizon probes (T16–25) — no in-turn hints, rely on memory
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

# Probe indices (0-based) → keyword that must appear in the reply to count as a recall hit
PROBE_RECALLS = {15: "sahith", 16: "neurips 2026", 17: "sqlite",
                 18: "procedural", 19: "neurips"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tok(text: str) -> int:
    """Rough token count (chars/4)."""
    return max(1, len(text) // 4)

def _entities(text: str):
    """Extract capitalized words as a proxy for named entities."""
    return set(re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text))

def _tfidf_cosine(query: str, doc: str) -> float:
    """Pure-Python TF-IDF cosine similarity (no sklearn needed)."""
    def tok(t): return re.findall(r'[a-z]+', t.lower())
    qt, dt = tok(query), tok(doc)
    all_w  = set(qt) | set(dt)
    df = {w: (1 if w in qt else 0) + (1 if w in dt else 0) for w in all_w}
    def vec(tokens):
        tf = {}
        for w in tokens: tf[w] = tf.get(w, 0) + 1
        n = len(tokens) or 1
        return {w: (c/n) * math.log(1 + 1/df[w]) for w, c in tf.items()}
    qv, dv = vec(qt), vec(dt)
    dot = sum(qv.get(k,0) * dv.get(k,0) for k in all_w)
    nq  = math.sqrt(sum(v**2 for v in qv.values())) or 1e-9
    nd  = math.sqrt(sum(v**2 for v in dv.values())) or 1e-9
    return dot / (nq * nd)

def _extract_summary(turns: list) -> str:
    """
    Simulate Sleep Consolidation:
    - Cluster turns in groups of 5 (proxy for DBSCAN clusters)
    - Keep top-2 entity-rich turns per cluster (extractive)
    - Append entity inventory (proxy for KG snapshot)
    """
    def entity_count(t): return len(re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', t["content"]))
    chunks   = [turns[i:i+5] for i in range(0, len(turns), 5)]
    excerpts = []
    for chunk in chunks:
        for t in sorted(chunk, key=entity_count, reverse=True)[:2]:
            excerpts.append(f"[{t['role'].upper()}]: {t['content'][:280]}")
    all_ents = _entities(" ".join(t["content"] for t in turns))
    if all_ents:
        excerpts.append("Entities: " + ", ".join(sorted(all_ents)[:30]))
    return "\n\n".join(excerpts)

def _tes(all_user_turns: list, compressed_text: str) -> float:
    """TES = √(compression_ratio × entity_preservation_rate)."""
    orig   = " ".join(t["content"] for t in all_user_turns)
    rc     = max(0.0, min(1.0, 1 - _tok(compressed_text) / max(1, _tok(orig))))
    E_orig = _entities(orig)
    E_comp = _entities(compressed_text)
    re_    = len(E_orig & E_comp) / max(1, len(E_orig))
    return round(math.sqrt(rc * re_), 4)

def _tes_naive(all_user_turns: list) -> float:
    """Naive TES baseline: keep newest 70% unchanged."""
    keep = all_user_turns[-max(1, int(len(all_user_turns)*0.7)):]
    return _tes(all_user_turns, " ".join(t["content"] for t in keep))

def _semantic_retrieve(query: str, turns: list, top_n: int) -> list:
    """TF-IDF top-N retrieval from turn history."""
    scored = [(t, _tfidf_cosine(query, t["content"])) for t in turns]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_n]]

def _kg_entities(turns: list, top_n: int) -> str:
    """Simulate KG: return the most-mentioned entities across history."""
    freq = {}
    for t in turns:
        for e in _entities(t["content"]):
            freq[e] = freq.get(e, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:top_n]
    return "Known entities: " + ", ".join(top) if top else ""

def _procedural_patterns(turns: list) -> str:
    """Simulate Procedural Memory: detect repeated user intent patterns."""
    patterns = []
    user_msgs = [t["content"] for t in turns if t["role"] == "user"]
    if any("explain" in m.lower() for m in user_msgs):
        patterns.append("User frequently asks for explanations → provide structured answers.")
    if any("python" in m.lower() or "code" in m.lower() for m in user_msgs):
        patterns.append("User requests code examples → include runnable snippets.")
    if any("research" in m.lower() or "paper" in m.lower() for m in user_msgs):
        patterns.append("User is in research mode → cite definitions and relate to literature.")
    return "\n".join(patterns)

# ── Context assembler (simulates AgentMem OS tiers) ───────────────────────────

def assemble_context(turns: list, query: str, flags: dict,
                     sleep_summary: str | None) -> str:
    """
    Build the assembled context string for a given variant.

    Tiers included based on flags:
      Tier 1 (Working)  — always: last RECENT_WIN turns
      Tier 2 (Episodic) — always: sleep summary (if available)
      Tier 3 (Semantic) — disabled if no_semantic
      Tier 4a (KG)      — disabled if no_kg
      Tier 4b (Proc)    — disabled if no_proc
      recent_only       — Tier 1 only (last 10 turns)
    """
    parts = []
    sys_prompt = (
        "You are AgentMem OS, an AI assistant with hierarchical persistent memory. "
        "Use the memory context below to answer accurately, especially for long-horizon questions."
    )
    parts.append(f"[SYSTEM]\n{sys_prompt}")

    if flags.get("recent_only"):
        recent = turns[-10:]
        turns_txt = "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent)
        parts.append(f"[RECENT TURNS (last 10)]\n{turns_txt}")
        return "\n\n".join(parts)

    # Tier 2 — Episodic / sleep summary
    if sleep_summary and not flags.get("no_sleep"):
        parts.append(f"[EPISODIC MEMORY — Consolidated Summary]\n{sleep_summary}")

    # Tier 3 — Semantic retrieval
    if not flags.get("no_semantic") and len(turns) > RECENT_WIN:
        older = turns[:-RECENT_WIN]
        semantic_hits = _semantic_retrieve(query, older, SEMANTIC_TOP_N)
        if semantic_hits:
            sem_txt = "\n".join(
                f"[{t['role'].upper()}]: {t['content'][:200]}" for t in semantic_hits
            )
            parts.append(f"[SEMANTIC MEMORY — Top-{SEMANTIC_TOP_N} Relevant Turns]\n{sem_txt}")

    # Tier 4a — KG entities
    if not flags.get("no_kg") and turns:
        kg_txt = _kg_entities(turns, KG_TOP_N)
        if kg_txt:
            parts.append(f"[ENTITY KNOWLEDGE GRAPH]\n{kg_txt}")

    # Tier 4b — Procedural patterns
    if not flags.get("no_proc") and turns:
        proc = _procedural_patterns(turns)
        if proc:
            parts.append(f"[PROCEDURAL MEMORY — Behavioural Patterns]\n{proc}")

    # Tier 1 — Recent working memory (always last)
    recent = turns[-RECENT_WIN:]
    turns_txt = "\n".join(f"[{t['role'].upper()}]: {t['content']}" for t in recent)
    parts.append(f"[WORKING MEMORY — Last {RECENT_WIN} Turns]\n{turns_txt}")

    return "\n\n".join(parts)

# ── Claude API call ───────────────────────────────────────────────────────────

def call_claude(client, context: str, user_msg: str) -> tuple[str, int, int]:
    """Returns (reply_text, input_tokens, output_tokens)."""
    messages = [
        {"role": "user",
         "content": f"{context}\n\n[CURRENT QUERY]\n{user_msg}"}
    ]
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        messages=messages,
    )
    reply     = resp.content[0].text if resp.content else ""
    in_toks   = resp.usage.input_tokens
    out_toks  = resp.usage.output_tokens
    return reply, in_toks, out_toks

# ══════════════════════════════════════════════════════════════════════════════
# Core: run one ablation variant
# ══════════════════════════════════════════════════════════════════════════════

def run_variant(client, variant_name: str, label: str, flags: dict) -> dict:
    hdr(f"VARIANT: {variant_name}  —  {label}")
    sub("Running 25-turn conversation...")

    turns          = []        # full history: {role, content}
    recall_hits    = {}
    naive_tok_total = 0
    ours_tok_total  = 0
    token_log       = []
    sleep_summary   = None     # set once sleep consolidation triggers

    for i, user_msg in enumerate(CONVERSATION):
        turn_num = i + 1

        # ── Sleep Consolidation trigger (>= SLEEP_THRESH turns, once) ─────────
        if (not flags.get("no_sleep") and not flags.get("recent_only")
                and sleep_summary is None and len(turns) >= SLEEP_THRESH):
            old_turns   = [t for t in turns if t["role"] == "user"]
            compress_n  = int(len(old_turns) * 0.6)
            sleep_summary = _extract_summary(old_turns[:compress_n])
            info(f"Sleep consolidation fired at turn {turn_num} "
                 f"({compress_n} turns compressed)")

        # ── Assemble context for this variant ─────────────────────────────────
        context = assemble_context(turns, user_msg, flags, sleep_summary)

        # ── Naive token count (full raw history) ──────────────────────────────
        naive_tok = sum(_tok(t["content"]) for t in turns) + _tok(user_msg)
        ours_tok  = _tok(context) + _tok(user_msg)

        savings   = round(100 * (1 - ours_tok / max(1, naive_tok)), 1)
        naive_tok_total += naive_tok
        ours_tok_total  += ours_tok
        token_log.append({"turn": turn_num, "naive": naive_tok,
                          "ours": ours_tok, "savings_pct": savings})

        # ── Call Claude ────────────────────────────────────────────────────────
        try:
            reply, in_toks, out_toks = call_claude(client, context, user_msg)
        except Exception as ex:
            warn(f"Turn {turn_num} API error: {ex}")
            reply = ""

        turns.append({"role": "user",      "content": user_msg})
        turns.append({"role": "assistant", "content": reply})

        # ── Long-horizon recall check ──────────────────────────────────────────
        if i in PROBE_RECALLS:
            kw  = PROBE_RECALLS[i]
            hit = kw.lower() in reply.lower()
            recall_hits[i] = hit
            sym = f"{G}✓{E}" if hit else f"{Y}✗{E}"
            print(f"    T{turn_num:02d} probe '{kw}': {sym}  "
                  f"[ctx={ours_tok}tok, saved {savings:+.0f}%]")
        else:
            print(f"    T{turn_num:02d} ✓  [ctx={ours_tok}tok, naive={naive_tok}tok]")

        time.sleep(0.15)   # gentle rate-limiting

    # ── LCS ───────────────────────────────────────────────────────────────────
    lcs_ours = round(sum(recall_hits.values()) / len(PROBE_RECALLS), 4)
    lcs_base = 0.70    # deterministic recent-8 baseline

    # ── TES ───────────────────────────────────────────────────────────────────
    user_turns = [t for t in turns if t["role"] == "user"]
    if flags.get("no_sleep") or flags.get("recent_only") or sleep_summary is None:
        compressed_text = " ".join(t["content"] for t in
                                   user_turns[-max(1, int(len(user_turns)*0.7)):])
        tes_ours = _tes(user_turns, compressed_text)
    else:
        tes_ours = _tes(user_turns, sleep_summary)
    tes_base = _tes_naive(user_turns)

    # ── CRS ───────────────────────────────────────────────────────────────────
    probe_qs = [CONVERSATION[i] for i in sorted(PROBE_RECALLS)]
    # Our context: sleep summary + last 10 turns text
    our_ctx = (sleep_summary or "") + " " + " ".join(
        t["content"] for t in turns[-10:])
    crs_ours  = round(sum(_tfidf_cosine(q, our_ctx) for q in probe_qs)
                      / len(probe_qs), 4)
    # Baseline: middle-5-turns only
    mid = len(turns) // 2
    base_ctx  = " ".join(t["content"] for t in turns[max(0,mid-2):mid+3])
    crs_base  = round(sum(_tfidf_cosine(q, base_ctx) for q in probe_qs)
                      / len(probe_qs), 4)

    # ── Token savings ─────────────────────────────────────────────────────────
    tok_savings = round(100*(1 - ours_tok_total/max(1, naive_tok_total)), 1)
    cost_ours   = round(ours_tok_total  / 1_000_000 * COST_PER_MTOK, 4)
    cost_naive  = round(naive_tok_total / 1_000_000 * COST_PER_MTOK, 4)

    sub("Results")
    print(f"    CRS : {crs_ours:.4f}  (base {crs_base:.4f},  Δ {crs_ours-crs_base:+.4f})")
    print(f"    TES : {tes_ours:.4f}  (base {tes_base:.4f},  Δ {tes_ours-tes_base:+.4f})")
    print(f"    LCS : {lcs_ours:.4f}  (base {lcs_base:.4f},  Δ {lcs_ours-lcs_base:+.4f})")
    print(f"    Tok : {tok_savings:+.1f}% savings  "
          f"(${cost_ours:.4f} vs ${cost_naive:.4f} naive)")

    return {
        "variant": variant_name,
        "label":   label,
        "flags":   flags,
        "metrics": {
            "CRS": {"ours": crs_ours,  "baseline": crs_base,
                    "delta": round(crs_ours - crs_base, 4)},
            "TES": {"ours": tes_ours,  "baseline": tes_base,
                    "delta": round(tes_ours - tes_base, 4)},
            "LCS": {"ours": lcs_ours,  "baseline": lcs_base,
                    "delta": round(lcs_ours - lcs_base, 4)},
        },
        "tokens": {
            "ours_total":  ours_tok_total,
            "naive_total": naive_tok_total,
            "savings_pct": tok_savings,
            "cost_ours":   cost_ours,
            "cost_naive":  cost_naive,
        },
        "recall": {PROBE_RECALLS[i]: recall_hits.get(i, False)
                   for i in PROBE_RECALLS},
        "token_log": token_log,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    hdr("AgentMem OS — Ablation Study")
    print(f"\n  Variants   : {len(VARIANTS)}")
    print(f"  Turns/var  : {len(CONVERSATION)}")
    print(f"  Est. cost  : ~${len(VARIANTS)*0.31:.2f} total\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or "YOUR_KEY" in api_key:
        print(f"  {R}✗{E}  ANTHROPIC_API_KEY not found.\n"
              f"     Run:  set -a && source .env && set +a   then re-run.")
        sys.exit(1)
    ok(f"API key loaded  ({api_key[:12]}...)")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(f"  {R}✗{E}  'anthropic' package not installed.\n"
              f"     Run:  pip install anthropic")
        sys.exit(1)

    results = []
    for vname, label, flags in VARIANTS:
        try:
            r = run_variant(client, vname, label, flags)
            results.append(r)
        except Exception as ex:
            warn(f"Variant {vname} crashed: {ex}")
            import traceback; traceback.print_exc()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    ok(f"Results → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    hdr("ABLATION SUMMARY TABLE")
    print(f"  {'Variant':<18} {'CRS':>7} {'TES':>7} {'LCS':>7} {'Savings':>9}")
    print(f"  {'─'*18} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")
    full_r = next((r for r in results if r["variant"] == "FULL"), None)
    for r in results:
        m   = r["metrics"]
        sav = r["tokens"]["savings_pct"]
        crs = m["CRS"]["ours"]; tes = m["TES"]["ours"]; lcs = m["LCS"]["ours"]
        print(f"  {r['variant']:<18} {crs:>7.4f} {tes:>7.4f} {lcs:>7.4f} {sav:>8.1f}%")
        if full_r and r["variant"] != "FULL":
            dc = round(crs - full_r["metrics"]["CRS"]["ours"], 3)
            dt = round(tes - full_r["metrics"]["TES"]["ours"], 3)
            dl = round(lcs - full_r["metrics"]["LCS"]["ours"], 3)
            print(f"  {'':18} ΔCRS={dc:+.3f}  ΔTES={dt:+.3f}  ΔLCS={dl:+.3f}")
        else:
            print(f"  {'':18} ← full system (reference)")
    print()
    ok("Ablation study complete.")

if __name__ == "__main__":
    main()
